"""
Project 5 -- Quantization and Efficiency Pareto Frontier, executed against the Project 1
Ninapro DB5 CNN checkpoint (subject S1, see train_cnn_checkpoint.py).

Configurations: FP32 baseline, FP16, dynamic int8, static int8 (calibrated), QAT, structured
pruning (30/50/70%), knowledge distillation into a small MLP on Hudgins+AR features, plus an
LDA-on-features baseline for comparison. Metrics: macro F1, size (KB), median/p95 latency
(1000 runs, single CPU thread), MACs. Also: per-layer quantization sensitivity, ONNX export
+ numerical parity check.
"""
import os
import sys
import io
import copy
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.ao.quantization as tq
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader

from ptflops import get_model_complexity_info
import onnx
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_project1_lean as p1

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SUBJECT = "S1"
RNG_SEED = 0

torch.manual_seed(RNG_SEED)
torch.set_num_threads(1)  # single-thread timing -- matches a microcontroller


def load_checkpoint():
    w = np.load(os.path.join(CKPT_DIR, f"cnn_{SUBJECT}_windows.npz"))
    norm = np.load(os.path.join(CKPT_DIR, f"cnn_{SUBJECT}_norm.npz"))
    n_classes = int(w["n_classes"])
    n_channels = int(w["n_channels"])
    model = p1.EMG1DCNN(n_channels, n_classes)
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, f"cnn_{SUBJECT}.pt")))
    model.eval()
    return model, norm["mu"], norm["sd"], w, n_classes, n_channels


def eval_macro_f1(model, X, y, dtype=torch.float32):
    from sklearn.metrics import f1_score
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X, dtype=dtype).permute(0, 2, 1)
        pred = model(x).float().argmax(dim=1).numpy()
    return f1_score(y, pred, average="macro")


def model_size_kb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return len(buf.getvalue()) / 1024.0


def bench_latency_ms(model, sample_x, n_runs=1000, dtype=torch.float32):
    model.eval()
    x = torch.tensor(sample_x[:1], dtype=dtype).permute(0, 2, 1)
    with torch.no_grad():
        for _ in range(20):
            model(x)
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))}


def count_macs(model, input_shape):
    try:
        macs, _ = get_model_complexity_info(model, input_shape, as_strings=False,
                                             print_per_layer_stat=False, verbose=False)
        return macs
    except Exception as e:
        print(f"  MAC counting failed ({e}), leaving blank")
        return None


def collect_metrics(name, model, X_eval, y_eval, input_shape, dtype=torch.float32, extra=None):
    row = {"config": name, "macro_f1": eval_macro_f1(model, X_eval, y_eval, dtype),
           "size_kb": model_size_kb(model), **bench_latency_ms(model, X_eval, dtype=dtype)}
    row["macs"] = count_macs(model, input_shape)
    if extra:
        row.update(extra)
    return row


def fuse_cnn(model, is_qat=False):
    """Fuse each [Conv1d, BatchNorm1d, ReLU] triple in EMG1DCNN.net in place.
    Required before static/QAT quantization -- torch's QuantizedCPU backend has no kernel
    for a bare nn.BatchNorm1d, so converting an unfused graph fails at inference with
    'Could not run aten::native_batch_norm with arguments from the QuantizedCPU backend'."""
    triples = [["net.0", "net.1", "net.2"], ["net.4", "net.5", "net.6"], ["net.8", "net.9", "net.10"]]
    fuse_fn = tq.fuse_modules_qat if is_qat else tq.fuse_modules
    fuse_fn(model, triples, inplace=True)
    return model


class QuantWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.quant = tq.QuantStub()
        self.model = model
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        return self.dequant(self.model(self.quant(x)))


class SmallMLP(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


def main():
    fp32_model, mu, sd, w, n_classes, n_channels = load_checkpoint()
    X, y, rep = w["X"], w["y"], w["rep"]
    tr_mask, te_mask = w["train_mask"], w["test_mask"]

    Xtr = ((X[tr_mask] - mu) / sd).astype(np.float32)
    Xte = ((X[te_mask] - mu) / sd).astype(np.float32)
    ytr, yte = y[tr_mask], y[te_mask]
    input_shape = (n_channels, X.shape[1])

    results = []
    print("=== FP32 baseline ===")
    results.append(collect_metrics("fp32_baseline", fp32_model, Xte, yte, input_shape))
    print(results[-1])

    print("=== FP16 ===")
    fp16_model = copy.deepcopy(fp32_model).half()
    row = {"config": "fp16", "macro_f1": eval_macro_f1(fp16_model, Xte, yte, dtype=torch.float16),
           "size_kb": model_size_kb(fp16_model)}
    try:
        row.update(bench_latency_ms(fp16_model, Xte, dtype=torch.float16))
    except Exception as e:
        row["median_ms"], row["p95_ms"] = None, None
        print(f"  fp16 CPU latency bench skipped: {e}")
    results.append(row)
    print(results[-1])

    print("=== dynamic int8 ===")
    dynamic_int8 = tq.quantize_dynamic(copy.deepcopy(fp32_model), {nn.Linear}, dtype=torch.qint8)
    results.append(collect_metrics("dynamic_int8", dynamic_int8, Xte, yte, input_shape))
    print(results[-1])

    print("=== static int8 (calibrated) ===")
    calib_idx = np.random.RandomState(RNG_SEED).choice(len(Xtr), size=min(200, len(Xtr)), replace=False)
    static_model = QuantWrapper(fuse_cnn(copy.deepcopy(fp32_model)))
    static_model.eval()
    static_model.qconfig = tq.get_default_qconfig("fbgemm")
    tq.prepare(static_model, inplace=True)
    with torch.no_grad():
        static_model(torch.tensor(Xtr[calib_idx], dtype=torch.float32).permute(0, 2, 1))
    static_int8 = tq.convert(static_model, inplace=False)
    results.append(collect_metrics("static_int8_calibrated", static_int8, Xte, yte, input_shape))
    print(results[-1])

    print("=== QAT ===")
    qat_base = copy.deepcopy(fp32_model)
    qat_base.train()
    fuse_cnn(qat_base, is_qat=True)
    qat_model = QuantWrapper(qat_base)
    qat_model.train()
    qat_model.qconfig = tq.get_default_qat_qconfig("fbgemm")
    tq.prepare_qat(qat_model, inplace=True)
    opt = torch.optim.Adam(qat_model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(p1.WindowDataset(Xtr, ytr), batch_size=64, shuffle=True)
    for _ in range(5):
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(qat_model(xb), yb)
            loss.backward()
            opt.step()
    qat_model.eval()
    qat_int8 = tq.convert(qat_model, inplace=False)
    results.append(collect_metrics("qat_int8", qat_int8, Xte, yte, input_shape))
    print(results[-1])

    print("=== structured pruning (30/50/70%) ===")
    for sparsity in (0.3, 0.5, 0.7):
        pruned = copy.deepcopy(fp32_model)
        for module in pruned.modules():
            if isinstance(module, nn.Conv1d):
                prune.ln_structured(module, name="weight", amount=sparsity, n=2, dim=0)
                prune.remove(module, "weight")
        opt = torch.optim.Adam(pruned.parameters(), lr=1e-4)
        loader = DataLoader(p1.WindowDataset(Xtr, ytr), batch_size=64, shuffle=True)
        pruned.train()
        for _ in range(5):
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(pruned(xb), yb)
                loss.backward()
                opt.step()
        pruned.eval()
        results.append(collect_metrics(f"pruned_{int(sparsity*100)}pct", pruned, Xte, yte, input_shape,
                                        extra={"sparsity": sparsity}))
        print(results[-1])

    print("=== knowledge distillation -> small MLP on Hudgins+AR features ===")
    F_train = p1.extract_feature_matrix(X[tr_mask])
    F_test = p1.extract_feature_matrix(X[te_mask])
    from sklearn.preprocessing import StandardScaler
    feat_scaler = StandardScaler().fit(F_train)
    F_train_s = feat_scaler.transform(F_train).astype(np.float32)
    F_test_s = feat_scaler.transform(F_test).astype(np.float32)

    T, alpha, epochs = 2.0, 0.5, 30
    student = SmallMLP(F_train_s.shape[1], n_classes)
    opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        teacher_logits = fp32_model(torch.tensor(Xtr, dtype=torch.float32).permute(0, 2, 1))
        teacher_soft = torch.softmax(teacher_logits / T, dim=1)
    y_t = torch.tensor(ytr, dtype=torch.long)
    x_t = torch.tensor(F_train_s, dtype=torch.float32)
    student.train()
    for _ in range(epochs):
        opt.zero_grad()
        student_logits = student(x_t)
        hard_loss = ce(student_logits, y_t)
        soft_loss = nn.functional.kl_div(torch.log_softmax(student_logits / T, dim=1), teacher_soft,
                                          reduction="batchmean") * (T ** 2)
        (alpha * hard_loss + (1 - alpha) * soft_loss).backward()
        opt.step()

    from sklearn.metrics import f1_score
    student.eval()
    with torch.no_grad():
        pred = student(torch.tensor(F_test_s)).argmax(dim=1).numpy()
    row = {"config": "distilled_mlp", "macro_f1": f1_score(yte, pred, average="macro"),
           "size_kb": model_size_kb(student)}
    x_bench = torch.tensor(F_test_s[:1])
    with torch.no_grad():
        for _ in range(20):
            student(x_bench)
        times = []
        for _ in range(1000):
            t0 = time.perf_counter()
            student(x_bench)
            times.append((time.perf_counter() - t0) * 1000)
    row["median_ms"], row["p95_ms"] = float(np.median(times)), float(np.percentile(times, 95))
    results.append(row)
    print(results[-1])

    print("=== LDA-on-features baseline ===")
    lda = p1.LinearDiscriminantAnalysis().fit(F_train_s, ytr)
    lda_pred = lda.predict(F_test_s)
    import pickle
    lda_size_kb = len(pickle.dumps(lda)) / 1024.0
    x_bench = F_test_s[:1]
    for _ in range(20):
        lda.predict(x_bench)
    times = []
    for _ in range(1000):
        t0 = time.perf_counter()
        lda.predict(x_bench)
        times.append((time.perf_counter() - t0) * 1000)
    results.append({"config": "lda_features_baseline", "macro_f1": f1_score(yte, lda_pred, average="macro"),
                     "size_kb": lda_size_kb, "median_ms": float(np.median(times)),
                     "p95_ms": float(np.percentile(times, 95))})
    print(results[-1])

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUT_DIR, "project5_results.csv"), index=False)
    print("\n=== full Pareto table ===")
    print(results_df.sort_values("macro_f1", ascending=False).to_string(index=False))

    print("\n=== per-layer quantization sensitivity ===")
    conv_layers = [name for name, mod in fp32_model.named_modules() if isinstance(mod, nn.Conv1d)]
    fp32_f1 = results_df.query("config == 'fp32_baseline'")["macro_f1"].iloc[0]
    sens_rows = []
    for layer_name in conv_layers:
        m = copy.deepcopy(fp32_model)
        target = dict(m.named_modules())[layer_name]
        fq = tq.FakeQuantize.with_args(observer=tq.MovingAverageMinMaxObserver,
                                        quant_min=-128, quant_max=127, dtype=torch.qint8)()
        orig_forward = target.forward
        target.forward = lambda x, _o=orig_forward, _f=fq: _f(_o(x))
        f1 = eval_macro_f1(m, Xte, yte)
        sens_rows.append({"layer": layer_name, "macro_f1_with_layer_quantized": f1,
                           "delta_vs_fp32": f1 - fp32_f1})
    sens_df = pd.DataFrame(sens_rows).sort_values("delta_vs_fp32")
    sens_df.to_csv(os.path.join(OUT_DIR, "project5_layer_sensitivity.csv"), index=False)
    print(sens_df.to_string(index=False))

    print("\n=== ONNX export + parity check ===")
    dummy = torch.randn(1, n_channels, X.shape[1])
    onnx_path = os.path.join(OUT_DIR, "cnn_S1.onnx")
    torch.onnx.export(fp32_model, dummy, onnx_path, input_names=["emg_window"], output_names=["logits"],
                       dynamic_axes={"emg_window": {0: "batch"}, "logits": {0: "batch"}}, opset_version=13,
                       dynamo=False)  # legacy exporter -- the new dynamo one prints a unicode checkmark
                                       # that crashes on Windows' cp1252 console encoding
    onnx.checker.check_model(onnx.load(onnx_path))
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    sample = Xte[:1].transpose(0, 2, 1).astype(np.float32)  # (N,T,C) -> (N,C,T), what the model expects
    onnx_out = sess.run(None, {"emg_window": sample})[0]
    with torch.no_grad():
        torch_out = fp32_model(torch.tensor(sample)).numpy()
    max_diff = np.max(np.abs(onnx_out - torch_out))
    print(f"ONNX vs PyTorch max abs logit diff: {max_diff:.2e} (exported to {onnx_path})")
    assert max_diff < 1e-3, "ONNX export diverges from PyTorch"

    print("\nALL DONE")


if __name__ == "__main__":
    main()
