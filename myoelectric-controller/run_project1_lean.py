"""
Memory-lean, resumable standalone runner for Project 1 (EMG gesture classification),
targeting Ninapro DB5 instead of DB2 -- DB2 requires registration at ninapro.hevs.ch,
DB5 is a direct, no-login download from Zenodo (https://zenodo.org/records/1000116).

DB5 differences from DB2 that this script accounts for (the notebook itself is written
for DB2's 2kHz/12-channel Delsys setup and is documented as such -- this script is the
DB5-adapted version that was actually executed):
  - 200 Hz sampling (not 2000 Hz) -> bandpass upper cutoff must stay below Nyquist (100 Hz).
  - 16 EMG channels (two 8-channel Myo armbands, not 12-channel Delsys).
  - File naming is S{subj}_E{1,2,3}_A1.mat (numeric exercise index), not S*_E{letter}_A1.mat --
    E1=Exercise A, E2=Exercise B, E3=Exercise C. The original notebook's `exercise: str = "B"`
    config field was a bug (it built a glob for "S*_EB_A1.mat", which never matches); fixed here.

Same memory-lean pattern as run_project2_lean.py: one subject at a time, incremental CSV
writes, explicit gc.collect(), and skip-what's-already-done resume logic.
"""
import gc
import os
import glob
import time
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, sosfiltfilt, iirnotch, tf2sos

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from lightgbm import LGBMClassifier

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

RNG_SEED = 0
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)
torch.set_num_threads(2)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(OUT_DIR, "data", "ninapro_db5")
RESULTS_CSV = os.path.join(OUT_DIR, "project1_results.csv")
LOSO_CSV = os.path.join(OUT_DIR, "project1_loso_results.csv")
PROGRESS_LOG = os.path.join(OUT_DIR, "project1_progress.log")

FS = 200
N_CHANNELS = 16
BANDPASS = (20.0, 95.0)   # DB5 Nyquist is 100 Hz -- DB2's 20-450 Hz range doesn't apply
NOTCH_FREQ = 50.0
WINDOW_MS = 200.0
INCREMENT_MS = 100.0
WINDOW_LEN = int(FS * WINDOW_MS / 1000)      # 40 samples
INCREMENT_LEN = int(FS * INCREMENT_MS / 1000)  # 20 samples
AR_ORDER = 4
N_CLASSES_SUBSET = 9   # + rest = 10 classes, matches the notebook's spec
TRAIN_REPS = (1, 3, 4, 6)
TEST_REPS = (2, 5)
EXERCISE_FILE_NUM = 2  # Exercise B (17 movements) = E2 in Ninapro's numeric file naming


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def append_csv(path, rows):
    df = pd.DataFrame(rows)
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)


def score(y_true, y_pred):
    return {"accuracy": accuracy_score(y_true, y_pred), "macro_f1": f1_score(y_true, y_pred, average="macro")}


# ---------------- data loading ----------------

def list_subject_files():
    pattern = os.path.join(DATA_DIR, "s*", f"S*_E{EXERCISE_FILE_NUM}_A1.mat")
    return sorted(glob.glob(pattern))


def load_subject(path):
    m = sio.loadmat(path)
    emg = m["emg"].astype(np.float32)
    restimulus = m["restimulus"].astype(np.int64).ravel()
    rerepetition = m["rerepetition"].astype(np.int64).ravel()
    subj_name = os.path.basename(path).split("_")[0]
    return emg, restimulus, rerepetition, subj_name


def subset_classes(restimulus, n_classes):
    labels, counts = np.unique(restimulus, return_counts=True)
    non_rest = sorted([(l, c) for l, c in zip(labels, counts) if l != 0], key=lambda lc: -lc[1])
    keep = [0] + [l for l, _ in non_rest[:n_classes]]
    mask = np.isin(restimulus, keep)
    remap = {old: new for new, old in enumerate(keep)}
    remapped = np.array([remap[v] for v in restimulus[mask]])
    return mask, remapped


# ---------------- preprocessing ----------------

def filter_emg(emg):
    bp = butter(4, list(BANDPASS), btype="bandpass", fs=FS, output="sos")
    nt = tf2sos(*iirnotch(NOTCH_FREQ, 30.0, FS))
    out = sosfiltfilt(bp, emg, axis=0)
    out = sosfiltfilt(nt, out, axis=0)
    return out.astype(np.float32)


# ---------------- windowing ----------------

def make_windows(emg, labels, repetition):
    win, inc = WINDOW_LEN, INCREMENT_LEN
    n = emg.shape[0]
    starts = np.arange(0, n - win + 1, inc)
    X, y, rep = [], [], []
    for s in starts:
        e = s + win
        seg = labels[s:e]
        if seg[0] != seg[-1]:
            continue
        X.append(emg[s:e])
        y.append(seg[win // 2])
        rep.append(repetition[s:e][win // 2])
    return np.stack(X).astype(np.float32), np.array(y), np.array(rep)


# ---------------- features ----------------

def feat_mav(x): return np.mean(np.abs(x), axis=0)
def feat_wl(x):  return np.sum(np.abs(np.diff(x, axis=0)), axis=0)
def feat_rms(x): return np.sqrt(np.mean(x ** 2, axis=0))


def feat_zc(x, th=1e-3):
    sc = (x[:-1] * x[1:]) < 0
    ok = np.abs(x[:-1] - x[1:]) > th
    return np.sum(sc & ok, axis=0)


def feat_ssc(x, th=1e-3):
    d1 = np.diff(x[:-1], axis=0)
    d2 = np.diff(x[1:], axis=0)
    return np.sum(((d1 * d2) < 0) & (np.abs(d1 - d2) > th), axis=0)


def feat_wamp(x, th=1e-2):
    return np.sum(np.abs(np.diff(x, axis=0)) > th, axis=0)


def feat_ar(x, order=AR_ORDER):
    C, out = x.shape[1], []
    for c in range(C):
        sig = x[:, c] - x[:, c].mean()
        r = np.correlate(sig, sig, mode="full")[len(sig) - 1:]
        r = r[: order + 1] / len(sig)
        R = np.array([[r[abs(i - j)] for j in range(order)] for i in range(order)])
        rhs = r[1: order + 1]
        try:
            coeffs = np.linalg.solve(R + 1e-8 * np.eye(order), rhs)
        except np.linalg.LinAlgError:
            coeffs = np.zeros(order)
        out.append(coeffs)
    return np.concatenate(out)


def extract_features(window):
    feats = [feat_mav(window), feat_wl(window), feat_zc(window), feat_ssc(window),
              feat_rms(window), feat_wamp(window)]
    return np.concatenate([np.concatenate(feats), feat_ar(window)])


def extract_feature_matrix(X_windows):
    return np.stack([extract_features(w) for w in X_windows]).astype(np.float32)


# ---------------- splits ----------------

def split_within_subject(rep):
    return np.isin(rep, TRAIN_REPS), np.isin(rep, TEST_REPS)


def split_random_shuffle(y, test_frac=0.3, rng=None):
    rng = rng or np.random.RandomState(RNG_SEED)
    n = len(y)
    idx = rng.permutation(n)
    n_test = int(n * test_frac)
    test_mask = np.zeros(n, dtype=bool)
    test_mask[idx[:n_test]] = True
    return ~test_mask, test_mask


# ---------------- models ----------------

class EMG1DCNN(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(64, n_classes)

    def forward(self, x):
        return self.head(self.net(x).squeeze(-1))


class WindowDataset(Dataset):
    def __init__(self, X_windows, y):
        self.X = torch.tensor(X_windows, dtype=torch.float32).permute(0, 2, 1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def fit_cnn(X_train, y_train, n_classes, epochs=15, batch_size=64, lr=1e-3):
    model = EMG1DCNN(N_CHANNELS, n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(WindowDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def predict_cnn(model, X_windows, batch_size=256):
    model.eval()
    loader = DataLoader(WindowDataset(X_windows, np.zeros(len(X_windows))), batch_size=batch_size)
    preds = []
    for xb, _ in loader:
        preds.append(model(xb).argmax(dim=1).numpy())
    return np.concatenate(preds)


def evaluate_split(X_train_w, y_train, X_test_w, y_test, n_classes, model_names):
    results = {}
    F_train = extract_feature_matrix(X_train_w)
    F_test = extract_feature_matrix(X_test_w)
    scaler = StandardScaler().fit(F_train)
    F_train_s, F_test_s = scaler.transform(F_train), scaler.transform(F_test)

    if "lda" in model_names:
        clf = LinearDiscriminantAnalysis().fit(F_train_s, y_train)
        results["lda"] = score(y_test, clf.predict(F_test_s))

    if "svm" in model_names:
        clf = LinearSVC(C=1.0, max_iter=5000).fit(F_train_s, y_train)
        results["svm"] = score(y_test, clf.predict(F_test_s))

    if "lgbm" in model_names:
        clf = LGBMClassifier(n_estimators=200, num_leaves=31, objective="multiclass",
                              num_class=n_classes, random_state=RNG_SEED, verbosity=-1)
        clf.fit(F_train_s, y_train)
        results["lgbm"] = score(y_test, clf.predict(F_test_s))

    if "cnn" in model_names:
        mu = X_train_w.mean(axis=(0, 1), keepdims=True)
        sd = X_train_w.std(axis=(0, 1), keepdims=True) + 1e-8
        model = fit_cnn((X_train_w - mu) / sd, y_train, n_classes)
        pred = predict_cnn(model, (X_test_w - mu) / sd)
        results["cnn"] = score(y_test, pred)
        del model

    del F_train, F_test, F_train_s, F_test_s
    return results


# ---------------- main ----------------

def main():
    open(PROGRESS_LOG, "a", encoding="utf-8").close()
    log("starting lean Project 1 (Ninapro DB5) run")

    files = list_subject_files()
    log(f"found {len(files)} subject files: {[os.path.basename(f) for f in files]}")
    if not files:
        log("no files found -- check DATA_DIR / EXERCISE_FILE_NUM")
        return

    n_classes = N_CLASSES_SUBSET + 1

    done_subjects = set()
    if os.path.exists(RESULTS_CSV):
        prev = pd.read_csv(RESULTS_CSV)
        needed = {(p, m) for p in ("within_subject", "random_shuffle_leaky")
                  for m in ("lda", "svm", "lgbm", "cnn")}
        for subj, g in prev.groupby("subject"):
            if needed.issubset(set(zip(g["protocol"], g["model"]))):
                done_subjects.add(subj)
        log(f"resume: {len(done_subjects)} subjects already fully done: {sorted(done_subjects)}")

    feat_cache = {}  # subject -> (feature_matrix, labels) -- tiny (~8600 x 160 floats, ~5MB/subject),
                      # computed once here and reused for LOSO so LOSO never touches raw windows again

    for path in files:
        t_subj = time.time()
        emg, restim, rerep, subj = load_subject(path)
        mask, y_sub = subset_classes(restim, N_CLASSES_SUBSET)
        emg_sub, rep_sub = emg[mask], rerep[mask]
        emg_f = filter_emg(emg_sub)
        Xw, yw, repw = make_windows(emg_f, y_sub, rep_sub)
        log(f"{subj}: loaded+windowed {Xw.shape} in {time.time()-t_subj:.1f}s")

        # cache extracted features for LOSO (tiny), then raw windows can be freed after this subject
        feat_cache[subj] = (extract_feature_matrix(Xw), yw)

        if subj in done_subjects:
            log(f"{subj}: baselines already complete, skipping (cached for LOSO)")
            del emg, restim, rerep, emg_sub, rep_sub, emg_f
            gc.collect()
            continue

        tr_mask, te_mask = split_within_subject(repw)
        if tr_mask.sum() > 0 and te_mask.sum() > 0:
            t0 = time.time()
            res = evaluate_split(Xw[tr_mask], yw[tr_mask], Xw[te_mask], yw[te_mask], n_classes,
                                  ("lda", "svm", "lgbm", "cnn"))
            for model, m in res.items():
                append_csv(RESULTS_CSV, [{"subject": subj, "protocol": "within_subject",
                                           "model": model, **m}])
            log(f"{subj}: within_subject done in {time.time()-t0:.1f}s -> {res}")

        t0 = time.time()
        tr_mask, te_mask = split_random_shuffle(yw)
        res = evaluate_split(Xw[tr_mask], yw[tr_mask], Xw[te_mask], yw[te_mask], n_classes,
                              ("lda", "svm", "lgbm", "cnn"))
        for model, m in res.items():
            append_csv(RESULTS_CSV, [{"subject": subj, "protocol": "random_shuffle_leaky",
                                       "model": model, **m}])
        log(f"{subj}: random_shuffle_leaky done in {time.time()-t0:.1f}s -> {res}")

        del emg, restim, rerep, emg_sub, rep_sub, emg_f, Xw, yw
        gc.collect()
        log(f"{subj}: TOTAL {time.time()-t_subj:.1f}s")

    # ---- LOSO on cheap models only (lda, svm), using cached windows ----
    log("starting LOSO (lda, svm only)")
    done_loso = set()
    if os.path.exists(LOSO_CSV):
        done_loso = set(pd.read_csv(LOSO_CSV)["subject"].unique())
        log(f"resume: LOSO already done for {sorted(done_loso)}")

    all_subj = list(feat_cache.keys())
    for held_out in all_subj:
        if held_out in done_loso:
            log(f"LOSO {held_out} already done, skipping")
            continue
        t0 = time.time()
        F_train = np.concatenate([feat_cache[s][0] for s in all_subj if s != held_out])
        y_train = np.concatenate([feat_cache[s][1] for s in all_subj if s != held_out])
        F_test, y_test = feat_cache[held_out]

        scaler = StandardScaler().fit(F_train)
        F_train_s, F_test_s = scaler.transform(F_train), scaler.transform(F_test)
        res = {}
        lda = LinearDiscriminantAnalysis().fit(F_train_s, y_train)
        res["lda"] = score(y_test, lda.predict(F_test_s))
        svm = LinearSVC(C=1.0, max_iter=5000).fit(F_train_s, y_train)
        res["svm"] = score(y_test, svm.predict(F_test_s))

        for model, m in res.items():
            append_csv(LOSO_CSV, [{"subject": held_out, "protocol": "loso", "model": model, **m}])
        log(f"LOSO {held_out} done in {time.time()-t0:.1f}s -> {res}")
        del F_train, y_train, F_train_s, F_test_s, scaler, lda, svm
        gc.collect()

    log("ALL DONE")


if __name__ == "__main__":
    main()
