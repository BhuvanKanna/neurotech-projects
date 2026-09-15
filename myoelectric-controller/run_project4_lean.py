"""
Project 4 -- Temporal Decision Aggregation, executed against the Project 1 Ninapro DB5
CNN checkpoint (subject S1, see train_cnn_checkpoint.py).

Real data constraint worth flagging up front: the notebook's spec sweeps k = 1,3,5,7,9 for
BOTH experiments. That's fine for experiment (a) -- a single 5s repetition holds ~45 windows
at a 100ms hop, plenty for k=9. It's NOT fine for experiment (b): Ninapro (DB2 and DB5 alike)
records only 6 repetitions per movement, so "k repetitions of the same gesture" tops out at
k=6. This script caps experiment (b)'s k grid at (1, 3, 5) rather than silently reusing fewer
repetitions than requested or padding with something synthetic.
"""
import os
import sys
import itertools
import numpy as np
import pandas as pd
import torch
from scipy.special import comb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_project1_lean as p1

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SUBJECT = "S1"

CONSECUTIVE_KS = (1, 3, 5, 7, 9)
ACROSS_REP_KS = (1, 3, 5)   # capped at 5 -- see module docstring


def load_checkpoint():
    windows = np.load(os.path.join(CKPT_DIR, f"cnn_{SUBJECT}_windows.npz"))
    norm = np.load(os.path.join(CKPT_DIR, f"cnn_{SUBJECT}_norm.npz"))
    n_classes = int(windows["n_classes"])
    model = p1.EMG1DCNN(int(windows["n_channels"]), n_classes)
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, f"cnn_{SUBJECT}.pt")))
    model.eval()
    return model, norm["mu"], norm["sd"], windows


@torch.no_grad()
def predict_proba(model, X_norm, batch_size=256):
    loader_X = torch.tensor(X_norm, dtype=torch.float32).permute(0, 2, 1)
    probs = []
    for i in range(0, len(loader_X), batch_size):
        out = model(loader_X[i:i + batch_size])
        probs.append(torch.softmax(out, dim=1).numpy())
    return np.concatenate(probs)


def find_segments(y, rep):
    """Run-length encode consecutive windows sharing the same (y, rep) -- each run is one
    continuous repetition-of-a-gesture (or one continuous rest stretch)."""
    segments = []
    start = 0
    for i in range(1, len(y) + 1):
        if i == len(y) or y[i] != y[start] or rep[i] != rep[start]:
            segments.append((start, i, y[start], rep[start]))
            start = i
    return segments


# ---------------- voting rules ----------------

def majority_vote(preds):
    vals, counts = np.unique(preds, return_counts=True)
    return vals[np.argmax(counts)]

def confidence_weighted_vote(probs):
    return probs.sum(axis=0).argmax()

def ewma_vote(probs, alpha=0.3):
    e = probs[0]
    for p in probs[1:]:
        e = alpha * p + (1 - alpha) * e
    return e.argmax()

def rejection_vote(probs, floor=0.5):
    mean_probs = probs.mean(axis=0)
    top = mean_probs.argmax()
    return top if mean_probs[top] >= floor else -1

def binomial_ceiling(p, k):
    thresh = k // 2
    return sum(comb(k, i) * (p ** i) * ((1 - p) ** (k - i)) for i in range(thresh + 1, k + 1))

def wolpaw_bits_per_trial(p, n_classes):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log2(n_classes) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / max(n_classes - 1, 1))

def itr_bits_per_min(p, n_classes, decision_interval_ms):
    return max(wolpaw_bits_per_trial(p, n_classes), 0.0) * (60_000.0 / decision_interval_ms)


def main():
    model, mu, sd, w = load_checkpoint()
    X, y, rep = w["X"], w["y"], w["rep"]
    test_mask = w["test_mask"]
    n_classes = int(w["n_classes"])

    X_test, y_test, rep_test = X[test_mask], y[test_mask], rep[test_mask]
    probs_test = predict_proba(model, (X_test - mu) / sd)
    preds_test = probs_test.argmax(axis=1)

    segments = find_segments(y_test, rep_test)
    gesture_segments = [(s, e, lab, r) for s, e, lab, r in segments if lab != 0 and (e - s) >= 9]
    print(f"{len(segments)} total segments in test set, {len(gesture_segments)} non-rest "
          f"gesture segments with >=9 windows (usable for consecutive-window voting)")

    # ---- experiment (a): consecutive-window voting within one repetition ----
    consec_rows = []
    for s, e, lab, r in gesture_segments:
        seg_preds, seg_probs = preds_test[s:e], probs_test[s:e]
        n = e - s
        for k in CONSECUTIVE_KS:
            for start in range(0, n - k + 1):
                idx = slice(start, start + k)
                consec_rows.append({
                    "k": k, "true": lab,
                    "majority": majority_vote(seg_preds[idx]),
                    "conf_weighted": confidence_weighted_vote(seg_probs[idx]),
                    "ewma": ewma_vote(seg_probs[idx]),
                    "rejection": rejection_vote(seg_probs[idx]),
                })
    consec_df = pd.DataFrame(consec_rows)
    consec_df.to_csv(os.path.join(OUT_DIR, "project4_consecutive_results.csv"), index=False)
    print(f"experiment (a): {len(consec_df)} voting-window rows across {len(gesture_segments)} segments")

    # ---- experiment (b): across-repetition voting ----
    # Test-only data respects the train/test split but only has 2 repetitions/movement
    # (TEST_REPS=(2,5)), so k=3 and k=5 have zero eligible groups -- that's a real finding
    # about this dataset's structure, not a script bug, and is reported as such below.
    # A second pass reuses ALL 6 repetitions (including the 4 the CNN trained on) purely to
    # get real k=3/k=5 numbers -- clearly labeled, with the train-leakage caveat that reps
    # 1/3/4/6 were seen during training so per-window accuracy there is inflated relative to
    # the genuinely held-out reps 2/5.
    def run_across_repetition(preds, probs, y_arr, rep_arr, label):
        segs = find_segments(y_arr, rep_arr)
        segs = [(s, e, lab, r) for s, e, lab, r in segs if lab != 0 and (e - s) >= 9]
        by_class = {}
        for s, e, lab, r in segs:
            by_class.setdefault(lab, []).append({
                "true": lab, "pred": majority_vote(preds[s:e]), "prob": probs[s:e].mean(axis=0),
            })
        rows = []
        for lab, entries in by_class.items():
            n_entries = len(entries)
            for k in ACROSS_REP_KS:
                if k > n_entries:
                    continue
                for combo in itertools.combinations(range(n_entries), k):
                    sel = [entries[i] for i in combo]
                    preds_k = np.array([e["pred"] for e in sel])
                    probs_k = np.stack([e["prob"] for e in sel])
                    rows.append({"k": k, "true": lab, "majority": majority_vote(preds_k),
                                 "conf_weighted": confidence_weighted_vote(probs_k)})
        df = pd.DataFrame(rows)
        n_instances = sum(len(v) for v in by_class.values())
        print(f"experiment (b) [{label}]: {len(df)} voting-group rows ({n_instances} "
              f"gesture-repetition instances across {len(by_class)} classes)")
        return df

    across_df = run_across_repetition(preds_test, probs_test, y_test, rep_test, "test-only, reps 2&5")

    probs_all = predict_proba(model, (X - mu) / sd)
    preds_all = probs_all.argmax(axis=1)
    across_df_allreps = run_across_repetition(preds_all, probs_all, y, rep,
                                                "ALL reps 1-6, includes train-seen data")

    across_df.to_csv(os.path.join(OUT_DIR, "project4_across_repetition_results.csv"), index=False)
    across_df_allreps.to_csv(os.path.join(OUT_DIR, "project4_across_repetition_allreps_results.csv"), index=False)

    # ---- empirical vs binomial ceiling ----
    base_p = float((consec_df.query("k == 1")["majority"] == consec_df.query("k == 1")["true"]).mean())
    print(f"base per-window accuracy (k=1): {base_p:.4f}")

    def summarize(df, ks):
        rows = []
        for k in ks:
            g = df[df["k"] == k]
            emp = (g["majority"] == g["true"]).mean()
            theo = binomial_ceiling(base_p, k) if k % 2 == 1 else np.nan
            rows.append({"k": k, "empirical": emp, "theoretical": theo, "n": len(g)})
        return pd.DataFrame(rows)

    consec_summary = summarize(consec_df, CONSECUTIVE_KS)
    across_summary = summarize(across_df, ACROSS_REP_KS)
    across_summary_allreps = summarize(across_df_allreps, ACROSS_REP_KS)
    consec_summary.to_csv(os.path.join(OUT_DIR, "project4_consecutive_summary.csv"), index=False)
    across_summary.to_csv(os.path.join(OUT_DIR, "project4_across_repetition_summary.csv"), index=False)
    across_summary_allreps.to_csv(os.path.join(OUT_DIR, "project4_across_repetition_allreps_summary.csv"), index=False)
    print("\nconsecutive-window (correlated errors):")
    print(consec_summary.to_string(index=False))
    print("\nacross-repetition, test-only reps 2&5 (near-independent errors, but only 2 reps/class):")
    print(across_summary.to_string(index=False))
    print("\nacross-repetition, ALL reps 1-6 (real k=3/5 numbers, but reps 1/3/4/6 were CNN-training data):")
    print(across_summary_allreps.to_string(index=False))

    # ---- Q-statistic (adjacent-window error correlation, consecutive case) ----
    def q_statistic(a, b):
        n11 = np.sum(a & b); n10 = np.sum(a & ~b); n01 = np.sum(~a & b); n00 = np.sum(~a & ~b)
        denom = n11 * n00 + n10 * n01
        return 0.0 if denom == 0 else (n11 * n00 - n10 * n01) / denom

    qs = []
    for s, e, lab, r in gesture_segments:
        correct = (preds_test[s:e] == lab)
        if len(correct) >= 2:
            qs.append(q_statistic(correct[:-1], correct[1:]))
    mean_q = float(np.nanmean(qs))
    print(f"\nmean adjacent-window Q-statistic (consecutive, correlated): {mean_q:.4f}")

    # ---- latency pricing + ITR ----
    consec_summary["latency_ms"] = (consec_summary["k"] - 1) * p1.INCREMENT_MS
    consec_summary["decision_interval_ms"] = consec_summary["k"] * p1.INCREMENT_MS
    consec_summary["itr_bpm"] = consec_summary.apply(
        lambda r: itr_bits_per_min(r["empirical"], n_classes, r["decision_interval_ms"]), axis=1)
    best_k = int(consec_summary.loc[consec_summary["itr_bpm"].idxmax(), "k"])
    print(f"\nk maximizing ITR (consecutive-window): {best_k}")
    consec_summary.to_csv(os.path.join(OUT_DIR, "project4_consecutive_summary.csv"), index=False)
    print(consec_summary[["k", "empirical", "latency_ms", "itr_bpm"]].to_string(index=False))

    # ---- voting-rule comparison at a representative k ----
    rule_rows = []
    for k in CONSECUTIVE_KS:
        g = consec_df[consec_df["k"] == k]
        for rule in ("majority", "conf_weighted", "ewma", "rejection"):
            valid = g[g[rule] != -1] if rule == "rejection" else g
            acc = (valid[rule] == valid["true"]).mean() if len(valid) else np.nan
            coverage = len(valid) / len(g) if len(g) else np.nan
            rule_rows.append({"k": k, "rule": rule, "accuracy": acc, "coverage": coverage})
    rule_df = pd.DataFrame(rule_rows)
    rule_df.to_csv(os.path.join(OUT_DIR, "project4_voting_rule_comparison.csv"), index=False)
    print("\nvoting rule comparison (consecutive-window):")
    print(rule_df.to_string(index=False))

    print(f"\nmean Q-statistic: {mean_q:.4f} (Q closer to 1 => errors highly correlated => "
          f"voting gain caps well below the binomial ceiling)")
    print("ALL DONE")


if __name__ == "__main__":
    main()
