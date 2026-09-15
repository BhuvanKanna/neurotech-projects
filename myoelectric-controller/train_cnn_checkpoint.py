"""
Trains and saves the Project 1 CNN checkpoint (+ its eval data) that Projects 4 and 5
both depend on but that run_project1_lean.py never persisted (it evaluated and discarded
each subject's CNN in memory). Reuses run_project1_lean.py's exact data pipeline so the
checkpoint is trained on the identical preprocessing/windowing/split as the real Project 1
results already committed.

Picks subject S1 (arbitrary but fixed choice, documented here): it has full within-subject
results in project1_results.csv, and is one of the higher-accuracy subjects, giving a more
useful checkpoint for the Project 4 voting analysis (voting matters more when it has room to
help) and Project 5 quantization sweep (a checkpoint too close to chance makes the whole
Pareto analysis uninformative).

Saves to myoelectric-controller/checkpoints/:
  - cnn_S1.pt              -- model state_dict
  - cnn_S1_norm.npz        -- mu, sd used for input normalization (must match at inference)
  - cnn_S1_windows.npz     -- ALL windows for S1 (X, y, repetition) so Project 4 can regroup
                               them by repetition, and Project 5 can build train/test/calib splits
                               identically to how the CNN itself was trained/evaluated.
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_project1_lean as p1

SUBJECT = "S1"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    files = p1.list_subject_files()
    path = next(f for f in files if os.path.basename(f).startswith(f"{SUBJECT}_"))
    print(f"loading {path}")

    emg, restim, rerep, subj = p1.load_subject(path)
    assert subj == SUBJECT
    mask, y_sub = p1.subset_classes(restim, p1.N_CLASSES_SUBSET)
    emg_sub, rep_sub = emg[mask], rerep[mask]
    emg_f = p1.filter_emg(emg_sub)
    Xw, yw, repw = p1.make_windows(emg_f, y_sub, rep_sub)
    print(f"{subj}: {Xw.shape} windows, classes {sorted(set(yw.tolist()))}, reps {sorted(set(repw.tolist()))}")

    tr_mask, te_mask = p1.split_within_subject(repw)
    n_classes = p1.N_CLASSES_SUBSET + 1

    mu = Xw[tr_mask].mean(axis=(0, 1), keepdims=True)
    sd = Xw[tr_mask].std(axis=(0, 1), keepdims=True) + 1e-8
    Xtr_norm = (Xw[tr_mask] - mu) / sd
    Xte_norm = (Xw[te_mask] - mu) / sd

    print("training CNN...")
    model = p1.fit_cnn(Xtr_norm, yw[tr_mask], n_classes, epochs=15)

    from sklearn.metrics import accuracy_score, f1_score
    pred = p1.predict_cnn(model, Xte_norm)
    acc = accuracy_score(yw[te_mask], pred)
    f1 = f1_score(yw[te_mask], pred, average="macro")
    print(f"held-out within_subject eval: accuracy={acc:.4f} macro_f1={f1:.4f}")

    torch.save(model.state_dict(), os.path.join(OUT_DIR, f"cnn_{SUBJECT}.pt"))
    np.savez(os.path.join(OUT_DIR, f"cnn_{SUBJECT}_norm.npz"), mu=mu, sd=sd)
    np.savez(os.path.join(OUT_DIR, f"cnn_{SUBJECT}_windows.npz"),
              X=Xw, y=yw, rep=repw, train_mask=tr_mask, test_mask=te_mask,
              n_classes=n_classes, n_channels=p1.N_CHANNELS)
    print(f"saved checkpoint + windows to {OUT_DIR}")


if __name__ == "__main__":
    main()
