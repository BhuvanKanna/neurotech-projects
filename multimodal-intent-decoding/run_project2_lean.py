"""
Memory-lean standalone runner for Project 2 (BCI-IV-2a / BNCI2014_001 via MOABB).

The system this runs on has very little free RAM (~2.7GB of 16GB at idle), so this
avoids the notebook's original approach of holding all 9 subjects' raw epoch arrays
(plus a concatenated duplicate for LOSO) in memory simultaneously. Instead:
  - one subject's raw epochs are in memory at a time,
  - per-trial Riemannian covariances (22x22, tiny) are cached for all subjects for LOSO
    instead of caching raw epochs (22x501, ~50x bigger),
  - results are appended to CSV incrementally so a kill partway through still leaves
    real partial results on disk,
  - float32 throughout instead of float64.
"""
import gc
import os
import sys
import time
import numpy as np
import pandas as pd

import mne
mne.set_log_level("WARNING")

import moabb
moabb.set_log_level("WARNING")
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score
from scipy.signal import butter, sosfiltfilt

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

RNG_SEED = 0
np.random.seed(RNG_SEED)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(OUT_DIR, "project2_results.csv")
LOSO_CSV = os.path.join(OUT_DIR, "project2_loso_results.csv")
DEEP_CSV = os.path.join(OUT_DIR, "project2_deep_results.csv")
PROGRESS_LOG = os.path.join(OUT_DIR, "project2_progress.log")

FBCSP_BANDS = tuple((lo, lo + 4) for lo in range(4, 40, 4))


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


def within_session_split(metadata, session_col="session"):
    rng = np.random.RandomState(RNG_SEED)
    train_mask = np.zeros(len(metadata), dtype=bool)
    for sess, idx in metadata.groupby(session_col).groups.items():
        idx = np.array(idx)
        rng.shuffle(idx)
        n_train = int(0.7 * len(idx))
        train_mask[idx[:n_train]] = True
    return train_mask, ~train_mask


def bandpass_epochs(X, sfreq, lo, hi, order=4):
    sos = butter(order, [lo, hi], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, X, axis=-1).astype(np.float32)


def fbcsp_features(X, y, sfreq, bands, n_components=4, fit_csps=None):
    feats = []
    csps = fit_csps if fit_csps is not None else []
    fitting = fit_csps is None
    for i, (lo, hi) in enumerate(bands):
        Xb = bandpass_epochs(X, sfreq, lo, hi)
        if fitting:
            csp = CSP(n_components=n_components, reg="ledoit_wolf", log=True, norm_trace=False)
            csp.fit(Xb, y)
            csps.append(csp)
        else:
            csp = csps[i]
        feats.append(csp.transform(Xb))
    return np.concatenate(feats, axis=1).astype(np.float32), csps


def load_subject(subject_id, dataset, paradigm):
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject_id])
    return X.astype(np.float32), y, metadata


def main():
    open(PROGRESS_LOG, "a", encoding="utf-8").close()
    log("starting lean Project 2 run")

    dataset = BNCI2014_001()
    paradigm = MotorImagery(fmin=8.0, fmax=30.0, tmin=0.5, tmax=2.5)
    subjects = dataset.subject_list
    log(f"subjects: {subjects}")

    sfreq = 250

    # ---- resume support: figure out what's already done from prior (killed) runs ----
    done_baseline_subjects = set()
    if os.path.exists(RESULTS_CSV):
        prev = pd.read_csv(RESULTS_CSV)
        needed = {("within_session", "csp_lda"), ("within_session", "fbcsp"),
                  ("within_session", "riemannian_ts_logreg"), ("cross_session", "riemannian_ts_logreg")}
        for subj, g in prev.groupby("subject"):
            have = set(zip(g["protocol"], g["model"]))
            if needed.issubset(have):
                done_baseline_subjects.add(subj)
        log(f"resume: {len(done_baseline_subjects)} subjects already have complete baselines: {sorted(done_baseline_subjects)}")
        # drop partial rows for subjects that weren't fully completed, so re-processing
        # them below doesn't create duplicate rows when we append
        clean = prev[prev["subject"].isin(done_baseline_subjects)]
        if len(clean) != len(prev):
            log(f"resume: dropping {len(prev) - len(clean)} partial rows for incomplete subjects, will redo them")
            clean.to_csv(RESULTS_CSV, index=False)

    done_loso_subjects = set()
    if os.path.exists(LOSO_CSV):
        done_loso_subjects = set(pd.read_csv(LOSO_CSV)["subject"].unique())
        log(f"resume: LOSO already done for {sorted(done_loso_subjects)}")

    done_deep = {}
    if os.path.exists(DEEP_CSV):
        prev_deep = pd.read_csv(DEEP_CSV)
        for subj, g in prev_deep.groupby("subject"):
            done_deep[subj] = set(g["model"].unique())
        log(f"resume: deep-model progress so far: { {k: sorted(v) for k, v in done_deep.items()} }")

    # ---- within-session + cross-session baselines, one subject at a time ----
    cov_cache = {}  # subject -> (cov_matrices float32, y, session_array) for LOSO reuse (tiny)

    for subj in subjects:
        t_subj = time.time()
        X, y, metadata = load_subject(subj, dataset, paradigm)
        log(f"subject {subj}: loaded X{X.shape} in {time.time()-t_subj:.1f}s")

        if subj in done_baseline_subjects:
            # baselines already on disk from a prior run — just rebuild the tiny covariance
            # cache entry needed for LOSO below, skip re-fitting classifiers.
            cov = Covariances(estimator="oas").transform(X)
            cov_cache[subj] = (cov.astype(np.float32), y, metadata["subject"].values)
            log(f"subject {subj}: baselines already complete, reused for LOSO cache only")
            del X, y, metadata
            gc.collect()
            continue

        tr_mask, te_mask = within_session_split(metadata)

        # CSP + LDA
        t0 = time.time()
        pipe = Pipeline([("csp", CSP(n_components=6, reg=None, log=True, norm_trace=False)),
                          ("lda", LinearDiscriminantAnalysis())])
        pipe.fit(X[tr_mask], y[tr_mask])
        pred = pipe.predict(X[te_mask])
        append_csv(RESULTS_CSV, [{"subject": subj, "protocol": "within_session", "model": "csp_lda",
                                   **score(y[te_mask], pred)}])
        log(f"subject {subj}: csp_lda done in {time.time()-t0:.1f}s")

        # FBCSP
        t0 = time.time()
        F_train, csps = fbcsp_features(X[tr_mask], y[tr_mask], sfreq, FBCSP_BANDS)
        sel = SelectKBest(mutual_info_classif, k=20).fit(F_train, y[tr_mask])
        clf = LinearDiscriminantAnalysis().fit(sel.transform(F_train), y[tr_mask])
        F_test, _ = fbcsp_features(X[te_mask], None, sfreq, FBCSP_BANDS, fit_csps=csps)
        pred = clf.predict(sel.transform(F_test))
        append_csv(RESULTS_CSV, [{"subject": subj, "protocol": "within_session", "model": "fbcsp",
                                   **score(y[te_mask], pred)}])
        log(f"subject {subj}: fbcsp done in {time.time()-t0:.1f}s")
        del F_train, F_test, csps, sel, clf
        gc.collect()

        # Riemannian tangent space + logreg
        t0 = time.time()
        riem = Pipeline([("cov", Covariances(estimator="oas")), ("ts", TangentSpace(metric="riemann")),
                          ("logreg", LogisticRegression(max_iter=1000))])
        riem.fit(X[tr_mask], y[tr_mask])
        pred = riem.predict(X[te_mask])
        append_csv(RESULTS_CSV, [{"subject": subj, "protocol": "within_session", "model": "riemannian_ts_logreg",
                                   **score(y[te_mask], pred)}])
        log(f"subject {subj}: riemannian within-session done in {time.time()-t0:.1f}s")

        # cross-session (train session 0, test session 1)
        sessions = metadata["session"].unique()
        if len(sessions) >= 2:
            cs_tr = (metadata["session"] == sessions[0]).values
            cs_te = (metadata["session"] == sessions[1]).values
            t0 = time.time()
            riem_cs = Pipeline([("cov", Covariances(estimator="oas")), ("ts", TangentSpace(metric="riemann")),
                                 ("logreg", LogisticRegression(max_iter=1000))])
            riem_cs.fit(X[cs_tr], y[cs_tr])
            pred_cs = riem_cs.predict(X[cs_te])
            append_csv(RESULTS_CSV, [{"subject": subj, "protocol": "cross_session", "model": "riemannian_ts_logreg",
                                       **score(y[cs_te], pred_cs)}])
            log(f"subject {subj}: cross-session done in {time.time()-t0:.1f}s")
            del riem_cs

        # cache tiny covariances (not raw epochs) for the LOSO pass below
        cov = Covariances(estimator="oas").transform(X)  # (n_trials, 22, 22) float32-ish, ~1MB/subject
        cov_cache[subj] = (cov.astype(np.float32), y, metadata["subject"].values)

        del X, y, metadata, pipe, riem, pred
        gc.collect()
        log(f"subject {subj}: TOTAL {time.time()-t_subj:.1f}s, memory freed")

    # ---- cross-subject LOSO on cached covariances only (tiny footprint) ----
    log("starting LOSO (cached covariances, riemannian tangent space)")
    all_subj_ids = list(cov_cache.keys())
    for held_out in all_subj_ids:
        if held_out in done_loso_subjects:
            log(f"LOSO held_out={held_out} already done, skipping")
            continue
        t0 = time.time()
        train_covs = np.concatenate([cov_cache[s][0] for s in all_subj_ids if s != held_out])
        train_y = np.concatenate([cov_cache[s][1] for s in all_subj_ids if s != held_out])
        test_covs, test_y, _ = cov_cache[held_out]

        ts = TangentSpace(metric="riemann").fit(train_covs)
        clf = LogisticRegression(max_iter=1000).fit(ts.transform(train_covs), train_y)
        pred = clf.predict(ts.transform(test_covs))
        append_csv(LOSO_CSV, [{"subject": held_out, "protocol": "cross_subject", "model": "riemannian_ts_logreg",
                                **score(test_y, pred)}])
        log(f"LOSO held_out={held_out} done in {time.time()-t0:.1f}s")
        del train_covs, train_y, ts, clf
        gc.collect()

    del cov_cache
    gc.collect()
    log("LOSO complete, covariance cache freed")

    # ---- deep models, one subject at a time, freshly loaded and discarded ----
    log("starting deep models (EEGNet, ShallowFBCSPNet), one subject at a time")
    import torch
    from braindecode.models import EEGNet, ShallowFBCSPNet
    from braindecode.classifier import EEGClassifier
    from skorch.callbacks import EarlyStopping
    from skorch.dataset import ValidSplit
    from sklearn.preprocessing import LabelEncoder

    torch.manual_seed(RNG_SEED)

    def make_clf(module, max_epochs=50):
        return EEGClassifier(
            module, criterion=torch.nn.CrossEntropyLoss, optimizer=torch.optim.Adam,
            optimizer__lr=1e-3, train_split=ValidSplit(cv=0.2, stratified=True, random_state=RNG_SEED),
            max_epochs=max_epochs, batch_size=32,
            callbacks=[EarlyStopping(patience=15, monitor="valid_loss")], verbose=0,
        )

    for subj in subjects:
        already = done_deep.get(subj, set())
        if {"eegnet", "shallow_fbcsp"}.issubset(already):
            log(f"subject {subj}: deep models already complete, skipping")
            continue

        t_subj = time.time()
        X, y, metadata = load_subject(subj, dataset, paradigm)
        tr_mask, te_mask = within_session_split(metadata)
        le = LabelEncoder().fit(y)
        yi = le.transform(y)
        n_ch, n_times = X.shape[1], X.shape[2]
        n_classes = len(le.classes_)

        for model_name, module in [
            ("eegnet", EEGNet(n_chans=n_ch, n_outputs=n_classes, n_times=n_times)),
            ("shallow_fbcsp", ShallowFBCSPNet(n_chans=n_ch, n_outputs=n_classes, n_times=n_times, final_conv_length="auto")),
        ]:
            if model_name in already:
                log(f"subject {subj}: {model_name} already done, skipping")
                continue
            t0 = time.time()
            clf = make_clf(module)
            clf.fit(X[tr_mask], yi[tr_mask])
            pred = clf.predict(X[te_mask])
            append_csv(DEEP_CSV, [{"subject": subj, "protocol": "within_session", "model": model_name,
                                    **score(yi[te_mask], pred)}])
            log(f"subject {subj}: {model_name} done in {time.time()-t0:.1f}s")
            del clf, module
            gc.collect()

        del X, y, metadata
        gc.collect()
        log(f"subject {subj}: deep models TOTAL {time.time()-t_subj:.1f}s")

    log("ALL DONE")


if __name__ == "__main__":
    main()
