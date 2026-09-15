"""
Project 3 -- EEG+EMG Fusion, executed against real Jeong et al. 2020 data (via MOABB's
Jeong2020 loader, cached from Zenodo/NEMAR). See module docstring at the bottom for the
scope reduction this required and why.

SCOPE: one subject (sub1), one session (session1), one task type (reaching -- 6 directions,
50 trials each = 300 trials), real-movement condition (not motor imagery -- needed for
genuine EMG signal to make the degradation sweep meaningful). The full Jeong et al. spec is
25 subjects x 3 sessions x 3 task types x 2 conditions; a single subject's raw sourcedata for
just ONE of those 18 recordings is ~960MB on disk (9.8GB for all of subject 1 alone -- see
../myoelectric-controller/README.md's discussion of Project 1's memory constraints on this
same machine, which apply here too). Loading and epoching more than one recording's worth
of continuous data at once risked repeating those OOM kills, so this reads directly from the
raw BrainVision file with preload=False and pulls each trial's ~4s window from disk on
demand (see extract_trial()) rather than ever holding the full ~1GB continuous recording in
memory -- 300 trials extracted this way took ~2 seconds and never touched more than a few MB
at once.

Trial structure (decoded from the .vmrk annotations, not documented anywhere machine-
readable): a class cue (Stimulus/S 1..6) precedes a "go" cue (Stimulus/S {class}1, e.g. S41
for class 4) by ~3s, which is followed by a trial-end marker (Stimulus/S 8) ~4s later. Trials
here are epoched around the go cue (true movement onset), not the class cue.
"""
import os
import re
import sys
import copy
import numpy as np
import pandas as pd
import mne

import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from scipy.signal import butter, sosfiltfilt, iirnotch, tf2sos

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

mne.set_log_level("WARNING")
RNG_SEED = 0
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(OUT_DIR, "..", "mne_data", "NEMAR", "nm000311", "sourcedata", "sub1",
                          "session1_sub1_reaching_realMove.vhdr")

EEG_BP = (8.0, 30.0)
EMG_BP = (20.0, 450.0)
EMG_NOTCH = 60.0
EPOCH_TMIN, EPOCH_TMAX = -1.0, 3.0   # seconds relative to the "go" cue
POST_CUE_START = 0.0                  # feature windows for the main classifiers use [0, TMAX]


def load_raw_and_trials():
    raw = mne.io.read_raw_brainvision(DATA_PATH, preload=False)
    sfreq = raw.info["sfreq"]

    eog_names = ["hEOG_L", "hEOG_R", "vEOG_U", "vEOG_D"]
    emg_names = [ch for ch in raw.ch_names if ch.startswith("EMG")]
    eeg_names = [ch for ch in raw.ch_names if ch not in eog_names and ch not in emg_names]
    print(f"{len(eeg_names)} EEG, {len(eog_names)} EOG, {len(emg_names)} EMG channels")

    trials = []
    go_codes = {"Stimulus/S 11": 1, "Stimulus/S 21": 2, "Stimulus/S 31": 3,
                "Stimulus/S 41": 4, "Stimulus/S 51": 5, "Stimulus/S 61": 6}
    for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
        cls = go_codes.get(desc.strip())
        if cls is not None:
            trials.append((float(onset), cls))
    print(f"{len(trials)} trials found (expect 300 = 6 classes x 50)")

    return raw, sfreq, eeg_names, eog_names, emg_names, trials


def extract_trial(raw, sfreq, onset, tmin=EPOCH_TMIN, tmax=EPOCH_TMAX):
    """Reads only this trial's ~4s window from disk -- raw stays preload=False throughout,
    so the full ~1GB continuous recording is never loaded into memory."""
    s = int(round((onset + tmin) * sfreq))
    e = int(round((onset + tmax) * sfreq))
    return raw.get_data(start=max(s, 0), stop=e).astype(np.float32)  # (n_channels_total, T)


def sos_bandpass(fs, lo, hi, order=4):
    return butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")

def sos_notch(fs, freq, q=30.0):
    return tf2sos(*iirnotch(freq, q, fs))


def regress_out_eog(eeg, eog):
    """Simplified EOG-contamination control: per-channel linear regression of EEG onto EOG,
    subtract the fitted projection. A lighter substitute for full ICA -- ICA needs many
    trials concatenated to fit reliably and was not practical within this session's time
    budget on top of everything else; this at minimum removes the linear (blink/saccade)
    component of EOG leakage into EEG, which is the dominant artifact source."""
    X = eog.T  # (T, 4)
    beta, *_ = np.linalg.lstsq(np.column_stack([X, np.ones(len(X))]), eeg.T, rcond=None)
    fitted = np.column_stack([X, np.ones(len(X))]) @ beta
    return (eeg.T - fitted).T.astype(np.float32)


def _mav(x): return np.mean(np.abs(x), axis=0)
def _wl(x):  return np.sum(np.abs(np.diff(x, axis=0)), axis=0)
def _rms(x): return np.sqrt(np.mean(x ** 2, axis=0))
def _zc(x, th=1e-4):
    sc = (x[:-1] * x[1:]) < 0
    ok = np.abs(x[:-1] - x[1:]) > th
    return np.sum(sc & ok, axis=0)

def emg_hudgins_features(emg_window):
    """emg_window: (C, T) -> flat feature vector."""
    x = emg_window.T  # (T, C)
    return np.concatenate([_mav(x), _wl(x), _rms(x), _zc(x)])


def score(y_true, y_pred):
    return {"accuracy": accuracy_score(y_true, y_pred), "macro_f1": f1_score(y_true, y_pred, average="macro")}


class TwoBranchGatedNet(nn.Module):
    def __init__(self, eeg_dim, emg_dim, n_classes, hidden=32):
        super().__init__()
        self.eeg_enc = nn.Sequential(nn.Linear(eeg_dim, hidden), nn.ReLU())
        self.emg_enc = nn.Sequential(nn.Linear(emg_dim, hidden), nn.ReLU())
        self.gate = nn.Sequential(nn.Linear(hidden * 2, 1), nn.Sigmoid())
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, eeg_x, emg_x):
        e, m = self.eeg_enc(eeg_x), self.emg_enc(emg_x)
        g = self.gate(torch.cat([e, m], dim=1))
        return self.head(g * e + (1 - g) * m), g.squeeze(-1)


def fit_gated_fusion(F_eeg, F_emg, y, n_classes, epochs=150, lr=1e-3):
    model = TwoBranchGatedNet(F_eeg.shape[1], F_emg.shape[1], n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    xe, xm = torch.tensor(F_eeg, dtype=torch.float32), torch.tensor(F_emg, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        out, _ = model(xe, xm)
        loss_fn(out, yt).backward()
        opt.step()
    return model


def main():
    raw, sfreq, eeg_names, eog_names, emg_names, trials = load_raw_and_trials()
    eeg_idx = mne.pick_channels(raw.ch_names, eeg_names, ordered=True)
    eog_idx = mne.pick_channels(raw.ch_names, eog_names, ordered=True)
    emg_idx = mne.pick_channels(raw.ch_names, emg_names, ordered=True)

    eeg_bp_sos = sos_bandpass(sfreq, *EEG_BP)
    emg_bp_sos = sos_bandpass(sfreq, *EMG_BP)
    emg_notch_sos = sos_notch(sfreq, EMG_NOTCH)

    eeg_raw_full, emg_raw_full, y_all = [], [], []
    for onset, cls in trials:
        window = extract_trial(raw, sfreq, onset)
        eeg_raw_full.append(window[eeg_idx])
        emg_raw_full.append(window[emg_idx])
        y_all.append(cls)
    y_all = np.array(y_all)
    print(f"extracted {len(y_all)} trials, {eeg_raw_full[0].shape[1]} samples/trial "
          f"(={EPOCH_TMAX - EPOCH_TMIN:.0f}s at {sfreq:.0f}Hz)")

    le = LabelEncoder().fit(y_all)
    y = le.transform(y_all)
    n_classes = len(le.classes_)

    idx_train, idx_test = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y,
                                            random_state=RNG_SEED)

    def preprocess_and_featurize(eeg_full, emg_full, eog_full_trials=None):
        post_start = int(round((POST_CUE_START - EPOCH_TMIN) * sfreq))
        F_eeg_cov, F_emg = [], []
        for i, (eeg, emg) in enumerate(zip(eeg_full, emg_full)):
            eeg_f = sosfiltfilt(eeg_bp_sos, eeg, axis=1)
            if eog_full_trials is not None:
                eeg_f = regress_out_eog(eeg_f, eog_full_trials[i])
            emg_f = sosfiltfilt(emg_bp_sos, emg, axis=1)
            emg_f = sosfiltfilt(emg_notch_sos, emg_f, axis=1)
            F_eeg_cov.append(eeg_f[:, post_start:])
            F_emg.append(emg_hudgins_features(emg_f[:, post_start:]))
        return np.stack(F_eeg_cov), np.stack(F_emg)

    print("preprocessing + featurizing (post-cue window)...")
    eog_raw_full = [extract_trial(raw, sfreq, onset)[eog_idx] for onset, _ in trials]
    eeg_epochs_f, F_emg_all = preprocess_and_featurize(eeg_raw_full, emg_raw_full, eog_raw_full)

    cov = Covariances(estimator="oas").fit(eeg_epochs_f[idx_train])
    covs_all = cov.transform(eeg_epochs_f)
    ts = TangentSpace(metric="riemann").fit(covs_all[idx_train])
    F_eeg_all = ts.transform(covs_all)

    emg_scaler = StandardScaler().fit(F_emg_all[idx_train])
    F_emg_s = emg_scaler.transform(F_emg_all).astype(np.float32)
    eeg_scaler = StandardScaler().fit(F_eeg_all[idx_train])
    F_eeg_s = eeg_scaler.transform(F_eeg_all).astype(np.float32)

    fusion_results = []
    emg_clf = LogisticRegression(max_iter=2000).fit(F_emg_s[idx_train], y[idx_train])
    proba_emg = emg_clf.predict_proba(F_emg_s[idx_test])
    fusion_results.append({"model": "emg_only", **score(y[idx_test], proba_emg.argmax(1))})

    eeg_clf = LogisticRegression(max_iter=2000).fit(F_eeg_s[idx_train], y[idx_train])
    proba_eeg = eeg_clf.predict_proba(F_eeg_s[idx_test])
    fusion_results.append({"model": "eeg_only", **score(y[idx_test], proba_eeg.argmax(1))})

    F_concat = np.concatenate([F_eeg_s, F_emg_s], axis=1)
    concat_clf = LogisticRegression(max_iter=2000).fit(F_concat[idx_train], y[idx_train])
    fusion_results.append({"model": "fusion_feature_concat",
                            **score(y[idx_test], concat_clf.predict(F_concat[idx_test]))})

    proba_avg = (proba_eeg + proba_emg) / 2
    fusion_results.append({"model": "fusion_decision_avg", **score(y[idx_test], proba_avg.argmax(1))})

    gated = fit_gated_fusion(F_eeg_s[idx_train], F_emg_s[idx_train], y[idx_train], n_classes)
    gated.eval()
    with torch.no_grad():
        out, gate_vals = gated(torch.tensor(F_eeg_s[idx_test]), torch.tensor(F_emg_s[idx_test]))
        pred_gated = out.argmax(dim=1).numpy()
    fusion_results.append({"model": "fusion_intermediate_gated", **score(y[idx_test], pred_gated)})
    print(f"mean gate weight on EEG branch (clean data): {gate_vals.mean().item():.3f}")

    fusion_df = pd.DataFrame(fusion_results)
    fusion_df.to_csv(os.path.join(OUT_DIR, "project3_fusion_results.csv"), index=False)
    print("\n=== fusion comparison (clean data) ===")
    print(fusion_df.to_string(index=False))

    # ---- degradation sweep on EMG ----
    print("\n=== EMG degradation sweep ===")
    def degrade_and_eval(degrade_fn, values, param_name):
        rows = []
        for v in values:
            emg_test_deg = [degrade_fn(emg, v) for emg in [emg_raw_full[i] for i in idx_test]]
            F_emg_deg = []
            for emg in emg_test_deg:
                emg_f = sosfiltfilt(emg_bp_sos, emg, axis=1)
                emg_f = sosfiltfilt(emg_notch_sos, emg_f, axis=1)
                post_start = int(round((POST_CUE_START - EPOCH_TMIN) * sfreq))
                F_emg_deg.append(emg_hudgins_features(emg_f[:, post_start:]))
            F_emg_deg_s = emg_scaler.transform(np.stack(F_emg_deg)).astype(np.float32)
            proba_emg_deg = emg_clf.predict_proba(F_emg_deg_s)
            proba_fused_deg = (proba_eeg + proba_emg_deg) / 2
            rows.append({param_name: v,
                         "emg_only": accuracy_score(y[idx_test], proba_emg_deg.argmax(1)),
                         "eeg_only": accuracy_score(y[idx_test], proba_eeg.argmax(1)),
                         "fusion_decision_avg": accuracy_score(y[idx_test], proba_fused_deg.argmax(1))})
        return pd.DataFrame(rows)

    rng = np.random.RandomState(RNG_SEED)
    def drop_channels(emg, n_keep):
        idx = rng.choice(emg.shape[0], size=n_keep, replace=False)
        out = np.zeros_like(emg)
        out[idx] = emg[idx]
        return out
    def add_noise(emg, snr_db):
        p = np.mean(emg ** 2)
        noise_p = p / (10 ** (snr_db / 10))
        return emg + rng.normal(0, np.sqrt(max(noise_p, 0)), emg.shape).astype(np.float32)
    def attenuate(emg, factor):
        return emg * factor

    channel_sweep = degrade_and_eval(drop_channels, list(range(7, 0, -1)), "n_channels_kept")
    snr_sweep = degrade_and_eval(add_noise, [20, 10, 5, 0, -5, -10], "snr_db")
    atten_sweep = degrade_and_eval(attenuate, [1.0, 0.5, 0.25, 0.1, 0.05, 0.0], "amplitude_factor")

    channel_sweep.to_csv(os.path.join(OUT_DIR, "project3_degradation_channels.csv"), index=False)
    snr_sweep.to_csv(os.path.join(OUT_DIR, "project3_degradation_snr.csv"), index=False)
    atten_sweep.to_csv(os.path.join(OUT_DIR, "project3_degradation_attenuation.csv"), index=False)
    print("channel dropout:\n", channel_sweep.to_string(index=False))
    print("gaussian noise:\n", snr_sweep.to_string(index=False))
    print("amplitude attenuation:\n", atten_sweep.to_string(index=False))

    for df, col in [(channel_sweep, "n_channels_kept"), (snr_sweep, "snr_db"), (atten_sweep, "amplitude_factor")]:
        crossover = df[df["fusion_decision_avg"] > df["emg_only"]]
        if len(crossover):
            print(f"crossover: fusion beats EMG-alone at {col} <= {crossover[col].iloc[0]}")
        else:
            print(f"no crossover found across swept {col} range")

    # ---- timing asymmetry ----
    print("\n=== timing asymmetry (accuracy vs. time relative to go-cue) ===")
    window_s, step_s = 0.4, 0.2
    offsets = np.arange(EPOCH_TMIN + window_s / 2, EPOCH_TMAX - window_s / 2, step_s)
    timing_rows = []
    for off in offsets:
        center = int(round((off - EPOCH_TMIN) * sfreq))
        half = int(round(window_s / 2 * sfreq))
        s, e = center - half, center + half
        if s < 0 or e > eeg_raw_full[0].shape[1]:
            continue
        eeg_win = np.stack([sosfiltfilt(eeg_bp_sos, tr[:, s:e], axis=1) for tr in eeg_raw_full])
        emg_win = np.stack([sosfiltfilt(emg_bp_sos, tr[:, s:e], axis=1) for tr in emg_raw_full])
        cov_t = Covariances(estimator="oas").fit_transform(eeg_win)
        F_eeg_t = TangentSpace(metric="riemann").fit_transform(cov_t)
        F_emg_t = np.stack([emg_hudgins_features(w) for w in emg_win])
        eeg_acc = cross_val_score(LogisticRegression(max_iter=1000), F_eeg_t, y, cv=3).mean()
        emg_acc = cross_val_score(LogisticRegression(max_iter=1000), F_emg_t, y, cv=3).mean()
        timing_rows.append({"t_rel_cue_s": off, "eeg_accuracy": eeg_acc, "emg_accuracy": emg_acc})
    timing_df = pd.DataFrame(timing_rows)
    timing_df.to_csv(os.path.join(OUT_DIR, "project3_timing_asymmetry.csv"), index=False)
    print(timing_df.to_string(index=False))

    pre_cue = timing_df[timing_df["t_rel_cue_s"] < 0]
    if len(pre_cue):
        lead = pre_cue[pre_cue["eeg_accuracy"] > pre_cue["emg_accuracy"] + 0.05]
        if len(lead):
            print(f"EEG leads EMG by margin>0.05 at t={lead['t_rel_cue_s'].iloc[0]:.2f}s pre-cue")
        else:
            print("no pre-cue window where EEG clearly leads EMG by >0.05 accuracy margin")

    # ---- mandatory control: EEG on pre-cue-only windows ----
    print("\n=== control: EEG accuracy restricted to pre-go-cue windows only ===")
    pre_end = int(round((0.0 - EPOCH_TMIN) * sfreq))  # sample index of the go cue itself
    eeg_pre = np.stack([sosfiltfilt(eeg_bp_sos, tr[:, :pre_end], axis=1) for tr in eeg_raw_full])
    cov_pre = Covariances(estimator="oas").fit_transform(eeg_pre)
    F_eeg_pre = TangentSpace(metric="riemann").fit_transform(cov_pre)
    pre_acc = cross_val_score(LogisticRegression(max_iter=1000), F_eeg_pre, y, cv=3).mean()
    print(f"EEG accuracy, pre-go-cue window only: {pre_acc:.4f} (chance = {1/n_classes:.4f})")
    print("Above-chance here supports genuine pre-movement EEG signal, not EMG bleed-through "
          "into the EEG channels (EMG should be near baseline before the go cue).")

    print("\nALL DONE")


if __name__ == "__main__":
    main()
