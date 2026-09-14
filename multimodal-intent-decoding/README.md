# Multimodal Intent Decoding

## Project 2 — EEG Motor Imagery: real results (executed 2026-09-14)

Dataset: BCI Competition IV 2a (`BNCI2014_001` via MOABB), 9 subjects, 22 channels, 4 classes (left hand / right hand / feet / tongue), 250 Hz, 2 sessions/subject. Chance = 25%.

Run via [`run_project2_lean.py`](run_project2_lean.py) (see top-level README for why — the notebook's own in-kernel execution hit an OOM kill twice on this machine; the lean runner does the identical computation one subject at a time). Raw output: [`project2_results.csv`](project2_results.csv) (within-session + cross-session), [`project2_loso_results.csv`](project2_loso_results.csv) (cross-subject), [`project2_deep_results.csv`](project2_deep_results.csv) (EEGNet / ShallowFBCSPNet).

### Within-session (mean over 9 subjects)

| Model | Accuracy | Macro F1 |
|---|---|---|
| CSP + LDA | 64.3% | 64.0% |
| FBCSP | 70.8% | 70.4% |
| Riemannian tangent-space + LogReg | 70.0% | 69.7% |
| ShallowFBCSPNet | 57.9% | 56.7% |
| EEGNet | 35.2% | 24.6% |

### Cross-session (Riemannian, train session 1 → test session 2)

67.1% accuracy / 66.8% macro F1 — close to within-session, a good sign the Riemannian features generalize across a subject's own sessions.

### Cross-subject LOSO (Riemannian)

**40.7% accuracy / 36.6% macro F1** — well above chance (25%) but below the literature ballpark I'd sketched in the notebook (50-65%) before running it. Real number, not adjusted to match the expectation.

### Findings worth keeping, not smoothing over

- **EEGNet overfits badly** on this dataset size (~250 training trials/subject): 35% accuracy, barely above chance, with training loss dropping to ~0.2 while validation loss climbs from epoch 3 onward. ShallowFBCSPNet (a shallower architecture designed for small EEG datasets) gets 58% — much better, but still behind the classical Riemannian pipeline (70%). This matches known BCI-IV-2a results: deep nets need more data or cross-subject pretraining to beat Riemannian at this dataset size.
- **Large subject variability**: subjects 1/3/7/8/9 hit 75-89% within-session; subjects 2/5/6 stall at 40-54%. This is the well-documented "BCI illiteracy" pattern for this exact dataset — not a bug in the pipeline.
- Per-subject detail and the false-activation-rate/confidence-threshold analysis are in the notebook itself ([`01_eeg_motor_imagery.ipynb`](01_eeg_motor_imagery.ipynb)); the lean runner above only reproduces the three split protocols + deep models, not the rest-class section.

## Project 3 — EEG+EMG Fusion

Pipeline complete in [`02_eeg_emg_fusion.ipynb`](02_eeg_emg_fusion.ipynb), not yet executed — requires converting Jeong et al. 2020's GigaDB distribution (doi: 10.1093/gigascience/giaa098) into the `.npz` layout `load_session()` expects (see the notebook's data-loading section for the exact schema).
