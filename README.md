# Neurotech Projects

Five myoelectric/EEG decoding projects, organized as two repos-in-one per the original sequencing plan (see [`projects.md`](projects.md) for the full spec each notebook implements):

- **[`myoelectric-controller/`](myoelectric-controller/)** — classify → aggregate decisions over time → compress. One coherent story about building a usable myoelectric controller.
  1. [`01_emg_gesture_classification.ipynb`](myoelectric-controller/01_emg_gesture_classification.ipynb) — Ninapro DB2, LDA/SVM/LightGBM/1D-CNN, three split protocols (including a deliberately-leaky control to quantify overlap-window leakage).
  2. [`02_temporal_decision_aggregation.ipynb`](myoelectric-controller/02_temporal_decision_aggregation.ipynb) — majority/confidence-weighted/EWMA/rejection voting, binomial-ceiling comparison, Wolpaw ITR optimization.
  3. [`03_quantization_efficiency.ipynb`](myoelectric-controller/03_quantization_efficiency.ipynb) — FP32/FP16/int8/QAT/pruning/distillation Pareto sweep, ONNX export + C++ harness.
- **[`multimodal-intent-decoding/`](multimodal-intent-decoding/)** — multimodal movement-intent decoding.
  1. [`01_eeg_motor_imagery.ipynb`](multimodal-intent-decoding/01_eeg_motor_imagery.ipynb) — BCI Competition IV 2a via MOABB, CSP+LDA/FBCSP/Riemannian/EEGNet/ShallowFBCSPNet, within/cross-session/LOSO, false-activation-rate analysis. **Fully executed — see [results](multimodal-intent-decoding/README.md).**
  2. [`02_eeg_emg_fusion.ipynb`](multimodal-intent-decoding/02_eeg_emg_fusion.ipynb) — EEG+EMG fusion (Jeong et al. 2020, GigaScience), framed as a robustness study: fusion vs. EMG-only under systematic degradation, timing-asymmetry analysis.

## Status

| # | Project | Status |
|---|---|---|
| 1 | EMG gesture classification | Pipeline complete, not yet run (Ninapro DB2 requires registration) |
| 2 | EEG motor imagery | **Executed end-to-end** — real results in [`multimodal-intent-decoding/README.md`](multimodal-intent-decoding/README.md) |
| 3 | EEG+EMG fusion | Pipeline complete, not yet run (requires converting Jeong et al.'s GigaDB distribution to the notebook's loader format) |
| 4 | Temporal decision aggregation | Pipeline complete, not yet run (depends on a trained Project 1 CNN) |
| 5 | Quantization/efficiency | Pipeline complete, not yet run (depends on a trained Project 1 CNN) |

## Reproducing Project 2

```bash
python -m venv .venv
source .venv/Scripts/activate   # or .venv\Scripts\activate on Windows
pip install moabb mne scikit-learn pyriemann scipy numpy pandas matplotlib torch braindecode skorch
python multimodal-intent-decoding/run_project2_lean.py
```

`run_project2_lean.py` is a memory-lean, resumable standalone runner (processes one subject at a time, caches only tiny per-trial covariance matrices for the cross-subject LOSO pass instead of holding all subjects' raw epochs in memory, and skips any subject/fold already present in its output CSVs on a re-run) — written after the notebook's in-kernel execution twice hit an OOM kill on a memory-constrained machine. The notebook itself (`01_eeg_motor_imagery.ipynb`) contains the same logic in its original in-notebook form for reference/adaptation.
