# Neurotech Projects

Five myoelectric/EEG decoding projects, organized as two repos-in-one per the original sequencing plan (see [`projects.md`](projects.md) for the full spec each notebook implements):

- **[`myoelectric-controller/`](myoelectric-controller/)** — classify → aggregate decisions over time → compress. One coherent story about building a usable myoelectric controller.
  1. [`01_emg_gesture_classification.ipynb`](myoelectric-controller/01_emg_gesture_classification.ipynb) — LDA/SVM/LightGBM/1D-CNN, three split protocols (including a deliberately-leaky control to quantify overlap-window leakage). **Fully executed against Ninapro DB5 — see [results](myoelectric-controller/README.md).**
  2. [`02_temporal_decision_aggregation.ipynb`](myoelectric-controller/02_temporal_decision_aggregation.ipynb) — majority/confidence-weighted/EWMA/rejection voting, binomial-ceiling comparison, Wolpaw ITR optimization.
  3. [`03_quantization_efficiency.ipynb`](myoelectric-controller/03_quantization_efficiency.ipynb) — FP32/FP16/int8/QAT/pruning/distillation Pareto sweep, ONNX export + C++ harness.
- **[`multimodal-intent-decoding/`](multimodal-intent-decoding/)** — multimodal movement-intent decoding.
  1. [`01_eeg_motor_imagery.ipynb`](multimodal-intent-decoding/01_eeg_motor_imagery.ipynb) — BCI Competition IV 2a via MOABB, CSP+LDA/FBCSP/Riemannian/EEGNet/ShallowFBCSPNet, within/cross-session/LOSO, false-activation-rate analysis. **Fully executed — see [results](multimodal-intent-decoding/README.md).**
  2. [`02_eeg_emg_fusion.ipynb`](multimodal-intent-decoding/02_eeg_emg_fusion.ipynb) — EEG+EMG fusion (Jeong et al. 2020, GigaScience), framed as a robustness study: fusion vs. EMG-only under systematic degradation, timing-asymmetry analysis.

## Status

| # | Project | Status |
|---|---|---|
| 1 | EMG gesture classification | **Executed end-to-end** (Ninapro DB5, not DB2 — see below) — real results in [`myoelectric-controller/README.md`](myoelectric-controller/README.md) |
| 2 | EEG motor imagery | **Executed end-to-end** — real results in [`multimodal-intent-decoding/README.md`](multimodal-intent-decoding/README.md) |
| 3 | EEG+EMG fusion | Pipeline complete, not yet run (requires converting Jeong et al.'s GigaDB distribution to the notebook's loader format) |
| 4 | Temporal decision aggregation | Pipeline complete, not yet run (depends on a saved Project 1 CNN checkpoint, which the lean DB5 runner doesn't currently persist) |
| 5 | Quantization/efficiency | Pipeline complete, not yet run (same dependency as #4) |

## Reproducing Projects 1 and 2

```bash
python -m venv .venv
source .venv/Scripts/activate   # or .venv\Scripts\activate on Windows
pip install moabb mne scikit-learn pyriemann scipy numpy pandas matplotlib torch braindecode skorch lightgbm

# Project 1 — downloads Ninapro DB5 (~200MB, no registration) to myoelectric-controller/data/ if not already present
python myoelectric-controller/run_project1_lean.py

# Project 2 — downloads BCI-IV-2a via MOABB on first run
python multimodal-intent-decoding/run_project2_lean.py
```

Both `run_project*_lean.py` scripts are memory-lean, resumable standalone runners (process one subject at a time, cache only small per-subject artifacts — covariance matrices for Project 2's LOSO, feature matrices for Project 1's — instead of holding every subject's raw data in memory, and skip any subject/fold already present in their output CSVs on a re-run). Both were written and needed after their notebooks' in-kernel execution repeatedly hit OOM kills on this machine (other running applications were consuming most of its 16GB RAM; free memory fluctuated between ~700MB and ~4GB across the runs). The notebooks themselves (`01_eeg_motor_imagery.ipynb`, `01_emg_gesture_classification.ipynb`) contain the same logic in their original in-notebook form for reference/adaptation — Project 1's notebook targets DB2 specifically (2kHz/12-channel Delsys) and wasn't itself re-executed; the lean runner is the DB5-adapted version that actually produced the results below.
