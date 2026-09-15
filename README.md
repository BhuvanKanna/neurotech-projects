# Neurotech Projects

Five myoelectric/EEG decoding projects, organized as two repos-in-one per the original sequencing plan (see [`projects.md`](projects.md) for the full spec each notebook implements). **All five have now been executed against real data** — see the Status table below for what ran at full spec vs. a documented reduced scope.

- **[`myoelectric-controller/`](myoelectric-controller/)** — classify → aggregate decisions over time → compress. One coherent story about building a usable myoelectric controller.
  1. [`01_emg_gesture_classification.ipynb`](myoelectric-controller/01_emg_gesture_classification.ipynb) — LDA/SVM/LightGBM/1D-CNN, three split protocols (including a deliberately-leaky control to quantify overlap-window leakage). **Fully executed against Ninapro DB5 — see [results](myoelectric-controller/README.md).**
  2. [`02_temporal_decision_aggregation.ipynb`](myoelectric-controller/02_temporal_decision_aggregation.ipynb) — majority/confidence-weighted/EWMA/rejection voting, binomial-ceiling comparison, Wolpaw ITR optimization. **Fully executed — see [results](myoelectric-controller/README.md).**
  3. [`03_quantization_efficiency.ipynb`](myoelectric-controller/03_quantization_efficiency.ipynb) — FP32/FP16/int8/QAT/pruning/distillation Pareto sweep, ONNX export + C++ harness. **Fully executed — see [results](myoelectric-controller/README.md).**
- **[`multimodal-intent-decoding/`](multimodal-intent-decoding/)** — multimodal movement-intent decoding.
  1. [`01_eeg_motor_imagery.ipynb`](multimodal-intent-decoding/01_eeg_motor_imagery.ipynb) — BCI Competition IV 2a via MOABB, CSP+LDA/FBCSP/Riemannian/EEGNet/ShallowFBCSPNet, within/cross-session/LOSO, false-activation-rate analysis. **Fully executed — see [results](multimodal-intent-decoding/README.md).**
  2. [`02_eeg_emg_fusion.ipynb`](multimodal-intent-decoding/02_eeg_emg_fusion.ipynb) — EEG+EMG fusion (Jeong et al. 2020, GigaScience), framed as a robustness study: fusion vs. EMG-only under systematic degradation, timing-asymmetry analysis. **Executed at reduced scope (1 subject, 1 session) — see [results](multimodal-intent-decoding/README.md).**

## Status

| # | Project | Status |
|---|---|---|
| 1 | EMG gesture classification | **Executed, full spec** (10/10 subjects, Ninapro DB5 not DB2 — see below) — [results](myoelectric-controller/README.md) |
| 2 | EEG motor imagery | **Executed, full spec** (9/9 subjects) — [results](multimodal-intent-decoding/README.md) |
| 3 | EEG+EMG fusion | **Executed, reduced scope** (1 of 25 subjects, 1 of 3 sessions, 1 of 3 task types — see why in [results](multimodal-intent-decoding/README.md)) |
| 4 | Temporal decision aggregation | **Executed, full spec** (against a trained Project 1 CNN checkpoint) — [results](myoelectric-controller/README.md) |
| 5 | Quantization/efficiency | **Executed, full spec** (same checkpoint as #4) — [results](myoelectric-controller/README.md) |

## Reproducing this

```bash
python -m venv .venv
source .venv/Scripts/activate   # or .venv\Scripts\activate on Windows
pip install moabb mne scikit-learn pyriemann scipy numpy pandas matplotlib torch braindecode skorch lightgbm onnx onnxruntime onnxscript ptflops

# Project 1 — downloads Ninapro DB5 (~200MB, no registration) to myoelectric-controller/data/ if not already present
python myoelectric-controller/run_project1_lean.py

# Project 2 — downloads BCI-IV-2a via MOABB on first run
python multimodal-intent-decoding/run_project2_lean.py

# Project 3 — downloads ~9.8GB of Jeong2020 sourcedata for subject 1 via MOABB/NEMAR on first run
python multimodal-intent-decoding/run_project3_lean.py

# Projects 4 & 5 need a trained Project 1 CNN checkpoint first:
python myoelectric-controller/train_cnn_checkpoint.py
python myoelectric-controller/run_project4_lean.py
python myoelectric-controller/run_project5_lean.py
```

All `run_project*_lean.py` scripts are memory-lean, resumable standalone runners (process one subject at a time, cache only small per-subject artifacts instead of holding every subject's raw data in memory, and skip any subject/fold already present in their output CSVs on a re-run). These were written and needed after the notebooks' in-kernel execution repeatedly hit OOM kills on this machine — other running applications were consuming most of its 16GB RAM, and free memory fluctuated wildly (from ~4GB down to ~700MB) across the runs, independent of anything these scripts were doing. The notebooks themselves (`01_eeg_motor_imagery.ipynb`, `01_emg_gesture_classification.ipynb`, `02_eeg_emg_fusion.ipynb`) contain the pipeline logic in its original in-notebook form for reference/adaptation at full spec (25 subjects for Project 3, DB2 instead of DB5 for Project 1) — none of the three were themselves re-executed; the lean runners are the versions that actually produced the committed results, at the dataset/scope substitutions documented in each project's own README.
