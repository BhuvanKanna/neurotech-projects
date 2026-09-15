# Myoelectric Controller

Classify → aggregate decisions over time → compress. Three notebooks chained into one repository, per the original spec in [`../projects.md`](../projects.md).

1. [`01_emg_gesture_classification.ipynb`](01_emg_gesture_classification.ipynb) — LDA/SVM/LightGBM/1D-CNN, three split protocols including a deliberately-leaky random-shuffle control to quantify overlap-window leakage. **Fully executed against Ninapro DB5 — see results below.**
2. [`02_temporal_decision_aggregation.ipynb`](02_temporal_decision_aggregation.ipynb) — consecutive-window vs. across-repetition majority voting, binomial-ceiling comparison, Q-statistic error correlation, latency pricing, Wolpaw information-transfer-rate optimization. Reuses Project 1's trained CNN — not yet run.
3. [`03_quantization_efficiency.ipynb`](03_quantization_efficiency.ipynb) — FP32/FP16/dynamic+static int8/QAT/pruning/distillation Pareto sweep, per-layer sensitivity analysis, ONNX export + C++ inference harness. Reuses Project 1's trained CNN — not yet run.

## Project 1 — EMG Gesture Classification: real results (executed 2026-09-14/15)

**Dataset: Ninapro DB5** (10 subjects, two 8-channel Myo armbands = 16 EMG channels, 200 Hz, Exercise B/17 movements), not DB2 — DB2 requires registration at ninapro.hevs.ch; DB5 is a direct, no-login download from [Zenodo](https://zenodo.org/records/1000116). The notebook itself (`01_emg_gesture_classification.ipynb`) is written for DB2's 2 kHz/12-channel Delsys setup and documented as such; [`run_project1_lean.py`](run_project1_lean.py) is the DB5-adapted version that was actually run — same pipeline (Butterworth bandpass + notch, 200 ms/100 ms windowing, Hudgins+AR features, same 4 models, same 3 split protocols), with DB5's sampling rate (bandpass capped below its 100 Hz Nyquist instead of DB2's 20-450 Hz) and 16-channel layout, plus the same memory-lean/resumable pattern used for Project 2 (this machine repeatedly hit OOM kills from other running applications during both projects — see git log for the full retry history).

10 classes (rest + 9 most frequent movements), chance ≈ 10% (but see the class-imbalance note below — rest alone is ~57% of samples, so *accuracy* chance is much higher than 10%).

Raw output: [`project1_results.csv`](project1_results.csv) (within-subject + leaky-shuffle control), [`project1_loso_results.csv`](project1_loso_results.csv) (cross-subject LOSO).

### Within-subject (standard protocol: reps 1/3/4/6 train, 2/5 test), mean over 10 subjects

| Model | Accuracy | Macro F1 |
|---|---|---|
| LDA | 88.8% | 64.9% |
| SVM | 88.3% | 62.4% |
| LightGBM | **91.4%** | **72.3%** |
| 1D CNN | 86.5% | 55.3% |

### The leakage control — this is the notebook's actual headline result

Random-shuffle split (intentionally wrong — ignores repetition structure) vs. the proper within-subject split, mean macro-F1 inflation across 10 subjects:

| Model | Macro F1 inflation |
|---|---|
| CNN | **+20.4 points** |
| LightGBM | +9.7 points |
| SVM | +9.2 points |
| LDA | +4.1 points |

The notebook predicted "expect random-shuffle to overstate accuracy by 20+ points" — that held almost exactly for the CNN (adjacent 100ms-hop windows from the same contraction leak hardest into a raw-window model) and was real but smaller for the feature-based models.

### Cross-subject LOSO

| Model | Accuracy | Macro F1 |
|---|---|---|
| LDA | 74.2% | **15.3%** |
| SVM | 74.8% | 15.5% |

**The accuracy/macro-F1 gap here is the finding, not a fluke.** ~74% accuracy looks fine until you notice rest is ~57% of all windows — a model trained on 9 subjects and tested on a held-out one is mostly just predicting "rest" for everything. Macro F1 (unweighted across all 10 classes) exposes that a LOSO-trained classifier is barely distinguishing actual gestures across subjects. This is well below the "40-60%" ballpark I'd sketched in the notebook before running it — a real number, not adjusted to match the expectation, and a much starker illustration of the cross-subject generalization problem than DB2 would likely have shown (DB5's 16-channel dry-electrode Myo armbands are noisier and less consistently placed across subjects than DB2's 12-channel Delsys Trigno setup).

### Bug found and fixed while adapting this pipeline

`01_emg_gesture_classification.ipynb`'s original `list_subject_files()` built a glob pattern from `cfg.exercise` (a letter, `"B"`) directly into the filename: `S*_EB_A1.mat`. Real Ninapro files are numbered (`S*_E2_A1.mat` for Exercise B), so that glob never matched anything. Fixed with an explicit letter→file-number map (`A→E1, B→E2, C→E3`), now in both the notebook and the lean runner.

## Project 3 — Quantization / Temporal Aggregation

Not yet executed — both depend on a trained Project 1 CNN artifact (`torch.save`d state dict), which the lean runner doesn't currently persist to disk (it evaluates and discards each subject's CNN). Adding that save/load step is the remaining work before 2 and 3 can run for real.
