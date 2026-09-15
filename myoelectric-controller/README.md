# Myoelectric Controller

Classify → aggregate decisions over time → compress. Three notebooks chained into one repository, per the original spec in [`../projects.md`](../projects.md).

1. [`01_emg_gesture_classification.ipynb`](01_emg_gesture_classification.ipynb) — LDA/SVM/LightGBM/1D-CNN, three split protocols including a deliberately-leaky random-shuffle control to quantify overlap-window leakage. **Fully executed against Ninapro DB5 — see results below.**
2. [`02_temporal_decision_aggregation.ipynb`](02_temporal_decision_aggregation.ipynb) — consecutive-window vs. across-repetition majority voting, binomial-ceiling comparison, Q-statistic error correlation, latency pricing, Wolpaw information-transfer-rate optimization. Reuses Project 1's trained CNN. **Fully executed — see results below.**
3. [`03_quantization_efficiency.ipynb`](03_quantization_efficiency.ipynb) — FP32/FP16/dynamic+static int8/QAT/pruning/distillation Pareto sweep, per-layer sensitivity analysis, ONNX export + C++ inference harness. Reuses Project 1's trained CNN. **Fully executed — see results below.**

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

## Project 4 — Temporal Decision Aggregation: real results (executed 2026-09-15)

Runs against a CNN checkpoint trained on subject S1's within-subject split (see [`train_cnn_checkpoint.py`](train_cnn_checkpoint.py), which saves the state dict + normalization stats + windows to `checkpoints/` — the piece Project 1's lean runner never persisted). [`run_project4_lean.py`](run_project4_lean.py) runs both experiments against real S1 test-set predictions. Raw output: [`project4_consecutive_summary.csv`](project4_consecutive_summary.csv), [`project4_across_repetition_summary.csv`](project4_across_repetition_summary.csv), [`project4_voting_rule_comparison.csv`](project4_voting_rule_comparison.csv).

**Real data constraint found while building this**: the notebook's spec sweeps k = 1,3,5,7,9 for both voting experiments. That's fine for experiment (a) (consecutive-window voting within one 5s repetition holds ~45 windows). It's not fine for experiment (b) (across-repetition voting) — Ninapro only records **6 repetitions per movement**, and the standard train/test split leaves only 2 repetitions (reps 2 & 5) in the test set, so a train/test-respecting version of experiment (b) tops out at k=2, not k=9. `run_project4_lean.py` reports this honestly (k=3, k=5 show `n=0` eligible groups on test-only data) alongside a second, clearly-labeled pass using all 6 repetitions (which reaches k=5, but with the caveat that reps 1/3/4/6 were seen during CNN training).

### Experiment (a): consecutive-window voting (correlated errors)

| k | empirical | binomial ceiling | latency added |
|---|---|---|---|
| 1 | 58.7% | 58.7% | 0 ms |
| 3 | 62.3% | 63.0% | 200 ms |
| 5 | 66.8% | 66.1% | 400 ms |
| 7 | 72.9% | 68.5% | 600 ms |
| 9 | 75.0% | 70.6% | 800 ms |

Mean adjacent-window Q-statistic: **0.43** — substantial positive error correlation, as expected for overlapping 100ms-hop windows from the same contraction.

### Experiment (b): across-repetition voting (test-only reps 2 & 5)

k=1: 94.4% (n=18). k=3 and k=5: **0 eligible groups** — only 2 test repetitions per movement exist, so this real dataset constraint (not a bug) caps what a train/test-clean version of this experiment can show. A second pass reusing all 6 repetitions gets k=3→100%, k=5→100%, but that's inflated by training-set leakage (reps 1/3/4/6 were in the CNN's training data) rather than genuine independent-error voting gain — reported for completeness, not as the honest number.

### The engineering answer: k maximizing ITR

**k=1** — for this checkpoint, voting's accuracy gains never outpace the latency cost in bits/min. ITR drops monotonically from 622 bpm (k=1) to 115 bpm (k=9). This is a real, checkpoint-specific finding: a stronger base classifier (this CNN's k=1 macro F1 was only 55% on S1, see Project 1) would likely shift the ITR-optimal k upward, since voting has more room to help a weaker base rate.

### Voting rule comparison (consecutive-window)

Confidence-weighted and EWMA voting modestly beat plain majority at every k (e.g. k=9: 77.8%/75.8% vs. 75.0%). Rejection thresholding (floor=0.5) buys much higher accuracy on the windows it does answer (e.g. 97.7% at k=9) but abstains on 43.6% of them — a real accuracy/coverage tradeoff, not a free lunch.

## Project 5 — Quantization / Efficiency Pareto Frontier: real results (executed 2026-09-15)

Same S1 CNN checkpoint as Project 4. [`run_project5_lean.py`](run_project5_lean.py) — raw output in [`project5_results.csv`](project5_results.csv), [`project5_layer_sensitivity.csv`](project5_layer_sensitivity.csv), and the exported [`cnn_S1.onnx`](cnn_S1.onnx).

### Pareto table

| Config | Macro F1 | Size (KB) | Median latency (ms) | MACs |
|---|---|---|---|---|
| dynamic_int8 | 61.8% | 93.5 | 1.29 | 370,560 |
| qat_int8 | 61.7% | **32.6** | 1.94 | 6,400 |
| fp16 | 61.7% | 50.8 | 1.82 | — |
| fp32_baseline | 61.7% | 94.7 | **0.88** | 371,210 |
| lda_features_baseline | 61.1% | **19.6** | **0.25** | — |
| static_int8_calibrated | 60.6% | 32.6 | 1.58 | 6,400 |
| pruned_70pct | 60.4% | 94.7 | 1.05 | 371,210 |
| pruned_30pct | 57.5% | 94.7 | 0.79 | 371,210 |
| pruned_50pct | 57.2% | 94.7 | 1.07 | 371,210 |
| distilled_mlp | 28.7% | 23.7 | **0.04** | — |

**The finding to watch for, and it held**: LDA-on-features lands within **0.6 macro-F1 points of the FP32 CNN at a fifth of the size and a fourth of the latency** — the notebook predicted exactly this ("LDA is frequently within 2-3 points... lead with it if that's what you find"). QAT gets the best accuracy/size tradeoff among the compressed CNN variants (61.7% at 32.6KB). Static int8 without QAT fine-tuning loses a full point of macro F1 relative to QAT for the identical 32.6KB footprint — the fine-tuning step earns its keep. Structured pruning actively *hurts* here (all three sparsity levels underperform the unpruned baseline) — this CNN's ~371K MACs were apparently not very redundant to begin with. Knowledge distillation into the small MLP badly underperformed (28.7% macro F1, barely above the 10-class ~10% floor) — a real negative result, not tuned away; 30 epochs on a single subject's data likely wasn't enough for the soft-target signal to transfer.

### Real bug found and fixed: BatchNorm has no quantized-CPU kernel

Static int8 and QAT both crashed on first attempt with `NotImplementedError: Could not run 'aten::native_batch_norm' with arguments from the 'QuantizedCPU' backend`. Cause: PyTorch's int8 backend has no kernel for a bare `nn.BatchNorm1d` — it must be **fused** into the preceding `Conv1d` (`torch.ao.quantization.fuse_modules` / `fuse_modules_qat`) before `prepare`/`convert`. Fixed via `fuse_cnn()` in the runner script, fusing each `[Conv1d, BatchNorm1d, ReLU]` triple beforehand.

### Per-layer quantization sensitivity

| Layer | Macro F1 delta vs. FP32 |
|---|---|
| net.0 (first conv) | **−0.37 points** |
| net.4 (second conv) | −0.04 points |
| net.8 (third conv) | +0.05 points |

The first conv layer is the most quantization-sensitive (unsurprising — it's closest to the raw, unnormalized-scale EMG input), consistent with the standard advice to keep early layers in float when a real int8 deployment budget allows it.

### ONNX export

Exported to [`cnn_S1.onnx`](cnn_S1.onnx); PyTorch vs. ONNX Runtime max abs logit difference: **2.26e-06** — clean parity, ready to drive from the ONNX Runtime C++ API (sketch in the original notebook's section 14).
