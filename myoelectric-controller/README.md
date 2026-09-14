# Myoelectric Controller

Classify → aggregate decisions over time → compress. Three notebooks chained into one repository, per the original spec in [`../projects.md`](../projects.md).

1. [`01_emg_gesture_classification.ipynb`](01_emg_gesture_classification.ipynb) — Ninapro DB2, Hudgins+AR features, LDA/SVM/LightGBM/1D-CNN, three split protocols including a deliberately-leaky random-shuffle control to quantify overlap-window leakage.
2. [`02_temporal_decision_aggregation.ipynb`](02_temporal_decision_aggregation.ipynb) — consecutive-window vs. across-repetition majority voting, binomial-ceiling comparison, Q-statistic error correlation, latency pricing, Wolpaw information-transfer-rate optimization. Reuses Project 1's trained CNN.
3. [`03_quantization_efficiency.ipynb`](03_quantization_efficiency.ipynb) — FP32/FP16/dynamic+static int8/QAT/pruning/distillation Pareto sweep, per-layer sensitivity analysis, ONNX export + C++ inference harness. Reuses Project 1's trained CNN.

## Status

Not yet executed — Ninapro DB2 requires registration at http://ninapro.hevs.ch/ before the `.mat` files needed by `01_emg_gesture_classification.ipynb`'s data loader can be downloaded. Once that's done, run notebook 1 first (it saves the trained CNN + z-score stats that 2 and 3 both load).
