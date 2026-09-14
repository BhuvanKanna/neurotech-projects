1. EMG gesture classification (foundation)

Dataset: Ninapro DB2. 40 healthy subjects performed 49 movements recorded with 12 Delsys Trigno wireless electrodes, with each movement repeated 6 times for 5 seconds followed by a 3 second rest period. Signals are sampled at 2 kHz, with electrodes on the forearm flexors and extensors plus biceps and triceps. Registration required, free. Ninapro DB5 (dual Myo armbands, 16 channels, 200 Hz, 10 subjects) is the lower-cost alternative if you want channel counts closer to what a student org would actually build. 
bioRxiv
arxiv

Why DB2 over a Kaggle Myo set: the 6-repetition structure is the unit you need for project 4, so picking DB2 now saves you a dataset migration later.

Approach

Load the .mat files. Use the restimulus and rerepetition fields, not stimulus and repetition. The re- versions are the relabeled ground truth with movement onset corrected against the glove data. Most public notebooks get this wrong and silently train on mislabeled transition periods.
Subset to Exercise B (17 movements) or a 8-to-10 class subset plus rest. 49 classes is a research problem, not a portfolio project.
Preprocess: 4th-order Butterworth bandpass 20 to 450 Hz, 50 Hz notch (recorded in Europe, not 60 Hz), per-channel z-score using training-fold statistics only.
Window: 200 ms analysis windows with 100 ms increment. Justify the choice against the roughly 300 ms controller response budget, which Englehart and Hudgins identified as the threshold for a system to feel intuitive. 
Academia.edu
Features, the Hudgins time-domain set plus extensions: MAV, waveform length, zero crossings, slope sign changes, RMS, Willison amplitude, and 4th-order autoregressive coefficients. Roughly 12 channels by 10 features.
Models in escalating order: LDA, linear SVM, LightGBM, 1D CNN on raw windows.
Three split protocols, all reported. (a) Within-subject, repetitions 1/3/4/6 train and 2/5 test, which is the standard Ninapro protocol. (b) Leave-one-subject-out. (c) Random window shuffle, which is wrong, included explicitly to quantify the leakage. Showing that (c) inflates accuracy by 20-plus points is the single most persuasive thing in the whole repo.

Expected honest numbers: within-subject 75 to 90 percent macro F1 on 10 classes, LOSO dropping to 40 to 60 percent.

2. EEG motor imagery classification

Dataset: BCI Competition IV dataset 2a, accessed through MOABB (Mother of All BCI Benchmarks). 9 subjects, 22 channels, four classes (left hand, right hand, feet, tongue), 250 Hz, two sessions per subject. Use MOABB rather than raw downloads because it handles fetching, applies a standardized evaluation protocol, and gives you published leaderboard numbers to benchmark against. Benchmarking against a public baseline instead of reporting a number in a vacuum is what separates this from a tutorial. 
ResearchGate

For scale, PhysioNet's EEG Motor Movement/Imagery database (109 subjects, 64 channels, no data use agreement) is the alternative.

Approach

MNE-Python: bandpass 8 to 30 Hz, epoch 0.5 to 2.5 s post-cue, apply ICA with the EOG channels for artifact removal.
Baselines: CSP plus LDA, then FBCSP (filter bank of 4 Hz sub-bands from 4 to 40 Hz with mutual-information feature selection), then Riemannian tangent space projection plus logistic regression using pyriemann. The Riemannian pipeline usually wins and is cheap, which is a useful thing to discover yourself.
Deep models via braindecode: EEGNet and ShallowFBCSPNet.
Evaluate within-session, cross-session (train on session T, test on session E), and cross-subject.
The differentiator: add a rest/no-control class and report false activations per minute at varying confidence thresholds, not just four-class accuracy. For any device that moves, a spurious command is a safety event, and accuracy alone hides it.

Expected: within-subject 70 to 80 percent, cross-subject 50 to 65 percent. Chance is 25 percent, so say so.

3. EEG plus EMG fusion

Dataset: Jeong et al. 2020, GigaScience. This is precisely what you described. It contains 60-channel EEG, 7-channel EMG, and 4-channel EOG from 25 healthy participants. Participants performed 11 movement tasks: arm-reaching in 6 directions, hand-grasping of 3 objects, and wrist-twisting with 2 motions, with EMG sensors placed on the right arm to capture the muscle activity corresponding to each movement, across 3 recording sessions one week apart. It totals 82,500 trials, roughly 3,300 per participant, and both real movement and motor imagery conditions were recorded. Same labels, same clock, both modalities. 
GigaScience, 9, 2020, 1–15 doi: 10.1093/gigascience/giaa098 DATA NOTE DATA NOTE +2

Backup option: WAY-EEG-GAL, which has 32-channel EEG, EMG from five arm and hand muscles, 3D hand and object position, and contact force across 3,936 grasp-and-lift trials from twelve participants. 
ResearchGate

The framing correction that makes this project good. If you just concatenate features and compare against EMG alone, you will find almost no gain, because EMG has vastly better SNR for movement classification and will dominate every fusion model. That is a known result, not a bug, and a naive version of this project reads as naive. Design it as a robustness study instead:

Train three models on identical splits: EMG-only, EEG-only, fused.
Fuse at three levels and compare: feature concatenation, decision-level (probability averaging and stacking), and intermediate (two-branch network with a learned gating weight).
Systematically degrade the EMG branch and plot accuracy against degradation for all three models. Degradations: drop channels from 7 down to 1, inject Gaussian noise at decreasing SNR, attenuate amplitude to simulate weak residual activation in an amputee or a fatigued user. The crossover point where fusion overtakes EMG-alone is your result, and it is the clinically meaningful one, since the whole reason to add EEG to a prosthetic is that EMG is unreliable in the population that needs it.
Exploit the timing asymmetry. Movement-related cortical activity precedes muscle activation by a few hundred milliseconds. Plot classification accuracy as a function of time relative to cue for each modality separately. If EEG carries usable intent before EMG onset, you have an argument for EEG as an early trigger and EMG as confirmation, which is a real control architecture.
Mandatory control: EMG contaminates EEG. Show your EEG branch is not just reading muscle artifact. Bandpass EEG to 8 to 30 Hz, run ICA against the EOG channels, and include an ablation where the EEG model is tested on pre-movement windows only.
4. Temporal decision aggregation (your repeated-classification idea, formalized)

This is the strongest idea you proposed and it has real prior art to anchor against. Englehart and Hudgins introduced majority voting as a post-processing step that uses the classification results of multiple consecutive samples to update the decision for the current sample, and overlapped windowing raises the frequency of class decisions, which is what makes majority voting tractable in the first place. 
DOI
PubMed Central

Dataset: Ninapro DB2 again, reusing the project 1 pipeline.

Run two experiments, because the contrast between them is the actual finding.

(a) Consecutive-window voting. Slide a 200 ms window at 100 ms increments through a single 5-second repetition and majority-vote over the last k decisions. Overlapping windows share the same underlying contraction, so their errors are strongly correlated and the gain will fall well short of theory.

(b) Across-repetition voting. Take k separate repetitions of the same gesture, produce one decision each, and vote. Errors here are much closer to independent, so gains should approach theory.

Quantify the gap against the binomial ceiling. For per-decision accuracy p and odd k, the independent-error prediction is the sum over i > k/2 of C(k,i)·p^i·(1−p)^(k−i). Plot empirical against theoretical for k = 1, 3, 5, 7, 9 in both experiments. The gap size is a direct measure of error correlation. Quantify it properly with the pairwise Q-statistic or the correlation of error indicator vectors.

Then price the latency. k votes at a 100 ms decision increment costs (k−1)·100 ms of added delay. Plot accuracy against total latency and draw the 300 ms usability line on it. Finally, collapse the whole tradeoff into a single number with Wolpaw information transfer rate in bits per minute, since more votes buy accuracy but cost decisions per minute. Finding the k that maximizes ITR is the actual engineering answer, and essentially no student project computes it.

Compare majority voting against: confidence-weighted voting (sum of softmax probabilities), exponentially weighted moving average over the probability stream, and a simple rejection threshold that emits nothing below a confidence floor.

5. Quantization and efficiency Pareto frontier

Take the best CNN from project 1 and compress it. Entirely software, no hardware purchase.

Configurations to sweep: FP32 baseline, FP16, post-training dynamic int8, post-training static int8 with a calibration set, quantization-aware training, structured pruning at 30/50/70 percent sparsity, and knowledge distillation from the CNN into a small MLP.

Tooling: torch.ao.quantization, ONNX Runtime, TFLite converter, ptflops or fvcore for MAC counts.

Metrics per configuration: macro F1, model size in KB, inference latency as median and p95 over 1000 runs pinned to a single CPU thread (a microcontroller is single-threaded, so multi-threaded timings are meaningless here), and MACs.

Present it as a Pareto plot of accuracy against size and accuracy against latency, and identify the knee. Add a per-layer sensitivity analysis where you quantize one layer at a time to show which layers must stay in floating point.

The finding to watch for: LDA on hand-crafted features is frequently within two or three points of the CNN at a tiny fraction of the compute. If that is what you find, lead with it. Reporting that your fancy model was not worth it demonstrates more judgment than any accuracy number.

Free C/C++ credential: export the int8 model to ONNX and write a small C++ harness using the ONNX Runtime C++ API, or hand-write the quantized convolution loop in C and assert it matches the Python output within tolerance. That satisfies the C/C++ line on their skills list with zero hardware spend.

How to sequence this

Projects 1, 4, and 5 chain into one repository: classify, then aggregate decisions over time, then compress. That is a single coherent story about building a usable myoelectric controller. Projects 2 and 3 form a second repository about multimodal intent decoding.

Two finished projects with honest metrics beat five described ones. If the application is near, build 1 and 4 completely, since together they already cover machine learning, time-series processing, and Python for ML, then describe 5 as in progress with the specific tooling named.
