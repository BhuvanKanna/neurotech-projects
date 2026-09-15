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

## Project 3 — EEG+EMG Fusion: real results (executed 2026-09-15)

**Scope reduction, stated up front**: the full spec is 25 subjects x 3 sessions x 3 task types x 2 conditions (MI + real movement). One subject's raw data for a single one of those 18 recordings is ~960MB on disk (9.8GB for all of subject 1 alone); given this machine's memory constraints (documented throughout — see the top-level README and Project 1/2 writeups), this run covers **one subject (sub1), one session (session1), one task type (reaching — 6 directions, 300 trials), real-movement condition** (not MI — needed for genuine EMG signal to make the degradation sweep meaningful). `02_eeg_emg_fusion.ipynb` is written for the full 25-subject `.npz`-based pipeline and was not itself re-executed; [`run_project3_lean.py`](run_project3_lean.py) is what actually ran, reading trials directly from the raw BrainVision file with `preload=False` so the ~1GB continuous recording is never fully loaded into memory (each ~4s trial window is pulled from disk on demand — 300 trials this way took about 2 seconds). Access note: MOABB's `Jeong2020` loader (Zenodo/NEMAR-backed) turned out far easier than GigaDB's own site, which is a JS-only SPA with no scriptable download path I could find.

Raw output: [`project3_fusion_results.csv`](project3_fusion_results.csv), [`project3_degradation_{channels,snr,attenuation}.csv`](project3_degradation_channels.csv), [`project3_timing_asymmetry.csv`](project3_timing_asymmetry.csv).

### Fusion comparison (clean data, 210 train / 90 test trials, 6 classes, chance = 16.7%)

| Model | Accuracy | Macro F1 |
|---|---|---|
| EMG-only | **91.1%** | 91.1% |
| Fusion, decision-avg | 85.6% | 85.5% |
| Fusion, feature-concat | 42.2% | 43.2% |
| EEG-only (Riemannian) | 20.0% | 20.6% |
| Fusion, intermediate gated | 20.0% | 19.1% |

The EEG-only Riemannian pipeline badly underperforms here relative to Project 2's BCI-IV-2a numbers (barely above chance vs. 70% there) — plausibly the single-session, no-ICA, EOG-linear-regression-only preprocessing (see below) and a much less standard "reaching" motor-execution paradigm than BCI-IV-2a's well-studied motor-imagery task. That weak EEG branch explains the rest of the table: **feature-concat fusion actively hurts** (a much higher-dimensional, noisier EEG tangent-space vector — 1830 dims — swamps a 28-dim EMG feature vector in only 210 training samples), and the **gated fusion network learned to weight EEG at 90%** despite EEG being the far worse branch — a real failure mode of naive learned gating, not a hand-picked example: the gate isn't guaranteed to discover which branch is actually informative, especially with this little training data.

### EMG degradation sweep

| Degradation | Crossover (fusion beats EMG-alone) |
|---|---|
| Channel dropout (7→1) | at ≤4 channels kept |
| Gaussian noise (20 to −10 dB SNR) | none found — both EMG-only and fusion collapse to ~chance together once SNR is very low |
| Amplitude attenuation (1.0→0.0) | at ≤0.5x amplitude |

The channel-dropout and attenuation crossovers are the clinically meaningful result the notebook's spec was after: once EMG signal quality degrades enough, blending in even this weak EEG branch starts to help. The SNR sweep didn't show a crossover — added Gaussian noise destroys the EMG feature set fast enough that fusion has nothing useful to blend in by the time EMG-alone drops, since EEG itself is only 20% accurate to begin with.

### Timing asymmetry — did NOT reproduce the expected EEG-leads-EMG finding

EMG decodability rises steadily after the go-cue (17% pre-cue → 68% by +2.2 to +2.4s), while EEG decodability stays flat and noisy (16-24%) across the entire −0.8s to +2.6s window, never clearly leading EMG. This is a **null result relative to the literature-motivated hypothesis** in the notebook (movement-related cortical activity preceding EMG onset) — most likely because a genuinely reliable version of this analysis needs more than one subject/session and full ICA-based EEG denoising, neither of which this reduced run had. Reported as-is rather than reframed to match the expectation.

### Mandatory control: EEG on pre-go-cue-only windows

18.3% accuracy vs. 16.7% chance — barely above chance. Weak evidence, consistent with (not strongly supporting) "not just EMG bleed-through," but nowhere near a confident result. A simplified EOG-linear-regression control was used in place of full ICA (fitting ICA reliably needs more trials/time than this session's budget allowed) — see `regress_out_eog()` in the runner script for exactly what it does and doesn't remove.

### Honest takeaway

This run demonstrates the pipeline runs correctly end-to-end on real data and reproduces one of the notebook's two headline predictions (EMG dominates naive fusion; degradation reveals a real crossover) — but not the other (EEG leading EMG in time), and the EEG branch itself underperformed enough to also break the gated-fusion architecture. Scaling to more subjects/sessions with full ICA preprocessing is the natural next step and would very plausibly change several of these numbers, especially the EEG-only and timing results.
