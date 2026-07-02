# BNCI2014-001 first-release verification

Run date: 2026-07-02

The paper CTNet protocol was run for 1,000 epochs on all nine subjects with
training seeds 0, 1, and 2. Each row uses session `0train` for training and
validation and session `1test` for the final held-out score. The run used
Python 3.14.6, PyTorch 2.12.1, Transformers 4.57.6, MOABB 1.5.0, MNE 1.12.1,
NumPy 2.4.6, and scikit-learn 1.9.0. Deterministic PyTorch algorithms were
enabled.

## Results

| Seed | Accuracy mean | Accuracy subject SD | Kappa mean | Kappa subject SD |
|---:|---:|---:|---:|---:|
| 0 | 0.77199 | 0.10651 | 0.69599 | 0.14202 |
| 1 | 0.75656 | 0.13678 | 0.67541 | 0.18238 |
| 2 | 0.79745 | 0.11411 | 0.72994 | 0.15215 |

Across all three seeds, mean accuracy was 0.77533. The standard deviation
across the nine subject means was 0.11682, while the mean within-subject
standard deviation across seeds was 0.02794. Subject variation was therefore
about 4.2 times the observed training-seed variation under this protocol.

The corresponding Cohen's kappa values were 0.70045 overall, 0.15576 across
subject means, and 0.03725 mean within-subject seed standard deviation.

The raw 27 rows and subject/seed summaries are frozen under `release/results/`.
They can be regenerated under `artifacts/release/results/` by:

```bash
python -m benchmarks.summarize_release \
  --input-dir artifacts/release/benchmarks \
  --output-dir artifacts/release/results \
  --seeds 0 1 2
```

## Bundle equivalence

All 27 fitted estimators were exported as a Hugging Face model bundle with a
serialized `CtnetPreprocessor`. Each bundle was reloaded and evaluated over all
288 trials in its held-out test session. All 7,776 class predictions matched,
all probability arrays were within `rtol=1e-5, atol=1e-6`, and the observed
maximum absolute probability difference was exactly 0 on the training device.

Every bundle records its fitted training-only Z-score, subject, seed, channel
order, microvolt input unit, 250 Hz sampling rate, four-second cue-relative
window, label mapping, hyperparameters, selected epoch, package versions, and
held-out prediction hash.

## Clean-environment inference

`scripts/check_clean_inference` built the project wheel, created a new virtual
environment, installed CPU PyTorch 2.3.1 and Transformers 4.57.6 from scratch,
loaded the seed-0/subject-1 bundle, and ran one real BNCI2014-001 test trial.
The clean process reproduced the stored predicted class (`feet`); CPU and GPU
probabilities differed only at normal cross-device floating-point precision.

The local-directory gate is complete. After uploading the selected bundle, run
the same command with its Hugging Face repository id to exercise Hub download
as the final publication gate.
