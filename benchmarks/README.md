# MOABB benchmarks

The single runner in `scripts/` evaluates CTNet on BNCI 2014-001. It sets local cache paths and writes raw and summary CSVs under `artifacts/benchmarks/`.

## Commands

Quick one-subject, one-epoch smoke test:

```bash
scripts/run
```

Full paper-compatible subject-specific benchmark:

```bash
scripts/run all
```

Three-seed release benchmark, bundle export, and equivalence checks:

```bash
scripts/run release --device cuda
```

Custom invocation:

```bash
scripts/run custom --subjects 1 --epochs 100 --device cuda
```

`PYTHON` selects an interpreter and `OUT` selects the result directory:

```bash
PYTHON=/path/to/python OUT=/tmp/results scripts/run all
```

The active environment must provide MOABB and the project dependencies. The convenience extra is `pip install -e .[benchmark]`.

## Paper protocol

The default `--architecture paper` path follows the published BCI IV-2a subject-specific setup:

- session `0train` trains and session `1test` tests;
- the 2-6 second task interval is cropped to exactly 1000 samples;
- no additional software band-pass filter is applied;
- the paper CTNet uses 8/16 convolution filters, two 8x pools, 15 tokens, 6 Transformer layers, and 2 heads;
- global Z-score statistics come only from the clean training subset;
- a fixed stratified 30% validation split selects the lowest-loss checkpoint;
- training uses S&R with 8 segments and augmentation factor 3;
- Adam uses learning rate `0.001`, betas `(0.5, 0.999)`, and no weight decay;
- the default run uses 1000 epochs and configured batch size 72; after the clean 30% split, each optimizer batch contains 51 real trials plus 216 synthetic trials.

The authors' paper-era code had overlapping train/validation slices and generated augmentation from validation trials. This runner deliberately fixes both leaks, so it is paper-compatible rather than bug-for-bug identical.

Use `--architecture compact` to select the repository's previous compact encoder. Its hyperparameters can be supplied through the same CLI.

## Results

Paper and compact outputs have distinct names, for example:

```text
moabb_BNCI2014_001_MotorImagery_TrainTest_paper_seed-0.csv
moabb_BNCI2014_001_MotorImagery_TrainTest_paper_seed-0_summary.csv
```

TrainTest results include accuracy, Cohen's kappa, seed, selected epoch, bundle
path, and the maximum export/reload probability difference. Per-seed summaries
report mean and sample standard deviation across subjects. The release runner
also writes three-seed summaries under `artifacts/release/results/`, separating
variation across subjects from variation across seeds within a subject.

Each exported bundle includes:

- `model.safetensors` and `config.json`;
- `preprocessor_config.json` with the fitted training-only Z-score, channel
  order, input units, sampling rate, and trial window;
- `training_metadata.json` with all hyperparameters and package versions;
- `export_reload_equivalence.json` with the held-out prediction check;
- `release_manifest.json` with SHA-256 checksums;
- the MIT license, third-party/data notices, and a subject-specific model card.

One raw held-out trial is kept outside the publishable bundle under
`artifacts/release/verification/` for the clean-environment inference gate.
The result tables used for the first card are frozen under `release/results/`.

Existing trained bundles can be upgraded to the current card, preprocessing
auto-class, legal files, and checksum manifest without retraining:

```bash
python -m benchmarks.refresh_release_bundles \
  --bundle-root artifacts/release/bundles \
  --results release/results/bnci2014_001_three_seed_results.csv
```

## Useful options

```text
--architecture paper          paper-compatible CTNet (default)
--architecture compact        earlier compact encoder
--epochs N                    training epochs (default 1000)
--validation-ratio 0.3        clean validation fraction
--augmentation-factor 3       S&R synthetic trials per raw batch ratio
--evaluation TrainTest        0train -> 1test (default)
--evaluation WithinSession    MOABB within-session evaluation
--evaluation CrossSession     MOABB cross-session folds, not cross-subject LOSO
--device cuda                 train on GPU
```
