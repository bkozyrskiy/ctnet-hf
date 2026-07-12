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

TrainTest results include accuracy, Cohen's kappa, seed, and selected epoch.
Per-seed summaries report mean and sample standard deviation across subjects.
Benchmark outputs remain local under `artifacts/` and are not part of the
untrained Hugging Face model repository.

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

## Pooled Hugging Face checkpoint

`scripts/train_hf` trains a single model on the complete training sessions of
all selected BNCI2014-001 subjects. Unlike the subject-specific benchmark, it
does not reserve 30% of the training trials: all `0train` trials enter the
optimizer and global Z-score fit. S&R sources are constrained to the same
subject and class, following the upstream repository's cross-subject
augmentation guidance.

At every epoch, the runner computes accuracy and Cohen's kappa separately for
each `1test` session. It selects the highest unweighted mean subject accuracy
(or mean kappa with `--selection-metric cohen_kappa`) and writes an
upload-ready Transformers bundle to `artifacts/trained_hf_model`.

This deliberately requested selection protocol consumes the nominal test
sessions during tuning. Its output must not be described as having an unbiased
held-out test score.
