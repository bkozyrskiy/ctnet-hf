# Frozen first-release evidence

These files record the paper-compatible BNCI2014-001 evaluation used for the
first CTNet-HF release. They are intentionally tracked separately from the
ignored training outputs and model weights.

The protocol trained one subject-specific model for each of nine subjects with
seeds 0, 1, and 2. Session `0train` supplied training and validation data;
session `1test` was held out for final evaluation. The grand mean accuracy was
0.77533 and mean Cohen's kappa was 0.70045. Variation across subjects was much
larger than the observed variation across these three seeds.

- `results/bnci2014_001_three_seed_results.csv`: all 27 checkpoint results
- `results/summary_by_seed.csv`: subject mean and SD for each seed
- `results/summary_by_subject.csv`: seed mean and SD for each subject
- `results/summary_overall.csv`: aggregate subject and seed variation
- `results/manifest.json`: SHA-256 provenance for source and generated tables

The `bundle_path` column is a portable logical path; weights remain outside the
Git repository and are intended to be published as separate Hugging Face model
repositories. Detailed protocol and bundle verification notes are in
[`benchmarks/RELEASE_RESULTS.md`](../benchmarks/RELEASE_RESULTS.md).

Regenerate the tables with:

```bash
python -m benchmarks.summarize_release \
  --input-dir artifacts/release/benchmarks \
  --output-dir artifacts/release/results \
  --seeds 0 1 2
```
