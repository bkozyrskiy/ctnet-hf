# CTNet-HF: Hugging Face-compatible CTNet for EEG motor imagery classification

`ctnet-hf` provides Hugging Face Transformers implementations of the published CTNet architecture and the repository's earlier compact CTNet-style variant. It supports training, evaluation, and complete model-plus-preprocessor release bundles.

This source repository does not commit pretrained weights. EEG decoding models are dataset-, montage-, session-, and subject-dependent; use only a released checkpoint whose documented input contract matches your data.

## What this is

- A `transformers`-compatible Python package named `ctnet_hf`
- A paper-compatible CTNet architecture plus the earlier compact variant
- A release path that exports weights, custom code, and fitted preprocessing together
- Tests, a comprehensive model-card template, and frozen three-seed evaluation evidence

## What this is not

- Not a pretrained EEG foundation model
- Not a benchmark claim or leaderboard submission
- Not a promise of cross-subject or cross-dataset generalization
- Not a replacement for dataset-specific preprocessing and evaluation design

## Installation

```bash
pip install -e .
```

For development tooling:

```bash
pip install -e .[dev]
```

## Quick start

```python
import torch
from ctnet_hf import CtnetConfig, CtnetForEEGClassification

config = CtnetConfig(
    architecture="paper",
    n_channels=22,
    n_times=1000,
    sampling_rate=250,
    num_labels=4,
)

model = CtnetForEEGClassification(config)
x = torch.randn(2, 22, 1000)
outputs = model(input_values=x)

print(outputs.logits.shape)  # torch.Size([2, 4])
```

## Input format

The external input shape is:

```python
(batch_size, n_channels, n_times)
```

Users do not need to reshape EEG into image-like tensors such as `(batch, 1, channels, time)`. The model handles internal reshaping.

## Configuration

The default configuration matches BCI Competition IV 2a style dimensions:

- `n_channels=22`
- `n_times=1000`
- `sampling_rate=250`
- `num_labels=4`

Set `architecture="paper"` for Zhao et al.'s CTNet: the three convolution stages, learned positions, six post-norm Transformer blocks, CNN residual, and flattened classifier. The default `architecture="compact"` preserves checkpoints made with this repository's earlier configurable encoder.

Both variants remain dataset-configurable through `CtnetConfig`, including EEG dimensions, convolution and pooling parameters, dropout, and Transformer dimensions.

## Save/load

Standard local save/load works with the package API. For publishable checkpoints,
use `export_huggingface_bundle(model, preprocessor, output_dir)` so the model and
its fitted input contract cannot drift apart.

```python
model.save_pretrained("./ctnet-local")

from transformers import AutoConfig, AutoModelForSequenceClassification

config = AutoConfig.from_pretrained("./ctnet-local", trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    "./ctnet-local",
    trust_remote_code=True,
)
```

A usable EEG checkpoint also needs its fitted preprocessing statistics:

```python
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForSequenceClassification

repo_id = "YOUR_ORG/YOUR_MODEL"  # or a local exported bundle
revision = "PINNED_HUB_COMMIT"
processor = AutoFeatureExtractor.from_pretrained(
    repo_id, trust_remote_code=True, revision=revision
)
model = AutoModelForSequenceClassification.from_pretrained(
    repo_id, trust_remote_code=True, revision=revision
).eval()
trial = np.load("trial.npy")  # raw shape and units must match the model card

with torch.no_grad():
    probabilities = torch.softmax(model(**processor(trial)).logits, dim=-1)
prediction = model.config.id2label[int(probabilities.argmax())]
```

The repository also includes `hf_model_repo_template/README.md` as a detailed
writing template. Configuration and preprocessing JSON are intentionally not
templated: they must be generated from the exact trained checkpoint.

## Training your own CTNet

The expected workflow is:

1. preprocess EEG externally using your own protocol;
2. instantiate `CtnetConfig` for your dataset layout;
3. train `CtnetForEEGClassification` on your own data;
4. save your checkpoint locally or publish it later.

The `examples/train_minimal_dummy.py` script is only a smoke test and produces meaningless dummy weights.

For local benchmarking, this repo includes a paper-compatible MOABB protocol under `benchmarks/`. It uses the competition train/test sessions, 1000-sample unfiltered trials, training-only S&R augmentation, clean stratified validation, and best-loss checkpoint selection. The known overlapping validation split in the released upstream code is intentionally not reproduced.

```bash
scripts/run
```

See `benchmarks/README.md` for the supported MOABB benchmark options.
The frozen first-release tables are tracked under `release/results/`.

## First-release checks

The release runner executes BNCI2014-001 subjects 1–9 with seeds 0, 1, and 2.
For every subject/seed it exports the model, fitted training-only Z-score, channel
contract, training manifest, and one export/reload equivalence report.

```bash
PYTHON=/path/to/python scripts/run release --device cuda
```

After choosing the checkpoint to publish, verify a held-out real trial in a
throwaway environment. `MODEL_OR_REPO_ID` may be a local bundle before upload
or the Hugging Face repository id after upload:

```bash
scripts/check_clean_inference \
  MODEL_OR_REPO_ID \
  artifacts/release/verification/BNCI2014_001/paper/seed-0/subject-1/real_test_trial.npy \
  EXPECTED_PREDICTED_LABEL
```

## License and citation

This implementation was written as a clean-room Hugging Face port and does not ship copied upstream training checkpoints.

The paper-compatible implementation follows:

- Wei Zhao, Xiaolu Jiang, Baocan Zhang, Shixiao Xiao, and Sujun Weng,
  "CTNet: a convolutional transformer network for EEG-based motor imagery
  classification," Scientific Reports 14, 20237 (2024):
  https://doi.org/10.1038/s41598-024-71118-7
- Authors' released implementation: https://github.com/snailpt/CTNet

No upstream training checkpoints or BNCI2014-001 recordings are redistributed
here. See `THIRD_PARTY_NOTICES.md` and `CITATION.cff`.
