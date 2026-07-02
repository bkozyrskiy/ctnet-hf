# CTNet-HF: Hugging Face-compatible CTNet for EEG motor imagery classification

`ctnet-hf` provides Hugging Face Transformers implementations of the published CTNet architecture and the repository's earlier compact CTNet-style variant. It is intended for training, evaluation, packaging, and later publishing your own CTNet checkpoints.

This repository currently provides the CTNet architecture only. It does not provide pretrained weights. EEG decoding models are often dataset-, montage-, session-, and subject-dependent; users should train or fine-tune the model under their own protocol.

## What this is

- A `transformers`-compatible Python package named `ctnet_hf`
- A paper-compatible CTNet architecture plus the earlier compact variant
- A repository scaffold suitable for later conversion into a Hugging Face model repository
- Tests, examples, and a lightweight architecture-only template model card

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

Standard local save/load works with the package API:

```python
model.save_pretrained("./ctnet-local")

from transformers import AutoConfig, AutoModelForSequenceClassification

config = AutoConfig.from_pretrained("./ctnet-local", trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    "./ctnet-local",
    trust_remote_code=True,
)
```

The repository also includes `hf_model_repo_template/` for an architecture-only Hugging Face model repo layout that intentionally excludes weights.

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

## License and citation

This implementation was written as a clean-room Hugging Face port and does not ship copied upstream training checkpoints.

The paper-compatible implementation follows:

- Zhao et al., "CTNet: a convolutional transformer network for EEG-based motor imagery classification," Scientific Reports 14, 20237 (2024): https://doi.org/10.1038/s41598-024-71118-7
- Authors' released implementation: https://github.com/snailpt/CTNet

No upstream training checkpoints are redistributed here.
