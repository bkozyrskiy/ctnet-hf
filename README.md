# CTNet-HF: Hugging Face-compatible CTNet for EEG motor imagery classification

`ctnet-hf` provides an architecture-only Hugging Face Transformers port of CTNet for EEG motor imagery classification. It is intended as a clean starting point for training, fine-tuning, packaging, and later publishing your own CTNet checkpoints.

This repository currently provides the CTNet architecture only. It does not provide pretrained weights. EEG decoding models are often dataset-, montage-, session-, and subject-dependent; users should train or fine-tune the model under their own protocol.

## What this is

- A `transformers`-compatible Python package named `ctnet_hf`
- A configurable CTNet-style hybrid convolution + transformer model for EEG classification
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

The implementation keeps the architecture dataset-configurable through `CtnetConfig`, including:

- EEG dimensions and labels
- Temporal convolution parameters
- Pooling and dropout
- Transformer depth, head count, embedding size, and MLP width

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

This repo does not bundle dataset loaders or MOABB training flows yet. The expected workflow is:

1. preprocess EEG externally using your own protocol;
2. instantiate `CtnetConfig` for your dataset layout;
3. train `CtnetForEEGClassification` on your own data;
4. save your checkpoint locally or publish it later.

The `examples/train_minimal_dummy.py` script is only a smoke test and produces meaningless dummy weights.

## License and citation

This implementation was written as a clean-room Hugging Face port and does not ship copied upstream training checkpoints.

The architecture is inspired by the CTNet family of EEG motor imagery models. Before publishing a downstream model repository, you should verify the upstream paper/repository attribution details that match the exact CTNet variant you trained against.

Useful references used while preparing this scaffold:

- EEGNet paper: https://arxiv.org/abs/1611.08024
- EEG-TCNet paper: https://arxiv.org/abs/2006.00622
- Subject-specific encoders paper discussing CTNet baselines: https://arxiv.org/abs/2606.16462

## Roadmap

- Finalize exact CTNet paper/repository citation metadata
- Add training and evaluation utilities in a separate milestone
- Add a Hub-ready example after trained checkpoints exist
- Add optional preprocessing adapters for common EEG datasets
