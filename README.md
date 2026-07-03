# CTNet-HF

Hugging Face Transformers implementation of CTNet for EEG motor-imagery
classification.

The public Hugging Face model is an **untrained initialization** intended to be
trained on the user's own EEG data. It is not pretrained on a particular
person and it does not contain subject-specific normalization statistics.

## Install

```bash
pip install -e .
```

## Create the architecture locally

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
print(model(input_values=x).logits.shape)  # torch.Size([2, 4])
```

## Load from Hugging Face

After `bkozyrskiy/ctnet-hf` is published, Transformers downloads and creates the
model in one call:

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "bkozyrskiy/ctnet-hf",
    trust_remote_code=True,
)
```

The stored weights are a deterministic random initialization so that
`from_pretrained()` works normally. Train all parameters before using the model
for classification.

To download the repository without loading it:

```bash
hf download bkozyrskiy/ctnet-hf --local-dir ctnet-hf-model
```

## Train on your own EEG

Inputs have shape `(batch_size, n_channels, n_times)`. Fit any normalization
only on your training split, then use an ordinary PyTorch training loop:

```python
import torch

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model.train()

for input_values, labels in train_loader:
    optimizer.zero_grad()
    loss = model(input_values=input_values, labels=labels).loss
    loss.backward()
    optimizer.step()
```

The Hub model defaults to the BCI Competition IV-2a layout: 22 channels,
1,000 samples at 250 Hz, and four output classes. For another acquisition
layout, construct a matching `CtnetConfig` before training.

## Architecture

`architecture="paper"` implements the CTNet topology described by Zhao et al.:

- temporal and depthwise spatial convolutions;
- a second local temporal feature convolution;
- two temporal pooling stages producing 15 tokens;
- learned positional embeddings;
- six post-norm Transformer blocks with two attention heads;
- a residual connection from positioned CNN tokens;
- a flattened 240-value classification representation.

The default four-class paper model has 27,284 trainable parameters. See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for layer-by-layer shapes, the
compact variant, and differences from the authors' implementation.

## Hugging Face bundle

[`hf_model/`](hf_model/) is the complete directory intended for the single
Hugging Face model repository. Regenerate its configuration, remote code, and
deterministic initialization with:

```bash
python examples/export_hf_model.py
```

Do not upload anything from `artifacts/`. That directory contains local
benchmark outputs and subject-specific experimental checkpoints only.

### Publish the model repository

```bash
python -m pip install -U huggingface_hub
hf auth login
python examples/export_hf_model.py

hf repo create bkozyrskiy/ctnet-hf \
  --repo-type model \
  --private \
  --exist-ok

hf upload bkozyrskiy/ctnet-hf hf_model . \
  --repo-type model \
  --commit-message "Publish untrained CTNet architecture"
```

Review the private repository at
`https://huggingface.co/bkozyrskiy/ctnet-hf`, verify the loading example, and
then change its visibility to public in the repository settings.

## Optional benchmark

The MOABB runner remains available for local experiments:

```bash
pip install -e .[benchmark]
scripts/run
```

It writes results under ignored `artifacts/`; it does not create Hugging Face
repositories.

## Paper and citation

- Wei Zhao, Xiaolu Jiang, Baocan Zhang, Shixiao Xiao, and Sujun Weng,
  “CTNet: a convolutional transformer network for EEG-based motor imagery
  classification,” *Scientific Reports* 14, 20237 (2024).
  https://doi.org/10.1038/s41598-024-71118-7
- Authors' implementation: https://github.com/snailpt/CTNet
- Software citation: [`CITATION.cff`](CITATION.cff)

CTNet-HF is MIT licensed. No upstream checkpoints or EEG recordings are
redistributed.
