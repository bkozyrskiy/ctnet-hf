---
library_name: transformers
license: mit
tags:
  - eeg
  - brain-computer-interface
  - motor-imagery
  - ct-net
  - pytorch
  - custom-code
---

# CTNet-HF: untrained CTNet for subject-specific EEG training

> **This model is not pretrained.** `model.safetensors` contains a deterministic
> random initialization only, allowing the CTNet architecture to be downloaded
> and instantiated with one standard Transformers call. Train all parameters on
> your own EEG training data before using the model for classification.

CTNet-HF implements the convolutional Transformer architecture introduced in
[Zhao et al. (2024)](https://doi.org/10.1038/s41598-024-71118-7) behind the
Hugging Face Transformers API.

- Implementation: https://github.com/bkozyrskiy/ctnet-hf
- Full architecture reference: https://github.com/bkozyrskiy/ctnet-hf/blob/main/docs/ARCHITECTURE.md
- Original CTNet code: https://github.com/snailpt/CTNet

## Load the model

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "bkozyrskiy/ctnet-hf",
    trust_remote_code=True,
)
```

Because the repository contains custom modeling code, inspect the files before
enabling `trust_remote_code`. For reproducible use, pin `revision` to a Hub
commit hash.

## What you receive

The default model is configured for four-class motor imagery with:

- input shape `(batch_size, 22, 1000)`;
- sampling rate metadata of 250 Hz;
- 27,284 trainable parameters;
- deterministic random initialization generated with PyTorch seed 0;
- generic output labels `LABEL_0` through `LABEL_3`.

There are no fitted preprocessing statistics and no subject data in this
repository. Fit normalization on your own training split; do not estimate it
from validation or test trials.

## Adapt CTNet to custom EEG data

The stored random initialization has shapes for 22 channels, 1,000 samples, and
four classes. If any of those dimensions changes, download the configuration
but create fresh weights with `from_config()`. Do not try to reuse or reshape
the default tensors: the spatial convolution depends on `n_channels`, and the
flattened classifier depends on both `n_times` and `num_labels`.

This example creates a two-class model for eight channels and two-second
windows at 250 Hz:

```python
from transformers import AutoConfig, AutoModelForSequenceClassification

repo_id = "bkozyrskiy/ctnet-hf"
channel_names = ["F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
labels = ["left", "right"]
sampling_rate = 250
window_seconds = 2.0

config = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
config.n_channels = len(channel_names)
config.n_times = round(sampling_rate * window_seconds)  # 500 samples
config.sampling_rate = sampling_rate
config.num_labels = len(labels)
config.id2label = dict(enumerate(labels))
config.label2id = {label: index for index, label in enumerate(labels)}

# Optional metadata: saved with config.json, but not enforced by the model.
config.channel_names = channel_names
config.window_seconds = window_seconds
config.input_unit = "microvolts"

model = AutoModelForSequenceClassification.from_config(
    config,
    trust_remote_code=True,
)
```

The resulting model expects exactly `(batch_size, 8, 500)` and returns two
logits. It is freshly initialized and must be trained from scratch.

### Core data settings

| Setting | Meaning | What to check |
|---|---|---|
| `n_channels` | Number of EEG channels | Must equal `input_values.shape[1]`; keep one documented channel order for every trial. |
| `n_times` | Samples in one EEG window | Must equal `input_values.shape[2]`; crop or pad variable-length trials before the model. |
| `sampling_rate` | Sampling-rate metadata in Hz | The model does not resample data. Resample externally when needed. |
| `num_labels` | Number of output classes | Labels supplied during training must be integers from `0` to `num_labels - 1`. |
| `id2label` / `label2id` | Human-readable class mapping | Define both mappings and keep them consistent with training targets. |

In EEG literature, an *epoch* often means one time window. Its duration is:

```text
window_seconds = n_times / sampling_rate
```

For example, 1,000 samples at 250 Hz is a four-second EEG window. This is
different from the number of training epochs, which is controlled by the
training loop rather than `CtnetConfig`.

### Sampling rate and temporal settings

`sampling_rate` is descriptive metadata. Convolution kernels and pooling
windows are expressed in samples, not seconds, so merely changing
`sampling_rate` does not preserve the published temporal receptive fields.

The safest way to follow the paper architecture is to resample EEG to 250 Hz
and keep the default temporal settings. If you intentionally use another rate,
review all sample-based fields:

- `filter_time_length`;
- `second_filter_time_length`;
- `pool_time_length` and `pool_time_stride`;
- `second_pool_time_length` and `second_pool_time_stride`.

For the paper path, the number of tokens after the two pools is:

```text
first_length = floor((n_times - pool_time_length) / pool_time_stride) + 1
token_count  = floor((first_length - second_pool_time_length)
                     / second_pool_time_stride) + 1
```

`token_count` must be at least 1 and must not exceed
`max_position_embeddings`. The model validates these constraints when it is
constructed.

### Architecture settings

Most custom datasets only require changing the core data settings above. These
advanced fields alter CTNet itself:

| Setting | Paper default | Effect |
|---|---:|---|
| `n_filters_time` | 8 | Number of first temporal convolution filters. |
| `filter_time_length` | 64 | First temporal kernel length in samples. |
| `depth_multiplier` | 2 | Spatial features per temporal filter. |
| `second_filter_time_length` | 16 | Local refinement kernel length in samples. |
| `att_depth` | 6 | Number of Transformer blocks. |
| `att_heads` | 2 | Number of attention heads. |
| `att_dim` | 16 | Token embedding dimension. |
| `att_mlp_dim` | 64 | Transformer feed-forward width. |
| `dropout` | 0.5 | Convolution and Transformer dropout. |
| `positional_dropout` | 0.1 | Dropout after learned positions are added. |
| `classifier_dropout` | 0.5 | Dropout immediately before the classifier. |
| `max_position_embeddings` | 100 | Capacity of the learned position table. |

The paper-compatible path requires:

```text
att_dim == n_filters_time * depth_multiplier
att_dim % att_heads == 0
```

Changing these fields creates a different-sized CTNet and therefore requires a
fresh initialization with `from_config()`.

## Architecture

For the default 22-channel, 1,000-sample input:

| Stage | Operation | Output shape |
|---|---|---:|
| Input | EEG trial | `(B, 22, 1000)` |
| Temporal features | Same-padded temporal convolution, 8 filters | `(B, 8, 22, 1000)` |
| Spatial features | Depthwise convolution across 22 channels | `(B, 16, 1, 1000)` |
| Local refinement | Two 8× pools around a temporal convolution | `(B, 16, 1, 15)` |
| Tokens | Transpose and add learned positions | `(B, 15, 16)` |
| Global context | Six post-norm Transformer blocks, two heads | `(B, 15, 16)` |
| Classifier | CNN residual, flatten, linear head | `(B, 4)` |

Attention follows the released CTNet implementation, including scaling by
`sqrt(embedding_dim)`. The public input is a normal three-dimensional EEG
tensor; the singleton convolution dimension is added internally.

## Preprocess and train on your own data

The model does not filter, resample, re-reference, reorder channels, reject
artifacts, crop windows, or normalize EEG. Perform those operations in the data
pipeline and apply the exact same transformation to validation and test data.

At minimum:

1. choose and record a fixed channel order;
2. resample every recording to one sampling rate;
3. extract fixed-length windows with one event-relative alignment;
4. use one amplitude unit consistently;
5. fit normalization on the training split only;
6. reuse those fitted training statistics for validation, test, and inference.

The model accepts `float32` EEG tensors and integer class labels:

```python
import torch

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model.train()

for input_values, labels in train_loader:
    # input_values: (batch, config.n_channels, config.n_times)
    # labels:       (batch,), integer class indices
    optimizer.zero_grad()
    loss = model(input_values=input_values, labels=labels).loss
    loss.backward()
    optimizer.step()
```

Learning rate, batch size, number of training epochs, validation strategy,
augmentation, class balancing, and early stopping are training choices; they
are not architecture fields in `CtnetConfig`. If class-weighted loss is needed,
compute cross-entropy from `model(input_values=...).logits` in the training
loop instead of passing `labels` directly.

After training, save the adapted configuration and learned weights together:

```python
model.save_pretrained("./my-trained-ctnet", safe_serialization=True)
```

Also document the channel names, reference, sampling rate, event-relative
window, amplitude unit, filters, normalization statistics, label mapping, and
train/validation/test split. The model cannot infer that acquisition contract
from the tensor alone.

### Common errors

| Error | Cause | Fix |
|---|---|---|
| `Expected ... EEG channels` | Input montage and `config.n_channels` differ. | Reorder/select channels consistently or create the model with the correct count. |
| `Expected ... time samples` | Window length and `config.n_times` differ. | Crop, pad, or create a configuration matching the intended window. |
| Weight-size mismatch in `from_pretrained()` | Channels, time length, or classes were changed while loading default-shape weights. | Load the config, modify it, and call `from_config()` instead. |
| Pooling collapsed the sequence | The EEG window is too short for the configured pools. | Use a longer window or reduce pool lengths/strides. |
| Too many tokens for learned positions | `token_count > max_position_embeddings`. | Increase position capacity or use stronger temporal pooling. |

## Intended use

This repository is intended for researchers who want to train CTNet on their
own subject-specific motor-imagery data or inspect the architecture through the
Transformers API.

It is not a trained classifier, foundation model, medical device, clinical
tool, or claim of cross-subject generalization. EEG performance depends on the
participant, session, montage, reference, hardware, preprocessing, and task.

## License and citation

CTNet-HF is available under the MIT License. If you use the architecture, cite
the original paper:

```bibtex
@article{zhao2024ctnet,
  title={CTNet: a convolutional transformer network for EEG-based motor imagery classification},
  author={Zhao, Wei and Jiang, Xiaolu and Zhang, Baocan and Xiao, Shixiao and Weng, Sujun},
  journal={Scientific Reports},
  volume={14},
  pages={20237},
  year={2024},
  doi={10.1038/s41598-024-71118-7}
}
```
