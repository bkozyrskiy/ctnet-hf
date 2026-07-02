"""Render checkpoint-specific Hugging Face model cards for benchmark exports."""

from __future__ import annotations

from typing import Any


def render_bnci2014_001_card(
    *,
    subject: int,
    seed: int,
    train_session: str,
    test_session: str,
    accuracy: float,
    cohen_kappa: float,
    best_epoch: int,
    channel_names: list[str] | None,
    id2label: dict[int, str],
    parameter_count: int,
    training_seconds: float,
) -> str:
    """Return a complete, deliberately conservative model card."""
    channels = ", ".join(channel_names or ["not recorded"])
    labels = ", ".join(
        f"`{index}` = `{label}`" for index, label in sorted(id2label.items())
    )
    return f"""---
library_name: transformers
license: mit
metrics:
  - accuracy
tags:
  - eeg
  - brain-computer-interface
  - motor-imagery
  - bnci2014-001
  - ct-net
  - pytorch
---

# CTNet for BNCI2014-001 — subject {subject}, seed {seed}

This repository contains a **subject-specific** four-class motor-imagery
checkpoint. It was trained on subject {subject}, session `{train_session}`, and
evaluated once on the same subject's held-out session `{test_session}`. It has
not been validated on other people, datasets, montages, or acquisition systems.

## Model details

- Architecture: paper-compatible CTNet
- Parameters: {parameter_count:,}
- Inputs: 22-channel EEG trials
- Outputs: feet, left hand, right hand, or tongue motor imagery
- Labels: {labels}
- Implementation: https://github.com/bkozyrskiy/ctnet-hf
- Reference: Zhao et al., *Scientific Reports* 14, 20237 (2024),
  https://doi.org/10.1038/s41598-024-71118-7

This is a research checkpoint, not a foundation model or a medical device.

## Intended use

The checkpoint is intended for reproducible research on subject {subject} of
BNCI2014-001 and for studying the CTNet architecture. It may be used as an
initialization for new experiments only after adapting and validating the
entire acquisition and preprocessing contract.

It must not be used for diagnosis, treatment, user assessment, identity
inference, or safety-critical BCI control. Performance on this subject does not
establish performance for another person.

## Input and preprocessing contract

- Shape: `(batch_size, 22, 1000)` or one trial as `(22, 1000)`
- Sampling rate: 250 Hz
- Values: microvolts, matching the MOABB array returned for BNCI2014-001
- Window: 0–4 s cue-relative, corresponding to 2–6 s in the original trial
- Software filtering: none beyond the dataset's acquisition filtering
- Channel order: {channels}
- Normalization: global Z-score fitted only on the training subset; exact
  statistics are serialized in `preprocessor_config.json`

The channel order, units, sampling rate, and time alignment are part of the
model. Silently changing any of them invalidates the reported evaluation.

## Usage

This repository contains custom model and feature-extractor code. Review it,
then pin `revision` to a Hub commit hash when enabling remote code.

```python
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForSequenceClassification

repo_id = "YOUR_ORG/YOUR_MODEL"
revision = "PINNED_HUB_COMMIT"
processor = AutoFeatureExtractor.from_pretrained(
    repo_id, trust_remote_code=True, revision=revision
)
model = AutoModelForSequenceClassification.from_pretrained(
    repo_id, trust_remote_code=True, revision=revision
).eval()

trial = np.load("trial.npy", allow_pickle=False)  # (22, 1000), microvolts
with torch.no_grad():
    logits = model(**processor(trial, return_tensors="pt")).logits
    probabilities = torch.softmax(logits, dim=-1)[0]
label = model.config.id2label[int(probabilities.argmax())]
print(label, probabilities.tolist())
```

## Training

The model used session `{train_session}` (288 trials). A fixed, stratified 30%
validation split selected epoch {best_epoch} by validation loss. Global Z-score
statistics and S&R augmentation sources came only from the remaining clean
training subset. Training used 1,000 maximum epochs, Adam with learning rate
0.001 and betas (0.5, 0.999), batch size 72, no weight decay, eight S&R
segments, and augmentation factor 3. The recorded training-and-evaluation wall
time was {training_seconds:.1f} seconds. See `training_metadata.json` for the
complete configuration, software versions, hardware, and source revision.

## Evaluation

Evaluation used all 288 trials from the held-out session `{test_session}`.

| Subject | Seed | Accuracy | Cohen's kappa |
|---:|---:|---:|---:|
| {subject} | {seed} | {accuracy:.4f} | {cohen_kappa:.4f} |

This row measures one trained subject-specific checkpoint. It is not a
cross-subject or cross-dataset score. The source repository's frozen release
results report all nine subjects and three training seeds without selecting the
best seed.

## Limitations and risks

- EEG varies substantially across people, sessions, hardware, referencing,
  electrode placement, artifacts, and cognitive state.
- BNCI2014-001 is a controlled research dataset and does not represent all
  populations or real-world operating conditions.
- The model can be confidently wrong and provides no calibrated safety or
  clinical guarantee.
- Aggregate accuracy can hide class-specific errors; inspect a confusion matrix
  for any new application.
- EEG is sensitive human data. Apply appropriate consent, privacy, retention,
  and governance requirements to any new recordings.

## Reproducibility files

- `config.json`: exact model configuration and label mapping
- `model.safetensors`: weights in non-pickle format
- `preprocessor_config.json`: channel contract and fitted normalization
- `training_metadata.json`: protocol, dependencies, hardware, and metrics
- `export_reload_equivalence.json`: held-out reload-equivalence evidence
- `release_manifest.json`: file sizes and SHA-256 checksums

## License and citation

Original CTNet-HF source and these released weights are provided under the MIT
License. The BNCI2014-001 data are not redistributed and remain subject to
their own terms. See `LICENSE` and `THIRD_PARTY_NOTICES.md`.

```bibtex
@article{{zhao2024ctnet,
  title={{CTNet: a convolutional transformer network for EEG-based motor imagery classification}},
  author={{Zhao, Wei and Jiang, Xiaolu and Zhang, Baocan and Xiao, Shixiao and Weng, Sujun}},
  journal={{Scientific Reports}},
  volume={{14}},
  pages={{20237}},
  year={{2024}},
  doi={{10.1038/s41598-024-71118-7}}
}}
```
"""
