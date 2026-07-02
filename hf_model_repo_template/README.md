---
license: mit
library_name: transformers
metrics:
  - accuracy
tags:
  - eeg
  - brain-computer-interface
  - motor-imagery
  - ct-net
  - pytorch
  - transformers
---

# CTNet for [DATASET] — [CHECKPOINT SCOPE]

> This file is a writing template. Replace every bracketed field. The release
> runner normally generates a checkpoint-specific card automatically.

This repository contains a [subject-specific/cross-subject] CTNet checkpoint
trained on [TRAINING DATA] and evaluated on [HELD-OUT DATA]. It has not been
validated outside [EXACT SCOPE].

## Model details

- Architecture: [paper-compatible CTNet or compact CTNet]
- Parameters: [COUNT]
- Input: [CHANNELS] EEG channels × [SAMPLES] samples at [RATE] Hz
- Labels: [ID-TO-LABEL MAPPING]
- Source: https://github.com/bkozyrskiy/ctnet-hf
- Paper: https://doi.org/10.1038/s41598-024-71118-7

## Intended use

Describe the narrow research use supported by the evaluation. State explicitly
that the model is not a medical device and is not validated for safety-critical
BCI control, diagnosis, treatment, identity inference, or untested subjects.

## Input and preprocessing contract

- Shape: `(batch_size, [CHANNELS], [SAMPLES])`
- Sampling rate: [RATE] Hz
- Units: [VOLTS/MICROVOLTS/NORMALIZED]
- Trial window and event alignment: [EXACT WINDOW]
- Acquisition and software filtering: [EXACT FILTERS]
- Reference/montage: [REFERENCE]
- Channel order: [ORDERED CHANNEL LIST]
- Normalization: [METHOD], fitted on [TRAINING SUBSET ONLY]

The generated `preprocessor_config.json` must contain the exact channel order
and fitted normalization statistics. Do not publish approximate placeholders.

## Usage

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
trial = np.load("trial.npy", allow_pickle=False)
with torch.no_grad():
    probabilities = torch.softmax(
        model(**processor(trial, return_tensors="pt")).logits, dim=-1
    )[0]
print(model.config.id2label[int(probabilities.argmax())])
```

## Training

Document the dataset version, train/validation split, leakage controls,
augmentation, optimizer, learning rate, batch size, epoch selection, random
seed, software versions, hardware, and wall time. The generated
`training_metadata.json` should contain the machine-readable values.

## Evaluation

Describe the held-out split and metrics. Report per-subject results where
appropriate, variation across training seeds, and any model-selection choices.

| Scope | Seed | Accuracy | Cohen's kappa |
|---|---:|---:|---:|
| [SUBJECT/COHORT] | [SEED] | [VALUE] | [VALUE] |

Do not present an architecture benchmark obtained by retraining several models
as though it were the score of one uploaded checkpoint.

## Limitations and risks

Document subject, session, montage, acquisition, population, artifact, and
dataset shift. Note that aggregate accuracy may hide class-specific failures
and that EEG recordings require appropriate privacy and consent controls.

## Reproducibility files

- `config.json`
- `model.safetensors`
- `preprocessor_config.json`
- `training_metadata.json`
- `export_reload_equivalence.json`
- `release_manifest.json`

## License and citation

State which license applies to the original implementation and weights, and
keep dataset terms separate. Include `LICENSE`, `THIRD_PARTY_NOTICES.md`, and a
complete citation for the CTNet paper.
