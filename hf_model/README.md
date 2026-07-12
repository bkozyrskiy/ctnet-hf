---
library_name: transformers
license: mit
metrics:
  - accuracy
  - cohen_kappa
tags:
  - eeg
  - brain-computer-interface
  - motor-imagery
  - bnci2014-001
  - ct-net
  - pytorch
  - custom-code
datasets:
  - BNCI2014_001
---

# CTNet-HF trained on BNCI2014-001

This repository contains a trained Hugging Face Transformers implementation of
CTNet, the convolutional Transformer architecture introduced by
[Zhao et al. (2024)](https://doi.org/10.1038/s41598-024-71118-7).

The checkpoint was trained as one pooled model on all BNCI2014-001 training
sessions: session `0train` from subjects 1-9. It expects 22 EEG channels,
1,000 samples at 250 Hz, and microvolt values in the channel order recorded in
`preprocessor_config.json`.

- Implementation: https://github.com/bkozyrskiy/ctnet-hf
- Original CTNet code: https://github.com/snailpt/CTNet
- Dataset interface: MOABB `BNCI2014_001`

## Important evaluation caveat

Session `1test` was evaluated during training and used to select the checkpoint.
The selected checkpoint is epoch 400, chosen by mean subject accuracy.

The scores below are therefore **checkpoint-selection metrics on the nominal
test sessions, not unbiased held-out test metrics**.

## Performance on BNCI2014-001 session `1test`

| Subject | Accuracy | Cohen's kappa |
|---:|---:|---:|
| 1 | 81.94% | 0.7593 |
| 2 | 57.99% | 0.4398 |
| 3 | 89.58% | 0.8611 |
| 4 | 76.74% | 0.6898 |
| 5 | 69.10% | 0.5880 |
| 6 | 58.33% | 0.4444 |
| 7 | 84.72% | 0.7963 |
| 8 | 78.82% | 0.7176 |
| 9 | 66.32% | 0.5509 |
| **Mean subject** | **73.73%** | **0.6497** |

Pooled accuracy across all session `1test` trials: 73.73%.

## Training protocol

- Dataset: BNCI2014-001.
- Subjects: 1-9.
- Training data: all trials from session `0train`, 2,592 total trials.
- Checkpoint-selection data: all trials from session `1test`, 2,592 total trials.
- Input window: 1,000 samples, 250 Hz, approximately 4 seconds.
- Labels: `feet`, `left_hand`, `right_hand`, `tongue`.
- Normalization: one global Z-score fitted on training-session trials only.
- Architecture: CTNet paper configuration.
- Optimizer: Adam, learning rate 0.001, betas `(0.5, 0.999)`.
- Batch size: 72.
- Epochs: 1,000.
- Augmentation: subject-aware segmentation-and-reconstruction, 8 segments,
  factor 3.
- Seed: 0.

See `training_metadata.json` and `training_history.csv` for the full run record.

## Load

```python
from transformers import AutoFeatureExtractor, AutoModelForSequenceClassification

repo_id = "likan-blk/ctnet-hf"

processor = AutoFeatureExtractor.from_pretrained(repo_id, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    repo_id,
    trust_remote_code=True,
).eval()

inputs = processor(eeg_trial, return_tensors="pt")  # eeg_trial: (22, 1000), microvolts
logits = model(**inputs).logits
```

Because this repository contains custom modeling and preprocessing code, inspect
the files before enabling `trust_remote_code`. For reproducible use, pin
`revision` to a Hub commit hash.

## Intended use

This checkpoint is intended for research and reproducibility experiments with
BNCI2014-001-style motor-imagery EEG. It is not a medical device, clinical
tool, safety-critical classifier, or claim of generalization to arbitrary EEG
recordings.
