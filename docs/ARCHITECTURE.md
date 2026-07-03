# CTNet-HF architecture

CTNet-HF implements two EEG classifiers behind the same Transformers API:

- `architecture="paper"` follows the CTNet topology published by Zhao et al.
  and used by the authors' subject-specific BCI Competition IV-2a code.
- `architecture="compact"` preserves the shallower configurable encoder from the
  first revision of this repository. It is not an architecture from the CTNet
  paper.

Both variants accept a floating-point tensor shaped
`(batch_size, n_channels, n_times)` and return classification logits. The
tables below use the repository defaults: 22 EEG channels, 1,000 samples at
250 Hz, and four output classes.

## Paper-compatible CTNet

The paper path combines an EEGNet-like convolutional front end with learned
positions, six Transformer blocks, an outer CNN residual, and a flattened
classifier.

| Stage | Operation | Output shape | Trainable parameters |
|---|---|---:|---:|
| Input | Add the internal singleton image channel | `(B, 1, 22, 1000)` | 0 |
| Temporal features | Same-padded `Conv2d`, `1 → 8`, kernel `(1, 64)`; batch norm | `(B, 8, 22, 1000)` | 528 |
| Spatial features | Depthwise `Conv2d`, `8 → 16`, kernel `(22, 1)`, 8 groups; batch norm; ELU | `(B, 16, 1, 1000)` | 384 |
| Patch pool 1 | Average pool `(1, 8)`, stride 8; dropout | `(B, 16, 1, 125)` | 0 |
| Local feature refinement | Same-padded `Conv2d`, `16 → 16`, kernel `(1, 16)`; batch norm; ELU | `(B, 16, 1, 125)` | 4,128 |
| Patch pool 2 | Average pool `(1, 8)`, stride 8; dropout | `(B, 16, 1, 15)` | 0 |
| Tokenization | Remove the height dimension and transpose | `(B, 15, 16)` | 0 |
| Positions | Scale by `sqrt(16)`, add a learned `100 × 16` position table, then dropout | `(B, 15, 16)` | 1,600 |
| Global context | 6 post-norm Transformer blocks, 2 heads, embedding 16, MLP 64 | `(B, 15, 16)` | 19,680 |
| CNN residual | Add positioned CNN tokens to the Transformer output | `(B, 15, 16)` | 0 |
| Classification | Flatten to 240, dropout, linear `240 → 4` | `(B, 4)` | 964 |
| **Total** | Four-class default configuration |  | **27,284** |

Each paper Transformer block uses independent query, key, value, and output
projections. Attention logits are divided by `sqrt(embedding_dim)`, matching
the released CTNet code, rather than the more usual `sqrt(head_dim)`. Both the
attention and feed-forward residual paths apply dropout, residual addition,
and then layer normalization.

The model exposes the token tensor as `last_hidden_state` and its flattened
240-value representation as `pooler_output`. The classification wrapper applies
the final dropout and linear head.

## Compact CTNet variant

The compact path has a wider first convolution but only one convolutional pool,
fixed sinusoidal positions, two pre-norm Transformer layers, and mean pooling.

| Stage | Operation | Output shape |
|---|---|---:|
| Input | Add the internal singleton image channel | `(B, 1, 22, 1000)` |
| Temporal features | Same-padded `Conv2d`, `1 → 40`, kernel `(1, 25)`; batch norm | `(B, 40, 22, 1000)` |
| Spatial features | Depthwise `Conv2d`, `40 → 40`, kernel `(22, 1)`, 40 groups; batch norm; ELU | `(B, 40, 1, 1000)` |
| Temporal pool | Average pool `(1, 75)`, stride 15; dropout | `(B, 40, 1, 62)` |
| Token projection | Transpose and linear projection `40 → 64` | `(B, 62, 64)` |
| Positions | Add fixed sinusoidal positions | `(B, 62, 64)` |
| Global context | 2 pre-norm Transformer layers, 4 heads, embedding 64, MLP 128; final layer norm | `(B, 62, 64)` |
| Pooling | Mean over the 62 tokens | `(B, 64)` |
| Classification | Dropout and linear `64 → 4` | `(B, 4)` |
| **Total** | Four-class default configuration | **71,996 parameters** |

## Paper and compact comparison

| Property | `paper` | `compact` |
|---|---|---|
| Provenance | Zhao et al. CTNet topology | Earlier CTNet-HF variant |
| Convolution widths | 8 temporal, 16 depthwise/refinement | 40 temporal/spatial |
| Temporal convolution kernels | 64 then 16 | 25 |
| Token count for 1,000 samples | 15 | 62 |
| Token dimension | 16 | 64 |
| Position representation | Learned, maximum length 100 | Fixed sinusoidal |
| Transformer | 6 post-norm blocks, 2 heads, MLP 64 | 2 pre-norm PyTorch encoder layers, 4 heads, MLP 128 |
| CNN-to-Transformer residual | Yes | No |
| Classifier representation | Flattened `15 × 16 = 240` | Mean-pooled 64 |
| Four-class parameters | 27,284 | 71,996 |
| Checkpoint compatibility | Paper-compatible CTNet-HF checkpoints | Compact CTNet-HF checkpoints only |

The variants are intentionally selected by configuration and are not
checkpoint-interchangeable.

## Differences from the authors' reference implementation

This comparison uses the authors' `main_subject_specific.py` at upstream commit
[`fb83814`](https://github.com/snailpt/CTNet/blob/fb83814abc55bafaea6e71cc18475bf331c8d28b/main_subject_specific.py).

The `paper` path preserves the released model's convolution widths and kernels,
two 8× temporal pools, 15 tokens, learned positions, six post-norm Transformer
blocks, two heads, embedding dimension 16, feed-forward dimension 64, outer CNN
residual, and flattened 240-value classifier. CTNet-HF changes the surrounding
engineering and evaluation behavior:

- The public input is `(batch, channels, time)`; the singleton convolution
  dimension is added internally.
- Model and position tensors are device-agnostic rather than containing
  hard-coded CUDA transfers.
- Even-kernel temporal `same` padding is explicit and deterministic.
- Dimensions, labels, dropout, convolution settings, attention settings, and
  position capacity are represented by `CtnetConfig` and validated early.
- Outputs use Transformers model-output objects and support the standard
  `save_pretrained` and AutoClass APIs.
- The Hugging Face model publishes a deterministic untrained initialization;
  subject-specific preprocessing must be fitted on the user's own training data.
- No checkpoint from the authors' repository is copied or converted.

The local benchmark is paper-compatible rather than bug-for-bug identical to
the released training script:

- It creates one fixed stratified validation split instead of rebuilding a
  batch-derived validation set during every epoch.
- Z-score statistics are fitted on the clean training subset only.
- S&R augmentation samples only from that training subset, never validation
  trials.
- The competition training session and test session remain separate, and the
  test session is used only for final evaluation.
- Seeds and deterministic PyTorch settings are explicit in the local benchmark.

These corrections make the evaluation easier to audit, but they also mean the
frozen CTNet-HF scores should not be presented as a direct reproduction of the
paper's reported numbers.

## Source map

- Configuration: [`src/ctnet_hf/configuration_ctnet.py`](../src/ctnet_hf/configuration_ctnet.py)
- Model implementation: [`src/ctnet_hf/modeling_ctnet.py`](../src/ctnet_hf/modeling_ctnet.py)
- Benchmark protocol: [`benchmarks/README.md`](../benchmarks/README.md)
- Hugging Face Model Card: [`hf_model/README.md`](../hf_model/README.md)
- Software citation: [`CITATION.cff`](../CITATION.cff)

## References

- Wei Zhao, Xiaolu Jiang, Baocan Zhang, Shixiao Xiao, and Sujun Weng,
  “CTNet: a convolutional transformer network for EEG-based motor imagery
  classification,” *Scientific Reports* 14, 20237 (2024).
  https://doi.org/10.1038/s41598-024-71118-7
- Authors' released implementation: https://github.com/snailpt/CTNet/tree/fb83814abc55bafaea6e71cc18475bf331c8d28b
