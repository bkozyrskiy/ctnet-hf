"""Configuration for CTNet Hugging Face models."""

from __future__ import annotations

from typing import Optional

from transformers.configuration_utils import PretrainedConfig as PreTrainedConfig


class CtnetConfig(PreTrainedConfig):
    """Configuration for a configurable CTNet-style EEG classifier."""

    model_type = "ctnet"

    def __init__(
        self,
        n_channels: int = 22,
        n_times: int = 1000,
        sampling_rate: Optional[int] = 250,
        num_labels: int = 4,
        label2id: Optional[dict[str, int]] = None,
        id2label: Optional[dict[int, str]] = None,
        n_filters_time: int = 40,
        filter_time_length: int = 25,
        pool_time_length: int = 75,
        pool_time_stride: int = 15,
        dropout: float = 0.5,
        att_depth: int = 2,
        att_heads: int = 4,
        att_dim: int = 64,
        att_mlp_dim: int = 128,
        **kwargs,
    ) -> None:
        if num_labels < 1:
            raise ValueError("num_labels must be at least 1.")
        if n_channels < 1:
            raise ValueError("n_channels must be at least 1.")
        if n_times < 1:
            raise ValueError("n_times must be at least 1.")
        if att_dim % att_heads != 0:
            raise ValueError("att_dim must be divisible by att_heads.")

        if label2id is None and id2label is None:
            id2label = {idx: f"LABEL_{idx}" for idx in range(num_labels)}
            label2id = {label: idx for idx, label in id2label.items()}
        elif label2id is None:
            id2label = {int(idx): label for idx, label in id2label.items()}
            label2id = {label: idx for idx, label in id2label.items()}
        elif id2label is None:
            label2id = {str(label): int(idx) for label, idx in label2id.items()}
            id2label = {idx: label for label, idx in label2id.items()}
        else:
            label2id = {str(label): int(idx) for label, idx in label2id.items()}
            id2label = {int(idx): str(label) for idx, label in id2label.items()}

        super().__init__(
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            **kwargs,
        )

        self.n_channels = n_channels
        self.n_times = n_times
        self.sampling_rate = sampling_rate
        self.n_filters_time = n_filters_time
        self.filter_time_length = filter_time_length
        self.pool_time_length = pool_time_length
        self.pool_time_stride = pool_time_stride
        self.dropout = dropout
        self.att_depth = att_depth
        self.att_heads = att_heads
        self.att_dim = att_dim
        self.att_mlp_dim = att_mlp_dim
        self.auto_map = {
            "AutoConfig": "configuration_ctnet.CtnetConfig",
            "AutoModel": "modeling_ctnet.CtnetModel",
            "AutoModelForSequenceClassification": (
                "modeling_ctnet.CtnetForEEGClassification"
            ),
        }
