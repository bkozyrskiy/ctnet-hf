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
        num_labels: Optional[int] = None,
        label2id: Optional[dict[str, int]] = None,
        id2label: Optional[dict[int, str]] = None,
        architecture: str = "compact",
        n_filters_time: Optional[int] = None,
        filter_time_length: Optional[int] = None,
        pool_time_length: Optional[int] = None,
        pool_time_stride: Optional[int] = None,
        depth_multiplier: int = 2,
        second_filter_time_length: int = 16,
        second_pool_time_length: int = 8,
        second_pool_time_stride: int = 8,
        dropout: float = 0.5,
        att_depth: Optional[int] = None,
        att_heads: Optional[int] = None,
        att_dim: Optional[int] = None,
        att_mlp_dim: Optional[int] = None,
        positional_dropout: float = 0.1,
        max_position_embeddings: int = 100,
        classifier_dropout: float = 0.5,
        **kwargs,
    ) -> None:
        defaults = {
            "paper": {
                "n_filters_time": 8,
                "filter_time_length": 64,
                "pool_time_length": 8,
                "pool_time_stride": 8,
                "att_depth": 6,
                "att_heads": 2,
                "att_dim": 16,
                "att_mlp_dim": 64,
            },
            "compact": {
                "n_filters_time": 40,
                "filter_time_length": 25,
                "pool_time_length": 75,
                "pool_time_stride": 15,
                "att_depth": 2,
                "att_heads": 4,
                "att_dim": 64,
                "att_mlp_dim": 128,
            },
        }
        if architecture not in defaults:
            raise ValueError("architecture must be 'compact' or 'paper'.")
        selected = defaults[architecture]
        n_filters_time = n_filters_time or selected["n_filters_time"]
        filter_time_length = filter_time_length or selected["filter_time_length"]
        pool_time_length = pool_time_length or selected["pool_time_length"]
        pool_time_stride = pool_time_stride or selected["pool_time_stride"]
        att_depth = att_depth or selected["att_depth"]
        att_heads = att_heads or selected["att_heads"]
        att_dim = att_dim or selected["att_dim"]
        att_mlp_dim = att_mlp_dim or selected["att_mlp_dim"]

        if num_labels is None:
            if id2label is not None:
                num_labels = len(id2label)
            elif label2id is not None:
                num_labels = len(label2id)
            else:
                num_labels = 4
        if num_labels < 1:
            raise ValueError("num_labels must be at least 1.")
        if n_channels < 1:
            raise ValueError("n_channels must be at least 1.")
        if n_times < 1:
            raise ValueError("n_times must be at least 1.")
        if att_dim % att_heads != 0:
            raise ValueError("att_dim must be divisible by att_heads.")
        if architecture == "paper" and att_dim != n_filters_time * depth_multiplier:
            raise ValueError(
                "Paper CTNet requires att_dim == n_filters_time * depth_multiplier."
            )

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
        self.architecture = architecture
        self.n_filters_time = n_filters_time
        self.filter_time_length = filter_time_length
        self.pool_time_length = pool_time_length
        self.pool_time_stride = pool_time_stride
        self.depth_multiplier = depth_multiplier
        self.second_filter_time_length = second_filter_time_length
        self.second_pool_time_length = second_pool_time_length
        self.second_pool_time_stride = second_pool_time_stride
        self.dropout = dropout
        self.att_depth = att_depth
        self.att_heads = att_heads
        self.att_dim = att_dim
        self.att_mlp_dim = att_mlp_dim
        self.positional_dropout = positional_dropout
        self.max_position_embeddings = max_position_embeddings
        self.classifier_dropout = classifier_dropout
        self.auto_map = {
            "AutoConfig": "configuration_ctnet.CtnetConfig",
            "AutoModel": "modeling_ctnet.CtnetModel",
            "AutoModelForSequenceClassification": (
                "modeling_ctnet.CtnetForEEGClassification"
            ),
        }
