"""PyTorch implementations of compact and paper-compatible CTNet models."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    SequenceClassifierOutput,
)

from .configuration_ctnet import CtnetConfig


def _ensure_eeg_tensor(
    input_values: torch.Tensor,
    *,
    expected_channels: int,
    expected_times: int,
) -> torch.Tensor:
    if not isinstance(input_values, torch.Tensor):
        raise TypeError("input_values must be a torch.Tensor.")
    if input_values.ndim != 3:
        raise ValueError(
            "Expected EEG input with shape (batch_size, n_channels, n_times)."
        )
    if input_values.shape[1] != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} EEG channels but received "
            f"{input_values.shape[1]}."
        )
    if input_values.shape[2] != expected_times:
        raise ValueError(
            f"Expected {expected_times} time samples but received "
            f"{input_values.shape[2]}."
        )
    return input_values.float()


def _sinusoidal_positions(
    seq_len: int,
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    position = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, hidden_size, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / hidden_size)
    )

    embeddings = torch.zeros(seq_len, hidden_size, device=device, dtype=torch.float32)
    embeddings[:, 0::2] = torch.sin(position * div_term)
    embeddings[:, 1::2] = torch.cos(position * div_term)
    return embeddings.unsqueeze(0).to(dtype=dtype)


def _pool_output_length(length: int, kernel_size: int, stride: int) -> int:
    return (length - kernel_size) // stride + 1


class TemporalConv2dSame(nn.Module):
    """Conv2d with explicit same-padding along the temporal axis only."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total_pad = self.kernel_size - 1
        left = total_pad // 2
        right = total_pad - left
        return self.conv(F.pad(x, (left, right, 0, 0)))


class CtnetPreTrainedModel(PreTrainedModel):
    config_class = CtnetConfig
    base_model_prefix = "ctnet"
    main_input_name = "input_values"

    def _init_weights(self, module: nn.Module) -> None:
        if self.config.architecture == "paper":
            return
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if module.weight is not None:
                nn.init.ones_(module.weight)


class CompactCtnetEncoder(nn.Module):
    """Original configurable compact encoder retained for compatibility."""

    def __init__(self, config: CtnetConfig) -> None:
        super().__init__()
        self.output_dim = config.att_dim
        self.temporal_conv = TemporalConv2dSame(
            in_channels=1,
            out_channels=config.n_filters_time,
            kernel_size=config.filter_time_length,
        )
        self.temporal_norm = nn.BatchNorm2d(config.n_filters_time)
        self.spatial_conv = nn.Conv2d(
            in_channels=config.n_filters_time,
            out_channels=config.n_filters_time,
            kernel_size=(config.n_channels, 1),
            groups=config.n_filters_time,
            bias=False,
        )
        self.spatial_norm = nn.BatchNorm2d(config.n_filters_time)
        self.activation = nn.ELU()
        self.pool = nn.AvgPool2d(
            kernel_size=(1, config.pool_time_length),
            stride=(1, config.pool_time_stride),
        )
        self.dropout = nn.Dropout(config.dropout)
        self.projection = nn.Linear(config.n_filters_time, config.att_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.att_dim,
            nhead=config.att_heads,
            dim_feedforward=config.att_mlp_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.att_depth,
            norm=nn.LayerNorm(config.att_dim),
            enable_nested_tensor=False,
        )

    def forward(self, input_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = input_values.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.temporal_norm(x)
        x = self.spatial_conv(x)
        x = self.spatial_norm(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.dropout(x)

        if x.shape[-1] < 1:
            raise ValueError(
                "Temporal pooling collapsed the sequence. "
                "Decrease pool_time_length or provide more time samples."
            )

        x = x.squeeze(2).transpose(1, 2)
        x = self.projection(x)
        x = x + _sinusoidal_positions(
            seq_len=x.shape[1],
            hidden_size=x.shape[2],
            device=x.device,
            dtype=x.dtype,
        )
        hidden_states = self.transformer(x)
        pooled_output = hidden_states.mean(dim=1)
        return hidden_states, pooled_output


class PaperMultiHeadAttention(nn.Module):
    """Attention used by the authors' released CTNet implementation."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.keys = nn.Linear(embed_dim, embed_dim)
        self.queries = nn.Linear(embed_dim, embed_dim)
        self.values = nn.Linear(embed_dim, embed_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        def split_heads(values: torch.Tensor) -> torch.Tensor:
            return values.view(
                batch_size, seq_len, self.num_heads, self.head_dim
            ).transpose(1, 2)

        queries = split_heads(self.queries(x))
        keys = split_heads(self.keys(x))
        values = split_heads(self.values(x))
        energy = torch.matmul(queries, keys.transpose(-2, -1))
        attention = F.softmax(energy / math.sqrt(self.embed_dim), dim=-1)
        attention = self.attention_dropout(attention)
        context = torch.matmul(attention, values)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )
        return self.projection(context)


class PaperResidualAdd(nn.Module):
    def __init__(self, module: nn.Module, embed_dim: int, dropout: float) -> None:
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(self.dropout(self.module(x)) + x)


class PaperTransformerBlock(nn.Module):
    def __init__(self, config: CtnetConfig) -> None:
        super().__init__()
        self.attention = PaperResidualAdd(
            PaperMultiHeadAttention(config.att_dim, config.att_heads, config.dropout),
            config.att_dim,
            config.dropout,
        )
        self.feed_forward = PaperResidualAdd(
            nn.Sequential(
                nn.Linear(config.att_dim, config.att_mlp_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.att_mlp_dim, config.att_dim),
            ),
            config.att_dim,
            config.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feed_forward(self.attention(x))


class PaperCtnetEncoder(nn.Module):
    """CTNet architecture from Zhao et al. and the released source code."""

    def __init__(self, config: CtnetConfig) -> None:
        super().__init__()
        feature_dim = config.n_filters_time * config.depth_multiplier
        first_length = _pool_output_length(
            config.n_times, config.pool_time_length, config.pool_time_stride
        )
        token_count = _pool_output_length(
            first_length,
            config.second_pool_time_length,
            config.second_pool_time_stride,
        )
        if token_count < 1:
            raise ValueError("Paper CTNet pooling collapsed the temporal sequence.")
        if token_count > config.max_position_embeddings:
            raise ValueError(
                f"Paper CTNet produces {token_count} tokens but "
                f"max_position_embeddings={config.max_position_embeddings}."
            )

        self.token_count = token_count
        self.output_dim = token_count * feature_dim
        self.temporal_conv = TemporalConv2dSame(
            1, config.n_filters_time, config.filter_time_length
        )
        self.temporal_norm = nn.BatchNorm2d(config.n_filters_time)
        self.depthwise_conv = nn.Conv2d(
            config.n_filters_time,
            feature_dim,
            kernel_size=(config.n_channels, 1),
            groups=config.n_filters_time,
            bias=False,
        )
        self.depthwise_norm = nn.BatchNorm2d(feature_dim)
        self.first_pool = nn.AvgPool2d(
            (1, config.pool_time_length),
            stride=(1, config.pool_time_stride),
        )
        self.first_dropout = nn.Dropout(config.dropout)
        self.feature_conv = TemporalConv2dSame(
            feature_dim, feature_dim, config.second_filter_time_length
        )
        self.feature_norm = nn.BatchNorm2d(feature_dim)
        self.second_pool = nn.AvgPool2d(
            (1, config.second_pool_time_length),
            stride=(1, config.second_pool_time_stride),
        )
        self.second_dropout = nn.Dropout(config.dropout)
        self.activation = nn.ELU()
        self.position_embeddings = nn.Parameter(
            torch.randn(1, config.max_position_embeddings, feature_dim)
        )
        self.position_dropout = nn.Dropout(config.positional_dropout)
        self.transformer = nn.ModuleList(
            [PaperTransformerBlock(config) for _ in range(config.att_depth)]
        )

    def forward(self, input_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = input_values.unsqueeze(1)
        x = self.temporal_norm(self.temporal_conv(x))
        x = self.activation(self.depthwise_norm(self.depthwise_conv(x)))
        x = self.first_dropout(self.first_pool(x))
        x = self.activation(self.feature_norm(self.feature_conv(x)))
        x = self.second_dropout(self.second_pool(x))
        x = x.squeeze(2).transpose(1, 2)
        x = x * math.sqrt(x.shape[-1])
        cnn_features = self.position_dropout(
            x + self.position_embeddings[:, : x.shape[1], :]
        )

        transformer_features = cnn_features
        for block in self.transformer:
            transformer_features = block(transformer_features)
        hidden_states = cnn_features + transformer_features
        return hidden_states, hidden_states.flatten(start_dim=1)


class CtnetModel(CtnetPreTrainedModel):
    """Base CTNet encoder model."""

    def __init__(self, config: CtnetConfig) -> None:
        super().__init__(config)
        if config.architecture == "paper":
            self.encoder = PaperCtnetEncoder(config)
        else:
            self.encoder = CompactCtnetEncoder(config)
        self.output_dim = self.encoder.output_dim
        self.post_init()

    def forward(
        self,
        input_values: torch.FloatTensor,
        return_dict: Optional[bool] = None,
    ) -> BaseModelOutputWithPooling | tuple[torch.Tensor, torch.Tensor]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        inputs = _ensure_eeg_tensor(
            input_values,
            expected_channels=self.config.n_channels,
            expected_times=self.config.n_times,
        )
        hidden_states, pooled_output = self.encoder(inputs)

        if not return_dict:
            return hidden_states, pooled_output

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled_output,
            hidden_states=None,
            attentions=None,
        )


class CtnetForEEGClassification(CtnetPreTrainedModel):
    """CTNet classifier with cross-entropy loss for EEG labels."""

    def __init__(self, config: CtnetConfig) -> None:
        super().__init__(config)
        self.ctnet = CtnetModel(config)
        dropout = (
            config.classifier_dropout
            if config.architecture == "paper"
            else config.dropout
        )
        self.classifier_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.ctnet.output_dim, config.num_labels)
        self.post_init()

    def forward(
        self,
        input_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> SequenceClassifierOutput | tuple[torch.Tensor, ...]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        outputs = self.ctnet(input_values=input_values, return_dict=True)
        logits = self.classifier(self.classifier_dropout(outputs.pooler_output))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels.long())

        if not return_dict:
            result = (logits, outputs.last_hidden_state)
            return ((loss,) + result) if loss is not None else result

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
