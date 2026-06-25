"""PyTorch implementation of a configurable CTNet-style EEG classifier."""

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
from .preprocessing import ensure_eeg_tensor


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
        x = F.pad(x, (left, right, 0, 0))
        return self.conv(x)


class CtnetPreTrainedModel(PreTrainedModel):
    config_class = CtnetConfig
    base_model_prefix = "ctnet"
    main_input_name = "input_values"

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if module.weight is not None:
                nn.init.ones_(module.weight)


class CtnetEncoder(nn.Module):
    """Convolutional feature extractor followed by a transformer encoder."""

    def __init__(self, config: CtnetConfig) -> None:
        super().__init__()
        self.config = config
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


class CtnetModel(CtnetPreTrainedModel):
    """Base CTNet encoder model."""

    def __init__(self, config: CtnetConfig) -> None:
        super().__init__(config)
        self.encoder = CtnetEncoder(config)
        self.post_init()

    def forward(
        self,
        input_values: torch.FloatTensor,
        return_dict: Optional[bool] = None,
    ) -> BaseModelOutputWithPooling | tuple[torch.Tensor, torch.Tensor]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        inputs = ensure_eeg_tensor(
            input_values,
            expected_channels=self.config.n_channels,
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
        self.classifier_dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.att_dim, config.num_labels)
        self.post_init()

    def forward(
        self,
        input_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> SequenceClassifierOutput | tuple[torch.Tensor, ...]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.ctnet(input_values=input_values, return_dict=True)
        pooled_output = self.classifier_dropout(outputs.pooler_output)
        logits = self.classifier(pooled_output)

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
