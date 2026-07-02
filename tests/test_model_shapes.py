from __future__ import annotations

import pytest
import torch

from ctnet_hf import CtnetConfig, CtnetForEEGClassification


@pytest.mark.parametrize(
    ("n_channels", "n_times", "num_labels"),
    [
        (22, 1000, 4),
        (3, 1000, 2),
    ],
)
def test_model_logits_shape(n_channels, n_times, num_labels):
    config = CtnetConfig(
        n_channels=n_channels,
        n_times=n_times,
        num_labels=num_labels,
    )
    model = CtnetForEEGClassification(config)
    x = torch.randn(2, n_channels, n_times)

    outputs = model(input_values=x)

    assert outputs.logits.shape == (2, num_labels)


def test_paper_model_uses_fifteen_flattened_tokens():
    config = CtnetConfig(
        architecture="paper",
        n_channels=22,
        n_times=1000,
        num_labels=4,
        n_filters_time=8,
        filter_time_length=64,
        depth_multiplier=2,
        pool_time_length=8,
        pool_time_stride=8,
        second_filter_time_length=16,
        second_pool_time_length=8,
        second_pool_time_stride=8,
        att_depth=6,
        att_heads=2,
        att_dim=16,
        att_mlp_dim=64,
    )
    model = CtnetForEEGClassification(config)

    outputs = model.ctnet(input_values=torch.randn(2, 22, 1000))

    assert outputs.last_hidden_state.shape == (2, 15, 16)
    assert outputs.pooler_output.shape == (2, 240)
    assert model.classifier.in_features == 240
