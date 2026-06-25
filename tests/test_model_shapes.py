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
