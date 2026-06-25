from __future__ import annotations

import torch

from ctnet_hf import CtnetConfig, CtnetForEEGClassification


def test_backward_populates_gradients():
    config = CtnetConfig()
    model = CtnetForEEGClassification(config)
    x = torch.randn(4, 22, 1000)
    labels = torch.randint(0, 4, (4,))

    outputs = model(input_values=x, labels=labels)
    outputs.loss.backward()

    assert outputs.loss is not None
    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )
