from __future__ import annotations

import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
)

from ctnet_hf import CtnetConfig, CtnetForEEGClassification


def test_model_save_and_reload(tmp_path):
    config = CtnetConfig()
    model = CtnetForEEGClassification(config)
    x = torch.randn(2, 22, 1000)

    model.eval()
    before = model(input_values=x)
    model.save_pretrained(tmp_path)

    reloaded = CtnetForEEGClassification.from_pretrained(tmp_path)
    reloaded.eval()
    after = reloaded(input_values=x)

    assert before.logits.shape == (2, 4)
    assert after.logits.shape == (2, 4)
    torch.testing.assert_close(before.logits, after.logits, rtol=0, atol=0)


def test_automodel_from_pretrained_with_remote_code(tmp_path):
    model = CtnetForEEGClassification(CtnetConfig())
    x = torch.randn(2, 22, 1000)
    model.save_pretrained(tmp_path)

    config = AutoConfig.from_pretrained(tmp_path, trust_remote_code=True)
    loaded = AutoModelForSequenceClassification.from_pretrained(
        tmp_path,
        trust_remote_code=True,
    )

    outputs = loaded(input_values=x)

    assert config.model_type == "ctnet"
    assert outputs.logits.shape == (2, 4)
