from __future__ import annotations

import torch
import numpy as np
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForSequenceClassification,
)

from ctnet_hf import CtnetConfig, CtnetForEEGClassification, CtnetPreprocessor


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


def test_preprocessor_save_reload_equivalence(tmp_path):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(3, 2, 16)).astype(np.float32)
    processor = CtnetPreprocessor(
        n_channels=2,
        n_times=16,
        sampling_rate=128,
        standardize=True,
        standardize_mode="channel",
        mean=x.mean(axis=(0, 2), keepdims=True),
        std=x.std(axis=(0, 2), keepdims=True),
        channel_names=["C3", "C4"],
        dataset="test",
    )

    before = processor(x, return_tensors="np")["input_values"]
    processor.save_pretrained(tmp_path)
    reloaded = CtnetPreprocessor.from_pretrained(tmp_path)
    after = reloaded(x, return_tensors="np")["input_values"]

    np.testing.assert_array_equal(before, after)
    assert reloaded.channel_names == ["C3", "C4"]

    auto_reloaded = AutoFeatureExtractor.from_pretrained(
        tmp_path,
        trust_remote_code=True,
    )
    auto_after = auto_reloaded(x, return_tensors="np")["input_values"]
    np.testing.assert_array_equal(before, auto_after)
