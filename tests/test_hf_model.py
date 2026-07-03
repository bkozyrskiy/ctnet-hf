from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import ModelCard
from transformers import AutoConfig, AutoModelForSequenceClassification


BUNDLE = Path("hf_model")


def test_hf_model_bundle_is_minimal_and_explicitly_untrained():
    assert {path.name for path in BUNDLE.iterdir() if path.is_file()} == {
        "LICENSE",
        "README.md",
        "config.json",
        "configuration_ctnet.py",
        "model.safetensors",
        "modeling_ctnet.py",
    }

    card = ModelCard.load(BUNDLE / "README.md")
    card.validate()
    assert "not pretrained" in card.text.lower()

    config = AutoConfig.from_pretrained(BUNDLE, trust_remote_code=True)
    assert config.architecture == "paper"
    assert config.pretrained is False
    assert config.initialization == "random"
    assert config.initialization_seed == 0


def test_hf_model_loads_through_auto_class():
    model = AutoModelForSequenceClassification.from_pretrained(
        BUNDLE,
        trust_remote_code=True,
    ).eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    with torch.no_grad():
        output = model(input_values=torch.zeros(1, 22, 1000))

    assert parameter_count == 27_284
    assert output.logits.shape == (1, 4)


def test_hf_model_supports_custom_data_shape_from_config():
    config = AutoConfig.from_pretrained(BUNDLE, trust_remote_code=True)
    config.n_channels = 8
    config.n_times = 500
    config.sampling_rate = 250
    config.num_labels = 2
    config.id2label = {0: "left", 1: "right"}
    config.label2id = {"left": 0, "right": 1}

    model = AutoModelForSequenceClassification.from_config(
        config,
        trust_remote_code=True,
    ).eval()

    with torch.no_grad():
        output = model(input_values=torch.zeros(2, 8, 500))

    assert model.config.n_channels == 8
    assert model.config.n_times == 500
    assert model.config.id2label == {0: "left", 1: "right"}
    assert output.logits.shape == (2, 2)
