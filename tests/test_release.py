from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from huggingface_hub import ModelCard

from benchmarks.model_card import render_bnci2014_001_card
from ctnet_hf import (
    CtnetConfig,
    CtnetForEEGClassification,
    CtnetPreprocessor,
    export_huggingface_bundle,
    validate_huggingface_bundle,
)


def test_exported_bundle_is_complete_and_uses_auto_feature_extractor(tmp_path):
    config = CtnetConfig(n_channels=3, n_times=128, sampling_rate=128)
    model = CtnetForEEGClassification(config)
    preprocessor = CtnetPreprocessor(
        n_channels=3,
        n_times=128,
        sampling_rate=128,
        standardize=True,
        mean=0.25,
        std=2.0,
        channel_names=["C3", "Cz", "C4"],
        unit="microvolts",
    )

    export_huggingface_bundle(model, preprocessor, tmp_path)
    contents = validate_huggingface_bundle(tmp_path)

    assert (tmp_path / "model.safetensors").is_file()
    assert not list(tmp_path.glob("pytorch_model*.bin"))
    assert contents["preprocessor"]["auto_map"] == {
        "AutoFeatureExtractor": "preprocessing.CtnetPreprocessor"
    }


def test_export_rejects_model_preprocessor_mismatch(tmp_path):
    model = CtnetForEEGClassification(
        CtnetConfig(n_channels=3, n_times=128, sampling_rate=128)
    )
    preprocessor = CtnetPreprocessor(
        n_channels=3,
        n_times=64,
        sampling_rate=128,
        standardize=False,
        channel_names=["C3", "Cz", "C4"],
    )

    with pytest.raises(ValueError, match="disagree on n_times"):
        export_huggingface_bundle(model, preprocessor, tmp_path)


def test_generated_model_card_metadata_is_valid(tmp_path):
    text = render_bnci2014_001_card(
        subject=1,
        seed=0,
        train_session="0train",
        test_session="1test",
        accuracy=0.8,
        cohen_kappa=0.7,
        best_epoch=20,
        channel_names=[f"C{index}" for index in range(22)],
        id2label={0: "feet", 1: "left_hand", 2: "right_hand", 3: "tongue"},
        parameter_count=1234,
        training_seconds=10.0,
    )
    path = tmp_path / "README.md"
    path.write_text(text, encoding="utf-8")

    card = ModelCard.load(path)
    card.validate()
    metadata = card.data.to_dict()
    assert metadata["library_name"] == "transformers"
    assert "pipeline_tag" not in metadata


def test_checked_in_card_template_metadata_is_valid():
    card = ModelCard.load("hf_model_repo_template/README.md")
    card.validate()
    assert "pipeline_tag" not in card.data.to_dict()


def test_frozen_release_result_hashes():
    results_dir = Path("release/results")
    manifest = json.loads((results_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["n_rows"] == 27
    for output in manifest["outputs"]:
        digest = hashlib.sha256((results_dir / output["path"]).read_bytes()).hexdigest()
        assert digest == output["sha256"]
