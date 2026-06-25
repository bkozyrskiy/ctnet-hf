from __future__ import annotations

from ctnet_hf import CtnetConfig


def test_config_round_trip(tmp_path):
    config = CtnetConfig(
        n_channels=22,
        n_times=1000,
        sampling_rate=250,
        num_labels=4,
        label2id={"left": 0, "right": 1, "foot": 2, "tongue": 3},
    )

    config.save_pretrained(tmp_path)
    loaded = CtnetConfig.from_pretrained(tmp_path)

    assert loaded.model_type == "ctnet"
    assert loaded.n_channels == 22
    assert loaded.n_times == 1000
    assert loaded.sampling_rate == 250
    assert loaded.num_labels == 4
    assert loaded.label2id["left"] == 0
    assert loaded.id2label[3] == "tongue"
    assert "AutoModelForSequenceClassification" in loaded.auto_map
