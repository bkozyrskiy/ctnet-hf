from __future__ import annotations

import numpy as np

from ctnet_hf import CtnetPreprocessor


def test_preprocessor_round_trip(tmp_path):
    processor = CtnetPreprocessor(
        n_channels=2,
        n_times=4,
        sampling_rate=128,
        mean=1.0,
        std=2.0,
        channel_names=["C3", "C4"],
        subjects=[1, 2],
    )
    processor.save_pretrained(tmp_path)
    loaded = CtnetPreprocessor.from_pretrained(tmp_path)

    result = loaded(np.full((2, 4), 3.0, dtype=np.float32), return_tensors="np")

    assert result["input_values"].shape == (1, 2, 4)
    assert np.all(result["input_values"] == 1.0)
    assert loaded.channel_names == ["C3", "C4"]
