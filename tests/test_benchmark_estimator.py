from __future__ import annotations

import numpy as np
import pytest
import torch

from benchmarks.ctnet_estimator import CtnetSklearnClassifier
from ctnet_hf import CtnetForEEGClassification, CtnetPreprocessor


def test_ctnet_estimator_fit_predict_score_and_export(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8, 3, 128)).astype(np.float32)
    y = np.array(["left", "right", "left", "right", "left", "right", "left", "right"])
    estimator = CtnetSklearnClassifier(
        architecture="compact",
        epochs=1,
        batch_size=4,
        validation_ratio=0,
        augmentation_factor=0,
        input_samples=None,
        seed=0,
        pool_time_length=16,
        pool_time_stride=8,
        att_depth=1,
        att_dim=16,
        att_heads=2,
        att_mlp_dim=32,
    )

    estimator.fit(x, y)
    predictions = estimator.predict(x[:2])
    probabilities = estimator.predict_proba(x[:2])
    score = estimator.score(x[:2], y[:2])

    assert predictions.shape == (2,)
    assert probabilities.shape == (2, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert 0.0 <= score <= 1.0
    assert estimator.label2id_ == {"left": 0, "right": 1}

    estimator.save_pretrained(
        str(tmp_path),
        channel_names=["a", "b", "c"],
        dataset="dummy",
    )
    processor = CtnetPreprocessor.from_pretrained(tmp_path)
    model = CtnetForEEGClassification.from_pretrained(tmp_path).to(
        estimator.device_
    ).eval()
    with torch.no_grad():
        inputs = processor(x[:2], return_tensors="pt")
        inputs = {key: value.to(estimator.device_) for key, value in inputs.items()}
        reloaded_probabilities = (
            torch.softmax(model(**inputs).logits, -1).cpu().numpy()
        )

    np.testing.assert_allclose(probabilities, reloaded_probabilities, rtol=1e-6)


def test_ctnet_estimator_rejects_non_eeg_shape():
    estimator = CtnetSklearnClassifier(epochs=1)

    with pytest.raises(ValueError, match="Expected EEG array"):
        estimator.fit(np.zeros((4, 128), dtype=np.float32), np.zeros(4))


def test_ctnet_estimator_get_set_params():
    estimator = CtnetSklearnClassifier(epochs=1)

    estimator.set_params(epochs=2, batch_size=8)

    assert estimator.get_params()["epochs"] == 2
    assert estimator.get_params()["batch_size"] == 8
