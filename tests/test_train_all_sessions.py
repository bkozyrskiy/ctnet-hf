from __future__ import annotations

import numpy as np

from benchmarks.train_all_sessions import (
    SubjectSplit,
    _cohen_kappa,
    _fit_global_standardizer,
    _make_subject_aware_synthetic,
    _standardize_in_place,
)


def test_subject_aware_augmentation_never_splices_people():
    # A trial's constant value identifies (subject, class). If segments crossed
    # people, at least one synthetic trial would contain multiple values.
    values = []
    labels = []
    subjects = []
    for subject in (1, 2):
        for class_id in (0, 1):
            for _ in range(3):
                values.append(np.full((2, 16), subject * 10 + class_id, np.float32))
                labels.append(class_id)
                subjects.append(subject)
    result = _make_subject_aware_synthetic(
        np.stack(values),
        np.asarray(labels),
        np.asarray(subjects),
        np.random.default_rng(0),
        factor=1,
        segments=4,
        configured_batch_size=4,
        n_classes=2,
    )

    assert result is not None
    synthetic, synthetic_labels = result
    assert synthetic.shape == (4, 2, 16)
    assert np.all(synthetic == synthetic[:, :1, :1])
    assert np.array_equal(synthetic_labels, [0, 0, 1, 1])


def test_global_standardizer_uses_all_training_splits():
    splits = [
        SubjectSplit(1, np.zeros((1, 1, 2), dtype=np.float32), np.array([0])),
        SubjectSplit(2, np.full((1, 1, 2), 2.0, dtype=np.float32), np.array([0])),
    ]

    mean, std = _fit_global_standardizer(splits)
    _standardize_in_place(splits, mean, std)

    assert mean == 1.0
    assert std == 1.0
    assert np.array_equal(splits[0].values, [[[-1.0, -1.0]]])
    assert np.array_equal(splits[1].values, [[[1.0, 1.0]]])


def test_cohen_kappa_perfect_and_chance():
    labels = np.array([0, 0, 1, 1])

    assert _cohen_kappa(labels, labels) == 1.0
    assert _cohen_kappa(labels, np.array([0, 1, 0, 1])) == 0.0
