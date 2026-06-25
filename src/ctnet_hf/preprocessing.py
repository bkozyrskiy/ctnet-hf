"""Lightweight EEG input validation helpers."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def ensure_eeg_tensor(
    input_values: torch.Tensor | np.ndarray,
    *,
    expected_channels: Optional[int] = None,
    expected_times: Optional[int] = None,
) -> torch.Tensor:
    """Validate `(batch, channels, time)` input and return a float tensor."""
    if isinstance(input_values, np.ndarray):
        tensor = torch.from_numpy(input_values)
    elif isinstance(input_values, torch.Tensor):
        tensor = input_values
    else:
        raise TypeError("input_values must be a torch.Tensor or numpy.ndarray.")

    if tensor.ndim != 3:
        raise ValueError(
            "Expected EEG input with shape (batch_size, n_channels, n_times)."
        )

    if expected_channels is not None and tensor.shape[1] != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} EEG channels but received {tensor.shape[1]}."
        )

    if expected_times is not None and tensor.shape[2] != expected_times:
        raise ValueError(
            f"Expected {expected_times} time samples but received {tensor.shape[2]}."
        )

    return tensor.float()
