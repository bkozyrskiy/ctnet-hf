"""Serializable preprocessing for trained CTNet checkpoints."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature, FeatureExtractionMixin


class CtnetPreprocessor(FeatureExtractionMixin):
    """Apply the training-set normalization and validate the EEG layout."""

    model_input_names = ["input_values"]

    def __init__(
        self,
        *,
        n_channels: int,
        n_times: int,
        sampling_rate: Optional[int] = None,
        mean: Any,
        std: Any,
        channel_names: Optional[list[str]] = None,
        dataset: Optional[str] = None,
        subjects: Optional[list[int]] = None,
        unit: str = "microvolts",
        train_session: Optional[str] = None,
        selection_session: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if n_channels < 1 or n_times < 1:
            raise ValueError("n_channels and n_times must be positive.")
        if channel_names is not None and len(channel_names) != n_channels:
            raise ValueError(
                f"Expected {n_channels} channel names, got {len(channel_names)}."
            )

        self.n_channels = int(n_channels)
        self.n_times = int(n_times)
        self.sampling_rate = sampling_rate
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        if np.any(self.std <= 0):
            raise ValueError("std must be positive.")
        self.channel_names = channel_names
        self.dataset = dataset
        self.subjects = subjects
        self.unit = unit
        self.train_session = train_session
        self.selection_session = selection_session
        super().__init__(**kwargs)

    def __call__(
        self,
        input_values: torch.Tensor | np.ndarray,
        *,
        return_tensors: str = "pt",
    ) -> BatchFeature:
        """Normalize one trial or a batch in ``(channels, time)`` layout."""
        if isinstance(input_values, torch.Tensor):
            values = input_values.detach().cpu().numpy()
        else:
            values = np.asarray(input_values)
        if values.ndim == 2:
            values = values[np.newaxis, ...]
        if values.ndim != 3:
            raise ValueError(
                "Expected EEG input with shape (channels, time) or "
                "(batch_size, channels, time)."
            )
        if values.shape[1] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} EEG channels but received "
                f"{values.shape[1]}."
            )
        if values.shape[2] != self.n_times:
            raise ValueError(
                f"Expected {self.n_times} time samples but received "
                f"{values.shape[2]}."
            )

        values = values.astype(np.float32, copy=True)
        values = (values - self.mean) / self.std
        if return_tensors not in {"np", "pt"}:
            raise ValueError("return_tensors must be 'pt' or 'np'.")
        return BatchFeature({"input_values": values}, tensor_type=return_tensors)

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "feature_extractor_type": self.__class__.__name__,
                "n_channels": self.n_channels,
                "n_times": self.n_times,
                "sampling_rate": self.sampling_rate,
                "mean": self.mean.tolist(),
                "std": self.std.tolist(),
                "channel_names": self.channel_names,
                "dataset": self.dataset,
                "subjects": self.subjects,
                "unit": self.unit,
                "train_session": self.train_session,
                "selection_session": self.selection_session,
            }
        )
        return payload
