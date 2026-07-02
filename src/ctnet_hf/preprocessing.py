"""Serializable preprocessing and EEG input validation helpers."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature, FeatureExtractionMixin


class CtnetPreprocessor(FeatureExtractionMixin):
    """Apply and serialize the exact normalization used to train a CTNet model.

    The preprocessor intentionally does not perform signal filtering. A release
    bundle records that external acquisition/cropping contract as metadata and
    stores only the deterministic sample selection and training-set Z-score
    needed immediately before model inference.
    """

    model_input_names = ["input_values"]

    def __init__(
        self,
        *,
        n_channels: int,
        n_times: int,
        sampling_rate: Optional[int] = None,
        input_samples: Optional[int] = None,
        standardize: bool = True,
        standardize_mode: str = "global",
        mean: Any = None,
        std: Any = None,
        channel_names: Optional[list[str]] = None,
        dataset: Optional[str] = None,
        subject: Optional[int] = None,
        unit: str = "volts",
        trial_start_seconds: Optional[float] = None,
        trial_duration_seconds: Optional[float] = None,
        software_filter: Optional[dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        if n_channels < 1 or n_times < 1:
            raise ValueError("n_channels and n_times must both be positive.")
        if standardize_mode not in {"global", "channel"}:
            raise ValueError("standardize_mode must be 'global' or 'channel'.")
        if channel_names is not None and len(channel_names) != n_channels:
            raise ValueError(
                f"Expected {n_channels} channel names, got {len(channel_names)}."
            )
        if standardize and (mean is None or std is None):
            raise ValueError("mean and std are required when standardize=True.")

        self.n_channels = int(n_channels)
        self.n_times = int(n_times)
        self.sampling_rate = sampling_rate
        self.input_samples = int(input_samples or n_times)
        self.standardize = bool(standardize)
        self.standardize_mode = standardize_mode
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)
        self.channel_names = channel_names
        self.dataset = dataset
        self.subject = subject
        self.unit = unit
        self.trial_start_seconds = trial_start_seconds
        self.trial_duration_seconds = trial_duration_seconds
        self.software_filter = software_filter
        super().__init__(**kwargs)

        if self.input_samples != self.n_times:
            raise ValueError(
                "input_samples must match the model's n_times in a release bundle."
            )
        if self.standardize and np.any(self.std <= 0):
            raise ValueError("All standard deviations must be positive.")

    def __call__(
        self,
        input_values: torch.Tensor | np.ndarray,
        *,
        return_tensors: str = "pt",
    ) -> BatchFeature:
        """Preprocess one trial or a batch in ``(channels, time)`` layout."""
        is_tensor = isinstance(input_values, torch.Tensor)
        if is_tensor:
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
        if values.shape[2] < self.input_samples:
            raise ValueError(
                f"Expected at least {self.input_samples} time samples but received "
                f"{values.shape[2]}."
            )

        values = values[..., : self.input_samples].astype(np.float32, copy=True)
        if self.standardize:
            values = (values - self.mean) / self.std
        if return_tensors not in {"np", "pt"}:
            raise ValueError("return_tensors must be 'pt' or 'np'.")
        return BatchFeature({"input_values": values}, tensor_type=return_tensors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_extractor_type": self.__class__.__name__,
            "input_shape": ["batch_size", self.n_channels, self.n_times],
            "n_channels": self.n_channels,
            "n_times": self.n_times,
            "sampling_rate": self.sampling_rate,
            "input_samples": self.input_samples,
            "standardize": self.standardize,
            "standardize_mode": self.standardize_mode,
            "mean": None if self.mean is None else self.mean.tolist(),
            "std": None if self.std is None else self.std.tolist(),
            "channel_names": self.channel_names,
            "dataset": self.dataset,
            "subject": self.subject,
            "unit": self.unit,
            "trial_start_seconds": self.trial_start_seconds,
            "trial_duration_seconds": self.trial_duration_seconds,
            "software_filter": self.software_filter,
            **{
                key: value
                for key, value in super().to_dict().items()
                if key
                not in {
                    "mean",
                    "std",
                    "feature_extractor_type",
                }
            },
        }


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
