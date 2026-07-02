"""Scikit-learn compatible CTNet estimator for MOABB benchmarks."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

from ctnet_hf import CtnetConfig, CtnetForEEGClassification


class CtnetSklearnClassifier(ClassifierMixin, BaseEstimator):
    """Sklearn wrapper with paper-compatible and compact CTNet variants."""

    def __init__(
        self,
        architecture: str = "paper",
        epochs: int = 1000,
        batch_size: int = 72,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        adam_beta1: float = 0.5,
        adam_beta2: float = 0.999,
        validation_ratio: float = 0.3,
        augmentation_factor: int = 3,
        augmentation_segments: int = 8,
        input_samples: int | None = 1000,
        seed: int = 0,
        device: str = "auto",
        num_workers: int = 0,
        sampling_rate: int | None = None,
        standardize: bool = True,
        standardize_mode: str = "global",
        dropout: float = 0.5,
        classifier_dropout: float = 0.5,
        positional_dropout: float = 0.1,
        att_depth: int = 6,
        att_heads: int = 2,
        att_dim: int = 16,
        att_mlp_dim: int = 64,
        n_filters_time: int = 8,
        filter_time_length: int = 64,
        depth_multiplier: int = 2,
        pool_time_length: int = 8,
        pool_time_stride: int = 8,
        second_filter_time_length: int = 16,
        second_pool_time_length: int = 8,
        second_pool_time_stride: int = 8,
        max_position_embeddings: int = 100,
        verbose: bool = False,
    ) -> None:
        self.architecture = architecture
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.validation_ratio = validation_ratio
        self.augmentation_factor = augmentation_factor
        self.augmentation_segments = augmentation_segments
        self.input_samples = input_samples
        self.seed = seed
        self.device = device
        self.num_workers = num_workers
        self.sampling_rate = sampling_rate
        self.standardize = standardize
        self.standardize_mode = standardize_mode
        self.dropout = dropout
        self.classifier_dropout = classifier_dropout
        self.positional_dropout = positional_dropout
        self.att_depth = att_depth
        self.att_heads = att_heads
        self.att_dim = att_dim
        self.att_mlp_dim = att_mlp_dim
        self.n_filters_time = n_filters_time
        self.filter_time_length = filter_time_length
        self.depth_multiplier = depth_multiplier
        self.pool_time_length = pool_time_length
        self.pool_time_stride = pool_time_stride
        self.second_filter_time_length = second_filter_time_length
        self.second_pool_time_length = second_pool_time_length
        self.second_pool_time_stride = second_pool_time_stride
        self.max_position_embeddings = max_position_embeddings
        self.verbose = verbose

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            key: getattr(self, key)
            for key in (
                "architecture",
                "epochs",
                "batch_size",
                "lr",
                "weight_decay",
                "adam_beta1",
                "adam_beta2",
                "validation_ratio",
                "augmentation_factor",
                "augmentation_segments",
                "input_samples",
                "seed",
                "device",
                "num_workers",
                "sampling_rate",
                "standardize",
                "standardize_mode",
                "dropout",
                "classifier_dropout",
                "positional_dropout",
                "att_depth",
                "att_heads",
                "att_dim",
                "att_mlp_dim",
                "n_filters_time",
                "filter_time_length",
                "depth_multiplier",
                "pool_time_length",
                "pool_time_stride",
                "second_filter_time_length",
                "second_pool_time_length",
                "second_pool_time_stride",
                "max_position_embeddings",
                "verbose",
            )
        }

    def set_params(self, **params: Any) -> "CtnetSklearnClassifier":
        for key, value in params.items():
            if key not in self.get_params():
                raise ValueError(f"Invalid parameter {key!r}.")
            setattr(self, key, value)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CtnetSklearnClassifier":
        X = _select_input_samples(_validate_eeg_array(X), self.input_samples)
        y = np.asarray(y).reshape(-1)
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} trials but y has {y.shape[0]} labels."
            )
        if not 0.0 <= self.validation_ratio < 1.0:
            raise ValueError("validation_ratio must be in [0, 1).")
        if self.standardize_mode not in {"global", "channel"}:
            raise ValueError("standardize_mode must be 'global' or 'channel'.")

        _set_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self.label2id_ = {
            str(label): int(index) for index, label in enumerate(self.classes_)
        }
        self.id2label_ = {
            int(index): str(label) for index, label in enumerate(self.classes_)
        }

        indices = np.arange(X.shape[0])
        if self.validation_ratio:
            train_indices, validation_indices = train_test_split(
                indices,
                test_size=self.validation_ratio,
                random_state=self.seed,
                stratify=y_encoded,
            )
        else:
            train_indices = indices
            validation_indices = np.empty(0, dtype=int)

        X_train = X[train_indices].astype(np.float32, copy=True)
        y_train = y_encoded[train_indices]
        X_validation = X[validation_indices].astype(np.float32, copy=True)
        y_validation = y_encoded[validation_indices]
        self._fit_standardizer(X_train)
        X_train = self._standardize(X_train)
        X_validation = self._standardize(X_validation)

        config = CtnetConfig(
            architecture=self.architecture,
            n_channels=int(X_train.shape[1]),
            n_times=int(X_train.shape[2]),
            sampling_rate=self.sampling_rate,
            num_labels=len(self.classes_),
            label2id=self.label2id_,
            id2label=self.id2label_,
            dropout=self.dropout,
            classifier_dropout=self.classifier_dropout,
            positional_dropout=self.positional_dropout,
            att_depth=self.att_depth,
            att_heads=self.att_heads,
            att_dim=self.att_dim,
            att_mlp_dim=self.att_mlp_dim,
            n_filters_time=self.n_filters_time,
            filter_time_length=self.filter_time_length,
            depth_multiplier=self.depth_multiplier,
            pool_time_length=self.pool_time_length,
            pool_time_stride=self.pool_time_stride,
            second_filter_time_length=self.second_filter_time_length,
            second_pool_time_length=self.second_pool_time_length,
            second_pool_time_stride=self.second_pool_time_stride,
            max_position_embeddings=self.max_position_embeddings,
        )
        self.device_ = _resolve_device(self.device)
        self.model_ = CtnetForEEGClassification(config).to(self.device_)
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.lr,
            betas=(self.adam_beta1, self.adam_beta2),
            weight_decay=self.weight_decay,
        )
        generator = torch.Generator().manual_seed(self.seed)
        train_batch_size = self.batch_size
        if self.validation_ratio:
            train_batch_size -= int(self.validation_ratio * self.batch_size)
        loader = DataLoader(
            TensorDataset(
                torch.as_tensor(X_train, dtype=torch.float32),
                torch.as_tensor(y_train, dtype=torch.long),
            ),
            batch_size=max(train_batch_size, 1),
            shuffle=True,
            num_workers=self.num_workers,
            generator=generator,
        )

        best_state = None
        best_loss = float("inf")
        self.best_epoch_ = self.epochs
        for epoch in range(1, self.epochs + 1):
            train_loss = _train_one_epoch(
                self.model_,
                loader,
                optimizer,
                self.device_,
                X_train,
                y_train,
                rng,
                self.augmentation_factor,
                self.augmentation_segments,
                self.batch_size,
                len(self.classes_),
            )
            if len(X_validation):
                validation_loss = _evaluate_loss(
                    self.model_,
                    X_validation,
                    y_validation,
                    self.batch_size,
                    self.num_workers,
                    self.device_,
                )
                if validation_loss < best_loss:
                    best_loss = validation_loss
                    self.best_epoch_ = epoch
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in self.model_.state_dict().items()
                    }
            else:
                validation_loss = float("nan")
            if self.verbose and (epoch == 1 or epoch % 25 == 0):
                print(
                    f"epoch={epoch:04d} train_loss={train_loss:.4f} "
                    f"val_loss={validation_loss:.4f}"
                )

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.best_validation_loss_ = (
            best_loss if best_state is not None else float("nan")
        )
        return self

    def _fit_standardizer(self, X: np.ndarray) -> None:
        if not self.standardize:
            self.mean_ = None
            self.std_ = None
        elif self.standardize_mode == "global":
            self.mean_ = np.asarray(X.mean(), dtype=np.float32)
            self.std_ = np.asarray(max(float(X.std()), 1e-6), dtype=np.float32)
        else:
            self.mean_ = X.mean(axis=(0, 2), keepdims=True)
            self.std_ = np.maximum(X.std(axis=(0, 2), keepdims=True), 1e-6)

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            return X
        return (X - self.mean_) / self.std_

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        encoded = probabilities.argmax(axis=1)
        return self.classes_[encoded]

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise ValueError("This CtnetSklearnClassifier instance is not fitted yet.")

        X_eval = _select_input_samples(
            _validate_eeg_array(X), self.input_samples
        ).astype(np.float32, copy=True)
        X_eval = self._standardize(X_eval)
        loader = DataLoader(
            torch.as_tensor(X_eval, dtype=torch.float32),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        self.model_.eval()
        probabilities = []
        for input_values in loader:
            logits = self.model_(input_values=input_values.to(self.device_)).logits
            probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(probabilities, axis=0)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        predictions = self.predict(X)
        return float(np.mean(predictions == np.asarray(y).reshape(-1)))


def _validate_eeg_array(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(
            f"Expected EEG array with shape (trials, channels, time), got {X.shape}."
        )
    return X


def _select_input_samples(X: np.ndarray, input_samples: int | None) -> np.ndarray:
    if input_samples is None:
        return X
    if X.shape[-1] < input_samples:
        raise ValueError(
            f"Expected at least {input_samples} time samples, got {X.shape[-1]}."
        )
    return X[..., :input_samples]


def _make_synthetic_trials(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    augmentation_factor: int,
    segments: int,
    batch_size: int,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    per_class = augmentation_factor * (batch_size // n_classes)
    if per_class < 1 or segments < 1:
        return None

    boundaries = np.linspace(0, X.shape[-1], segments + 1, dtype=int)
    augmented_data = []
    augmented_labels = []
    for class_id in range(n_classes):
        class_data = X[y == class_id]
        if not len(class_data):
            continue
        source_indices = rng.integers(0, len(class_data), size=(per_class, segments))
        synthetic = np.empty((per_class, X.shape[1], X.shape[2]), dtype=np.float32)
        for segment_id, (start, stop) in enumerate(
            zip(boundaries[:-1], boundaries[1:])
        ):
            synthetic[..., start:stop] = class_data[
                source_indices[:, segment_id], :, start:stop
            ]
        augmented_data.append(synthetic)
        augmented_labels.append(np.full(per_class, class_id, dtype=np.int64))
    if not augmented_data:
        return None
    return np.concatenate(augmented_data), np.concatenate(augmented_labels)


def _train_one_epoch(
    model: CtnetForEEGClassification,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    augmentation_data: np.ndarray,
    augmentation_labels: np.ndarray,
    rng: np.random.Generator,
    augmentation_factor: int,
    augmentation_segments: int,
    batch_size: int,
    n_classes: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for input_values, labels in loader:
        augmented = _make_synthetic_trials(
            augmentation_data,
            augmentation_labels,
            rng,
            augmentation_factor,
            augmentation_segments,
            batch_size,
            n_classes,
        )
        if augmented is not None:
            synthetic_values, synthetic_labels = augmented
            input_values = torch.cat(
                [input_values, torch.from_numpy(synthetic_values)], dim=0
            )
            labels = torch.cat([labels, torch.from_numpy(synthetic_labels)], dim=0)
        input_values = input_values.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(input_values=input_values, labels=labels)
        outputs.loss.backward()
        optimizer.step()
        current_batch_size = int(labels.shape[0])
        total_loss += float(outputs.loss.detach()) * current_batch_size
        total_examples += current_batch_size
    return total_loss / max(total_examples, 1)


@torch.no_grad()
def _evaluate_loss(
    model: CtnetForEEGClassification,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> float:
    if not len(X):
        return float("-inf")
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(X, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    model.eval()
    total_loss = 0.0
    total_examples = 0
    for input_values, labels in loader:
        labels = labels.to(device)
        loss = model(input_values=input_values.to(device), labels=labels).loss
        current_batch_size = int(labels.shape[0])
        total_loss += float(loss) * current_batch_size
        total_examples += current_batch_size
    return total_loss / max(total_examples, 1)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
