"""Train one pooled CTNet checkpoint on all BNCI2014-001 training sessions."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ctnet_hf import CtnetConfig, CtnetForEEGClassification, CtnetPreprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHANNEL_NAMES = [
    "Fz",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "P1",
    "Pz",
    "P2",
    "POz",
]


@dataclass
class SubjectSplit:
    subject: int
    values: np.ndarray
    labels: np.ndarray


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one paper CTNet on every selected subject's training session "
            "and select the checkpoint by mean subject accuracy on the test sessions."
        )
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="*",
        help="Subject ids (default: every BNCI2014-001 subject).",
    )
    parser.add_argument("--train-session", default="0train")
    parser.add_argument("--test-session", default="1test")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=72)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.5)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--augmentation-factor", type=int, default=3)
    parser.add_argument("--augmentation-segments", type=int, default=8)
    parser.add_argument(
        "--selection-metric",
        choices=("accuracy", "cohen_kappa"),
        default="accuracy",
    )
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sampling-rate", type=int, default=250)
    parser.add_argument("--input-samples", type=int, default=1000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/trained_hf_model"),
    )
    parser.add_argument(
        "--non-deterministic",
        action="store_true",
        help="Allow faster nondeterministic CUDA operations.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if args.eval_every < 1:
        parser.error("--eval-every must be positive")
    if args.augmentation_factor < 0 or args.augmentation_segments < 1:
        parser.error("augmentation factor must be non-negative and segments positive")
    if args.sampling_rate != 250:
        parser.error("BNCI2014-001 is fixed at 250 Hz; --sampling-rate must be 250")
    return args


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    _set_seed(args.seed, deterministic=not args.non_deterministic)
    device = _resolve_device(args.device)
    dataset, paradigm = _make_dataset_and_paradigm(args)
    subjects = list(args.subjects or dataset.subject_list)

    print(f"Loading BNCI2014-001 subjects: {subjects}")
    train_splits, test_splits, label_names = _load_splits(
        dataset, paradigm, subjects, args
    )
    mean, std = _fit_global_standardizer(train_splits)
    _standardize_in_place(train_splits + test_splits, mean, std)
    train_values = np.concatenate([split.values for split in train_splits])
    train_labels = np.concatenate([split.labels for split in train_splits])
    train_subjects = np.concatenate(
        [
            np.full(len(split.labels), split.subject, dtype=np.int64)
            for split in train_splits
        ]
    )

    id2label = {index: label for index, label in enumerate(label_names)}
    label2id = {label: index for index, label in id2label.items()}
    config = CtnetConfig(
        architecture="paper",
        n_channels=int(train_values.shape[1]),
        n_times=int(train_values.shape[2]),
        sampling_rate=args.sampling_rate,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
    )
    config.initialization = "trained"
    config.pretrained = True
    config.training_dataset = "BNCI2014_001"
    config.training_subjects = subjects
    config.train_session = args.train_session
    config.checkpoint_selection_session = args.test_session
    config.checkpoint_selection_metric = f"mean_subject_{args.selection_metric}"
    config.channel_names = CHANNEL_NAMES
    config.input_unit = "microvolts"

    model = CtnetForEEGClassification(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_values),
            torch.from_numpy(train_labels),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=generator,
    )
    rng = np.random.default_rng(args.seed)
    preprocessor = CtnetPreprocessor(
        n_channels=config.n_channels,
        n_times=config.n_times,
        sampling_rate=config.sampling_rate,
        mean=mean,
        std=std,
        channel_names=CHANNEL_NAMES,
        dataset="BNCI2014_001",
        subjects=subjects,
        unit="microvolts",
        train_session=args.train_session,
        selection_session=args.test_session,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "training_history.csv"
    checkpoint_dir = args.output_dir / "checkpoints" / "best"
    best_state = None
    best_epoch = 0
    best_score = -float("inf")
    best_metrics = None
    history_fields = _history_fields(subjects)
    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=history_fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_loss = _train_one_epoch(
                model=model,
                loader=loader,
                optimizer=optimizer,
                device=device,
                augmentation_values=train_values,
                augmentation_labels=train_labels,
                augmentation_subjects=train_subjects,
                rng=rng,
                augmentation_factor=args.augmentation_factor,
                augmentation_segments=args.augmentation_segments,
                configured_batch_size=args.batch_size,
                n_classes=len(label_names),
            )
            if epoch % args.eval_every != 0 and epoch != args.epochs:
                if not args.quiet:
                    print(f"epoch={epoch:04d} train_loss={train_loss:.6f}")
                continue

            metrics = _evaluate_subjects(model, test_splits, args.batch_size, device)
            score = metrics[f"mean_subject_{args.selection_metric}"]
            row = {"epoch": epoch, "train_loss": train_loss, **metrics}
            writer.writerow(row)
            history_file.flush()
            improved = score > best_score
            if improved:
                best_score = score
                best_epoch = epoch
                best_metrics = metrics
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                _save_hf_checkpoint(model, preprocessor, checkpoint_dir)
                (checkpoint_dir / "selection_metrics.json").write_text(
                    json.dumps(
                        {"epoch": epoch, "selection_score": score, **metrics},
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            if not args.quiet:
                marker = " *" if improved else ""
                print(
                    f"epoch={epoch:04d} train_loss={train_loss:.6f} "
                    f"mean_accuracy={metrics['mean_subject_accuracy']:.4f} "
                    f"mean_kappa={metrics['mean_subject_cohen_kappa']:.4f}{marker}"
                )

    if best_state is None or best_metrics is None:
        raise RuntimeError("No checkpoint was evaluated.")
    model.load_state_dict(best_state)
    model.to(device)
    _save_hf_checkpoint(model, preprocessor, args.output_dir)
    elapsed = time.perf_counter() - started
    metadata = _training_metadata(
        args=args,
        subjects=subjects,
        label_names=label_names,
        train_splits=train_splits,
        test_splits=test_splits,
        mean=mean,
        std=std,
        best_epoch=best_epoch,
        best_score=best_score,
        best_metrics=best_metrics,
        elapsed=elapsed,
        device=device,
    )
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "README.md").write_text(
        _render_model_card(metadata), encoding="utf-8"
    )
    for filename in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        shutil.copy2(PROJECT_ROOT / filename, args.output_dir / filename)
    shutil.rmtree(args.output_dir / "checkpoints", ignore_errors=True)

    print(f"Best epoch: {best_epoch} ({args.selection_metric}={best_score:.4f})")
    print(f"Saved upload-ready Hugging Face checkpoint to {args.output_dir.resolve()}")
    print(
        "Warning: the test sessions selected the checkpoint, so their metrics are "
        "tuning metrics rather than an unbiased final test result."
    )


def _make_dataset_and_paradigm(args: argparse.Namespace):
    try:
        import moabb
        from moabb.datasets import BNCI2014_001
        from moabb.datasets.preprocessing import make_fixed_pipeline
        from moabb.paradigms import MotorImagery
    except ImportError as exc:
        raise SystemExit(
            "MOABB is required. Install the training dependencies with "
            "`pip install -e '.[benchmark]'`."
        ) from exc

    moabb.set_log_level("warning")

    class PaperMotorImagery(MotorImagery):
        def _get_raw_pipelines(self):
            return [make_fixed_pipeline(None)]

    dataset = BNCI2014_001()
    paradigm = PaperMotorImagery(
        n_classes=None,
        tmax=(args.input_samples - 1) / args.sampling_rate,
    )
    return dataset, paradigm


def _load_splits(dataset, paradigm, subjects, args):
    raw_splits = []
    label_set: set[str] = set()
    for subject in subjects:
        values, labels, metadata = paradigm.get_data(dataset=dataset, subjects=[subject])
        if values.shape[1] != len(CHANNEL_NAMES):
            raise ValueError(
                f"Expected {len(CHANNEL_NAMES)} EEG channels, got {values.shape[1]}."
            )
        if values.shape[2] < args.input_samples:
            raise ValueError(
                f"Expected at least {args.input_samples} samples, got {values.shape[2]}."
            )
        labels = np.asarray(labels).astype(str)
        sessions = metadata["session"].astype(str).to_numpy()
        train_mask = sessions == args.train_session
        test_mask = sessions == args.test_session
        if not train_mask.any() or not test_mask.any():
            available = ", ".join(sorted(set(sessions)))
            raise SystemExit(
                f"Subject {subject} lacks {args.train_session!r} or "
                f"{args.test_session!r}; available sessions: {available}"
            )
        label_set.update(labels.tolist())
        raw_splits.append(
            (
                subject,
                np.asarray(values[train_mask, :, : args.input_samples], dtype=np.float32),
                labels[train_mask],
                np.asarray(values[test_mask, :, : args.input_samples], dtype=np.float32),
                labels[test_mask],
            )
        )

    label_names = sorted(label_set)
    label2id = {label: index for index, label in enumerate(label_names)}
    train_splits = []
    test_splits = []
    for subject, train_x, train_y, test_x, test_y in raw_splits:
        train_splits.append(
            SubjectSplit(subject, train_x, _encode_labels(train_y, label2id))
        )
        test_splits.append(
            SubjectSplit(subject, test_x, _encode_labels(test_y, label2id))
        )
    return train_splits, test_splits, label_names


def _encode_labels(labels: np.ndarray, label2id: dict[str, int]) -> np.ndarray:
    try:
        return np.asarray([label2id[str(label)] for label in labels], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Unknown label: {exc.args[0]}") from exc


def _fit_global_standardizer(splits: list[SubjectSplit]) -> tuple[float, float]:
    count = sum(split.values.size for split in splits)
    total = sum(split.values.sum(dtype=np.float64) for split in splits)
    mean = total / count
    squared_error = sum(
        np.square(split.values.astype(np.float64) - mean).sum() for split in splits
    )
    std = max(float(np.sqrt(squared_error / count)), 1e-6)
    return float(mean), std


def _standardize_in_place(
    splits: list[SubjectSplit], mean: float, std: float
) -> None:
    for split in splits:
        np.subtract(split.values, mean, out=split.values)
        np.divide(split.values, std, out=split.values)


def _make_subject_aware_synthetic(
    values: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    rng: np.random.Generator,
    factor: int,
    segments: int,
    configured_batch_size: int,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    per_class = factor * (configured_batch_size // n_classes)
    if per_class < 1:
        return None
    boundaries = np.linspace(0, values.shape[-1], segments + 1, dtype=int)
    synthetic = []
    synthetic_labels = []
    for class_id in range(n_classes):
        class_subjects = np.unique(subjects[labels == class_id])
        if not len(class_subjects):
            continue
        chosen_subjects = rng.choice(class_subjects, size=per_class, replace=True)
        class_values = np.empty(
            (per_class, values.shape[1], values.shape[2]), dtype=np.float32
        )
        for subject in class_subjects:
            rows = np.flatnonzero(chosen_subjects == subject)
            if not len(rows):
                continue
            source_pool = np.flatnonzero((labels == class_id) & (subjects == subject))
            source_rows = rng.choice(
                source_pool, size=(len(rows), segments), replace=True
            )
            for segment_id, (start, stop) in enumerate(
                zip(boundaries[:-1], boundaries[1:])
            ):
                class_values[rows, :, start:stop] = values[
                    source_rows[:, segment_id], :, start:stop
                ]
        synthetic.append(class_values)
        synthetic_labels.append(np.full(per_class, class_id, dtype=np.int64))
    if not synthetic:
        return None
    return np.concatenate(synthetic), np.concatenate(synthetic_labels)


def _train_one_epoch(
    *,
    model,
    loader,
    optimizer,
    device,
    augmentation_values,
    augmentation_labels,
    augmentation_subjects,
    rng,
    augmentation_factor,
    augmentation_segments,
    configured_batch_size,
    n_classes,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for input_values, labels in loader:
        augmented = _make_subject_aware_synthetic(
            augmentation_values,
            augmentation_labels,
            augmentation_subjects,
            rng,
            augmentation_factor,
            augmentation_segments,
            configured_batch_size,
            n_classes,
        )
        if augmented is not None:
            synthetic_values, synthetic_labels = augmented
            input_values = torch.cat((input_values, torch.from_numpy(synthetic_values)))
            labels = torch.cat((labels, torch.from_numpy(synthetic_labels)))
        input_values = input_values.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_values=input_values, labels=labels).loss
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * len(labels)
        total_examples += len(labels)
    return total_loss / total_examples


@torch.no_grad()
def _evaluate_subjects(model, splits, batch_size, device) -> dict[str, float]:
    model.eval()
    metrics = {}
    accuracies = []
    kappas = []
    losses = []
    pooled_correct = 0
    pooled_count = 0
    for split in splits:
        loader = DataLoader(
            TensorDataset(torch.from_numpy(split.values), torch.from_numpy(split.labels)),
            batch_size=batch_size,
            shuffle=False,
        )
        predictions = []
        total_loss = 0.0
        for values, labels in loader:
            labels = labels.to(device)
            output = model(input_values=values.to(device), labels=labels)
            total_loss += float(output.loss) * len(labels)
            predictions.append(output.logits.argmax(dim=-1).cpu().numpy())
        predicted = np.concatenate(predictions)
        accuracy = float(np.mean(predicted == split.labels))
        kappa = _cohen_kappa(split.labels, predicted)
        loss = total_loss / len(split.labels)
        metrics[f"subject_{split.subject}_accuracy"] = accuracy
        metrics[f"subject_{split.subject}_cohen_kappa"] = kappa
        metrics[f"subject_{split.subject}_loss"] = loss
        accuracies.append(accuracy)
        kappas.append(kappa)
        losses.append(loss)
        pooled_correct += int(np.sum(predicted == split.labels))
        pooled_count += len(split.labels)
    metrics["mean_subject_accuracy"] = float(np.mean(accuracies))
    metrics["mean_subject_cohen_kappa"] = float(np.mean(kappas))
    metrics["mean_subject_loss"] = float(np.mean(losses))
    metrics["pooled_accuracy"] = pooled_correct / pooled_count
    return metrics


def _cohen_kappa(labels: np.ndarray, predictions: np.ndarray) -> float:
    n_classes = int(max(labels.max(), predictions.max()) + 1)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    observed = np.trace(confusion) / confusion.sum()
    expected = (
        np.dot(confusion.sum(axis=0), confusion.sum(axis=1))
        / confusion.sum() ** 2
    )
    return float((observed - expected) / (1.0 - expected)) if expected < 1.0 else 0.0


def _history_fields(subjects: list[int]) -> list[str]:
    fields = [
        "epoch",
        "train_loss",
        "mean_subject_accuracy",
        "mean_subject_cohen_kappa",
        "mean_subject_loss",
        "pooled_accuracy",
    ]
    for subject in subjects:
        fields.extend(
            [
                f"subject_{subject}_accuracy",
                f"subject_{subject}_cohen_kappa",
                f"subject_{subject}_loss",
            ]
        )
    return fields


def _save_hf_checkpoint(model, preprocessor, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination, safe_serialization=True)
    preprocessor.save_pretrained(destination)


def _training_metadata(**values) -> dict:
    args = values["args"]
    return {
        "dataset": "BNCI2014_001",
        "architecture": "paper",
        "subjects": values["subjects"],
        "train_session": args.train_session,
        "checkpoint_selection_session": args.test_session,
        "selection_warning": (
            "The nominal test sessions were inspected every evaluation epoch and "
            "used for model selection; reported scores are tuning metrics."
        ),
        "selection_metric": f"mean_subject_{args.selection_metric}",
        "best_epoch": values["best_epoch"],
        "best_score": values["best_score"],
        "best_metrics": values["best_metrics"],
        "label_names": values["label_names"],
        "channel_names": CHANNEL_NAMES,
        "input_unit": "microvolts",
        "sampling_rate": args.sampling_rate,
        "input_samples": args.input_samples,
        "normalization": {
            "type": "global_zscore",
            "mean": values["mean"],
            "std": values["std"],
        },
        "n_train_trials": sum(len(split.labels) for split in values["train_splits"]),
        "n_selection_trials": sum(len(split.labels) for split in values["test_splits"]),
        "train_trials_by_subject": {
            str(split.subject): len(split.labels) for split in values["train_splits"]
        },
        "selection_trials_by_subject": {
            str(split.subject): len(split.labels) for split in values["test_splits"]
        },
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "adam_betas": [args.adam_beta1, args.adam_beta2],
            "augmentation": "subject-aware segmentation-and-reconstruction",
            "augmentation_factor": args.augmentation_factor,
            "augmentation_segments": args.augmentation_segments,
            "eval_every": args.eval_every,
            "seed": args.seed,
            "deterministic": not args.non_deterministic,
        },
        "device": str(values["device"]),
        "training_seconds": values["elapsed"],
    }


def _render_model_card(metadata: dict) -> str:
    rows = []
    for subject in metadata["subjects"]:
        accuracy = metadata["best_metrics"][f"subject_{subject}_accuracy"]
        kappa = metadata["best_metrics"][f"subject_{subject}_cohen_kappa"]
        rows.append(f"| {subject} | {accuracy:.4f} | {kappa:.4f} |")
    return f"""---
library_name: transformers
license: mit
metrics:
  - accuracy
tags:
  - eeg
  - brain-computer-interface
  - motor-imagery
  - bnci2014-001
  - ct-net
---

# CTNet trained on all BNCI2014-001 training sessions

This checkpoint is one paper-architecture CTNet trained on session
`{metadata['train_session']}` from subjects {metadata['subjects']}. It expects
22 channels, {metadata['input_samples']:,} samples at
{metadata['sampling_rate']} Hz, and microvolt values in the channel order recorded by
`preprocessor_config.json`.

## Important evaluation caveat

Session `{metadata['checkpoint_selection_session']}` was evaluated during training and
selected epoch {metadata['best_epoch']} by `{metadata['selection_metric']}`. These are
therefore **checkpoint-tuning metrics, not unbiased held-out test metrics**.

## Load

```python
from transformers import AutoFeatureExtractor, AutoModelForSequenceClassification

processor = AutoFeatureExtractor.from_pretrained(REPO_ID, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    REPO_ID, trust_remote_code=True
).eval()
inputs = processor(eeg_trial, return_tensors="pt")  # (22, 1000), microvolts
logits = model(**inputs).logits
```

## Checkpoint-selection metrics

| Subject | Accuracy | Cohen's kappa |
|---:|---:|---:|
{chr(10).join(rows)}

Mean subject accuracy: {metadata['best_metrics']['mean_subject_accuracy']:.4f}  
Mean subject Cohen's kappa: {metadata['best_metrics']['mean_subject_cohen_kappa']:.4f}

See `training_metadata.json` and `training_history.csv` for the complete protocol.
This research checkpoint is not a medical device and has not been validated for
clinical or safety-critical use.
"""


def _set_seed(seed: int, *, deterministic: bool) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


if __name__ == "__main__":
    main()
