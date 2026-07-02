"""Refresh existing trained bundles without rerunning expensive experiments."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

from ctnet_hf import (
    CtnetForEEGClassification,
    CtnetPreprocessor,
    export_huggingface_bundle,
)
from ctnet_hf.release import (
    install_release_documents,
    validate_huggingface_bundle,
    write_release_manifest,
)

from .model_card import render_bnci2014_001_card


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()

    results = _read_results(args.results)
    metadata_paths = sorted(args.bundle_root.rglob("training_metadata.json"))
    if not metadata_paths:
        raise SystemExit(f"No bundles found under {args.bundle_root}.")

    for metadata_path in metadata_paths:
        bundle = metadata_path.parent
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        key = (int(metadata["seed"]), int(metadata["subject"]))
        if key not in results:
            raise SystemExit(f"No result row for seed {key[0]}, subject {key[1]}.")
        result = results[key]

        model = CtnetForEEGClassification.from_pretrained(bundle)
        preprocessor = CtnetPreprocessor.from_pretrained(bundle)
        export_huggingface_bundle(model, preprocessor, bundle)

        training_seconds = float(result["time"])
        metadata["training_and_evaluation_seconds"] = training_seconds
        metadata.setdefault(
            "hardware",
            {
                "device": str(metadata.get("hyperparameters", {}).get("device")),
                "name": "not recorded",
            },
        )
        metadata.setdefault("source", _source_metadata(args.project_root))
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        metrics = metadata["metrics"]
        card = render_bnci2014_001_card(
            subject=key[1],
            seed=key[0],
            train_session=str(metadata["train_session"]),
            test_session=str(metadata["test_session"]),
            accuracy=float(metrics["accuracy"]),
            cohen_kappa=float(metrics["cohen_kappa"]),
            best_epoch=int(metadata["best_epoch"]),
            channel_names=metadata.get("channel_names"),
            id2label={int(index): label for index, label in model.config.id2label.items()},
            parameter_count=sum(parameter.numel() for parameter in model.parameters()),
            training_seconds=training_seconds,
        )
        (bundle / "README.md").write_text(card, encoding="utf-8")
        install_release_documents(
            bundle,
            license_path=args.project_root / "LICENSE",
            notices_path=args.project_root / "THIRD_PARTY_NOTICES.md",
        )
        write_release_manifest(bundle)
        validate_huggingface_bundle(bundle, require_documentation=True)
        print(f"Refreshed {bundle}")


def _read_results(path: Path) -> dict[tuple[int, int], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return {(int(row["seed"]), int(row["subject"])): row for row in rows}


def _source_metadata(project_root: Path) -> dict[str, object]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return {
            "git_commit": revision,
            "git_dirty": True,
            "note": "Training used uncommitted release tooling later captured in this source tree.",
        }
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": True}


if __name__ == "__main__":
    main()
