"""Export and validate self-contained Hugging Face CTNet bundles."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

from .modeling_ctnet import CtnetForEEGClassification
from .preprocessing import CtnetPreprocessor


CORE_BUNDLE_FILES = (
    "config.json",
    "configuration_ctnet.py",
    "model.safetensors",
    "modeling_ctnet.py",
    "preprocessing.py",
    "preprocessor_config.json",
)

DOCUMENTED_BUNDLE_FILES = (
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "training_metadata.json",
    "export_reload_equivalence.json",
    "release_manifest.json",
)


def export_huggingface_bundle(
    model: CtnetForEEGClassification,
    preprocessor: CtnetPreprocessor,
    save_directory: str | Path,
) -> Path:
    """Save weights, custom code, configuration, and preprocessing together."""
    _validate_model_preprocessor_contract(model, preprocessor)
    destination = Path(save_directory)
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination, safe_serialization=True)
    preprocessor.save_pretrained(destination)
    validate_huggingface_bundle(destination)
    return destination


def install_release_documents(
    bundle_directory: str | Path,
    *,
    license_path: str | Path,
    notices_path: str | Path,
) -> None:
    """Copy the source and third-party notices into a publishable bundle."""
    bundle_directory = Path(bundle_directory)
    shutil.copy2(license_path, bundle_directory / "LICENSE")
    shutil.copy2(notices_path, bundle_directory / "THIRD_PARTY_NOTICES.md")


def validate_huggingface_bundle(
    bundle_directory: str | Path,
    *,
    require_documentation: bool = False,
) -> dict[str, Any]:
    """Validate bundle completeness and its model/preprocessor contract."""
    bundle_directory = Path(bundle_directory)
    required = list(CORE_BUNDLE_FILES)
    if require_documentation:
        required.extend(DOCUMENTED_BUNDLE_FILES)
    missing = [name for name in required if not (bundle_directory / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete Hugging Face bundle; missing: {', '.join(missing)}")
    if list(bundle_directory.glob("pytorch_model*.bin")):
        raise ValueError("Release bundles must use safetensors, not pickle weights.")

    config = _read_json(bundle_directory / "config.json")
    preprocessing = _read_json(bundle_directory / "preprocessor_config.json")
    for field in ("n_channels", "n_times", "sampling_rate"):
        if config.get(field) != preprocessing.get(field):
            raise ValueError(
                f"Model and preprocessor disagree on {field}: "
                f"{config.get(field)!r} != {preprocessing.get(field)!r}."
            )
    if not preprocessing.get("channel_names"):
        raise ValueError("A release preprocessor must record channel_names.")
    if preprocessing.get("standardize") and (
        preprocessing.get("mean") is None or preprocessing.get("std") is None
    ):
        raise ValueError("A standardized release must include its fitted mean and std.")

    _validate_auto_map_files(bundle_directory, config.get("auto_map", {}))
    _validate_auto_map_files(bundle_directory, preprocessing.get("auto_map", {}))
    if "AutoFeatureExtractor" not in preprocessing.get("auto_map", {}):
        raise ValueError("preprocessor_config.json must map AutoFeatureExtractor.")
    return {"config": config, "preprocessor": preprocessing}


def write_release_manifest(
    bundle_directory: str | Path,
    *,
    filename: str = "release_manifest.json",
) -> Path:
    """Write portable SHA-256 and size records for every publishable file."""
    bundle_directory = Path(bundle_directory)
    manifest_path = bundle_directory / filename
    files = []
    for path in sorted(bundle_directory.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        files.append(
            {
                "path": path.relative_to(bundle_directory).as_posix(),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    payload = {"format_version": 1, "files": files}
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _validate_model_preprocessor_contract(
    model: CtnetForEEGClassification,
    preprocessor: CtnetPreprocessor,
) -> None:
    for field in ("n_channels", "n_times", "sampling_rate"):
        model_value = getattr(model.config, field)
        preprocessor_value = getattr(preprocessor, field)
        if model_value != preprocessor_value:
            raise ValueError(
                f"Model and preprocessor disagree on {field}: "
                f"{model_value!r} != {preprocessor_value!r}."
            )


def _validate_auto_map_files(
    bundle_directory: Path,
    auto_map: dict[str, Any],
) -> None:
    references: Iterable[Any] = auto_map.values()
    for reference in references:
        if isinstance(reference, (list, tuple)):
            values = reference
        else:
            values = (reference,)
        for value in values:
            if not value:
                continue
            module = str(value).split("--")[-1].rsplit(".", 1)[0]
            module_path = bundle_directory / f"{module}.py"
            if not module_path.is_file():
                raise ValueError(
                    f"auto_map references {value!r}, but {module_path.name} is missing."
                )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read valid JSON from {path}.") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
