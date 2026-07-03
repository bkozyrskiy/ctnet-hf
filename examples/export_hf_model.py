"""Build the single untrained Hugging Face CTNet model repository."""

from __future__ import annotations

from pathlib import Path
import shutil

import torch

from ctnet_hf import CtnetConfig, CtnetForEEGClassification


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "hf_model"


def main() -> None:
    for filename in (
        "config.json",
        "configuration_ctnet.py",
        "model.safetensors",
        "modeling_ctnet.py",
        "preprocessing.py",
        "LICENSE",
    ):
        path = OUTPUT_DIR / filename
        if path.exists():
            path.unlink()

    torch.manual_seed(0)
    config = CtnetConfig(
        architecture="paper",
        n_channels=22,
        n_times=1000,
        sampling_rate=250,
        num_labels=4,
    )
    config.initialization = "random"
    config.initialization_seed = 0
    config.pretrained = False

    model = CtnetForEEGClassification(config)
    model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
    shutil.copy2(PROJECT_ROOT / "LICENSE", OUTPUT_DIR / "LICENSE")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Wrote {parameter_count:,}-parameter untrained model to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
