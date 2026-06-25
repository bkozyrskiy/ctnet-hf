from __future__ import annotations

import json
from pathlib import Path

from ctnet_hf import CtnetConfig, CtnetForEEGClassification


def main() -> None:
    output_dir = Path("artifacts/ctnet_architecture_only")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = CtnetConfig()
    _ = CtnetForEEGClassification(config)

    config.save_pretrained(output_dir)
    preprocessor_config = {
        "input_shape": ["batch_size", "n_channels", "n_times"],
        "expected_units": (
            "microvolts or normalized EEG values depending on training protocol"
        ),
        "sampling_rate": config.sampling_rate,
        "filtering": "not applied by this model; preprocessing must be handled externally",
        "normalization": (
            "not applied by this model; normalization must match the training protocol"
        ),
    }
    (output_dir / "preprocessor_config.json").write_text(
        json.dumps(preprocessor_config, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved architecture-only bundle to {output_dir.resolve()}")
    print("No pretrained weights are included in this export.")


if __name__ == "__main__":
    main()
