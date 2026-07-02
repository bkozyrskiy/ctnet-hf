"""Re-export a trained CTNet checkpoint as a validated Hugging Face bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from ctnet_hf import (
    CtnetForEEGClassification,
    CtnetPreprocessor,
    export_huggingface_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Load trained CTNet weights and their fitted preprocessor, then "
            "write a self-contained safetensors bundle."
        )
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Directory or Hub id containing trained weights and preprocessing.",
    )
    parser.add_argument("output", type=Path, help="Destination bundle directory.")
    args = parser.parse_args()

    model = CtnetForEEGClassification.from_pretrained(args.source)
    preprocessor = CtnetPreprocessor.from_pretrained(args.source)
    export_huggingface_bundle(model, preprocessor, args.output)
    print(f"Exported validated bundle to {args.output.resolve()}")
    print("Add the checkpoint-specific card and release evidence before upload.")


if __name__ == "__main__":
    main()
