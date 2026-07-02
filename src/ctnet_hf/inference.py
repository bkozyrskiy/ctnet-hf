"""Minimal command-line inference for a released CTNet bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .modeling_ctnet import CtnetForEEGClassification
from .preprocessing import CtnetPreprocessor


@torch.no_grad()
def predict_trial(
    pretrained_model_name_or_path: str | Path,
    trial: np.ndarray,
    **from_pretrained_kwargs: Any,
) -> dict[str, Any]:
    """Load a complete bundle and predict one real or synthetic EEG trial."""
    processor = CtnetPreprocessor.from_pretrained(
        pretrained_model_name_or_path,
        **from_pretrained_kwargs,
    )
    model = CtnetForEEGClassification.from_pretrained(
        pretrained_model_name_or_path,
        **from_pretrained_kwargs,
    )
    model.eval()
    inputs = processor(trial, return_tensors="pt")
    probabilities = torch.softmax(model(**inputs).logits, dim=-1)[0].cpu().numpy()
    predicted_id = int(probabilities.argmax())
    return {
        "predicted_id": predicted_id,
        "predicted_label": str(model.config.id2label[predicted_id]),
        "probabilities": [float(value) for value in probabilities],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a CTNet model+preprocessor bundle and infer one EEG trial."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Local bundle directory or Hugging Face Hub repository id.",
    )
    parser.add_argument(
        "--trial",
        type=Path,
        required=True,
        help="NumPy .npy file with shape (channels, time).",
    )
    parser.add_argument(
        "--expected-prediction",
        help="Fail if the predicted label does not match this value.",
    )
    args = parser.parse_args()

    trial = np.load(args.trial, allow_pickle=False)
    result = predict_trial(args.model, trial)
    if (
        args.expected_prediction is not None
        and result["predicted_label"] != args.expected_prediction
    ):
        raise SystemExit(
            f"Expected prediction {args.expected_prediction!r}, got "
            f"{result['predicted_label']!r}."
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
