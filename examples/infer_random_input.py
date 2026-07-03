from __future__ import annotations

import torch

from ctnet_hf import CtnetConfig, CtnetForEEGClassification


def main() -> None:
    config = CtnetConfig(architecture="paper")
    model = CtnetForEEGClassification(config)
    x = torch.randn(2, config.n_channels, config.n_times)

    with torch.no_grad():
        outputs = model(input_values=x)

    print(f"logits shape: {tuple(outputs.logits.shape)}")


if __name__ == "__main__":
    main()
