from __future__ import annotations

from pathlib import Path

import torch

from ctnet_hf import CtnetConfig, CtnetForEEGClassification


def main() -> None:
    torch.manual_seed(0)

    config = CtnetConfig()
    model = CtnetForEEGClassification(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(12, config.n_channels, config.n_times)
    labels = torch.randint(0, config.num_labels, (12,))

    losses: list[float] = []
    for step in range(3):
        optimizer.zero_grad()
        outputs = model(input_values=x, labels=labels)
        outputs.loss.backward()
        optimizer.step()
        losses.append(float(outputs.loss.detach()))
        print(f"step={step} loss={losses[-1]:.4f}")

    checkpoint_dir = Path("artifacts/ctnet_dummy_checkpoint")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)

    if losses[-1] <= losses[0]:
        print("Loss decreased during the dummy run.")
    else:
        print("Loss did not decrease, but the dummy training loop completed successfully.")

    print(f"Saved dummy checkpoint to {checkpoint_dir.resolve()}")
    print("These weights are only for smoke testing and are not useful pretrained weights.")


if __name__ == "__main__":
    main()
