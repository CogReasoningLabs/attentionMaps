"""Training-metric visualization helpers."""

from __future__ import annotations

import os
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MATPLOTLIB_CACHE = _PROJECT_ROOT / "data" / "cache" / "matplotlib"
_MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CACHE))

import matplotlib.pyplot as plt


def save_loss_curve(
    train_steps: list[int],
    train_losses: list[float],
    val_steps: list[int],
    val_losses: list[float],
    save_path: str | Path,
    title: str,
) -> Path:
    """Save training and validation loss on a shared step axis."""

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    if train_steps and train_losses:
        axis.plot(train_steps, train_losses, label="train_loss")
    if val_steps and val_losses:
        axis.plot(val_steps, val_losses, label="val_loss")
    axis.set_xlabel("Step")
    axis.set_ylabel("Loss")
    axis.set_title(title)
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path

