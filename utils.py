from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_checkpoint(path: str, payload: dict) -> None:
    ensure_dir(Path(path).parent)
    torch.save(payload, path)


def save_json(path: str, payload: dict) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def make_run_dir(base_dir: str | Path = "runs", run_name: str | None = None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name is None:
        run_name = f"attention_run_{timestamp}"
    run_dir = Path(base_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def compute_global_attention_scale(attentions, percentile=95):
    values = []

    for attn in attentions:
        if attn is None:
            continue
        values.append(attn.detach().cpu().float().reshape(-1).numpy())

    if not values:
        raise ValueError("No attention tensors found to compute scale.")

    all_values = np.concatenate(values)
    vmax = np.percentile(all_values, percentile)
    return 0.0, float(vmax)

def plot_attention(
    tokens,
    attn,
    save_path,
    title,
    cmap="magma",
    vmin=0.0,
    vmax=None,
):
    attn_np = attn.detach().cpu().float().numpy()

    plt.figure(figsize=(8, 7))
    plt.imshow(
        attn_np,
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(tokens)), tokens, fontsize=8)
    plt.title(title)
    plt.xlabel("Key positions")
    plt.ylabel("Query positions")
    plt.colorbar()
    plt.tight_layout(pad=0.5)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()

def save_all_attention_maps(
    tokens,
    attentions,
    run_dir,
    cmap="magma",
    variant_name="unknown",
):
    run_dir = Path(run_dir)

    vmin, vmax = compute_global_attention_scale(attentions, percentile=95)

    for layer_idx, attn in enumerate(attentions):
        if attn is None:
            continue

        if attn.dim() != 4:
            raise ValueError(
                f"Expected attention tensor of shape (B, H, T, T), got shape {tuple(attn.shape)}"
            )

        _, num_heads, _, _ = attn.shape
        layer_dir = run_dir / f"layer_{layer_idx:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        for head_idx in range(num_heads):
            attn_map = attn[0, head_idx, :, :]
            save_path = layer_dir / f"head_{head_idx:02d}.png"
            title = f"Attention map | variant={variant_name} | layer={layer_idx} | head={head_idx}"

            plot_attention(
                tokens=tokens,
                attn=attn_map,
                save_path=save_path,
                title=title,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )

def save_loss_curve(
    train_steps: list[int],
    train_losses: list[float],
    val_steps: list[int],
    val_losses: list[float],
    save_path: str,
    title: str,
) -> None:
    plt.figure(figsize=(8, 5))

    if train_steps and train_losses:
        plt.plot(train_steps, train_losses, label="train_loss")

    if val_steps and val_losses:
        plt.plot(val_steps, val_losses, label="val_loss")

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()



def make_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_generation_dir(run_dir: str | Path, gen_name: str | None = None) -> Path:
    run_dir = Path(run_dir)
    generations_dir = run_dir / "generations"
    generations_dir.mkdir(parents=True, exist_ok=True)

    if gen_name is None:
        gen_name = f"gen_{make_timestamp()}"

    gen_dir = generations_dir / gen_name
    gen_dir.mkdir(parents=True, exist_ok=True)
    return gen_dir