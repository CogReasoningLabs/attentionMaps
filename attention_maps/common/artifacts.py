from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


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


def find_run_dir_from_checkpoint(checkpoint_path: str | Path) -> Path:
    """Resolve ``runs/<name>`` from its ``checkpoints/*.pt`` artifact."""

    checkpoint = Path(checkpoint_path).resolve()
    if checkpoint.parent.name != "checkpoints":
        raise ValueError(
            f"Could not infer run directory from {checkpoint}. Expected "
            "runs/<run_name>/checkpoints/<checkpoint>.pt"
        )
    return checkpoint.parent.parent


def make_attention_dir(
    run_dir: str | Path,
    export_name: str | None = None,
) -> Path:
    """Create a run-scoped directory for an attention inspection."""

    name = export_name or f"attention_{make_timestamp()}"
    output = Path(run_dir) / "attention_maps" / name
    output.mkdir(parents=True, exist_ok=True)
    return output
