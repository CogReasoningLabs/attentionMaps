"""Reusable, Unicode-aware attention-map rendering functions."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MATPLOTLIB_CACHE = _PROJECT_ROOT / "data" / "cache" / "matplotlib"
_MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CACHE))

import matplotlib.pyplot as plt
from matplotlib import font_manager, ft2font


def format_token_labels(
    tokens: Sequence[str],
    special_tokens: dict[str, str | None] | None = None,
) -> list[str]:
    """Convert serialized subword pieces into compact human-readable labels."""

    role_labels = {
        token: f"<{role.upper()}>"
        for role, token in (special_tokens or {}).items()
        if token is not None
    }
    labels = []
    for raw_token in tokens:
        label = role_labels.get(raw_token, raw_token)
        label = label.replace("\r", "\\r").replace("\n", "\\n")
        if label not in role_labels.values():
            label = label.replace("</w>", "")
            if label.startswith("Ġ"):
                label = "␠" + label[1:]
        labels.append(label or "∅")
    return labels


def compute_global_attention_scale(
    attentions: Iterable[torch.Tensor | None],
    percentile: float = 95.0,
) -> tuple[float, float]:
    """Return a shared color scale for comparable layer/head heatmaps."""

    if not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must be in (0, 100]")
    values = [
        attention.detach().cpu().float().reshape(-1).numpy()
        for attention in attentions
        if attention is not None
    ]
    if not values:
        raise ValueError("No attention tensors found to compute scale")
    vmax = float(np.percentile(np.concatenate(values), percentile))
    return 0.0, vmax


@lru_cache(maxsize=32)
def _font_for_codepoints(codepoints: tuple[int, ...]):
    """Find an installed font covering every non-ASCII label character."""

    if not codepoints:
        return None
    fonts = sorted(
        font_manager.fontManager.ttflist,
        key=lambda entry: (
            "last resort" in entry.name.lower(),
            not entry.name.lower().startswith("noto"),
            not entry.name.lower().startswith("free"),
            entry.name,
        ),
    )
    required = set(codepoints)
    for entry in fonts:
        if "last resort" in entry.name.lower():
            continue
        try:
            if required <= set(ft2font.FT2Font(entry.fname).get_charmap()):
                return font_manager.FontProperties(fname=entry.fname)
        except (OSError, RuntimeError):
            continue
    return None


def unicode_font_for_tokens(tokens: Sequence[str]):
    """Return one font covering every printable character in the labels."""

    codepoints = tuple(
        sorted(
            {
                ord(character)
                for token in tokens
                for character in str(token)
                if not character.isspace()
            }
        )
    )
    return _font_for_codepoints(codepoints)


def _unicode_font_for_token(token: str):
    """Choose a script font per label so ASCII controls keep a Latin font."""

    codepoints = tuple(
        sorted(
            {
                ord(character)
                for character in str(token)
                if ord(character) > 127 and not character.isspace()
            }
        )
    )
    return _font_for_codepoints(codepoints) if codepoints else None


def _validate_single_map(tokens: Sequence[str], attention: torch.Tensor) -> None:
    if attention.ndim != 2:
        raise ValueError(
            "A single attention map must have shape (query_tokens, key_tokens); "
            f"received {tuple(attention.shape)}"
        )
    query_count, key_count = attention.shape
    if query_count != key_count:
        raise ValueError(
            "Token-to-token heatmaps require a square attention matrix; "
            f"received {tuple(attention.shape)}"
        )
    if len(tokens) != key_count:
        raise ValueError(
            f"Received {len(tokens)} token labels for a {key_count}x{key_count} map"
        )


def plot_attention(
    tokens: Sequence[str],
    attention: torch.Tensor,
    save_path: str | Path,
    title: str,
    cmap: str = "magma",
    vmin: float = 0.0,
    vmax: float | None = None,
) -> Path:
    """Render one square token-to-token attention matrix."""

    _validate_single_map(tokens, attention)
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    attention_array = attention.detach().cpu().float().numpy()
    side = min(18.0, max(8.0, 5.0 + 0.3 * len(tokens)))
    figure, axis = plt.subplots(figsize=(side, max(7.0, side - 1.0)))
    image = axis.imshow(
        attention_array,
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    positions = range(len(tokens))
    axis.set_xticks(
        positions,
        tokens,
        rotation=45,
        ha="right",
        fontsize=8,
    )
    axis.set_yticks(
        positions,
        tokens,
        fontsize=8,
    )
    for tick, token in zip(axis.get_xticklabels(), tokens):
        token_font = _unicode_font_for_token(str(token))
        if token_font is not None:
            tick.set_fontproperties(token_font)
    for tick, token in zip(axis.get_yticklabels(), tokens):
        token_font = _unicode_font_for_token(str(token))
        if token_font is not None:
            tick.set_fontproperties(token_font)
    axis.set_title(title)
    axis.set_xlabel("Key positions")
    axis.set_ylabel("Query positions")
    figure.colorbar(image, ax=axis)
    figure.tight_layout(pad=0.5)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_all_attention_maps(
    tokens: Sequence[str],
    attentions: Iterable[torch.Tensor | None],
    output_dir: str | Path,
    cmap: str = "magma",
    variant_name: str = "unknown",
    percentile: float = 95.0,
) -> list[Path]:
    """Save every layer/head with one shared color scale and return its paths."""

    attention_list = list(attentions)
    vmin, vmax = compute_global_attention_scale(attention_list, percentile=percentile)
    root = Path(output_dir)
    saved_paths: list[Path] = []

    for layer_index, attention in enumerate(attention_list):
        if attention is None:
            continue
        if attention.ndim != 4:
            raise ValueError(
                "Expected attention shape (batch, heads, query_tokens, key_tokens); "
                f"received {tuple(attention.shape)}"
            )
        if attention.shape[0] < 1:
            raise ValueError("Attention batch dimension must contain at least one item")
        layer_dir = root / f"layer_{layer_index:02d}"
        for head_index in range(attention.shape[1]):
            title = (
                f"Attention map | variant={variant_name} | "
                f"layer={layer_index} | head={head_index}"
            )
            saved_paths.append(
                plot_attention(
                    tokens=tokens,
                    attention=attention[0, head_index],
                    save_path=layer_dir / f"head_{head_index:02d}.png",
                    title=title,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
            )
    return saved_paths
