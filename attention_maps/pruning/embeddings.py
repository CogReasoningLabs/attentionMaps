"""Batched transformer embedding extraction for cleaned text documents."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .contracts import D2PruningConfig


EmbeddingProgress = Callable[[int, int], None]


def generate_transformer_embeddings(
    documents: Iterable[tuple[str, str]],
    total_documents: int,
    destination: Path,
    config: D2PruningConfig,
    *,
    progress: EmbeddingProgress | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Encode documents without retaining their text in memory."""

    if total_documents < 2:
        raise ValueError("D2 embedding extraction requires at least two documents")
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "D2 transformer embeddings require torch and transformers"
        ) from error

    device = _resolve_device(config.embedding_device, torch)
    load_options = {
        "revision": config.embedding_revision,
        "local_files_only": config.local_files_only,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        config.embedding_model_id, **load_options
    )
    model = AutoModel.from_pretrained(config.embedding_model_id, **load_options)
    model.to(device)
    model.eval()

    destination.parent.mkdir(parents=True, exist_ok=True)
    output = None
    row_ids: list[str] = []
    pending_ids: list[str] = []
    pending_text: list[str] = []
    written = 0

    def flush() -> None:
        nonlocal output, written
        if not pending_text:
            return
        encoded = tokenizer(
            pending_text,
            padding=True,
            truncation=True,
            max_length=config.embedding_max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state
        if config.embedding_pooling == "cls":
            pooled = hidden[:, 0, :]
        else:
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        values = pooled.detach().float().cpu().numpy()
        if config.normalize_embeddings:
            norms = np.linalg.norm(values, axis=1, keepdims=True)
            values = values / np.maximum(norms, 1e-12)
        if output is None:
            output = np.lib.format.open_memmap(
                destination,
                mode="w+",
                dtype=np.float32,
                shape=(total_documents, values.shape[1]),
            )
        end = written + len(values)
        if end > total_documents:
            raise ValueError("embedding input exceeded declared document count")
        output[written:end] = values
        row_ids.extend(pending_ids)
        written = end
        pending_ids.clear()
        pending_text.clear()
        if progress:
            progress(written, total_documents)

    for row_id, text in documents:
        pending_ids.append(str(row_id))
        pending_text.append(str(text))
        if len(pending_text) >= config.embedding_batch_size:
            flush()
    flush()
    if written != total_documents or output is None:
        raise ValueError(
            f"expected {total_documents:,} documents for D2 embeddings, got {written:,}"
        )
    output.flush()
    return output, tuple(row_ids)


def _resolve_device(requested: str, torch: object) -> str:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("D2 embedding_device=cuda but CUDA is unavailable")
        return "cuda"
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"
