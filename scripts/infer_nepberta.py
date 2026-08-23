from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MODEL_ID = "NepBERTa/NepBERTa"
DEFAULT_TEXT = "नेपाल एक {mask} देश हो।"
DEFAULT_CACHE_DIR = Path("data/cache/huggingface")


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run fill-mask inference with NepBERTa. Use exactly one [MASK] "
            "token or the model-independent {mask} placeholder."
        )
    )
    parser.add_argument(
        "text",
        nargs="?",
        default=DEFAULT_TEXT,
        help="Nepali text containing exactly one [MASK] or {mask}",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--revision",
        default="main",
        help="Hugging Face branch, tag, or commit (default: main)",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--target",
        action="append",
        dest="targets",
        help="Optionally score only this candidate token; may be repeated",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="cpu",
        help="TensorFlow inference device",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Model cache directory (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a table",
    )
    args = parser.parse_args(arguments)
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    return args


def prepare_masked_text(text: str, mask_token: str) -> str:
    """Resolve the portable placeholder and enforce one-mask MLM inference."""

    if not text.strip():
        raise ValueError("Input text must not be empty")
    if not mask_token:
        raise ValueError("The selected tokenizer does not define a mask token")
    prepared = text.replace("{mask}", mask_token)
    count = prepared.count(mask_token)
    if count != 1:
        raise ValueError(
            f"Expected exactly one mask token ({mask_token!r}); found {count}. "
            "Use the portable {mask} placeholder if unsure."
        )
    return prepared


def load_model(args: argparse.Namespace):
    """Load the repository's tokenizer and TensorFlow MLM checkpoint."""

    # This repository only publishes TensorFlow weights. Explicitly disable
    # Transformers' PyTorch backend before import: otherwise a mixed TF/PyTorch
    # environment may import torch._dynamo and Triton even though this script
    # never uses them. A broken native Triton install can then crash Python.
    os.environ["USE_TORCH"] = "0"
    os.environ["USE_TF"] = "1"
    os.environ["USE_FLAX"] = "0"
    # The optional hf-xet client can stall on this repository in restricted or
    # proxied environments. Standard Hub HTTP downloads are slower but robust.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    # Transformers uses the legacy Keras compatibility package for TF models.
    # These must also be set before importing TensorFlow or Transformers.
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    try:
        import tensorflow as tf
        from transformers import (
            AutoTokenizer,
            TFAutoModelForMaskedLM,
        )
        from transformers.utils import logging as transformers_logging
    except ImportError as exc:
        raise RuntimeError(
            "NepBERTa inference dependencies are missing. Install them with "
            "`.venv/bin/pip install -r requirements-nepberta.txt`."
        ) from exc

    transformers_logging.set_verbosity_error()
    load_kwargs: dict[str, Any] = {
        "revision": args.revision,
        "local_files_only": args.local_files_only,
        "cache_dir": str(args.cache_dir),
    }

    if args.device == "cpu":
        tf.config.set_visible_devices([], "GPU")
    else:
        has_gpu = bool(tf.config.list_physical_devices("GPU"))
        if args.device == "gpu" and not has_gpu:
            raise RuntimeError("--device gpu was requested, but TensorFlow found no GPU")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, **load_kwargs)
    model = TFAutoModelForMaskedLM.from_pretrained(args.model_id, **load_kwargs)
    return model, tokenizer, tf


def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, tf = load_model(args)
    text = prepare_masked_text(args.text, tokenizer.mask_token)

    token_count = len(tokenizer(text, add_special_tokens=True)["input_ids"])
    max_positions = getattr(model.config, "max_position_embeddings", None)
    if max_positions is not None and token_count > max_positions:
        raise ValueError(
            f"Input has {token_count} tokens, exceeding the model limit of "
            f"{max_positions}; shorten the text."
        )

    inputs = tokenizer(text, return_tensors="tf")
    mask_positions = tf.where(inputs["input_ids"][0] == tokenizer.mask_token_id)
    if int(tf.size(mask_positions).numpy()) != 1:
        raise ValueError("Tokenization did not produce exactly one mask token")
    mask_index = int(mask_positions[0, 0].numpy())

    logits = model(**inputs, training=False).logits[0, mask_index, :]
    probabilities = tf.nn.softmax(logits)

    if args.targets:
        candidate_ids = []
        for target in args.targets:
            target_ids = tokenizer.encode(target, add_special_tokens=False)
            if len(target_ids) != 1:
                raise ValueError(
                    f"Target {target!r} maps to {len(target_ids)} WordPiece tokens; "
                    "--target candidates must each map to exactly one token."
                )
            candidate_ids.append(target_ids[0])
        candidate_scores = tf.gather(probabilities, candidate_ids)
        order = tf.argsort(candidate_scores, direction="DESCENDING")[: args.top_k]
        prediction_ids = [candidate_ids[int(index)] for index in order.numpy()]
        prediction_scores = [
            float(candidate_scores[int(index)].numpy()) for index in order.numpy()
        ]
    else:
        count = min(args.top_k, int(model.config.vocab_size))
        top = tf.math.top_k(probabilities, k=count)
        prediction_ids = [int(token_id) for token_id in top.indices.numpy()]
        prediction_scores = [float(score) for score in top.values.numpy()]

    input_ids = inputs["input_ids"][0].numpy().tolist()
    predictions = []
    for token_id, score in zip(prediction_ids, prediction_scores):
        completed_ids = input_ids.copy()
        completed_ids[mask_index] = token_id
        predictions.append(
            {
                "score": score,
                "token": token_id,
                "token_str": tokenizer.decode([token_id]),
                "sequence": tokenizer.decode(
                    completed_ids,
                    skip_special_tokens=True,
                ),
            }
        )

    return {
        "model": args.model_id,
        "revision": args.revision,
        "task": "fill-mask",
        "input": text,
        "mask_token": tokenizer.mask_token,
        "predictions": predictions,
    }


def print_result(result: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"Input: {result['input']}")
    print("\nPredictions:")
    for rank, prediction in enumerate(result["predictions"], start=1):
        print(
            f"{rank:>2}. {prediction['token_str'].strip():<20} "
            f"p={prediction['score']:.6f}  {prediction['sequence']}"
        )


def main() -> None:
    args = parse_args()
    try:
        result = run_inference(args)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    print_result(result, as_json=args.json)


if __name__ == "__main__":
    main()
