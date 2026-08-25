#!/usr/bin/env python3
"""Train or load a BPE and tokenize processed pretraining splits."""

from __future__ import annotations

import argparse
from pathlib import Path

from attention_maps.tokenization.pipeline import (
    load_tokenization_config,
    run_tokenization,
    with_path_overrides,
)


DEFAULT_CONFIG = Path("configs/tokenizer/huggingface_nepali_bpe.yaml")


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a tokenizer from Hugging Face or a local directory, or train "
            "a custom BPE, then encode all configured Parquet splits."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML configuration file (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Override input.dataset_dir from the YAML config",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override output.output_dir from the YAML config",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        help=(
            "Override tokenizer.source with a local tokenizer.json directory and "
            "skip tokenizer training or Hub loading"
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tokenizer trainer's progress display",
    )
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> None:
    args = parse_args(arguments)
    config = load_tokenization_config(args.config)
    config = with_path_overrides(config, args.input_dir, args.output_dir)
    manifest = run_tokenization(
        config,
        show_progress=not args.no_progress,
        existing_tokenizer_dir=args.tokenizer_dir,
    )

    print(f"Tokenizer and encoded splits: {config.output.output_dir}")
    print(f"Tokenizer mode               : {manifest['tokenizer']['origin']['mode']}")
    print(f"Vocabulary size             : {manifest['tokenizer']['vocab_size']:,}")
    for split, stats in manifest["split_stats"].items():
        print(
            f"  {split:10s}: {stats['documents']:,} documents, "
            f"{stats['tokens']:,} tokens, "
            f"UNK={stats['unknown_token_rate']:.6%}"
        )


if __name__ == "__main__":
    main()
