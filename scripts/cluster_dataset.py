#!/usr/bin/env python3
"""Script-owned corpus embeddings, clustering, pair similarity, and B1–B3 sampling."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attention_maps.explorer.semantic_models import MODEL_PRESETS, DocumentEncoder
from attention_maps.explorer.semantic_instances import SCHEMAS
from attention_maps.explorer.semantic_artifacts import build_run, load_run, pair_similarity
from attention_maps.explorer.semantic_sampling import curate_run
from attention_maps.explorer.semantic_source import source_settings, resolve_source
from attention_maps.explorer.semantic_settings import load_embedding_settings, apply_cli_overrides, effective_run_settings


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def fraction(value):
    number = float(value)
    if not math.isfinite(number) or not 0 < number <= 1:
        raise argparse.ArgumentTypeError("must be in (0, 1]")
    return number


def parser(run_defaults=None):
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    commands.add_parser("models", help="List model presets without loading models")
    run = commands.add_parser("run", help="Embed the whole selected source and cluster every non-empty record")
    run.add_argument("--settings", type=Path, help="Central embedding YAML/JSON config; CLI flags override its values")
    run.add_argument("--source-settings", type=Path, help="Reuse dataset/text selection from inspection YAML/JSON")
    source = run.add_mutually_exclusive_group()
    source.add_argument("--local", type=Path)
    source.add_argument("--dataset")
    run.add_argument("--provider", choices=("huggingface", "kaggle"), help="Required for remote datasets, here or in the settings file")
    run.add_argument("--dataset-file")
    run.add_argument("--config")
    run.add_argument("--split")
    run.add_argument("--revision")
    run.add_argument("--shard", dest="shards", action="append")
    run.add_argument("--text-column", dest="text_columns", action="append")
    run.add_argument("--language", dest="languages", action="append")
    run.add_argument("--training-schema", choices=("auto", *SCHEMAS), default="auto", help="Canonical atomic data-instance schema")
    run.add_argument("--field-map", dest="field_mapping", action="append", metavar="CANONICAL=SOURCE.PATH", help="Map schema fields to source columns; repeat as needed")
    run.add_argument("--field-parser", dest="field_parsers", action="append", metavar="FIELD=string|join_strings", help="Parse mapped text fields; repeat as needed")
    run.add_argument("--task-name", help="Task name for supervised sources without a task column")
    run.add_argument("--text-record-unit", choices=("line", "blank_line"), default="line", help="TXT instance boundary; never inferred from titles or sentences")
    run.add_argument("--model", choices=tuple(MODEL_PRESETS), default="nepali-bert")
    run.add_argument("--model-id", help="Override model repository/local path within the chosen backend")
    run.add_argument("--model-revision", default="main")
    run.add_argument("--embedding-task", choices=("clustering", "similarity"), default="clustering")
    run.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    run.add_argument("--batch-size", type=positive, default=16)
    run.add_argument("--max-length", type=positive, help="Chunk token limit, never a document truncation limit")
    run.add_argument("--max-records", type=positive, help="Explicit smoke-test prefix; omit for whole selected dataset")
    run.add_argument("--clusters", type=positive, default=10)
    run.add_argument("--cluster-batch-size", type=positive, default=1024)
    run.add_argument("--cluster-epochs", type=positive, default=3)
    run.add_argument("--wordcloud-font", type=Path, help="Font supporting the corpus scripts; Noto Devanagari is auto-detected")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--output-dir", type=Path, required=not bool((run_defaults or {}).get("output_dir")))
    run.set_defaults(**(run_defaults or {}))
    recluster = commands.add_parser("recluster", help="Save one new cluster count and word clouds using existing embeddings; no model loading")
    recluster.add_argument("--run", type=Path, required=True, help="Completed embedding run")
    recluster.add_argument("--clusters", type=positive, required=True)
    recluster.add_argument("--cluster-batch-size", type=positive, default=1024)
    recluster.add_argument("--cluster-epochs", type=positive, default=3)
    recluster.add_argument("--seed", type=int, default=42)
    recluster.add_argument("--name", help="Saved result name within RUN/clusterings (default: kN-seedS)")
    recluster.add_argument("--wordcloud-font", type=Path)
    clouds = commands.add_parser("wordclouds", help="Add missing word clouds to existing cluster assignments; no embedding or clustering")
    clouds.add_argument("--run", type=Path, required=True, help="Completed embedding run")
    clouds.add_argument("--clustering", help="Saved result name, e.g. k20-seed42; omit for the original clustering")
    clouds.add_argument("--wordcloud-font", type=Path)
    clouds.add_argument("--refresh", action="store_true", help="Regenerate saved clouds with current stopwords, keeping embeddings and cluster assignments")
    pair = commands.add_parser("pair", help="Compare any two saved record embeddings; no model inference")
    pair.add_argument("--run", type=Path, required=True)
    pair.add_argument("--row-a", type=int, required=True, help="Zero-based embedded record ID")
    pair.add_argument("--row-b", type=int, required=True)
    compare = commands.add_parser("compare", help="Score the same pair across models for the identical saved corpus")
    compare.add_argument("--run", type=Path, action="append", required=True)
    compare.add_argument("--row-a", type=int, required=True)
    compare.add_argument("--row-b", type=int, required=True)
    sample = commands.add_parser("sample", help="B1 random, B2 density/IPS, or B3 within-cluster SemDeDup")
    sample.add_argument("--run", type=Path, required=True)
    sample.add_argument("--method", choices=("random", "density", "semdedup"), required=True)
    sample.add_argument("--sample-fraction", type=fraction, default=0.2)
    sample.add_argument("--sampling-runs", type=positive, default=5)
    sample.add_argument("--seed", type=int, default=42)
    sample.add_argument("--density-rows", type=positive, default=64)
    sample.add_argument("--density-bins", type=positive, default=2048)
    sample.add_argument("--density-width", type=float, default=0.5)
    sample.add_argument("--similarity-threshold", type=float, default=0.95)
    sample.add_argument("--block-size", type=positive, default=512)
    sample.add_argument("--output-dir", type=Path, required=True)
    return cli


def parse_arguments(arguments=None):
    raw = list(sys.argv[1:] if arguments is None else arguments)
    preliminary = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    preliminary.add_argument("--settings", type=Path)
    selected, _ = preliminary.parse_known_args(raw)
    defaults = load_embedding_settings(selected.settings) if selected.settings else {}
    args = parser(apply_cli_overrides(defaults, raw)).parse_args(raw)
    if args.command == "run" and isinstance(args.field_mapping, list):
        mapping = {}
        for item in args.field_mapping:
            name, separator, source = item.partition("=")
            if not separator or not name.strip() or not source.strip():
                raise ValueError("--field-map requires CANONICAL=SOURCE.PATH")
            if name.strip() in mapping:
                raise ValueError(f"Repeated --field-map for {name.strip()}")
            mapping[name.strip()] = source.strip()
        args.field_mapping = mapping
    if args.command == "run" and isinstance(args.field_parsers, list):
        parsers = {}
        for item in args.field_parsers:
            name, separator, mode = item.partition("=")
            if not separator or not name.strip() or not mode.strip():
                raise ValueError("--field-parser requires FIELD=PARSER")
            if name.strip() in parsers:
                raise ValueError(f"Repeated --field-parser for {name.strip()}")
            parsers[name.strip()] = mode.strip()
        args.field_parsers = parsers
    return args


def main(arguments=None):
    try:
        args = parse_arguments(arguments)
        if args.command == "models":
            result = MODEL_PRESETS
        elif args.command == "recluster":
            from attention_maps.explorer.semantic_variants import recluster_run
            output, report = recluster_run(args)
            result = {"output_dir": str(output), "clustering": report["clustering"], "wordclouds": report["wordclouds"]}
        elif args.command == "wordclouds":
            from attention_maps.explorer.semantic_variants import backfill_wordclouds
            output, summary = backfill_wordclouds(args)
            result = {"output_dir": str(output), "wordclouds": summary}
        elif args.command == "pair":
            result = pair_similarity(args.run, args.row_a, args.row_b)
        elif args.command == "compare":
            reports = [load_run(path)[1] for path in args.run]
            if len({r["corpus_fingerprint"] for r in reports}) != 1:
                raise ValueError("Model comparison requires identical corpus fingerprints (records, ordering, and selected text)")
            result = [pair_similarity(path, args.row_a, args.row_b) for path in args.run]
        elif args.command == "sample":
            if args.seed < 0:
                raise ValueError("seed must be non-negative")
            output, report = curate_run(args)
            result = {"output_dir": str(output), "method": report["method"], "sampling": report["language_status"]["sampling"]}
        else:
            if args.seed < 0:
                raise ValueError("seed must be non-negative")
            if args.output_dir.expanduser().exists():
                raise ValueError("Output directory already exists; choose a new run name")
            try:
                from dotenv import load_dotenv
                load_dotenv(Path(__file__).resolve().parents[1] / ".env")
            except ImportError:
                pass
            token = os.getenv("HF_TOKEN") or os.getenv("HF_token")
            settings = source_settings(args)
            print(f"Datasource: {settings['provider']} · {settings['dataset'] or settings['local']}", file=sys.stderr)
            args.effective_settings = effective_run_settings(args, settings)
            inventory, fields = resolve_source(settings, token, validate_text_fields=False)
            from attention_maps.explorer.semantic_wordclouds import find_font
            find_font(args.wordcloud_font)  # Validate explicit font paths before expensive inference.
            encoder = DocumentEncoder(args, token)
            output, report = build_run(inventory, fields, encoder, args, token)
            result = {"output_dir": str(output), "embedded_records": report["embedded_records"],
                      "scope": report["scope"], "instance_definition": report["instance_definition"],
                      "model": report["model"], "clustering": report["clustering"]}
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
