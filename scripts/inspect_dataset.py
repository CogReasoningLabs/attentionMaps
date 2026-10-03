#!/usr/bin/env python3
"""Inspect selected dataset files; optionally filter whole records by script."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attention_maps.explorer.huggingface import (
    discover_huggingface_dataset, inspect_huggingface_configuration,
    inspect_huggingface_selection,
)
from attention_maps.explorer.kaggle_inspection import (
    discover_kaggle_dataset, kaggle_selection, stage_kaggle_selection, inspect_kaggle_selection,
)
from attention_maps.explorer.file_formats import inspect_format_metadata
from attention_maps.explorer.inspection import file_signatures, inspect_dataset
from attention_maps.explorer.inspection_runs import (
    build_inspection_report, load_inspection_report,
)
from attention_maps.explorer.inspection_history import DEFAULT_HISTORY_SHEET, append_history, read_history
from attention_maps.explorer.sampling_vote import SAMPLING_STRATEGIES
from attention_maps.explorer.row_selection import parse_row_filters
from attention_maps.explorer.classification_thresholds import DEFAULT_THRESHOLDS
from attention_maps.explorer.language_detection import DETECTION_DEFAULTS
from attention_maps.datasets.schemas import STANDARD_DATASET_SCHEMAS
from attention_maps.explorer.inspection_settings import (
    SETTINGS_KEYS, load_inspection_settings, validate_inspection_settings, validate_output_paths,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, help="YAML/JSON settings; CLI flags override its values")
    parser.add_argument("--provider", choices=("huggingface", "kaggle"), default="huggingface",
                        help="Provider for --dataset (default: huggingface)")
    parser.add_argument("--dataset-file", help="Exact Kaggle file path from --list")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset", help="Hugging Face ID or Kaggle owner/dataset[/versions/N]")
    source.add_argument("--local", type=Path, help="Local supported file or Parquet directory")
    parser.add_argument("--revision", help="Commit, tag, or branch; resolved to a commit")
    parser.add_argument("--config", help="Hugging Face dataset configuration")
    parser.add_argument("--split", help="Dataset split")
    parser.add_argument("--shard", dest="shards", action="append", help="Shard name/path from --list; repeat to combine")
    parser.add_argument("--list", action="store_true", help="List Hub configurations/splits/shards or Kaggle files without downloading data")
    parser.add_argument("--formats-only", action=argparse.BooleanOptionalAction, default=False,
                        help="Report file formats without reading records; local directories include all files")
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--sample-fraction", type=float, help="Fraction per random run (default: 0.2)")
    sampling.add_argument("--sample-size", type=int, help="Legacy single sample of 0..1000 rows")
    parser.add_argument("--sampling-runs", type=int, help="Independent voting runs (default: 5; legacy sample-size: 1)")
    parser.add_argument("--sampling-method", choices=tuple(SAMPLING_STRATEGIES), default="random")
    parser.add_argument("--concurrency", type=int, default=1, help="Worker processes for selected-record analysis (default: 1)")
    parser.add_argument("--batch-size", type=int, default=1024, help="Source row batches and selected records per worker task (default: 1024)")
    parser.add_argument("--text-column", dest="text_columns", action="append", help="Field to inspect/filter; repeat for multiple fields")
    parser.add_argument("--training-schema", choices=tuple(schema.key for schema in STANDARD_DATASET_SCHEMAS),
                        help="Atomic instance category; set this for each dataset before analysis")
    parser.add_argument("--invalid-instance-policy", choices=("error", "skip"), default="error",
                        help="Fail on malformed instances (default), or count and exclude them before sampling")
    parser.add_argument("--text-record-unit", choices=("line", "blank_line"), default="line",
                        help="TXT document boundary (default: one non-empty line)")
    parser.add_argument("--field-map", action="append", help="Map canonical field to source path: FIELD=PATH; repeat as needed")
    parser.add_argument("--field-parser", action="append", help="Parser for a mapped text field: FIELD=string|join_strings; repeat as needed")
    parser.add_argument("--task-name", help="Task for supervised examples without a task column")
    parser.add_argument("--language", dest="languages", action="append", help="Deprecated: retained in settings only; manual declarations are not language evidence")
    parser.set_defaults(row_filters=None, field_mapping=None, field_parsers=None)
    parser.add_argument("--row-filter", action="append", help="Select matching records BEFORE sampling: COLUMN=VALUE; repeat for alternatives (OR), different columns combine with AND")
    parser.add_argument("--no-row-filters", action="store_true", help="Clear YAML row filters for this run")
    for name, default in DEFAULT_THRESHOLDS.items():
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=default,
                            help=f"Classification/voting ratio (default: {default}; see dataset YAML)")
    for name, default in DETECTION_DEFAULTS.items():
        kind = Path if name.endswith("_model") else type(default)
        parser.add_argument("--" + name.replace("_", "-"), type=kind, default=default,
                            help="Text language fallback setting; see dataset YAML")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-devanagari-ratio", type=float, help="Keep whole records whose Devanagari letter/mark share meets this threshold")
    parser.add_argument("--max-records", type=int, help="Limit filtering to the first N selected records; default scans all")
    parser.add_argument("--filtered-output", type=Path, help="Destination JSONL for qualifying whole records")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False,
                        help="Replace an existing filtered output (source files are always protected)")
    parser.add_argument("--output", type=Path, help="Single CSV report path (defaults to the shared inspection history)")
    parser.add_argument("--history-sheet", type=Path, default=DEFAULT_HISTORY_SHEET, help="Shared CSV sheet; every completed inspection appends a run")
    parser.add_argument("--history", dest="record_history", action=argparse.BooleanOptionalAction, default=True,
                        help="Record completed inspections in the single CSV (default: enabled)")
    return parser


def _arguments(arguments: list[str] | None) -> tuple[argparse.Namespace, dict]:
    parser = _parser()
    raw = list(sys.argv[1:] if arguments is None else arguments)
    preliminary, _ = parser.parse_known_args(raw)
    defaults = load_inspection_settings(preliminary.settings) if preliminary.settings else {}
    # Switching providers clears provider-specific defaults; explicit CLI flags
    # are still validated, so inapplicable flags are never silently ignored.
    if any(item == "--provider" or item.startswith("--provider=") for item in raw):
        if preliminary.provider != defaults.get("provider", "huggingface"):
            for key in ("config", "split", "revision", "shards", "dataset_file"):
                defaults[key] = None
    if "--local" in raw or any(item.startswith("--local=") for item in raw):
        for key in ("config", "split", "revision", "shards", "dataset_file"):
            defaults[key] = None
    if any(item == "--sample-fraction" or item.startswith("--sample-fraction=") for item in raw):
        defaults["sample_size"] = None
    if any(item == "--sample-size" or item.startswith("--sample-size=") for item in raw):
        defaults["sample_fraction"] = None
        defaults["sampling_runs"] = None
    # Explicit CLI source and repeated lists replace their settings-file values.
    for flag, other in (("--local", "dataset"), ("--dataset", "local")):
        if any(item == flag or item.startswith(flag + "=") for item in raw):
            defaults[other] = None
    for flag, key in (("--shard", "shards"), ("--text-column", "text_columns"), ("--language", "languages"), ("--field-map", "field_mapping"), ("--field-parser", "field_parsers")):
        if any(item == flag or item.startswith(flag + "=") for item in raw):
            defaults[key] = None
    if any(item == "--history-sheet" or item.startswith("--history-sheet=") for item in raw):
        defaults["output"] = None  # Explicit alias replaces a YAML output path.
    parser.set_defaults(**defaults)
    args = parser.parse_args(raw)
    if args.field_map:
        mapping = {}
        for entry in args.field_map:
            name, separator, source = entry.partition("=")
            if not separator or not name.strip() or not source.strip():
                raise ValueError("--field-map requires FIELD=PATH")
            mapping[name.strip()] = source.strip()
        args.field_mapping = mapping
    if args.field_parser:
        parsers = {}
        for entry in args.field_parser:
            name, separator, mode = entry.partition("=")
            if not separator or not name.strip() or not mode.strip():
                raise ValueError("--field-parser requires FIELD=PARSER")
            parsers[name.strip()] = mode.strip()
        args.field_parsers = parsers
    settings = {key: getattr(args, key) for key in SETTINGS_KEYS}
    if args.no_row_filters and args.row_filter:
        raise ValueError("Choose --row-filter or --no-row-filters, not both")
    if args.no_row_filters:
        settings["row_filters"] = {}
    elif args.row_filter:
        settings["row_filters"] = parse_row_filters(args.row_filter)
    for key in ("local", "output", "filtered_output", "history_sheet", "language_detection_model"):
        if settings[key] is not None:
            settings[key] = str(Path(settings[key]).expanduser().resolve())
    if settings["sample_size"] is None and settings["sample_fraction"] is None:
        settings["sample_fraction"] = 0.2
    if settings["sampling_runs"] is None:
        settings["sampling_runs"] = 1 if settings["sample_size"] is not None else 5
    if settings["local"]:
        settings["provider"] = "local"
    if settings["output"]:
        settings["history_sheet"] = settings["output"]
    validate_inspection_settings(settings, listing=args.list)
    return args, settings


def main(arguments: list[str] | None = None) -> int:
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except ImportError:
        pass
    token = os.getenv("HF_TOKEN") or os.getenv("HF_token")
    try:
        args, settings = _arguments(arguments)
        if (settings["record_history"] or settings["output"]) and not args.list:
            read_history(settings["history_sheet"])  # Reject incompatible sheets before source processing.
        source_files = []
        if settings["local"]:
            path = Path(settings["local"])
            pattern = "*" if settings["formats_only"] else "*.parquet"
            excluded = set()
            if path.is_dir() and settings["formats_only"] and settings.get("output"):
                previous_report = Path(settings["output"])
                try:
                    load_inspection_report(previous_report)
                except (OSError, ValueError):
                    pass
                else:
                    excluded.add(previous_report)
            if args.settings:
                excluded.add(args.settings.expanduser().resolve())
            paths = tuple(sorted(
                item for item in path.rglob(pattern) if item.is_file() and item.resolve() not in excluded
            )) if path.is_dir() else (path,)
            if not paths:
                raise ValueError("No files found in local directory" if settings["formats_only"] else "No Parquet files found in local directory")
            source_files = [str(item) for item in paths]
        if args.settings:
            source_files.append(str(args.settings.expanduser().resolve()))
        validate_output_paths(settings, source_files)
        if settings["provider"] == "kaggle":
            report = _kaggle_report(settings, listing=args.list, source_files=source_files)
        elif settings["dataset"]:
            catalog = discover_huggingface_dataset(settings["dataset"], revision=settings["revision"], token=token)
            settings["revision"] = catalog.get("revision")
            if args.list and not settings["config"]:
                report = {key: value for key, value in catalog.items() if key != "file_sizes"}
            else:
                config = settings["config"]
                if config is None:
                    if len(catalog["configs"]) != 1:
                        raise ValueError("Choose --config; use --list to see all configurations.")
                    config = catalog["configs"][0]
                settings["config"] = config
                configuration = inspect_huggingface_configuration(catalog, config, token=token)
                if args.list:
                    report = configuration
                else:
                    split = settings["split"]
                    if split is None:
                        if len(configuration["splits"]) != 1:
                            raise ValueError("Choose --split; use --config NAME --list to see splits/shards.")
                        split = next(iter(configuration["splits"]))
                    settings["split"] = split
                    shards = None
                    if settings["shards"]:
                        available = configuration["splits"].get(split, {}).get("shards", [])
                        by_name = {item["name"]: item["path"] for item in available}
                        shards = [by_name.get(value, value) for value in settings["shards"]]
                    inventory = inspect_huggingface_selection(
                        configuration, split, shards=shards, token=token, metadata_only=settings["formats_only"],
                    )
                    settings["shards"] = inventory["dataset_shards"]
                    report = build_inspection_report(inventory, settings, token=token)
        else:
            signatures = file_signatures(paths)
            inventory = inspect_format_metadata(signatures) if settings["formats_only"] else inspect_dataset(signatures, show_progress=True)
            inventory["source_files"] = [str(item) for item in paths]
            inventory["source_signatures"] = [list(item) for item in signatures]
            report = build_inspection_report(inventory, settings, token=token)
            if file_signatures(paths) != signatures:
                raise ValueError("Source files changed during inspection; report was not published")
        if not args.list:
            report["started_at"] = started_at
            report["processing_seconds"] = round(time.perf_counter() - start, 3)
            report["created_at"] = datetime.now(timezone.utc).isoformat()
        if (settings["record_history"] or settings["output"]) and not args.list:
            report = append_history(report, settings["history_sheet"])
            print(f"Inspection history: {settings['history_sheet']} · run {report['history']['run_id']}", file=sys.stderr)
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    except (OSError, ValueError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def _kaggle_report(settings: dict, *, listing: bool, source_files: list[str]) -> dict:
    catalog = discover_kaggle_dataset(settings["dataset"])
    if listing:
        return catalog
    selected = kaggle_selection(catalog, settings["dataset_file"], metadata_only=settings["formats_only"])
    settings["dataset"] = catalog["dataset_handle"]
    settings["dataset_file"] = selected[0]["name"] if len(selected) == 1 else None
    cached_path = None
    if not settings["formats_only"]:
        cached_path = stage_kaggle_selection(catalog, selected)
        validate_output_paths(settings, [*source_files, str(cached_path)])
    inventory = inspect_kaggle_selection(catalog, selected, cached_path=cached_path, show_progress=True)
    report = build_inspection_report(inventory, settings)
    if cached_path is not None and inventory["source_signatures"] != [list(item) for item in file_signatures((cached_path,))]:
        raise ValueError("Kaggle cached file changed during inspection; report was not published")
    return report


if __name__ == "__main__":
    raise SystemExit(main())
