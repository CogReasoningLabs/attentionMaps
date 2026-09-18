#!/usr/bin/env python3
"""Export Hugging Face dataset sizes from a newline-delimited identifier list."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SIZE_ENDPOINT = "https://datasets-server.huggingface.co/size"


@dataclass(frozen=True)
class DatasetTarget:
    dataset_id: str
    config: str | None = None
    split: str | None = None


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read Hugging Face dataset IDs or URLs, query metadata without "
            "downloading datasets, and write spreadsheet-ready CSV."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Text file containing one dataset ID, URL, or ID|config|split per line",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("huggingface_dataset_sizes.csv"),
        help="Output CSV path (default: huggingface_dataset_sizes.csv)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Metadata request timeout in seconds (default: 60)",
    )
    return parser.parse_args(arguments)


def parse_target(value: str) -> DatasetTarget:
    """Parse an ID, Hub URL, or ID|config|split target."""

    text = value.strip()
    if not text:
        raise ValueError("dataset identifier cannot be empty")
    if "|" in text:
        parts = [part.strip() for part in text.split("|")]
        if len(parts) > 3 or not parts[0]:
            raise ValueError("use dataset_id|config|split")
        parts.extend([""] * (3 - len(parts)))
        target = DatasetTarget(parts[0], parts[1] or None, parts[2] or None)
        return _validate_target(target)
    if "://" not in text:
        return _validate_target(DatasetTarget(text))

    parsed = urllib.parse.urlparse(text)
    if parsed.scheme not in {"http", "https"} or parsed.netloc not in {
        "huggingface.co",
        "www.huggingface.co",
    }:
        raise ValueError("only huggingface.co dataset URLs are supported")
    parts = [urllib.parse.unquote(part) for part in parsed.path.split("/") if part]
    if len(parts) < 3 or parts[0] != "datasets":
        raise ValueError("Hugging Face URL must point to /datasets/owner/name")
    config = split = None
    if len(parts) >= 6 and parts[3] == "viewer":
        config, split = parts[4], parts[5]
    return _validate_target(DatasetTarget(f"{parts[1]}/{parts[2]}", config, split))


def _validate_target(target: DatasetTarget) -> DatasetTarget:
    parts = target.dataset_id.strip("/").split("/")
    if len(parts) != 2 or not all(parts) or any(
        character.isspace() for character in target.dataset_id
    ):
        raise ValueError("dataset ID must look like owner/dataset")
    if target.split and not target.config:
        raise ValueError("a split requires a configuration")
    return DatasetTarget("/".join(parts), target.config, target.split)


def read_targets(path: Path) -> list[tuple[str, DatasetTarget]]:
    """Read non-empty, non-comment targets while retaining their input text."""

    try:
        lines = path.expanduser().read_text(encoding="utf-8-sig").splitlines()
    except OSError as error:
        raise ValueError(f"could not read input list {path}: {error}") from error
    targets = []
    for line_number, line in enumerate(lines, start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            targets.append((text, parse_target(text)))
        except ValueError as error:
            raise ValueError(f"line {line_number}: {error}") from error
    if not targets:
        raise ValueError("input list contains no dataset identifiers")
    return targets


def query_size(
    target: DatasetTarget,
    *,
    token: str | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Query Dataset Viewer size metadata for one target."""

    url = f"{SIZE_ENDPOINT}?{urllib.parse.urlencode({'dataset': target.dataset_id})}"
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        raise ValueError(f"HTTP {error.code}: {error.reason}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"metadata request failed: {error}") from error

    size = payload.get("size") if isinstance(payload, dict) else None
    if not isinstance(size, dict):
        raise ValueError("Dataset Viewer returned no size metadata")
    if target.split:
        candidates = [
            row
            for row in size.get("splits", [])
            if isinstance(row, dict)
            and row.get("config") == target.config
            and row.get("split") == target.split
        ]
        scope = "split"
    elif target.config:
        candidates = [
            row
            for row in size.get("configs", [])
            if isinstance(row, dict) and row.get("config") == target.config
        ]
        scope = "configuration"
    else:
        dataset = size.get("dataset")
        candidates = [dataset] if isinstance(dataset, dict) else []
        scope = "dataset"
    if len(candidates) != 1:
        raise ValueError(
            f"no unambiguous {scope} size for "
            f"{target.config or '<all>'}/{target.split or '<all>'}"
        )

    metadata = candidates[0]
    original_bytes = metadata.get("num_bytes_original_files")
    parquet_bytes = metadata.get("num_bytes_parquet_files")
    if original_bytes is not None:
        file_bytes = int(original_bytes)
        basis = "original files"
    elif parquet_bytes is not None:
        file_bytes = int(parquet_bytes)
        basis = "Parquet files"
    else:
        file_bytes = None
        basis = "unavailable"
    memory_bytes = metadata.get("num_bytes_memory")
    return {
        "scope": scope,
        "num_rows": metadata.get("num_rows"),
        "file_size_bytes": file_bytes,
        "file_size": human_size(file_bytes),
        "file_size_basis": basis,
        "original_size_bytes": original_bytes,
        "parquet_size_bytes": parquet_bytes,
        "decoded_size_bytes": memory_bytes,
        "decoded_size": human_size(memory_bytes),
    }


def human_size(value: Any) -> str:
    """Format a byte count using decimal storage units."""

    if value is None:
        return "Unknown"
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(size) < 1_000 or unit == "PB":
            return f"{size:.2f} {unit}"
        size /= 1_000
    raise AssertionError("unreachable")


def export_sizes(
    targets: list[tuple[str, DatasetTarget]],
    output: Path,
    *,
    token: str | None,
    timeout: float,
) -> tuple[int, int]:
    """Query targets sequentially and write one CSV row per input line."""

    checked_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    cache: dict[DatasetTarget, dict[str, Any] | Exception] = {}
    errors = 0
    for index, (input_value, target) in enumerate(targets, start=1):
        print(f"[{index}/{len(targets)}] {target.dataset_id}", flush=True)
        if target not in cache:
            try:
                cache[target] = query_size(target, token=token, timeout=timeout)
            except Exception as error:
                cache[target] = error
        result = cache[target]
        base = {
            "input": input_value,
            "dataset_id": target.dataset_id,
            "config": target.config or "",
            "split": target.split or "",
            "checked_at_utc": checked_at,
        }
        if isinstance(result, Exception):
            errors += 1
            rows.append({**base, "status": "error", "error": str(result)})
        else:
            rows.append({**base, **result, "status": "ok", "error": ""})

    fields = (
        "input",
        "dataset_id",
        "config",
        "split",
        "scope",
        "num_rows",
        "file_size",
        "file_size_bytes",
        "file_size_basis",
        "original_size_bytes",
        "parquet_size_bytes",
        "decoded_size",
        "decoded_size_bytes",
        "status",
        "error",
        "checked_at_utc",
    )
    destination = output.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows), errors


def _load_project_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def main(arguments: Iterable[str] | None = None) -> int:
    args = parse_args(arguments)
    if args.timeout <= 0:
        print("error: --timeout must be positive", file=sys.stderr)
        return 2
    _load_project_env()
    try:
        targets = read_targets(args.input)
        count, errors = export_sizes(
            targets,
            args.output,
            token=os.getenv("HF_TOKEN") or None,
            timeout=args.timeout,
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"Wrote {count} row(s) to {args.output} ({errors} error(s)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
