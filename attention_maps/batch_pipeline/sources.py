"""Bounded readers for local or downloaded corpus files."""

from __future__ import annotations

import csv
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


SUPPORTED_SUFFIXES = {".parquet", ".jsonl", ".ndjson", ".json", ".csv", ".txt"}


@dataclass(frozen=True)
class SourceRecord:
    row_id: str
    record: Mapping[str, Any]
    source_file: str
    source_row: int


def iter_record_batches(
    source: Path,
    *,
    batch_size: int,
    extraction_dir: Path,
) -> Iterator[list[SourceRecord]]:
    """Read supported corpus formats without retaining the complete corpus."""

    files = _discover_files(source, extraction_dir)
    if not files:
        raise ValueError(f"No supported dataset files found under {source}")
    for path, display_name in files:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            yield from _parquet_batches(path, display_name, batch_size)
        elif suffix in {".jsonl", ".ndjson"}:
            yield from _jsonl_batches(path, display_name, batch_size)
        elif suffix == ".json":
            yield from _json_batches(path, display_name, batch_size)
        elif suffix == ".csv":
            yield from _csv_batches(path, display_name, batch_size)
        else:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            yield [SourceRecord(f"{display_name}:0", {"text": text}, display_name, 0)]


def _discover_files(source: Path, extraction_dir: Path) -> list[tuple[Path, str]]:
    source = source.expanduser().resolve()
    if not source.exists():
        raise ValueError(f"Input path does not exist: {source}")
    candidates = [source] if source.is_file() else sorted(source.rglob("*"))
    files: list[tuple[Path, str]] = []
    extraction_dir.mkdir(parents=True, exist_ok=True)
    for path in candidates:
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.name if source.is_file() else str(path.relative_to(source))
        if path.suffix.lower() == ".zip":
            target = extraction_dir / f"{path.stem}-{len(files):04d}"
            _safe_extract_zip(path, target)
            for extracted in sorted(target.rglob("*")):
                if extracted.is_file() and extracted.suffix.lower() in SUPPORTED_SUFFIXES:
                    files.append((extracted, f"{relative}!{extracted.relative_to(target)}"))
        elif path.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append((path, relative))
    return files


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    marker = destination / ".extracted"
    if marker.is_file():
        return
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            target = (destination / info.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"Unsafe ZIP member path: {info.filename!r}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source_stream, target.open("wb") as target_stream:
                while chunk := source_stream.read(8 * 1024**2):
                    target_stream.write(chunk)
    marker.touch()


def _parquet_batches(
    path: Path, display_name: str, batch_size: int
) -> Iterator[list[SourceRecord]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("Parquet input requires pyarrow") from error
    row_offset = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size):
        values = batch.to_pylist()
        yield [
            SourceRecord(
                f"{display_name}:{row_offset + index}",
                record,
                display_name,
                row_offset + index,
            )
            for index, record in enumerate(values)
        ]
        row_offset += len(values)


def _jsonl_batches(
    path: Path, display_name: str, batch_size: int
) -> Iterator[list[SourceRecord]]:
    pending: list[SourceRecord] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for index, line in enumerate(stream):
            if not line.strip():
                continue
            value = json.loads(line)
            record = value if isinstance(value, Mapping) else {"text": value}
            pending.append(SourceRecord(f"{display_name}:{index}", record, display_name, index))
            if len(pending) >= batch_size:
                yield pending
                pending = []
    if pending:
        yield pending


def _json_batches(
    path: Path, display_name: str, batch_size: int
) -> Iterator[list[SourceRecord]]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(value, Mapping):
        for key in ("records", "data", "documents", "items"):
            if isinstance(value.get(key), Sequence):
                value = value[key]
                break
        else:
            value = [value]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"JSON dataset must contain an array of records: {path}")
    for start in range(0, len(value), batch_size):
        yield [
            SourceRecord(
                f"{display_name}:{index}",
                row if isinstance(row, Mapping) else {"text": row},
                display_name,
                index,
            )
            for index, row in enumerate(value[start : start + batch_size], start=start)
        ]


def _csv_batches(
    path: Path, display_name: str, batch_size: int
) -> Iterator[list[SourceRecord]]:
    pending: list[SourceRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for index, record in enumerate(csv.DictReader(stream)):
            pending.append(SourceRecord(f"{display_name}:{index}", record, display_name, index))
            if len(pending) >= batch_size:
                yield pending
                pending = []
    if pending:
        yield pending
