"""CPU worker functions and incremental Parquet/audit writers."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from attention_maps.eda.text import (
    devanagari_ratio,
    extract_source,
    extract_text,
    normalize_structured_text,
    tokenize,
)

from .contracts import CleaningConfig, InputConfig, SamplingConfig
from .sources import SourceRecord


@dataclass(frozen=True)
class PreparedRecord:
    doc_id: str
    text: str
    source: str
    source_file: str
    source_row: int
    text_sha256: str
    token_count: int
    character_count: int
    devanagari_ratio: float


@dataclass(frozen=True)
class RejectedRecord:
    row_id: str
    source_file: str
    source_row: int
    stage: str
    reason: str


@dataclass(frozen=True)
class PreparedBatch:
    batch_id: int
    records: tuple[PreparedRecord, ...]
    rejections: tuple[RejectedRecord, ...]


def selected_by_sampling(row_id: str, config: SamplingConfig) -> bool:
    if config.fraction >= 1:
        return True
    digest = hashlib.sha256(f"{config.seed}:{row_id}".encode("utf-8")).digest()
    score = int.from_bytes(digest[:8], "big") / 2**64
    return score < config.fraction


def prepare_batch(
    batch_id: int,
    rows: Sequence[SourceRecord],
    input_config: InputConfig,
    cleaning: CleaningConfig,
) -> PreparedBatch:
    """Normalize and validate a batch; safe to execute in another process."""

    records: list[PreparedRecord] = []
    rejected: list[RejectedRecord] = []
    for row in rows:
        text = extract_text(row.record, input_config.text_columns)
        normalized = normalize_structured_text(text) if text else ""
        if not normalized:
            rejected.append(_rejection(row, "cleaning", "missing_or_empty_text"))
            continue
        tokens = tokenize(normalized)
        if len(tokens) < cleaning.min_tokens:
            rejected.append(_rejection(row, "cleaning", "below_min_tokens"))
            continue
        ratio = devanagari_ratio(normalized)
        if ratio < cleaning.min_devanagari_ratio:
            rejected.append(_rejection(row, "cleaning", "below_devanagari_ratio"))
            continue
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        records.append(
            PreparedRecord(
                doc_id=hashlib.sha256(row.row_id.encode("utf-8")).hexdigest(),
                text=normalized,
                source=extract_source(row.record, input_config.source_columns),
                source_file=row.source_file,
                source_row=row.source_row,
                text_sha256=digest,
                token_count=len(tokens),
                character_count=len(normalized),
                devanagari_ratio=round(ratio, 8),
            )
        )
    return PreparedBatch(batch_id, tuple(records), tuple(rejected))


def ordered_prepared_batches(
    batches: Iterable[tuple[int, Sequence[SourceRecord]]],
    input_config: InputConfig,
    cleaning: CleaningConfig,
    *,
    workers: int,
    max_pending: int,
) -> Iterator[PreparedBatch]:
    """Bound the process-pool queue and always yield source order."""

    if workers == 1:
        for batch_id, rows in batches:
            yield prepare_batch(batch_id, rows, input_config, cleaning)
        return
    iterator = iter(batches)
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        pending: list[Future[PreparedBatch]] = []
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < max_pending:
                try:
                    batch_id, rows = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                pending.append(
                    executor.submit(
                        prepare_batch, batch_id, rows, input_config, cleaning
                    )
                )
            if pending:
                yield pending.pop(0).result()


class ParquetShardWriter:
    """Write records into bounded, immutable Parquet part files."""

    def __init__(self, output_dir: Path, shard_rows: int):
        self.output_dir = output_dir
        self.shard_rows = shard_rows
        self.buffer: list[dict] = []
        self.part = 0
        self.current_rows = 0
        self.writer = None
        self.current_path: Path | None = None
        self.rows_written = 0
        self.paths: list[Path] = []
        output_dir.mkdir(parents=True, exist_ok=True)

    def add(self, record: PreparedRecord) -> None:
        self.buffer.append(asdict(record))
        if len(self.buffer) >= min(1_024, self.shard_rows - self.current_rows):
            self._write_buffer()

    def _write_buffer(self) -> None:
        if not self.buffer:
            return
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError("Writing clean data requires pyarrow") from error
        if self.writer is None:
            path = self.output_dir / f"part-{self.part:05d}.parquet"
            self.current_path = path.with_suffix(".parquet.tmp")
            schema = pa.schema(
                [
                    ("doc_id", pa.string()),
                    ("text", pa.string()),
                    ("source", pa.string()),
                    ("source_file", pa.string()),
                    ("source_row", pa.int64()),
                    ("text_sha256", pa.string()),
                    ("token_count", pa.int64()),
                    ("character_count", pa.int64()),
                    ("devanagari_ratio", pa.float64()),
                ]
            )
            self.writer = pq.ParquetWriter(
                self.current_path, schema, compression="zstd"
            )
        self.writer.write_table(pa.Table.from_pylist(self.buffer, schema=self.writer.schema))
        self.rows_written += len(self.buffer)
        self.current_rows += len(self.buffer)
        self.buffer.clear()
        if self.current_rows >= self.shard_rows:
            self._close_shard()

    def _close_shard(self) -> None:
        if self.writer is None or self.current_path is None:
            return
        self.writer.close()
        final_path = self.current_path.with_suffix("")
        self.current_path.replace(final_path)
        self.paths.append(final_path)
        self.writer = None
        self.current_path = None
        self.current_rows = 0
        self.part += 1

    def close(self) -> tuple[Path, ...]:
        self._write_buffer()
        self._close_shard()
        return tuple(self.paths)


class AuditWriter:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("w", encoding="utf-8")
        self.events = 0

    def add(self, event: RejectedRecord) -> None:
        self.stream.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
        self.events += 1

    def close(self) -> None:
        self.stream.close()

    def __enter__(self) -> "AuditWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def iter_parquet_records(paths: Iterable[Path], batch_size: int) -> Iterator[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("Reading clean data requires pyarrow") from error
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size):
            yield from batch.to_pylist()


def prepared_from_mapping(value: dict) -> PreparedRecord:
    return PreparedRecord(**{key: value[key] for key in PreparedRecord.__dataclass_fields__})


def _rejection(row: SourceRecord, stage: str, reason: str) -> RejectedRecord:
    return RejectedRecord(row.row_id, row.source_file, row.source_row, stage, reason)
