"""CLI pipeline for converting the three raw sources to cleaned Parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

if __package__:
    from scripts.build_pretraining_dataset import (
        Candidate,
        iter_lyrics_candidates,
        iter_parquet_candidates,
    )
    from scripts.utils.nepali_text import (
        CleaningConfig,
        clean_text_with_result,
        devanagari_letter_stats,
    )
else:
    from build_pretraining_dataset import (
        Candidate,
        iter_lyrics_candidates,
        iter_parquet_candidates,
    )
    from utils.nepali_text import (
        CleaningConfig,
        clean_text_with_result,
        devanagari_letter_stats,
    )


DEFAULT_PDF_DIR = Path("data/raw/nepali_pdf_corpus/data")
DEFAULT_NEWS_DIR = Path("data/raw/nepali_news_corpus/data")
DEFAULT_LYRICS_DIR = Path("data/raw/nepali_music_lyrics/archive/Final Dataset")
DEFAULT_OUTPUT_ROOT = Path("data/cleaned")
SOURCE_DIRECTORY = {
    "nepali_pdf": "nepali_pdf_corpus",
    "nepali_news": "nepali_news_corpus",
    "nepali_lyrics": "nepali_music_lyrics",
}


@dataclass
class CleaningStageStats:
    scanned: int = 0
    written: int = 0
    input_characters: int = 0
    output_characters: int = 0
    rejected: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict:
        return {
            "scanned": self.scanned,
            "written": self.written,
            "rejected": dict(sorted(self.rejected.items())),
            "input_characters": self.input_characters,
            "output_characters": self.output_characters,
            "character_retention": self.output_characters / max(1, self.input_characters),
        }


class ShardedParquetWriter:
    def __init__(
        self,
        output_dir: Path,
        schema,
        *,
        rows_per_shard: int,
        buffer_size: int,
        compression: str,
    ) -> None:
        import pyarrow.parquet as pq

        self.output_dir = output_dir
        self.schema = schema
        self.rows_per_shard = rows_per_shard
        self.buffer_size = buffer_size
        self.compression = compression
        self._pq = pq
        self._writer = None
        self._buffer: list[dict] = []
        self._shard_index = 0
        self._rows_in_shard = 0
        self.total_rows = 0
        self.files: list[str] = []
        output_dir.mkdir(parents=True, exist_ok=True)

    def _open_writer(self) -> None:
        path = self.output_dir / f"part-{self._shard_index:05d}.parquet"
        self._writer = self._pq.ParquetWriter(
            path,
            self.schema,
            compression=self.compression,
            use_dictionary=True,
        )
        self.files.append(path.name)
        self._shard_index += 1
        self._rows_in_shard = 0

    def append(self, row: dict) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        import pyarrow as pa

        if self._writer is None:
            self._open_writer()
        self._writer.write_table(pa.Table.from_pylist(self._buffer, schema=self.schema))
        count = len(self._buffer)
        self._rows_in_shard += count
        self.total_rows += count
        self._buffer.clear()
        if self._rows_in_shard >= self.rows_per_shard:
            self._writer.close()
            self._writer = None

    def close(self) -> None:
        self.flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean the raw PDF, news, and music-lyrics sources into separate "
            "canonical Parquet datasets. This stage does not split or tokenize."
        )
    )
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--news-dir", type=Path, default=DEFAULT_NEWS_DIR)
    parser.add_argument("--lyrics-dir", type=Path, default=DEFAULT_LYRICS_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=sorted(SOURCE_DIRECTORY),
        default=list(SOURCE_DIRECTORY),
    )
    parser.add_argument("--mode", choices=["strict", "preserve"], default="strict")
    parser.add_argument("--normalization", choices=["NFC", "NFKC"], default="NFC")
    parser.add_argument("--min-devanagari-ratio", type=float, default=0.80)
    parser.add_argument("--lyrics-min-devanagari-ratio", type=float, default=0.10)
    parser.add_argument("--min-devanagari-letters", type=int, default=10)
    parser.add_argument("--min-characters", type=int, default=20)
    parser.add_argument("--lyrics-min-characters", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--write-buffer-size", type=int, default=2_048)
    parser.add_argument("--rows-per-shard", type=int, default=100_000)
    parser.add_argument("--compression", choices=["zstd", "snappy"], default="zstd")
    parser.add_argument("--log-every", type=int, default=100_000)
    return parser.parse_args(arguments)


def validate_args(args: argparse.Namespace) -> None:
    for name in ("min_devanagari_ratio", "lyrics_min_devanagari_ratio"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    for name in (
        "min_devanagari_letters",
        "min_characters",
        "lyrics_min_characters",
        "batch_size",
        "write_buffer_size",
        "rows_per_shard",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.log_every < 0:
        raise ValueError("--log-every must be non-negative")


def clean_candidate(candidate: Candidate, config: CleaningConfig) -> tuple[dict | None, str | None]:
    result = clean_text_with_result(candidate.text, config)
    if not result.accepted:
        return None, result.reason

    output_ratio, output_letters = devanagari_letter_stats(result.text)
    raw_digest = hashlib.sha256(candidate.text.encode("utf-8")).hexdigest()
    clean_digest = hashlib.sha256(result.text.encode("utf-8")).hexdigest()
    metadata = {
        **candidate.metadata,
        "cleaning": {
            "input_devanagari_ratio": round(result.devanagari_ratio, 6),
            "input_devanagari_letters": result.devanagari_letters,
            "output_devanagari_ratio": round(output_ratio, 6),
            "output_devanagari_letters": output_letters,
        },
    }
    return {
        "source": candidate.source,
        "source_id": candidate.source_id,
        "text": result.text,
        "language": candidate.language or "ne",
        "url": candidate.url,
        "raw_text_sha256": raw_digest,
        "text_sha256": clean_digest,
        "metadata_json": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
    }, None


def raw_sources(args: argparse.Namespace) -> dict[str, tuple[Path, Iterator[Candidate]]]:
    return {
        "nepali_pdf": (
            args.pdf_dir,
            iter_parquet_candidates(args.pdf_dir, "nepali_pdf", args.batch_size),
        ),
        "nepali_news": (
            args.news_dir,
            iter_parquet_candidates(args.news_dir, "nepali_news", args.batch_size),
        ),
        "nepali_lyrics": (args.lyrics_dir, iter_lyrics_candidates(args.lyrics_dir)),
    }


def source_download_manifest(input_dir: Path) -> dict | None:
    manifest_path = input_dir.parent / "download_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"path": str(manifest_path.resolve()), "unreadable": True}


def clean_sources(args: argparse.Namespace) -> dict[str, dict]:
    validate_args(args)
    import pyarrow as pa

    for source_name in args.sources:
        output_dir = args.output_root / SOURCE_DIRECTORY[source_name]
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"Cleaned output is not empty: {output_dir}. "
                "Choose another --output-root to preserve reproducibility."
            )

    schema = pa.schema(
        [
            ("source", pa.string()),
            ("source_id", pa.string()),
            ("text", pa.string()),
            ("language", pa.string()),
            ("url", pa.string()),
            ("raw_text_sha256", pa.string()),
            ("text_sha256", pa.string()),
            ("metadata_json", pa.string()),
        ]
    )
    configurations = {
        "nepali_pdf": CleaningConfig(
            args.min_devanagari_ratio,
            args.min_devanagari_letters,
            args.min_characters,
            args.mode,
            args.normalization,
        ),
        "nepali_news": CleaningConfig(
            args.min_devanagari_ratio,
            args.min_devanagari_letters,
            args.min_characters,
            args.mode,
            args.normalization,
        ),
        "nepali_lyrics": CleaningConfig(
            args.lyrics_min_devanagari_ratio,
            args.min_devanagari_letters,
            args.lyrics_min_characters,
            args.mode,
            args.normalization,
        ),
    }

    sources = raw_sources(args)
    manifests: dict[str, dict] = {}
    for source_name in args.sources:
        input_dir, candidates = sources[source_name]
        output_dir = args.output_root / SOURCE_DIRECTORY[source_name]
        writer = ShardedParquetWriter(
            output_dir / "data",
            schema,
            rows_per_shard=args.rows_per_shard,
            buffer_size=args.write_buffer_size,
            compression=args.compression,
        )
        stats = CleaningStageStats()
        config = configurations[source_name]
        print(f"Cleaning {source_name} ...")
        try:
            for candidate in candidates:
                stats.scanned += 1
                stats.input_characters += len(candidate.text) if isinstance(candidate.text, str) else 0
                row, reason = clean_candidate(candidate, config)
                if row is None:
                    stats.rejected[reason or "empty"] += 1
                else:
                    writer.append(row)
                    stats.written += 1
                    stats.output_characters += len(row["text"])
                if args.log_every and stats.scanned % args.log_every == 0:
                    print(f"  scanned={stats.scanned:,} written={stats.written:,}")
        finally:
            writer.close()

        manifest = {
            "schema_version": 1,
            "status": "complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": source_name,
            "input_path": str(input_dir.resolve()),
            "input_download_manifest": source_download_manifest(input_dir),
            "output_path": str(output_dir.resolve()),
            "cleaning": asdict(config),
            "operations": [
                "Unicode normalization",
                "HTML entity decoding and tag/script/style removal",
                "URL and email removal",
                "unsafe control-character removal",
                "Devanagari-dominance and minimum-content filtering",
                "Latin/unsupported character stripping when mode=strict",
                "whitespace normalization",
            ],
            "canonical_columns": schema.names,
            "output_files": writer.files,
            "stats": stats.to_dict(),
        }
        (output_dir / "cleaning_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifests[source_name] = manifest
        print(f"Finished {source_name}: scanned={stats.scanned:,}, written={stats.written:,}")
    return manifests


def main() -> None:
    args = parse_args()
    manifests = clean_sources(args)
    print(f"Cleaned sources written under: {args.output_root.resolve()}")
    for source, manifest in manifests.items():
        stats = manifest["stats"]
        print(f"  {source:14s}: {stats['written']:,}/{stats['scanned']:,} documents")


if __name__ == "__main__":
    main()
