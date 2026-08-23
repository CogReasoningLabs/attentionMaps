from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

if __package__:
    from scripts.utils.nepali_text import (
        devanagari_letter_stats,
    )
else:
    from utils.nepali_text import (
        devanagari_letter_stats,
    )


DEFAULT_PDF_DIR = Path("data/cleaned/nepali_pdf_corpus/data")
DEFAULT_NEWS_DIR = Path("data/cleaned/nepali_news_corpus/data")
DEFAULT_LYRICS_DIR = Path("data/cleaned/nepali_music_lyrics/data")
DEFAULT_OUTPUT_DIR = Path("data/processed/nepali_pretraining")
SOURCE_NAMES = ("nepali_pdf", "nepali_news", "nepali_lyrics")
SPLIT_NAMES = ("train", "validation", "test")
LYRICS_COLUMNS = {"SegmentFilePath", "SegmentLength", "Lyrics"}


@dataclass(frozen=True)
class Candidate:
    source: str
    source_id: str
    text: str
    language: str = "ne"
    url: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class SourceStats:
    scanned: int = 0
    selected_by_sampling: int = 0
    sampled_out: int = 0
    accepted_after_cleaning: int = 0
    exact_duplicates: int = 0
    written: int = 0
    rejected: Counter = field(default_factory=Counter)
    splits: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict:
        return {
            "scanned": self.scanned,
            "selected_by_sampling": self.selected_by_sampling,
            "sampled_out": self.sampled_out,
            "accepted_after_cleaning": self.accepted_after_cleaning,
            "exact_duplicates": self.exact_duplicates,
            "written": self.written,
            "rejected": dict(sorted(self.rejected.items())),
            "splits": {name: self.splits[name] for name in SPLIT_NAMES},
        }


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standardize already-cleaned PDF, news, and music-lyrics sources "
            "into deterministic train/validation/test Parquet files."
        )
    )
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--news-dir", type=Path, default=DEFAULT_NEWS_DIR)
    parser.add_argument("--lyrics-dir", type=Path, default=DEFAULT_LYRICS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=1.0,
        help="Global deterministic fraction applied to every source (0..1)",
    )
    parser.add_argument("--pdf-fraction", type=float, default=1.0)
    parser.add_argument("--news-fraction", type=float, default=1.0)
    parser.add_argument("--lyrics-fraction", type=float, default=1.0)

    parser.add_argument("--train-ratio", type=float, default=0.98)
    parser.add_argument("--validation-ratio", type=float, default=0.01)
    parser.add_argument("--test-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--deduplicate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove exact normalized-text duplicates across all selected sources",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--write-buffer-size", type=int, default=2_048)
    parser.add_argument(
        "--log-every",
        type=int,
        default=100_000,
        help="Report source scan progress every N records; use 0 to disable",
    )
    parser.add_argument("--compression", choices=["zstd", "snappy"], default="zstd")
    return parser.parse_args(arguments)


def validate_args(args: argparse.Namespace) -> None:
    fractions = {
        "sample_fraction": args.sample_fraction,
        "pdf_fraction": args.pdf_fraction,
        "news_fraction": args.news_fraction,
        "lyrics_fraction": args.lyrics_fraction,
    }
    for name, value in fractions.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")

    ratios = [args.train_ratio, args.validation_ratio, args.test_ratio]
    if any(ratio < 0.0 for ratio in ratios):
        raise ValueError("Split ratios must be non-negative")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("Train, validation, and test ratios must sum to 1")
    if args.train_ratio <= 0:
        raise ValueError("--train-ratio must be positive")
    if args.batch_size <= 0 or args.write_buffer_size <= 0:
        raise ValueError("Batch and write-buffer sizes must be positive")
    if args.log_every < 0:
        raise ValueError("--log-every must be non-negative")


def stable_unit_interval(seed: int, namespace: str, key: str) -> float:
    digest = hashlib.blake2b(
        f"{seed}:{namespace}:{key}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big") / 2**64


def include_sample(source: str, source_id: str, fraction: float, seed: int) -> bool:
    if fraction >= 1.0:
        return True
    if fraction <= 0.0:
        return False
    return stable_unit_interval(seed, "sample", f"{source}:{source_id}") < fraction


def assign_split(
    doc_id: str,
    *,
    train_ratio: float,
    validation_ratio: float,
    seed: int,
) -> str:
    value = stable_unit_interval(seed, "split", doc_id)
    if value < train_ratio:
        return "train"
    if value < train_ratio + validation_ratio:
        return "validation"
    return "test"


def parquet_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.rglob("*.parquet")) if input_dir.is_dir() else []
    if not files:
        raise FileNotFoundError(f"No Parquet files found under {input_dir}")
    return files


def iter_parquet_candidates(
    input_dir: Path,
    source: str,
    batch_size: int,
) -> Iterator[Candidate]:
    import pyarrow.dataset as pads

    dataset = pads.dataset(
        [str(path) for path in parquet_files(input_dir)],
        format="parquet",
    )
    required = {"id", "text", "url", "language"}
    missing = required - set(dataset.schema.names)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")

    scanner = dataset.scanner(
        columns=["id", "text", "url", "language"],
        batch_size=batch_size,
    )
    for batch in scanner.to_batches():
        columns = {
            name: batch.column(name).to_pylist()
            for name in ("id", "text", "url", "language")
        }
        for index in range(batch.num_rows):
            yield Candidate(
                source=source,
                source_id=str(columns["id"][index] or ""),
                text=columns["text"][index],
                language=str(columns["language"][index] or "ne"),
                url=columns["url"][index],
            )


def iter_cleaned_candidates(
    input_dir: Path,
    expected_source: str,
    batch_size: int,
) -> Iterator[Candidate]:
    """Stream canonical records emitted by clean_pretraining_sources.py."""

    import pyarrow.dataset as pads

    dataset = pads.dataset(
        [str(path) for path in parquet_files(input_dir)],
        format="parquet",
    )
    columns = [
        "source",
        "source_id",
        "text",
        "language",
        "url",
        "text_sha256",
        "metadata_json",
    ]
    missing = set(columns) - set(dataset.schema.names)
    if missing:
        raise ValueError(
            f"Cleaned {expected_source} input is missing columns: {sorted(missing)}"
        )

    for batch in dataset.scanner(columns=columns, batch_size=batch_size).to_batches():
        values = {name: batch.column(name).to_pylist() for name in columns}
        for index in range(batch.num_rows):
            source = str(values["source"][index] or "")
            if source != expected_source:
                raise ValueError(
                    f"Expected source {expected_source!r}, found {source!r} in {input_dir}"
                )
            raw_metadata = values["metadata_json"][index]
            try:
                metadata = json.loads(raw_metadata) if raw_metadata else {}
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid metadata_json for {source}:{values['source_id'][index]}"
                ) from exc
            if not isinstance(metadata, dict):
                raise ValueError("metadata_json must decode to an object")
            metadata["_cleaned_text_sha256"] = values["text_sha256"][index]
            yield Candidate(
                source=source,
                source_id=str(values["source_id"][index] or ""),
                text=values["text"][index],
                language=str(values["language"][index] or "ne"),
                url=values["url"][index],
                metadata=metadata,
            )


def natural_segment_key(segment_path: str) -> tuple:
    name = Path(segment_path).stem
    match = re.search(r"(\d+)$", name)
    return (int(match.group(1)) if match else 10**12, name)


def iter_lyrics_candidates(lyrics_dir: Path) -> Iterator[Candidate]:
    metadata_files = sorted(lyrics_dir.glob("*/metadata.csv"))
    if not metadata_files:
        raise FileNotFoundError(
            f"No artist-level metadata.csv files found under {lyrics_dir}"
        )

    for metadata_path in metadata_files:
        artist = metadata_path.parent.name
        songs: dict[str, list[dict]] = defaultdict(list)
        with metadata_path.open(encoding="utf-8-sig", newline="") as source_file:
            reader = csv.DictReader(source_file)
            missing = LYRICS_COLUMNS - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"{metadata_path} is missing columns: {sorted(missing)}"
                )
            for row in reader:
                segment_path = str(row.get("SegmentFilePath") or "").replace("\\", "/")
                song = Path(segment_path).parent.name.strip()
                if not segment_path or not song:
                    continue
                try:
                    duration = float(row.get("SegmentLength") or 0.0)
                except ValueError:
                    duration = 0.0
                audio_path = metadata_path.parent / segment_path
                songs[song].append(
                    {
                        "segment_path": segment_path,
                        "duration": duration,
                        "lyrics": str(row.get("Lyrics") or ""),
                        "audio_exists": audio_path.is_file(),
                    }
                )

        for song, segments in sorted(songs.items()):
            segments.sort(key=lambda row: natural_segment_key(row["segment_path"]))
            text = "\n".join(row["lyrics"].strip() for row in segments if row["lyrics"].strip())
            source_id = f"{artist}/{song}"
            yield Candidate(
                source="nepali_lyrics",
                source_id=source_id,
                text=text,
                language="ne",
                metadata={
                    "artist": artist,
                    "song": song,
                    "segment_count": len(segments),
                    "duration_seconds": round(
                        sum(row["duration"] for row in segments), 3
                    ),
                    "missing_audio_segments": sum(
                        not row["audio_exists"] for row in segments
                    ),
                },
            )


def make_doc_id(source: str, source_id: str) -> str:
    return hashlib.sha256(f"{source}\0{source_id}".encode("utf-8")).hexdigest()


def serialize_path(path: Path) -> str:
    return str(path.resolve())


def source_manifest(data_dir: Path) -> dict | None:
    for name in ("cleaning_manifest.json", "download_manifest.json"):
        manifest_path = data_dir.parent / name
        if not manifest_path.is_file():
            continue
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"manifest_path": serialize_path(manifest_path), "unreadable": True}
    return None


def build_dataset(args: argparse.Namespace) -> dict:
    validate_args(args)
    parquet_files(args.pdf_dir)
    parquet_files(args.news_dir)
    parquet_files(args.lyrics_dir)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}. "
            "Choose a new directory to preserve reproducibility."
        )

    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            ("doc_id", pa.string()),
            ("text", pa.string()),
            ("source", pa.string()),
            ("source_id", pa.string()),
            ("language", pa.string()),
            ("url", pa.string()),
            ("text_sha256", pa.string()),
            ("metadata_json", pa.string()),
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    writers = {
        split: pq.ParquetWriter(
            args.output_dir / f"{split}.parquet",
            schema,
            compression=args.compression,
            use_dictionary=True,
        )
        for split in SPLIT_NAMES
    }
    buffers: dict[str, list[dict]] = {split: [] for split in SPLIT_NAMES}
    stats = {source: SourceStats() for source in SOURCE_NAMES}
    seen_text_hashes: set[bytes] = set()
    seen_doc_ids: set[str] = set()
    global_splits: Counter = Counter()

    source_fractions = {
        "nepali_pdf": args.sample_fraction * args.pdf_fraction,
        "nepali_news": args.sample_fraction * args.news_fraction,
        "nepali_lyrics": args.sample_fraction * args.lyrics_fraction,
    }

    def flush(split: str) -> None:
        if not buffers[split]:
            return
        writers[split].write_table(
            pa.Table.from_pylist(buffers[split], schema=schema)
        )
        buffers[split].clear()

    def process(candidate: Candidate) -> None:
        source_stats = stats[candidate.source]
        source_stats.scanned += 1
        if not candidate.source_id.strip():
            source_stats.rejected["missing_source_id"] += 1
            return
        if not include_sample(
            candidate.source,
            candidate.source_id,
            source_fractions[candidate.source],
            args.seed,
        ):
            source_stats.sampled_out += 1
            return
        source_stats.selected_by_sampling += 1

        cleaned_text = candidate.text
        if not isinstance(cleaned_text, str) or not cleaned_text.strip():
            source_stats.rejected["empty_cleaned_text"] += 1
            return
        ratio, devanagari_letters = devanagari_letter_stats(cleaned_text)
        source_stats.accepted_after_cleaning += 1

        text_digest = hashlib.sha256(cleaned_text.encode("utf-8")).digest()
        expected_digest = candidate.metadata.get("_cleaned_text_sha256")
        if expected_digest and expected_digest != text_digest.hex():
            source_stats.rejected["cleaned_text_hash_mismatch"] += 1
            source_stats.accepted_after_cleaning -= 1
            return
        if args.deduplicate and text_digest in seen_text_hashes:
            source_stats.exact_duplicates += 1
            return

        doc_id = make_doc_id(candidate.source, candidate.source_id)
        if doc_id in seen_doc_ids:
            source_stats.rejected["duplicate_source_id"] += 1
            return
        seen_text_hashes.add(text_digest)
        seen_doc_ids.add(doc_id)
        split = assign_split(
            doc_id,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
        )
        metadata = {
            **{
                key: value
                for key, value in candidate.metadata.items()
                if key != "_cleaned_text_sha256"
            },
            "devanagari_ratio": round(ratio, 6),
            "devanagari_letters": devanagari_letters,
        }
        buffers[split].append(
            {
                "doc_id": doc_id,
                "text": cleaned_text,
                "source": candidate.source,
                "source_id": candidate.source_id,
                "language": candidate.language or "ne",
                "url": candidate.url,
                "text_sha256": text_digest.hex(),
                "metadata_json": json.dumps(
                    metadata,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
        source_stats.written += 1
        source_stats.splits[split] += 1
        global_splits[split] += 1
        if len(buffers[split]) >= args.write_buffer_size:
            flush(split)

    sources = (
        (
            "nepali_pdf",
            iter_cleaned_candidates(args.pdf_dir, "nepali_pdf", args.batch_size),
        ),
        (
            "nepali_news",
            iter_cleaned_candidates(args.news_dir, "nepali_news", args.batch_size),
        ),
        (
            "nepali_lyrics",
            iter_cleaned_candidates(args.lyrics_dir, "nepali_lyrics", args.batch_size),
        ),
    )
    try:
        for source_name, candidates in sources:
            print(f"Scanning {source_name} ...")
            for candidate in candidates:
                process(candidate)
                if (
                    args.log_every
                    and stats[source_name].scanned % args.log_every == 0
                ):
                    source_stats = stats[source_name]
                    print(
                        f"  scanned={source_stats.scanned:,} "
                        f"selected={source_stats.selected_by_sampling:,} "
                        f"written={source_stats.written:,}"
                    )
            print(
                f"Finished {source_name}: scanned={stats[source_name].scanned:,}, "
                f"written={stats[source_name].written:,}"
            )
        for split in SPLIT_NAMES:
            flush(split)
    finally:
        for writer in writers.values():
            writer.close()

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "inputs": {
            "nepali_pdf": {
                "path": serialize_path(args.pdf_dir),
                "source_manifest": source_manifest(args.pdf_dir),
            },
            "nepali_news": {
                "path": serialize_path(args.news_dir),
                "source_manifest": source_manifest(args.news_dir),
            },
            "nepali_lyrics": {
                "path": serialize_path(args.lyrics_dir),
                "source_manifest": source_manifest(args.lyrics_dir),
            },
        },
        "sampling": {
            "global_fraction": args.sample_fraction,
            "source_fractions": {
                "nepali_pdf": args.pdf_fraction,
                "nepali_news": args.news_fraction,
                "nepali_lyrics": args.lyrics_fraction,
            },
            "effective_fractions": source_fractions,
        },
        "split_ratios": {
            "train": args.train_ratio,
            "validation": args.validation_ratio,
            "test": args.test_ratio,
        },
        "input_stage": "cleaned",
        "processing_operations": [
            "deterministic source sampling",
            "exact cleaned-text deduplication across sources",
            "stable canonical document IDs",
            "deterministic document-level split assignment",
        ],
        "deduplicate_exact_text": args.deduplicate,
        "canonical_columns": schema.names,
        "source_stats": {source: stats[source].to_dict() for source in SOURCE_NAMES},
        "split_counts": {split: global_splits[split] for split in SPLIT_NAMES},
        "total_documents": sum(global_splits.values()),
    }
    (args.output_dir / "build_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_dataset(args)
    print(f"Wrote standardized corpus to: {args.output_dir.resolve()}")
    for split in SPLIT_NAMES:
        print(f"  {split:10s}: {manifest['split_counts'][split]:,} documents")
    print(f"  total     : {manifest['total_documents']:,} documents")
    for source, source_stats in manifest["source_stats"].items():
        print(
            f"  {source:14s}: scanned={source_stats['scanned']:,} "
            f"selected={source_stats['selected_by_sampling']:,} "
            f"written={source_stats['written']:,}"
        )


if __name__ == "__main__":
    main()
