"""Interactive, read-only explorer for pretraining and finetuning datasets.

Run with:

    streamlit run apps/dataset_explorer.py

The app discovers the repository's raw, cleaned, processed, and tokenized
Parquet datasets. It also discovers future Parquet datasets placed under
``data/finetuning``, ``data/sft``, or ``data/preference``.
The repository's paired LIMA English/Nepali translation JSON is also exposed
as a dedicated dataset.
"""

from __future__ import annotations

import bisect
import csv
import hashlib
import io
import json
import os
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from attention_maps.inference.comparison import (
    ComparisonConfigurationError,
    DEFAULT_GEMINI_FLASH_LITE_MODEL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GOOGLE_GEMMA_MODEL,
    GoogleGenAIBackend,
    HuggingFaceBackend,
    LocalPeftBackend,
    build_decoding_grid,
    build_prompt,
    run_comparison,
)
from attention_maps.inference.arkios import (
    DEFAULT_ARKIOS_MODEL_ID,
    DEFAULT_ARKIOS_REVISION,
    ArkiosBackend,
    load_arkios,
)
from attention_maps.inference.himalayagpt import (
    DEFAULT_HIMALAYAGPT_MODEL_ID,
    DEFAULT_HIMALAYAGPT_REVISION,
    HimalayaGPTBackend,
    load_himalayagpt,
)
from attention_maps.inference.local_comparison import (
    LocalAdapterSpec,
    LocalInferenceError,
    build_local_decoding_grid,
    discover_local_adapters,
    load_local_model_pair,
    local_comparison_csv,
    run_local_comparison,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_FINETUNED_MODELS_ROOT = PROJECT_ROOT / "finetuned_models"
ARKIOS_BACKEND_NAME = "Arkios 1B Chat · local"
HIMALAYAGPT_BACKEND_NAME = "HimalayaGPT 0.5B Instruct · local"
DEFAULT_LIMA_TRANSLATIONS_PATH = (
    PROJECT_ROOT / "data_generation_pipeline" / "lima_translations.json"
)
SPLIT_NAMES = ("train", "validation", "test")
FUTURE_DATA_STAGES = ("finetuning", "sft", "preference", "reward_modeling")
MANIFEST_NAMES = (
    "download_manifest.json",
    "cleaning_manifest.json",
    "build_manifest.json",
    "tokenization_manifest.json",
    "dataset_info.json",
)
DATASET_LABELS = {
    "nepcov19tweets": "NepCOV19Tweets · sentiment",
}
SENTIMENT_LABELS = {-1: "negative", 0: "neutral", 1: "positive"}
VIEWER_PREFIX = "__viewer_"
TEXT_FIELD_NAMES = (
    "source_text",
    "translation",
    "text",
    "content",
    "article",
    "lyrics",
    "Lyrics",
    "messages",
    "prompt",
    "completion",
    "response",
    "chosen",
    "rejected",
)
DEFAULT_WORDCLOUD_STOPWORDS = (
    "अनि",
    "अब",
    "अथवा",
    "अरू",
    "एक",
    "एउटा",
    "का",
    "कि",
    "की",
    "को",
    "छ",
    "छन्",
    "तथा",
    "तर",
    "त्यो",
    "नै",
    "पनि",
    "भएको",
    "भने",
    "मा",
    "र",
    "लाई",
    "लागि",
    "ले",
    "वा",
    "यो",
    "हो",
)
ENGLISH_WORDCLOUD_STOPWORDS = (
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
    "you",
)
DEVANAGARI_FONT_CANDIDATES = (
    Path("/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoSerifDevanagari-Regular.ttf"),
    Path("/usr/share/fonts/truetype/lohit-nepali/Lohit-Nepali.ttf"),
    Path("/usr/share/fonts/truetype/freefont/FreeSans.ttf"),
)
LATIN_FONT_CANDIDATES = (
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    Path("/usr/share/fonts/truetype/freefont/FreeSans.ttf"),
)


@dataclass(frozen=True)
class DatasetSpec:
    """A logical dataset represented by one or more data files."""

    key: str
    label: str
    stage: str
    files: tuple[Path, ...]
    format: str = "parquet"

    @property
    def location(self) -> Path:
        if len(self.files) == 1:
            return self.files[0]
        return Path(_common_path(self.files))


def _common_path(paths: Sequence[Path]) -> str:
    """Return a common path without importing platform-specific helpers."""

    import os

    return os.path.commonpath([str(path) for path in paths])


def _parquet_files(path: Path, *, recursive: bool = True) -> tuple[Path, ...]:
    if not path.exists():
        return ()
    if path.is_file():
        return (path.resolve(),) if path.suffix.lower() == ".parquet" else ()
    iterator = path.rglob("*.parquet") if recursive else path.glob("*.parquet")
    return tuple(sorted(item.resolve() for item in iterator if item.is_file()))


def _add_spec(
    specs: list[DatasetSpec],
    seen: set[tuple[str, ...]],
    *,
    key: str,
    label: str,
    stage: str,
    files: Iterable[Path],
) -> None:
    resolved = tuple(sorted(Path(path).resolve() for path in files))
    signature = tuple(str(path) for path in resolved)
    if not resolved or signature in seen:
        return
    seen.add(signature)
    specs.append(DatasetSpec(key=key, label=label, stage=stage, files=resolved))


def discover_datasets(data_root: Path) -> list[DatasetSpec]:
    """Discover current and future logical Parquet datasets under ``data_root``."""

    root = data_root.expanduser().resolve()
    specs: list[DatasetSpec] = []
    seen: set[tuple[str, ...]] = set()

    raw_root = root / "raw"
    if raw_root.is_dir():
        for source_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
            files = _parquet_files(source_dir / "data")
            _add_spec(
                specs,
                seen,
                key=f"raw:{source_dir.name}",
                label=DATASET_LABELS.get(source_dir.name, source_dir.name),
                stage="raw",
                files=files,
            )

    cleaned_root = root / "cleaned"
    if cleaned_root.is_dir():
        for source_dir in sorted(
            path for path in cleaned_root.iterdir() if path.is_dir()
        ):
            _add_spec(
                specs,
                seen,
                key=f"cleaned:{source_dir.name}",
                label=source_dir.name,
                stage="cleaned",
                files=_parquet_files(source_dir / "data"),
            )

    processed_root = root / "processed"
    if processed_root.is_dir():
        for dataset_dir in sorted(
            path for path in processed_root.iterdir() if path.is_dir()
        ):
            for split in SPLIT_NAMES:
                split_file = dataset_dir / f"{split}.parquet"
                _add_spec(
                    specs,
                    seen,
                    key=f"processed:{dataset_dir.name}:{split}",
                    label=f"{dataset_dir.name} · {split}",
                    stage="processed",
                    files=_parquet_files(split_file),
                )

    tokenized_root = root / "tokenized"
    if tokenized_root.is_dir():
        for dataset_dir in sorted(
            path for path in tokenized_root.iterdir() if path.is_dir()
        ):
            for split in SPLIT_NAMES:
                _add_spec(
                    specs,
                    seen,
                    key=f"tokenized:{dataset_dir.name}:{split}",
                    label=f"{dataset_dir.name} · {split}",
                    stage="tokenized",
                    files=_parquet_files(dataset_dir / split),
                )

    for stage in FUTURE_DATA_STAGES:
        stage_root = root / stage
        if not stage_root.is_dir():
            continue
        grouped: dict[Path, list[Path]] = defaultdict(list)
        for parquet_path in stage_root.rglob("*.parquet"):
            grouped[parquet_path.parent].append(parquet_path)
        for parent, files in sorted(grouped.items(), key=lambda item: str(item[0])):
            relative = parent.relative_to(stage_root)
            label = str(relative) if str(relative) != "." else stage
            _add_spec(
                specs,
                seen,
                key=f"{stage}:{relative}",
                label=label,
                stage=stage,
                files=files,
            )

    return sorted(specs, key=lambda spec: (spec.stage, spec.label))


def custom_dataset(path: Path) -> DatasetSpec | None:
    """Create a dataset specification from a custom Parquet file or directory."""

    files = _parquet_files(path.expanduser())
    if not files:
        return None
    return DatasetSpec(
        key=f"custom:{path.expanduser().resolve()}",
        label=path.expanduser().resolve().name,
        stage="custom",
        files=files,
    )


def lima_translation_dataset(
    path: Path = DEFAULT_LIMA_TRANSLATIONS_PATH,
) -> DatasetSpec | None:
    """Return the repository's paired English/Nepali LIMA translation dataset."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        return None
    return DatasetSpec(
        key=f"translations:{resolved}",
        label="LIMA · Gemini/Gemma · English → Nepali",
        stage="translations",
        files=(resolved,),
        format="json",
    )


def file_signatures(files: Sequence[Path]) -> tuple[tuple[str, int, int], ...]:
    """Build a cache key that changes when a source file changes."""

    signatures = []
    for path in files:
        stat = path.stat()
        signatures.append((str(path), stat.st_size, stat.st_mtime_ns))
    return tuple(signatures)


def inspect_parquet(
    signatures: tuple[tuple[str, int, int], ...],
) -> dict[str, Any]:
    """Read Parquet metadata only; no dataset rows are materialized."""

    import pyarrow.parquet as pq

    row_groups: list[dict[str, Any]] = []
    common_columns: list[str] | None = None
    first_schema: list[dict[str, str]] = []
    schema_variants: set[tuple[tuple[str, str], ...]] = set()
    total_rows = 0

    for file_index, (path_text, _, _) in enumerate(signatures):
        parquet = pq.ParquetFile(path_text)
        arrow_schema = parquet.schema_arrow
        schema_tuple = tuple((field.name, str(field.type)) for field in arrow_schema)
        schema_variants.add(schema_tuple)
        names = list(arrow_schema.names)
        if common_columns is None:
            common_columns = names
            first_schema = [
                {"column": field.name, "type": str(field.type), "nullable": str(field.nullable)}
                for field in arrow_schema
            ]
        else:
            name_set = set(names)
            common_columns = [name for name in common_columns if name in name_set]

        for group_index in range(parquet.num_row_groups):
            rows = parquet.metadata.row_group(group_index).num_rows
            row_groups.append(
                {
                    "path": path_text,
                    "file_index": file_index,
                    "row_group": group_index,
                    "start": total_rows,
                    "rows": rows,
                }
            )
            total_rows += rows

    return {
        "format": "parquet",
        "rows": total_rows,
        "files": len(signatures),
        "bytes": sum(size for _, size, _ in signatures),
        "row_groups": row_groups,
        "columns": common_columns or [],
        "schema": first_schema,
        "schema_variants": len(schema_variants),
    }


def _json_value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError("JSON dataset must be a top-level array of objects")
    return data


def inspect_json(
    signatures: tuple[tuple[str, int, int], ...],
) -> dict[str, Any]:
    """Inspect a single JSON array dataset and infer its top-level schema."""

    if len(signatures) != 1:
        raise ValueError("JSON datasets must contain exactly one file")
    path_text, size, _ = signatures[0]
    records = _read_json_records(Path(path_text))
    columns = list(dict.fromkeys(key for record in records for key in record))
    schema = []
    for column in columns:
        types = {
            _json_value_type(record.get(column))
            for record in records
            if column in record and record.get(column) is not None
        }
        schema.append(
            {
                "column": column,
                "type": " | ".join(sorted(types)) if types else "null",
                "nullable": str(
                    any(column not in record or record.get(column) is None for record in records)
                ),
            }
        )
    return {
        "format": "json",
        "path": path_text,
        "rows": len(records),
        "files": 1,
        "bytes": size,
        "row_groups": [],
        "columns": columns,
        "schema": schema,
        "schema_variants": 1,
    }


def inspect_dataset(
    signatures: tuple[tuple[str, int, int], ...],
) -> dict[str, Any]:
    """Inspect a supported dataset using its file extension."""

    suffixes = {Path(path_text).suffix.lower() for path_text, _, _ in signatures}
    if suffixes == {".parquet"}:
        return inspect_parquet(signatures)
    if suffixes == {".json"}:
        return inspect_json(signatures)
    raise ValueError(f"Unsupported or mixed dataset formats: {sorted(suffixes)}")


def sample_parquet_rows(
    inventory: dict[str, Any],
    sample_size: int,
    seed: int,
    columns: Sequence[str],
) -> list[dict[str, Any]]:
    """Uniformly sample logical rows while reading only selected row groups."""

    import pyarrow.parquet as pq

    total_rows = int(inventory["rows"])
    if total_rows <= 0 or sample_size <= 0:
        return []

    selected_columns = list(columns)
    invalid = set(selected_columns) - set(inventory["columns"])
    if invalid:
        raise ValueError(f"Columns are not shared by every shard: {sorted(invalid)}")

    rng = random.Random(seed)
    global_indices = rng.sample(range(total_rows), k=min(sample_size, total_rows))
    row_groups = inventory["row_groups"]
    group_ends = [group["start"] + group["rows"] for group in row_groups]
    requested: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)

    for global_index in global_indices:
        group_position = bisect.bisect_right(group_ends, global_index)
        group = row_groups[group_position]
        local_index = global_index - group["start"]
        requested[(group["path"], group["row_group"])].append(
            (global_index, local_index)
        )

    sampled_by_index: dict[int, dict[str, Any]] = {}
    for (path_text, group_index), positions in requested.items():
        parquet = pq.ParquetFile(path_text)
        table = parquet.read_row_group(group_index, columns=selected_columns)
        for global_index, local_index in positions:
            record = table.slice(local_index, 1).to_pylist()[0]
            record[f"{VIEWER_PREFIX}row_index"] = global_index
            record[f"{VIEWER_PREFIX}file"] = path_text
            record[f"{VIEWER_PREFIX}row_group"] = group_index
            sampled_by_index[global_index] = record

    return [sampled_by_index[index] for index in global_indices]


def sample_json_rows(
    inventory: dict[str, Any],
    sample_size: int,
    seed: int,
    columns: Sequence[str],
) -> list[dict[str, Any]]:
    """Uniformly sample selected fields from a JSON array dataset."""

    selected_columns = list(columns)
    invalid = set(selected_columns) - set(inventory["columns"])
    if invalid:
        raise ValueError(f"Unknown JSON columns: {sorted(invalid)}")
    records = _read_json_records(Path(inventory["path"]))
    if not records or sample_size <= 0:
        return []
    rng = random.Random(seed)
    indices = rng.sample(range(len(records)), k=min(sample_size, len(records)))
    sampled = []
    for index in indices:
        record = {column: records[index].get(column) for column in selected_columns}
        record[f"{VIEWER_PREFIX}row_index"] = index
        record[f"{VIEWER_PREFIX}file"] = inventory["path"]
        sampled.append(record)
    return sampled


def sample_dataset_rows(
    inventory: dict[str, Any],
    sample_size: int,
    seed: int,
    columns: Sequence[str],
) -> list[dict[str, Any]]:
    """Uniformly sample a supported dataset without changing the UI contract."""

    if inventory.get("format") == "json":
        return sample_json_rows(inventory, sample_size, seed, columns)
    return sample_parquet_rows(inventory, sample_size, seed, columns)


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def preview_value(value: Any, max_characters: int) -> Any:
    if isinstance(value, (list, tuple, dict)):
        value = json.dumps(value, ensure_ascii=False, default=str)
    if isinstance(value, bytes):
        value = value.hex()
    if isinstance(value, str) and len(value) > max_characters:
        return f"{value[:max_characters]}…"
    return value


def preview_records(
    records: Sequence[dict[str, Any]], max_characters: int
) -> list[dict[str, Any]]:
    return [
        {
            key: preview_value(value, max_characters)
            for key, value in record.items()
            if not key.startswith(VIEWER_PREFIX)
        }
        for record in records
    ]


def text_columns(schema: Sequence[dict[str, str]]) -> list[str]:
    """Return columns that can plausibly contain natural-language text."""

    preferred = [
        name
        for name in TEXT_FIELD_NAMES
        if any(field["column"] == name for field in schema)
    ]
    other_strings = [
        field["column"]
        for field in schema
        if "string" in field["type"].lower()
        and field["column"] not in preferred
        and not field["column"].startswith(VIEWER_PREFIX)
    ]
    return preferred + other_strings


def sentiment_column(schema: Sequence[dict[str, str]]) -> str | None:
    """Return a conventional sentiment/label column when one is present."""

    candidates = ("sentiment", "sentiment_label", "label")
    by_normalized_name = {
        field["column"].casefold(): field["column"] for field in schema
    }
    return next(
        (by_normalized_name[name] for name in candidates if name in by_normalized_name),
        None,
    )


def sentiment_label(value: Any) -> str | None:
    """Normalize the numeric labels used by NepCOV19Tweets."""

    try:
        numeric_value = int(value)
    except (TypeError, ValueError):
        return (
            str(value).strip().casefold()
            if isinstance(value, str) and value.strip()
            else None
        )
    return SENTIMENT_LABELS.get(numeric_value)


def sentiment_prediction(text: str) -> str | None:
    """Extract the first canonical sentiment label from model output."""

    if not isinstance(text, str):
        return None
    match = re.search(r"(?<![a-z])(negative|neutral|positive)(?![a-z])", text.casefold())
    return match.group(1) if match else None


def extract_text(value: Any) -> list[str]:
    """Extract text from strings or common nested chat/instruction values."""

    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        extracted: list[str] = []
        for item in value:
            extracted.extend(extract_text(item))
        return extracted
    if isinstance(value, dict):
        content_keys = (
            "content",
            "text",
            "prompt",
            "completion",
            "response",
            "chosen",
            "rejected",
        )
        selected = [value[key] for key in content_keys if key in value]
        extracted = []
        for item in selected:
            extracted.extend(extract_text(item))
        return extracted
    return []


def split_human_assistant_example(text: str) -> tuple[str, str | None]:
    """Split a serialized single-turn LIMA example into prompt and reference."""

    stripped = text.strip()
    match = re.match(
        r"^HUMAN:\s*(.*?)\n\s*ASSISTANT:\s*(.*)$",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return stripped, None
    prompt, reference = (part.strip() for part in match.groups())
    return prompt, reference or None


def unicode_words(text: str, *, include_numbers: bool = False) -> list[str]:
    """Tokenize words while retaining Unicode combining marks used by Nepali."""

    normalized = unicodedata.normalize("NFC", text).casefold()
    words: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        token = "".join(buffer).strip("-'’")
        buffer.clear()
        if not token:
            return
        categories = [unicodedata.category(character) for character in token]
        if any(category.startswith("L") for category in categories) or (
            include_numbers and any(category.startswith("N") for category in categories)
        ):
            words.append(token)

    for character in normalized:
        category = unicodedata.category(character)
        is_letter_or_mark = category.startswith(("L", "M"))
        is_number = include_numbers and category.startswith("N")
        is_joiner = character in {"\u200c", "\u200d"}
        if is_letter_or_mark or is_number or is_joiner:
            buffer.append(character)
        elif character in {"-", "'", "’"} and buffer:
            buffer.append(character)
        else:
            flush()
    flush()
    return words


def parse_stopwords(value: str) -> set[str]:
    """Parse editable comma/whitespace-separated stopwords."""

    return {
        word
        for chunk in value.replace(",", " ").split()
        for word in unicode_words(chunk, include_numbers=True)
    }


def word_frequencies(
    records: Sequence[dict[str, Any]],
    columns: Sequence[str],
    *,
    stopwords: set[str] | None = None,
    min_characters: int = 2,
    min_frequency: int = 1,
    include_numbers: bool = False,
) -> Counter[str]:
    """Count Unicode words in selected columns of sampled dataset records."""

    excluded = stopwords or set()
    counts: Counter[str] = Counter()
    for record in records:
        for column in columns:
            for text in extract_text(record.get(column)):
                counts.update(
                    word
                    for word in unicode_words(text, include_numbers=include_numbers)
                    if len(word) >= min_characters and word not in excluded
                )
    return Counter(
        {
            word: count
            for word, count in counts.items()
            if count >= min_frequency
        }
    )


def find_devanagari_font() -> Path | None:
    """Return the first commonly installed font capable of rendering Nepali."""

    return next((path for path in DEVANAGARI_FONT_CANDIDATES if path.is_file()), None)


def find_latin_font() -> Path | None:
    """Return the first commonly installed font with reliable Latin coverage."""

    return next((path for path in LATIN_FONT_CANDIDATES if path.is_file()), None)


def create_wordcloud(
    frequencies: dict[str, int],
    *,
    font_path: Path | None,
    max_words: int,
    width: int = 1_400,
    height: int = 700,
    background_color: str = "white",
    colormap: str = "viridis",
    seed: int = 42,
) -> Any:
    """Create a WordCloud image from already-tokenized word frequencies."""

    from wordcloud import WordCloud

    cloud = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        colormap=colormap,
        max_words=max_words,
        font_path=str(font_path) if font_path else None,
        random_state=seed,
        collocations=False,
    )
    return cloud.generate_from_frequencies(frequencies)


def parse_number_list(
    value: str,
    *,
    value_type: type[float] | type[int],
    name: str,
) -> list[float] | list[int]:
    """Parse comma/whitespace-separated numeric UI values."""

    parts = value.replace(",", " ").split()
    if not parts:
        raise ComparisonConfigurationError(f"{name} cannot be empty")
    try:
        return [value_type(part) for part in parts]
    except ValueError as error:
        raise ComparisonConfigurationError(
            f"{name} must contain only {value_type.__name__} values"
        ) from error


def parse_model_ids(value: str) -> list[str]:
    """Parse one model repository ID per line or comma."""

    return [item.strip() for item in value.replace(",", "\n").splitlines() if item.strip()]


def comparison_csv(results: Sequence[Any]) -> bytes:
    """Serialize comparison dataclasses as a UTF-8 CSV download."""

    if not results:
        return b""
    output = io.StringIO()
    rows = [result.as_dict() for result in results]
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def secret_fingerprint(secret: str) -> str:
    """Provide cache invalidation without using a secret as a visible cache key."""

    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def find_manifest(spec: DatasetSpec, data_root: Path) -> Path | None:
    """Find the closest known manifest associated with a dataset."""

    start = spec.location if spec.location.is_dir() else spec.location.parent
    boundary = data_root.expanduser().resolve()
    current = start.resolve()
    while True:
        for name in MANIFEST_NAMES:
            candidate = current / name
            if candidate.is_file():
                return candidate
        if current == boundary or current.parent == current:
            return None
        try:
            current.relative_to(boundary)
        except ValueError:
            return None
        current = current.parent


def default_columns(columns: Sequence[str]) -> list[str]:
    preferred = (
        "source_text",
        "translation",
        "doc_id",
        "text",
        "messages",
        "prompt",
        "completion",
        "chosen",
        "rejected",
        "source",
        "source_id",
        "language",
        "url",
        "metadata_json",
        "input_ids",
        "num_tokens",
        "text_sha256",
    )
    selected = [name for name in preferred if name in columns]
    return selected or list(columns[: min(8, len(columns))])


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bytes):
        return value.hex()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def render_messages(st: Any, messages: Any) -> bool:
    """Render a standard chat record and report whether it was recognized."""

    if not isinstance(messages, list) or not all(
        isinstance(message, dict) for message in messages
    ):
        return False
    st.markdown("#### Conversation")
    for message in messages:
        role = str(message.get("role", "unknown")).upper()
        content = message.get("content", "")
        st.markdown(f"**{role}**")
        st.text(str(content))
    return True


def render_full_record(st: Any, record: dict[str, Any]) -> None:
    """Render long text/chat fields without truncation, followed by full JSON."""

    viewer_fields = {
        key.removeprefix(VIEWER_PREFIX): value
        for key, value in record.items()
        if key.startswith(VIEWER_PREFIX)
    }
    data_fields = {
        key: value for key, value in record.items() if not key.startswith(VIEWER_PREFIX)
    }

    if viewer_fields:
        location_parts = [f"Global row {viewer_fields.get('row_index')}"]
        if viewer_fields.get("row_group") is not None:
            location_parts.append(f"row group {viewer_fields.get('row_group')}")
        location_parts.append(str(viewer_fields.get("file")))
        st.caption(" · ".join(location_parts))

    render_messages(st, data_fields.get("messages"))
    long_text_fields = tuple(field for field in TEXT_FIELD_NAMES if field != "messages")
    shown: set[str] = set()
    if "source_text" in data_fields or "translation" in data_fields:
        st.markdown("#### Original and translated text")
        original_column, translation_column = st.columns(2)
        original_column.markdown("##### Original (English)")
        original_column.text(str(data_fields.get("source_text") or "—"))
        translation_column.markdown("##### Translation (Nepali)")
        translation_column.text(str(data_fields.get("translation") or "—"))
        shown.update({"source_text", "translation"})
    for field in long_text_fields:
        value = data_fields.get(field)
        if value is None or field == "messages" or field in shown:
            continue
        st.markdown(f"#### `{field}`")
        st.text(str(value))
        shown.add(field)

    with st.expander("Complete record as JSON", expanded=not bool(shown)):
        st.json(_jsonable(data_fields), expanded=True)


def run_app() -> None:
    try:
        import streamlit as st
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Streamlit is not installed. Run `pip install -r requirements.txt`."
        ) from error

    st.set_page_config(
        page_title="Dataset Explorer",
        page_icon="🔎",
        layout="wide",
    )
    st.title("Pretraining & Finetuning Dataset Explorer")
    st.caption(
        "Read-only inspection of full Nepali records, schemas, manifests, and "
        "uniform random samples, including paired English/Nepali translations. "
        "Future English and finetuning Parquet data use the same viewer."
    )

    @st.cache_data(show_spinner=False)
    def cached_inventory(
        signatures: tuple[tuple[str, int, int], ...],
    ) -> dict[str, Any]:
        return inspect_dataset(signatures)

    @st.cache_data(show_spinner=False)
    def cached_sample(
        inventory: dict[str, Any],
        sample_size: int,
        seed: int,
        columns: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        return sample_dataset_rows(inventory, sample_size, seed, columns)

    @st.cache_resource(show_spinner=False)
    def cached_google_backend(
        model_id: str,
        credential_fingerprint: str,
        _api_key: str,
    ) -> GoogleGenAIBackend:
        del credential_fingerprint
        return GoogleGenAIBackend(model_id, _api_key)

    @st.cache_resource(show_spinner=False)
    def cached_huggingface_backend(
        model_id: str,
        provider: str,
        credential_fingerprint: str,
        _token: str,
    ) -> HuggingFaceBackend:
        del credential_fingerprint
        return HuggingFaceBackend(model_id, _token, provider)

    @st.cache_resource(show_spinner=False)
    def cached_local_model_pair(
        adapter_key: str,
        adapter_label: str,
        adapter_path: str,
        base_model_id: str,
        device: str,
        dtype: str,
        local_files_only: bool,
    ) -> Any:
        adapter_spec = LocalAdapterSpec(
            key=adapter_key,
            label=adapter_label,
            path=Path(adapter_path),
            base_model_id=base_model_id,
        )
        return load_local_model_pair(
            adapter_spec,
            device=device,
            dtype=dtype,
            local_files_only=local_files_only,
        )

    @st.cache_resource(show_spinner=False)
    def cached_arkios(
        model_id: str,
        revision: str,
        device: str,
        dtype: str,
        local_files_only: bool,
    ) -> Any:
        return load_arkios(
            model_id=model_id,
            revision=revision,
            device=device,
            dtype=dtype,
            local_files_only=local_files_only,
        )

    @st.cache_resource(show_spinner=False)
    def cached_himalayagpt(
        model_id: str,
        revision: str,
        device: str,
        dtype: str,
        local_files_only: bool,
    ) -> Any:
        return load_himalayagpt(
            model_id=model_id,
            revision=revision,
            device=device,
            dtype=dtype,
            local_files_only=local_files_only,
        )

    st.sidebar.header("Dataset")
    data_root_text = st.sidebar.text_input("Data root", str(DEFAULT_DATA_ROOT))
    data_root = Path(data_root_text).expanduser()
    use_custom = st.sidebar.checkbox("Use custom Parquet path")

    spec: DatasetSpec | None = None
    if use_custom:
        custom_path_text = st.sidebar.text_input("Parquet file or directory")
        if custom_path_text:
            spec = custom_dataset(Path(custom_path_text))
            if spec is None:
                st.sidebar.error("No Parquet files were found at that path.")
    else:
        specs = discover_datasets(data_root)
        translation_spec = lima_translation_dataset()
        if translation_spec is not None:
            specs.append(translation_spec)
        specs.sort(key=lambda item: (item.stage, item.label))
        if not specs:
            st.warning(f"No supported datasets found under `{data_root}`.")
            st.stop()
        stages = sorted({item.stage for item in specs})
        stage = st.sidebar.selectbox("Pipeline stage", stages)
        stage_specs = [item for item in specs if item.stage == stage]
        spec = st.sidebar.selectbox(
            "Dataset / split",
            stage_specs,
            format_func=lambda item: item.label,
        )

    if spec is None:
        st.info("Select a valid dataset to begin.")
        st.stop()

    try:
        signatures = file_signatures(spec.files)
        with st.spinner("Reading dataset metadata…"):
            inventory = cached_inventory(signatures)
    except (OSError, ValueError, ImportError) as error:
        st.error(f"Could not inspect this dataset: {error}")
        st.stop()

    st.subheader(f"{spec.stage.title()} · {spec.label}")
    st.code(str(spec.location), language=None)
    metric_columns = st.columns(5)
    metric_columns[0].metric("Rows", f"{inventory['rows']:,}")
    metric_columns[1].metric("Files", f"{inventory['files']:,}")
    grouping_label = "JSON documents" if inventory["format"] == "json" else "Row groups"
    grouping_value = inventory["files"] if inventory["format"] == "json" else len(
        inventory["row_groups"]
    )
    metric_columns[2].metric(grouping_label, f"{grouping_value:,}")
    metric_columns[3].metric("Columns", f"{len(inventory['columns']):,}")
    metric_columns[4].metric("Disk size", format_bytes(inventory["bytes"]))

    if inventory["schema_variants"] > 1:
        st.warning(
            f"The files contain {inventory['schema_variants']} schema variants. "
            "Sampling is limited to columns shared by every shard."
        )

    (
        details_tab,
        sample_tab,
        wordcloud_tab,
        local_inference_tab,
        inference_tab,
        manifest_tab,
    ) = st.tabs(
        [
            "Schema",
            "Random records",
            "Word cloud",
            "Local base vs finetuned",
            "Model comparison",
            "Manifest",
        ]
    )

    with details_tab:
        st.dataframe(inventory["schema"], use_container_width=True, hide_index=True)
        with st.expander("Dataset files"):
            st.code("\n".join(str(path) for path in spec.files), language=None)

    with sample_tab:
        controls = st.columns([1, 1, 3])
        sample_size = controls[0].number_input(
            "Records", min_value=1, max_value=20, value=5, step=1
        )
        base_seed = controls[1].number_input(
            "Random seed", min_value=0, value=42, step=1
        )
        selected_columns = controls[2].multiselect(
            "Columns to read",
            inventory["columns"],
            default=default_columns(inventory["columns"]),
        )
        preview_limit = st.slider(
            "Table preview characters per field", 100, 2_000, 400, 100
        )

        refresh_key = f"refresh:{spec.key}"
        if refresh_key not in st.session_state:
            st.session_state[refresh_key] = 0
        if st.button("New random sample", type="primary"):
            st.session_state[refresh_key] += 1
        effective_seed = int(base_seed) + st.session_state[refresh_key]

        if not selected_columns:
            st.info("Select at least one column.")
        else:
            try:
                with st.spinner("Reading sampled records…"):
                    records = cached_sample(
                        inventory,
                        int(sample_size),
                        effective_seed,
                        tuple(selected_columns),
                    )
            except (OSError, ValueError, ImportError) as error:
                st.error(f"Could not read the sample: {error}")
                records = []

            if records:
                st.caption(
                    f"Uniform random sample using effective seed {effective_seed}. "
                    "The table is shortened only for display; the full record below is not."
                )
                st.dataframe(
                    preview_records(records, preview_limit),
                    use_container_width=True,
                    hide_index=True,
                )
                record_index = st.selectbox(
                    "Full record",
                    range(len(records)),
                    format_func=lambda index: (
                        f"Sample {index + 1} · global row "
                        f"{records[index].get(f'{VIEWER_PREFIX}row_index')}"
                    ),
                )
                render_full_record(st, records[record_index])

    with wordcloud_tab:
        candidate_columns = text_columns(inventory["schema"])
        paired_translation_clouds = {"source_text", "translation"}.issubset(
            inventory["columns"]
        )
        if not candidate_columns:
            st.info("This dataset has no detectable text-bearing columns.")
        else:
            st.markdown(
                "The cloud is computed from a deterministic random sample and reads "
                "only the selected columns. Increase the sample carefully for large corpora."
            )
            cloud_controls = st.columns([2, 1, 1, 1])
            if paired_translation_clouds:
                cloud_columns = ["source_text", "translation"]
                cloud_controls[0].markdown(
                    "**Text columns**  \nOriginal (`source_text`) and Nepali (`translation`)"
                )
            else:
                cloud_columns = cloud_controls[0].multiselect(
                    "Text columns",
                    candidate_columns,
                    default=candidate_columns[:1],
                    key=f"wordcloud-columns:{spec.key}",
                )
            maximum_sample = max(1, min(5_000, int(inventory["rows"])))
            default_sample = min(500, maximum_sample)
            cloud_sample_size = cloud_controls[1].number_input(
                "Sampled rows",
                min_value=1,
                max_value=maximum_sample,
                value=default_sample,
                step=1,
            )
            maximum_words = cloud_controls[2].number_input(
                "Maximum words", min_value=10, max_value=500, value=150, step=10
            )
            minimum_frequency = cloud_controls[3].number_input(
                "Minimum frequency", min_value=1, max_value=100, value=2, step=1
            )

            option_columns = st.columns(3)
            minimum_characters = option_columns[0].number_input(
                "Minimum word characters", min_value=1, max_value=20, value=2
            )
            include_numbers = option_columns[1].checkbox("Include numeric tokens")
            colormap = option_columns[2].selectbox(
                "Colour map", ("viridis", "plasma", "magma", "cividis", "turbo")
            )
            if paired_translation_clouds:
                font_columns = st.columns(2)
                default_latin_font = find_latin_font()
                english_font_text = font_columns[0].text_input(
                    "Original English font path",
                    str(default_latin_font) if default_latin_font else "",
                    help="Use a font with Latin glyph coverage, such as DejaVu Sans.",
                )
                default_devanagari_font = find_devanagari_font()
                nepali_font_text = font_columns[1].text_input(
                    "Nepali translation font path",
                    str(default_devanagari_font) if default_devanagari_font else "",
                    help="Use a Devanagari-capable .ttf/.otf font.",
                )
                stopword_columns = st.columns(2)
                english_stopword_text = stopword_columns[0].text_area(
                    "Original-text stopwords",
                    ", ".join(ENGLISH_WORDCLOUD_STOPWORDS),
                    height=100,
                )
                nepali_stopword_text = stopword_columns[1].text_area(
                    "Nepali-translation stopwords",
                    ", ".join(DEFAULT_WORDCLOUD_STOPWORDS),
                    height=100,
                )
                cloud_groups = (
                    (
                        "Original (English)",
                        ("source_text",),
                        english_stopword_text,
                        english_font_text,
                    ),
                    (
                        "Translation (Nepali)",
                        ("translation",),
                        nepali_stopword_text,
                        nepali_font_text,
                    ),
                )
            else:
                default_devanagari_font = find_devanagari_font()
                font_text = st.text_input(
                    "Font path",
                    str(default_devanagari_font) if default_devanagari_font else "",
                    help="Use a Devanagari-capable .ttf/.otf font for Nepali text.",
                )
                stopword_text = st.text_area(
                    "Stopwords (comma, space, or newline separated)",
                    ", ".join(DEFAULT_WORDCLOUD_STOPWORDS),
                    height=100,
                )
                cloud_groups = (
                    ("Selected text", tuple(cloud_columns), stopword_text, font_text),
                )

            if not cloud_columns:
                st.info("Select at least one text column.")
            elif st.button("Generate word cloud", type="primary"):
                invalid_fonts = [
                    Path(group_font_text).expanduser()
                    for _, _, _, group_font_text in cloud_groups
                    if group_font_text.strip()
                    and not Path(group_font_text).expanduser().is_file()
                ]
                if invalid_fonts:
                    st.error(f"Font file not found: {invalid_fonts[0]}")
                else:
                    try:
                        with st.spinner("Sampling text and building word frequencies…"):
                            cloud_records = cached_sample(
                                inventory,
                                int(cloud_sample_size),
                                int(base_seed),
                                tuple(cloud_columns),
                            )
                            cloud_outputs = []
                            for (
                                title,
                                columns,
                                group_stopwords,
                                group_font_text,
                            ) in cloud_groups:
                                frequencies = word_frequencies(
                                    cloud_records,
                                    columns,
                                    stopwords=parse_stopwords(group_stopwords),
                                    min_characters=int(minimum_characters),
                                    min_frequency=int(minimum_frequency),
                                    include_numbers=include_numbers,
                                )
                                cloud = create_wordcloud(
                                    frequencies,
                                    font_path=(
                                        Path(group_font_text).expanduser()
                                        if group_font_text.strip()
                                        else None
                                    ),
                                    max_words=int(maximum_words),
                                    colormap=colormap,
                                    seed=int(base_seed),
                                )
                                cloud_outputs.append((title, frequencies, cloud))
                    except (ImportError, OSError, ValueError) as error:
                        st.error(f"Could not generate the word cloud: {error}")
                    else:
                        output_columns = st.columns(len(cloud_outputs))
                        for output_column, (title, frequencies, cloud) in zip(
                            output_columns, cloud_outputs
                        ):
                            output_column.markdown(f"#### {title}")
                            output_column.image(
                                cloud.to_array(), use_container_width=True
                            )
                            output_column.caption(
                                f"{len(frequencies):,} retained word types from "
                                f"{len(cloud_records):,} sampled records."
                            )
                            top_words = [
                                {"word": word, "count": count}
                                for word, count in frequencies.most_common(50)
                            ]
                            output_column.markdown("##### Most frequent words")
                            output_column.dataframe(
                                top_words, use_container_width=True, hide_index=True
                            )

    with local_inference_tab:
        st.markdown(
            "Compare a local base model with its finetuned PEFT adapter on exactly "
            "the same prompt and decoding parameters. No inference API is used."
        )
        local_adapters = discover_local_adapters(DEFAULT_FINETUNED_MODELS_ROOT)
        local_text_columns = text_columns(inventory["schema"])
        if not local_adapters:
            st.warning(
                f"No complete PEFT adapters were found under "
                f"`{DEFAULT_FINETUNED_MODELS_ROOT}`."
            )
        elif not local_text_columns:
            st.warning(
                "This dataset has no detectable text field. Select another dataset "
                "or use custom prompt text below."
            )

        selected_adapter = st.selectbox(
            "Local finetuned model",
            local_adapters,
            format_func=lambda item: item.label,
            disabled=not local_adapters,
            key="local-adapter",
        ) if local_adapters else None
        if selected_adapter is not None:
            st.caption(
                f"Base: `{selected_adapter.base_model_id}` · Adapter: "
                f"`{selected_adapter.path}`"
            )

        local_prompt_source = st.radio(
            "Evaluation input",
            ("Selected dataset instance", "Custom prompt"),
            horizontal=True,
            key=f"local-prompt-source:{spec.key}",
        )
        local_source_text = ""
        local_reference_text: str | None = None
        local_record_label = "custom"
        if local_prompt_source == "Selected dataset instance":
            if local_text_columns:
                local_source_controls = st.columns([2, 2, 1, 1])
                default_local_column = (
                    local_text_columns.index("translation")
                    if "translation" in local_text_columns
                    else 0
                )
                local_source_column = local_source_controls[0].selectbox(
                    "Prompt field",
                    local_text_columns,
                    index=default_local_column,
                    key=f"local-source-column:{spec.key}",
                )
                reference_options = ["Auto / none"] + [
                    column
                    for column in local_text_columns
                    if column != local_source_column
                ]
                local_reference_column = local_source_controls[1].selectbox(
                    "Reference field (optional)",
                    reference_options,
                    key=f"local-reference-column:{spec.key}:{local_source_column}",
                    help=(
                        "Auto extracts the ASSISTANT portion of a serialized "
                        "HUMAN/ASSISTANT example when available."
                    ),
                )
                local_sample_seed = local_source_controls[2].number_input(
                    "Instance seed",
                    min_value=0,
                    value=42,
                    step=1,
                    key=f"local-sample-seed:{spec.key}",
                )
                local_maximum_chars = local_source_controls[3].number_input(
                    "Maximum characters",
                    min_value=50,
                    max_value=20_000,
                    value=2_000,
                    step=50,
                    key=f"local-maximum-chars:{spec.key}",
                )
                local_sample_columns = [local_source_column]
                if local_reference_column != "Auto / none":
                    local_sample_columns.append(local_reference_column)
                try:
                    local_records = cached_sample(
                        inventory,
                        1,
                        int(local_sample_seed),
                        tuple(local_sample_columns),
                    )
                except (OSError, ValueError, ImportError) as error:
                    st.error(f"Could not read the evaluation instance: {error}")
                    local_records = []
                if local_records:
                    local_record = local_records[0]
                    source_parts = extract_text(local_record.get(local_source_column))
                    serialized_source = "\n\n".join(source_parts)
                    local_source_text, embedded_reference = split_human_assistant_example(
                        serialized_source
                    )
                    if local_reference_column == "Auto / none":
                        local_reference_text = embedded_reference
                    else:
                        reference_parts = extract_text(
                            local_record.get(local_reference_column)
                        )
                        local_reference_text = "\n\n".join(reference_parts) or None
                    local_source_text = local_source_text[: int(local_maximum_chars)]
                    if local_reference_text:
                        local_reference_text = local_reference_text[
                            : int(local_maximum_chars)
                        ]
                    global_row = local_record.get(f"{VIEWER_PREFIX}row_index")
                    local_record_label = (
                        f"{local_source_column}:{global_row}:{int(local_sample_seed)}"
                    )
                    st.caption(f"Selected dataset global row: {global_row}")
            else:
                st.info("Choose Custom prompt because this dataset has no text field.")
        else:
            local_source_text = "नेपालका हिमालहरूको महत्त्वबारे छोटकरीमा लेख्नुहोस्।"

        editable_local_source = st.text_area(
            "Model input text",
            value=local_source_text,
            height=150,
            key=(
                f"local-source-text:{spec.key}:{local_prompt_source}:"
                f"{local_record_label}"
            ),
        )
        if local_reference_text:
            with st.expander("Dataset reference answer", expanded=True):
                st.text(local_reference_text)

        local_prompt_columns = st.columns(2)
        local_prompt_template = local_prompt_columns[0].text_area(
            "Prompt template",
            value="{text}",
            height=100,
            key=f"local-prompt-template:{spec.key}",
            help="Use {text} where the selected dataset text should be inserted.",
        )
        local_system_prompt = local_prompt_columns[1].text_area(
            "System prompt (optional)",
            value="",
            height=100,
            key=f"local-system-prompt:{spec.key}",
        )
        try:
            local_prompt_preview = build_prompt(
                local_prompt_template, editable_local_source
            )
        except ComparisonConfigurationError as error:
            st.error(str(error))
            local_prompt_preview = ""
        if local_prompt_preview:
            with st.expander("Final local prompt preview"):
                st.text(local_prompt_preview)

        st.markdown("#### Local decoding grid")
        local_decoding_columns = st.columns(6)
        local_temperatures_text = local_decoding_columns[0].text_input(
            "Temperatures",
            "0, 0.7",
            key="local-temperatures",
            help="Temperature 0 uses greedy decoding.",
        )
        local_top_ps_text = local_decoding_columns[1].text_input(
            "Top-p values", "0.95", key="local-top-ps"
        )
        local_top_ks_text = local_decoding_columns[2].text_input(
            "Top-k values", "40", key="local-top-ks"
        )
        local_max_tokens = local_decoding_columns[3].number_input(
            "Maximum new tokens",
            min_value=1,
            max_value=1_024,
            value=128,
            key="local-max-new-tokens",
        )
        local_repetition_penalty = local_decoding_columns[4].number_input(
            "Repetition penalty",
            min_value=0.1,
            max_value=5.0,
            value=1.05,
            step=0.05,
            key="local-repetition-penalty",
        )
        local_generation_seed = local_decoding_columns[5].number_input(
            "Generation seed",
            min_value=0,
            value=42,
            key="local-generation-seed",
        )

        local_runtime_columns = st.columns(3)
        local_device = local_runtime_columns[0].selectbox(
            "Device", ("auto", "cuda", "cpu"), key="local-device"
        )
        local_dtype = local_runtime_columns[1].selectbox(
            "Weight dtype",
            ("auto", "float32", "bfloat16", "float16"),
            key="local-dtype",
        )
        local_files_only = local_runtime_columns[2].checkbox(
            "Use cached base weights only",
            value=False,
            key="local-files-only",
            help=(
                "Adapters and tokenizers are local. Base weights may be downloaded "
                "from Hugging Face once when this is off."
            ),
        )

        local_configs = []
        local_grid_error = None
        try:
            local_temperatures = parse_number_list(
                local_temperatures_text,
                value_type=float,
                name="temperatures",
            )
            local_top_ps = parse_number_list(
                local_top_ps_text, value_type=float, name="top-p"
            )
            local_top_ks = parse_number_list(
                local_top_ks_text, value_type=int, name="top-k"
            )
            local_configs = build_local_decoding_grid(
                local_temperatures,
                local_top_ps,
                local_top_ks,
                max_new_tokens=int(local_max_tokens),
                repetition_penalty=float(local_repetition_penalty),
                seed=int(local_generation_seed),
            )
        except (ComparisonConfigurationError, LocalInferenceError) as error:
            local_grid_error = str(error)
            st.error(local_grid_error)

        local_generation_count = len(local_configs) * 2
        st.info(
            f"This run will create {local_generation_count} generation(s): base and "
            "finetuned output for each decoding configuration."
        )
        if local_generation_count > 12:
            st.error("Reduce the local decoding grid to at most 12 generations.")

        local_result_context = hashlib.sha256(
            (
                f"{selected_adapter.key if selected_adapter else ''}\0"
                f"{local_prompt_preview}\0{local_system_prompt}\0"
                f"{local_configs}\0{local_device}\0{local_dtype}\0"
                f"{local_files_only}"
            ).encode("utf-8")
        ).hexdigest()[:16]
        local_results_key = (
            f"local-inference-results:{spec.key}:{local_result_context}"
        )
        local_run_disabled = (
            selected_adapter is None
            or not local_prompt_preview
            or not local_configs
            or local_generation_count > 12
            or bool(local_grid_error)
        )
        if st.button(
            "Run local base vs finetuned comparison",
            type="primary",
            disabled=local_run_disabled,
        ):
            try:
                with st.spinner(
                    "Loading the base model and local adapter, then generating…"
                ):
                    local_bundle = cached_local_model_pair(
                        selected_adapter.key,
                        selected_adapter.label,
                        str(selected_adapter.path),
                        selected_adapter.base_model_id,
                        local_device,
                        local_dtype,
                        local_files_only,
                    )
                    st.session_state[local_results_key] = run_local_comparison(
                        local_bundle,
                        local_prompt_preview,
                        local_configs,
                        system_prompt=local_system_prompt,
                    )
            except (LocalInferenceError, RuntimeError, OSError) as error:
                st.error(f"Could not run local comparison: {error}")

        local_results = st.session_state.get(local_results_key, [])
        if local_results:
            st.markdown("#### Local comparison results")
            local_summary_rows = [
                {
                    "variant": result.variant,
                    "decoding": result.decoding,
                    "latency_seconds": round(result.latency_seconds, 3),
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "status": "error" if result.error else "ok",
                    "output_preview": (result.output or result.error or "")[:240],
                }
                for result in local_results
            ]
            st.dataframe(
                local_summary_rows, use_container_width=True, hide_index=True
            )
            results_by_decoding: dict[str, dict[str, Any]] = defaultdict(dict)
            for result in local_results:
                results_by_decoding[result.decoding][result.variant] = result
            for decoding_name, variant_results in results_by_decoding.items():
                st.markdown(f"##### `{decoding_name}`")
                base_output_column, finetuned_output_column = st.columns(2)
                for output_column, variant in (
                    (base_output_column, "Base"),
                    (finetuned_output_column, "Finetuned"),
                ):
                    result = variant_results.get(variant)
                    output_column.markdown(f"**{variant} model**")
                    if result is None:
                        output_column.warning("No result")
                    elif result.error:
                        output_column.error(result.error)
                    else:
                        output_column.text(result.output or "—")
                        output_column.caption(
                            f"{result.output_tokens} tokens · "
                            f"{result.latency_seconds:.2f} seconds"
                        )
            st.download_button(
                "Download local comparison as CSV",
                local_comparison_csv(local_results),
                file_name="local_base_vs_finetuned.csv",
                mime="text/csv",
            )

    with inference_tab:
        st.markdown(
            "Compare the same Nepali prompt across the local finetuned GPT-2 and "
            "TinyLlama adapters, Gemini, Google-hosted Gemma, and optional Hugging "
            "Face Inference Provider models. Hosted API calls may be billed."
        )
        inference_text_columns = text_columns(inventory["schema"])
        reference_sentiment_column = sentiment_column(inventory["schema"])
        task_options = (
            "Sentiment classification",
            "Summarization",
            "Custom instruction",
        )
        task_preset = st.selectbox(
            "Task preset",
            task_options,
            index=0 if reference_sentiment_column else 1,
            help=(
                "Sentiment classification asks for exactly one of: negative, "
                "neutral, or positive. Dataset labels are shown only as references."
            ),
        )
        prompt_source = st.radio(
            "Prompt source",
            ("Selected dataset sample", "Custom source text", "Prompt only"),
            horizontal=True,
        )

        source_text: str | None = ""
        reference_sentiment: str | None = None
        reference_sentiment_value: Any = None
        source_widget_context = "custom"
        if prompt_source == "Selected dataset sample":
            if not inference_text_columns:
                st.warning(
                    "This dataset has no detectable natural-language column. "
                    "Choose Custom source text, Prompt only, or a pre-tokenization dataset."
                )
            else:
                source_controls = st.columns([2, 1, 1])
                inference_column = source_controls[0].selectbox(
                    "Dataset text column",
                    inference_text_columns,
                    key=f"inference-column:{spec.key}",
                )
                inference_sample_seed = source_controls[1].number_input(
                    "Sample seed",
                    min_value=0,
                    value=42,
                    step=1,
                )
                maximum_prompt_chars = source_controls[2].number_input(
                    "Maximum source characters",
                    min_value=50,
                    max_value=20_000,
                    value=1_000,
                    step=50,
                )
                inference_sample_columns = [inference_column]
                if (
                    reference_sentiment_column
                    and reference_sentiment_column != inference_column
                ):
                    inference_sample_columns.append(reference_sentiment_column)
                inference_records = cached_sample(
                    inventory,
                    1,
                    int(inference_sample_seed),
                    tuple(inference_sample_columns),
                )
                if inference_records:
                    inference_record = inference_records[0]
                    extracted = extract_text(inference_record.get(inference_column))
                    source_text = "\n\n".join(extracted)[: int(maximum_prompt_chars)]
                    if reference_sentiment_column:
                        reference_sentiment_value = inference_record.get(
                            reference_sentiment_column
                        )
                        reference_sentiment = sentiment_label(
                            reference_sentiment_value
                        )
                    source_widget_context = (
                        f"{inference_column}:{int(inference_sample_seed)}:"
                        f"{int(maximum_prompt_chars)}"
                    )
        elif prompt_source == "Custom source text":
            source_text = "नेपाल एउटा विविध भाषा र संस्कृतिले भरिएको देश हो।"
        else:
            source_text = None

        editable_source: str | None = source_text
        if prompt_source != "Prompt only":
            editable_source = st.text_area(
                "Source text",
                value=source_text or "",
                height=180,
                key=f"inference-source:{spec.key}:{prompt_source}:{source_widget_context}",
            )
        if reference_sentiment:
            st.info(
                f"Dataset reference sentiment: **{reference_sentiment}** "
                f"(raw label: `{reference_sentiment_value}`). This label is not "
                "included in the model prompt."
            )

        sentiment_system_prompt = (
            "तपाईं नेपाली पाठको भावना वर्गीकरण गर्ने सहायक हुनुहुन्छ। "
            "उत्तरमा negative, neutral, वा positive मध्ये ठीक एउटा अंग्रेजी "
            "लेबल मात्र दिनुहोस्; व्याख्या नदिनुहोस्।"
        )
        system_prompt = st.text_area(
            "System prompt (optional)",
            value=sentiment_system_prompt if task_preset == "Sentiment classification" else "",
            height=90,
            key=f"system-prompt:{task_preset}",
            help=(
                "Sets model behavior independently from the user prompt. "
                "It is sent through the provider's system-instruction mechanism."
            ),
        )
        if prompt_source == "Prompt only":
            default_user_prompt = (
                "यो नेपाली वाक्यको भावना वर्गीकरण गर्नुहोस्: आजको खबरले मलाई खुसी बनायो।"
                if task_preset == "Sentiment classification"
                else "नेपाली भाषामा नेपालको सांस्कृतिक विविधताबारे छोटो अनुच्छेद लेख्नुहोस्।"
            )
        elif task_preset == "Sentiment classification":
            default_user_prompt = "निम्न नेपाली पाठको भावना वर्गीकरण गर्नुहोस्:\n\n{text}"
        elif task_preset == "Summarization":
            default_user_prompt = "निम्न पाठलाई नेपालीमा संक्षेप गर्नुहोस्:\n\n{text}"
        else:
            default_user_prompt = "{text}"
        prompt_template = st.text_area(
            "User prompt" if prompt_source == "Prompt only" else "User prompt template",
            value=default_user_prompt,
            height=110,
            key=f"user-prompt:{task_preset}:{prompt_source}",
            help=(
                "Use {text} where the source document should be inserted."
                if prompt_source != "Prompt only"
                else "This prompt is sent directly without a source document."
            ),
        )
        try:
            prompt_preview = build_prompt(prompt_template, editable_source)
        except ComparisonConfigurationError as error:
            st.error(str(error))
            prompt_preview = ""
        if prompt_preview:
            with st.expander("Final prompt preview"):
                st.text(prompt_preview)

        comparison_local_adapters = discover_local_adapters(
            DEFAULT_FINETUNED_MODELS_ROOT
        )
        local_backend_specs = {
            f"Local · {adapter.label}": adapter
            for adapter in comparison_local_adapters
        }
        backend_names = st.multiselect(
            "Inference backends",
            (
                "Gemini API",
                "Gemini Flash Lite API",
                "Google Gemma API",
                "Hugging Face Inference API",
                HIMALAYAGPT_BACKEND_NAME,
                ARKIOS_BACKEND_NAME,
                *local_backend_specs,
            ),
            default=(),
            help=(
                "Select hosted APIs and/or local finetuned adapters. Only selected "
                "models run."
            ),
        )
        if not backend_names:
            st.caption("Select one or more inference backends before running.")
        gemini_model = DEFAULT_GEMINI_MODEL
        gemini_flash_lite_model = DEFAULT_GEMINI_FLASH_LITE_MODEL
        google_gemma_model = DEFAULT_GOOGLE_GEMMA_MODEL
        hf_models_text = ""
        hf_provider = "auto"
        selected_comparison_local_specs = [
            local_backend_specs[name]
            for name in backend_names
            if name in local_backend_specs
        ]
        comparison_local_device = "auto"
        comparison_local_dtype = "auto"
        comparison_local_files_only = False
        selected_arkios = ARKIOS_BACKEND_NAME in backend_names
        selected_himalayagpt = HIMALAYAGPT_BACKEND_NAME in backend_names
        selected_full_local_models = int(selected_arkios) + int(
            selected_himalayagpt
        )
        if backend_names:
            backend_columns = st.columns(len(backend_names))
            for backend_column, backend_name in zip(backend_columns, backend_names):
                with backend_column:
                    if backend_name == "Gemini API":
                        gemini_model = st.text_input(
                            "Gemini model", DEFAULT_GEMINI_MODEL
                        )
                        gemini_env_available = bool(os.getenv("GEMINI_API_KEY"))
                        st.caption(
                            "GEMINI_API_KEY detected in environment"
                            if gemini_env_available
                            else "GEMINI_API_KEY is not available to this process"
                        )
                    elif backend_name == "Gemini Flash Lite API":
                        gemini_flash_lite_model = st.text_input(
                            "Gemini Flash Lite model",
                            DEFAULT_GEMINI_FLASH_LITE_MODEL,
                        )
                        st.caption("Uses GEMINI_API_KEY through the Interactions API")
                    elif backend_name == "Google Gemma API":
                        google_gemma_model = st.text_input(
                            "Google Gemma model", DEFAULT_GOOGLE_GEMMA_MODEL
                        )
                        st.caption(
                            "Uses GEMINI_API_KEY through Gemma's generate_content endpoint"
                        )
                    elif backend_name == "Hugging Face Inference API":
                        hf_models_text = st.text_area(
                            "Hugging Face model IDs",
                            "",
                            height=70,
                            placeholder="account/model-name",
                            help=(
                                "Enter one Hub model ID per line. Each model must "
                                "have coverage from the selected Inference Provider."
                            ),
                        )
                        hf_provider = st.text_input("Inference Provider", "auto")
                        hf_env_available = bool(os.getenv("HF_TOKEN"))
                        st.caption(
                            "HF_TOKEN detected in environment"
                            if hf_env_available
                            else "HF_TOKEN is not available to this process"
                        )
                    elif backend_name in local_backend_specs:
                        local_spec = local_backend_specs[backend_name]
                        st.caption(
                            f"Base: `{local_spec.base_model_id}`  \n"
                            f"Adapter: `{local_spec.path.name}`"
                        )
                    elif backend_name == ARKIOS_BACKEND_NAME:
                        st.caption(
                            f"Model: `{DEFAULT_ARKIOS_MODEL_ID}`  \n"
                            "Published ChatML template · 4,096-token context"
                        )
                    elif backend_name == HIMALAYAGPT_BACKEND_NAME:
                        st.caption(
                            f"Model: `{DEFAULT_HIMALAYAGPT_MODEL_ID}`  \n"
                            "Pinned custom code · 2,048-token context"
                        )
            if selected_comparison_local_specs or selected_full_local_models:
                st.markdown("##### Local model runtime")
                comparison_runtime_columns = st.columns(3)
                comparison_local_device = comparison_runtime_columns[0].selectbox(
                    "Local device",
                    ("auto", "cuda", "cpu"),
                    key="comparison-local-device",
                )
                comparison_dtype_options = (
                    ("auto", "float32", "bfloat16")
                    if selected_himalayagpt
                    else ("auto", "float32", "bfloat16", "float16")
                )
                comparison_local_dtype = comparison_runtime_columns[1].selectbox(
                    "Local weight dtype",
                    comparison_dtype_options,
                    key="comparison-local-dtype",
                )
                comparison_local_files_only = comparison_runtime_columns[2].checkbox(
                    "Cached base weights only",
                    value=False,
                    key="comparison-local-files-only",
                )

        st.markdown("#### Decoding grid")
        decoding_profile = "himalayagpt" if selected_himalayagpt else "general"
        default_temperatures = "0.8" if selected_himalayagpt else "0.2, 0.7"
        default_top_ps = "1.0" if selected_himalayagpt else "0.95"
        default_top_ks = "50" if selected_himalayagpt else "40"
        default_max_tokens = 96 if selected_himalayagpt else 256
        st.caption(
            "Every temperature × top-p × top-k combination runs against every "
            "selected model. Hugging Face top-k support is provider-dependent."
        )
        if selected_himalayagpt:
            st.caption(
                "HimalayaGPT reference profile: temperature 0.8, top-k 50, "
                "maximum 96 tokens, repetition penalty 1.08, and special-token "
                "stopping. Its reference generation loop does not apply top-p."
            )
        decoding_columns = st.columns(6)
        temperatures_text = decoding_columns[0].text_input(
            "Temperatures",
            default_temperatures,
            key=f"comparison-temperatures:{decoding_profile}",
        )
        top_ps_text = decoding_columns[1].text_input(
            "Top-p values",
            default_top_ps,
            key=f"comparison-top-ps:{decoding_profile}",
        )
        top_ks_text = decoding_columns[2].text_input(
            "Top-k values",
            default_top_ks,
            key=f"comparison-top-ks:{decoding_profile}",
        )
        inference_max_tokens = decoding_columns[3].number_input(
            "Maximum new tokens",
            min_value=1,
            max_value=4_096,
            value=default_max_tokens,
            key=f"comparison-max-new-tokens:{decoding_profile}",
        )
        inference_seed = decoding_columns[4].number_input(
            "Generation seed", min_value=0, value=42
        )
        inference_thinking_level = decoding_columns[5].selectbox(
            "Thinking level",
            ("minimal", "low", "medium", "high"),
            index=0,
            help=(
                "Google thinking tokens share the maximum-output budget. Use "
                "minimal for short generation limits and comparison tasks."
            ),
        )

        configs = []
        grid_error = None
        try:
            temperatures = parse_number_list(
                temperatures_text, value_type=float, name="temperatures"
            )
            top_ps = parse_number_list(top_ps_text, value_type=float, name="top-p")
            top_ks = parse_number_list(top_ks_text, value_type=int, name="top-k")
            configs = build_decoding_grid(
                temperatures,
                top_ps,
                top_ks,
                max_new_tokens=int(inference_max_tokens),
                seed=int(inference_seed),
                thinking_level=inference_thinking_level,
            )
        except ComparisonConfigurationError as error:
            grid_error = str(error)
            st.error(grid_error)

        hf_model_ids = parse_model_ids(hf_models_text)
        hosted_model_count = (
            int("Gemini API" in backend_names)
            + int("Gemini Flash Lite API" in backend_names)
            + int("Google Gemma API" in backend_names)
            + (
                len(hf_model_ids)
                if "Hugging Face Inference API" in backend_names
                else 0
            )
        )
        local_model_count = (
            len(selected_comparison_local_specs) + selected_full_local_models
        )
        model_count = hosted_model_count + local_model_count
        request_count = model_count * len(configs)
        remote_request_count = hosted_model_count * len(configs)
        local_comparison_count = local_model_count * len(configs)
        st.info(
            f"This comparison will run {request_count} generation(s): "
            f"{remote_request_count} hosted request(s) and "
            f"{local_comparison_count} local generation(s)."
        )
        if request_count > 60:
            st.error("Reduce the grid to at most 60 generations per comparison.")

        local_context_error = ""
        for local_spec in selected_comparison_local_specs:
            context_limit = 1_024 if local_spec.base_model_id == "gpt2" else 2_048
            if int(inference_max_tokens) >= context_limit:
                local_context_error = (
                    f"Maximum new tokens must be below {context_limit:,} for "
                    f"{local_spec.label}."
                )
                st.error(local_context_error)
                break
        if (
            not local_context_error
            and selected_himalayagpt
            and int(inference_max_tokens) >= 2_048
        ):
            local_context_error = (
                "Maximum new tokens must be below 2,048 for HimalayaGPT 0.5B."
            )
            st.error(local_context_error)
        if (
            not local_context_error
            and selected_arkios
            and int(inference_max_tokens) >= 4_096
        ):
            local_context_error = (
                "Maximum new tokens must be below 4,096 for Arkios 1B Chat."
            )
            st.error(local_context_error)

        result_context = hashlib.sha256(
            (
                f"{task_preset}\0{prompt_preview}\0{system_prompt}\0"
                f"{backend_names}\0{gemini_model}\0{gemini_flash_lite_model}\0"
                f"{google_gemma_model}\0{hf_model_ids}\0{hf_provider}\0"
                f"{configs}\0{comparison_local_device}\0"
                f"{comparison_local_dtype}\0{comparison_local_files_only}"
            ).encode("utf-8")
        ).hexdigest()[:16]
        results_key = f"inference-results:{spec.key}:{result_context}"
        run_disabled = (
            request_count <= 0
            or request_count > 60
            or bool(grid_error)
            or bool(local_context_error)
            or not prompt_preview
        )
        if st.button(
            "Run inference comparison",
            type="primary",
            disabled=run_disabled,
        ):
            effective_gemini_key = os.getenv("GEMINI_API_KEY", "")
            effective_hf_token = os.getenv("HF_TOKEN", "")
            try:
                if grid_error:
                    raise ComparisonConfigurationError(grid_error)
                if not prompt_preview:
                    raise ComparisonConfigurationError("provide a valid source prompt")
                if request_count <= 0:
                    raise ComparisonConfigurationError("select at least one model")
                if request_count > 60:
                    raise ComparisonConfigurationError(
                        "comparison exceeds the 60-generation safety limit"
                    )
                if local_context_error:
                    raise ComparisonConfigurationError(local_context_error)

                backends = []
                if {
                    "Gemini API",
                    "Gemini Flash Lite API",
                    "Google Gemma API",
                }.intersection(backend_names):
                    if not effective_gemini_key:
                        raise ComparisonConfigurationError(
                            "GEMINI_API_KEY is missing for Google inference"
                        )
                if "Gemini API" in backend_names:
                    backends.append(
                        cached_google_backend(
                            gemini_model,
                            secret_fingerprint(effective_gemini_key),
                            effective_gemini_key,
                        )
                    )
                if "Gemini Flash Lite API" in backend_names:
                    backends.append(
                        cached_google_backend(
                            gemini_flash_lite_model,
                            secret_fingerprint(effective_gemini_key),
                            effective_gemini_key,
                        )
                    )
                if "Google Gemma API" in backend_names:
                    backends.append(
                        cached_google_backend(
                            google_gemma_model,
                            secret_fingerprint(effective_gemini_key),
                            effective_gemini_key,
                        )
                    )
                if "Hugging Face Inference API" in backend_names:
                    if not effective_hf_token:
                        raise ComparisonConfigurationError("Hugging Face token is missing")
                    for model_id in hf_model_ids:
                        backends.append(
                            cached_huggingface_backend(
                                model_id,
                                hf_provider,
                                secret_fingerprint(effective_hf_token),
                                effective_hf_token,
                            )
                        )
                for local_spec in selected_comparison_local_specs:
                    local_bundle = cached_local_model_pair(
                        local_spec.key,
                        local_spec.label,
                        str(local_spec.path),
                        local_spec.base_model_id,
                        comparison_local_device,
                        comparison_local_dtype,
                        comparison_local_files_only,
                    )
                    backends.append(LocalPeftBackend(local_bundle))
                if selected_himalayagpt:
                    backends.append(
                        HimalayaGPTBackend(
                            cached_himalayagpt(
                                DEFAULT_HIMALAYAGPT_MODEL_ID,
                                DEFAULT_HIMALAYAGPT_REVISION,
                                comparison_local_device,
                                comparison_local_dtype,
                                comparison_local_files_only,
                            )
                        )
                    )
                if selected_arkios:
                    backends.append(
                        ArkiosBackend(
                            cached_arkios(
                                DEFAULT_ARKIOS_MODEL_ID,
                                DEFAULT_ARKIOS_REVISION,
                                comparison_local_device,
                                comparison_local_dtype,
                                comparison_local_files_only,
                            )
                        )
                    )

                with st.spinner(f"Running {request_count} model generations…"):
                    st.session_state[results_key] = run_comparison(
                        backends,
                        [editable_source],
                        configs,
                        prompt_template=prompt_template,
                        system_prompt=system_prompt,
                    )
            except (
                ComparisonConfigurationError,
                LocalInferenceError,
                ImportError,
                RuntimeError,
                OSError,
            ) as error:
                st.error(f"Could not start comparison: {error}")

        comparison_results = st.session_state.get(results_key, [])
        if comparison_results:
            st.markdown("#### Comparison results")
            summary_rows = [
                {
                    "model": result.model,
                    "decoding": result.decoding,
                    "latency_seconds": result.latency_seconds,
                    "status": "error" if result.error else "ok",
                    "prediction": sentiment_prediction(result.output)
                    if task_preset == "Sentiment classification"
                    else None,
                    "reference": reference_sentiment,
                    "matches_reference": (
                        sentiment_prediction(result.output) == reference_sentiment
                        if reference_sentiment and not result.error
                        else None
                    ),
                    "output_preview": (result.output or result.error or "")[:300],
                }
                for result in comparison_results
            ]
            st.dataframe(summary_rows, use_container_width=True, hide_index=True)
            selected_result = st.selectbox(
                "Inspect complete output",
                range(len(comparison_results)),
                format_func=lambda index: (
                    f"{comparison_results[index].model} · "
                    f"{comparison_results[index].decoding}"
                ),
            )
            result = comparison_results[selected_result]
            if result.error:
                st.error(result.error)
            else:
                st.text(result.output)
            st.download_button(
                "Download results as CSV",
                comparison_csv(comparison_results),
                file_name="nepali_inference_comparison.csv",
                mime="text/csv",
            )

    with manifest_tab:
        manifest_path = find_manifest(spec, data_root)
        if manifest_path is None:
            st.info("No associated manifest was found.")
        else:
            st.caption(str(manifest_path))
            try:
                st.json(json.loads(manifest_path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError) as error:
                st.error(f"Could not read manifest: {error}")


if __name__ == "__main__":
    run_app()
