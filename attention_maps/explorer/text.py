"""Text extraction, analysis-input parsing, and word-cloud utilities."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from attention_maps.eda.contracts import DatasetSpec as EDADatasetSpec
from attention_maps.eda.text import clean_devanagari_text, strip_nepali_suffix
from attention_maps.inference.comparison import ComparisonConfigurationError

from .catalog import (
    DEVANAGARI_FONT_CANDIDATES,
    LATIN_FONT_CANDIDATES,
    SENTIMENT_LABELS,
    TEXT_FIELD_NAMES,
    VIEWER_PREFIX,
    DatasetSpec,
)

def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def format_decimal_bytes(size: int) -> str:
    """Format storage bytes with the decimal units used by the Hugging Face UI."""

    value = float(size)
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if value < 1000 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1000
    return f"{value:.1f} TB"


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

    string_columns = [
        field["column"]
        for field in schema
        if "string" in field["type"].lower()
        and not field["column"].startswith(VIEWER_PREFIX)
    ]
    actual_by_casefold = {column.casefold(): column for column in string_columns}
    preferred = list(
        dict.fromkeys(
            actual_by_casefold[name.casefold()]
            for name in TEXT_FIELD_NAMES
            if name.casefold() in actual_by_casefold
        )
    )
    metadata_names = {
        "id",
        "uuid",
        "guid",
        "index",
        "row_id",
        "record_id",
        "label",
        "class",
        "category",
        "sentiment",
        "source",
        "url",
        "domain",
        "language",
        "language_code",
    }
    other_strings = [
        column
        for column in string_columns
        if column not in preferred and column.casefold() not in metadata_names
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
    match = re.search(
        r"(?<![a-z])(negative|neutral|positive)(?![a-z])", text.casefold()
    )
    return match.group(1) if match else None


def extract_text(value: Any) -> list[str]:
    """Extract text from strings or common nested chat/instruction values."""

    if isinstance(value, str):
        return [unicodedata.normalize("NFC", value).replace("\ufeff", "")]
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
            "value",
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

    normalized = unicodedata.normalize("NFC", value).replace("\ufeff", " ")
    return {
        word
        for chunk in normalized.replace(",", " ").split()
        for word in unicode_words(chunk, include_numbers=True)
    }


def parse_eda_terms(value: str) -> tuple[str, ...]:
    """Parse ordered, unique seed tokens for the co-occurrence network."""

    return tuple(
        dict.fromkeys(
            word
            for chunk in value.replace(",", " ").split()
            for word in unicode_words(chunk, include_numbers=True)
        )
    )


def eda_dataset_spec(
    spec: DatasetSpec,
    text_fields: Sequence[str],
    source_fields: Sequence[str],
    sample_size: int,
    population_rows: int | None = None,
) -> EDADatasetSpec:
    """Convert a viewer dataset into the stable EDA pipeline contract."""

    normalized_key = re.sub(r"[^A-Za-z0-9._-]+", "-", spec.key).strip("-.")
    digest = hashlib.sha256(spec.key.encode("utf-8")).hexdigest()[:10]
    key = f"{(normalized_key or 'dataset')[:50]}-{digest}"
    return EDADatasetSpec(
        key=key,
        dataset_id=spec.dataset_id or f"local/{key}",
        config_name=spec.dataset_config,
        split=spec.dataset_split or "local",
        text_columns=tuple(text_fields),
        source_columns=tuple(source_fields),
        sample_size=sample_size,
        population_rows=population_rows,
    )


def word_frequencies(
    records: Sequence[dict[str, Any]],
    columns: Sequence[str],
    *,
    stopwords: set[str] | None = None,
    min_characters: int = 2,
    min_frequency: int = 1,
    include_numbers: bool = False,
    devanagari_only: bool = False,
    strip_nepali_suffixes: bool = False,
) -> Counter[str]:
    """Count Unicode words in selected columns of sampled dataset records."""

    excluded = {
        unicodedata.normalize("NFC", word).replace("\ufeff", "").strip().casefold()
        for word in (stopwords or set())
        if word.strip().replace("\ufeff", "")
    }
    counts: Counter[str] = Counter()
    for record in records:
        for column in columns:
            for text in extract_text(record.get(column)):
                prepared = clean_devanagari_text(text) if devanagari_only else text
                for word in unicode_words(prepared, include_numbers=include_numbers):
                    normalized_word = unicodedata.normalize("NFC", word).casefold()
                    if normalized_word in excluded:
                        continue
                    if strip_nepali_suffixes:
                        normalized_word = strip_nepali_suffix(normalized_word)
                    if (
                        len(normalized_word) >= min_characters
                        and normalized_word not in excluded
                    ):
                        counts[normalized_word] += 1
    return Counter(
        {word: count for word, count in counts.items() if count >= min_frequency}
    )


def find_devanagari_font() -> Path | None:
    """Return the first commonly installed font capable of rendering Nepali."""

    return next(
        (
            path
            for path in DEVANAGARI_FONT_CANDIDATES
            if path.is_file() and font_supports_devanagari(path)
        ),
        None,
    )


def font_supports_devanagari(path: Path) -> bool:
    """Verify that a font maps representative Nepali base and combining glyphs."""

    try:
        from matplotlib import ft2font

        characters = ft2font.FT2Font(str(path)).get_charmap()
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        return False
    return all(ord(character) in characters for character in "कसम्बन्धी")


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

    return [
        item.strip() for item in value.replace(",", "\n").splitlines() if item.strip()
    ]


def secret_fingerprint(secret: str) -> str:
    """Provide cache invalidation without using a secret as a visible cache key."""

    return hashlib.sha256(secret.encode("utf-8")).hexdigest()
