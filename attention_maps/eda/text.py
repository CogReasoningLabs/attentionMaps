"""Schema extraction and Unicode-aware text metrics."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


TOKEN_PATTERN = re.compile(
    r"[\u0900-\u0963\u0971-\u097F]+|[A-Za-z]+(?:['’][A-Za-z]+)?|\d+(?:[.,]\d+)*"
)
SENTENCE_BOUNDARY = re.compile(r"[।॥!?]+|(?<!\d)\.(?!\d)|\n+")
NON_DEVANAGARI_PATTERN = re.compile(r"[^\u0900-\u097F\s]")
NEPALI_SUFFIXES = (
    "हरूबाट",
    "हरुबाट",
    "हरूसँग",
    "हरुसँग",
    "हरूलाई",
    "हरुलाई",
    "बाट",
    "सँग",
    "सम्म",
    "लाई",
    "देखि",
    "को",
    "का",
    "की",
    "ले",
    "मा",
)
TEXT_CANDIDATES = (
    "text",
    "content",
    "document",
    "article",
    "sentence",
    "body",
    "messages",
    "conversations",
)
SOURCE_CANDIDATES = (
    "source",
    "dataset",
    "domain",
    "url",
    "link",
    "metadata.source",
    "metadata.url",
)


def normalize_text(text: str) -> str:
    """Apply canonical Unicode normalization and collapse whitespace."""

    return " ".join(unicodedata.normalize("NFC", text).replace("\ufeff", " ").split())


def normalize_structured_text(text: str) -> str:
    """Normalize Unicode and spaces while retaining non-empty line boundaries."""

    normalized = unicodedata.normalize("NFC", text).replace("\ufeff", " ")
    return "\n".join(
        line
        for line in (" ".join(line.split()) for line in normalized.splitlines())
        if line
    )


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFC", text).replace("\ufeff", " ")
    return [token.casefold() for token in TOKEN_PATTERN.findall(normalized)]


def clean_devanagari_text(text: str) -> str:
    """Retain only Devanagari-block characters and whitespace after NFC."""

    normalized = unicodedata.normalize("NFC", text).replace("\ufeff", " ")
    return NON_DEVANAGARI_PATTERN.sub(" ", normalized)


def strip_nepali_suffix(token: str, *, minimum_stem_length: int = 4) -> str:
    """Remove one common attached postposition from a normalized Nepali token."""

    normalized = unicodedata.normalize("NFC", token).replace("\ufeff", "").casefold()
    for suffix in NEPALI_SUFFIXES:
        if normalized.endswith(suffix):
            stem = normalized[: -len(suffix)]
            if len(stem) >= minimum_stem_length:
                return stem
    return normalized


def parse_stopword_text(text: str) -> tuple[str, ...]:
    """Parse comma- or whitespace-separated stopwords with stable ordering."""

    normalized = unicodedata.normalize("NFC", text).replace("\ufeff", " ")
    return tuple(dict.fromkeys(tokenize(normalized.replace(",", " "))))


def load_stopwords_file(path: Path) -> tuple[str, ...]:
    """Load a UTF-8 stopword resource with BOM-safe Unicode normalization."""

    try:
        content = path.read_text(encoding="utf-8-sig")
    except OSError as error:
        raise ValueError(f"could not read stopword file {path}: {error}") from error
    return parse_stopword_text(content)


def sentence_token_lengths(text: str) -> list[int]:
    """Return token counts for non-empty sentence-like segments."""

    return [
        len(tokens)
        for segment in SENTENCE_BOUNDARY.split(text)
        if (tokens := tokenize(segment))
    ]


def line_character_lengths(text: str) -> list[int]:
    """Return normalized character counts for non-empty physical lines."""

    return [
        len(line)
        for raw_line in text.splitlines() or [text]
        if (line := normalize_text(raw_line))
    ]


def token_ngrams(tokens: Sequence[str], order: int) -> list[str]:
    """Return whitespace-joined token n-grams of the requested order."""

    return [
        " ".join(tokens[index : index + order])
        for index in range(len(tokens) - order + 1)
    ]


def devanagari_ratio(text: str) -> float:
    """Share of Unicode letters/marks that belong to the Devanagari block."""

    script_characters = [
        character
        for character in text
        if unicodedata.category(character).startswith(("L", "M"))
    ]
    if not script_characters:
        return 0.0
    devanagari = sum(
        "\u0900" <= character <= "\u097f" for character in script_characters
    )
    return devanagari / len(script_characters)


def extract_text(record: Mapping[str, Any], columns: Sequence[str] = ()) -> str:
    """Extract plain, nested, conversational, or instruction/response text."""

    if columns:
        return _join_values(_path_value(record, column) for column in columns)

    for column in ("messages", "conversations"):
        text = _flatten_text(record.get(column))
        if text:
            return text

    instruction_parts = [
        record.get(column)
        for column in ("instruction", "input", "prompt", "output", "response", "answer")
        if record.get(column) not in (None, "")
    ]
    if instruction_parts and any(
        column in record for column in ("instruction", "prompt")
    ):
        return _join_values(instruction_parts)

    for column in TEXT_CANDIDATES:
        text = _flatten_text(record.get(column))
        if text:
            return text
    return ""


def extract_source(record: Mapping[str, Any], columns: Sequence[str] = ()) -> str:
    candidates = columns or SOURCE_CANDIDATES
    for column in candidates:
        value = _path_value(record, column)
        if value in (None, ""):
            continue
        text = _flatten_text(value)
        if not text:
            continue
        if column.rsplit(".", 1)[-1] in {"url", "link"}:
            domain = urlparse(text).netloc
            if domain:
                return domain.lower().removeprefix("www.")
        return normalize_text(text)[:300]
    return ""


def _path_value(record: Mapping[str, Any], path: str) -> Any:
    value: Any = record
    for part in path.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def _join_values(values: Any) -> str:
    return "\n".join(filter(None, (_flatten_text(value) for value in values)))


def _flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return normalize_structured_text(value)
    if isinstance(value, Mapping):
        for key in ("content", "text", "value"):
            if key in value:
                return _flatten_text(value[key])
        return ""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _join_values(value)
    return ""
