"""Record cleaning, schema rendering, and public export serialization."""

from __future__ import annotations

import csv
import io
import json
import unicodedata
from typing import Any

from attention_maps.inference.comparison import ComparisonConfigurationError
from scripts.utils.nepali_text import CleaningConfig, clean_text_with_result, normalize_whitespace

def prepare_source(text: str, cleaning: CleaningConfig | None) -> str:
    """Reuse the project's canonical Nepali cleaner when it is enabled."""

    if contains_corrupted_text(text):
        return ""
    if cleaning is None:
        return normalize_whitespace(text)
    result = clean_text_with_result(text, cleaning)
    return result.text if result.accepted else ""


def contains_corrupted_text(text: str) -> bool:
    """Identify replacement glyphs and binary control bytes masquerading as text."""

    allowed_controls = {"\n", "\t", "\u200c", "\u200d"}
    return "\ufffd" in text or any(
        unicodedata.category(character).startswith("C")
        and character not in allowed_controls
        for character in text
    )


def parse_output_schema(value: str) -> dict[str, str]:
    """Parse a flat output-record schema whose values are format templates."""

    try:
        schema = json.loads(value)
    except json.JSONDecodeError as error:
        raise ComparisonConfigurationError(f"output schema must be valid JSON: {error}") from error
    if not isinstance(schema, dict) or not schema:
        raise ComparisonConfigurationError("output schema must be a non-empty JSON object")
    if not all(
        isinstance(key, str)
        and key.strip()
        and isinstance(template, str)
        and template.strip()
        for key, template in schema.items()
    ):
        raise ComparisonConfigurationError(
            "each output-schema field and its template must be a non-empty string"
        )
    return schema


def dataset_records_for_export(
    generated_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Return only successful user-schema records from internal result rows."""

    return [
        dict(row["_dataset_record"])
        for row in generated_rows
        if row.get("_status") == "ok"
        and isinstance(row.get("_dataset_record"), dict)
    ]


def records_jsonl(records: list[dict[str, str]]) -> str:
    """Serialize public dataset records without internal generation metadata."""

    if not records:
        return ""
    return "\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n"


def records_csv(records: list[dict[str, str]]) -> str:
    """Serialize records to CSV while preserving schema field order."""

    if not records:
        return ""
    fieldnames = list(dict.fromkeys(field for record in records for field in record))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)
    return buffer.getvalue()


def result_preview_rows(
    generated_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build a readable UI table while retaining provider diagnostics."""

    return [
        {
            **dict(row.get("_dataset_record", {})),
            "status": row.get("_status", ""),
            "error": row.get("_error", ""),
            "latency_seconds": row.get("_latency_seconds", 0.0),
        }
        for row in generated_rows
    ]


def record_from_schema(
    schema: dict[str, str], *, source: str, output: str, prompt: str,
    system_prompt: str, model: str, decoding: str, row_index: object,
) -> dict[str, str]:
    values = {
        "source": source, "output": output, "prompt": prompt,
        "system_prompt": system_prompt, "model": model,
        "decoding": decoding, "row_index": str(row_index),
    }
    try:
        return {field: template.format(**values) for field, template in schema.items()}
    except (KeyError, ValueError) as error:
        raise ComparisonConfigurationError(
            f"invalid output-schema placeholder: {error}. Use source, output, prompt, "
            "system_prompt, model, decoding, or row_index."
        ) from error
