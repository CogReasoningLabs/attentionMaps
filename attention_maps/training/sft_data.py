"""Schema-flexible normalization for supervised instruction-tuning datasets."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from scripts.utils.nepali_text import CleaningConfig, clean_text_with_result


CanonicalExample = dict[str, Any]
SchemaAdapter = Callable[
    [Mapping[str, Any]],
    tuple[list[dict[str, str]], list[dict[str, str]]],
]

_TAGGED_DIALOGUE = re.compile(
    r"^\s*HUMAN\s*:\s*(?P<user>.*?)\s*^\s*ASSISTANT\s*:\s*(?P<assistant>.*)\s*$",
    flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
_ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "instruction": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "bot": "assistant",
    "model": "assistant",
    "system": "system",
}


def clean_scalar(value: Any) -> str:
    """Convert a scalar to text while treating common missing values as empty."""

    if value is None:
        return ""
    try:
        if bool(math.isnan(value)):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _message(role: str, content: Any) -> dict[str, str] | None:
    normalized_role = _ROLE_ALIASES.get(clean_scalar(role).lower())
    normalized_content = clean_scalar(content)
    if normalized_role is None or not normalized_content:
        return None
    return {"role": normalized_role, "content": normalized_content}


def _normalize_messages(value: Any) -> list[dict[str, str]]:
    if isinstance(value, str):
        return [{"role": "user", "content": value.strip()}] if value.strip() else []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    normalized = []
    for index, item in enumerate(value):
        if isinstance(item, Mapping):
            role = item.get("role", item.get("from", ""))
            content = item.get("content", item.get("value", item.get("text", "")))
        else:
            role = "user" if index % 2 == 0 else "assistant"
            content = item
        message = _message(str(role), content)
        if message is not None:
            normalized.append(message)
    return normalized


def _split_messages(
    messages: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    assistant_positions = [
        index for index, message in enumerate(messages) if message["role"] == "assistant"
    ]
    if not assistant_positions:
        return [], []
    completion_index = assistant_positions[-1]
    prompt = messages[:completion_index]
    completion = [messages[completion_index]]
    if not any(message["role"] == "user" for message in prompt):
        return [], []
    return prompt, completion


def _alpaca_adapter(
    example: Mapping[str, Any],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    instruction = clean_scalar(example.get("instruction"))
    additional_input = clean_scalar(example.get("input"))
    output = clean_scalar(example.get("output"))
    if additional_input:
        instruction = f"{instruction}\n\nथप जानकारी:\n{additional_input}"
    if not instruction or not output:
        return [], []
    return (
        [{"role": "user", "content": instruction}],
        [{"role": "assistant", "content": output}],
    )


def _lima_adapter(
    example: Mapping[str, Any],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if clean_scalar(example.get("status")).lower() not in {"", "success"}:
        return [], []
    tagged = clean_scalar(example.get("translation", example.get("source_text")))
    match = _TAGGED_DIALOGUE.match(tagged)
    if match is None:
        return [], []
    return (
        [{"role": "user", "content": match.group("user").strip()}],
        [{"role": "assistant", "content": match.group("assistant").strip()}],
    )


def _prompt_completion_adapter(
    example: Mapping[str, Any],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    prompt = _normalize_messages(example.get("prompt"))
    completion = _normalize_messages(example.get("completion"))
    if completion and completion[0]["role"] == "user":
        completion[0]["role"] = "assistant"
    return prompt, completion[:1]


def _chat_adapter(
    example: Mapping[str, Any],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    messages = example.get("messages", example.get("conversations"))
    return _split_messages(_normalize_messages(messages))


SCHEMA_ADAPTERS: dict[str, SchemaAdapter] = {
    "alpaca": _alpaca_adapter,
    "lima": _lima_adapter,
    "prompt_completion": _prompt_completion_adapter,
    "chat": _chat_adapter,
}


def register_sft_schema(name: str, adapter: SchemaAdapter) -> None:
    """Register an adapter without changing the normalization pipeline."""

    normalized_name = name.strip().lower()
    if not normalized_name or normalized_name == "auto":
        raise ValueError("A custom schema name must be non-empty and cannot be 'auto'")
    SCHEMA_ADAPTERS[normalized_name] = adapter


def _mapped_example(
    example: Mapping[str, Any], field_map: Mapping[str, str] | None
) -> Mapping[str, Any]:
    if not field_map:
        return example
    allowed = {
        "instruction",
        "input",
        "output",
        "translation",
        "status",
        "prompt",
        "completion",
        "messages",
        "conversations",
        "source_text",
    }
    unknown = set(field_map) - allowed
    if unknown:
        raise ValueError(f"Unsupported logical field names: {sorted(unknown)}")
    return {
        logical_name: example.get(source_name)
        for logical_name, source_name in field_map.items()
    }


def detect_sft_schema(example: Mapping[str, Any]) -> str | None:
    """Detect known instruction, translated LIMA, or chat-shaped records."""

    keys = set(example)
    if "translation" in keys or "source_text" in keys:
        tagged = clean_scalar(example.get("translation", example.get("source_text")))
        if _TAGGED_DIALOGUE.match(tagged):
            return "lima"
    if {"instruction", "output"}.issubset(keys):
        return "alpaca"
    if {"prompt", "completion"}.issubset(keys):
        return "prompt_completion"
    if keys.intersection({"messages", "conversations"}):
        return "chat"
    return None


def _clean_messages(
    messages: list[dict[str, str]], config: CleaningConfig | None
) -> tuple[list[dict[str, str]], str | None]:
    if config is None:
        return messages, None
    cleaned = []
    for message in messages:
        result = clean_text_with_result(message["content"], config)
        if not result.accepted:
            return [], f"{message['role']}_{result.reason}"
        cleaned.append({"role": message["role"], "content": result.text})
    return cleaned, None


def normalize_sft_example(
    example: Mapping[str, Any],
    *,
    schema: str = "auto",
    field_map: Mapping[str, str] | None = None,
    cleaning_config: CleaningConfig | None = None,
) -> CanonicalExample:
    """Convert a source row to TRL conversational prompt/completion columns.

    The returned shape is stable even for rejected rows, which makes the
    function safe to use with ``datasets.Dataset.map`` followed by ``filter``.
    ``cleaning_config=None`` performs schema conversion only.
    """

    projected = _mapped_example(example, field_map)
    requested_schema = schema.strip().lower()
    schema_name = (
        detect_sft_schema(projected)
        if requested_schema == "auto"
        else requested_schema
    )
    adapter = SCHEMA_ADAPTERS.get(schema_name or "")
    if adapter is None:
        return {
            "prompt": [],
            "completion": [],
            "source_schema": schema_name or "unknown",
            "normalization_status": "unsupported_schema",
        }

    prompt, completion = adapter(projected)
    if not prompt or not completion:
        return {
            "prompt": [],
            "completion": [],
            "source_schema": schema_name,
            "normalization_status": "missing_prompt_or_completion",
        }
    prompt, prompt_reason = _clean_messages(prompt, cleaning_config)
    completion, completion_reason = _clean_messages(completion, cleaning_config)
    rejection = prompt_reason or completion_reason
    if rejection:
        return {
            "prompt": [],
            "completion": [],
            "source_schema": schema_name,
            "normalization_status": f"cleaning_rejected:{rejection}",
        }
    return {
        "prompt": prompt,
        "completion": completion,
        "source_schema": schema_name,
        "normalization_status": "accepted",
    }
