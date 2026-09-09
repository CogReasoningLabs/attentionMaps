"""Small Streamlit-independent helpers for explorer presentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .catalog import MANIFEST_NAMES, TEXT_FIELD_NAMES, VIEWER_PREFIX, DatasetSpec

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
        "inputs",
        "targets",
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
        role = str(message.get("role", message.get("from", "unknown"))).upper()
        content = message.get("content", message.get("value", ""))
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

    chat_field = data_fields.get("messages", data_fields.get("conversations"))
    render_messages(st, chat_field)
    long_text_fields = tuple(
        field
        for field in TEXT_FIELD_NAMES
        if field not in {"messages", "conversations"}
    )
    shown: set[str] = set()
    if "source_text" in data_fields and "translation" in data_fields:
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
