"""Interactive structured-dataset generator.

Run with:

    python -m streamlit run apps/dataset_generator.py
"""

from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Any

from attention_maps.inference.comparison import (
    ComparisonConfigurationError,
    DEFAULT_GEMINI_FLASH_LITE_MODEL,
    GoogleGenAIBackend,
    OpenAIBackend,
    build_decoding_grid,
    build_prompt,
)
from scripts.utils.nepali_text import (
    CleaningConfig,
    clean_text_with_result,
    normalize_whitespace,
)
from apps.dataset_explorer import (
    DEFAULT_DATA_ROOT,
    PROJECT_ROOT,
    VIEWER_PREFIX,
    aya_nepali_dataset_specs,
    custom_dataset,
    discover_datasets,
    extract_text,
    file_signatures,
    himalaya_nepali_sft_dataset,
    inspect_dataset,
    inspect_huggingface_dataset,
    inspect_kaggle_text,
    inspect_kaggle_workbook,
    iriis_nepali_text_corpus_specs,
    kaggle_dataset_specs,
    lima_original_dataset,
    lima_translation_dataset,
    parse_number_list,
    sample_dataset_rows,
    secret_fingerprint,
    text_columns,
)


def _prepare_source(text: str, cleaning: CleaningConfig | None) -> str:
    """Reuse the project's canonical Nepali cleaner when it is enabled."""

    if _contains_corrupted_text(text):
        return ""
    if cleaning is None:
        return normalize_whitespace(text)
    result = clean_text_with_result(text, cleaning)
    return result.text if result.accepted else ""


def _contains_corrupted_text(text: str) -> bool:
    """Identify replacement glyphs and binary control bytes masquerading as text."""

    allowed_controls = {"\n", "\t", "\u200c", "\u200d"}
    return "\ufffd" in text or any(
        unicodedata.category(character).startswith("C")
        and character not in allowed_controls
        for character in text
    )


def _parse_output_schema(value: str) -> dict[str, str]:
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


def _dataset_records_for_export(
    generated_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Return only successful user-schema records from internal result rows."""

    return [
        dict(row["_dataset_record"])
        for row in generated_rows
        if row.get("_status") == "ok"
        and isinstance(row.get("_dataset_record"), dict)
    ]


def _records_jsonl(records: list[dict[str, str]]) -> str:
    """Serialize public dataset records without internal generation metadata."""

    if not records:
        return ""
    return "\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n"


def _records_csv(records: list[dict[str, str]]) -> str:
    """Serialize records to CSV while preserving schema field order."""

    if not records:
        return ""
    fieldnames = list(dict.fromkeys(field for record in records for field in record))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)
    return buffer.getvalue()


def _result_preview_rows(
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


def _record_from_schema(
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


class _GenerationRateLedger:
    """A tiny persistent request ledger; provider limits are never modified."""

    def __init__(self, path: Path) -> None:
        self.path = path
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS requests (scope TEXT, created REAL)"
            )

    def reset(self) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute("DELETE FROM requests")

    def usage(self, scope: str, now: float | None = None) -> tuple[int, int]:
        now = now or time.time()
        with sqlite3.connect(self.path) as connection:
            minute = connection.execute(
                "SELECT COUNT(*) FROM requests WHERE scope = ? AND created > ?",
                (scope, now - 60),
            ).fetchone()[0]
            day = connection.execute(
                "SELECT COUNT(*) FROM requests WHERE scope = ? AND created > ?",
                (scope, now - 86_400),
            ).fetchone()[0]
        return int(minute), int(day)

    def wait_and_record(self, scope: str, per_minute: int, per_day: int) -> None:
        """Block until a locally configured slot is available, then reserve it."""

        while True:
            now = time.time()
            with sqlite3.connect(self.path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                minute_rows = connection.execute(
                    "SELECT created FROM requests WHERE scope = ? AND created > ? ORDER BY created",
                    (scope, now - 60),
                ).fetchall()
                day_rows = connection.execute(
                    "SELECT created FROM requests WHERE scope = ? AND created > ? ORDER BY created",
                    (scope, now - 86_400),
                ).fetchall()
                if len(day_rows) >= per_day:
                    raise ComparisonConfigurationError(
                        f"local daily request limit reached for {scope}; reset the "
                        "local counter only if that is intentional"
                    )
                if len(minute_rows) < per_minute:
                    connection.execute("INSERT INTO requests VALUES (?, ?)", (scope, now))
                    return
                waits = []
                if len(minute_rows) >= per_minute:
                    waits.append(minute_rows[0][0] + 60 - now)
            time.sleep(max(0.1, min(waits) if waits else 0.1))


class _GenerationStore:
    """Persistent local archive of generator runs and every generated record."""

    def __init__(self, path: Path) -> None:
        self.path = path
        with sqlite3.connect(self.path) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS generator_runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    dataset_label TEXT NOT NULL,
                    source_column TEXT,
                    task_preset TEXT NOT NULL,
                    system_prompt TEXT NOT NULL,
                    user_prompt TEXT NOT NULL,
                    output_schema_json TEXT NOT NULL,
                    hyperparameters_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS generated_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES generator_runs(run_id),
                    source_row_index TEXT,
                    source_text TEXT NOT NULL,
                    rendered_prompt TEXT NOT NULL,
                    model TEXT NOT NULL,
                    decoding TEXT NOT NULL,
                    generated_output TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT NOT NULL,
                    latency_seconds REAL NOT NULL
                );
                """
            )

    def save_run(self, metadata: dict[str, Any], rows: list[dict[str, Any]]) -> str:
        run_id = uuid.uuid4().hex
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """INSERT INTO generator_runs VALUES (?, datetime('now'), ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    metadata["dataset_label"], metadata["source_column"],
                    metadata["task_preset"], metadata["system_prompt"],
                    metadata["user_prompt"], json.dumps(metadata["output_schema"], ensure_ascii=False),
                    json.dumps(metadata["hyperparameters"], ensure_ascii=False),
                ),
            )
            connection.executemany(
                """INSERT INTO generated_records (
                    run_id, source_row_index, source_text, rendered_prompt, model,
                    decoding, generated_output, record_json, status, error, latency_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        run_id, str(row.get("_source_row_index", "")), row["_source_text"],
                        row["_rendered_prompt"], row["_model"], row["_decoding"],
                        row["_generated_output"], json.dumps(row["_dataset_record"], ensure_ascii=False),
                        row["_status"], row["_error"], row["_latency_seconds"],
                    )
                    for row in rows
                ],
            )
        return run_id

    def recent_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            return [dict(row) for row in connection.execute(
                """SELECT run_id, created_at, dataset_label, task_preset,
                   (SELECT COUNT(*) FROM generated_records WHERE run_id = generator_runs.run_id) AS records
                   FROM generator_runs ORDER BY created_at DESC LIMIT ?""",
                (limit,),
            )]

    def records_for_run(
        self, run_id: str, *, successful_only: bool = True
    ) -> list[dict[str, str]]:
        """Load public dataset records for a persisted generation run."""

        query = "SELECT record_json FROM generated_records WHERE run_id = ?"
        parameters: list[object] = [run_id]
        if successful_only:
            query += " AND status = ?"
            parameters.append("ok")
        query += " ORDER BY id"
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(query, parameters).fetchall()
        records = [json.loads(row[0]) for row in rows]
        return [record for record in records if isinstance(record, dict)]


def run_generator_app() -> None:
    """Focused, persistent dataset generator UI."""

    try:
        import streamlit as st
    except ModuleNotFoundError as error:
        raise SystemExit("Streamlit is not installed. Run `pip install -r requirements.txt`.") from error
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        # Keep the app usable before the newly added optional helper is installed.
        env_path = PROJECT_ROOT / ".env"
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                key, separator, value = line.partition("=")
                if separator and key.strip() and not key.lstrip().startswith("#"):
                    os.environ.setdefault(key.strip(), value.strip())
    else:
        load_dotenv(PROJECT_ROOT / ".env")

    st.set_page_config(page_title="Dataset Generator", page_icon="✦", layout="wide")
    # Nirmala UI ships with Windows and has full Devanagari shaping support.
    # The fallbacks keep the app legible on Linux/macOS installations.
    st.markdown(
        """
        <style>
        :root, body, input, textarea, button, select, code, pre,
        [data-testid="stApp"] *, [data-testid="stDataFrame"] * {
            font-family: "Nirmala UI", "Nirmala", "Noto Sans Devanagari",
                         "Noto Serif Devanagari", sans-serif !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Dataset Generator")
    st.caption("Generate structured training records from a bounded dataset batch. API calls may be billed and source text is sent to the selected hosted provider.")

    @st.cache_data(show_spinner=False)
    def cached_generator_inventory(signatures: tuple[tuple[str, int, int], ...]) -> dict[str, Any]:
        return inspect_dataset(signatures)

    @st.cache_data(show_spinner=False)
    def cached_generator_batch(inventory: dict[str, Any], count: int, seed: int, columns: tuple[str, ...]) -> list[dict[str, Any]]:
        return sample_dataset_rows(inventory, count, seed, columns)

    @st.cache_resource(show_spinner=False)
    def cached_openai_backend(model: str, fingerprint: str, _key: str) -> OpenAIBackend:
        del fingerprint
        return OpenAIBackend(model, _key)

    @st.cache_resource(show_spinner=False)
    def cached_google_generator_backend(model: str, fingerprint: str, _key: str) -> GoogleGenAIBackend:
        del fingerprint
        return GoogleGenAIBackend(model, _key)

    @st.cache_data(show_spinner=False)
    def cached_huggingface_inventory(
        dataset_id: str, config: str | None, split: str, filter_column: str | None,
        filter_value: str | None, credential_fingerprint: str, _token: str,
    ) -> dict[str, Any]:
        del credential_fingerprint
        return inspect_huggingface_dataset(
            dataset_id, split, config=config, token=_token or None,
            filter_column=filter_column, filter_value=filter_value,
        )

    @st.cache_data(show_spinner=False)
    def cached_kaggle_inventory(dataset_id: str, dataset_file: str) -> dict[str, Any]:
        return inspect_kaggle_workbook(dataset_id, dataset_file)

    @st.cache_data(show_spinner=False)
    def cached_kaggle_text_inventory(dataset_id: str, dataset_file: str) -> dict[str, Any]:
        return inspect_kaggle_text(dataset_id, dataset_file)

    specs = discover_datasets(DEFAULT_DATA_ROOT)
    specs.extend(item for item in (lima_translation_dataset(), lima_original_dataset()) if item)
    specs.extend([himalaya_nepali_sft_dataset(), *aya_nepali_dataset_specs(), *iriis_nepali_text_corpus_specs(), *kaggle_dataset_specs()])
    source_mode = st.radio(
        "Prompt source", ("Dataset batch", "Single dataset record", "Custom text"), horizontal=True,
        help="Dataset modes use the same local and remote sources exposed by Dataset Explorer.",
    )
    records: list[dict[str, Any]] = []
    source_column = ""
    dataset_label = "Custom text"
    selected_preview = ""
    st.markdown("### Source preprocessing")
    cleaning_mode = st.selectbox(
        "Cleaning profile",
        ("Normalize only", "Nepali preserve", "Nepali strict"),
        help=(
            "Nepali profiles reuse scripts.utils.nepali_text.clean_text_with_result: "
            "they remove markup, URLs, unsafe controls, and reject rows below the "
            "configured Devanagari/language thresholds."
        ),
    )
    cleaning_config: CleaningConfig | None = None
    cleaning_metadata: dict[str, Any] = {"profile": cleaning_mode}
    if cleaning_mode != "Normalize only":
        cleaning_controls = st.columns(3)
        min_ratio = cleaning_controls[0].number_input(
            "Minimum Devanagari ratio", 0.0, 1.0, 0.80, step=0.05
        )
        min_letters = cleaning_controls[1].number_input(
            "Minimum Devanagari letters", 1, 10_000, 10
        )
        cleaner_min_chars = cleaning_controls[2].number_input(
            "Cleaner minimum characters", 1, 20_000, 20
        )
        cleaning_config = CleaningConfig(
            min_devanagari_ratio=float(min_ratio),
            min_devanagari_letters=int(min_letters),
            min_characters=int(cleaner_min_chars),
            mode="strict" if cleaning_mode == "Nepali strict" else "preserve",
        )
        cleaning_metadata["config"] = {
            "min_devanagari_ratio": float(min_ratio),
            "min_devanagari_letters": int(min_letters),
            "min_characters": int(cleaner_min_chars),
            "mode": cleaning_config.mode,
            "normalization": cleaning_config.normalization,
        }
    if source_mode != "Custom text":
        path_text = st.text_input("Optional local Parquet file or directory", "")
        if path_text.strip():
            custom = custom_dataset(Path(path_text))
            if custom:
                specs = [custom]
            else:
                st.error("No Parquet files found at that path.")
        spec_by_label = {f"{item.label} ({item.stage})": item for item in specs}
        selected_spec = st.selectbox("Dataset", list(spec_by_label))
        spec = spec_by_label[selected_spec]
        dataset_label = selected_spec
        try:
            if spec.format in {"huggingface", "kaggle", "kaggle_text"}:
                hf_token = os.getenv("HF_TOKEN") or os.getenv("HF_token") or ""
                if spec.format == "huggingface":
                    inventory = cached_huggingface_inventory(
                        spec.dataset_id or "", spec.dataset_config, spec.dataset_split or "train",
                        spec.filter_column, spec.filter_value, secret_fingerprint(hf_token), hf_token,
                    )
                elif spec.format == "kaggle":
                    inventory = cached_kaggle_inventory(spec.dataset_id or "", spec.dataset_file or "")
                else:
                    inventory = cached_kaggle_text_inventory(spec.dataset_id or "", spec.dataset_file or "")
            else:
                inventory = cached_generator_inventory(file_signatures(spec.files))
        except (OSError, ValueError, ImportError) as error:
            st.error(f"Could not inspect dataset: {error}")
            inventory = None
        if inventory:
            candidates = text_columns(inventory["schema"])
            if not candidates:
                st.error("The selected dataset has no usable text column.")
            else:
                controls = st.columns(5)
                source_column = controls[0].selectbox("Dataset text column", candidates)
                count = controls[1].number_input(
                    "Batch size" if source_mode == "Dataset batch" else "Records",
                    1, 100 if source_mode == "Dataset batch" else 1,
                    10 if source_mode == "Dataset batch" else 1,
                )
                sample_seed = controls[2].number_input("Sample seed", 0, value=42)
                min_chars = controls[3].number_input("Min source characters", 1, 20_000, 50, step=25)
                max_chars = controls[4].number_input("Max source characters", 50, 20_000, 4_000, step=50)
                remove_duplicates = st.checkbox("Remove duplicate preprocessed text", value=True)
                sampled = cached_generator_batch(inventory, int(count), int(sample_seed), (source_column,))
                corrupt_sources = sum(
                    _contains_corrupted_text(
                        "\n\n".join(extract_text(record.get(source_column)))
                    )
                    for record in sampled
                )
                prepared = [
                    {
                        **record,
                        "__source": _prepare_source(
                            "\n\n".join(extract_text(record.get(source_column))),
                            cleaning_config,
                        )[:int(max_chars)],
                    }
                    for record in sampled
                ]
                prepared = [record for record in prepared if len(record["__source"]) >= int(min_chars)]
                if remove_duplicates:
                    seen_sources: set[str] = set()
                    prepared = [record for record in prepared if not (record["__source"] in seen_sources or seen_sources.add(record["__source"]))]
                if corrupt_sources:
                    st.warning(
                        f"{corrupt_sources} sampled record(s) in {source_column!r} contain "
                        "invalid Unicode or binary data and were excluded from prompts. "
                        "Choose a different text column or a clean dataset source."
                    )
                if not prepared:
                    st.warning("No usable text records remained after preprocessing.")
                else:
                    preview_index = st.selectbox(
                        "Preview one record from this batch", range(len(prepared)),
                        format_func=lambda index: f"Record {index + 1} · row {prepared[index].get(f'{VIEWER_PREFIX}row_index')}",
                    )
                    selected_preview = prepared[preview_index]["__source"]
                    records = prepared
                    st.caption(f"{len(records)} prepared record(s) will generate; only the selected record is previewed.")
    else:
        raw_source = st.text_area("Source text", height=180, value="")
        selected_preview = _prepare_source(raw_source, cleaning_config)
        if raw_source and _contains_corrupted_text(raw_source):
            st.error(
                "The source contains invalid Unicode or binary data, so it cannot be "
                "used as a prompt. Paste clean UTF-8 text instead."
            )
        records = [{"__source": selected_preview, f"{VIEWER_PREFIX}row_index": "manual"}] if selected_preview else []

    presets = {
        "Instruction / answer": ("You create accurate, helpful Nepali training data.", "Using the source below, write one high-quality instruction and its answer.\n\nSource:\n{text}"),
        "Question / answer": ("You create grounded question-answer examples.", "Create one question and answer grounded only in this source.\n\n{text}"),
        "Summarization": ("You write concise, factual Nepali summaries.", "Summarize this text in Nepali.\n\n{text}"),
        "Custom": ("", "{text}"),
    }
    task = st.selectbox("Task preset", list(presets))
    defaults = presets[task]
    prompt_columns = st.columns(2)
    system_prompt = prompt_columns[0].text_area("System prompt", defaults[0], height=150)
    user_prompt = prompt_columns[1].text_area("User prompt template", defaults[1], height=150, help="Use {text} for each preprocessed dataset value.")
    try:
        final_prompt = build_prompt(user_prompt, selected_preview) if selected_preview else ""
    except ComparisonConfigurationError as error:
        st.error(str(error)); final_prompt = ""
    if final_prompt:
        with st.expander("Final prompt preview", expanded=True):
            st.markdown("**System**"); st.code(system_prompt or "(none)")
            st.markdown("**User**"); st.code(final_prompt)

    schema_default = json.dumps({
        "instruction": "{prompt}",
        "response": "{output}",
        "source": "{source}",
        "model": "{model}",
        "row_index": "{row_index}",
    }, ensure_ascii=False, indent=2)
    st.markdown("### Output schema")
    schema_text = st.text_area("JSON object: output field → template", schema_default, height=220, help="Available placeholders: source, output, prompt, system_prompt, model, decoding, row_index. Values are strings; use a `response`: `{output}` field for simple SFT data.")
    try:
        output_schema = _parse_output_schema(schema_text)
        schema_error = ""
    except ComparisonConfigurationError as error:
        output_schema = {}; schema_error = str(error); st.error(schema_error)

    st.markdown("### Inference and decoding")
    backend_names = st.multiselect("Inference backends", ("OpenAI API", "Gemini API"))
    backend_cols = st.columns(2)
    openai_model = backend_cols[0].text_input("OpenAI model", "gpt-4.1-mini")
    gemini_model = backend_cols[1].text_input("Gemini model", DEFAULT_GEMINI_FLASH_LITE_MODEL)
    grid_cols = st.columns(5)
    temperatures_text = grid_cols[0].text_input("Temperatures", "0.7")
    top_ps_text = grid_cols[1].text_input("Top-p", "0.95")
    top_ks_text = grid_cols[2].text_input("Top-k (blank = off)", "")
    max_tokens = grid_cols[3].number_input("Max output tokens", 1, 8_192, 512)
    seed = grid_cols[4].number_input("Seed", 0, value=42)
    try:
        top_ks: list[int | None] = [int(item) for item in parse_number_list(top_ks_text, value_type=int, name="top-k")] if top_ks_text.strip() else [None]
        configs = build_decoding_grid(parse_number_list(temperatures_text, value_type=float, name="temperatures"), parse_number_list(top_ps_text, value_type=float, name="top-p"), top_ks, max_new_tokens=int(max_tokens), seed=int(seed))
        grid_error = ""
    except ComparisonConfigurationError as error:
        configs = []; grid_error = str(error); st.error(grid_error)

    limiter_cols = st.columns(3)
    per_minute = limiter_cols[0].number_input("Max requests / minute / model", 1, 1_000, 20)
    per_day = limiter_cols[1].number_input("Max requests / day / model", 1, 100_000, 1_000)
    ledger = _GenerationRateLedger(PROJECT_ROOT / ".dataset_generator_rate_limits.sqlite3")
    store = _GenerationStore(PROJECT_ROOT / ".dataset_generator.sqlite3")
    if limiter_cols[2].button("Reset local rate counters"):
        ledger.reset(); st.success("Local counters reset. Provider-side limits are unchanged.")
    for name in backend_names:
        scope = (
            "openai:" + openai_model
            if name == "OpenAI API"
            else ("google-gemma:" if gemini_model.removeprefix("models/").startswith("gemma-") else "google-gemini:") + gemini_model
        )
        minute_used, day_used = ledger.usage(scope)
        st.caption(f"{scope}: {minute_used}/{int(per_minute)} this minute · {day_used}/{int(per_day)} today (local tracker)")
    with st.expander("Saved generation runs"):
        saved_runs = store.recent_runs()
        if saved_runs:
            st.dataframe(saved_runs, width="stretch", hide_index=True)
            saved_by_label = {
                (
                    f"{run['created_at']} · {run['dataset_label']} · "
                    f"{run['records']} result(s)"
                ): run["run_id"]
                for run in saved_runs
            }
            selected_saved_label = st.selectbox(
                "Export a saved run", list(saved_by_label)
            )
            saved_records = store.records_for_run(
                saved_by_label[selected_saved_label]
            )
            if saved_records:
                saved_exports = st.columns(2)
                saved_exports[0].download_button(
                    "Download saved JSONL",
                    _records_jsonl(saved_records),
                    "generated_dataset.jsonl",
                    "application/jsonl",
                    key="saved-jsonl",
                )
                saved_exports[1].download_button(
                    "Download saved CSV",
                    _records_csv(saved_records),
                    "generated_dataset.csv",
                    "text/csv",
                    key="saved-csv",
                )
            else:
                st.caption("This run has no successful records to export.")
        else:
            st.caption("No generator runs have been saved yet.")

    request_count = len(records) * len(backend_names) * len(configs)
    st.info(f"This run will make {request_count} request(s): {len(records)} records × {len(backend_names)} models × {len(configs)} decoding settings.")
    can_run = bool(records and backend_names and configs and final_prompt and output_schema and not schema_error and not grid_error)
    if st.button("Generate dataset", type="primary", disabled=not can_run):
        try:
            backends = []
            if "OpenAI API" in backend_names:
                key = os.getenv("OPENAI_API_KEY", "")
                if not key: raise ComparisonConfigurationError("OPENAI_API_KEY is missing. Add it to .env and restart Streamlit.")
                backends.append(cached_openai_backend(openai_model, secret_fingerprint(key), key))
            if "Gemini API" in backend_names:
                key = os.getenv("GEMINI_API_KEY", "")
                if not key: raise ComparisonConfigurationError("GEMINI_API_KEY is missing. Add it to .env and restart Streamlit.")
                backends.append(cached_google_generator_backend(gemini_model, secret_fingerprint(key), key))
            generated = []
            progress = st.progress(0, text="Preparing generation…")
            total = len(records) * len(backends) * len(configs); completed = 0
            for record in records:
                prompt = build_prompt(user_prompt, record["__source"])
                for backend in backends:
                    for config in configs:
                        ledger.wait_and_record(backend.label, int(per_minute), int(per_day))
                        started = time.perf_counter()
                        try:
                            output, error = backend.generate(prompt, config, system_prompt), ""
                        except Exception as generation_error:
                            output, error = "", f"{type(generation_error).__name__}: {generation_error}"
                        dataset_record = _record_from_schema(
                            output_schema, source=record["__source"], output=output,
                            prompt=prompt, system_prompt=system_prompt, model=backend.label,
                            decoding=config.name, row_index=record.get(f"{VIEWER_PREFIX}row_index"),
                        )
                        generated.append({
                            **dataset_record,
                            "_source_row_index": record.get(f"{VIEWER_PREFIX}row_index"),
                            "_source_text": record["__source"],
                            "_rendered_prompt": prompt,
                            "_model": backend.label,
                            "_decoding": config.name,
                            "_generated_output": output,
                            "_dataset_record": dataset_record,
                            "_status": "error" if error else "ok", "_error": error,
                            "_latency_seconds": round(time.perf_counter() - started, 3),
                        })
                        completed += 1; progress.progress(completed / total, text=f"Generated {completed}/{total}")
            st.session_state["dataset_generator_results"] = generated
            try:
                run_id = store.save_run(
                    {
                        "dataset_label": dataset_label,
                        "source_column": source_column,
                        "task_preset": task,
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "output_schema": output_schema,
                        "hyperparameters": {
                            "backends": backend_names,
                            "openai_model": openai_model,
                            "gemini_model": gemini_model,
                            "decoding_grid": [config.__dict__ for config in configs],
                            "source_cleaning": cleaning_metadata,
                            "requests_per_minute_per_model": int(per_minute),
                            "requests_per_day_per_model": int(per_day),
                        },
                    },
                    generated,
                )
                st.session_state["dataset_generator_run_id"] = run_id
                st.success(f"Saved run {run_id} to .dataset_generator.sqlite3")
            except sqlite3.Error as error:
                st.error(f"Generation finished, but could not save the database run: {error}")
        except (ComparisonConfigurationError, RuntimeError, ImportError, OSError) as error:
            st.error(f"Could not generate dataset: {error}")

    results = st.session_state.get("dataset_generator_results", [])
    if results:
        st.markdown("### Generated records")
        st.dataframe(_result_preview_rows(results), width="stretch", hide_index=True)
        successful_records = _dataset_records_for_export(results)
        failed_count = len(results) - len(successful_records)
        if failed_count:
            st.warning(
                f"{failed_count} generation(s) failed and are excluded from downloads."
            )
        if successful_records:
            exports = st.columns(2)
            exports[0].download_button(
                "Download JSONL",
                _records_jsonl(successful_records),
                "generated_dataset.jsonl",
                "application/jsonl",
            )
            exports[1].download_button(
                "Download CSV",
                _records_csv(successful_records),
                "generated_dataset.csv",
                "text/csv",
            )
        else:
            st.error("No successful records are available to download.")


if __name__ == "__main__":
    run_generator_app()
