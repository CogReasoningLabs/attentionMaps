"""SQLite-backed rate accounting and generated-dataset run storage."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from attention_maps.inference.comparison import ComparisonConfigurationError

class GenerationRateLedger:
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

    def wait_and_record(
        self,
        scope: str,
        per_minute: int,
        per_day: int,
        on_wait: Callable[[float], None] | None = None,
    ) -> None:
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
            wait_seconds = max(0.1, min(waits) if waits else 0.1)
            if on_wait is not None:
                on_wait(wait_seconds)
            time.sleep(wait_seconds)


class GenerationStore:
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
