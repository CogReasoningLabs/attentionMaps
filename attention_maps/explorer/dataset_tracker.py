"""Read an Excel-exported tracker and atomically edit only reviewed CSV cells."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import fcntl
import io
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import unquote, urlsplit

from attention_maps.datasets.schemas import DATASET_ROLES, parse_dataset_roles


DEFAULT_TRACKER = Path(__file__).resolve().parents[2] / "llm_dataset_tracker - To-go-datasets.csv"
ROLE_COLUMN = "Dataset Role (LLM Lifecycle)"
TASK_COLUMN = "Primary Task"
ACQUISITION_COLUMNS = ("Data Acquisition Method", "Data Acquision Method", "Dataset Generation Method")
EDITABLE_COLUMNS = (ROLE_COLUMN, TASK_COLUMN, *ACQUISITION_COLUMNS)
PRIMARY_TASKS = (
    "Language modeling", "Instruction following", "Question answering", "Machine translation",
    "Text classification", "Sentiment analysis", "Named entity recognition", "Part-of-speech tagging",
    "Summarization", "Dialogue", "Natural language inference", "Information extraction",
    "Reasoning", "Code generation", "Tool use", "Preference ranking",
)
ACQUISITION_METHODS = (
    "Web crawled", "Human Generated", "Synthetic", "Human+Model", "Derived Dataset",
    "Manual collection", "Human annotation", "Machine translation", "OCR extraction",
    "Speech transcription", "Mixed methods", "Not documented",
)


@dataclass(frozen=True)
class TrackerSheet:
    path: Path
    raw: bytes
    text: str
    rows: tuple[tuple[str, ...], ...]
    spans: tuple[tuple[int, int], ...]
    header_index: int

    @property
    def columns(self):
        return tuple(value.strip() for value in self.rows[self.header_index])

    @property
    def dataset_rows(self):
        name = self.columns.index("Dataset Name")
        return tuple(index for index in range(self.header_index + 1, len(self.rows))
                     if len(self.rows[index]) > name and self.rows[index][name].strip())

    @property
    def acquisition_column(self):
        return next((column for column in ACQUISITION_COLUMNS if column in self.columns), None)

    def cell(self, index, column):
        position = self.columns.index(column)
        return self.rows[index][position] if position < len(self.rows[index]) else ""


def read_tracker(path=DEFAULT_TRACKER):
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    stream = io.StringIO(text, newline="")
    reader = csv.reader(stream, strict=True)
    rows, spans = [], []
    start = 0
    for values in reader:
        rows.append(tuple(values))
        end = stream.tell()
        spans.append((start, end))
        start = end
    headers = [i for i, row in enumerate(rows) if {"Dataset Name", ROLE_COLUMN, TASK_COLUMN}.issubset(
        value.strip() for value in row)]
    if len(headers) != 1:
        raise ValueError("Tracker must have one header containing Dataset Name, Dataset Role (LLM Lifecycle), and Primary Task")
    sheet = TrackerSheet(path, raw, text, tuple(rows), tuple(spans), headers[0])
    nonempty = [column for column in sheet.columns if column]
    if len(nonempty) != len(set(nonempty)):
        raise ValueError("Tracker contains duplicate column names")
    if "URL(s)" not in sheet.columns:
        raise ValueError("Tracker needs a URL(s) column to identify dataset rows")
    return sheet


def _source_identity(value):
    value = value.strip().rstrip("/.,;)")
    parsed = urlsplit(value)
    host = parsed.netloc.lower().removeprefix("www.")
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if host in {"huggingface.co", "hf.co"} or parsed.scheme == "hf":
        if parts and parts[0] == "datasets":
            parts = parts[1:]
        if len(parts) >= 2:
            return "hf:" + "/".join(parts[:2])
    if host == "kaggle.com" or parsed.scheme == "kaggle":
        if parts and parts[0] == "datasets":
            parts = parts[1:]
        if len(parts) >= 2:
            return "kaggle:" + "/".join(parts[:2])
    if host:
        return parsed._replace(fragment="", query="", netloc=host, path=parsed.path.rstrip("/")).geturl()
    return value


def matching_tracker_rows(sheet, spec):
    """Suggest only exact source identities; duplicated URLs remain ambiguous."""
    values = [str(spec.location), spec.source_uri or ""]
    if spec.dataset_id:
        prefix = "kaggle" if spec.format.startswith("kaggle") else "hf"
        values.append(f"{prefix}://datasets/{spec.dataset_id}")
    identities = {_source_identity(value) for value in values if value}
    return tuple(index for index in sheet.dataset_rows
                 if any(_source_identity(url) in identities
                        for url in re.findall(r"(?:https?://|hf://|kaggle://|s3://)[^\s<>\"]+", sheet.cell(index, "URL(s)"))))


def format_role_cell(keys):
    labels = {role.key: role.label for role in DATASET_ROLES}
    if any(key not in labels for key in keys):
        raise ValueError("Choose standardized dataset roles; remove unrecognized role labels before saving")
    return ", ".join(labels[key] for key in dict.fromkeys(keys))


def update_tracker_cells(snapshot, row_index, updates):
    """Merge into the latest file, rejecting conflicting edits to the target cells.

    The directory lock survives os.replace and coordinates concurrent UI saves.
    Untouched CSV records, including the spreadsheet preamble, remain byte-identical.
    """
    if row_index not in snapshot.dataset_rows or not updates:
        raise ValueError("Choose a tracker dataset row and columns to update")
    if set(updates) - set(EDITABLE_COLUMNS) or set(updates) - set(snapshot.columns):
        raise ValueError("Only the selected role, primary task, and acquisition-method columns can be edited")
    if any(not isinstance(value, str) for value in updates.values()):
        raise ValueError("Tracker cells must contain text")
    updates = dict(updates)
    if ROLE_COLUMN in updates:
        updates[ROLE_COLUMN] = format_role_cell(parse_dataset_roles(updates[ROLE_COLUMN]))
    path = snapshot.path
    descriptor = os.open(path.parent, os.O_RDONLY)
    temporary = None
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        current = read_tracker(path)
        if current.columns != snapshot.columns:
            raise ValueError("Tracker columns changed. Reload the tracker before saving")
        if row_index not in current.dataset_rows or any(
            current.cell(row_index, field) != snapshot.cell(row_index, field)
            for field in ("Dataset Name", "URL(s)")
        ):
            raise ValueError("The tracker row moved or its identity changed. Reload the tracker before saving")
        for column in updates:
            if column not in current.columns or current.cell(row_index, column) != snapshot.cell(row_index, column):
                raise ValueError(f"{column} changed since you opened this row. Reload the tracker before saving")
        if all(current.cell(row_index, column) == value for column, value in updates.items()):
            return current
        values = list(current.rows[row_index])
        values.extend([""] * (len(current.columns) - len(values)))
        for column, value in updates.items():
            values[current.columns.index(column)] = value
        start, end = current.spans[row_index]
        original = current.text[start:end]
        terminator = "\r\n" if original.endswith("\r\n") else "\n"
        output = io.StringIO(newline="")
        csv.writer(output, lineterminator=terminator).writerow(values)
        replacement = output.getvalue()
        if not original.endswith(("\r", "\n")):
            replacement = replacement[:-len(terminator)]
        payload = (current.text[:start] + replacement + current.text[end:]).encode("utf-8")
        if current.raw.startswith(b"\xef\xbb\xbf"):
            payload = b"\xef\xbb\xbf" + payload
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), path.stat().st_mode & 0o777)
        temporary.replace(path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)
        os.close(descriptor)
    return read_tracker(path)
