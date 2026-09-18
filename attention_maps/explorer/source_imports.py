"""Identifier-driven dataset sources for the interactive explorer."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from urllib.parse import urlparse

from attention_maps.common.google_drive import (
    download_google_drive_path,
    extract_google_drive_id,
)
from attention_maps.datasets.schemas import STANDARD_TRAINING_SCHEMAS

from .catalog import DEFAULT_DATA_ROOT, DatasetSpec


SOURCE_TYPES = ("Hugging Face", "Kaggle", "Google Drive", "S3", "Local")
SUPPORTED_IMPORT_SUFFIXES = {
    ".parquet",
    ".json",
    ".jsonl",
    ".ndjson",
    ".csv",
    ".txt",
    ".xlsx",
}
DEFAULT_IMPORT_CACHE = DEFAULT_DATA_ROOT / "cache" / "source-imports"

_SCHEMA_PURPOSES = {
    schema.key: (
        "Task-specific fine-tuning"
        if schema.key == "task_specific_supervised"
        else schema.label
    )
    for schema in STANDARD_TRAINING_SCHEMAS
}


def normalize_huggingface_id(value: str) -> str:
    """Validate a Hub dataset ID without accepting an arbitrary URL."""

    dataset_id = value.strip().strip("/")
    if not dataset_id or any(character.isspace() for character in dataset_id):
        raise ValueError("Enter a Hugging Face dataset ID such as owner/dataset.")
    if "://" in dataset_id or dataset_id.count("/") > 1:
        raise ValueError("Use the Hugging Face dataset ID, not its web URL.")
    return dataset_id


def normalize_kaggle_handle(value: str) -> str:
    """Validate owner/dataset with an optional /versions/N suffix."""

    handle = value.strip().strip("/")
    if not re.fullmatch(r"[^/\s]+/[^/\s]+(?:/versions/\d+)?", handle):
        raise ValueError(
            "Enter a Kaggle handle such as owner/dataset or "
            "owner/dataset/versions/3."
        )
    return handle


def parse_s3_uri(value: str) -> tuple[str, str]:
    """Return the bucket and exact object key from an s3:// URI."""

    parsed = urlparse(value.strip())
    bucket = parsed.netloc.strip()
    key = parsed.path.lstrip("/")
    if parsed.scheme != "s3" or not bucket or not key or key.endswith("/"):
        raise ValueError("Enter an exact S3 object URI such as s3://bucket/data.csv.")
    return bucket, key


def discover_import_files(path: Path) -> tuple[Path, ...]:
    """Return supported immutable files below one staged source."""

    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"Dataset source does not exist: {resolved}")
    candidates = [resolved] if resolved.is_file() else sorted(resolved.rglob("*"))
    return tuple(
        candidate
        for candidate in candidates
        if candidate.is_file()
        and not candidate.is_symlink()
        and candidate.suffix.lower() in SUPPORTED_IMPORT_SUFFIXES
    )


def stage_kaggle_source(dataset_id: str, dataset_path: str | None = None) -> Path:
    """Download one Kaggle dataset/file through KaggleHub's managed cache."""

    handle = normalize_kaggle_handle(dataset_id)
    requested_path = (dataset_path or "").strip() or None
    try:
        import kagglehub
    except ImportError as error:
        raise ValueError("Kaggle loading requires kagglehub.") from error
    try:
        return Path(
            kagglehub.dataset_download(handle, path=requested_path)
        ).expanduser().resolve()
    except Exception as error:
        raise ValueError(f"Could not load Kaggle dataset {handle!r}: {error}") from error


def stage_google_drive_source(
    identifier: str, *, cache_root: Path = DEFAULT_IMPORT_CACHE
) -> Path:
    """Download a Drive file/folder into a stable resumable cache directory."""

    source_id = extract_google_drive_id(identifier)
    destination = cache_root / "drive" / _short_hash(source_id)
    download_google_drive_path(
        identifier,
        destination,
        credentials_file=os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or None,
        oauth_client_secrets_file=os.getenv("ATTENTION_MAPS_DRIVE_OAUTH_CLIENT")
        or None,
        oauth_token_file=os.getenv(
            "ATTENTION_MAPS_DRIVE_OAUTH_TOKEN", ".google-drive-token.json"
        ),
        full_drive_access=True,
    )
    return destination.resolve()


def stage_s3_object(
    uri: str,
    *,
    cache_root: Path = DEFAULT_IMPORT_CACHE,
    client: object | None = None,
) -> Path:
    """Download one S3 object atomically, reusing a matching cached copy."""

    bucket, key = parse_s3_uri(uri)
    if client is None:
        try:
            import boto3
        except ImportError as error:
            raise ValueError("S3 loading requires boto3; install requirements.txt.") from error
        client = boto3.client(
            "s3",
            endpoint_url=os.getenv("AWS_ENDPOINT_URL")
            or os.getenv("S3_ENDPOINT_URL")
            or None,
        )
    try:
        metadata = client.head_object(Bucket=bucket, Key=key)  # type: ignore[attr-defined]
        expected_size = int(metadata["ContentLength"])
        etag = str(metadata.get("ETag", "")).strip('"')
        name = Path(key).name
        destination = cache_root / "s3" / _short_hash(f"{bucket}/{key}") / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        etag_path = destination.with_suffix(destination.suffix + ".etag")
        cached_etag = etag_path.read_text(encoding="utf-8").strip() if etag_path.is_file() else ""
        if (
            destination.is_file()
            and destination.stat().st_size == expected_size
            and (not etag or cached_etag == etag)
        ):
            return destination.resolve()
        partial = destination.with_suffix(destination.suffix + ".part")
        partial.unlink(missing_ok=True)
        client.download_file(bucket, key, str(partial))  # type: ignore[attr-defined]
        if partial.stat().st_size != expected_size:
            raise ValueError(
                f"S3 object size mismatch: expected {expected_size}, got "
                f"{partial.stat().st_size}."
            )
        partial.replace(destination)
        if etag:
            etag_path.write_text(etag + "\n", encoding="utf-8")
        return destination.resolve()
    except Exception as error:
        if isinstance(error, ValueError):
            raise
        raise ValueError(f"Could not load S3 object {uri!r}: {error}") from error


def huggingface_source_spec(
    dataset_id: str,
    *,
    schema: str,
    config: str | None = None,
    split: str = "train",
    revision: str | None = None,
) -> DatasetSpec:
    """Create a streamed Hugging Face source specification."""

    normalized = normalize_huggingface_id(dataset_id)
    clean_split = split.strip()
    if not clean_split:
        raise ValueError("A Hugging Face split is required.")
    purpose = _purpose_for_schema(schema)
    return DatasetSpec(
        key=f"dynamic:huggingface:{normalized}:{config or ''}:{clean_split}:{revision or ''}",
        label=f"Hugging Face · {normalized} · {clean_split}",
        stage="dataset",
        files=(),
        format="huggingface",
        dataset_id=normalized,
        dataset_config=(config or "").strip() or None,
        dataset_split=clean_split,
        dataset_revision=(revision or "").strip() or None,
        provider="Hugging Face Hub",
        purpose_path=(purpose,),
        tags=("dynamic source", schema),
    )


def staged_source_spec(
    path: Path,
    *,
    source_type: str,
    schema: str,
    source_uri: str | None = None,
) -> DatasetSpec:
    """Create a source spec for one selected local or staged remote file."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.suffix.lower() not in SUPPORTED_IMPORT_SUFFIXES:
        raise ValueError(f"Unsupported dataset file: {resolved}")
    purpose = _purpose_for_schema(schema)
    format_name = {
        ".jsonl": "jsonl",
        ".ndjson": "jsonl",
        ".txt": "text",
    }.get(resolved.suffix.lower(), resolved.suffix.lower().lstrip("."))
    identity = source_uri or str(resolved)
    return DatasetSpec(
        key=f"dynamic:{source_type.casefold()}:{_short_hash(identity)}:{resolved.name}",
        label=f"{source_type} · {resolved.name}",
        stage="dataset",
        files=(resolved,),
        format=format_name,
        source_uri=source_uri,
        provider=source_type,
        purpose_path=(purpose,),
        tags=("dynamic source", schema),
    )


def _purpose_for_schema(schema: str) -> str:
    try:
        return _SCHEMA_PURPOSES[schema]
    except KeyError as error:
        raise ValueError(f"Unknown standard dataset schema: {schema!r}") from error


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
