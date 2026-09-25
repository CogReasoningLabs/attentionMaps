"""Physical file formats and local batch-reader requirements, inferred from names."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Iterable, Mapping, Any
from urllib.parse import unquote, urlsplit

_FORMATS = {
    ".parquet": "parquet", ".pq": "parquet", ".jsonl": "jsonl", ".ndjson": "jsonl",
    ".json": "json", ".csv": "csv", ".tsv": "tsv", ".txt": "text",
    ".xlsx": "xlsx", ".xls": "xls", ".arrow": "arrow", ".feather": "feather",
    ".avro": "avro", ".orc": "orc", ".xml": "xml",
}
_COMPRESSION = {".gz": "gzip", ".bz2": "bzip2", ".xz": "xz", ".lzma": "lzma",
                ".zst": "zstd", ".zstd": "zstd", ".lz4": "lz4", ".br": "brotli"}
_ARCHIVES = {".zip": "zip", ".tar": "tar", ".7z": "7z", ".rar": "rar"}
_ARCHIVE_ALIASES = {".tgz": ("tar", "gzip"), ".tbz2": ("tar", "bzip2"),
                    ".txz": ("tar", "xz")}
_BATCH_READERS = {
    ".parquet": ("parquet_batches", "Read Parquet row batches with pyarrow.parquet.ParquetFile.iter_batches."),
    ".jsonl": ("json_lines", "Parse one JSON value per non-empty line and collect records into batches."),
    ".ndjson": ("json_lines", "Parse one JSON value per non-empty line and collect records into batches."),
    ".json": ("json_document", "Parse a JSON document, select its record array/object, then batch; the current reader loads the document into memory."),
    ".csv": ("csv_comma", "Read a comma-delimited table with csv.DictReader; the first row supplies headers."),
    ".txt": ("whole_text_file", "The current batch reader loads the entire text file as ONE record; the inspector uses non-empty lines."),
}


def describe_file_format(path: str, *, name: str | None = None, size: int | None = None) -> dict:
    """Keep provider, file extension, compression, and record parser distinct.

    Query strings on remote URLs do not affect the extension. Archive member
    formats and internal Parquet codecs cannot be established from a filename.
    """
    path = str(path)
    filename = PurePosixPath(unquote(urlsplit(path).path) if "://" in path else path).name
    remaining = filename.lower()
    compression = []
    wrappers = []
    suffix = PurePosixPath(remaining).suffix
    while suffix in _COMPRESSION:
        compression.append(_COMPRESSION[suffix])  # Outermost first: decompression order.
        wrappers.insert(0, suffix)
        remaining = remaining[:-len(suffix)]
        suffix = PurePosixPath(remaining).suffix
    container = _ARCHIVES.get(suffix)
    if suffix in _ARCHIVE_ALIASES:
        container, codec = _ARCHIVE_ALIASES[suffix]
        compression.append(codec)
    format_name = "unknown" if container else _FORMATS.get(suffix, "unknown")
    extension = suffix + "".join(wrappers)
    reader, hint = _BATCH_READERS.get(suffix, (None, "A format-specific reader or conversion is required."))
    if container:
        reader = "zip_members" if container == "zip" and not compression else None
        support = "inspect_archive_members"
        hint = ("Inspect archive members to determine their record formats. The local batch loader extracts ZIP members with supported extensions."
                if reader else "Unpack the archive and inspect member formats before selecting a record reader.")
    elif format_name == "unknown":
        support = "unknown_format"
        hint = "Inspect file content or dataset documentation to identify its record format before choosing a reader."
    elif compression:
        support = "decompress_first"
        hint = "Decompress (" + " → ".join(compression) + ") before parsing. " + hint
    elif suffix in _BATCH_READERS:
        support = "supported_extension"
    else:
        support = "reader_not_implemented"
        if format_name == "tsv":
            hint = "Use a tab-delimited reader; the current local batch loader only handles comma-delimited CSV."
    return {
        "path": path, "name": name or filename, "bytes": size,
        "extension": extension, "format": format_name,
        "compression": compression, "container": container,
        "detection_basis": "filename_extension", "batch_reader": reader,
        "batch_loader_support": support, "parsing_hint": hint,
    }


def describe_dataset_formats(files: Iterable[Mapping[str, Any]]) -> dict:
    """Summarize selected physical files; never infer a single format from shard zero."""
    entries = [describe_file_format(str(item["path"]), name=item.get("name"), size=item.get("bytes"))
               for item in files]
    formats = sorted({item["format"] for item in entries})
    extensions = sorted({item["extension"] for item in entries})
    groups: dict[tuple, dict] = {}
    for entry in entries:
        key = (entry["extension"], entry["format"], tuple(entry["compression"]), entry["container"])
        if key not in groups:
            groups[key] = {field: entry[field] for field in (
                "extension", "format", "compression", "container", "batch_reader",
                "batch_loader_support", "parsing_hint",
            )}
            groups[key].update(files=0, bytes=0)
        group = groups[key]
        group["files"] += 1
        group["bytes"] = group["bytes"] + entry["bytes"] if group["bytes"] is not None and entry["bytes"] is not None else None
    return {
        "format": formats[0] if len(formats) == 1 else "mixed" if formats else "unknown",
        "formats": formats, "extensions": extensions,
        "mixed_formats": len(formats) > 1, "mixed_extensions": len(extensions) > 1,
        "requires_preparation": any(item["batch_loader_support"] != "supported_extension" for item in entries),
        "groups": list(groups.values()), "files": entries,
        "note": "Formats are inferred from filenames, not validated content. Batch support refers to the current local/staged-file batch loader; Hugging Face uses its configured datasets loader. Compression lists describe outer wrappers, not internal Parquet codecs. Archive member formats remain unknown until inspected.",
    }


def inspect_format_metadata(signatures: tuple[tuple[str, int, int], ...]) -> dict:
    """Inventory any local file selection without parsing records or archives."""
    return {
        "format": "files", "metadata_only": True, "rows": None,
        "files": len(signatures), "bytes": sum(size for _, size, _ in signatures),
        "columns": [], "schema": [], "schema_variants": 0, "row_groups": [],
        "file_formats": describe_dataset_formats({"path": path, "bytes": size} for path, size, _ in signatures),
    }


def validate_format_metadata(metadata: dict) -> None:
    """Validate the optional saved format section before read-only presentation."""
    if (not isinstance(metadata, dict) or not isinstance(metadata.get("format"), str)
            or not isinstance(metadata.get("note"), str) or type(metadata.get("mixed_formats")) is not bool
            or not isinstance(metadata.get("extensions"), list)
            or any(not isinstance(item, str) for item in metadata["extensions"])
            or not isinstance(metadata.get("groups"), list) or not isinstance(metadata.get("files"), list)):
        raise ValueError("Inspection report has invalid file-format metadata")
    for collection in ("groups", "files"):
        for item in metadata[collection]:
            if (not isinstance(item, dict)
                    or not all(isinstance(item.get(key), str) for key in ("extension", "format", "parsing_hint"))
                    or item.get("batch_loader_support") not in {
                        "supported_extension", "decompress_first", "inspect_archive_members",
                        "reader_not_implemented", "unknown_format",
                    }
                    or not isinstance(item.get("compression"), list)
                    or any(not isinstance(codec, str) for codec in item["compression"])
                    or any(key not in item or (item[key] is not None and not isinstance(item[key], str))
                           for key in ("container", "batch_reader"))
                    or "bytes" not in item or (item["bytes"] is not None and (type(item["bytes"]) is not int or item["bytes"] < 0))):
                raise ValueError("Inspection report has invalid file-format details")
            if collection == "groups" and (type(item.get("files")) is not int or item["files"] < 0):
                raise ValueError("Inspection report has invalid format group counts")
            if collection == "files" and not isinstance(item.get("path"), str):
                raise ValueError("Inspection report has invalid format file paths")
