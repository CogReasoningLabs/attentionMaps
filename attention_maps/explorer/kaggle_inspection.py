"""Versioned Kaggle file discovery and inspection through the shared readers."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from .file_formats import describe_dataset_formats
from .inspection import file_signatures, inspect_dataset
from .source_imports import SUPPORTED_IMPORT_SUFFIXES, normalize_kaggle_handle, stage_kaggle_source


def discover_kaggle_dataset(handle: str) -> dict:
    """List all pages of file metadata at one version without downloading data."""
    from kagglehub.clients import build_kaggle_client
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest, ApiListDatasetFilesRequest

    handle = normalize_kaggle_handle(handle)
    parts = handle.split("/")
    owner, slug = parts[:2]
    version = int(parts[3]) if len(parts) == 4 else None
    if version is not None and version < 1:
        raise ValueError("Kaggle version must be a positive integer")
    files = []
    try:
        # KaggleHub can emit cache/version messages; keep CLI stdout valid JSON.
        with contextlib.redirect_stdout(sys.stderr), build_kaggle_client() as client:
            api = client.datasets.dataset_api_client
            if version is None:
                request = ApiGetDatasetRequest()
                request.owner_slug, request.dataset_slug = owner, slug
                version = api.get_dataset(request).current_version_number
                if not version or version < 1:
                    raise ValueError("Could not resolve the current Kaggle dataset version")
            pinned = f"{owner}/{slug}/versions/{version}"
            page_token = None
            seen_tokens = set()
            while True:
                request = ApiListDatasetFilesRequest()
                request.owner_slug, request.dataset_slug = owner, slug
                request.dataset_version_number = version
                request.page_size, request.page_token = 100, page_token
                response = api.list_dataset_files(request)
                if response.error_message:
                    raise ValueError(response.error_message)
                for item in response.dataset_files:
                    name = item.name
                    if not name or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts or "\\" in name:
                        raise ValueError("Kaggle returned an invalid dataset file path")
                    files.append({"name": name, "path": f"kaggle://datasets/{pinned}/{quote(name, safe='/')}",
                                  "bytes": item.total_bytes})
                page_token = response.next_page_token
                if not page_token:
                    break
                if page_token in seen_tokens:
                    raise ValueError("Kaggle file listing returned a repeated page token")
                seen_tokens.add(page_token)
    except Exception as error:
        raise ValueError(f"Could not discover Kaggle dataset {handle!r}: {error}") from error
    by_name = {item["name"]: item for item in files}
    files = sorted(by_name.values(), key=lambda item: item["name"])
    return {"provider": "kaggle", "dataset_id": f"{owner}/{slug}", "requested_handle": handle,
            "dataset_handle": pinned, "dataset_version": version, "files": files,
            "file_formats": describe_dataset_formats(files)}


def kaggle_selection(catalog: dict, dataset_file: str | None, *, metadata_only: bool = False) -> list[dict]:
    available = {item["name"]: item for item in catalog["files"]}
    if dataset_file:
        if dataset_file not in available:
            raise ValueError(f"Unknown Kaggle file {dataset_file!r}; use --list to see exact paths")
        selected = [available[dataset_file]]
    elif metadata_only:
        selected = list(available.values())
    else:
        candidates = [item for item in available.values() if PurePosixPath(item["name"]).suffix.lower() in SUPPORTED_IMPORT_SUFFIXES]
        if len(candidates) != 1:
            raise ValueError("Choose --dataset-file; use --list to see Kaggle files and --formats-only for unsupported formats")
        selected = candidates
    if not selected:
        raise ValueError("No Kaggle dataset files were found")
    if not metadata_only and PurePosixPath(selected[0]["name"]).suffix.lower() not in SUPPORTED_IMPORT_SUFFIXES:
        raise ValueError("This Kaggle file cannot be parsed directly; use --formats-only or stage and convert it first")
    return selected


def stage_kaggle_selection(catalog: dict, selected: list[dict]) -> Path:
    """Fetch only the exact selected file from the version used in discovery."""
    name = selected[0]["name"]
    with contextlib.redirect_stdout(sys.stderr):
        downloaded = stage_kaggle_source(catalog["dataset_handle"], name)
    # KaggleHub normally returns the file; tolerate a returned dataset cache root.
    if downloaded.is_dir():
        downloaded = downloaded / name
    if not downloaded.is_file():
        raise ValueError(f"Kaggle did not cache the requested file {name!r}")
    return downloaded.resolve()


def inspect_kaggle_selection(catalog: dict, selected: list[dict], *, cached_path: Path | None = None, show_progress: bool = False) -> dict:
    """Use local parsing for cached records; metadata-only mode never downloads."""
    if cached_path is None:
        sizes = [item["bytes"] for item in selected]
        inventory = {"format": "files", "metadata_only": True, "rows": None,
                     "files": len(selected), "bytes": sum(sizes) if all(value is not None for value in sizes) else None,
                     "schema": [], "columns": [], "schema_variants": 0, "row_groups": []}
    else:
        signatures = file_signatures((cached_path,))
        inventory = inspect_dataset(signatures, show_progress=show_progress)
        inventory["source_files"] = [str(cached_path)]
        inventory["source_signatures"] = [list(item) for item in signatures]
        inventory["cached_file_formats"] = inventory["file_formats"]
    inventory.update(
        provider="kaggle", dataset_id=catalog["dataset_id"], dataset_handle=catalog["dataset_handle"],
        dataset_version=catalog["dataset_version"], dataset_file=selected[0]["name"] if len(selected) == 1 else None,
        source_uri=selected[0]["path"] if len(selected) == 1 else f"kaggle://datasets/{catalog['dataset_handle']}",
        file_formats=describe_dataset_formats(selected),
        selected_files=[item["name"] for item in selected],
        remote_bytes=sum(item["bytes"] for item in selected) if all(item["bytes"] is not None for item in selected) else None,
    )
    return inventory
