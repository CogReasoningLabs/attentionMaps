from __future__ import annotations

import argparse
import fnmatch
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlparse


DEFAULT_DATASET_ID = "himalaya-ai/nepali-pretrain-corpus"
DEFAULT_PATTERNS = (
    "*.parquet",
    "**/*.parquet",
    "README.md",
    "**/*.json",
)


@dataclass(frozen=True)
class DatasetReference:
    dataset_id: str
    revision: str | None = None
    path_prefix: str | None = None


@dataclass(frozen=True)
class DownloadRequest:
    dataset_id: str
    revision: str
    path_prefix: str | None
    output_dir: Path


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download every Parquet shard from a Hugging Face dataset repository "
            "and save a reproducibility manifest."
        )
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        help=(
            "Dataset ID or Hugging Face URL. Defaults to "
            f"{DEFAULT_DATASET_ID}."
        ),
    )
    parser.add_argument(
        "--dataset-id",
        help="Dataset ID flag retained for compatibility; do not use with dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to data/raw/<dataset-name>",
    )
    parser.add_argument(
        "--revision",
        help="Override the branch, tag, or commit from the URL (default: main)",
    )
    parser.add_argument(
        "--path-prefix",
        help="Only download files beneath this repository folder",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching files and required space without downloading",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Download the complete dataset repository instead of Parquet/metadata only",
    )
    return parser.parse_args(arguments)


def parse_dataset_reference(value: str) -> DatasetReference:
    """Parse either `organization/name` or a Hugging Face dataset URL."""

    if "://" not in value:
        parts = value.strip("/").split("/")
        if len(parts) != 2 or not all(parts):
            raise ValueError(
                "Dataset IDs must look like 'organization/dataset-name'"
            )
        return DatasetReference(dataset_id="/".join(parts))

    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or parsed.netloc not in {
        "huggingface.co",
        "www.huggingface.co",
    }:
        raise ValueError("Only huggingface.co dataset URLs are supported")
    parts = [unquote(part) for part in parsed.path.strip("/").split("/")]
    if len(parts) < 3 or parts[0] != "datasets":
        raise ValueError(
            "Expected a URL like https://huggingface.co/datasets/org/name"
        )

    reference = DatasetReference(dataset_id=f"{parts[1]}/{parts[2]}")
    remainder = parts[3:]
    if not remainder:
        return reference
    if len(remainder) < 2 or remainder[0] != "tree":
        raise ValueError("Dataset URLs may optionally contain /tree/<revision>/<folder>")
    return DatasetReference(
        dataset_id=reference.dataset_id,
        revision=remainder[1],
        path_prefix="/".join(remainder[2:]).strip("/") or None,
    )


def resolve_download_request(args: argparse.Namespace) -> DownloadRequest:
    if args.dataset and args.dataset_id:
        raise ValueError("Pass the dataset either positionally or with --dataset-id, not both")
    reference = parse_dataset_reference(
        args.dataset_id or args.dataset or DEFAULT_DATASET_ID
    )
    path_prefix = args.path_prefix or reference.path_prefix
    if path_prefix:
        path_prefix = path_prefix.strip("/")
        if not path_prefix or path_prefix.startswith(".."):
            raise ValueError("--path-prefix must be a repository-relative folder")
    dataset_directory = re.sub(
        r"[^A-Za-z0-9._]+",
        "_",
        reference.dataset_id.rsplit("/", 1)[-1],
    ).strip("._")
    return DownloadRequest(
        dataset_id=reference.dataset_id,
        revision=args.revision or reference.revision or "main",
        path_prefix=path_prefix,
        output_dir=args.output_dir
        or Path("data/raw") / dataset_directory,
    )


def build_patterns(path_prefix: str | None, all_files: bool) -> tuple[str, ...] | None:
    if all_files and path_prefix is None:
        return None
    if all_files:
        return (f"{path_prefix}/**", "README.md")
    if path_prefix is None:
        return DEFAULT_PATTERNS
    return (
        f"{path_prefix}/*.parquet",
        f"{path_prefix}/**/*.parquet",
        f"{path_prefix}/**/*.json",
        "README.md",
    )


def matches_patterns(path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def human_size(size: int | None) -> str:
    if size is None:
        return "unknown"
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def get_huggingface_api():
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install project dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc
    return HfApi()


def select_files(dataset_info, patterns: tuple[str, ...] | None) -> list[dict]:
    selected = []
    for sibling in dataset_info.siblings or []:
        path = sibling.rfilename
        if patterns is None or matches_patterns(path, patterns):
            selected.append(
                {
                    "path": path,
                    "size": getattr(sibling, "size", None),
                    "blob_id": getattr(sibling, "blob_id", None),
                    "lfs": getattr(sibling, "lfs", None),
                }
            )
    return sorted(selected, key=lambda item: item["path"])


def validate_destination(
    output_dir: Path, dataset_id: str, path_prefix: str | None
) -> None:
    manifest_path = output_dir / "download_manifest.json"
    if not output_dir.exists() or not any(output_dir.iterdir()):
        return
    if not manifest_path.is_file():
        raise FileExistsError(
            f"Output directory is not empty and has no download manifest: {output_dir}. "
            "Choose another --output-dir to avoid mixing datasets."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read existing manifest: {manifest_path}") from exc
    if manifest.get("dataset_id") != dataset_id:
        raise ValueError(
            f"Output directory belongs to {manifest.get('dataset_id')!r}, "
            f"not {dataset_id!r}"
        )
    if manifest.get("path_prefix") != path_prefix:
        raise ValueError(
            f"Output directory was created for path prefix "
            f"{manifest.get('path_prefix')!r}, not {path_prefix!r}"
        )


def available_space(path: Path) -> int:
    existing_parent = path
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    return shutil.disk_usage(existing_parent).free


def write_manifest(
    output_dir: Path,
    *,
    dataset_id: str,
    requested_revision: str,
    resolved_revision: str,
    path_prefix: str | None,
    files: list[dict],
    status: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    known_size = sum(item["size"] or 0 for item in files)
    manifest = {
        "dataset_id": dataset_id,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "path_prefix": path_prefix,
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files),
        "known_download_bytes": known_size,
        "files": files,
    }
    (output_dir / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    request = resolve_download_request(args)
    if args.max_workers < 1:
        raise ValueError("--max-workers must be positive")

    api = get_huggingface_api()
    print(f"Resolving {request.dataset_id}@{request.revision} ...")
    info = api.dataset_info(
        repo_id=request.dataset_id,
        revision=request.revision,
        files_metadata=True,
    )
    resolved_revision = info.sha
    patterns = build_patterns(request.path_prefix, all_files=args.all_files)
    files = select_files(info, patterns=patterns)
    parquet_files = [item for item in files if item["path"].endswith(".parquet")]
    if not parquet_files:
        raise RuntimeError(
            f"No Parquet files found in {request.dataset_id}@{resolved_revision}"
        )

    known_bytes = sum(item["size"] or 0 for item in files)
    free_bytes = available_space(request.output_dir)
    print(f"Resolved commit : {resolved_revision}")
    print(f"Path prefix     : {request.path_prefix or '<entire repository>'}")
    print(f"Output directory: {request.output_dir}")
    print(f"Parquet shards  : {len(parquet_files):,}")
    print(f"Selected files  : {len(files):,}")
    print(f"Known size      : {human_size(known_bytes)}")
    print(f"Available space : {human_size(free_bytes)}")
    for item in files:
        print(f"  {human_size(item['size']):>12}  {item['path']}")

    if args.dry_run:
        print("Dry run complete; nothing downloaded.")
        return
    if known_bytes and free_bytes < int(known_bytes * 1.10):
        raise OSError(
            "Insufficient free space. Keep at least 10% more than the reported "
            f"download size ({human_size(known_bytes)})."
        )

    validate_destination(
        request.output_dir, request.dataset_id, request.path_prefix
    )
    write_manifest(
        request.output_dir,
        dataset_id=request.dataset_id,
        requested_revision=request.revision,
        resolved_revision=resolved_revision,
        path_prefix=request.path_prefix,
        files=files,
        status="downloading",
    )

    from huggingface_hub import snapshot_download

    print(f"Downloading to {request.output_dir.resolve()} ...")
    snapshot_download(
        repo_id=request.dataset_id,
        repo_type="dataset",
        revision=resolved_revision,
        local_dir=request.output_dir,
        allow_patterns=list(patterns) if patterns is not None else None,
        max_workers=args.max_workers,
    )

    missing = [
        item["path"]
        for item in files
        if not (request.output_dir / item["path"]).is_file()
    ]
    if missing:
        raise RuntimeError(f"Download finished with {len(missing)} missing files: {missing}")

    write_manifest(
        request.output_dir,
        dataset_id=request.dataset_id,
        requested_revision=request.revision,
        resolved_revision=resolved_revision,
        path_prefix=request.path_prefix,
        files=files,
        status="complete",
    )
    print(f"Download complete: {request.output_dir.resolve()}")


if __name__ == "__main__":
    main()
