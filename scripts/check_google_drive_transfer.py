#!/usr/bin/env python3
"""Download one Drive file/folder, upload it elsewhere, and report timings."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attention_maps.common.google_drive import (
    DEFAULT_CHUNK_SIZE,
    DriveDownloadProgress,
    DriveDownloadSummary,
    DriveUploadProgress,
    DriveUploadSummary,
    download_google_drive_path,
    upload_path_to_google_drive,
)


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Source Drive file/folder ID or sharing URL")
    parser.add_argument(
        "--destination-folder-id",
        required=True,
        help="Destination Google Drive folder ID",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/drive-transfer-checks"),
        help="Local root for downloaded data and transfer_check.json",
    )
    parser.add_argument(
        "--run-name",
        help="Unique local run name; defaults to a UTC timestamp",
    )
    parser.add_argument("--credentials-file", type=Path)
    parser.add_argument("--oauth-client-secrets", type=Path)
    parser.add_argument(
        "--oauth-token-file", type=Path, default=Path(".google-drive-token.json")
    )
    parser.add_argument("--impersonate-user")
    parser.add_argument(
        "--chunk-size-mib",
        type=int,
        default=DEFAULT_CHUNK_SIZE // 1024**2,
        help="Download/upload chunk size in MiB (default: 8)",
    )
    return parser.parse_args(arguments)


def run_transfer_check(
    *,
    source: str,
    destination_folder_id: str,
    run_dir: Path,
    credentials_file: Path | None = None,
    oauth_client_secrets_file: Path | None = None,
    oauth_token_file: Path = Path(".google-drive-token.json"),
    impersonate_user: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    download_func: Callable[..., DriveDownloadSummary] = download_google_drive_path,
    upload_func: Callable[..., DriveUploadSummary] = upload_path_to_google_drive,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Run a transfer check; injectable functions keep it integration-testable."""

    if not destination_folder_id.strip():
        raise ValueError("A destination Drive folder ID is required.")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"Transfer-check directory is not empty: {run_dir}")
    download_dir = run_dir / "downloaded"
    run_dir.mkdir(parents=True, exist_ok=True)

    download_state = {"last_percent": -1}

    def download_progress(update: DriveDownloadProgress) -> None:
        percent = round(100 * update.downloaded_bytes / max(1, update.total_bytes))
        if percent != download_state["last_percent"]:
            print(
                f"[download] {percent:3d}% files "
                f"{update.completed_files}/{update.total_files}",
                flush=True,
            )
            download_state["last_percent"] = percent

    common = {
        "credentials_file": credentials_file,
        "oauth_client_secrets_file": oauth_client_secrets_file,
        "oauth_token_file": oauth_token_file,
        "full_drive_access": True,
        "impersonate_user": impersonate_user,
        "chunk_size": chunk_size,
    }
    total_started = clock()
    download_started = clock()
    download = download_func(
        source,
        download_dir,
        progress=download_progress,
        **common,
    )
    download_seconds = clock() - download_started

    upload_state = {"last_percent": -1}

    def upload_progress(update: DriveUploadProgress) -> None:
        percent = round(100 * update.uploaded_bytes / max(1, update.total_bytes))
        if percent != upload_state["last_percent"]:
            print(
                f"[upload]   {percent:3d}% files "
                f"{update.completed_files}/{update.total_files}",
                flush=True,
            )
            upload_state["last_percent"] = percent

    upload_started = clock()
    upload = upload_func(
        download_dir,
        destination_folder_id,
        progress=upload_progress,
        **common,
    )
    upload_seconds = clock() - upload_started
    total_seconds = clock() - total_started

    report_path = run_dir / "transfer_check.json"
    report = {
        "schema_version": 1,
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_drive_id": download.source_drive_id,
        "destination_parent_drive_id": destination_folder_id,
        "uploaded_root_drive_id": upload.root_drive_id,
        "local_download_dir": str(download.destination),
        "chunk_size_bytes": chunk_size,
        "download": {
            "seconds": round(download_seconds, 6),
            "files_downloaded": download.files_downloaded,
            "files_reused": download.files_reused,
            "bytes": download.bytes_downloaded,
            "mib_per_second": _throughput(download.bytes_downloaded, download_seconds),
        },
        "upload": {
            "seconds": round(upload_seconds, 6),
            "files_uploaded": upload.files_uploaded,
            "folders_created": upload.folders_created,
            "bytes": upload.bytes_uploaded,
            "mib_per_second": _throughput(upload.bytes_uploaded, upload_seconds),
        },
        "total_seconds": round(total_seconds, 6),
        "uploaded_items": [asdict(item) for item in upload.items],
        "report_path": str(report_path.resolve()),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return report


def _throughput(byte_count: int, seconds: float) -> float | None:
    if seconds <= 0:
        return None
    return round(byte_count / 1024**2 / seconds, 4)


def main(arguments: Iterable[str] | None = None) -> int:
    args = parse_args(arguments)
    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if any(part in run_name for part in ("/", "\\", "..")):
        print("Transfer check failed: --run-name must be a safe directory name", file=sys.stderr)
        return 2
    try:
        report = run_transfer_check(
            source=args.source,
            destination_folder_id=args.destination_folder_id,
            run_dir=(args.output_root / run_name).expanduser().resolve(),
            credentials_file=args.credentials_file,
            oauth_client_secrets_file=args.oauth_client_secrets,
            oauth_token_file=args.oauth_token_file,
            impersonate_user=args.impersonate_user,
            chunk_size=args.chunk_size_mib * 1024**2,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Transfer check failed: {error}", file=sys.stderr)
        return 1
    print(f"Download: {report['download']['seconds']:.2f}s")
    print(f"Upload:   {report['upload']['seconds']:.2f}s")
    print(f"Total:    {report['total_seconds']:.2f}s")
    print(f"Report:   {report['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
