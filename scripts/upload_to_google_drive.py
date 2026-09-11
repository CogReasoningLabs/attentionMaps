#!/usr/bin/env python3
"""Upload a dataset file or folder to Google Drive with resumable transfers."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attention_maps.common.google_drive import (
    DEFAULT_CHUNK_SIZE,
    DriveUploadProgress,
    upload_path_to_google_drive,
)
from attention_maps.common.pipeline_logging import pipeline_logger

LOGGER = pipeline_logger("drive_upload_cli")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Local file or folder to upload")
    parser.add_argument(
        "--parent-folder-id",
        default=os.getenv("ATTENTION_MAPS_DRIVE_PARENT_ID", ""),
        help="Destination Drive folder ID (or ATTENTION_MAPS_DRIVE_PARENT_ID)",
    )
    parser.add_argument(
        "--credentials-file",
        type=Path,
        help="Optional service-account JSON; otherwise use ADC",
    )
    parser.add_argument(
        "--oauth-client-secrets",
        type=Path,
        help="Desktop OAuth client JSON; opens a browser on first use",
    )
    parser.add_argument(
        "--oauth-token-file",
        type=Path,
        default=Path(".google-drive-token.json"),
        help="Local OAuth token cache (default: .google-drive-token.json)",
    )
    parser.add_argument(
        "--chunk-size-mib",
        type=int,
        default=DEFAULT_CHUNK_SIZE // 1024**2,
        help="Resumable upload chunk size in MiB (default: 8)",
    )
    parser.add_argument(
        "--full-drive-access",
        action="store_true",
        help="Request full Drive scope instead of the default drive.file scope",
    )
    parser.add_argument(
        "--impersonate-user",
        help="Workspace user email for configured domain-wide delegation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.parent_folder_id:
        raise SystemExit(
            "Set --parent-folder-id or ATTENTION_MAPS_DRIVE_PARENT_ID."
        )

    last_percent = -1

    def report(update: DriveUploadProgress) -> None:
        nonlocal last_percent
        percent = round(100 * update.uploaded_bytes / max(1, update.total_bytes))
        if percent != last_percent:
            LOGGER.info(
                "UPLOAD PROGRESS percent=%d files=%d/%d current=%s",
                percent,
                update.completed_files,
                update.total_files,
                update.local_path,
            )
            last_percent = percent

    result = upload_path_to_google_drive(
        args.source,
        args.parent_folder_id,
        credentials_file=args.credentials_file,
        oauth_client_secrets_file=args.oauth_client_secrets,
        oauth_token_file=args.oauth_token_file,
        full_drive_access=args.full_drive_access,
        impersonate_user=args.impersonate_user,
        chunk_size=args.chunk_size_mib * 1024**2,
        progress=report,
    )
    LOGGER.info(
        "DONE root_drive_id=%s files=%d folders=%d bytes=%d",
        result.root_drive_id,
        result.files_uploaded,
        result.folders_created,
        result.bytes_uploaded,
    )


if __name__ == "__main__":
    main()
