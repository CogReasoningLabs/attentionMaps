#!/usr/bin/env python3
"""Run the batched local/Google-Drive clean corpus and EDA pipeline."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attention_maps.batch_pipeline import load_pipeline_config, run_pipeline
from attention_maps.batch_pipeline.runner import DriveAuth


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, help="Override process worker count")
    parser.add_argument("--destination-drive-folder-id", help="Override upload folder")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--credentials-file", type=Path)
    parser.add_argument("--oauth-client-secrets", type=Path)
    parser.add_argument(
        "--oauth-token-file", type=Path, default=Path(".google-drive-token.json")
    )
    parser.add_argument("--impersonate-user")
    return parser.parse_args(arguments)


def main(arguments: Iterable[str] | None = None) -> int:
    args = parse_args(arguments)
    try:
        config = load_pipeline_config(args.config)
        if args.workers is not None:
            config = replace(
                config, execution=replace(config.execution, workers=args.workers)
            )
        if args.no_upload:
            config = replace(config, destination_drive_folder_id=None)
        elif args.destination_drive_folder_id:
            config = replace(
                config,
                destination_drive_folder_id=args.destination_drive_folder_id,
            )
        auth = DriveAuth(
            credentials_file=args.credentials_file,
            oauth_client_secrets_file=args.oauth_client_secrets,
            oauth_token_file=args.oauth_token_file,
            impersonate_user=args.impersonate_user,
        )

        def progress(stage: str, completed: int, total: int | None) -> None:
            suffix = f"/{total:,}" if total is not None else ""
            print(f"[{stage}] {completed:,}{suffix}", flush=True)

        result = run_pipeline(config, drive_auth=auth, progress=progress)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Pipeline failed: {error}", file=sys.stderr)
        return 1
    print(f"Clean documents: {result.clean_documents:,}")
    print(f"Package: {result.package_path}")
    print(f"Manifest: {result.manifest_path}")
    if result.upload:
        print(f"Drive item: {result.upload.root_drive_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
