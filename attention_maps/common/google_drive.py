"""Resumable Google Drive uploads for files and recursive dataset folders."""

from __future__ import annotations

import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .pipeline_logging import pipeline_logger

DRIVE_FILE_SCOPE = "https://www.googleapis.com/auth/drive.file"
DRIVE_FULL_SCOPE = "https://www.googleapis.com/auth/drive"
GOOGLE_FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
MIN_CHUNK_SIZE = 256 * 1024
DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024

LOGGER = pipeline_logger("google_drive")


class GoogleDriveUploadError(RuntimeError):
    """Raised when Drive authentication or upload fails."""


@dataclass(frozen=True)
class DriveUploadProgress:
    local_path: Path
    uploaded_bytes: int
    total_bytes: int
    completed_files: int
    total_files: int


@dataclass(frozen=True)
class DriveUploadedItem:
    local_path: Path
    drive_id: str
    name: str
    mime_type: str
    size_bytes: int
    web_view_link: str | None = None


@dataclass(frozen=True)
class DriveUploadSummary:
    source: Path
    root_drive_id: str
    files_uploaded: int
    folders_created: int
    bytes_uploaded: int
    items: tuple[DriveUploadedItem, ...]


ProgressCallback = Callable[[DriveUploadProgress], None]


def build_google_drive_service(
    *,
    credentials_file: str | Path | None = None,
    oauth_client_secrets_file: str | Path | None = None,
    oauth_token_file: str | Path = ".google-drive-token.json",
    full_drive_access: bool = False,
    impersonate_user: str | None = None,
) -> Any:
    """Build Drive v3 using ADC, service-account JSON, or desktop OAuth.

    An API key is not sufficient because uploading modifies private user or
    shared-drive data. Prefer ADC; an explicit key file is supported for teams
    that already manage service-account credentials securely.
    """

    scope = DRIVE_FULL_SCOPE if full_drive_access else DRIVE_FILE_SCOPE
    if credentials_file is not None and oauth_client_secrets_file is not None:
        raise ValueError(
            "Choose either service-account credentials or desktop OAuth, not both."
        )
    try:
        import google.auth
        from googleapiclient.discovery import build
    except ImportError as error:
        raise GoogleDriveUploadError(
            "Google Drive dependencies are missing. Install requirements.txt."
        ) from error

    try:
        if oauth_client_secrets_file is not None:
            credentials = _desktop_oauth_credentials(
                Path(oauth_client_secrets_file), Path(oauth_token_file), scope
            )
        elif credentials_file is None:
            credentials, _ = google.auth.default(scopes=[scope])
        else:
            from google.oauth2 import service_account

            credentials = service_account.Credentials.from_service_account_file(
                str(Path(credentials_file).expanduser()), scopes=[scope]
            )
        if impersonate_user:
            if not hasattr(credentials, "with_subject"):
                raise GoogleDriveUploadError(
                    "User impersonation requires service-account credentials."
                )
            credentials = credentials.with_subject(impersonate_user)
        return build("drive", "v3", credentials=credentials, cache_discovery=False)
    except GoogleDriveUploadError:
        raise
    except Exception as error:
        raise GoogleDriveUploadError(
            "Could not authenticate to Google Drive with the configured credentials."
        ) from error


def upload_path_to_google_drive(
    source: str | Path,
    parent_folder_id: str,
    *,
    credentials_file: str | Path | None = None,
    oauth_client_secrets_file: str | Path | None = None,
    oauth_token_file: str | Path = ".google-drive-token.json",
    full_drive_access: bool = False,
    impersonate_user: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    progress: ProgressCallback | None = None,
    service: Any | None = None,
    media_upload_factory: Callable[..., Any] | None = None,
) -> DriveUploadSummary:
    """Upload one file or a directory tree with resumable file transfers."""

    local_source = Path(source).expanduser().resolve()
    if not local_source.exists():
        raise ValueError(f"Upload source does not exist: {local_source}")
    if not local_source.is_file() and not local_source.is_dir():
        raise ValueError(f"Upload source must be a regular file or folder: {local_source}")
    if not parent_folder_id.strip():
        raise ValueError("A destination Google Drive folder ID is required.")
    if chunk_size < MIN_CHUNK_SIZE or chunk_size % MIN_CHUNK_SIZE:
        raise ValueError("Drive chunk size must be a positive multiple of 256 KiB.")

    drive = service or build_google_drive_service(
        credentials_file=credentials_file,
        oauth_client_secrets_file=oauth_client_secrets_file,
        oauth_token_file=oauth_token_file,
        full_drive_access=full_drive_access,
        impersonate_user=impersonate_user,
    )
    if media_upload_factory is None:
        try:
            from googleapiclient.http import MediaFileUpload
        except ImportError as error:
            raise GoogleDriveUploadError(
                "Google Drive dependencies are missing. Install requirements.txt."
            ) from error
        media_upload_factory = MediaFileUpload

    files = (
        (local_source,)
        if local_source.is_file()
        else tuple(
            sorted(
                path
                for path in local_source.rglob("*")
                if path.is_file() and not path.is_symlink()
            )
        )
    )
    total_bytes = sum(path.stat().st_size for path in files)
    LOGGER.info(
        "UPLOAD START source=%s files=%d bytes=%d destination_parent=%s",
        local_source,
        len(files),
        total_bytes,
        parent_folder_id,
    )

    uploaded: list[DriveUploadedItem] = []
    folders_created = 0
    parent_by_directory: dict[Path, str] = {}
    try:
        if local_source.is_dir():
            root_id = _create_drive_folder(drive, local_source.name, parent_folder_id)
            parent_by_directory[local_source] = root_id
            folders_created += 1
            directories = sorted(
                (
                    path
                    for path in local_source.rglob("*")
                    if path.is_dir() and not path.is_symlink()
                ),
                key=lambda path: (len(path.relative_to(local_source).parts), str(path)),
            )
            for directory in directories:
                drive_id = _create_drive_folder(
                    drive,
                    directory.name,
                    parent_by_directory[directory.parent],
                )
                parent_by_directory[directory] = drive_id
                folders_created += 1
        else:
            root_id = ""

        bytes_before = 0
        for file_index, path in enumerate(files, start=1):
            destination_parent = (
                parent_folder_id
                if local_source.is_file()
                else parent_by_directory[path.parent]
            )
            item = _upload_drive_file(
                drive,
                path,
                destination_parent,
                chunk_size,
                media_upload_factory,
                bytes_before=bytes_before,
                total_bytes=total_bytes,
                completed_files=file_index - 1,
                total_files=len(files),
                progress=progress,
            )
            uploaded.append(item)
            bytes_before += path.stat().st_size
            if local_source.is_file():
                root_id = item.drive_id
            LOGGER.info(
                "UPLOAD FILE COMPLETE file=%d/%d bytes=%d path=%s drive_id=%s",
                file_index,
                len(files),
                item.size_bytes,
                path,
                item.drive_id,
            )
            if progress:
                progress(
                    DriveUploadProgress(
                        path, bytes_before, total_bytes, file_index, len(files)
                    )
                )
    except Exception as error:
        LOGGER.exception("UPLOAD FAILED source=%s", local_source)
        if isinstance(error, (GoogleDriveUploadError, ValueError)):
            raise
        raise GoogleDriveUploadError(f"Google Drive upload failed: {error}") from error

    LOGGER.info(
        "UPLOAD COMPLETE source=%s files=%d folders=%d bytes=%d root_drive_id=%s",
        local_source,
        len(uploaded),
        folders_created,
        total_bytes,
        root_id,
    )
    return DriveUploadSummary(
        source=local_source,
        root_drive_id=root_id,
        files_uploaded=len(uploaded),
        folders_created=folders_created,
        bytes_uploaded=total_bytes,
        items=tuple(uploaded),
    )


def _desktop_oauth_credentials(
    client_secrets_file: Path,
    token_file: Path,
    scope: str,
) -> Any:
    """Authorize a local user in their browser and securely cache the token."""

    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as error:
        raise GoogleDriveUploadError(
            "Desktop OAuth dependencies are missing. Install requirements.txt."
        ) from error

    client_path = client_secrets_file.expanduser().resolve()
    token_path = token_file.expanduser().resolve()
    if not client_path.is_file():
        raise GoogleDriveUploadError(
            f"OAuth desktop-client JSON was not found: {client_path}"
        )

    credentials = None
    if token_path.is_file():
        credentials = Credentials.from_authorized_user_file(
            str(token_path), [scope]
        )
        if not credentials.has_scopes([scope]):
            credentials = None
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            LOGGER.info("OAUTH BROWSER LOGIN START scope=%s", scope)
            flow = InstalledAppFlow.from_client_secrets_file(
                str(client_path), [scope]
            )
            credentials = flow.run_local_server(port=0)
            LOGGER.info("OAUTH BROWSER LOGIN COMPLETE")
        token_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if token_path.exists():
                os.chmod(token_path, 0o600)
            descriptor = os.open(
                token_path,
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                0o600,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as token_stream:
                token_stream.write(credentials.to_json())
        except OSError:
            raise GoogleDriveUploadError(
                f"Could not securely save the OAuth token: {token_path}"
            )
    return credentials


def _create_drive_folder(service: Any, name: str, parent_id: str) -> str:
    response = (
        service.files()
        .create(
            body={
                "name": name,
                "mimeType": GOOGLE_FOLDER_MIME_TYPE,
                "parents": [parent_id],
            },
            fields="id",
            supportsAllDrives=True,
        )
        .execute()
    )
    LOGGER.info("UPLOAD FOLDER CREATED name=%s drive_id=%s", name, response["id"])
    return str(response["id"])


def _upload_drive_file(
    service: Any,
    path: Path,
    parent_id: str,
    chunk_size: int,
    media_upload_factory: Callable[..., Any],
    *,
    bytes_before: int,
    total_bytes: int,
    completed_files: int,
    total_files: int,
    progress: ProgressCallback | None,
) -> DriveUploadedItem:
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    media = media_upload_factory(
        str(path), mimetype=mime_type, chunksize=chunk_size, resumable=True
    )
    request = service.files().create(
        body={"name": path.name, "parents": [parent_id]},
        media_body=media,
        fields="id,name,mimeType,size,webViewLink",
        supportsAllDrives=True,
    )
    response = None
    last_reported = -1
    while response is None:
        status, response = request.next_chunk(num_retries=5)
        if status is not None and progress:
            resumable_progress = getattr(status, "resumable_progress", None)
            file_bytes = int(
                resumable_progress
                if resumable_progress is not None
                else status.progress() * path.stat().st_size
            )
            aggregate = min(total_bytes, bytes_before + file_bytes)
            if aggregate != last_reported:
                progress(
                    DriveUploadProgress(
                        path,
                        aggregate,
                        total_bytes,
                        completed_files,
                        total_files,
                    )
                )
                last_reported = aggregate
    return DriveUploadedItem(
        local_path=path,
        drive_id=str(response["id"]),
        name=str(response.get("name", path.name)),
        mime_type=str(response.get("mimeType", mime_type)),
        size_bytes=int(response.get("size", path.stat().st_size)),
        web_view_link=response.get("webViewLink"),
    )
