"""Resumable Google Drive uploads for files and recursive dataset folders."""

from __future__ import annotations

import mimetypes
import os
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from .pipeline_logging import pipeline_logger

DRIVE_FILE_SCOPE = "https://www.googleapis.com/auth/drive.file"
DRIVE_FULL_SCOPE = "https://www.googleapis.com/auth/drive"
GOOGLE_FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
GOOGLE_NATIVE_MIME_PREFIX = "application/vnd.google-apps."
GOOGLE_NATIVE_EXPORTS = {
    "application/vnd.google-apps.document": ("application/pdf", ".pdf"),
    "application/vnd.google-apps.spreadsheet": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xlsx",
    ),
    "application/vnd.google-apps.presentation": ("application/pdf", ".pdf"),
    "application/vnd.google-apps.drawing": ("application/pdf", ".pdf"),
    "application/vnd.google-apps.script": (
        "application/vnd.google-apps.script+json",
        ".json",
    ),
}
MIN_CHUNK_SIZE = 256 * 1024
DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024

LOGGER = pipeline_logger("google_drive")


class GoogleDriveUploadError(RuntimeError):
    """Raised when Drive authentication or upload fails."""


class GoogleDriveDownloadError(RuntimeError):
    """Raised when a Drive file or folder cannot be downloaded safely."""


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
    md5_checksum: str | None = None


@dataclass(frozen=True)
class DriveUploadSummary:
    source: Path
    root_drive_id: str
    files_uploaded: int
    folders_created: int
    bytes_uploaded: int
    items: tuple[DriveUploadedItem, ...]


@dataclass(frozen=True)
class DriveDownloadProgress:
    local_path: Path
    downloaded_bytes: int
    total_bytes: int
    completed_files: int
    total_files: int


@dataclass(frozen=True)
class DriveDownloadSummary:
    source_drive_id: str
    destination: Path
    files_downloaded: int
    files_reused: int
    bytes_downloaded: int


ProgressCallback = Callable[[DriveUploadProgress], None]
DownloadProgressCallback = Callable[[DriveDownloadProgress], None]


def extract_google_drive_id(value: str) -> str:
    """Accept a raw Drive ID or a standard file/folder sharing URL."""

    candidate = value.strip()
    if not candidate:
        raise ValueError("A Google Drive file/folder ID or URL is required.")
    if "://" not in candidate:
        return candidate
    parsed = urlparse(candidate)
    query_id = parse_qs(parsed.query).get("id", ())
    if query_id and query_id[0].strip():
        return query_id[0].strip()
    match = re.search(r"/(?:folders|d)/([^/?#]+)", parsed.path)
    if match:
        return match.group(1)
    raise ValueError("Could not extract a Google Drive ID from the supplied URL.")


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


def download_google_drive_path(
    source: str,
    destination: str | Path,
    *,
    credentials_file: str | Path | None = None,
    oauth_client_secrets_file: str | Path | None = None,
    oauth_token_file: str | Path = ".google-drive-token.json",
    full_drive_access: bool = True,
    impersonate_user: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    progress: DownloadProgressCallback | None = None,
    service: Any | None = None,
    media_download_factory: Callable[..., Any] | None = None,
) -> DriveDownloadSummary:
    """Recursively download one Drive file/folder with retryable chunks.

    A partially written ``.part`` file is resumed on the next invocation. Existing
    completed files with the expected byte size are reused.
    """

    source_id = extract_google_drive_id(source)
    target = Path(destination).expanduser().resolve()
    if chunk_size < MIN_CHUNK_SIZE or chunk_size % MIN_CHUNK_SIZE:
        raise ValueError("Drive chunk size must be a positive multiple of 256 KiB.")
    drive = service or build_google_drive_service(
        credentials_file=credentials_file,
        oauth_client_secrets_file=oauth_client_secrets_file,
        oauth_token_file=oauth_token_file,
        full_drive_access=full_drive_access,
        impersonate_user=impersonate_user,
    )
    if media_download_factory is None:
        try:
            from googleapiclient.http import MediaIoBaseDownload
        except ImportError as error:
            raise GoogleDriveDownloadError(
                "Google Drive dependencies are missing. Install requirements.txt."
            ) from error
        media_download_factory = MediaIoBaseDownload

    try:
        root = _drive_metadata(drive, source_id)
        if root.get("mimeType") != GOOGLE_FOLDER_MIME_TYPE:
            root = _downloadable_drive_item(root)
        inventory = _drive_download_inventory(drive, root)
        total_bytes = sum(int(item.get("size", 0)) for item, _ in inventory)
        target.mkdir(parents=True, exist_ok=True)
        downloaded = reused = downloaded_bytes = 0
        assigned_paths: set[Path] = set()
        for index, (item, relative_parent) in enumerate(inventory, start=1):
            parent = target / relative_parent
            parent.mkdir(parents=True, exist_ok=True)
            name = _safe_drive_name(str(item["name"]), str(item["id"]))
            local_path = parent / name
            if local_path in assigned_paths:
                local_path = parent / (
                    f"{local_path.stem}-{item['id']}{local_path.suffix}"
                )
            assigned_paths.add(local_path)
            expected_size = int(item.get("size", 0))
            if _matches_drive_file(local_path, item):
                reused += 1
            else:
                _download_drive_file(
                    drive,
                    item,
                    local_path,
                    chunk_size,
                    media_download_factory,
                    bytes_before=downloaded_bytes,
                    total_bytes=total_bytes,
                    completed_files=index - 1,
                    total_files=len(inventory),
                    progress=progress,
                )
                downloaded += 1
            downloaded_bytes += local_path.stat().st_size
            if progress:
                effective_total = max(total_bytes, downloaded_bytes)
                progress(
                    DriveDownloadProgress(
                        local_path,
                        downloaded_bytes,
                        effective_total,
                        index,
                        len(inventory),
                    )
                )
        return DriveDownloadSummary(
            source_drive_id=source_id,
            destination=target,
            files_downloaded=downloaded,
            files_reused=reused,
            bytes_downloaded=downloaded_bytes,
        )
    except Exception as error:
        if isinstance(error, (GoogleDriveDownloadError, ValueError)):
            raise
        raise GoogleDriveDownloadError(f"Google Drive download failed: {error}") from error


def _drive_metadata(service: Any, file_id: str) -> dict[str, Any]:
    return dict(
        service.files()
        .get(
            fileId=file_id,
            fields="id,name,mimeType,size,md5Checksum",
            supportsAllDrives=True,
        )
        .execute()
    )


def _drive_download_inventory(
    service: Any, root: dict[str, Any]
) -> list[tuple[dict[str, Any], Path]]:
    if root.get("mimeType") != GOOGLE_FOLDER_MIME_TYPE:
        return [(root, Path())]
    files: list[tuple[dict[str, Any], Path]] = []

    def visit(folder_id: str, relative: Path) -> None:
        token = None
        while True:
            response = (
                service.files()
                .list(
                    q=f"'{folder_id}' in parents and trashed = false",
                    fields="nextPageToken,files(id,name,mimeType,size,md5Checksum)",
                    pageToken=token,
                    pageSize=1_000,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
            for item in sorted(response.get("files", ()), key=lambda row: row["name"]):
                if item.get("mimeType") == GOOGLE_FOLDER_MIME_TYPE:
                    folder = _safe_drive_name(str(item["name"]), str(item["id"]))
                    visit(str(item["id"]), relative / folder)
                else:
                    files.append((_downloadable_drive_item(item), relative))
            token = response.get("nextPageToken")
            if not token:
                return

    visit(str(root["id"]), Path())
    return files


def _downloadable_drive_item(item: dict[str, Any]) -> dict[str, Any]:
    """Return blob metadata or add a deterministic native-file export format."""

    downloadable = dict(item)
    mime_type = str(downloadable.get("mimeType", ""))
    if not mime_type.startswith(GOOGLE_NATIVE_MIME_PREFIX):
        return downloadable
    export = GOOGLE_NATIVE_EXPORTS.get(mime_type)
    if export is None:
        raise GoogleDriveDownloadError(
            f"Native Google file {downloadable.get('name', downloadable.get('id'))!r} "
            f"has unsupported type {mime_type!r}; export it to a regular file first."
        )
    export_mime_type, extension = export
    name = str(downloadable.get("name", downloadable.get("id", "drive-item")))
    if not name.lower().endswith(extension):
        downloadable["name"] = name + extension
    downloadable["_export_mime_type"] = export_mime_type
    # Drive reports neither the exported byte size nor a checksum in metadata.
    downloadable.pop("size", None)
    downloadable.pop("md5Checksum", None)
    return downloadable


def _safe_drive_name(name: str, drive_id: str) -> str:
    safe = Path(name).name.replace("\x00", "").strip()
    if safe in {"", ".", ".."}:
        safe = f"drive-item-{drive_id}"
    return safe


def _download_drive_file(
    service: Any,
    item: dict[str, Any],
    path: Path,
    chunk_size: int,
    media_download_factory: Callable[..., Any],
    *,
    bytes_before: int,
    total_bytes: int,
    completed_files: int,
    total_files: int,
    progress: DownloadProgressCallback | None,
) -> None:
    partial = path.with_name(path.name + ".part")
    expected_size = int(item.get("size", 0))
    if expected_size and partial.exists() and partial.stat().st_size > expected_size:
        partial.unlink()
    offset = partial.stat().st_size if partial.exists() else 0
    export_mime_type = item.get("_export_mime_type")
    if export_mime_type:
        request = service.files().export_media(
            fileId=str(item["id"]), mimeType=str(export_mime_type)
        )
    else:
        request = service.files().get_media(
            fileId=str(item["id"]), supportsAllDrives=True
        )
    with partial.open("ab") as stream:
        downloader = media_download_factory(stream, request, chunksize=chunk_size)
        if offset:
            # MediaIoBaseDownload exposes no public resume hook, but its range
            # requests are driven by this progress offset.
            downloader._progress = offset
        done = offset == expected_size and expected_size > 0
        while not done:
            status, done = downloader.next_chunk(num_retries=5)
            if status is not None and progress:
                current = int(getattr(status, "resumable_progress", 0))
                progress(
                    DriveDownloadProgress(
                        path,
                        min(total_bytes, bytes_before + current),
                        total_bytes,
                        completed_files,
                        total_files,
                    )
                )
    if expected_size and partial.stat().st_size != expected_size:
        raise GoogleDriveDownloadError(
            f"Downloaded size mismatch for {item['name']!r}: "
            f"expected {expected_size}, got {partial.stat().st_size}."
        )
    os.replace(partial, path)
    if not _matches_drive_file(path, item):
        path.unlink(missing_ok=True)
        raise GoogleDriveDownloadError(
            f"Checksum validation failed for {item['name']!r}."
        )


def _matches_drive_file(path: Path, item: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    if item.get("_export_mime_type"):
        return path.stat().st_size > 0
    expected_size = int(item.get("size", 0))
    if expected_size and path.stat().st_size != expected_size:
        return False
    expected_md5 = item.get("md5Checksum")
    if not expected_md5:
        return not expected_size or path.stat().st_size == expected_size
    return _file_md5(path) == expected_md5


def _file_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


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
        fields="id,name,mimeType,size,md5Checksum,webViewLink",
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
    response_size = int(response.get("size", path.stat().st_size))
    if response_size != path.stat().st_size:
        raise GoogleDriveUploadError(
            f"Uploaded size mismatch for {path.name!r}: "
            f"local={path.stat().st_size}, remote={response_size}."
        )
    remote_md5 = response.get("md5Checksum")
    if remote_md5 and _file_md5(path) != remote_md5:
        raise GoogleDriveUploadError(
            f"Uploaded checksum mismatch for {path.name!r}."
        )
    return DriveUploadedItem(
        local_path=path,
        drive_id=str(response["id"]),
        name=str(response.get("name", path.name)),
        mime_type=str(response.get("mimeType", mime_type)),
        size_bytes=response_size,
        web_view_link=response.get("webViewLink"),
        md5_checksum=remote_md5,
    )
