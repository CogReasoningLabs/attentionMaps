"""Reusable Google Drive uploader for a web-server OAuth client.

Keep ``credentials.json`` and the generated ``token.json`` on the server only.
They must never be sent to a browser or committed to Git.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import Resource, build
from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload


# Lets the app create and manage the files it uploads, without all-Drive access.
DRIVE_FILE_SCOPE = "https://www.googleapis.com/auth/drive.file"
SCOPES = [DRIVE_FILE_SCOPE]
DEFAULT_CREDENTIALS_FILE = Path("credentials.json")
DEFAULT_TOKEN_FILE = Path("token.json")


def get_authorization_url(
    redirect_uri: str,
    state: str,
    *,
    credentials_file: str | Path = DEFAULT_CREDENTIALS_FILE,
) -> str:
    """Return the Google consent URL for a *Web application* OAuth client.

    ``redirect_uri`` must exactly match an Authorized redirect URI in Google
    Cloud. Generate a cryptographically random ``state`` per browser session,
    save it in that session, and verify it in the callback before exchanging
    the code.
    """
    flow = Flow.from_client_secrets_file(str(credentials_file), scopes=SCOPES)
    flow.redirect_uri = redirect_uri
    authorization_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
        state=state,
    )
    return authorization_url


def save_credentials_from_callback(
    authorization_response: str,
    redirect_uri: str,
    state: str,
    *,
    credentials_file: str | Path = DEFAULT_CREDENTIALS_FILE,
    token_file: str | Path = DEFAULT_TOKEN_FILE,
) -> Credentials:
    """Exchange a complete callback URL for tokens and save them server-side."""
    flow = Flow.from_client_secrets_file(
        str(credentials_file), scopes=SCOPES, state=state
    )
    flow.redirect_uri = redirect_uri
    flow.fetch_token(authorization_response=authorization_response)
    credentials = flow.credentials
    _write_token(credentials, token_file)
    return credentials


def get_drive_service(*, token_file: str | Path = DEFAULT_TOKEN_FILE) -> Resource:
    """Build an authenticated Drive v3 service using a saved refresh token."""
    token_path = Path(token_file)
    if not token_path.exists():
        raise RuntimeError(
            "Google Drive is not connected. Complete the web OAuth callback first."
        )

    credentials = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if not credentials.valid:
        if not credentials.expired or not credentials.refresh_token:
            raise RuntimeError("Saved Google credentials are invalid; authorize again.")
        credentials.refresh(Request())
        _write_token(credentials, token_path)

    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def upload_file(
    file_path: str | Path,
    *,
    folder_id: str,
    name: str | None = None,
    mime_type: str | None = None,
    token_file: str | Path = DEFAULT_TOKEN_FILE,
) -> dict[str, Any]:
    """Upload one local file into ``folder_id`` and return its Drive metadata."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Upload file does not exist: {path}")

    media = MediaFileUpload(
        str(path), mimetype=mime_type or mimetypes.guess_type(path.name)[0], resumable=True
    )
    return _create_file(name=name or path.name, folder_id=folder_id, media=media, token_file=token_file)


def upload_bytes(
    content: bytes,
    *,
    name: str,
    folder_id: str,
    mime_type: str = "application/octet-stream",
    token_file: str | Path = DEFAULT_TOKEN_FILE,
) -> dict[str, Any]:
    """Upload data created in memory, for example CSV or JSON from your app."""
    from io import BytesIO

    media = MediaIoBaseUpload(BytesIO(content), mimetype=mime_type, resumable=True)
    return _create_file(name=name, folder_id=folder_id, media=media, token_file=token_file)


def upload_text(
    content: str,
    *,
    name: str,
    folder_id: str,
    mime_type: str = "text/plain; charset=utf-8",
    token_file: str | Path = DEFAULT_TOKEN_FILE,
) -> dict[str, Any]:
    """Upload UTF-8 text created in memory."""
    return upload_bytes(content.encode("utf-8"), name=name, folder_id=folder_id,
                        mime_type=mime_type, token_file=token_file)


def _create_file(*, name: str, folder_id: str, media: Any, token_file: str | Path) -> dict[str, Any]:
    if not folder_id.strip():
        raise ValueError("folder_id is required (copy it from the Google Drive folder URL).")
    return (
        get_drive_service(token_file=token_file).files().create(
            body={"name": name, "parents": [folder_id]}, media_body=media,
            fields="id,name,mimeType,webViewLink,parents", supportsAllDrives=True,
        ).execute()
    )


def _write_token(credentials: Credentials, token_file: str | Path) -> None:
    path = Path(token_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(credentials.to_json(), encoding="utf-8")


def main() -> None:
    """Upload a file after the web app has completed OAuth once."""
    parser = argparse.ArgumentParser(description="Upload a local file to a Google Drive folder.")
    parser.add_argument("file", help="Path of the local file to upload")
    parser.add_argument("--folder-id", required=True, help="Target Google Drive folder ID")
    parser.add_argument("--name", help="Optional name to use in Google Drive")
    parser.add_argument("--token-file", default=str(DEFAULT_TOKEN_FILE))
    args = parser.parse_args()
    print(json.dumps(upload_file(args.file, folder_id=args.folder_id, name=args.name,
                                 token_file=args.token_file), indent=2))


if __name__ == "__main__":
    main()
