"""Authorize Google Drive and upload a local file to a folder.

This local script requires a Google OAuth client of type ``Desktop app``.
For a hosted web application using a Web application OAuth client, use
``upload_to_drive.py`` instead; see ``docs/google-drive-upload.md``.
"""

from __future__ import annotations

import argparse
import mimetypes
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


CREDENTIALS_FILE = Path("credentials.json")
TOKEN_FILE = Path("token.json")
DEFAULT_FOLDER_ID = "1fFWVdaIbOV0JVH1SQUpRAAUSY_6TtwmV"
SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def get_drive_service():
	"""Authorize this computer once, then reuse the saved refresh token."""
	credentials = None
	if TOKEN_FILE.exists():
		credentials = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

	if not credentials or not credentials.valid:
		if credentials and credentials.expired and credentials.refresh_token:
			credentials.refresh(Request())
		else:
			flow = InstalledAppFlow.from_client_secrets_file(
				str(CREDENTIALS_FILE), SCOPES
			)
			credentials = flow.run_local_server(port=0)
		TOKEN_FILE.write_text(credentials.to_json(), encoding="utf-8")

	return build("drive", "v3", credentials=credentials, cache_discovery=False)


def upload_file(file_path: Path, folder_id: str) -> dict:
	"""Upload one file and return its Google Drive metadata."""
	if not file_path.is_file():
		raise FileNotFoundError(f"File not found: {file_path}")
	if not folder_id.strip():
		raise ValueError("A Google Drive folder ID is required.")

	media = MediaFileUpload(
		str(file_path),
		mimetype=mimetypes.guess_type(file_path.name)[0]
		or "application/octet-stream",
		resumable=True,
	)
	return (
		get_drive_service()
		.files()
		.create(
			body={"name": file_path.name, "parents": [folder_id]},
			media_body=media,
			fields="id,name,mimeType,webViewLink",
			supportsAllDrives=True,
		)
		.execute()
	)


def main() -> None:
	parser = argparse.ArgumentParser(description="Upload a file to Google Drive.")
	parser.add_argument("file", nargs="?", default="data.py")
	parser.add_argument("--folder-id", default=DEFAULT_FOLDER_ID)
	args = parser.parse_args()

	result = upload_file(Path(args.file), args.folder_id)
	print(f"Uploaded: {result['name']}")
	print(f"File ID: {result['id']}")
	if result.get("webViewLink"):
		print(f"Open: {result['webViewLink']}")


if __name__ == "__main__":
	main()
