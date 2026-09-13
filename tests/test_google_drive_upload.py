import tempfile
import unittest
from pathlib import Path

from attention_maps.common.google_drive import (
    MIN_CHUNK_SIZE,
    build_google_drive_service,
    download_google_drive_path,
    extract_google_drive_id,
    upload_path_to_google_drive,
)


class _ExecuteRequest:
    def __init__(self, response):
        self.response = response

    def execute(self):
        return self.response


class _UploadRequest:
    def __init__(self, response):
        self.response = response

    def next_chunk(self, num_retries):
        assert num_retries == 5
        return None, self.response


class _FilesResource:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        body = kwargs["body"]
        item_id = f"item-{len(self.calls)}"
        if "media_body" not in kwargs:
            return _ExecuteRequest({"id": item_id})
        return _UploadRequest(
            {
                "id": item_id,
                "name": body["name"],
                "mimeType": "application/octet-stream",
                "size": "4",
                "webViewLink": f"https://drive.example/{item_id}",
            }
        )


class _DriveService:
    def __init__(self):
        self.resource = _FilesResource()

    def files(self):
        return self.resource


class GoogleDriveUploadTests(unittest.TestCase):
    def test_extracts_file_and_folder_ids_from_sharing_links(self):
        self.assertEqual(
            extract_google_drive_id("https://drive.google.com/drive/folders/folder-1"),
            "folder-1",
        )
        self.assertEqual(
            extract_google_drive_id("https://drive.google.com/file/d/file-1/view"),
            "file-1",
        )

    def test_rejects_multiple_authentication_modes(self):
        with self.assertRaisesRegex(ValueError, "either service-account"):
            build_google_drive_service(
                credentials_file="service-account.json",
                oauth_client_secrets_file="desktop-client.json",
            )

    def test_recursively_uploads_folder_and_preserves_hierarchy(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "clean-run"
            nested = root / "eda"
            nested.mkdir(parents=True)
            (root / "clean.jsonl").write_text("data", encoding="utf-8")
            (nested / "audit.csv").write_text("data", encoding="utf-8")
            service = _DriveService()
            media_calls = []
            updates = []

            def media_factory(path, **kwargs):
                media_calls.append((path, kwargs))
                return object()

            summary = upload_path_to_google_drive(
                root,
                "team-parent",
                service=service,
                media_upload_factory=media_factory,
                progress=updates.append,
            )

        self.assertEqual(summary.files_uploaded, 2)
        self.assertEqual(summary.folders_created, 2)
        self.assertEqual(summary.bytes_uploaded, 8)
        self.assertEqual(summary.root_drive_id, "item-1")
        self.assertEqual(len(media_calls), 2)
        self.assertTrue(all(call[1]["resumable"] for call in media_calls))
        self.assertTrue(
            all(call[1]["chunksize"] == 8 * 1024**2 for call in media_calls)
        )
        self.assertTrue(
            all(call["supportsAllDrives"] for call in service.resource.calls)
        )
        file_calls = [
            call for call in service.resource.calls if "media_body" in call
        ]
        self.assertEqual(file_calls[0]["body"]["parents"], ["item-1"])
        self.assertEqual(file_calls[1]["body"]["parents"], ["item-2"])
        self.assertEqual(updates[-1].completed_files, 2)

    def test_rejects_invalid_chunk_size_before_authentication(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "data.jsonl"
            source.write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "multiple of 256 KiB"):
                upload_path_to_google_drive(
                    source,
                    "parent",
                    chunk_size=MIN_CHUNK_SIZE + 1,
                    service=_DriveService(),
                    media_upload_factory=lambda *args, **kwargs: object(),
                )

    def test_downloads_a_drive_file_to_a_bounded_local_target(self):
        class DownloadFiles:
            def get(self, **kwargs):
                return _ExecuteRequest(
                    {
                        "id": kwargs["fileId"],
                        "name": "dataset.jsonl",
                        "mimeType": "application/json",
                        "size": "4",
                    }
                )

            def get_media(self, **kwargs):
                return kwargs

        class DownloadService:
            def files(self):
                return DownloadFiles()

        class Downloader:
            def __init__(self, stream, request, chunksize):
                self.stream = stream
                self._progress = 0

            def next_chunk(self, num_retries):
                self.stream.write(b"data")
                return None, True

        with tempfile.TemporaryDirectory() as directory:
            result = download_google_drive_path(
                "file-1",
                Path(directory) / "input",
                service=DownloadService(),
                media_download_factory=Downloader,
            )
            content = (result.destination / "dataset.jsonl").read_bytes()

        self.assertEqual(content, b"data")
        self.assertEqual(result.files_downloaded, 1)

    def test_exports_a_native_google_document_as_pdf(self):
        class DownloadFiles:
            def __init__(self):
                self.export_calls = []

            def get(self, **kwargs):
                return _ExecuteRequest(
                    {
                        "id": kwargs["fileId"],
                        "name": "Survey Notes",
                        "mimeType": "application/vnd.google-apps.document",
                    }
                )

            def export_media(self, **kwargs):
                self.export_calls.append(kwargs)
                return kwargs

        class DownloadService:
            def __init__(self):
                self.resource = DownloadFiles()

            def files(self):
                return self.resource

        class Downloader:
            def __init__(self, stream, request, chunksize):
                self.stream = stream

            def next_chunk(self, num_retries):
                self.stream.write(b"%PDF")
                return None, True

        service = DownloadService()
        with tempfile.TemporaryDirectory() as directory:
            result = download_google_drive_path(
                "native-doc-1",
                Path(directory) / "input",
                service=service,
                media_download_factory=Downloader,
            )
            content = (result.destination / "Survey Notes.pdf").read_bytes()

        self.assertEqual(content, b"%PDF")
        self.assertEqual(result.bytes_downloaded, 4)
        self.assertEqual(
            service.resource.export_calls,
            [{"fileId": "native-doc-1", "mimeType": "application/pdf"}],
        )


if __name__ == "__main__":
    unittest.main()
