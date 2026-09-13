import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from attention_maps.common.google_drive import (
    DriveDownloadSummary,
    DriveUploadSummary,
)


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_google_drive_transfer.py"
SPEC = importlib.util.spec_from_file_location("check_google_drive_transfer", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class GoogleDriveTransferCheckTests(unittest.TestCase):
    def test_records_independent_download_upload_and_total_timings(self):
        calls = []

        def download(source, destination, **kwargs):
            calls.append(("download", source, kwargs["chunk_size"]))
            destination.mkdir(parents=True)
            (destination / "data.bin").write_bytes(b"x" * 1024**2)
            return DriveDownloadSummary(source, destination, 1, 0, 1024**2)

        def upload(source, destination, **kwargs):
            calls.append(("upload", destination, kwargs["chunk_size"]))
            return DriveUploadSummary(source, "uploaded-root", 1, 1, 1024**2, ())

        values = iter((0.0, 0.0, 2.0, 2.0, 5.0, 5.0))
        with tempfile.TemporaryDirectory() as directory:
            report = MODULE.run_transfer_check(
                source="source-id",
                destination_folder_id="destination-id",
                run_dir=Path(directory) / "run",
                chunk_size=8 * 1024**2,
                download_func=download,
                upload_func=upload,
                clock=lambda: next(values),
            )
            persisted = json.loads(
                Path(report["report_path"]).read_text(encoding="utf-8")
            )

        self.assertEqual([call[0] for call in calls], ["download", "upload"])
        self.assertEqual(report["download"]["seconds"], 2.0)
        self.assertEqual(report["download"]["mib_per_second"], 0.5)
        self.assertEqual(report["upload"]["seconds"], 3.0)
        self.assertEqual(report["total_seconds"], 5.0)
        self.assertEqual(persisted["uploaded_root_drive_id"], "uploaded-root")


if __name__ == "__main__":
    unittest.main()
