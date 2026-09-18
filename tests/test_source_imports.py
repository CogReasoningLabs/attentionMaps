import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from attention_maps.explorer import (
    discover_import_files,
    file_signatures,
    huggingface_source_spec,
    inspect_dataset,
    normalize_huggingface_id,
    normalize_kaggle_handle,
    parse_s3_uri,
    sample_dataset_rows,
    stage_kaggle_source,
    stage_s3_object,
    staged_source_spec,
)


class SourceImportTests(unittest.TestCase):
    def test_validates_provider_identifiers(self):
        self.assertEqual(normalize_huggingface_id("owner/data"), "owner/data")
        self.assertEqual(
            normalize_kaggle_handle("owner/data/versions/3"),
            "owner/data/versions/3",
        )
        self.assertEqual(
            parse_s3_uri("s3://bucket/path/data.jsonl"),
            ("bucket", "path/data.jsonl"),
        )
        with self.assertRaises(ValueError):
            normalize_huggingface_id("https://huggingface.co/datasets/owner/data")
        with self.assertRaises(ValueError):
            normalize_kaggle_handle("dataset-only")
        with self.assertRaises(ValueError):
            parse_s3_uri("s3://bucket/folder/")

    def test_builds_revision_pinned_huggingface_spec(self):
        spec = huggingface_source_spec(
            "owner/data",
            schema="task_specific_supervised",
            config="default",
            split="validation",
            revision="abc123",
        )
        self.assertEqual(spec.dataset_revision, "abc123")
        self.assertEqual(spec.primary_purpose, "Task-specific fine-tuning")
        self.assertIn("@abc123", str(spec.location))

    def test_discovers_and_builds_staged_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "records.csv"
            csv_path.write_text("text,label\na,0\nb,1\n", encoding="utf-8")
            (root / "ignored.bin").write_bytes(b"x")

            self.assertEqual(discover_import_files(root), (csv_path,))
            spec = staged_source_spec(
                csv_path,
                source_type="S3",
                schema="task_specific_supervised",
                source_uri="s3://bucket/records.csv",
            )

        self.assertEqual(spec.format, "csv")
        self.assertEqual(spec.location, "s3://bucket/records.csv")

    def test_inspects_and_reservoir_samples_streaming_flat_files(self):
        fixtures = {
            "rows.csv": "text,label\na,0\nb,1\nc,0\n",
            "rows.jsonl": (
                '{"text":"a","label":0}\n'
                '{"text":"b","label":1}\n'
                '{"text":"c","label":0}\n'
            ),
            "rows.txt": "a\n\nb\nc\n",
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, content in fixtures.items():
                path = Path(directory) / name
                path.write_text(content, encoding="utf-8")
                inventory = inspect_dataset(file_signatures((path,)))
                columns = tuple(inventory["columns"])
                first = sample_dataset_rows(inventory, 2, 42, columns)
                second = sample_dataset_rows(inventory, 2, 42, columns)
                self.assertEqual(inventory["rows"], 3)
                self.assertEqual(first, second)
                self.assertEqual(len(first), 2)

    def test_stages_kaggle_via_standard_handle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            path.touch()
            with patch("kagglehub.dataset_download", return_value=str(path)) as loader:
                result = stage_kaggle_source("owner/dataset", "data.csv")
        self.assertEqual(result, path.resolve())
        loader.assert_called_once_with("owner/dataset", path="data.csv")

    def test_stages_s3_object_and_reuses_matching_cache(self):
        class FakeS3:
            downloads = 0

            def head_object(self, *, Bucket, Key):
                self.request = (Bucket, Key)
                return {"ContentLength": 4}

            def download_file(self, bucket, key, destination):
                self.downloads += 1
                Path(destination).write_bytes(b"data")

        with tempfile.TemporaryDirectory() as directory:
            client = FakeS3()
            first = stage_s3_object(
                "s3://bucket/path/data.csv",
                cache_root=Path(directory),
                client=client,
            )
            second = stage_s3_object(
                "s3://bucket/path/data.csv",
                cache_root=Path(directory),
                client=client,
            )
        self.assertEqual(first, second)
        self.assertEqual(client.request, ("bucket", "path/data.csv"))
        self.assertEqual(client.downloads, 1)


if __name__ == "__main__":
    unittest.main()
