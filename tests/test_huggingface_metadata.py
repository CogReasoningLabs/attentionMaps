"""File-size lookup is independent from row and decoded-size metadata."""

import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from attention_maps.explorer.inspection import inspect_huggingface_dataset


VIEWER = "attention_maps.explorer.inspection.urllib.request.urlopen"
LOOKUP = "attention_maps.explorer.inspection._huggingface_viewer_split_size"


def known_stream(download_size=None, splits=None):
    return SimpleNamespace(
        num_shards=1, features={"text": "string"},
        info=SimpleNamespace(
            config_name="default", download_size=download_size, download_checksums=None,
            splits=splits or {"train": SimpleNamespace(num_examples=22_554, num_bytes=184_209_536)},
        ),
    )


def viewer_response():
    return io.BytesIO(json.dumps({"size": {"splits": [
        {"config": "default", "split": "train", "num_rows": 22_554,
         "num_bytes_memory": 184_209_536, "num_bytes_parquet_files": 151_890_111},
        {"config": "default", "split": "validation", "num_rows": 3_982,
         "num_bytes_parquet_files": 26_941_429},
        {"config": "gliner", "split": "train", "num_rows": 22_889,
         "num_bytes_parquet_files": 75_250_502},
    ]}}).encode())


class HuggingFaceMetadataTests(unittest.TestCase):
    def test_known_rows_still_resolve_missing_file_bytes_for_selected_split(self):
        with patch("datasets.load_dataset", return_value=known_stream()), patch(VIEWER, return_value=viewer_response()):
            inventory = inspect_huggingface_dataset("Sidharth1743/indicphi", "train")
        self.assertEqual(inventory["rows"], 22_554)
        self.assertEqual(inventory["memory_bytes"], 184_209_536)
        self.assertEqual(inventory["hub_file_bytes"], 151_890_111)
        self.assertEqual(inventory["hub_file_size_basis"], "converted Parquet")
        self.assertEqual(inventory["hub_file_size_scope"], "selected split")
        self.assertEqual(inventory["dataset_config"], "default")
        self.assertIsNone(inventory["hub_file_size_error"])

    def test_optional_file_size_failure_keeps_usable_inventory(self):
        with patch("datasets.load_dataset", return_value=known_stream()), patch(VIEWER, side_effect=OSError("Viewer unavailable")):
            inventory = inspect_huggingface_dataset("org/data", "train")
        self.assertEqual(inventory["rows"], 22_554)
        self.assertEqual(inventory["memory_bytes"], 184_209_536)
        self.assertIsNone(inventory["hub_file_bytes"])
        self.assertEqual(inventory["hub_file_size_basis"], "unavailable")
        self.assertIn("Viewer unavailable", inventory["hub_file_size_error"])

    def test_ambiguous_or_failed_viewer_does_not_break_known_row_inspection(self):
        for splits in ([], [{"config": "other", "split": "train", "num_rows": 99}]):
            response = io.BytesIO(json.dumps({"size": {"splits": splits}}).encode())
            with patch("datasets.load_dataset", return_value=known_stream()), patch(VIEWER, return_value=response):
                inventory = inspect_huggingface_dataset("org/data", "train")
            self.assertEqual(inventory["rows"], 22_554)
            self.assertIsNone(inventory["hub_file_bytes"])

    def test_known_download_size_does_not_query_viewer(self):
        with patch("datasets.load_dataset", return_value=known_stream(1234)), patch(LOOKUP) as lookup:
            inventory = inspect_huggingface_dataset("org/data", "train")
        lookup.assert_not_called()
        self.assertEqual(inventory["hub_file_bytes"], 1234)

    def test_known_download_size_for_multiple_splits_is_labelled_configuration(self):
        stream = known_stream(2000, {name: SimpleNamespace(num_examples=2, num_bytes=20) for name in ("train", "test")})
        with patch("datasets.load_dataset", return_value=stream), patch(LOOKUP) as lookup:
            inventory = inspect_huggingface_dataset("org/data", "train")
        lookup.assert_not_called()
        self.assertEqual(inventory["hub_file_size_scope"], "selected configuration")

    def test_pinned_revision_is_not_assigned_main_file_sizes(self):
        with patch("datasets.load_dataset", return_value=known_stream()), patch(LOOKUP) as lookup:
            inventory = inspect_huggingface_dataset("org/data", "train", revision="commit123")
        lookup.assert_not_called()
        self.assertIsNone(inventory["hub_file_bytes"])

    def test_original_size_is_preferred_when_viewer_provides_it(self):
        with patch("datasets.load_dataset", return_value=known_stream()), patch(LOOKUP, return_value={
            "num_rows": 22_554, "num_bytes_original_files": 194_790_963,
            "num_bytes_parquet_files": 151_890_111,
        }):
            inventory = inspect_huggingface_dataset("org/data", "train")
        self.assertEqual(inventory["hub_file_bytes"], 194_790_963)
        self.assertEqual(inventory["hub_file_size_basis"], "original files")


if __name__ == "__main__":
    unittest.main()
