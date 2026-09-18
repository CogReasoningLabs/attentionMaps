from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.get_huggingface_dataset_sizes import (
    DatasetTarget,
    export_sizes,
    human_size,
    parse_target,
    query_size,
    read_targets,
)


class _JsonResponse(io.BytesIO):
    def __enter__(self) -> "_JsonResponse":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class HuggingFaceDatasetSizeTests(unittest.TestCase):
    def test_parses_identifier_and_scoped_target(self) -> None:
        self.assertEqual(
            parse_target("himalaya-ai/nepali-news-corpus"),
            DatasetTarget("himalaya-ai/nepali-news-corpus"),
        )
        self.assertEqual(
            parse_target("uonlp/CulturaX|ne|train"),
            DatasetTarget("uonlp/CulturaX", "ne", "train"),
        )

    def test_parses_dataset_viewer_url(self) -> None:
        target = parse_target(
            "https://huggingface.co/datasets/uonlp/CulturaX/viewer/ne/train"
        )
        self.assertEqual(target, DatasetTarget("uonlp/CulturaX", "ne", "train"))

    def test_rejects_split_without_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "split requires"):
            parse_target("uonlp/CulturaX||train")

    def test_reads_list_and_ignores_comments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "datasets.txt"
            source.write_text(
                "# tracked datasets\n\nhimalaya-ai/nepali-news-corpus\n",
                encoding="utf-8",
            )
            targets = read_targets(source)
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0][1].dataset_id, "himalaya-ai/nepali-news-corpus")

    @patch("scripts.get_huggingface_dataset_sizes.urllib.request.urlopen")
    def test_queries_split_metadata_without_downloading_data(self, urlopen: object) -> None:
        payload = {
            "size": {
                "dataset": {"num_rows": 30},
                "configs": [],
                "splits": [
                    {
                        "config": "ne",
                        "split": "train",
                        "num_rows": 10,
                        "num_bytes_original_files": 1_500_000,
                        "num_bytes_parquet_files": 1_000_000,
                        "num_bytes_memory": 2_000_000,
                    }
                ],
            }
        }
        urlopen.return_value = _JsonResponse(json.dumps(payload).encode())

        result = query_size(
            DatasetTarget("uonlp/CulturaX", "ne", "train"),
            token="secret",
            timeout=12,
        )

        self.assertEqual(result["scope"], "split")
        self.assertEqual(result["num_rows"], 10)
        self.assertEqual(result["file_size"], "1.50 MB")
        request = urlopen.call_args.args[0]
        self.assertIn("dataset=uonlp%2FCulturaX", request.full_url)
        self.assertEqual(request.headers["Authorization"], "Bearer secret")

    @patch("scripts.get_huggingface_dataset_sizes.query_size")
    def test_export_continues_after_one_dataset_fails(self, query: object) -> None:
        query.side_effect = [
            {
                "scope": "dataset",
                "num_rows": 4,
                "file_size_bytes": 2_000,
                "file_size": "2.00 KB",
                "file_size_basis": "original files",
                "original_size_bytes": 2_000,
                "parquet_size_bytes": 1_500,
                "decoded_size_bytes": 3_000,
                "decoded_size": "3.00 KB",
            },
            ValueError("not found"),
        ]
        targets = [
            ("org/one", DatasetTarget("org/one")),
            ("org/two", DatasetTarget("org/two")),
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sizes.csv"
            count, errors = export_sizes(targets, output, token=None, timeout=10)
            with output.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual((count, errors), (2, 1))
        self.assertEqual(rows[0]["status"], "ok")
        self.assertEqual(rows[0]["file_size"], "2.00 KB")
        self.assertEqual(rows[1]["status"], "error")
        self.assertEqual(rows[1]["error"], "not found")

    def test_human_size_uses_spreadsheet_friendly_units(self) -> None:
        self.assertEqual(human_size(999), "999.00 B")
        self.assertEqual(human_size(1_500_000_000), "1.50 GB")
        self.assertEqual(human_size(None), "Unknown")


if __name__ == "__main__":
    unittest.main()
