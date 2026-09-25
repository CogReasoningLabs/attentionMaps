import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from attention_maps.explorer.file_formats import describe_file_format, describe_dataset_formats
from attention_maps.explorer.huggingface import inspect_huggingface_selection
from attention_maps.explorer.inspection_runs import load_inspection_report
from tests.inspection_helpers import isolated_main as main


class FileFormatTests(unittest.TestCase):
    def test_compression_compound_extensions_and_urls_are_separate(self):
        cases = [
            ("hf://datasets/owner/data@commit/part.v1.JSONL.GZ", ".jsonl.gz", "jsonl", ["gzip"], None),
            ("https://host/data%20file.ndjson?download=.parquet#anchor", ".ndjson", "jsonl", [], None),
            ("/data/part.001.snappy.parquet", ".parquet", "parquet", [], None),
            ("/data/part.jsonl.gz.bz2", ".jsonl.gz.bz2", "jsonl", ["bzip2", "gzip"], None),
            ("/data/corpus.tar.gz", ".tar.gz", "unknown", ["gzip"], "tar"),
            ("/data/corpus.tgz", ".tgz", "unknown", ["gzip"], "tar"),
            ("/data/corpus.zip", ".zip", "unknown", [], "zip"),
            ("/data/README", "", "unknown", [], None),
            ("https://host/download?filename=data.jsonl", "", "unknown", [], None),
        ]
        for path, extension, format_name, compression, archive in cases:
            with self.subTest(path=path):
                result = describe_file_format(path)
                self.assertEqual((result["extension"], result["format"], result["compression"], result["container"]),
                                 (extension, format_name, compression, archive))
        self.assertEqual(describe_file_format("/local/data?label.jsonl")["format"], "jsonl")

    def test_batch_support_matches_actual_reader_and_exposes_record_semantics(self):
        from attention_maps.batch_pipeline.sources import SUPPORTED_SUFFIXES

        for extension in SUPPORTED_SUFFIXES:
            self.assertEqual(describe_file_format("data" + extension)["batch_loader_support"], "supported_extension")
        self.assertEqual(describe_file_format("data.jsonl.gz")["batch_loader_support"], "decompress_first")
        self.assertEqual(describe_file_format("data.tsv")["batch_loader_support"], "reader_not_implemented")
        self.assertEqual(describe_file_format("data.xlsx")["batch_loader_support"], "reader_not_implemented")
        self.assertEqual(describe_file_format("data.zip")["batch_loader_support"], "inspect_archive_members")
        self.assertEqual(describe_file_format("data.unknown")["batch_loader_support"], "unknown_format")
        self.assertEqual(describe_file_format("data.json")["batch_reader"], "json_document")
        self.assertEqual(describe_file_format("data.jsonl")["batch_reader"], "json_lines")
        self.assertIn("ONE record", describe_file_format("data.txt")["parsing_hint"])

    def test_mixed_formats_and_alias_extensions_are_not_conflated(self):
        equivalent = describe_dataset_formats([
            {"path": "a.jsonl", "bytes": 10}, {"path": "b.ndjson", "bytes": 20},
        ])
        self.assertFalse(equivalent["mixed_formats"])
        self.assertTrue(equivalent["mixed_extensions"])
        self.assertEqual(equivalent["format"], "jsonl")
        mixed = describe_dataset_formats([
            {"path": "first.parquet", "bytes": 10}, {"path": "second.jsonl.gz", "bytes": None},
            {"path": "third.jsonl.gz", "bytes": 20},
        ])
        self.assertTrue(mixed["mixed_formats"])
        self.assertEqual(mixed["format"], "mixed")
        self.assertEqual(mixed["groups"][1]["files"], 2)
        self.assertIsNone(mixed["groups"][1]["bytes"])
        self.assertTrue(mixed["requires_preparation"])

    def test_huggingface_formats_only_uses_exact_selection_without_reading_records(self):
        config = {"dataset_id": "owner/data", "revision": "pinned", "config": "ne", "schema": [],
                  "splits": {"train": {"rows": None, "memory_bytes": None, "shards": [
                      {"path": "first.parquet", "name": "first.parquet", "bytes": 100},
                      {"path": "second.jsonl.gz", "name": "second.jsonl.gz", "bytes": 40},
                  ]}}}
        with patch("attention_maps.explorer.huggingface._parquet_metadata", side_effect=AssertionError("No footer reads")), \
             patch("attention_maps.explorer.huggingface._viewer_split_metadata", side_effect=AssertionError("No row count requests")), \
             patch("datasets.load_dataset", side_effect=AssertionError("No records")):
            all_files = inspect_huggingface_selection(config, "train", metadata_only=True)
            second = inspect_huggingface_selection(config, "train", shards=["second.jsonl.gz"], metadata_only=True)
        self.assertEqual(all_files["file_formats"]["format"], "mixed")
        self.assertEqual(second["file_formats"]["format"], "jsonl")
        self.assertEqual(second["file_formats"]["extensions"], [".jsonl.gz"])
        self.assertEqual(second["bytes"], 40)
        self.assertEqual(second["file_formats"]["files"][0]["path"], "second.jsonl.gz")


class FileFormatCLITests(unittest.TestCase):
    def test_normal_inspection_reports_source_and_filtered_output_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, filtered, report_path = root / "source.ndjson", root / "kept.jsonl", root / "report.csv"
            source.write_text(json.dumps({"text": "नेपाल"}) + "\n")
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["--local", str(source), "--min-devanagari-ratio", "1",
                             "--filtered-output", str(filtered), "--output", str(report_path)])
            self.assertEqual(code, 0)
            report = load_inspection_report(report_path)
            self.assertEqual(report["inventory"]["file_formats"]["extensions"], [".ndjson"])
            self.assertEqual(report["filter_result"]["file_formats"]["extensions"], [".jsonl"])

    def test_formats_only_catalogs_mixed_local_files_without_parsing_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            for filename in ("records.jsonl.gz", "records.csv", "records.parquet", "archive.zip", "unlabelled"):
                (source / filename).write_text("deliberately not valid data for this extension")
            output = source / "report.csv"
            with patch("scripts.inspect_dataset.inspect_dataset", side_effect=AssertionError("No parsing")), \
                 patch("attention_maps.explorer.inspection_runs.sample_dataset_rows", side_effect=AssertionError("No sampling")), \
                 contextlib.redirect_stdout(io.StringIO()):
                for _ in range(2):  # Repeated runs exclude their own report from the source directory.
                    self.assertEqual(main(["--local", str(source), "--formats-only", "--output", str(output)]), 0)
            report = load_inspection_report(output)
            self.assertEqual(report["inventory"]["files"], 5)
            self.assertIsNone(report["inventory"]["rows"])
            self.assertEqual(report["sample_scope"], "metadata_only")
            self.assertEqual(report["language_status"]["sampled_records"], 0)
            self.assertTrue(report["inventory"]["file_formats"]["mixed_formats"])

    def test_directory_report_path_cannot_overwrite_an_input_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename, flags in (("input.json", ["--formats-only"]), ("input.parquet", [])):
                source = root / filename
                source.write_text('[{"text": "source data"}]')
                original = source.read_bytes()
                with contextlib.redirect_stderr(io.StringIO()) as errors:
                    self.assertEqual(main(["--local", str(root), "--output", str(source), *flags]), 1)
                self.assertRegex(errors.getvalue(), "must not overwrite a source file|output must be a .csv")
                self.assertEqual(source.read_bytes(), original)

    def test_formats_only_settings_reject_filtering(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Path(directory) / "settings.yaml"
            settings.write_text("local: data.jsonl\nformats_only: true\nmin_devanagari_ratio: 0.8\nfiltered_output: kept.jsonl\n")
            with contextlib.redirect_stderr(io.StringIO()) as errors:
                self.assertEqual(main(["--settings", str(settings)]), 1)
            self.assertIn("cannot be combined with filtering", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
