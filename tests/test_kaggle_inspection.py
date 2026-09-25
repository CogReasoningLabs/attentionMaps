import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from attention_maps.explorer.inspection_runs import load_inspection_report
from attention_maps.explorer.kaggle_inspection import discover_kaggle_dataset
from scripts.inspect_dataset import _arguments
from tests.inspection_helpers import isolated_main as main


class KaggleInspectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.report_path = self.root / "report.csv"

    def catalog(self, *names):
        return {"provider": "kaggle", "dataset_id": "owner/data", "dataset_handle": "owner/data/versions/7",
                "dataset_version": 7, "files": [
                    {"name": name, "path": f"kaggle://datasets/owner/data/versions/7/{name}", "bytes": 100}
                    for name in names]}

    def run_cli(self, *flags):
        with contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(io.StringIO()) as errors:
            code = main(["--provider", "kaggle", "--dataset", "owner/data", "--output", str(self.report_path), *flags])
        return code, output.getvalue(), errors.getvalue()

    def test_discovery_pins_version_and_lists_all_metadata_pages_without_download(self):
        client = MagicMock()
        api = client.__enter__.return_value.datasets.dataset_api_client
        api.get_dataset.return_value = SimpleNamespace(current_version_number=7)
        api.list_dataset_files.side_effect = [
            SimpleNamespace(dataset_files=[SimpleNamespace(name="first.csv", total_bytes=10)], error_message=None, next_page_token="page-2"),
            SimpleNamespace(dataset_files=[SimpleNamespace(name="later.jsonl", total_bytes=20)], error_message=None, next_page_token=None),
        ]
        with patch("kagglehub.clients.build_kaggle_client", return_value=client), \
             patch("kagglehub.dataset_download", side_effect=AssertionError("Metadata must not download")):
            catalog = discover_kaggle_dataset("owner/data")
        self.assertEqual(catalog["dataset_handle"], "owner/data/versions/7")
        self.assertEqual([item["name"] for item in catalog["files"]], ["first.csv", "later.jsonl"])
        self.assertTrue(catalog["file_formats"]["mixed_formats"])
        requests = [call.args[0] for call in api.list_dataset_files.call_args_list]
        self.assertEqual([request.dataset_version_number for request in requests], [7, 7])
        self.assertFalse(requests[0].page_token)
        self.assertEqual(requests[1].page_token, "page-2")

    def test_explicit_version_does_not_resolve_latest_and_repeated_pages_fail(self):
        client = MagicMock()
        api = client.__enter__.return_value.datasets.dataset_api_client
        api.get_dataset.side_effect = AssertionError("Version already supplied")
        api.list_dataset_files.return_value = SimpleNamespace(dataset_files=[], error_message=None, next_page_token="same")
        with patch("kagglehub.clients.build_kaggle_client", return_value=client):
            with self.assertRaisesRegex(ValueError, "repeated page token"):
                discover_kaggle_dataset("owner/data/versions/3")
        self.assertEqual(api.list_dataset_files.call_args.args[0].dataset_version_number, 3)

    def test_csv_language_inspection_and_filtering_use_only_requested_file(self):
        cached = self.root / "later.csv"
        cached.write_text("text,language,id\nनेपाल सुन्दर छ।,ne,1\nEnglish text,en,2\n", encoding="utf-8")
        filtered = self.root / "kept.jsonl"
        with patch("scripts.inspect_dataset.discover_kaggle_dataset", return_value=self.catalog("first.csv", "later.csv")), \
             patch("kagglehub.dataset_download", return_value=str(cached)) as download:
            code, output, errors = self.run_cli("--dataset-file", "later.csv", "--text-column", "text", "--sample-fraction", "1",
                "--min-devanagari-ratio", "0.8", "--filtered-output", str(filtered))
        self.assertEqual(code, 0, errors)
        self.assertEqual(json.loads(output)["inventory"]["provider"], "kaggle")
        download.assert_called_once_with("owner/data/versions/7", path="later.csv")
        report = load_inspection_report(self.report_path)
        self.assertEqual(report["language_status"]["language_coverage"], "Bilingual (Nepali-English)")
        self.assertEqual(report["language_status"]["script"], "Mixed (Nepali + English)")
        self.assertEqual(report["filter_result"]["rows_kept"], 1)
        self.assertEqual(json.loads(filtered.read_text())["id"], "1")
        self.assertEqual(report["inventory"]["file_formats"]["extensions"], [".csv"])
        self.assertEqual(report["settings"]["dataset"], "owner/data/versions/7")

    def test_metadata_language_filter_precedes_percentage_sampling(self):
        cached = self.root / "mixed.csv"
        cached.write_text("text,language_code\n" + "English,en\n" * 80 + "नेपाल,npi\n" * 20, encoding="utf-8")
        with patch("scripts.inspect_dataset.discover_kaggle_dataset", return_value=self.catalog("mixed.csv")), \
             patch("kagglehub.dataset_download", return_value=str(cached)) as download:
            code, _, errors = self.run_cli("--dataset-file", "mixed.csv", "--row-filter", "language_code=npi")
        self.assertEqual(code, 0, errors)
        download.assert_called_once_with("owner/data/versions/7", path="mixed.csv")
        report = load_inspection_report(self.report_path)
        self.assertEqual(report["inventory"]["row_selection"]["rows_scanned"], 100)
        self.assertEqual(report["language_status"]["sampling"]["population_rows"], 20)
        self.assertEqual(report["language_status"]["sampling"]["rows_per_run"], 4)
        self.assertEqual(report["language_status"]["voting"]["language_coverage"]["counts"], {"Nepali-only": 5})

    def test_workbook_language_inspection_uses_shared_reader(self):
        from openpyxl import Workbook

        cached = self.root / "reviews.xlsx"
        book = Workbook()
        book.active.append(["Reviews", "language"])
        book.active.append(["नेपाल सुन्दर छ।", "ne"])
        book.save(cached)
        book.close()
        with patch("scripts.inspect_dataset.discover_kaggle_dataset", return_value=self.catalog("reviews.xlsx")), \
             patch("kagglehub.dataset_download", return_value=str(cached)):
            code, _, errors = self.run_cli("--text-column", "Reviews")
        self.assertEqual(code, 0, errors)
        report = load_inspection_report(self.report_path)
        self.assertEqual(report["inventory"]["format"], "xlsx")
        self.assertEqual(report["language_status"]["script"], "Devanagari")
        self.assertEqual(report["language_status"]["language_coverage"], "Nepali-only")

    def test_ambiguous_or_foreign_file_fails_without_downloading(self):
        with patch("scripts.inspect_dataset.discover_kaggle_dataset", return_value=self.catalog("first.csv", "second.csv")), \
             patch("kagglehub.dataset_download", side_effect=AssertionError("Do not select a default file")):
            for flags in ((), ("--dataset-file", "missing.csv"), ("--dataset-file", "../first.csv")):
                code, _, errors = self.run_cli(*flags)
                self.assertEqual(code, 1)
                self.assertIn("--list", errors)

    def test_formats_only_and_listing_never_download_and_preserve_selection(self):
        catalog = self.catalog("first.csv", "archive.zip")
        with patch("scripts.inspect_dataset.discover_kaggle_dataset", return_value=catalog), \
             patch("kagglehub.dataset_download", side_effect=AssertionError("No download")):
            code, output, _ = self.run_cli("--list")
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output), catalog)
            self.assertFalse(self.report_path.exists())
            code, _, errors = self.run_cli("--formats-only")
            self.assertEqual(code, 0, errors)
            self.assertEqual(load_inspection_report(self.report_path)["inventory"]["files"], 2)
            self.assertEqual(self.run_cli("--formats-only", "--dataset-file", "archive.zip")[0], 0)
            report = load_inspection_report(self.report_path)
            self.assertEqual(report["inventory"]["file_formats"]["extensions"], [".zip"])
            self.assertEqual(report["inventory"]["bytes"], 100)
            self.assertEqual(self.run_cli("--dataset-file", "archive.zip")[0], 1)

    def test_cli_provider_override_clears_only_inapplicable_settings_defaults(self):
        settings = self.root / "settings.yaml"
        settings.write_text("dataset: owner/data\nconfig: ne\nsplit: train\nrevision: main\nshards: [first.parquet]\n")
        _, effective = _arguments(["--settings", str(settings), "--provider", "kaggle", "--dataset-file", "later.csv"])
        self.assertEqual(effective["provider"], "kaggle")
        self.assertTrue(all(effective[key] is None for key in ("config", "split", "revision", "shards")))
        with self.assertRaisesRegex(ValueError, "only to Hugging Face"):
            _arguments(["--provider", "kaggle", "--dataset", "owner/data", "--split", "train"])
        with self.assertRaisesRegex(ValueError, "only to Kaggle"):
            _arguments(["--dataset", "owner/data", "--dataset-file", "later.csv"])

    def test_cached_source_cannot_be_overwritten_by_report(self):
        cached = self.root / "data.csv"
        original = "text\nनेपाल\n"
        cached.write_text(original)
        with patch("scripts.inspect_dataset.discover_kaggle_dataset", return_value=self.catalog("data.csv")), \
             patch("kagglehub.dataset_download", return_value=str(cached)):
            code, _, errors = self.run_cli("--output", str(cached))
        self.assertEqual(code, 1)
        self.assertRegex(errors, "must not overwrite a source file|incompatible header")
        self.assertEqual(cached.read_text(), original)


if __name__ == "__main__":
    unittest.main()
