import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from apps.components.inspection_reports import report_matches_source
from attention_maps.explorer.inspection_runs import load_inspection_report
from tests.inspection_helpers import isolated_main as main


class InspectionReportUITests(unittest.TestCase):
    def test_script_results_show_exact_saved_values_and_never_rerun_processing(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            root = Path(directory)
            source, output, report_path = root / "source.jsonl", root / "kept.jsonl", root / "report.csv"
            rows = [{"text": "नेपाल", "language": "ne"}, {"text": "English", "language": "en"}]
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--local", str(source), "--text-column", "text",
                    "--min-devanagari-ratio", "0.8", "--filtered-output", str(output),
                    "--output", str(report_path)]), 0)
            report = load_inspection_report(report_path)
            signatures = (report_path.stat().st_mtime_ns, output.stat().st_mtime_ns)
            source.unlink()  # Viewing historical results must work without source access.
            stack.enter_context(patch.dict(os.environ, {"DATASET_INSPECTION_REPORT": str(report_path)}))
            for target in ("attention_maps.explorer.discover_datasets",
                           "attention_maps.explorer.huggingface.discover_huggingface_dataset",
                           "attention_maps.explorer.inspection_runs.build_inspection_report",
                           "attention_maps.explorer.inspection_runs.iter_inspection_records",
                           "attention_maps.explorer.inspection.sample_dataset_rows"):
                stack.enter_context(patch(target, side_effect=AssertionError("Viewer must not execute processing")))
            st.cache_data.clear()
            app_path = Path(__file__).resolve().parents[1] / "apps/dataset_explorer.py"
            app = AppTest.from_file(str(app_path), default_timeout=20).run()
            self.assertFalse(app.exception)

            def values(label):
                return [item.value for item in app.metric if item.label == label]

            self.assertEqual(values("Language coverage"), [report["language_status"]["language_coverage"],
                report["filter_result"]["language_status"]["language_coverage"]])
            self.assertEqual(values("Script"), ["—", report["filter_result"]["language_status"]["script"]])
            for label, key in (("Records scanned", "rows_scanned"), ("Records kept", "rows_kept"),
                               ("Records dropped", "rows_dropped")):
                self.assertEqual(values(label), [str(report["filter_result"][key])])
            self.assertEqual(values("File format"), ["JSONL"])
            self.assertEqual(values("File extensions"), [".jsonl"])
            format_rows = app.dataframe[0].value
            self.assertEqual(format_rows.iloc[0]["Extension"], ".jsonl")
            self.assertEqual(format_rows.iloc[0]["Parsing requirement"], report["inventory"]["file_formats"]["groups"][0]["parsing_hint"])
            self.assertFalse(app.selectbox)  # No separate filter settings or category overrides.
            next(item for item in app.button if item.label == "Reload report").click().run()
            self.assertFalse(app.exception)
            self.assertEqual((report_path.stat().st_mtime_ns, output.stat().st_mtime_ns), signatures)
            # Reload reads replacement report bytes, not a stale cached result.
            report["created_at"] = "replacement run"
            report_path.write_text(json.dumps(report))
            next(item for item in app.button if item.label == "Reload report").click().run()
            self.assertTrue(any("replacement run" in item.value for item in app.caption))
            report_path.write_text('{"report_version": 99}')
            next(item for item in app.button if item.label == "Reload report").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.warning)
            self.assertFalse(app.metric)
            st.cache_data.clear()

    def test_formats_only_report_shows_mixed_extensions_and_parsing_requirements(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            for filename in ("records.jsonl.gz", "table.csv"):
                (source / filename).write_text("filename-only inspection")
            report_path = root / "report.csv"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--local", str(source), "--formats-only", "--output", str(report_path)]), 0)
            app_path = Path(__file__).resolve().parents[1] / "apps/dataset_explorer.py"
            with patch.dict(os.environ, {"DATASET_INSPECTION_REPORT": str(report_path)}):
                app = AppTest.from_file(str(app_path), default_timeout=20).run()
                self.assertFalse(app.exception)
                self.assertEqual(next(item.value for item in app.metric if item.label == "File format"), "MIXED")
                rows = app.dataframe[0].value
                self.assertEqual(set(rows["Extension"]), {".csv", ".jsonl.gz"})
                self.assertEqual(set(rows["Local batch support"]), {"Supported extension", "Decompress first"})
                self.assertTrue(any("records were not parsed" in item.value for item in app.caption))
                report = load_inspection_report(report_path)
                # Older version-1 reports remain readable and never invent format evidence.
                del report["inventory"]["file_formats"]
                report_path.write_text(json.dumps(report))
                next(item for item in app.button if item.label == "Reload report").click().run()
                self.assertFalse(app.exception)
                self.assertTrue(any("not saved" in item.value for item in app.caption))
                report["inventory"]["file_formats"] = {"format": "invalid"}
                report_path.write_text(json.dumps(report))
                next(item for item in app.button if item.label == "Reload report").click().run()
                self.assertFalse(app.exception)
                self.assertTrue(app.warning)
                self.assertFalse(app.metric)
            st.cache_data.clear()

    def test_kaggle_saved_report_shows_provider_version_and_language_without_network(self):
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cached, report_path = root / "data.csv", root / "report.csv"
            cached.write_text("text,language\nनेपाल,ne\n", encoding="utf-8")
            catalog = {"provider": "kaggle", "dataset_id": "owner/data", "dataset_handle": "owner/data/versions/3",
                       "dataset_version": 3, "files": [{"name": "data.csv", "bytes": cached.stat().st_size,
                       "path": "kaggle://datasets/owner/data/versions/3/data.csv"}]}
            with patch("scripts.inspect_dataset.discover_kaggle_dataset", return_value=catalog), \
                 patch("kagglehub.dataset_download", return_value=str(cached)), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--provider", "kaggle", "--dataset", "owner/data", "--output", str(report_path)]), 0)
            cached.unlink()
            with patch.dict(os.environ, {"DATASET_INSPECTION_REPORT": str(report_path)}), \
                 patch("kagglehub.dataset_download", side_effect=AssertionError("Viewer must not download")):
                app_path = Path(__file__).resolve().parents[1] / "apps/dataset_explorer.py"
                app = AppTest.from_file(str(app_path), default_timeout=20).run()
                self.assertFalse(app.exception)
                self.assertTrue(any(item.value == "Source provider: kaggle" for item in app.caption))
                self.assertEqual(next(item.value for item in app.metric if item.label == "Language coverage"), "Nepali-only")
                self.assertEqual(next(item.value for item in app.metric if item.label == "Script"), "Devanagari")
                metadata = json.loads(app.json[0].value)
                self.assertEqual(metadata["Version"], 3)
                self.assertEqual(metadata["Selected files"], ["data.csv"])

    def test_voting_view_displays_stored_coverage_and_inconclusive_result(self):
        from streamlit.testing.v1 import AppTest
        from attention_maps.explorer.sampling_vote import RandomWithoutReplacement

        # Pick a deterministic seed for two different one-record samples.
        base_seed = next(seed for seed in range(100) if
            RandomWithoutReplacement(2, 0.5, f"{seed}:1").select(0) !=
            RandomWithoutReplacement(2, 0.5, f"{seed}:2").select(0))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, report_path = root / "data.jsonl", root / "report.csv"
            source.write_text(json.dumps({"text": "नेपाल", "language": "ne"}) + "\n" + json.dumps({"text": "English", "language": "en"}) + "\n")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--local", str(source), "--sample-fraction", "0.5", "--sampling-runs", "2",
                    "--seed", str(base_seed), "--output", str(report_path)]), 0)
            report = load_inspection_report(report_path)
            source.unlink()
            with patch.dict(os.environ, {"DATASET_INSPECTION_REPORT": str(report_path)}), \
                 patch("attention_maps.explorer.sampling_vote.RepeatedSampleAnalysis.observe", side_effect=AssertionError("Display must not sample")):
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "apps/dataset_explorer.py"), default_timeout=20).run()
                self.assertFalse(app.exception)
                self.assertEqual(next(item.value for item in app.metric if item.label == "Unique sample coverage"), "100.00%")
                self.assertEqual(next(item.value for item in app.metric if item.label == "Script"), "—")
                self.assertTrue(any("Script vote is inconclusive" in item.value for item in app.info))
                table = next(item.value for item in app.dataframe if "Run" in item.value.columns and "Script" in item.value.columns)
                self.assertEqual(list(table["Records"]), [1, 1])
                self.assertEqual(set(table["Script"]), {"Devanagari", "—"})
                report["language_status"]["voting"]["script"]["label"] = "Devanagari"
                report_path.write_text(json.dumps(report))
                next(item for item in app.button if item.label == "Reload report").click().run()
                self.assertFalse(app.exception)
                self.assertTrue(app.warning)
                self.assertFalse(app.metric)

    def test_status_cannot_follow_a_different_configuration_split_or_shard(self):
        inventory = {"format": "huggingface", "dataset_id": "owner/data", "dataset_config": "ne",
                     "dataset_split": "train", "dataset_revision": "commit", "dataset_shards": ["second.parquet"]}
        report = {"inventory": inventory}
        spec = SimpleNamespace(files=())
        self.assertTrue(report_matches_source(report, inventory, spec))
        for key, value in (("dataset_config", "en"), ("dataset_split", "test"),
                           ("dataset_revision", "new"), ("dataset_shards", ["first.parquet"])):
            with self.subTest(key=key):
                self.assertFalse(report_matches_source(report, {**inventory, key: value}, spec))


if __name__ == "__main__":
    unittest.main()
