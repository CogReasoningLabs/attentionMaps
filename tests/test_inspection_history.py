import csv
import contextlib
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from attention_maps.explorer.inspection_history import append_history, consolidate_inspection_reports, decision_text, load_history_report, read_history, LEGACY_FIELDS, PRE_PROPORTION_FIELDS
from attention_maps.explorer.sampling_vote import majority_vote
from scripts.inspect_dataset import main, _arguments


class InspectionHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.sheet = self.root / "history.csv"
        self.output = self.sheet

    def inspect(self, name="first", text="नेपाल", *extra):
        source = self.root / f"{name}.jsonl"
        source.write_text("".join(json.dumps({"text": text, "language": "ne" if text == "नेपाल" else "en"}, ensure_ascii=False) + "\n" for _ in range(10)))
        with contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(io.StringIO()):
            code = main(["--local", str(source), "--history-sheet", str(self.sheet), *extra])
        return code, json.loads(output.getvalue()) if code == 0 else None

    def test_each_manual_run_keeps_its_own_inline_report_and_exact_votes(self):
        reports = []
        for name, text, seed in (("first", "नेपाल", "1"), ("second", "English", "2"), ("first", "नेपाल", "3")):
            code, report = self.inspect(name, text, "--seed", seed)
            self.assertEqual(code, 0)
            reports.append(report)
        rows = read_history(self.sheet)
        self.assertEqual(len(rows), 3)
        self.assertEqual(len({row["Run ID"] for row in rows}), 3)
        self.assertTrue(self.sheet.read_bytes().startswith(b"\xef\xbb\xbf"))
        for row, expected in zip(rows, reports):
            self.assertEqual(load_history_report(self.sheet, row), expected)
            self.assertEqual(row["Language Coverage"], expected["language_status"]["language_coverage"])
            expected_script = expected["language_status"]["script"]
            self.assertEqual(row["Script"], expected_script if expected_script != "Not applicable" else "")
            self.assertEqual(json.loads(row["Per-run votes"]), expected["language_status"]["sampling"]["run_results"])
            self.assertEqual(json.loads(row["Script votes"]), expected["language_status"]["voting"]["script"]["counts"])
            self.assertIn("at least 3", row["Script decision"])
        self.assertEqual(rows[0]["Dataset"], rows[2]["Dataset"])
        self.assertNotEqual(rows[0]["Seed"], rows[2]["Seed"])

    def test_inconclusive_and_unsupported_categories_are_recorded_without_false_summary_labels(self):
        source = self.root / "categories.jsonl"
        for record, expected_language, expected_script in (
            ({"text": "नेपाल"}, "", ""),
            ({"text": "हिन्दी", "language": "hi"}, "", ""),
            ({"text": "中文", "language": "ne"}, "Nepali-only", ""),
            ({"text": "नेपाल", "language": "ne"}, "Nepali-only", "Devanagari"),
        ):
            with self.subTest(record=record):
                source.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as errors:
                    code = main(["--local", str(source), "--history-sheet", str(self.sheet),
                                 "--sample-fraction", "1", "--sampling-runs", "3"])
                self.assertEqual(code, 0, errors.getvalue())
                saved = read_history(self.sheet)[-1]
                self.assertEqual((saved["Language Coverage"], saved["Script"]),
                                 (expected_language, expected_script))
                self.assertEqual(json.loads(saved["Language category %"])["Nepali-only"],
                                 100.0 if record.get("language") == "ne" else 0.0)
        self.assertEqual(len(read_history(self.sheet)), 4)

    def test_prior_csv_header_upgrades_in_place_when_new_run_adds_proportions(self):
        self.assertEqual(self.inspect("prior")[0], 0)
        original = read_history(self.sheet)[0]
        with self.sheet.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=PRE_PROPORTION_FIELDS)
            writer.writeheader()
            writer.writerow({name: original[name] for name in PRE_PROPORTION_FIELDS})
        self.assertEqual(self.inspect("new")[0], 0)
        rows = read_history(self.sheet)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["Run ID"], original["Run ID"])
        self.assertEqual(rows[0]["Report JSON"], original["Report JSON"])
        self.assertEqual(json.loads(rows[1]["Language category %"])["Nepali-only"], 100.0)

    def test_default_history_and_yaml_paths_and_collision_validation(self):
        with patch("scripts.inspect_dataset.DEFAULT_HISTORY_SHEET", self.sheet), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            source = self.root / "local.csv"
            source.write_text("text,language\nनेपाल,ne\n")
            self.assertEqual(main(["--local", str(source)]), 0)
            self.assertEqual(len(read_history(self.sheet)), 1)
            before = self.sheet.read_bytes()
            self.assertEqual(main(["--local", str(source), "--output", str(self.sheet.with_suffix(".json"))]), 1)
            self.assertEqual(self.sheet.read_bytes(), before)
            original_source = source.read_bytes()
            self.assertEqual(main(["--local", str(source), "--history-sheet", str(source)]), 1)
            self.assertEqual(source.read_bytes(), original_source)
        settings = self.root / "settings.yaml"
        settings.write_text("local: local.csv\nhistory_sheet: logs/sheet.csv\n")
        self.assertEqual(_arguments(["--settings", str(settings)])[1]["history_sheet"], str(self.root / "logs/sheet.csv"))

    def test_metadata_only_legacy_optout_and_failed_runs_do_not_fabricate_votes(self):
        self.assertEqual(self.inspect("metadata", "नेपाल", "--formats-only")[0], 0)
        self.assertEqual(self.inspect("legacy", "नेपाल", "--sample-size", "2")[0], 0)
        self.assertEqual(self.inspect("unrecorded", "नेपाल", "--no-history")[0], 0)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--local", str(self.root / "missing.jsonl"), "--history-sheet", str(self.sheet)]), 1)
        rows = read_history(self.sheet)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["Sample scope"], "metadata_only")
        for row in rows:
            self.assertEqual(row["Sampling runs"], "0")
            self.assertEqual(json.loads(row["Per-run votes"]), [])
            self.assertIn("No repeated-sample vote", row["Script decision"])

    def test_concurrent_appends_and_failed_publication_preserve_history(self):
        _, report = self.inspect()
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda _: append_history(report, self.sheet), range(8)))
        rows = read_history(self.sheet)
        self.assertEqual(len(rows), 9)
        self.assertEqual(len({row["Run ID"] for row in rows}), 9)
        before = self.sheet.read_bytes()
        snapshots = set(self.root.glob("history_reports/*.json"))
        replace = Path.replace

        def fail_csv(path, target):
            if Path(target) == self.sheet:
                raise OSError("simulated CSV publication failure")
            return replace(path, target)

        with patch.object(Path, "replace", fail_csv), self.assertRaises(OSError):
            append_history(report, self.sheet)
        self.assertEqual(before, self.sheet.read_bytes())
        self.assertEqual(snapshots, set(self.root.glob("history_reports/*.json")))

    def test_spreadsheet_padding_does_not_break_csv_or_saved_votes(self):
        _, report = self.inspect()
        rows = read_history(self.sheet)
        fields = list(rows[0])
        with self.sheet.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow([f" {name}  " for name in fields])
            writer.writerow([f" {rows[0][name]}  " for name in fields])
        self.assertEqual(load_history_report(self.sheet, read_history(self.sheet)[0]), report)

    def test_csv_is_portable_without_any_companion_files(self):
        code, report = self.inspect()
        self.assertEqual(code, 0)
        self.assertFalse(list(self.root.glob("*.lock")))
        self.assertFalse(list(self.root.glob("*_reports")))
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "history.csv"
            shutil.copyfile(self.sheet, copied)
            self.assertEqual(load_history_report(copied, read_history(copied)[0]), report)
            self.assertEqual([p.name for p in Path(directory).iterdir()], ["history.csv"])

    def test_legacy_migration_verifies_reports_before_deleting_json(self):
        _, report = self.inspect()
        row = read_history(self.sheet)[0]
        archive = self.root / "history_reports"
        archive.mkdir()
        snapshot = archive / (row["Run ID"] + ".json")
        report["history"]["report"] = str(snapshot)
        snapshot.write_text(json.dumps(report))
        latest = self.root / "report.json"
        latest.write_text(json.dumps(report))
        second = self.root / "earlier-report.json"
        second_report = {**report, "created_at": "2026-01-01T00:00:00+00:00"}
        second.write_text(json.dumps(second_report))
        unrelated = self.root / "source.json"
        unrelated.write_text('{"text": "Keep source data"}')
        row.pop("Report JSON")
        row["Report"] = str(snapshot.relative_to(self.root))
        with self.sheet.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=LEGACY_FIELDS)
            writer.writeheader()
            writer.writerow({field: row[field] for field in LEGACY_FIELDS})
        (self.root / "history.csv.lock").touch()
        before = self.sheet.read_bytes()
        with patch("attention_maps.explorer.inspection_history.write_history", side_effect=OSError("failed")), self.assertRaises(OSError):
            consolidate_inspection_reports(self.root, self.sheet)
        self.assertEqual(before, self.sheet.read_bytes())
        self.assertTrue(snapshot.exists() and latest.exists() and second.exists())
        result = consolidate_inspection_reports(self.root, self.sheet)
        self.assertEqual(result["runs"], 2)
        self.assertEqual(result["removed_json_reports"], 3)
        self.assertFalse(snapshot.exists() or latest.exists() or second.exists() or archive.exists())
        self.assertFalse((self.root / "history.csv.lock").exists())
        self.assertTrue(unrelated.exists())
        recovered = [load_history_report(self.sheet, item) for item in read_history(self.sheet)]
        self.assertEqual({item["created_at"] for item in recovered}, {report["created_at"], second_report["created_at"]})
        self.assertTrue(all(item["language_status"] == report["language_status"] for item in recovered))

    def test_majority_and_inconclusive_explanations_follow_saved_votes(self):
        for labels, expected in ((["Devanagari"] * 3 + ["Latin"] * 2, "3/5"),
                                 (["Devanagari", "Latin"], "No category assigned"),
                                 (["Unknown"] * 3 + ["Latin"] * 2, "No category assigned")):
            status = {"voting": {"script": majority_vote(labels)}}
            self.assertIn(expected, decision_text(status, "script"))

    def test_streamlit_browses_history_and_displays_two_vote_cards_without_processing(self):
        from streamlit.testing.v1 import AppTest
        reports = [self.inspect("first")[1], self.inspect("second", "English")[1]]
        for path in self.root.glob("*.jsonl"):
            path.unlink()
        before = self.sheet.read_bytes()
        with patch.dict(os.environ, {"DATASET_INSPECTION_HISTORY": str(self.sheet), "DATASET_INSPECTION_REPORT": "", "DATASET_EMBEDDING_RUNS": ""}), \
             patch("attention_maps.explorer.inspection_runs.iter_inspection_records", side_effect=AssertionError("No source scan")), \
             patch("attention_maps.explorer.sampling_vote.RepeatedSampleAnalysis.observe", side_effect=AssertionError("No sampling")):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "apps/dataset_explorer.py"), default_timeout=25).run()
            for report in reports:
                next(x for x in app.selectbox if x.label == "Inspection run").set_value(report["history"]["run_id"]).run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                self.assertEqual(next(x.value for x in app.metric if x.label == "Language coverage"), report["language_status"]["language_coverage"])
                expected_script = report["language_status"]["script"]
                self.assertEqual(next(x.value for x in app.metric if x.label == "Script"),
                                 expected_script if expected_script != "Not applicable" else "—")
                tables = [x.value for x in app.dataframe if "Vote" in x.value.columns]
                self.assertEqual(len(tables), 2)
                self.assertTrue(all(len(table) == 5 for table in tables))
                self.assertTrue(any("vote rule needs at least 3" in x.value for x in app.markdown))
        self.assertEqual(before, self.sheet.read_bytes())


if __name__ == "__main__":
    unittest.main()
