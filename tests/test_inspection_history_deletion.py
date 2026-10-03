"""Deleting a saved run must preserve all other research history."""

import contextlib
from concurrent.futures import ThreadPoolExecutor
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from attention_maps.explorer.inspection_history import (
    FIELDS, LEGACY_FIELDS, PRE_INSTANCE_FIELDS, PRE_TIMING_FIELDS,
    PRE_PROPORTION_FIELDS, PRE_SELECTION_FIELDS,
    append_history, delete_history_run, load_history_report, read_history,
)
from scripts.inspect_dataset import main


class InspectionHistoryDeletionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.sheet = self.root / "history.csv"
        self.source = self.root / "source.jsonl"
        self.source.write_text(json.dumps({"text": "नेपाल", "language": "ne"}) + "\n")
        with contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--local", str(self.source), "--output", str(self.sheet)]), 0)
        self.report = json.loads(output.getvalue())
        for _ in range(2):
            append_history(self.report, self.sheet)
        self.rows = read_history(self.sheet, preserve_values=True)

    def test_delete_preserves_exact_bytes_across_headers_quoting_and_line_endings(self):
        headers = (FIELDS, PRE_INSTANCE_FIELDS, PRE_TIMING_FIELDS, PRE_PROPORTION_FIELDS,
                   PRE_SELECTION_FIELDS, LEGACY_FIELDS, (*FIELDS, "Future metric"))
        for fields in headers:
            for bom, ending in ((b"", "\n"), (b"\xef\xbb\xbf", "\r\n"), (b"", "\r")):
                with self.subTest(fields=fields, bom=bom, ending=ending):
                    def encoded(values):
                        stream = io.StringIO(newline="")
                        csv.writer(stream, quoting=csv.QUOTE_ALL, lineterminator=ending).writerow(values)
                        return stream.getvalue().encode("utf-8")

                    header = bom + encoded([f" {field} " for field in fields])
                    records = []
                    for original in self.rows:
                        row = {**original, "Language evidence": '  नेपाल, "quoted"\nsecond line\r\nthird  ',
                               "Report": "untouched.json", "Future metric": "  unchanged  "}
                        records.append(encoded([f" {row[field]} " for field in fields]))
                    self.sheet.write_bytes(header + b"".join(records))
                    self.assertTrue(delete_history_run(self.sheet, self.rows[1]["Run ID"]))
                    self.assertEqual(self.sheet.read_bytes(), header + records[0] + records[2])
                    if fields == FIELDS:
                        remaining = read_history(self.sheet)
                        self.assertEqual(load_history_report(self.sheet, remaining[0]), self.report)

    def test_missing_run_is_noop_and_duplicate_or_malformed_sheet_is_rejected(self):
        before = self.sheet.read_bytes()
        self.assertFalse(delete_history_run(self.sheet, "missing-run"))
        self.assertEqual(self.sheet.read_bytes(), before)
        self.assertFalse(delete_history_run(self.root / "missing.csv", "missing-run"))
        self.assertFalse((self.root / "missing.csv").exists())
        for run_id in ("", "  ", None):
            with self.assertRaises(ValueError):
                delete_history_run(self.sheet, run_id)
        with self.sheet.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows([self.rows[0], self.rows[0]])
        before = self.sheet.read_bytes()
        with self.assertRaisesRegex(ValueError, "more than once"):
            delete_history_run(self.sheet, self.rows[0]["Run ID"])
        self.assertEqual(self.sheet.read_bytes(), before)
        for data in (b"unrelated,csv\nx,y\n", before + b"incomplete,row\n"):
            self.sheet.write_bytes(data)
            with self.assertRaises(ValueError):
                delete_history_run(self.sheet, self.rows[0]["Run ID"])
            self.assertEqual(self.sheet.read_bytes(), data)

    def test_last_deletion_keeps_header_and_allows_new_run(self):
        for row in self.rows:
            self.assertTrue(delete_history_run(self.sheet, row["Run ID"]))
        self.assertEqual(read_history(self.sheet), [])
        self.assertTrue(self.sheet.read_bytes().startswith(b"\xef\xbb\xbf"))
        saved = append_history(self.report, self.sheet)
        self.assertEqual(len(read_history(self.sheet)), 1)
        self.assertEqual(load_history_report(self.sheet, read_history(self.sheet)[0]), saved)

    def test_failed_atomic_replacement_keeps_original_and_removes_temporary_file(self):
        before = self.sheet.read_bytes()
        with patch.object(Path, "replace", side_effect=OSError("disk error")):
            with self.assertRaisesRegex(OSError, "disk error"):
                delete_history_run(self.sheet, self.rows[1]["Run ID"])
        self.assertEqual(self.sheet.read_bytes(), before)
        self.assertFalse(list(self.root.glob(".history.csv-*.tmp")))

    def test_concurrent_appends_and_deletes_do_not_lose_other_runs(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            additions = [pool.submit(append_history, self.report, self.sheet) for _ in range(4)]
            removals = [pool.submit(delete_history_run, self.sheet, row["Run ID"]) for row in self.rows[1:]]
            saved = [future.result() for future in additions]
            self.assertTrue(all(future.result() for future in removals))
        remaining = read_history(self.sheet, preserve_values=True)
        self.assertEqual(len(remaining), 5)
        self.assertEqual(remaining[0], self.rows[0])
        self.assertEqual({row["Run ID"] for row in remaining},
                         {self.rows[0]["Run ID"], *(report["history"]["run_id"] for report in saved)})

    def app(self):
        from streamlit.testing.v1 import AppTest

        return AppTest.from_string("""
import streamlit as st
from apps.components.inspection_reports import render_script_results
render_script_results(st)
""", default_timeout=20).run()

    def test_streamlit_deletes_selected_run_refreshes_and_handles_empty_history(self):
        before, source = self.sheet.read_bytes(), self.source.read_bytes()
        with patch.dict(os.environ, {"DATASET_INSPECTION_HISTORY": str(self.sheet)}), \
             patch("attention_maps.explorer.inspection_runs.build_inspection_report",
                   side_effect=AssertionError("Deleting history must not analyze data")):
            app = self.app()
            self.assertFalse(app.exception)
            self.assertEqual(self.sheet.read_bytes(), before)
            # Delete the oldest run, not the latest run initially selected by the UI.
            next(item for item in app.selectbox if item.label == "Inspection run").set_value(
                self.rows[0]["Run ID"]).run()
            self.assertEqual(self.sheet.read_bytes(), before)
            next(item for item in app.button if item.label == "Delete selected run").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("Deleted inspection run" in item.value for item in app.success))
            self.assertEqual(read_history(self.sheet, preserve_values=True), self.rows[1:])
            selector = next(item for item in app.selectbox if item.label == "Inspection run")
            self.assertNotEqual(selector.value, self.rows[0]["Run ID"])
            self.assertEqual(len(selector.options), 2)
            for _ in range(2):
                next(item for item in app.button if item.label == "Delete selected run").click().run()
                self.assertFalse(app.exception)
            self.assertEqual(read_history(self.sheet), [])
            self.assertTrue(any("No recorded inspections" in item.value for item in app.info))
            self.assertFalse(app.metric)
            self.assertFalse(any(item.label == "Delete selected run" for item in app.button))
        self.assertEqual(self.source.read_bytes(), source)

    def test_streamlit_reports_write_failure_without_removing_any_run(self):
        before = self.sheet.read_bytes()
        with patch.dict(os.environ, {"DATASET_INSPECTION_HISTORY": str(self.sheet)}), \
             patch("apps.components.inspection_history.delete_history_run", side_effect=OSError("read-only file")):
            app = self.app()
            next(item for item in app.button if item.label == "Delete selected run").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Could not delete" in item.value for item in app.error))
        self.assertEqual(self.sheet.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
