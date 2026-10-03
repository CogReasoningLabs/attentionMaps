"""Historical research results retain their original cells and displayed taxonomy."""

import contextlib
import copy
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from apps.components.inspection_history import _language_distribution
from apps.components.inspection_reports import _category_value, _display_vote_counts
from attention_maps.explorer.inspection_history import (
    FIELDS, PRE_TIMING_FIELDS, PRE_PROPORTION_FIELDS, PRE_SELECTION_FIELDS,
    append_history, history_row, load_history_report, read_history,
)
from attention_maps.explorer.inspection_runs import validate_inspection_report
from tests.inspection_helpers import isolated_main as main


class InspectionResultPreservationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        source = self.root / "source.jsonl"
        source.write_text(json.dumps({"text": "हिन्दी", "language": "hi"}) + "\n")
        self.sheet = self.root / "history.csv"
        with contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--local", str(source), "--sample-fraction", "1",
                                   "--sampling-runs", "3", "--output", str(self.sheet)]), 0)
        self.current = json.loads(output.getvalue())
        self.earlier = copy.deepcopy(self.current)
        status = self.earlier["language_status"]
        status.pop("record_examples", None)
        for item in [status, *status["sampling"]["run_results"]]:
            for prefix in ("language", "script"):
                for bucket in ("Other", "Unknown"):
                    item[f"{prefix}_category_counts"].pop(bucket)
                    item[f"{prefix}_category_percentages"].pop(bucket)
        validate_inspection_report(self.earlier)

    def write_earlier(self, fields=FIELDS):
        row = history_row(self.earlier, self.earlier["history"]["run_id"])
        with self.sheet.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerow({name: row[name] for name in fields})
        return row

    def test_append_preserves_every_existing_cell_across_supported_inline_headers(self):
        for fields in (FIELDS, PRE_TIMING_FIELDS, PRE_PROPORTION_FIELDS, PRE_SELECTION_FIELDS):
            with self.subTest(fields=fields):
                original = self.write_earlier(fields)
                # Historic values must survive verbatim, even if today's display
                # would leave this unsupported label blank or trim whitespace.
                original.update({"Script": "Not applicable", "Language evidence": "  saved evidence  ",
                                 "Report JSON": "\n" + original["Report JSON"] + "\n"})
                with self.sheet.open("w", encoding="utf-8-sig", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerow({name: original[name] for name in fields})
                append_history(self.current, self.sheet)
                with self.sheet.open(encoding="utf-8-sig", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 2)
                self.assertEqual({key: rows[0][key] for key in fields},
                                 {key: original[key] for key in fields})
                self.assertEqual(load_history_report(self.sheet, rows[0]), self.earlier)
                self.assertNotEqual(rows[0]["Run ID"], rows[1]["Run ID"])
                self.assertEqual(json.loads(rows[1]["Language category %"])["Other"], 100.0)

    def test_old_other_votes_are_not_promoted_to_new_summary_categories(self):
        old, new = self.earlier["language_status"], self.current["language_status"]
        self.assertEqual(old["voting"], new["voting"])
        self.assertEqual(_category_value(old, "language_coverage"), "—")
        self.assertEqual(_category_value(new, "language_coverage"), "—")
        self.assertEqual(_display_vote_counts({"Other": 3}, "language_coverage", old),
                         {"No supported category": 3})
        self.assertEqual(history_row(self.earlier, "old")["Language Coverage"], "")
        self.assertEqual(history_row(self.current, "new")["Language Coverage"], "")

    def test_mixed_history_percentages_are_arrow_compatible_without_changing_saved_values(self):
        import pandas as pd
        import pyarrow as pa
        from streamlit.testing.v1 import AppTest

        self.write_earlier()
        append_history(self.current, self.sheet)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--local", str(self.root / "source.jsonl"), "--formats-only",
                                   "--output", str(self.sheet)]), 0)
        before = self.sheet.read_bytes()
        with patch.dict(os.environ, {"DATASET_INSPECTION_HISTORY": str(self.sheet)}), \
             patch("streamlit.dataframe_util.fix_arrow_incompatible_column_types",
                   side_effect=AssertionError("History must serialize without Arrow type repair")):
            app = AppTest.from_string("""
import streamlit as st
from apps.components.inspection_history import select_history_report
select_history_report(st)
""", default_timeout=20).run()
        self.assertFalse(app.exception)
        history = app.dataframe[0].value
        arrow = pa.Table.from_pandas(history)
        # Newest first: metadata-only, current taxonomy, original taxonomy.
        self.assertEqual(arrow.column("Other %").to_pylist(), [None, 100.0, None])
        self.assertEqual(arrow.column("Unknown %").to_pylist(), [None, 0.0, None])
        self.assertEqual(arrow.column("Outside listed categories %").to_pylist(), [None, None, 100.0])
        self.assertEqual(arrow.column("No language evidence %").to_pylist(), [None, None, 0.0])
        self.assertTrue(pd.api.types.is_numeric_dtype(history["Nepali-only %"]))
        self.assertEqual(self.sheet.read_bytes(), before)

    def test_viewing_old_results_never_backfills_other_or_changes_the_file(self):
        from streamlit.testing.v1 import AppTest

        original = self.write_earlier()
        before = self.sheet.read_bytes()
        percentages, _ = _language_distribution(original)
        for name, value in self.earlier["language_status"]["language_category_percentages"].items():
            self.assertEqual(percentages[name], value)
        self.assertEqual(percentages["Outside listed categories"], 100.0)
        self.assertNotIn("Other", percentages)
        app_path = Path(__file__).resolve().parents[1] / "apps/dataset_explorer.py"
        with patch.dict(os.environ, {"DATASET_INSPECTION_REPORT": str(self.sheet)}), \
             patch("attention_maps.explorer.inspection_runs.build_inspection_report",
                   side_effect=AssertionError("Viewing old results must never recompute them")):
            app = AppTest.from_file(str(app_path), default_timeout=20).run()
        self.assertFalse(app.exception)
        category_tables = [item.value for item in app.dataframe
                           if {"Category", "Records", "Percent"}.issubset(item.value.columns)]
        self.assertNotIn("Other", set(category_tables[0]["Category"]))
        per_run = next(item.value for item in app.dataframe
                       if "No language evidence (records)" in item.value.columns)
        self.assertNotIn("Other", per_run.columns)
        self.assertEqual(list(per_run["Outside listed categories (records)"]), [1, 1, 1])
        self.assertEqual(self.sheet.read_bytes(), before)
        self.assertEqual(load_history_report(self.sheet, read_history(self.sheet)[0]), self.earlier)


if __name__ == "__main__":
    unittest.main()
