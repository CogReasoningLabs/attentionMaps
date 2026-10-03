import contextlib
import copy
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from attention_maps.explorer.inspection_history import read_history
from attention_maps.explorer.inspection_runs import load_inspection_report
from attention_maps.explorer.language_status import analyze_language_status
from attention_maps.explorer.sampling_vote import RepeatedSampleAnalysis, validate_sampling_result
from apps.components.inspection_reports import _render_category_proportions
from tests.inspection_helpers import isolated_main as main


class CategoryProportionTests(unittest.TestCase):
    def test_other_and_unknown_script_records_keep_the_original_eligible_denominator(self):
        status = analyze_language_status([
            {"text": "中文", "language": "ne"},
            {"text": "123", "language": "ne"},
            {"text": "English", "language": "en"},
        ], text_columns=["text"])
        self.assertEqual(status["script_eligible_records"], 2)
        self.assertEqual(status["script_outside_categories_records"], 1)
        self.assertEqual(status["script_no_evidence_records"], 1)
        self.assertEqual(status["script_category_percentages"]["Other"], 50.0)
        self.assertEqual(status["script_category_percentages"]["Unknown"], 50.0)
        validate_sampling_result(status)
        # Moving a record between remainder buckets must not silently pass validation.
        status["script_category_counts"].update(Other=0, Unknown=2)
        with self.assertRaisesRegex(ValueError, "inconsistent category counts"):
            validate_sampling_result(status)

    def test_rounding_is_explained_without_adjusting_any_existing_percentage(self):
        status = analyze_language_status([
            {"text": "नेपाल", "language": "ne"},
            {"text": "English", "language": "en"},
            {"text": "हिन्दी", "language": "hi"},
        ], text_columns=["text"])
        before = copy.deepcopy(status)
        st = Mock()
        _render_category_proportions(st, status, "language_coverage")
        rows = st.dataframe.call_args.args[0]
        shares = {row["Category"]: row["Percent"] for row in rows}
        self.assertEqual(shares["Nepali-only"], 33.33)
        self.assertEqual(shares["English-only"], 33.33)
        self.assertEqual(shares["Other"], 33.33)
        self.assertEqual(shares["Total"], 100.0)
        self.assertTrue(any("99.99%" in call.args[0] and "rounded" in call.args[0]
                            for call in st.caption.call_args_list))
        self.assertEqual(status, before)

    def test_empty_denominators_do_not_claim_one_hundred_percent(self):
        status = analyze_language_status([])
        for key in ("language_coverage", "script"):
            st = Mock()
            _render_category_proportions(st, status, key)
            self.assertEqual(st.dataframe.call_args.args[0][-1],
                             {"Category": "Total", "Records": 0, "Percent": None})
        validate_sampling_result(status)

    def test_record_categories_keep_unclassified_records_in_denominators(self):
        records = [
            {"text": "नेपाल", "language": "ne"},
            {"text": "namaste", "language": "ne"},
            {"text": "नेपाल namaste", "language": "ne"},
            {"text": "नेपाल hello", "language_pair": "en-ne"},
            {"text": "Hello", "language": "en"},
            {"text": "हिन्दी", "language": "hi"},
            {"text": "123 !!!"},
            {"text": "123", "language": "ne"},
        ]
        status = analyze_language_status(records, text_columns=["text"])
        self.assertEqual(status["sampled_records"], 8)
        self.assertEqual(status["language_category_counts"], {
            "Nepali-only": 4, "Bilingual (Nepali-English)": 1,
            "Multilingual": 0, "English-only": 1, "Other": 1, "Unknown": 1,
        })
        self.assertEqual(status["language_category_percentages"]["Nepali-only"], 50.0)
        self.assertEqual((status["language_no_evidence_records"],
                          status["language_outside_categories_records"]), (1, 1))
        self.assertEqual(status["language_category_percentages"]["Other"], 12.5)
        self.assertEqual(status["script_eligible_records"], 5)
        self.assertEqual(status["script_category_counts"], {
            "Devanagari": 1, "Mixed (Devanagari + romanized)": 1,
            "Romanized": 1, "Mixed (Nepali + English)": 1, "Other": 0, "Unknown": 1,
        })
        self.assertEqual(status["script_category_percentages"]["Unknown"], 20.0)
        self.assertEqual(sum(status["language_category_counts"].values()), 8)
        self.assertEqual(sum(status["script_category_counts"].values()), 5)
        self.assertEqual(sum(status["language_category_percentages"].values()), 100.0)
        self.assertEqual(sum(status["script_category_percentages"].values()), 100.0)
        self.assertEqual((status["script_no_evidence_records"],
                          status["script_outside_categories_records"]), (1, 0))

    def test_unique_sample_counts_overlapping_runs_once(self):
        records = [{"text": "नेपाल", "language": "ne"} for _ in range(20)]
        analysis = RepeatedSampleAnalysis(
            {"format": "local"},
            {"seed": 42, "sampling_runs": 5, "sample_fraction": .5, "sampling_method": "random"},
            ["text"], len(records),
        )
        for record in records:
            analysis.observe(record)
        status = analysis.finish()
        self.assertEqual(status["language_category_counts"]["Nepali-only"],
                         status["sampling"]["unique_sampled_records"])
        self.assertEqual(status["script_category_counts"]["Devanagari"],
                         status["sampling"]["unique_sampled_records"])
        self.assertEqual(status["language_category_percentages"]["Nepali-only"], 100.0)
        self.assertTrue(all(run["language_category_counts"]["Nepali-only"] == 10
                            for run in status["sampling"]["run_results"]))
        validate_sampling_result(status)
        status["language_category_percentages"]["Nepali-only"] = 0.0
        with self.assertRaisesRegex(ValueError, "inconsistent category percentages"):
            validate_sampling_result(status)

    def test_inconclusive_report_still_saves_category_proportions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, sheet = root / "source.jsonl", root / "history.csv"
            records = [
                {"text": "नेपाल", "language": "ne"},
                {"text": "Hello", "language": "en"},
                {"text": "123 !!!"},
                {"text": "हिन्दी", "language": "hi"},
            ]
            source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--local", str(source), "--sample-fraction", "1",
                                       "--sampling-runs", "3", "--output", str(sheet)]), 0)
            report = load_inspection_report(sheet)
            row = read_history(sheet)[0]
            self.assertEqual(report["language_status"]["sampled_records"], 4)
            self.assertEqual(row["Language Coverage"], "")
            self.assertEqual(row["Script"], "")
            self.assertEqual(json.loads(row["Language category %"])["Nepali-only"], 25.0)
            self.assertEqual(json.loads(row["Nepali script category %"])["Devanagari"], 100.0)
            self.assertEqual(report["language_status"]["language_no_evidence_records"], 1)
            self.assertEqual(report["language_status"]["language_outside_categories_records"], 1)
            self.assertEqual(json.loads(row["Language category %"])["Other"], 25.0)
            self.assertEqual(json.loads(row["Language category %"])["Unknown"], 25.0)
            from streamlit.testing.v1 import AppTest
            app_path = Path(__file__).resolve().parents[1] / "apps/dataset_explorer.py"
            with patch.dict(os.environ, {"DATASET_INSPECTION_REPORT": str(sheet)}):
                app = AppTest.from_file(str(app_path), default_timeout=20).run()
            self.assertFalse(app.exception)
            distributions = [item.value for item in app.dataframe
                             if {"Category", "Records", "Percent"}.issubset(item.value.columns)]
            self.assertEqual(len(distributions), 2)
            self.assertEqual(distributions[0].set_index("Category").loc["Nepali-only", "Percent"], 25.0)
            self.assertEqual(distributions[0].set_index("Category").loc["Other", "Percent"], 25.0)
            self.assertEqual(distributions[0].set_index("Category").loc["Unknown", "Percent"], 25.0)
            self.assertEqual(distributions[0].set_index("Category").loc["Total", "Records"], 4)
            self.assertEqual(distributions[0].set_index("Category").loc["Total", "Percent"], 100.0)
            self.assertEqual(distributions[1].set_index("Category").loc["Devanagari", "Percent"], 100.0)
            per_run = [item.value for item in app.dataframe
                       if "Run" in item.value.columns and
                       "No language evidence (records)" in item.value.columns]
            self.assertEqual(len(per_run), 1)
            self.assertEqual(list(per_run[0]["No language evidence (records)"]), [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
