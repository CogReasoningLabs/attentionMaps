import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from attention_maps.explorer.inspection_examples import InspectionExamples, TEXT_LIMIT
from attention_maps.explorer.sampling_vote import RepeatedSampleAnalysis, validate_sampling_result
from tests.inspection_helpers import isolated_main as main


def analyze(records, seed=42, detector=None, inventory=None):
    settings = {"seed": seed, "sample_fraction": 1.0, "sampling_runs": 5}
    with patch("attention_maps.explorer.sampling_vote.make_language_detector", return_value=detector):
        analysis = RepeatedSampleAnalysis(inventory or {}, settings, ["text"], len(records))
    for record in records:
        analysis.observe(record)
    return analysis.finish()


class InspectionExampleTests(unittest.TestCase):
    def test_examples_cover_computed_categories_and_retain_full_script_evidence(self):
        text = "नेपाल " * 3000 + "English tail"
        records = [
            {"text": text, "language": "ne"},
            {"text": "namaste", "language": "ne"},
            {"text": "नेपाल namaste", "language": "ne"},
            {"text": "नेपाल hello", "language_pair": "en-ne"},
            {"text": "English", "language": "en"},
            {"text": "हिन्दी", "language": "hi"},
            {"text": "中文", "language": "ne"},
            {"text": "123", "language": "ne"},
            {"text": "नेपाल"},
        ]
        status = analyze(records)
        validate_sampling_result(status)
        groups = status["record_examples"]["groups"]
        for kind in ("language", "script"):
            for category, examples in groups[kind].items():
                for item in examples:
                    self.assertEqual(item[f"{kind}_category"], category)
                    self.assertEqual(item["text"], records[item["population_row"] - 1]["text"][:TEXT_LIMIT])
        long = groups["script"]["Devanagari"][0]
        self.assertTrue(long["text_truncated"])
        self.assertEqual(long["text_characters"], len(text))
        self.assertEqual(long["script_counts"]["Latin"], 11)
        unknown = groups["language"]["Unknown"][0]
        self.assertEqual(unknown["script_category"], "Not applicable")
        self.assertEqual(unknown["script_percentages"]["Devanagari"], 100.0)

    def test_random_examples_are_bounded_unique_reproducible_and_not_a_prefix(self):
        records = [{"text": f"नेपाल {index}", "language": "ne"} for index in range(200)]
        first, second = analyze(records), analyze(records)
        self.assertEqual(first, second)
        examples = first["record_examples"]["groups"]["language"]["Nepali-only"]
        positions = {item["population_row"] for item in examples}
        self.assertEqual(len(positions), 10)
        self.assertTrue(any(position > 10 for position in positions))
        self.assertNotEqual(first["record_examples"], analyze(records, seed=9)["record_examples"])
        validate_sampling_result(first)
        bad = copy.deepcopy(first)
        bad["record_examples"]["groups"]["language"]["Nepali-only"][1] = examples[0]
        with self.assertRaisesRegex(ValueError, "invalid example evidence"):
            validate_sampling_result(bad)

    def test_recording_examples_does_not_change_counts_votes_or_detector_calls(self):
        class Detector:
            metadata = {"method": "fasttext"}
            def __init__(self):
                self.calls = []
            def detect(self, text):
                self.calls.append(text)
                reason = "too_short" if text == "क" else "low_confidence" if text == "uncertain" else "accepted"
                return {"language": "ne" if reason == "accepted" else None,
                        "score": None if reason == "too_short" else .3 if reason == "low_confidence" else .99,
                        "truncated": False, "reason": reason}
        records = [{"text": "नेपाल"}, {"text": "क"}, {"text": "uncertain"}]
        one, two = Detector(), Detector()
        status = analyze(records, detector=one)
        with patch.object(InspectionExamples, "observe", return_value=None):
            baseline = analyze(records, detector=two)
        examples = status.pop("record_examples")
        baseline.pop("record_examples")
        self.assertEqual(status, baseline)
        self.assertEqual(one.calls, two.calls)
        self.assertEqual(len(one.calls), len(records))
        unknown = examples["groups"]["language"]["Unknown"]
        self.assertEqual({item["language_reason"] for item in unknown}, {"too_short", "low_confidence"})
        self.assertIn(.3, [item["prediction"]["score"] for item in unknown])

    def test_huggingface_partition_examples_use_the_saved_fallback_evidence(self):
        status = analyze([{"text": "namaste"}], inventory={"format": "huggingface", "dataset_split": "nep"})
        example = status["record_examples"]["groups"]["script"]["Romanized"][0]
        self.assertEqual(example["languages"], ["ne"])
        self.assertEqual(example["language_origin"], "Hugging Face partition")
        validate_sampling_result(status)

    def test_streamlit_switches_five_ten_and_category_without_source_access(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.jsonl", root / "history.csv"
            records = [{"text": f"नेपाल {index}", "language": "ne"} for index in range(20)]
            records += [{"text": "namaste", "language": "ne"}, {"text": "क"}]
            source.write_text("".join(json.dumps(row) + "\n" for row in records))
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--local", str(source), "--sample-fraction", "1", "--output", str(output)]), 0)
            before = output.read_bytes()
            source.unlink()
            with patch.dict(os.environ, {"DATASET_INSPECTION_REPORT": str(output)}), \
                 patch("attention_maps.explorer.inspection_runs.build_inspection_report", side_effect=AssertionError("No reanalysis")):
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "apps/dataset_explorer.py"), default_timeout=20).run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.text), 5)
                next(item for item in app.radio if item.label == "Examples to show").set_value(10).run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.text), 10)
                next(item for item in app.selectbox if item.label == "Example category").set_value("Romanized").run()
                self.assertEqual(app.text[0].value, "namaste")
                next(item for item in app.radio if item.label == "Inspect by").set_value("Language").run()
                next(item for item in app.selectbox if item.label == "Example category").set_value("Unknown").run()
                self.assertFalse(app.exception)
                self.assertEqual(app.text[0].value, "क")
                self.assertTrue(any("no metadata or detector" in item.value for item in app.caption))
                self.assertTrue(any("Observed writing system: Devanagari (100.00%)" in item.value for item in app.markdown))
                self.assertFalse(any("Script: Not applicable" in item.value for item in app.markdown))
                self.assertTrue(any("Nepali-specific script category: not assigned" in item.value for item in app.caption))
                next(item for item in app.radio if item.label == "Inspect by").set_value("Script").run()
                selector = next(item for item in app.selectbox if item.label == "Example category")
                self.assertIn("Excluded: no accepted Nepali evidence", selector.options)
            self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
