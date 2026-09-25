import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from attention_maps.explorer.sampling_vote import RandomWithoutReplacement, RepeatedSampleAnalysis, majority_vote
from attention_maps.explorer.inspection_runs import _observe_records, build_inspection_report, load_inspection_report
from scripts.inspect_dataset import _arguments
from tests.inspection_helpers import isolated_main as main


def settings(**overrides):
    return {"sample_fraction": 0.2, "sampling_runs": 5, "sampling_method": "random", "seed": 42,
            "sample_size": None, "text_columns": ["text"], "languages": [], **overrides}


class RepeatedSamplingTests(unittest.TestCase):
    def test_exact_random_samples_are_reproducible_without_replacement(self):
        def selected(seed):
            sampler = RandomWithoutReplacement(101, 0.2, seed)
            return [index for index in range(101) if sampler.select(index)]

        first = selected(42)
        self.assertEqual(len(first), 21)
        self.assertEqual(len(set(first)), 21)
        self.assertEqual(first, selected(42))
        self.assertNotEqual(first, selected(43))
        self.assertTrue(any(index > 90 for index in first))
        sampler = RandomWithoutReplacement(100, 0.07, 42)
        self.assertEqual(sum(sampler.select(i) for i in range(100)), 7)

    def test_random_selection_is_not_biased_to_prefix_positions(self):
        counts = [0] * 10
        for seed in range(1000):
            selector = RandomWithoutReplacement(10, 0.2, seed)
            for index in range(10):
                counts[index] += selector.select(index)
        self.assertTrue(all(150 < count < 250 for count in counts), counts)

    def test_unique_coverage_matches_union_and_aggregate_has_no_overlap_double_counting(self):
        analysis = RepeatedSampleAnalysis({}, settings(), ["text"], 100)
        selectors = [RandomWithoutReplacement(100, 0.2, seed) for seed in analysis.seeds]
        union = set()
        for index in range(100):
            membership = [selector.select(index) for selector in selectors]
            if any(membership):
                union.add(index)
            analysis.observe({"text": "क", "language": "ne"})
        result = analysis.finish()
        sample = result["sampling"]
        self.assertEqual(sample["rows_per_run"], 20)
        self.assertEqual(sample["sampled_record_occurrences"], 100)
        self.assertEqual(sample["unique_sampled_records"], len(union))
        self.assertLess(len(union), 100)
        self.assertEqual(result["sampled_characters"], len(union))
        self.assertEqual(result["script_counts"], {"Devanagari": len(union)})
        self.assertAlmostEqual(sample["expected_unique_coverage"], 1 - 0.8**5)
        self.assertEqual(result["voting"]["script"]["winning_votes"], 5)

    def test_parallel_batches_match_serial_counts_votes_and_samples(self):
        rows = [
            {"text": "नेपाल र Kathmandu" if index % 3 == 0 else "English text" if index % 3 == 1 else "नेपाली पाठ",
             "language": "ne" if index % 3 != 1 else "en"}
            for index in range(137)
        ]
        results = []
        for workers, batch_size in ((1, 1024), (6, 7), (3, 1)):
            analysis = RepeatedSampleAnalysis(
                {}, settings(concurrency=workers, batch_size=batch_size), ["text"], len(rows)
            )
            _observe_records(analysis, iter(rows))
            result = analysis.finish()
            self.assertEqual(result["sampling"]["concurrency"], workers)
            self.assertEqual(result["sampling"]["batch_size"], batch_size)
            result["sampling"].pop("concurrency")
            result["sampling"].pop("batch_size")
            results.append(result)
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0], results[2])

    def test_all_selected_text_is_analyzed_beyond_old_character_cap(self):
        analysis = RepeatedSampleAnalysis({}, settings(sample_fraction=1), ["text"], 1)
        text = "नेपाल " * 30000 + "English"
        analysis.observe({"text": text, "language": "ne"})
        report = analysis.finish()
        self.assertEqual(report["script"], "Devanagari")
        self.assertGreater(report["script_counts"]["Latin"], 0)  # Counted but below the mixed-script threshold.
        self.assertEqual(report["sampled_characters"], len(text))
        self.assertIsNone(report["character_limit"])
        self.assertTrue(all(run["sampled_characters"] == len(text) for run in report["sampling"]["run_results"]))

    def test_vote_requires_strict_majority_and_keeps_unknown_votes_in_denominator(self):
        self.assertEqual(majority_vote(["Devanagari"] * 3 + ["Latin"] * 2)["label"], "Devanagari")
        for labels in (["Devanagari", "Latin"], ["Devanagari"] * 2 + ["Latin"] * 2 + ["Other"],
                       ["Unknown"] * 3 + ["Devanagari"] * 2):
            with self.subTest(labels=labels):
                result = majority_vote(labels)
                self.assertEqual(result["label"], "Unknown")
                self.assertEqual(result["decision"], "inconclusive")
        self.assertEqual(majority_vote(["Latin"] * 3 + ["Unknown"] * 2)["agreement"], 0.6)

    def test_empty_population_abstains_even_with_declared_language(self):
        result = RepeatedSampleAnalysis({}, settings(languages=["ne"]), ["text"], 0).finish()
        self.assertEqual(result["language_coverage"], "Unknown")
        self.assertEqual(result["script"], "Unknown")
        self.assertEqual(result["sampling"]["unique_coverage"], 0)

    def test_population_mismatch_fails_instead_of_reporting_false_coverage(self):
        analysis = RepeatedSampleAnalysis({}, settings(), ["text"], 2)
        analysis.observe({"text": "नेपाल"})
        with self.assertRaisesRegex(ValueError, "row count changed"):
            analysis.finish()
        analysis.observe({"text": "नेपाल"})
        with self.assertRaisesRegex(ValueError, "row count changed"):
            analysis.observe({"text": "नेपाल"})

    def test_huggingface_unknown_population_uses_count_then_one_analysis_pass(self):
        inventory = {"format": "huggingface", "dataset_id": "owner/data", "dataset_config": "ne",
                     "dataset_revision": "pinned", "dataset_split": "test", "dataset_shards": ["second.jsonl"],
                     "rows": None, "columns": ["text"], "schema": [{"column": "text", "type": "string"}]}
        records = [{"text": "नेपाल"}] * 50
        with patch("datasets.load_dataset", side_effect=lambda *args, **kwargs: iter(records)) as loader, \
             contextlib.redirect_stderr(io.StringIO()):
            report = build_inspection_report(inventory, settings())
        self.assertEqual(loader.call_count, 2)
        for call in loader.call_args_list:
            self.assertEqual(call.kwargs["data_files"], {"test": ["second.jsonl"]})
            self.assertEqual(call.kwargs["revision"], "pinned")
        sampling = report["language_status"]["sampling"]
        self.assertEqual(sampling["population_basis"], "counting pass")
        self.assertEqual(sampling["rows_per_run"], 10)


class RepeatedSamplingCLITests(unittest.TestCase):
    def test_defaults_and_legacy_override_are_explicit(self):
        _, values = _arguments(["--local", "example.jsonl"])
        self.assertEqual((values["sample_fraction"], values["sampling_runs"]), (0.2, 5))
        self.assertEqual((values["concurrency"], values["batch_size"]), (1, 1024))
        _, values = _arguments(["--local", "example.jsonl", "--concurrency", "6", "--batch-size", "64"])
        self.assertEqual((values["concurrency"], values["batch_size"]), (6, 64))
        _, values = _arguments(["--local", "example.jsonl", "--sample-size", "100"])
        self.assertIsNone(values["sample_fraction"])
        self.assertEqual(values["sampling_runs"], 1)
        for flags in (("--sample-fraction", "0"), ("--sample-fraction", "nan"),
                      ("--sample-fraction", "1.1"), ("--sampling-runs", "0"),
                      ("--sample-size", "20", "--sampling-runs", "3"),
                      ("--concurrency", "0"), ("--batch-size", "0")):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                _arguments(["--local", "example.jsonl", *flags])

    def test_percentage_overrides_legacy_yaml_and_roundtrips_saved_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output, config = root / "data.jsonl", root / "report.csv", root / "settings.yaml"
            source.write_text("".join(json.dumps({"text": "नेपाल", "language": "ne"}) + "\n" for _ in range(50)))
            config.write_text("local: data.jsonl\nsample_size: 100\noutput: report.csv\n")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--settings", str(config), "--sample-fraction", "0.2", "--sampling-runs", "3"]), 0)
            report = load_inspection_report(output)
            self.assertEqual(report["language_status"]["sampling"]["rows_per_run"], 10)
            self.assertEqual(report["language_status"]["voting"]["script"]["winning_votes"], 3)
            self.assertEqual(report["language_status"]["sampling"]["runs"], 3)

    def test_filter_voting_uses_scanned_prefix_and_retained_output_as_separate_populations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output, kept = root / "data.jsonl", root / "report.csv", root / "kept.jsonl"
            source.write_text("".join(json.dumps({"text": "नेपाल" if i % 2 else "English", "language": "ne" if i % 2 else "en"}) + "\n" for i in range(100)))
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["--local", str(source), "--sample-fraction", "0.2", "--sampling-runs", "4",
                             "--min-devanagari-ratio", "1", "--max-records", "20", "--filtered-output", str(kept),
                             "--concurrency", "6", "--batch-size", "3", "--output", str(output)])
            self.assertEqual(code, 0)
            report = load_inspection_report(output)
            self.assertEqual(report["sample_scope"], "limited_prefix")
            self.assertEqual(report["language_status"]["sampling"]["population_rows"], 20)
            self.assertEqual(report["language_status"]["sampling"]["rows_per_run"], 4)
            filtered = report["filter_result"]
            self.assertEqual(filtered["rows_kept"], 10)
            self.assertEqual(filtered["language_status"]["sampling"]["population_rows"], 10)
            self.assertEqual(filtered["language_status"]["sampling"]["rows_per_run"], 2)
            self.assertEqual(filtered["language_status"]["voting"]["script"]["winning_votes"], 4)


if __name__ == "__main__":
    unittest.main()
