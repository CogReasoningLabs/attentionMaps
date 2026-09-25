import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from attention_maps.explorer.inspection_runs import load_inspection_report
from scripts.inspect_dataset import _arguments
from tests.inspection_helpers import isolated_main as main


class InspectionRunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.jsonl"
        self.filtered = self.root / "kept.jsonl"
        self.report = self.root / "report.csv"

    def write_rows(self, rows):
        self.source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    def run_filter(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as errors:
            code = main(["--local", str(self.source), "--min-devanagari-ratio", "0.8",
                         "--text-column", "text", "--filtered-output", str(self.filtered),
                         "--output", str(self.report), *extra])
        return code, errors.getvalue()

    def test_settings_file_filters_all_rows_and_keeps_complete_records(self):
        rows = [
            {"text": "नेपाल सुन्दर छ।", "language": "ne", "metadata": {"id": 1}},
            {"text": "English text", "language": "en", "metadata": {"id": 2}},
            {"text": "नेपाली भाषा", "language": "ne", "metadata": {"id": 3}},
            {"text": "123!", "language": "ne", "metadata": {"id": 4}},
        ]
        self.write_rows(rows)
        settings = self.root / "run.yaml"
        settings.write_text("local: source.jsonl\ntext_columns: [text]\nsample_size: 1\n"
                            "min_devanagari_ratio: 0.8\nfiltered_output: kept.jsonl\noutput: report.csv\n")
        original = self.source.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--settings", str(settings)]), 0)
        report = load_inspection_report(self.report)
        result = report["filter_result"]
        self.assertEqual([json.loads(line) for line in self.filtered.read_text().splitlines()], [rows[0], rows[2]])
        self.assertEqual((result["rows_scanned"], result["rows_kept"], result["rows_dropped"]), (4, 2, 2))
        self.assertEqual(result["scope"], "complete_selection")
        self.assertTrue(result["selection_exhausted"])
        self.assertEqual(result["language_status"]["script"], "Devanagari")
        self.assertEqual(report["language_status"]["sampled_records"], 1)
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(report["settings"]["local"], str(self.source))

    def test_limit_is_explicit_and_sample_size_does_not_limit_filtering(self):
        self.write_rows([{"text": "नेपाल", "language": "ne"}, {"text": "English", "language": "en"}, {"text": "नेपाली", "language": "ne"}])
        self.assertEqual(self.run_filter("--max-records", "2", "--sample-fraction", "1")[0], 0)
        report = load_inspection_report(self.report)
        result = report["filter_result"]
        self.assertEqual((result["rows_scanned"], result["rows_kept"]), (2, 1))
        self.assertFalse(result["selection_exhausted"])
        self.assertEqual(report["sample_scope"], "limited_prefix")
        self.assertEqual(report["language_status"]["script"], "Mixed (Nepali + English)")
        self.assertEqual(self.run_filter("--max-records", "3", "--sample-fraction", "1", "--overwrite")[0], 0)
        self.assertTrue(load_inspection_report(self.report)["filter_result"]["selection_exhausted"])

    def test_ratio_boundary_and_empty_text(self):
        self.write_rows([{"text": text, "language": "ne"} for text in ("कa", "नेपाल", "123!!!", "abc", "")])
        self.assertEqual(self.run_filter("--min-devanagari-ratio", "0.5", "--sample-fraction", "1")[0], 0)
        self.assertEqual([json.loads(line)["text"] for line in self.filtered.read_text().splitlines()], ["कa", "नेपाल"])
        self.assertEqual(self.run_filter("--min-devanagari-ratio", "0", "--sample-fraction", "1", "--overwrite")[0], 0)
        self.assertEqual(load_inspection_report(self.report)["filter_result"]["rows_kept"], 2)

    def test_nested_text_fields_preserve_conversation_and_metadata(self):
        rows = [{"payload": {"messages": [{"role": "user", "content": "नेपाल"}]}, "id": 1, "language": "ne"},
                {"payload": {"messages": [{"role": "user", "content": "English"}]}, "id": 2, "language": "en"}]
        self.write_rows(rows)
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(["--local", str(self.source), "--text-column", "payload.messages",
                         "--min-devanagari-ratio", "1", "--filtered-output", str(self.filtered),
                         "--output", str(self.report), "--sample-fraction", "1"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(self.filtered.read_text()), rows[0])

    def test_huggingface_filter_reads_exact_config_split_revision_and_second_shard(self):
        catalog = {"dataset_id": "owner/data", "configs": ["ne"], "revision": "pinned"}
        configuration = {"dataset_id": "owner/data", "config": "ne", "revision": "pinned",
                         "languages": ["ne"], "schema": [{"column": "text", "type": "string"}],
                         "splits": {"test": {"rows": 200, "memory_bytes": 500,
                             "shards": [{"name": f"part-{n}.jsonl", "path": f"hf://part-{n}.jsonl", "bytes": n * 100}
                                        for n in (1, 2)]}}}
        with patch("scripts.inspect_dataset.discover_huggingface_dataset", return_value=catalog), \
             patch("scripts.inspect_dataset.inspect_huggingface_configuration", return_value=configuration), \
             patch("datasets.load_dataset", return_value=[{"text": "नेपाल"}, {"text": "English"}]) as loader, \
             contextlib.redirect_stdout(io.StringIO()):
            code = main(["--dataset", "owner/data", "--config", "ne", "--split", "test",
                         "--shard", "part-2.jsonl", "--min-devanagari-ratio", "0.8",
                         "--filtered-output", str(self.filtered), "--output", str(self.report)])
        self.assertEqual(code, 0)
        self.assertEqual(loader.call_args.args, ("owner/data", "ne"))
        self.assertEqual(loader.call_args.kwargs["data_files"], {"test": ["hf://part-2.jsonl"]})
        self.assertEqual(loader.call_args.kwargs["revision"], "pinned")
        report = load_inspection_report(self.report)
        self.assertIsNone(report["inventory"]["rows"])
        self.assertEqual(report["inventory"]["bytes"], 200)
        self.assertEqual(report["filter_result"]["rows_scanned"], 2)

    def test_invalid_settings_and_output_collisions_fail_before_writing(self):
        self.write_rows([{"text": "नेपाल", "language": "ne"}])
        original = self.source.read_bytes()
        for flags in (("--min-devanagari-ratio", "nan"), ("--max-records", "0"),
                      ("--output", str(self.source)), ("--filtered-output", str(self.source)),
                      ("--output", str(self.filtered))):
            with self.subTest(flags=flags):
                self.assertEqual(self.run_filter(*flags)[0], 1)
                self.assertFalse(self.report.exists())
                self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(self.run_filter()[0], 0)
        self.assertEqual(self.run_filter()[0], 1)  # Reusing filtered output requires explicit --overwrite.

    def test_failed_stream_leaves_previous_artifacts_intact(self):
        self.write_rows([{"text": "नेपाल", "language": "ne"}])
        self.assertEqual(self.run_filter()[0], 0)
        previous = (self.filtered.read_bytes(), self.report.read_bytes())

        def failing_stream(*_args, **_kwargs):
            yield {"text": "नयाँ"}
            raise OSError("interrupted read")

        with patch("attention_maps.explorer.inspection_runs.iter_inspection_records", side_effect=failing_stream):
            self.assertEqual(self.run_filter("--overwrite")[0], 1)
        self.assertEqual((self.filtered.read_bytes(), self.report.read_bytes()), previous)
        self.assertFalse(list(self.root.glob(".*.tmp")))

    def test_cli_overrides_settings_lists_and_rejects_typos(self):
        settings = self.root / "run.yaml"
        settings.write_text("dataset: owner/data\nshards: [first.jsonl]\ntext_columns: [old]\nlanguages: [ne]\n")
        _, effective = _arguments(["--settings", str(settings), "--shard", "second.jsonl",
                                   "--text-column", "new", "--language", "en"])
        self.assertEqual(effective["shards"], ["second.jsonl"])
        self.assertEqual(effective["text_columns"], ["new"])
        self.assertEqual(effective["languages"], ["en"])
        settings.write_text("local: source.jsonl\nsample_sze: 12\n")
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(main(["--settings", str(settings)]), 1)
        self.assertIn("sample_sze", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
