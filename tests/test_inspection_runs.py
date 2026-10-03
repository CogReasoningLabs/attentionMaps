import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from attention_maps.explorer.inspection_runs import (
    _observe_records, build_inspection_report, iter_inspection_records, load_inspection_report,
)
from attention_maps.explorer.inspection_history import read_history
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

    def test_huggingface_native_batches_preserve_rows_and_use_requested_size(self):
        class BatchedStream:
            def __init__(self):
                self.requested = []

            def iter(self, batch_size):
                self.requested.append(batch_size)
                yield {"text": ["नेपाल", "English"], "language": ["ne", "en"]}
                yield {"text": ["नेपाली"], "language": ["ne"]}

        stream = BatchedStream()
        inventory = {
            "format": "huggingface", "dataset_id": "owner/data", "dataset_config": "ne",
            "dataset_split": "train", "dataset_shards": ["second.parquet"],
            "filter_column": "language", "filter_value": "ne",
        }
        with patch("datasets.load_dataset", return_value=stream):
            rows = list(iter_inspection_records(inventory, batch_size=7))
        self.assertEqual(stream.requested, [7])
        self.assertEqual(rows, [
            {"text": "नेपाल", "language": "ne"},
            {"text": "नेपाली", "language": "ne"},
        ])

    def test_interrupted_huggingface_batch_reader_is_closed(self):
        events = []

        class BatchedStream:
            def iter(self, batch_size):
                try:
                    yield {"text": ["first", "second"]}
                finally:
                    events.append("batch reader closed")

        inventory = {
            "format": "huggingface", "dataset_id": "owner/data", "dataset_config": "default",
            "dataset_split": "nep", "dataset_shards": ["one.parquet"],
        }
        with patch("datasets.load_dataset", return_value=BatchedStream()):
            rows = iter_inspection_records(inventory, batch_size=2)
            self.assertEqual(next(rows), {"text": "first"})
            rows.close()
        self.assertEqual(events, ["batch reader closed"])

    def test_nepali_source_split_can_be_sampled_with_parallel_workers(self):
        class BatchedStream:
            def iter(self, batch_size):
                yield {"doc_id": ["a", "b", "c"],
                       "text": ["नेपाल राम्रो छ", "अर्को नेपाली लेख", "यो पनि नेपाली हो"]}

        inventory = {
            "format": "huggingface", "provider": "huggingface", "dataset_id": "owner/corpus",
            "dataset_config": "verified", "dataset_split": "nep", "dataset_revision": "frozen",
            "dataset_shards": ["one.parquet"], "rows": 3,
            "schema": [{"column": "doc_id", "type": "string"}, {"column": "text", "type": "string"}],
            "columns": ["doc_id", "text"],
        }
        settings = {
            "training_schema": "pretraining", "field_mapping": {"id": "doc_id"},
            "text_columns": ["text"], "language_detection": "none",
            "sample_fraction": 1.0, "sampling_runs": 1, "concurrency": 2,
            "batch_size": 2, "seed": 42,
        }
        with patch("datasets.load_dataset", return_value=BatchedStream()):
            report = build_inspection_report(inventory, settings)
        self.assertEqual(report["language_status"]["sampling"]["rows_scanned"], 3)
        self.assertEqual(report["language_status"]["sampling"]["unique_sampled_records"], 3)
        self.assertEqual(report["inventory"]["dataset_split"], "nep")

    def test_source_reader_closes_before_worker_pool_on_error(self):
        events = []

        class Records:
            def __iter__(self):
                yield {"text": "bad"}

            def close(self):
                events.append("records closed")

        class Processor:
            def __init__(self, _analysis):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                events.append("pool closed")

            def observe(self, _record):
                raise ValueError("invalid instance")

        with patch("attention_maps.explorer.inspection_runs._EvidenceBatchProcessor", Processor):
            with self.assertRaisesRegex(ValueError, "invalid instance"):
                _observe_records(type("Analysis", (), {"population": 1})(), Records())
        self.assertEqual(events, ["records closed", "pool closed"])

    def test_staged_kaggle_parquet_uses_the_same_row_batch_size(self):
        parquet = MagicMock()
        parquet.__enter__.return_value = parquet
        row_batch = MagicMock()
        row_batch.to_pylist.return_value = [{"text": "नेपाल"}, {"text": "नेपाली"}]
        parquet.iter_batches.return_value = [row_batch]
        inventory = {
            "format": "parquet", "provider": "kaggle",
            "source_files": ["cached.parquet"],
        }
        with patch("pyarrow.parquet.ParquetFile", return_value=parquet) as open_file:
            rows = list(iter_inspection_records(inventory, batch_size=7))
        open_file.assert_called_once_with("cached.parquet")
        parquet.iter_batches.assert_called_once_with(batch_size=7)
        self.assertEqual(rows, [{"text": "नेपाल"}, {"text": "नेपाली"}])

    def test_declared_instruction_instance_analyzes_all_turns(self):
        self.write_rows([{"messages": [
            {"role": "user", "content": "नेपाल"},
            {"role": "assistant", "content": "राम्रो"},
        ], "language": "ne"}])
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(["--local", str(self.source), "--training-schema", "instruction_finetuning",
                         "--text-column", "stale_text_column", "--sample-fraction", "1",
                         "--sampling-runs", "1", "--output", str(self.report)])
        self.assertEqual(code, 0)
        report = load_inspection_report(self.report)
        self.assertEqual(report["instance_definition"]["unit"], "one complete conversation")
        self.assertEqual(report["language_status"]["sampled_characters"], len("नेपाल\n\nराम्रो"))
        self.assertEqual(report["language_status"]["script"], "Devanagari")

    def test_explicit_skip_policy_samples_only_complete_instruction_pairs(self):
        self.write_rows([
            {"instruction": "नेपालको राजधानी?", "output": "काठमाडौं", "language": "ne"},
            {"instruction": "अर्को प्रश्न", "output": None, "language": "ne"},
            {"instruction": "", "output": "उत्तर", "language": "ne"},
            {"instruction": "नेपालीमा लेख", "input": "विषय", "output": "नेपाली उत्तर", "language": "ne"},
        ])
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(["--local", str(self.source), "--training-schema", "instruction_finetuning",
                         "--invalid-instance-policy", "skip", "--sample-fraction", "1",
                         "--sampling-runs", "2", "--concurrency", "2", "--batch-size", "1",
                         "--output", str(self.report)])
        self.assertEqual(code, 0)
        report = load_inspection_report(self.report)
        selection = report["instance_selection"]
        self.assertEqual((selection["source_rows"], selection["eligible_rows"], selection["skipped_rows"]),
                         (4, 2, 2))
        self.assertEqual(selection["reason_counts"],
                         {"assistant message has empty content": 1, "instruction must be non-empty text": 1})
        self.assertEqual([item["source_row"] for item in selection["examples"]], [1, 2])
        sampling = report["language_status"]["sampling"]
        self.assertEqual((sampling["population_rows"], sampling["rows_per_run"], sampling["unique_sampled_records"]),
                         (2, 2, 2))
        self.assertEqual(report["language_status"]["language_coverage"], "Nepali-only")
        self.assertEqual(report["language_status"]["script"], "Devanagari")
        saved = read_history(self.report)[0]
        self.assertEqual((saved["Eligible instances"], saved["Skipped instances"]), ("2", "2"))

    def test_mapped_string_list_is_parsed_as_one_pretraining_document(self):
        self.write_rows([{"paragraphs": ["नेपाल राम्रो छ", "अर्को अनुच्छेद"], "language": "ne"}])
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(["--local", str(self.source), "--training-schema", "pretraining",
                         "--field-map", "text=paragraphs", "--field-parser", "text=join_strings",
                         "--sample-fraction", "1", "--sampling-runs", "1", "--output", str(self.report)])
        self.assertEqual(code, 0)
        report = load_inspection_report(self.report)
        self.assertEqual(report["instance_definition"]["field_parsers"], {"text": "join_strings"})
        self.assertEqual(report["language_status"]["sampled_characters"], len("नेपाल राम्रो छ\n\nअर्को अनुच्छेद"))
        self.assertEqual(report["language_status"]["script"], "Devanagari")

    def test_schema_mismatch_fails_before_saving_classification(self):
        self.write_rows([{"text": "नेपाल", "language": "ne"}])
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as errors:
            code = main(["--local", str(self.source), "--training-schema", "instruction_finetuning",
                         "--sample-fraction", "1", "--sampling-runs", "1", "--output", str(self.report)])
        self.assertEqual(code, 1)
        self.assertIn("Invalid instruction_finetuning instance", errors.getvalue())
        self.assertFalse(self.report.exists())

    def test_supervised_mapping_counts_input_text_without_label_or_task(self):
        self.write_rows([{"payload": {"body": "नेपाल", "category": "positive"}, "language": "ne"}])
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(["--local", str(self.source), "--training-schema", "task_specific_supervised",
                         "--field-map", "text=payload.body", "--field-map", "label=payload.category",
                         "--task-name", "sentiment", "--sample-fraction", "1", "--sampling-runs", "1",
                         "--output", str(self.report)])
        self.assertEqual(code, 0)
        report = load_inspection_report(self.report)
        self.assertEqual(report["instance_definition"]["field_mapping"],
                         {"text": "payload.body", "label": "payload.category"})
        self.assertEqual(report["language_status"]["sampled_characters"], len("नेपाल"))
        self.assertNotIn("Latin", report["language_status"]["script_counts"])

    def test_evaluation_mapping_inspects_both_sides_without_task_marker(self):
        self.write_rows([{"corrupted": "नेपाल सन्दर छ", "clean": "नेपाल सुन्दर छ", "language": "ne"}])
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(["--local", str(self.source), "--training-schema", "evaluation",
                         "--field-map", "input=corrupted", "--field-map", "reference=clean",
                         "--task-name", "ocr_proofreading", "--sample-fraction", "1",
                         "--sampling-runs", "1", "--output", str(self.report)])
        self.assertEqual(code, 0)
        report = load_inspection_report(self.report)
        self.assertEqual(report["instance_definition"]["schema"], "evaluation")
        self.assertEqual(report["instance_definition"]["unit"], "one input/reference example")
        self.assertEqual(report["language_status"]["sampled_characters"],
                         len("नेपाल सन्दर छ\n\nनेपाल सुन्दर छ"))
        self.assertEqual(report["language_status"]["script"], "Devanagari")
        self.write_rows([{"corrupted": "नेपाल", "clean": None, "language": "ne"}])
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(main(["--local", str(self.source), "--training-schema", "evaluation",
                                   "--field-map", "input=corrupted", "--field-map", "reference=clean",
                                   "--sample-fraction", "1", "--sampling-runs", "1",
                                   "--output", str(self.root / "invalid.csv")]), 1)
        self.assertIn("Invalid evaluation instance", errors.getvalue())
        self.assertFalse((self.root / "invalid.csv").exists())

    def test_preference_pair_counts_prompt_and_both_branches(self):
        self.write_rows([{"prompt": "नेपाल", "chosen": "राम्रो", "rejected": "नराम्रो", "language": "ne"}])
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(["--local", str(self.source), "--training-schema", "preference_tuning",
                         "--sample-fraction", "1", "--sampling-runs", "1", "--concurrency", "2",
                         "--batch-size", "1", "--output", str(self.report)])
        self.assertEqual(code, 0)
        report = load_inspection_report(self.report)
        self.assertEqual(report["instance_definition"]["unit"], "one prompt/chosen/rejected group")
        self.assertEqual(report["language_status"]["sampled_characters"], len("नेपाल\n\nराम्रो\n\nनराम्रो"))
        self.assertEqual(report["language_status"]["script"], "Devanagari")

    def test_blank_line_documents_are_sampled_as_atomic_instances(self):
        source = self.root / "documents.txt"
        source.write_text("पहिलो\nदोस्रो\n\nतेस्रो\nचौथो\n", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(["--local", str(source), "--training-schema", "pretraining",
                         "--text-record-unit", "blank_line", "--sample-fraction", "1",
                         "--sampling-runs", "1", "--language-detection", "none",
                         "--output", str(self.report)])
        self.assertEqual(code, 0)
        report = load_inspection_report(self.report)
        self.assertEqual(report["inventory"]["rows"], 4)
        self.assertEqual(report["language_status"]["sampling"]["population_rows"], 2)
        self.assertEqual(report["language_status"]["sampled_records"], 2)

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
