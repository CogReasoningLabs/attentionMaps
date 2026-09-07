import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from apps.dataset_generator import (
    _GenerationRateLedger,
    _GenerationStore,
    _contains_corrupted_text,
    _dataset_records_for_export,
    _parse_output_schema,
    _record_from_schema,
    _records_csv,
    _records_jsonl,
)
from attention_maps.inference.comparison import ComparisonConfigurationError


class DatasetGeneratorSchemaTests(unittest.TestCase):
    def test_parses_and_renders_output_schema(self):
        schema = _parse_output_schema(
            '{"instruction": "{prompt}", "response": "{output}"}'
        )

        record = _record_from_schema(
            schema,
            source="नेपाल",
            output="उत्तर",
            prompt="प्रश्न",
            system_prompt="",
            model="fake:model",
            decoding="T0.7",
            row_index=3,
        )

        self.assertEqual(record, {"instruction": "प्रश्न", "response": "उत्तर"})

    def test_rejects_invalid_or_empty_schema_templates(self):
        for value in ("[]", "{}", '{"response": 1}', '{"response": ""}'):
            with self.subTest(value=value):
                with self.assertRaises(ComparisonConfigurationError):
                    _parse_output_schema(value)

    def test_rejects_unknown_schema_placeholder(self):
        with self.assertRaisesRegex(
            ComparisonConfigurationError, "invalid output-schema placeholder"
        ):
            _record_from_schema(
                {"response": "{unknown}"},
                source="नेपाल",
                output="उत्तर",
                prompt="प्रश्न",
                system_prompt="",
                model="fake:model",
                decoding="T0.7",
                row_index=3,
            )

    def test_detects_replacement_characters_and_binary_controls(self):
        self.assertTrue(_contains_corrupted_text("bad\ufffdtext"))
        self.assertTrue(_contains_corrupted_text("bad\x00text"))
        self.assertFalse(_contains_corrupted_text("नेपाली\nपाठ"))


class DatasetGeneratorExportTests(unittest.TestCase):
    def setUp(self):
        self.generated = [
            {
                "_dataset_record": {"instruction": "प्रश्न", "response": "उत्तर"},
                "_status": "ok",
                "_error": "",
                "_model": "fake:model",
            },
            {
                "_dataset_record": {"instruction": "अर्को", "response": ""},
                "_status": "error",
                "_error": "provider unavailable",
                "_model": "fake:model",
            },
        ]

    def test_exports_only_successful_public_records(self):
        records = _dataset_records_for_export(self.generated)

        self.assertEqual(
            records, [{"instruction": "प्रश्न", "response": "उत्तर"}]
        )
        exported = json.loads(_records_jsonl(records))
        self.assertNotIn("_model", exported)

    def test_csv_uses_union_of_public_schema_fields(self):
        text = _records_csv(
            [
                {"instruction": "प्रश्न", "response": "उत्तर"},
                {"instruction": "अर्को", "source": "नेपाल"},
            ]
        )

        rows = list(csv.DictReader(io.StringIO(text)))
        self.assertEqual(
            list(rows[0]), ["instruction", "response", "source"]
        )
        self.assertEqual(rows[1]["source"], "नेपाल")

    def test_empty_exports_are_empty_strings(self):
        self.assertEqual(_records_jsonl([]), "")
        self.assertEqual(_records_csv([]), "")


class DatasetGeneratorPersistenceTests(unittest.TestCase):
    def test_rate_ledger_records_scoped_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = _GenerationRateLedger(Path(directory) / "limits.sqlite3")

            ledger.wait_and_record("fake:model", per_minute=2, per_day=5)

            self.assertEqual(ledger.usage("fake:model"), (1, 1))
            self.assertEqual(ledger.usage("other:model"), (0, 0))
            ledger.reset()
            self.assertEqual(ledger.usage("fake:model"), (0, 0))

    def test_store_round_trips_successful_records(self):
        with tempfile.TemporaryDirectory() as directory:
            store = _GenerationStore(Path(directory) / "runs.sqlite3")
            rows = [
                self._stored_row("ok", "उत्तर", ""),
                self._stored_row("error", "", "provider unavailable"),
            ]

            run_id = store.save_run(self._metadata(), rows)

            self.assertEqual(len(store.recent_runs()), 1)
            self.assertEqual(
                store.records_for_run(run_id),
                [{"instruction": "प्रश्न", "response": "उत्तर"}],
            )
            self.assertEqual(
                len(store.records_for_run(run_id, successful_only=False)), 2
            )

    @staticmethod
    def _metadata():
        return {
            "dataset_label": "test",
            "source_column": "text",
            "task_preset": "Question / answer",
            "system_prompt": "",
            "user_prompt": "{text}",
            "output_schema": {"instruction": "{prompt}", "response": "{output}"},
            "hyperparameters": {"model": "fake:model"},
        }

    @staticmethod
    def _stored_row(status, output, error):
        record = {"instruction": "प्रश्न", "response": output}
        return {
            "_source_row_index": "1",
            "_source_text": "नेपाल",
            "_rendered_prompt": "प्रश्न",
            "_model": "fake:model",
            "_decoding": "T0.7",
            "_generated_output": output,
            "_dataset_record": record,
            "_status": status,
            "_error": error,
            "_latency_seconds": 0.1,
        }


if __name__ == "__main__":
    unittest.main()
