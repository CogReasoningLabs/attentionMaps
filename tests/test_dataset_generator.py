import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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

    def test_rate_ledger_reports_live_wait_time(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = _GenerationRateLedger(Path(directory) / "limits.sqlite3")
            waits = []
            with patch(
                "apps.dataset_generator.time.time",
                side_effect=[100.0, 100.0, 161.0],
            ), patch("apps.dataset_generator.time.sleep") as sleep:
                ledger.wait_and_record("fake:model", per_minute=1, per_day=5)
                ledger.wait_and_record(
                    "fake:model",
                    per_minute=1,
                    per_day=5,
                    on_wait=waits.append,
                )

            self.assertEqual(waits, [60.0])
            sleep.assert_called_once_with(60.0)

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


class DatasetGeneratorAppTests(unittest.TestCase):
    def test_inference_dropdown_includes_api_and_local_gemma(self):
        try:
            from streamlit.testing.v1 import AppTest
        except ModuleNotFoundError:
            self.skipTest("Streamlit is not installed")

        app = AppTest.from_file(
            Path(__file__).resolve().parents[1] / "apps/dataset_generator.py"
        ).run(timeout=30)

        model_selector = next(
            item for item in app.multiselect if item.label == "Inference backends"
        )
        self.assertIn("Gemma 4 · Google API", model_selector.options)
        self.assertIn(
            "Gemma 4 E2B Base · Hugging Face local", model_selector.options
        )
        self.assertIn(
            "IRIIS GPT-2 Instruct Nepali 124M · Hugging Face local",
            model_selector.options,
        )
        self.assertIn(
            "IRIIS GPT-2 Nepali 124M Base · Hugging Face local",
            model_selector.options,
        )
        model_selector.set_value(
            ["IRIIS GPT-2 Instruct Nepali 124M · Hugging Face local"]
        ).run(timeout=30)
        max_tokens = next(
            item for item in app.number_input if item.label == "Max output tokens"
        )
        self.assertEqual(max_tokens.value, 256)
        self.assertTrue(
            any(
                item.label == "Local model device"
                for item in app.selectbox
            )
        )

    def test_inference_dropdown_discovers_llama7b_checkpoint(self):
        try:
            from streamlit.testing.v1 import AppTest
        except ModuleNotFoundError:
            self.skipTest("Streamlit is not installed")

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "ckpt-504-llama7b"
            checkpoint.mkdir()
            (checkpoint / "adapter_config.json").write_text(
                json.dumps(
                    {
                        "base_model_name_or_path": (
                            "meta-llama/Llama-2-7b-chat-hf"
                        )
                    }
                ),
                encoding="utf-8",
            )
            (checkpoint / "adapter_model.safetensors").touch()
            with patch.dict(
                "os.environ",
                {"ATTENTION_MAPS_FINETUNED_MODELS_ROOT": temporary_directory},
            ):
                app = AppTest.from_file(
                    Path(__file__).resolve().parents[1]
                    / "apps/dataset_generator.py"
                ).run(timeout=30)
                model_selector = next(
                    item
                    for item in app.multiselect
                    if item.label == "Inference backends"
                )
                self.assertIn(
                    "Finetuned · Llama 2 7B Chat · Nepali Multi-Dataset QLoRA",
                    model_selector.options,
                )
                model_selector.set_value(
                    ["Finetuned · Llama 2 7B Chat · Nepali Multi-Dataset QLoRA"]
                ).run(timeout=30)
                quantization_selector = next(
                    item
                    for item in app.selectbox
                    if item.label == "PEFT quantization"
                )
                self.assertEqual(quantization_selector.value, "auto")


if __name__ == "__main__":
    unittest.main()
