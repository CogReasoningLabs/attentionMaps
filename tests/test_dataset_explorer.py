import tempfile
import unittest
import json
from pathlib import Path

from apps.dataset_explorer import (
    DatasetSpec,
    custom_dataset,
    discover_datasets,
    find_manifest,
    format_bytes,
    extract_text,
    inspect_dataset,
    lima_translation_dataset,
    parse_model_ids,
    parse_number_list,
    parse_stopwords,
    preview_records,
    sentiment_column,
    sentiment_label,
    sentiment_prediction,
    sample_dataset_rows,
    text_columns,
    split_human_assistant_example,
    unicode_words,
    word_frequencies,
)


class DatasetDiscoveryTests(unittest.TestCase):
    def test_discovers_current_and_future_pipeline_stages(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = (
                root / "raw/pdf/data/train.parquet",
                root / "cleaned/pdf/data/part-00000.parquet",
                root / "processed/corpus/train.parquet",
                root / "tokenized/bpe/train/part-00000.parquet",
                root / "finetuning/domain/train/part-00000.parquet",
            )
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            specs = discover_datasets(root)

            self.assertEqual(
                {spec.key for spec in specs},
                {
                    "raw:pdf",
                    "cleaned:pdf",
                    "processed:corpus:train",
                    "tokenized:bpe:train",
                    "finetuning:domain/train",
                },
            )

    def test_custom_dataset_accepts_file_or_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "part-00000.parquet"
            second = root / "nested/part-00001.parquet"
            second.parent.mkdir()
            first.touch()
            second.touch()

            directory_spec = custom_dataset(root)
            file_spec = custom_dataset(first)

            self.assertEqual(len(directory_spec.files), 2)
            self.assertEqual(file_spec.files, (first.resolve(),))
            self.assertIsNone(custom_dataset(root / "missing"))

    def test_finds_nearest_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_file = root / "cleaned/pdf/data/part-00000.parquet"
            manifest = root / "cleaned/pdf/cleaning_manifest.json"
            data_file.parent.mkdir(parents=True)
            data_file.touch()
            manifest.write_text("{}", encoding="utf-8")
            spec = DatasetSpec("cleaned:pdf", "pdf", "cleaned", (data_file,))

            self.assertEqual(find_manifest(spec, root), manifest)

    def test_loads_lima_translation_json_as_a_dataset(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "lima_translations.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "index": 0,
                            "source_text": "Hello world",
                            "translation": "नमस्ते संसार",
                            "status": "success",
                        },
                        {
                            "index": 1,
                            "source_text": "How are you?",
                            "translation": None,
                            "status": "failed",
                        },
                    ]
                ),
                encoding="utf-8",
            )

            spec = lima_translation_dataset(path)
            self.assertIsNotNone(spec)
            self.assertEqual(spec.label, "LIMA · Gemini/Gemma · English → Nepali")
            self.assertEqual(spec.stage, "translations")
            self.assertEqual(spec.format, "json")
            inventory = inspect_dataset(
                ((str(path), path.stat().st_size, path.stat().st_mtime_ns),)
            )
            self.assertEqual(inventory["rows"], 2)
            self.assertIn("translation", inventory["columns"])

            sampled = sample_dataset_rows(
                inventory, 2, 42, ("source_text", "translation")
            )
            self.assertEqual(len(sampled), 2)
            self.assertTrue(all("source_text" in record for record in sampled))


class DatasetDisplayTests(unittest.TestCase):
    def test_previews_nested_and_long_values(self):
        records = preview_records(
            [{"text": "नेपाल" * 10, "messages": [{"role": "user"}]}],
            max_characters=10,
        )

        self.assertTrue(records[0]["text"].endswith("…"))
        self.assertIsInstance(records[0]["messages"], str)

    def test_formats_binary_sizes(self):
        self.assertEqual(format_bytes(1024), "1.0 KiB")

    def test_preserves_devanagari_combining_marks_in_words(self):
        self.assertEqual(
            unicode_words("नेपालको सुन्दरता र नेपालको"),
            ["नेपालको", "सुन्दरता", "र", "नेपालको"],
        )

    def test_extracts_chat_content_without_roles(self):
        messages = [
            {"role": "user", "content": "नेपालबारे बताऊ"},
            {"role": "assistant", "content": "नेपाल सुन्दर देश हो।"},
        ]
        self.assertEqual(
            extract_text(messages),
            ["नेपालबारे बताऊ", "नेपाल सुन्दर देश हो।"],
        )

    def test_splits_lima_prompt_from_reference_answer(self):
        prompt, reference = split_human_assistant_example(
            "HUMAN: नेपालको राजधानी के हो?\n\nASSISTANT: नेपालको राजधानी काठमाडौं हो।"
        )

        self.assertEqual(prompt, "नेपालको राजधानी के हो?")
        self.assertEqual(reference, "नेपालको राजधानी काठमाडौं हो।")

        plain_prompt, no_reference = split_human_assistant_example("नेपालबारे लेख।")
        self.assertEqual(plain_prompt, "नेपालबारे लेख।")
        self.assertIsNone(no_reference)

    def test_detects_and_normalizes_sentiment_fields(self):
        schema = [
            {"column": "Sentiment", "type": "int32"},
            {"column": "Sentences", "type": "string"},
        ]

        self.assertEqual(sentiment_column(schema), "Sentiment")
        self.assertEqual(sentiment_label(-1), "negative")
        self.assertEqual(sentiment_label(0), "neutral")
        self.assertEqual(sentiment_label(1), "positive")
        self.assertEqual(sentiment_prediction("Prediction: POSITIVE."), "positive")

    def test_word_frequencies_support_stopwords_and_nested_text(self):
        records = [
            {
                "messages": [
                    {"role": "user", "content": "नेपाल र नेपाली"},
                    {"role": "assistant", "content": "नेपाल सुन्दर छ"},
                ]
            }
        ]
        frequencies = word_frequencies(
            records,
            ["messages"],
            stopwords=parse_stopwords("र, छ"),
        )

        self.assertEqual(frequencies["नेपाल"], 2)
        self.assertEqual(frequencies["नेपाली"], 1)
        self.assertNotIn("र", frequencies)

    def test_detects_plain_and_nested_text_columns(self):
        schema = [
            {"column": "doc_id", "type": "string", "nullable": "True"},
            {"column": "text", "type": "string", "nullable": "True"},
            {
                "column": "messages",
                "type": "list<item: struct<role: string, content: string>>",
                "nullable": "True",
            },
        ]

        self.assertEqual(text_columns(schema)[:2], ["text", "messages"])

    def test_parses_inference_grid_and_model_fields(self):
        self.assertEqual(
            parse_number_list("0.2, 0.7 1.0", value_type=float, name="temperature"),
            [0.2, 0.7, 1.0],
        )
        self.assertEqual(
            parse_model_ids("org/model-a\norg/model-b, org/model-c"),
            ["org/model-a", "org/model-b", "org/model-c"],
        )


if __name__ == "__main__":
    unittest.main()
