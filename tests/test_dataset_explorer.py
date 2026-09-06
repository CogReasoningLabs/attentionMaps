import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from apps.dataset_explorer import (
    aya_nepali_dataset_specs,
    DatasetSpec,
    custom_dataset,
    discover_datasets,
    find_manifest,
    format_bytes,
    himalaya_nepali_sft_dataset,
    extract_text,
    inspect_huggingface_dataset,
    inspect_dataset,
    iriis_nepali_text_corpus_specs,
    kaggle_dataset_specs,
    lima_original_dataset,
    lima_translation_dataset,
    parse_model_ids,
    parse_number_list,
    parse_stopwords,
    preview_records,
    project_inventory,
    sentiment_column,
    sentiment_label,
    sentiment_prediction,
    sample_dataset_rows,
    sample_huggingface_rows,
    text_columns,
    split_human_assistant_example,
    unicode_words,
    word_frequencies,
)


class DatasetDiscoveryTests(unittest.TestCase):
    def test_exposes_iriis_nepali_corpus_splits_as_streaming_datasets(self):
        specs = iriis_nepali_text_corpus_specs()

        self.assertEqual([spec.dataset_split for spec in specs], ["train", "test"])
        self.assertTrue(
            all(
                spec.dataset_id == "IRIIS-RESEARCH/Nepali-Text-Corpus"
                for spec in specs
            )
        )
        self.assertTrue(all(spec.format == "huggingface" for spec in specs))

    def test_exposes_aya_train_as_nepali_only(self):
        specs = aya_nepali_dataset_specs()

        self.assertEqual([spec.dataset_split for spec in specs], ["train"])
        self.assertTrue(
            all(spec.dataset_id == "CohereLabs/aya_dataset" for spec in specs)
        )
        self.assertTrue(
            all(spec.filter_column == "language_code" for spec in specs)
        )
        self.assertTrue(all(spec.filter_value == "npi" for spec in specs))
        self.assertTrue(
            all("language_code=npi" in str(spec.location) for spec in specs)
        )

    def test_exposes_kaggle_classification_workbooks(self):
        specs = kaggle_dataset_specs()
        workbook_specs = [spec for spec in specs if spec.format == "kaggle"]

        self.assertEqual(len(workbook_specs), 3)
        self.assertEqual(
            {spec.dataset_file for spec in workbook_specs},
            {
                "nepalimoviereviews.csv.xlsx",
                "Nepali hate speech.xlsx",
                "brb.xlsx",
            },
        )
        self.assertTrue(all(spec.format == "kaggle" for spec in workbook_specs))
        self.assertTrue(
            all(str(spec.location).startswith("kaggle://") for spec in specs)
        )

    def test_exposes_deduplicated_oscar_with_download_confirmation_metadata(self):
        spec = next(
            spec
            for spec in kaggle_dataset_specs()
            if spec.dataset_id == "hsebarp/oscar-corpus-nepali"
        )

        self.assertEqual(spec.dataset_file, "ne_dedup.txt")
        self.assertEqual(spec.format, "kaggle_text")
        self.assertGreater(spec.download_bytes, 1_000_000_000)

    def test_exposes_himalaya_sft_as_remote_dataset(self):
        spec = himalaya_nepali_sft_dataset()

        self.assertEqual(spec.label, "Himalaya AI · Nepali SFT Dataset")
        self.assertEqual(spec.format, "huggingface")
        self.assertEqual(spec.dataset_split, "train")
        self.assertIn("himalaya-ai/nepali-sft-dataset", str(spec.location))

    def test_discovers_only_supported_pipeline_stages(self):
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

    def test_exposes_separate_original_and_translated_lima_views(self):
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

            original = lima_original_dataset(path)
            translated = lima_translation_dataset(path)
            self.assertIsNotNone(original)
            self.assertIsNotNone(translated)
            self.assertEqual(original.label, "LIMA · Original English")
            self.assertEqual(
                translated.label, "LIMA · Translated Nepali (Gemini/Gemma)"
            )
            self.assertEqual(original.stage, "dataset")
            self.assertEqual(translated.stage, "dataset")
            inventory = inspect_dataset(
                ((str(path), path.stat().st_size, path.stat().st_mtime_ns),)
            )
            self.assertEqual(inventory["rows"], 2)
            original_inventory = project_inventory(
                inventory, original.visible_columns
            )
            translated_inventory = project_inventory(
                inventory, translated.visible_columns
            )
            self.assertEqual(
                original_inventory["columns"], ["index", "source_text"]
            )
            self.assertEqual(
                translated_inventory["columns"], ["index", "translation", "status"]
            )

            sampled = sample_dataset_rows(
                translated_inventory, 2, 42, ("translation",)
            )
            self.assertEqual(len(sampled), 2)
            self.assertTrue(all("translation" in record for record in sampled))


class DatasetDisplayTests(unittest.TestCase):
    def test_inspects_and_samples_only_filtered_huggingface_rows(self):
        class FakeStream:
            num_shards = 46
            features = {
                "inputs": "string",
                "targets": "string",
                "language_code": "string",
            }
            info = SimpleNamespace(
                splits={
                    "train": SimpleNamespace(
                        num_examples=202_362,
                        num_bytes=254_591_851,
                        shard_lengths=[],
                    )
                }
            )

            def __iter__(self):
                return (
                    {"language_code": "npi"}
                    for _ in range(4_002)
                )

        with patch("datasets.load_dataset", return_value=FakeStream()) as loader:
            inventory = inspect_huggingface_dataset(
                "CohereLabs/aya_dataset",
                "train",
                config="default",
                filter_column="language_code",
                filter_value="npi",
            )
        self.assertEqual(
            loader.call_args.kwargs["filters"],
            [("language_code", "==", "npi")],
        )

        class FilteredStream:
            def shuffle(self, **kwargs):
                self.shuffle_arguments = kwargs
                return self

            def take(self, count):
                return [
                    {
                        "inputs": "नृत्य प्रतियोगिताबारे चर्चा गर्नुहोस्।",
                        "targets": "नृत्यले सीप विकास गर्छ।",
                        "language_code": "npi",
                    }
                ][:count]

        filtered_stream = FilteredStream()
        with patch(
            "datasets.load_dataset", return_value=filtered_stream
        ) as filtered_loader:
            records = sample_huggingface_rows(
                inventory, 1, 42, ("inputs", "targets")
            )

        self.assertEqual(inventory["rows"], 4_002)
        self.assertEqual(inventory["files"], 46)
        self.assertEqual(inventory["source_rows"], 202_362)
        self.assertTrue(inventory["bytes_estimated"])
        self.assertEqual(records[0]["__viewer_row_index"], 0)
        self.assertNotIn("language_code", records[0])
        self.assertEqual(
            filtered_loader.call_args.kwargs["filters"],
            [("language_code", "==", "npi")],
        )
        self.assertEqual(filtered_stream.shuffle_arguments["buffer_size"], 200)

    def test_inspects_and_samples_huggingface_dataset_with_bounded_streaming(self):
        class FakeStream:
            features = {"conversations": "list<struct>", "source": "string", "id": "string"}
            info = SimpleNamespace(
                splits={
                    "train": SimpleNamespace(
                        num_examples=1_112_863,
                        num_bytes=3_662_367_076,
                        shard_lengths=[2, 2],
                    )
                }
            )

            def shuffle(self, **kwargs):
                self.shuffle_arguments = kwargs
                return self

            def take(self, count):
                rows = [
                    {
                        "conversations": [
                            {"from": "human", "value": "प्रश्न"},
                            {"from": "gpt", "value": "उत्तर"},
                        ],
                        "source": "test",
                        "id": "row-1",
                    }
                ]
                return rows[:count]

        stream = FakeStream()
        with patch("datasets.load_dataset", return_value=stream):
            inventory = inspect_huggingface_dataset(
                "himalaya-ai/nepali-sft-dataset", "train"
            )
            records = sample_huggingface_rows(
                inventory, 1, 42, ("conversations",), shuffle_buffer=50
            )

        self.assertEqual(inventory["rows"], 1_112_863)
        self.assertEqual(inventory["files"], 2)
        self.assertEqual(records[0]["conversations"][0]["value"], "प्रश्न")
        self.assertEqual(stream.shuffle_arguments["buffer_size"], 50)

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
        sharegpt_messages = [
            {"from": "human", "value": "प्रश्न"},
            {"from": "gpt", "value": "उत्तर"},
        ]
        self.assertEqual(extract_text(sharegpt_messages), ["प्रश्न", "उत्तर"])

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
