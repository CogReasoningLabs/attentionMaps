import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from attention_maps.explorer import (
    aya_nepali_dataset_specs,
    available_dataset_purposes,
    configured_finetuned_models_root,
    configured_eda_output_root,
    configured_nepali_stopwords,
    DatasetSpec,
    custom_dataset,
    discover_datasets,
    eda_dataset_spec,
    filter_dataset_specs,
    find_manifest,
    format_bytes,
    himalaya_nepali_sft_dataset,
    himalaya_ai_dataset_specs,
    extract_text,
    find_devanagari_font,
    font_supports_devanagari,
    inspect_huggingface_dataset,
    inspect_dataset,
    iriis_nepali_text_corpus_specs,
    kaggle_dataset_specs,
    lima_original_dataset,
    lima_synthetic_dataset_specs,
    lima_translation_dataset,
    load_eda_cleaning_notes,
    parse_model_ids,
    parse_number_list,
    parse_eda_terms,
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
from attention_maps.generation import synthetic_dataset_families


class DatasetDiscoveryTests(unittest.TestCase):
    def test_registers_lima_as_an_extensible_synthetic_dataset_family(self):
        families = synthetic_dataset_families()

        self.assertEqual([family.key for family in families], ["lima-translation"])
        self.assertEqual(families[0].label, "LIMA translation")
        self.assertEqual(
            [variant.label for variant in families[0].variants],
            [
                "LIMA · Translated Nepali (Gemini/Gemma)",
                "LIMA · Original English",
            ],
        )

    def test_lists_only_purposes_available_for_selected_provider(self):
        specs = [
            *himalaya_ai_dataset_specs(),
            *iriis_nepali_text_corpus_specs(),
        ]

        self.assertEqual(
            available_dataset_purposes(specs, provider="IRIIS Research"),
            ("All purposes", "Pretraining corpus"),
        )
        himalaya_purposes = available_dataset_purposes(
            specs,
            provider="Himalaya AI",
        )
        self.assertIn("Instruction fine-tuning", himalaya_purposes)
        self.assertIn("Evaluation / benchmark", himalaya_purposes)
        self.assertNotIn("Preference tuning", himalaya_purposes)

    def test_catalogs_himalaya_datasets_by_independent_purpose(self):
        specs = himalaya_ai_dataset_specs()

        self.assertEqual(len(specs), 17)
        self.assertTrue(all(spec.provider == "Himalaya AI" for spec in specs))
        purposes = {spec.primary_purpose for spec in specs}
        self.assertIn("Pretraining corpus", purposes)
        self.assertIn("Instruction fine-tuning", purposes)
        self.assertIn("Tokenizer development", purposes)
        self.assertIn("Evaluation / benchmark", purposes)
        instruction_specs = filter_dataset_specs(
            specs,
            provider="Himalaya AI",
            purpose="Instruction fine-tuning",
        )
        self.assertGreaterEqual(len(instruction_specs), 6)
        self.assertTrue(
            all(
                spec.primary_purpose == "Instruction fine-tuning"
                for spec in instruction_specs
            )
        )

    def test_tracks_dataset_provider_source_and_adapter_independently(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "lima.json"
            path.write_text("[]", encoding="utf-8")
            translated = lima_translation_dataset(path)

        self.assertIsNotNone(translated)
        self.assertEqual(translated.provider, "Local project")
        self.assertEqual(translated.source_provider, "GAIR / LIMA")
        self.assertIn("Google (Gemini / Gemma)", translated.adapted_by)
        self.assertEqual(
            filter_dataset_specs(
                [translated],
                provider="Google (Gemini / Gemma)",
            ),
            [translated],
        )

    def test_loads_version_controlled_eda_cleaning_notes(self):
        notes = load_eda_cleaning_notes()

        self.assertIn("## Core EDA logic", notes)
        self.assertIn("## Planned cleaning and deduplication pipeline", notes)
        self.assertIn("SimHash", notes)
        self.assertIn("quarantine", notes)

    def test_uses_repository_nepali_stopword_resource(self):
        stopwords = configured_nepali_stopwords()

        self.assertGreaterEqual(len(stopwords), 190)
        self.assertIn("सम्बन्धी", stopwords)

    def test_allows_eda_output_root_override(self):
        with tempfile.TemporaryDirectory() as temporary_directory, patch.dict(
            "os.environ",
            {"ATTENTION_MAPS_EDA_OUTPUT_ROOT": temporary_directory},
        ):
            self.assertEqual(
                configured_eda_output_root(),
                Path(temporary_directory).resolve(),
            )

    def test_allows_shared_finetuned_models_root_override(self):
        with tempfile.TemporaryDirectory() as temporary_directory, patch.dict(
            "os.environ",
            {"ATTENTION_MAPS_FINETUNED_MODELS_ROOT": temporary_directory},
        ):
            self.assertEqual(
                configured_finetuned_models_root(),
                Path(temporary_directory).resolve(),
            )

    def test_exposes_iriis_nepali_corpus_splits_as_streaming_datasets(self):
        specs = iriis_nepali_text_corpus_specs()

        self.assertEqual([spec.dataset_split for spec in specs], ["train", "test"])
        self.assertTrue(
            all(
                spec.dataset_id == "IRIIS-RESEARCH/Nepali-Text-Corpus" for spec in specs
            )
        )
        self.assertTrue(all(spec.format == "huggingface" for spec in specs))

    def test_exposes_aya_train_as_nepali_only(self):
        specs = aya_nepali_dataset_specs()

        self.assertEqual([spec.dataset_split for spec in specs], ["train"])
        self.assertTrue(
            all(spec.dataset_id == "CohereLabs/aya_dataset" for spec in specs)
        )
        self.assertTrue(all(spec.filter_column == "language_code" for spec in specs))
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
            synthetic_artifacts = lima_synthetic_dataset_specs(path)
            self.assertIsNotNone(original)
            self.assertIsNotNone(translated)
            self.assertEqual(
                [artifact.label for artifact in synthetic_artifacts],
                [
                    "LIMA · Translated Nepali (Gemini/Gemma)",
                    "LIMA · Original English",
                ],
            )
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
            original_inventory = project_inventory(inventory, original.visible_columns)
            translated_inventory = project_inventory(
                inventory, translated.visible_columns
            )
            self.assertEqual(original_inventory["columns"], ["index", "source_text"])
            self.assertEqual(
                translated_inventory["columns"], ["index", "translation", "status"]
            )

            sampled = sample_dataset_rows(translated_inventory, 2, 42, ("translation",))
            self.assertEqual(len(sampled), 2)
            self.assertTrue(all("translation" in record for record in sampled))


class DatasetDisplayTests(unittest.TestCase):
    def test_builds_stable_eda_contract_and_parses_seed_terms(self):
        viewer_spec = DatasetSpec(
            "huggingface:org/corpus:train",
            "Corpus",
            "dataset",
            (),
            format="huggingface",
            dataset_id="org/corpus",
            dataset_split="train",
        )

        first = eda_dataset_spec(viewer_spec, ("text",), ("source",), 500)
        second = eda_dataset_spec(viewer_spec, ("text",), ("source",), 500)

        self.assertEqual(first, second)
        self.assertEqual(first.dataset_id, "org/corpus")
        self.assertEqual(first.text_columns, ("text",))
        self.assertEqual(first.source_columns, ("source",))
        self.assertEqual(first.sample_size, 500)
        self.assertNotIn(":", first.key)
        self.assertEqual(
            parse_eda_terms("मन्त्रालय, लिलाम मन्त्रालय सरकार"),
            ("मन्त्रालय", "लिलाम", "सरकार"),
        )

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
                return ({"language_code": "npi"} for _ in range(4_002))

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
            records = sample_huggingface_rows(inventory, 1, 42, ("inputs", "targets"))

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
            features = {
                "conversations": "list<struct>",
                "source": "string",
                "id": "string",
            }
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

    def test_nepali_word_frequencies_normalize_clean_and_strip_suffixes(self):
        records = [
            {"text": ("गाउँपालिकाको सम्बन्धी मन्त्रालयबाट ASCII 123 ☒ " "गाउँपालिकाको")}
        ]

        frequencies = word_frequencies(
            records,
            ["text"],
            stopwords=parse_stopwords("\ufeff सम्बन्धी  \n"),
            devanagari_only=True,
            strip_nepali_suffixes=True,
        )

        self.assertEqual(frequencies["गाउँपालिका"], 2)
        self.assertEqual(frequencies["मन्त्रालय"], 1)
        self.assertNotIn("सम्बन्धी", frequencies)
        self.assertNotIn("ascii", frequencies)
        self.assertNotIn("123", frequencies)

    def test_default_wordcloud_font_has_devanagari_glyphs(self):
        font = find_devanagari_font()

        self.assertIsNotNone(font)
        self.assertTrue(font_supports_devanagari(font))

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


class DatasetExplorerAppTests(unittest.TestCase):
    def test_runs_eda_from_ui_and_exposes_detailed_results(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            from streamlit.testing.v1 import AppTest
        except ModuleNotFoundError:
            self.skipTest("Streamlit and PyArrow are required")

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_path = root / "fixture.parquet"
            pq.write_table(
                pa.table(
                    {
                        "text": [
                            "नेपाल सरकारको मन्त्रालयले सूचना जारी गर्यो।",
                            "गाउँ कार्यपालिकाको कार्यालय नेपालमा छ।",
                            "लिलाम सम्बन्धी सूचना मन्त्रालयबाट आयो।",
                        ],
                        "source": ["a", "b", "a"],
                    }
                ),
                dataset_path,
            )
            with patch.dict(
                "os.environ",
                {"ATTENTION_MAPS_EDA_OUTPUT_ROOT": str(root / "eda-output")},
            ):
                app = AppTest.from_file(
                    Path(__file__).resolve().parents[1] / "apps/dataset_explorer.py"
                ).run(timeout=30)
                next(
                    item
                    for item in app.checkbox
                    if item.label == "Use custom Parquet path"
                ).set_value(True).run(timeout=30)
                next(
                    item
                    for item in app.text_input
                    if item.label == "Parquet file or directory"
                ).set_value(str(dataset_path)).run(timeout=30)
                next(
                    item
                    for item in app.button
                    if item.label == "Run EDA for this dataset"
                ).click().run(timeout=30)

                result_key = f"eda-result:custom:{dataset_path.resolve()}"
                self.assertIn(result_key, app.session_state)
                self.assertFalse(app.exception)
                self.assertTrue(
                    {
                        "Usable rows",
                        "Median words",
                        "Median sentence",
                        "Duplicates",
                    }.issubset({metric.label for metric in app.metric})
                )
                output_dir = Path(app.session_state[result_key]["output_dir"])
                self.assertTrue(
                    (output_dir / "document_size_distribution.png").is_file()
                )
                self.assertTrue((output_dir / "top_2grams.csv").is_file())

    def test_nlue_benchmark_exposes_collection_tasks_and_local_models(self):
        try:
            from streamlit.testing.v1 import AppTest
        except ModuleNotFoundError:
            self.skipTest("Streamlit is not installed")

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "ckpt-504-llama7b"
            checkpoint.mkdir()
            (checkpoint / "adapter_config.json").write_text(
                json.dumps(
                    {"base_model_name_or_path": ("meta-llama/Llama-2-7b-chat-hf")}
                ),
                encoding="utf-8",
            )
            (checkpoint / "adapter_model.safetensors").touch()
            with patch.dict(
                "os.environ",
                {"ATTENTION_MAPS_FINETUNED_MODELS_ROOT": temporary_directory},
            ):
                app = AppTest.from_file(
                    Path(__file__).resolve().parents[1] / "apps/dataset_explorer.py"
                ).run(timeout=30)

        task_selector = next(
            item for item in app.selectbox if item.label == "Decoder benchmark task"
        )
        self.assertEqual(len(task_selector.options), 16)
        self.assertTrue(any("Belebele" in option for option in task_selector.options))
        self.assertTrue(
            any("Global-MMLU" in option for option in task_selector.options)
        )
        self.assertTrue(any("XL-Sum" in option for option in task_selector.options))
        model_selector = next(
            item
            for item in app.multiselect
            if item.label == "Decoder models to benchmark"
        )
        self.assertTrue(any(tab.label == "Survey EDA" for tab in app.tabs))
        self.assertTrue(any(tab.label == "EDA & cleaning notes" for tab in app.tabs))
        provider_selector = next(
            item for item in app.selectbox if item.label == "Provider / lineage"
        )
        self.assertIn("Himalaya AI", provider_selector.options)
        self.assertIn("Arkios", provider_selector.options)
        self.assertIn("Google (Gemini / Gemma)", provider_selector.options)
        purpose_selector = next(
            item for item in app.selectbox if item.label == "Dataset purpose"
        )
        self.assertIn("Pretraining corpus", purpose_selector.options)
        self.assertIn("Instruction fine-tuning", purpose_selector.options)
        self.assertNotIn("Preference tuning", purpose_selector.options)
        self.assertTrue(
            any(button.label == "Run EDA for this dataset" for button in app.button)
        )
        self.assertTrue(
            any(field.label == "Sampled rows" for field in app.number_input)
        )
        llama_label = "Llama 2 7B Chat · Nepali Multi-Dataset QLoRA"
        self.assertIn(f"Base · {llama_label}", model_selector.options)
        self.assertIn(f"Finetuned · {llama_label}", model_selector.options)
        self.assertIn("Gemma 4 · Google API", model_selector.options)
        self.assertIn("Gemma 4 E2B Base · Hugging Face local", model_selector.options)
        self.assertIn(
            "IRIIS GPT-2 Instruct Nepali 124M · Hugging Face local",
            model_selector.options,
        )
        self.assertIn(
            "IRIIS GPT-2 Nepali 124M Base · Hugging Face local",
            model_selector.options,
        )

        provider_selector.set_value("Arkios").run(timeout=30)
        filtered_purpose_selector = next(
            item for item in app.selectbox if item.label == "Dataset purpose"
        )
        self.assertEqual(filtered_purpose_selector.options, ["All purposes"])

        workspace_selector = next(
            item for item in app.radio if item.label == "Dataset workspace"
        )
        workspace_selector.set_value("Synthetic data generation").run(
            timeout=30
        )
        family_selector = next(
            item
            for item in app.selectbox
            if item.label == "Synthetic dataset family"
        )
        self.assertEqual(family_selector.options, ["LIMA translation"])
        lima_selector = next(
            item for item in app.selectbox if item.label == "Dataset variant"
        )
        self.assertEqual(
            lima_selector.options,
            [
                "LIMA · Translated Nepali (Gemini/Gemma)",
                "LIMA · Original English",
            ],
        )
        self.assertTrue(
            any(
                item.value
                == "2 materialized variant(s) detected for LIMA translation"
                for item in app.success
            )
        )

    def test_evaluation_dropdown_includes_google_gemma_api(self):
        try:
            from streamlit.testing.v1 import AppTest
        except ModuleNotFoundError:
            self.skipTest("Streamlit is not installed")
        from attention_maps.evaluation.flores import FloresExample

        app = AppTest.from_file(
            Path(__file__).resolve().parents[1] / "apps/dataset_explorer.py"
        )
        app.session_state["flores-examples:devtest:0:5"] = [
            FloresExample(0, 0, "Hello", "नमस्ते")
        ]
        app.run(timeout=30)

        model_selector = next(
            item for item in app.multiselect if item.label == "Models to benchmark"
        )
        self.assertIn("Gemma 4 · Google API", model_selector.options)
        self.assertIn("Gemma 4 E2B Base · Hugging Face local", model_selector.options)
        iriis_model_options = (
            "IRIIS GPT-2 Instruct Nepali 124M · Hugging Face local",
            "IRIIS GPT-2 Nepali 124M Base · Hugging Face local",
        )
        for option in iriis_model_options:
            self.assertIn(option, model_selector.options)
        comparison_selector = next(
            item for item in app.multiselect if item.label == "Inference backends"
        )
        self.assertIn("Gemma 4 · Google API", comparison_selector.options)
        self.assertIn(
            "Gemma 4 E2B Base · Hugging Face local",
            comparison_selector.options,
        )
        for option in iriis_model_options:
            self.assertIn(option, comparison_selector.options)
        model_selector.set_value(
            [
                "Teacher · gemini-3.5-flash-lite",
                "Other · Arkios 1B Chat",
                "Gemma 4 · Google API",
            ]
        ).run(timeout=30)
        model_selector = next(
            item for item in app.multiselect if item.label == "Models to benchmark"
        )
        self.assertIn("Gemma 4 · Google API", model_selector.value)
        gemma_model_field = next(
            item
            for item in app.text_input
            if item.label == "Evaluation Google Gemma model"
        )
        self.assertEqual(gemma_model_field.value, "gemma-4-26b-a4b-it")

    def test_llama_checkpoint_appears_in_all_inference_selectors(self):
        try:
            from streamlit.testing.v1 import AppTest
        except ModuleNotFoundError:
            self.skipTest("Streamlit is not installed")
        from attention_maps.evaluation.flores import FloresExample

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "ckpt-504-llama7b"
            checkpoint.mkdir()
            (checkpoint / "adapter_config.json").write_text(
                json.dumps(
                    {"base_model_name_or_path": ("meta-llama/Llama-2-7b-chat-hf")}
                ),
                encoding="utf-8",
            )
            (checkpoint / "adapter_model.safetensors").touch()
            with patch.dict(
                "os.environ",
                {"ATTENTION_MAPS_FINETUNED_MODELS_ROOT": temporary_directory},
            ):
                app = AppTest.from_file(
                    Path(__file__).resolve().parents[1] / "apps/dataset_explorer.py"
                )
                app.session_state["flores-examples:devtest:0:5"] = [
                    FloresExample(0, 0, "Hello", "नमस्ते")
                ]
                app.run(timeout=30)

            label = "Llama 2 7B Chat · Nepali Multi-Dataset QLoRA"
            local_selector = next(
                item for item in app.selectbox if item.label == "Local finetuned model"
            )
            self.assertIn(label, local_selector.options)
            quantization_selector = next(
                item for item in app.selectbox if item.label == "Quantization"
            )
            self.assertEqual(quantization_selector.value, "auto")
            self.assertEqual(
                quantization_selector.options,
                ["auto", "4bit", "8bit", "none"],
            )
            comparison_selector = next(
                item for item in app.multiselect if item.label == "Inference backends"
            )
            self.assertIn(f"Local · {label}", comparison_selector.options)
            evaluation_selector = next(
                item for item in app.multiselect if item.label == "Models to benchmark"
            )
            self.assertIn(f"Base · {label}", evaluation_selector.options)
            self.assertIn(f"Finetuned · {label}", evaluation_selector.options)


if __name__ == "__main__":
    unittest.main()
