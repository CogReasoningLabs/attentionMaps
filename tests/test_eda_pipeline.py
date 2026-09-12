import json
import tempfile
import unittest
from pathlib import Path

from attention_maps.eda.cli import load_plan
from attention_maps.eda.contracts import AnalysisConfig, DatasetSpec, SurveyPlan
from attention_maps.eda.pipeline import analyze_records, run_survey, select_datasets
from attention_maps.eda.reporting import (
    _eligible_wordcloud_frequencies,
    write_survey_report,
)
from attention_maps.eda.deduplication import (
    DeduplicationConfig,
    MultiStageDeduplicator,
    normalize_for_deduplication,
)
from attention_maps.eda.sampling import fold_seed, plan_repeated_sampling
from attention_maps.eda.workspace import (
    deduplicate_workspace_documents,
    normalize_workspace_sample,
    persist_workspace_artifacts,
    workspace_audit_csv,
    workspace_documents_jsonl,
)
from attention_maps.eda.script import (
    DEVANAGARI,
    LATIN,
    MIXED_DEVANAGARI_ROMANIZED,
    MIXED_NEPALI_ENGLISH,
    OTHER,
    ROMANIZED,
    SCRIPT_CATEGORIES,
    identify_script_category,
)
from attention_maps.eda.text import (
    extract_source,
    extract_text,
    line_character_lengths,
    load_stopwords_file,
    normalize_text,
    sentence_token_lengths,
    tokenize,
)


class EDASchemaTests(unittest.TestCase):
    def test_workspace_normalizes_then_materializes_deduplicated_clean_data(self):
        common = "यो वेबसाइटमा दोहोरिने साझा सूचना अनुच्छेद पर्याप्त लामो छ।"
        records = [
            {
                "text": f"{common}\nनेपालको पहिलो समाचार सामग्री।",
                "source": "a",
                "__viewer_row_index": 1,
            },
            {
                "text": f"{common}\nअर्को फरक समाचार सामग्री।",
                "source": "b",
                "__viewer_row_index": 2,
            },
            {
                "text": f"  {common}\nनेपालको पहिलो समाचार सामग्री।  ",
                "source": "a",
                "__viewer_row_index": 3,
            },
            {"missing": "text"},
        ]
        normalized = normalize_workspace_sample(records, ("text",), ("source",))

        self.assertEqual(normalized.normalization, "NFC")
        self.assertEqual(normalized.normalized_rows, 3)
        self.assertEqual(normalized.missing_text_rows, 1)
        result = deduplicate_workspace_documents(
            normalized.documents,
            DeduplicationConfig(
                near_duplicate_threshold=1.0,
                boilerplate_min_documents=2,
                boilerplate_min_characters=20,
            ),
        )
        self.assertEqual(result.exact_documents_removed, 1)
        self.assertEqual(result.repeated_paragraph_patterns, 1)
        self.assertEqual(result.paragraphs_removed, 2)
        self.assertEqual(result.retained_documents, 2)
        self.assertNotIn(common, workspace_documents_jsonl(result.documents).decode())
        self.assertIn("exact_document", workspace_audit_csv(result).decode())
        with tempfile.TemporaryDirectory() as directory:
            paths = persist_workspace_artifacts(
                result,
                Path(directory) / "run",
                metadata={"dataset_key": "fixture"},
            )
            manifest = json.loads(
                (Path(directory) / "run" / "workspace_manifest.json").read_text()
            )
        self.assertEqual(len(paths), 3)
        self.assertFalse(manifest["source_dataset_modified"])
        self.assertEqual(manifest["deduplication_result"]["retained_documents"], 2)

    def test_plans_percentage_based_repeated_samples_with_visible_cap(self):
        plan = plan_repeated_sampling(
            1_000_000,
            10,
            folds=5,
            max_rows_per_fold=50_000,
        )

        self.assertEqual(plan.requested_rows_per_fold, 100_000)
        self.assertEqual(plan.rows_per_fold, 50_000)
        self.assertEqual(plan.effective_percentage_per_fold, 5.0)
        self.assertEqual(plan.total_rows_read, 250_000)
        self.assertTrue(plan.capped)
        self.assertGreater(plan.expected_population_coverage_pct, 20)
        self.assertNotEqual(fold_seed(42, 1), fold_seed(42, 2))

    def test_plans_disjoint_folds_without_repeated_population_reads(self):
        plan = plan_repeated_sampling(
            1_666,
            20,
            folds=5,
            disjoint_folds=True,
        )

        self.assertEqual(plan.rows_per_fold, 334)
        self.assertEqual(plan.total_rows_read, 1_666)
        self.assertEqual(plan.expected_unique_rows, 1_666)
        self.assertEqual(plan.expected_population_coverage_pct, 100.0)
        self.assertTrue(plan.full_population)
        self.assertTrue(plan.disjoint_folds)

    def test_normalizes_unicode_whitespace_and_case_before_sha256(self):
        first = normalize_for_deduplication("  CAFÉ\n नेपाल  ")
        second = normalize_for_deduplication("cafe\u0301 नेपाल")

        self.assertEqual(first, second)

    def test_runs_exact_near_and_repeated_paragraph_stages_sequentially(self):
        config = DeduplicationConfig(
            minhash_permutations=64,
            minhash_bands=16,
            near_duplicate_threshold=0.70,
            boilerplate_min_documents=2,
            boilerplate_min_characters=10,
        )
        deduplicator = MultiStageDeduplicator(config)
        boilerplate = "यो वेबसाइटको साझा लामो सूचना अनुच्छेद हो।"
        original = f"{boilerplate}\nनेपालको विस्तृत समाचार सामग्री यहाँ छ।"
        exact = "  " + original.replace("नेपालको", "नेपालको") + "  "
        near = original + " थप"
        distinct = f"{boilerplate}\nअर्को पूर्णतः फरक दस्तावेज यहाँ छ।"

        self.assertFalse(deduplicator.observe(original).duplicate)
        self.assertTrue(deduplicator.observe(exact).exact_duplicate)
        self.assertTrue(deduplicator.observe(near).near_duplicate)
        self.assertFalse(deduplicator.observe(distinct).duplicate)
        boilerplate_result = deduplicator.boilerplate_summary()
        self.assertEqual(boilerplate_result.unique_repeated_paragraphs, 1)
        self.assertEqual(boilerplate_result.affected_documents, 2)

    def test_identifies_all_supported_script_categories(self):
        cases = {
            DEVANAGARI: "नेपाल सुन्दर देश हो",
            MIXED_DEVANAGARI_ROMANIZED: "नेपाल ramro chha mero desh",
            ROMANIZED: "mero naam ram ho ani yo ramro chha",
            LATIN: "This is a plain English dataset",
            OTHER: "中文数据集資料語言文本範例 with",
            MIXED_NEPALI_ENGLISH: "नेपाल is a beautiful country",
        }

        self.assertEqual(set(cases), set(SCRIPT_CATEGORIES))
        for expected, text in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(identify_script_category(text), expected)

    def test_aggregates_script_evidence_across_dataset_rows(self):
        self.assertEqual(
            identify_script_category(["नेपाल राम्रो छ", "mero desh ramro chha"]),
            MIXED_DEVANAGARI_ROMANIZED,
        )

    def test_loads_bom_safe_comma_or_line_separated_stopwords(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "stopwords.txt"
            path.write_text("\ufeff सम्बन्धी, को\nबाट  सम्बन्धी ", encoding="utf-8")

            self.assertEqual(
                load_stopwords_file(path),
                ("सम्बन्धी", "को", "बाट"),
            )

    def test_extracts_plain_nested_chat_and_instruction_text(self):
        self.assertEqual(extract_text({"text": " नेपाल  राम्रो "}), "नेपाल राम्रो")
        self.assertEqual(
            extract_text(
                {
                    "messages": [
                        {"role": "user", "content": "प्रश्न"},
                        {"role": "assistant", "content": "उत्तर"},
                    ]
                }
            ),
            "प्रश्न\nउत्तर",
        )
        self.assertEqual(
            extract_text({"instruction": "लेख", "input": "नेपाल", "output": "उत्तर"}),
            "लेख\nनेपाल\nउत्तर",
        )
        self.assertEqual(
            extract_text({"nested": {"body": "सामग्री"}}, ("nested.body",)),
            "सामग्री",
        )

    def test_extracts_domain_or_nested_source(self):
        self.assertEqual(
            extract_source({"url": "https://www.example.com/news/1"}),
            "example.com",
        )
        self.assertEqual(
            extract_source({"metadata": {"source": "पत्रिका"}}),
            "पत्रिका",
        )

    def test_preserves_and_measures_sentence_and_line_structure(self):
        text = extract_text({"text": "नेपाल राम्रो छ। अर्को वाक्य!\nतेस्रो लाइन"})

        self.assertEqual(text.count("\n"), 1)
        self.assertEqual(sentence_token_lengths(text), [3, 2, 2])
        self.assertEqual(
            line_character_lengths(text),
            [len("नेपाल राम्रो छ। अर्को वाक्य!"), len("तेस्रो लाइन")],
        )
        self.assertEqual(
            tokenize("मन्त्रालयले सूचना दियो। १२ पटक"),
            ["मन्त्रालयले", "सूचना", "दियो", "१२", "पटक"],
        )
        self.assertEqual(normalize_text("Cafe\u0301\ufeff  पाठ"), "Café पाठ")


class EDAAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.spec = DatasetSpec("fixture", "example/fixture")
        self.config = AnalysisConfig(
            sample_size=10,
            min_tokens=2,
            reservoir_size=3,
            max_vocabulary=100,
            top_tokens=5,
            cooccurrence_terms=("नेपाल",),
            cooccurrence_stopwords=("हो",),
        )

    def test_computes_quality_provenance_and_duplicates(self):
        nepali_document = " ".join(["नेपाल सुन्दर देश हो"] * 10)
        records = [
            {"text": nepali_document, "source": "a"},
            {"text": nepali_document, "source": "a"},
            {"text": f"{nepali_document} सुन्दर देश", "source": "b"},
            {"text": "english only words here", "source": "b"},
            {"other": "missing"},
        ]

        profile = analyze_records(self.spec, records, self.config)

        self.assertEqual(profile.summary.rows_seen, 5)
        self.assertEqual(profile.summary.usable_rows, 4)
        self.assertEqual(profile.summary.missing_text_rows, 1)
        self.assertEqual(profile.summary.exact_duplicate_rows, 1)
        self.assertGreaterEqual(profile.summary.near_duplicate_rows, 1)
        self.assertEqual(profile.summary.quality_pass_rows, 3)
        self.assertEqual(profile.summary.script_category, MIXED_NEPALI_ENGLISH)
        self.assertGreater(profile.summary.devanagari_letter_share_pct, 0)
        self.assertGreater(profile.summary.latin_letter_share_pct, 0)
        self.assertEqual(profile.summary.source_categories_observed, 2)
        self.assertLessEqual(len(profile.samples), 3)
        self.assertTrue(profile.top_tokens)
        self.assertTrue(profile.top_ngrams[2])
        self.assertTrue(profile.top_ngrams[3])
        self.assertTrue(profile.top_ngrams[4])
        self.assertTrue(profile.sentence_length_samples)
        self.assertTrue(profile.line_length_samples)
        self.assertTrue(profile.cooccurrence_edges)

    def test_sample_limit_and_progress_are_deterministic(self):
        events = []
        config = AnalysisConfig(
            sample_size=2,
            reservoir_size=1,
            max_vocabulary=20,
            top_tokens=2,
        )
        records = ({"text": f"नेपाल पाठ {index}"} for index in range(10))

        first = analyze_records(
            self.spec, records, config, progress=lambda *x: events.append(x)
        )
        second = analyze_records(
            self.spec,
            ({"text": f"नेपाल पाठ {index}"} for index in range(10)),
            config,
        )

        self.assertEqual(first.summary.rows_seen, 2)
        self.assertEqual(first.samples, second.samples)
        self.assertIn("bounded sample", " ".join(first.summary.warnings))

        full_profile = analyze_records(
            DatasetSpec(
                "full-fixture",
                "example/fixture",
                sample_size=2,
                population_rows=2,
            ),
            ({"text": f"नेपाल पाठ {index}"} for index in range(2)),
            config,
        )
        self.assertNotIn("bounded sample", " ".join(full_profile.summary.warnings))

    def test_classifies_other_scripts_even_when_regex_tokenizer_cannot_use_them(self):
        profile = analyze_records(
            self.spec,
            [{"text": "中文数据集資料"}],
            self.config,
        )

        self.assertEqual(profile.summary.usable_rows, 0)
        self.assertEqual(profile.summary.script_category, OTHER)
        self.assertEqual(profile.summary.other_letter_share_pct, 100.0)

    def test_cooccurrence_normalizes_attached_nepali_postpositions(self):
        profile = analyze_records(
            self.spec,
            [{"text": "नेपालको मन्त्रालयबाट सरकारको कार्यालयमा सूचना आयो।"}],
            AnalysisConfig(
                sample_size=1,
                reservoir_size=10,
                max_vocabulary=100,
                cooccurrence_terms=("नेपाल", "मन्त्रालय", "सरकार", "कार्यालय"),
                cooccurrence_stopwords=("सूचना",),
            ),
        )

        edges = {(source, target) for source, target, _ in profile.cooccurrence_edges}
        self.assertIn(("नेपाल", "मन्त्रालय"), edges)
        self.assertIn(("मन्त्रालय", "सरकार"), edges)
        self.assertNotIn(("कार्यालय", "सूचना"), edges)

    def test_wordcloud_never_restores_filtered_stopwords(self):
        frequencies = _eligible_wordcloud_frequencies(
            (
                ("सबै", 100),
                ("सबैलाई", 90),
                ("गर्दा", 80),
                ("नेपाल", 20),
            ),
            ("सबै", "गर्दा"),
        )

        self.assertEqual(frequencies, {"नेपाल": 20})
        self.assertEqual(
            _eligible_wordcloud_frequencies((("सबै", 100),), ("सबै",)),
            {},
        )

    def test_survey_isolates_dataset_failure(self):
        plan = SurveyPlan(
            "fixture survey",
            (
                DatasetSpec("good", "example/good"),
                DatasetSpec("bad", "example/bad"),
            ),
            self.config,
        )

        def source(spec, config):
            if spec.key == "bad":
                raise RuntimeError("unavailable")
            return [{"text": "नेपाल राम्रो देश हो"}]

        run = run_survey(plan, record_source=source)

        self.assertEqual(len(run.profiles), 1)
        self.assertIn("bad", run.failures)
        self.assertIn("RuntimeError", run.failures["bad"])

    def test_writes_derived_artifacts_without_raw_text(self):
        plan = SurveyPlan("fixture survey", (self.spec,), self.config)
        run = run_survey(
            plan,
            record_source=lambda spec, config: [
                {"text": "गोप्य कच्चा पाठ यहाँ छ", "source": "fixture"}
            ],
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            written = write_survey_report(run, output, plots=True)
            manifest = json.loads(
                (output / "survey_manifest.json").read_text(encoding="utf-8")
            )
            combined = "".join(
                path.read_text(encoding="utf-8")
                for path in written
                if path.suffix in {".json", ".csv"}
            )
            artifact_names = {path.name for path in written}

        self.assertFalse(manifest["raw_text_persisted"])
        self.assertIn("git_commit", manifest["runtime"])
        self.assertIn("datasets", manifest["runtime"]["packages"])
        self.assertNotIn("गोप्य कच्चा पाठ यहाँ छ", combined)
        self.assertTrue(any(path.suffix == ".png" for path in written))
        self.assertTrue(
            {
                "document_size_distribution.png",
                "segment_length_kde.png",
                "top_ngrams.png",
                "term_cooccurrence_network.png",
                "deduplication_stages.png",
                "deduplication_stages.csv",
                "top_2grams.csv",
                "top_3grams.csv",
                "top_4grams.csv",
                "cooccurrence_edges.csv",
            }.issubset(artifact_names)
        )


class EDAConfigurationTests(unittest.TestCase):
    def test_loads_default_plan_and_selects_subset(self):
        root = Path(__file__).resolve().parents[1]
        plan = load_plan(root / "configs/eda/nepali_corpus_survey.json")

        self.assertEqual(len(plan.datasets), 17)
        self.assertEqual(plan.analysis.ngram_orders, (2, 3, 4))
        self.assertIn("मन्त्रालय", plan.analysis.cooccurrence_terms)
        self.assertGreaterEqual(len(plan.analysis.cooccurrence_stopwords), 190)
        self.assertIn("सम्बन्धी", plan.analysis.cooccurrence_stopwords)
        selected = select_datasets(plan, {"cc100-nepali"})
        self.assertEqual(
            [dataset.key for dataset in selected.datasets], ["cc100-nepali"]
        )

    def test_rejects_duplicate_dataset_keys(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            SurveyPlan(
                "invalid",
                (
                    DatasetSpec("same", "example/one"),
                    DatasetSpec("same", "example/two"),
                ),
            )


if __name__ == "__main__":
    unittest.main()
