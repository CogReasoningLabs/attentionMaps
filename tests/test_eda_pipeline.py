import json
import tempfile
import unittest
from pathlib import Path

from attention_maps.eda.cli import load_plan
from attention_maps.eda.contracts import AnalysisConfig, DatasetSpec, SurveyPlan
from attention_maps.eda.pipeline import analyze_records, run_survey, select_datasets
from attention_maps.eda.reporting import write_survey_report
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
            near_duplicate_distance=4,
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
