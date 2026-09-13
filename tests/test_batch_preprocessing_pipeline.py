import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from attention_maps.batch_pipeline.contracts import (
    DeduplicationConfig,
    EdaConfig,
    ExecutionConfig,
    InputConfig,
    PipelineConfig,
)
from attention_maps.batch_pipeline.processing import selected_by_sampling
from attention_maps.batch_pipeline.processing import iter_parquet_records
from attention_maps.batch_pipeline.runner import run_pipeline


class BatchPreprocessingPipelineTests(unittest.TestCase):
    def test_local_pipeline_batches_deduplicates_reports_and_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            rows = (
                {"text": "नेपालमा आज राम्रो मौसम छ।", "source": "news"},
                {"text": "नेपालमा आज राम्रो मौसम छ।", "source": "copy"},
                {"text": "काठमाडौँ नेपालको राजधानी हो।", "source": "reference"},
            )
            source.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            config = PipelineConfig(
                run_name="test-run",
                input=InputConfig(
                    local_path=source,
                    text_columns=("text",),
                    source_columns=("source",),
                ),
                output_root=root / "runs",
                deduplication=DeduplicationConfig(mode="exact"),
                execution=ExecutionConfig(
                    batch_size=2,
                    workers=2,
                    max_pending_batches=2,
                    shard_rows=1,
                ),
                eda=EdaConfig(
                    enabled=True,
                    top_items=10,
                    max_vocabulary=100,
                    max_ngrams=100,
                    reservoir_size=10,
                    max_pattern_tokens_per_document=50,
                ),
            )

            result = run_pipeline(config)

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            audit = (
                result.run_dir / "output/audit/preprocessing_audit.jsonl"
            ).read_text(encoding="utf-8")
            with zipfile.ZipFile(result.package_path) as bundle:
                members = set(bundle.namelist())

        self.assertEqual(result.clean_documents, 2)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["counts"]["exact_duplicates"], 1)
        self.assertIn('"reason": "exact_duplicates"', audit)
        self.assertIn("report/eda_report.pdf", members)
        self.assertIn("run_manifest.json", members)
        self.assertIn("checksums.sha256", members)
        self.assertTrue(any(name.endswith("top_1grams.csv") for name in members))
        self.assertTrue(any(name.endswith("top_2grams.csv") for name in members))
        self.assertTrue(any(name.endswith("wordcloud.png") for name in members))
        self.assertTrue(any(name.startswith("clean-data/part-") for name in members))

    def test_sampling_decision_is_stable(self):
        from attention_maps.batch_pipeline.contracts import SamplingConfig

        config = SamplingConfig(fraction=0.25, seed=9)
        first = [selected_by_sampling(f"row-{index}", config) for index in range(100)]
        second = [selected_by_sampling(f"row-{index}", config) for index in range(100)]
        self.assertEqual(first, second)
        self.assertGreater(sum(first), 0)
        self.assertLess(sum(first), 100)

    def test_multistage_mode_strips_repeated_boilerplate(self):
        repeated = "यो सबै दस्तावेजमा दोहोरिने लामो साझा परिचयात्मक अनुच्छेद हो।"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            rows = [
                {"text": f"{repeated}\nपहिलो दस्तावेजको बिलकुल फरक मूल सामग्री।"},
                {"text": f"{repeated}\nदोस्रो दस्तावेजको अर्को बिलकुल फरक सामग्री।"},
            ]
            source.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            config = PipelineConfig(
                run_name="boilerplate-run",
                input=InputConfig(local_path=source, text_columns=("text",)),
                output_root=root / "runs",
                deduplication=DeduplicationConfig(
                    mode="multistage",
                    near_duplicate_threshold=1.0,
                    minhash_permutations=16,
                    minhash_bands=4,
                    boilerplate_min_documents=2,
                    boilerplate_min_characters=20,
                ),
                execution=ExecutionConfig(batch_size=2, workers=1, shard_rows=2),
                eda=EdaConfig(enabled=False),
            )

            result = run_pipeline(config)
            clean = list(
                iter_parquet_records(
                    sorted((result.run_dir / "output/clean-data").glob("*.parquet")),
                    10,
                )
            )
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(len(clean), 2)
        self.assertTrue(all(repeated not in record["text"] for record in clean))
        self.assertEqual(manifest["deduplication"]["repeated_paragraph_patterns"], 1)
        self.assertEqual(
            manifest["counts"]["boilerplate_paragraphs_removed"], 2
        )


if __name__ == "__main__":
    unittest.main()
