from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_pretraining_dataset import Candidate, build_dataset, parse_args
from scripts.utils.nepali_text import CleaningConfig
from scripts.clean_pretraining_sources import clean_candidate


class PretrainingDataStageTests(unittest.TestCase):
    def test_strict_cleaning_removes_markup_latin_and_web_debris(self) -> None:
        candidate = Candidate(
            source="nepali_news",
            source_id="story-1",
            text=(
                "<article>नेपाल एउटा सुन्दर देश हो। AI test@example.com</article>"
                "<script>tracking payload</script>"
            ),
        )
        row, reason = clean_candidate(
            candidate,
            CleaningConfig(
                min_devanagari_ratio=0.10,
                min_devanagari_letters=1,
                min_characters=1,
                mode="strict",
            ),
        )
        self.assertIsNone(reason)
        self.assertIsNotNone(row)
        text = row["text"]
        self.assertIn("नेपाल एउटा सुन्दर देश हो।", text)
        self.assertNotIn("article", text)
        self.assertNotIn("AI", text)
        self.assertNotIn("tracking", text)
        self.assertNotIn("example.com", text)
        metadata = json.loads(row["metadata_json"])
        self.assertEqual(
            metadata["cleaning"]["input_script_category"],
            "Mixed(Nepali+English)",
        )
        self.assertEqual(metadata["cleaning"]["output_script_category"], "Devanagari")

    def test_processed_builder_consumes_canonical_cleaned_sources(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        schema = pa.schema(
            [
                ("source", pa.string()),
                ("source_id", pa.string()),
                ("text", pa.string()),
                ("language", pa.string()),
                ("url", pa.string()),
                ("raw_text_sha256", pa.string()),
                ("text_sha256", pa.string()),
                ("metadata_json", pa.string()),
            ]
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_specs = {
                "nepali_pdf": ("pdf", "नेपालको एउटा लामो दस्तावेज सामग्री हो।"),
                # Exact duplicate of the PDF text verifies cross-source deduplication.
                "nepali_news": ("news", "नेपालको एउटा लामो दस्तावेज सामग्री हो।"),
                "nepali_lyrics": ("lyrics", "नेपाली गीतको पहिलो हरफ।"),
            }
            input_dirs = {}
            for source, (source_id, text) in source_specs.items():
                data_dir = root / source / "data"
                data_dir.mkdir(parents=True)
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                row = {
                    "source": source,
                    "source_id": source_id,
                    "text": text,
                    "language": "ne",
                    "url": None,
                    "raw_text_sha256": digest,
                    "text_sha256": digest,
                    "metadata_json": json.dumps({}, ensure_ascii=False),
                }
                pq.write_table(
                    pa.Table.from_pylist([row], schema=schema),
                    data_dir / "part-00000.parquet",
                )
                input_dirs[source] = data_dir

            output_dir = root / "processed"
            args = parse_args(
                [
                    "--pdf-dir", str(input_dirs["nepali_pdf"]),
                    "--news-dir", str(input_dirs["nepali_news"]),
                    "--lyrics-dir", str(input_dirs["nepali_lyrics"]),
                    "--output-dir", str(output_dir),
                    "--train-ratio", "0.8",
                    "--validation-ratio", "0.1",
                    "--test-ratio", "0.1",
                    "--log-every", "0",
                ]
            )
            manifest = build_dataset(args)

            self.assertEqual(manifest["input_stage"], "cleaned")
            self.assertEqual(manifest["total_documents"], 2)
            self.assertEqual(
                manifest["source_stats"]["nepali_news"]["exact_duplicates"],
                1,
            )
            self.assertTrue((output_dir / "train.parquet").is_file())
            self.assertTrue((output_dir / "validation.parquet").is_file())
            self.assertTrue((output_dir / "test.parquet").is_file())
            self.assertTrue((output_dir / "build_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
