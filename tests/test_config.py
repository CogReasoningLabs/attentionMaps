from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from attention_maps.config import ExperimentProfile, load_experiment_profile


ROOT = Path(__file__).resolve().parents[1]


class ExperimentProfileTests(unittest.TestCase):
    def test_bundled_profiles_load(self) -> None:
        english = load_experiment_profile(ROOT / "profiles/en_wikitext103.json")
        nepali = load_experiment_profile(ROOT / "profiles/ne_wikipedia.json")
        local = load_experiment_profile(ROOT / "profiles/local_text.example.json")
        pdf = load_experiment_profile(ROOT / "profiles/ne_pdf_local.json")

        self.assertEqual(english.dataset.language, "en")
        self.assertEqual(english.dataset.splits.validation, "validation")
        self.assertEqual(nepali.dataset.language, "ne")
        self.assertIsNone(nepali.dataset.splits.validation)
        self.assertEqual(nepali.dataset.splits.validation_fraction, 0.01)
        self.assertEqual(english.tokenizer.backend, "sentencepiece")
        self.assertEqual(nepali.tokenizer.backend, "sentencepiece")
        self.assertEqual(local.dataset.source, "local")
        self.assertEqual(local.dataset.language, "und")
        self.assertEqual(pdf.dataset.loader, "parquet")
        self.assertEqual(pdf.tokenizer.vocab_size, 8000)

    def test_profile_round_trip(self) -> None:
        profile = load_experiment_profile(ROOT / "profiles/ne_wikipedia.json")
        restored = ExperimentProfile.from_dict(profile.to_dict())
        self.assertEqual(restored, profile)

    def test_rejects_conflicting_split_configuration(self) -> None:
        payload = load_experiment_profile(
            ROOT / "profiles/en_wikitext103.json"
        ).to_dict()
        payload["dataset"]["splits"]["validation_fraction"] = 0.1

        with self.assertRaisesRegex(ValueError, "either an explicit validation"):
            ExperimentProfile.from_dict(payload)

    def test_local_source_requires_files(self) -> None:
        payload = load_experiment_profile(
            ROOT / "profiles/en_wikitext103.json"
        ).to_dict()
        payload["dataset"]["source"] = "local"

        with self.assertRaisesRegex(ValueError, "require dataset.data_files"):
            ExperimentProfile.from_dict(payload)

    def test_loader_reports_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "broken.json"
            path.write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Could not load profile"):
                load_experiment_profile(path)


if __name__ == "__main__":
    unittest.main()
