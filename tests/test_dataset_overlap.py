import unittest

from attention_maps.eda.deduplication import DeduplicationConfig
from attention_maps.eda.overlap import analyze_dataset_overlap


class DatasetOverlapTests(unittest.TestCase):
    def test_reports_internal_duplicates_and_both_containment_directions(self):
        shared = "नेपालमा धेरै भाषा बोलिन्छ र यहाँ विविध संस्कृति पाइन्छ।"
        punctuation_variant = (
            "नेपालमा धेरै भाषा बोलिन्छ, र यहाँ विविध संस्कृति पाइन्छ!"
        )
        capital = "काठमाडौँ नेपालको राजधानी तथा ठूलो ऐतिहासिक सहर हो।"
        other = "हिमाली क्षेत्रमा धेरै सुन्दर ताल र हिमालहरू रहेका छन्।"
        another = "तराई क्षेत्रमा उष्ण हावापानी र उर्वर खेतीयोग्य जमिन पाइन्छ।"
        datasets = {
            "dataset-a": [shared, shared, capital],
            "dataset-b": [shared, punctuation_variant, other, another],
        }
        original = {key: list(value) for key, value in datasets.items()}
        config = DeduplicationConfig(
            minhash_permutations=16,
            minhash_bands=4,
            near_duplicate_threshold=1.0,
            edit_similarity_threshold=1.0,
        )

        result = analyze_dataset_overlap(datasets, config)

        summaries = {summary.dataset_id: summary for summary in result.datasets}
        self.assertEqual(summaries["dataset-a"].exact_duplicates, 1)
        self.assertEqual(summaries["dataset-a"].unique_documents, 2)
        self.assertEqual(summaries["dataset-b"].near_duplicates, 1)
        self.assertEqual(summaries["dataset-b"].unique_documents, 3)
        self.assertAlmostEqual(summaries["dataset-a"].internal_duplicate_ratio, 1 / 3)
        self.assertEqual(summaries["dataset-b"].internal_duplicate_ratio, 0.25)
        directions = {
            (row.source_dataset_id, row.covering_dataset_id): row
            for row in result.directional_containment
        }
        self.assertEqual(directions[("dataset-a", "dataset-b")].matched_documents, 1)
        self.assertEqual(directions[("dataset-b", "dataset-a")].matched_documents, 1)
        self.assertEqual(directions[("dataset-a", "dataset-b")].containment_ratio, 0.5)
        self.assertAlmostEqual(
            directions[("dataset-b", "dataset-a")].containment_ratio, 1 / 3
        )
        self.assertEqual(datasets, original)

    def test_emits_every_ordered_direction_for_selected_datasets(self):
        result = analyze_dataset_overlap(
            {
                "a": ["नेपालमा विभिन्न भाषा र संस्कृतिको विविधता पाइन्छ।"],
                "b": ["काठमाडौँ नेपालको ऐतिहासिक राजधानी सहर हो।"],
                "c": ["हिमाली क्षेत्रमा धेरै हिमाल र ताल रहेका छन्।"],
            },
            DeduplicationConfig(minhash_permutations=16, minhash_bands=4),
        )

        self.assertEqual(len(result.directional_containment), 6)
        self.assertEqual(
            {
                (row.source_dataset_id, row.covering_dataset_id)
                for row in result.directional_containment
            },
            {
                ("a", "b"),
                ("a", "c"),
                ("b", "a"),
                ("b", "c"),
                ("c", "a"),
                ("c", "b"),
            },
        )

    def test_requires_two_datasets(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            analyze_dataset_overlap({"only": ["नेपाल एउटा सुन्दर देश हो।"]})


if __name__ == "__main__":
    unittest.main()
