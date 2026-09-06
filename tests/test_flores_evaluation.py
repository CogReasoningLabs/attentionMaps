import unittest
from unittest.mock import patch

from attention_maps.evaluation.flores import (
    FLORES_CONFIG,
    FLORES_DATASET_REVISION,
    FloresEvaluationError,
    load_flores_examples,
    score_flores_results,
)
from attention_maps.inference.comparison import ComparisonResult


class FakeScore:
    score = 100.0


class FakeSacreBleu:
    @staticmethod
    def sentence_chrf(*args, **kwargs):
        return FakeScore()

    @staticmethod
    def corpus_chrf(*args, **kwargs):
        return FakeScore()


class FloresLoadingTests(unittest.TestCase):
    def test_streams_only_requested_slice(self):
        rows = (
            {
                "id": index,
                "eng_Latn": f"English {index}",
                "npi_Deva": f"नेपाली {index}",
            }
            for index in range(20)
        )
        with patch("datasets.load_dataset", return_value=rows) as load:
            examples = load_flores_examples(split="dev", offset=4, limit=3)

        self.assertEqual([example.position for example in examples], [4, 5, 6])
        self.assertEqual(examples[0].source, "English 4")
        load.assert_called_once_with(
            "yash9439/flores200",
            FLORES_CONFIG,
            split="dev",
            streaming=True,
            token=None,
            revision=FLORES_DATASET_REVISION,
        )

    def test_rejects_out_of_range_slice(self):
        with self.assertRaises(FloresEvaluationError):
            load_flores_examples(split="dev", offset=997, limit=1)


class FloresScoringTests(unittest.TestCase):
    def test_scores_sentence_and_corpus_chrfpp(self):
        rows = [
            {
                "id": 1,
                "eng_Latn": "Hello",
                "npi_Deva": "नमस्ते",
            }
        ]
        with patch("datasets.load_dataset", return_value=rows):
            examples = load_flores_examples(split="dev", limit=1)
        results = [
            ComparisonResult(
                model="teacher",
                decoding="standard",
                sample_index=0,
                prompt="Translate Hello",
                output="नमस्ते",
                latency_seconds=0.5,
            )
        ]

        with patch(
            "attention_maps.evaluation.flores._sacrebleu",
            return_value=FakeSacreBleu(),
        ):
            details, summaries = score_flores_results(results, examples)

        self.assertEqual(details[0]["chrF++"], 100.0)
        self.assertEqual(summaries[0]["chrF++"], 100.0)
        self.assertEqual(summaries[0]["coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
