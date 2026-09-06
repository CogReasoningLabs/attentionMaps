import unittest
from pathlib import Path

from attention_maps.tokenization.analysis import (
    TokenizerSpec,
    analyses_csv,
    analyze_tokenizer,
    discover_repository_tokenizers,
    nepali_words,
)


class FakeTokenizer:
    all_special_ids = [0]
    unk_token_id = 0

    def __init__(self):
        self.vocab = {
            "<unk>": 0,
            "नेपाल": 1,
            "को": 2,
            "राज": 3,
            "धानी": 4,
            "hello": 5,
        }

    def get_vocab(self):
        return self.vocab

    def encode(self, text, add_special_tokens=False):
        mapping = {
            "नेपाल": [1],
            "को": [2],
            "राजधानी": [3, 4],
            "नेपाल को राजधानी": [1, 2, 3, 4],
        }
        return mapping.get(text, [0])

    def decode(self, token_ids, **kwargs):
        inverse = {value: key for key, value in self.vocab.items()}
        return "".join(inverse[token_id] for token_id in token_ids)

    def convert_ids_to_tokens(self, token_id):
        inverse = {value: key for key, value in self.vocab.items()}
        return inverse[token_id]


class TokenizerAnalysisTests(unittest.TestCase):
    def test_measures_vocabulary_and_sample_efficiency(self):
        analysis = analyze_tokenizer(
            TokenizerSpec("fake", "Fake", "fake"),
            FakeTokenizer(),
            "नेपाल को राजधानी",
        )

        self.assertEqual(analysis.vocabulary_tokens, 5)
        self.assertEqual(analysis.devanagari_vocabulary_tokens, 4)
        self.assertEqual(analysis.devanagari_vocabulary_percent, 80.0)
        self.assertEqual(analysis.sample_tokens, 4)
        self.assertEqual(analysis.tokens_per_nepali_word, 1.3333)
        self.assertEqual(analysis.single_token_nepali_word_percent, round(200 / 3, 4))
        self.assertEqual(analysis.unknown_token_percent, 0.0)
        self.assertEqual(len(analysis.pieces), 4)

    def test_nepali_word_extraction_excludes_danda_and_latin(self):
        self.assertEqual(
            nepali_words("नेपाल सुन्दर छ। AI उपयोगी छ।"),
            ["नेपाल", "सुन्दर", "छ", "उपयोगी", "छ"],
        )

    def test_discovers_only_complete_repository_tokenizers(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            complete = root / "nepali" / "tokenizer"
            complete.mkdir(parents=True)
            (complete / "tokenizer.json").write_text("{}", encoding="utf-8")
            (root / "incomplete" / "tokenizer").mkdir(parents=True)

            specs = discover_repository_tokenizers(root)

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].key, "repository:nepali")
        self.assertTrue(specs[0].local)

    def test_csv_contains_decision_metrics(self):
        analysis = analyze_tokenizer(
            TokenizerSpec("fake", "Fake", "fake"),
            FakeTokenizer(),
            "नेपाल को राजधानी",
        )
        csv_text = analyses_csv([analysis])
        self.assertIn("Devanagari vocabulary %", csv_text)
        self.assertIn("tokens / Nepali word", csv_text)


if __name__ == "__main__":
    unittest.main()
