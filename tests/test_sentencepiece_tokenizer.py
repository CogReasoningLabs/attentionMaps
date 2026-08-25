from __future__ import annotations

import unittest

from attention_maps.config import TokenizerConfig
from attention_maps.tokenization.models import (
    tokenizer_from_state,
    train_sentencepiece_tokenizer,
)


class SentencePieceTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = TokenizerConfig(
            backend="sentencepiece",
            vocab_size=128,
            model_type="bpe",
            normalization="nmt_nfkc",
            character_coverage=1.0,
            byte_fallback=False,
            hard_vocab_limit=False,
        )
        sentences = [
            "नेपाल एउटा सुन्दर देश हो।",
            "काठमाडौं नेपालको राजधानी हो।",
            "नेपाली भाषामा पुस्तक र समाचार लेखिन्छन्।",
            "यो परीक्षणमा English र नेपाली दुवै शब्द छन्।",
        ] * 30
        cls.tokenizer = train_sentencepiece_tokenizer(iter(sentences), config)

    def test_special_token_ids_are_stable(self) -> None:
        self.assertEqual(self.tokenizer.pad_id, 0)
        self.assertEqual(self.tokenizer.unk_id, 1)
        self.assertEqual(self.tokenizer.bos_id, 2)
        self.assertEqual(self.tokenizer.eos_id, 3)

    def test_nepali_round_trip(self) -> None:
        text = "नेपाल एउटा सुन्दर देश हो।"
        ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        self.assertEqual(self.tokenizer.decode_to_string(ids), text)

    def test_serialized_state_round_trip(self) -> None:
        restored = tokenizer_from_state(self.tokenizer.to_state())
        text = "काठमाडौं नेपालको राजधानी हो।"
        self.assertEqual(restored.encode(text), self.tokenizer.encode(text))


if __name__ == "__main__":
    unittest.main()
