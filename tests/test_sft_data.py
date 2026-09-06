import unittest

from attention_maps.training.sft_data import (
    detect_sft_schema,
    normalize_sft_example,
)
from scripts.utils.nepali_text import CleaningConfig


class SFTDataTests(unittest.TestCase):
    def test_alpaca_conversion_does_not_clean_hub_text_by_default(self):
        row = {
            "instruction": "यो URL व्याख्या गर्नुहोस्: https://example.com",
            "input": "थप सन्दर्भ",
            "output": "यो उत्तर हो।",
        }

        normalized = normalize_sft_example(row, schema="alpaca")

        self.assertEqual(normalized["normalization_status"], "accepted")
        self.assertIn("https://example.com", normalized["prompt"][0]["content"])
        self.assertIn("थप जानकारी", normalized["prompt"][0]["content"])

    def test_lima_translation_uses_shared_nepali_cleaning(self):
        row = {
            "status": "success",
            "translation": (
                "HUMAN: यो पृष्ठ हेर्नुहोस् https://example.com र उत्तर दिनुहोस्।\n\n"
                "ASSISTANT: यो नेपाली भाषामा दिइएको उपयोगी उत्तर हो।"
            ),
        }
        config = CleaningConfig(
            min_devanagari_ratio=0.1,
            min_devanagari_letters=1,
            min_characters=1,
            mode="preserve",
        )

        normalized = normalize_sft_example(
            row,
            schema="auto",
            cleaning_config=config,
        )

        self.assertEqual(detect_sft_schema(row), "lima")
        self.assertEqual(normalized["normalization_status"], "accepted")
        self.assertNotIn("https://", normalized["prompt"][0]["content"])

    def test_custom_field_map_handles_a_future_schema(self):
        row = {
            "question_text": "नेपालको राजधानी के हो?",
            "context_text": "देशको प्रशासनिक केन्द्र बताउनुहोस्।",
            "answer_text": "काठमाडौं नेपालको राजधानी हो।",
        }

        normalized = normalize_sft_example(
            row,
            schema="alpaca",
            field_map={
                "instruction": "question_text",
                "input": "context_text",
                "output": "answer_text",
            },
        )

        self.assertEqual(normalized["normalization_status"], "accepted")
        self.assertIn("प्रशासनिक केन्द्र", normalized["prompt"][0]["content"])
        self.assertEqual(
            normalized["completion"][0]["content"],
            "काठमाडौं नेपालको राजधानी हो।",
        )

    def test_chat_schema_uses_final_assistant_message_as_completion(self):
        row = {
            "messages": [
                {"role": "system", "content": "नेपालीमा उत्तर दिनुहोस्।"},
                {"role": "user", "content": "नमस्ते"},
                {"role": "assistant", "content": "नमस्ते!"},
                {"role": "user", "content": "नेपालको राजधानी?"},
                {"role": "assistant", "content": "काठमाडौं।"},
            ]
        }

        normalized = normalize_sft_example(row)

        self.assertEqual(normalized["source_schema"], "chat")
        self.assertEqual(len(normalized["prompt"]), 4)
        self.assertEqual(normalized["completion"][0]["content"], "काठमाडौं।")


if __name__ == "__main__":
    unittest.main()
