from __future__ import annotations

import unittest

from scripts.utils.nepali_text import (
    CleaningConfig,
    clean_text,
    clean_text_with_result,
    devanagari_letter_stats,
)


NEPALI = "नेपाल एउटा सुन्दर देश हो र यहाँ धेरै भाषा बोलिन्छन्।"


class NepaliCleaningTests(unittest.TestCase):
    def test_ratio_ignores_punctuation_and_digits(self) -> None:
        ratio, count = devanagari_letter_stats("नेपाल २०८१, 2024!")
        self.assertEqual(ratio, 1.0)
        self.assertGreater(count, 0)

    def test_preserve_mode_keeps_mixed_terms_but_removes_url(self) -> None:
        text = f"{NEPALI} AI 2025 https://example.com/page"
        cleaned = clean_text(text)
        self.assertIn("AI 2025", cleaned)
        self.assertNotIn("example.com", cleaned)

    def test_markup_entities_and_unsafe_controls_are_cleaned(self) -> None:
        text = (
            "<article>नेपाल एउटा सुन्दर देश हो&nbsp;र यहाँ धेरै भाषा बोलिन्छन्।</article>"
            "<script>English tracking payload</script>\ufeff"
        )
        cleaned = clean_text(text)
        self.assertIn("देश हो र यहाँ", cleaned)
        self.assertNotIn("article", cleaned)
        self.assertNotIn("tracking", cleaned)
        self.assertNotIn("\ufeff", cleaned)

    def test_strict_mode_removes_latin_but_keeps_digits(self) -> None:
        text = f"{NEPALI} AI 2025 test@example.com"
        cleaned = clean_text(text, CleaningConfig(mode="strict"))
        self.assertNotIn("AI", cleaned)
        self.assertNotIn("example", cleaned)
        self.assertIn("2025", cleaned)
        self.assertIn("।", cleaned)

    def test_ratio_rejects_english_dominant_document(self) -> None:
        result = clean_text_with_result(
            "This is mostly English with नेपाल appended at the end.",
            CleaningConfig(min_devanagari_letters=1, min_characters=1),
        )
        self.assertEqual(result.reason, "low_ratio")

    def test_minimum_content_rejects_single_nepali_character(self) -> None:
        result = clean_text_with_result("न", CleaningConfig())
        self.assertEqual(result.reason, "few_devanagari")

if __name__ == "__main__":
    unittest.main()
