"""Pure Nepali text normalization and filtering primitives; no file I/O or CLI."""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass

from .script import identify_script_category


URL_PATTERN = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
EMAIL_PATTERN = re.compile(r"(?<!\S)[^\s@]+@[^\s@]+\.[^\s@]+(?!\S)")
SCRIPT_STYLE_PATTERN = re.compile(
    r"(?is)<(script|style)\b[^>]*>.*?</\1\s*>"
)
HTML_TAG_PATTERN = re.compile(r"(?s)<!--.*?-->|<[^>]+>")

# Punctuation and symbols that commonly occur in Nepali prose and official
# documents. ASCII digits are handled separately so dates and measurements are
# not silently destroyed.
ALLOWED_PUNCTUATION = set(
    "।॥,.!?:;()[]{}\"'/%&@₹+–—-_…“”‘’"
)
ORPHAN_PUNCTUATION = ALLOWED_PUNCTUATION - {"।", "॥"}
ORPHAN_PUNCT_TOKEN = re.compile(
    rf"(?<!\S)[{re.escape(''.join(sorted(ORPHAN_PUNCTUATION)))}]+(?!\S)"
)
REPEATED_SEPARATOR = re.compile(r"([,;:/-])(?:\s+\1)+")


@dataclass(frozen=True)
class CleaningConfig:
    min_devanagari_ratio: float = 0.80
    min_devanagari_letters: int = 10
    min_characters: int = 20
    mode: str = "preserve"
    normalization: str = "NFC"

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_devanagari_ratio <= 1.0:
            raise ValueError("min_devanagari_ratio must be in [0, 1]")
        if self.min_devanagari_letters < 1:
            raise ValueError("min_devanagari_letters must be positive")
        if self.min_characters < 1:
            raise ValueError("min_characters must be positive")
        if self.mode not in {"preserve", "strict"}:
            raise ValueError("mode must be 'preserve' or 'strict'")
        if self.normalization not in {"NFC", "NFKC"}:
            raise ValueError("normalization must be NFC or NFKC")


@dataclass(frozen=True)
class CleaningResult:
    text: str
    reason: str | None
    devanagari_ratio: float
    devanagari_letters: int
    script_category: str = "Other"

    @property
    def accepted(self) -> bool:
        return self.reason is None


def is_devanagari(character: str) -> bool:
    """Return whether a character belongs to a Devanagari Unicode block."""

    codepoint = ord(character)
    return (
        0x0900 <= codepoint <= 0x097F
        or 0xA8E0 <= codepoint <= 0xA8FF
        or 0x11B00 <= codepoint <= 0x11B5F
    )


def devanagari_letter_stats(text: str) -> tuple[float, int]:
    """Compute Devanagari share over letters, ignoring digits and punctuation."""

    total_letters = 0
    devanagari_letters = 0
    for character in text:
        if not unicodedata.category(character).startswith("L"):
            continue
        total_letters += 1
        devanagari_letters += is_devanagari(character)
    ratio = devanagari_letters / total_letters if total_letters else 0.0
    return ratio, devanagari_letters


def remove_web_artifacts(text: str) -> str:
    """Remove complete URLs/emails before any character-level filtering."""

    text = URL_PATTERN.sub(" ", text)
    return EMAIL_PATTERN.sub(" ", text)


def remove_markup_artifacts(text: str) -> str:
    """Decode HTML entities and remove tags plus script/style payloads."""

    text = SCRIPT_STYLE_PATTERN.sub(" ", text)
    text = HTML_TAG_PATTERN.sub(" ", text)
    return html.unescape(text)


def remove_unsafe_controls(text: str) -> str:
    """Remove BOM/control/format debris while retaining layout and conjunct marks."""

    retained_controls = {"\n", "\t", "\u200c", "\u200d"}
    return "".join(
        character
        for character in text
        if character in retained_controls
        or not unicodedata.category(character).startswith("C")
    )


def is_allowed_strict_character(character: str) -> bool:
    return (
        is_devanagari(character)
        or character.isspace()
        or character.isascii() and character.isdigit()
        or character in ALLOWED_PUNCTUATION
        or character in {"\u200c", "\u200d"}  # ZWNJ and ZWJ in conjuncts
    )


def normalize_whitespace(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    text = "\n".join(lines)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def clean_text_with_result(
    text: str,
    config: CleaningConfig = CleaningConfig(),
) -> CleaningResult:
    """Filter one document and return its cleaned text plus rejection reason."""

    if not isinstance(text, str) or not text.strip():
        return CleaningResult("", "empty", 0.0, 0)

    normalized = unicodedata.normalize(config.normalization, text)
    without_markup = remove_markup_artifacts(normalized)
    sanitized = remove_unsafe_controls(remove_web_artifacts(without_markup))
    ratio, devanagari_letters = devanagari_letter_stats(sanitized)
    script_category = identify_script_category(sanitized)

    if ratio < config.min_devanagari_ratio:
        return CleaningResult(
            "", "low_ratio", ratio, devanagari_letters, script_category
        )
    if devanagari_letters < config.min_devanagari_letters:
        return CleaningResult(
            "", "few_devanagari", ratio, devanagari_letters, script_category
        )

    if config.mode == "strict":
        cleaned = "".join(
            character
            for character in sanitized
            if is_allowed_strict_character(character)
        )
        cleaned = ORPHAN_PUNCT_TOKEN.sub("", cleaned)
        cleaned = REPEATED_SEPARATOR.sub(r"\1", cleaned)
    else:
        # Preserve legitimate English names, acronyms, and technical terms in a
        # document that has already passed the Nepali-dominance threshold.
        cleaned = sanitized

    cleaned = normalize_whitespace(cleaned)
    visible_characters = sum(not char.isspace() for char in cleaned)
    if not cleaned:
        return CleaningResult("", "empty", ratio, devanagari_letters, script_category)
    if visible_characters < config.min_characters:
        return CleaningResult("", "short", ratio, devanagari_letters, script_category)
    return CleaningResult(cleaned, None, ratio, devanagari_letters, script_category)


def clean_text(text: str, config: CleaningConfig = CleaningConfig()) -> str:
    """Convenience API compatible with simple dataframe/string workflows."""

    return clean_text_with_result(text, config).text
