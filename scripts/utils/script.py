"""Explainable script and Romanized-Nepali classification primitives."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable


DEVANAGARI = "Devanagari"
MIXED_DEVANAGARI_ROMANIZED = "Mixed (Devanagari + romanized)"
ROMANIZED = "Romanized"
LATIN = "Latin"
OTHER = "Other"
MIXED_NEPALI_ENGLISH = "Mixed(Nepali+English)"
SCRIPT_CATEGORIES = (
    DEVANAGARI,
    MIXED_DEVANAGARI_ROMANIZED,
    ROMANIZED,
    LATIN,
    OTHER,
    MIXED_NEPALI_ENGLISH,
)

# This intentionally contains signals that are useful in Romanized Nepali and
# uncommon in ordinary English. Ambiguous English tokens such as "to", "me",
# and "is" are excluded even though they can occur in transliterations.
ROMANIZED_NEPALI_TERMS = frozenset(
    {
        "aaja",
        "aama",
        "aba",
        "afno",
        "ani",
        "baba",
        "bahira",
        "bhanchha",
        "bhane",
        "bhayo",
        "bholi",
        "cha",
        "chha",
        "chhaina",
        "dai",
        "dherai",
        "didi",
        "garchha",
        "garchhu",
        "garda",
        "garnu",
        "ghar",
        "hajur",
        "hamile",
        "hamro",
        "haru",
        "huncha",
        "hunchha",
        "kahile",
        "kata",
        "kina",
        "lai",
        "maile",
        "malai",
        "manchhe",
        "mero",
        "naam",
        "namaste",
        "nepali",
        "pachi",
        "pani",
        "ramro",
        "sabai",
        "sanga",
        "tapai",
        "tapain",
        "timi",
        "timilai",
        "timro",
        "yaha",
        "yo",
    }
)
ROMANIZED_NEPALI_PATTERN = re.compile(
    r"(?:chh|bhay|bhan|garch|garn|hunch|hund|haru|tapai|timro|malai|"
    r"maile|hamro|sanga|dekhi|samma|wala|wali)"
)
LATIN_TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)?")
MIN_MIXED_LETTER_SHARE = 0.05
MIN_ROMANIZED_TOKEN_SHARE = 0.35


@dataclass(frozen=True)
class ScriptEvidence:
    """Counts sufficient to classify one document or an entire stream."""

    devanagari_letters: int = 0
    latin_letters: int = 0
    other_letters: int = 0
    latin_tokens: int = 0
    romanized_nepali_tokens: int = 0

    def __add__(self, other: "ScriptEvidence") -> "ScriptEvidence":
        if not isinstance(other, ScriptEvidence):
            return NotImplemented
        return ScriptEvidence(
            self.devanagari_letters + other.devanagari_letters,
            self.latin_letters + other.latin_letters,
            self.other_letters + other.other_letters,
            self.latin_tokens + other.latin_tokens,
            self.romanized_nepali_tokens + other.romanized_nepali_tokens,
        )

    @property
    def total_letters(self) -> int:
        return self.devanagari_letters + self.latin_letters + self.other_letters

    @property
    def romanized_token_share(self) -> float:
        if not self.latin_tokens:
            return 0.0
        return self.romanized_nepali_tokens / self.latin_tokens

    def letter_share(self, count: int) -> float:
        return count / self.total_letters if self.total_letters else 0.0


def is_devanagari_character(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x0900 <= codepoint <= 0x097F
        or 0xA8E0 <= codepoint <= 0xA8FF
        or 0x11B00 <= codepoint <= 0x11B5F
    )


def is_latin_character(character: str) -> bool:
    return "LATIN" in unicodedata.name(character, "")


def is_romanized_nepali_token(token: str) -> bool:
    """Return a conservative lexical/transliteration signal for one token."""

    normalized = "".join(
        character
        for character in unicodedata.normalize("NFKD", token).casefold()
        if character.isascii() and character.isalpha()
    )
    return bool(
        normalized in ROMANIZED_NEPALI_TERMS
        or len(normalized) >= 4
        and ROMANIZED_NEPALI_PATTERN.search(normalized)
    )


def script_evidence(text: str) -> ScriptEvidence:
    """Extract Unicode-script counts and conservative Romanized-Nepali signals."""

    devanagari = latin = other = 0
    for character in text if isinstance(text, str) else "":
        if not unicodedata.category(character).startswith("L"):
            continue
        if is_devanagari_character(character):
            devanagari += 1
        elif is_latin_character(character):
            latin += 1
        else:
            other += 1
    latin_tokens = LATIN_TOKEN_PATTERN.findall(text if isinstance(text, str) else "")
    return ScriptEvidence(
        devanagari_letters=devanagari,
        latin_letters=latin,
        other_letters=other,
        latin_tokens=len(latin_tokens),
        romanized_nepali_tokens=sum(
            is_romanized_nepali_token(token) for token in latin_tokens
        ),
    )


def classify_script_evidence(evidence: ScriptEvidence) -> str:
    """Map aggregate evidence to the dataset taxonomy's six script labels."""

    if not evidence.total_letters:
        return OTHER
    devanagari_share = evidence.letter_share(evidence.devanagari_letters)
    latin_share = evidence.letter_share(evidence.latin_letters)
    other_share = evidence.letter_share(evidence.other_letters)
    if other_share > max(devanagari_share, latin_share):
        return OTHER
    has_devanagari = (
        evidence.devanagari_letters > 0 and devanagari_share >= MIN_MIXED_LETTER_SHARE
    )
    has_latin = evidence.latin_letters > 0 and latin_share >= MIN_MIXED_LETTER_SHARE
    romanized = (
        evidence.romanized_nepali_tokens > 0
        and evidence.romanized_token_share >= MIN_ROMANIZED_TOKEN_SHARE
    )

    if has_devanagari and has_latin:
        return MIXED_DEVANAGARI_ROMANIZED if romanized else MIXED_NEPALI_ENGLISH
    if has_devanagari:
        return DEVANAGARI
    if has_latin:
        return ROMANIZED if romanized else LATIN
    return OTHER


def identify_script_category(texts: str | Iterable[str]) -> str:
    """Classify one string or aggregate an iterable without retaining its text."""

    values = (texts,) if isinstance(texts, str) else texts
    evidence = ScriptEvidence()
    for text in values:
        evidence += script_evidence(text)
    return classify_script_evidence(evidence)
