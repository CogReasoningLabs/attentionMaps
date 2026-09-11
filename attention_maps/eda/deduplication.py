"""Sequential exact, MinHash-LSH, and boilerplate duplicate diagnostics."""

from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass


_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 64) - 1
_PARAGRAPH_BOUNDARY = re.compile(r"\n+")


@dataclass(frozen=True)
class DeduplicationConfig:
    normalization: str = "NFC"
    lowercase: bool = True
    collapse_whitespace: bool = True
    hash_algorithm: str = "sha256"
    shingle_size: int = 5
    minhash_permutations: int = 128
    minhash_bands: int = 16
    near_duplicate_threshold: float = 0.80
    boilerplate_min_documents: int = 3
    boilerplate_min_characters: int = 40
    seed: int = 42

    def __post_init__(self) -> None:
        if self.normalization not in {"NFC", "NFKC"}:
            raise ValueError("dedup normalization must be NFC or NFKC")
        if self.hash_algorithm != "sha256":
            raise ValueError("only SHA-256 exact hashing is supported")
        if self.shingle_size <= 0:
            raise ValueError("shingle_size must be positive")
        if self.minhash_permutations <= 0:
            raise ValueError("minhash_permutations must be positive")
        if self.minhash_bands <= 0:
            raise ValueError("minhash_bands must be positive")
        if self.minhash_permutations % self.minhash_bands:
            raise ValueError("minhash_bands must divide minhash_permutations")
        if not 0 < self.near_duplicate_threshold <= 1:
            raise ValueError("near_duplicate_threshold must be in (0, 1]")
        if self.boilerplate_min_documents < 2:
            raise ValueError("boilerplate_min_documents must be at least 2")
        if self.boilerplate_min_characters <= 0:
            raise ValueError("boilerplate_min_characters must be positive")


@dataclass(frozen=True)
class DuplicateDecision:
    exact_duplicate: bool = False
    near_duplicate: bool = False

    @property
    def duplicate(self) -> bool:
        return self.exact_duplicate or self.near_duplicate


@dataclass(frozen=True)
class BoilerplateSummary:
    unique_repeated_paragraphs: int
    repeated_paragraph_occurrences: int
    affected_documents: int


def normalize_for_deduplication(
    text: str,
    *,
    normalization: str = "NFC",
    lowercase: bool = True,
    collapse_whitespace: bool = True,
) -> str:
    """Create comparison text; Unicode normalization always precedes hashing."""

    if normalization not in {"NFC", "NFKC"}:
        raise ValueError("normalization must be NFC or NFKC")
    normalized = unicodedata.normalize(normalization, text).replace("\ufeff", " ")
    if collapse_whitespace:
        normalized = " ".join(normalized.split())
    else:
        normalized = normalized.strip()
    return normalized.casefold() if lowercase else normalized


def exact_sha256(text: str, config: DeduplicationConfig) -> bytes:
    normalized = normalize_for_deduplication(
        text,
        normalization=config.normalization,
        lowercase=config.lowercase,
        collapse_whitespace=config.collapse_whitespace,
    )
    return hashlib.sha256(normalized.encode("utf-8")).digest()


def character_shingles(text: str, size: int) -> set[str]:
    if not text:
        return set()
    if len(text) <= size:
        return {text}
    return {text[index : index + size] for index in range(len(text) - size + 1)}


class _MinHashLSHIndex:
    def __init__(self, config: DeduplicationConfig):
        self.config = config
        randomizer = random.Random(config.seed)
        self.coefficients = tuple(
            (
                randomizer.randrange(1, _MERSENNE_PRIME),
                randomizer.randrange(0, _MERSENNE_PRIME),
            )
            for _ in range(config.minhash_permutations)
        )
        self.signatures: list[tuple[int, ...]] = []
        self.buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
        self.rows_per_band = config.minhash_permutations // config.minhash_bands

    def signature(self, text: str) -> tuple[int, ...]:
        shingles = character_shingles(text, self.config.shingle_size)
        if not shingles:
            return tuple([_MAX_HASH] * self.config.minhash_permutations)
        hashes = tuple(
            int.from_bytes(
                hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(),
                "big",
            )
            for value in shingles
        )
        return tuple(
            min((coefficient * value + offset) % _MERSENNE_PRIME for value in hashes)
            for coefficient, offset in self.coefficients
        )

    def contains_near(self, signature: tuple[int, ...]) -> bool:
        candidates: set[int] = set()
        for band in range(self.config.minhash_bands):
            start = band * self.rows_per_band
            key = (band, signature[start : start + self.rows_per_band])
            candidates.update(self.buckets[key])
        for candidate in candidates:
            other = self.signatures[candidate]
            estimated_jaccard = sum(
                left == right for left, right in zip(signature, other)
            ) / len(signature)
            if estimated_jaccard >= self.config.near_duplicate_threshold:
                return True
        return False

    def add(self, signature: tuple[int, ...]) -> None:
        index = len(self.signatures)
        self.signatures.append(signature)
        for band in range(self.config.minhash_bands):
            start = band * self.rows_per_band
            key = (band, signature[start : start + self.rows_per_band])
            self.buckets[key].append(index)


class MultiStageDeduplicator:
    """Observe documents sequentially, retaining only bounded comparison state."""

    def __init__(self, config: DeduplicationConfig):
        self.config = config
        self.exact_hashes: set[bytes] = set()
        self.near_index = _MinHashLSHIndex(config)
        self.document_paragraphs: list[frozenset[bytes]] = []
        self.paragraph_counts: Counter[bytes] = Counter()

    def observe(self, text: str) -> DuplicateDecision:
        normalized = normalize_for_deduplication(
            text,
            normalization=self.config.normalization,
            lowercase=self.config.lowercase,
            collapse_whitespace=self.config.collapse_whitespace,
        )
        digest = hashlib.sha256(normalized.encode("utf-8")).digest()
        if digest in self.exact_hashes:
            return DuplicateDecision(exact_duplicate=True)
        self.exact_hashes.add(digest)

        signature = self.near_index.signature(normalized)
        near_duplicate = self.near_index.contains_near(signature)
        self.near_index.add(signature)
        if near_duplicate:
            return DuplicateDecision(near_duplicate=True)

        paragraphs = frozenset(self._paragraph_hashes(text))
        self.document_paragraphs.append(paragraphs)
        self.paragraph_counts.update(paragraphs)
        return DuplicateDecision()

    def boilerplate_summary(self) -> BoilerplateSummary:
        repeated = self.boilerplate_hashes()
        return BoilerplateSummary(
            unique_repeated_paragraphs=len(repeated),
            repeated_paragraph_occurrences=sum(
                self.paragraph_counts[digest] for digest in repeated
            ),
            affected_documents=sum(
                bool(paragraphs.intersection(repeated))
                for paragraphs in self.document_paragraphs
            ),
        )

    def boilerplate_hashes(self) -> frozenset[bytes]:
        return frozenset(
            digest
            for digest, count in self.paragraph_counts.items()
            if count >= self.config.boilerplate_min_documents
        )

    def _paragraph_hashes(self, text: str):
        normalized = unicodedata.normalize(self.config.normalization, text).replace(
            "\ufeff", " "
        )
        for paragraph in _PARAGRAPH_BOUNDARY.split(normalized):
            paragraph = normalize_for_deduplication(
                paragraph,
                normalization=self.config.normalization,
                lowercase=self.config.lowercase,
                collapse_whitespace=True,
            )
            if len(paragraph) >= self.config.boilerplate_min_characters:
                yield hashlib.sha256(paragraph.encode("utf-8")).digest()


def strip_boilerplate_paragraphs(
    text: str,
    repeated_hashes: frozenset[bytes],
    config: DeduplicationConfig,
) -> tuple[str, int]:
    """Remove paragraphs identified by a completed deduplicator pass."""

    retained: list[str] = []
    removed = 0
    normalized = unicodedata.normalize(config.normalization, text).replace("\ufeff", " ")
    for paragraph in _PARAGRAPH_BOUNDARY.split(normalized):
        comparison = normalize_for_deduplication(
            paragraph,
            normalization=config.normalization,
            lowercase=config.lowercase,
            collapse_whitespace=True,
        )
        digest = hashlib.sha256(comparison.encode("utf-8")).digest()
        if (
            len(comparison) >= config.boilerplate_min_characters
            and digest in repeated_hashes
        ):
            removed += 1
            continue
        visible = " ".join(paragraph.split())
        if visible:
            retained.append(visible)
    return "\n".join(retained), removed
