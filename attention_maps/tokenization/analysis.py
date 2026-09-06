"""Tokenizer-only Nepali vocabulary and sample-efficiency analysis."""

from __future__ import annotations

import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


class TokenizerAnalysisError(ValueError):
    """Raised when a tokenizer cannot be loaded or analyzed."""


@dataclass(frozen=True)
class TokenizerSpec:
    key: str
    label: str
    source: str
    revision: str = "main"
    trust_remote_code: bool = False
    local: bool = False


@dataclass(frozen=True)
class TokenPiece:
    position: int
    token_id: int
    raw_token: str
    decoded_piece: str
    contains_devanagari: bool
    is_unknown: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TokenizerAnalysis:
    key: str
    tokenizer: str
    source: str
    vocabulary_tokens: int
    devanagari_vocabulary_tokens: int
    devanagari_vocabulary_percent: float
    sample_characters: int
    sample_tokens: int
    sample_nepali_token_percent: float
    nepali_word_occurrences: int
    unique_nepali_words: int
    tokens_per_nepali_word: float
    single_token_nepali_word_percent: float
    unknown_token_percent: float
    devanagari_vocabulary_examples: tuple[str, ...]
    pieces: tuple[TokenPiece, ...]

    def summary(self) -> dict[str, object]:
        return {
            "tokenizer": self.tokenizer,
            "source": self.source,
            "vocabulary": self.vocabulary_tokens,
            "Devanagari vocabulary tokens": self.devanagari_vocabulary_tokens,
            "Devanagari vocabulary %": self.devanagari_vocabulary_percent,
            "sample characters": self.sample_characters,
            "sample tokens": self.sample_tokens,
            "sample Devanagari token %": self.sample_nepali_token_percent,
            "tokens / Nepali word": self.tokens_per_nepali_word,
            "single-token Nepali word %": self.single_token_nepali_word_percent,
            "unknown token %": self.unknown_token_percent,
        }


DEFAULT_TOKENIZER_SPECS = (
    TokenizerSpec("gpt2", "GPT-2 · base", "gpt2"),
    TokenizerSpec(
        "tinyllama",
        "TinyLlama 1.1B · base",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ),
    TokenizerSpec(
        "arkios",
        "Arkios 1B Chat",
        "sajalregmi4/arkios-1b-chat",
    ),
    TokenizerSpec(
        "himalayagpt",
        "HimalayaGPT 0.5B Instruct",
        "himalaya-ai/himalayagpt-0.5b-it",
        revision="10ef5130093789db77b186fc37e524754848bb3a",
        trust_remote_code=True,
    ),
    TokenizerSpec(
        "gemma4-e2b",
        "Gemma 4 E2B · base",
        "google/gemma-4-E2B",
    ),
    TokenizerSpec(
        "himalaya-gemma4-e2b",
        "Himalaya Gemma 4 E2B Instruct",
        "himalaya-ai/himalaya-gemma-4-e2b-it",
    ),
    TokenizerSpec(
        "llama2-7b",
        "Llama 2 7B Chat · base",
        "meta-llama/Llama-2-7b-chat-hf",
    ),
)


def discover_repository_tokenizers(root: Path) -> list[TokenizerSpec]:
    """Discover portable tokenizers created by the materialized pipeline."""

    root = root.expanduser().resolve()
    if not root.is_dir():
        return []
    specs = []
    for tokenizer_path in sorted(root.glob("*/tokenizer")):
        if not (tokenizer_path / "tokenizer.json").is_file():
            continue
        dataset_name = tokenizer_path.parent.name
        specs.append(
            TokenizerSpec(
                key=f"repository:{dataset_name}",
                label=f"Repository tokenizer · {dataset_name}",
                source=str(tokenizer_path),
                local=True,
            )
        )
    return specs


def tokenizer_specs(repository_tokenizer_root: Path | None = None) -> list[TokenizerSpec]:
    specs = list(DEFAULT_TOKENIZER_SPECS)
    if repository_tokenizer_root is not None:
        specs.extend(discover_repository_tokenizers(repository_tokenizer_root))
    return specs


def load_tokenizer(
    spec: TokenizerSpec,
    *,
    token: str | None = None,
    local_files_only: bool = False,
) -> Any:
    """Load one tokenizer without loading any model weights."""

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise TokenizerAnalysisError(
            "Tokenizer analysis requires Transformers; install requirements.txt"
        ) from error
    try:
        return AutoTokenizer.from_pretrained(
            spec.source,
            revision=None if spec.local else spec.revision,
            trust_remote_code=spec.trust_remote_code,
            token=token or None,
            local_files_only=spec.local or local_files_only,
        )
    except Exception as error:
        if local_files_only and not spec.local:
            hint = " Disable cached-only mode to download missing tokenizer files."
        elif not spec.local:
            hint = " For gated models, accept their terms and set HF_TOKEN."
        else:
            hint = ""
        raise TokenizerAnalysisError(
            f"Could not load tokenizer {spec.label!r} from {spec.source!r}: "
            f"{error}.{hint}"
        ) from error


def is_devanagari(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x0900 <= codepoint <= 0x097F
        or 0xA8E0 <= codepoint <= 0xA8FF
        or 0x11B00 <= codepoint <= 0x11B5F
    )


def contains_devanagari(text: str) -> bool:
    return any(is_devanagari(character) for character in text)


def nepali_words(text: str) -> list[str]:
    """Extract Devanagari letter/mark sequences without treating danda as a word."""

    words = []
    current = []
    for character in text:
        category = unicodedata.category(character)
        if is_devanagari(character) and category[0] in {"L", "M"}:
            current.append(character)
            continue
        if current:
            word = "".join(current)
            if any(unicodedata.category(item).startswith("L") for item in word):
                words.append(word)
            current = []
    if current:
        word = "".join(current)
        if any(unicodedata.category(item).startswith("L") for item in word):
            words.append(word)
    return words


def _vocabulary(tokenizer: Any) -> dict[str, int]:
    try:
        vocabulary = tokenizer.get_vocab()
    except (AttributeError, NotImplementedError) as error:
        raise TokenizerAnalysisError(
            "This tokenizer does not expose a vocabulary for analysis"
        ) from error
    if not isinstance(vocabulary, dict) or not vocabulary:
        raise TokenizerAnalysisError("Tokenizer returned an empty vocabulary")
    return {str(token): int(token_id) for token, token_id in vocabulary.items()}


def _raw_token_by_id(vocabulary: dict[str, int]) -> dict[int, str]:
    by_id = {}
    for token, token_id in vocabulary.items():
        by_id.setdefault(token_id, token)
    return by_id


def _encode(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    return [int(token_id) for token_id in encoded]


def _decoded_piece(tokenizer: Any, token_id: int) -> str:
    try:
        return str(
            tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
    except TypeError:
        return str(tokenizer.decode([token_id], skip_special_tokens=False))


def analyze_tokenizer(
    spec: TokenizerSpec,
    tokenizer: Any,
    text: str,
    *,
    detail_limit: int = 400,
) -> TokenizerAnalysis:
    """Measure vocabulary coverage and fragmentation on the same Nepali sample."""

    if not text.strip():
        raise TokenizerAnalysisError("Provide non-empty text for tokenizer analysis")
    if detail_limit <= 0:
        raise TokenizerAnalysisError("detail_limit must be positive")

    vocabulary = _vocabulary(tokenizer)
    by_id = _raw_token_by_id(vocabulary)
    special_ids = {int(value) for value in (getattr(tokenizer, "all_special_ids", []) or [])}
    lexical_vocabulary = {
        token_id: raw_token
        for token_id, raw_token in by_id.items()
        if token_id not in special_ids
    }
    devanagari_vocabulary = [
        raw_token
        for raw_token in lexical_vocabulary.values()
        if contains_devanagari(raw_token)
    ]
    vocabulary_count = len(lexical_vocabulary)

    token_ids = _encode(tokenizer, text)
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    pieces = []
    devanagari_sample_tokens = 0
    unknown_tokens = 0
    for position, token_id in enumerate(token_ids):
        raw_token = by_id.get(token_id, str(tokenizer.convert_ids_to_tokens(token_id)))
        decoded = _decoded_piece(tokenizer, token_id)
        has_devanagari = contains_devanagari(raw_token) or contains_devanagari(decoded)
        devanagari_sample_tokens += has_devanagari
        is_unknown = unknown_id is not None and token_id == int(unknown_id)
        unknown_tokens += is_unknown
        if position < detail_limit:
            pieces.append(
                TokenPiece(
                    position=position,
                    token_id=token_id,
                    raw_token=raw_token,
                    decoded_piece=decoded,
                    contains_devanagari=has_devanagari,
                    is_unknown=is_unknown,
                )
            )

    words = nepali_words(text)
    unique_words = sorted(set(words))
    single_token_words = sum(len(_encode(tokenizer, word)) == 1 for word in unique_words)
    token_count = len(token_ids)
    return TokenizerAnalysis(
        key=spec.key,
        tokenizer=spec.label,
        source=spec.source,
        vocabulary_tokens=vocabulary_count,
        devanagari_vocabulary_tokens=len(devanagari_vocabulary),
        devanagari_vocabulary_percent=round(
            100.0 * len(devanagari_vocabulary) / max(1, vocabulary_count), 4
        ),
        sample_characters=len(text),
        sample_tokens=token_count,
        sample_nepali_token_percent=round(
            100.0 * devanagari_sample_tokens / max(1, token_count), 4
        ),
        nepali_word_occurrences=len(words),
        unique_nepali_words=len(unique_words),
        tokens_per_nepali_word=round(token_count / max(1, len(words)), 4),
        single_token_nepali_word_percent=round(
            100.0 * single_token_words / max(1, len(unique_words)), 4
        ),
        unknown_token_percent=round(100.0 * unknown_tokens / max(1, token_count), 4),
        devanagari_vocabulary_examples=tuple(
            sorted(set(devanagari_vocabulary), key=lambda item: (len(item), item))[:30]
        ),
        pieces=tuple(pieces),
    )


def analyses_csv(analyses: Iterable[TokenizerAnalysis]) -> str:
    import csv
    import io

    output = io.StringIO()
    rows = [analysis.summary() for analysis in analyses]
    if not rows:
        return ""
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()
