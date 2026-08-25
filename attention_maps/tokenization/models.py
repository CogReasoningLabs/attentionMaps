from __future__ import annotations

import re
import unicodedata
from io import BytesIO
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Any

from attention_maps.config import TokenizerConfig


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

# ── Punctuation-aware tokenization regex ─────────────────────────────────────
# Splits on whitespace but also splits punctuation OFF word boundaries,
# so "hello." → ["hello", "."] instead of keeping "hello." as one OOV token.
_TOKENIZE_RE = re.compile(
    r"""
    \d+(?:[.,]\d+)*     # numbers with optional decimal/comma  e.g. 3.14, 1,000
    | [a-zA-Z]+(?:'[a-zA-Z]+)*   # words with optional contractions  e.g. don't
    | [^\s]             # any remaining single non-whitespace char (punct, symbols)
    """,
    re.VERBOSE,
)


def _normalize(text: str) -> str:
    """
    Light normalization that improves token coverage without being destructive:
      - Unicode NFC normalization  (é stays é, not e + combining accent)
      - Collapse repeated whitespace
      - Lowercase  (optional — remove if your task is case-sensitive)
    """
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> list[str]:
    """Normalize then split into tokens."""
    return _TOKENIZE_RE.findall(_normalize(text))


@dataclass
class SimpleTokenizer:
    stoi: dict[str, int]
    itos: list[str]
    # character-level fallback vocab (built automatically if use_char_fallback=True)
    char_stoi: dict[str, int] = field(default_factory=dict)
    use_char_fallback: bool = False

    # ── special token ids ────────────────────────────────────────────────────
    @property
    def pad_id(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    # ── encoding ─────────────────────────────────────────────────────────────
    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = True,
    ) -> list[int]:
        tokens = _tokenize(text)
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for tok in tokens:
            if tok in self.stoi:
                ids.append(self.stoi[tok])
            elif self.use_char_fallback and self.char_stoi:
                # represent OOV token as its character sequence
                # each character maps to its own id (or unk if truly unseen)
                ids.extend(
                    self.char_stoi.get(ch, self.unk_id) for ch in tok
                )
            else:
                ids.append(self.unk_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def encode_batch(
        self,
        texts: list[str],
        add_bos: bool = False,
        add_eos: bool = True,
        pad: bool = True,
    ) -> tuple[list[list[int]], list[int]]:
        """
        Encode a batch of texts.
        Returns (padded_ids, lengths) where lengths are the unpadded sequence lengths.
        Useful for downstream collation without re-writing padding logic everywhere.
        """
        encoded = [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]
        lengths = [len(e) for e in encoded]
        if pad:
            max_len = max(lengths)
            encoded = [e + [self.pad_id] * (max_len - len(e)) for e in encoded]
        return encoded, lengths

    # ── decoding ─────────────────────────────────────────────────────────────
    def decode(self, ids: list[int]) -> list[str]:
        return [self.itos[i] for i in ids]

    def decode_to_string(
        self,
        ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode ids back to a human-readable string.
        Handles punctuation spacing so output reads naturally:
          ["hello", ",", "world", "!"] → "hello, world!"
        """
        skip = set(SPECIAL_TOKENS) if skip_special_tokens else set()
        tokens = [self.itos[i] for i in ids if self.itos[i] not in skip]
        return _detokenize(tokens)

    # ── introspection ────────────────────────────────────────────────────────
    def unk_rate(self, texts: list[str]) -> float:
        """Fraction of tokens that map to <unk>. Useful for vocab diagnostics."""
        total = unk = 0
        for text in texts:
            for tok in _tokenize(text):
                total += 1
                if tok not in self.stoi:
                    unk += 1
        return unk / total if total else 0.0

    def coverage(self, texts: list[str]) -> float:
        return 1.0 - self.unk_rate(texts)

    def to_state(self) -> dict[str, Any]:
        return {
            "backend": "legacy_word",
            "stoi": self.stoi,
            "itos": self.itos,
            "char_stoi": self.char_stoi,
            "use_char_fallback": self.use_char_fallback,
        }


class SentencePieceTokenizer:
    """Reversible Unicode subword tokenizer backed by a serialized model."""

    def __init__(self, model_proto: bytes):
        try:
            import sentencepiece as spm
        except ImportError as exc:
            raise RuntimeError(
                "SentencePiece is required; install dependencies with "
                "`pip install -r requirements.txt`."
            ) from exc
        self.model_proto = model_proto
        self.processor = spm.SentencePieceProcessor(model_proto=model_proto)

    @property
    def pad_id(self) -> int:
        return self.processor.pad_id()

    @property
    def unk_id(self) -> int:
        return self.processor.unk_id()

    @property
    def bos_id(self) -> int:
        return self.processor.bos_id()

    @property
    def eos_id(self) -> int:
        return self.processor.eos_id()

    @property
    def vocab_size(self) -> int:
        return self.processor.vocab_size()

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = True,
    ) -> list[int]:
        return list(
            self.processor.encode(
                text,
                out_type=int,
                add_bos=add_bos,
                add_eos=add_eos,
            )
        )

    def decode(self, ids: list[int]) -> list[str]:
        return [self.processor.id_to_piece(token_id) for token_id in ids]

    def decode_to_string(
        self,
        ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        if skip_special_tokens:
            special_ids = {self.pad_id, self.bos_id, self.eos_id}
            ids = [token_id for token_id in ids if token_id not in special_ids]
        return self.processor.decode(ids)

    def unk_rate(self, texts: Iterable[str]) -> float:
        total = unknown = 0
        for text in texts:
            ids = self.encode(text, add_eos=False)
            total += len(ids)
            unknown += sum(token_id == self.unk_id for token_id in ids)
        return unknown / total if total else 0.0

    def coverage(self, texts: Iterable[str]) -> float:
        return 1.0 - self.unk_rate(texts)

    def to_state(self) -> dict[str, Any]:
        return {
            "backend": "sentencepiece",
            "model_proto": self.model_proto,
        }


class HuggingFaceTokenizer:
    """Portable runtime adapter around any serialized Hugging Face tokenizer."""

    def __init__(
        self,
        tokenizer_json: str,
        special_tokens: dict[str, str | None],
    ):
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "tokenizers is required; install dependencies with "
                "`pip install -r requirements.txt`."
            ) from exc
        self.tokenizer_json = tokenizer_json
        self.special_tokens = dict(special_tokens)
        self.processor = Tokenizer.from_str(tokenizer_json)
        for role, token in self.special_tokens.items():
            if token is not None and self.processor.token_to_id(token) is None:
                raise ValueError(f"Tokenizer is missing its configured {role} token")

    @classmethod
    def from_directory(cls, directory: str | Path) -> HuggingFaceTokenizer:
        directory = Path(directory)
        tokenizer_path = directory / "tokenizer.json"
        config_path = directory / "tokenizer_config.json"
        runtime_config_path = directory / "attention_maps_tokenizer_config.json"
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"Missing tokenizer artifact: {tokenizer_path}")
        import json

        if runtime_config_path.is_file():
            runtime_metadata = json.loads(
                runtime_config_path.read_text(encoding="utf-8")
            )
            special_tokens = runtime_metadata["token_roles"]
        else:
            if not config_path.is_file():
                raise FileNotFoundError(f"Missing tokenizer metadata: {config_path}")
            metadata = json.loads(config_path.read_text(encoding="utf-8"))
            special_tokens = {
                role: metadata.get(f"{role}_token")
                for role in ("pad", "unk", "bos", "eos")
            }
        return cls(tokenizer_path.read_text(encoding="utf-8"), special_tokens)

    def _special_id(self, role: str) -> int:
        token = self.special_tokens.get(role)
        if token is None:
            raise ValueError(f"Tokenizer has no configured {role} token")
        token_id = self.processor.token_to_id(token)
        if token_id is None:
            raise RuntimeError(f"Tokenizer lost its {role} token")
        return token_id

    @property
    def pad_id(self) -> int:
        return self._special_id("pad")

    @property
    def unk_id(self) -> int:
        return self._special_id("unk")

    @property
    def bos_id(self) -> int:
        return self._special_id("bos")

    @property
    def eos_id(self) -> int:
        return self._special_id("eos")

    @property
    def vocab_size(self) -> int:
        return self.processor.get_vocab_size()

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = True,
    ) -> list[int]:
        ids = self.processor.encode(text, add_special_tokens=False).ids
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> list[str]:
        return [self.processor.id_to_token(token_id) for token_id in ids]

    def decode_to_string(
        self,
        ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        return self.processor.decode(ids, skip_special_tokens=skip_special_tokens)

    def unk_rate(self, texts: Iterable[str]) -> float:
        if self.special_tokens.get("unk") is None:
            return 0.0
        total = unknown = 0
        for text in texts:
            ids = self.encode(text, add_eos=False)
            total += len(ids)
            unknown += sum(token_id == self.unk_id for token_id in ids)
        return unknown / total if total else 0.0

    def coverage(self, texts: Iterable[str]) -> float:
        return 1.0 - self.unk_rate(texts)

    def to_state(self) -> dict[str, Any]:
        return {
            "backend": "huggingface_tokenizer",
            "tokenizer_json": self.tokenizer_json,
            "special_tokens": self.special_tokens,
        }


# Backward-compatible name for checkpoints and imports created before the
# adapter was generalized beyond BPE models.
HuggingFaceBPETokenizer = HuggingFaceTokenizer


def _detokenize(tokens: list[str]) -> str:
    """
    Rejoin tokens into natural text.
    Rules:
      - No space before punctuation: , . ! ? ; : ) ] }
      - No space after opening brackets: ( [ {
      - Contractions stay glued: don ' t → don't  (already split correctly)
    """
    NO_SPACE_BEFORE = set(".,!?;:)]}%")
    NO_SPACE_AFTER  = set("([{")
    APOSTROPHE_AFTER = {"'"}

    out = []
    for i, tok in enumerate(tokens):
        if i == 0:
            out.append(tok)
        elif tok in NO_SPACE_BEFORE:
            out.append(tok)
        elif tok in APOSTROPHE_AFTER and out:
            out.append(tok)
        elif out and out[-1] in NO_SPACE_AFTER:
            out.append(tok)
        elif out and out[-1] in APOSTROPHE_AFTER:
            out.append(tok)
        else:
            out.append(" " + tok)
    return "".join(out)


# ── builder ──────────────────────────────────────────────────────────────────

def build_tokenizer(
    texts: list[str],
    max_vocab_size: int = 20_000,
    min_freq: int = 2,
    use_char_fallback: bool = False,
) -> SimpleTokenizer:
    """
    Build a word-level tokenizer from a list of raw text strings.

    Improvements over the original:
      1. Punctuation-aware splitting  — "hello." no longer becomes one OOV token
      2. Lowercasing + Unicode NFC    — reduces spurious duplicates in the vocab
      3. Frequency-sorted vocab       — most common tokens get lowest ids
         (very slight embedding lookup cache benefit; also aids debugging)
      4. Optional char-level fallback — OOV words are split into known characters
         instead of blindly emitting <unk>, preserving some signal for rare words

    Args:
        texts:             Raw training corpus strings.
        max_vocab_size:    Hard cap on vocabulary size (includes special tokens).
        min_freq:          Tokens seen fewer times than this are excluded.
        use_char_fallback: If True, also build a character vocab for OOV fallback.
    """
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(_tokenize(text))

    vocab = SPECIAL_TOKENS.copy()
    for token, freq in counter.most_common(max_vocab_size - len(SPECIAL_TOKENS)):
        if freq < min_freq:
            break
        vocab.append(token)

    stoi = {token: idx for idx, token in enumerate(vocab)}

    # ── optional character fallback vocab ────────────────────────────────────
    char_stoi: dict[str, int] = {}
    if use_char_fallback:
        char_counter: Counter[str] = Counter()
        for tok in counter:
            char_counter.update(tok)
        # reuse the same stoi so char ids live in the same embedding space
        # only add characters not already in the word vocab
        for ch, _ in char_counter.most_common():
            if ch not in stoi:
                stoi[ch] = len(stoi)
                vocab.append(ch)
        char_stoi = {ch: stoi[ch] for ch in char_counter if ch in stoi}

    return SimpleTokenizer(
        stoi=stoi,
        itos=vocab,
        char_stoi=char_stoi,
        use_char_fallback=use_char_fallback,
    )


def train_sentencepiece_tokenizer(
    sentences: Iterable[str],
    config: TokenizerConfig,
) -> SentencePieceTokenizer:
    """Train a self-contained SentencePiece model from an iterable of text."""

    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError(
            "SentencePiece is required; install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    model_buffer = BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=sentences,
        model_writer=model_buffer,
        vocab_size=config.vocab_size,
        model_type=config.model_type,
        normalization_rule_name=config.normalization,
        character_coverage=config.character_coverage,
        byte_fallback=config.byte_fallback,
        hard_vocab_limit=config.hard_vocab_limit,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece=PAD_TOKEN,
        unk_piece=UNK_TOKEN,
        bos_piece=BOS_TOKEN,
        eos_piece=EOS_TOKEN,
        max_sentence_length=16_384,
        shuffle_input_sentence=False,
        minloglevel=1,
    )
    return SentencePieceTokenizer(model_buffer.getvalue())


def tokenizer_from_state(
    state: dict[str, Any],
) -> SimpleTokenizer | SentencePieceTokenizer | HuggingFaceTokenizer:
    backend = state.get("backend")
    if backend == "sentencepiece":
        return SentencePieceTokenizer(model_proto=state["model_proto"])
    if backend == "legacy_word":
        return SimpleTokenizer(
            stoi=state["stoi"],
            itos=state["itos"],
            char_stoi=state.get("char_stoi", {}),
            use_char_fallback=state.get("use_char_fallback", False),
        )
    if backend in {"huggingface_bpe", "huggingface_tokenizer"}:
        return HuggingFaceTokenizer(
            tokenizer_json=state["tokenizer_json"],
            special_tokens=state["special_tokens"],
        )
    raise ValueError(f"Unsupported tokenizer backend in checkpoint: {backend!r}")


def tokenizer_from_checkpoint(checkpoint: dict[str, Any]):
    """Load current tokenizer state or migrate a legacy checkpoint in memory."""

    if "tokenizer" in checkpoint:
        return tokenizer_from_state(checkpoint["tokenizer"])
    if "tokenizer_stoi" in checkpoint and "tokenizer_itos" in checkpoint:
        return SimpleTokenizer(
            stoi=checkpoint["tokenizer_stoi"],
            itos=checkpoint["tokenizer_itos"],
        )
    raise ValueError("Checkpoint does not contain tokenizer state")
