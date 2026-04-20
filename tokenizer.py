from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


@dataclass
class SimpleTokenizer:
    stoi: dict[str, int]
    itos: list[str]

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

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = True) -> list[int]:
        tokens = text.strip().split()
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(self.stoi.get(tok, self.unk_id) for tok in tokens)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> list[str]:
        return [self.itos[i] for i in ids]


def build_tokenizer(texts: list[str], max_vocab_size: int = 20000, min_freq: int = 2) -> SimpleTokenizer:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(text.strip().split())

    vocab = SPECIAL_TOKENS.copy()
    for token, freq in counter.most_common(max_vocab_size - len(SPECIAL_TOKENS)):
        if freq < min_freq:
            break
        vocab.append(token)

    stoi = {token: idx for idx, token in enumerate(vocab)}
    return SimpleTokenizer(stoi=stoi, itos=vocab)
