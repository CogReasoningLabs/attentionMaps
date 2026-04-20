from __future__ import annotations

from dataclasses import dataclass

import torch
from datasets import load_dataset

from tokenizer import SimpleTokenizer, build_tokenizer


@dataclass
class EncodedDataset:
    train_ids: torch.Tensor
    valid_ids: torch.Tensor
    test_ids: torch.Tensor
    tokenizer: SimpleTokenizer


def _clean_texts(texts: list[str]) -> list[str]:
    cleaned = []
    for t in texts:
        t = t.strip()
        if t:
            cleaned.append(t)
    return cleaned


# def load_wikitext2(max_vocab_size: int = 20000, min_freq: int = 2) -> EncodedDataset:
#     # ds = load_dataset("wikitext", "wikitext-2-raw-v1")
#     ds= load_dataset("wikitext", "wikitext-103-raw-v1")
#     train_texts = _clean_texts(ds["train"]["text"])
#     valid_texts = _clean_texts(ds["validation"]["text"])
#     test_texts = _clean_texts(ds["test"]["text"])
#     tokenizer = build_tokenizer(train_texts, max_vocab_size=max_vocab_size, min_freq=min_freq)

#     def encode_split(texts: list[str]) -> torch.Tensor:
#         ids: list[int] = []
#         for text in texts:
#             ids.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
#         return torch.tensor(ids, dtype=torch.long)

#     return EncodedDataset(
#         train_ids=encode_split(train_texts),
#         valid_ids=encode_split(valid_texts),
#         test_ids=encode_split(test_texts),
#         tokenizer=tokenizer,
#     )

def load_wikitext2(max_vocab_size: int = 20000, min_freq: int = 2) -> EncodedDataset:
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    train_texts = _clean_texts(ds["train"]["text"])
    valid_texts = _clean_texts(ds["validation"]["text"])
    test_texts = _clean_texts(ds["test"]["text"])
    tokenizer = build_tokenizer(train_texts, max_vocab_size=max_vocab_size, min_freq=min_freq)

    def encode_split(texts: list[str]) -> torch.Tensor:
        ids: list[int] = []
        for text in texts:
            ids.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
        t = torch.tensor(ids, dtype=torch.long)
        return t.pin_memory() if torch.cuda.is_available() else t

    return EncodedDataset(
        train_ids=encode_split(train_texts),
        valid_ids=encode_split(valid_texts),
        test_ids=encode_split(test_texts),
        tokenizer=tokenizer,
    )

def get_batch(data: torch.Tensor, batch_size: int, seq_len: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(data) - seq_len - 1
    if max_start <= 0:
        raise ValueError("Dataset too small for the configured sequence length.")

    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[s : s + seq_len] for s in starts])
    y = torch.stack([data[s + 1 : s + seq_len + 1] for s in starts])
    return x.to(device), y.to(device)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect WikiText-2 loading and sampled batches.")
    parser.add_argument("--max-vocab-size", type=int, default=20000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--show-train-text", action="store_true")
    args = parser.parse_args()

    print("Loading WikiText-2...")
    encoded = load_wikitext2(
        max_vocab_size=args.max_vocab_size,
        min_freq=args.min_freq,
    )

    print("\n=== Dataset Summary ===")
    print(f"Vocab size      : {encoded.tokenizer.vocab_size}")
    print(f"Train tokens    : {len(encoded.train_ids)}")
    print(f"Valid tokens    : {len(encoded.valid_ids)}")
    print(f"Test tokens     : {len(encoded.test_ids)}")

    print("\n=== First 30 Train Token IDs ===")
    print(encoded.train_ids[:30].tolist())

    print("\n=== Decoded First 30 Train Tokens ===")
    print(encoded.tokenizer.decode(encoded.train_ids[:30].tolist()))

    x, y = get_batch(
        encoded.train_ids,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=args.device,
    )

    print("\n=== Sample Batch Shapes ===")
    print(f"x shape: {tuple(x.shape)}")
    print(f"y shape: {tuple(y.shape)}")

    for i in range(args.batch_size):
        x_ids = x[i].tolist()
        y_ids = y[i].tolist()

        print(f"\n--- Sample {i} ---")
        print("Input IDs: ", x_ids)
        print("Target IDs:", y_ids)

        print("Input Tokens:")
        print(encoded.tokenizer.decode(x_ids))

        print("Target Tokens:")
        print(encoded.tokenizer.decode(y_ids))

        print("Input -> Target pairs:")
        for xin, yin in zip(encoded.tokenizer.decode(x_ids), encoded.tokenizer.decode(y_ids)):
            print(f"  {xin:>12}  ->  {yin}")

    if args.show_train_text:
        print("\n=== Decoded Longer Train Example ===")
        sample_ids = encoded.train_ids[:100].tolist()
        print(encoded.tokenizer.decode(sample_ids))