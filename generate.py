from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import ModelConfig
from model import TinyTransformerLM
from tokenizer import SimpleTokenizer
from utils import save_json, make_generation_dir
from visualize_attention import save_all_attention_maps


def load_checkpoint(path: str, device: str):
    ckpt = torch.load(path, map_location=device)
    cfg = ModelConfig(**ckpt["model_config"])
    tokenizer = SimpleTokenizer(stoi=ckpt["tokenizer_stoi"], itos=ckpt["tokenizer_itos"])

    model = TinyTransformerLM(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, tokenizer, cfg, ckpt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained tiny transformer LM.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--save-attention", action="store_true")
    parser.add_argument("--cmap", type=str, default="magma")
    parser.add_argument("--gen-name", type=str, default=None)

    return parser.parse_args()


def find_run_dir_from_checkpoint(checkpoint_path: str | Path) -> Path:
    checkpoint_path = Path(checkpoint_path).resolve()
    # expected: runs/<run_name>/checkpoints/best.pt
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    raise ValueError(
        f"Could not infer run dir from checkpoint path: {checkpoint_path}. "
        "Expected something like runs/<run_name>/checkpoints/best.pt"
    )


def clean_tokens(tokens: list[str]) -> list[str]:
    return [tok for tok in tokens if tok not in {"<bos>", "<eos>", "<pad>"}]


def main() -> None:
    args = parse_args()

    model, tokenizer, cfg, ckpt = load_checkpoint(args.checkpoint, args.device)
    run_dir = find_run_dir_from_checkpoint(args.checkpoint)
    gen_dir = make_generation_dir(run_dir, args.gen_name)

    input_ids = tokenizer.encode(args.prompt, add_bos=True, add_eos=False)
    x = torch.tensor([input_ids], dtype=torch.long, device=args.device)

    with torch.inference_mode():
        out = model.generate(
            idx=x,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            do_sample=args.do_sample,
            eos_token_id=tokenizer.eos_id,
            return_attention=args.save_attention,
        )

    output_ids = out["sequences"][0].tolist()
    output_tokens = tokenizer.decode(output_ids)
    cleaned_output_tokens = clean_tokens(output_tokens)
    generated_text = " ".join(cleaned_output_tokens)

    record = {
        "prompt": args.prompt,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "attention_type": cfg.attention_type,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "do_sample": args.do_sample,
        "output_ids": output_ids,
        "output_tokens": output_tokens,
        "generated_text": generated_text,
    }

    save_json(str(gen_dir / "generation.json"), record)
    (gen_dir / "generation.txt").write_text(generated_text, encoding="utf-8")

    print("\n=== Generated Text ===")
    print(generated_text)
    print(f"\nSaved generation record to: {gen_dir}")

    if args.save_attention:
        step_attentions = out["step_attentions"]
        if step_attentions:
            last_step_attentions = step_attentions[-1]
            attn_dir = gen_dir / "attention_maps"
            attn_dir.mkdir(parents=True, exist_ok=True)

            # visualize attention from the final decoding step context
            final_tokens = tokenizer.decode(output_ids[-cfg.max_seq_len :])

            save_all_attention_maps(
                tokens=final_tokens,
                attentions=last_step_attentions,
                run_dir=attn_dir,
                cmap=args.cmap,
                variant_name=cfg.attention_type,
            )
            print(f"Saved attention maps to: {attn_dir}")


if __name__ == "__main__":
    main()