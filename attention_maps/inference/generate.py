from __future__ import annotations

import argparse
from pathlib import Path

import torch

from attention_maps.common.artifacts import (
    find_run_dir_from_checkpoint,
    make_generation_dir,
    save_json,
    set_seed,
)
from attention_maps.config import ModelConfig
from attention_maps.tokenization.models import tokenizer_from_checkpoint
from attention_maps.training.model import TinyTransformerLM
from attention_maps.visualization.attention import save_all_attention_maps
from attention_maps.visualization.attention import format_token_labels


def load_checkpoint(path: str, device: str):
    ckpt = torch.load(path, map_location=device)
    cfg = ModelConfig(**ckpt["model_config"])
    tokenizer = tokenizer_from_checkpoint(ckpt)

    model = TinyTransformerLM(cfg).to(device)
    state = ckpt["model_state_dict"]
    if state and all(key.startswith("_orig_mod.") for key in state):
        state = {key.removeprefix("_orig_mod."): value for key, value in state.items()}
    model.load_state_dict(state)
    model.eval()

    return model, tokenizer, cfg, ckpt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained tiny transformer LM.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-bos",
        action="store_true",
        help="Do not prepend the tokenizer's configured BOS token",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--save-attention", action="store_true")
    parser.add_argument("--cmap", type=str, default="magma")
    parser.add_argument("--gen-name", type=str, default=None)

    return parser.parse_args()


def final_attention_context_ids(
    output_ids: list[int],
    context_length: int,
) -> list[int]:
    """Return the exact pre-sampling context represented by final-step attention."""

    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if len(output_ids) < context_length + 1:
        raise ValueError(
            "Generated output must include the context and its subsequently "
            "sampled token"
        )
    return output_ids[-(context_length + 1) : -1]


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.top_k is not None and args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.top_p is not None and not 0.0 < args.top_p <= 1.0:
        raise ValueError("--top-p must be in (0, 1]")
    set_seed(args.seed)

    model, tokenizer, cfg, ckpt = load_checkpoint(args.checkpoint, args.device)
    run_dir = find_run_dir_from_checkpoint(args.checkpoint)
    gen_dir = make_generation_dir(run_dir, args.gen_name)

    input_ids = tokenizer.encode(
        args.prompt,
        add_bos=not args.no_bos,
        add_eos=False,
    )
    if len(input_ids) > cfg.max_seq_len:
        print(
            f"Warning: prompt has {len(input_ids)} tokens; generation uses only "
            f"the latest {cfg.max_seq_len} tokens as model context."
        )
    x = torch.tensor([input_ids], dtype=torch.long, device=args.device)

    use_amp = args.device.startswith("cuda")
    with torch.inference_mode(), torch.autocast(
        device_type="cuda" if use_amp else "cpu",
        dtype=torch.float16 if use_amp else torch.bfloat16,
        enabled=use_amp,
    ):
        out = model.generate(
            idx=x,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=args.do_sample,
            eos_token_id=tokenizer.eos_id,
            return_attention=args.save_attention,
        )

    output_ids = out["sequences"][0].tolist()
    output_tokens = tokenizer.decode(output_ids)
    output_token_labels = format_token_labels(
        output_tokens,
        tokenizer.special_tokens,
    )
    completion_ids = output_ids[len(input_ids) :]
    generated_text = tokenizer.decode_to_string(
        output_ids,
        skip_special_tokens=True,
    )
    completion_text = tokenizer.decode_to_string(
        completion_ids,
        skip_special_tokens=True,
    )

    record = {
        "prompt": args.prompt,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "attention_type": cfg.attention_type,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "do_sample": args.do_sample,
        "seed": args.seed,
        "prompt_ids": input_ids,
        "completion_ids": completion_ids,
        "output_ids": output_ids,
        "output_tokens": output_tokens,
        "output_token_labels": output_token_labels,
        "completion_text": completion_text,
        "generated_text": generated_text,
    }

    save_json(str(gen_dir / "generation.json"), record)
    (gen_dir / "generation.txt").write_text(generated_text, encoding="utf-8")

    print("\n=== Generated Text ===")
    print(generated_text)
    print("\n=== Completion Only ===")
    print(completion_text)
    print(f"\nSaved generation record to: {gen_dir}")

    if args.save_attention:
        last_step_attentions = out["last_step_attentions"]
        if last_step_attentions:
            attn_dir = gen_dir / "attention_maps"
            attn_dir.mkdir(parents=True, exist_ok=True)

            # The final generated token was sampled after this attention pass,
            # so label the matrix with the exact preceding model context.
            context_length = last_step_attentions[0].shape[-1]
            context_ids = final_attention_context_ids(output_ids, context_length)
            final_tokens = format_token_labels(
                tokenizer.decode(context_ids),
                tokenizer.special_tokens,
            )

            save_all_attention_maps(
                tokens=final_tokens,
                attentions=last_step_attentions,
                output_dir=attn_dir,
                cmap=args.cmap,
                variant_name=cfg.attention_type,
            )
            print(f"Saved attention maps to: {attn_dir}")


if __name__ == "__main__":
    main()
