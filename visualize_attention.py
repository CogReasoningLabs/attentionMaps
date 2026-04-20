from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from config import ModelConfig
from model import TinyTransformerLM
from tokenizer import SimpleTokenizer
from utils import ensure_dir, make_run_dir, plot_attention, save_all_attention_maps


def load_checkpoint(path: str, device: str):
    ckpt = torch.load(path, map_location=device)
    cfg = ModelConfig(**ckpt["model_config"])
    tokenizer = SimpleTokenizer(stoi=ckpt["tokenizer_stoi"], itos=ckpt["tokenizer_itos"])
    model = TinyTransformerLM(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, tokenizer, cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize attention maps from a trained checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--cmap", type=str, default="magma")

    # optional single-map mode
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--head", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer, cfg = load_checkpoint(args.checkpoint, args.device)

    ids = tokenizer.encode(args.text, add_bos=True, add_eos=True)
    if len(ids) > cfg.max_seq_len:
        raise ValueError(
            f"Input has {len(ids)} tokens but model max_seq_len is {cfg.max_seq_len}. Shorten the text or retrain with larger seq_len."
        )

    x = torch.tensor(ids, dtype=torch.long, device=args.device).unsqueeze(0)

    with torch.inference_mode():
        out = model(x, return_attention=True)

    attentions = out["attentions"]
    if attentions is None:
        raise RuntimeError("No attention tensors were returned.")

    tokens = tokenizer.decode(ids)

    if args.layer is not None and args.head is not None:
        if not (0 <= args.layer < len(attentions)):
            raise ValueError(f"Requested layer {args.layer}, but model has {len(attentions)} layers.")

        attn = attentions[args.layer]
        if attn is None:
            raise RuntimeError("This attention variant did not return attention weights.")

        if not (0 <= args.head < attn.shape[1]):
            raise ValueError(f"Requested head {args.head}, but layer has {attn.shape[1]} heads.")

        matrix = attn[0, args.head, :, :]
        ensure_dir("artifacts")
        output_path = args.output or str(Path("artifacts") / f"attention_layer{args.layer}_head{args.head}.png")
        title = f"Attention map | variant={cfg.attention_type} | layer={args.layer} | head={args.head}"
        plot_attention(tokens, matrix, output_path, title, cmap=args.cmap)
        print(f"Saved attention visualization to {output_path}")
    else:
        run_dir = make_run_dir(base_dir=args.runs_dir, run_name=args.run_name)
        save_all_attention_maps(
            tokens=tokens,
            attentions=attentions,
            run_dir=run_dir,
            cmap=args.cmap,
            variant_name=cfg.attention_type,
        )
        print(f"Saved all attention visualizations to {run_dir}")


if __name__ == "__main__":
    main()