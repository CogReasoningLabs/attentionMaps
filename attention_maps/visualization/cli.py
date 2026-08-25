from __future__ import annotations

import argparse
from pathlib import Path

import torch

from attention_maps.common.artifacts import (
    find_run_dir_from_checkpoint,
    make_attention_dir,
    save_json,
)
from attention_maps.config import ModelConfig
from attention_maps.tokenization.models import tokenizer_from_checkpoint
from attention_maps.training.model import TinyTransformerLM
from attention_maps.visualization.attention import (
    compute_global_attention_scale,
    format_token_labels,
    plot_attention,
    save_all_attention_maps,
)


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
    return model, tokenizer, cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize attention maps from a trained checkpoint."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this inspection under the trained run's attention_maps/",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Explicit directory for all-map output instead of the run directory",
    )
    parser.add_argument("--cmap", type=str, default="magma")
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument(
        "--max-display-tokens",
        type=int,
        default=64,
        help="Reject unreadably long heatmaps; use 0 to disable the limit",
    )

    # optional single-map mode
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--head", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.layer is None) != (args.head is None):
        raise ValueError("--layer and --head must be provided together")
    if args.output_dir and args.output:
        raise ValueError("Use either --output-dir or --output, not both")
    if not 0.0 < args.percentile <= 100.0:
        raise ValueError("--percentile must be in (0, 100]")
    model, tokenizer, cfg = load_checkpoint(args.checkpoint, args.device)

    ids = tokenizer.encode(args.text, add_bos=True, add_eos=True)
    if len(ids) > cfg.max_seq_len:
        raise ValueError(
            f"Input has {len(ids)} tokens but model max_seq_len is "
            f"{cfg.max_seq_len}. Shorten the text or retrain with larger seq_len."
        )
    if args.max_display_tokens > 0 and len(ids) > args.max_display_tokens:
        raise ValueError(
            f"Input produces {len(ids)} labels, exceeding --max-display-tokens="
            f"{args.max_display_tokens}. Use shorter text or explicitly raise the limit."
        )

    x = torch.tensor(ids, dtype=torch.long, device=args.device).unsqueeze(0)

    with torch.inference_mode():
        out = model(x, return_attention=True)

    attentions = out["attentions"]
    if attentions is None:
        raise RuntimeError("No attention tensors were returned.")

    tokens = format_token_labels(
        tokenizer.decode(ids),
        tokenizer.special_tokens,
    )
    common_record = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "text": args.text,
        "input_ids": ids,
        "token_labels": tokens,
        "attention_type": cfg.attention_type,
        "layers": len(attentions),
        "heads_per_layer": attentions[0].shape[1] if attentions else 0,
        "color_scale_percentile": args.percentile,
    }

    if args.layer is not None and args.head is not None:
        if not (0 <= args.layer < len(attentions)):
            raise ValueError(
                f"Requested layer {args.layer}, but model has {len(attentions)} layers."
            )

        attn = attentions[args.layer]
        if attn is None:
            raise RuntimeError("This attention variant did not return attention weights.")

        if not (0 <= args.head < attn.shape[1]):
            raise ValueError(
                f"Requested head {args.head}, but layer has {attn.shape[1]} heads."
            )

        matrix = attn[0, args.head, :, :]
        if args.output:
            output_path = Path(args.output)
        else:
            run_dir = find_run_dir_from_checkpoint(args.checkpoint)
            inspection_dir = make_attention_dir(run_dir, args.run_name)
            output_path = inspection_dir / (
                f"layer_{args.layer:02d}_head_{args.head:02d}.png"
            )
        title = (
            f"Attention map | variant={cfg.attention_type} | "
            f"layer={args.layer} | head={args.head}"
        )
        _, vmax = compute_global_attention_scale(
            [matrix],
            percentile=args.percentile,
        )
        plot_attention(
            tokens,
            matrix,
            output_path,
            title,
            cmap=args.cmap,
            vmax=vmax,
        )
        metadata_path = output_path.with_suffix(".json")
        save_json(
            str(metadata_path),
            {
                **common_record,
                "layer": args.layer,
                "head": args.head,
                "image": str(output_path.resolve()),
            },
        )
        print(f"Saved attention visualization to {output_path}")
        print(f"Saved attention metadata to {metadata_path}")
    else:
        if args.output_dir:
            inspection_dir = Path(args.output_dir).resolve()
            inspection_dir.mkdir(parents=True, exist_ok=True)
        else:
            run_dir = find_run_dir_from_checkpoint(args.checkpoint)
            inspection_dir = make_attention_dir(run_dir, args.run_name)
        paths = save_all_attention_maps(
            tokens=tokens,
            attentions=attentions,
            output_dir=inspection_dir,
            cmap=args.cmap,
            variant_name=cfg.attention_type,
            percentile=args.percentile,
        )
        save_json(
            str(inspection_dir / "attention_metadata.json"),
            {
                **common_record,
                "images": [str(path.resolve()) for path in paths],
            },
        )
        print(f"Saved {len(paths)} attention visualizations to {inspection_dir}")


if __name__ == "__main__":
    main()
