from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from config import ModelConfig, TrainConfig
from data import get_batch, load_wikitext2
from model import TinyTransformerLM
from utils import save_checkpoint, save_json, set_seed, save_loss_curve


@torch.no_grad()
def estimate_loss(
    model: TinyTransformerLM,
    data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: str,
    eval_iters: int = 20,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data, batch_size=batch_size, seq_len=seq_len, device=device)
        out = model(x, targets=y)
        losses.append(out["loss"].item())
    model.train()
    return sum(losses) / len(losses)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny transformer LM on WikiText-2.")
    parser.add_argument("--attention", type=str, default="softmax", choices=["softmax", "cosine", "linear"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-vocab-size", type=int, default=100000000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print("Loading WikiText-2...")
    encoded = load_wikitext2(max_vocab_size=args.max_vocab_size, min_freq=args.min_freq)

    model_cfg = ModelConfig(
        vocab_size=encoded.tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
        attention_type=args.attention,
    )
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        eval_every=args.eval_every,
        max_steps=args.max_steps,
        device=args.device,
        seed=args.seed,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_run_name = (
        f"{timestamp}_"
        f"{args.attention}_"
        f"L{args.n_layers}_H{args.n_heads}_"
        f"D{args.d_model}_FF{args.d_ff}_"
        f"S{args.seq_len}_BS{args.batch_size}_LR{args.lr}"
    )
    run_name = args.run_name or default_run_name

    run_dir = Path(args.runs_dir) / run_name
    ckpt_dir = run_dir / "checkpoints"
    artifact_dir = run_dir / "artifacts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    config_path = artifact_dir / "config.json"
    save_json(
        str(config_path),
        {
            "run_name": run_name,
            "args": vars(args),
            "model_config": model_cfg.__dict__,
            "train_config": train_cfg.__dict__,
        },
    )

    print(f"Run directory: {run_dir}")

    model = TinyTransformerLM(model_cfg).to(train_cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)

    tokens_per_epoch = len(encoded.train_ids)
    steps_per_epoch = max(1, tokens_per_epoch // (train_cfg.batch_size * model_cfg.max_seq_len))

    global_step = 0
    best_val = float("inf")
    history = []

    train_steps = []
    train_losses = []
    val_steps = []
    val_losses = []

    print(f"Training on device={train_cfg.device} with attention={args.attention}")

    for epoch in range(train_cfg.epochs):
        progress = tqdm(range(steps_per_epoch), desc=f"epoch {epoch + 1}/{train_cfg.epochs}")
        for _ in progress:
            x, y = get_batch(
                encoded.train_ids,
                batch_size=train_cfg.batch_size,
                seq_len=model_cfg.max_seq_len,
                device=train_cfg.device,
            )
            out = model(x, targets=y)
            loss = out["loss"]
            if loss is None:
                raise RuntimeError("Loss should not be None during training.")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()

            global_step += 1
            progress.set_postfix(train_loss=f"{loss.item():.4f}")
            train_steps.append(global_step)
            train_losses.append(float(loss.item()))

            if global_step % train_cfg.eval_every == 0:
                val_loss = estimate_loss(
                    model,
                    encoded.valid_ids,
                    batch_size=train_cfg.batch_size,
                    seq_len=model_cfg.max_seq_len,
                    device=train_cfg.device,
                )

                train_loss_value = float(loss.item())
                record = {
                    "step": global_step,
                    "train_loss": round(train_loss_value, 6),
                    "val_loss": round(val_loss, 6),
                }
                history.append(record)

                val_steps.append(global_step)
                val_losses.append(float(val_loss))
                print(f"\nstep={global_step} train_loss={loss.item():.4f} val_loss={val_loss:.4f}")

                if val_loss < best_val:
                    best_val = val_loss
                    ckpt_path = ckpt_dir / "best.pt"
                    save_checkpoint(
                        str(ckpt_path),
                        {
                            "model_state_dict": model.state_dict(),
                            "model_config": model_cfg.__dict__,
                            "tokenizer_stoi": encoded.tokenizer.stoi,
                            "tokenizer_itos": encoded.tokenizer.itos,
                            "best_val_loss": best_val,
                            "run_name": run_name,
                        },
                    )
                    print(f"Saved best checkpoint to {ckpt_path}")

            if train_cfg.max_steps is not None and global_step >= train_cfg.max_steps:
                break

        if train_cfg.max_steps is not None and global_step >= train_cfg.max_steps:
            break

    metrics_path = artifact_dir / "train_metrics.json"
    curve_path = artifact_dir / "loss_curve.png"

    save_json(
        str(metrics_path),
        {
            "run_name": run_name,
            "history": history,
            "best_val_loss": best_val,
            "train_steps": train_steps,
            "train_losses": train_losses,
            "val_steps": val_steps,
            "val_losses": val_losses,
        },
    )

    save_loss_curve(
        train_steps=train_steps,
        train_losses=train_losses,
        val_steps=val_steps,
        val_losses=val_losses,
        save_path=str(curve_path),
        title=f"Training Curve ({args.attention})",
    )

    print(f"Saved training metrics to {metrics_path}")
    print(f"Saved loss curve to {curve_path}")
    print(f"Best checkpoint saved under {ckpt_dir}")

if __name__ == "__main__":
    main()