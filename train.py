from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR

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


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Linear warmup then cosine decay to min_lr_ratio × peak_lr.
    Much more stable than jumping straight to peak LR with AdamW.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item())
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
 
    return LambdaLR(optimizer, lr_lambda)
 

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny transformer LM on wikitext-103-raw-v1.")
    parser.add_argument("--attention", type=str, default="softmax", choices=["softmax", "cosine", "linear"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps(effective batch = batch size * grad_Accum)")

    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=200, help="Number of steps to warm up the learning rate")
    parser.add_argument("--min-lr-ratio",    type=float, default=0.1, help="Minimum LR as fraction of peak (cosine decay floor)")
    parser.add_argument("--moe-aux-weight",  type=float, default=0.01, help="Weight for the MoE load-balancing auxiliary loss")

    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-vocab-size", type=int, default=50000)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to latest.pt checkpoint to resume from")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
 
    # ── data ──────────────────────────────────────────────────────────────────
    print("Loading WikiText-103...")
    encoded = load_wikitext2(max_vocab_size=args.max_vocab_size, min_freq=args.min_freq)
 
    # ── configs ───────────────────────────────────────────────────────────────
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
 
    # ── run directory ─────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = (
        f"{timestamp}_{args.attention}_"
        f"L{args.n_layers}_H{args.n_heads}_"
        f"D{args.d_model}_FF{args.d_ff}_"
        f"S{args.seq_len}_BS{args.batch_size}_"
        f"LR{args.lr}_DR{args.dropout}_"
        f"E{args.epochs}_EVAL{args.eval_every}_"
        f"V{args.max_vocab_size}_MF{args.min_freq}_"
        f"SEED{args.seed}"
    )
    run_name = args.run_name or default_name
    run_dir = Path(args.runs_dir) / run_name
    ckpt_dir = run_dir / "checkpoints"
    art_dir = run_dir / "artifacts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)
 
    save_json(str(art_dir / "config.json"), {
        "run_name": run_name,
        "args": vars(args),
        "model_config": model_cfg.__dict__,
        "train_config": train_cfg.__dict__,
    })
    print(f"Run directory: {run_dir}")
 
    # ── model + optimiser ─────────────────────────────────────────────────────
    model = TinyTransformerLM(model_cfg).to(train_cfg.device)
    model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=0.1, betas=(0.9, 0.95))
 
    # AMP: use float16 on CUDA, bfloat16 on MPS (if available), skip on CPU
    use_amp = train_cfg.device.startswith("cuda")
    amp_dtype = torch.float16 if use_amp else torch.bfloat16
    # device_type for autocast must be "cuda" or "cpu" only (not full device string)
    amp_device_type = "cuda" if train_cfg.device.startswith("cuda") else "cpu"
    scaler = torch.amp.GradScaler(enabled=use_amp)
 
    # ── LR schedule ───────────────────────────────────────────────────────────
    tokens_per_epoch = len(encoded.train_ids)
    steps_per_epoch = max(1, tokens_per_epoch // (train_cfg.batch_size * model_cfg.max_seq_len))
    total_steps = train_cfg.epochs * steps_per_epoch
    # account for gradient accumulation — optimizer steps are less frequent
    optim_steps_total = total_steps // args.grad_accum
 
    scheduler = build_lr_scheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=optim_steps_total,
        min_lr_ratio=args.min_lr_ratio,
    )
 
    # ── state ─────────────────────────────────────────────────────────────────
    global_step = 0
    optim_step = 0
    start_epoch = 0
    best_val = float("inf")
    history: list[dict] = []
    train_steps: list[int] = []
    train_losses: list[float] = []
    val_steps: list[int] = []
    val_losses: list[float] = []
 
    # ── resume ────────────────────────────────────────────────────────────────
    if args.resume:
        print(f"Resuming from {args.resume} ...")
        ckpt = torch.load(args.resume, map_location=train_cfg.device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        global_step = ckpt.get("global_step", 0)
        optim_step = ckpt.get("optim_step",  0)
        best_val = ckpt.get("best_val_loss", float("inf"))
        start_epoch = global_step // steps_per_epoch
        print(f"Resumed at step={global_step}, optim_step={optim_step}, best_val={best_val:.4f}")
 
    print(f"Training device={train_cfg.device}  attention={args.attention}  "
          f"grad_accum={args.grad_accum}  warmup={args.warmup_steps}")
    print(f"  steps/epoch={steps_per_epoch}  total={total_steps}  optim_steps={optim_steps_total}")
 
    # ── training loop ─────────────────────────────────────────────────────────
    try:
        for epoch in range(start_epoch, train_cfg.epochs):
            pbar = tqdm(range(steps_per_epoch), desc=f"epoch {epoch+1}/{train_cfg.epochs}")
 
            for micro_step in pbar:
                x, y = get_batch(
                    encoded.train_ids,
                    batch_size=train_cfg.batch_size,
                    seq_len=model_cfg.max_seq_len,
                    device=train_cfg.device,
                )
 
                # ── forward + backward ───────────────────────────────────────
                # Note: zero_grad BEFORE forward, not after backward (original bug)
                is_accum_step = (micro_step + 1) % args.grad_accum != 0
 
                with torch.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=use_amp):
                    out  = model(x, targets=y, moe_aux_weight=args.moe_aux_weight)
                    loss = out["loss"] / args.grad_accum   # scale for accumulation
 
                scaler.scale(loss).backward()
 
                if not is_accum_step:
                    # ── optimiser step ───────────────────────────────────────
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    optim_step += 1
 
                global_step += 1
                raw_loss = loss.item() * args.grad_accum   # unscale for display
                pbar.set_postfix(
                    train_loss=f"{raw_loss:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )
                train_steps.append(global_step)
                train_losses.append(raw_loss)
 
                # ── evaluation ───────────────────────────────────────────────
                if global_step % train_cfg.eval_every == 0:
                    val_loss = estimate_loss(
                        model,
                        encoded.valid_ids,
                        batch_size=train_cfg.batch_size,
                        seq_len=model_cfg.max_seq_len,
                        device=train_cfg.device,
                    )
                    record = {
                        "step": global_step,
                        "train_loss": round(raw_loss, 6),
                        "val_loss": round(val_loss, 6),
                        "lr": round(scheduler.get_last_lr()[0], 8),
                    }
                    history.append(record)
                    val_steps.append(global_step)
                    val_losses.append(val_loss)
                    print(f"\nstep={global_step} train={raw_loss:.4f} val={val_loss:.4f} "
                          f"lr={scheduler.get_last_lr()[0]:.2e}")
 
                    if val_loss < best_val:
                        best_val = val_loss
                        ckpt_path = ckpt_dir / "best.pt"
                        save_checkpoint(str(ckpt_path), {
                            "model_state_dict":model.state_dict(),
                            "model_config":model_cfg.__dict__,
                            "tokenizer_stoi":encoded.tokenizer.stoi,
                            "tokenizer_itos":encoded.tokenizer.itos,
                            "best_val_loss":best_val,
                            "run_name":run_name,
                        })
                        print(f"Saved best checkpoint  val={best_val:.4f}")
 
                if train_cfg.max_steps and global_step >= train_cfg.max_steps:
                    break
 
            if train_cfg.max_steps and global_step >= train_cfg.max_steps:
                break
 
    finally:
        print("\nSaving latest checkpoint and metrics...")
 
        latest_path = ckpt_dir / "latest.pt"
        save_checkpoint(str(latest_path), {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_step": global_step,
            "optim_step": optim_step,
            "model_config": model_cfg.__dict__,
            "tokenizer_stoi": encoded.tokenizer.stoi,
            "tokenizer_itos": encoded.tokenizer.itos,
            "best_val_loss": best_val,
            "run_name": run_name,
        })
        print(f"Saved latest → {latest_path}")
 
        metrics_path = art_dir / "train_metrics.json"
        save_json(str(metrics_path), {
            "run_name": run_name,
            "history": history,
            "best_val_loss": best_val,
            "train_steps": train_steps,
            "train_losses": train_losses,
            "val_steps": val_steps,
            "val_losses": val_losses,
        })
        print(f"Saved metrics → {metrics_path}")
 
        if train_steps:
            curve_path = art_dir / "loss_curve.png"
            save_loss_curve(
                train_steps=train_steps,
                train_losses=train_losses,
                val_steps=val_steps,
                val_losses=val_losses,
                save_path=str(curve_path),
                title=f"Training Curve ({args.attention})",
            )
            print(f"Saved loss curve → {curve_path}")
 
        print(f"\nBest val loss : {best_val:.4f}")
        print(f"Stopped at step: {global_step}  (optim steps: {optim_step})")
 
 
if __name__ == "__main__":
    main()