from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR

from attention_maps.common.artifacts import (
    save_checkpoint,
    save_json,
    set_seed,
)
from attention_maps.config import ModelConfig, TrainConfig, load_experiment_profile
from attention_maps.training.configuration import load_decoder_training_config
from attention_maps.training.data import get_batch, load_profile_dataset, load_wikitext2
from attention_maps.training.metrics import save_loss_curve
from attention_maps.training.model import TinyTransformerLM
from attention_maps.training.tokenized_data import (
    PackedDataLoaders,
    create_packed_dataloaders,
    load_materialized_tokenized_corpus,
)


@torch.no_grad()
def estimate_loss(
    model: TinyTransformerLM,
    data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: str,
    eval_iters: int = 20,
    moe_aux_weight: float = 0.01,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data, batch_size=batch_size, seq_len=seq_len, device=device)
        out = model(x, targets=y, moe_aux_weight=moe_aux_weight)
        losses.append(out["loss"].item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def estimate_loader_loss(
    model: TinyTransformerLM,
    loader,
    device: str,
    eval_iters: int = 20,
    moe_aux_weight: float = 0.01,
) -> float:
    """Estimate loss from deterministic packed evaluation batches."""

    model.eval()
    losses = []
    non_blocking = device.startswith("cuda")
    for index, (x, y) in enumerate(loader):
        if index >= eval_iters:
            break
        x = x.to(device, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)
        out = model(x, targets=y, moe_aux_weight=moe_aux_weight)
        losses.append(out["loss"].item())
    model.train()
    if not losses:
        raise ValueError("Evaluation DataLoader did not yield any batches")
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
 

def _training_config_defaults(arguments: list[str] | None) -> dict:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--training-config")
    known, _ = pre_parser.parse_known_args(arguments)
    if not known.training_config:
        return {}
    return load_decoder_training_config(known.training_config).cli_defaults()


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    defaults = _training_config_defaults(arguments)
    parser = argparse.ArgumentParser(
        description=(
            "Train a tiny transformer LM from materialized token IDs, a dataset "
            "profile, or the legacy WikiText path."
        )
    )
    parser.add_argument(
        "--training-config",
        type=str,
        help="YAML config for materialized token-ID training",
    )
    parser.add_argument(
        "--tokenized-dir",
        type=str,
        default=defaults.get("tokenized_dir"),
        help="Directory containing tokenizer/, split shards, and a manifest",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Versioned JSON dataset/tokenizer profile",
    )
    parser.add_argument("--no-data-cache", action="store_true")
    parser.add_argument(
        "--attention",
        type=str,
        default=defaults.get("attention", "softmax"),
        choices=["softmax", "cosine", "linear"],
    )
    parser.add_argument("--epochs", type=int, default=defaults.get("epochs", 5))
    parser.add_argument(
        "--batch-size", type=int, default=defaults.get("batch_size", 32)
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=defaults.get("grad_accum", 1),
        help="Gradient accumulation steps",
    )

    parser.add_argument("--seq-len", type=int, default=defaults.get("seq_len", 128))
    parser.add_argument("--d-model", type=int, default=defaults.get("d_model", 128))
    parser.add_argument("--n-heads", type=int, default=defaults.get("n_heads", 4))
    parser.add_argument("--n-layers", type=int, default=defaults.get("n_layers", 2))
    parser.add_argument("--d-ff", type=int, default=defaults.get("d_ff", 256))
    parser.add_argument("--dropout", type=float, default=defaults.get("dropout", 0.2))
    parser.add_argument("--lr", type=float, default=defaults.get("lr", 3e-4))
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=defaults.get("warmup_steps", 200),
    )
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=defaults.get("min_lr_ratio", 0.1),
    )
    parser.add_argument(
        "--moe-aux-weight",
        type=float,
        default=defaults.get("moe_aux_weight", 0.01),
    )
    parser.add_argument(
        "--no-moe",
        action="store_true",
        default=defaults.get("no_moe", False),
    )
    parser.add_argument(
        "--num-experts", type=int, default=defaults.get("num_experts", 4)
    )
    parser.add_argument("--top-k", type=int, default=defaults.get("top_k", 2))

    parser.add_argument(
        "--eval-every", type=int, default=defaults.get("eval_every", 200)
    )
    parser.add_argument(
        "--eval-iters", type=int, default=defaults.get("eval_iters", 20)
    )
    parser.add_argument("--max-steps", type=int, default=defaults.get("max_steps"))
    parser.add_argument(
        "--grad-clip", type=float, default=defaults.get("grad_clip", 1.0)
    )
    parser.add_argument(
        "--weight-decay", type=float, default=defaults.get("weight_decay", 0.1)
    )
    parser.add_argument("--max-vocab-size", type=int, default=50000)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument(
        "--device", type=str, default=defaults.get("device", "auto")
    )
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    parser.add_argument(
        "--runs-dir", type=str, default=defaults.get("runs_dir", "runs")
    )
    parser.add_argument("--run-name", type=str, default=defaults.get("run_name"))
    parser.add_argument(
        "--num-workers", type=int, default=defaults.get("num_workers", 0)
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("pin_memory", True),
    )
    parser.add_argument(
        "--drop-last-batch",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("drop_last_batch", True),
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to latest.pt checkpoint to resume from")
    parser.add_argument(
        "--compile",
        action="store_true",
        default=defaults.get("compile", False),
        help="Enable torch.compile (off by default for portable smoke runs)",
    )
    return parser.parse_args(arguments)


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def unwrap_model(model: TinyTransformerLM) -> TinyTransformerLM:
    return getattr(model, "_orig_mod", model)


def portable_model_state_dict(model: TinyTransformerLM) -> dict:
    return unwrap_model(model).state_dict()


def load_portable_model_state(model: TinyTransformerLM, state_dict: dict) -> None:
    if state_dict and all(key.startswith("_orig_mod.") for key in state_dict):
        state_dict = {
            key.removeprefix("_orig_mod."): value for key, value in state_dict.items()
        }
    unwrap_model(model).load_state_dict(state_dict)


def main() -> None:
    args = parse_args()
    args.device = resolve_device(args.device)
    if args.tokenized_dir and args.profile:
        raise ValueError("Choose either --tokenized-dir or --profile, not both")
    if args.training_config and not args.tokenized_dir:
        raise ValueError("A training YAML must configure data.tokenized_dir")
    if args.batch_size <= 0 or args.seq_len <= 0 or args.grad_accum <= 0:
        raise ValueError("batch size, sequence length, and grad accumulation must be positive")
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
 
    # ── data ──────────────────────────────────────────────────────────────────
    packed_loaders: PackedDataLoaders | None = None
    materialized_config = (
        load_decoder_training_config(args.training_config)
        if args.training_config
        else None
    )
    if args.tokenized_dir:
        profile = None
        print(f"Loading materialized token IDs: {args.tokenized_dir}")
        encoded = load_materialized_tokenized_corpus(args.tokenized_dir)
        packed_loaders = create_packed_dataloaders(
            encoded,
            sequence_length=args.seq_len,
            batch_size=args.batch_size,
            seed=args.seed,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory and args.device.startswith("cuda"),
            drop_last_batch=args.drop_last_batch,
        )
        print(
            "Packed blocks: "
            + ", ".join(
                f"{split}={count:,}"
                for split, count in packed_loaders.blocks.items()
            )
        )
    elif args.profile:
        profile = load_experiment_profile(args.profile)
        print(f"Loading profile: {profile.name}")
        encoded = load_profile_dataset(
            args.profile,
            use_cache=not args.no_data_cache,
        )
    else:
        profile = None
        print("Loading WikiText-103...")
        encoded = load_wikitext2(
            max_vocab_size=args.max_vocab_size,
            min_freq=args.min_freq,
        )
 
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
        use_moe=not args.no_moe,
        num_experts=args.num_experts,
        top_k=args.top_k,
    )
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        eval_every=args.eval_every,
        max_steps=args.max_steps,
        grad_clip=args.grad_clip,
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
        f"V{encoded.tokenizer.vocab_size}_"
        f"SEED{args.seed}"
    )
    if args.tokenized_dir:
        default_name = f"materialized_bpe_{default_name}"
    elif profile is not None:
        default_name = f"{profile.name}_{default_name}"
    run_name = args.run_name or default_name
    run_dir = Path(args.runs_dir) / run_name
    ckpt_dir = run_dir / "checkpoints"
    art_dir = run_dir / "artifacts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)
 
    save_json(
        str(art_dir / "config.json"),
        {
            "run_name": run_name,
            "args": vars(args),
            "training_yaml": (
                materialized_config.to_dict() if materialized_config else None
            ),
            "model_config": model_cfg.__dict__,
            "train_config": train_cfg.__dict__,
            "data_metadata": encoded.metadata,
        },
    )
    print(f"Run directory: {run_dir}")
 
    # ── model + optimiser ─────────────────────────────────────────────────────
    model = TinyTransformerLM(model_cfg).to(train_cfg.device)
    if args.compile:
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
 
    # AMP: use float16 on CUDA, bfloat16 on MPS (if available), skip on CPU
    use_amp = train_cfg.device.startswith("cuda")
    amp_dtype = torch.float16 if use_amp else torch.bfloat16
    # device_type for autocast must be "cuda" or "cpu" only (not full device string)
    amp_device_type = "cuda" if train_cfg.device.startswith("cuda") else "cpu"
    scaler = torch.amp.GradScaler(enabled=use_amp)
 
    # ── LR schedule ───────────────────────────────────────────────────────────
    tokens_per_epoch = len(encoded.train_ids)
    if packed_loaders is not None:
        steps_per_epoch = len(packed_loaders.train)
        if steps_per_epoch == 0:
            raise ValueError(
                "Training DataLoader has zero batches; reduce batch size or disable "
                "drop_last_batch"
            )
    else:
        steps_per_epoch = max(
            1,
            tokens_per_epoch // (train_cfg.batch_size * model_cfg.max_seq_len),
        )
    total_steps = train_cfg.epochs * steps_per_epoch
    optim_steps_total = train_cfg.epochs * math.ceil(
        steps_per_epoch / args.grad_accum
    )
 
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
        load_portable_model_state(model, ckpt["model_state_dict"])
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
    optimizer.zero_grad(set_to_none=True)
 
    # ── training loop ─────────────────────────────────────────────────────────
    try:
        for epoch in range(start_epoch, train_cfg.epochs):
            epoch_batches = (
                packed_loaders.train
                if packed_loaders is not None
                else range(steps_per_epoch)
            )
            pbar = tqdm(
                epoch_batches,
                total=steps_per_epoch,
                desc=f"epoch {epoch+1}/{train_cfg.epochs}",
            )
 
            for micro_step, batch in enumerate(pbar):
                if packed_loaders is not None:
                    x, y = batch
                    non_blocking = train_cfg.device.startswith("cuda")
                    x = x.to(train_cfg.device, non_blocking=non_blocking)
                    y = y.to(train_cfg.device, non_blocking=non_blocking)
                else:
                    x, y = get_batch(
                        encoded.train_ids,
                        batch_size=train_cfg.batch_size,
                        seq_len=model_cfg.max_seq_len,
                        device=train_cfg.device,
                    )
 
                # ── forward + backward ───────────────────────────────────────
                # Note: zero_grad BEFORE forward, not after backward (original bug)
                is_accum_step = (
                    (micro_step + 1) % args.grad_accum != 0
                    and (micro_step + 1) < steps_per_epoch
                )
 
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
                    if packed_loaders is not None:
                        val_loss = estimate_loader_loss(
                            model,
                            packed_loaders.validation,
                            device=train_cfg.device,
                            eval_iters=args.eval_iters,
                            moe_aux_weight=args.moe_aux_weight,
                        )
                    else:
                        val_loss = estimate_loss(
                            model,
                            encoded.valid_ids,
                            batch_size=train_cfg.batch_size,
                            seq_len=model_cfg.max_seq_len,
                            device=train_cfg.device,
                            eval_iters=args.eval_iters,
                            moe_aux_weight=args.moe_aux_weight,
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
                            "model_state_dict": portable_model_state_dict(model),
                            "model_config":model_cfg.__dict__,
                            "tokenizer": encoded.tokenizer.to_state(),
                            "data_metadata": encoded.metadata,
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
            "model_state_dict": portable_model_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_step": global_step,
            "optim_step": optim_step,
            "model_config": model_cfg.__dict__,
            "tokenizer": encoded.tokenizer.to_state(),
            "data_metadata": encoded.metadata,
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
