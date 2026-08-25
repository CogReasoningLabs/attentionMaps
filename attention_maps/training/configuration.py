"""YAML configuration for training from materialized token IDs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TokenizedDataConfig:
    tokenized_dir: Path
    sequence_length: int = 512
    batch_size: int = 4
    num_workers: int = 0
    pin_memory: bool = True
    drop_last_batch: bool = True

    def validate(self) -> None:
        if not self.tokenized_dir.is_dir():
            raise FileNotFoundError(
                f"Tokenized dataset directory not found: {self.tokenized_dir}"
            )
        if self.sequence_length <= 0:
            raise ValueError("data.sequence_length must be positive")
        if self.batch_size <= 0:
            raise ValueError("data.batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("data.num_workers must be non-negative")


@dataclass(frozen=True)
class DecoderModelConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    attention: str = "softmax"
    use_moe: bool = False
    num_experts: int = 4
    top_k: int = 2

    def validate(self) -> None:
        for name, value in (
            ("d_model", self.d_model),
            ("n_heads", self.n_heads),
            ("n_layers", self.n_layers),
            ("d_ff", self.d_ff),
        ):
            if value <= 0:
                raise ValueError(f"model.{name} must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("model.d_model must be divisible by model.n_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("model.dropout must be in [0, 1)")
        if self.attention not in {"softmax", "cosine", "linear"}:
            raise ValueError("model.attention must be softmax, cosine, or linear")
        if self.num_experts <= 0 or self.top_k <= 0:
            raise ValueError("model.num_experts and model.top_k must be positive")
        if self.top_k > self.num_experts:
            raise ValueError("model.top_k cannot exceed model.num_experts")


@dataclass(frozen=True)
class OptimizationConfig:
    epochs: int = 1
    learning_rate: float = 3e-4
    grad_accumulation: int = 4
    grad_clip: float = 1.0
    warmup_steps: int = 200
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    eval_every: int = 200
    eval_iters: int = 20
    max_steps: int | None = None
    moe_aux_weight: float = 0.01

    def validate(self) -> None:
        for name, value in (
            ("epochs", self.epochs),
            ("grad_accumulation", self.grad_accumulation),
            ("eval_every", self.eval_every),
            ("eval_iters", self.eval_iters),
        ):
            if value <= 0:
                raise ValueError(f"optimization.{name} must be positive")
        if self.learning_rate <= 0 or self.grad_clip <= 0:
            raise ValueError(
                "optimization.learning_rate and grad_clip must be positive"
            )
        if self.warmup_steps < 0:
            raise ValueError("optimization.warmup_steps must be non-negative")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("optimization.min_lr_ratio must be in [0, 1]")
        if self.weight_decay < 0 or self.moe_aux_weight < 0:
            raise ValueError(
                "optimization.weight_decay and moe_aux_weight must be non-negative"
            )
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("optimization.max_steps must be positive when set")


@dataclass(frozen=True)
class RunConfig:
    runs_dir: Path = Path("runs")
    run_name: str | None = None
    device: str = "auto"
    seed: int = 42
    compile: bool = False

    def validate(self) -> None:
        if self.device != "auto" and not self.device.strip():
            raise ValueError("run.device must be 'auto' or a device string")
        if self.run_name is not None and not self.run_name.strip():
            raise ValueError("run.run_name must be non-empty when set")


@dataclass(frozen=True)
class DecoderTrainingConfig:
    data: TokenizedDataConfig
    model: DecoderModelConfig
    optimization: OptimizationConfig
    run: RunConfig
    schema_version: int = 1

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Only training config schema_version=1 is supported")
        self.data.validate()
        self.model.validate()
        self.optimization.validate()
        self.run.validate()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["data"]["tokenized_dir"] = str(self.data.tokenized_dir)
        payload["run"]["runs_dir"] = str(self.run.runs_dir)
        return payload

    def cli_defaults(self) -> dict[str, Any]:
        """Map the structured YAML fields onto the existing training CLI."""

        return {
            "tokenized_dir": str(self.data.tokenized_dir),
            "seq_len": self.data.sequence_length,
            "batch_size": self.data.batch_size,
            "num_workers": self.data.num_workers,
            "pin_memory": self.data.pin_memory,
            "drop_last_batch": self.data.drop_last_batch,
            "d_model": self.model.d_model,
            "n_heads": self.model.n_heads,
            "n_layers": self.model.n_layers,
            "d_ff": self.model.d_ff,
            "dropout": self.model.dropout,
            "attention": self.model.attention,
            "no_moe": not self.model.use_moe,
            "num_experts": self.model.num_experts,
            "top_k": self.model.top_k,
            "epochs": self.optimization.epochs,
            "lr": self.optimization.learning_rate,
            "grad_accum": self.optimization.grad_accumulation,
            "grad_clip": self.optimization.grad_clip,
            "warmup_steps": self.optimization.warmup_steps,
            "min_lr_ratio": self.optimization.min_lr_ratio,
            "weight_decay": self.optimization.weight_decay,
            "eval_every": self.optimization.eval_every,
            "eval_iters": self.optimization.eval_iters,
            "max_steps": self.optimization.max_steps,
            "moe_aux_weight": self.optimization.moe_aux_weight,
            "runs_dir": str(self.run.runs_dir),
            "run_name": self.run.run_name,
            "device": self.run.device,
            "seed": self.run.seed,
            "compile": self.run.compile,
        }


def _mapping(payload: Any, name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    return dict(payload)


def _relative_path(value: Any, config_dir: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path string")
    path = Path(value)
    return (config_dir / path).resolve() if not path.is_absolute() else path.resolve()


def load_decoder_training_config(path: str | Path) -> DecoderTrainingConfig:
    """Load and validate a training YAML with paths relative to its location."""

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required; install requirements.txt") from exc

    config_path = Path(path).resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read training config {config_path}: {exc}") from exc
    root = _mapping(payload, "config")
    try:
        data = _mapping(root.pop("data"), "data")
        model = _mapping(root.pop("model"), "model")
        optimization = _mapping(root.pop("optimization"), "optimization")
        run = _mapping(root.pop("run"), "run")
        data["tokenized_dir"] = _relative_path(
            data["tokenized_dir"], config_path.parent, "data.tokenized_dir"
        )
        run["runs_dir"] = _relative_path(
            run.get("runs_dir", "../../runs"), config_path.parent, "run.runs_dir"
        )
        config = DecoderTrainingConfig(
            data=TokenizedDataConfig(**data),
            model=DecoderModelConfig(**model),
            optimization=OptimizationConfig(**optimization),
            run=RunConfig(**run),
            **root,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid training config {config_path}: {exc}") from exc
    config.validate()
    return config

