from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    max_seq_len: int = 64
    dropout: float = 0.1
    attention_type: str = "softmax"


@dataclass
class TrainConfig:
    batch_size: int = 16
    lr: float = 3e-4
    epochs: int = 3
    eval_every: int = 200
    max_steps: int | None = None
    grad_clip: float = 1.0
    device: str = "cpu"
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    artifact_dir: str = "artifacts"
