from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))


class BaseAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        x = x.view(bsz, seq_len, self.n_heads, self.head_dim)
        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_heads, seq_len, head_dim = x.shape
        x = x.transpose(1, 2).contiguous().view(bsz, seq_len, n_heads * head_dim)
        return x


class SoftmaxCausalAttention(BaseAttention):
    def forward(self, x: torch.Tensor, need_weights: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        q = self.split_heads(self.q_proj(x))
        k = self.split_heads(self.k_proj(x))
        v = self.split_heads(self.v_proj(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = causal_mask(scores.size(-1), scores.device)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out = torch.matmul(weights, v)
        out = self.out_proj(self.combine_heads(out))
        return out, weights if need_weights else None


class CosineCausalAttention(BaseAttention):
    def forward(self, x: torch.Tensor, need_weights: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        q = self.split_heads(self.q_proj(x))
        k = self.split_heads(self.k_proj(x))
        v = self.split_heads(self.v_proj(x))

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        scores = torch.matmul(q, k.transpose(-2, -1))
        mask = causal_mask(scores.size(-1), scores.device)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out = torch.matmul(weights, v)
        out = self.out_proj(self.combine_heads(out))
        return out, weights if need_weights else None


class LinearCausalAttention(BaseAttention):
    """
    Educational linear attention using a positive feature map phi(x) = elu(x) + 1.
    This avoids forming the full attention matrix during the main forward pass.
    """

    @staticmethod
    def phi(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor, need_weights: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        q = self.split_heads(self.q_proj(x))
        k = self.split_heads(self.k_proj(x))
        v = self.split_heads(self.v_proj(x))

        qp = self.phi(q)
        kp = self.phi(k)

        bsz, n_heads, seq_len, head_dim = qp.shape
        device = qp.device

        kv_prefix = torch.zeros(bsz, n_heads, head_dim, head_dim, device=device)
        k_prefix = torch.zeros(bsz, n_heads, head_dim, device=device)
        outputs = []

        for t in range(seq_len):
            kt = kp[:, :, t, :]
            vt = v[:, :, t, :]
            kv_prefix = kv_prefix + torch.einsum("bhd,bhe->bhde", kt, vt)
            k_prefix = k_prefix + kt

            qt = qp[:, :, t, :]
            numerator = torch.einsum("bhd,bhde->bhe", qt, kv_prefix)
            denominator = torch.einsum("bhd,bhd->bh", qt, k_prefix).unsqueeze(-1) + 1e-6
            out_t = numerator / denominator
            outputs.append(out_t.unsqueeze(2))

        out = torch.cat(outputs, dim=2)
        out = self.out_proj(self.combine_heads(out))

        weights = None
        if need_weights:
            weights = self.compute_prefix_influence_map(qp, kp)
        return out, weights

    def compute_prefix_influence_map(self, qp: torch.Tensor, kp: torch.Tensor) -> torch.Tensor:
        """
        Returns an approximate token-token influence matrix of shape [B, H, T, T]
        for visualization. Each row is normalized over the allowed prefix.
        """
        scores = torch.einsum("bhtd,bhsd->bhts", qp, kp)
        mask = causal_mask(scores.size(-1), scores.device)
        scores = scores.masked_fill(~mask, 0.0)
        denom = scores.sum(dim=-1, keepdim=True) + 1e-6
        return scores / denom


ATTENTION_REGISTRY = {
    "softmax": SoftmaxCausalAttention,
    "cosine": CosineCausalAttention,
    "linear": LinearCausalAttention,
}
