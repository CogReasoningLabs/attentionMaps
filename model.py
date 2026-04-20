from __future__ import annotations

import torch
import torch.nn as nn

from attention_variants import ATTENTION_REGISTRY
from config import ModelConfig


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        attn_cls = ATTENTION_REGISTRY[cfg.attention_type]
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = attn_cls(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, need_weights: bool = False):
        attn_out, attn_weights = self.attn(self.ln1(x), need_weights=need_weights)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, attn_weights


class TinyTransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_attention: bool = False,
    ):
        bsz, seq_len = idx.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.cfg.max_seq_len}.")

        positions = torch.arange(0, seq_len, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(positions)
        x = self.drop(x)

        attentions = []
        for block in self.blocks:
            x, attn = block(x, need_weights=return_attention)
            if return_attention:
                attentions.append(attn)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return {
            "logits": logits,
            "loss": loss,
            "attentions": attentions if return_attention else None,
        }
    
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        do_sample: bool = True,
        eos_token_id: int | None = None,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
        self.eval()

        collected_attentions: list[list[torch.Tensor]] = []

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len :]

            out = self(idx_cond, targets=None, return_attention=return_attention)
            logits = out["logits"]  # (B, T, vocab_size)
            next_token_logits = logits[:, -1, :]  # (B, vocab_size)

            if temperature <= 0:
                raise ValueError("temperature must be > 0")

            next_token_logits = next_token_logits / temperature

            if top_k is not None:
                k = min(top_k, next_token_logits.size(-1))
                values, _ = torch.topk(next_token_logits, k=k)
                threshold = values[:, -1].unsqueeze(-1)
                next_token_logits = torch.where(
                    next_token_logits < threshold,
                    torch.full_like(next_token_logits, float("-inf")),
                    next_token_logits,
                )

            probs = torch.softmax(next_token_logits, dim=-1)

            if do_sample:
                next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)  # (B, 1)

            if return_attention and out["attentions"] is not None:
                collected_attentions.append(out["attentions"])

            idx = torch.cat([idx, next_token], dim=1)

            if eos_token_id is not None and torch.all(next_token.squeeze(-1) == eos_token_id):
                break

        return {
            "sequences": idx,
            "step_attentions": collected_attentions if return_attention else None,
        }