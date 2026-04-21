from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention_variants import ATTENTION_REGISTRY
from config import ModelConfig


class Expert(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_experts = cfg.num_experts
        self.top_k = cfg.top_k
        self.d_model = cfg.d_model
        moe_hidden = cfg.moe_hidden_dim or cfg.d_ff
        
        # Create experts
        self.experts = nn.ModuleList([
            Expert(cfg.d_model, moe_hidden, cfg.dropout) 
            for _ in range(self.num_experts)
        ])
        
        # Gating network
        self.gate = nn.Linear(cfg.d_model, self.num_experts)
        self.aux_loss: torch.Tensor = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x_flat = x.view(-1, D)          # (N, D)  N = B*T
        N = x_flat.size(0)
 
        gate_logits = self.gate(x_flat) # (N, E)
        top_k_logits, top_k_idx = torch.topk(gate_logits, self.top_k, dim=-1)
        gate_weights = F.softmax(top_k_logits, dim=-1)  # (N, top_k)
 
        # ── auxiliary load-balancing loss ────────────────────────────────────
        # Penalises uneven expert usage: encourages each expert to receive ~1/E
        # of all tokens. 
        # Classic formulation from Switch Transformer.
        with torch.no_grad():
            # fraction of tokens routed to each expert
            onehot = torch.zeros(N, self.num_experts, device=x.device)
            onehot.scatter_(1, top_k_idx, 1.0)
            tokens_per_expert = onehot.sum(0) / (N * self.top_k)   # (E,)
        # router probability per expert (soft, so gradients flow)
        router_prob = F.softmax(gate_logits, dim=-1).mean(0)        # (E,)
        self.aux_loss = self.num_experts * (tokens_per_expert * router_prob).sum()
 
        # ── efficient expert dispatch ────────────────────────────────────────
        # Instead of running all E experts on all N tokens (original bug),
        # run expert i only on the tokens assigned to it.
        output = torch.zeros_like(x_flat)
 
        # flatten top_k selections: each token appears top_k times
        token_idx  = torch.arange(N, device=x.device).unsqueeze(1).expand(N, self.top_k).reshape(-1)
        expert_idx = top_k_idx.reshape(-1)
        weights    = gate_weights.reshape(-1)
 
        for e in range(self.num_experts):
            mask   = expert_idx == e
            if not mask.any():
                continue
            toks   = x_flat[token_idx[mask]]       # tokens sent to expert e
            wts    = weights[mask].unsqueeze(-1)    # their gate weights
            result = self.experts[e](toks) * wts    # weighted output
            output.scatter_add_(0, token_idx[mask].unsqueeze(1).expand_as(result), result)
 
        return output.view(B, T, D)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        attn_cls = ATTENTION_REGISTRY[cfg.attention_type]
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = attn_cls(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        
        if cfg.use_moe:
            print(f"Using MoE with {cfg.num_experts} experts, top_k={cfg.top_k}, hidden_dim={cfg.moe_hidden_dim or cfg.d_ff}")
            self.ff = MoE(cfg)
            self.is_moe = True
        else:
            self.ff = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_ff),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.d_ff, cfg.d_model),
                nn.Dropout(cfg.dropout),
            )
            self.is_moe= False

    def forward(self, x: torch.Tensor, need_weights: bool = False):
        attn_out, attn_weights = self.attn(self.ln1(x), need_weights=need_weights)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, attn_weights
    
    @property
    def aux_loss(self) -> torch.Tensor:
        if self.is_moe:
            return self.ff.aux_loss
        return torch.tensor(0.0)


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
        self.head.weight = self.token_emb.weight  # weight tying

        self._init_weights()


    def _init_weights(self) -> None:
        """
        GPT-2-style initialisation:
          - Embeddings: N(0, 0.02)
          - Linear weights: N(0, 0.02)
          - Linear biases: 0
          - Residual projections scaled by 1/sqrt(2 * n_layers) so that the
            signal variance stays ~1 as depth grows.
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
 
        # scale residual projections
        scale = (2 * self.cfg.n_layers) ** -0.5
        for block in self.blocks:
            # the second linear of the FFN projects back into the residual stream
            if not block.is_moe:
                nn.init.normal_(block.ff[-2].weight, mean=0.0, std=0.02 * scale)
 

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_attention: bool = False,
        moe_aux_weight: float = 0.01,
    ) -> dict:
        B, T = idx.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(f"seq len {T} > max_seq_len {self.cfg.max_seq_len}")
 
        pos = torch.arange(T, device=idx.device).unsqueeze(0)  # (1, T)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
 
        attentions: list[torch.Tensor] = []
        aux_loss = torch.tensor(0.0, device=idx.device)
 
        for block in self.blocks:
            x, attn = block(x, need_weights=return_attention)
            if return_attention and attn is not None:
                attentions.append(attn)
            aux_loss = aux_loss + block.aux_loss
 
        x = self.ln_f(x)
        logits = self.head(x)           # (B, T, vocab_size)
 
        loss = None
        if targets is not None:
            ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            loss = ce + moe_aux_weight * aux_loss
 
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
        top_p: float | None = None,        # nucleus sampling
        do_sample: bool = True,
        eos_token_id: int | None = None,
        return_attention: bool = False,
    ) -> dict:
        """
        Autoregressive generation with optional top-k and nucleus (top-p) filtering.
 
        top_p: float in (0, 1] — keep the smallest set of tokens whose
               cumulative probability exceeds top_p.  Use together with
               top_k or alone.
        """
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
 
        self.eval()
        collected_attentions: list[list[torch.Tensor]] = []
 
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len:]
            out = self(idx_cond, return_attention=return_attention)
            logits = out["logits"][:, -1, :] / temperature   # (B, V)
 
            # top-k filtering
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits = logits.masked_fill(logits < v[:, -1:], float("-inf"))
 
            # nucleus (top-p) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = sorted_logits.softmax(-1).cumsum(-1)
                # remove tokens beyond the nucleus
                remove = cum_probs - sorted_logits.softmax(-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)
 
            probs = logits.softmax(-1)
 
            if do_sample:
                next_tok = torch.multinomial(probs, num_samples=1)
            else:
                next_tok = probs.argmax(-1, keepdim=True)
 
            if return_attention and out["attentions"]:
                collected_attentions.append(out["attentions"])
 
            idx = torch.cat([idx, next_tok], dim=1)
 
            if eos_token_id is not None and (next_tok.squeeze(-1) == eos_token_id).all():
                break
 
        return {
            "sequences": idx,
            "step_attentions": collected_attentions if return_attention else None,
        }