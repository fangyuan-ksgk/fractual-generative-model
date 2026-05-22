"""Vanilla GPT baseline with the same API as FractalSequenceModel.

- Same KV-cache + state contract: `model(idx, targets, state=...)` returns
  `(logits, loss, state)` when state is given, else `(logits, loss)`.
- A constant-length per-layer KV cache (`cfg.mem_len`) gives a
  Transformer-XL-style streaming baseline directly comparable to FSM's
  memory mechanism.
- Reuses `GPTBlock`, `sinusoidal_pos_embed` from src.fsm so the only
  difference vs FSM is the architecture (homogeneous stack vs fractal).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fsm import KV, GPTBlock


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 10
    n_embd: int = 256
    n_head: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    bias: bool = True
    mem_len: int = 0


@dataclass
class GPTState:
    kv_caches: List[Optional[KV]] = field(default_factory=list)
    pos_offset: int = 0
    mem_len: int = 0

    def update_cache(self, block_idx: int, new_kv: KV) -> None:
        past = self.kv_caches[block_idx]
        if past is not None:
            k = torch.cat([past[0], new_kv[0]], dim=-2)
            v = torch.cat([past[1], new_kv[1]], dim=-2)
        else:
            k, v = new_kv
        if k.size(-2) > self.mem_len:
            k = k[..., -self.mem_len:, :]
            v = v[..., -self.mem_len:, :]
        self.kv_caches[block_idx] = (k.detach(), v.detach())

    def detach(self) -> "GPTState":
        new_kv = [
            (kv[0].detach(), kv[1].detach()) if kv is not None else None
            for kv in self.kv_caches
        ]
        return GPTState(kv_caches=new_kv,
                        pos_offset=self.pos_offset,
                        mem_len=self.mem_len)


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        # GPTBlock is duck-compatible with this config (uses n_embd, n_head,
        # mlp_ratio, dropout, bias).
        self.blocks = nn.ModuleList([GPTBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def init_state(self) -> GPTState:
        return GPTState(
            kv_caches=[None] * self.cfg.n_layer,
            pos_offset=0,
            mem_len=self.cfg.mem_len,
        )

    def forward(self,
                idx: torch.Tensor,
                targets: Optional[torch.Tensor] = None,
                state: Optional[GPTState] = None):
        B, T = idx.shape
        assert T == self.cfg.block_size, (
            f"input length {T} != block_size {self.cfg.block_size}"
        )

        offset = 0 if state is None else state.pos_offset
        x = self.drop(self.tok_emb(idx))                  # no abs-pos: RoPE inside block

        for i, block in enumerate(self.blocks):
            past_kv = None if state is None else state.kv_caches[i]
            x, new_kv = block(x, past_kv=past_kv, pos_offset=offset)
            if state is not None:
                state.update_cache(i, new_kv)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )

        if state is None:
            return logits, loss
        state.pos_offset = offset + T
        return logits, loss, state

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        self.eval()
        T = self.cfg.block_size
        for _ in range(max_new_tokens):
            ctx = idx[:, -T:]
            if ctx.size(1) < T:
                pad = ctx.new_zeros(ctx.size(0), T - ctx.size(1))
                ctx_in = torch.cat([pad, ctx], dim=1)
            else:
                ctx_in = ctx
            logits, _ = self(ctx_in)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx
