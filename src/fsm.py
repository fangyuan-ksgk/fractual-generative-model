"""Fractal Sequence Model (FSM).

A minimal prototype that replaces the homogeneous stack of Transformer layers
in a GPT with a *symmetric* U-Net-like stack of length-changing layers:

    embed -> [Compress]^N_fractal -> [Decompress]^N_fractal -> unembed

Each Compress layer turns a sequence of length L into one of length L/K,
each Decompress layer turns L into K*L. Length conversion is performed by
a simple (de)convolution with kernel=K and stride=K. The transformer block
itself is the canonical GPT block (causal multi-head self-attention + MLP).

Causality.
- Attention uses a causal mask at every level.
- Compress uses a non-overlapping stride-K conv: compressed position p
  aggregates input positions [p*K, p*K + K - 1]. This is causal at the
  *chunk* granularity: position p's content is fully determined by tokens
  with indices <= p*K + K - 1.
- Decompress upsamples (transpose conv stride=K) and is then right-shifted
  by K-1, so that output position j reads from a compressed position whose
  source tokens have indices <= j. This guarantees no information from
  future tokens leaks into position j at the original resolution.

Positional encoding.
- We use RoPE (Su et al., 2021) applied to Q and K inside each attention
  block. No absolute positional embedding is added to the token embedding.
- In streaming mode, queries / new keys are rotated at absolute positions
  [pos_offset, pos_offset + T); cached keys are stored *already rotated*
  at the positions they were written at, so cross-chunk concatenation
  needs no re-rotation. ``pos_offset`` is tracked per scale in FSMState.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

KV = Tuple[torch.Tensor, torch.Tensor]  # (k, v), each shape (B, n_head, M, head_dim)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class FSMConfig:
    vocab_size: int = 50257
    block_size: int = 1024          # input sequence length (must be divisible by K**n_levels)
    n_levels: int = 3               # number of compress (== decompress) layers
    K: int = 4                      # length factor per level
    n_embd: int = 256
    n_head: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    use_skip: bool = True           # U-Net-style skip connections
    bias: bool = True
    mem_len: int = 0                # constant per-layer KV-cache length (0 = disabled)


# ---------------------------------------------------------------------------
# Per-layer state: constant-length KV cache + per-scale position offset.
#
# Memory design: every attention block keeps a KV cache of length cfg.mem_len.
# Because layer s operates on a sequence whose tokens each summarise K**s
# original tokens, the same M-slot cache covers M*K**s original tokens at
# layer s -> recent layers store verbatim memory, deep layers store
# heavily-compressed distant memory.
#
# Notation: N_fractal = cfg.n_levels (number of compress/decompress layers).
# Total cache memory is O(M * (2*N_fractal + 1)), but the effective
# receptive field at the deepest layer is O(M * K**N_fractal).
# ---------------------------------------------------------------------------


@dataclass
class FSMState:
    # One KV cache per attention block, in order:
    #   compress[0..N_fractal-1], bottleneck, decompress[0..N_fractal-1]
    # so kv_caches has length 2 * N_fractal + 1.
    #
    # Each entry is either None (cache empty) or a tuple (k, v) where k and v
    # are tensors of shape (B, n_head, T_cache, head_dim) holding up to
    # `mem_len` past key/value vectors along the sequence dim.
    kv_caches: List[Optional[KV]] = field(default_factory=list)
    # one absolute-position counter per scale s in [0..N_fractal] (tokens
    # already consumed at that scale)
    pos_offsets: List[int] = field(default_factory=list)
    # constant per-layer cache length (0 = caching disabled)
    mem_len: int = 0

    def update_cache(self, block_idx: int, new_kv: KV) -> None:
        """Merge `new_kv` (current chunk) with the past cache for this block,
        FIFO-trim to `mem_len`, detach (Transformer-XL style), and store back.
        """
        past = self.kv_caches[block_idx]
        if past is not None:
            k = torch.cat([past[0], new_kv[0]], dim=-2)
            v = torch.cat([past[1], new_kv[1]], dim=-2)
        else:
            k, v = new_kv
        if k.size(-2) > self.mem_len:                  # -- truncate kv cache
            k = k[..., -self.mem_len:, :]
            v = v[..., -self.mem_len:, :]
        self.kv_caches[block_idx] = (k.detach(), v.detach())

    def detach(self) -> "FSMState":
        new_kv = [
            (kv[0].detach(), kv[1].detach()) if kv is not None else None
            for kv in self.kv_caches
        ]
        return FSMState(kv_caches=new_kv,
                        pos_offsets=list(self.pos_offsets),
                        mem_len=self.mem_len)


# ---------------------------------------------------------------------------
# Standard GPT block
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: FSMConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.proj_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor,
                past_kv: Optional[KV] = None,
                pos_offset: int = 0) -> Tuple[torch.Tensor, KV]:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and the *new* K at absolute positions
        # [pos_offset, pos_offset + T). Cached K were already rotated at
        # the positions they were originally written at, so they need no
        # further rotation.
        cos, sin = build_rope_cache(T, self.head_dim, q.device, q.dtype,
                                    offset=pos_offset)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if past_kv is None:
            # Fast path: no cache, fully causal attention.
            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Concatenate past keys/values; build mask: queries attend to all
            # past KV slots (no causal restriction wrt the past) and causally
            # to the current chunk.
            pk, pv = past_kv
            k_full = torch.cat([pk, k], dim=-2)
            v_full = torch.cat([pv, v], dim=-2)
            M = pk.size(-2)
            mask = torch.zeros(T, M + T, device=q.device, dtype=q.dtype)
            causal = torch.triu(
                torch.full((T, T), float("-inf"), device=q.device, dtype=q.dtype),
                diagonal=1,
            )
            mask[:, M:] = causal
            y = F.scaled_dot_product_attention(
                q, k_full, v_full,
                attn_mask=mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=False,
            )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # Cache the *rotated* K so future chunks can concat without
        # re-rotating. V is position-free.
        return self.proj_drop(self.proj(y)), (k, v)


class MLP(nn.Module):
    def __init__(self, cfg: FSMConfig):
        super().__init__()
        hidden = int(cfg.n_embd * cfg.mlp_ratio)
        self.fc1 = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.fc2 = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class GPTBlock(nn.Module):
    def __init__(self, cfg: FSMConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor,
                past_kv: Optional[KV] = None,
                pos_offset: int = 0) -> Tuple[torch.Tensor, KV]:
        a, new_kv = self.attn(self.ln1(x), past_kv=past_kv, pos_offset=pos_offset)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, new_kv


# ---------------------------------------------------------------------------
# Rotary Positional Embedding (RoPE, Llama-style "rotate-half" variant).
# ---------------------------------------------------------------------------


def build_rope_cache(seq_len: int, head_dim: int, device, dtype,
                     offset: int = 0, base: float = 10000.0):
    """Return (cos, sin) of shape ``(seq_len, head_dim)`` for positions
    ``[offset, offset + seq_len)``.
    """
    assert head_dim % 2 == 0, "RoPE requires even head_dim"
    inv_freq = 1.0 / (base ** (
        torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
    ))                                                                     # (head_dim/2,)
    t = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)                                       # (T, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)                                # (T, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.size(-1) // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` (B, H, T, D) by RoPE angles (cos, sin of shape (T, D))."""
    return x * cos + _rotate_half(x) * sin


# ---------------------------------------------------------------------------
# Resolution-change modules (no attention; thin wrappers around (de)conv).
# ---------------------------------------------------------------------------


class Downsample(nn.Module):
    """Non-overlapping stride-K conv: ``(B, L, C) -> (B, L/K, C)``."""

    def __init__(self, cfg: FSMConfig):
        super().__init__()
        self.K = cfg.K
        self.ln = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.conv = nn.Conv1d(cfg.n_embd, cfg.n_embd,
                              kernel_size=cfg.K, stride=cfg.K, bias=cfg.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x).transpose(1, 2)            # (B, C, L)
        h = self.conv(h).transpose(1, 2)          # (B, L/K, C)
        return h


class Upsample(nn.Module):
    """Stride-K transpose conv + right-shift by K-1 for causality:
    ``(B, L, C) -> (B, K*L, C)``.
    """

    def __init__(self, cfg: FSMConfig):
        super().__init__()
        self.K = cfg.K
        self.ln = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.deconv = nn.ConvTranspose1d(cfg.n_embd, cfg.n_embd,
                                         kernel_size=cfg.K, stride=cfg.K, bias=cfg.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x).transpose(1, 2)            # (B, C, L)
        h = self.deconv(h).transpose(1, 2)        # (B, K*L, C)
        # Right-shift by K-1: output position j reads from coarser position
        # floor((j - K + 1) / K), whose source tokens are all at indices <= j
        # in the original sequence.
        if self.K > 1:
            pad = h.new_zeros(h.size(0), self.K - 1, h.size(2))
            h = torch.cat([pad, h[:, :-(self.K - 1)]], dim=1)
        return h


# ---------------------------------------------------------------------------
# Fractal Sequence Model
# ---------------------------------------------------------------------------


class FractalSequenceModel(nn.Module):
    def __init__(self, cfg: FSMConfig):
        super().__init__()
        assert cfg.block_size % (cfg.K ** cfg.n_levels) == 0, (
            f"block_size={cfg.block_size} must be divisible by K**n_levels="
            f"{cfg.K ** cfg.n_levels}"
        )
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        N = cfg.n_levels
        # U-Net layout:
        #   enc_blocks[0..N]  : GPT blocks at scales 0..N (the last is the bottleneck)
        #   downs[0..N-1]     : conv stride-K, scale s -> s+1
        #   ups[0..N-1]       : deconv stride-K + right-shift, scale s+1 -> s
        #   dec_blocks[0..N-1]: GPT blocks at scales N-1, N-2, ..., 0
        # Skips connect enc_blocks[s] output to dec_blocks (the one at scale s),
        # fused U-Net style after the upsample and before the decoder block.
        self.enc_blocks = nn.ModuleList([GPTBlock(cfg) for _ in range(N + 1)])
        self.downs = nn.ModuleList([Downsample(cfg) for _ in range(N)])
        self.ups = nn.ModuleList([Upsample(cfg) for _ in range(N)])
        self.dec_blocks = nn.ModuleList([GPTBlock(cfg) for _ in range(N)])

        if cfg.use_skip:
            self.skip_projs = nn.ModuleList([
                nn.Linear(2 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
                for _ in range(N)
            ])

        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # weight tying
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # ---- helpers -----------------------------------------------------------

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def n_attn_blocks(self) -> int:
        # (N+1) encoder + N decoder GPT blocks = 2*N + 1
        return 2 * self.cfg.n_levels + 1

    def init_state(self) -> FSMState:
        """Allocate an empty memory state (all caches None, all offsets 0)."""
        return FSMState(
            kv_caches=[None] * self.n_attn_blocks,
            pos_offsets=[0] * (self.cfg.n_levels + 1),
            mem_len=self.cfg.mem_len,
        )

    def _attend(self, block_idx: int, layer, x, state, pos_offset: int = 0):
        """Run a sub-layer with optional per-layer KV cache.

        layer is one of CompressLayer / DecompressLayer / GPTBlock; their
        forwards all accept ``(x, past_kv, pos_offset)`` and return
        ``(..., new_kv)``.
        """
        past_kv = None if state is None else state.kv_caches[block_idx]
        out = layer(x, past_kv=past_kv, pos_offset=pos_offset)
        if state is not None:
            state.update_cache(block_idx, out[-1])
        return out

    # ---- forward -----------------------------------------------------------

    def forward(self,
                idx: torch.Tensor,
                targets: Optional[torch.Tensor] = None,
                state: Optional[FSMState] = None):
        """Forward pass.

        If `state` is None, runs as a stateless model (back-compat) and
        returns ``(logits, loss)``. If `state` is provided, the per-layer KV
        caches and per-scale position offsets are read & updated *in place*
        and the call returns ``(logits, loss, state)``.
        """
        B, T = idx.shape
        assert T == self.cfg.block_size, (
            f"input length {T} != block_size {self.cfg.block_size}"
        )
        N_fractal = self.cfg.n_levels
        K = self.cfg.K

        # Per-scale position offsets (scale s in [0..N_fractal]); RoPE
        # uses these to rotate Q / new K at absolute positions per scale.
        if state is None:
            offs = [0] * (N_fractal + 1)
        else:
            offs = list(state.pos_offsets)

        x = self.drop(self.tok_emb(idx))                  # no abs-pos added: RoPE only

        # Encoder: GPT block at scale s, then downsample to scale s+1.
        # The block at s=N_fractal is the bottleneck (no downsample after).
        skips = []
        for s in range(N_fractal + 1):
            x, _ = self._attend(s, self.enc_blocks[s], x, state, pos_offset=offs[s])
            if s < N_fractal:
                skips.append(x)                            # encoder output at scale s
                x = self.downs[s](x)                       # -> scale s+1

        # Decoder: upsample scale s+1 -> s, fuse U-Net skip, then GPT block at scale s.
        for j in range(N_fractal):
            s = N_fractal - 1 - j                          # decoder operates at this scale
            x = self.ups[j](x)                             # scale s+1 -> s
            if self.cfg.use_skip:
                x = self.skip_projs[j](torch.cat([x, skips[s]], dim=-1))
            x, _ = self._attend(N_fractal + 1 + j, self.dec_blocks[j], x, state,
                                pos_offset=offs[s])

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

        # Advance per-scale offsets by tokens-seen-at-that-scale-this-chunk.
        for s in range(N_fractal + 1):
            state.pos_offsets[s] = offs[s] + self.cfg.block_size // (K ** s)
        return logits, loss, state

    # ---- generation --------------------------------------------------------

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        """Naive generation: pad/crop to block_size each step, take last logit."""
        self.eval()
        T = self.cfg.block_size
        for _ in range(max_new_tokens):
            ctx = idx[:, -T:]
            if ctx.size(1) < T:
                pad = ctx.new_zeros(ctx.size(0), T - ctx.size(1))
                ctx_in = torch.cat([pad, ctx], dim=1)
                last_pos = T - 1
            else:
                ctx_in = ctx
                last_pos = T - 1
            logits, _ = self(ctx_in)
            logits = logits[:, last_pos, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx
