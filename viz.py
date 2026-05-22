"""Render a diagram of the Fractal Sequence Model architecture.

Usage
-----
    python viz.py                              # uses notebook defaults
    python viz.py --block-size 256 --K 4 --n-levels 3 --mem-len 64
    python viz.py --out architecture.png

Produces a U-Net-shaped figure:
- Each row is a scale s in [0..N]. The sequence at scale s is drawn as boxes
  whose count = block_size / K**s.
- Each row holds one GPT block per side: an encoder block on the left and
  (for s < N) a decoder block on the right. The deepest encoder block
  (s = N) is the bottleneck and sits in the middle.
- Arrows between rows are pure resolution changes: Downsample (Conv1d k=s=K)
  on the encoder side, Upsample (ConvT k=s=K + right-shift K-1) on the decoder.
- U-Net skips run horizontally from each encoder block's output to the
  matching decoder block's input (after Upsample, before the decoder block).
- A KV-cache strip is annotated per row with its effective receptive
  field: ``mem_len * K**s`` original tokens.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ---- styling ---------------------------------------------------------------

ENC_COLOR  = "#4C78A8"   # blue
DEC_COLOR  = "#F58518"   # orange
BOTTLE_COLOR = "#54A24B" # green
SKIP_COLOR = "#B279A2"   # purple
CACHE_COLOR = "#999999"  # grey
TOKEN_EDGE = "#222222"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--n-levels", type=int, default=3)
    p.add_argument("--mem-len", type=int, default=64)
    p.add_argument("--max-boxes", type=int, default=32,
                   help="max boxes drawn at the input scale (visual only)")
    p.add_argument("--out", default="architecture.png")
    return p.parse_args()


def draw_token_row(ax, *, y, n_tokens, n_boxes, x_center, width, color,
                   label_left=None, label_above=None):
    """Draw a row of ``n_boxes`` boxes centred at ``x_center`` with total width ``width``."""
    box_w = width / n_boxes
    box_h = 0.45
    x0 = x_center - width / 2
    for i in range(n_boxes):
        rect = mpatches.Rectangle(
            (x0 + i * box_w, y - box_h / 2),
            box_w * 0.92, box_h,
            facecolor=color, edgecolor=TOKEN_EDGE, linewidth=0.6, alpha=0.85,
        )
        ax.add_patch(rect)
    if label_left:
        ax.text(x0 - 0.25, y, label_left, va="center", ha="right",
                fontsize=10, color=color, weight="bold")
    if label_above:
        ax.text(x_center, y + box_h / 2 + 0.18, label_above,
                va="bottom", ha="center", fontsize=8.5, color="#444")


def vert_arrow(ax, x, y0, y1, color, label=None, side="left"):
    ax.annotate(
        "", xy=(x, y1), xytext=(x, y0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8,
                        mutation_scale=14),
    )
    if label:
        ha = "right" if side == "left" else "left"
        dx = -0.15 if side == "left" else 0.15
        ax.text(x + dx, (y0 + y1) / 2, label, va="center", ha=ha,
                fontsize=8.5, color=color)


def main():
    args = parse_args()
    L = args.block_size
    K = args.K
    N = args.n_levels
    M = args.mem_len

    assert L % (K ** N) == 0, "block_size must be divisible by K**n_levels"

    # Per-scale info
    scales = list(range(N + 1))
    lengths = [L // (K ** s) for s in scales]
    rf      = [M * (K ** s) for s in scales]   # mem_len receptive field at scale s

    # Layout
    enc_x = -4.5
    dec_x =  4.5
    bottleneck_x = 0.0
    row_y = [-(s * 2.0) for s in scales]      # top = input (s=0), bottom = bottleneck
    base_w = 6.0                               # widest row (input)

    fig, ax = plt.subplots(figsize=(15, 2.4 + 1.6 * N))
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- draw rows --------------------------------------------------------
    for s in scales:
        y = row_y[s]
        n_tok = lengths[s]
        # visual box count: shrink with scale, capped at args.max_boxes
        n_boxes = min(n_tok, max(2, args.max_boxes // (K ** s)))
        # width shrinks with scale (rough visual proxy for sequence length)
        w = base_w * (n_tok / lengths[0]) ** 0.5
        w = max(w, 0.6)

        is_top    = (s == 0)
        is_bottom = (s == N)

        if is_top:
            draw_token_row(ax, y=y, n_tokens=n_tok, n_boxes=n_boxes,
                           x_center=enc_x, width=w, color=ENC_COLOR,
                           label_left=f"L={n_tok}",
                           label_above="input tokens  ➜  enc_blocks[0]")
            draw_token_row(ax, y=y, n_tokens=n_tok, n_boxes=n_boxes,
                           x_center=dec_x, width=w, color=DEC_COLOR,
                           label_above=f"dec_blocks[{N - 1}]  ➜  output / logits")
        elif is_bottom:
            draw_token_row(ax, y=y, n_tokens=n_tok, n_boxes=n_boxes,
                           x_center=bottleneck_x, width=w, color=BOTTLE_COLOR,
                           label_left=f"L={n_tok}")
            ax.text(bottleneck_x, y - 0.5,
                    f"enc_blocks[{N}]  (bottleneck GPT block)",
                    va="top", ha="center", fontsize=8.5, color="#444")
        else:
            draw_token_row(ax, y=y, n_tokens=n_tok, n_boxes=n_boxes,
                           x_center=enc_x, width=w, color=ENC_COLOR,
                           label_left=f"L={n_tok}",
                           label_above=f"enc_blocks[{s}]")
            draw_token_row(ax, y=y, n_tokens=n_tok, n_boxes=n_boxes,
                           x_center=dec_x, width=w, color=DEC_COLOR,
                           label_above=f"dec_blocks[{N - 1 - s}]")

        # cache annotation on the far right
        if M > 0:
            tag = "verbatim" if s == 0 else f"compressed ×{K**s}"
            cache_text = (f"KV cache: {M} slots  →  {rf[s]:,} original tokens  ({tag})")
            ax.text(10.5, y, cache_text, va="center", ha="left",
                    fontsize=9, color=CACHE_COLOR,
                    family="monospace")

    # ---- downsample / upsample arrows (pure resolution change) -----------
    for s in range(N):
        y_top = row_y[s]
        y_bot = row_y[s + 1]
        is_into_bottleneck = (s + 1 == N)

        # Encoder side: enc_blocks[s] output --downs[s]--> input of enc_blocks[s+1]
        x_to = bottleneck_x if is_into_bottleneck else enc_x
        ax.annotate("", xy=(x_to, y_bot + 0.30),
                    xytext=(enc_x, y_top - 0.30),
                    arrowprops=dict(arrowstyle="-|>", color=ENC_COLOR, lw=2.0,
                                    mutation_scale=14))
        if s == 0:
            ax.text(enc_x - base_w / 2 - 0.4, (y_top + y_bot) / 2,
                    f"downs[{s}]\nConv1d (k={K}, s={K})",
                    va="center", ha="right", fontsize=8.5, color=ENC_COLOR)

        # Decoder side: output of enc_blocks[s+1] (or previous dec block)
        # --ups[N-1-s]--> input of dec_blocks[N-1-s]
        x_from = bottleneck_x if is_into_bottleneck else dec_x
        ax.annotate("", xy=(dec_x, y_top - 0.30),
                    xytext=(x_from, y_bot + 0.30),
                    arrowprops=dict(arrowstyle="-|>", color=DEC_COLOR, lw=2.0,
                                    mutation_scale=14))
        if s == 0:
            ax.text(dec_x + base_w / 2 + 0.4, (y_top + y_bot) / 2,
                    f"ups[{N - 1 - s}]\nConvT (k={K}, s={K})\n+ right-shift K−1",
                    va="center", ha="left", fontsize=8.5, color=DEC_COLOR)

    # ---- skip connections (curved above each row) ------------------------
    for s in range(N):                       # skips at scales 0..N-1
        y = row_y[s]
        # half-row width at this scale
        w = base_w * (lengths[s] / lengths[0]) ** 0.5
        w = max(w, 0.6)
        x0 = enc_x + w / 2 + 0.05
        x1 = dec_x - w / 2 - 0.05
        # curved arrow above the row
        bump = 0.55
        arrow = mpatches.FancyArrowPatch(
            (x0, y + 0.25), (x1, y + 0.25),
            connectionstyle=f"arc3,rad=-{bump / max(x1 - x0, 1):.3f}",
            arrowstyle="-|>", mutation_scale=12,
            color=SKIP_COLOR, lw=1.5, linestyle="--",
        )
        ax.add_patch(arrow)
        if s == 0:
            ax.text((x0 + x1) / 2, y + 1.05,
                    "U-Net skip:  enc_blocks[s].out  ⨁  ups[…].out  →  skip_projs[…]",
                    va="bottom", ha="center", fontsize=9, color=SKIP_COLOR)

    # ---- title / legend ---------------------------------------------------
    ax.set_title(
        f"Fractal Sequence Model  —  block={L}, K={K}, N={N}, mem_len={M}",
        fontsize=12, weight="bold", pad=14,
    )

    legend_handles = [
        mpatches.Patch(color=ENC_COLOR,    label="Encoder GPT block + Downsample"),
        mpatches.Patch(color=BOTTLE_COLOR, label="Bottleneck GPT block (enc_blocks[N])"),
        mpatches.Patch(color=DEC_COLOR,    label="Decoder GPT block + Upsample"),
        Line2D([0], [0], color=SKIP_COLOR, ls="--", label="U-Net skip (concat + linear)"),
        Line2D([0], [0], color=CACHE_COLOR, lw=4, label="constant-length KV cache"),
    ]
    ax.legend(handles=legend_handles, loc="lower left",
              bbox_to_anchor=(0.0, -0.05), ncol=3, fontsize=9, frameon=False)

    # frame
    ax.set_xlim(-11.0, 19.0)
    ax.set_ylim(row_y[-1] - 1.6, row_y[0] + 1.8)

    out = Path(args.out)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved -> {out.resolve()}")


if __name__ == "__main__":
    main()
