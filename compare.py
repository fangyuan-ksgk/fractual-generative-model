"""Plot val/train loss curves across one or more `runs/<name>/metrics.jsonl` files.

Usage
-----
    python compare.py runs/fsm_plain runs/gpt_plain
    python compare.py runs/* --metric val_eval --out runs/compare.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_run(run_dir: Path):
    iters, vals, trains = [], [], []
    cfg = None
    with open(run_dir / "metrics.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r.get("event") == "config":
                cfg = r
            elif r.get("event") == "eval":
                iters.append(r["iter"])
                vals.append(r["val_eval"])
                trains.append(r["train_eval"])
    return cfg, iters, trains, vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run directories under runs/")
    ap.add_argument("--metric", choices=["val_eval", "train_eval", "both"],
                    default="both")
    ap.add_argument("--out", default=None, help="path to save png; default: show()")
    args = ap.parse_args()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))

    for run in args.runs:
        run = Path(run)
        if not (run / "metrics.jsonl").exists():
            print(f"skip {run}: no metrics.jsonl"); continue
        cfg, its, tr, va = load_run(run)
        label = run.name
        if args.metric in ("val_eval", "both"):
            ax.plot(its, va, label=f"{label} val", linewidth=2)
        if args.metric in ("train_eval", "both"):
            ax.plot(its, tr, label=f"{label} train", linestyle="--", alpha=0.6)

    ax.set_xlabel("iteration")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("FSM vs GPT on TinyStories")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=120)
        print(f"saved -> {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
