"""Unified training script for the FSM vs GPT comparison on TinyStories.

Examples
--------

# 1. Plain (single-window) training
python train.py --model fsm --mode plain --name fsm_plain
python train.py --model gpt --mode plain --name gpt_plain

# 2. Streaming with constant-length KV cache (Transformer-XL style for GPT)
python train.py --model fsm --mode stream --name fsm_stream
python train.py --model gpt --mode stream --name gpt_stream

# 3. Long-context GPT (no cache, larger window) vs default FSM
python train.py --model gpt --mode long --block-size 1024 --name gpt_long

All runs write metrics to ``runs/<name>/metrics.jsonl`` and a final
checkpoint to ``runs/<name>/ckpt.pt``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from src.data import get_batch, get_stream_batch, prepare_tinystories
from src.fsm import FractalSequenceModel, FSMConfig
from src.gpt import GPT, GPTConfig


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["fsm", "gpt"], required=True)
    p.add_argument("--mode", choices=["plain", "stream", "long"], default="plain",
                   help="plain: one window/step. "
                        "stream: chunked w/ KV cache passed across chunks. "
                        "long: single window of size block_size (use a large value).")
    p.add_argument("--name", required=True, help="run name; logs to runs/<name>/")

    # data
    p.add_argument("--n-train-docs", type=int, default=20000)
    p.add_argument("--n-val-docs", type=int, default=200)
    p.add_argument("--data-dir", default="data")

    # shared model hparams
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--n-embd", type=int, default=256)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--mem-len", type=int, default=64)

    # FSM-specific
    p.add_argument("--n-levels", type=int, default=3)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--no-skip", action="store_true")

    # GPT-specific
    p.add_argument("--n-layer", type=int, default=9,
                   help="GPT depth; default 9 ~= 20.0M params, matching FSM (20.4M) at default settings.")

    # optimisation
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-chunks", type=int, default=4,
                   help="chunks per stream sample (stream mode only)")
    p.add_argument("--max-iters", type=int, default=2000)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup-iters", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--device", default=None)
    p.add_argument("--compile", action="store_true")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def auto_device(d: str | None) -> str:
    if d is not None:
        return d
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def lr_at(it: int, args) -> float:
    if it < args.warmup_iters:
        return args.lr * it / max(args.warmup_iters, 1)
    if it >= args.max_iters:
        return args.min_lr
    progress = (it - args.warmup_iters) / max(args.max_iters - args.warmup_iters, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_lr + coeff * (args.lr - args.min_lr)


def build_model(args, vocab_size: int):
    if args.model == "fsm":
        cfg = FSMConfig(
            vocab_size=vocab_size,
            block_size=args.block_size,
            n_levels=args.n_levels, K=args.K,
            n_embd=args.n_embd, n_head=args.n_head,
            mlp_ratio=args.mlp_ratio, dropout=args.dropout,
            use_skip=not args.no_skip,
            mem_len=args.mem_len if args.mode == "stream" else 0,
        )
        return FractalSequenceModel(cfg), cfg
    else:
        cfg = GPTConfig(
            vocab_size=vocab_size,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_embd=args.n_embd, n_head=args.n_head,
            mlp_ratio=args.mlp_ratio, dropout=args.dropout,
            mem_len=args.mem_len if args.mode == "stream" else 0,
        )
        return GPT(cfg), cfg


@torch.no_grad()
def estimate_loss(model, train_data, val_data, args, device) -> dict:
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = []
        for _ in range(args.eval_iters):
            if args.mode == "stream":
                xs, ys = get_stream_batch(data, args.block_size,
                                          args.n_chunks, args.batch_size, device)
                state = model.init_state()
                for x, y in zip(xs, ys):
                    _, loss, state = model(x, y, state=state)
                    losses.append(loss.item())
            else:
                x, y = get_batch(data, args.block_size, args.batch_size, device)
                _, loss = model(x, y)
                losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = auto_device(args.device)

    run_dir = Path("runs") / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    metrics_f = open(metrics_path, "w")

    print(f"== run: {args.name} | model={args.model} mode={args.mode} device={device} ==")

    # data ------------------------------------------------------------------
    train_data, val_data, vocab_size = prepare_tinystories(
        cache_dir=args.data_dir,
        n_train_docs=args.n_train_docs,
        n_val_docs=args.n_val_docs,
    )
    print(f"train tokens: {len(train_data):,}  val tokens: {len(val_data):,}  vocab: {vocab_size}")

    # model -----------------------------------------------------------------
    model, cfg = build_model(args, vocab_size)
    model = model.to(device)
    print(f"model params: {model.num_params() / 1e6:.2f} M")
    print(f"config: {asdict(cfg)}")

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    # optimiser -------------------------------------------------------------
    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(0.9, 0.95), weight_decay=args.weight_decay,
    )

    # log header ------------------------------------------------------------
    metrics_f.write(json.dumps({
        "event": "config", "args": vars(args), "cfg": asdict(cfg),
        "n_params": model.num_params() if hasattr(model, "num_params") else None,
    }) + "\n"); metrics_f.flush()

    # training loop ---------------------------------------------------------
    model.train()
    t0 = time.time()
    for it in range(1, args.max_iters + 1):
        # LR schedule
        lr = lr_at(it, args)
        for g in optim.param_groups:
            g["lr"] = lr

        optim.zero_grad(set_to_none=True)

        if args.mode == "stream":
            xs, ys = get_stream_batch(train_data, args.block_size,
                                      args.n_chunks, args.batch_size, device)
            state = model.init_state()
            last_loss = 0.0
            for x, y in zip(xs, ys):
                _, loss, state = model(x, y, state=state)
                loss.backward()
                last_loss = loss.item()
            train_loss = last_loss
        else:
            x, y = get_batch(train_data, args.block_size, args.batch_size, device)
            _, loss = model(x, y)
            loss.backward()
            train_loss = loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()

        if it % args.eval_every == 0 or it == 1:
            m = estimate_loss(model, train_data, val_data, args, device)
            dt = time.time() - t0
            print(f"iter {it:5d} | lr {lr:.2e} | train(running) {train_loss:.3f} | "
                  f"eval train {m['train']:.3f} val {m['val']:.3f} | {dt:.1f}s")
            metrics_f.write(json.dumps({
                "event": "eval", "iter": it, "lr": lr,
                "train_running": train_loss,
                "train_eval": m["train"], "val_eval": m["val"],
                "elapsed": dt,
            }) + "\n"); metrics_f.flush()

    # checkpoint ------------------------------------------------------------
    ckpt_path = run_dir / "ckpt.pt"
    torch.save({
        "model_state": (model._orig_mod.state_dict()
                        if hasattr(model, "_orig_mod") else model.state_dict()),
        "cfg": asdict(cfg),
        "args": vars(args),
    }, ckpt_path)
    print(f"saved checkpoint -> {ckpt_path}")
    metrics_f.close()


if __name__ == "__main__":
    main()
