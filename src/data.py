"""TinyStories tokenize-and-cache + batch samplers shared by both models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import torch


def prepare_tinystories(
    cache_dir: str | os.PathLike = "data",
    n_train_docs: int = 20000,
    n_val_docs: int = 200,
    encoding_name: str = "gpt2",
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Tokenize a slice of TinyStories once and cache to disk.

    Returns ``(train_ids, val_ids, vocab_size)``.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{encoding_name}_t{n_train_docs}_v{n_val_docs}"
    train_path = cache_dir / f"train_{tag}.pt"
    val_path = cache_dir / f"val_{tag}.pt"
    meta_path = cache_dir / f"meta_{tag}.pt"

    if train_path.exists() and val_path.exists() and meta_path.exists():
        train = torch.load(train_path, map_location="cpu")
        val = torch.load(val_path, map_location="cpu")
        meta = torch.load(meta_path, map_location="cpu")
        return train, val, int(meta["vocab_size"])

    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding(encoding_name)
    eot = enc.eot_token

    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    train_ids: list[int] = []
    val_ids: list[int] = []
    for i, row in enumerate(ds):
        if i >= n_train_docs + n_val_docs:
            break
        toks = enc.encode_ordinary(row["text"]) + [eot]
        (val_ids if i < n_val_docs else train_ids).extend(toks)

    train = torch.tensor(train_ids, dtype=torch.long)
    val = torch.tensor(val_ids, dtype=torch.long)
    torch.save(train, train_path)
    torch.save(val, val_path)
    torch.save({"vocab_size": enc.n_vocab}, meta_path)
    return train, val, enc.n_vocab


def get_batch(data: torch.Tensor, block_size: int, batch_size: int,
              device: str | torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(0, data.size(0) - block_size - 1, (batch_size,))
    x = torch.stack([data[i     : i + block_size    ] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def get_stream_batch(data: torch.Tensor, block_size: int, n_chunks: int,
                     batch_size: int, device: str | torch.device
                     ) -> Tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Sample contiguous windows of ``n_chunks * block_size + 1`` tokens and
    split into chunk-aligned ``(x, y)`` pairs."""
    span = n_chunks * block_size
    ix = torch.randint(0, data.size(0) - span - 1, (batch_size,))
    seq = torch.stack([data[i : i + span + 1] for i in ix])     # (B, span+1)
    xs, ys = [], []
    for c in range(n_chunks):
        xs.append(seq[:, c * block_size     : (c + 1) * block_size    ].to(device))
        ys.append(seq[:, c * block_size + 1 : (c + 1) * block_size + 1].to(device))
    return xs, ys
