#!/usr/bin/env python
"""Raw stats on chunk_transitions_{split}.pt: reward sparsity, exec_chunk vs ref_chunk."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    return p.parse_args()


def inspect_split(cache_dir: Path, split: str) -> None:
    path = cache_dir / f"chunk_transitions_{split}.pt"
    if not path.exists():
        print(f"[{split}] SKIP: {path} not found")
        return

    data = torch.load(path, weights_only=False)
    n = len(data)
    print(f"\n=== [{split}] n={n} ===")
    if n == 0:
        return

    reward = torch.stack([t["reward_seq"] for t in data])
    n_nonzero = int((reward != 0).any(dim=1).sum().item())
    print(f"reward_seq: nonzero_transitions={n_nonzero}/{n} sum={reward.sum().item():.4f}")

    done = torch.stack([t["done"] for t in data]).bool()
    n_done = int(done.sum().item())
    n_done_reward = int((reward[done] != 0).any(dim=1).sum().item()) if n_done else 0
    print(f"done=1: {n_done}/{n}, of those with nonzero reward: {n_done_reward}")

    n_identical = sum(1 for t in data if (t["exec_chunk"] - t["ref_chunk"]).abs().max().item() == 0.0)
    mean_diff = sum((t["exec_chunk"] - t["ref_chunk"]).abs().mean().item() for t in data) / n
    print(f"exec_chunk == ref_chunk (bit-exact): {n_identical}/{n}  mean_abs_diff={mean_diff:.6f}")

    exec_stack = torch.stack([t["exec_chunk"] for t in data])
    print(f"exec_chunk range: [{exec_stack.min().item():.4f}, {exec_stack.max().item():.4f}]")

    state = torch.stack([t["state_vec"] for t in data])
    print(f"state_vec: nan={bool(torch.isnan(state).any())} inf={bool(torch.isinf(state).any())}")


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    for split in args.splits:
        inspect_split(cache_dir, split)


if __name__ == "__main__":
    main()
