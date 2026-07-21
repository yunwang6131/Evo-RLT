#!/usr/bin/env python
"""Patch reward_seq in cached chunk-transition .pt files in place, without
re-running the VLA/RL-token encoding pass.

Fixes caches built by the reward-blind build_transition_cache_v2.py (which
hardcoded reward_seq=0 for every transition): sets reward_seq[actual_steps-1]
= 1.0 on each terminal chunk (done==1), matching evo_rlt.core.rewards.
build_reward_seq exactly, and leaves non-terminal chunks untouched (they were
already zero, and stay zero). Loads each <cache_dir>/chunk_transitions_<split>.pt
produced by save_transition_cache (a list of dicts), and writes the file back.
The original is renamed to <file><backup-suffix> the first time only, so
re-running never clobbers the true original.

Only correct when every episode contributing to the cache is a success — pass
--assume-success (the default) to confirm that. If the source dataset can
contain failed episodes, rebuild via build_transition_cache_v2.py instead,
which now reads episode_success per episode.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from evo_rlt.core.rewards import build_reward_seq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--backup-suffix", default=".bak_zeroreward")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--assume-success", action="store_true", default=True,
        help="All episodes in this cache are successful demos (required flag, see module docstring).",
    )
    return p.parse_args()


def patch_split(cache_dir: Path, split: str, backup_suffix: str, dry_run: bool) -> None:
    file = cache_dir / f"chunk_transitions_{split}.pt"
    if not file.exists():
        print(f"[{split}] SKIP (not found): {file}")
        return

    transitions = torch.load(file, weights_only=False)
    n = len(transitions)
    if n == 0:
        print(f"[{split}] EMPTY: {file}")
        return

    nz_before = sum(1 for t in transitions if bool((t["reward_seq"] != 0).any().item()))
    n_done = sum(1 for t in transitions if bool(t["done"].item()))

    for t in transitions:
        is_terminal = bool(t["done"].item())
        C = t["reward_seq"].shape[0]
        t["reward_seq"] = build_reward_seq(
            chunk_length=C,
            is_terminal_chunk=is_terminal,
            episode_success=True,
            actual_steps=t.get("actual_steps"),
            device=t["reward_seq"].device,
        ).to(t["reward_seq"].dtype)

    nz_after = sum(1 for t in transitions if bool((t["reward_seq"] != 0).any().item()))

    print(f"[{split}] n={n} done={n_done} nonzero_reward {nz_before}->{nz_after}")
    if nz_after != n_done:
        raise RuntimeError(
            f"[{split}] expected nonzero_reward == done count ({n_done}), got {nz_after}"
        )

    if dry_run:
        print(f"[{split}] DRY-RUN: not writing")
        return

    backup = file.with_name(file.name + backup_suffix)
    if not backup.exists():
        file.rename(backup)
        print(f"[{split}] backed up original -> {backup.name}")
    else:
        print(f"[{split}] backup {backup.name} exists, skipping backup")

    torch.save(transitions, file)
    print(f"[{split}] wrote {file.name}")


def main() -> None:
    args = parse_args()
    if not args.assume_success:
        raise SystemExit(
            "--assume-success must be true: this patch sets reward=1.0 on every terminal "
            "chunk regardless of episode_success. If the cache can contain failed episodes, "
            "rebuild with build_transition_cache_v2.py instead."
        )
    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        raise FileNotFoundError(cache_dir)
    print(f"cache_dir={cache_dir} dry_run={args.dry_run}")
    for split in args.splits:
        patch_split(cache_dir, split, args.backup_suffix, args.dry_run)


if __name__ == "__main__":
    main()
