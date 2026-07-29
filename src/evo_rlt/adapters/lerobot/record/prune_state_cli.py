"""Remove specific episode(s) from a saved online-RL state's replay buffer
(latest_online_state.pt) -- e.g. an episode later found to have a bad
outcome label or corrupted data. This only stops those transitions from
being sampled in future training; it does not undo any gradient step already
taken while they were still in the buffer (see the module-level warning
printed at runtime).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch


def prune_episodes(state: dict, episode_ids: set[int]) -> tuple[dict, int]:
    buffer = state["replay_buffer"]
    kept = [t for t in buffer if int(t.episode_id.item()) not in episode_ids]
    n_removed = len(buffer) - len(kept)
    state["replay_buffer"] = kept
    # total_added must stay an accurate count of what's actually in the
    # buffer's history (see ReplayBuffer.rollback()'s same convention) --
    # decrement by what was actually removed, don't just recompute from len().
    state["replay_buffer_total_added"] = state["replay_buffer_total_added"] - n_removed
    return state, n_removed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remove one or more episode_id's transitions from a saved "
        "online-RL latest_online_state.pt."
    )
    parser.add_argument("--state-path", required=True, help="Path to latest_online_state.pt.")
    parser.add_argument(
        "--episode-id", type=int, action="append", required=True, dest="episode_ids",
        help="Episode id to remove (repeat --episode-id N for multiple). This is the "
        "recorded-episode index shown in training logs, e.g. 'Recording episode 41' -> 41.",
    )
    parser.add_argument(
        "--in-place", action="store_true", default=False,
        help="Overwrite --state-path (a .bak backup of the original is written first). "
        "Default: write to --output instead, leaving the original untouched.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path when not --in-place. Defaults to <state-path>.pruned.pt.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    state_path = Path(args.state_path)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    episode_ids = set(args.episode_ids)
    buffer_len_before = len(state["replay_buffer"])
    state, n_removed = prune_episodes(state, episode_ids)

    if n_removed == 0:
        print(f"No transitions found with episode_id in {sorted(episode_ids)} in "
              f"{buffer_len_before} total -- nothing removed, no file written.")
        return

    if args.in_place:
        backup_path = state_path.with_suffix(state_path.suffix + ".bak")
        shutil.copy2(state_path, backup_path)
        out_path = state_path
        print(f"Backed up original ({buffer_len_before} transitions) to {backup_path}")
    else:
        out_path = Path(args.output) if args.output else state_path.with_suffix(state_path.suffix + ".pruned.pt")

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(out_path)
    print(
        f"Removed {n_removed} transition(s) with episode_id in {sorted(episode_ids)} "
        f"({buffer_len_before} -> {len(state['replay_buffer'])} transitions). Wrote {out_path}\n"
        "Note: gradient steps already taken using this data before now cannot be undone -- "
        "this only prevents future sampling of it."
    )


if __name__ == "__main__":
    main()
