"""Prepare, validate, and roll out SmolVLA on the blue-screw simulation data.

Training is not wrapped here: `lerobot-train --config_path=configs/smolvla/train_config.json`
does it directly. That json pins the four settings SmolVLA fails *silently*
without -- fine-tuning from the base rather than a random init, both freeze
flags, `load_vlm_weights=false`, and the camera rename map -- which is the
whole reason a wrapper existed. A wrapper additionally hid every lerobot flag
it did not re-export, which is how `--resume` went missing.

Rollout stays here, because lerobot has no notion of this rig's simulator,
key-labelled episodes or part resets. It reads the camera rename map from the
checkpoint's own train_config.json, so the map used at rollout is by
construction the one that checkpoint was trained with.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from evo_rlt.cli.act import (
    _print_command,
    _repo_path,
    build_prepare_argv,
    load_profile as _load_profile,
    validate_dataset_root,
    validate_sources,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROFILE = REPO_ROOT / "configs" / "blue_screw_sim_v1.json"
LABEL = "SmolVLA"


def load_profile(path: Path) -> dict[str, Any]:
    return _load_profile(path)


def _check_dataset(profile: dict[str, Any], *, check_totals: bool = True) -> tuple[int, int]:
    return validate_dataset_root(
        _repo_path(profile["merged_root"]),
        profile["expected"],
        check_totals=check_totals,
        label=LABEL,
    )


def checkpoint_rename_map(checkpoint: Path) -> dict[str, str]:
    """Read the camera rename map from the checkpoint's own train_config.json.

    The map has to match the one training used or every camera feeds a
    different policy input slot than it did during training, and the only
    symptom is a policy that behaves as if it had never been trained. Taking
    it from the checkpoint makes that agreement structural rather than
    something two config files have to be kept in sync about: this file is
    written by the very run that produced these weights.
    """
    config_path = checkpoint / "train_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"{config_path} not found -- cannot tell which camera fed which input slot "
            "during training. Point --checkpoint at a directory written by lerobot-train."
        )
    return json.loads(config_path.read_text()).get("rename_map") or {}


def _require_smolvla_deps() -> None:
    """SmolVLA's processor imports num2words at load time.

    lerobot's `pi` extra (what this repo installs) does not pull it in, so a
    missing num2words surfaces as an ImportError several minutes into a train
    run rather than at install time. Fail here instead, with the fix.
    """
    try:
        import num2words  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "SmolVLA needs num2words (and accelerate). Install them with:\n"
            '  pip install -e ".[smolvla]"'
        ) from exc


def run_prepare(args: argparse.Namespace, profile: dict[str, Any]) -> None:
    validate_sources(profile)
    argv = build_prepare_argv(profile, overwrite=args.overwrite)
    if args.dry_run:
        _print_command(["python", "-m", "evo_rlt.cli.merge_lerobot_datasets", *argv])
        return
    from evo_rlt.cli.merge_lerobot_datasets import main as merge_main

    previous = sys.argv
    try:
        sys.argv = ["evo-rlt-merge", *argv]
        merge_main()
    finally:
        sys.argv = previous
    _check_dataset(profile)


def run_check(args: argparse.Namespace, profile: dict[str, Any]) -> None:
    _require_smolvla_deps()
    if args.sources:
        validate_sources(profile)
        return
    root = _repo_path(profile["merged_root"])
    episodes, frames = _check_dataset(profile)
    print(f"[OK] SmolVLA-ready dataset: {root} ({episodes} episodes, {frames} frames)")


def _validate_smolvla_checkpoint(path: Path) -> None:
    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"SmolVLA checkpoint config not found: {config_path}")
    config = json.loads(config_path.read_text())
    if config.get("type") != "smolvla":
        raise ValueError(f"Checkpoint is type {config.get('type')!r}, expected 'smolvla': {path}")
    for required in ("policy_preprocessor.json", "policy_postprocessor.json"):
        if not (path / required).is_file():
            raise FileNotFoundError(f"Incomplete SmolVLA checkpoint; missing {path / required}")


def run_rollout(args: argparse.Namespace, profile: dict[str, Any]) -> None:
    _require_smolvla_deps()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not args.dry_run:
        _validate_smolvla_checkpoint(checkpoint)
    record_argv = [
        "full",
        "--initial-source",
        "vla",
        "--policy-path",
        str(checkpoint),
        "--sim",
        args.sim_endpoint,
        "--setup-json",
        str(_repo_path(args.setup_json)),
        "--task",
        profile["task"],
        "--dataset-tag",
        args.dataset_tag,
        "--num-episodes",
        str(args.num_episodes),
        "--episode-time-s",
        str(args.episode_time_s),
        "--reset-time-s",
        str(args.reset_time_s),
        "--fps",
        str(profile["expected"]["fps"]),
        "--rename-map",
        json.dumps(checkpoint_rename_map(checkpoint), separators=(",", ":")),
        "--no-teleop",
        # RTC constrains each new chunk to continue the actions still queued
        # from the previous one, instead of drawing an independent sample. On
        # this checkpoint two draws from one observation differ about as much
        # as either differs from the demonstration, so re-planning without it
        # swaps trajectories mid-motion -- the step at a chunk boundary
        # measures 2.6x the steps inside a chunk. --no-rtc restores the old
        # behaviour for an A/B against earlier evaluations.
        "--rtc" if args.rtc else "--no-rtc",
    ]
    if args.dry_run:
        record_argv.append("--dry-run")
    from evo_rlt.adapters.lerobot.record.cli import main as record_main

    record_main(record_argv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Validate and merge the nine blue-screw sessions.")
    prepare.add_argument("--overwrite", action="store_true")
    prepare.add_argument("--dry-run", action="store_true")
    prepare.set_defaults(func=run_prepare)

    check = sub.add_parser("check", help="Run SmolVLA schema, totals, success-label and dependency checks.")
    check.add_argument("--sources", action="store_true", help="Check the nine sources instead of merged output.")
    check.set_defaults(func=run_check)

    rollout = sub.add_parser("rollout", help="Load a fine-tuned SmolVLA checkpoint in the current simulator.")
    rollout.add_argument("--checkpoint", type=Path, required=True)
    rollout.add_argument("--sim-endpoint", default="tcp://127.0.0.1:5555")
    rollout.add_argument("--setup-json", default="configs/my_so101_manifest.json")
    rollout.add_argument("--dataset-tag", default="smolvla_blue_screw_eval")
    rollout.add_argument("--num-episodes", type=int, default=10)
    # Wall-clock, not sim time: the record loop ends an episode on
    # `time.perf_counter() - start > episode_time_s` while the simulator
    # advances a fixed 1/fps per step. A policy that runs the loop at 9 Hz only
    # gets episode_time_s * 9 / 30 seconds of sim time, so this default is
    # deliberately generous -- see README_SMOLVLA.md.
    rollout.add_argument("--episode-time-s", type=int, default=150)
    rollout.add_argument("--reset-time-s", type=int, default=3)
    rollout.add_argument(
        "--rtc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Real-Time Chunking: guide each new action chunk with the actions still "
            "queued from the previous one. On by default; --no-rtc reproduces the "
            "independent-sampling behaviour earlier evaluations were run with."
        ),
    )
    rollout.add_argument("--dry-run", action="store_true")
    rollout.set_defaults(func=run_rollout)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    profile = load_profile(args.profile)
    args.func(args, profile)


if __name__ == "__main__":
    main()
