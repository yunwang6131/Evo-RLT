"""Prepare, validate, and roll out ACT on the blue-screw simulation data.

Training is not wrapped here: `lerobot-train --config_path=configs/act/train_config.json`
does it directly. A wrapper around it would only fix argument values -- which
that json already does -- while hiding every lerobot flag it does not
re-export (`--resume` was lost that way). What stays here is the part lerobot
has no notion of: this rig's dataset preflight and the simulator rollout.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROFILE = REPO_ROOT / "configs" / "blue_screw_sim_v1.json"


def _repo_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_profile(path: Path) -> dict[str, Any]:
    """Load a dataset profile.

    A profile describes the *dataset* only -- sources, merged root, task
    string, expected shape -- and is policy-independent, which is what lets
    both CLIs share one preflight and guarantees the two policies are judged
    on the same episodes. Training hyper-parameters live in
    `configs/<policy>/train_config.json` and are read by `lerobot-train`
    directly; nothing in this module needs them.
    """
    profile = json.loads(path.expanduser().read_text())
    if profile.get("schema_version") != 1:
        raise ValueError(f"Unsupported profile schema in {path}")
    required = {"source_parent", "source_sessions", "merged_root", "repo_id", "task", "expected"}
    missing = sorted(required - profile.keys())
    if missing:
        raise ValueError(f"Profile is missing fields: {missing}")
    return profile


def source_roots(profile: dict[str, Any]) -> list[Path]:
    parent = _repo_path(profile["source_parent"])
    return [parent / name for name in profile["source_sessions"]]


def _read_info(root: Path) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot metadata not found: {info_path}")
    return json.loads(info_path.read_text())


def _episode_outcomes(root: Path) -> Counter[str]:
    from pyarrow import parquet as pq

    outcomes: Counter[str] = Counter()
    episode_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"Episode metadata not found under {root / 'meta' / 'episodes'}")
    for path in episode_files:
        table = pq.read_table(path, columns=["episode_success"])
        outcomes.update(value for value in table.column("episode_success").to_pylist() if value is not None)
    return outcomes


def validate_dataset_root(
    root: Path,
    expected: dict[str, Any],
    *,
    check_totals: bool,
    check_outcomes: bool = True,
    label: str = "ACT",
) -> tuple[int, int]:
    info = _read_info(root)
    errors: list[str] = []
    if info.get("robot_type") != expected["robot_type"]:
        errors.append(f"robot_type={info.get('robot_type')!r}, expected {expected['robot_type']!r}")
    if info.get("fps") != expected["fps"]:
        errors.append(f"fps={info.get('fps')}, expected {expected['fps']}")

    features = info.get("features", {})
    state = features.get("observation.state", {})
    action = features.get("action", {})
    if state.get("shape") != [expected["state_dim"]]:
        errors.append(f"observation.state shape={state.get('shape')}")
    if action.get("shape") != [expected["action_dim"]]:
        errors.append(f"action shape={action.get('shape')}")
    if state.get("names") != action.get("names"):
        errors.append("observation.state and action joint names/order differ")
    actual_images = sorted(key for key, spec in features.items() if spec.get("dtype") == "video")
    wanted_images = sorted(expected["image_features"])
    if actual_images != wanted_images:
        errors.append(f"image features={actual_images}, expected {wanted_images}")
    for key in wanted_images:
        if features.get(key, {}).get("shape") != expected["image_shape"]:
            errors.append(f"{key} shape={features.get(key, {}).get('shape')}")

    episodes = int(info.get("total_episodes", 0))
    frames = int(info.get("total_frames", 0))
    if episodes <= 0 or frames <= 0:
        errors.append(f"empty dataset ({episodes} episodes/{frames} frames)")
    if check_totals and (episodes != expected["episodes"] or frames != expected["frames"]):
        errors.append(
            f"totals={episodes} episodes/{frames} frames, expected "
            f"{expected['episodes']}/{expected['frames']}"
        )
    if check_outcomes:
        outcomes = _episode_outcomes(root)
        required_outcome = expected.get("episode_success")
        if required_outcome and outcomes != Counter({required_outcome: episodes}):
            errors.append(f"episode outcomes={dict(outcomes)}, expected all {required_outcome!r}")
    if errors:
        raise ValueError(f"{label} dataset preflight failed for {root}:\n  - " + "\n  - ".join(errors))
    return episodes, frames


def validate_sources(profile: dict[str, Any]) -> tuple[int, int]:
    total_episodes = 0
    total_frames = 0
    for root in source_roots(profile):
        episodes, frames = validate_dataset_root(root, profile["expected"], check_totals=False)
        total_episodes += episodes
        total_frames += frames
        print(f"[OK] {root.name}: {episodes} episodes, {frames} frames")
    expected = profile["expected"]
    if (total_episodes, total_frames) != (expected["episodes"], expected["frames"]):
        raise ValueError(
            f"Source totals are {total_episodes} episodes/{total_frames} frames, expected "
            f"{expected['episodes']}/{expected['frames']}"
        )
    print(f"[OK] source total: {total_episodes} episodes, {total_frames} frames")
    return total_episodes, total_frames


def build_prepare_argv(profile: dict[str, Any], *, overwrite: bool) -> list[str]:
    argv = [
        "--input-parent",
        str(_repo_path(profile["source_parent"])),
        "--output-root",
        str(_repo_path(profile["merged_root"])),
        "--output-repo-id",
        profile["repo_id"],
        "--repo-id-prefix",
        f"local/{profile['name']}_source_",
    ]
    for name in profile["source_sessions"]:
        argv += ["--include", name]
    if overwrite:
        argv.append("--overwrite")
    return argv


def _print_command(argv: list[str]) -> None:
    import shlex

    separator = " " + "\\" + "\n  "
    print(separator.join(shlex.quote(arg) for arg in argv))


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
    validate_dataset_root(_repo_path(profile["merged_root"]), profile["expected"], check_totals=True)


def run_check(args: argparse.Namespace, profile: dict[str, Any]) -> None:
    if args.sources:
        validate_sources(profile)
        return
    root = _repo_path(profile["merged_root"])
    episodes, frames = validate_dataset_root(root, profile["expected"], check_totals=True)
    print(f"[OK] ACT-ready dataset: {root} ({episodes} episodes, {frames} frames)")


def _validate_act_checkpoint(path: Path) -> None:
    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"ACT checkpoint config not found: {config_path}")
    config = json.loads(config_path.read_text())
    if config.get("type") != "act":
        raise ValueError(f"Checkpoint is type {config.get('type')!r}, expected 'act': {path}")
    for required in ("policy_preprocessor.json", "policy_postprocessor.json"):
        if not (path / required).is_file():
            raise FileNotFoundError(f"Incomplete ACT checkpoint; missing {path / required}")


def run_rollout(args: argparse.Namespace, profile: dict[str, Any]) -> None:
    checkpoint = args.checkpoint.expanduser().resolve()
    if not args.dry_run:
        _validate_act_checkpoint(checkpoint)
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
        "--no-teleop",
        "--no-rtc",
    ]
    if args.dry_run:
        record_argv.append("--dry-run")
    from evo_rlt.adapters.lerobot.record.cli import main as record_main

    record_main(record_argv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Validate and merge the nine blue-screw sessions.")
    prepare.add_argument("--overwrite", action="store_true")
    prepare.add_argument("--dry-run", action="store_true")
    prepare.set_defaults(func=run_prepare)

    check = sub.add_parser("check", help="Run ACT schema, totals, and success-label preflight checks.")
    check.add_argument("--sources", action="store_true", help="Check the nine sources instead of merged output.")
    check.set_defaults(func=run_check)

    rollout = sub.add_parser("rollout", help="Load a trained ACT checkpoint in the current simulator.")
    rollout.add_argument("--checkpoint", type=Path, required=True)
    rollout.add_argument("--sim-endpoint", default="tcp://127.0.0.1:5555")
    rollout.add_argument("--setup-json", default="configs/my_so101_manifest.json")
    rollout.add_argument("--dataset-tag", default="act_blue_screw_eval")
    rollout.add_argument("--num-episodes", type=int, default=10)
    rollout.add_argument("--episode-time-s", type=int, default=45)
    rollout.add_argument("--reset-time-s", type=int, default=3)
    rollout.add_argument("--dry-run", action="store_true")
    rollout.set_defaults(func=run_rollout)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    profile = load_profile(args.profile)
    args.func(args, profile)


if __name__ == "__main__":
    main()
