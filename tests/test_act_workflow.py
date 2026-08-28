from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from evo_rlt.cli.act import (
    DEFAULT_PROFILE,
    build_prepare_argv,
    load_profile,
    validate_dataset_root,
)
from evo_rlt.cli.act import REPO_ROOT
from evo_rlt.cli.merge_lerobot_datasets import discover_datasets


def _write_dataset(root: Path, *, outcome: str = "success") -> dict:
    expected = {
        "robot_type": "sim_bi_so_follower",
        "episodes": 1,
        "frames": 10,
        "fps": 30,
        "state_dim": 12,
        "action_dim": 12,
        "image_features": [
            "observation.images.left_wrist",
            "observation.images.right_wrist",
            "observation.images.right_front",
        ],
        "image_shape": [480, 640, 3],
        "episode_success": "success",
    }
    names = [f"joint_{i}" for i in range(12)]
    features = {
        "observation.state": {"dtype": "float32", "shape": [12], "names": names},
        "action": {"dtype": "float32", "shape": [12], "names": names},
    }
    for key in expected["image_features"]:
        features[key] = {"dtype": "video", "shape": [480, 640, 3]}
    info = {
        "robot_type": expected["robot_type"],
        "total_episodes": 1,
        "total_frames": 10,
        "fps": 30,
        "features": features,
    }
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(info))
    pq.write_table(
        pa.table({"episode_success": [outcome]}),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    return expected


def test_blue_screw_profile_pins_only_blue_sessions() -> None:
    profile = load_profile(DEFAULT_PROFILE)
    assert len(profile["source_sessions"]) == 9
    assert profile["source_sessions"][0].endswith("162809")
    assert profile["expected"]["episodes"] == 122
    assert profile["expected"]["frames"] == 83247


def test_validate_act_dataset_checks_schema_and_success(tmp_path: Path) -> None:
    expected = _write_dataset(tmp_path)
    assert validate_dataset_root(tmp_path, expected, check_totals=True) == (1, 10)


def test_validate_act_dataset_rejects_failure_episode(tmp_path: Path) -> None:
    expected = _write_dataset(tmp_path, outcome="failure")
    with pytest.raises(ValueError, match="episode outcomes"):
        validate_dataset_root(tmp_path, expected, check_totals=True)


def test_train_config_pins_contact_safe_replanning() -> None:
    """`lerobot-train --config_path` reads this file; nothing in the CLI does.

    n_action_steps is the only value here that differs from ACTConfig's own
    default (100), and it is the load-bearing one: 100 means 3.33 s of
    open-loop execution, far too long for a contact-sensitive insertion.
    """
    cfg = json.loads((REPO_ROOT / "configs" / "act" / "train_config.json").read_text())
    assert cfg["policy"]["type"] == "act"
    assert cfg["policy"]["chunk_size"] == 100
    assert cfg["policy"]["n_action_steps"] == 10
    assert cfg["policy"]["push_to_hub"] is False


def test_train_config_uses_the_profile_dataset() -> None:
    """Both policies must train on the episodes the profile pins, or the two
    success rates are not comparable."""
    profile = load_profile(DEFAULT_PROFILE)
    cfg = json.loads((REPO_ROOT / "configs" / "act" / "train_config.json").read_text())
    assert cfg["dataset"]["repo_id"] == profile["repo_id"]
    assert cfg["dataset"]["root"] == profile["merged_root"]


def test_prepare_argv_lists_exact_sessions() -> None:
    profile = load_profile(DEFAULT_PROFILE)
    argv = build_prepare_argv(profile, overwrite=False)
    includes = [argv[i + 1] for i, value in enumerate(argv) if value == "--include"]
    assert includes == profile["source_sessions"]
    assert "record_teleop_full_132606" not in includes


def test_discover_datasets_honors_explicit_order(tmp_path: Path) -> None:
    for name in ("a", "b", "c"):
        (tmp_path / name / "meta").mkdir(parents=True)
        (tmp_path / name / "meta" / "info.json").write_text("{}")
    roots = discover_datasets(tmp_path, ["c", "a"])
    assert [root.name for root in roots] == ["c", "a"]


def test_discover_datasets_rejects_missing_include(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_datasets(tmp_path, ["missing"])
