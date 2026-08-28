"""增广数据集的列必须和源数据集逐字一致。

增广出来的 episode 是要和源数据 **合并后一起训** 的。列名、dtype、shape、names
差一个字,炸点在几十分钟后的 merge 或 train 预检里,而不是在写数据的时候 ——
所以这里在写之前就守住。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evo_rlt.cli.augment import REPO_ROOT, _dataset_features

SOURCE = REPO_ROOT / "data" / "bimanual" / "blue_screw_sim_v1"

pytestmark = pytest.mark.skipif(
    not (SOURCE / "meta" / "info.json").is_file(),
    reason="源数据集不在这台机器上",
)


def test_features_drop_only_the_columns_lerobot_adds_itself():
    from lerobot.datasets.utils import DEFAULT_FEATURES

    info = json.loads((SOURCE / "meta" / "info.json").read_text())
    features = _dataset_features(SOURCE)
    assert set(features) == set(info["features"]) - set(DEFAULT_FEATURES)
    for key, spec in features.items():
        assert spec == info["features"][key], key


def test_created_dataset_reproduces_the_source_schema(tmp_path: Path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    want = json.loads((SOURCE / "meta" / "info.json").read_text())
    LeRobotDataset.create(
        "local/schema_probe",
        want["fps"],
        features=_dataset_features(SOURCE),
        root=tmp_path / "ds",
        robot_type=want["robot_type"],
        use_videos=True,
    )
    got = json.loads((tmp_path / "ds" / "meta" / "info.json").read_text())
    assert list(got["features"]) == list(want["features"])
    for key, spec in want["features"].items():
        assert got["features"][key] == spec, key
    assert got["robot_type"] == want["robot_type"]
    assert got["fps"] == want["fps"]
