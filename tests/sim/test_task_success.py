"""自动成功判据。

盯的是"判据本身会不会说谎":增广和脚本采集会成千上万条地依赖它,判错一次
不会报错,只会往数据集里塞一条失败的演示 —— 而失败的演示比没有演示更糟。

几何常量来自 ``configs/task_scene.json``,构造用例时直接读它,这样改了孔径或
杆长之后测试跟着动,不会留下一组和场景对不上的硬编码数字。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from evo_rlt.sim import task_success as ts


@pytest.fixture(scope="module")
def config() -> dict:
    return ts.load_config()


def _upright(position, yaw: float = 0.0) -> list[float]:
    return [*position, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def _bolt_over_socket(config, socket_xyz, depth, lateral=0.0, yaw=0.0):
    """构造"螺栓杆尖插入螺套 ``depth`` 米"的一对位姿。

    螺栓 euler [pi,0,0] 即杆朝下,四元数 ``[0,1,0,0]``;此时杆尖在 body 系 +z
    方向、世界系 -z 方向,故杆尖世界 z = bolt_z - tip_z。
    """
    mouth = float(config["socket"]["hole_mouth_z"])
    tip = float(config["bolt"]["tip_z"])
    tip_z = socket_xyz[2] + mouth - depth
    return {
        "socket": _upright(socket_xyz, yaw),
        "bolt": [socket_xyz[0] + lateral, socket_xyz[1], tip_z + tip, 0.0, 1.0, 0.0, 0.0],
    }


def test_config_has_every_threshold(config):
    for key in ("socket", "bolt", "inserted", "aligned", "table_top_z"):
        assert key in config, key
    assert config["socket"]["hole_radius"] > config["bolt"]["shank_radius"], "孔必须比杆粗"


def test_inserted_when_deep_and_centred(config):
    depth = float(config["inserted"]["min_depth"]) + 0.005
    state = ts.evaluate(_bolt_over_socket(config, [0.30, 0.0, 0.20], depth), config)
    assert state.inserted and state.stage == "inserted"
    assert state.depth == pytest.approx(depth)
    assert state.lateral == pytest.approx(0.0, abs=1e-9)


def test_not_inserted_when_too_shallow(config):
    depth = float(config["inserted"]["min_depth"]) - 0.002
    state = ts.evaluate(_bolt_over_socket(config, [0.30, 0.0, 0.20], depth), config)
    assert not state.inserted


def test_not_inserted_when_laterally_off(config):
    """孔口平面以下但横向偏出去 —— 几何上不可能在孔里,判据必须挡住。

    这条对应真实失败:螺栓从螺套旁边擦过去时深度是正的,只有横向那一项能
    分开"插进去了"和"从边上过去了"。
    """
    depth = float(config["inserted"]["min_depth"]) + 0.005
    lateral = float(config["inserted"]["max_lateral"]) + 0.002
    state = ts.evaluate(_bolt_over_socket(config, [0.30, 0.0, 0.20], depth, lateral), config)
    assert not state.inserted
    assert state.lateral == pytest.approx(lateral)


def test_not_inserted_when_axes_disagree(config):
    """两轴夹角过大 —— 实测的主要失败模式,不是横偏。

    重放里见过螺套只倾 1.8 度而螺栓倾 13.3 度、横偏只有 0.38mm 的情形:
    横向判据完全通过,可杆根本插不进去。
    """
    poses = _bolt_over_socket(config, [0.30, 0.0, 0.20], 0.012)
    angle = math.radians(float(config["inserted"]["max_angle_deg"]) + 10.0)
    # 绕 x 轴把螺栓再转一个角度(原本是 [0,1,0,0],即绕 x 转 180 度)
    poses["bolt"] = [
        *poses["bolt"][:3],
        math.cos((math.pi + angle) / 2), math.sin((math.pi + angle) / 2), 0.0, 0.0,
    ]
    state = ts.evaluate(poses, config)
    assert state.angle_deg > float(config["inserted"]["max_angle_deg"])
    assert not state.inserted


def test_lateral_offset_points_from_hole_axis_to_tip(config):
    """偏移向量的方向必须对:解析修正整个建立在它身上,反了就会越修越偏。"""
    lateral = 0.004
    state = ts.evaluate(
        _bolt_over_socket(config, [0.30, 0.0, 0.20], 0.005, lateral), config
    )
    assert state.lateral_offset[0] == pytest.approx(lateral)
    assert state.lateral_offset[1] == pytest.approx(0.0, abs=1e-9)
    assert state.lateral_offset[2] == pytest.approx(0.0, abs=1e-9)


def test_lateral_offset_follows_socket_yaw(config):
    """螺套转了,偏移向量也要跟着转到世界系 —— 它是在螺套系里量的。"""
    lateral = 0.004
    yaw = math.pi / 2
    state = ts.evaluate(
        _bolt_over_socket(config, [0.30, 0.0, 0.20], 0.005, lateral, yaw=yaw), config
    )
    # 螺套绕 z 转 90 度后,螺套系的 +x 指向世界 +y;但杆尖是在世界 +x 上偏的,
    # 所以螺套系里的偏移是 -y,换回世界系仍是 +x。模长不变是核心不变量。
    assert np.linalg.norm(state.lateral_offset) == pytest.approx(lateral, abs=1e-9)


def test_stage_ladder_reports_progress(config):
    rest = float(config["socket"]["rest_z"])
    table = float(config["table_top_z"])
    idle = ts.evaluate(
        {"socket": _upright([0.35, -0.06, rest]), "bolt": [0.32, 0.085, 0.1324, 0, 1, 0, 0]},
        config,
    )
    assert idle.stage == "idle"
    lifted = ts.evaluate(
        {
            "socket": _upright([0.35, -0.06, rest + float(config["socket_lifted_z"]) + 0.01]),
            "bolt": [0.32, 0.085, 0.1324, 0, 1, 0, 0],
        },
        config,
    )
    assert lifted.socket_lifted and lifted.stage == "socket_lifted"
    assert lifted.stage_index > idle.stage_index
    assert table > 0.0


def test_episode_needs_a_sustained_insertion(config):
    """一帧就算的话,穿模弹开的瞬间会被记成成功。"""
    deep = ts.evaluate(_bolt_over_socket(config, [0.30, 0.0, 0.20], 0.015), config)
    shallow = ts.evaluate(_bolt_over_socket(config, [0.30, 0.0, 0.20], 0.001), config)
    assert not ts.episode_succeeded([shallow] * 50 + [deep] + [shallow] * 50, hold_frames=10)
    assert ts.episode_succeeded([shallow] * 5 + [deep] * 10 + [shallow] * 5, hold_frames=10)


def test_furthest_stage_is_the_high_water_mark(config):
    deep = ts.evaluate(_bolt_over_socket(config, [0.30, 0.0, 0.20], 0.015), config)
    idle = ts.evaluate(
        {"socket": _upright([0.35, -0.06, 0.1145]), "bolt": [0.32, 0.085, 0.1324, 0, 1, 0, 0]},
        config,
    )
    assert ts.furthest_stage([idle, deep, idle]) == "inserted"
    assert ts.furthest_stage([idle, idle]) == "idle"


def test_missing_object_is_an_error(config):
    with pytest.raises(ts.TaskSuccessError, match="socket"):
        ts.evaluate({"bolt": [0, 0, 0, 1, 0, 0, 0]}, config)


def test_malformed_pose_is_an_error(config):
    with pytest.raises(ts.TaskSuccessError):
        ts.evaluate({"socket": [0, 0, 0], "bolt": [0, 0, 0, 1, 0, 0, 0]}, config)
