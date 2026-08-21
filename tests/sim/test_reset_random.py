"""复位随机化的配置契约。

采样行为本身要跑 MuJoCo(见 SimulatorState._randomize),这里守的是配置里的
几何不变量 —— 那几个数一改,零件就会压在凹槽边沿上初始就歪掉,而现象是
"偶尔有几条 episode 一开始零件就是斜的",很难反查到配置。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_SCENE = REPO_ROOT / "configs" / "task_scene.json"

#: 射线扫台面实测:圆形凹槽中心 (0.3500, -0.0623),最大半径 38.9mm,
#: 面积/圆面积 0.97(确实是圆),凹面 z=115.8mm。
RECESS_CENTRE = (0.3500, -0.0623)
RECESS_RADIUS = 0.0389

#: 螺套六角的外接圆半径(网格包围盒 x ±13mm)。
SOCKET_CIRCUMRADIUS = 0.013


@pytest.fixture(scope="module")
def task():
    return json.loads(TASK_SCENE.read_text())


def test_socket_has_a_random_region(task):
    """没有它,每条 episode 的起点都一样,VLA 只会记住那一个位置。"""
    spec = task["socket"]["reset_random"]
    assert spec["radius"] > 0
    assert len(spec["center"]) == 2


def test_random_region_keeps_the_socket_inside_the_recess(task):
    """圆心可动范围 + 零件半径不能超出凹槽,否则零件会骑在凹槽壁上。"""
    spec = task["socket"]["reset_random"]
    cx, cy = spec["center"]
    offset = math.hypot(cx - RECESS_CENTRE[0], cy - RECESS_CENTRE[1])
    reach = offset + float(spec["radius"]) + SOCKET_CIRCUMRADIUS
    assert reach <= RECESS_RADIUS, (
        f"最远能到距凹槽中心 {reach * 1000:.1f}mm,超过凹槽半径 "
        f"{RECESS_RADIUS * 1000:.1f}mm —— 零件会压在边沿上"
    )


def test_random_region_tracks_the_table(task):
    """桌子挪过位(x 0.20 -> 0.30),随机区域必须跟着挪。

    只改 table.pos 而忘了这里的话,零件会被复位到桌子外面的空中。
    """
    table_x = task["table"]["pos"][0]
    centre_x = task["socket"]["reset_random"]["center"][0]
    # 凹槽在桌心前方 5cm 处(台面特征相对桌心是固定的)
    assert centre_x == pytest.approx(table_x + 0.05, abs=0.005)


def test_nominal_socket_pose_is_inside_the_random_region(task):
    """keyframe 里那个落稳位姿本身也该在可随机区域内。

    不在的话,``--random-seed`` 关掉随机化时用的起点和开着时的分布对不上,
    两种模式采出来的数据不可比。
    """
    spec = task["socket"]["reset_random"]
    pos = task["socket"]["pos"]
    offset = math.hypot(pos[0] - spec["center"][0], pos[1] - spec["center"][1])
    assert offset <= spec["radius"]
