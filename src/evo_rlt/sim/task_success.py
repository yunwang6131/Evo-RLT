"""插销任务的自动成功判据 —— 从零件真值位姿判断这一帧任务进行到哪一步。

在此之前 ``episode_success`` 只有人工标注一条路:采集时按 s / f,122 条源数据
就是这么标出来的。人标不了增广和脚本采集 —— 那两条路一次产出上千条 episode,
没有自动判据就只能全部当成功收下,而失败的演示比没有演示更糟。

**为什么判据只看位姿、不看接触。** 螺套的孔半径 5.20mm、螺栓杆半径 4.75mm,
单边间隙 0.45mm(几何取自场景网格实测,见 ``configs/task_scene.json`` 的
``success`` 段)。杆尖一旦越过孔口平面且横向偏移在间隙量级,它在几何上就只能
在孔里 —— 孔壁不可能被穿过。于是"插入深度 + 横向偏移 + 轴线夹角"三个量就够,
不必去数接触点。接触判据反而脆:凸分解后的孔壁是 200 多块凸包,接触对的数量
随分解参数变。

**milestone 不是附赠品。** rollout 全失败时,"完全没学会"和"学会了但最后对不
准"要做的事完全不同 —— 前者加多少数据都没用。这两种情况在成功率上都是 0%,
只有分阶段的判据能把它们分开,所以 :func:`evaluate` 一次把四级进度都算出来。

只依赖 numpy:它要能在客户端(评测/回放)和仿真进程(诊断脚本)两边跑,而后者
装不了 torch。
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]

#: 判据参数。刻意**不放进** ``configs/task_scene.json`` —— 那个文件连同整套仿真
#: 资产被 ``evo-rlt-sim-snapshot`` 按字节封存,是"122 条演示当时的那个环境"的
#: 定义;判据是事后评判用的旋钮,改它不该让快照校验失败,更不该让人误以为环境
#: 变了。这里的几何常量是从场景网格量出来的副本,改了场景要跟着核对。
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "task_success.json"

#: 进度阶梯,从低到高。``stage`` 取当前成立的最高一级。
STAGES: tuple[str, ...] = ("idle", "socket_lifted", "bolt_pulled", "aligned", "inserted")

#: 判定 episode 成功所需的连续插入帧数。一帧就算的话,穿模弹开的瞬间会被
#: 记成成功;30 Hz 下 10 帧 = 0.33 秒,足够排除瞬态又不至于漏掉真的插入。
DEFAULT_HOLD_FRAMES = 10


class TaskSuccessError(ValueError):
    """判据配置缺失或零件位姿不全。"""


@dataclass(frozen=True)
class TaskState:
    """一帧的任务状态。

    ``depth`` / ``lateral`` / ``angle_deg`` 都是"螺栓杆尖相对螺套孔"的量,
    在螺套 body 系里算:depth 从孔口往下为正,lateral 是到孔轴的距离,
    angle_deg 是杆轴与孔轴的夹角(插入方向为 0)。
    """

    stage: str
    inserted: bool
    aligned: bool
    socket_lifted: bool
    bolt_pulled: bool
    depth: float
    lateral: float
    angle_deg: float
    #: 世界系下"从孔轴指向杆尖"的向量。``lateral`` 是它的模。
    #:
    #: 保留方向而不只是大小,是因为重放的对不准可以**解析地**修回去:零件被夹爪
    #: 刚性带着走,手走多少它就走多少,所以量到的错位直接就是该给手臂加的位移
    #: (见 ``evo_rlt.sim.augment.socket_pose_correction``)。
    lateral_offset: tuple[float, float, float]
    #: 世界系下的转动向量(轴×角,弧度):把**螺套的孔轴**转到**螺栓的插入方向**
    #: 所需的那个转动。``angle_deg`` 是它的模。
    #:
    #: 和 ``lateral_offset`` 同理,方向是为了能反过来修:实测重放失败的主因不是
    #: 横向偏移(那个能收敛到 0.1mm)而是两轴夹角 —— 螺套在钳口里的倾斜和源演示
    #: 当时那一次不同。修正办法是把握着螺套的那只手整体转一下,而不是挪一下。
    axis_offset: tuple[float, float, float]

    @property
    def stage_index(self) -> int:
        return STAGES.index(self.stage)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """读 ``configs/task_success.json``。"""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise TaskSuccessError(f"找不到判据配置 {path}")
    config = json.loads(path.read_text())
    missing = sorted({"socket", "bolt", "inserted", "aligned"} - config.keys())
    if missing:
        raise TaskSuccessError(f"{path} 缺少 {missing}")
    return config


def _quat_matrix(quat: Sequence[float]) -> np.ndarray:
    """``[w, x, y, z]`` 单位四元数 -> 3x3 旋转矩阵(列是 body 系各轴的世界方向)。"""
    w, x, y, z = (float(v) for v in quat)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        raise TaskSuccessError("四元数是零向量")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _rotation_vector(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """把单位向量 ``source`` 转到 ``target`` 的最小转动,写成 轴×角 向量。"""
    axis = np.cross(source, target)
    sin = float(np.linalg.norm(axis))
    cos = float(np.dot(source, target))
    angle = math.atan2(sin, cos)
    if sin < 1e-12:
        return np.zeros(3)
    return axis / sin * angle


def _split(pose: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    if len(pose) != 7:
        raise TaskSuccessError(f"位姿要 7 个数 [x,y,z,qw,qx,qy,qz],给了 {len(pose)}")
    return np.asarray(pose[:3], dtype=float), _quat_matrix(pose[3:])


def evaluate(
    object_poses: dict[str, Sequence[float]], config: dict[str, Any] | None = None
) -> TaskState:
    """按这一帧的零件位姿算任务状态。

    ``object_poses`` 就是 :attr:`evo_rlt.sim.sim_robot.SimRobot.object_poses`
    的格式:``{"socket": [x,y,z,qw,qx,qy,qz], "bolt": [...]}``。
    """
    config = config if config is not None else load_config()
    for name in ("socket", "bolt"):
        if name not in object_poses:
            raise TaskSuccessError(f"缺少零件 {name} 的位姿;拿到的是 {sorted(object_poses)}")

    socket_cfg, bolt_cfg = config["socket"], config["bolt"]
    socket_pos, socket_rot = _split(object_poses["socket"])
    bolt_pos, bolt_rot = _split(object_poses["bolt"])

    # 杆尖世界坐标,再换到螺套 body 系 —— 孔轴就是那个系的 +z,换过去之后
    # "深度/横向"就是简单的分量,不必再处理螺套自己的倾斜。
    tip_world = bolt_pos + bolt_rot @ np.array([0.0, 0.0, float(bolt_cfg["tip_z"])])
    tip_local = socket_rot.T @ (tip_world - socket_pos)

    depth = float(socket_cfg["hole_mouth_z"]) - float(tip_local[2])
    lateral = float(np.linalg.norm(tip_local[:2]))
    lateral_world = socket_rot @ np.array([tip_local[0], tip_local[1], 0.0])
    # 杆的 +z 是"头指向尖";插入时它与孔轴(螺套 +z)反向,故取负号后再求夹角。
    hole_axis = socket_rot[:, 2]
    insert_axis = -bolt_rot[:, 2]
    cos_axis = float(np.dot(hole_axis, insert_axis))
    angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_axis))))
    axis_world = _rotation_vector(hole_axis, insert_axis)

    ins = config["inserted"]
    inserted = (
        depth >= float(ins["min_depth"])
        and lateral <= float(ins["max_lateral"])
        and angle_deg <= float(ins["max_angle_deg"])
    )
    ali = config["aligned"]
    # depth 为负 = 杆尖还在孔口上方;-max_height <= depth <= 0 即"悬在孔口上方"。
    aligned = inserted or (
        -float(ali["max_height"]) <= depth <= 0.0
        and lateral <= float(ali["max_lateral"])
        and angle_deg <= float(ali["max_angle_deg"])
    )

    socket_lifted = bool(
        socket_pos[2] - float(socket_cfg["rest_z"]) >= float(config["socket_lifted_z"])
    )
    # 螺栓杆尖高过台面 = 已经从孔里拔出来了。初始时杆尖穿在台面板以下 6.5mm。
    bolt_pulled = bool(
        tip_world[2] >= float(config["table_top_z"]) + float(config["bolt_pulled_clearance"])
    )

    if inserted:
        stage = "inserted"
    elif aligned:
        stage = "aligned"
    elif bolt_pulled:
        stage = "bolt_pulled"
    elif socket_lifted:
        stage = "socket_lifted"
    else:
        stage = "idle"

    return TaskState(
        stage=stage,
        inserted=inserted,
        aligned=aligned,
        socket_lifted=socket_lifted,
        bolt_pulled=bolt_pulled,
        depth=depth,
        lateral=lateral,
        angle_deg=angle_deg,
        lateral_offset=(
            float(lateral_world[0]),
            float(lateral_world[1]),
            float(lateral_world[2]),
        ),
        axis_offset=(float(axis_world[0]), float(axis_world[1]), float(axis_world[2])),
    )


def episode_succeeded(
    states: Iterable[TaskState], hold_frames: int = DEFAULT_HOLD_FRAMES
) -> bool:
    """连续 ``hold_frames`` 帧都判为已插入才算这条 episode 成功。

    要求"连续"而不是"曾经出现过":插入的瞬间如果是穿模再弹开,单帧判据会把它
    记成成功,而那条演示教给策略的是一次失败的碰撞。
    """
    run = 0
    for state in states:
        run = run + 1 if state.inserted else 0
        if run >= hold_frames:
            return True
    return False


def furthest_stage(states: Iterable[TaskState]) -> str:
    """整条 episode 走到过的最高阶段 —— 失败时用它定位卡在哪一步。"""
    best = 0
    for state in states:
        best = max(best, state.stage_index)
    return STAGES[best]
