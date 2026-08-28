"""把已有的人类演示增广成更多条 —— MimicGen 式的物体中心重放。

**问题。** 蓝色螺栓那批仿真数据只有 122 条 episode。人再采一批的成本是线性的,
而这个任务里真正随机的只有一件事:螺套复位时落在圆形凹槽里的位置和朝向
(``configs/task_scene.json`` 的 ``socket.reset_random``)。螺栓的初始位姿是固定
的。也就是说,122 条演示是对一个三维随机量(x, y, yaw)的 122 次采样。

**做法。** 对每条源演示,把它重放到一个**新的螺套位姿**上:抓取之前的那段末端
轨迹整体平移,抓到之后连着零件一起平移,于是插入动作发生在工作空间的另一个
位置。人类演示里最难的那一段 —— 对准和插入 —— 逐帧照抄,不是脚本编出来的。

**为什么位移只取平移。** SO-101 只有 5 个本体关节,够不到任意 6D 位姿,实测
缺的那一维几乎纯粹是**绕世界 z 轴的偏航**,残差 0.135 度/毫米平移。若把螺套的
偏航差也做成末端目标,30 度偏航等价于要求手臂多走 ~200mm(绕螺套中心转和绕
底座转差一个 ``(I-Rz)(c-b)`` 的平移),IK 直接崩掉。所以:

* 位移只做平移,IK 用"位置精确、偏航放开"的权重解;
* 抓取那一帧解完之后,**反过来**读回夹爪实际到达的位姿,再由它推出螺套该摆在
  哪 —— 偏航残差就这样被吸收进"螺套摆哪"里。螺套的偏航本来就是均匀随机的,
  被残差改掉几度不损失任何东西。

**偏航残差为什么不影响插入。** 螺套的孔轴就是它自己的 z 轴,螺栓杆也是轴对称
的;绕各自竖直轴转几度,孔轴和杆轴都不动。残差只影响"钳口相对六角面正不正",
而那一项由上面的反推补掉了。

**为什么还需要成功过滤。** 上面每一步都可能不成立:新位姿够不着、平移后夹爪
撞到台面、六角被钳口推歪。重放完拿 :mod:`evo_rlt.sim.task_success` 判一次,
不成功的直接丢 —— 失败的演示比没有演示更糟。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from evo_rlt.sim.calib import ARM_SIDES, MOTOR_NAMES

#: 夹爪在 ``JOINT_ORDER`` 里的下标(左、右)。
GRIPPER_INDEX = {side: idx * len(MOTOR_NAMES) + MOTOR_NAMES.index("gripper")
                 for idx, side in enumerate(ARM_SIDES)}

#: 一条臂的 5 个本体关节在 12 维向量里的下标。
BODY_INDEX = {
    side: [idx * len(MOTOR_NAMES) + j for j, name in enumerate(MOTOR_NAMES) if name != "gripper"]
    for idx, side in enumerate(ARM_SIDES)
}


class AugmentError(RuntimeError):
    """增广过程中数据本身不成立时抛出(分段失败、标定发散等)。"""


# -- 分段 -------------------------------------------------------------------


@dataclass(frozen=True)
class Segments:
    """一条 episode 里两只夹爪各自的抓/放帧号。"""

    right_close: int
    right_open: int
    left_close: int
    left_open: int
    length: int


def find_closed_span(gripper: Sequence[float], warmup: int = 5) -> tuple[int, int] | None:
    """夹爪信号里**最长的一段闭合**,返回 ``(闭合帧, 张开帧)``。

    取最长一段而不是第一次跌破阈值:开头几帧的数值是遥操还没接上时的残值
    (实测有整条 episode 以 0 起头的),而人中途也会有试探性的半闭合。真正
    "抓着零件"的那段一定是最长的。``warmup`` 跳过开头那几帧。
    """
    signal = np.asarray(gripper, dtype=float)[warmup:]
    if signal.size == 0:
        return None
    low, high = np.percentile(signal, 5), np.percentile(signal, 95)
    if high - low < 4.0:  # 全程没开合过
        return None
    closed = signal < low + 0.45 * (high - low)
    best_len = best_start = best_end = 0
    run = 0
    for i, is_closed in enumerate(closed):
        if is_closed:
            run += 1
            continue
        if run > best_len:
            best_len, best_start, best_end = run, i - run, i
        run = 0
    if run > best_len:
        best_len, best_start, best_end = run, len(closed) - run, len(closed)
    if best_len == 0:
        return None
    return best_start + warmup, best_end + warmup


def segment_episode(actions: np.ndarray) -> Segments:
    """按两只夹爪的开合把一条 episode 切成抓取/搬运/插入几段。

    ``actions`` 是 ``(T, 12)`` 的原始电机值(未转弧度) —— 夹爪那一维是 0..100 的
    开度百分比,阈值判据直接在这上面做最稳。
    """
    right = find_closed_span(actions[:, GRIPPER_INDEX["right"]])
    left = find_closed_span(actions[:, GRIPPER_INDEX["left"]])
    if right is None or left is None:
        raise AugmentError("夹爪信号里找不到成段的闭合,这条 episode 无法分段")
    return Segments(
        right_close=right[0], right_open=right[1],
        left_close=left[0], left_open=left[1], length=len(actions),
    )


# -- 螺套初始位姿的反推 -----------------------------------------------------


@dataclass
class GraspCalibration:
    """螺套相对右夹爪 ``gripper_link`` 的固定位姿(抓稳之后)。

    源数据里**没有记录螺套的初始位姿** —— 采集时零件位姿从没送出过仿真进程。
    但抓取那一帧夹爪在哪是知道的(关节角做 FK),而人每次都以大致相同的方式
    去抓那个六棱柱,于是"螺套原点在夹爪坐标系里的位置"是个常量,标定出来就能
    反推每条演示当时的螺套在哪。

    标定用的是位置那一半:螺套的 z 恒为凹槽底(它就立在那儿),xy 均匀落在半径
    25mm 的圆盘内且以圆盘中心为期望 —— 所以让"推出来的螺套位置"整体最贴近圆盘
    中心的那个 ``translation`` 就是最小二乘解。解完必须核对残差:若推出的 xy
    确实像一个 25mm 均匀圆盘(分位数对得上),说明抓取帧找对了、常量假设成立。
    """

    translation: np.ndarray                  # 螺套原点在夹爪系下的位置 (3,)
    rest_z: float                            # 螺套静止时的世界 z
    disk_center: np.ndarray                  # 凹槽中心 (2,)
    disk_radius: float
    residual_radius: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def socket_position(self, ee_pos: np.ndarray, ee_rot: np.ndarray) -> np.ndarray:
        """由夹爪位姿推出螺套原点的世界位置。"""
        return np.asarray(ee_pos, dtype=float) + np.asarray(ee_rot, dtype=float) @ self.translation

    def to_dict(self) -> dict[str, Any]:
        return {
            "translation": [float(v) for v in self.translation],
            "rest_z": float(self.rest_z),
            "disk_center": [float(v) for v in self.disk_center],
            "disk_radius": float(self.disk_radius),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "GraspCalibration":
        return cls(
            translation=np.asarray(raw["translation"], dtype=float),
            rest_z=float(raw["rest_z"]),
            disk_center=np.asarray(raw["disk_center"], dtype=float),
            disk_radius=float(raw["disk_radius"]),
        )


def fit_grasp_calibration(
    ee_positions: np.ndarray,
    ee_rotations: np.ndarray,
    disk_center: Sequence[float],
    disk_radius: float,
    rest_z: float,
) -> GraspCalibration:
    """最小二乘拟合螺套在夹爪系下的位置。

    目标:``min_t  Σ_i || p_i + R_i t - c ||²``,其中 ``c`` 是凹槽中心加静止高度。
    真实螺套位置以 ``c`` 为中心对称分布,所以这个解是无偏的;若换成"只让 z 对
    上"的一维约束,方程会秩亏(各帧的 R_i 只差一个小偏航),解出来是最小范数的
    垃圾值 —— 那条弯路走过,推出的螺套落在离凹槽 106mm 的地方。
    """
    positions = np.asarray(ee_positions, dtype=float)
    rotations = np.asarray(ee_rotations, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise AugmentError(f"ee_positions 形状应为 (N,3),给了 {positions.shape}")
    if rotations.shape != (len(positions), 3, 3):
        raise AugmentError(f"ee_rotations 形状应为 (N,3,3),给了 {rotations.shape}")

    target = np.array([disk_center[0], disk_center[1], rest_z], dtype=float)
    design = rotations.reshape(-1, 3)
    rhs = (np.tile(target, (len(positions), 1)) - positions).reshape(-1)
    translation, *_ = np.linalg.lstsq(design, rhs, rcond=None)

    implied = positions + np.einsum("nij,j->ni", rotations, translation)
    residual = np.linalg.norm(implied[:, :2] - np.asarray(disk_center, dtype=float), axis=1)
    return GraspCalibration(
        translation=translation,
        rest_z=float(rest_z),
        disk_center=np.asarray(disk_center, dtype=float),
        disk_radius=float(disk_radius),
        residual_radius=residual,
    )


def disk_quantiles(radius: float) -> dict[str, float]:
    """半径 ``radius`` 的均匀圆盘上,到圆心距离的理论分位数。

    标定的自检基准:推出来的螺套位置若真是那个随机化产生的,它到凹槽中心的
    距离就该服从这个分布(``P(r<x) = x²/R²``,故 ``q_p = R·sqrt(p)``)。
    """
    return {f"q{int(p * 100):02d}": radius * math.sqrt(p) for p in (0.25, 0.5, 0.75, 0.9)}


# -- 姿态:由夹爪推出螺套的朝向 ---------------------------------------------


def rotation_between(source: Sequence[float], target: Sequence[float]) -> np.ndarray:
    """把单位向量 ``source`` 转到 ``target`` 的最小转角旋转矩阵。"""
    a = np.asarray(source, dtype=float)
    b = np.asarray(target, dtype=float)
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    b = b / max(float(np.linalg.norm(b)), 1e-12)
    axis = np.cross(a, b)
    sin = float(np.linalg.norm(axis))
    cos = float(np.dot(a, b))
    if sin < 1e-9:
        if cos > 0:
            return np.eye(3)
        # 反向:绕任一垂直轴转 180 度
        seed = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, seed)
        axis /= np.linalg.norm(axis)
        skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + 2.0 * skew @ skew
    axis /= sin
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + sin * skew + (1.0 - cos) * skew @ skew


def rotation_from_vector(vector: Sequence[float]) -> np.ndarray:
    """轴×角向量 -> 旋转矩阵(罗德里格斯)。"""
    vec = np.asarray(vector, dtype=float)
    angle = float(np.linalg.norm(vec))
    if angle < 1e-12:
        return np.eye(3)
    axis = vec / angle
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * skew @ skew


def quaternion_from_matrix(rot: np.ndarray) -> list[float]:
    """3x3 旋转矩阵 -> ``[w, x, y, z]``。"""
    trace = float(rot[0, 0] + rot[1, 1] + rot[2, 2])
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = [
            0.25 * scale,
            (rot[2, 1] - rot[1, 2]) / scale,
            (rot[0, 2] - rot[2, 0]) / scale,
            (rot[1, 0] - rot[0, 1]) / scale,
        ]
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        scale = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        quat = [
            (rot[2, 1] - rot[1, 2]) / scale, 0.25 * scale,
            (rot[0, 1] + rot[1, 0]) / scale, (rot[0, 2] + rot[2, 0]) / scale,
        ]
    elif rot[1, 1] > rot[2, 2]:
        scale = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        quat = [
            (rot[0, 2] - rot[2, 0]) / scale, (rot[0, 1] + rot[1, 0]) / scale,
            0.25 * scale, (rot[1, 2] + rot[2, 1]) / scale,
        ]
    else:
        scale = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        quat = [
            (rot[1, 0] - rot[0, 1]) / scale, (rot[0, 2] + rot[2, 0]) / scale,
            (rot[1, 2] + rot[2, 1]) / scale, 0.25 * scale,
        ]
    norm = math.sqrt(sum(v * v for v in quat))
    return [v / norm for v in quat]


def socket_up_axis_in_gripper(ee_rotations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """螺套的竖直轴在右夹爪坐标系里的方向,以及各帧相对它的偏离角(度)。

    螺套永远是立着的,所以它的 z 轴在世界系里就是 ``e_z``;换到夹爪系就是
    ``R_ee^T e_z``。人每次都以同样的姿势去抓,这个向量因此近乎常量 —— 偏离角
    的分布正好用来验证这个假设(实测中位数 5.3 度,是人手的抖动量级)。
    """
    rotations = np.asarray(ee_rotations, dtype=float)
    axes = np.einsum("nji,j->ni", rotations, np.array([0.0, 0.0, 1.0]))
    mean = axes.mean(axis=0)
    mean /= max(float(np.linalg.norm(mean)), 1e-12)
    spread = np.degrees(np.arccos(np.clip(axes @ mean, -1.0, 1.0)))
    return mean, spread


def gripper_yaw(ee_rotation: np.ndarray, up_axis: Sequence[float]) -> float:
    """夹爪当前对应的螺套偏航角(弧度),尚未加上全局偏置 ``yaw_offset``。

    ``R_ee @ R_a`` 的 z 轴按构造就是世界 ``e_z``,所以它是一个纯偏航,角度直接从
    第一列读出来。
    """
    aligned = np.asarray(ee_rotation, dtype=float) @ rotation_between([0.0, 0.0, 1.0], up_axis)
    return float(math.atan2(aligned[1, 0], aligned[0, 0]))


def yaw_quaternion(yaw: float) -> list[float]:
    """绕世界 z 轴转 ``yaw`` 的四元数 ``[w, x, y, z]``。"""
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def socket_pose_from_grasp(
    ee_pos: Sequence[float],
    ee_rot: np.ndarray,
    calibration: GraspCalibration,
    up_axis: Sequence[float],
    yaw_offset: float,
) -> list[float]:
    """由抓取那一帧的右夹爪位姿,推出螺套该摆在哪(7 维位姿)。

    z 强制取 ``rest_z``:螺套是立在凹槽底上的,那个高度由几何决定,不该由标定
    残差决定 —— 让它浮起来或者陷进去都会在复位时被物理弹开。
    """
    position = calibration.socket_position(np.asarray(ee_pos, dtype=float), ee_rot)
    yaw = gripper_yaw(ee_rot, up_axis) + yaw_offset
    return [float(position[0]), float(position[1]), calibration.rest_z, *yaw_quaternion(yaw)]


# -- 位移排程与末端轨迹变换 -------------------------------------------------


def smoothstep(x: np.ndarray | float) -> np.ndarray:
    """``3x²-2x³``,在 0 和 1 两端一阶导为零。

    用平滑过渡而不是线性:位移是加在位置指令上的,线性斜坡在起止两点有速度
    突变,30 Hz 下表现为可见的顿挫,而这些帧是要拿去当演示的。
    """
    t = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _ramp(length: int, start: int, stop: int) -> np.ndarray:
    """长度 ``length`` 的序列,在 ``[start, stop)`` 内由 0 平滑升到 1,之后保持 1。"""
    out = np.zeros(length)
    if stop <= start:
        out[max(start, 0):] = 1.0
        return out
    idx = np.arange(length)
    out = smoothstep((idx - start) / float(stop - start))
    out[idx < start] = 0.0
    out[idx >= stop] = 1.0
    return out


def displacement_schedule(
    segments: Segments, lift_frame: int, bridge_frames: int = 30
) -> dict[str, np.ndarray]:
    """每条臂逐帧的位移权重(0=照抄源轨迹,1=整体位移 Δ)。

    右臂:从起点平滑升到 1,在**抓取前**就到位并保持 —— 最后一段接近必须是源
    轨迹的干净平移,否则夹爪是斜着扑上去的。抓到之后一直保持 1,螺套就被搬到
    了新位置,插入动作因此发生在工作空间的另一处(这才是增广的价值所在)。

    左臂:螺栓的初始位姿是固定的,所以拔出来之前必须逐帧照抄(权重 0);
    ``lift_frame`` 之后再平滑升到 1,去和被搬走的螺套会合。

    两条斜坡都不与"抓"或"拔"的瞬间重叠:位移在接触期间变化会把零件从钳口里
    蹭出去。
    """
    length = segments.length
    right_hold = max(1, int(0.7 * segments.right_close))
    settle = segments.right_close + 5
    return {
        "right": _ramp(length, 0, right_hold),
        "left": _ramp(length, lift_frame, lift_frame + max(1, bridge_frames)),
        # 抓稳之后才起效的那一路,给"握持修正"用 —— 它必须完全避开抓取瞬间,
        # 在钳口正在合拢时平移末端会把零件蹭出去。
        "hold": _ramp(length, settle, settle + max(1, bridge_frames)),
    }


def offset_ee_targets(
    ee_poses: Sequence[Sequence[float]],
    offsets: np.ndarray,
    rotations: np.ndarray | None = None,
    pivot_in_gripper: Sequence[float] | None = None,
) -> list[list[float]]:
    """逐帧变换一条末端位姿轨迹。

    ``offsets`` 是逐帧位移。``rotations`` 是逐帧的 轴×角 向量,**绕握着的零件转**
    而不是绕世界原点转 —— 支点取零件在夹爪系里的位置 ``pivot_in_gripper``
    (就是标定出来的那个常量),这样转动只改零件的朝向,不把它甩到别处去。
    """
    offsets = np.asarray(offsets, dtype=float)
    pivot = None if pivot_in_gripper is None else np.asarray(pivot_in_gripper, dtype=float)
    out = []
    for index, (pose, offset) in enumerate(zip(ee_poses, offsets)):
        position, rot = pose_matrix(pose)
        if rotations is not None:
            turn = rotation_from_vector(rotations[index])
            if pivot is not None:
                # 支点是零件,不是夹爪原点:p' = pivot + turn·(p - pivot),
                # 而 pivot = p + rot·pivot_in_gripper,代入即下式。
                arm = rot @ pivot
                position = position + arm - turn @ arm
            rot = turn @ rot
        out.append([*(position + offset), *quaternion_from_matrix(rot)])
    return [[float(v) for v in pose] for pose in out]


def translate_ee_targets(
    ee_poses: Sequence[Sequence[float]], weights: np.ndarray, delta: Sequence[float]
) -> list[list[float]]:
    """把一条末端位姿轨迹按逐帧权重平移,姿态原样保留。

    只平移不转:见模块开头 —— 偏航差会让 5 自由度的 IK 崩掉,而它在这个任务里
    可以通过"改摆零件的朝向"来吸收。
    """
    offset = np.asarray(delta, dtype=float)
    out = []
    for pose, weight in zip(ee_poses, weights):
        pose = np.asarray(pose, dtype=float)
        shifted = pose.copy()
        shifted[:3] = pose[:3] + weight * offset
        out.append([float(v) for v in shifted])
    return out


# -- 源数据读取 -------------------------------------------------------------


@dataclass
class SourceEpisode:
    """一条源演示:原始电机值 + 切好的段。"""

    index: int
    actions: np.ndarray      # (T, 12) LeRobot 电机值
    states: np.ndarray       # (T, 12) 同上,实测位置
    segments: Segments
    task: str = ""

    def __len__(self) -> int:
        return len(self.actions)


def read_source_episodes(root: str | Path, task: str = "") -> list[SourceEpisode]:
    """读一个 LeRobot 数据集里的全部 episode。

    直接读 parquet 而不是走 ``LeRobotDataset``:这里只要 ``action`` /
    ``observation.state`` 两列,而 ``LeRobotDataset`` 会连视频索引一起建起来,
    对 122 条 × 83k 帧是白花几十秒。
    """
    from pyarrow import parquet as pq

    root = Path(root)
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise AugmentError(f"{root} 下没有找到 data/**.parquet")

    episodes: list[SourceEpisode] = []
    for path in files:
        table = pq.read_table(
            path, columns=["action", "observation.state", "episode_index"]
        )
        actions = np.asarray(table.column("action").to_pylist(), dtype=np.float64)
        states = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float64)
        index = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
        for episode_index in np.unique(index):
            mask = index == episode_index
            try:
                segments = segment_episode(actions[mask])
            except AugmentError:
                continue  # 分不出段的丢掉,后面统计里会报出来
            episodes.append(
                SourceEpisode(
                    index=int(episode_index),
                    actions=actions[mask],
                    states=states[mask],
                    segments=segments,
                    task=task,
                )
            )
    if not episodes:
        raise AugmentError(f"{root} 里没有一条 episode 能分段")
    return episodes


def to_radians(rows: np.ndarray, bridge, joint_order: Sequence[str]) -> np.ndarray:
    """``(T, 12)`` 电机值 -> ``(T, 12)`` 关节角(弧度),按 ``joint_order`` 排列。

    ``bridge`` 是 :class:`evo_rlt.sim.calib.BimanualCalibration`。这里不直接
    import 它是为了让本模块也能被仿真进程按路径加载(那边没有 torch,
    ``evo_rlt`` 包导不进去)。
    """
    keys = [f"{name}.pos" for name in joint_order]
    out = np.empty((len(rows), len(joint_order)), dtype=np.float64)
    for i, row in enumerate(rows):
        rads = bridge.action_to_rad({key: float(v) for key, v in zip(keys, row)})
        out[i] = [rads[name] for name in joint_order]
    return out


def to_motor_values(rads: np.ndarray, bridge, joint_order: Sequence[str]) -> np.ndarray:
    """:func:`to_radians` 的逆。"""
    out = np.empty((len(rads), len(joint_order)), dtype=np.float64)
    for i, row in enumerate(rads):
        values = bridge.rad_to_observation({name: float(v) for name, v in zip(joint_order, row)})
        out[i] = [values[f"{name}.pos"] for name in joint_order]
    return out


def pose_matrix(pose: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """7 维位姿 -> ``(位置, 3x3 旋转)``。"""
    from evo_rlt.sim.task_success import _quat_matrix

    pose = np.asarray(pose, dtype=float)
    return pose[:3].copy(), _quat_matrix(pose[3:])


# -- 规划一条增广 episode ---------------------------------------------------


@dataclass
class Plan:
    """一条增广 episode 的全部输入:摆哪、发什么指令。"""

    source_index: int
    socket_pose: list[float]          # 复位时把螺套摆到这里
    actions: np.ndarray               # (T, 12) LeRobot 电机值,逐帧 send_action
    delta: np.ndarray                 # 实际用的平移量 (3,)
    lift_frame: int                   # 左臂开始跟着位移的帧
    ik_pos_err_max: float             # IK 位置残差上界(米)
    ik_yaw_err_deg: float             # 抓取帧的偏航残差(度),已被摆件朝向吸收

    def __len__(self) -> int:
        return len(self.actions)


def _wrap_pi(angle: float) -> float:
    """把角度折回 ``(-pi, pi]``。"""
    return math.atan2(math.sin(angle), math.cos(angle))


def _first_lift_frame(
    ee_poses: Sequence[Sequence[float]], close_frame: int, clearance: float
) -> int:
    """螺栓被提起 ``clearance`` 之后的第一帧。

    左臂在这之前必须逐帧照抄:螺栓是**插在台面孔里**的固定初始位姿,给指令加
    平移会让夹爪去够一个空位,或者把还没拔出来的螺栓往侧面掰。
    """
    base = float(ee_poses[close_frame][2])
    for index in range(close_frame, len(ee_poses)):
        if float(ee_poses[index][2]) - base >= clearance:
            return index
    return min(close_frame + 1, len(ee_poses) - 1)


def plan_episode(
    robot,
    episode: SourceEpisode,
    calibration: GraspCalibration,
    up_axis: Sequence[float],
    yaw_offset: float,
    delta: Sequence[float],
    joint_order: Sequence[str],
    hold_correction: Sequence[float] | None = None,
    hold_rotation: Sequence[float] | None = None,
    bridge_frames: int = 30,
    lift_clearance: float = 0.04,
    rotation_weight: float | None = None,
) -> Plan:
    """把一条源演示搬到"螺套平移了 ``delta``"的场景上。

    ``robot`` 是连着仿真器的 :class:`~evo_rlt.sim.sim_robot.SimRobot`,只用它的
    ``fk`` / ``ik``,不动仿真状态。

    ``hold_rotation`` 是同样只在抓稳之后加的一个常量转动(轴×角,世界系),绕握着
    的螺套转,用来补掉"零件在钳口里的**倾斜**"和源演示当时那一次的差别 —— 实测
    横向偏移修好之后,剩下的失败几乎全是两轴夹角(典型"横偏 0.11mm 夹角 19.1°")。

    ``hold_correction`` 是**抓稳之后**只加在右臂上的一个常量平移,用来补掉"零件
    在钳口里的实际位置"和源演示当时那一次的差别。这个差别没法从摆件位姿上修:
    钳口合拢时会把六角重新坐正,摆哪儿它都被推到大致同一个地方 —— 实测只改摆
    件位姿,重放成功率从 32% 只能推到 42%。而直接平移握着零件的那只手是精确的:
    零件被刚性握着,手走多少它就走多少。修正量由一次不录的重放量出来
    (:func:`socket_pose_correction` 量的就是它)。
    """
    delta = np.asarray(delta, dtype=float)
    correction = np.zeros(3) if hold_correction is None else np.asarray(hold_correction, dtype=float)
    turn = np.zeros(3) if hold_rotation is None else np.asarray(hold_rotation, dtype=float)
    rads = to_radians(episode.actions, robot.calibration_bridge, joint_order)
    poses = robot.fk(rads.tolist())
    ee = {side: [frame[side] for frame in poses] for side in ARM_SIDES}

    lift_frame = _first_lift_frame(ee["left"], episode.segments.left_close, lift_clearance)
    weights = displacement_schedule(episode.segments, lift_frame, bridge_frames)

    # 每条臂逐帧的总位移:右臂另外叠一路只在抓稳之后起效的握持修正。
    offsets = {
        side: np.outer(weights[side], delta) for side in ARM_SIDES
    }
    offsets["right"] = offsets["right"] + np.outer(weights["hold"], correction)
    turns = {"right": np.outer(weights["hold"], turn), "left": None}

    solved = rads.copy()
    ik_pos_err = 0.0
    ik_kwargs = {} if rotation_weight is None else {"rotation_weight": rotation_weight}
    for side in ARM_SIDES:
        active = np.linalg.norm(offsets[side], axis=1)
        if turns[side] is not None:
            active = active + np.linalg.norm(turns[side], axis=1)
        moving = np.flatnonzero(active > 0.0)
        if moving.size == 0:
            continue
        start = int(moving[0])
        targets = offset_ee_targets(
            ee[side][start:],
            offsets[side][start:],
            rotations=None if turns[side] is None else turns[side][start:],
            pivot_in_gripper=calibration.translation if side == "right" else None,
        )
        result = robot.ik(side, targets, rads[start].tolist(), **ik_kwargs)
        solved[start:, BODY_INDEX[side]] = np.asarray(result["qpos"], dtype=float)
        ik_pos_err = max(ik_pos_err, float(np.max(result["pos_err"])))

    # 抓取那一帧夹爪**实际**到达的位姿 —— 不是我们要求的那个。5 自由度做不到
    # 任意 6D 位姿,差的那点几乎全在绕竖直轴的偏航上。螺套就按实际位姿反推着
    # 摆,残差于是被"零件摆哪"吸收掉,而不是留成一个抓偏的抓取。
    grasp_frame = episode.segments.right_close
    achieved = robot.fk([solved[grasp_frame].tolist()])[0]["right"]
    achieved_pos, achieved_rot = pose_matrix(achieved)
    socket_pose = socket_pose_from_grasp(
        achieved_pos, achieved_rot, calibration, up_axis, yaw_offset
    )
    # 位移是纯平移,所以"想要的"姿态就是源姿态;两者的偏航之差即残差。
    wanted_rot = pose_matrix(ee["right"][grasp_frame])[1]
    yaw_err = math.degrees(
        abs(
            _wrap_pi(gripper_yaw(achieved_rot, up_axis) - gripper_yaw(wanted_rot, up_axis))
        )
    )

    actions = to_motor_values(solved, robot.calibration_bridge, joint_order)
    # 夹爪不进 IK(它不影响 gripper_link 的位姿),原样照抄源指令。
    for side in ARM_SIDES:
        actions[:, GRIPPER_INDEX[side]] = episode.actions[:, GRIPPER_INDEX[side]]
    return Plan(
        source_index=episode.index,
        socket_pose=socket_pose,
        actions=actions,
        delta=delta,
        lift_frame=lift_frame,
        ik_pos_err_max=ik_pos_err,
        ik_yaw_err_deg=yaw_err,
    )


def sample_delta(
    rng: np.random.Generator,
    calibration: GraspCalibration,
    source_socket_xy: Sequence[float],
    max_delta: float,
) -> np.ndarray:
    """给一条源演示抽一个位移:目标位姿在凹槽圆盘内均匀采样。

    在圆盘内采**目标**而不是直接采位移,是为了让增广出来的螺套位置分布和真实
    随机化一致 —— 直接采位移会让分布向圆盘中心堆积(两个均匀圆盘之差的和)。
    ``max_delta`` 是位移上限:平移越大,5 自由度带来的偏航残差越大
    (0.135 度/毫米),摆件朝向虽然能吸收它,但夹爪相对台面的姿态也跟着变,
    过大就会蹭到台面。超限就重采。
    """
    source = np.asarray(source_socket_xy, dtype=float)
    for _ in range(64):
        radius = calibration.disk_radius * math.sqrt(rng.random())
        angle = rng.uniform(0.0, 2.0 * math.pi)
        target = calibration.disk_center + radius * np.array([math.cos(angle), math.sin(angle)])
        delta = target - source
        if float(np.linalg.norm(delta)) <= max_delta:
            return np.array([delta[0], delta[1], 0.0])
    return np.zeros(3)


# -- 用重放本身把螺套位姿标准确 ---------------------------------------------


def closest_approach(
    states: Sequence, window: float = 0.05, max_angle_deg: float = 30.0
) -> int | None:
    """杆尖离孔最近、且确实处在"准备插入"姿态的那一帧。

    三道闸都是必需的:

    * 两件都已拿起 —— 否则量到的是两个零件各自摆在台面上的距离;
    * 杆尖在孔口 ``window`` 米以内 —— 搬运途中它可能从螺套旁边扫过去;
    * 两轴夹角小于 ``max_angle_deg`` —— 这道最容易漏。实测有帧夹角 89 度却
      "很近",那是螺栓横躺着从孔口边上过,此时把杆尖投影到孔平面上得到的
      横向偏移毫无意义,拿它去修正会把轨迹带得更偏(实测 0.70mm 修成 2.06mm)。
    """
    best_index, best_lateral = None, float("inf")
    for index, state in enumerate(states):
        if not (state.bolt_pulled and state.socket_lifted):
            continue
        if abs(state.depth) > window or state.angle_deg > max_angle_deg:
            continue
        if state.lateral < best_lateral:
            best_index, best_lateral = index, state.lateral
    return best_index


def socket_pose_correction(
    states: Sequence,
    right_rotations: Sequence[np.ndarray],
    grasp_frame: int,
) -> np.ndarray | None:
    """由一次重放的对不准量,解析地反推该把螺套往哪挪。

    螺套从抓起那一刻起就被夹爪刚性带着走,所以插入时刻它相对源演示的偏差,就是
    复位时摆件误差经 ``A = R(t_插) · R(t_抓)ᵀ`` 这个已知旋转的像::

        miss ≈ -A · (摆的位置 - 真实位置)   =>   真实位置 ≈ 摆的位置 + Aᵀ · miss

    于是一次重放就能把摆件误差标出来,不用迭代搜索。这一步是必需的:源数据里
    螺套的初始位姿从没被记录过,只能由抓取时的夹爪位姿反推,而人每次抓的相对
    位置有毫米级的抖动 —— 实测未修正时纯重放只有 32% 能复现成功,失败的那些
    横向偏移正好落在 0.8~3.7mm,和单边间隙 0.45mm 一比就知道是它。
    """
    index = closest_approach(states)
    if index is None:
        return None
    miss = np.asarray(states[index].lateral_offset, dtype=float)
    transform = np.asarray(right_rotations[index], dtype=float) @ np.asarray(
        right_rotations[grasp_frame], dtype=float
    ).T
    correction = transform.T @ miss
    # 只修 xy:螺套立在凹槽底上,z 由几何定死,改它只会让复位时物理把它弹开。
    return np.array([correction[0], correction[1], 0.0])


def socket_tilt_correction(states: Sequence) -> np.ndarray | None:
    """由一次重放量出"握着的螺套该再转多少"(轴×角,世界系)。

    横向偏移修好之后剩下的失败几乎全是这一项:螺套在钳口里的倾斜和源演示当时
    那一次不同。实测成功的那条在插入瞬间螺套倾 12.6 度、螺栓倾 12.7 度、两轴
    夹角 0.1 度 —— 人是把两件一起斜着对上的;失败的那条螺套只倾 1.8 度而螺栓
    13.3 度,横偏只有 0.38mm,却怎么也插不进去。

    修在手上是精确的(零件被刚性握着),而且这个转动是绕水平轴的 —— 正好不是
    SO-101 缺的那一维(缺的是绕竖直轴的偏航),分轴权重的 IK 能精确跟随。
    """
    index = closest_approach(states)
    if index is None:
        return None
    return np.asarray(states[index].axis_offset, dtype=float)


# -- 重放 -------------------------------------------------------------------


def replay(
    robot,
    actions: np.ndarray,
    socket_pose: Sequence[float],
    joint_order: Sequence[str],
    success_config: dict[str, Any],
    settle_frames: int = 40,
    on_frame=None,
) -> list:
    """把一条指令轨迹放到仿真里跑一遍,返回逐帧的任务状态。

    ``settle_frames``:重放之前先把手臂开到轨迹起点并保持若干帧。整体 reset 会
    把手臂弹回 home 姿态,而源演示的第一帧不一定在那儿 —— 不先开过去的话,第一帧
    就是一个大台阶,手臂会甩着把零件扫飞。这几帧不进数据集。

    ``on_frame(index, observation, action)`` 给上层挂数据集写入用;为 None 时
    只跑不录,标定和成功过滤走的就是这条路(不渲染额外的观测,快一倍)。
    """
    from evo_rlt.sim.task_success import evaluate

    keys = [f"{name}.pos" for name in joint_order]
    robot.reset()
    robot.reset_objects(poses={"socket": [float(v) for v in socket_pose]})

    first = {key: float(value) for key, value in zip(keys, actions[0])}
    for _ in range(settle_frames):
        robot.send_action(first)

    states = []
    for index, row in enumerate(actions):
        action = {key: float(value) for key, value in zip(keys, row)}
        observation = robot.get_observation() if on_frame is not None else None
        sent = robot.send_action(action)
        if on_frame is not None:
            on_frame(index, observation, sent)
        states.append(evaluate(robot.object_poses, success_config))
    return states
