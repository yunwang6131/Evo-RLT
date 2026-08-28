"""Wire protocol between the record loop and the simulator process.

This module is deliberately dependency-free -- no torch, no lerobot, no mujoco --
because it is imported by *both* sides of a process boundary that exists
precisely because those dependency sets do not coexist. The simulator runs on
its own interpreter (MuJoCo today, Isaac Lab later, potentially on another
machine); this file is the only thing they must agree on.

The boundary is drawn at **radians**. Everything about LeRobot's calibrated
motor values -- the DEGREES mid-point offset, the gripper's 0..100 percentage,
the per-arm tick ranges -- is resolved on the client side by
`evo_rlt.sim.calib`. The simulator is handed plain joint angles and hands back
plain joint angles, so swapping the physics backend cannot reintroduce a
calibration bug, and the simulator needs no knowledge of the real hardware.

Messages are JSON (small, self-describing, easy to debug over the wire); camera
frames travel separately as raw bytes to keep them out of the JSON encoder.
"""

from __future__ import annotations

import math

# Kept in sync with `evo_rlt.sim.calib.MOTOR_NAMES` by
# `tests/sim/test_protocol.py`. Duplicated rather than imported so the
# simulator process never has to import the (torch-dependent) evo_rlt package.
JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
ARM_SIDES: tuple[str, ...] = ("left", "right")

#: Prefixed joint names, ordered. This is the canonical ordering for every
#: position vector on the wire, so both sides must never sort independently.
JOINT_ORDER: tuple[str, ...] = tuple(
    f"{side}_{joint}" for side in ARM_SIDES for joint in JOINT_NAMES
)

#: Camera keys, matching the real rig's recorded observation keys.
CAMERA_KEYS: tuple[str, ...] = ("left_wrist", "right_wrist", "right_front")

#: Joint limits the scene builder enforces, radians. Mirrors
#: `evo_rlt.sim.calib.SIM_JOINT_LIMITS`; kept in sync by
#: `tests/sim/test_protocol.py`. These must match the client's clip, or MuJoCo
#: stops a joint short of what the bridge commands.
SIM_JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-1.91986, 1.91986),
    "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-1.65806, 1.65806),
    "wrist_roll": (-2 * math.pi, 2 * math.pi),
    # 见 calib.GRIPPER_CLOSED_RAD:URDF 的 -10 度并非真正贴合位置
    "gripper": (math.radians(-13.5), math.radians(-13.5 + 136.4)),
}

DEFAULT_ENDPOINT = "tcp://127.0.0.1:5555"
DEFAULT_IMAGE_WIDTH = 640
DEFAULT_IMAGE_HEIGHT = 480
DEFAULT_CONTROL_HZ = 30.0
DEFAULT_PHYSICS_HZ = 500.0

#: 指令纯延迟步数 @30Hz。
#:
#: 真机总滞后约 135 ms(纯延迟 50 ms + 时间常数 85 ms);仿真在 kp=50 下时间
#: 常数只有 33 ms,故用 3 帧(100 ms)纯延迟补齐,合计 133 ms。
#:
#: 刻意用纯延迟而非压低增益来对齐:低增益会带来真机没有的重力下垂(真机舵机
#: 的 PID 有积分项补偿),kp=20 时 0.67 度、kp=5 时 6.5 度。
DEFAULT_ACTION_DELAY_STEPS = 3

#: Client gives up on a request after this long, so a wedged or dead simulator
#: surfaces as an error in the record loop instead of an indefinite hang.
DEFAULT_TIMEOUT_S = 5.0

#: 位姿在线上的编码:``[x, y, z, qw, qx, qy, qz]``,世界系,米 + 单位四元数。
#: 和 MuJoCo 的 ``qpos`` 里自由关节的排布一致,两边都不必再换序。
POSE_LEN = 7

#: 各臂末端参考系。取 ``gripper_link`` 而不是钳口中点:它是 URDF 里真实存在的
#: body,FK/IK 两边指的是同一个东西;抓取点相对它的偏移是任务侧的事,由调用者
#: 自己带(见 ``evo_rlt.sim.grasp_frame``)。
EE_BODIES: dict[str, str] = {side: f"{side}_gripper_link" for side in ARM_SIDES}

#: IK 的姿态权重,**按世界坐标轴分开给**(位置权重固定为 1)。
#:
#: SO-101 只有 5 个本体关节,够不到任意 6D 位姿。缺的是哪一维不是玄学,是实测
#: 出来的:对一个纯平移目标解 IK,姿态残差的转轴 z 分量恒为 0.94,大小随平移
#: 线性增长(0.135 度/毫米)—— **缺的就是绕世界 z 轴的偏航**。
#:
#: 于是正确的提法是:位置 3 维 + 工具倾角 2 维 = 5,和自由度数正好相等,是一个
#: 恰定问题;只把绕 z 的那一维放开。所以 x/y 分量给 1、z 分量给 0.02。
#:
#: 早先用的是各向同性的小权重(三个都 0.02),那等于"位置精确、姿态完全不管",
#: 手腕的倾角会自己漂 —— 而插销任务最后失败的主因正是螺栓和螺套的轴线对不上,
#: 不是横向偏移。另一个极端(三个都给 1)更糟:求解器会拿位置去换姿态,25mm 的
#: 平移目标解出来位置就差 25mm。
DEFAULT_IK_ROTATION_WEIGHT: tuple[float, float, float] = (1.0, 1.0, 0.02)
DEFAULT_IK_ITERS = 250
DEFAULT_IK_DAMPING = 1e-5

#: v2 起 observation 带 ``object_poses`` / ``ee_poses``,并新增 FK / IK 两条命令。
PROTOCOL_VERSION = 2


class Command:
    """Request types. Every request is answered exactly once (REQ/REP)."""

    HANDSHAKE = "handshake"
    OBSERVE = "observe"
    STEP = "step"
    RESET = "reset"
    #: 只把任务零件放回初始位姿,手臂原地不动。采数据/调试时零件被碰歪了要重摆,
    #: 用整体 RESET 会连手臂一起弹回复位姿态,遥操的手感就断了。
    #:
    #: 带 ``poses`` 时改为把零件放到指定位姿,跳过随机化 —— 演示增广要能复现
    #: 一个算好的初始位姿,靠随机种子碰是碰不到的。
    RESET_OBJECTS = "reset_objects"
    #: 批量正运动学:关节角 -> 夹爪位姿。放在仿真器这边是因为只有它有模型;
    #: 客户端那个环境里没有 MuJoCo(见模块开头)。
    FK = "fk"
    #: 批量逆运动学。同上,而且逐点用上一解做种子,单条轨迹一次往返即可。
    IK = "ik"
    CLOSE = "close"


class Status:
    OK = "ok"
    ERROR = "error"


class ProtocolError(RuntimeError):
    """Raised when the peer answers with an error or an unusable payload."""


def check_version(reply: dict) -> None:
    """Reject a simulator speaking a different protocol version.

    A silent mismatch would show up as subtly wrong joint ordering or image
    layout, which is far harder to debug than an immediate refusal.
    """
    peer = reply.get("protocol_version")
    if peer != PROTOCOL_VERSION:
        raise ProtocolError(
            f"simulator speaks protocol v{peer}, client expects v{PROTOCOL_VERSION}"
        )


def frame_nbytes(width: int, height: int) -> int:
    """Bytes in one uint8 RGB frame."""
    return width * height * 3
