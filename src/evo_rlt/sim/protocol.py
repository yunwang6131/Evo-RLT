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

PROTOCOL_VERSION = 1


class Command:
    """Request types. Every request is answered exactly once (REQ/REP)."""

    HANDSHAKE = "handshake"
    OBSERVE = "observe"
    STEP = "step"
    RESET = "reset"
    #: 只把任务零件放回初始位姿,手臂原地不动。采数据/调试时零件被碰歪了要重摆,
    #: 用整体 RESET 会连手臂一起弹回复位姿态,遥操的手感就断了。
    RESET_OBJECTS = "reset_objects"
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
