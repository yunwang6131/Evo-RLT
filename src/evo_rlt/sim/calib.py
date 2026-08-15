"""Convert between LeRobot's calibrated motor values and simulator joint angles.

This is the foundation of the sim/real bridge. LeRobot does **not** emit radians:
`send_action` / `get_observation` carry per-motor values that have already been
normalized through the arm's calibration file. Mapping those onto a URDF joint
requires replaying that same calibration in reverse -- naively rescaling the
value range onto the URDF joint limits produces a pose that silently disagrees
with the real arm, and disagrees *differently* for the left and right arm.

Two normalization modes are in play, because `SOFollower` hardcodes the gripper:

* the five body joints run in ``MotorNormMode.DEGREES`` (the record pipeline
  builds every follower with ``use_degrees=true``), so the transported value is
  degrees away from the *calibrated mid-point*, ``mid = (range_min + range_max)/2``::

      value_deg = (ticks - mid) * 360 / MAX_RES

* the gripper is pinned to ``MotorNormMode.RANGE_0_100`` regardless of
  ``use_degrees``, so its value is a 0..100 opening percentage::

      value = (ticks - range_min) / (range_max - range_min) * 100

Ticks become a joint angle by measuring against the servo's mechanical centre
(2048 of 4096), which the calibration's ``homing_offset`` has already aligned to
the arm's zero pose::

      rad = (ticks - 2048) * 2*pi / 4096

For the gripper that last step is meaningless -- its calibrated travel differs
between the two arms (the left spans 1652..3119 ticks, the right 1369..2915),
because the jaw's closed position is set by assembly, not by a shared zero. Its
percentage is therefore mapped straight onto the URDF jaw limits instead.

`sign` is deliberately left as a per-joint knob defaulting to +1: whether a
motor's positive direction matches its URDF axis is a property of the physical
build, and must be confirmed against the real arm rather than assumed. Run
``diagnostics/check_sim_calib.py`` to inspect the mapping and, with ``--live``,
to round-trip it against the connected followers.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

# Feetech STS3215 servo. LeRobot's DEGREES formula divides by resolution - 1,
# while the tick-to-angle conversion uses the full resolution; keep both.
STS3215_RESOLUTION = 4096
STS3215_MAX_RES = STS3215_RESOLUTION - 1
MECHANICAL_CENTER_TICKS = STS3215_RESOLUTION // 2

BODY_JOINTS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
GRIPPER_JOINT = "gripper"
MOTOR_NAMES: tuple[str, ...] = (*BODY_JOINTS, GRIPPER_JOINT)

ARM_SIDES: tuple[str, ...] = ("left", "right")

# Joint limits as declared in third_party/SO101/so101_new_calib.urdf, radians.
# Kept as a faithful record of the file; see SIM_JOINT_LIMITS for what is used.
URDF_JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-1.91986, 1.91986),
    "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-1.65806, 1.65806),
    "wrist_roll": (-2.74385, 2.84121),
    "gripper": (-0.174533, 1.74533),
}

#: 真机零位与 URDF 零位的定义差,单位弧度,按 ``urdf = calib + offset`` 相加。
#:
#: ``wrist_roll`` 差 90 度:真机标定时夹爪是**水平**的(固定爪在机身左、活动爪
#: 在机身右),而 URDF 零位时夹爪**竖立**。这个关节恰好是 LeRobot 标定流程里
#: 唯一不记录行程、直接固定成 0~4095 的那个,所以没有任何标定数据能暴露这个
#: 差异 —— 它只会表现为腕相机视野不对、以及 wrist_roll 行程一侧提前撞限位。
JOINT_ZERO_OFFSETS: dict[str, float] = {
    "wrist_roll": -math.pi / 2,
}

#: 夹爪贴合时的关节角(弧度)。URDF 声明的下限 -10 度并非真正的闭合位置:
#: 实测爪尖间距在 -13.5 度取到最小的 4.19 mm(那 4.19 mm 是两爪尖错开的形状,
#: 不是缝),再往负转爪子就转过头、间距反而变大。停在 -10 度会让仿真的夹爪
#: 始终合不拢,夹小物件时抓不住。
GRIPPER_CLOSED_RAD = math.radians(-13.5)

#: 夹爪张开上限。按真机标定行程(左 136 度 / 右 134 度)从贴合位置推算,
#: 取较大者,这样两臂都能走完各自的行程而不被模型截断。
GRIPPER_OPEN_RAD = GRIPPER_CLOSED_RAD + math.radians(136.4)

#: 仿真实际执行的限位。
#:
#: ``wrist_roll`` 放宽到整两圈,原因有二:一是真机实测行程 333 度、超过 URDF
#: 声明的 320 度(遥操时该关节的跟随相关性是十二个里最差的,就是被截断所致);
#: 二是叠加 -90 度零位偏移后需要 -270~+90 度,只给 ±180 会把负方向再截掉
#: 90 度。舵机标定跨满整个 4096 tick,硬件本来就报告可整圈旋转。
SIM_JOINT_LIMITS: dict[str, tuple[float, float]] = {
    **URDF_JOINT_LIMITS,
    "wrist_roll": (-2 * math.pi, 2 * math.pi),
    "gripper": (GRIPPER_CLOSED_RAD, GRIPPER_OPEN_RAD),
}

_TICKS_PER_RAD = STS3215_RESOLUTION / (2 * math.pi)


class CalibrationError(ValueError):
    """Raised when a calibration file is missing, malformed, or degenerate."""


@dataclass(frozen=True)
class MotorCalibration:
    """One motor's entry from a LeRobot follower calibration JSON."""

    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int

    @property
    def mid(self) -> float:
        """Calibrated mid-point in ticks -- the zero of the DEGREES mode."""
        return (self.range_min + self.range_max) / 2.0

    @property
    def span(self) -> int:
        return self.range_max - self.range_min


@dataclass
class ArmCalibration:
    """Calibration for one SO-101 arm, plus the sim-side conventions."""

    motors: dict[str, MotorCalibration]
    signs: dict[str, float] = field(default_factory=dict)
    source: Path | None = None

    def __post_init__(self) -> None:
        missing = [name for name in MOTOR_NAMES if name not in self.motors]
        if missing:
            raise CalibrationError(
                f"calibration {self.source or '<inline>'} is missing motors: {missing}"
            )
        for name, motor in self.motors.items():
            if motor.span == 0:
                raise CalibrationError(
                    f"motor '{name}' in {self.source or '<inline>'} has range_min == range_max"
                )
        self.signs = {name: float(self.signs.get(name, 1.0)) for name in MOTOR_NAMES}

    @classmethod
    def from_file(cls, path: str | Path, signs: dict[str, float] | None = None) -> ArmCalibration:
        path = Path(path).expanduser()
        if not path.is_file():
            raise CalibrationError(f"calibration file not found: {path}")
        with open(path) as f:
            raw = json.load(f)
        motors = {
            name: MotorCalibration(
                id=int(entry["id"]),
                drive_mode=int(entry["drive_mode"]),
                homing_offset=int(entry["homing_offset"]),
                range_min=int(entry["range_min"]),
                range_max=int(entry["range_max"]),
            )
            for name, entry in raw.items()
            if name in MOTOR_NAMES
        }
        return cls(motors=motors, signs=dict(signs or {}), source=path)

    # -- ticks <-> LeRobot transported value ------------------------------

    def value_to_ticks(self, name: str, value: float) -> float:
        """Invert LeRobot's ``_unnormalize`` for one motor (kept in float)."""
        motor = self.motors[name]
        if name == GRIPPER_JOINT:
            bounded = min(100.0, max(0.0, value))
            if motor.drive_mode:
                bounded = 100.0 - bounded
            return bounded / 100.0 * motor.span + motor.range_min
        return value * STS3215_MAX_RES / 360.0 + motor.mid

    def ticks_to_value(self, name: str, ticks: float) -> float:
        """Mirror LeRobot's ``_normalize`` for one motor."""
        motor = self.motors[name]
        if name == GRIPPER_JOINT:
            bounded = min(float(motor.range_max), max(float(motor.range_min), ticks))
            norm = (bounded - motor.range_min) / motor.span * 100.0
            return 100.0 - norm if motor.drive_mode else norm
        return (ticks - motor.mid) * 360.0 / STS3215_MAX_RES

    # -- ticks <-> joint angle --------------------------------------------

    def ticks_to_rad(self, name: str, ticks: float) -> float:
        if name == GRIPPER_JOINT:
            # Jaw travel is set by assembly and differs per arm; go through the
            # opening percentage rather than a shared mechanical centre.
            return self._gripper_value_to_rad(self.ticks_to_value(name, ticks))
        rad = (ticks - MECHANICAL_CENTER_TICKS) / _TICKS_PER_RAD * self.signs[name]
        return rad + JOINT_ZERO_OFFSETS.get(name, 0.0)

    def rad_to_ticks(self, name: str, rad: float) -> float:
        if name == GRIPPER_JOINT:
            return self.value_to_ticks(name, self._gripper_rad_to_value(rad))
        rad = rad - JOINT_ZERO_OFFSETS.get(name, 0.0)
        return rad * self.signs[name] * _TICKS_PER_RAD + MECHANICAL_CENTER_TICKS

    def _gripper_value_to_rad(self, value: float) -> float:
        # 0% 落在贴合角、100% 落在张开上限
        lo, hi = GRIPPER_CLOSED_RAD, GRIPPER_OPEN_RAD
        frac = min(100.0, max(0.0, value)) / 100.0
        rad = lo + frac * (hi - lo)
        return rad * self.signs[GRIPPER_JOINT]

    def _gripper_rad_to_value(self, rad: float) -> float:
        lo, hi = GRIPPER_CLOSED_RAD, GRIPPER_OPEN_RAD
        rad = rad * self.signs[GRIPPER_JOINT]
        frac = (min(hi, max(lo, rad)) - lo) / (hi - lo)
        return frac * 100.0

    # -- the pair the bridge actually calls --------------------------------

    def value_to_rad(self, name: str, value: float, clip: bool = True) -> float:
        """LeRobot transported value -> URDF joint angle in radians."""
        rad = self.ticks_to_rad(name, self.value_to_ticks(name, value))
        if clip:
            lo, hi = SIM_JOINT_LIMITS[name]
            rad = min(hi, max(lo, rad))
        return rad

    def rad_to_value(self, name: str, rad: float) -> float:
        """URDF joint angle in radians -> LeRobot transported value."""
        return self.ticks_to_value(name, self.rad_to_ticks(name, rad))

    def action_to_rad(self, action: dict[str, float], clip: bool = True) -> dict[str, float]:
        """Map a single arm's ``{motor}.pos`` dict onto joint angles.

        Keys arrive unprefixed, as `BiSOFollower` strips ``left_``/``right_``
        before handing an action to the per-arm follower.
        """
        return {
            name: self.value_to_rad(name, float(action[f"{name}.pos"]), clip=clip)
            for name in MOTOR_NAMES
            if f"{name}.pos" in action
        }

    def rad_to_observation(self, rads: dict[str, float]) -> dict[str, float]:
        """Inverse of :meth:`action_to_rad`, producing ``{motor}.pos`` keys."""
        return {
            f"{name}.pos": self.rad_to_value(name, float(rads[name]))
            for name in MOTOR_NAMES
            if name in rads
        }


@dataclass
class BimanualCalibration:
    """Both arms, keyed the way `BiSOFollower` prefixes its features."""

    left: ArmCalibration
    right: ArmCalibration

    @classmethod
    def from_dir(
        cls,
        calibration_dir: str | Path,
        left_id: str = "left_follower_arm",
        right_id: str = "right_follower_arm",
        signs: dict[str, dict[str, float]] | None = None,
    ) -> BimanualCalibration:
        calibration_dir = Path(calibration_dir).expanduser()
        signs = signs or {}
        return cls(
            left=ArmCalibration.from_file(
                calibration_dir / f"{left_id}.json", signs.get("left")
            ),
            right=ArmCalibration.from_file(
                calibration_dir / f"{right_id}.json", signs.get("right")
            ),
        )

    def arm(self, side: str) -> ArmCalibration:
        if side not in ARM_SIDES:
            raise KeyError(f"unknown arm side {side!r}, expected one of {ARM_SIDES}")
        return self.left if side == "left" else self.right

    def action_to_rad(self, action: dict[str, float], clip: bool = True) -> dict[str, float]:
        """Prefixed 12-DoF action -> prefixed joint angles, e.g. ``left_elbow_flex``."""
        out: dict[str, float] = {}
        for side in ARM_SIDES:
            prefix = f"{side}_"
            arm_action = {
                key.removeprefix(prefix): value
                for key, value in action.items()
                if key.startswith(prefix)
            }
            for name, rad in self.arm(side).action_to_rad(arm_action, clip=clip).items():
                out[f"{prefix}{name}"] = rad
        return out

    def rad_to_observation(self, rads: dict[str, float]) -> dict[str, float]:
        """Prefixed joint angles -> prefixed ``{motor}.pos`` observation."""
        out: dict[str, float] = {}
        for side in ARM_SIDES:
            prefix = f"{side}_"
            arm_rads = {
                key.removeprefix(prefix): value
                for key, value in rads.items()
                if key.startswith(prefix)
            }
            for key, value in self.arm(side).rad_to_observation(arm_rads).items():
                out[f"{prefix}{key}"] = value
        return out
