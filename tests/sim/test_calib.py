"""Check the sim calibration bridge against LeRobot's own normalization.

The value of these tests is that the oracle is not a second copy of the formula:
`FeetechMotorsBus` can be constructed without touching hardware, so the
tick conversions are compared against the exact code path the real arm runs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from evo_rlt.sim.calib import (
    ARM_SIDES,
    BODY_JOINTS,
    GRIPPER_CLOSED_RAD,
    GRIPPER_JOINT,
    GRIPPER_OPEN_RAD,
    JOINT_ZERO_OFFSETS,
    MOTOR_NAMES,
    SIM_JOINT_LIMITS,
    URDF_JOINT_LIMITS,
    ArmCalibration,
    BimanualCalibration,
    CalibrationError,
)

# 用仓库内的 fixture,不依赖机器上的标定状态:标定文件会被删除、重标、
# 或被别的项目改写,测试不该因此静默跳过。fixture 取自真实标定数据,
# 保留了左右臂的差异 —— 那正是这些测试要守住的东西。
CALIB_DIR = Path(__file__).parent / "fixtures"
ARM_FILES = {side: CALIB_DIR / f"{side}_follower.json" for side in ARM_SIDES}


def _lerobot_bus(calib_path: Path):
    """Build a real FeetechMotorsBus on a dummy port, purely to reuse its math."""
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech.feetech import FeetechMotorsBus

    raw = json.loads(calib_path.read_text())
    calibration = {name: MotorCalibration(**entry) for name, entry in raw.items()}
    motors = {
        name: Motor(
            calibration[name].id,
            "sts3215",
            MotorNormMode.RANGE_0_100 if name == GRIPPER_JOINT else MotorNormMode.DEGREES,
        )
        for name in MOTOR_NAMES
    }
    return FeetechMotorsBus(port="/dev/null", motors=motors, calibration=calibration)


@pytest.fixture(scope="module")
def arms() -> dict[str, ArmCalibration]:
    return {side: ArmCalibration.from_file(path) for side, path in ARM_FILES.items()}


@pytest.fixture(scope="module")
def buses():
    return {side: _lerobot_bus(path) for side, path in ARM_FILES.items()}


def _sample_values(name: str) -> list[float]:
    if name == GRIPPER_JOINT:
        return [0.0, 12.5, 50.0, 87.5, 100.0]
    return [-90.0, -30.0, -5.5, 0.0, 5.5, 30.0, 90.0]


@pytest.mark.parametrize("side", ARM_SIDES)
@pytest.mark.parametrize("name", MOTOR_NAMES)
def test_value_to_ticks_matches_lerobot(side, name, arms, buses):
    """Our value->ticks must reproduce LeRobot's `_unnormalize` exactly."""
    arm, bus = arms[side], buses[side]
    motor_id = arm.motors[name].id
    for value in _sample_values(name):
        expected = bus._unnormalize({motor_id: value})[motor_id]
        got = arm.value_to_ticks(name, value)
        # LeRobot truncates to int at the very end; compare pre-truncation.
        assert int(got) == expected, f"{side}/{name} value={value}: {int(got)} != {expected}"


@pytest.mark.parametrize("side", ARM_SIDES)
@pytest.mark.parametrize("name", MOTOR_NAMES)
def test_ticks_to_value_matches_lerobot(side, name, arms, buses):
    """Our ticks->value must reproduce LeRobot's `_normalize` exactly."""
    arm, bus = arms[side], buses[side]
    motor = arm.motors[name]
    lo, hi = motor.range_min, motor.range_max
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        ticks = int(lo + frac * (hi - lo))
        expected = bus._normalize({motor.id: ticks})[motor.id]
        got = arm.ticks_to_value(name, ticks)
        assert got == pytest.approx(expected, abs=1e-9), f"{side}/{name} ticks={ticks}"


@pytest.mark.parametrize("side", ARM_SIDES)
@pytest.mark.parametrize("name", MOTOR_NAMES)
def test_value_rad_round_trip(side, name, arms):
    """value -> rad -> value must return the original, or the bridge loses pose."""
    arm = arms[side]
    for value in _sample_values(name):
        rad = arm.value_to_rad(name, value, clip=False)
        back = arm.rad_to_value(name, rad)
        assert back == pytest.approx(value, abs=1e-6), f"{side}/{name} value={value} -> {back}"


@pytest.mark.parametrize("side", ARM_SIDES)
def test_body_zero_maps_to_calibrated_midpoint(side, arms):
    """A DEGREES value of 0 means "at the calibrated mid-point", not "URDF zero".

    This is the whole reason the bridge exists, so pin it down explicitly.
    """
    arm = arms[side]
    for name in BODY_JOINTS:
        if name in JOINT_ZERO_OFFSETS:
            continue  # 有零位偏移,见 test_wrist_roll_zero_maps_to_horizontal_jaw
        rad = arm.value_to_rad(name, 0.0, clip=False)
        expected_ticks = arm.motors[name].mid
        expected_rad = (expected_ticks - 2048) * 2 * math.pi / 4096
        assert rad == pytest.approx(expected_rad, abs=1e-9)


def test_left_and_right_disagree_on_the_same_action(arms):
    """The same transported value lands on different angles per arm.

    A shared or URDF-limit-based mapping would make this test pass trivially by
    returning equal angles; it exists to stop that simplification from creeping
    back in.
    """
    left, right = arms["left"], arms["right"]
    differing = [
        name
        for name in MOTOR_NAMES
        if abs(left.value_to_rad(name, 0.0, clip=False) - right.value_to_rad(name, 0.0, clip=False))
        > 1e-6
    ]
    assert differing, "expected per-arm calibration to produce per-arm angles"


@pytest.mark.parametrize("side", ARM_SIDES)
def test_gripper_percentage_spans_the_real_jaw_travel(side, arms):
    """0%/100% 必须精确落在贴合角和张开上限。

    刻意不用 URDF 声明的限位:实测 URDF 下限 -10 度时爪尖还差 6.9 mm,夹爪
    合不拢,小物件抓不住。真正的贴合位置在 -13.5 度。
    """
    arm = arms[side]
    lo, hi = GRIPPER_CLOSED_RAD, GRIPPER_OPEN_RAD
    closed = arm.value_to_rad(GRIPPER_JOINT, 0.0, clip=False)
    opened = arm.value_to_rad(GRIPPER_JOINT, 100.0, clip=False)
    drive_mode = arm.motors[GRIPPER_JOINT].drive_mode
    # drive_mode flips which end of the travel counts as "0 percent".
    expected = (hi, lo) if drive_mode else (lo, hi)
    assert (closed, opened) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("side", ARM_SIDES)
@pytest.mark.parametrize("name", MOTOR_NAMES)
def test_clip_keeps_angles_inside_sim_limits(side, name, arms):
    """Out-of-range values must be clamped, so the simulator never gets a
    target its joint cannot represent.

    Clipped against SIM_JOINT_LIMITS, not the URDF's: wrist_roll is
    deliberately wider than the URDF declares, because the real wrist reaches
    further and truncating it would make the simulated arm unable to reproduce
    real poses.
    """
    arm = arms[side]
    lo, hi = SIM_JOINT_LIMITS[name]
    for value in (-10_000.0, 10_000.0):
        rad = arm.value_to_rad(name, value, clip=True)
        assert lo - 1e-9 <= rad <= hi + 1e-9


@pytest.mark.parametrize("side", ARM_SIDES)
def test_wrist_roll_limit_exceeds_the_urdf(side, arms):
    """限位必须比 URDF 声明的宽,否则真机能到的姿态仿真到不了。

    实测右腕行程 333 度 > URDF 的 320 度;再叠加零位偏移,需要的范围更大。
    """
    arm = arms[side]
    urdf_lo, urdf_hi = URDF_JOINT_LIMITS["wrist_roll"]
    lo = arm.value_to_rad("wrist_roll", -10_000.0, clip=True)
    hi = arm.value_to_rad("wrist_roll", 10_000.0, clip=True)
    assert hi - lo > urdf_hi - urdf_lo


def test_sign_flip_negates_body_joint(arms):
    """`sign` must flip a body joint's direction, since URDF axis orientation
    is a build property that has to be confirmed on the real arm."""
    plus = ArmCalibration.from_file(ARM_FILES["right"], signs={"elbow_flex": 1.0})
    minus = ArmCalibration.from_file(ARM_FILES["right"], signs={"elbow_flex": -1.0})
    for value in (-30.0, 0.0, 30.0):
        assert plus.value_to_rad("elbow_flex", value, clip=False) == pytest.approx(
            -minus.value_to_rad("elbow_flex", value, clip=False), abs=1e-9
        )


def test_sign_flip_survives_round_trip(arms):
    """A flipped sign must still round-trip, or the observation path desyncs."""
    arm = ArmCalibration.from_file(ARM_FILES["left"], signs=dict.fromkeys(MOTOR_NAMES, -1.0))
    for name in MOTOR_NAMES:
        for value in _sample_values(name):
            rad = arm.value_to_rad(name, value, clip=False)
            assert arm.rad_to_value(name, rad) == pytest.approx(value, abs=1e-6)


# -- bimanual, prefixed keys ------------------------------------------------


def _full_action() -> dict[str, float]:
    action = {}
    for side in ARM_SIDES:
        for name in BODY_JOINTS:
            action[f"{side}_{name}.pos"] = 10.0
        action[f"{side}_{GRIPPER_JOINT}.pos"] = 50.0
    return action


def test_bimanual_round_trip_preserves_all_twelve_keys():
    bimanual = BimanualCalibration.from_dir(CALIB_DIR, left_id="left_follower", right_id="right_follower")
    action = _full_action()
    rads = bimanual.action_to_rad(action, clip=False)
    assert len(rads) == 12
    assert set(rads) == {f"{side}_{name}" for side in ARM_SIDES for name in MOTOR_NAMES}

    observation = bimanual.rad_to_observation(rads)
    assert set(observation) == set(action)
    for key, value in action.items():
        assert observation[key] == pytest.approx(value, abs=1e-6), key


def test_bimanual_routes_each_prefix_to_its_own_arm():
    """A left-prefixed key must use the left file; crossing them silently
    mis-poses one arm, which is exactly the failure this bridge prevents."""
    bimanual = BimanualCalibration.from_dir(CALIB_DIR, left_id="left_follower", right_id="right_follower")
    rads = bimanual.action_to_rad(_full_action(), clip=False)
    for name in MOTOR_NAMES:
        assert rads[f"left_{name}"] == pytest.approx(
            bimanual.left.value_to_rad(name, 10.0 if name in BODY_JOINTS else 50.0, clip=False)
        )
        assert rads[f"right_{name}"] == pytest.approx(
            bimanual.right.value_to_rad(name, 10.0 if name in BODY_JOINTS else 50.0, clip=False)
        )


def test_partial_action_is_passed_through_untouched():
    """The record loop may send a subset; absent motors must not be invented."""
    bimanual = BimanualCalibration.from_dir(CALIB_DIR, left_id="left_follower", right_id="right_follower")
    rads = bimanual.action_to_rad({"left_elbow_flex.pos": 12.0}, clip=False)
    assert set(rads) == {"left_elbow_flex"}


# -- failure modes ----------------------------------------------------------


def test_missing_motor_is_rejected(tmp_path):
    raw = json.loads(ARM_FILES["right"].read_text())
    raw.pop("wrist_roll")
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(CalibrationError, match="missing motors"):
        ArmCalibration.from_file(path)


def test_degenerate_range_is_rejected(tmp_path):
    raw = json.loads(ARM_FILES["right"].read_text())
    raw["elbow_flex"]["range_max"] = raw["elbow_flex"]["range_min"]
    path = tmp_path / "degenerate.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(CalibrationError, match="range_min == range_max"):
        ArmCalibration.from_file(path)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(CalibrationError, match="not found"):
        ArmCalibration.from_file(tmp_path / "nope.json")


# -- 真机零位与 URDF 零位的定义差 ------------------------------------------


@pytest.mark.parametrize("side", ARM_SIDES)
def test_wrist_roll_zero_maps_to_horizontal_jaw(side, arms):
    """真机 wrist_roll 读数 0 必须落在 URDF 的 -90 度。

    标定时夹爪是水平的(固定爪在机身左、活动爪在机身右),而 URDF 零位夹爪竖立。
    wrist_roll 恰好是 LeRobot 标定流程里唯一不记录行程的关节,没有任何标定数据
    能暴露这个差异 —— 丢了这个偏移,腕相机视野和该关节行程都会错,而且不报错。
    """
    rad = arms[side].value_to_rad("wrist_roll", 0.0, clip=False)
    assert math.degrees(rad) == pytest.approx(-90.0, abs=0.5)


@pytest.mark.parametrize("side", ARM_SIDES)
def test_other_joints_have_no_zero_offset(side, arms):
    """除 wrist_roll 外都不该有偏移,否则是误加。"""
    for name in MOTOR_NAMES:
        if name in ("wrist_roll", GRIPPER_JOINT):
            continue
        arm = arms[side]
        expected = (arm.motors[name].mid - 2048) * 2 * math.pi / 4096
        assert arm.value_to_rad(name, 0.0, clip=False) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("side", ARM_SIDES)
def test_wrist_roll_full_travel_survives_the_offset(side, arms):
    """真机整圈行程叠加偏移后不能被限位截断。

    ±180 度的可达范围加 -90 度偏移需要 -270~+90 度;限位只给 ±180 就会把负方向
    截掉 90 度 —— 修好一个行程问题的同时制造另一个。
    """
    arm = arms[side]
    lo = arm.value_to_rad("wrist_roll", -180.0, clip=True)
    hi = arm.value_to_rad("wrist_roll", 180.0, clip=True)
    assert math.degrees(lo) == pytest.approx(-270.0, abs=1.0)
    assert math.degrees(hi) == pytest.approx(90.0, abs=1.0)
    assert math.degrees(hi - lo) == pytest.approx(360.0, abs=1.0)


@pytest.mark.parametrize("side", ARM_SIDES)
def test_gripper_closes_further_than_the_urdf_allows(side, arms):
    """闭合端必须比 URDF 的 -10 度更负,否则夹爪永远合不拢。"""
    urdf_lo, _ = URDF_JOINT_LIMITS[GRIPPER_JOINT]
    assert GRIPPER_CLOSED_RAD < urdf_lo
    assert math.degrees(GRIPPER_CLOSED_RAD) == pytest.approx(-13.5, abs=0.1)


@pytest.mark.parametrize("side", ARM_SIDES)
def test_gripper_travel_covers_the_calibrated_range(side, arms):
    """模型行程不得窄于真机标定行程,否则一端会被截断。"""
    motor = arms[side].motors[GRIPPER_JOINT]
    real_deg = motor.span * 360.0 / 4095
    sim_deg = math.degrees(GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD)
    assert sim_deg >= real_deg - 0.5
