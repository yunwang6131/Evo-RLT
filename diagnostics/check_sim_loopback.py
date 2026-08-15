#!/usr/bin/env python
"""End-to-end check of the sim bridge, with no real hardware and no policy.

Starts nothing itself -- run the simulator first::

    ~/anaconda3/envs/tutorial_for_mujoco/bin/python src/evo_rlt/sim/mj_server.py --build
    ~/anaconda3/envs/evo-rlt/bin/python diagnostics/check_sim_loopback.py

What it proves, in order:

1. `SimRobot` presents the same observation/action schema the real
   `BiSOFollower` does -- same 12 motor keys, same three camera keys.
2. A commanded pose comes back through physics, so the calibration round-trip
   survives the process boundary rather than just the unit tests.
3. The loop sustains the target control rate with images attached.

It deliberately does *not* check joint direction or camera framing: those are
properties of the physical build and mount, and need the real rig to settle.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from evo_rlt.sim.calib import GRIPPER_JOINT, MOTOR_NAMES  # noqa: E402
from evo_rlt.sim.protocol import DEFAULT_ENDPOINT, JOINT_ORDER  # noqa: E402
from evo_rlt.sim.sim_robot import make_sim_robot  # noqa: E402

EXPECTED_CAMERAS = ("left_wrist", "right_wrist", "right_front")


def _neutral_action() -> dict[str, float]:
    action = {}
    for side in ("left", "right"):
        for motor in MOTOR_NAMES:
            action[f"{side}_{motor}.pos"] = 50.0 if motor == GRIPPER_JOINT else 0.0
    return action


def check_schema(robot) -> list[str]:
    failures = []
    action_keys = set(robot.action_features)
    expected_motors = {f"{side}_{m}.pos" for side in ("left", "right") for m in MOTOR_NAMES}
    if action_keys != expected_motors:
        failures.append(f"action_features mismatch: {sorted(action_keys ^ expected_motors)}")

    obs_keys = set(robot.observation_features)
    missing_cams = [c for c in EXPECTED_CAMERAS if c not in obs_keys]
    if missing_cams:
        failures.append(f"observation_features missing cameras {missing_cams}")
    if not expected_motors <= obs_keys:
        failures.append("observation_features missing motor keys")

    print(f"  action features : {len(action_keys)} motor keys")
    print(f"  obs features    : {len(obs_keys)} keys, cameras {[c for c in EXPECTED_CAMERAS if c in obs_keys]}")
    for cam in EXPECTED_CAMERAS:
        if cam in robot.observation_features:
            print(f"    {cam:12s} shape {robot.observation_features[cam]}")
    return failures


def check_observation(robot) -> list[str]:
    failures = []
    obs = robot.get_observation()

    for cam in EXPECTED_CAMERAS:
        if cam not in obs:
            failures.append(f"observation missing camera {cam}")
            continue
        img = obs[cam]
        expected = robot.observation_features[cam]
        if img.shape != expected or img.dtype != np.uint8:
            failures.append(f"{cam}: got {img.shape}/{img.dtype}, expected {expected}/uint8")
        elif img.std() < 1.0:
            failures.append(f"{cam}: image is nearly uniform (std={img.std():.2f}); camera may see nothing")
        else:
            print(f"    {cam:12s} mean={img.mean():6.1f} std={img.std():5.1f}")

    motor_keys = [k for k in obs if k.endswith(".pos")]
    if len(motor_keys) != len(JOINT_ORDER):
        failures.append(f"expected {len(JOINT_ORDER)} motor readings, got {len(motor_keys)}")
    return failures


def check_tracking(robot, settle_steps: int, tolerance: float) -> list[str]:
    """Command a pose, let physics settle, and see if it is reached.

    This is the round-trip the unit tests cannot cover: value -> radians ->
    simulator -> radians -> value, across a process boundary and through
    actuator dynamics.
    """
    failures = []
    target = _neutral_action()
    # Nudge a few joints so this is not the trivial "already there" case.
    target["left_shoulder_lift.pos"] = -20.0
    target["right_elbow_flex.pos"] = 25.0
    target["left_gripper.pos"] = 20.0

    for _ in range(settle_steps):
        robot.send_action(target)

    obs = robot.get_observation()
    print(f"    {'key':<26}{'commanded':>11}{'measured':>11}{'error':>9}")
    worst = 0.0
    for key in sorted(target):
        measured = obs[key]
        error = abs(measured - target[key])
        worst = max(worst, error)
        flag = "" if error <= tolerance else "  <-- off"
        print(f"    {key:<26}{target[key]:>11.2f}{measured:>11.2f}{error:>9.2f}{flag}")
    print(f"    worst tracking error: {worst:.2f}")
    if worst > tolerance:
        failures.append(
            f"pose not tracked within {tolerance} (worst {worst:.2f}). "
            "Check actuator gain (control_kp) or that the target is reachable."
        )
    return failures


def check_rate(robot, fps: float, steps: int) -> list[str]:
    action = _neutral_action()
    budget_ms = 1000.0 / fps
    durations = []
    for _ in range(steps):
        t0 = time.perf_counter()
        robot.send_action(action)
        robot.get_observation()
        durations.append((time.perf_counter() - t0) * 1000)

    mean = statistics.mean(durations)
    p95 = sorted(durations)[int(0.95 * len(durations)) - 1]
    print(f"    mean {mean:.2f} ms | p95 {p95:.2f} ms | budget {budget_ms:.1f} ms at {fps:g} Hz")
    if p95 > budget_ms:
        return [
            f"cannot sustain {fps:g} Hz: p95 {p95:.1f} ms > {budget_ms:.1f} ms budget. "
            "If the simulator's own benchmark was fast, the cost is transport, not physics."
        ]
    print(f"    headroom {budget_ms / mean:.1f}x")
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--rate-steps", type=int, default=60)
    parser.add_argument("--settle-steps", type=int, default=60)
    parser.add_argument("--tolerance", type=float, default=3.0,
                        help="max |commanded - measured|, in transported units")
    args = parser.parse_args()

    robot = make_sim_robot(endpoint=args.endpoint, fps=int(args.fps))
    print(f"connecting to {args.endpoint} ...")
    try:
        robot.connect()
    except Exception as exc:
        print(f"FAILED to connect: {exc}")
        print("\nIs the simulator running?")
        print("  ~/anaconda3/envs/tutorial_for_mujoco/bin/python src/evo_rlt/sim/mj_server.py --build")
        return 1

    failures: list[str] = []
    try:
        print("\n[1/4] schema matches BiSOFollower")
        failures += check_schema(robot)

        print("\n[2/4] observation content")
        robot.reset()
        failures += check_observation(robot)

        print("\n[3/4] commanded pose survives the round-trip")
        failures += check_tracking(robot, args.settle_steps, args.tolerance)

        print("\n[4/4] sustained control rate")
        failures += check_rate(robot, args.fps, args.rate_steps)
    finally:
        robot.disconnect()

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("OK: schema, observations, tracking and rate all check out.")
    print("Still unverified (needs the real rig): joint directions, camera mounts, arm spacing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
