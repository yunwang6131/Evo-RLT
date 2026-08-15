#!/usr/bin/env python
"""SO-101 标定:重新标定各臂、检查、同步到项目快照。

    python diagnostics/calibration.py --arm left_follower   # 标一条
    python diagnostics/calibration.py --all                 # 四条依次标
    python diagnostics/calibration.py --check               # 全部检查
    python diagnostics/calibration.py --check --live        # 加上真机往返
    python diagnostics/calibration.py --sync                # 同步到项目快照

--check 依次查四件事:

1. 左右臂行程是否统一 —— 窄的那条遥操时会提前触顶
2. 项目快照与系统标定是否同步 —— 不同步则仿真和真机落在不同姿态空间
3. 映射表 —— 每个关节的模式、行程、零位对应的弧度
4. 左右臂同值不同角 —— 这个差异是真实的,不是 bug

标定流程(LeRobot 交互式,两步):

1. 把该臂摆到各关节行程中间,回车 —— 定零位
2. 除 wrist_roll 外每个关节推过完整行程,回车 —— 记录行程

左右统一的关键在第 2 步:每个关节都推到真正的机械硬限位,两条臂用同样力度。
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evo_rlt.sim.calib import (  # noqa: E402
    ARM_SIDES,
    GRIPPER_JOINT,
    MOTOR_NAMES,
    URDF_JOINT_LIMITS,
    ArmCalibration,
    BimanualCalibration,
)
from evo_rlt.sim.arms import (  # noqa: E402
    FOLLOWER_CALIBRATION_DIR,
    ArmResolveError,
    build_device,
    calibration_status,
    load_arms,
    resolve_port,
)

DEFAULT_CALIB_DIR = str(FOLLOWER_CALIBRATION_DIR)

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = REPO_ROOT / "configs" / "calibration"

#: 左右行程差超过这个度数就认为不统一,值得重标。
TRAVEL_TOLERANCE_DEG = 8.0

#: wrist_roll 的行程由 LeRobot 固定成整圈,不参与一致性比较。
FULL_TURN_MOTOR = "wrist_roll"


def calibrate_one(alias: str) -> Path:
    """标定一条臂。端口按序列号自动定位,标定写进本项目目录。"""
    port = resolve_port(alias)
    device = build_device(alias, port)

    print("\n" + "=" * 70)
    print(f"  标定 {alias}   端口 {port}")
    print(f"  写入 {device.calibration_fpath}")
    print("=" * 70)
    print("  第 1 步: 把该臂摆到各关节行程的中间,然后回车")
    print("  第 2 步: 把除 wrist_roll 外每个关节推到两端硬限位,然后回车")
    print("           推到底 —— 左右臂行程能否统一全看这一步")
    if device.calibration:
        print("\n  已有标定,LeRobot 会先问是否复用。要重标就输入 c 再回车。")
    print()

    device.connect(calibrate=False)
    try:
        device.calibrate()
    finally:
        device.disconnect()
    print(f"\n已写入 {device.calibration_fpath}")
    return device.calibration_fpath


def load_ranges(alias: str) -> dict[str, tuple[int, int]] | None:
    """读本项目标定的行程。只看项目内文件,不碰共享缓存。"""
    from evo_rlt.sim.arms import arm

    path = arm(alias).calibration_path
    if not path.is_file():
        return None
    raw = json.loads(path.read_text())
    return {k: (v["range_min"], v["range_max"]) for k, v in raw.items()}


def check_symmetry() -> int:
    """比较左右臂各关节的行程宽度。

    行程是人推出来的,不是机械参数,所以左右不一致意味着标定时用力不同 ——
    窄的那条会在遥操时提前触顶,而另一条还能继续走。
    """
    problems = 0
    for kind in ("follower", "leader"):
        left = load_ranges(f"left_{kind}")
        right = load_ranges(f"right_{kind}")
        print(f"\n===== {kind} 左右行程对比 =====")
        if left is None or right is None:
            missing = [s for s, v in (("left", left), ("right", right)) if v is None]
            print(f"  {missing} 还没在本项目标定过")
            problems += 1
            continue

        print(f"{'关节':<15}{'左(度)':>10}{'右(度)':>10}{'差':>9}")
        print("-" * 46)
        for motor in left:
            if motor == FULL_TURN_MOTOR:
                continue
            lo = (left[motor][1] - left[motor][0]) * 360 / 4095
            ro = (right[motor][1] - right[motor][0]) * 360 / 4095
            diff = ro - lo
            flag = "" if abs(diff) <= TRAVEL_TOLERANCE_DEG else "  <-- 不统一"
            if abs(diff) > TRAVEL_TOLERANCE_DEG:
                problems += 1
            print(f"{motor:<15}{lo:>10.1f}{ro:>10.1f}{diff:>9.1f}{flag}")

    if problems:
        print(f"\n{problems} 处问题(行程阈值 {TRAVEL_TOLERANCE_DEG} 度)。")
        print("窄的那条臂遥操时会提前触顶。重标时把该关节推到真正的硬限位。")
    else:
        print("\n左右行程已统一。")
    return problems


def report_status() -> None:
    """列出每条臂:端口、标定文件、标定时间。

    以后想确认"仿真到底在用哪份标定"就跑这个,不用去翻目录或猜。
    """
    import datetime
    import hashlib

    from evo_rlt.sim.arms import by_id_map

    devices = by_id_map()
    print(f"{'臂':<16}{'序列号':<12}{'端口':<15}{'标定时间':<14}{'内容':<8}{'文件'}")
    print("-" * 108)
    for alias, a in sorted(load_arms().items()):
        port = devices.get(a.serial, "不在线")
        path = a.calibration_path
        if path.is_file():
            when = datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime("%m-%d %H:%M")
            digest = hashlib.md5(path.read_bytes()).hexdigest()[:6]
            try:
                shown = str(path.relative_to(Path.cwd()))
            except ValueError:
                shown = str(path)
        else:
            when, digest, shown = "未标定", "-", str(path)
        print(f"{alias:<16}{a.serial:<12}{port:<15}{when:<14}{digest:<8}{shown}")

    cams = Path.cwd() / "configs" / "cameras.json"
    if cams.is_file():
        when = datetime.datetime.fromtimestamp(cams.stat().st_mtime).strftime("%m-%d %H:%M")
        print(f"\n相机外参: configs/cameras.json  ({when})")
    else:
        print("\n相机外参: 未标定,仍是占位值 —— 跑 diagnostics/tune_cameras.py")


def _parse_signs(raw: list[str] | None) -> dict[str, dict[str, float]]:
    """Parse ``--sign left/elbow_flex=-1`` into nested per-arm overrides."""
    signs: dict[str, dict[str, float]] = {side: {} for side in ARM_SIDES}
    for item in raw or []:
        try:
            target, value = item.split("=")
            side, joint = target.split("/")
        except ValueError:
            raise SystemExit(f"--sign expects side/joint=value, got {item!r}")
        if side not in ARM_SIDES:
            raise SystemExit(f"--sign: unknown arm {side!r}")
        if joint not in MOTOR_NAMES:
            raise SystemExit(f"--sign: unknown joint {joint!r}")
        signs[side][joint] = float(value)
    return signs


def _report_arm(side: str, arm: ArmCalibration) -> None:
    print(f"\n===== {side} arm  ({arm.source}) =====")
    header = (
        f"{'joint':<15}{'mode':<9}{'ticks range':>14}{'mid':>8}{'sign':>6}"
        f"{'rad @ value=0':>15}{'urdf limit (rad)':>22}{'reachable':>11}"
    )
    print(header)
    print("-" * len(header))
    for name in MOTOR_NAMES:
        motor = arm.motors[name]
        mode = "0..100" if name == GRIPPER_JOINT else "degrees"
        zero_rad = arm.value_to_rad(name, 50.0 if name == GRIPPER_JOINT else 0.0, clip=False)
        lo, hi = URDF_JOINT_LIMITS[name]
        # How much of the URDF joint range the calibrated travel actually covers.
        span_lo = arm.value_to_rad(name, 0.0 if name == GRIPPER_JOINT else -180.0, clip=False)
        span_hi = arm.value_to_rad(name, 100.0 if name == GRIPPER_JOINT else 180.0, clip=False)
        reach = (min(hi, max(span_lo, span_hi)) - max(lo, min(span_lo, span_hi))) / (hi - lo)
        print(
            f"{name:<15}{mode:<9}{motor.range_min:>6}~{motor.range_max:<7}{motor.mid:>8.0f}"
            f"{arm.signs[name]:>6.0f}{zero_rad:>15.4f}"
            f"{f'{lo:.3f} ~ {hi:.3f}':>22}{f'{100 * min(1.0, max(0.0, reach)):.0f}%':>11}"
        )


def _report_arm_disagreement(bimanual: BimanualCalibration) -> None:
    print("\n===== left vs right, same transported value =====")
    print("A shared mapping would show all zeros here. It does not.")
    print(f"{'joint':<15}{'value':>8}{'left (rad)':>13}{'right (rad)':>13}{'delta (deg)':>13}")
    print("-" * 62)
    worst = 0.0
    for name in MOTOR_NAMES:
        value = 50.0 if name == GRIPPER_JOINT else 0.0
        left = bimanual.left.value_to_rad(name, value, clip=False)
        right = bimanual.right.value_to_rad(name, value, clip=False)
        delta = math.degrees(right - left)
        worst = max(worst, abs(delta))
        print(f"{name:<15}{value:>8.1f}{left:>13.4f}{right:>13.4f}{delta:>13.1f}")
    print(f"\nlargest left/right disagreement: {worst:.1f} deg")


def _load_setup(setup_json: Path) -> tuple[list[dict], str]:
    data = json.loads(setup_json.expanduser().read_text())
    arms = data.get("arms", [])
    followers = [a for a in arms if "follower" in a.get("type", "").lower()]
    if len(followers) != 2:
        raise SystemExit(f"expected 2 follower arms in {setup_json}, found {len(followers)}")
    cal_dir = followers[0].get("calibration_dir", DEFAULT_CALIB_DIR)
    return followers, cal_dir


def _check_live(followers: list[dict], cal_dir: str, bimanual: BimanualCalibration) -> int:
    """Read the real arms and round-trip their pose through the bridge."""
    from lerobot.robots.bi_so_follower import BiSOFollower
    from lerobot.robots.bi_so_follower.config_bi_so_follower import BiSOFollowerConfig
    from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig

    left_port = next(f["port"] for f in followers if "left" in f.get("alias", "").lower())
    right_port = next(f["port"] for f in followers if "right" in f.get("alias", "").lower())

    # use_degrees must match build_robot_argv, or every body joint is off.
    robot = BiSOFollower(
        BiSOFollowerConfig(
            id="bimanual",
            calibration_dir=Path(cal_dir).expanduser(),
            left_arm_config=SOFollowerConfig(port=left_port, use_degrees=True),
            right_arm_config=SOFollowerConfig(port=right_port, use_degrees=True),
        )
    )
    print(f"\n===== live round-trip ({left_port}, {right_port}) =====")
    robot.connect(calibrate=False)
    try:
        observation = robot.get_observation()
    finally:
        robot.disconnect()

    action = {key: value for key, value in observation.items() if key.endswith(".pos")}
    rads = bimanual.action_to_rad(action, clip=False)
    restored = bimanual.rad_to_observation(rads)

    print(f"{'key':<26}{'measured':>11}{'rad':>11}{'restored':>11}{'err':>11}")
    print("-" * 70)
    worst = 0.0
    for key in sorted(action):
        joint = key.removesuffix(".pos")
        err = abs(restored[key] - action[key])
        worst = max(worst, err)
        print(
            f"{key:<26}{action[key]:>11.4f}{rads[joint]:>11.4f}"
            f"{restored[key]:>11.4f}{err:>11.2e}"
        )

    clipped = [
        joint
        for joint, rad in rads.items()
        for lo, hi in [URDF_JOINT_LIMITS[joint.split("_", 1)[1]]]
        if not lo <= rad <= hi
    ]
    print(f"\nworst round-trip error: {worst:.2e}")
    if clipped:
        print(f"WARNING: current pose is outside URDF limits for: {clipped}")
        print("         the simulator will clamp these; re-check the URDF or the calibration.")
    if worst > 1e-6:
        print("FAIL: round-trip is lossy, the bridge would desync from the real arm")
        return 1
    print("OK: measured pose survives the round-trip")
    print("\nNote: this does NOT verify joint direction. Compare the printed radians")
    print("against the simulator at a known pose, and set --sign for any that invert.")
    return 0




def run_all_checks(setup: Path, live: bool, signs: list[str] | None) -> int:
    """把各项检查串成一条命令。"""
    from evo_rlt.sim.arms import FOLLOWER_CALIBRATION_DIR, arm

    print("===== 臂与标定状态 =====")
    report_status()

    problems = check_symmetry()

    missing = [a for a, ok in calibration_status().items() if not ok]
    if missing:
        print(f"\n{sorted(missing)} 尚未标定,跳过映射检查。")
        return 1

    bimanual = BimanualCalibration.from_dir(
        FOLLOWER_CALIBRATION_DIR,
        left_id=arm("left_follower").calibration_id,
        right_id=arm("right_follower").calibration_id,
        signs=_parse_signs(signs),
    )
    for side in ARM_SIDES:
        _report_arm(side, bimanual.arm(side))
    _report_arm_disagreement(bimanual)

    if live:
        if not setup.exists():
            raise SystemExit(f"--live 需要 manifest,{setup} 不存在")
        followers, live_cal_dir = _load_setup(setup)
        problems += _check_live(followers, live_cal_dir, bimanual)

    print("\n" + "=" * 70)
    if problems:
        print(f"共 {problems} 处问题,见上。")
        return 1
    print("全部检查通过。")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--setup", type=Path, default=Path("configs/my_so101_manifest.json"))
    parser.add_argument("--arm", help="标定单条臂")
    parser.add_argument("--all", action="store_true", help="四条臂依次标定")
    parser.add_argument("--check", action="store_true", help="检查全部")
    parser.add_argument("--status", action="store_true", help="只看端口和标定状态")
    parser.add_argument("--live", action="store_true", help="配合 --check,连真机做往返")
    parser.add_argument("--sign", action="append", metavar="SIDE/JOINT=VALUE",
                        help="翻转某关节方向,如 left/elbow_flex=-1")
    args = parser.parse_args()

    try:
        if args.status:
            report_status()
            return 0
        if args.check:
            return run_all_checks(args.setup, args.live, args.sign)

        arms = load_arms()
        if args.all:
            targets = sorted(arms)
        elif args.arm:
            if args.arm not in arms:
                raise SystemExit(f"未知的臂 {args.arm!r},可选: {sorted(arms)}")
            targets = [args.arm]
        else:
            parser.print_help()
            print(f"\n可标定的臂: {sorted(arms)}")
            return 1

        for alias in targets:
            calibrate_one(alias)
    except ArmResolveError as exc:
        print(f"\n{exc}")
        return 1

    print("\n\n" + "=" * 70)
    return 1 if check_symmetry() else 0


if __name__ == "__main__":
    raise SystemExit(main())
