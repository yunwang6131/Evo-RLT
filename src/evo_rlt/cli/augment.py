"""用已有的人类演示增广出更多仿真 VLA 数据。

两步:

``calibrate``
    源数据里**没有记录螺套的初始位姿** —— 采集时零件坐标从没送出过仿真进程。
    这一步先由抓取那一帧的夹爪位姿反推一个估计,再拿重放本身把它标准确:摆上
    去跑一遍,量出插入时刻的对不准量,解析地反解回摆件误差,修一次再跑。跑通了
    (自动判据判成功)才把这条源演示连同它的螺套位姿记进标定文件。**跑不通的
    源演示不进标定文件**,因为它的几何没被复现出来,拿去增广只会批量产出失败。

``run``
    对每条已标定的源演示,在凹槽圆盘内另选若干螺套位置重放:抓取前的末端轨迹
    整体平移,抓到之后连零件一起平移,于是插入发生在工作空间的另一处。每条跑完
    用自动判据过一遍,成功的才写进数据集。

原理、以及为什么位移只取平移(SO-101 只有 5 个自由度),见
``evo_rlt.sim.augment`` 的模块说明。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROFILE = REPO_ROOT / "configs" / "blue_screw_sim_v1.json"
DEFAULT_CALIBRATION = REPO_ROOT / "configs" / "sim_augment_calibration.json"

#: 凹槽随机化的圆盘,和 ``configs/task_scene.json`` 的 ``socket.reset_random``
#: 同源 —— 从那里读,免得两处各写一份而悄悄分叉。
def _disk_from_scene() -> tuple[np.ndarray, float, float]:
    scene = json.loads((REPO_ROOT / "configs" / "task_scene.json").read_text())
    random_spec = scene["socket"]["reset_random"]
    return (
        np.asarray(random_spec["center"], dtype=float),
        float(random_spec["radius"]),
        float(scene["socket"]["pos"][2]),
    )


def _repo_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _connect(endpoint: str):
    from evo_rlt.sim.sim_robot import make_sim_robot

    robot = make_sim_robot(endpoint=endpoint)
    robot.connect()
    return robot


def _fit_calibration(robot, episodes, joint_order):
    """由每条演示抓取帧的夹爪位姿,拟合"螺套在夹爪系下的位置"。"""
    from evo_rlt.sim import augment as A

    center, radius, rest_z = _disk_from_scene()
    rows = np.array([ep.states[ep.segments.right_close] for ep in episodes])
    rads = A.to_radians(rows, robot.calibration_bridge, joint_order)
    poses = robot.fk(rads.tolist())
    positions = np.array([frame["right"][:3] for frame in poses])
    rotations = np.array([A.pose_matrix(frame["right"])[1] for frame in poses])
    calibration = A.fit_grasp_calibration(positions, rotations, center, radius, rest_z)
    up_axis, spread = A.socket_up_axis_in_gripper(rotations)
    return calibration, up_axis, spread, positions, rotations


def run_calibrate(args: argparse.Namespace, profile: dict[str, Any]) -> None:
    from evo_rlt.sim import augment as A
    from evo_rlt.sim import task_success as TS
    from evo_rlt.sim.protocol import JOINT_ORDER

    source = _repo_path(args.source or profile["merged_root"])
    episodes = A.read_source_episodes(source, task=profile["task"])
    if args.limit:
        episodes = episodes[: args.limit]
    print(f"源数据 {source}: {len(episodes)} 条可分段")

    robot = _connect(args.endpoint)
    try:
        calibration, up_axis, spread, positions, rotations = _fit_calibration(
            robot, episodes, JOINT_ORDER
        )
        residual = calibration.residual_radius * 1000.0
        theory = A.disk_quantiles(calibration.disk_radius)
        print(f"\n螺套竖直轴在夹爪系 = {np.round(up_axis, 4)}")
        print(f"各帧偏离: 中位 {np.median(spread):.1f}°  p90 {np.percentile(spread, 90):.1f}°")
        print(f"t_rel = {np.round(calibration.translation, 4)} (|t| = {np.linalg.norm(calibration.translation) * 1000:.1f} mm)")
        print("推出的螺套到凹槽中心的距离(自检:该像一个均匀圆盘)")
        print(
            "  实测 q25 %.1f  q50 %.1f  q75 %.1f  q90 %.1f mm"
            % tuple(np.percentile(residual, p) for p in (25, 50, 75, 90))
        )
        print(
            "  理论 q25 %.1f  q50 %.1f  q75 %.1f  q90 %.1f mm"
            % tuple(theory[k] * 1000 for k in ("q25", "q50", "q75", "q90"))
        )

        success_config = TS.load_config()
        verified: dict[str, Any] = {}
        stages: dict[str, int] = {}
        started = time.perf_counter()
        for index, episode in enumerate(episodes):
            pose = A.socket_pose_from_grasp(
                positions[index], rotations[index], calibration, up_axis, args.yaw_offset
            )
            # 摆件位姿一次定死,之后只修**抓稳后的右臂轨迹**。钳口合拢会把六角
            # 重新坐正,所以摆件误差和抓后位姿不是一一对应的,改摆件位置只能把
            # 重放成功率从 32% 推到 42%;而握着零件的手走多少零件就走多少,
            # 修在轨迹上是精确的(实测横偏能收敛到 0.1mm)。
            correction = np.zeros(3)
            outcome, states, nearest = "未通过", None, None
            for round_index in range(args.rounds):
                actions = episode.actions
                if round_index:
                    actions = A.plan_episode(
                        robot, episode, calibration, up_axis, args.yaw_offset,
                        np.zeros(3), JOINT_ORDER, hold_correction=correction,
                    ).actions
                states = A.replay(robot, actions, pose, JOINT_ORDER, success_config)
                if TS.episode_succeeded(states):
                    verified[str(episode.index)] = {
                        "socket_pose": [float(v) for v in pose],
                        "hold_correction": [float(v) for v in correction],
                        "rounds": round_index + 1,
                    }
                    outcome = f"第 {round_index} 轮通过"
                    break
                nearest = A.closest_approach(states)
                if nearest is None:
                    outcome = "没走到孔口附近,量不到对不准量"
                    break
                correction = correction + np.asarray(states[nearest].lateral_offset, dtype=float)
            if states is not None:
                stage = TS.furthest_stage(states)
                stages[stage] = stages.get(stage, 0) + 1
                if str(episode.index) not in verified:
                    outcome += f" (停在 {stage}"
                    if nearest is not None:
                        outcome += (
                            f":横偏 {states[nearest].lateral * 1000:.2f}mm"
                            f" 夹角 {states[nearest].angle_deg:.1f}°"
                        )
                    outcome += ")"
            elapsed = time.perf_counter() - started
            print(
                f"  [{index + 1}/{len(episodes)}] ep{episode.index}: {outcome}"
                f"   ({elapsed / (index + 1):.0f} s/条)",
                flush=True,
            )
        print("\n各条最终停在: " + "  ".join(f"{k}×{v}" for k, v in sorted(stages.items())))
    finally:
        robot.disconnect()

    out = _repo_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": str(source.relative_to(REPO_ROOT)) if source.is_relative_to(REPO_ROOT) else str(source),
                "grasp": calibration.to_dict(),
                "up_axis": [float(v) for v in up_axis],
                "yaw_offset": float(args.yaw_offset),
                "episodes": verified,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\n标定通过 {len(verified)}/{len(episodes)} 条 -> {out}")
    if not verified:
        raise SystemExit("没有一条源演示能被复现,增广无从谈起;先查仿真器和场景是否是采集时那一份")


# -- 生成增广数据集 ---------------------------------------------------------


def _dataset_features(source_root: Path) -> dict[str, Any]:
    """照抄源数据集的特征表(去掉 lerobot 自己会加的那几列)。

    不自己拼:增广出来的 episode 要能和源数据 **合并** 后一起训,列名、dtype、
    shape、names 差一个字都会在 merge 或 train 的预检里炸出来。从源 info.json
    抄是唯一能保证一致的做法。
    """
    from lerobot.datasets.utils import DEFAULT_FEATURES

    info = json.loads((source_root / "meta" / "info.json").read_text())
    return {
        key: value
        for key, value in info["features"].items()
        if key not in DEFAULT_FEATURES
    }


def _make_dataset(repo_id: str, root: Path, fps: int, robot, features: dict[str, Any]):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # episode_success 要写进 meta/episodes,而装着的 lerobot 的 save_episode
    # 不认这个参数 —— 本仓库用一个 monkey patch 加上去,采集主流程走的也是它。
    from evo_rlt.adapters.lerobot.record.runner import _patch_save_episode_extra_metadata

    _patch_save_episode_extra_metadata()
    return LeRobotDataset.create(
        repo_id,
        fps,
        features=features,
        root=root,
        robot_type=robot.name,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4 * len(robot.cameras),
    )


def _frame_writer(dataset, features: dict[str, Any], task: str):
    """返回一个 ``on_frame`` 回调,把一帧观测/指令写进数据集缓冲。"""
    from lerobot.datasets.feature_utils import build_dataset_frame
    from lerobot.utils.constants import ACTION, OBS_STR

    zeros = np.zeros(features["action"]["shape"][0], dtype=np.float32)

    def on_frame(_index: int, observation: dict, action: dict) -> None:
        frame = {
            **build_dataset_frame(dataset.features, observation, prefix=OBS_STR),
            **build_dataset_frame(dataset.features, action, prefix=ACTION),
            "task": task,
        }
        # 这几列源数据里有,合并时列必须对齐。增广数据没有策略/干预的概念,
        # 全部填成"人类采集、无干预"—— 和被增广的那批源演示同一档。
        for key, value in (
            ("complementary_info.policy_action", zeros),
            ("complementary_info.is_intervention", np.zeros(1, dtype=np.float32)),
            ("complementary_info.state", np.zeros(1, dtype=np.float32)),
            ("complementary_info.phase", np.zeros(1, dtype=np.float32)),
            ("complementary_info.collector_policy_id", np.zeros(1, dtype=np.int64)),
        ):
            if key in dataset.features:
                frame[key] = value
        dataset.add_frame(frame)

    return on_frame


def run_generate(args: argparse.Namespace, profile: dict[str, Any]) -> None:
    from evo_rlt.sim import augment as A
    from evo_rlt.sim import task_success as TS
    from evo_rlt.sim.protocol import JOINT_ORDER

    calibration_path = _repo_path(args.calibration)
    if not calibration_path.is_file():
        raise SystemExit(f"找不到标定文件 {calibration_path};先跑 `evo-rlt-sim-augment calibrate`")
    raw = json.loads(calibration_path.read_text())
    calibration = A.GraspCalibration.from_dict(raw["grasp"])
    up_axis = np.asarray(raw["up_axis"], dtype=float)
    yaw_offset = float(raw["yaw_offset"])
    verified = raw["episodes"]
    if not verified:
        raise SystemExit("标定文件里没有通过的源演示")

    source = _repo_path(args.source or profile["merged_root"])
    episodes = [
        ep for ep in A.read_source_episodes(source, task=profile["task"])
        if str(ep.index) in verified
    ]
    print(f"可用源演示 {len(episodes)} 条,每条生成 {args.per_source} 个变体")

    out_root = _repo_path(args.out_root)
    if out_root.exists() and not args.resume:
        raise SystemExit(f"{out_root} 已存在;换个 --out-root 或加 --resume")

    features = _dataset_features(source)
    success_config = TS.load_config()
    rng = np.random.default_rng(args.seed)

    robot = _connect(args.endpoint)
    dataset = _make_dataset(
        profile["repo_id"] + args.repo_id_suffix, out_root,
        profile["expected"]["fps"], robot, features,
    )
    on_frame = _frame_writer(dataset, features, profile["task"])

    kept = attempted = 0
    stages: dict[str, int] = {}
    started = time.perf_counter()
    try:
        for episode in episodes:
            entry = verified[str(episode.index)]
            socket_pose = np.asarray(entry["socket_pose"], dtype=float)
            hold = np.asarray(entry.get("hold_correction", [0.0, 0.0, 0.0]), dtype=float)
            for _ in range(args.per_source):
                delta = A.sample_delta(rng, calibration, socket_pose[:2], args.max_delta)
                plan = A.plan_episode(
                    robot, episode, calibration, up_axis, yaw_offset, delta, JOINT_ORDER,
                    hold_correction=hold, bridge_frames=args.bridge_frames,
                )
                states = A.replay(
                    robot, plan.actions, plan.socket_pose, JOINT_ORDER, success_config,
                    on_frame=on_frame,
                )
                attempted += 1
                ok = TS.episode_succeeded(states)
                stage = TS.furthest_stage(states)
                stages[stage] = stages.get(stage, 0) + 1
                if ok:
                    dataset.save_episode(extra_episode_metadata={"episode_success": "success"})
                    kept += 1
                else:
                    dataset.clear_episode_buffer()
                print(
                    f"  ep{episode.index} Δ={np.linalg.norm(delta) * 1000:5.1f}mm "
                    f"IK残差 {plan.ik_pos_err_max * 1000:.2f}mm/{plan.ik_yaw_err_deg:.1f}° "
                    f"-> {'留' if ok else '弃'} ({stage})   "
                    f"累计 {kept}/{attempted}  {(time.perf_counter() - started) / attempted:.0f} s/条",
                    flush=True,
                )
    finally:
        robot.disconnect()

    print(f"\n保留 {kept}/{attempted} = {100 * kept / max(attempted, 1):.0f}%  -> {out_root}")
    print("失败停在: " + "  ".join(f"{k}×{v}" for k, v in sorted(stages.items())))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    sub = parser.add_subparsers(dest="command", required=True)

    calibrate = sub.add_parser("calibrate", help="反推每条源演示的螺套位姿,并用重放验证。")
    calibrate.add_argument("--source", type=Path, default=None, help="源数据集根目录(默认取 profile)")
    calibrate.add_argument("--out", type=Path, default=DEFAULT_CALIBRATION)
    calibrate.add_argument("--rounds", type=int, default=3, help="每条最多修正几轮")
    calibrate.add_argument("--limit", type=int, default=0, help="只标定前 N 条(调试用)")
    calibrate.add_argument("--yaw-offset", type=float, default=0.0,
                           help="摆件偏航的全局偏置(弧度)。钳口会把六角重新坐正,实测不敏感。")
    calibrate.set_defaults(func=run_calibrate)

    generate = sub.add_parser("run", help="按标定文件生成增广数据集。")
    generate.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    generate.add_argument("--source", type=Path, default=None)
    generate.add_argument("--out-root", type=Path, required=True)
    generate.add_argument("--repo-id-suffix", default="_aug")
    generate.add_argument("--per-source", type=int, default=8)
    generate.add_argument("--max-delta", type=float, default=0.030,
                          help="单次位移上限(米)。越大越有多样性,IK 的偏航残差也越大(0.135 度/毫米)。")
    generate.add_argument("--bridge-frames", type=int, default=30)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--resume", action="store_true")
    generate.set_defaults(func=run_generate)
    return parser


def main(argv: list[str] | None = None) -> None:
    from evo_rlt.cli.act import load_profile

    args = build_parser().parse_args(argv)
    profile = load_profile(args.profile)
    args.func(args, profile)


if __name__ == "__main__":
    main()
