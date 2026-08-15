#!/usr/bin/env python
"""让任务物体在物理里自然落稳,把最终位姿写回配置。

    python diagnostics/settle_objects.py --apply

手算摆放高度很难对:零件表面有倒角、凸分解后的孔壁与原始几何有出入,差零点几
毫米就会在复位瞬间嵌进对方,接触力把物体弹飞 —— 而现象看起来像"位置放错了"。

这里改成:把物体抬高一点自由落下,跑到静止,读出稳定位姿。物理引擎自己解出的
位置一定是无穿透的。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENE = Path("~/.cache/evo_rlt/sim_assets/scene.xml").expanduser()
CONFIG = REPO_ROOT / "configs" / "task_scene.json"

#: 抬高多少米后自由落下。太高会砸偏,太低则可能一开始就嵌住。
DROP_HEIGHT = 0.012

#: 判定静止的速度阈值(米/秒)
STILL_SPEED = 1e-3


def quat_to_euler(q) -> list[float]:
    """MuJoCo 四元数 (w,x,y,z) -> MJCF 的 XYZ 欧拉角。"""
    w, x, y, z = q
    sinr = 2 * (w * x + y * z)
    cosr = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    sinp = 2 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    siny = 2 * (w * z + x * y)
    cosy = 1 - 2 * (y * y + z * z)
    return [round(v, 5) for v in (roll, pitch, math.atan2(siny, cosy))]


def settle(steps: int, verbose: bool) -> dict:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    names = ["socket", "bolt"]
    joints = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_free") for n in names}
    bodies = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in names}

    # 抬高后再落,避开复位瞬间的嵌入
    for name, jid in joints.items():
        adr = model.jnt_qposadr[jid]
        data.qpos[adr + 2] += DROP_HEIGHT
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    start = {n: data.xpos[b].copy() for n, b in bodies.items()}
    for step in range(steps):
        mujoco.mj_step(model, data)
        if step > 500 and np.abs(data.qvel).max() < STILL_SPEED:
            if verbose:
                print(f"  第 {step} 步已静止")
            break

    result = {}
    print(f"{'物体':<8}{'落稳位置 (x, y, z)':>30}{'水平漂移':>10}{'最大穿透':>10}")
    print("-" * 60)
    for name, bid in bodies.items():
        pos = data.xpos[bid]
        drift = np.linalg.norm(pos[:2] - start[name][:2]) * 1000
        pen = [
            data.contact[i].dist * 1000
            for i in range(data.ncon)
            if bid in (model.geom_bodyid[data.contact[i].geom1],
                       model.geom_bodyid[data.contact[i].geom2])
        ]
        worst = min(pen) if pen else 0.0
        print(f"{name:<8}({pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}){drift:>10.1f}{worst:>10.2f}")
        result[name] = {
            "pos": [round(float(v), 4) for v in pos],
            "euler": quat_to_euler(data.xquat[bid]),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--apply", action="store_true", help="把结果写回 configs/task_scene.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not SCENE.is_file():
        raise SystemExit(f"场景不存在 {SCENE},先 mj_server.py --build")

    settled = settle(args.steps, args.verbose)

    if args.apply:
        cfg = json.loads(CONFIG.read_text())
        for name, state in settled.items():
            cfg[name]["pos"] = state["pos"]
            cfg[name]["euler"] = state["euler"]
        CONFIG.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
        print(f"\n已写回 {CONFIG}")
        print("重建场景生效: mj_server.py --build")
    else:
        print("\n加 --apply 写回配置")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
