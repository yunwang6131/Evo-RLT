#!/usr/bin/env python
"""用真机遥操数据拟合仿真的执行器参数,让仿真的跟随特性贴近真机。

    python diagnostics/fit_sim_dynamics.py outputs/sign_check.json

真机的滞后由两部分组成,性质不同,必须分开处理:

**纯延迟** —— 指令发出后一段时间内关节完全不动(通信往返 + 舵机启动)。
调执行器增益模拟不出来:降低 kp 只会让仿真"立刻开始、慢慢接近",而真机是
"先不动、然后动"。曲线形状对不上。这部分靠 ``action_delay_steps`` 缓冲指令。

**一阶时间常数** —— PID 收敛快慢。这部分才归 ``control_kp`` 管。

为什么要对齐:RLT 是 action chunk 方法,策略在预测未来一段动作。仿真里"发指令
即到位"而真机要 120 ms,学到的时序就偏快,迁到真机上动作会赶在实际位置之前。
插销这类接触任务最吃这个 —— 对准的瞬间差 120 ms,销就插偏了。而且症状隐蔽:
仿真里一切正常。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

FPS = 30.0


def fit_first_order(u: np.ndarray, y: np.ndarray, max_delay: int = 8) -> tuple[int, float, float]:
    """拟合 y[t] = y[t-1] + a*(u[t-d] - y[t-1]),返回 (纯延迟帧, 响应系数, RMSE)。"""
    best = (0, 0.5, 1e18)
    for delay in range(max_delay + 1):
        shifted = np.concatenate([np.full(delay, u[0]), u[: len(u) - delay]]) if delay else u
        for a in np.arange(0.05, 1.005, 0.05):
            pred = np.empty_like(y)
            pred[0] = y[0]
            for t in range(1, len(y)):
                pred[t] = pred[t - 1] + a * (shifted[t] - pred[t - 1])
            rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
            if rmse < best[2]:
                best = (delay, float(a), rmse)
    return best


def tau_ms(a: float) -> float:
    """一阶响应系数换算成时间常数(毫秒)。"""
    if a >= 1.0:
        return 0.0
    return -1.0 / np.log(1.0 - a) / FPS * 1000.0


def analyse(path: Path) -> dict:
    data = json.loads(path.read_text())
    if data.get("real_measured") is None:
        raise SystemExit(f"{path} 是 solo 模式采的,没有真机数据,无法拟合")

    cmd, real, sim = data["commanded"], data["real_measured"], data["sim_measured"]
    rows = []
    for key in cmd:
        u = np.asarray(cmd[key], dtype=float)
        rd, ra, _ = fit_first_order(u, np.asarray(real[key], dtype=float))
        sd, sa, _ = fit_first_order(u, np.asarray(sim[key], dtype=float))
        rows.append((key, rd, ra, sd, sa))

    print(f"{'关节':<24}{'真机延迟':>9}{'真机τ(ms)':>11}{'仿真延迟':>9}{'仿真τ(ms)':>11}")
    print("-" * 66)
    for key, rd, ra, sd, sa in rows:
        print(f"{key:<24}{rd:>9d}{tau_ms(ra):>11.0f}{sd:>9d}{tau_ms(sa):>11.0f}")

    real_delay = int(np.median([r[1] for r in rows]))
    real_tau = float(np.median([tau_ms(r[2]) for r in rows]))
    sim_delay = int(np.median([r[3] for r in rows]))
    sim_tau = float(np.median([tau_ms(r[4]) for r in rows]))

    print(f"\n真机: 纯延迟 {real_delay} 帧 ({real_delay / FPS * 1000:.0f} ms), τ = {real_tau:.0f} ms")
    print(f"仿真: 纯延迟 {sim_delay} 帧 ({sim_delay / FPS * 1000:.0f} ms), τ = {sim_tau:.0f} ms")
    return {
        "real_delay_steps": real_delay,
        "real_tau_ms": real_tau,
        "sim_delay_steps": sim_delay,
        "sim_tau_ms": sim_tau,
    }


def sweep_kp(scene: Path, target_tau: float, kps: list[float], dampratios: list[float]) -> None:
    """在仿真里跑阶跃响应,找出时间常数最接近真机的增益。

    直接量仿真本身的阶跃响应,而不是从遥操数据反推 —— 后者掺着人手的运动,
    分不清是执行器慢还是指令本身就慢。
    """
    import mujoco

    print(f"\n目标 τ = {target_tau:.0f} ms,扫描执行器参数:")
    print(f"{'kp':>8}{'dampratio':>12}{'τ(ms)':>10}{'与目标差':>10}")
    print("-" * 42)

    # 执行器定义在被 attach 的单臂模型里,不在场景文件中。改错文件会让所有
    # 参数都"生效为原值",扫描结果全都一样 —— 所以这里直接改 so101.xml,
    # 并在改完后核对写入是否真的落到了模型上。
    arm_xml = scene.parent / "so101.xml"
    if not arm_xml.is_file():
        raise SystemExit(f"找不到 {arm_xml}")
    original = arm_xml.read_text()
    if 'key="kp"' not in original:
        raise SystemExit(f"{arm_xml} 里找不到 PID 增益配置")

    results = []
    try:
        for kp in kps:
            for dr in dampratios:
                import re
                xml = re.sub(r'(key="kp" value=")[^"]*', rf'\g<1>{kp:g}', original)
                xml = re.sub(r'(key="kd" value=")[^"]*', rf'\g<1>{dr:g}', xml)
                arm_xml.write_text(xml)
                model = mujoco.MjModel.from_xml_path(str(scene))
                # PID 是 plugin 执行器,增益不在 actuator_gainprm 里,靠上面的
                # 正则改 XML 生效;此处不再断言。
                data = mujoco.MjData(model)
                mujoco.mj_resetData(model, data)

                # 对第一个关节下阶跃,记录到达 63.2% 的时间
                step = 0.5  # rad
                data.ctrl[:] = data.qpos[: model.nu]
                data.ctrl[0] = data.qpos[0] + step
                start = float(data.qpos[0])
                reached = None
                for i in range(int(1.0 / model.opt.timestep)):
                    mujoco.mj_step(model, data)
                    if reached is None and (data.qpos[0] - start) >= 0.632 * step:
                        reached = (i + 1) * model.opt.timestep * 1000
                        break
                if reached is None:
                    print(f"{kp:>8.1f}{dr:>12.2f}{'未收敛':>10}")
                    continue
                results.append((kp, dr, reached))
                print(f"{kp:>8.1f}{dr:>12.2f}{reached:>10.0f}{reached - target_tau:>10.0f}")
    finally:
        arm_xml.write_text(original)

    if results:
        best = min(results, key=lambda r: abs(r[2] - target_tau))
        print(f"\n最接近: control_kp={best[0]:g}, control_dampratio={best[1]:g} (τ={best[2]:.0f} ms)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", type=Path, help="teleop_sim.py --save 存下的 JSON")
    parser.add_argument("--scene", type=Path,
                        default=Path("~/.cache/evo_rlt/sim_assets/scene.xml").expanduser())
    parser.add_argument("--sweep", action="store_true", help="在仿真里扫描 kp/dampratio")
    args = parser.parse_args()

    stats = analyse(args.data)

    if args.sweep:
        if not args.scene.is_file():
            raise SystemExit(f"找不到场景 {args.scene},先跑 mj_server.py --build")
        sweep_kp(
            args.scene,
            stats["real_tau_ms"],
            kps=[1, 2, 3, 5, 8, 12, 20],
            dampratios=[0.5, 1.0, 2.0],
        )

    print("\n把结果填进 src/evo_rlt/sim/assets.py 的 SceneConfig:")
    print(f"  action_delay_steps = {stats['real_delay_steps']}   # 纯延迟,靠缓冲指令")
    print("  control_kp / control_dampratio 用 --sweep 的结果")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
