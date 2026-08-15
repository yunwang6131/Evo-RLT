#!/usr/bin/env python
"""逐个串口探测舵机应答,定位是哪条臂出问题。

只读,不写任何寄存器,不使能扭矩,机械臂不会动。

用法:
    python diagnostics/probe_arms.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

EXPECTED_IDS = [1, 2, 3, 4, 5, 6]


def probe(port: str) -> tuple[list[int], str | None]:
    """返回 (应答的电机 ID, 错误信息)。"""
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech.feetech import FeetechMotorsBus

    motors = {f"m{i}": Motor(i, "sts3215", MotorNormMode.DEGREES) for i in EXPECTED_IDS}
    bus = FeetechMotorsBus(port=port, motors=motors)
    try:
        # handshake=False: 不校验电机表,否则缺一个就直接抛错,看不到实际应答了谁
        bus.connect(handshake=False)
    except Exception as exc:
        return [], f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
    try:
        found = bus.broadcast_ping() or {}
        return sorted(found), None
    except Exception as exc:
        return [], f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
    finally:
        try:
            bus.disconnect()
        except Exception:
            pass


def _open_all(ports: list[str]) -> dict:
    """同时打开所有端口用于读取,返回 {port: bus}。"""
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech.feetech import FeetechMotorsBus

    buses = {}
    for port in ports:
        motors = {f"m{i}": Motor(i, "sts3215", MotorNormMode.DEGREES) for i in EXPECTED_IDS}
        bus = FeetechMotorsBus(port=port, motors=motors)
        try:
            bus.connect(handshake=False)
            bus.sync_read("Present_Position", normalize=False)
            buses[port] = bus
        except Exception:
            try:
                bus.disconnect()
            except Exception:
                pass
    return buses


def _read_all(buses: dict) -> dict[str, dict]:
    out = {}
    for port, bus in buses.items():
        try:
            out[port] = bus.sync_read("Present_Position", normalize=False)
        except Exception:
            pass
    return out


def identify(setup: Path) -> int:
    """靠物理移动认臂:动哪条,哪个端口的读数就变。

    端口号按 USB 插入顺序分配,重启或重插就会变,而四条臂的电机 ID 完全相同,
    仅凭通信无法区分。标错臂不会报错,只会把左臂的标定写进右臂的文件。
    """
    import time

    ports = sorted(str(p) for p in Path("/dev").glob("ttyACM*"))
    buses = _open_all(ports)
    if len(buses) < 2:
        print(f"只有 {len(buses)} 个端口可读,先跑一次不带 --identify 的探测")
        return 1

    by_id = {}
    for link in Path("/dev/serial/by-id").glob("*") if Path("/dev/serial/by-id").is_dir() else []:
        by_id[str(link.resolve())] = link.name

    print(f"已打开 {len(buses)} 个端口: {sorted(buses)}")
    print("接下来逐条确认。每次只动一条臂,幅度大一点。\n")

    found: dict[str, str] = {}
    try:
        for alias in ARMS_ORDER:
            input(f"  请抓住 【{alias}】 并保持不动,然后回车开始检测...")
            base = _read_all(buses)
            print(f"  现在慢慢来回晃动 【{alias}】 ...", end="", flush=True)

            moved: dict[str, float] = {}
            for _ in range(60):  # 约 6 秒
                time.sleep(0.1)
                now = _read_all(buses)
                for port in now:
                    if port not in base:
                        continue
                    delta = max(abs(now[port][m] - base[port][m]) for m in now[port])
                    moved[port] = max(moved.get(port, 0.0), delta)
            print(" 完成")

            ranked = sorted(moved.items(), key=lambda kv: -kv[1])
            best, best_delta = ranked[0]
            second = ranked[1][1] if len(ranked) > 1 else 0.0
            for port, delta in ranked:
                mark = " <-- 就是它" if port == best else ""
                print(f"      {port}  变化 {delta:6.0f} tick{mark}")

            if best_delta < 50:
                print(f"  没检测到明显移动(最大 {best_delta:.0f} tick),重来一次\n")
                return 1
            if best_delta < second * 3:
                print(f"  两个端口变化接近({best_delta:.0f} vs {second:.0f}),可能同时动了,重来\n")
                return 1
            found[alias] = best
            print(f"  {alias} = {best}\n")
    finally:
        for bus in buses.values():
            try:
                bus.disconnect()
            except Exception:
                pass

    print("=" * 66)
    print("识别结果:\n")
    print(f"{'臂':<16}{'端口':<16}{'稳定路径(by-id)'}")
    print("-" * 66)
    for alias, port in found.items():
        stable = by_id.get(port, "(无)")
        print(f"{alias:<16}{port:<16}{stable}")

    current = {}
    if setup.exists():
        for arm in json.loads(setup.read_text()).get("arms", []):
            current[arm.get("alias")] = arm.get("port")
    wrong = {a: (current.get(a), p) for a, p in found.items() if current.get(a) != p}
    if wrong:
        print("\n与 manifest 不符,标定前必须先改 manifest:")
        for alias, (had, now) in wrong.items():
            print(f"  {alias}: manifest 写的是 {had},实际是 {now}")
    else:
        print("\n与 manifest 一致。")

    print("\n把 manifest 的 port 换成 by-id 路径,重启后就不会错位:")
    for alias, port in found.items():
        stable = by_id.get(port)
        if stable:
            print(f'  "{alias}": "/dev/serial/by-id/{stable}"')
    return 0


#: 识别时的提问顺序。
ARMS_ORDER = ("left_follower", "right_follower", "left_leader", "right_leader")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setup", type=Path, default=Path("configs/my_so101_manifest.json"))
    parser.add_argument("--identify", action="store_true",
                        help="逐条晃动机械臂,认出每条臂对应哪个端口")
    args = parser.parse_args()

    if args.identify:
        return identify(args.setup)

    alias = {}
    if args.setup.exists():
        for arm in json.loads(args.setup.read_text()).get("arms", []):
            alias[arm["port"]] = arm.get("alias", "?")

    ports = sorted(str(p) for p in Path("/dev").glob("ttyACM*"))
    if not ports:
        print("没有找到 /dev/ttyACM* —— 检查 USB 连接")
        return 1

    print(f"{'端口':<16}{'manifest 别名':<18}{'应答电机':<24}状态")
    print("-" * 72)
    bad = 0
    for port in ports:
        found, error = probe(port)
        name = alias.get(port, "(不在 manifest)")
        if error:
            status, shown = f"打不开: {error}", "-"
            bad += 1
        elif not found:
            status, shown = "无应答 —— 舵机没通电?", "无"
            bad += 1
        elif found != EXPECTED_IDS:
            status, shown = f"缺 {sorted(set(EXPECTED_IDS) - set(found))}", str(found)
            bad += 1
        else:
            status, shown = "正常", str(found)
        print(f"{port:<16}{name:<18}{shown:<24}{status}")

    missing = [p for p in alias if p not in ports]
    if missing:
        print(f"\nmanifest 里有但系统上不存在的端口: {missing}")
        bad += 1

    if bad:
        print("\n串口是 USB 供电,舵机是外部 12V 供电 —— 臂没电时端口照样存在但不应答。")
        print("先查电源,再查线缆。刚上电时偶发不应答,重跑一次通常就好。")
        return 1
    print("\n四条臂全部正常。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
