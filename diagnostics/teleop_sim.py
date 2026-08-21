#!/usr/bin/env python
"""Drive the simulator from the real SO-101 leaders, and pin down joint directions.

``sign`` in `calib.py` defaults to +1 for every joint, which encodes an
assumption that each motor's positive direction matches its URDF axis. That is a
property of the physical build, so it has to be measured. Until it is, a wrong
sign means the simulated arm mirrors the real one on that joint and every
recorded action is wrong in a way no amount of training will fix.

端口按 configs/arms.json 里的 USB 序列号自动定位,不依赖 ttyACM 序号。
标定读本项目 configs/calibration/,不碰其他项目的共享缓存。

Two modes, picked automatically from what is plugged in:

**paired** (leaders + real followers) -- the leaders drive both the real arms
and the simulator. Both report through the same calibrated value space, so the
two streams should track each other; a mirrored joint shows up as a negative
correlation. Fully automatic, and the recommended way to do this.

**solo** (leaders only) -- the leaders drive the simulator alone. Correlation
against the commanded value cannot detect a sign error, because the command is
what produced the angle in the first place. So this mode instead reports how far
each joint moved and leaves the judgement to you: watch the viewer window
(``mj_server.py --viewer``) and confirm each joint turns the same way the real
arm would.

Safety: in paired mode the real followers move. This is ordinary teleop -- the
same thing recording does -- but the arms do move, so keep the workspace clear
and keep a hand near the leaders. Ctrl-C stops immediately.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

# Reuse the record pipeline's own id constants and calibration staging, so a
# leader driving the simulator is set up exactly like a leader driving a
# recording session.
from evo_rlt.adapters.lerobot.record.common import (  # noqa: E402
    ARM_FREEZE_KEY,
    ArmFreeze,
    install_safe_follower_torque_enable,
)
from evo_rlt.sim.arms import (  # noqa: E402
    ArmResolveError,
    build_device,
    calibration_status,
    resolve_all,
)
from evo_rlt.sim.calib import MOTOR_NAMES  # noqa: E402
from evo_rlt.sim.feedback import FeedbackGains, LeaderForceFeedback, LeaderLock  # noqa: E402
from evo_rlt.sim.protocol import ARM_SIDES, DEFAULT_ENDPOINT  # noqa: E402
from evo_rlt.sim.sim_robot import make_sim_robot  # noqa: E402

#: `build_robot_argv` passes --robot.id=bimanual for the dual follower.
FOLLOWER_ID = "bimanual"

#: STS3215 servos intermittently miss a status packet right after power-up,
#: and LeRobot does not retry by default (its errors read "after 1 tries").
CONNECT_ATTEMPTS = 3
CONNECT_RETRY_S = 1.5


class ServoError(RuntimeError):
    """A real arm failed to talk, as opposed to the simulator being absent."""


def _connect_with_retry(device, label: str):
    """Connect to a real arm, retrying transient bus failures.

    A missed status packet on power-up is not a real fault, but LeRobot gives up
    after one attempt and the whole session dies on it.
    """
    for attempt in range(1, CONNECT_ATTEMPTS + 1):
        try:
            device.connect(calibrate=False)
            return
        except Exception as exc:
            try:
                device.disconnect()
            except Exception:
                pass  # never connected, or already torn down
            if attempt == CONNECT_ATTEMPTS:
                raise ServoError(f"{label} 连接失败({attempt} 次尝试): {exc}") from exc
            print(f"  {label} 第 {attempt} 次连接失败,{CONNECT_RETRY_S}s 后重试: "
                  f"{str(exc).splitlines()[0]}")
            time.sleep(CONNECT_RETRY_S)

#: A joint must move at least this much (in transported units) before its
#: direction is called: below it, the correlation is just noise.
MIN_TRAVEL = 4.0


def require_calibrated(aliases: list[str]) -> None:
    """本项目没标定过就直接停,不要退回去读别的项目的标定。"""
    status = calibration_status()
    missing = [a for a in aliases if not status.get(a)]
    if missing:
        raise SystemExit(
            f"{missing} 还没在本项目标定过。先跑:\n"
            + "\n".join(f"  python diagnostics/calibration.py --arm {a}" for a in missing)
        )


def follower_safe_bounds(bridge) -> dict[str, tuple[float, float]]:
    """每个关节允许下发的取值区间,取自 follower 自己的标定行程。

    DEGREES 模式下 LeRobot 的 ``_unnormalize`` 不做钳位,而 leader 是被动的、
    能被推到任意角度。leader 的极限映射到 follower 可能落在它机械限位之外,
    follower 跟随时就会堵转,STS3215 触发过载保护后直接停止应答 —— 表现为
    跑到一半整条总线失联。
    """
    bounds: dict[str, tuple[float, float]] = {}
    for side in ARM_SIDES:
        arm = bridge.arm(side)
        for name in MOTOR_NAMES:
            motor = arm.motors[name]
            lo = arm.ticks_to_value(name, motor.range_min)
            hi = arm.ticks_to_value(name, motor.range_max)
            bounds[f"{side}_{name}.pos"] = (min(lo, hi), max(lo, hi))
    return bounds


def clip_action(action: dict, bounds: dict, hits: dict[str, int]) -> dict:
    """钳位到安全行程,并统计每个关节被钳了多少次。"""
    out = {}
    for key, value in action.items():
        lo, hi = bounds[key]
        clipped = min(hi, max(lo, value))
        if clipped != value:
            hits[key] = hits.get(key, 0) + 1
        out[key] = clipped
    return out


def show_frames(observation: dict, window: str, info: str) -> bool:
    """把三路相机画面并排显示。返回 False 表示用户按了 q/ESC。

    显示的是 observation 里的图像,也就是策略真正会看到的输入 —— 比另开
    viewer 看仿真全景更贴近训练时的视角。
    """
    import cv2
    import numpy as np

    frames = []
    for key in ("left_wrist", "right_wrist", "right_front"):
        img = observation.get(key)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
        cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(img, key, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        frames.append(img)
    if not frames:
        return True

    canvas = np.hstack(frames)
    bar = np.zeros((26, canvas.shape[1], 3), np.uint8)
    cv2.putText(bar, info, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow(window, np.vstack([bar, canvas]))
    return (cv2.waitKey(1) & 0xFF) not in (27, ord("q"))


class KeyWatcher:
    """让遥操循环里能非阻塞地读单个按键。

    遥操时两只手都在主臂上,而仿真器和遥操各占着一个终端 —— 想复位零件就得
    再开第三个终端去跑 reset_objects.py。改成按键,松一只手敲一下就行。

    把终端切到 cbreak:按键立即可读,不用等回车,同时保留 ISIG 所以 Ctrl-C
    照常生效。**退出时一定要还原** —— cbreak 泄漏到 shell 会变成不回显、
    Ctrl-C 失效,得盲敲 reset 才能救回来,所以用 finally 保证还原。

    stdin 不是 TTY(输出被重定向、在 IDE 里跑)时整个功能自动关掉,不报错。
    """

    def __init__(self) -> None:
        self.enabled = False
        self._fd = None
        self._saved = None

    def open(self) -> KeyWatcher:
        try:
            import termios
            import tty
        except ImportError:      # 非 POSIX
            return self
        if not sys.stdin.isatty():
            return self
        self._fd = sys.stdin.fileno()
        try:
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except Exception:
            self._fd = self._saved = None
            return self
        self.enabled = True
        return self

    def close(self) -> None:
        if self._saved is None:
            return
        import termios

        try:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
        finally:
            self.enabled = False
            self._saved = None

    def pressed(self) -> list[str]:
        """这一轮按下的键,没有就是空表。绝不阻塞。

        直接 ``os.read`` 读文件描述符,不走 ``sys.stdin``:后者带缓冲,
        ``read(1)`` 会一次性从 fd 抓走一批数据却只返回一个字符,而 select
        查的是 fd —— 剩下的字符卡在 Python 缓冲区里,select 报"无数据",
        连按两下就只认第一下。实测按 "rbq" 只读到 "r"。
        """
        if not self.enabled:
            return []
        import os
        import select

        out: list[str] = []
        try:
            while select.select([self._fd], [], [], 0)[0]:
                chunk = os.read(self._fd, 16)
                if not chunk:
                    break
                out.extend(chunk.decode("utf-8", "ignore"))
        except OSError:
            # 终端没了(窗口被关、进程被切到后台)。这只是个便利功能,不值得
            # 让整轮遥操崩掉把已采集的数据一起丢了 —— 自己关掉,继续跑。
            self.enabled = False
            self._saved = None
            print("\n  (stdin 已失效,按键功能关闭;遥操继续)")
        return out


#: 复位全部零件的键。**只占这一个键** —— 后面 RLT 的人工干预还要在遥操里绑
#: 按键,键位留给它,别在这儿铺开。提前结束用 Ctrl-C,不另占键。
RESET_KEY = "b"


def _blend(start: dict, target: dict, alpha: float) -> dict:
    """Interpolate from a measured pose toward the leaders' pose."""
    return {k: start[k] * (1.0 - alpha) + v * alpha for k, v in target.items() if k in start}


def report_pose_gap(teleop, real, sim, keys: list[str]) -> float:
    """Print how far the arms have to travel to reach the leaders, and warn.

    A large gap is the moment where teleop can hurt something, so it is shown
    before anything moves rather than discovered afterwards.
    """
    leader = teleop.get_action()
    target = real if real is not None else sim
    label = "real follower" if real is not None else "sim"
    measured = target.get_observation()

    gaps = {k: abs(leader[k] - measured[k]) for k in keys if k in leader and k in measured}
    worst_key = max(gaps, key=gaps.get)
    worst = gaps[worst_key]
    if worst > 20.0:
        print(f"\n  WARNING: {label} is far from the leaders "
              f"({worst_key} off by {worst:.0f}). Largest movers:")
        for k in sorted(gaps, key=gaps.get, reverse=True)[:4]:
            print(f"      {k:<26} {measured[k]:7.1f} -> {leader[k]:7.1f}  ({gaps[k]:.0f})")
        print("  The ramp will take it there gradually. Ctrl-C now if that is unsafe.")
    return worst


class ArmPair:
    """把两条单臂设备合成双臂接口,键名带 left_/right_ 前缀。

    不用 LeRobot 的 BiSOLeader/BiSOFollower:那两个类会把父 id 拼成
    ``{id}_left``,标定文件名因此受双臂 id 影响,难以和本项目的
    ``evosim_<alias>`` 命名对齐。分开构造则每条臂各读各的标定。
    """

    def __init__(self, left, right):
        self.left, self.right = left, right

    def _merge(self, method: str) -> dict:
        out = {}
        for side, dev in (("left", self.left), ("right", self.right)):
            for key, value in getattr(dev, method)().items():
                out[f"{side}_{key}"] = value
        return out

    def get_action(self) -> dict:
        return self._merge("get_action")

    def get_observation(self) -> dict:
        return self._merge("get_observation")

    def send_action(self, action: dict) -> dict:
        out = {}
        for side, dev in (("left", self.left), ("right", self.right)):
            prefix = f"{side}_"
            sub = {k.removeprefix(prefix): v for k, v in action.items() if k.startswith(prefix)}
            for key, value in dev.send_action(sub).items():
                out[f"{prefix}{key}"] = value
        return out

    def disconnect(self) -> None:
        for dev in (self.left, self.right):
            try:
                dev.disconnect()
            except Exception:
                pass


def _check_calibration_loaded(dev, alias: str) -> None:
    """未加载标定的总线会退回原始 tick,每个值都带巨大的静默偏移。"""
    if not dev.bus.calibration:
        raise ServoError(f"{alias} 没加载到标定,应在 {dev.calibration_fpath}")


def connect_pair(kind: str, ports: dict[str, str]) -> ArmPair:
    """连接一对臂(follower 或 leader)。"""
    devices = {}
    for side in ("left", "right"):
        alias = f"{side}_{kind}"
        dev = build_device(alias, ports[alias])
        if kind == "follower":
            # 必须在 connect 之前:STS3215 的 Goal_Position 断电后可能读成 0,
            # torque 一开所有关节就冲向零位。
            install_safe_follower_torque_enable(dev)
        _connect_with_retry(dev, alias)
        _check_calibration_loaded(dev, alias)
        devices[side] = dev
    return ArmPair(devices["left"], devices["right"])


def run(args) -> int:
    ports = resolve_all()
    leaders = ["left_leader", "right_leader"]
    followers = ["left_follower", "right_follower"]

    if not all(a in ports for a in leaders):
        raise ArmResolveError(f"leader 不在线,当前可用: {sorted(ports)}")
    paired = not args.no_followers and all(a in ports for a in followers)

    require_calibrated(leaders + (followers if paired else []))

    print(f"mode: {'paired' if paired else 'solo'}")
    for alias in leaders + (followers if paired else []):
        print(f"  {alias:<16} {ports[alias]}")
    if paired:
        print("\n  leader 同时驱动真机和仿真,真机会动,确认工作区无障碍。")
    else:
        print("\n  follower 不在线,只驱动仿真。方向要看 viewer 判断。")

    robot = make_sim_robot(endpoint=args.endpoint, fps=int(args.fps))
    robot.connect()
    teleop = connect_pair("leader", ports)
    real = connect_pair("follower", ports) if paired else None

    keys = [f"{side}_{motor}.pos" for side in ARM_SIDES for motor in MOTOR_NAMES]
    commanded: dict[str, list[float]] = {k: [] for k in keys}
    sim_measured: dict[str, list[float]] = {k: [] for k in keys}
    real_measured: dict[str, list[float]] = {k: [] for k in keys}

    period = 1.0 / args.fps
    bounds = follower_safe_bounds(robot.calibration_bridge)
    clip_hits: dict[str, int] = {}

    # Ramp onto the leaders' pose instead of jumping to it. The followers start
    # wherever they were left, which can be far from where the leaders are
    # holding; commanding that difference in one step is a hard lunge. Ramping
    # also keeps the startup transient -- driven by the reset pose, not by the
    # leaders -- out of the correlation, where it could mask or invent a flip.
    gap = report_pose_gap(teleop, real, robot, keys)
    print(f"\nramping onto the leaders' pose over {args.warmup} steps "
          f"(largest gap {gap:.1f})...")
    start_real = real.get_observation() if real is not None else None
    start_sim = robot.get_observation()
    for i in range(args.warmup):
        loop_start = time.perf_counter()
        alpha = (i + 1) / args.warmup
        leader_now = clip_action(
            {k: v for k, v in teleop.get_action().items() if k in commanded}, bounds, clip_hits
        )
        robot.send_action(_blend(start_sim, leader_now, alpha))
        if real is not None:
            real.send_action(_blend(start_real, leader_now, alpha))
        time.sleep(max(0.0, period - (time.perf_counter() - loop_start)))

    # 力反馈在 ramp **之后**才通电。ramp 期间从臂离主臂还很远,这时候把从臂的
    # 位置写给主臂,主臂会朝那个远处的位姿猛拽操作者的手。
    # 冻结右臂:双臂任务单人采数据时,一只手当夹具腾出另一只手。
    # 只做右臂 —— 左臂是主操作臂,而且 RLT 的 rl_action_arms=left 也只管左臂。
    # 冻结时把**右主臂**也锁住(通电顶住),否则操作者的手会在冻结期间把主臂
    # 带到别处,解冻那一刻两者差几十度。锁住之后解冻是无缝的。
    freeze = ArmFreeze("right", fps=args.fps,
                       leader_lock=(None if args.no_arm_lock
                                    else LeaderLock(teleop.right, args.lock_torque or 80.0)))

    feedback: LeaderForceFeedback | None = None
    if args.force_feedback:
        feedback = LeaderForceFeedback(teleop, FeedbackGains(
            gain=args.fb_gain, deadband=args.fb_deadband, torque_percent=args.fb_torque,
        ))
        limits = feedback.engage()
        source = "真机 follower" if real is not None else "仿真"
        print(f"\n  力反馈已启用(阻力来自{source}被挡住的程度)。")
        print(f"  gain={args.fb_gain:g} deadband={args.fb_deadband:g} "
              f"力矩上限={args.fb_torque:g}% (寄存器读回 {sorted(set(limits.values()))})")
        print("  自由运动时主臂是断电的,和平时一样;只有从臂真被挡住才通电出力。")
        print("  通电时手要握住。嗡嗡震颤=环路振荡,降 gain;咔咔通断=死区太小,加大 deadband。")

    print(f"\nteleoperating at {args.fps:g} Hz for {args.duration:g}s -- move every joint through its range")

    started = time.perf_counter()
    ticks = 0
    failure: Exception | None = None
    show_ok = args.show
    if args.show:
        # 这个进程装的是 lerobot 依赖的 headless opencv,没有 GUI。
        import cv2

        if not hasattr(cv2, "imshow") or "headless" in getattr(cv2, "__file__", ""):
            show_ok = False
        try:
            cv2.namedWindow("probe")
            cv2.destroyWindow("probe")
        except Exception:
            show_ok = False
        if not show_ok:
            print("本环境的 opencv 是 headless 版(lerobot 依赖),无法显示。")
            print("改在仿真进程开窗口: mj_server.py --show-cameras\n")
    keyboard = KeyWatcher().open()
    if keyboard.enabled:
        print(f"按 {RESET_KEY} 复位全部零件,按 {ARM_FREEZE_KEY} 冻结/解冻右臂(绿)。"
              f"Ctrl-C 停止。\n")
    else:
        print("(stdin 不是终端,按键已关闭;复位改用 diagnostics/reset_objects.py)")
        print("Ctrl-C 停止。\n")

    try:
        while time.perf_counter() - started < args.duration:
            loop_start = time.perf_counter()

            keys = [ch.lower() for ch in keyboard.pressed()]
            if RESET_KEY in keys:
                # 只动零件,手臂不受影响 —— 遥操的手感不会断
                done = robot.reset_objects()
                print(f"\n  [{ticks}] 已复位 {' '.join(done)}")

            action = teleop.get_action()
            action = {k: v for k, v in action.items() if k in commanded}
            if ARM_FREEZE_KEY in keys:
                state = freeze.toggle(action)
                label = {"frozen": "已冻结", "blending": "解冻中(平滑交回)"}[state]
                extra = ""
                if state == "frozen" and freeze._lock_readback:
                    vals = sorted(set(freeze._lock_readback.values()))
                    extra = f"  主臂力矩上限读回 {vals}"
                print(f"\n  [{ticks}] 右臂{label}{extra}")
            # 冻结要在钳位**之前**:钳位是按 follower 行程做的,而冻结值本身就是
            # 钳过的,再钳一次无害;反过来先钳后冻则会把过渡段的插值算错。
            action = freeze.apply(action)
            # Clip before anything is sent, and record the clipped values, so
            # commanded and measured stay comparable at the limits.
            action = clip_action(action, bounds, clip_hits)

            robot.send_action(action)
            if real is not None:
                real.send_action(action)

            sim_obs = robot.get_observation()
            real_obs = real.get_observation() if real is not None else None

            if feedback is not None:
                # 有真机从臂时用它的实测 —— 那才是"真的被挡住了"的物理来源;
                # solo 模式下退回仿真的实测,同样是 qpos 不是 ctrl。
                # 传的是**指令**而不是主臂位置:见 feedback.py 模块说明,
                # 接在主臂-从臂位置差上会把正常跟随的滞后当成被挡住。
                feedback.update(real_obs if real_obs is not None else sim_obs, action)

            for key in keys:
                if key in action:
                    commanded[key].append(action[key])
                    sim_measured[key].append(sim_obs[key])
                    if real_obs is not None:
                        real_measured[key].append(real_obs[key])

            ticks += 1
            if args.show and ticks % 2 == 0 and show_ok:
                # 隔帧显示:cv2.imshow 本身要几毫秒,每帧都画会吃掉控制周期预算
                elapsed = time.perf_counter() - started
                if not show_frames(
                    sim_obs, "sim cameras (q 退出)",
                    f"{elapsed:5.1f}s / {args.duration:g}s   {'paired' if real is not None else 'solo'}",
                ):
                    print("\n窗口关闭,提前结束")
                    break
            if ticks % int(args.fps) == 0:
                elapsed = time.perf_counter() - started
                print(f"  {elapsed:5.1f}s / {args.duration:g}s", end="\r", flush=True)

            time.sleep(max(0.0, period - (time.perf_counter() - loop_start)))
    except KeyboardInterrupt:
        print("\n已中断,用已采集的数据出报告")
    except Exception as exc:
        # Report on whatever was collected rather than throwing it away: the
        # arms were moved by hand and that effort should not be wasted just
        # because the bus dropped a packet near the end.
        failure = exc
        print(f"\n\n采集中断于第 {ticks} 步: {type(exc).__name__}: {str(exc).splitlines()[0]}")
    finally:
        # Each teardown step is isolated: a servo that fails to disable torque
        # must not mask the error that actually stopped the run, nor prevent
        # the remaining devices from being released.
        #
        # 终端设置**第一个**还原:后面几步任何一个卡住或抛异常,都不能把 shell
        # 留在 cbreak 模式里 —— 那会变成不回显、Ctrl-C 失效,只能盲敲 reset。
        keyboard.close()
        # 断力矩排在 disconnect **之前**:总线关掉之后就写不进寄存器了,
        # 主臂会带着力矩留在原地,手一推就顶。
        freeze.release()
        if feedback is not None:
            feedback.release()
            print("  力反馈已关闭,主臂恢复自由拖动。")
            for line in feedback.summary():
                print(line)
        print(freeze.summary())
        if args.show:
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:
                pass
        for label, fn in (
            ("sim", robot.disconnect),
            ("leader", teleop.disconnect),
            *((("follower", real.disconnect),) if real is not None else ()),
        ):
            try:
                fn()
            except Exception as exc:
                print(f"  断开 {label} 时出错(已忽略): {str(exc).splitlines()[0]}")

    if failure is not None:
        print("\n注意: 本次是异常结束,下面的结论基于中断前的数据。")
        print("若某关节此时还没扳到,它的判定不可信。")

    print(f"\ncollected {ticks} steps\n")
    if clip_hits:
        print("以下关节被推超了 follower 的标定行程,已钳位(这正是会让舵机堵转的操作):")
        for key, count in sorted(clip_hits.items(), key=lambda kv: -kv[1]):
            lo, hi = bounds[key]
            print(f"  {key:<26} {count:5d} 步  安全区间 [{lo:.1f}, {hi:.1f}]")
        print()
    if args.save:
        # 原始序列也一并存下:报告是聚合结论,出了争议要回看逐步数据。
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps({
            "mode": "paired" if paired else "solo",
            "steps": ticks,
            "fps": args.fps,
            "clip_hits": clip_hits,
            "bounds": {k: list(v) for k, v in bounds.items()},
            "commanded": commanded,
            "sim_measured": sim_measured,
            "real_measured": real_measured if paired else None,
        }, indent=1))
        print(f"原始数据已存到 {args.save}\n")

    if ticks < 10:
        print("too few samples to judge anything")
        return 1

    return report(commanded, sim_measured, real_measured if paired else None, args)


def report(commanded, sim_measured, real_measured, args) -> int:
    """Compare the streams and, in paired mode, recommend joint_signs."""
    suggested: dict[str, dict[str, float]] = {}
    print(f"{'joint':<26}{'travel':>9}{'sim travel':>12}", end="")
    print(f"{'corr(real,sim)':>16}{'verdict':>12}" if real_measured else f"{'verdict':>12}")
    print("-" * (75 if real_measured else 59))

    problems = 0
    for side in ARM_SIDES:
        for motor in MOTOR_NAMES:
            key = f"{side}_{motor}.pos"
            cmd = np.array(commanded[key])
            sim = np.array(sim_measured[key])
            travel = float(cmd.max() - cmd.min()) if cmd.size else 0.0
            sim_travel = float(sim.max() - sim.min()) if sim.size else 0.0

            if real_measured is None:
                verdict = "moved" if travel >= MIN_TRAVEL else "NOT MOVED"
                if travel < MIN_TRAVEL:
                    problems += 1
                print(f"{key:<26}{travel:>9.1f}{sim_travel:>12.1f}{verdict:>12}")
                continue

            real = np.array(real_measured[key])
            if travel < MIN_TRAVEL:
                print(f"{key:<26}{travel:>9.1f}{sim_travel:>12.1f}{'--':>16}{'NOT MOVED':>12}")
                problems += 1
                continue

            # Correlate the two *measured* streams. Both are in the same
            # calibrated value space, so a mirrored joint reads as negative.
            if real.std() < 1e-6 or sim.std() < 1e-6:
                corr = float("nan")
                verdict = "NO SIGNAL"
                problems += 1
            else:
                corr = float(np.corrcoef(real, sim)[0, 1])
                if corr < -args.corr_threshold:
                    verdict = "FLIP"
                    suggested.setdefault(side, {})[motor] = -1.0
                    problems += 1
                elif corr > args.corr_threshold:
                    verdict = "ok"
                else:
                    verdict = "UNCLEAR"
                    problems += 1
            print(f"{key:<26}{travel:>9.1f}{sim_travel:>12.1f}{corr:>16.3f}{verdict:>12}")

    print()
    if real_measured is None:
        print("Solo mode: travel only. Watch the viewer and confirm each joint turns")
        print("the same way the real arm does, then set joint_signs by hand.")
        if problems:
            print(f"\n{problems} joint(s) barely moved -- exercise them and re-run.")
        return 0

    if suggested:
        print("Mirrored joints found. Put this in your SimRobotConfig:\n")
        print(f"  joint_signs={json.dumps(suggested, indent=4)}")
        print("\nThen re-run: every joint should come back 'ok'.")
    elif problems:
        print(f"{problems} joint(s) inconclusive -- move them through a wider range and re-run.")
    else:
        print("All joints track the real arms. joint_signs can stay empty.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--warmup", type=int, default=90,
                        help="ramp steps before recording (90 @ 30 Hz = 3 s); also drops "
                             "the startup transient out of the correlation")
    parser.add_argument("--no-followers", action="store_true",
                        help="never drive the real arms, even if they are connected")
    parser.add_argument("--corr-threshold", type=float, default=0.5)
    parser.add_argument("--show", action="store_true",
                        help="(已废弃)改用 mj_server.py --show-cameras")
    parser.add_argument("--save", type=Path, default=None,
                        help="把逐步原始数据存成 JSON,便于事后复查或让人代为分析")
    parser.add_argument("--lock-torque", type=float, default=None,
                        help="按 p 锁住右主臂时的力矩上限(占满量程%%)。"
                             "默认 80。塌下去就调高,拧不动就调低")
    parser.add_argument("--no-arm-lock", action="store_true",
                        help="按 p 冻结右臂时,不锁住右主臂。默认会锁(通电顶住),"
                             "免得解冻时主从差太多")
    parser.add_argument("--force-feedback", action="store_true",
                        help="从臂被挡住时让主臂给手阻力。**主臂会通电主动出力**,"
                             "默认关闭,开之前先看 sim/feedback.py 的说明")
    parser.add_argument("--fb-gain", type=float, default=FeedbackGains.gain,
                        help="主臂朝从臂位置移动的比例(默认 %(default)s,越大越硬也越易振荡)")
    parser.add_argument("--fb-deadband", type=float, default=FeedbackGains.deadband,
                        help="死区,位置差小于它不出力(默认 %(default)s)")
    parser.add_argument("--fb-torque", type=float, default=FeedbackGains.torque_percent,
                        help="主臂力矩上限,占满量程百分比(默认 %(default)s)")
    args = parser.parse_args()
    try:
        return run(args)
    except ServoError as exc:
        # Servo faults also surface as ConnectionError, so they must be caught
        # first -- otherwise a dead motor gets blamed on the simulator.
        print(f"\n真机通信失败: {exc}")
        print("\n舵机没应答。逐端口探测哪条臂有问题:")
        print("  python diagnostics/probe_arms.py")
        print("\n常见原因: 舵机没通电(串口是 USB 供电,臂没电时端口照样在);")
        print("刚上电时偶发不应答,直接重跑一次通常就好。")
        return 1
    except ConnectionError as exc:
        print(f"\n仿真器连接失败: {exc}")
        print("\n仿真器在跑吗?")
        print("  ~/anaconda3/envs/rlt_sim/bin/python src/evo_rlt/sim/mj_server.py --viewer")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
