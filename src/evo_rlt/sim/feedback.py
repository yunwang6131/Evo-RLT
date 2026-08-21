"""让主臂在从臂被挡住时给手一个阻力(双边遥操)。

阻力接在**从臂的"实测 - 指令"**上,即"我让它去哪、它实际到了哪"的差。
自由运动时这个差只有伺服滞后那一点点;被桌子顶住或夹爪夹到东西时,指令继续
往前走而实测停住,差持续变大 —— 把它乘个增益加到主臂当前位置上写进
``Goal_Position``,主臂舵机就往回拽操作者的手。上限由 ``Torque_Limit`` 卡死。

**不能接在"主臂位置 - 从臂位置"上。** 第一版就是这么写的,结果全程都有很大
阻力:实测纯延迟是 3 个控制步(100 ms),操作者以 1.5 rad/s(约 86°/s)挥臂时
从臂就落后 8.6°,远超 2° 的死区 —— 于是"正常跟随的滞后"被当成了"被挡住"。
用"实测 - 指令"就没有这个问题,因为纯延迟在两边同时出现,对齐后自己抵消。
对齐用的步数见 ``DEFAULT_DELAY_STEPS``,残余误差实测 p95 = 2.3、p99 = 4.8。

**只在被挡住时才通电。** 这是第二版的教训:通电的位置伺服本身就有黏滞阻力,
和反馈力无关 —— ``Goal_Position`` 每 33 ms 才刷新一次,操作者一动,伺服的目标
就成了一个周期之前的旧位置,于是往回拽。以 86°/s 挥臂时目标落后 2.9°,乘上
舵机的位置增益就是持续的"费力",全程都有。所以自由运动时必须真的**断力矩**,
只有误差超过死区才通电,回落到 ``RELEASE_RATIO`` 倍死区以下就断开。

通断目前是**按整条臂**的:该臂任一关节超死区,整臂通电。粒度可以再细到按
关节(``enable_torque`` 支持传关节名列表),好处是夹爪夹住东西时不会连累同臂
其余 5 个关节 —— 实测一次遥操里 ``right_gripper`` 超死区 141 步,而同臂的
wrist_flex/wrist_roll 最大误差只有 1.6/1.1。没做是因为按臂已经够用,
真觉得夹住东西时整条臂发紧再改。

数据本来就有:``mj_server.joint_positions()`` 读的是 ``data.qpos`` 而不是回显
``ctrl``,真机 follower 的 ``get_observation()`` 同理。缺的只是把主臂通上电。

**为什么增益要小、还要留死区**

这是 position-position 双边环,在 30 Hz 控制率 + 实测约 100 ms 纯延迟下,
刚性耦合必然自激振荡(触觉回路通常要 500~1000 Hz)。所以:

* ``gain`` < 1:主臂只朝从臂的位置走一部分,做成"提示性阻力"而不是"硬墙"。
* ``deadband``:正常跟随时两者本来就差零点几度(伺服滞后),不设死区的话
  主臂会一直微幅出力,握着的手感觉是持续的嗡嗡震颤。
* ``torque_limit``:硬上限。人任何时候都要能轻松拧过它。

**安全**

主臂会通电主动出力,而操作者的手正握着它。所以:

* 默认不启用,要显式打开。
* ``release()`` 必须在任何退出路径上跑到 —— 用 ``with`` 语句,别手工配对。
* 力矩上限默认压到很低。调高之前先在从臂是仿真的情况下试。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

#: 主臂力矩上限的寄存器单位。Feetech STS 系列的 ``Torque_Limit``(地址 48,
#: 2 字节)是千分比:1000 = 100%,和 EEPROM 里的 ``Max_Torque_Limit`` 同制。
#: 上电时 ``Torque_Limit`` 由 ``Max_Torque_Limit`` 初始化。
TORQUE_LIMIT_FULL = 1000

#: 关节位置的键后缀,和 LeRobot 的 observation/action 一致。
POS_SUFFIX = ".pos"

#: 指令到实测之间的纯延迟(控制步)。和 ``protocol.DEFAULT_ACTION_DELAY_STEPS``
#: 同源:用 outputs/sign_check.json 的 209 秒真机数据扫出来的(3 步 = 100 ms,
#: RMSE 在 0/1/2/3/4/5/6 步里最低)。对齐用它,不对齐的话正常跟随的滞后会被
#: 当成"被挡住"。
#:
#: 对齐之后的残余误差(同一份数据,真机 follower,|实测-指令|):
#:   延迟步   中位数   p95    p99
#:     0      0.35    5.10   13.80
#:     3      0.35    2.33    4.79   <- 现值
#:     4      0.35    2.37    4.44
#:     6      0.35    3.91    9.80
#: 死区必须大于这个残余,否则正常跟随会被判成被挡住。
DEFAULT_DELAY_STEPS = 3

#: 松开的滞回比例:误差回落到 ``deadband * RELEASE_RATIO`` 以下才断力矩。
#: 不留滞回的话误差在死区边界附近抖动时会反复通断,主臂咔咔响。
RELEASE_RATIO = 0.5


@dataclass(frozen=True)
class FeedbackGains:
    """双边环的三个旋钮。默认值按"宁可太弱"选。

    ``gain`` 和 ``deadband`` 的单位跟随 LeRobot 的归一化值空间:非夹爪关节
    是度,夹爪是开度百分比。两者量纲不同但量级接近(行程都是 100 出头),
    所以共用一组阈值是够用的;要分开调再拆。
    """

    #: 主臂朝从臂实测位置移动的比例。1.0 = 完全跟随(硬),0 = 不出力。
    gain: float = 0.3
    #: 死区。误差小于它时主臂**完全断电**,和没开这个功能一样自由。
    #: 必须大于正常跟随的残余误差(见 DEFAULT_DELAY_STEPS 那张表:p95=2.3、
    #: p99=4.8),否则自由运动时会被误判成被挡住。取 5.0 覆盖到 p99。
    deadband: float = 5.0
    #: 力矩上限,占舵机满量程的百分比。人必须能轻松拧过它。
    torque_percent: float = 15.0

    def torque_register(self) -> int:
        """换算成 ``Torque_Limit`` 寄存器的整数值。"""
        if not 0.0 <= self.torque_percent <= 100.0:
            raise ValueError(f"torque_percent 应在 0~100,给的是 {self.torque_percent}")
        return int(round(self.torque_percent / 100.0 * TORQUE_LIMIT_FULL))


def blocking_error(
    measured: dict[str, float],
    commanded: dict[str, float],
) -> dict[str, float]:
    """从臂"被挡住"的程度 = 实测 - 指令。纯函数。

    两个字典都是 ``{关节名}.pos -> 归一化值``,而且**必须是时间对齐的** ——
    ``commanded`` 要取纯延迟之前的那一条,否则这个差里仍然混着延迟(调用方
    用 ``LeaderForceFeedback`` 的队列对齐,别自己传当前指令)。

    符号:从臂停在指令后面时为负(沿运动方向),前面时为正。
    """
    out: dict[str, float] = {}
    for key, value in measured.items():
        if not key.endswith(POS_SUFFIX):
            continue
        target = commanded.get(key)
        if target is None:
            continue
        out[key] = float(value) - float(target)
    return out


def goal_positions(
    leader: dict[str, float],
    error: dict[str, float],
    gains: FeedbackGains,
) -> dict[str, float]:
    """算出要写给主臂的目标位置。纯函数,不碰硬件。

    ``leader`` 是主臂当前位置,``error`` 是 `blocking_error` 的输出,键都是
    ``{关节名}.pos``(调用方负责剥掉 ``left_``/``right_`` 前缀)。

    死区内返回主臂**当前**位置 —— 目标等于现状,位置环误差为零,不出力。
    死区外把误差减掉死区宽度再乘增益,这样力是从 0 连续长起来的;直接乘
    增益的话跨过死区边界会有一个力的跳变,手上是"咔"的一下。
    """
    out: dict[str, float] = {}
    for key, lead in leader.items():
        if not key.endswith(POS_SUFFIX):
            continue
        err = error.get(key)
        if err is None:
            continue
        if abs(err) <= gains.deadband:
            out[key] = float(lead)
            continue
        trimmed = err - math.copysign(gains.deadband, err)
        out[key] = float(lead) + gains.gain * trimmed
    return out


class LeaderForceFeedback:
    """给一对主臂加力反馈。用 ``with`` 语句,退出时保证断力矩。

    ``pair`` 是 `teleop_sim.ArmPair`(有 ``.left`` / ``.right``,各自有 ``.bus``)。
    """

    def __init__(self, pair, gains: FeedbackGains | None = None,
                 delay_steps: int = DEFAULT_DELAY_STEPS) -> None:
        self.pair = pair
        self.gains = gains or FeedbackGains()
        self.delay_steps = delay_steps
        self._engaged = False
        #: 指令历史,用来和实测对齐(见模块说明里"不能接在主臂-从臂位置差上")。
        self._history: deque[dict[str, float]] = deque(maxlen=delay_steps + 1)
        #: 每个关节"实际出过力"的步数和最大误差。没有这个,跑完只能靠手感
        #: 判断力反馈起没起作用 —— 而从臂压根没被挡住过的那种情况(手臂没碰到
        #: 任何东西)和"功能坏了"在输出里长得一模一样。
        self.engaged_steps: dict[str, int] = {}
        self.max_error: dict[str, float] = {}
        self.steps = 0
        #: 每条主臂当前通电没有。自由运动时必须是 False,否则有黏滞阻力。
        self._powered: dict[str, bool] = {"left": False, "right": False}
        #: 通断次数。频繁通断说明死区取小了,主臂会咔咔响。
        self.transitions = 0

    # -- 生命周期 ------------------------------------------------------------

    def engage(self) -> dict[str, int]:
        """压低主臂的力矩上限,**但不通电**。返回各舵机读回的上限,供核对。

        这里只做准备:力矩要等到从臂真被挡住那一刻才开(见 ``update``)。
        一直通着电的话,自由运动时也会有黏滞阻力 —— ``Goal_Position`` 每周期
        才刷新一次,伺服总在往一个周期前的旧位置拽。
        """
        readback: dict[str, int] = {}
        limit = self.gains.torque_register()
        for side, dev in (("left", self.pair.left), ("right", self.pair.right)):
            bus = dev.bus
            for motor in bus.motors:
                bus.write("Torque_Limit", motor, limit, normalize=False)
                readback[f"{side}_{motor}"] = int(
                    bus.read("Torque_Limit", motor, normalize=False)
                )
        self._engaged = True
        return readback

    def release(self) -> None:
        """断主臂力矩,恢复成可以自由拖动。异常路径上也必须跑到。"""
        for side, dev in (("left", self.pair.left), ("right", self.pair.right)):
            try:
                dev.bus.disable_torque()
            except Exception:  # pragma: no cover - 断电/拔线时尽力而为
                pass
            self._powered[side] = False
        self._engaged = False

    def _set_powered(self, side: str, dev, on: bool) -> None:
        """给一条主臂通/断电。已经是目标状态就不重复写总线。

        通电前必须先把 ``Goal_Position`` 设成当前位置:舵机里存的还是上次断电
        前的旧目标,直接开力矩会朝那个旧位置弹一下。
        """
        if self._powered[side] == on:
            return
        if on:
            dev.bus.sync_write("Goal_Position", dev.bus.sync_read("Present_Position"))
            dev.bus.enable_torque()
        else:
            dev.bus.disable_torque()
        self._powered[side] = on
        self.transitions += 1

    def __enter__(self) -> LeaderForceFeedback:
        self.engage()
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    # -- 每周期调用 ----------------------------------------------------------

    def update(self, measured: dict[str, float], commanded: dict[str, float]) -> None:
        """按从臂的"实测 - 指令"更新主臂的目标。

        两个参数都是带 ``left_``/``right_`` 前缀的字典(相机图像等非 ``.pos``
        的键会被忽略)。``commanded`` 传**本周期**下发的那条即可,延迟对齐由
        内部队列负责。

        队列没攒满之前不出力:开头几步没有可对齐的历史指令,拿当前指令去比
        会把纯延迟整个算成"被挡住",主臂会猛拽一下。
        """
        if not self._engaged:
            raise RuntimeError("先 engage() 或用 with 语句")
        self._history.append({
            key: float(value) for key, value in commanded.items()
            if key.endswith(POS_SUFFIX)
        })
        self.steps += 1
        if len(self._history) <= self.delay_steps:
            return
        aligned = self._history[0]

        for side, dev in (("left", self.pair.left), ("right", self.pair.right)):
            prefix = f"{side}_"
            strip = lambda d: {  # noqa: E731
                k.removeprefix(prefix): v for k, v in d.items()
                if k.startswith(prefix) and k.endswith(POS_SUFFIX)
            }
            follower, target = strip(measured), strip(aligned)
            if not follower or not target:
                continue
            error = blocking_error(follower, target)
            worst = 0.0
            for key, err in error.items():
                name = f"{prefix}{key}"
                self.max_error[name] = max(self.max_error.get(name, 0.0), abs(err))
                worst = max(worst, abs(err))
                if abs(err) > self.gains.deadband:
                    self.engaged_steps[name] = self.engaged_steps.get(name, 0) + 1

            # 滞回:超过死区才通电,回落到一半以下才断开。只用一个阈值的话
            # 误差在边界抖动时会反复通断,主臂咔咔响。
            if worst > self.gains.deadband:
                self._set_powered(side, dev, True)
            elif worst < self.gains.deadband * RELEASE_RATIO:
                self._set_powered(side, dev, False)
            if not self._powered[side]:
                continue

            bus = dev.bus
            leader = {f"{m}{POS_SUFFIX}": v for m, v in bus.sync_read("Present_Position").items()}
            goals = goal_positions(leader, error, self.gains)
            if goals:
                bus.sync_write(
                    "Goal_Position",
                    {k.removesuffix(POS_SUFFIX): v for k, v in goals.items()},
                )

    def summary(self) -> list[str]:
        """跑完打印用:哪些关节真的出过力、误差多大。

        全是 0 说明从臂一次都没被挡住(手臂没碰到任何东西),不是功能坏了 ——
        这两种情况没有这张表就分不开。
        """
        if not self.max_error:
            return ["  力反馈:一次都没触发(队列没攒满或没有关节数据)"]
        lines = [f"  力反馈:{self.steps} 步,死区 {self.gains.deadband:g},"
                 f"通断 {self.transitions} 次(次数多说明死区取小了,主臂会咔咔响)",
                 f"  {'关节':<26}{'出力步数':>9}{'最大误差':>10}"]
        for name in sorted(self.max_error, key=lambda k: -self.max_error[k]):
            hits = self.engaged_steps.get(name, 0)
            lines.append(f"  {name:<26}{hits:>9}{self.max_error[name]:>10.1f}")
        return lines

#: 锁住主臂时的力矩上限,占满量程百分比。
#:
#: 30% 试过,**顶不住自重,主臂会塌下去**。主臂的舵机比从臂弱得多:按 Feetech
#: 的标称,Leader 的 shoulder_lift 是 C001(额定 0.490 / 堵转 1.912 N·m)、
#: wrist 三轴是 C046(0.471 / 1.412),而且 Leader 按 SO-101 设计跑 **5V**,
#: 实际出力还低于这些标称值。30% 之后只剩零点几 N·m,举不动自己。
#:
#: 现在取 80%。锁住时目标就是当前位姿,不会有运动,所以不像"跟随"那样有窜出去
#: 的风险;但**长时间保持高力矩会发热**,STS3215 堵转电流很大。冻结几十秒没事,
#: 挂几分钟不动要留意。人仍然能拧过去,只是要用点力。
LOCK_TORQUE_PERCENT = 80.0


class LeaderLock:
    """把一条主臂钉在当前位姿,让操作者推不动它。

    配合 `record.common.ArmFreeze` 用:从臂冻结时,对应的主臂也锁住。不锁的话
    操作者的手会在冻结期间把主臂带到别处,解冻那一刻两者差几十度 —— 从臂要么
    猛窜过去,要么(靠 ArmFreeze 的平滑过渡)慢慢滑过去,两种都不是操作者想要的。
    锁住之后两者始终一致,解冻就是无缝的。

    人仍然能拧过去(力矩上限只有 30%),所以 ArmFreeze 的平滑过渡要留着兜底。

    **安全**:主臂会通电。``unlock()`` 必须在任何退出路径上跑到 —— 用 ``with``,
    或者把它放进 finally。
    """

    def __init__(self, dev, torque_percent: float = LOCK_TORQUE_PERCENT) -> None:
        self.bus = getattr(dev, "bus", dev)
        if not 0.0 < torque_percent <= 100.0:
            raise ValueError(f"torque_percent 应在 (0, 100],给的是 {torque_percent}")
        self.torque_percent = torque_percent
        self.locked = False

    def lock(self) -> dict[str, int]:
        """钉在当前位姿。已经锁着就不重复写总线。

        返回各舵机**读回**的力矩上限。读回是刻意的:"写了但没生效"和"写了、
        生效了、可就是不够顶住自重"在现象上都是"主臂塌下去",光看代码分不出来。
        """
        if self.locked:
            return {}
        limit = int(round(self.torque_percent / 100.0 * TORQUE_LIMIT_FULL))
        readback: dict[str, int] = {}
        for motor in self.bus.motors:
            self.bus.write("Torque_Limit", motor, limit, normalize=False)
            try:
                readback[motor] = int(self.bus.read("Torque_Limit", motor, normalize=False))
            except Exception:  # pragma: no cover - 读不回来不该挡住加锁
                readback[motor] = -1
        # 先把目标设成当前位置再通电,否则舵机会朝上次断电前的旧目标弹一下
        self.bus.sync_write("Goal_Position", self.bus.sync_read("Present_Position"))
        self.bus.enable_torque()
        self.locked = True
        bad = {m: v for m, v in readback.items() if v != limit}
        if bad:
            print(f"[lock] ** 力矩上限没写进去 ** 想写 {limit},读回 {bad}", flush=True)
        return readback

    def unlock(self) -> None:
        """断电,恢复自由拖动。异常路径上也必须跑到。"""
        if not self.locked:
            return
        try:
            self.bus.disable_torque()
        except Exception:  # pragma: no cover - 拔线/断电时尽力而为
            pass
        self.locked = False

    def __enter__(self) -> "LeaderLock":
        return self

    def __exit__(self, *exc) -> None:
        self.unlock()
