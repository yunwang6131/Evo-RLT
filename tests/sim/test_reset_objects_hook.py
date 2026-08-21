"""采集的 reset 阶段要自动把零件摆回去。

盯的是"每条 episode 的初始位姿都一样"这个数据质量问题:遥操时靠人按 b,
采集时没人按 —— 80 条 episode 会有 79 条从上一条的终态开始。
"""

from __future__ import annotations


class _SimRobot:
    """有 reset_objects 的机器人(仿真)。"""

    name = "sim_bi_so_follower"

    def __init__(self, fail: bool = False):
        self.calls = 0
        self.fail = fail

    def reset_objects(self):
        self.calls += 1
        if self.fail:
            raise RuntimeError("simulator went away")
        return ["socket", "bolt"]


class _RealRobot:
    """真机没有这个方法,不能因此报错。"""

    name = "bi_so_follower"


def _reset_hook(robot, log):
    """复刻 backend._run_reset_loop_if_needed 里那段。

    这里复制而不是 import:那段在一个深层闭包里,拿不到。测试因此只能守住
    行为契约,改了 backend 要同步改这里 —— 下面那条源码检查负责提醒。
    """
    fn = getattr(robot, "reset_objects", None)
    if callable(fn):
        try:
            log.append(fn())
        except Exception as exc:  # noqa: BLE001
            log.append(f"failed: {exc}")


def test_sim_robot_gets_its_objects_reset():
    robot, log = _SimRobot(), []
    _reset_hook(robot, log)
    assert robot.calls == 1
    assert log == [["socket", "bolt"]]


def test_real_robot_is_untouched():
    robot, log = _RealRobot(), []
    _reset_hook(robot, log)          # 不能抛
    assert log == []


def test_reset_failure_does_not_abort_recording():
    """复位失败最多让这条 episode 的起点和上一条一样,不该中断采集。"""
    robot, log = _SimRobot(fail=True), []
    _reset_hook(robot, log)
    assert log and log[0].startswith("failed:")


def test_backend_still_calls_reset_objects_in_the_reset_phase():
    """钉住 backend 里那段没被删。上面的测试是复刻的,防不住原件被改。"""
    import inspect

    from evo_rlt.adapters.lerobot.record import backend

    src = inspect.getsource(backend)
    assert "reset_objects" in src, "backend 不再在 reset 阶段复位零件"
    # 必须在 reset 那一段里,不是随便哪儿
    reset_block = src[src.index("Reset the environment"):]
    assert "reset_objects" in reset_block[:2000]


def test_go_home_is_disabled_by_default():
    """项目规则:任何时候都不复位手臂。自动归位默认必须关着。

    归位在遥操里是有害的:从臂被拉回 home,而操作者手上的主臂不动 —— 一旦错开,
    下一帧指令就让从臂猛窜回主臂那边。
    """
    from evo_rlt.adapters.lerobot.record.backend import OnlineRLConfig

    assert OnlineRLConfig().go_home_time_s == 0.0


def test_reset_phase_never_resets_the_arm_for_our_robots():
    """reset 阶段只许调 reset_objects;robot.reset() 只留给 unitree_g1。"""
    import inspect

    from evo_rlt.adapters.lerobot.record import backend

    src = inspect.getsource(backend)
    block = src[src.index("Reset the environment"):]
    block = block[:block.index("record_loop(")]
    # 注释里也写着 robot.reset()(解释为什么不能用),按字面找会先命中它。
    # 只看真正的代码行。
    code = "\n".join(
        line for line in block.splitlines() if not line.strip().startswith("#")
    )
    if "robot.reset()" in code:
        guard = code[:code.index("robot.reset()")]
        assert "unitree_g1" in guard, "reset 阶段有不受 unitree_g1 保护的整臂复位"
    assert "reset_objects" in code, "reset 阶段没有复位零件"
