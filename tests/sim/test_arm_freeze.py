"""冻结一条臂:单人采双臂数据用。

盯的是两件容易写错的事:解冻要平滑交回(不然从臂猛窜),以及冻结期间
另一条臂必须完全不受影响。
"""

from __future__ import annotations

import pytest

from evo_rlt.adapters.lerobot.record.common import ARM_FREEZE_KEY, ArmFreeze


def _act(left: float, right: float) -> dict[str, float]:
    return {"left_shoulder_pan.pos": left, "right_shoulder_pan.pos": right}


def test_follow_passes_through_untouched():
    f = ArmFreeze("right")
    a = _act(1.0, 2.0)
    assert f.apply(a) is a          # 不复制,遥操回路每帧都走这里
    assert not f.frozen


def test_freeze_holds_the_pose_the_key_was_pressed_at():
    f = ArmFreeze("right", blend_s=0.1, fps=30)
    f.toggle(_act(1.0, 10.0))
    assert f.frozen
    # 主臂继续走,冻结那条臂不动
    out = f.apply(_act(5.0, 99.0))
    assert out["right_shoulder_pan.pos"] == 10.0


def test_the_other_arm_keeps_following_while_frozen():
    """冻结右臂不能影响左臂 —— 左臂正是操作者此刻专心在用的那条。"""
    f = ArmFreeze("right", blend_s=0.1, fps=30)
    f.toggle(_act(1.0, 10.0))
    out = f.apply(_act(42.0, 99.0))
    assert out["left_shoulder_pan.pos"] == 42.0


def test_unfreeze_blends_instead_of_jumping():
    """**这条是安全项。**

    冻结期间操作者的手一直在动,解冻那一刻主臂可能已经和从臂差几十度。
    直接交回去从臂会猛窜一下 —— 既危险,也在数据里留一个非物理的跳变。
    """
    f = ArmFreeze("right", blend_s=0.1, fps=30)      # 3 步
    f.toggle(_act(0.0, 10.0))
    f.toggle(_act(0.0, 100.0))                        # 解冻
    seen = [f.apply(_act(0.0, 100.0))["right_shoulder_pan.pos"] for _ in range(3)]
    assert seen[0] == pytest.approx(10.0)             # 从冻结位姿起步
    assert seen == sorted(seen)                       # 单调爬向主臂
    assert seen[-1] < 100.0                           # 中途还没到
    assert f.state == ArmFreeze.FOLLOW                # 三步后交回
    assert f.apply(_act(0.0, 100.0))["right_shoulder_pan.pos"] == 100.0


def test_toggling_during_blend_refreezes_here():
    """过渡途中再按一次,应当就地重新冻结,而不是继续滑回去。"""
    f = ArmFreeze("right", blend_s=0.2, fps=30)
    f.toggle(_act(0.0, 10.0))
    f.toggle(_act(0.0, 100.0))
    f.apply(_act(0.0, 100.0))
    f.toggle(_act(0.0, 55.0))
    assert f.frozen
    assert f.apply(_act(0.0, 999.0))["right_shoulder_pan.pos"] == 55.0


def test_summary_separates_unused_from_used():
    """全 0 说明这个功能没被用上,和"用了但没效果"要分得开。"""
    f = ArmFreeze("right")
    assert "没用过" in f.summary()
    f.toggle(_act(0.0, 1.0)); f.apply(_act(0.0, 2.0))
    assert "切换 1 次" in f.summary() and "冻结 1 帧" in f.summary()


def test_freeze_key_does_not_collide_with_the_reset_key():
    """两个键都在遥操回路里读,撞了就会一按同时触发两件事。"""
    import diagnostics.teleop_sim as ts  # noqa: PLC0415

    assert ARM_FREEZE_KEY != ts.RESET_KEY


class _FakeBus:
    def __init__(self):
        self.motors = ["shoulder_pan", "gripper"]
        self.powered = False
        self.writes: list[tuple] = []
        self.regs: dict[tuple, int] = {}

    def write(self, name, motor, value, **kw):
        self.writes.append((name, motor, value))
        self.regs[(name, motor)] = value

    def read(self, name, motor, **kw):
        return self.regs[(name, motor)]

    def sync_read(self, _name):
        return {m: 0.0 for m in self.motors}

    def sync_write(self, _name, _values, **kw):
        pass

    def enable_torque(self, *a, **kw):
        self.powered = True

    def disable_torque(self, *a, **kw):
        self.powered = False


class _FakeDev:
    def __init__(self):
        self.bus = _FakeBus()


def test_freezing_also_locks_the_leader():
    """**这条是这个功能的要点。**

    不锁主臂的话,操作者的手会在冻结期间把主臂带到别处,解冻那一刻主从差
    几十度 —— 要么从臂猛窜,要么靠平滑过渡慢慢滑过去,两种都不是想要的。
    """
    from evo_rlt.sim.feedback import LeaderLock

    dev = _FakeDev()
    f = ArmFreeze("right", leader_lock=LeaderLock(dev))
    assert not dev.bus.powered
    f.toggle(_act(0.0, 10.0))
    assert dev.bus.powered, "冻结时主臂没锁住"
    f.toggle(_act(0.0, 10.0))
    assert not dev.bus.powered, "解冻时主臂没断电"


def test_lock_writes_the_configured_torque_limit_and_reads_it_back():
    """读回是刻意的:"写了没生效"和"生效了但不够顶住自重"现象一样,都是主臂塌。

    30% 试过顶不住(主臂舵机弱,而且 Leader 跑 5V),所以默认调到了 80%。
    不再断言"人能轻松拧过去" —— 锁住时目标就是当前位姿、不会有运动,
    真正的约束是别长时间高力矩发热,那个卡不到测试里。
    """
    from evo_rlt.sim.feedback import LOCK_TORQUE_PERCENT, TORQUE_LIMIT_FULL, LeaderLock

    dev = _FakeDev()
    want = int(round(LOCK_TORQUE_PERCENT / 100 * TORQUE_LIMIT_FULL))
    readback = LeaderLock(dev).lock()
    assert {v for n, _m, v in dev.bus.writes if n == "Torque_Limit"} == {want}
    assert set(readback.values()) == {want}
    assert LOCK_TORQUE_PERCENT >= 50.0, "30% 实测顶不住自重"


def test_release_cuts_leader_power_even_if_never_frozen():
    from evo_rlt.sim.feedback import LeaderLock

    dev = _FakeDev()
    ArmFreeze("right", leader_lock=LeaderLock(dev)).release()
    assert not dev.bus.powered


def test_works_without_a_leader_lock():
    """没有可锁的主臂时(比如 teleop 不是 BiSOLeader)也要能用。"""
    f = ArmFreeze("right", blend_s=0.1, fps=30)
    f.toggle(_act(0.0, 10.0))
    assert f.apply(_act(0.0, 99.0))["right_shoulder_pan.pos"] == 10.0
