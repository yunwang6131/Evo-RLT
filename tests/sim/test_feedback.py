"""力反馈的纯计算部分。硬件那半(通电、写寄存器)在这里测不了。"""

from __future__ import annotations

import pytest

from evo_rlt.sim.feedback import (
    TORQUE_LIMIT_FULL,
    FeedbackGains,
    blocking_error,
    goal_positions,
)


def test_tracking_lag_alone_produces_no_force():
    """**这条盯的是第一版的 bug。**

    第一版把阻力接在"主臂位置 - 从臂位置"上。正常跟随时这两者本来就差一个
    纯延迟(实测 3 步 = 100ms):以 86°/s 挥臂时差 8.6°,远超死区,于是全程
    都有很大阻力。改成"实测 - 指令"后,延迟在两边同时出现,对齐后自己抵消。

    这里模拟自由运动:从臂精确执行了 3 步之前的指令,没有任何阻挡。
    """
    gains = FeedbackGains(gain=0.3, deadband=2.0)
    commanded_then = {"elbow_flex.pos": 20.0}       # 3 步之前下发的
    measured_now = {"elbow_flex.pos": 20.0}         # 从臂原样到位了
    error = blocking_error(measured_now, commanded_then)
    assert error == {"elbow_flex.pos": 0.0}
    # 此刻主臂已经走到 28.6(那 8.6° 就是纯延迟造成的领先),仍然不该出力
    leader = {"elbow_flex.pos": 28.6}
    assert goal_positions(leader, error, gains) == {"elbow_flex.pos": 28.6}


def test_blocked_follower_produces_force():
    """指令继续往前、实测停住 —— 这才是"被挡住",必须出力。"""
    gains = FeedbackGains(gain=0.3, deadband=2.0)
    error = blocking_error({"gripper.pos": 30.0}, {"gripper.pos": 10.0})
    assert error == {"gripper.pos": 20.0}
    leader = {"gripper.pos": 5.0}
    goal = goal_positions(leader, error, gains)["gripper.pos"]
    assert goal == pytest.approx(5.0 + 0.3 * (20.0 - 2.0))


def test_deadband_holds_position():
    """死区内目标必须等于主臂**当前**位置 —— 位置环误差为零才不出力。"""
    gains = FeedbackGains(gain=0.5, deadband=2.0)
    leader = {"elbow_flex.pos": 10.0}
    error = {"elbow_flex.pos": 1.5}              # 在死区内
    assert goal_positions(leader, error, gains) == {"elbow_flex.pos": 10.0}


def test_force_grows_from_zero_at_the_deadband_edge():
    """跨过死区边界时力不能跳变,否则手上是"咔"的一下。

    做法是先减掉死区宽度再乘增益,所以恰好在边界上偏移量为 0。
    """
    gains = FeedbackGains(gain=0.5, deadband=2.0)
    leader = {"gripper.pos": 0.0}
    at_edge = goal_positions(leader, {"gripper.pos": 2.0}, gains)["gripper.pos"]
    assert at_edge == pytest.approx(0.0)
    just_past = goal_positions(leader, {"gripper.pos": 2.2}, gains)["gripper.pos"]
    assert just_past == pytest.approx(0.5 * 0.2)


def test_direction_follows_the_sign_of_the_block():
    """从臂卡在指令后面和卡在前面,主臂被推的方向必须相反。"""
    gains = FeedbackGains(gain=0.5, deadband=1.0)
    leader = {"shoulder_lift.pos": 30.0}
    behind = goal_positions(leader, {"shoulder_lift.pos": -10.0}, gains)["shoulder_lift.pos"]
    assert behind < 30.0
    ahead = goal_positions(leader, {"shoulder_lift.pos": 10.0}, gains)["shoulder_lift.pos"]
    assert ahead > 30.0


def test_gain_scales_the_pull():
    """增益越小,主臂被拉走得越少 —— 这是"软提示 vs 硬墙"那个旋钮。"""
    leader, error = {"a.pos": 0.0}, {"a.pos": 10.0}
    soft = goal_positions(leader, error, FeedbackGains(gain=0.2, deadband=0.0))["a.pos"]
    hard = goal_positions(leader, error, FeedbackGains(gain=0.9, deadband=0.0))["a.pos"]
    assert soft < hard < 10.0


def test_missing_and_non_position_keys_are_skipped():
    """observation 里混着相机图像等非 .pos 的键,不能当关节处理。"""
    error = blocking_error({"a.pos": 50.0, "front": 0.0}, {"a.pos": 0.0})
    assert set(error) == {"a.pos"}
    out = goal_positions({"a.pos": 0.0, "b.pos": 0.0}, error, FeedbackGains())
    assert set(out) == {"a.pos"}          # b 没有对应误差,跳过


def test_torque_register_is_per_mille():
    """Feetech 的 Torque_Limit 是千分比,1000 = 满量程。"""
    assert FeedbackGains(torque_percent=100.0).torque_register() == TORQUE_LIMIT_FULL
    assert FeedbackGains(torque_percent=15.0).torque_register() == 150
    assert FeedbackGains(torque_percent=0.0).torque_register() == 0


@pytest.mark.parametrize("bad", [-1.0, 100.1])
def test_torque_percent_out_of_range_is_rejected(bad):
    """力矩上限写错量纲(比如填了寄存器值 150)会让主臂满力矩顶人手。"""
    with pytest.raises(ValueError):
        FeedbackGains(torque_percent=bad).torque_register()


def test_defaults_are_conservative():
    """默认值必须是"宁可太弱":增益远小于 1,力矩上限远小于满量程。"""
    gains = FeedbackGains()
    assert 0.0 < gains.gain < 0.5
    assert gains.deadband > 0.0
    assert 0.0 < gains.torque_percent <= 25.0


class _FakeBus:
    """够用的假总线:只记下写了什么和通断电,不碰串口。"""

    def __init__(self, motors, position):
        self.motors = list(motors)
        self.position = dict(position)
        self.written: list[dict] = []
        self.powered = False
        self.power_log: list[bool] = []

    def enable_torque(self, *_a, **_kw):
        self.powered = True
        self.power_log.append(True)

    def disable_torque(self, *_a, **_kw):
        self.powered = False
        self.power_log.append(False)

    def sync_read(self, _name):
        return dict(self.position)

    def sync_write(self, _name, values, **_kw):
        self.written.append(dict(values))


class _FakeDev:
    def __init__(self, bus):
        self.bus = bus


class _FakePair:
    def __init__(self, left, right):
        self.left, self.right = left, right


def _pair():
    return _FakePair(_FakeDev(_FakeBus(["gripper"], {"gripper": 5.0})),
                     _FakeDev(_FakeBus(["gripper"], {"gripper": 5.0})))


def test_no_force_until_the_delay_queue_fills():
    """开头几步没有可对齐的历史指令。

    拿当前指令去比会把整个纯延迟算成"被挡住",主臂在启用那一刻猛拽一下。
    """
    from evo_rlt.sim.feedback import LeaderForceFeedback

    pair = _pair()
    fb = LeaderForceFeedback(pair, FeedbackGains(deadband=1.0), delay_steps=3)
    fb._engaged = True
    for step in range(3):
        fb.update({"left_gripper.pos": 0.0, "right_gripper.pos": 0.0},
                  {"left_gripper.pos": 50.0, "right_gripper.pos": 50.0})
        assert pair.left.bus.written == [], f"第 {step} 步就出力了"
    fb.update({"left_gripper.pos": 0.0, "right_gripper.pos": 0.0},
              {"left_gripper.pos": 50.0, "right_gripper.pos": 50.0})
    assert pair.left.bus.written, "队列攒满后应该开始出力"


def test_summary_separates_never_blocked_from_broken():
    """全 0 的表说明"从臂没被挡过",和"功能坏了"必须能分开。"""
    from evo_rlt.sim.feedback import LeaderForceFeedback

    pair = _pair()
    fb = LeaderForceFeedback(pair, FeedbackGains(deadband=1.0), delay_steps=0)
    fb._engaged = True
    for _ in range(5):
        fb.update({"left_gripper.pos": 10.0, "right_gripper.pos": 10.0},
                  {"left_gripper.pos": 10.0, "right_gripper.pos": 10.0})
    text = "\n".join(fb.summary())
    assert "left_gripper.pos" in text
    assert fb.engaged_steps == {}          # 一次都没超过死区
    assert fb.max_error["left_gripper.pos"] == 0.0


def _power_pair():
    return _pair()


def _feed(fb, measured, commanded, n):
    for _ in range(n):
        fb.update({"left_gripper.pos": measured, "right_gripper.pos": measured},
                  {"left_gripper.pos": commanded, "right_gripper.pos": commanded})


def test_free_motion_leaves_the_leader_unpowered():
    """**这条盯的是第二版的 bug。**

    第二版一启用就通电。通电的位置伺服本身有黏滞阻力:Goal_Position 每 33ms
    才刷新,操作者一动伺服就往一个周期前的旧位置拽 —— 全程"十分费力",
    和有没有被挡住无关。自由运动时必须真的断电。
    """
    from evo_rlt.sim.feedback import LeaderForceFeedback

    pair = _power_pair()
    fb = LeaderForceFeedback(pair, FeedbackGains(deadband=5.0), delay_steps=0)
    fb._engaged = True
    _feed(fb, measured=10.0, commanded=10.0, n=10)      # 从臂完全跟上,没被挡
    assert pair.left.bus.powered is False
    assert pair.left.bus.written == []


def test_blocked_follower_powers_the_leader():
    from evo_rlt.sim.feedback import LeaderForceFeedback

    pair = _power_pair()
    fb = LeaderForceFeedback(pair, FeedbackGains(deadband=5.0), delay_steps=0)
    fb._engaged = True
    _feed(fb, measured=10.0, commanded=40.0, n=3)       # 指令跑了,实测停住
    assert pair.left.bus.powered is True
    assert pair.left.bus.written


def test_hysteresis_stops_power_chatter():
    """误差在死区边界附近抖动时不能反复通断,否则主臂咔咔响。"""
    from evo_rlt.sim.feedback import LeaderForceFeedback, RELEASE_RATIO

    pair = _power_pair()
    fb = LeaderForceFeedback(pair, FeedbackGains(deadband=10.0), delay_steps=0)
    fb._engaged = True
    _feed(fb, measured=0.0, commanded=12.0, n=1)        # 超死区 -> 通电
    assert pair.left.bus.powered is True
    # 回落到死区之下、但还在松开阈值之上:保持通电,不许抖
    _feed(fb, measured=0.0, commanded=8.0, n=5)
    assert pair.left.bus.powered is True
    assert pair.left.bus.power_log == [True]
    # 掉到松开阈值以下才断
    _feed(fb, measured=0.0, commanded=10.0 * RELEASE_RATIO - 1.0, n=1)
    assert pair.left.bus.powered is False
    assert fb.transitions == 4        # 左右各通断一次


def test_release_cuts_power_even_if_never_engaged():
    from evo_rlt.sim.feedback import LeaderForceFeedback

    pair = _power_pair()
    fb = LeaderForceFeedback(pair)
    fb.release()
    assert pair.left.bus.powered is False and pair.right.bus.powered is False


def test_deadband_default_clears_the_measured_tracking_residual():
    """死区必须大于正常跟随的残余误差,否则自由运动被误判成被挡住。

    真机 6258 帧实测:对齐 3 步后 p95=2.33、p99=4.79。
    """
    assert FeedbackGains().deadband >= 4.79
