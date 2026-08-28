"""演示增广的纯计算部分(不需要 MuJoCo,也不连仿真器)。

这里守的是几条"错了不会报错、只会悄悄产出坏数据"的规则:分段挑错帧、标定的
最小二乘退化、位移斜坡压在抓取瞬间上、对不准量的方向反了。每一条都在开发
过程中真实踩到过,注释里记着当时的症状。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from evo_rlt.sim import augment as A
from evo_rlt.sim.task_success import TaskState


# -- 分段 -------------------------------------------------------------------


def _gripper_signal(length, close_at, open_at, high=40.0, low=12.0, lead=None):
    signal = np.full(length, high)
    signal[close_at:open_at] = low
    if lead is not None:
        signal[:lead] = 0.0
    return signal


def test_find_closed_span_locates_the_grasp():
    span = A.find_closed_span(_gripper_signal(600, 180, 520))
    assert span is not None
    assert abs(span[0] - 180) <= 2 and abs(span[1] - 520) <= 2


def test_find_closed_span_ignores_the_uninitialised_lead():
    """开头几帧的 0 是遥操还没接上的残值,不是"夹爪闭合"。

    第一版按"首次跌破阈值"找,这些 episode 的抓取帧全部被判在第 0 帧,推出的
    螺套位置离凹槽中心 106mm —— 数值上毫无异常,只是全错。
    """
    span = A.find_closed_span(_gripper_signal(600, 180, 520, lead=3))
    assert span is not None
    assert abs(span[0] - 180) <= 2


def test_find_closed_span_prefers_the_longest_run():
    """人中途会有试探性的半闭合,真正抓着零件的一定是最长那段。"""
    signal = np.full(600, 40.0)
    signal[100:120] = 12.0      # 试探
    signal[300:560] = 12.0      # 真抓
    span = A.find_closed_span(signal)
    assert span is not None
    assert abs(span[0] - 300) <= 2


def test_find_closed_span_returns_none_when_never_closed():
    assert A.find_closed_span(np.full(300, 40.0)) is None


def test_segment_episode_reads_both_arms():
    actions = np.zeros((600, 12))
    actions[:, A.GRIPPER_INDEX["right"]] = _gripper_signal(600, 100, 560)
    actions[:, A.GRIPPER_INDEX["left"]] = _gripper_signal(600, 300, 480)
    segments = A.segment_episode(actions)
    assert abs(segments.right_close - 100) <= 2
    assert abs(segments.left_close - 300) <= 2
    assert segments.length == 600


def test_segment_episode_rejects_an_unsegmentable_episode():
    actions = np.zeros((300, 12))
    actions[:, A.GRIPPER_INDEX["right"]] = 40.0
    actions[:, A.GRIPPER_INDEX["left"]] = 40.0
    with pytest.raises(A.AugmentError):
        A.segment_episode(actions)


# -- 标定 -------------------------------------------------------------------


def _synthetic_grasps(translation, count=5000, radius=0.025, center=(0.35, -0.0623), rest_z=0.1145):
    """造一批"已知答案"的抓取:螺套在圆盘内均匀,夹爪只差一个绕 z 的偏航。"""
    rng = np.random.default_rng(0)
    positions, rotations = [], []
    for _ in range(count):
        r = radius * math.sqrt(rng.random())
        a = rng.uniform(0, 2 * math.pi)
        socket = np.array([center[0] + r * math.cos(a), center[1] + r * math.sin(a), rest_z])
        yaw = rng.uniform(-math.pi, math.pi)
        rot = np.array(
            [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1.0]]
        )
        positions.append(socket - rot @ translation)
        rotations.append(rot)
    return np.array(positions), np.array(rotations)


def test_fit_grasp_calibration_recovers_a_known_offset():
    """估计量是无偏的,但只在期望意义上 —— 容差按有限样本误差给。

    目标函数是"让推出的螺套位置整体最贴近圆盘中心",而真实螺套在圆盘内均匀
    分布,所以每轴的标准误差约 ``R / (2√N)``:25mm 圆盘、5000 条样本下是
    0.18mm。收得比这更紧就是在测随机种子,不是在测算法。
    """
    truth = np.array([0.011, -0.014, -0.090])
    positions, rotations = _synthetic_grasps(truth)
    fitted = A.fit_grasp_calibration(positions, rotations, [0.35, -0.0623], 0.025, 0.1145)
    assert np.allclose(fitted.translation, truth, atol=5e-4)


def test_fit_grasp_calibration_residual_looks_like_the_disk():
    """自检指标:拟合完的残差分布该和均匀圆盘的理论分位数对得上。

    这是判断"抓取帧找对了没有"的唯一手段 —— 源数据里没有螺套位姿的真值。
    """
    positions, rotations = _synthetic_grasps(np.array([0.011, -0.014, -0.090]))
    fitted = A.fit_grasp_calibration(positions, rotations, [0.35, -0.0623], 0.025, 0.1145)
    theory = A.disk_quantiles(0.025)
    for percentile, key in ((50, "q50"), (75, "q75")):
        assert np.percentile(fitted.residual_radius, percentile) == pytest.approx(
            theory[key], abs=0.004
        )


def test_fit_grasp_calibration_rejects_bad_shapes():
    with pytest.raises(A.AugmentError):
        A.fit_grasp_calibration(np.zeros((4, 2)), np.zeros((4, 3, 3)), [0, 0], 0.025, 0.1)


def test_calibration_round_trips_through_json():
    positions, rotations = _synthetic_grasps(np.array([0.011, -0.014, -0.090]))
    fitted = A.fit_grasp_calibration(positions, rotations, [0.35, -0.0623], 0.025, 0.1145)
    again = A.GraspCalibration.from_dict(fitted.to_dict())
    assert np.allclose(again.translation, fitted.translation)
    assert again.disk_radius == fitted.disk_radius


# -- 姿态工具 ---------------------------------------------------------------


def test_rotation_between_is_a_proper_rotation():
    rng = np.random.default_rng(3)
    for _ in range(50):
        a, b = rng.normal(size=3), rng.normal(size=3)
        rot = A.rotation_between(a, b)
        assert np.allclose(rot @ (a / np.linalg.norm(a)), b / np.linalg.norm(b), atol=1e-9)
        assert np.allclose(rot.T @ rot, np.eye(3), atol=1e-9)
        assert np.linalg.det(rot) == pytest.approx(1.0)


def test_rotation_between_handles_the_antiparallel_case():
    rot = A.rotation_between([0, 0, 1], [0, 0, -1])
    assert np.allclose(rot @ np.array([0, 0, 1.0]), [0, 0, -1], atol=1e-9)
    assert np.linalg.det(rot) == pytest.approx(1.0)


def test_socket_up_axis_is_the_world_z_seen_from_the_gripper():
    _, rotations = _synthetic_grasps(np.array([0.0, 0.0, -0.09]))
    up, spread = A.socket_up_axis_in_gripper(rotations)
    # 造的数据里夹爪只绕 z 转,所以 R^T e_z 恒为 e_z,离散度为 0
    assert np.allclose(up, [0, 0, 1.0], atol=1e-9)
    assert spread.max() < 1e-6


def test_socket_pose_from_grasp_recovers_the_placement():
    truth = np.array([0.011, -0.014, -0.090])
    positions, rotations = _synthetic_grasps(truth)
    fitted = A.fit_grasp_calibration(positions, rotations, [0.35, -0.0623], 0.025, 0.1145)
    up, _ = A.socket_up_axis_in_gripper(rotations)
    pose = A.socket_pose_from_grasp(positions[0], rotations[0], fitted, up, 0.0)
    assert len(pose) == 7
    # 用拟合值而不是真值来对:这条测的是"由标定推位姿"这一步,标定本身的
    # 有限样本误差由上一条测试负责。
    expected = positions[0] + rotations[0] @ fitted.translation
    assert pose[0] == pytest.approx(expected[0], abs=1e-5)
    assert pose[1] == pytest.approx(expected[1], abs=1e-5)
    # z 被强制取凹槽底高度:螺套是立在那儿的,让它浮起来复位时会被物理弹开
    assert pose[2] == pytest.approx(0.1145)
    assert np.linalg.norm(pose[3:]) == pytest.approx(1.0)


# -- 位移排程 ---------------------------------------------------------------


def test_schedule_reaches_full_offset_before_the_grasp():
    """最后一段接近必须是源轨迹的干净平移,否则夹爪是斜着扑上去的。"""
    segments = A.Segments(right_close=100, right_open=600, left_close=280, left_open=500, length=700)
    weights = A.displacement_schedule(segments, lift_frame=330)
    assert weights["right"][segments.right_close] == pytest.approx(1.0)
    assert weights["right"][0] == pytest.approx(0.0)


def test_left_arm_holds_still_until_the_bolt_is_clear():
    """螺栓插在台面孔里,拔出来之前给左臂加平移会把它往侧面掰。"""
    segments = A.Segments(right_close=100, right_open=600, left_close=280, left_open=500, length=700)
    weights = A.displacement_schedule(segments, lift_frame=330)
    assert np.all(weights["left"][: segments.left_close + 1] == 0.0)
    assert weights["left"][330] == pytest.approx(0.0)
    assert weights["left"][-1] == pytest.approx(1.0)


def test_hold_ramp_starts_after_the_jaws_have_closed():
    """握持修正若在钳口合拢期间起变化,会把零件从钳口里蹭出去。"""
    segments = A.Segments(right_close=100, right_open=600, left_close=280, left_open=500, length=700)
    weights = A.displacement_schedule(segments, lift_frame=330)
    assert np.all(weights["hold"][: segments.right_close + 5] == 0.0)
    assert weights["hold"][-1] == pytest.approx(1.0)


def test_smoothstep_has_zero_slope_at_both_ends():
    """线性斜坡在起止两点有速度突变,30 Hz 下是可见的顿挫,而这些帧要当演示用。"""
    assert A.smoothstep(0.0) == pytest.approx(0.0)
    assert A.smoothstep(1.0) == pytest.approx(1.0)
    assert A.smoothstep(0.5) == pytest.approx(0.5)
    step = 1e-4
    assert A.smoothstep(step) < step / 10
    assert 1.0 - A.smoothstep(1.0 - step) < step / 10


def test_offset_ee_targets_moves_position_and_keeps_orientation():
    poses = [[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]] * 3
    offsets = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.02, 0.0]])
    out = A.offset_ee_targets(poses, offsets)
    assert out[0][:3] == pytest.approx([0.1, 0.2, 0.3])
    assert out[1][:3] == pytest.approx([0.11, 0.2, 0.3])
    assert out[2][:3] == pytest.approx([0.1, 0.22, 0.3])
    for pose in out:
        assert pose[3:] == pytest.approx([1.0, 0.0, 0.0, 0.0])


# -- 对不准量的测量与修正 ---------------------------------------------------


def _state(depth, lateral, angle, offset=(0.0, 0.0, 0.0), held=True):
    return TaskState(
        stage="aligned", inserted=False, aligned=True,
        socket_lifted=held, bolt_pulled=held,
        depth=depth, lateral=lateral, angle_deg=angle, lateral_offset=offset,
    )


def test_closest_approach_picks_the_best_aligned_frame():
    states = [_state(-0.01, 0.004, 5.0), _state(-0.002, 0.001, 3.0), _state(0.004, 0.003, 4.0)]
    assert A.closest_approach(states) == 1


def test_closest_approach_skips_a_bolt_lying_across_the_mouth():
    """实测见过夹角 89 度却"很近"的帧 —— 那是螺栓横躺着从孔口边上过。

    拿它去修正会把轨迹带得更偏(0.70mm 修成 2.06mm),所以夹角必须是一道硬闸。
    """
    states = [_state(-0.001, 0.0002, 89.0), _state(-0.004, 0.003, 6.0)]
    assert A.closest_approach(states) == 1


def test_closest_approach_ignores_frames_before_both_parts_are_held():
    states = [_state(0.0, 0.0001, 1.0, held=False), _state(-0.004, 0.003, 6.0)]
    assert A.closest_approach(states) == 1


def test_closest_approach_returns_none_when_it_never_gets_there():
    assert A.closest_approach([_state(0.5, 0.4, 80.0)]) is None


def test_socket_pose_correction_inverts_the_carry_transform():
    """零件被夹爪刚性带着走,所以修正量 = 对不准量经 ``A = R(插)·R(抓)ᵀ`` 的逆像。"""
    yaw = math.radians(35.0)
    grasp = np.eye(3)
    insert = np.array(
        [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1.0]]
    )
    miss = np.array([0.003, -0.001, 0.0])
    # 第 0 帧刻意不可测量(夹角过大),这样最接近帧是第 1 帧,抓取帧是第 0 帧 ——
    # 两者的旋转必须不同,否则变换退化成单位阵,测不出方向对不对。
    states = [
        _state(0.0, 0.02, 80.0),
        _state(0.0, float(np.linalg.norm(miss)), 2.0, tuple(miss)),
    ]
    correction = A.socket_pose_correction(states, [grasp, insert], grasp_frame=0)
    assert correction is not None
    expected = (insert @ grasp.T).T @ miss
    assert correction[:2] == pytest.approx(expected[:2], abs=1e-12)
    # z 不修:螺套立在凹槽底上,高度由几何定死
    assert correction[2] == pytest.approx(0.0)


def test_socket_pose_correction_needs_a_measurable_frame():
    assert A.socket_pose_correction([_state(0.5, 0.4, 80.0)], [np.eye(3)], 0) is None


def test_socket_pose_correction_is_identity_when_the_arm_has_not_turned():
    """抓和插时手腕朝向一样,修正量就等于量到的对不准量本身。"""
    miss = np.array([0.002, 0.001, 0.0])
    states = [
        _state(0.0, 0.02, 80.0),
        _state(0.0, float(np.linalg.norm(miss)), 2.0, tuple(miss)),
    ]
    correction = A.socket_pose_correction(states, [np.eye(3), np.eye(3)], grasp_frame=0)
    assert correction[:2] == pytest.approx(miss[:2], abs=1e-12)


# -- 目标位姿采样 -----------------------------------------------------------


def test_sample_delta_lands_inside_the_disk_and_respects_the_cap():
    positions, rotations = _synthetic_grasps(np.array([0.0, 0.0, -0.09]))
    calibration = A.fit_grasp_calibration(positions, rotations, [0.35, -0.0623], 0.025, 0.1145)
    rng = np.random.default_rng(1)
    source = np.array([0.35, -0.0623])
    for _ in range(200):
        delta = A.sample_delta(rng, calibration, source, max_delta=0.02)
        assert np.linalg.norm(delta) <= 0.02 + 1e-9
        assert delta[2] == 0.0
        target = source + delta[:2]
        assert np.linalg.norm(target - calibration.disk_center) <= 0.025 + 1e-9
