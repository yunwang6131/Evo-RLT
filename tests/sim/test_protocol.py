"""Guard the constants that are duplicated across the process boundary.

`protocol.py` restates the joint names instead of importing them from
`calib.py`, because the simulator process cannot import `evo_rlt` (its
``__init__`` pulls in torch). That duplication is only safe if something checks
it, and a drift here would misroute joint targets between arms -- silently.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import pytest

from evo_rlt.sim import assets, calib, protocol


def test_joint_names_match_calibration_module():
    assert protocol.JOINT_NAMES == calib.MOTOR_NAMES


def test_arm_sides_match_calibration_module():
    assert protocol.ARM_SIDES == calib.ARM_SIDES


def test_joint_order_matches_bimanual_action_keys():
    """The wire ordering must be exactly what `BimanualCalibration` produces."""
    expected = tuple(
        f"{side}_{joint}" for side in calib.ARM_SIDES for joint in calib.MOTOR_NAMES
    )
    assert protocol.JOINT_ORDER == expected
    assert len(protocol.JOINT_ORDER) == 12


def test_assets_uses_the_protocol_joint_names():
    assert assets.JOINT_NAMES == protocol.JOINT_NAMES


def test_sim_joint_limits_match_calibration_module():
    """The scene's physics limits and the client's clip must be identical.

    If MuJoCo stops a joint short of what the bridge is willing to command, the
    simulated arm cannot reach poses the real one does -- which is exactly the
    wrist_roll truncation this constant was introduced to fix.
    """
    assert protocol.SIM_JOINT_LIMITS == calib.SIM_JOINT_LIMITS


def test_wrist_roll_is_wider_than_the_urdf_declares():
    """真机行程超过 URDF 声明的 320 度,再叠加 -90 度零位偏移后需要更宽。

    给到整两圈:可达 ±180 度加偏移后是 -270~+90 度,±180 会截掉负方向 90 度。
    """
    urdf_lo, urdf_hi = calib.URDF_JOINT_LIMITS["wrist_roll"]
    sim_lo, sim_hi = calib.SIM_JOINT_LIMITS["wrist_roll"]
    assert sim_hi - sim_lo > urdf_hi - urdf_lo
    assert (sim_hi - sim_lo) == pytest.approx(4 * math.pi)


def test_zero_offset_only_on_wrist_roll():
    """零位偏移只该有 wrist_roll 一项 —— 其余关节的零位由标定文件确定。"""
    assert set(calib.JOINT_ZERO_OFFSETS) == {"wrist_roll"}
    assert calib.JOINT_ZERO_OFFSETS["wrist_roll"] == pytest.approx(-math.pi / 2)


def test_only_known_joints_differ_from_the_urdf():
    """只有 wrist_roll 和 gripper 允许偏离 URDF,其余必须逐字相同。

    两者都是实测发现 URDF 声明的限位卡早了:wrist_roll 差一个 90 度零位,
    gripper 的下限 -10 度并非真正贴合位置。其它关节若也偏离,多半是误改。
    """
    differing = sorted(
        name
        for name in calib.MOTOR_NAMES
        if calib.SIM_JOINT_LIMITS[name] != calib.URDF_JOINT_LIMITS[name]
    )
    assert differing == ["gripper", "wrist_roll"]


def test_camera_keys_match_the_real_rig():
    """These are the keys the real manifest records; datasets key off them."""
    assert protocol.CAMERA_KEYS == ("left_wrist", "right_wrist", "right_front")


def test_frame_nbytes():
    assert protocol.frame_nbytes(640, 480) == 640 * 480 * 3


def test_check_version_accepts_matching_peer():
    protocol.check_version({"protocol_version": protocol.PROTOCOL_VERSION})


def test_check_version_rejects_mismatched_peer():
    import pytest

    with pytest.raises(protocol.ProtocolError, match="protocol"):
        protocol.check_version({"protocol_version": protocol.PROTOCOL_VERSION + 1})


def test_check_version_rejects_missing_version():
    import pytest

    with pytest.raises(protocol.ProtocolError):
        protocol.check_version({})


# -- MJCF post-processing, the parts that need no MuJoCo --------------------


def _arm_like_mjcf() -> ET.Element:
    """A miniature stand-in for the converted URDF: loose base geoms in the
    worldbody, a child body, and paired visual/collision geoms."""
    root = ET.fromstring(
        """
        <mujoco>
          <worldbody>
            <geom name="base_vis" contype="0" conaffinity="0" group="1" type="mesh"/>
            <geom name="base_col" type="mesh"/>
            <body name="shoulder_link">
              <geom name="sh_vis" contype="0" conaffinity="0" group="1" type="mesh"/>
              <geom name="sh_col" type="mesh"/>
              <body name="upper_arm_link">
                <geom name="ua_col" type="mesh"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    return root


def test_wrap_base_body_creates_an_attachable_base():
    root = _arm_like_mjcf()
    assets._wrap_base_body(root)
    worldbody = root.find("worldbody")
    assert [child.tag for child in worldbody] == ["body"]
    base = worldbody.find("body")
    assert base.get("name") == "base_link"
    assert {g.get("name") for g in base.findall("geom")} == {"base_vis", "base_col"}
    assert base.find("body").get("name") == "shoulder_link"


def test_wrap_base_body_is_idempotent_on_a_body_tree():
    root = _arm_like_mjcf()
    assets._wrap_base_body(root)
    before = ET.tostring(root)
    assets._wrap_base_body(root)
    assert ET.tostring(root) == before


def test_separate_visual_collision_moves_only_collision_geoms():
    root = _arm_like_mjcf()
    assets._wrap_base_body(root)
    moved = assets._separate_visual_collision(root)
    assert moved == 3  # base_col, sh_col, ua_col

    by_name = {g.get("name"): g for g in root.iter("geom")}
    for name in ("base_col", "sh_col", "ua_col"):
        assert by_name[name].get("group") == "3", name
    for name in ("base_vis", "sh_vis"):
        assert by_name[name].get("group") == "1", name


def test_exclude_adjacent_collisions_covers_every_linked_pair():
    """Every parent/child pair must be excluded, or the overlap at the joint
    becomes a permanent contact force that pushes the joint off target."""
    root = _arm_like_mjcf()
    assets._wrap_base_body(root)
    count = assets._exclude_adjacent_collisions(root)

    pairs = {
        (e.get("body1"), e.get("body2")) for e in root.find("contact").findall("exclude")
    }
    assert pairs == {
        ("base_link", "shoulder_link"),
        ("shoulder_link", "upper_arm_link"),
    }
    assert count == len(pairs)


# -- 动力学校准 ------------------------------------------------------------


def test_action_delay_default_is_nonzero():
    """仿真必须复现真机的指令滞后,否则 action chunk 学到的时序会偏早。

    真机总滞后约 135 ms,仿真执行器只有 33 ms,差额用纯延迟补。仿真"发指令即
    到位"而真机要等,迁到真机上动作会赶在实际位置之前 —— 而仿真里看不出异常。
    """
    assert protocol.DEFAULT_ACTION_DELAY_STEPS == 3


def test_delayed_step_holds_pose_until_queue_fills(tmp_path):
    """队列未满时保持当前位姿,不能提前下发。"""
    from collections import deque

    class FakeSim:
        """只复现 SimulatorState.step 的排队逻辑,不需要 MuJoCo。"""

        def __init__(self, delay):
            self.action_delay_steps = delay
            self._pending = deque()
            self.applied = []

        def step(self, targets):
            self._pending.append(list(targets))
            if len(self._pending) > self.action_delay_steps:
                self.applied.append(self._pending.popleft())
            else:
                self.applied.append(None)

    sim = FakeSim(delay=2)
    for value in (1.0, 2.0, 3.0, 4.0, 5.0):
        sim.step([value])

    # 前两步无指令下发,之后下发的是 2 步之前那条
    assert sim.applied[0] is None
    assert sim.applied[1] is None
    assert sim.applied[2] == [1.0]
    assert sim.applied[3] == [2.0]
    assert sim.applied[4] == [3.0]


def test_scene_gains_match_the_fitted_values():
    """增益按稳态误差选定(kp=50 -> 0.27 度,真机 0.31 度),改动应是有意识的。

    压低 kp 去凑响应速度会引入真机没有的重力下垂 —— 真机舵机的 PID 带积分项。
    """
    from evo_rlt.sim.assets import SceneConfig

    cfg = SceneConfig()
    assert cfg.control_kp == 50.0
    assert cfg.control_dampratio == 1.0


# -- v2:零件位姿与运动学 ---------------------------------------------------


def test_protocol_version_is_two():
    """v2 起 observation 带零件/夹爪位姿,并新增 FK / IK 两条命令。

    版本号必须跟着涨:旧仿真器不回 ``object_poses``,自动成功判据会静默地拿到
    空字典 —— 那是"每条 episode 都判失败",而不是一个错误。
    """
    assert protocol.PROTOCOL_VERSION == 2


def test_pose_layout_matches_mujoco_free_joints():
    """位姿用 ``[x,y,z,qw,qx,qy,qz]``,和 MuJoCo 自由关节的 qpos 排布一致。"""
    assert protocol.POSE_LEN == 7


def test_ee_bodies_cover_both_arms_and_name_real_links():
    assert set(protocol.EE_BODIES) == set(protocol.ARM_SIDES)
    for side, body in protocol.EE_BODIES.items():
        assert body == f"{side}_gripper_link"


def test_kinematics_commands_exist_and_are_distinct():
    names = [
        protocol.Command.HANDSHAKE, protocol.Command.OBSERVE, protocol.Command.STEP,
        protocol.Command.RESET, protocol.Command.RESET_OBJECTS,
        protocol.Command.FK, protocol.Command.IK, protocol.Command.CLOSE,
    ]
    assert len(set(names)) == len(names)


def test_ik_rotation_weight_frees_only_the_vertical_yaw():
    """位置 3 维 + 倾角 2 维 = 5,正好是 SO-101 的自由度数,只放开绕世界 z 的偏航。

    三个分量都给大(各向同性 1.0)的话,求解器会拿位置去换姿态 —— 实测一个 25mm
    的平移目标解出来位置就差 25mm。三个都给小(各向同性 0.02)则手腕的倾角自己
    漂:同样的平移下倾角误差 0.46~2.11 度,而插销任务最后失败的主因正是螺栓和
    螺套的轴线对不上。分轴之后倾角误差是 0.00 度,位置仍在 0.16mm 以内。
    """
    weight = protocol.DEFAULT_IK_ROTATION_WEIGHT
    assert len(weight) == 3
    assert weight[0] == weight[1] >= 1.0, "倾角必须严格跟随"
    assert weight[2] <= 0.05, "绕竖直轴的偏航必须放开,那是这条臂做不到的一维"
