from types import SimpleNamespace

import pytest


def _robot(action_names):
    return SimpleNamespace(action_features={name: object() for name in action_names})


def _policy_cfg(actor_rl_arm, action_dim=None):
    return SimpleNamespace(actor_rl_arm=actor_rl_arm, action_dim=action_dim)


LEFT_RIGHT_NAMES = [
    "left_shoulder_pan.pos", "left_shoulder_lift.pos", "left_elbow_flex.pos",
    "left_wrist_flex.pos", "left_wrist_roll.pos", "left_gripper.pos",
    "right_shoulder_pan.pos", "right_shoulder_lift.pos", "right_elbow_flex.pos",
    "right_wrist_flex.pos", "right_wrist_roll.pos", "right_gripper.pos",
]


def test_left_right_block_order_passes():
    backend = pytest.importorskip("evo_rlt.adapters.lerobot.record.backend")
    backend._validate_actor_rl_arm_action_order(
        _robot(LEFT_RIGHT_NAMES), _policy_cfg("left", action_dim=12)
    )


def test_both_mode_skips_the_check_entirely():
    backend = pytest.importorskip("evo_rlt.adapters.lerobot.record.backend")
    # A single-arm (non-block-ordered) robot would fail the "left" check, but
    # "both" doesn't rely on the ordering assumption at all -- must be a no-op.
    backend._validate_actor_rl_arm_action_order(
        _robot(["shoulder_pan.pos", "gripper.pos"]), _policy_cfg("both")
    )


def test_right_arm_first_is_rejected():
    backend = pytest.importorskip("evo_rlt.adapters.lerobot.record.backend")
    reversed_names = LEFT_RIGHT_NAMES[6:] + LEFT_RIGHT_NAMES[:6]
    with pytest.raises(ValueError, match="left_\\* dims"):
        backend._validate_actor_rl_arm_action_order(
            _robot(reversed_names), _policy_cfg("left", action_dim=12)
        )


def test_interleaved_left_right_is_rejected():
    backend = pytest.importorskip("evo_rlt.adapters.lerobot.record.backend")
    interleaved = [
        "left_shoulder_pan.pos", "right_shoulder_pan.pos",
        "left_shoulder_lift.pos", "right_shoulder_lift.pos",
    ]
    with pytest.raises(ValueError, match="ordered as"):
        backend._validate_actor_rl_arm_action_order(
            _robot(interleaved), _policy_cfg("left", action_dim=4)
        )


def test_odd_action_count_is_rejected():
    backend = pytest.importorskip("evo_rlt.adapters.lerobot.record.backend")
    with pytest.raises(ValueError, match="even number"):
        backend._validate_actor_rl_arm_action_order(
            _robot(LEFT_RIGHT_NAMES[:11]), _policy_cfg("left", action_dim=11)
        )


def test_action_dim_mismatch_is_rejected():
    backend = pytest.importorskip("evo_rlt.adapters.lerobot.record.backend")
    with pytest.raises(ValueError, match="action_dim"):
        backend._validate_actor_rl_arm_action_order(
            _robot(LEFT_RIGHT_NAMES), _policy_cfg("left", action_dim=10)
        )
