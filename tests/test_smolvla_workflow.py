from __future__ import annotations

import json
from pathlib import Path

import pytest

from evo_rlt.cli import act as act_cli
from evo_rlt.cli.smolvla import (
    DEFAULT_PROFILE,
    REPO_ROOT,
    checkpoint_rename_map,
    load_profile,
)

TRAIN_CONFIG = REPO_ROOT / "configs" / "smolvla" / "train_config.json"


def _train_config() -> dict:
    return json.loads(TRAIN_CONFIG.read_text())


def test_both_policies_share_one_dataset_profile() -> None:
    """Two profiles could drift; one cannot.

    The profile describes the dataset, not the policy, and the two success
    rates are only comparable if both policies were trained and evaluated on
    the same episodes.
    """
    assert DEFAULT_PROFILE == act_cli.DEFAULT_PROFILE
    profile = load_profile(DEFAULT_PROFILE)
    assert profile["expected"]["episodes"] == 122
    assert profile["expected"]["frames"] == 83247


def test_train_config_uses_the_profile_dataset() -> None:
    profile = load_profile(DEFAULT_PROFILE)
    dataset = _train_config()["dataset"]
    assert dataset["repo_id"] == profile["repo_id"]
    assert dataset["root"] == profile["merged_root"]


def test_train_config_finetunes_instead_of_random_init() -> None:
    """`pretrained_path`, not a bare `type`, is what loads the base weights.

    Without it lerobot builds a randomly initialised 450M model and trains it
    happily -- the logs are identical and only the success rate ever tells you.
    122 demonstrations cannot train a VLA from scratch.
    """
    policy = _train_config()["policy"]
    assert policy["type"] == "smolvla"
    assert policy["pretrained_path"] == "lerobot/smolvla_base"


def test_train_config_trains_the_vision_encoder() -> None:
    """Both flags, always.

    set_requires_grad() freezes the entire VLM -- vision encoder included --
    whenever train_expert_only is true, so setting only freeze_vision_encoder
    to false trains no vision at all and gives no sign of it.
    """
    policy = _train_config()["policy"]
    assert policy["freeze_vision_encoder"] is False
    assert policy["train_expert_only"] is False


def test_train_config_does_not_reload_vlm_weights_over_the_checkpoint() -> None:
    """The VLM weights come from pretrained_path; a second copy would overwrite
    them, and needs a repo that is only cached here for its processor."""
    assert _train_config()["policy"]["load_vlm_weights"] is False


def test_train_config_keeps_the_pretrained_chunk_size() -> None:
    """chunk_size sets the action expert's sequence length: changing it
    discards the pretrained expert while still looking like a fine-tune."""
    assert _train_config()["policy"]["chunk_size"] == 50


def test_train_config_keeps_contact_safe_replanning() -> None:
    """The base ships n_action_steps=50 -- 1.67 s open-loop, far too long for a
    contact-sensitive insertion. 10 replans every 0.33 s."""
    assert _train_config()["policy"]["n_action_steps"] == 10


def test_train_config_spells_out_what_the_base_config_used_to_supply() -> None:
    """Three fields whose SmolVLAConfig defaults differ from the base's config.

    `--policy.path` used to load the base's config.json and inherit these. A
    config_path run builds the policy config from this json alone, so anything
    omitted silently falls back to the dataclass default and changes the model:
    num_expert_layers -1 vs 0, prefix_length -1 vs 0, pad_language_to
    "longest" vs "max_length".
    """
    policy = _train_config()["policy"]
    assert policy["num_expert_layers"] == 0
    assert policy["prefix_length"] == 0
    assert policy["pad_language_to"] == "max_length"


def test_train_config_matches_the_lr_decay_to_the_step_budget() -> None:
    """A mismatch either decays the LR to its floor long before training ends,
    or never finishes decaying."""
    cfg = _train_config()
    assert cfg["policy"]["scheduler_decay_steps"] == cfg["steps"]


def test_train_config_pins_the_camera_rename_map() -> None:
    """The base hard-codes camera1/2/3; this rig records left_wrist /
    right_wrist / right_front. Without a map lerobot-train refuses to start."""
    rename = _train_config()["rename_map"]
    assert set(rename.values()) == {
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.images.camera3",
    }
    assert set(rename) == set(load_profile(DEFAULT_PROFILE)["expected"]["image_features"])


def test_rollout_reads_the_rename_map_from_the_checkpoint(tmp_path: Path) -> None:
    """Rollout must use the map its checkpoint was trained with.

    Reading it from the checkpoint's own train_config.json makes that
    structural: the file is written by the run that produced these weights, so
    there is no second copy to keep in sync. Disagreement would feed every
    camera to the wrong input slot, and the only symptom is a policy that
    behaves as if it had never been trained.
    """
    rename = {"observation.images.right_front": "observation.images.camera1"}
    (tmp_path / "train_config.json").write_text(json.dumps({"rename_map": rename}))
    assert checkpoint_rename_map(tmp_path) == rename


def test_rollout_refuses_a_checkpoint_without_a_train_config(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="which camera fed which input slot"):
        checkpoint_rename_map(tmp_path)


def test_rollout_default_episode_time_accounts_for_slow_loops() -> None:
    """episode_time_s is wall clock while the simulator advances 1/fps per step,
    so a default sized for 30 Hz would cut episodes off partway through."""
    from evo_rlt.cli.smolvla import build_parser

    args = build_parser().parse_args(["rollout", "--checkpoint", "x"])
    assert args.episode_time_s >= 120


def test_rollout_passes_the_rename_map_to_make_policy() -> None:
    """The camera rename map has to reach make_policy(), not just the preprocessor.

    make_policy() validates the policy's expected visual features against the
    dataset's *original* camera keys and aborts with "Feature mismatch" before
    the rollout starts -- exactly the case for a renaming policy like SmolVLA
    (camera1/2/3). A non-empty rename_map skips that check. Upstream's
    lerobot_record.py, which backend.record() is forked from, omits the
    argument, so this is the kind of line a resync silently reverts.
    """
    import ast
    import inspect

    from evo_rlt.adapters.lerobot.record import backend

    tree = ast.parse(inspect.getsource(backend))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "make_policy"
    ]
    assert calls, "make_policy() is no longer called in backend.py"
    for call in calls:
        assert "rename_map" in {kw.arg for kw in call.keywords}, (
            "make_policy() must be called with rename_map=cfg.dataset.rename_map, "
            "or SmolVLA rollouts die on 'Feature mismatch'"
        )


def test_image_feature_check_applies_the_rename_map() -> None:
    """record_loop's own image-feature check must compare renamed keys too.

    It exists to catch --dataset.video=false wiping out every image feature,
    which is still worth catching -- but it ran against the dataset's original
    camera names, so a renaming policy tripped it on all three cameras right
    after the first episode started.
    """
    from evo_rlt.adapters.lerobot.record.loop import _validate_policy_image_features

    class _Type:
        value = "VISUAL"

    class _Feature:
        type = _Type()

    class _Config:
        input_features = {f"observation.images.camera{i}": _Feature() for i in (1, 2, 3)}

    class _Policy:
        config = _Config()

    dataset_features = {
        "observation.images.right_front": {"dtype": "video"},
        "observation.images.left_wrist": {"dtype": "video"},
        "observation.images.right_wrist": {"dtype": "video"},
    }
    rename_map = _train_config()["rename_map"]

    _validate_policy_image_features(_Policy(), dataset_features, rename_map)

    # Without the map the same rollout is rejected -- and an empty map must
    # still reject a genuinely image-less dataset.
    with pytest.raises(ValueError, match="camera1"):
        _validate_policy_image_features(_Policy(), dataset_features)
    with pytest.raises(ValueError, match="dataset.video"):
        _validate_policy_image_features(_Policy(), {}, rename_map)
