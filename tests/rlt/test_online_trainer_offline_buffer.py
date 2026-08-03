from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from evo_rlt.adapters.lerobot.record.online_trainer import OnlineRLTrainer
from evo_rlt.core.interfaces import ChunkTransition
from evo_rlt.core.replay_buffer import ReplayBuffer


def _transition(value: float, *, chunk_length: int = 2, action_dim: int = 4) -> ChunkTransition:
    return ChunkTransition(
        state_vec=torch.full((3,), value),
        exec_chunk=torch.full((chunk_length, action_dim), value),
        ref_chunk=torch.zeros(chunk_length, action_dim),
        reward_seq=torch.zeros(chunk_length),
        next_state_vec=torch.full((3,), value + 1),
        next_ref_chunk=torch.zeros(chunk_length, action_dim),
        done=torch.tensor(0.0),
        intervention=torch.tensor(0.0),
        actual_steps=torch.tensor(chunk_length),
        episode_id=torch.tensor(int(value)),
    )


def _outcome_transition(
    episode_id: int,
    *,
    success: bool,
    intervention: bool,
) -> ChunkTransition:
    transition = _transition(float(episode_id))
    transition.episode_id = torch.tensor(episode_id)
    transition.done = torch.tensor(1.0)
    transition.intervention = torch.tensor(float(intervention))
    transition.outcome = torch.tensor(float(success))
    if success:
        transition.reward_seq[-1] = 1.0
    return transition


def _buffer(values: range) -> ReplayBuffer:
    replay = ReplayBuffer(capacity=len(values))
    for value in values:
        replay.add(_transition(float(value)))
    return replay


def test_mixed_batch_has_fixed_offline_online_split():
    trainer = object.__new__(OnlineRLTrainer)
    trainer.cfg = SimpleNamespace(
        batch_size=10,
        offline_batch_fraction=0.4,
        use_stratified_sampling=False,
    )
    trainer.offline_buffer = _buffer(range(20))
    trainer.replay_buffer = _buffer(range(20, 40))

    batch, offline_n, online_n = trainer._sample_training_batch()

    assert offline_n == 4
    assert online_n == 6
    assert batch["state_vec"].shape[0] == 10


def test_offline_cache_loader_accepts_v2_dict_cache(tmp_path):
    transition = _transition(1.0)
    cache_dict = {
        key: getattr(transition, key)
        for key in ChunkTransition.__dataclass_fields__
        if isinstance(getattr(transition, key), torch.Tensor)
    }
    cache_path = tmp_path / "chunk_transitions_train.pt"
    torch.save([cache_dict], cache_path)
    (tmp_path / "cache_metadata.json").write_text(
        json.dumps(
            {
                "exec_chunk_source": "demonstrated_action",
                "rl_action_arms": "both",
            }
        )
    )

    replay = OnlineRLTrainer._load_offline_buffer(
        str(tmp_path),
        policy_cfg=SimpleNamespace(chunk_length=2, action_dim=4),
    )

    assert replay is not None
    assert len(replay) == 1
    assert torch.equal(replay.buffer[0].exec_chunk, transition.exec_chunk)


def test_autonomous_success_metrics_exclude_human_rescues():
    trainer = object.__new__(OnlineRLTrainer)
    trainer.replay_buffer = ReplayBuffer(capacity=10)
    trainer.replay_buffer.add(_outcome_transition(0, success=True, intervention=False))
    trainer.replay_buffer.add(_outcome_transition(1, success=False, intervention=False))
    trainer.replay_buffer.add(_outcome_transition(2, success=True, intervention=True))

    metrics = trainer._autonomous_success_metrics(rolling_window=2)

    assert metrics["online_rl/episode_reward"] == 1.0
    assert metrics["online_rl/episode_intervened"] == 1.0
    assert metrics["online_rl/episode_autonomous_success"] == 0.0
    assert metrics["online_rl/autonomous_episodes"] == 2
    assert metrics["online_rl/autonomous_successes"] == 1
    assert metrics["online_rl/autonomous_success_rate"] == 0.5
    assert metrics["online_rl/autonomous_success_rate_rolling_2"] == 0.0
    assert metrics["online_rl/autonomous_episodes_rolling_2"] == 1
    assert metrics["online_rl/intervention_rate"] == 1 / 3
    assert metrics["online_rl/intervention_rate_rolling_2"] == 0.5


def _write_offline_cache(tmp_path, *, extra_metadata: dict) -> None:
    transition = _transition(1.0)
    cache_dict = {
        key: getattr(transition, key)
        for key in ChunkTransition.__dataclass_fields__
        if isinstance(getattr(transition, key), torch.Tensor)
    }
    torch.save([cache_dict], tmp_path / "chunk_transitions_train.pt")
    (tmp_path / "cache_metadata.json").write_text(
        json.dumps(
            {
                "exec_chunk_source": "demonstrated_action",
                "rl_action_arms": "both",
                **extra_metadata,
            }
        )
    )


class TestRewardSchemaConsistency:
    def test_matching_reward_params_load_fine(self, tmp_path):
        _write_offline_cache(
            tmp_path,
            extra_metadata={
                "reward_schema_version": 2,
                "milestone_reward": 0.3,
                "terminal_reward": 1.0,
                "time_decay": 0.995,
            },
        )
        replay = OnlineRLTrainer._load_offline_buffer(
            str(tmp_path),
            policy_cfg=SimpleNamespace(chunk_length=2, action_dim=4),
            online_rl_cfg=SimpleNamespace(milestone_reward=0.3, terminal_reward=1.0, time_decay=0.995),
        )
        assert replay is not None

    def test_mismatched_milestone_reward_raises(self, tmp_path):
        _write_offline_cache(
            tmp_path,
            extra_metadata={
                "reward_schema_version": 2,
                "milestone_reward": 0.3,
                "terminal_reward": 1.0,
                "time_decay": 0.995,
            },
        )
        with pytest.raises(ValueError, match="milestone_reward"):
            OnlineRLTrainer._load_offline_buffer(
                str(tmp_path),
                policy_cfg=SimpleNamespace(chunk_length=2, action_dim=4),
                online_rl_cfg=SimpleNamespace(milestone_reward=0.5, terminal_reward=1.0, time_decay=0.995),
            )

    def test_mismatched_time_decay_raises(self, tmp_path):
        _write_offline_cache(
            tmp_path,
            extra_metadata={
                "reward_schema_version": 2,
                "milestone_reward": 0.3,
                "terminal_reward": 1.0,
                "time_decay": 0.995,
            },
        )
        with pytest.raises(ValueError, match="time_decay"):
            OnlineRLTrainer._load_offline_buffer(
                str(tmp_path),
                policy_cfg=SimpleNamespace(chunk_length=2, action_dim=4),
                online_rl_cfg=SimpleNamespace(milestone_reward=0.3, terminal_reward=1.0, time_decay=1.0),
            )

    def test_old_cache_without_schema_raises_if_online_uses_shaping(self, tmp_path):
        """A cache built before milestone/time-decay support has no
        reward_schema_version -- refuse to silently mix it with an online run
        that actually uses milestone/decay shaping."""
        _write_offline_cache(tmp_path, extra_metadata={})
        with pytest.raises(ValueError, match="predates milestone/time-decay"):
            OnlineRLTrainer._load_offline_buffer(
                str(tmp_path),
                policy_cfg=SimpleNamespace(chunk_length=2, action_dim=4),
                online_rl_cfg=SimpleNamespace(milestone_reward=0.3, terminal_reward=1.0, time_decay=0.995),
            )

    def test_old_cache_without_schema_loads_fine_if_online_shaping_disabled(self, tmp_path):
        """time_decay=1.0 and milestone_reward=0 reproduces the old fixed-
        reward-on-success behavior exactly, so an old-schema cache is still
        valid in that configuration."""
        _write_offline_cache(tmp_path, extra_metadata={})
        replay = OnlineRLTrainer._load_offline_buffer(
            str(tmp_path),
            policy_cfg=SimpleNamespace(chunk_length=2, action_dim=4),
            online_rl_cfg=SimpleNamespace(milestone_reward=0.0, terminal_reward=1.0, time_decay=1.0),
        )
        assert replay is not None

    def test_no_online_rl_cfg_skips_check(self, tmp_path):
        """Bare cache-inspection callers (no online_rl_cfg) aren't blocked by
        this check -- it has nothing to compare against."""
        _write_offline_cache(tmp_path, extra_metadata={})
        replay = OnlineRLTrainer._load_offline_buffer(
            str(tmp_path),
            policy_cfg=SimpleNamespace(chunk_length=2, action_dim=4),
        )
        assert replay is not None


def test_episode_reward_sums_milestone_and_terminal_chunks():
    trainer = object.__new__(OnlineRLTrainer)
    trainer.replay_buffer = ReplayBuffer(capacity=10)
    milestone = _transition(7.0)
    milestone.episode_id = torch.tensor(7)
    milestone.reward_seq[-1] = 0.3
    milestone.outcome = torch.tensor(1.0)
    terminal = _outcome_transition(7, success=True, intervention=False)
    trainer.replay_buffer.add(milestone)
    trainer.replay_buffer.add(terminal)

    metrics = trainer._autonomous_success_metrics()

    assert metrics["online_rl/episode_reward"] == pytest.approx(1.3)
