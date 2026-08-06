from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from evo_rlt.adapters.lerobot.record.online_trainer import OnlineRLTrainer
from evo_rlt.core.actor import ChunkActor
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
    assert torch.equal(batch["rankq_outcome"][:offline_n], torch.full((offline_n,), -1.0))


def test_mixed_batch_keeps_offline_success_for_bc_but_excludes_it_from_rankq():
    trainer = object.__new__(OnlineRLTrainer)
    trainer.cfg = SimpleNamespace(
        batch_size=4,
        offline_batch_fraction=0.5,
        use_stratified_sampling=False,
    )
    trainer.offline_buffer = ReplayBuffer(capacity=2)
    for episode_id in (10, 11):
        trainer.offline_buffer.add(
            _outcome_transition(episode_id, success=True, intervention=False)
        )
    trainer.replay_buffer = ReplayBuffer(capacity=2)
    trainer.replay_buffer.add(
        _outcome_transition(20, success=False, intervention=False)
    )
    trainer.replay_buffer.add(
        _outcome_transition(21, success=True, intervention=False)
    )

    batch, offline_n, online_n = trainer._sample_training_batch()

    assert (offline_n, online_n) == (2, 2)
    assert torch.equal(batch["outcome"][:offline_n], torch.ones(offline_n))
    assert torch.equal(
        batch["rankq_outcome"][:offline_n], torch.full((offline_n,), -1.0)
    )
    assert torch.equal(
        batch["rankq_outcome"][offline_n:], batch["outcome"][offline_n:]
    )


def test_sampling_preserves_transition_outcome_when_episode_id_is_shared():
    """A critical attempt's label belongs to each transition, not to the
    final terminal transition that happened to reuse the same dataset episode
    id. RankQ must see both labels rather than overwriting both with the last.
    """
    trainer = object.__new__(OnlineRLTrainer)
    trainer.cfg = SimpleNamespace(
        batch_size=2,
        offline_batch_fraction=0.0,
        use_stratified_sampling=False,
    )
    trainer.offline_buffer = None
    trainer.replay_buffer = ReplayBuffer(capacity=2)
    failed = _outcome_transition(7, success=False, intervention=False)
    succeeded = _outcome_transition(7, success=True, intervention=False)
    trainer.replay_buffer.add(failed)
    trainer.replay_buffer.add(succeeded)

    batch, _, _ = trainer._sample_training_batch()

    assert sorted(batch["outcome"].tolist()) == [0.0, 1.0]
    assert torch.equal(batch["rankq_outcome"], batch["outcome"])


def test_split_batch_sizes_matches_sample_training_batch():
    """maybe_update()'s 'enough data yet' gate and _sample_training_batch()
    must use the identical split, or the gate can require far more online
    data than a batch actually needs once an offline buffer covers part of
    it (batch_size=10, offline_fraction=0.4 only ever needs 6 online
    transitions, not a full 10)."""
    trainer = object.__new__(OnlineRLTrainer)
    trainer.cfg = SimpleNamespace(batch_size=10, offline_batch_fraction=0.4)
    trainer.offline_buffer = _buffer(range(20))

    online_n, offline_n = trainer._split_batch_sizes()

    assert (online_n, offline_n) == (6, 4)


def test_split_batch_sizes_no_offline_buffer_needs_full_batch():
    trainer = object.__new__(OnlineRLTrainer)
    trainer.cfg = SimpleNamespace(batch_size=10, offline_batch_fraction=0.4)
    trainer.offline_buffer = None

    online_n, offline_n = trainer._split_batch_sizes()

    assert (online_n, offline_n) == (10, 0)


def test_offline_cache_loader_accepts_actor_supervised_dict_cache(tmp_path):
    transition = _transition(1.0)
    transition.outcome = torch.tensor(1.0)
    transition.intervention_mask = torch.ones_like(transition.exec_chunk)
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
                "build_complete": True,
                "actor_supervision_schema_version": 1,
                "actor_supervision_source": "successful_demonstration",
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
    sampled = replay.sample(1)
    assert torch.equal(sampled["intervention_mask_flat"], torch.ones(1, 8))
    assert torch.equal(sampled["outcome"], torch.ones(1))


def test_offline_cache_loader_rejects_cache_without_actor_supervision_schema(tmp_path):
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
                "build_complete": True,
                "rl_action_arms": "both",
            }
        )
    )

    with pytest.raises(ValueError, match="train only the Critic"):
        OnlineRLTrainer._load_offline_buffer(
            str(tmp_path),
            policy_cfg=SimpleNamespace(chunk_length=2, action_dim=4),
        )


def test_offline_cache_loader_rejects_incomplete_cache(tmp_path):
    transition = _transition(1.0)
    transition.outcome = torch.tensor(1.0)
    transition.intervention_mask = torch.ones_like(transition.exec_chunk)
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
                "build_complete": False,
                "actor_supervision_schema_version": 1,
                "rl_action_arms": "both",
            }
        )
    )

    with pytest.raises(ValueError, match="incomplete"):
        OnlineRLTrainer._load_offline_buffer(
            str(tmp_path),
            policy_cfg=SimpleNamespace(chunk_length=2, action_dim=4),
        )


def test_offline_bc_warmstart_updates_actor_before_online_warmup():
    trainer = object.__new__(OnlineRLTrainer)
    trainer.cfg = SimpleNamespace(batch_size=4)
    trainer.policy_cfg = SimpleNamespace(
        beta=0.0,
        actor_demo_bc_weight=1.0,
        actor_action_clip_delta=0.2,
        chunk_length=2,
    )
    trainer.online_actor_update_interval = 2
    trainer.offline_buffer = ReplayBuffer(capacity=4)
    for value in (0.02, 0.04, 0.06, 0.08):
        transition = _transition(value)
        transition.outcome = torch.tensor(1.0)
        transition.intervention_mask = torch.ones_like(transition.exec_chunk)
        trainer.offline_buffer.add(transition)

    actor = ChunkActor(
        state_dim=3,
        chunk_dim=8,
        hidden_dim=8,
        num_layers=1,
        fixed_std=0.0,
        ref_dropout_p=0.0,
        residual_to_ref=True,
    )

    class _ActorOnlyPolicy(torch.nn.Module):
        def __init__(self, actor):
            super().__init__()
            self.actor = actor

    trainer.policy = _ActorOnlyPolicy(actor)
    trainer.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-2)
    before = {name: value.detach().clone() for name, value in actor.named_parameters()}

    num_updates, info = trainer._run_offline_bc_warmstart(requested_updates=4)

    assert num_updates == 2
    assert info["loss_actor_demo_bc"] > 0
    assert info["actor_grad_norm"] > 0
    assert any(
        not torch.equal(value, before[name])
        for name, value in actor.named_parameters()
    )


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
    transition.outcome = torch.tensor(1.0)
    transition.intervention_mask = torch.ones_like(transition.exec_chunk)
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
                "build_complete": True,
                "actor_supervision_schema_version": 1,
                "actor_supervision_source": "successful_demonstration",
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


class TestActorUnfreezeRamp:
    """OnlineRLTrainer._actor_update_interval_for: actor_update_interval
    should stay frozen through critic_only, then ramp down to
    online_actor_update_interval over actor_unfreeze_ramp_episodes episodes
    instead of snapping straight to it (see OnlineRLConfig.
    actor_unfreeze_ramp_episodes for why the hard flip this replaces caused
    actor jitter)."""

    def _trainer(self, *, online_actor_update_interval: int, ramp_episodes: int) -> OnlineRLTrainer:
        trainer = object.__new__(OnlineRLTrainer)
        trainer.cfg = SimpleNamespace(actor_unfreeze_ramp_episodes=ramp_episodes)
        trainer.online_actor_update_interval = online_actor_update_interval
        return trainer

    def test_still_critic_only_is_frozen(self):
        trainer = self._trainer(online_actor_update_interval=2, ramp_episodes=10)
        assert trainer._actor_update_interval_for(recorded_episodes=5, critic_only_until=10) == 10**9

    def test_first_episode_of_ramp_is_the_widest_interval(self):
        trainer = self._trainer(online_actor_update_interval=2, ramp_episodes=10)
        # episodes_since_unfreeze=0 -> multiplier=ramp_episodes=10
        assert trainer._actor_update_interval_for(recorded_episodes=10, critic_only_until=10) == 20

    def test_mid_ramp_is_between_start_and_target(self):
        trainer = self._trainer(online_actor_update_interval=2, ramp_episodes=10)
        # episodes_since_unfreeze=5 -> multiplier=5
        assert trainer._actor_update_interval_for(recorded_episodes=15, critic_only_until=10) == 10

    def test_last_ramp_episode_already_hits_target(self):
        trainer = self._trainer(online_actor_update_interval=2, ramp_episodes=10)
        # episodes_since_unfreeze=9 -> multiplier=1 -> exactly the target, no
        # discontinuity at the boundary with the post-ramp branch below.
        assert trainer._actor_update_interval_for(recorded_episodes=19, critic_only_until=10) == 2

    def test_past_ramp_window_stays_at_target(self):
        trainer = self._trainer(online_actor_update_interval=2, ramp_episodes=10)
        assert trainer._actor_update_interval_for(recorded_episodes=20, critic_only_until=10) == 2
        assert trainer._actor_update_interval_for(recorded_episodes=100, critic_only_until=10) == 2

    def test_zero_ramp_episodes_reproduces_old_hard_flip(self):
        trainer = self._trainer(online_actor_update_interval=2, ramp_episodes=0)
        assert trainer._actor_update_interval_for(recorded_episodes=9, critic_only_until=10) == 10**9
        assert trainer._actor_update_interval_for(recorded_episodes=10, critic_only_until=10) == 2
        assert trainer._actor_update_interval_for(recorded_episodes=50, critic_only_until=10) == 2


class TestActorDeployScaleRamp:
    def _trainer(self, *, warmup_completed=5, critic_only=10, ramp=10):
        trainer = object.__new__(OnlineRLTrainer)
        trainer.warmup_completed_at_episode = warmup_completed
        trainer.cfg = SimpleNamespace(
            critic_only_episodes=critic_only,
            actor_unfreeze_ramp_episodes=ramp,
        )
        return trainer

    def test_warmup_and_critic_only_are_exact_vla(self):
        trainer = self._trainer()
        trainer.warmup_completed_at_episode = None
        assert trainer._actor_deploy_scale_for(100) == 0.0
        trainer.warmup_completed_at_episode = 5
        assert trainer._actor_deploy_scale_for(14) == 0.0
        assert trainer._actor_deploy_scale_for(15) == 0.0

    def test_deploy_scale_ramps_after_zero_boundary_episode(self):
        trainer = self._trainer()
        assert trainer._actor_deploy_scale_for(16) == pytest.approx(0.1)
        assert trainer._actor_deploy_scale_for(20) == pytest.approx(0.5)
        assert trainer._actor_deploy_scale_for(25) == 1.0
        assert trainer._actor_deploy_scale_for(100) == 1.0

    def test_zero_ramp_hard_flips_only_after_critic_only_boundary(self):
        trainer = self._trainer(ramp=0)
        assert trainer._actor_deploy_scale_for(15) == 0.0
        assert trainer._actor_deploy_scale_for(16) == 1.0


class TestInterventionCorrectionMetrics:
    """OnlineRLTrainer._intervention_correction_metrics: how far human
    corrections actually move exec away from ref, compared against
    actor_action_clip_delta -- the residual actor can never autonomously
    reproduce a correction bigger than that bound, regardless of training."""

    def test_computes_mean_and_frac_exceeding_clip(self):
        trainer = object.__new__(OnlineRLTrainer)
        trainer.replay_buffer = ReplayBuffer(capacity=10)
        trainer.policy_cfg = SimpleNamespace(actor_action_clip_delta=0.1)

        small = _transition(0.05)  # within clip
        small.intervention = torch.tensor(1.0)
        big = _transition(0.2)  # exceeds clip
        big.intervention = torch.tensor(1.0)
        autonomous = _transition(0.3)  # not an intervention chunk -- must be excluded
        trainer.replay_buffer.add(small)
        trainer.replay_buffer.add(big)
        trainer.replay_buffer.add(autonomous)

        metrics = trainer._intervention_correction_metrics()

        assert metrics["online_rl/intervention_correction_max_mean"] == pytest.approx((0.05 + 0.2) / 2)
        assert metrics["online_rl/intervention_correction_frac_exceeds_clip"] == pytest.approx(0.5)

    def test_no_interventions_returns_zeros(self):
        trainer = object.__new__(OnlineRLTrainer)
        trainer.replay_buffer = ReplayBuffer(capacity=10)
        trainer.policy_cfg = SimpleNamespace(actor_action_clip_delta=0.1)
        trainer.replay_buffer.add(_transition(0.5))  # intervention=0.0 by default

        metrics = trainer._intervention_correction_metrics()

        assert metrics["online_rl/intervention_correction_max_mean"] == 0.0
        assert metrics["online_rl/intervention_correction_frac_exceeds_clip"] == 0.0

    def test_no_clip_configured_frac_is_zero_not_crash(self):
        trainer = object.__new__(OnlineRLTrainer)
        trainer.replay_buffer = ReplayBuffer(capacity=10)
        trainer.policy_cfg = SimpleNamespace(actor_action_clip_delta=None)
        t = _transition(0.5)
        t.intervention = torch.tensor(1.0)
        trainer.replay_buffer.add(t)

        metrics = trainer._intervention_correction_metrics()

        assert metrics["online_rl/intervention_correction_max_mean"] == pytest.approx(0.5)
        assert metrics["online_rl/intervention_correction_frac_exceeds_clip"] == 0.0


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
