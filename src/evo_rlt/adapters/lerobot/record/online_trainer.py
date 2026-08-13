"""Standalone online-RL "brain": replay buffer + TD3+BC gradient updates for
`rlt_ac`, extracted out of `backend.record()`'s closures into their own
testable unit (see `backend.py`, which calls this unchanged).

This is a pure extraction: the control flow and math below are copied
verbatim from the closures that used to live inside `record()`
(`_run_online_rl_update`, `_warmup_satisfied`, `_save_online_rl_latest_state`)
-- only the closure variables became `self` attributes. Do not "clean up" the
ordering here without re-checking against RLT's TD3+BC correctness notes
(critic-before-actor step order, the tau=0/actor_update_interval=10**9
freeze-and-restore trick, warmup anchored to when it was actually satisfied
rather than a fixed offset) -- see the comments inline and on
`OnlineRLConfig` in `backend.py`.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import time
from collections import deque
from pathlib import Path
from typing import Any

import torch

from evo_rlt.adapters.lerobot.online_collector import RLTOnlineCollector
from evo_rlt.core.interfaces import ChunkTransition
from evo_rlt.core.losses import (
    _apply_slew_rate_limit_flat,
    _valid_action_mask,
    actor_behavior_cloning_loss,
    q_action_sensitivity,
)
from evo_rlt.core.replay_buffer import ReplayBuffer
from evo_rlt.core.utils import project_action_delta, soft_update, unflatten_chunk


class OnlineRLTrainer:
    """Owns the replay buffer, collector, optimizers, and warmup/critic-only
    state for one online-RL session, and can run gradient updates on `policy`
    in place. `policy` must already be a live `ChunkACPolicy` on the target
    device -- this class does not build or move it."""

    def __init__(self, policy: Any, online_rl_cfg: Any, policy_cfg: Any):
        self.policy = policy
        self.cfg = online_rl_cfg
        self.policy_cfg = policy_cfg

        self.replay_buffer = ReplayBuffer(capacity=online_rl_cfg.replay_capacity)
        self.offline_buffer = self._load_offline_buffer(
            online_rl_cfg.offline_cache_path,
            policy_cfg=policy_cfg,
            online_rl_cfg=online_rl_cfg,
        )
        self.collector = RLTOnlineCollector(
            replay_buffer=self.replay_buffer,
            chunk_length=policy_cfg.chunk_length,
            action_dim=policy_cfg.action_dim,
            milestone_reward=online_rl_cfg.milestone_reward,
            terminal_reward=online_rl_cfg.terminal_reward,
            time_decay=online_rl_cfg.time_decay,
        )
        # Separate optimizers (not just param groups on one Adam): actor lr
        # << critic lr since the critic should adapt quickly while the actor
        # -- which directly drives the robot -- should not, and this also
        # makes actor/critic state separately checkpointable/restartable.
        self.actor_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=online_rl_cfg.lr_actor)
        self.critic_optimizer = torch.optim.Adam(policy.critic.parameters(), lr=online_rl_cfg.lr_critic)
        # policy_cfg.actor_update_interval is read live by ChunkACPolicy.forward()
        # on every call; remembered here so the critic-only warmup window (below)
        # can temporarily inflate it and then restore the real value.
        self.online_actor_update_interval = policy_cfg.actor_update_interval
        # ChunkACPolicy.forward() soft-updates target_critic BEFORE the
        # caller's optimizer.step() actually applies this iteration's
        # critic gradient -- i.e. one iteration too early, off standard
        # TD3 order. forward() is shared with offline training
        # (lerobot-train), so it isn't touched; instead policy_cfg.tau is
        # temporarily zeroed around each online forward() call (making its
        # internal soft_update(..., tau=0) an exact no-op, since
        # target = (1-tau)*target + tau*source), and the real soft update
        # runs here afterwards, once critic.step() has actually happened.
        self.online_tau = policy_cfg.tau
        # Set once, the episode warmup actually finished on -- see
        # warmup_satisfied(). None means warmup hasn't completed yet.
        self.warmup_completed_at_episode: int | None = None

        # Actor weights may learn trusted demonstrations during warmup, but
        # those weights must not immediately drive real hardware. Deployment
        # starts as exact VLA pass-through and is scheduled in start_episode.
        self.actor_deploy_scale = 0.0
        self.policy.set_actor_deploy_scale(0.0)

        self.wandb_run = self._init_wandb(online_rl_cfg, policy_cfg)

    @staticmethod
    def _resolve_offline_cache_path(path: str | Path) -> Path:
        resolved = Path(path).expanduser()
        if resolved.is_dir():
            resolved = resolved / "chunk_transitions_train.pt"
        return resolved

    @staticmethod
    def _checkpoint_identity(path: str) -> str:
        candidate = Path(path).expanduser()
        return str(candidate.resolve()) if candidate.exists() else path

    @staticmethod
    def _check_reward_schema(metadata: dict, cache_path: Path, online_rl_cfg: Any) -> None:
        """Refuse to silently mix offline and online transitions that were
        built under different reward scales.

        RLTOnlineCollector rewards a critical-phase attempt with
        milestone_reward/terminal_reward * time_decay ** (chunks closed) --
        see OnlineRLConfig in backend.py. build_transition_cache_v2 mirrors
        that formula (reward_schema_version>=2 caches), so the two only
        agree if built/run with matching milestone_reward/terminal_reward/
        time_decay values. Skipped entirely if `online_rl_cfg` is None (no
        online-side config to compare against, e.g. a bare cache-inspection
        call) or if online_rl_cfg has no milestone/decay attributes.
        """
        if online_rl_cfg is None:
            return
        milestone_reward = getattr(online_rl_cfg, "milestone_reward", None)
        time_decay = getattr(online_rl_cfg, "time_decay", None)
        if milestone_reward is None and time_decay is None:
            return
        online_uses_shaping = (milestone_reward or 0.0) != 0.0 or (time_decay or 1.0) != 1.0
        if metadata.get("reward_schema_version") is None:
            if online_uses_shaping:
                raise ValueError(
                    f"Offline cache {cache_path} predates milestone/time-decay reward "
                    "support (no reward_schema_version in cache_metadata.json), but this "
                    f"online run uses milestone_reward={milestone_reward} "
                    f"time_decay={time_decay} -- rebuild the cache with the current "
                    "evo-rlt-build-transition-cache-v2 (--milestone-reward/--terminal-reward/"
                    "--time-decay matching these online_rl flags), or set "
                    "online_rl.milestone_reward=0 and online_rl.time_decay=1.0 to match "
                    "the cache's old fixed-reward-on-success behavior."
                )
            return
        for metadata_key, cfg_attr in (
            ("milestone_reward", "milestone_reward"),
            ("terminal_reward", "terminal_reward"),
            ("time_decay", "time_decay"),
        ):
            cached_value = metadata.get(metadata_key)
            current_value = getattr(online_rl_cfg, cfg_attr, None)
            if cached_value is None or current_value is None:
                continue
            if not math.isclose(cached_value, current_value, rel_tol=1e-6, abs_tol=1e-9):
                raise ValueError(
                    f"Offline cache {cache_path} {metadata_key}={cached_value!r} != "
                    f"online_rl.{cfg_attr}={current_value!r} -- offline and online reward "
                    f"scales must match. Rebuild the cache with --{metadata_key.replace('_', '-')}"
                    f"={current_value}, or change online_rl.{cfg_attr} to {cached_value}."
                )

    @classmethod
    def _load_offline_buffer(
        cls,
        path: str | None,
        *,
        policy_cfg: Any,
        online_rl_cfg: Any = None,
    ) -> ReplayBuffer | None:
        if path is None:
            return None
        cache_path = cls._resolve_offline_cache_path(path)
        if not cache_path.exists():
            raise FileNotFoundError(f"Offline transition cache not found: {cache_path}")
        metadata_path = cache_path.parent / "cache_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("exec_chunk_source") != "demonstrated_action":
                raise ValueError(
                    f"Offline cache {cache_path} does not contain demonstrated exec_chunk data"
                )
            if metadata.get("build_complete") is not True:
                raise ValueError(
                    f"Offline cache {cache_path} is incomplete or predates the build "
                    "completion marker. Let evo-rlt-build-transition-cache-v2 finish "
                    "successfully and write build_complete=true before starting online "
                    "training."
                )
            if metadata.get("actor_supervision_schema_version") != 1:
                raise ValueError(
                    f"Offline cache {cache_path} does not mark successful demonstrations "
                    "for direct Actor supervision. Rebuild it with the current "
                    "evo-rlt-build-transition-cache-v2; otherwise offline data would "
                    "train only the Critic."
                )
            cached_arms = metadata.get("rl_action_arms")
            policy_arms = getattr(policy_cfg, "actor_rl_arm", "both")
            if cached_arms is not None and cached_arms != policy_arms:
                raise ValueError(
                    f"Offline cache was built for rl_action_arms={cached_arms!r}, "
                    f"but policy actor_rl_arm={policy_arms!r}"
                )
            for metadata_key, policy_key in (
                ("vla_pretrained_path", "vla_pretrained_path"),
                ("rl_token_policy_path", "rl_token_pretrained_path"),
            ):
                cached_checkpoint = metadata.get(metadata_key)
                current_checkpoint = getattr(policy_cfg, policy_key, None)
                if (
                    cached_checkpoint
                    and current_checkpoint
                    and cls._checkpoint_identity(cached_checkpoint)
                    != cls._checkpoint_identity(current_checkpoint)
                ):
                    raise ValueError(
                        f"Offline cache {metadata_key}={cached_checkpoint!r} does not "
                        f"match policy {policy_key}={current_checkpoint!r}; state_vec and "
                        "VLA references must be encoded by the deployed checkpoints"
                    )
            cls._check_reward_schema(metadata, cache_path, online_rl_cfg)
        else:
            raise ValueError(
                f"Offline cache {cache_path} has no cache_metadata.json. It may be an "
                "old v2 cache with exec_chunk == ref_chunk; rebuild it with the "
                "current evo-rlt-build-transition-cache-v2."
            )
        raw_transitions = torch.load(cache_path, map_location="cpu", weights_only=False)
        if not isinstance(raw_transitions, list) or not raw_transitions:
            raise ValueError(f"Offline transition cache is empty or invalid: {cache_path}")

        buffer = ReplayBuffer(capacity=len(raw_transitions))
        required = {
            "state_vec", "exec_chunk", "ref_chunk", "reward_seq",
            "next_state_vec", "next_ref_chunk", "done", "intervention",
            "actual_steps",
        }
        for index, raw in enumerate(raw_transitions):
            if isinstance(raw, ChunkTransition):
                transition = raw
            elif isinstance(raw, dict):
                missing = required - raw.keys()
                if missing:
                    raise ValueError(
                        f"Offline transition {index} in {cache_path} is missing {sorted(missing)}"
                    )
                transition = ChunkTransition(
                    **{
                        key: value
                        for key, value in raw.items()
                        if key in ChunkTransition.__dataclass_fields__
                    }
                )
            else:
                raise TypeError(
                    f"Offline transition {index} has unsupported type {type(raw).__name__}"
                )
            if transition.exec_chunk.shape != (
                policy_cfg.chunk_length,
                policy_cfg.action_dim,
            ):
                raise ValueError(
                    f"Offline transition {index} exec_chunk shape "
                    f"{tuple(transition.exec_chunk.shape)} != "
                    f"({policy_cfg.chunk_length}, {policy_cfg.action_dim})"
                )
            outcome = getattr(transition, "outcome", None)
            if outcome is None or float(outcome.item()) < 0.0:
                raise ValueError(
                    f"Offline transition {index} in {cache_path} has no resolved outcome "
                    "for gating Actor demonstration loss"
                )
            supervision_mask = getattr(transition, "intervention_mask", None)
            if (
                supervision_mask is None
                or supervision_mask.shape != transition.exec_chunk.shape
            ):
                raise ValueError(
                    f"Offline transition {index} in {cache_path} has no valid "
                    "per-element Actor supervision mask"
                )
            if float(outcome.item()) >= 0.5 and not bool(supervision_mask.any().item()):
                raise ValueError(
                    f"Successful offline transition {index} in {cache_path} has an empty "
                    "Actor supervision mask"
                )
            buffer.add(transition)

        logging.info(
            "Loaded fixed offline replay: %d transitions from %s",
            len(buffer),
            cache_path,
        )
        return buffer

    @staticmethod
    def _concat_replay_batches(
        first: dict[str, torch.Tensor],
        second: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if first.keys() != second.keys():
            raise ValueError("Offline and online replay batches have different fields")
        return {key: torch.cat([first[key], second[key]], dim=0) for key in first}

    def _split_batch_sizes(self) -> tuple[int, int]:
        """(online_n, offline_n): how a full training batch splits between
        the online replay buffer and the fixed offline demo buffer (all
        online, 0 offline, if no offline buffer is configured). Shared by
        _sample_training_batch() and maybe_update()'s "is there enough data
        yet" gate so the two can't drift apart -- that gate must check the
        actual online_n needed, not the full batch_size, or it needlessly
        delays the first update whenever an offline buffer is configured to
        cover part of the batch (e.g. offline_batch_fraction=0.5 only ever
        needs batch_size/2 online transitions, not a full batch_size worth).
        """
        batch_size = self.cfg.batch_size
        if self.offline_buffer is None:
            return batch_size, 0
        offline_n = round(batch_size * self.cfg.offline_batch_fraction)
        offline_n = min(max(offline_n, 0), batch_size - 1)
        return batch_size - offline_n, offline_n

    def _sample_training_batch(self) -> tuple[dict[str, torch.Tensor], int, int]:
        """Sample a controlled offline/online mixture.

        Offline demonstrations are fixed and uniformly sampled. Online data
        keeps its outcome/intervention/recent stratification. Both sources
        contribute ordinary TD samples and retain their real outcome for
        demo BC, while rankq_outcome marks only online rows as RankQ-eligible.
        The warmup and UTD budget remain based solely on online experience.
        """
        online_n, offline_n = self._split_batch_sizes()

        if self.cfg.use_stratified_sampling:
            online = self.replay_buffer.sample_stratified(online_n)
        else:
            online = self.replay_buffer.sample(online_n)
        # Fresh transitions carry their own resolved critical-attempt outcome.
        # Fall back to episode-id lookup only for legacy replay entries.
        resolved_online = self.replay_buffer.outcome_labels(online["episode_id"])
        online["outcome"] = torch.where(
            online["outcome"] >= 0.0,
            online["outcome"],
            resolved_online,
        )
        # RankQ should learn its local action ordering from real online
        # outcomes. Keep this separate from ``outcome`` because the latter is
        # also the trust gate for successful offline demonstration BC.
        online["rankq_outcome"] = online["outcome"].clone()
        if offline_n == 0:
            actual_online_n = next(iter(online.values())).shape[0]
            return online, 0, actual_online_n
        assert self.offline_buffer is not None
        offline = self.offline_buffer.sample(offline_n)
        # Offline demonstrations still contribute TD targets and direct Actor
        # BC, but do not create RankQ pairs. ``-1`` is RankQ's existing
        # unresolved/ignored label.
        offline["rankq_outcome"] = torch.full_like(offline["outcome"], -1.0)
        actual_offline_n = next(iter(offline.values())).shape[0]
        actual_online_n = next(iter(online.values())).shape[0]
        return (
            self._concat_replay_batches(offline, online),
            actual_offline_n,
            actual_online_n,
        )

    def _actor_bc_step(
        self,
        raw: dict[str, torch.Tensor],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """One Critic-free Actor BC step on a replay-format batch."""
        batch = {
            "state_vec": raw["state_vec"].to(device),
            "ref_chunk_flat": raw["ref_chunk_flat"].to(device),
            "exec_chunk_flat": raw["exec_chunk_flat"].to(device),
            "actual_steps": raw["actual_steps"].to(device),
            "outcome": raw["outcome"].to(device),
            "intervention_mask_flat": raw["intervention_mask_flat"].to(device),
        }
        info: dict[str, torch.Tensor] = {}
        loss = actor_behavior_cloning_loss(
            self.policy.actor,
            batch,
            beta=self.policy_cfg.beta,
            demo_bc_weight=getattr(self.policy_cfg, "actor_demo_bc_weight", 1.0),
            action_clip_delta=getattr(self.policy_cfg, "actor_action_clip_delta", None),
            chunk_length=self.policy_cfg.chunk_length,
            info=info,
        )
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("Non-finite Actor BC warm-start loss")
        self.actor_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.actor.parameters(), max_norm=1.0
        )
        if not bool(torch.isfinite(grad_norm).item()):
            raise FloatingPointError("Non-finite Actor BC warm-start gradient")
        self.actor_optimizer.step()
        return {
            "loss_actor": loss.detach(),
            "actor_grad_norm": grad_norm.detach(),
            **info,
        }

    def _run_offline_bc_warmstart(self, requested_updates: int) -> tuple[int, dict]:
        """Train Actor from trusted offline demos while online RL is warming up.

        Match the eventual delayed-Actor cadence: ``requested_updates`` is a
        critic-equivalent budget, and one BC step runs per configured Actor
        interval. No Critic/Q term is consulted in this phase.
        """
        if self.offline_buffer is None or requested_updates <= 0:
            return 0, {}
        interval = max(int(self.online_actor_update_interval), 1)
        num_bc_updates = (requested_updates + interval - 1) // interval
        device = next(self.policy.parameters()).device
        latest: dict[str, torch.Tensor] = {}
        self.policy.train()
        try:
            for _ in range(num_bc_updates):
                raw = self.offline_buffer.sample(self.cfg.batch_size)
                latest = self._actor_bc_step(raw, device)
        finally:
            self.policy.eval()
        return num_bc_updates, {
            key: value.item() if torch.is_tensor(value) else value
            for key, value in latest.items()
        }

    def _init_wandb(self, online_rl_cfg: Any, policy_cfg: Any) -> Any | None:
        if not getattr(online_rl_cfg, "wandb", False):
            return None
        try:
            import wandb
        except ImportError:
            logging.warning(
                "online_rl.wandb=true but the `wandb` package is not installed "
                "(pip install evo-rlt[wandb]) -- continuing without wandb logging."
            )
            return None
        return wandb.init(
            project=online_rl_cfg.wandb_project,
            entity=online_rl_cfg.wandb_entity,
            name=online_rl_cfg.wandb_run_name,
            id=online_rl_cfg.wandb_run_id,
            resume=online_rl_cfg.wandb_resume,
            config={
                "warmup_episodes": online_rl_cfg.warmup_episodes,
                "critic_only_episodes": online_rl_cfg.critic_only_episodes,
                "actor_unfreeze_ramp_episodes": online_rl_cfg.actor_unfreeze_ramp_episodes,
                "min_warmup_transitions": online_rl_cfg.min_warmup_transitions,
                "min_warmup_successes": online_rl_cfg.min_warmup_successes,
                "min_warmup_failures": online_rl_cfg.min_warmup_failures,
                "terminal_reward": online_rl_cfg.terminal_reward,
                "milestone_reward": online_rl_cfg.milestone_reward,
                "time_decay": online_rl_cfg.time_decay,
                "replay_capacity": online_rl_cfg.replay_capacity,
                "batch_size": online_rl_cfg.batch_size,
                "offline_cache_path": online_rl_cfg.offline_cache_path,
                "offline_batch_fraction": online_rl_cfg.offline_batch_fraction,
                "actor_demo_bc_weight": getattr(policy_cfg, "actor_demo_bc_weight", 1.0),
                "offline_buffer_transitions": (
                    len(self.offline_buffer) if self.offline_buffer is not None else 0
                ),
                "lr_actor": online_rl_cfg.lr_actor,
                "lr_critic": online_rl_cfg.lr_critic,
                "utd_ratio": policy_cfg.utd_ratio,
                "max_updates_per_episode": online_rl_cfg.max_updates_per_episode,
                "use_stratified_sampling": online_rl_cfg.use_stratified_sampling,
                "chunk_length": policy_cfg.chunk_length,
                "action_dim": policy_cfg.action_dim,
                "actor_hidden_dim": policy_cfg.actor_hidden_dim,
                "actor_layer_norm": policy_cfg.actor_layer_norm,
                "critic_hidden_dim": policy_cfg.critic_hidden_dim,
                "critic_layer_norm": policy_cfg.critic_layer_norm,
                "rankq_alpha_success": getattr(policy_cfg, "rankq_alpha_success", None),
                "rankq_alpha_failure": getattr(policy_cfg, "rankq_alpha_failure", None),
                "rankq_noise_scale": getattr(policy_cfg, "rankq_noise_scale", None),
                "rankq_margin": getattr(policy_cfg, "rankq_margin", None),
                "target_noise_std": getattr(policy_cfg, "target_noise_std", None),
                "target_noise_clip": getattr(policy_cfg, "target_noise_clip", None),
                "actor_action_clip_delta": getattr(policy_cfg, "actor_action_clip_delta", None),
                "actor_slew_rate_limit": getattr(policy_cfg, "actor_slew_rate_limit", None),
                "actor_smoothness_weight": getattr(policy_cfg, "actor_smoothness_weight", None),
            },
            settings=wandb.Settings(save_code=False),
        )

    def _log(self, data: dict[str, Any], step: int) -> None:
        if self.wandb_run is None:
            return
        self.wandb_run.log(data, step=step)

    def _autonomous_success_metrics(self, rolling_window: int = 20) -> dict[str, float | int]:
        """Summarize success without counting human-rescued episodes as autonomous.

        An episode is considered intervened when any of its replay chunks has
        ``intervention=1``. This follows the collector's existing chunk-level
        dominant-source annotation. ``autonomous_success_rate`` uses only
        autonomous episodes as its denominator.
        """
        episodes: dict[int, dict[str, bool | float]] = {}
        for transition in self.replay_buffer.buffer:
            episode_id = int(transition.episode_id.item())
            episode = episodes.setdefault(
                episode_id,
                {
                    "done": False,
                    "success": False,
                    "intervention": False,
                    "reward": 0.0,
                },
            )
            episode["intervention"] |= bool(transition.intervention.item())
            episode["reward"] += float(transition.reward_seq.sum().item())
            if float(transition.done.item()) == 1.0:
                episode["done"] = True
                explicit = getattr(transition, "outcome", None)
                explicit_value = float(explicit.item()) if explicit is not None else -1.0
                episode["success"] = (
                    explicit_value >= 0.5
                    if explicit_value >= 0.0
                    else float(transition.reward_seq.sum().item()) > 0.0
                )

        labeled = [episodes[eid] for eid in sorted(episodes) if episodes[eid]["done"]]
        autonomous = [episode for episode in labeled if not episode["intervention"]]
        autonomous_successes = sum(episode["success"] for episode in autonomous)
        intervened = sum(episode["intervention"] for episode in labeled)

        recent_labeled = labeled[-rolling_window:]
        recent_autonomous = [
            episode for episode in recent_labeled if not episode["intervention"]
        ]
        recent_autonomous_successes = sum(
            episode["success"] for episode in recent_autonomous
        )

        labeled_count = len(labeled)
        autonomous_count = len(autonomous)
        recent_count = len(recent_labeled)
        recent_autonomous_count = len(recent_autonomous)
        latest = labeled[-1] if labeled else None
        return {
            "online_rl/episode_reward": float(latest["reward"]) if latest else 0.0,
            "online_rl/episode_intervened": float(latest["intervention"]) if latest else 0.0,
            "online_rl/episode_autonomous_success": (
                float(bool(latest["success"]) and not bool(latest["intervention"]))
                if latest
                else 0.0
            ),
            "online_rl/autonomous_episodes": autonomous_count,
            "online_rl/autonomous_successes": autonomous_successes,
            "online_rl/autonomous_success_rate": (
                autonomous_successes / autonomous_count if autonomous_count else 0.0
            ),
            f"online_rl/autonomous_success_rate_rolling_{rolling_window}": (
                recent_autonomous_successes / recent_autonomous_count
                if recent_autonomous_count
                else 0.0
            ),
            f"online_rl/autonomous_episodes_rolling_{rolling_window}": recent_autonomous_count,
            "online_rl/intervention_rate": intervened / labeled_count if labeled_count else 0.0,
            f"online_rl/intervention_rate_rolling_{rolling_window}": (
                sum(episode["intervention"] for episode in recent_labeled) / recent_count
                if recent_count
                else 0.0
            ),
        }

    def _intervention_correction_metrics(self) -> dict[str, float]:
        """How far human interventions actually move the executed action
        away from ref (the frozen VLA reference) -- i.e. the correction
        magnitude the residual actor (mu = ref + delta) would need to
        reproduce on its own to close the same gap autonomously. Compared
        against actor_action_clip_delta, the hard bound on how far the
        actor's own output may ever deviate from ref: if real corrections
        routinely exceed that bound, no amount of training closes the gap,
        since the actor is structurally incapable of ever reaching that
        action regardless of how good the learning signal is otherwise.
        Scans the whole buffer, same as buffer_successes/buffer_failures
        (ReplayBuffer.count_outcomes()) -- cheap relative to the gradient
        updates this runs alongside.
        """
        deltas = []
        for transition in self.replay_buffer.buffer:
            mask = getattr(transition, "intervention_mask", None)
            if mask is None or mask.numel() == 0:
                # Legacy cache/checkpoint compatibility. Fresh collectors
                # always persist the exact per-element teaching mask.
                if transition.intervention is None or transition.intervention.item() != 1.0:
                    continue
                mask = torch.ones_like(transition.exec_chunk)
            if not bool(mask.any().item()):
                continue
            valid_steps = int(transition.actual_steps.item())
            valid_mask = mask[:valid_steps].bool()
            correction = (transition.exec_chunk - transition.ref_chunk).abs()
            deltas.append(correction[:valid_steps][valid_mask].max().item())
        if not deltas:
            return {
                "online_rl/intervention_correction_max_mean": 0.0,
                "online_rl/intervention_correction_frac_exceeds_clip": 0.0,
            }
        action_clip_delta = getattr(self.policy_cfg, "actor_action_clip_delta", None)
        frac_exceeds = (
            sum(d > action_clip_delta for d in deltas) / len(deltas)
            if action_clip_delta is not None and action_clip_delta > 0
            else 0.0
        )
        return {
            "online_rl/intervention_correction_max_mean": sum(deltas) / len(deltas),
            "online_rl/intervention_correction_frac_exceeds_clip": frac_exceeds,
        }

    @staticmethod
    def _buffer_return_mean(transitions, gamma: float) -> float:
        """Mean *discounted* per-episode return, measured from each episode's
        first step -- the quantity Q(s0, .) is defined to predict.

        Discounting matters more here than it looks. gamma applies per
        PHYSICAL step, not per chunk: within a chunk via
        losses.discounted_chunk_return's gamma^i, and across chunks via
        critic_loss's gamma^actual_steps bootstrap. At ~783 steps per episode
        an undiscounted sum would overstate the target by 1.5x at
        gamma=0.9995 and by ~2600x at gamma=0.99, which would make the ratio
        read as pathological overestimation purely as an artifact of the
        discount setting.
        """
        per_episode: dict[int, float] = {}
        elapsed: dict[int, int] = {}
        # Buffer order is collection order, so the running step count per
        # episode is the transition's true offset from that episode's start.
        for transition in transitions:
            episode_id = int(transition.episode_id.item())
            steps = int(transition.actual_steps.item())
            offset = elapsed.get(episode_id, 0)
            rewards = transition.reward_seq[:steps]
            if rewards.numel():
                discounts = gamma ** (
                    offset + torch.arange(rewards.numel(), dtype=torch.float32)
                )
                per_episode[episode_id] = per_episode.get(episode_id, 0.0) + float(
                    (rewards.to(torch.float32) * discounts).sum().item()
                )
            elapsed[episode_id] = offset + steps
        if not per_episode:
            return 0.0
        return sum(per_episode.values()) / len(per_episode)

    def _empirical_return_mean(self) -> float:
        """Return yardstick for q_vs_return_ratio, matched to the batch mix.

        Q is measured on batches that are `offline_batch_fraction` demos and
        the rest online experience, so the denominator has to be the same
        blend. Using the online buffer alone silently biases the ratio by
        however far the two sources' returns differ -- measured on this
        project, online 0.374 vs offline 0.181, which understated the
        overestimate by 1.35x at a 50/50 mix.

        The online part is rescanned each call (it grows); the offline cache
        is fixed, so its mean is computed once and cached.

        Same whole-buffer scan as _intervention_correction_metrics(), and
        cheap for the same reason (once per episode, alongside hundreds of
        gradient steps).
        """
        gamma = float(self.policy_cfg.gamma)
        online_mean = self._buffer_return_mean(self.replay_buffer.buffer, gamma)
        if self.offline_buffer is None or not len(self.offline_buffer):
            return online_mean
        cached = getattr(self, "_offline_return_mean_cache", None)
        if cached is None or cached[0] != gamma:
            value = self._buffer_return_mean(self.offline_buffer.buffer, gamma)
            self._offline_return_mean_cache = (gamma, value)
            cached = self._offline_return_mean_cache
        offline_mean = cached[1]
        # Weight by the split maybe_update() actually samples, not by how many
        # transitions each buffer holds.
        online_n, offline_n = self._split_batch_sizes()
        total = online_n + offline_n
        if total <= 0:
            return online_mean
        return (online_n * online_mean + offline_n * offline_mean) / total

    def _q_calibration_metrics(
        self, raw: dict[str, torch.Tensor], device: torch.device
    ) -> dict[str, float]:
        """Two things plain loss_critic cannot show, both learned the hard way
        on this project (150 episodes of a flat 0.43 intervention rate before
        either was visible in hindsight):

        `q_vs_return_ratio` -- Q(s, ref) against the mean *discounted*
        episode return the buffer actually contains (see
        _empirical_return_mean: discounted, because that is what Q predicts).
        TD bootstrap bias accumulates silently: a run that looked converged
        by every logged loss had Q ~= 4.8 against a discounted mean return of
        0.374 -- a 12.8x overestimate that no logged loss revealed. It should
        sit near 1; a persistent climb means the critic is inventing value.

        `q_rank_margin` -- Q(s, human takeover action) minus Q(s, the action
        the actor would actually deploy), on intervened rows only. This is the
        one ordering the actor's whole learning signal depends on, and it is
        NOT implied by a small TD loss: measured at -0.337 on that same run,
        i.e. the critic preferred the actor's own action over the human
        correction that had just rescued the episode -- while human takeovers
        accounted for 94.9% of all successes and the actor's autonomous
        success rate was 3.5%. Negative here means actor gradients point the
        wrong way and no amount of further training helps.
        """
        chunk_length = self.policy_cfg.chunk_length
        clip_delta = getattr(self.policy_cfg, "actor_action_clip_delta", None)
        slew = getattr(self.policy_cfg, "actor_slew_rate_limit", None)
        state = raw["state_vec"].to(device)
        ref = raw["ref_chunk_flat"].to(device)
        exec_chunk = raw["exec_chunk_flat"].to(device)
        metrics: dict[str, float] = {}
        with torch.no_grad():
            q_ref = self.policy.critic.min_q(state, ref).mean().item()
            empirical_return = self._empirical_return_mean()
            metrics["online_rl/q_ref_mean"] = q_ref
            metrics["online_rl/empirical_return_mean"] = empirical_return
            metrics["online_rl/q_vs_return_ratio"] = (
                q_ref / empirical_return if abs(empirical_return) > 1e-6 else 0.0
            )

            intervened = raw.get("intervention_mask_flat")
            if intervened is None:
                return metrics
            rows = (intervened.to(device) > 0).any(dim=-1)
            if not bool(rows.any()):
                # No intervened rows in this batch -- report nothing rather
                # than a 0.0 that would read as "ordering is exactly neutral".
                return metrics
            # Reproduce the action that actually reaches the robot, matching
            # losses.actor_loss's construction -- ranking the human action
            # against an unconstrained mu the actor could never deploy would
            # measure the wrong gap.
            mu, _ = self.policy.actor.forward(state, ref, training=False)
            mu_deployed = ref + self.actor_deploy_scale * (mu - ref)
            mu_safe = project_action_delta(mu_deployed, ref, clip_delta)
            if clip_delta is None:
                mu_safe = mu_safe.clamp(-1.0, 1.0)
            mu_safe = _apply_slew_rate_limit_flat(mu_safe, ref, chunk_length, slew)
            q_human = self.policy.critic.min_q(state[rows], exec_chunk[rows])
            q_actor = self.policy.critic.min_q(state[rows], mu_safe[rows])
            metrics["online_rl/q_rank_margin"] = (q_human - q_actor).mean().item()
            metrics["online_rl/q_rank_correct_frac"] = (
                (q_human > q_actor).float().mean().item()
            )
        return metrics

    def close(self) -> None:
        if self.wandb_run is not None:
            self.wandb_run.finish()

    def start_episode(self, episode_id: int) -> int:
        """Call at the start of every rollout episode. Returns the replay
        buffer's `total_added` baseline to pass back into `maybe_update()`."""
        self.actor_deploy_scale = self._actor_deploy_scale_for(episode_id)
        self.policy.set_actor_deploy_scale(self.actor_deploy_scale)
        logging.info(
            "Online RL episode %d: actor_deploy_scale=%.3f",
            episode_id,
            self.actor_deploy_scale,
        )
        self.collector.start_episode(episode_id)
        return self.replay_buffer.total_added

    def _actor_deploy_scale_for(self, episode_id: int) -> float:
        """Residual fraction used by the upcoming physical rollout.

        Actor training is allowed in the background, but deployment remains
        exact VLA throughout warmup and critic-only. At the unfreeze boundary
        the first rollout is still scale 0; subsequent episodes linearly ramp
        to full Actor control over ``actor_unfreeze_ramp_episodes``.
        """
        if self.warmup_completed_at_episode is None:
            return 0.0
        critic_only_until = (
            self.warmup_completed_at_episode + self.cfg.critic_only_episodes
        )
        if episode_id <= critic_only_until:
            return 0.0
        ramp_episodes = self.cfg.actor_unfreeze_ramp_episodes
        if ramp_episodes <= 0:
            return 1.0
        return min((episode_id - critic_only_until) / ramp_episodes, 1.0)

    def discard_episode(self, buffer_total_added_before: int) -> None:
        """Roll back every transition this cycle's critical-phase attempt(s)
        already flushed into the replay buffer -- call this instead of
        maybe_update() when the whole episode is being rerecorded/discarded
        (left arrow), so a redo doesn't leave the discarded attempt's
        transitions (correctly labeled or not) permanently mixed into
        training data."""
        n_removed = self.replay_buffer.rollback(buffer_total_added_before)
        if n_removed:
            logging.info("Online RL: discarded %d transition(s) from rerecorded episode.", n_removed)

    def warmup_satisfied(self, recorded_episodes: int) -> bool:
        if self.warmup_completed_at_episode is not None:
            return True  # sticky: once satisfied, stays satisfied
        if recorded_episodes < self.cfg.warmup_episodes:
            return False
        if len(self.replay_buffer) < self.cfg.min_warmup_transitions:
            return False
        successes, failures = self.replay_buffer.count_outcomes()
        if not (successes >= self.cfg.min_warmup_successes and failures >= self.cfg.min_warmup_failures):
            return False
        # Record when warmup actually finished -- the critic-only window
        # below anchors to this, not a fixed offset (see its comment).
        self.warmup_completed_at_episode = recorded_episodes
        logging.info(
            "=" * 70 + "\nOnline RL: warmup satisfied after episode %d -- gradient updates "
            "start now (critic-only for the next %d episodes, then actor unfreezes).\n" + "=" * 70,
            recorded_episodes, self.cfg.critic_only_episodes,
        )
        return True

    def _actor_update_interval_for(self, recorded_episodes: int, critic_only_until: int) -> int:
        """actor_update_interval for this update cycle: still frozen (10**9)
        before critic_only_until, then ramped down to
        online_actor_update_interval over cfg.actor_unfreeze_ramp_episodes
        episodes instead of snapping straight to it -- see
        OnlineRLConfig.actor_unfreeze_ramp_episodes for why the hard flip
        this replaces is a direct contributor to actor jitter. The interval
        decreases by one online_actor_update_interval-sized step per
        episode, reaching the target exactly at the end of the ramp (no
        discontinuity at the boundary). ramp_episodes <= 0 reproduces the
        exact old hard-flip behavior.
        """
        if recorded_episodes < critic_only_until:
            return 10**9
        episodes_since_unfreeze = recorded_episodes - critic_only_until
        ramp_episodes = self.cfg.actor_unfreeze_ramp_episodes
        if ramp_episodes > 0 and episodes_since_unfreeze < ramp_episodes:
            multiplier = ramp_episodes - episodes_since_unfreeze
            return self.online_actor_update_interval * multiplier
        return self.online_actor_update_interval

    def maybe_update(self, recorded_episodes: int, buffer_total_added_before: int) -> dict | None:
        """Run gradient steps scaled to how much data this cycle actually
        added, in place on `policy` -- the very next rollout episode uses
        the updated weights directly, no checkpoint save/reload.

        The caller (or the collector's flush_episode()) is responsible for
        having already committed this cycle's transitions to the replay
        buffer -- this only checks whether that happened (via the
        total_added delta) and, if so, trains. Returns a stats dict on
        update (with `checkpoint_path` set if a periodic checkpoint was
        saved this call), or None if no update ran.
        """
        # total_added is monotonic (unlike len(), which stops growing once
        # the deque hits capacity and starts evicting) -- see ReplayBuffer.
        new_transitions = self.replay_buffer.total_added - buffer_total_added_before
        if new_transitions == 0:
            return None  # no critical-phase attempt was flushed this cycle

        # Use the same per-new-transition budget before and after warmup. If
        # warmup has not completed yet, this budget drives safe offline BC
        # instead of being discarded completely.
        requested_updates = max(new_transitions, 0) * self.policy_cfg.utd_ratio
        num_updates = min(requested_updates, self.cfg.max_updates_per_episode)

        if not self.warmup_satisfied(recorded_episodes):
            bc_updates, bc_info = self._run_offline_bc_warmstart(num_updates)
            successes, failures = self.replay_buffer.count_outcomes()
            logging.info(
                "Online RL warmup not yet satisfied after episode %d: "
                "transitions=%d (need %d), successes=%d (need %d), failures=%d (need %d), "
                "offline_bc_updates=%d",
                recorded_episodes, len(self.replay_buffer), self.cfg.min_warmup_transitions,
                successes, self.cfg.min_warmup_successes,
                failures, self.cfg.min_warmup_failures,
                bc_updates,
            )
            self._log(
                {
                    "online_rl/warmup_satisfied": 0,
                    "online_rl/actor_deploy_scale": self.actor_deploy_scale,
                    "online_rl/offline_bc_warmstart_updates": bc_updates,
                    "online_rl/buffer_transitions": len(self.replay_buffer),
                    "online_rl/buffer_successes": successes,
                    "online_rl/buffer_failures": failures,
                    **self._autonomous_success_metrics(),
                    **self._intervention_correction_metrics(),
                    **{f"online_rl/{key}": value for key, value in bc_info.items()},
                },
                step=recorded_episodes,
            )
            return None
        online_n, _offline_n = self._split_batch_sizes()
        if len(self.replay_buffer) < online_n:
            self._log(
                {**self._autonomous_success_metrics(), **self._intervention_correction_metrics()},
                step=recorded_episodes,
            )
            return None

        # Critic-only window: freeze actor updates so the critic gets a
        # chance to form a non-random value estimate before the actor
        # starts moving away from its safe, VLA-equivalent starting
        # point. Implemented by inflating actor_update_interval (read
        # live by ChunkACPolicy.forward()) rather than toggling
        # requires_grad, so no actor gradient is even computed. Anchored
        # to warmup_completed_at_episode (set by warmup_satisfied(), which
        # already returned True above) rather than a fixed
        # warmup_episodes + critic_only_episodes offset -- if warmup took
        # longer than warmup_episodes to actually satisfy its
        # transition/success/failure thresholds, the fixed-offset version
        # would already be in the past, skipping critic-only entirely.
        assert self.warmup_completed_at_episode is not None
        critic_only_until = self.warmup_completed_at_episode + self.cfg.critic_only_episodes
        # Captured separately from self.policy_cfg.actor_update_interval because
        # the finally block below restores that attribute to
        # online_actor_update_interval before this cycle's stats get logged --
        # without this local, online_rl/actor_update_interval would always
        # report the post-ramp target instead of what this cycle actually used.
        actor_update_interval_this_cycle = self._actor_update_interval_for(
            recorded_episodes, critic_only_until
        )
        self.policy_cfg.actor_update_interval = actor_update_interval_this_cycle
        # See the comment at online_tau's definition: disable forward()'s
        # own (mistimed) soft update for the duration of this call, then
        # do it correctly ourselves after each real critic.step() below.
        self.policy_cfg.tau = 0.0

        device = next(self.policy.parameters()).device
        self.policy.train()
        start_t = time.perf_counter()
        info = None
        # policy.forward()'s do_actor gate (critic_step % actor_update_interval)
        # fires on only some iterations of this loop, and `info` gets
        # overwritten every iteration -- so if the very last iteration isn't
        # a do_actor step, "loss_actor" silently disappears from the stats
        # below even though the actor genuinely was updated earlier in this
        # same loop. Remember the most recent do_actor iteration's info
        # separately so its loss_actor survives to the log/wandb output.
        last_actor_info = None
        last_actor_grad_norm = None
        last_critic_grad_norm = None
        try:
            for _ in range(num_updates):
                # ReplayBuffer.sample()/sample_stratified() return core/losses.py's
                # flat-key format (exec_chunk_flat/ref_chunk_flat/next_ref_flat);
                # ChunkACPolicy's forward()/_coerce_batch expects unflattened
                # (B, C, action_dim) exec_chunk/ref_chunk/next_ref_chunk and
                # flattens internally.
                raw, offline_batch_size, online_batch_size = self._sample_training_batch()
                chunk_length = self.policy_cfg.chunk_length
                batch = {
                    "state_vec": raw["state_vec"],
                    "exec_chunk": unflatten_chunk(raw["exec_chunk_flat"], chunk_length),
                    "ref_chunk": unflatten_chunk(raw["ref_chunk_flat"], chunk_length),
                    "reward_seq": raw["reward_seq"],
                    "next_state_vec": raw["next_state_vec"],
                    "next_ref_chunk": unflatten_chunk(raw["next_ref_flat"], chunk_length),
                    "done": raw["done"],
                    "actual_steps": raw["actual_steps"],
                    "outcome": raw["outcome"],
                    "rankq_outcome": raw["rankq_outcome"],
                    "intervention_mask_flat": raw["intervention_mask_flat"],
                }
                batch = {k: v.to(device) for k, v in batch.items()}
                loss, info = self.policy.forward(batch)
                if "loss_actor" in info:
                    last_actor_info = info
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError("Non-finite online Actor/Critic loss")
                loss.backward()
                # Matches ChunkACPolicyConfig.get_optimizer_preset()'s
                # grad_clip_norm=1.0 (applied automatically for offline
                # training by lerobot's generic train loop; this custom
                # online loop bypasses that loop entirely, so it has to be
                # applied explicitly here to get the same protection).
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.actor.parameters(), max_norm=1.0
                )
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.critic.parameters(), max_norm=1.0
                )
                if not bool(torch.isfinite(critic_grad_norm).item()):
                    raise FloatingPointError("Non-finite Critic gradient")
                if "loss_actor" in info and not bool(torch.isfinite(actor_grad_norm).item()):
                    raise FloatingPointError("Non-finite Actor gradient")
                self.critic_optimizer.step()
                self.actor_optimizer.step()
                last_critic_grad_norm = critic_grad_norm.detach()
                if "loss_actor" in info:
                    last_actor_grad_norm = actor_grad_norm.detach()

                # During the nominal critic-only window, freeze only the
                # untrusted -Q policy update. Continue safe supervised Actor
                # learning from successful offline/human demonstrations at
                # the normal delayed-Actor cadence.
                if (
                    recorded_episodes < critic_only_until
                    and int(self.policy._critic_step.item())
                    % max(self.online_actor_update_interval, 1)
                    == 0
                ):
                    last_actor_info = self._actor_bc_step(raw, device)
                    last_actor_grad_norm = last_actor_info["actor_grad_norm"]
                # Correct TD3 order: soft-update target only now, after
                # critic.step() has actually applied this iteration's gradient.
                soft_update(self.policy.target_critic, self.policy.critic, self.online_tau)
        finally:
            # Always restore, even on exception (CUDA OOM, bad sample, NaN
            # loss, ...): policy_cfg is the same object make_policy() gave
            # the live ChunkACPolicy, so a corrupted tau=0 /
            # actor_update_interval=10**9 left behind by an unhandled
            # exception would silently disable target updates / actor
            # training for the rest of the session (and leak into
            # policy.save_pretrained()'s config.json below).
            self.policy_cfg.tau = self.online_tau
            self.policy_cfg.actor_update_interval = self.online_actor_update_interval
            self.policy.eval()
        training_time_s = time.perf_counter() - start_t

        loss_dict = {k: (v.item() if torch.is_tensor(v) else v) for k, v in (info or {}).items()}
        q_calibration: dict[str, float] = {}
        if num_updates > 0:
            # Diagnostic (see losses.q_action_sensitivity): computed once per
            # update cycle, not every gradient step, on the last sampled
            # batch -- cheap enough for periodic logging, and catches the
            # critic collapsing into an action-insensitive V(s)-like function
            # (low TD loss but no useful signal for the actor) that plain
            # loss_critic alone doesn't reveal.
            loss_dict["q_action_sensitivity"] = q_action_sensitivity(
                self.policy.critic,
                raw["state_vec"].to(device),
                raw["exec_chunk_flat"].to(device),
                action_mask=self.policy.actor.action_mask,
            ).item()
            q_calibration = self._q_calibration_metrics(raw, device)
        if last_actor_info is not None and "loss_actor" not in loss_dict:
            # The last iteration of this call didn't happen to be a do_actor
            # step, but an earlier one in the same call was -- surface that
            # actor loss instead of silently dropping it (see the comment
            # where last_actor_info is set, above).
            loss_dict["loss_actor"] = (
                last_actor_info["loss_actor"].item()
                if torch.is_tensor(last_actor_info["loss_actor"])
                else last_actor_info["loss_actor"]
            )
            for key in (
                "loss_actor_q",
                "loss_actor_vla_bc",
                "loss_actor_demo_bc",
                "loss_actor_smoothness",
                "loss_actor_bc_only",
            ):
                if key in last_actor_info:
                    value = last_actor_info[key]
                    loss_dict[key] = value.item() if torch.is_tensor(value) else value
        if last_actor_grad_norm is not None:
            loss_dict["actor_grad_norm"] = (
                last_actor_grad_norm.item()
                if torch.is_tensor(last_actor_grad_norm)
                else last_actor_grad_norm
            )
        if last_critic_grad_norm is not None:
            loss_dict["critic_grad_norm"] = last_critic_grad_norm.item()

        logging.info(
            "Online RL update after episode %d: new_transitions=%d requested_updates=%d "
            "actual_updates=%d effective_utd=%.2f batch_mix=%d_offline/%d_online "
            "training_time=%.1fs loss=%s",
            recorded_episodes, new_transitions, requested_updates, num_updates,
            (num_updates / new_transitions) if new_transitions > 0 else 0.0,
            offline_batch_size, online_batch_size,
            training_time_s,
            loss_dict,
        )
        if q_calibration:
            # Surfaced on its own line, not folded into loss=...: these two
            # are the health check that decides whether the losses above mean
            # anything (see _q_calibration_metrics).
            logging.info(
                "Online RL critic health: Q(ref)=%.3f vs empirical_return=%.3f "
                "(ratio=%.2f, want ~1) | q_rank_margin=%s (want > 0)",
                q_calibration.get("online_rl/q_ref_mean", float("nan")),
                q_calibration.get("online_rl/empirical_return_mean", float("nan")),
                q_calibration.get("online_rl/q_vs_return_ratio", float("nan")),
                (
                    f"{q_calibration['online_rl/q_rank_margin']:+.3f}"
                    if "online_rl/q_rank_margin" in q_calibration
                    else "n/a (no intervened rows in batch)"
                ),
            )
        stats: dict[str, Any] = {
            "new_transitions": new_transitions,
            "requested_updates": requested_updates,
            "actual_updates": num_updates,
            "training_time_s": training_time_s,
            "offline_batch_size": offline_batch_size,
            "online_batch_size": online_batch_size,
            "actor_deploy_scale": self.actor_deploy_scale,
            "loss": loss_dict,
            "checkpoint_path": None,
        }
        successes, failures = self.replay_buffer.count_outcomes()
        self._log(
            {
                "online_rl/warmup_satisfied": 1,
                "online_rl/critic_only": float(recorded_episodes < critic_only_until),
                "online_rl/actor_deploy_scale": self.actor_deploy_scale,
                "online_rl/actor_update_interval": actor_update_interval_this_cycle,
                "online_rl/new_transitions": new_transitions,
                "online_rl/actual_updates": num_updates,
                "online_rl/effective_utd": (num_updates / new_transitions) if new_transitions > 0 else 0.0,
                "online_rl/training_time_s": training_time_s,
                "online_rl/buffer_transitions": len(self.replay_buffer),
                "online_rl/offline_buffer_transitions": (
                    len(self.offline_buffer) if self.offline_buffer is not None else 0
                ),
                "online_rl/offline_batch_size": offline_batch_size,
                "online_rl/online_batch_size": online_batch_size,
                "online_rl/buffer_successes": successes,
                "online_rl/buffer_failures": failures,
                **self._autonomous_success_metrics(),
                **self._intervention_correction_metrics(),
                **q_calibration,
                **{f"online_rl/{k}": v for k, v in loss_dict.items() if k != "critic_step"},
            },
            step=recorded_episodes,
        )
        completed_episodes = recorded_episodes + 1
        if completed_episodes % self.cfg.save_every_episodes == 0:
            save_path = Path(self.cfg.save_dir) / f"step_{completed_episodes:06d}"
            self.policy.save_pretrained(save_path)
            logging.info("Online RL checkpoint saved to %s", save_path)
            stats["checkpoint_path"] = str(save_path)
        return stats

    def save_latest_state(self, completed_episodes: int) -> None:
        """Overwrite a single, internally-consistent 'latest' snapshot every
        episode -- model weights AND optimizer state AND replay buffer AND
        counters together, not just the optimizer/buffer/counters. The
        periodic step_NNNNNN saves above (policy.save_pretrained(), every
        `save_every_episodes`) run on a DIFFERENT cadence than this
        per-episode save; if this file held only optimizer/buffer/counters,
        a crash between two step_NNNNNN saves would leave weights from
        episode N sitting next to optimizer/buffer state from episode N+2,
        which can't be recombined into a valid training state. Written to a
        temp file then atomically renamed so a crash mid-write never
        leaves a half-written file behind.

        See load_latest_state() for the corresponding resume path.
        """
        save_dir = Path(self.cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "recorded_episodes": completed_episodes,
            "actor_state_dict": self.policy.actor.state_dict(),
            "critic_state_dict": self.policy.critic.state_dict(),
            "target_critic_state_dict": self.policy.target_critic.state_dict(),
            "critic_step": self.policy._critic_step,
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "replay_buffer": list(self.replay_buffer.buffer),
            "replay_buffer_total_added": self.replay_buffer.total_added,
            "replay_capacity": self.replay_buffer.capacity,
            "offline_cache_path": (
                str(self._resolve_offline_cache_path(self.cfg.offline_cache_path).resolve())
                if self.cfg.offline_cache_path is not None
                else None
            ),
            "offline_batch_fraction": self.cfg.offline_batch_fraction,
            "warmup_completed_at_episode": self.warmup_completed_at_episode,
            "torch_rng_state": torch.get_rng_state(),
        }
        final_path = save_dir / "latest_online_state.pt"
        tmp_path = save_dir / "latest_online_state.pt.tmp"
        torch.save(state, tmp_path)
        os.replace(tmp_path, final_path)
        archive_path = None
        if completed_episodes % self.cfg.save_every_episodes == 0:
            # Keep a selectable, internally-consistent training snapshot next
            # to the inference-only policy checkpoint. A hard link avoids
            # writing the often-large replay buffer twice; later replacement
            # of latest_online_state.pt does not modify this historical inode.
            archive_path = save_dir / f"step_{completed_episodes:06d}" / "online_state.pt"
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            archive_tmp = archive_path.with_suffix(".pt.tmp")
            try:
                archive_tmp.unlink(missing_ok=True)
                os.link(final_path, archive_tmp)
            except OSError:
                # Some filesystems do not support hard links. Preserve the
                # same interface with a regular copy in that case.
                shutil.copy2(final_path, archive_tmp)
            os.replace(archive_tmp, archive_path)
        logging.info(
            "Online RL state saved to %s%s (episodes=%d, buffer=%d transitions)",
            final_path,
            f" and selectable snapshot {archive_path}" if archive_path is not None else "",
            completed_episodes,
            len(self.replay_buffer),
        )

    def load_latest_state(self, path: str | Path) -> int:
        """Restore a snapshot written by save_latest_state(): model weights,
        optimizer momentum, the full replay buffer, and the warmup/critic-only
        anchor -- so training resumes exactly where it left off instead of
        re-entering warmup (or re-freezing the actor for another
        critic_only_episodes window) from the resume point. Does NOT restore
        torch_rng_state onto CUDA generators if the snapshot was saved on a
        different device configuration; this only affects exploration noise
        reproducibility, not correctness.

        Returns the recorded_episodes count to resume the outer episode loop
        from.
        """
        path = Path(path)
        # map_location="cpu", NOT the policy's device: the replay buffer's
        # ChunkTransition tensors must stay on CPU, matching the invariant
        # every other write path relies on (_emit_transition() always calls
        # .cpu() before storing; maybe_update() is what moves a *batch* to
        # device, right before the forward pass). Loading straight onto the
        # policy's device would leave resumed transitions on GPU while
        # newly-collected ones are CPU, and ReplayBuffer._collate()'s
        # torch.stack() crashes the moment a sampled batch mixes both.
        # actor_state_dict/critic_state_dict/optimizer state loaded below are
        # unaffected -- load_state_dict() copies values onto the existing
        # (already correctly-placed) parameters regardless of the source
        # tensors' device.
        #
        # weights_only=False: this snapshot embeds ChunkTransition dataclass
        # instances (the replay buffer) and optimizer state, not just plain
        # tensors -- torch>=2.6 defaults weights_only=True and refuses to
        # unpickle those. Safe here since this is our own save_latest_state()
        # output, not a third-party checkpoint.
        state = torch.load(path, map_location="cpu", weights_only=False)
        saved_offline_path = state.get("offline_cache_path")
        current_offline_path = (
            str(self._resolve_offline_cache_path(self.cfg.offline_cache_path).resolve())
            if self.cfg.offline_cache_path is not None
            else None
        )
        if saved_offline_path != current_offline_path:
            raise ValueError(
                "Resumed online state used a different offline cache: "
                f"saved={saved_offline_path!r}, current={self.cfg.offline_cache_path!r}. "
                "Pass the same --offline-cache-path used by the original run."
            )
        saved_offline_fraction = state.get("offline_batch_fraction")
        if (
            saved_offline_fraction is not None
            and saved_offline_fraction != self.cfg.offline_batch_fraction
        ):
            logging.warning(
                "Changing offline_batch_fraction across resume: saved=%.3f current=%.3f",
                saved_offline_fraction,
                self.cfg.offline_batch_fraction,
            )
        self.policy.actor.load_state_dict(state["actor_state_dict"])
        self.policy.critic.load_state_dict(state["critic_state_dict"])
        self.policy.target_critic.load_state_dict(state["target_critic_state_dict"])
        self.policy._critic_step = state["critic_step"]
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.replay_buffer.buffer = deque(state["replay_buffer"], maxlen=self.replay_buffer.capacity)
        self.replay_buffer.total_added = state["replay_buffer_total_added"]
        self.warmup_completed_at_episode = state.get("warmup_completed_at_episode")
        if "torch_rng_state" in state:
            torch.set_rng_state(state["torch_rng_state"].cpu())
        recorded_episodes = state["recorded_episodes"]
        logging.info(
            "Online RL resumed from %s (episodes=%d, buffer=%d transitions, warmup_completed_at_episode=%s)",
            path, recorded_episodes, len(self.replay_buffer), self.warmup_completed_at_episode,
        )
        return recorded_episodes
