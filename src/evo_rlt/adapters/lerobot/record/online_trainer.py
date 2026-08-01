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
import os
import shutil
import time
from collections import deque
from pathlib import Path
from typing import Any

import torch

from evo_rlt.adapters.lerobot.online_collector import RLTOnlineCollector
from evo_rlt.core.interfaces import ChunkTransition
from evo_rlt.core.replay_buffer import ReplayBuffer
from evo_rlt.core.utils import soft_update, unflatten_chunk


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
        )
        self.collector = RLTOnlineCollector(
            replay_buffer=self.replay_buffer,
            chunk_length=policy_cfg.chunk_length,
            action_dim=policy_cfg.action_dim,
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

    @classmethod
    def _load_offline_buffer(
        cls,
        path: str | None,
        *,
        policy_cfg: Any,
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

    def _sample_training_batch(self) -> tuple[dict[str, torch.Tensor], int, int]:
        """Sample a controlled offline/online mixture.

        Offline demonstrations are fixed and uniformly sampled. Online data
        keeps its outcome/intervention/recent stratification. The warmup and
        UTD budget remain based solely on online experience.
        """
        batch_size = self.cfg.batch_size
        if self.offline_buffer is None:
            online_n = batch_size
            offline_n = 0
        else:
            offline_n = round(batch_size * self.cfg.offline_batch_fraction)
            offline_n = min(max(offline_n, 0), batch_size - 1)
            online_n = batch_size - offline_n

        if self.cfg.use_stratified_sampling:
            online = self.replay_buffer.sample_stratified(online_n)
        else:
            online = self.replay_buffer.sample(online_n)
        # RankQ ranking loss (see losses.rankq_ranking_loss): tag each online
        # transition with its trajectory's resolved success/failure outcome.
        online["outcome"] = self.replay_buffer.outcome_labels(online["episode_id"])
        if offline_n == 0:
            actual_online_n = next(iter(online.values())).shape[0]
            return online, 0, actual_online_n
        assert self.offline_buffer is not None
        offline = self.offline_buffer.sample(offline_n)
        # The offline cache is contractually "demonstrated_action" only (see
        # _load_offline_buffer's exec_chunk_source check), so every demo
        # transition counts as a resolved success -- its episode_id is not
        # reliable for episode_outcomes() lookup (often left at the default
        # -1 sentinel by the cache builder).
        offline["outcome"] = torch.ones(offline["state_vec"].shape[0])
        actual_offline_n = next(iter(offline.values())).shape[0]
        actual_online_n = next(iter(online.values())).shape[0]
        return (
            self._concat_replay_batches(offline, online),
            actual_offline_n,
            actual_online_n,
        )

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
                "min_warmup_transitions": online_rl_cfg.min_warmup_transitions,
                "min_warmup_successes": online_rl_cfg.min_warmup_successes,
                "min_warmup_failures": online_rl_cfg.min_warmup_failures,
                "replay_capacity": online_rl_cfg.replay_capacity,
                "batch_size": online_rl_cfg.batch_size,
                "offline_cache_path": online_rl_cfg.offline_cache_path,
                "offline_batch_fraction": online_rl_cfg.offline_batch_fraction,
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
            },
            settings=wandb.Settings(save_code=False),
        )

    def _log(self, data: dict[str, Any], step: int) -> None:
        if self.wandb_run is None:
            return
        self.wandb_run.log(data, step=step)

    def close(self) -> None:
        if self.wandb_run is not None:
            self.wandb_run.finish()

    def start_episode(self, episode_id: int) -> int:
        """Call at the start of every rollout episode. Returns the replay
        buffer's `total_added` baseline to pass back into `maybe_update()`."""
        self.collector.start_episode(episode_id)
        return self.replay_buffer.total_added

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

        if not self.warmup_satisfied(recorded_episodes):
            successes, failures = self.replay_buffer.count_outcomes()
            logging.info(
                "Online RL warmup not yet satisfied after episode %d: "
                "transitions=%d (need %d), successes=%d (need %d), failures=%d (need %d)",
                recorded_episodes, len(self.replay_buffer), self.cfg.min_warmup_transitions,
                successes, self.cfg.min_warmup_successes,
                failures, self.cfg.min_warmup_failures,
            )
            self._log(
                {
                    "online_rl/warmup_satisfied": 0,
                    "online_rl/buffer_transitions": len(self.replay_buffer),
                    "online_rl/buffer_successes": successes,
                    "online_rl/buffer_failures": failures,
                },
                step=recorded_episodes,
            )
            return None
        if len(self.replay_buffer) < self.cfg.batch_size:
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
        self.policy_cfg.actor_update_interval = (
            10**9 if recorded_episodes < critic_only_until else self.online_actor_update_interval
        )
        # See the comment at online_tau's definition: disable forward()'s
        # own (mistimed) soft update for the duration of this call, then
        # do it correctly ourselves after each real critic.step() below.
        self.policy_cfg.tau = 0.0

        # UTD: gradient updates scaled to how much new data this episode
        # added, not a fixed count regardless of episode length.
        requested_updates = max(new_transitions, 0) * self.policy_cfg.utd_ratio
        num_updates = min(requested_updates, self.cfg.max_updates_per_episode)

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
                }
                batch = {k: v.to(device) for k, v in batch.items()}
                loss, info = self.policy.forward(batch)
                if "loss_actor" in info:
                    last_actor_info = info
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                # Matches ChunkACPolicyConfig.get_optimizer_preset()'s
                # grad_clip_norm=1.0 (applied automatically for offline
                # training by lerobot's generic train loop; this custom
                # online loop bypasses that loop entirely, so it has to be
                # applied explicitly here to get the same protection).
                torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), max_norm=1.0)
                self.critic_optimizer.step()
                self.actor_optimizer.step()
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
        stats: dict[str, Any] = {
            "new_transitions": new_transitions,
            "requested_updates": requested_updates,
            "actual_updates": num_updates,
            "training_time_s": training_time_s,
            "offline_batch_size": offline_batch_size,
            "online_batch_size": online_batch_size,
            "loss": loss_dict,
            "checkpoint_path": None,
        }
        successes, failures = self.replay_buffer.count_outcomes()
        self._log(
            {
                "online_rl/warmup_satisfied": 1,
                "online_rl/critic_only": float(recorded_episodes < critic_only_until),
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
