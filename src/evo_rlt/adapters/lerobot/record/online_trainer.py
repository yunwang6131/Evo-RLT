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

import logging
import os
import time
from pathlib import Path
from typing import Any

import torch

from evo_rlt.adapters.lerobot.online_collector import RLTOnlineCollector
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

    def start_episode(self, episode_id: int) -> int:
        """Call at the start of every rollout episode. Returns the replay
        buffer's `total_added` baseline to pass back into `maybe_update()`."""
        self.collector.start_episode(episode_id)
        return self.replay_buffer.total_added

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
        try:
            for _ in range(num_updates):
                # ReplayBuffer.sample()/sample_stratified() return core/losses.py's
                # flat-key format (exec_chunk_flat/ref_chunk_flat/next_ref_flat);
                # ChunkACPolicy's forward()/_coerce_batch expects unflattened
                # (B, C, action_dim) exec_chunk/ref_chunk/next_ref_chunk and
                # flattens internally.
                if self.cfg.use_stratified_sampling:
                    raw = self.replay_buffer.sample_stratified(self.cfg.batch_size)
                else:
                    raw = self.replay_buffer.sample(self.cfg.batch_size)
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
                }
                batch = {k: v.to(device) for k, v in batch.items()}
                loss, info = self.policy.forward(batch)
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
        logging.info(
            "Online RL update after episode %d: new_transitions=%d requested_updates=%d "
            "actual_updates=%d effective_utd=%.2f training_time=%.1fs loss=%s",
            recorded_episodes, new_transitions, requested_updates, num_updates,
            (num_updates / new_transitions) if new_transitions > 0 else 0.0,
            training_time_s,
            {k: (v.item() if torch.is_tensor(v) else v) for k, v in (info or {}).items()},
        )

        stats: dict[str, Any] = {
            "new_transitions": new_transitions,
            "requested_updates": requested_updates,
            "actual_updates": num_updates,
            "training_time_s": training_time_s,
            "loss": {k: (v.item() if torch.is_tensor(v) else v) for k, v in (info or {}).items()},
            "checkpoint_path": None,
        }
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

        Loading this back into a running session (full resume) is not
        wired up yet -- this is crash-survivability for the data, not a
        `--resume` CLI path.
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
            "torch_rng_state": torch.get_rng_state(),
        }
        final_path = save_dir / "latest_online_state.pt"
        tmp_path = save_dir / "latest_online_state.pt.tmp"
        torch.save(state, tmp_path)
        os.replace(tmp_path, final_path)
        logging.info(
            "Online RL latest state saved to %s (episodes=%d, buffer=%d transitions)",
            final_path, completed_episodes, len(self.replay_buffer),
        )
