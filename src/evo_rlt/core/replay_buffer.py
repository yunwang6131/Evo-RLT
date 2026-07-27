from __future__ import annotations

import random
from collections import deque

import torch

from evo_rlt.core.interfaces import (
    ACTUAL_STEPS,
    DONE,
    EPISODE_ID,
    EXEC_CHUNK_FLAT,
    IS_CRITICAL,
    NEXT_REF_FLAT,
    NEXT_STATE_VEC,
    REF_CHUNK_FLAT,
    REWARD_SEQ,
    SOURCE,
    STATE_VEC,
    ChunkTransition,
)


class ReplayBuffer:
    """Deque-based chunk-level replay buffer.

    Stores single (unbatched) ChunkTransition objects and collates them
    into batched dicts at sample time.

    TODO: Replace with tensor-backed circular buffer for performance at scale.
    """

    def __init__(self, capacity: int = 200_000):
        self.buffer: deque[ChunkTransition] = deque(maxlen=capacity)
        # Monotonic count of every transition ever added, unaffected by the
        # deque's automatic eviction once full. len(buffer) alone understates
        # "new" additions after the buffer fills (old entries silently drop
        # off, so len() stops growing) -- callers computing "how many
        # transitions did this episode add" (e.g. UTD scaling) must diff
        # total_added, not len().
        self.total_added: int = 0

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def capacity(self) -> int:
        if self.buffer.maxlen is None:
            raise RuntimeError("Buffer has no capacity limit")
        return self.buffer.maxlen

    def add(self, transition: ChunkTransition) -> None:
        self.buffer.append(transition)
        self.total_added += 1

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample a batch uniformly and collate into a dict of stacked tensors."""
        n = min(batch_size, len(self.buffer))
        indices = random.sample(range(len(self.buffer)), n)
        return self._collate([self.buffer[i] for i in indices])

    def episode_outcomes(self) -> dict[int, str]:
        """Map episode_id -> "success"/"failure" for every episode that has a
        terminal (done=1) transition currently in the buffer."""
        outcomes: dict[int, str] = {}
        for t in self.buffer:
            if t.done.item() == 1.0:
                eid = int(t.episode_id.item())
                outcomes[eid] = "success" if t.reward_seq.sum().item() > 0 else "failure"
        return outcomes

    def count_outcomes(self) -> tuple[int, int]:
        """Return (num_success_episodes, num_failure_episodes) represented in the buffer."""
        outcomes = self.episode_outcomes()
        successes = sum(1 for v in outcomes.values() if v == "success")
        failures = sum(1 for v in outcomes.values() if v == "failure")
        return successes, failures

    def sample_stratified(
        self,
        batch_size: int,
        success_frac: float = 0.4,
        failure_frac: float = 0.3,
        intervention_frac: float = 0.2,
        recent_frac: float = 0.1,
        recent_window: int = 500,
    ) -> dict[str, torch.Tensor]:
        """Sample a batch stratified across success/failure/intervention/recent
        transitions so sparse positive-reward terminal transitions (and their
        episode's lead-up transitions, via episode_id) aren't drowned out by
        uniform sampling of a mostly-zero-reward buffer.

        Buckets are populated per-transition (not just terminal transitions):
        every transition belonging to an episode that eventually succeeded or
        failed is tagged via `episode_id`, so the critic also sees the
        non-terminal lead-up steps of successful/failed episodes, not only the
        final reward step. Falls back to uniform sampling to fill any shortfall
        when a bucket is too small (e.g. no failures seen yet early in training).
        """
        n = len(self.buffer)
        if n == 0:
            raise ValueError("Cannot sample from an empty replay buffer")
        outcomes = self.episode_outcomes()
        recent_start = max(0, n - recent_window)

        success_idx: list[int] = []
        failure_idx: list[int] = []
        intervention_idx: list[int] = []
        other_idx: list[int] = []
        for i, t in enumerate(self.buffer):
            if t.intervention.item() == 1.0:
                intervention_idx.append(i)
                continue
            outcome = outcomes.get(int(t.episode_id.item()))
            if outcome == "success":
                success_idx.append(i)
            elif outcome == "failure":
                failure_idx.append(i)
            else:
                other_idx.append(i)
        recent_idx = list(range(recent_start, n))

        def _take(pool: list[int], k: int) -> list[int]:
            if not pool or k <= 0:
                return []
            return random.sample(pool, k) if len(pool) >= k else random.choices(pool, k=k)

        quotas = {
            "success": round(batch_size * success_frac),
            "failure": round(batch_size * failure_frac),
            "intervention": round(batch_size * intervention_frac),
            "recent": round(batch_size * recent_frac),
        }
        indices = (
            _take(success_idx, quotas["success"])
            + _take(failure_idx, quotas["failure"])
            + _take(intervention_idx, quotas["intervention"])
            + _take(recent_idx, quotas["recent"])
        )
        shortfall = batch_size - len(indices)
        if shortfall > 0:
            indices += _take(other_idx or list(range(n)), shortfall)
        indices = indices[:batch_size]

        return self._collate([self.buffer[i] for i in indices])

    def _collate(self, batch: list[ChunkTransition]) -> dict[str, torch.Tensor]:
        stacked_exec = torch.stack([t.exec_chunk for t in batch])
        stacked_ref = torch.stack([t.ref_chunk for t in batch])
        stacked_next_ref = torch.stack([t.next_ref_chunk for t in batch])
        return {
            STATE_VEC: torch.stack([t.state_vec for t in batch]),
            EXEC_CHUNK_FLAT: stacked_exec.flatten(start_dim=-2),
            REF_CHUNK_FLAT: stacked_ref.flatten(start_dim=-2),
            REWARD_SEQ: torch.stack([t.reward_seq for t in batch]),
            NEXT_STATE_VEC: torch.stack([t.next_state_vec for t in batch]),
            NEXT_REF_FLAT: stacked_next_ref.flatten(start_dim=-2),
            DONE: torch.stack([t.done for t in batch]),
            ACTUAL_STEPS: torch.stack([t.actual_steps for t in batch]),
            SOURCE: torch.stack([t.source for t in batch]),
            EPISODE_ID: torch.stack([t.episode_id for t in batch]),
            IS_CRITICAL: torch.stack([t.is_critical for t in batch]),
        }
