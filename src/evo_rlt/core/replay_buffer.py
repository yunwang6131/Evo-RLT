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

    def rollback(self, total_added_before: int) -> int:
        """Undo every add() since `total_added_before` by popping them back
        off the right end of the deque -- add() only ever appends, so the
        most recent entries are exactly the ones to remove (e.g. a
        rerecorded/discarded episode's already-flushed transitions). If some
        of those were already evicted by capacity, only what's still
        physically present is removed (total_added is decremented to match,
        not reset to total_added_before, so it stays an accurate count of
        what's actually in the buffer's history). Returns how many were
        removed."""
        n_to_remove = self.total_added - total_added_before
        if n_to_remove <= 0:
            return 0
        n_removed = min(n_to_remove, len(self.buffer))
        for _ in range(n_removed):
            self.buffer.pop()
        self.total_added -= n_removed
        return n_removed

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample a batch uniformly and collate into a dict of stacked tensors."""
        n = min(batch_size, len(self.buffer))
        indices = random.sample(range(len(self.buffer)), n)
        return self._collate([self.buffer[i] for i in indices])

    def episode_outcomes(self) -> dict[int, str]:
        """Map episode_id -> "success"/"failure" for every episode that has a
        terminal (done=1) transition currently in the buffer.

        New transitions carry an explicit outcome so positive milestone
        shaping cannot turn a failed episode into a success.  The reward
        fallback keeps old checkpoints/caches readable.
        """
        outcomes: dict[int, str] = {}
        for t in self.buffer:
            if t.done.item() == 1.0:
                eid = int(t.episode_id.item())
                explicit = getattr(t, "outcome", None)
                explicit_value = float(explicit.item()) if explicit is not None else -1.0
                if explicit_value >= 0.0:
                    outcomes[eid] = "success" if explicit_value >= 0.5 else "failure"
                else:
                    outcomes[eid] = "success" if t.reward_seq.sum().item() > 0 else "failure"
        return outcomes

    def outcome_labels(self, episode_id: torch.Tensor) -> torch.Tensor:
        """Map each transition's episode_id to its trajectory outcome, for
        the RankQ ranking loss (see losses.rankq_ranking_loss): 1.0 =
        success, 0.0 = failure, -1.0 = unresolved/unknown (e.g. the episode
        has no terminal transition in the buffer yet). Looks up outcomes via
        episode_outcomes(), so it reflects the *current* buffer contents,
        not necessarily the episode_id tensor's own source buffer -- pass
        the episode_id of a batch already sampled from this buffer.
        """
        outcomes = self.episode_outcomes()
        return torch.tensor(
            [
                1.0 if outcomes.get(int(eid.item())) == "success"
                else 0.0 if outcomes.get(int(eid.item())) == "failure"
                else -1.0
                for eid in episode_id
            ],
            dtype=torch.float32,
        )

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
            explicit = getattr(t, "outcome", None)
            explicit_value = float(explicit.item()) if explicit is not None else -1.0
            outcome = (
                "success" if explicit_value >= 0.5
                else "failure" if explicit_value >= 0.0
                else outcomes.get(int(t.episode_id.item()))
            )
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
        intervention_masks = []
        for t in batch:
            mask = getattr(t, "intervention_mask", None)
            if mask is None or mask.numel() == 0:
                mask = torch.full_like(t.exec_chunk, float(t.intervention.item()))
            intervention_masks.append(mask)
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
            "outcome": torch.stack(
                [
                    getattr(t, "outcome", torch.tensor(-1.0))
                    for t in batch
                ]
            ),
            "intervention_mask_flat": torch.stack(intervention_masks).flatten(start_dim=-2),
        }
