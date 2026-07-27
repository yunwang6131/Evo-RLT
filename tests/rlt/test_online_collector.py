from __future__ import annotations

import torch
import pytest

from evo_rlt.adapters.lerobot.online_collector import RLTOnlineCollector
from evo_rlt.adapters.lerobot.record.annotations import SOURCE_RL
from evo_rlt.core.replay_buffer import ReplayBuffer

CHUNK_LENGTH = 4
ACTION_DIM = 6
STATE_DIM = 10


def _make_collector() -> tuple[RLTOnlineCollector, ReplayBuffer]:
    buffer = ReplayBuffer(capacity=100)
    collector = RLTOnlineCollector(replay_buffer=buffer, chunk_length=CHUNK_LENGTH, action_dim=ACTION_DIM)
    collector.start_episode(episode_id=0)
    return collector, buffer


def _feed_frame(collector: RLTOnlineCollector, chunk_idx: int) -> None:
    is_chunk_start = chunk_idx == 0
    state = torch.randn(STATE_DIM) if is_chunk_start else None
    ref = torch.randn(CHUNK_LENGTH, ACTION_DIM) if is_chunk_start else None
    collector.on_frame(
        action=torch.randn(ACTION_DIM),
        state_vec=state,
        ref_chunk=ref,
        source_type=SOURCE_RL,
        is_critical=1.0,
    )


class TestFlushEpisodeReward:
    def test_success_sets_terminal_reward_partial_chunk(self):
        """Episode ends mid-chunk (not an exact multiple of C): flush_episode
        emits the terminal transition itself."""
        collector, buffer = _make_collector()
        for i in range(CHUNK_LENGTH - 1):  # one short of a full chunk
            _feed_frame(collector, i)
        collector.flush_episode(episode_success=True)

        assert len(buffer) == 1
        transition = buffer.buffer[0]
        actual = int(transition.actual_steps.item())
        assert actual == CHUNK_LENGTH - 1
        assert transition.done.item() == 1.0
        assert transition.reward_seq[actual - 1].item() == 1.0
        assert transition.reward_seq[:actual - 1].sum().item() == 0.0

    def test_failure_leaves_reward_zero_partial_chunk(self):
        collector, buffer = _make_collector()
        for i in range(CHUNK_LENGTH - 1):
            _feed_frame(collector, i)
        collector.flush_episode(episode_success=False)

        transition = buffer.buffer[0]
        assert transition.done.item() == 1.0
        assert transition.reward_seq.sum().item() == 0.0

    def test_success_patches_prev_transition_exact_multiple(self):
        """Episode length is an exact multiple of C: the last chunk was already
        emitted (staged, not yet committed to the buffer) with done=False by
        on_frame; flush_episode must patch it and commit it."""
        collector, buffer = _make_collector()
        for i in range(CHUNK_LENGTH):  # exactly one full chunk, already emitted (staged)
            _feed_frame(collector, i)
        assert len(buffer) == 0  # staged, not committed until flush_episode
        assert collector._episode_staging[0].done.item() == 0.0

        collector.flush_episode(episode_success=True)

        assert len(buffer) == 1  # no extra transition emitted
        transition = buffer.buffer[0]
        assert transition.done.item() == 1.0
        actual = int(transition.actual_steps.item())
        assert actual == CHUNK_LENGTH
        assert transition.reward_seq[actual - 1].item() == 1.0

    def test_failure_patches_prev_transition_exact_multiple(self):
        collector, buffer = _make_collector()
        for i in range(CHUNK_LENGTH):
            _feed_frame(collector, i)
        collector.flush_episode(episode_success=False)

        transition = buffer.buffer[0]
        assert transition.done.item() == 1.0
        assert transition.reward_seq.sum().item() == 0.0

    def test_non_terminal_chunks_keep_zero_reward(self):
        """Only the terminal transition's reward should ever be nonzero."""
        collector, buffer = _make_collector()
        for i in range(CHUNK_LENGTH):  # first full chunk (non-terminal)
            _feed_frame(collector, i)
        for i in range(CHUNK_LENGTH - 1):  # second, partial, terminal chunk
            _feed_frame(collector, i)
        collector.flush_episode(episode_success=True)

        assert len(buffer) == 2
        first, second = buffer.buffer[0], buffer.buffer[1]
        assert first.done.item() == 0.0
        assert first.reward_seq.sum().item() == 0.0
        assert second.done.item() == 1.0
        actual = int(second.actual_steps.item())
        assert second.reward_seq[actual - 1].item() == 1.0


class TestEpisodeStaging:
    """Transitions must stay out of the global replay buffer until
    flush_episode() commits them, so a rerecorded/discarded/never-labeled
    episode (flush_episode never called) can't leave dangling, unlabeled,
    no-valid-next-state transitions in the buffer forever."""

    def test_unflushed_episode_never_reaches_global_buffer(self):
        collector, buffer = _make_collector()
        for i in range(CHUNK_LENGTH):  # a full non-terminal chunk gets staged
            _feed_frame(collector, i)
        for i in range(CHUNK_LENGTH - 2):  # partial second chunk, never completes
            _feed_frame(collector, i)
        # Episode gets rerecorded/discarded: caller never calls flush_episode().
        assert len(buffer) == 0

    def test_next_start_episode_drops_unflushed_staging(self):
        collector, buffer = _make_collector()
        for i in range(CHUNK_LENGTH):
            _feed_frame(collector, i)
        assert len(collector._episode_staging) == 1
        # Rerecord: start the retry without ever flushing the bad attempt.
        collector.start_episode(episode_id=0)
        assert collector._episode_staging == []
        for i in range(CHUNK_LENGTH - 1):
            _feed_frame(collector, i)
        collector.flush_episode(episode_success=True)
        # Only the retry's transition made it to the buffer.
        assert len(buffer) == 1
