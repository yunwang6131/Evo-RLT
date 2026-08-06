from __future__ import annotations

import torch
import pytest

from evo_rlt.adapters.lerobot.online_collector import RLTOnlineCollector
from evo_rlt.adapters.lerobot.record.annotations import SOURCE_HUMAN, SOURCE_RL
from evo_rlt.core.replay_buffer import ReplayBuffer

CHUNK_LENGTH = 4
ACTION_DIM = 6
STATE_DIM = 10


def _make_collector() -> tuple[RLTOnlineCollector, ReplayBuffer]:
    buffer = ReplayBuffer(capacity=100)
    # time_decay=1.0: these tests are about staging/flush mechanics, not
    # decay -- see TestMilestoneReward for decay-specific behavior, which
    # uses its own helper with an explicit time_decay per test.
    collector = RLTOnlineCollector(
        replay_buffer=buffer, chunk_length=CHUNK_LENGTH, action_dim=ACTION_DIM, time_decay=1.0
    )
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


class TestFlushedGuard:
    """After flush_episode() (critical phase resolved), the recorded episode
    may keep going (e.g. VLA autonomously finishing a subsequent step) --
    on_frame() must ignore all of that so the RL reward reflects only what
    the actor actually controlled, not whatever happens afterward."""

    def test_on_frame_is_noop_after_flush(self):
        collector, buffer = _make_collector()
        for i in range(CHUNK_LENGTH - 1):
            _feed_frame(collector, i)
        collector.flush_episode(episode_success=True)
        assert len(buffer) == 1

        # Episode keeps recording under VLA afterward -- fed to on_frame as
        # usual by loop.py, but must not affect the buffer at all.
        for i in range(CHUNK_LENGTH * 3):
            _feed_frame(collector, i % CHUNK_LENGTH)
        assert len(buffer) == 1
        assert collector._episode_staging == []

    def test_next_start_episode_clears_flushed_guard(self):
        collector, buffer = _make_collector()
        for i in range(CHUNK_LENGTH - 1):
            _feed_frame(collector, i)
        collector.flush_episode(episode_success=True)
        assert collector._flushed is True

        collector.start_episode(episode_id=1)
        assert collector._flushed is False
        for i in range(CHUNK_LENGTH - 1):
            _feed_frame(collector, i)
        collector.flush_episode(episode_success=False)
        assert len(buffer) == 2


class TestMilestoneReward:
    def _make_collector_with_shaping(self, **kwargs) -> tuple[RLTOnlineCollector, ReplayBuffer]:
        buffer = ReplayBuffer(capacity=100)
        collector = RLTOnlineCollector(
            replay_buffer=buffer, chunk_length=CHUNK_LENGTH, action_dim=ACTION_DIM, **kwargs
        )
        collector.start_episode(episode_id=0)
        return collector, buffer

    def test_time_decay_of_one_reproduces_old_fixed_reward(self):
        """time_decay=1.0 must reproduce the pre-shaping behavior exactly:
        terminal reward is always exactly 1.0/0.0 no matter how many chunks
        closed. (The class default below is < 1.0 -- see its own test.)"""
        collector, buffer = self._make_collector_with_shaping(time_decay=1.0)
        for i in range(CHUNK_LENGTH - 1):
            _feed_frame(collector, i)
        collector.flush_episode(episode_success=True)
        transition = buffer.buffer[0]
        assert transition.reward_seq.sum().item() == pytest.approx(1.0)

    def test_default_time_decay_is_enabled_not_a_no_op(self):
        """The class default (0.995, decaying per closed chunk) must actually
        shrink the terminal reward once at least one chunk has closed -- this
        flag is on by default, not an inert opt-in."""
        collector, buffer = self._make_collector_with_shaping()  # default time_decay
        for i in range(CHUNK_LENGTH):  # one full chunk closes
            _feed_frame(collector, i)
        for i in range(CHUNK_LENGTH - 1):  # partial terminal chunk
            _feed_frame(collector, i)
        collector.flush_episode(episode_success=True)
        terminal = buffer.buffer[-1]
        actual = int(terminal.actual_steps.item())
        # 2 chunks closed by the time the terminal reward is computed (the
        # first full chunk, then this terminal chunk itself).
        assert terminal.reward_seq[actual - 1].item() == pytest.approx(1.0 * 0.995 ** 2)
        assert terminal.reward_seq[actual - 1].item() < 1.0

    def test_milestone_fires_once_and_lands_on_next_chunk_close(self):
        collector, buffer = self._make_collector_with_shaping(milestone_reward=0.5, time_decay=1.0)
        _feed_frame(collector, 0)
        bonus = collector.mark_milestone()
        assert bonus == pytest.approx(0.5)
        # Second press this attempt must be a no-op.
        assert collector.mark_milestone() == 0.0

        for i in range(1, CHUNK_LENGTH):  # close out the chunk the milestone landed in
            _feed_frame(collector, i)
        assert len(buffer) == 0  # first chunk auto-emitted but only staged, not committed yet
        first_staged = collector._episode_staging[0]
        assert first_staged.done.item() == 0.0
        assert first_staged.reward_seq.sum().item() == pytest.approx(0.5)

        for i in range(CHUNK_LENGTH - 1):  # second, partial, terminal chunk
            _feed_frame(collector, i)
        collector.flush_episode(episode_success=True)

        assert len(buffer) == 2
        assert buffer.buffer[0].reward_seq.sum().item() == pytest.approx(0.5)  # milestone chunk unaffected
        assert buffer.buffer[1].reward_seq.sum().item() == pytest.approx(1.0)  # terminal chunk, no bonus left
        assert collector.last_episode_reward == pytest.approx(1.5)

    def test_milestone_and_terminal_additive_on_same_chunk(self):
        """Milestone pressed in the same chunk that immediately closes the
        episode: both rewards must land on that one chunk, summed."""
        collector, buffer = self._make_collector_with_shaping(milestone_reward=0.5, time_decay=1.0)
        for i in range(CHUNK_LENGTH - 1):
            _feed_frame(collector, i)
        collector.mark_milestone()
        collector.flush_episode(episode_success=True)

        assert len(buffer) == 1
        transition = buffer.buffer[0]
        assert transition.reward_seq.sum().item() == pytest.approx(1.5)
        assert collector.last_episode_reward == pytest.approx(1.5)

    def test_milestone_does_not_turn_terminal_failure_into_success(self):
        collector, buffer = self._make_collector_with_shaping(
            milestone_reward=0.5, time_decay=1.0
        )
        for i in range(CHUNK_LENGTH - 1):
            _feed_frame(collector, i)
        collector.mark_milestone()
        collector.flush_episode(episode_success=False)

        transition = buffer.buffer[0]
        assert transition.reward_seq.sum().item() == pytest.approx(0.5)
        assert transition.outcome.item() == 0.0
        assert buffer.episode_outcomes() == {0: "failure"}

    def test_flush_episode_is_idempotent(self):
        collector, buffer = self._make_collector_with_shaping(time_decay=1.0)
        for i in range(CHUNK_LENGTH - 1):
            _feed_frame(collector, i)
        collector.flush_episode(episode_success=False)
        assert collector.flush_episode(episode_success=True) is None
        assert len(buffer) == 1
        assert buffer.buffer[0].outcome.item() == 0.0
        assert collector.last_episode_reward == 0.0

    def test_milestone_ignored_after_flush(self):
        collector, buffer = self._make_collector_with_shaping(milestone_reward=0.5, time_decay=1.0)
        for i in range(CHUNK_LENGTH - 1):
            _feed_frame(collector, i)
        collector.flush_episode(episode_success=True)
        assert collector.mark_milestone() == 0.0
        assert collector.last_episode_reward == pytest.approx(1.0)

    def test_time_decay_shrinks_reward_by_chunks_closed_not_frames(self):
        """The decay exponent must be chunks closed, not raw frame count --
        a milestone pressed mid-chunk (no chunk closed yet) gets exponent 0
        (undiscounted); one pressed after N chunks have already closed gets
        exponent N regardless of how many frames those N chunks contained."""
        collector, buffer = self._make_collector_with_shaping(
            milestone_reward=1.0, terminal_reward=1.0, time_decay=0.9
        )
        for i in range(CHUNK_LENGTH - 1):  # partial chunk, nothing closed yet
            _feed_frame(collector, i)
        bonus_step0 = collector.mark_milestone()
        assert bonus_step0 == pytest.approx(1.0)  # 0.9 ** 0 == 1.0, no discount yet

        collector.flush_episode(episode_success=True)  # closes 1 chunk -> _chunks_closed=1
        expected_terminal = 1.0 * 0.9 ** 1
        assert collector.last_episode_reward == pytest.approx(bonus_step0 + expected_terminal)
        assert expected_terminal < 1.0

    def test_default_time_decay_shrinks_a_milestone_pressed_after_chunks_close(self):
        collector, buffer = self._make_collector_with_shaping(milestone_reward=1.0)  # default time_decay
        for i in range(CHUNK_LENGTH):  # 2 full chunks close before the milestone press
            _feed_frame(collector, i)
        for i in range(CHUNK_LENGTH):
            _feed_frame(collector, i)
        bonus = collector.mark_milestone()
        assert bonus == pytest.approx(1.0 * 0.995 ** 2)
        assert bonus < 1.0

    def test_milestone_bonus_dropped_by_next_start_episode_if_never_flushed(self):
        """A rerecorded attempt's milestone bonus must not leak into the retry."""
        collector, buffer = self._make_collector_with_shaping(milestone_reward=0.5, time_decay=1.0)
        _feed_frame(collector, 0)
        collector.mark_milestone()
        collector.start_episode(episode_id=1)  # rerecord, no flush_episode() call
        for i in range(CHUNK_LENGTH - 1):
            _feed_frame(collector, i)
        collector.flush_episode(episode_success=True)
        assert buffer.buffer[0].reward_seq.sum().item() == pytest.approx(1.0)  # no stray 0.5


class TestHumanChunkPreservesVLAReference:
    """Regression test: a human-dominant chunk used to overwrite ref with
    exec_chunk, erasing the "VLA would have done X, the correction was
    delta=Y" signal the residual actor (mu = ref + delta) is supposed to
    learn from a successful intervention -- turning it into a tautological
    ref==exec, delta==0 pair instead. ref must stay the true, current VLA
    reference regardless of who executed the chunk."""

    def test_human_dominant_chunk_keeps_vla_ref_not_exec(self):
        collector, buffer = _make_collector()
        vla_ref = torch.randn(CHUNK_LENGTH, ACTION_DIM)
        human_action = torch.randn(ACTION_DIM) + 5.0  # far from vla_ref, unmistakably distinct
        for i in range(CHUNK_LENGTH):
            collector.on_frame(
                action=human_action,
                state_vec=torch.randn(STATE_DIM) if i == 0 else None,
                ref_chunk=vla_ref if i == 0 else None,
                source_type=SOURCE_HUMAN,
                is_critical=1.0,
            )
        collector.flush_episode(episode_success=True)

        transition = buffer.buffer[0]
        assert torch.equal(transition.ref_chunk, vla_ref)
        assert not torch.allclose(transition.ref_chunk, transition.exec_chunk)
        assert transition.intervention.item() == 1.0

    def test_human_chunk_without_current_context_is_dropped_not_fabricated(self):
        """A caller must supply current counterfactual VLA context.
        The collector never reuses the prior chunk's state/ref just to keep a
        human replay item; that tuple would be temporally false."""
        collector, buffer = _make_collector()
        first_ref = torch.randn(CHUNK_LENGTH, ACTION_DIM)
        for i in range(CHUNK_LENGTH):
            collector.on_frame(
                action=torch.randn(ACTION_DIM),
                state_vec=torch.randn(STATE_DIM) if i == 0 else None,
                ref_chunk=first_ref if i == 0 else None,
                source_type=SOURCE_RL,
                is_critical=1.0,
            )
        human_action = torch.randn(ACTION_DIM) + 5.0
        for i in range(CHUNK_LENGTH):
            collector.on_frame(
                action=human_action,
                state_vec=None,
                ref_chunk=None,
                source_type=SOURCE_HUMAN,
                is_critical=1.0,
            )
        collector.flush_episode(episode_success=True)

        assert len(buffer) == 1
        first_transition = buffer.buffer[0]
        assert torch.equal(first_transition.next_ref_chunk, first_ref)
        # The context-free human frames were dropped completely.
        assert first_transition.intervention.item() == 0.0
        assert first_transition.intervention_mask.sum().item() == 0.0

    def test_any_human_element_marks_the_actual_transition_intervened(self):
        collector, buffer = _make_collector()
        state = torch.randn(STATE_DIM)
        ref = torch.randn(CHUNK_LENGTH, ACTION_DIM)
        for i in range(CHUNK_LENGTH):
            mask = torch.zeros(ACTION_DIM)
            source = SOURCE_RL
            action = torch.zeros(ACTION_DIM)
            if i == CHUNK_LENGTH - 1:
                mask[0] = 1.0
                source = SOURCE_HUMAN
                action[0] = 0.4
            collector.on_frame(
                action=action,
                state_vec=state if i == 0 else None,
                ref_chunk=ref if i == 0 else None,
                source_type=source,
                is_critical=1.0,
                intervention_mask=mask,
            )
        collector.flush_episode(True)

        transition = buffer.buffer[0]
        assert transition.intervention.item() == 1.0
        # Source remains the dominant RL source, demonstrating that the
        # intervention tag now follows the exact mask rather than majority.
        assert transition.source.item() == SOURCE_RL
        assert transition.intervention_mask.sum().item() == 1.0
        assert transition.outcome.item() == 1.0


class TestControlBoundaryAlignment:
    def test_begin_attempt_drops_prefix_residue_and_aligns_first_rl_chunk(self):
        collector, buffer = _make_collector()
        for _ in range(2):
            collector.on_frame(
                action=torch.full((ACTION_DIM,), -1.0),
                state_vec=None,
                ref_chunk=None,
                source_type=SOURCE_RL,
                is_critical=0.0,
            )

        collector.begin_attempt()
        state = torch.randn(STATE_DIM)
        ref = torch.randn(CHUNK_LENGTH, ACTION_DIM)
        actions = [torch.full((ACTION_DIM,), float(i)) for i in range(CHUNK_LENGTH)]
        for i, action in enumerate(actions):
            collector.on_frame(
                action=action,
                state_vec=state if i == 0 else None,
                ref_chunk=ref if i == 0 else None,
                source_type=SOURCE_RL,
                is_critical=1.0,
            )
        collector.flush_episode(True)

        assert len(buffer) == 1
        assert torch.equal(buffer.buffer[0].exec_chunk, torch.stack(actions))
        assert torch.equal(buffer.buffer[0].state_vec, state)

    def test_cut_chunk_keeps_partial_validity_and_new_context_separate(self):
        collector, buffer = _make_collector()
        first_ref = torch.randn(CHUNK_LENGTH, ACTION_DIM)
        first_actions = [torch.randn(ACTION_DIM) for _ in range(2)]
        for i, action in enumerate(first_actions):
            collector.on_frame(
                action=action,
                state_vec=torch.randn(STATE_DIM) if i == 0 else None,
                ref_chunk=first_ref if i == 0 else None,
                source_type=SOURCE_RL,
                is_critical=1.0,
            )
        collector.cut_chunk()

        second_ref = torch.randn(CHUNK_LENGTH, ACTION_DIM)
        human = torch.randn(ACTION_DIM)
        human_mask = torch.cat([torch.ones(ACTION_DIM // 2), torch.zeros(ACTION_DIM // 2)])
        collector.on_frame(
            action=human,
            state_vec=torch.randn(STATE_DIM),
            ref_chunk=second_ref,
            source_type=SOURCE_HUMAN,
            is_critical=1.0,
            intervention_mask=human_mask,
        )
        collector.flush_episode(True)

        first, second = buffer.buffer
        assert first.actual_steps.item() == 2
        assert torch.equal(first.exec_chunk[2:], first_ref[2:])
        assert torch.equal(second.intervention_mask[0], human_mask)
        assert second.intervention_mask[1:].sum().item() == 0.0
