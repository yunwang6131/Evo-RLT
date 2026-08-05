from __future__ import annotations

import torch
import pytest

from evo_rlt.adapters.lerobot.policies.action_modifier import RLTActionModifier
from evo_rlt.core.phase_controller import PhaseController

ACTION_DIM = 4
CHUNK_LENGTH = 3
PROPRIO_DIM = 2
RL_TOKEN_DIM = 5


class _FakeRLToken:
    def encode(self, prefix_tokens):
        return torch.zeros(prefix_tokens.shape[0], RL_TOKEN_DIM)


class _FixedActor:
    """Duck-typed stand-in for ChunkActor: always returns a configured mu,
    regardless of state/ref, so tests can dictate the actor's raw output
    exactly instead of depending on a real (trained or zero-init) network."""

    def __init__(self, mu_chunk: torch.Tensor):
        self._mu_flat = mu_chunk.flatten(start_dim=-2)

    def __call__(self, state_vec, ref_flat, training=False):
        return self._mu_flat, torch.zeros_like(self._mu_flat)


def _make_modifier(
    mu_chunk: torch.Tensor,
    action_clip_delta: float | None = None,
    slew_rate_limit: float | None = None,
    phase_ctrl: PhaseController | None = None,
) -> RLTActionModifier:
    if phase_ctrl is None:
        phase_ctrl = PhaseController(mode="manual")
        phase_ctrl.trigger_critical()
    return RLTActionModifier(
        rl_token=_FakeRLToken(),
        actor=_FixedActor(mu_chunk),
        phase_ctrl=phase_ctrl,
        chunk_length=CHUNK_LENGTH,
        action_dim=ACTION_DIM,
        proprio_dim=PROPRIO_DIM,
        chunk_exec_steps=CHUNK_LENGTH,
        action_clip_delta=action_clip_delta,
        slew_rate_limit=slew_rate_limit,
    )


def _compute(modifier: RLTActionModifier, ref_chunk: torch.Tensor) -> torch.Tensor:
    B = ref_chunk.shape[0]
    proprio = torch.zeros(B, PROPRIO_DIM)
    prefix_tokens = torch.zeros(B, 1, RL_TOKEN_DIM)
    return modifier.compute_chunk(ref_chunk, proprio, prefix_tokens)


def _run_full_chunk(modifier: RLTActionModifier, ref_chunk: torch.Tensor) -> torch.Tensor:
    """Simulate real per-frame consumption -- compute the chunk, enqueue it,
    then pop every frame one at a time, as select_action() would each real
    control step (see modeling_rlt_ac.py's needs_new_chunk/enqueue/
    pop_action). _last_executed_action is only ever updated by a real
    pop_action() (or record_executed_action()), never by compute_chunk()
    alone -- calling compute_chunk() directly without this, as the tests
    below used to, does not exercise the actual continuity-tracking path."""
    chunk = _compute(modifier, ref_chunk)
    modifier.enqueue(chunk)
    popped = [modifier.pop_action() for _ in range(chunk.shape[1])]
    return torch.stack(popped, dim=1)


class TestComputeChunkActionClipDelta:
    def test_zero_residual_respects_delta_even_when_ref_exceeds_unit_range(self):
        """Regression test for the exact deployment-side bug: actor exactly
        equals ref (raw residual == 0) but ref itself exceeds [-1,1] (a real,
        common QUANTILES-normalization occurrence). The executed chunk must
        still land within action_clip_delta of ref -- the old
        clamp(-1,1)->delta->clamp(-1,1) sequence would visibly jump here even
        though the actor contributed nothing."""
        ref_chunk = torch.full((2, CHUNK_LENGTH, ACTION_DIM), 1.5)
        modifier = _make_modifier(mu_chunk=ref_chunk, action_clip_delta=0.1)
        chunk = _compute(modifier, ref_chunk)
        assert (chunk - ref_chunk).abs().max().item() < 0.1 + 1e-4

    def test_large_residual_is_bounded_by_delta(self):
        ref_chunk = torch.zeros(2, CHUNK_LENGTH, ACTION_DIM)
        mu_chunk = torch.full((2, CHUNK_LENGTH, ACTION_DIM), 50.0)
        modifier = _make_modifier(mu_chunk=mu_chunk, action_clip_delta=0.1)
        chunk = _compute(modifier, ref_chunk)
        assert (chunk - ref_chunk).abs().max().item() < 0.1 + 1e-4

    def test_none_delta_falls_back_to_plain_unit_clamp(self):
        """No action_clip_delta configured -> exact prior behavior (plain
        clamp(-1,1), independent of ref)."""
        ref_chunk = torch.zeros(2, CHUNK_LENGTH, ACTION_DIM)
        mu_chunk = torch.full((2, CHUNK_LENGTH, ACTION_DIM), 5.0)
        modifier = _make_modifier(mu_chunk=mu_chunk, action_clip_delta=None)
        chunk = _compute(modifier, ref_chunk)
        assert torch.allclose(chunk, torch.ones_like(chunk))


class TestSlewRateLimit:
    def test_limits_within_chunk_jumps(self):
        ref_chunk = torch.zeros(1, CHUNK_LENGTH, ACTION_DIM)
        mu_chunk = torch.zeros(1, CHUNK_LENGTH, ACTION_DIM)
        mu_chunk[0, 1, :] = 1.0  # a hard jump mid-chunk, then back to 0
        modifier = _make_modifier(mu_chunk=mu_chunk, slew_rate_limit=0.2)
        chunk = _compute(modifier, ref_chunk)
        steps = chunk[0]
        adjacent_diffs = (steps[1:] - steps[:-1]).abs()
        assert adjacent_diffs.max().item() <= 0.2 + 1e-4

    def test_limits_across_chunk_boundary(self):
        """The delta bound only constrains an action relative to its own
        ref, not relative to the immediately preceding physical frame --
        nothing else stops a new chunk from starting far from where the
        previous one ended. Must be limited across the boundary too. Uses
        _run_full_chunk (compute + enqueue + pop every frame) since
        _last_executed_action is only updated by real pop_action() calls,
        not by compute_chunk() alone."""
        ref_chunk = torch.zeros(1, CHUNK_LENGTH, ACTION_DIM)
        modifier = _make_modifier(mu_chunk=ref_chunk, slew_rate_limit=0.05)
        chunk_1 = _run_full_chunk(modifier, ref_chunk)
        last_step_of_chunk_1 = chunk_1[:, -1, :].clone()

        modifier.actor = _FixedActor(torch.ones(1, CHUNK_LENGTH, ACTION_DIM))
        chunk_2 = _run_full_chunk(modifier, ref_chunk)

        first_step_jump = (chunk_2[:, 0, :] - last_step_of_chunk_1).abs()
        assert first_step_jump.max().item() <= 0.05 + 1e-4

    def test_tracks_last_action_through_vla_phase_too(self):
        """The first RL-phase step right after a VLA->RL handoff must be
        limited against the last VLA action, not against nothing (which
        would let the handoff itself produce an unlimited jump)."""
        phase_ctrl = PhaseController(mode="manual")  # starts in VLA phase
        modifier = _make_modifier(
            mu_chunk=torch.ones(1, CHUNK_LENGTH, ACTION_DIM),
            slew_rate_limit=0.05,
            phase_ctrl=phase_ctrl,
        )
        vla_chunk = torch.zeros(1, CHUNK_LENGTH, ACTION_DIM)
        vla_out = _run_full_chunk(modifier, vla_chunk)
        last_vla_step = vla_out[:, -1, :].clone()

        phase_ctrl.trigger_critical()
        rl_chunk = _run_full_chunk(modifier, vla_chunk)  # ref stays 0; actor wants 1.0
        first_rl_step_jump = (rl_chunk[:, 0, :] - last_vla_step).abs()
        assert first_rl_step_jump.max().item() <= 0.05 + 1e-4

    def test_none_slew_rate_is_a_no_op(self):
        ref_chunk = torch.zeros(1, CHUNK_LENGTH, ACTION_DIM)
        mu_chunk = torch.zeros(1, CHUNK_LENGTH, ACTION_DIM)
        mu_chunk[0, 1, :] = 1.0
        modifier = _make_modifier(mu_chunk=mu_chunk, slew_rate_limit=None)
        chunk = _compute(modifier, ref_chunk)
        assert torch.allclose(chunk, mu_chunk)


class TestReset:
    def test_reset_clears_last_executed_action(self):
        ref_chunk = torch.zeros(1, CHUNK_LENGTH, ACTION_DIM)
        modifier = _make_modifier(mu_chunk=ref_chunk, slew_rate_limit=0.05)
        _run_full_chunk(modifier, ref_chunk)
        assert modifier._last_executed_action is not None
        modifier.reset()
        assert modifier._last_executed_action is None


class TestLastExecutedActionTiming:
    """Regression tests for the exact reported bug: _last_executed_action
    must reflect what was really popped/dispatched for execution, not what
    a chunk happened to contain when merely computed -- a chunk can be
    interrupted (queue cleared) after only some of its frames were ever
    actually consumed, e.g. the user switches VLA->RL after only the first
    of a 25-frame chunk has run."""

    def test_compute_chunk_alone_does_not_set_last_executed_action(self):
        """Computing a chunk (without enqueueing/popping anything from it)
        must not by itself update the slew-rate continuity reference."""
        ref_chunk = torch.zeros(1, CHUNK_LENGTH, ACTION_DIM)
        modifier = _make_modifier(mu_chunk=ref_chunk, slew_rate_limit=0.05)
        _compute(modifier, ref_chunk)
        assert modifier._last_executed_action is None

    def test_interrupted_chunk_uses_the_real_last_popped_frame(self):
        """A 3-frame chunk is computed with a big internal swing (0 -> 1 ->
        -1), but only frame 0 is actually popped/executed before the chunk
        is interrupted (queue cleared, frames 1-2 discarded, never run).
        The next chunk's slew-limiting must be anchored to frame 0 (what
        really happened), not frame -1 at index 2 (the chunk's last frame,
        never executed) -- the exact bug reported."""
        ref_chunk = torch.zeros(1, CHUNK_LENGTH, ACTION_DIM)
        mu_chunk = torch.zeros(1, CHUNK_LENGTH, ACTION_DIM)
        mu_chunk[0, 1, :] = 1.0
        mu_chunk[0, 2, :] = -1.0  # chunk's last frame -- must never be used
        modifier = _make_modifier(mu_chunk=mu_chunk, slew_rate_limit=None)
        chunk = _compute(modifier, ref_chunk)
        modifier.enqueue(chunk)
        frame_0 = modifier.pop_action()  # only one frame actually consumed
        assert torch.allclose(modifier._last_executed_action, frame_0)
        modifier._action_queue.clear()  # simulates interrupt_chunk()

        modifier.slew_rate_limit = 0.05
        modifier.actor = _FixedActor(torch.ones(1, CHUNK_LENGTH, ACTION_DIM))
        next_chunk = _compute(modifier, ref_chunk)
        first_step_jump = (next_chunk[:, 0, :] - frame_0).abs()
        assert first_step_jump.max().item() <= 0.05 + 1e-4

    def test_record_executed_action_updates_continuity_reference(self):
        """Simulates the human-intervention path (loop.py calls this
        directly; pop_action() is never invoked while intervention is
        active, since policy inference is skipped entirely then)."""
        modifier = _make_modifier(
            mu_chunk=torch.zeros(1, CHUNK_LENGTH, ACTION_DIM), slew_rate_limit=0.05,
        )
        human_action = torch.full((1, ACTION_DIM), 0.9)
        modifier.record_executed_action(human_action)
        assert torch.equal(modifier._last_executed_action, human_action)

        ref_chunk = torch.zeros(1, CHUNK_LENGTH, ACTION_DIM)
        modifier.actor = _FixedActor(torch.zeros(1, CHUNK_LENGTH, ACTION_DIM))
        chunk = _compute(modifier, ref_chunk)
        first_step_jump = (chunk[:, 0, :] - human_action).abs()
        assert first_step_jump.max().item() <= 0.05 + 1e-4

    def test_recorded_action_is_aligned_to_inference_dtype(self):
        """Robot-reported actions can differ from inference tensors in
        device/dtype (CPU vs CUDA in deployment). The slew reference must be
        aligned without changing the generated chunk's dtype."""
        modifier = _make_modifier(
            mu_chunk=torch.zeros(1, CHUNK_LENGTH, ACTION_DIM), slew_rate_limit=0.05,
        )
        modifier.record_executed_action(torch.full((1, ACTION_DIM), 0.9, dtype=torch.float64))

        ref_chunk = torch.zeros(1, CHUNK_LENGTH, ACTION_DIM, dtype=torch.float32)
        chunk = _compute(modifier, ref_chunk)

        assert chunk.dtype == ref_chunk.dtype
        assert (chunk[:, 0, :] - 0.9).abs().max().item() <= 0.05 + 1e-4
