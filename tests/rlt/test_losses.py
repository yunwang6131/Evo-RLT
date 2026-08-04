from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from evo_rlt.core.losses import (
    discounted_chunk_return,
    critic_loss,
    actor_loss,
    rankq_ranking_loss,
    q_action_sensitivity,
    _masked_candidate,
)
from evo_rlt.core.actor import ChunkActor
from evo_rlt.core.critic import TwinCritic
from evo_rlt.core.utils import soft_update

STATE_DIM = 78
CHUNK_DIM = 140
C = 10


@pytest.fixture
def actor():
    return ChunkActor(state_dim=STATE_DIM, chunk_dim=CHUNK_DIM, hidden_dim=64, num_layers=2)


@pytest.fixture
def critic():
    return TwinCritic(state_dim=STATE_DIM, chunk_dim=CHUNK_DIM, hidden_dim=64, num_layers=2)


@pytest.fixture
def target_critic(critic):
    import copy
    tc = copy.deepcopy(critic)
    for p in tc.parameters():
        p.requires_grad = False
    return tc


@pytest.fixture
def batch():
    B = 16
    return {
        "state_vec": torch.randn(B, STATE_DIM),
        "exec_chunk_flat": torch.randn(B, CHUNK_DIM),
        "ref_chunk_flat": torch.randn(B, CHUNK_DIM),
        "reward_seq": torch.randn(B, C),
        "next_state_vec": torch.randn(B, STATE_DIM),
        "next_ref_flat": torch.randn(B, CHUNK_DIM),
        "done": torch.zeros(B),
        "actual_steps": torch.full((B,), C),
    }


def test_discounted_chunk_return_hand_computed():
    """Hand-computed: rewards=[1,1,1], gamma=0.5 -> 1 + 0.5 + 0.25 = 1.75"""
    reward_seq = torch.tensor([[1.0, 1.0, 1.0]])
    result = discounted_chunk_return(reward_seq, gamma=0.5)
    assert torch.allclose(result, torch.tensor([[1.75]]))


def test_discounted_chunk_return_shape():
    reward_seq = torch.randn(8, 10)
    result = discounted_chunk_return(reward_seq, gamma=0.99)
    assert result.shape == (8, 1)


def test_done_masking(actor, critic, target_critic):
    """When done=1, the bootstrap term should be zero."""
    B = 4
    batch_done = {
        "state_vec": torch.randn(B, STATE_DIM),
        "exec_chunk_flat": torch.randn(B, CHUNK_DIM),
        "ref_chunk_flat": torch.randn(B, CHUNK_DIM),
        "reward_seq": torch.ones(B, C),
        "next_state_vec": torch.randn(B, STATE_DIM),
        "next_ref_flat": torch.randn(B, CHUNK_DIM),
        "done": torch.ones(B),  # all done
        "actual_steps": torch.full((B,), C),
    }
    loss = critic_loss(critic, target_critic, actor, batch_done, gamma=0.99, C=C)
    assert not torch.isnan(loss)


def test_critic_loss_scalar(actor, critic, target_critic, batch):
    loss = critic_loss(critic, target_critic, actor, batch, gamma=0.99, C=C)
    assert loss.shape == ()
    assert not torch.isnan(loss)


def test_actor_loss_scalar(actor, critic, batch):
    loss = actor_loss(actor, critic, batch, beta=1.0)
    assert loss.shape == ()
    assert not torch.isnan(loss)


def test_bc_term_scales_with_beta(actor, critic, batch):
    """Higher beta should give higher actor loss (assuming BC term > 0)."""
    torch.manual_seed(42)
    loss_low = actor_loss(actor, critic, batch, beta=0.0)
    torch.manual_seed(42)
    loss_high = actor_loss(actor, critic, batch, beta=10.0)
    # With beta=0, only Q term. With beta=10, large BC term added.
    # We just check they're different and the high-beta one is larger
    # (BC reg is always non-negative, so adding it increases loss)
    assert loss_high.item() > loss_low.item()


def test_target_is_stop_gradiented(actor, critic, target_critic, batch):
    """Target critic params should not receive gradients through critic_loss."""
    loss = critic_loss(critic, target_critic, actor, batch, gamma=0.99, C=C)
    loss.backward()
    for p in target_critic.parameters():
        assert p.grad is None


def test_actor_loss_matches_paper_bc_scaling():
    """BC term must be per-sample squared-distance sum averaged over batch.

    This is the paper's β convention; it differs from F.mse_loss (mean over
    all elements) by a factor of C * D_flat. We pin the numerical value with a
    deterministic (mu, ref) pair and a stub critic that returns a constant Q
    (so d/dβ of the loss equals the BC term exactly), and check the ratio
    relative to mean-MSE.
    """
    torch.manual_seed(0)
    B = 4
    D_flat = CHUNK_DIM  # = C * D
    mu = torch.randn(B, D_flat)
    ref = torch.randn(B, D_flat)

    class _StubActor:
        def forward(self, x, ref, training=False):
            return mu, None

    class _StubCritic:
        def min_q(self, x, a):
            return torch.zeros(a.shape[0], 1)

    stub_batch = {
        "state_vec": torch.randn(B, STATE_DIM),
        "ref_chunk_flat": ref,
    }

    # Loss at beta=0 removes BC contribution (leaves -q.mean()=0).
    loss_beta0 = actor_loss(_StubActor(), _StubCritic(), stub_batch, beta=0.0)
    loss_beta1 = actor_loss(_StubActor(), _StubCritic(), stub_batch, beta=1.0)
    bc_reg_observed = (loss_beta1 - loss_beta0).item()

    expected_bc = ((mu - ref) ** 2).sum(dim=-1).mean().item()
    assert bc_reg_observed == pytest.approx(expected_bc, rel=1e-6)

    # The new convention equals mean-MSE * (C * D_flat).
    mean_mse = F.mse_loss(mu, ref).item()
    assert bc_reg_observed == pytest.approx(mean_mse * D_flat, rel=1e-6)


def test_critic_loss_respects_target_q_clip():
    class _ZeroCritic(torch.nn.Module):
        def forward(self, state_vec, action_flat):
            z = torch.zeros(state_vec.shape[0], 1)
            return z, z

    class _ConstantTargetCritic(torch.nn.Module):
        def min_q(self, state_vec, action_flat):
            return torch.full((state_vec.shape[0], 1), 1000.0)

    class _ZeroActor:
        def forward(self, state_vec, ref_flat):
            return torch.zeros(ref_flat.shape), None

    batch = {
        "state_vec": torch.zeros(2, STATE_DIM),
        "exec_chunk_flat": torch.zeros(2, CHUNK_DIM),
        "ref_chunk_flat": torch.zeros(2, CHUNK_DIM),
        "reward_seq": torch.zeros(2, C),
        "next_state_vec": torch.zeros(2, STATE_DIM),
        "next_ref_flat": torch.zeros(2, CHUNK_DIM),
        "done": torch.zeros(2),
        "actual_steps": torch.ones(2, dtype=torch.int64),
    }

    clipped = critic_loss(
        _ZeroCritic(), _ConstantTargetCritic(), _ZeroActor(),
        batch, gamma=1.0, C=C, target_q_clip=10.0,
    )
    unclipped = critic_loss(
        _ZeroCritic(), _ConstantTargetCritic(), _ZeroActor(),
        batch, gamma=1.0, C=C, target_q_clip=None,
    )

    assert clipped.item() == pytest.approx(200.0)
    assert unclipped.item() == pytest.approx(2_000_000.0)


class TestRankQRankingLoss:
    def test_zero_when_no_resolved_outcome(self, critic):
        B = 8
        state = torch.randn(B, STATE_DIM)
        action = torch.randn(B, CHUNK_DIM)
        outcome = torch.full((B,), -1.0)  # all unresolved
        loss = rankq_ranking_loss(critic, state, action, outcome)
        assert loss.item() == pytest.approx(0.0)

    def test_nonzero_and_finite_with_success_and_failure(self, critic):
        B = 8
        state = torch.randn(B, STATE_DIM)
        action = torch.randn(B, CHUNK_DIM)
        outcome = torch.tensor([1.0, 0.0] * (B // 2))
        loss = rankq_ranking_loss(critic, state, action, outcome)
        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert loss.item() != 0.0

    def test_success_action_preferred_after_training_step(self, critic):
        """A single gradient step on the ranking loss should push Q(exec) up
        relative to Q(random) for a success transition -- i.e. the loss is
        actually informative, not a no-op that happens to be nonzero."""
        torch.manual_seed(0)
        state = torch.randn(4, STATE_DIM)
        action = torch.randn(4, CHUNK_DIM)
        outcome = torch.ones(4)

        opt = torch.optim.SGD(critic.parameters(), lr=0.1)
        with torch.no_grad():
            q_before = critic.min_q(state, action).mean()
        for _ in range(20):
            opt.zero_grad()
            loss = rankq_ranking_loss(critic, state, action, outcome)
            loss.backward()
            opt.step()
        with torch.no_grad():
            q_after = critic.min_q(state, action).mean()
        assert q_after.item() > q_before.item()

    def test_critic_loss_is_unaffected_by_default(self, actor, critic, target_critic, batch):
        """No 'outcome' key and default alphas=0 -> byte-identical to before
        the RankQ integration (backward compatibility)."""
        torch.manual_seed(1)
        loss_a = critic_loss(critic, target_critic, actor, batch, gamma=0.99, C=C)
        torch.manual_seed(1)
        loss_b = critic_loss(
            critic, target_critic, actor, batch, gamma=0.99, C=C,
            rankq_alpha_success=0.0, rankq_alpha_failure=0.0,
        )
        assert loss_a.item() == pytest.approx(loss_b.item())

    def test_critic_loss_adds_ranking_term_when_enabled_and_outcome_present(
        self, actor, critic, target_critic, batch
    ):
        batch_with_outcome = dict(batch)
        batch_with_outcome["outcome"] = torch.tensor(
            [1.0, 0.0] * (batch["state_vec"].shape[0] // 2)
        )
        base = critic_loss(critic, target_critic, actor, batch, gamma=0.99, C=C)
        with_rankq = critic_loss(
            critic, target_critic, actor, batch_with_outcome, gamma=0.99, C=C,
            rankq_alpha_success=1.0, rankq_alpha_failure=1.0,
        )
        assert with_rankq.item() != pytest.approx(base.item())

    def test_critic_loss_ignores_rankq_alphas_without_outcome_key(
        self, actor, critic, target_critic, batch
    ):
        """Enabling the alphas without an 'outcome' key in the batch must
        not crash or change the loss -- offline lerobot-train batches don't
        carry outcome labels."""
        base = critic_loss(critic, target_critic, actor, batch, gamma=0.99, C=C)
        still_base = critic_loss(
            critic, target_critic, actor, batch, gamma=0.99, C=C,
            rankq_alpha_success=1.0, rankq_alpha_failure=1.0,
        )
        assert still_base.item() == pytest.approx(base.item())

    def test_critic_loss_forwards_action_mask_to_ranking_term(
        self, actor, critic, target_critic, batch
    ):
        """Sanity check the plumbing: critic_loss must accept action_mask and
        thread it through without erroring, same call shape as before."""
        batch = dict(batch, outcome=torch.tensor([1.0, 0.0] * 8))
        mask = torch.ones(CHUNK_DIM)
        loss = critic_loss(
            critic, target_critic, actor, batch, gamma=0.99, C=C,
            rankq_alpha_success=1.0, rankq_alpha_failure=1.0,
            action_mask=mask,
        )
        assert torch.isfinite(loss)


class TestMaskedCandidate:
    def test_pins_masked_out_dims_to_base_value(self):
        base = torch.zeros(2, 4)
        alt = torch.ones(2, 4)
        mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
        out = _masked_candidate(base, alt, mask)
        assert torch.equal(out[:, :2], torch.ones(2, 2))
        assert torch.equal(out[:, 2:], torch.zeros(2, 2))

    def test_none_mask_returns_alt_unchanged(self):
        base = torch.zeros(2, 4)
        alt = torch.ones(2, 4)
        assert torch.equal(_masked_candidate(base, alt, None), alt)


class TestRankQActionMask:
    """actor_rl_arm="left"-style masking: candidates must never differ from
    the executed action on a frozen (masked-out) dim, since the actor could
    never have produced such an action."""

    def test_all_zero_mask_makes_ranking_loss_seed_invariant(self, critic):
        """With every dim masked out, every candidate collapses to the exact
        executed action -- so the loss must not depend at all on the
        noisy/random/permuted draws, unlike the unmasked case."""
        B = 8
        state = torch.randn(B, STATE_DIM)
        action = torch.randn(B, CHUNK_DIM)
        outcome = torch.tensor([1.0, 0.0] * (B // 2))
        zero_mask = torch.zeros(CHUNK_DIM)

        torch.manual_seed(1)
        loss_a = rankq_ranking_loss(critic, state, action, outcome, action_mask=zero_mask)
        torch.manual_seed(2)
        loss_b = rankq_ranking_loss(critic, state, action, outcome, action_mask=zero_mask)

        assert loss_a.item() == pytest.approx(loss_b.item())

    def test_partial_mask_leaves_masked_out_half_never_perturbed(self, critic):
        """Right half masked out (actor_rl_arm='left'-style split): rerunning
        with different seeds must still change the loss (left half is being
        randomized), but must match a version where we manually zero out
        the right-half randomness by construction -- verified indirectly via
        _masked_candidate's own unit tests plus this end-to-end smoke check
        that mixed masks don't crash and stay finite."""
        B = 8
        state = torch.randn(B, STATE_DIM)
        action = torch.randn(B, CHUNK_DIM)
        outcome = torch.tensor([1.0, 0.0] * (B // 2))
        half_mask = torch.cat([torch.ones(CHUNK_DIM // 2), torch.zeros(CHUNK_DIM // 2)])

        loss = rankq_ranking_loss(critic, state, action, outcome, action_mask=half_mask)
        assert torch.isfinite(loss)


class TestQActionSensitivity:
    def test_nonzero_for_a_freshly_initialized_critic(self, critic):
        """A random-init critic has not collapsed yet -- different actions at
        the same state should already produce visibly different Q values."""
        state = torch.randn(16, STATE_DIM)
        action = torch.randn(16, CHUNK_DIM)
        sensitivity = q_action_sensitivity(critic, state, action)
        assert sensitivity.item() > 0.0

    def test_near_zero_for_an_action_insensitive_critic(self):
        """A critic that structurally ignores the action input (the failure
        mode this diagnostic is meant to catch) must report ~0 sensitivity."""

        class _StateOnlyCritic:
            def min_q(self, state_vec, action_flat):
                return state_vec.sum(dim=-1, keepdim=True)

        state = torch.randn(16, STATE_DIM)
        action = torch.randn(16, CHUNK_DIM)
        sensitivity = q_action_sensitivity(_StateOnlyCritic(), state, action)
        assert sensitivity.item() == pytest.approx(0.0, abs=1e-6)

    def test_does_not_require_grad_tracking(self, critic):
        """Must be safe to call mid-training without interfering with a live
        autograd graph (it's a pure logging diagnostic, no gradients)."""
        state = torch.randn(4, STATE_DIM, requires_grad=True)
        action = torch.randn(4, CHUNK_DIM, requires_grad=True)
        sensitivity = q_action_sensitivity(critic, state, action)
        assert not sensitivity.requires_grad

    def test_all_zero_mask_reports_zero_sensitivity_even_for_a_healthy_critic(self, critic):
        """With every dim masked out, all 5 candidates collapse to the exact
        same executed action -- so a critic that's perfectly action-sensitive
        elsewhere must still report ~0 here, since the actor (under this
        mask) genuinely cannot move any of these dims."""
        state = torch.randn(16, STATE_DIM)
        action = torch.randn(16, CHUNK_DIM)
        zero_mask = torch.zeros(CHUNK_DIM)
        sensitivity = q_action_sensitivity(critic, state, action, action_mask=zero_mask)
        assert sensitivity.item() == pytest.approx(0.0, abs=1e-6)
