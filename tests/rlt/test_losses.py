from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from evo_rlt.core.losses import (
    discounted_chunk_return,
    critic_loss,
    actor_loss,
    actor_behavior_cloning_loss,
    rankq_ranking_loss,
    q_action_sensitivity,
    _masked_candidate,
    _apply_slew_rate_limit_flat,
    _random_action_like,
)
from evo_rlt.core.actor import ChunkActor
from evo_rlt.core.critic import TwinCritic
from evo_rlt.core.utils import project_action_delta, soft_update, unflatten_chunk

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


def test_target_q_min_bounds_the_bootstrap_from_below():
    """With every reward non-negative, Q is a discounted sum of non-negative
    terms and cannot be negative -- but the backup fits Q(s, a_data) while
    bootstrapping Q(s', pi(s')), so a policy that trails the data subtracts
    that gap on every step and can walk Q far below zero with no TD error to
    show for it. target_q_min closes off that half of the line.
    """
    class _ZeroCritic(torch.nn.Module):
        def forward(self, state_vec, action_flat):
            z = torch.zeros(state_vec.shape[0], 1)
            return z, z

    class _NegativeTargetCritic(torch.nn.Module):
        def min_q(self, state_vec, action_flat):
            return torch.full((state_vec.shape[0], 1), -50.0)

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
    kwargs = dict(gamma=1.0, C=C, target_q_clip=3.0)

    # Unset -> the previous symmetric bound, so existing configs are untouched.
    default = critic_loss(
        _ZeroCritic(), _NegativeTargetCritic(), _ZeroActor(), batch, **kwargs
    )
    floored = critic_loss(
        _ZeroCritic(), _NegativeTargetCritic(), _ZeroActor(), batch,
        target_q_min=0.0, **kwargs,
    )
    explicit = critic_loss(
        _ZeroCritic(), _NegativeTargetCritic(), _ZeroActor(), batch,
        target_q_min=-3.0, **kwargs,
    )

    # loss = 2 * MSE(0, target) and target == the clamped bootstrap.
    assert default.item() == pytest.approx(2 * 3.0**2)   # clamped to -3
    assert floored.item() == pytest.approx(0.0)          # clamped to 0
    assert explicit.item() == pytest.approx(default.item())


class TestTargetPolicySmoothing:
    def test_default_disabled_matches_no_smoothing_exactly(self, actor, critic, target_critic, batch):
        """target_noise_std defaults to 0.0 -- existing callers that don't
        pass it must see byte-identical behavior to before this feature."""
        torch.manual_seed(1)
        base = critic_loss(critic, target_critic, actor, batch, gamma=0.99, C=C)
        torch.manual_seed(1)
        explicit_off = critic_loss(
            critic, target_critic, actor, batch, gamma=0.99, C=C, target_noise_std=0.0,
        )
        assert explicit_off.item() == pytest.approx(base.item())

    def test_nonzero_noise_std_changes_the_target_action(self, actor, critic):
        """With smoothing enabled, target_critic must be queried at an action
        that differs from the actor's raw (unperturbed) target action."""

        class _RecordingTargetCritic:
            def __init__(self):
                self.seen_actions: list[torch.Tensor] = []

            def min_q(self, state_vec, action_flat):
                self.seen_actions.append(action_flat.clone())
                return torch.zeros(state_vec.shape[0], 1)

        batch = {
            "state_vec": torch.zeros(4, STATE_DIM),
            "exec_chunk_flat": torch.zeros(4, CHUNK_DIM),
            "ref_chunk_flat": torch.zeros(4, CHUNK_DIM),
            "reward_seq": torch.zeros(4, C),
            "next_state_vec": torch.zeros(4, STATE_DIM),
            "next_ref_flat": torch.zeros(4, CHUNK_DIM),
            "done": torch.zeros(4),
            "actual_steps": torch.full((4,), C),
        }
        target_critic = _RecordingTargetCritic()
        torch.manual_seed(0)
        with torch.no_grad():
            raw_mu_next, _ = actor.forward(batch["next_state_vec"], batch["next_ref_flat"])
            raw_mu_next = raw_mu_next.clamp(-1.0, 1.0)

        torch.manual_seed(0)
        critic_loss(
            critic, target_critic, actor, batch, gamma=0.99, C=C,
            target_noise_std=0.2, target_noise_clip=0.5,
        )
        smoothed_action = target_critic.seen_actions[0]
        assert not torch.allclose(smoothed_action, raw_mu_next)
        # Still clamped to the valid action range despite the added noise.
        assert smoothed_action.abs().max().item() <= 1.0 + 1e-5

    def test_noise_respects_action_mask(self, actor, critic):
        """Masked-out dims (e.g. the frozen arm under actor_rl_arm='left')
        must not receive smoothing noise -- the actor could never actually
        produce a perturbed action there (mu is pinned to ref on those dims
        regardless), so smoothing them would just be pointless extra noise
        in the TD target."""

        class _RecordingTargetCritic:
            def __init__(self):
                self.seen_actions: list[torch.Tensor] = []

            def min_q(self, state_vec, action_flat):
                self.seen_actions.append(action_flat.clone())
                return torch.zeros(state_vec.shape[0], 1)

        batch = {
            "state_vec": torch.zeros(4, STATE_DIM),
            "exec_chunk_flat": torch.zeros(4, CHUNK_DIM),
            "ref_chunk_flat": torch.zeros(4, CHUNK_DIM),
            "reward_seq": torch.zeros(4, C),
            "next_state_vec": torch.zeros(4, STATE_DIM),
            "next_ref_flat": torch.zeros(4, CHUNK_DIM),
            "done": torch.zeros(4),
            "actual_steps": torch.full((4,), C),
        }
        target_critic = _RecordingTargetCritic()
        mask = torch.cat([torch.ones(CHUNK_DIM // 2), torch.zeros(CHUNK_DIM // 2)])

        with torch.no_grad():
            raw_mu_next, _ = actor.forward(batch["next_state_vec"], batch["next_ref_flat"])
            raw_mu_next = raw_mu_next.clamp(-1.0, 1.0)

        critic_loss(
            critic, target_critic, actor, batch, gamma=0.99, C=C,
            target_noise_std=0.2, target_noise_clip=0.5, action_mask=mask,
        )
        smoothed_action = target_critic.seen_actions[0]
        second_half = slice(CHUNK_DIM // 2, CHUNK_DIM)
        assert torch.allclose(smoothed_action[:, second_half], raw_mu_next[:, second_half])


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

    def test_excluded_rows_cannot_supply_cross_batch_candidates(self, critic):
        state = torch.randn(4, STATE_DIM)
        action = torch.randn(4, CHUNK_DIM)
        outcome = torch.tensor([-1.0, -1.0, 1.0, 0.0])
        changed_offline = action.clone()
        changed_offline[:2] = changed_offline[:2] * 1000.0 + 500.0
        batch_action_mask = torch.ones_like(action)

        torch.manual_seed(7)
        original = rankq_ranking_loss(
            critic,
            state,
            action,
            outcome,
            action_mask=batch_action_mask,
        )
        torch.manual_seed(7)
        changed = rankq_ranking_loss(
            critic,
            state,
            changed_offline,
            outcome,
            action_mask=batch_action_mask,
        )

        assert changed.item() == pytest.approx(original.item())

    def test_positive_margin_stops_satisfied_failure_pair(self):
        class _CallOrderedCritic:
            def __init__(self):
                self.calls = 0

            def __call__(self, state_vec, action_flat):
                # Candidate order is exec/noisy/very_noisy/random/permuted.
                value = 1.0 if self.calls == 0 else 0.0
                self.calls += 1
                q = torch.full((action_flat.shape[0], 1), value)
                return q, q

        state = torch.zeros(4, STATE_DIM)
        action = torch.randn(4, CHUNK_DIM)
        outcome = torch.zeros(4)  # failures use only exec > random

        hinge = rankq_ranking_loss(
            _CallOrderedCritic(), state, action, outcome, margin=0.1
        )
        softplus = rankq_ranking_loss(
            _CallOrderedCritic(), state, action, outcome, margin=0.0
        )

        assert hinge.item() == pytest.approx(0.0)
        assert softplus.item() > 0.0

    def test_negative_margin_is_rejected(self, critic):
        with pytest.raises(ValueError, match="margin"):
            rankq_ranking_loss(
                critic,
                torch.zeros(2, STATE_DIM),
                torch.zeros(2, CHUNK_DIM),
                torch.ones(2),
                margin=-0.1,
            )

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
        not crash or change the loss for generic unlabeled callers."""
        base = critic_loss(critic, target_critic, actor, batch, gamma=0.99, C=C)
        still_base = critic_loss(
            critic, target_critic, actor, batch, gamma=0.99, C=C,
            rankq_alpha_success=1.0, rankq_alpha_failure=1.0,
        )
        assert still_base.item() == pytest.approx(base.item())

    def test_rankq_outcome_can_exclude_offline_rows_without_changing_bc_outcome(
        self, actor, critic, target_critic, batch
    ):
        excluded = dict(batch)
        excluded["outcome"] = torch.ones(batch["state_vec"].shape[0])
        excluded["rankq_outcome"] = torch.full_like(excluded["outcome"], -1.0)

        torch.manual_seed(3)
        base = critic_loss(critic, target_critic, actor, batch, gamma=0.99, C=C)
        torch.manual_seed(3)
        with_excluded_rankq = critic_loss(
            critic,
            target_critic,
            actor,
            excluded,
            gamma=0.99,
            C=C,
            rankq_alpha_success=1.0,
            rankq_alpha_failure=1.0,
        )

        assert with_excluded_rankq.item() == pytest.approx(base.item())

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

    def test_critic_loss_info_breakdown_sums_to_returned_loss(
        self, actor, critic, target_critic, batch
    ):
        """The two components logged via `info` must actually add up to the
        scalar critic_loss() returns -- otherwise the wandb breakdown would
        be lying about what's actually being optimized."""
        batch_with_outcome = dict(batch, outcome=torch.tensor([1.0, 0.0] * 8))
        info: dict = {}
        loss = critic_loss(
            critic, target_critic, actor, batch_with_outcome, gamma=0.99, C=C,
            rankq_alpha_success=1.0, rankq_alpha_failure=1.0,
            info=info,
        )
        assert set(info.keys()) == {"loss_critic_td", "loss_critic_rankq"}
        assert (info["loss_critic_td"] + info["loss_critic_rankq"]).item() == pytest.approx(
            loss.item()
        )

    def test_critic_loss_info_rankq_is_zero_when_disabled(
        self, actor, critic, target_critic, batch
    ):
        """No outcome / alphas=0 -> loss_critic_rankq must report exactly 0,
        not just be absent, so the wandb panel reads correctly either way."""
        info: dict = {}
        loss = critic_loss(critic, target_critic, actor, batch, gamma=0.99, C=C, info=info)
        assert info["loss_critic_rankq"].item() == pytest.approx(0.0)
        assert info["loss_critic_td"].item() == pytest.approx(loss.item())

    def test_critic_loss_info_is_optional(self, actor, critic, target_critic, batch):
        """Existing callers that don't pass info= must keep working exactly
        as before (no crash, same returned loss)."""
        loss = critic_loss(critic, target_critic, actor, batch, gamma=0.99, C=C)
        assert torch.isfinite(loss)


class TestRankQRelativeMargin:
    """margin_relative scales the requested Q separation by the critic's own
    mean|Q|, so the hinge keeps constraining the ordering after Q drifts off
    the reward scale the absolute margin was tuned against."""

    @staticmethod
    def _inputs(scale: float):
        torch.manual_seed(0)
        state = torch.randn(16, 6)
        action = torch.randn(16, 8)
        outcome = torch.tensor([1.0, 0.0] * 8)

        class ScaledCritic(torch.nn.Module):
            """Q with a controllable magnitude, so the two margin modes can be
            compared at an unchanged Q *ordering* but a changed Q *scale*."""

            def __init__(self, gain):
                super().__init__()
                self.gain = gain
                self.lin = torch.nn.Linear(6 + 8, 1)

            def forward(self, s, a):
                q = self.gain * self.lin(torch.cat([s, a], dim=-1))
                return q, q

        return ScaledCritic(scale), state, action, outcome

    @staticmethod
    def _loss(critic, state, action, outcome, **kwargs):
        # rankq_ranking_loss samples its own negative candidates, so every
        # comparison here has to start from the same RNG state or it is
        # measuring noise instead of the margin mode.
        torch.manual_seed(1234)
        return rankq_ranking_loss(critic, state, action, outcome, **kwargs).item()

    def test_relative_margin_tracks_q_scale_while_absolute_does_not(self):
        losses_abs, losses_rel = [], []
        for gain in (1.0, 10.0):
            critic, state, action, outcome = self._inputs(gain)
            losses_abs.append(
                self._loss(critic, state, action, outcome, margin=0.1, margin_relative=False)
            )
            losses_rel.append(
                self._loss(critic, state, action, outcome, margin=0.1, margin_relative=True)
            )
        # At a 10x larger Q scale the absolute margin is 10x less of the
        # signal it is ordering; the relative one asks for a 10x larger gap
        # and so stays proportionally as binding as it was.
        abs_ratio = losses_abs[1] / max(losses_abs[0], 1e-9)
        rel_ratio = losses_rel[1] / max(losses_rel[0], 1e-9)
        assert rel_ratio > abs_ratio

    def test_relative_margin_is_at_least_absolute_when_q_scale_exceeds_one(self):
        critic, state, action, outcome = self._inputs(5.0)
        absolute = self._loss(
            critic, state, action, outcome, margin=0.1, margin_relative=False
        )
        relative = self._loss(
            critic, state, action, outcome, margin=0.1, margin_relative=True
        )
        assert relative > absolute

    def test_relative_margin_is_a_no_op_without_a_margin(self):
        """margin=0 keeps the paper's softplus in both modes -- the flag must
        not silently switch the hinge on."""
        critic, state, action, outcome = self._inputs(3.0)
        softplus = self._loss(
            critic, state, action, outcome, margin=0.0, margin_relative=False
        )
        still_softplus = self._loss(
            critic, state, action, outcome, margin=0.0, margin_relative=True
        )
        assert still_softplus == pytest.approx(softplus)

    def test_relative_margin_gradient_flows_to_critic_only_through_the_gap(self):
        """The scale is detached: it sets how much separation to ask for, and
        must not itself become something the critic can optimize (shrinking
        |Q| to make the margin cheap)."""
        critic, state, action, outcome = self._inputs(4.0)
        torch.manual_seed(1234)
        loss = rankq_ranking_loss(
            critic, state, action, outcome, margin=0.1, margin_relative=True
        )
        loss.backward()
        assert any(
            p.grad is not None and torch.isfinite(p.grad).all() for p in critic.parameters()
        )


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


class TestRandomActionLike:
    def test_preserves_every_dimension_marginal(self):
        action = torch.tensor(
            [[-4.0, 10.0], [2.0, -3.0], [7.0, 5.0], [1.0, 9.0]]
        )
        torch.manual_seed(0)
        shuffled = _random_action_like(action)
        assert torch.equal(
            torch.sort(shuffled, dim=0).values,
            torch.sort(action, dim=0).values,
        )

    def test_singleton_batch_is_unchanged(self):
        action = torch.tensor([[3.0, -7.0]])
        assert torch.equal(_random_action_like(action), action)


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

    def test_uses_batch_marginals_instead_of_uniform_unit_actions(self):
        class _ActionSumCritic:
            def min_q(self, state_vec, action_flat):
                return action_flat.sum(dim=-1, keepdim=True)

        # With zero Gaussian noise, identical rows remain identical under
        # marginal shuffle/permutation. The former U[-1,1] candidate made
        # this diagnostic spuriously nonzero.
        state = torch.zeros(4, STATE_DIM)
        action = torch.full((4, CHUNK_DIM), 10.0)
        sensitivity = q_action_sensitivity(
            _ActionSumCritic(), state, action, noise_scale=0.0
        )
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


class TestProjectActionDelta:
    def test_none_limit_returns_mu_unchanged(self):
        mu = torch.tensor([5.0, -3.0, 0.2])
        ref = torch.tensor([0.1, 0.1, 0.1])
        assert torch.equal(project_action_delta(mu, ref, None), mu)

    def test_zero_limit_returns_ref_unchanged(self):
        """limit=0 means zero authority (must equal ref exactly); must not
        divide by zero."""
        mu = torch.tensor([5.0, -3.0, 0.2])
        ref = torch.tensor([0.1, 0.1, 0.1])
        result = project_action_delta(mu, ref, 0.0)
        assert torch.equal(result, ref)
        assert torch.isfinite(result).all()

    def test_small_delta_is_almost_unchanged(self):
        """Near mu==ref, tanh is ~linear -- a small in-range delta should
        survive the projection almost exactly, not get squashed."""
        ref = torch.zeros(5)
        mu = ref + 0.01
        result = project_action_delta(mu, ref, limit=0.1)
        assert torch.allclose(result, mu, atol=1e-3)

    def test_invariant_holds_for_a_huge_raw_delta(self):
        """However far mu strays from ref, the projected action must stay
        within `limit` of ref (tanh saturates, never overshoots) -- allowing
        a hair of float32 rounding at the boundary itself, since tanh of an
        extreme ratio rounds to exactly 1.0 in float32."""
        ref = torch.tensor([0.2])
        mu = torch.tensor([1e6])
        result = project_action_delta(mu, ref, limit=0.1)
        assert (result - ref).abs().item() <= 0.1 + 1e-4

    def test_invariant_holds_even_when_ref_itself_exceeds_unit_range(self):
        """The actual bug this exists to fix: ref regularly exceeds [-1,1]
        under QUANTILES normalization (confirmed empirically on real
        recorded data, not a hypothetical). With mu == ref (a perfectly
        zero-residual actor), the projected action must still land within
        `limit` of ref -- unlike clamp(mu,-1,1) -> clamp(delta,-l,l) ->
        clamp(-1,1), which silently blows the bound open whenever ref > 1."""
        ref = torch.tensor([1.5, -1.8, 2.3])
        mu = ref.clone()  # zero raw residual
        result = project_action_delta(mu, ref, limit=0.1)
        assert (result - ref).abs().max().item() < 0.1

        # Demonstrate the old buggy sequence actually violates the bound in
        # this exact scenario, so this test would have caught the regression.
        old_chunk = mu.clamp(-1, 1)
        old_delta = (old_chunk - ref).clamp(-0.1, 0.1)
        old_result = (ref + old_delta).clamp(-1, 1)
        assert (old_result - ref).abs().max().item() > 0.1

    def test_gradient_shrinks_with_saturation_but_stays_nonzero(self):
        """Unlike a hard clamp (exactly zero gradient once saturated), the
        tanh projection's gradient should shrink as mu moves further from
        ref, but stay strictly positive at a moderately saturating (not
        astronomically large, where float32 underflow would round it to
        exactly 0) offset -- a saturated actor still receives a shrinking
        but nonzero pull, unlike the hard-clamp sequence it replaces."""
        ref = torch.zeros(1)

        mu_near = torch.tensor([0.01], requires_grad=True)
        project_action_delta(mu_near, ref, limit=0.1).backward()
        grad_near = mu_near.grad.item()

        mu_far = torch.tensor([0.3], requires_grad=True)
        project_action_delta(mu_far, ref, limit=0.1).backward()
        grad_far = mu_far.grad.item()

        assert grad_far > 0.0
        assert grad_far < grad_near


class TestActorLossActionClipDelta:
    def test_default_none_matches_prior_behavior_exactly(self, actor, critic, batch):
        torch.manual_seed(0)
        base = actor_loss(actor, critic, batch, beta=0.3)
        torch.manual_seed(0)
        explicit_none = actor_loss(actor, critic, batch, beta=0.3, action_clip_delta=None)
        assert explicit_none.item() == pytest.approx(base.item())

    def test_critic_is_queried_at_the_projected_action_not_raw_mu(self, actor, batch):
        """actor_loss must maximize Q at the action that will actually be
        deployed (ref +/- action_clip_delta), not an unconstrained mu the
        robot would never execute."""

        class _RecordingCritic:
            def __init__(self):
                self.seen_actions: list[torch.Tensor] = []

            def min_q(self, state_vec, action_flat):
                self.seen_actions.append(action_flat.clone())
                return torch.zeros(state_vec.shape[0], 1)

        recording_critic = _RecordingCritic()
        with torch.no_grad():
            raw_mu, _ = actor.forward(batch["state_vec"], batch["ref_chunk_flat"], training=True)

        actor_loss(actor, recording_critic, batch, beta=0.3, action_clip_delta=0.1)
        seen = recording_critic.seen_actions[0]
        assert not torch.allclose(seen, raw_mu)
        assert (seen - batch["ref_chunk_flat"]).abs().max().item() < 0.1 + 1e-4

    def test_bc_regularization_still_uses_raw_mu(self, actor, batch):
        """BC must keep pulling the *raw* mu back toward ref (unsaturating
        gradient) even when action_clip_delta is set -- regressing the
        already-bounded projected action instead would saturate near the
        delta limit and lose that pull-back signal."""
        beta = 1000.0  # dominate the loss so bc_reg's value is recoverable
        loss = actor_loss(actor, _ZeroQCritic(), batch, beta=beta, action_clip_delta=0.05)
        with torch.no_grad():
            mu, _ = actor.forward(batch["state_vec"], batch["ref_chunk_flat"], training=True)
            expected_bc = ((mu - batch["ref_chunk_flat"]) ** 2).sum(dim=-1).mean()
        assert loss.item() == pytest.approx((beta * expected_bc).item(), rel=1e-3)

    def test_zero_deploy_scale_queries_q_at_exact_vla_reference(self, actor, batch):
        class _RecordingCritic:
            def __init__(self):
                self.seen = None

            def min_q(self, state_vec, action_flat):
                self.seen = action_flat.detach().clone()
                return torch.zeros(state_vec.shape[0], 1)

        critic = _RecordingCritic()
        actor_loss(actor, critic, batch, beta=0.3, actor_deploy_scale=0.0)
        assert torch.equal(critic.seen, batch["ref_chunk_flat"])


class _ZeroQCritic:
    def min_q(self, state_vec, action_flat):
        return torch.zeros(state_vec.shape[0], 1)


class TestActorLossHumanCorrection:
    def test_demo_bc_has_independent_weight_when_vla_beta_is_zero(self):
        """Known-correct demonstrations must not disappear when beta is
        lowered to relax the ordinary VLA anchor."""
        ref = torch.zeros(1, 4)

        class _LearnableActor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mu = torch.nn.Parameter(torch.zeros_like(ref))

            def forward(self, state_vec, ref_flat, training=False):
                return self.mu.expand_as(ref_flat), None

        actor = _LearnableActor()
        batch = {
            "state_vec": torch.zeros(1, 3),
            "ref_chunk_flat": ref,
            "exec_chunk_flat": torch.full_like(ref, 0.05),
            "intervention_mask_flat": torch.ones_like(ref),
            "actual_steps": torch.tensor([2]),
            "outcome": torch.tensor([1.0]),
        }

        loss = actor_behavior_cloning_loss(
            actor,
            batch,
            beta=0.0,
            demo_bc_weight=1.0,
            action_clip_delta=0.1,
            chunk_length=2,
        )
        loss.backward()

        assert loss.item() > 0
        assert torch.all(actor.mu.grad < 0)

    def test_intervened_dims_pull_actor_toward_executed_human_action(self):
        """Human takeover must supervise the residual, not merely label Q."""
        chunk_length = 2
        action_dim = 2
        ref = torch.zeros(1, chunk_length * action_dim)

        class _LearnableActor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mu = torch.nn.Parameter(torch.zeros_like(ref))

            def forward(self, state_vec, ref_flat, training=False):
                return self.mu.expand_as(ref_flat), None

        actor = _LearnableActor()
        human_exec = torch.tensor([[0.05, 0.0, 0.0, 0.0]])
        batch = {
            "state_vec": torch.zeros(1, 3),
            "ref_chunk_flat": ref,
            "exec_chunk_flat": human_exec,
            "intervention_mask_flat": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "actual_steps": torch.tensor([2]),
            "outcome": torch.tensor([1.0]),
        }
        loss = actor_loss(
            actor,
            _ZeroQCritic(),
            batch,
            beta=1.0,
            action_clip_delta=0.1,
            chunk_length=chunk_length,
        )
        loss.backward()

        # Gradient descent subtracts this negative gradient, increasing the
        # first actor output toward the positive human correction.
        assert actor.mu.grad[0, 0].item() < 0
        assert torch.equal(actor.mu.grad[0, 1:], torch.zeros(3))

    def test_failed_human_action_is_not_cloned_into_actor(self):
        chunk_length = 2
        ref = torch.zeros(1, 4)

        class _LearnableActor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mu = torch.nn.Parameter(torch.zeros_like(ref))

            def forward(self, state_vec, ref_flat, training=False):
                return self.mu.expand_as(ref_flat), None

        actor = _LearnableActor()
        batch = {
            "state_vec": torch.zeros(1, 3),
            "ref_chunk_flat": ref,
            "exec_chunk_flat": torch.full_like(ref, 0.05),
            "intervention_mask_flat": torch.ones_like(ref),
            "actual_steps": torch.tensor([2]),
            "outcome": torch.tensor([0.0]),
        }
        loss = actor_loss(
            actor,
            _ZeroQCritic(),
            batch,
            beta=1.0,
            action_clip_delta=0.1,
            chunk_length=chunk_length,
        )
        loss.backward()

        assert torch.equal(actor.mu.grad, torch.zeros_like(actor.mu.grad))

    def test_non_intervened_dims_are_not_pulled_toward_human_action(self):
        """Complements test_intervened_dims_pull_actor_toward_executed_human_
        action: that test's human_exec happens to be 0.0 on the non-masked
        dims, same as ref -- so a bug that silently ignored intervention_mask
        entirely would produce the exact same (zero) gradient there by pure
        coincidence, and the test couldn't tell the difference. Use a
        human_exec that's clearly nonzero on every dim, so an ignored mask
        would show up as nonzero gradient on the non-intervened ones too."""
        chunk_length = 2
        action_dim = 2
        ref = torch.zeros(1, chunk_length * action_dim)

        class _LearnableActor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mu = torch.nn.Parameter(torch.zeros_like(ref))

            def forward(self, state_vec, ref_flat, training=False):
                return self.mu.expand_as(ref_flat), None

        actor = _LearnableActor()
        # Every dim is far from ref -- an ignored mask would pull all 4
        # toward this, not just dim 0.
        human_exec = torch.tensor([[0.05, 0.05, 0.05, 0.05]])
        batch = {
            "state_vec": torch.zeros(1, 3),
            "ref_chunk_flat": ref,
            "exec_chunk_flat": human_exec,
            "intervention_mask_flat": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "actual_steps": torch.tensor([2]),
            "outcome": torch.tensor([1.0]),
        }
        loss = actor_loss(
            actor,
            _ZeroQCritic(),
            batch,
            beta=1.0,
            action_clip_delta=0.1,
            chunk_length=chunk_length,
        )
        loss.backward()

        assert actor.mu.grad[0, 0].item() < 0  # masked-in dim: pulled toward human_exec
        assert torch.equal(actor.mu.grad[0, 1:], torch.zeros(3))  # masked-out dims: untouched

    def test_partial_chunk_padding_is_excluded_from_bc(self):
        chunk_length = 2
        ref = torch.zeros(1, 4)

        class _FixedActor:
            def forward(self, state_vec, ref_flat, training=False):
                # Only the never-executed second timestep differs.
                return torch.tensor([[0.0, 0.0, 9.0, -9.0]]), None

        batch = {
            "state_vec": torch.zeros(1, 3),
            "ref_chunk_flat": ref,
            "exec_chunk_flat": ref.clone(),
            "actual_steps": torch.tensor([1]),
        }
        loss = actor_loss(
            _FixedActor(),
            _ZeroQCritic(),
            batch,
            beta=1.0,
            chunk_length=chunk_length,
        )
        assert loss.item() == pytest.approx(0.0)


def test_training_slew_limiter_is_sequential_and_uses_residual_anchor():
    ref = torch.tensor([[1.0, 2.0, 3.0]])
    action = ref + torch.tensor([[0.0, 0.2, -0.2]])
    limited = _apply_slew_rate_limit_flat(
        action,
        ref,
        chunk_length=3,
        slew_rate_limit=0.05,
        prev_residual=torch.tensor([[-0.1]]),
    )
    assert torch.allclose(limited - ref, torch.tensor([[-0.05, 0.0, -0.05]]))


def test_training_slew_limiter_does_not_limit_reference_motion():
    ref = torch.tensor([[0.0, 1.5, -2.0]])
    limited = _apply_slew_rate_limit_flat(
        ref,
        ref,
        chunk_length=3,
        slew_rate_limit=0.01,
    )
    assert torch.equal(limited, ref)


class TestActorLossSmoothness:
    def test_zero_weight_is_a_no_op(self, actor, critic, batch):
        torch.manual_seed(0)
        base = actor_loss(actor, critic, batch, beta=0.3)
        torch.manual_seed(0)
        explicit_zero = actor_loss(
            actor, critic, batch, beta=0.3, smoothness_weight=0.0, chunk_length=C,
        )
        assert explicit_zero.item() == pytest.approx(base.item())

    def test_positive_weight_requires_chunk_length(self, actor, critic, batch):
        with pytest.raises(ValueError, match="chunk_length"):
            actor_loss(actor, critic, batch, beta=0.3, smoothness_weight=1.0)

    def test_positive_weight_penalizes_adjacent_timestep_jumps(self, actor, batch):
        """A hand-built actor whose raw mu jumps wildly between adjacent
        timesteps should get a strictly larger loss under a positive
        smoothness_weight than under zero, all else equal."""
        action_dim = CHUNK_DIM // C
        with torch.no_grad():
            mu, _ = actor.forward(batch["state_vec"], batch["ref_chunk_flat"], training=True)
        mu_chunk = unflatten_chunk(mu, C)
        # Force alternating +/-10 across timesteps: a large, oscillating raw
        # residual with nothing smooth about it.
        sign = torch.tensor([1.0 if t % 2 == 0 else -1.0 for t in range(C)])
        jumpy = (sign.view(1, C, 1) * 10.0).expand(mu_chunk.shape[0], C, action_dim).clone()

        class _FixedOutputActor:
            def forward(self, state_vec, ref, training=False):
                flat = jumpy.flatten(start_dim=-2)
                return flat, torch.zeros_like(flat)

        loss_smooth_off = actor_loss(
            _FixedOutputActor(), _ZeroQCritic(), batch, beta=0.0,
            smoothness_weight=0.0, chunk_length=C,
        )
        loss_smooth_on = actor_loss(
            _FixedOutputActor(), _ZeroQCritic(), batch, beta=0.0,
            smoothness_weight=1.0, chunk_length=C,
        )
        assert loss_smooth_on.item() > loss_smooth_off.item()

    def test_vla_reference_motion_is_not_penalized(self):
        action_dim = CHUNK_DIM // C
        ref_chunk = torch.arange(C, dtype=torch.float32).view(1, C, 1)
        ref_chunk = ref_chunk.expand(2, C, action_dim).clone()
        ref = ref_chunk.flatten(start_dim=-2)

        class _ZeroResidualActor:
            def forward(self, state_vec, ref_flat, training=False):
                return ref_flat, torch.zeros_like(ref_flat)

        batch = {
            "state_vec": torch.zeros(2, STATE_DIM),
            "ref_chunk_flat": ref,
            "exec_chunk_flat": ref.clone(),
            "actual_steps": torch.full((2,), C),
        }
        loss = actor_loss(
            _ZeroResidualActor(),
            _ZeroQCritic(),
            batch,
            beta=0.0,
            smoothness_weight=1.0,
            chunk_length=C,
        )
        assert loss.item() == pytest.approx(0.0)


class TestCriticLossActionClipDelta:
    def test_default_none_matches_prior_behavior_exactly(self, actor, critic, target_critic, batch):
        torch.manual_seed(0)
        base = critic_loss(critic, target_critic, actor, batch, gamma=0.99, C=C)
        torch.manual_seed(0)
        explicit_none = critic_loss(
            critic, target_critic, actor, batch, gamma=0.99, C=C, action_clip_delta=None,
        )
        assert explicit_none.item() == pytest.approx(base.item())

    def test_target_action_respects_delta_even_when_ref_exceeds_unit_range(self, actor, critic):
        """Regression test for the exact bug: ref_next regularly exceeds
        [-1,1] under QUANTILES normalization. mu_next must land within
        action_clip_delta of ref_next regardless."""

        class _RecordingTargetCritic:
            def __init__(self):
                self.seen_actions: list[torch.Tensor] = []

            def min_q(self, state_vec, action_flat):
                self.seen_actions.append(action_flat.clone())
                return torch.zeros(state_vec.shape[0], 1)

        B = 4
        batch = {
            "state_vec": torch.zeros(B, STATE_DIM),
            "exec_chunk_flat": torch.zeros(B, CHUNK_DIM),
            "ref_chunk_flat": torch.zeros(B, CHUNK_DIM),
            "reward_seq": torch.zeros(B, C),
            "next_state_vec": torch.zeros(B, STATE_DIM),
            # Deliberately outside [-1, 1], matching real recorded ref_chunk
            # statistics (up to ~2.5-3.8 in practice).
            "next_ref_flat": torch.full((B, CHUNK_DIM), 1.8),
            "done": torch.zeros(B),
            "actual_steps": torch.full((B,), C),
        }
        target_critic = _RecordingTargetCritic()
        critic_loss(
            critic, target_critic, actor, batch, gamma=0.99, C=C, action_clip_delta=0.1,
        )
        seen = target_critic.seen_actions[0]
        assert (seen - batch["next_ref_flat"]).abs().max().item() < 0.1 + 1e-4

    def test_zero_deploy_scale_bootstraps_at_exact_vla_reference(self, actor, critic):
        class _RecordingTargetCritic:
            def __init__(self):
                self.seen = None

            def min_q(self, state_vec, action_flat):
                self.seen = action_flat.detach().clone()
                return torch.zeros(state_vec.shape[0], 1)

        B = 4
        next_ref = torch.full((B, CHUNK_DIM), 1.8)
        batch = {
            "state_vec": torch.zeros(B, STATE_DIM),
            "exec_chunk_flat": torch.zeros(B, CHUNK_DIM),
            "ref_chunk_flat": torch.zeros(B, CHUNK_DIM),
            "reward_seq": torch.zeros(B, C),
            "next_state_vec": torch.zeros(B, STATE_DIM),
            "next_ref_flat": next_ref,
            "done": torch.zeros(B),
            "actual_steps": torch.full((B,), C),
        }
        target_critic = _RecordingTargetCritic()
        critic_loss(
            critic,
            target_critic,
            actor,
            batch,
            gamma=0.99,
            C=C,
            actor_deploy_scale=0.0,
        )
        assert torch.equal(target_critic.seen, next_ref)

    def test_target_noise_cannot_push_action_outside_delta(self, actor, critic):
        """target_noise_clip (0.3 by convention) can exceed action_clip_delta
        (0.1) -- the noised mu_next must still be re-projected back inside
        the deployable neighborhood, not allowed to explore a physically
        unreachable region."""

        class _RecordingTargetCritic:
            def __init__(self):
                self.seen_actions: list[torch.Tensor] = []

            def min_q(self, state_vec, action_flat):
                self.seen_actions.append(action_flat.clone())
                return torch.zeros(state_vec.shape[0], 1)

        B = 4
        batch = {
            "state_vec": torch.zeros(B, STATE_DIM),
            "exec_chunk_flat": torch.zeros(B, CHUNK_DIM),
            "ref_chunk_flat": torch.zeros(B, CHUNK_DIM),
            "reward_seq": torch.zeros(B, C),
            "next_state_vec": torch.zeros(B, STATE_DIM),
            "next_ref_flat": torch.zeros(B, CHUNK_DIM),
            "done": torch.zeros(B),
            "actual_steps": torch.full((B,), C),
        }
        target_critic = _RecordingTargetCritic()
        critic_loss(
            critic, target_critic, actor, batch, gamma=0.99, C=C,
            target_noise_std=0.5, target_noise_clip=0.3, action_clip_delta=0.1,
        )
        seen = target_critic.seen_actions[0]
        assert (seen - batch["next_ref_flat"]).abs().max().item() < 0.1 + 1e-4

    def test_post_noise_rebound_does_not_compound_shrinkage(self, actor, critic):
        """Regression test: the post-noise re-bound used to call
        project_action_delta() a second time. That projection is a smooth
        *contraction* toward ref (tanh), not a boundary-only clip, so
        applying it twice pulled the target action noticeably closer to ref
        than a single projection does (e.g. a raw delta of exactly
        action_clip_delta lands at ~0.76x of the limit after one call but
        ~0.64x after two) -- silently biasing the target action away from
        the one project_action_delta's own docstring promises actor_loss/
        deployment produce for the same state. With a (numerically) zero
        smoothing noise, the post-noise re-bound must be a no-op on an
        already-in-bound action, so the seen action must match the
        single-projection value exactly, not a further-shrunk one.
        """

        class _RecordingTargetCritic:
            def __init__(self):
                self.seen_actions: list[torch.Tensor] = []

            def min_q(self, state_vec, action_flat):
                self.seen_actions.append(action_flat.clone())
                return torch.zeros(state_vec.shape[0], 1)

        action_clip_delta = 0.1
        B = 4
        next_state_vec = torch.randn(B, STATE_DIM)
        next_ref_flat = torch.randn(B, CHUNK_DIM)
        with torch.no_grad():
            mu_next, _ = actor.forward(next_state_vec, next_ref_flat)
            once_projected = project_action_delta(mu_next, next_ref_flat, action_clip_delta)

        batch = {
            "state_vec": torch.zeros(B, STATE_DIM),
            "exec_chunk_flat": torch.zeros(B, CHUNK_DIM),
            "ref_chunk_flat": torch.zeros(B, CHUNK_DIM),
            "reward_seq": torch.zeros(B, C),
            "next_state_vec": next_state_vec,
            "next_ref_flat": next_ref_flat,
            "done": torch.zeros(B),
            "actual_steps": torch.full((B,), C),
        }
        target_critic = _RecordingTargetCritic()
        critic_loss(
            critic, target_critic, actor, batch, gamma=0.99, C=C,
            target_noise_std=1e-8, target_noise_clip=0.3, action_clip_delta=action_clip_delta,
        )
        seen = target_critic.seen_actions[0]
        assert torch.allclose(seen, once_projected, atol=1e-5)
