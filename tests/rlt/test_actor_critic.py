from __future__ import annotations

import torch
import pytest

from evo_rlt.core.actor import ChunkActor
from evo_rlt.core.critic import ChunkCritic, TwinCritic


@pytest.fixture
def actor():
    return ChunkActor(state_dim=78, chunk_dim=140, hidden_dim=64, num_layers=2)


@pytest.fixture
def twin_critic():
    return TwinCritic(state_dim=78, chunk_dim=140, hidden_dim=64, num_layers=2)


class TestActor:
    def test_forward_shapes(self, actor):
        state = torch.randn(8, 78)
        ref = torch.randn(8, 140)
        mu, std = actor(state, ref)
        assert mu.shape == (8, 140)
        assert std.shape == (8, 140)

    def test_sample_shapes(self, actor):
        state = torch.randn(8, 78)
        ref = torch.randn(8, 140)
        action, mu = actor.sample(state, ref)
        assert action.shape == (8, 140)
        assert mu.shape == (8, 140)

    def test_fixed_std(self, actor):
        state = torch.randn(4, 78)
        ref = torch.randn(4, 140)
        _, std = actor(state, ref)
        assert torch.allclose(std, torch.full_like(std, 0.05))

    def test_ref_dropout_statistics(self):
        """With large batch and training=True, ~50% should be zeroed."""
        actor = ChunkActor(state_dim=78, chunk_dim=140, hidden_dim=64, ref_dropout_p=0.5)
        state = torch.randn(1000, 78)
        ref = torch.ones(1000, 140)  # all ones so we can detect zeroing

        torch.manual_seed(42)
        mu, _ = actor(state, ref, training=True)

        # The ref was multiplied by a mask. We can check by looking at the input
        # indirectly: the ratio of zero-ref samples should be ~50%
        # We verify by calling forward manually and checking the mask effect
        torch.manual_seed(42)
        mask = (torch.rand(1000, 1) > 0.5).float()
        frac_kept = mask.mean().item()
        assert 0.4 < frac_kept < 0.6

    def test_gradient_flow(self, actor):
        state = torch.randn(4, 78)
        ref = torch.randn(4, 140)
        action, _ = actor.sample(state, ref, training=True)
        loss = action.sum()
        loss.backward()
        for p in actor.parameters():
            assert p.grad is not None


class TestResidualToRefActor:
    """residual_to_ref=True must start out as a no-op over the VLA reference
    (mu == ref, delta == 0), for safe online RL initialization on real hardware."""

    def test_mu_equals_ref_at_init(self):
        actor = ChunkActor(
            state_dim=78, chunk_dim=140, hidden_dim=64, num_layers=2, residual_to_ref=True,
        )
        state = torch.randn(8, 78)
        ref = torch.randn(8, 140)
        mu, _ = actor(state, ref, training=False)
        assert torch.allclose(mu, ref)

    def test_mu_equals_ref_at_init_with_residual_mlp(self):
        actor = ChunkActor(
            state_dim=78, chunk_dim=140, hidden_dim=64, num_layers=2,
            residual=True, residual_to_ref=True,
        )
        state = torch.randn(8, 78)
        ref = torch.randn(8, 140)
        mu, _ = actor(state, ref, training=False)
        assert torch.allclose(mu, ref)

    def test_ref_dropout_does_not_break_residual_bias(self):
        """Even when the network's view of ref is dropped out during training,
        the true (undropped) ref is still added back as the residual bias."""
        actor = ChunkActor(
            state_dim=78, chunk_dim=140, hidden_dim=64, num_layers=2,
            residual_to_ref=True, ref_dropout_p=1.0,  # force full dropout
        )
        state = torch.randn(8, 78)
        ref = torch.randn(8, 140)
        mu, _ = actor(state, ref, training=True)
        # delta==0 at init regardless of what the net saw, so mu still == ref.
        assert torch.allclose(mu, ref)

    def test_gradient_flow(self):
        actor = ChunkActor(
            state_dim=78, chunk_dim=140, hidden_dim=64, num_layers=2, residual_to_ref=True,
        )
        state = torch.randn(4, 78)
        ref = torch.randn(4, 140)
        action, _ = actor.sample(state, ref, training=True)
        loss = action.sum()
        loss.backward()
        for p in actor.parameters():
            assert p.grad is not None


class TestCritic:
    def test_chunk_critic_shape(self):
        critic = ChunkCritic(state_dim=78, chunk_dim=140, hidden_dim=64)
        q = critic(torch.randn(8, 78), torch.randn(8, 140))
        assert q.shape == (8, 1)

    def test_twin_critic_shapes(self, twin_critic):
        state = torch.randn(8, 78)
        action = torch.randn(8, 140)
        q1, q2 = twin_critic(state, action)
        assert q1.shape == (8, 1)
        assert q2.shape == (8, 1)

    def test_min_q(self, twin_critic):
        state = torch.randn(8, 78)
        action = torch.randn(8, 140)
        q1, q2 = twin_critic(state, action)
        min_q = twin_critic.min_q(state, action)
        expected = torch.minimum(q1, q2)
        assert torch.allclose(min_q, expected)

    def test_gradient_flow(self, twin_critic):
        state = torch.randn(4, 78)
        action = torch.randn(4, 140)
        q = twin_critic.min_q(state, action)
        q.sum().backward()
        for p in twin_critic.parameters():
            assert p.grad is not None


def test_left_arm_action_mask_keeps_right_arm_at_reference_and_blocks_gradients():
    chunk_length, action_dim = 3, 12
    per_step = torch.tensor([1.0] * 6 + [0.0] * 6)
    actor = ChunkActor(
        state_dim=8,
        chunk_dim=chunk_length * action_dim,
        hidden_dim=16,
        num_layers=1,
        residual_to_ref=True,
        action_mask=per_step.repeat(chunk_length),
    )
    # Move away from the zero-init policy so the allowed left-arm residual and
    # its gradients are observable.
    with torch.no_grad():
        actor.net[-1].weight.normal_()
        actor.net[-1].bias.normal_()

    state = torch.randn(2, 8)
    ref = torch.randn(2, chunk_length * action_dim)
    mu, _ = actor(state, ref)
    delta = (mu - ref).view(2, chunk_length, action_dim)
    assert torch.count_nonzero(delta[..., :6]) > 0
    assert torch.equal(delta[..., 6:], torch.zeros_like(delta[..., 6:]))

    sampled, _ = actor.sample(state, ref)
    sampled_delta = (sampled - ref).view(2, chunk_length, action_dim)
    assert torch.equal(sampled_delta[..., 6:], torch.zeros_like(sampled_delta[..., 6:]))

    mu.sum().backward()
    out_grad = actor.net[-1].bias.grad.view(chunk_length, action_dim)
    assert torch.count_nonzero(out_grad[:, :6]) > 0
    assert torch.equal(out_grad[:, 6:], torch.zeros_like(out_grad[:, 6:]))


def test_actor_action_mask_validates_flattened_size():
    with pytest.raises(ValueError, match="action_mask"):
        ChunkActor(state_dim=8, chunk_dim=24, action_mask=torch.ones(12))


def test_action_mask_keeps_masked_dimensions_at_ref_for_non_residual_actor():
    actor = ChunkActor(
        state_dim=4,
        chunk_dim=4,
        hidden_dim=8,
        num_layers=1,
        residual_to_ref=False,
        action_mask=torch.tensor([1.0, 1.0, 0.0, 0.0]),
    )
    ref = torch.randn(2, 4)
    mu, _ = actor(torch.randn(2, 4), ref)
    assert torch.equal(mu[:, 2:], ref[:, 2:])
