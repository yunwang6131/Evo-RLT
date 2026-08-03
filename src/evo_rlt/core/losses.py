from __future__ import annotations

import torch
import torch.nn.functional as F

from evo_rlt.core.actor import ChunkActor
from evo_rlt.core.critic import TwinCritic
from evo_rlt.core.utils import compute_discount_vector


def discounted_chunk_return(
    reward_seq: torch.Tensor, gamma: float, actual_steps: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute discounted return over a chunk of rewards.

    Args:
        reward_seq: (B, C) rewards for each timestep (padded with 0 beyond actual_steps)
        gamma: discount factor
        actual_steps: (B,) number of valid steps per chunk (if None, assume all C are valid)

    Returns:
        (B, 1) discounted return
    """
    C = reward_seq.shape[1]
    discounts = compute_discount_vector(gamma, C, device=reward_seq.device)
    return (reward_seq * discounts.unsqueeze(0)).sum(dim=1, keepdim=True)


def rankq_ranking_loss(
    critic: TwinCritic,
    state_vec: torch.Tensor,
    action_flat: torch.Tensor,
    outcome: torch.Tensor,
    noise_scale: float = 0.15,
    alpha_success: float = 1.0,
    alpha_failure: float = 1.0,
) -> torch.Tensor:
    """RankQ (Choi & Xu, 2026) self-supervised action-ranking loss.

    Rather than uniformly penalizing every unseen action (CQL/Cal-QL-style
    pessimism), this shapes the Q-landscape so that gradients w.r.t. action
    point toward higher-quality regions: for a transition from a
    *successful* trajectory, the executed action must outrank a small
    perturbation of itself, which must outrank a larger perturbation, which
    must outrank a random action -- plus the executed action must also
    outrank an unrelated (permuted) action. For a *failed* trajectory's
    action, only the weak constraint "beats a random action" is enforced,
    since a failure action isn't necessarily bad everywhere. Transitions
    whose trajectory outcome hasn't resolved (outcome not in {0, 1}) are
    excluded. Mirrors RankQ Eq. 4-6 / Appendix A.

    `outcome`: (B,) with 1.0 = success, 0.0 = failure, anything else =
    unresolved/unknown (excluded).
    """
    success_mask = outcome == 1
    failure_mask = outcome == 0
    if not (bool(success_mask.any()) or bool(failure_mask.any())):
        return action_flat.new_zeros(())

    bs = action_flat.shape[0]
    eps = torch.randn_like(action_flat)
    perm = torch.roll(torch.arange(bs, device=action_flat.device), shifts=-1)
    candidates = {
        "exec": action_flat,
        "noisy": action_flat + eps * noise_scale,
        "very_noisy": action_flat + eps * (2.0 * noise_scale),
        "random": torch.rand_like(action_flat) * 2.0 - 1.0,
        "permuted": action_flat[perm],
    }
    q1, q2 = {}, {}
    for name, action in candidates.items():
        q1[name], q2[name] = critic(state_vec, action)

    def _rank(q: dict[str, torch.Tensor], pos: str, neg: str, mask: torch.Tensor) -> torch.Tensor:
        if not bool(mask.any()):
            return action_flat.new_zeros(())
        return F.softplus(q[neg][mask] - q[pos][mask]).mean()

    # L^succ + L^chain (Eq. 4-5): executed action beats every suboptimal
    # variant, and the variants are themselves chained in quality order.
    success_pairs = [
        ("exec", "noisy"), ("exec", "very_noisy"), ("exec", "random"), ("exec", "permuted"),
        ("noisy", "very_noisy"), ("very_noisy", "random"),
    ]
    loss = action_flat.new_zeros(())
    for q in (q1, q2):
        for pos, neg in success_pairs:
            loss = loss + alpha_success * _rank(q, pos, neg, success_mask)
        # L^fail (Eq. 6): only the weak "beats random" constraint.
        loss = loss + alpha_failure * _rank(q, "exec", "random", failure_mask)
    return loss


def critic_loss(
    critic: TwinCritic,
    target_critic: TwinCritic,
    actor: ChunkActor,
    batch: dict[str, torch.Tensor],
    gamma: float,
    C: int,
    target_q_clip: float | None = 100.0,
    rankq_noise_scale: float = 0.15,
    rankq_alpha_success: float = 0.0,
    rankq_alpha_failure: float = 0.0,
) -> torch.Tensor:
    """TD3-style chunk-level TD loss with correct truncated-chunk handling.

    Uses actual_steps to compute the correct bootstrap exponent gamma^k
    instead of always using gamma^C.

    If `batch["outcome"]` is present and either rankq_alpha_* is > 0, adds
    RankQ's self-supervised ranking loss (see rankq_ranking_loss above) as
    an additional term. Fully backward compatible: with the default alphas
    (or no "outcome" key in batch) this is a no-op.
    """
    x = batch["state_vec"]
    a = batch["exec_chunk_flat"]
    x_next = batch["next_state_vec"]
    ref_next = batch["next_ref_flat"]
    reward_seq = batch["reward_seq"]
    done = batch["done"]
    actual_steps = batch.get("actual_steps")

    with torch.no_grad():
        # Use deterministic mean for target action (TD3-style), clamped to [-1,1]
        mu_next, _ = actor.forward(x_next, ref_next)
        mu_next = mu_next.clamp(-1.0, 1.0)
        q_next = target_critic.min_q(x_next, mu_next)
        if target_q_clip is not None and target_q_clip > 0:
            q_next = q_next.clamp(-target_q_clip, target_q_clip)
        r = discounted_chunk_return(reward_seq, gamma, actual_steps)

        # Bootstrap with gamma^k where k = actual steps executed
        if actual_steps is not None:
            bootstrap_exp = actual_steps.unsqueeze(-1).float()
        else:
            bootstrap_exp = torch.full_like(done.unsqueeze(-1), C, dtype=torch.float32)
        bootstrap = (gamma ** bootstrap_exp) * (1.0 - done.unsqueeze(-1)) * q_next
        target = r + bootstrap

    q1, q2 = critic(x, a)
    loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

    outcome = batch.get("outcome")
    if outcome is not None and (rankq_alpha_success > 0 or rankq_alpha_failure > 0):
        loss = loss + rankq_ranking_loss(
            critic, x, a, outcome,
            noise_scale=rankq_noise_scale,
            alpha_success=rankq_alpha_success,
            alpha_failure=rankq_alpha_failure,
        )
    return loss


def actor_loss(
    actor: ChunkActor,
    critic: TwinCritic,
    batch: dict[str, torch.Tensor],
    beta: float,
) -> torch.Tensor:
    """Q-maximization + BC regularization toward VLA reference.

    Uses deterministic mean (not noisy samples) for stable optimization.
    BC term is the per-sample squared distance summed across action dims, then
    averaged over the batch — matching the paper's β-scaling convention. This
    differs from mean-MSE by a factor of C*D_flat.
    """
    x = batch["state_vec"]
    ref = batch["ref_chunk_flat"]
    mu, _ = actor.forward(x, ref, training=True)
    q = critic.min_q(x, mu)
    bc_reg = ((mu - ref) ** 2).sum(dim=-1).mean()
    return -q.mean() + beta * bc_reg


def q_action_sensitivity(
    critic: TwinCritic,
    state_vec: torch.Tensor,
    action_flat: torch.Tensor,
    noise_scale: float = 0.15,
) -> torch.Tensor:
    """Diagnostic: how much does critic.min_q(s, ·) actually vary across
    different actions at the same state?

    A critic trained under sparse, long-horizon reward can trivially
    minimize TD-MSE by learning a state-only baseline and ignoring the
    action input almost entirely -- functionally collapsing into a value
    function V(s) even though it's architecturally still Q(s,a) (see RankQ
    paper Fig.1's ∂Q/∂a visualization for the same failure mode via a true
    gradient; this is a cheaper no-autograd proxy for periodic logging).
    Reuses the same candidate-action construction as rankq_ranking_loss so
    the comparison is apples-to-apples.

    Returns a scalar: mean over the batch of the per-sample std across 5
    candidate actions' (exec/noisy/very_noisy/random/permuted) Q values.
    Near 0 means Q is action-insensitive (collapsed toward V(s)); a healthy
    value should track Q's own scale and stay well above 0.
    """
    with torch.no_grad():
        bs = action_flat.shape[0]
        eps = torch.randn_like(action_flat)
        perm = torch.roll(torch.arange(bs, device=action_flat.device), shifts=-1)
        candidates = [
            action_flat,
            action_flat + eps * noise_scale,
            action_flat + eps * (2.0 * noise_scale),
            torch.rand_like(action_flat) * 2.0 - 1.0,
            action_flat[perm],
        ]
        q_values = torch.stack(
            [critic.min_q(state_vec, a).squeeze(-1) for a in candidates], dim=0
        )  # (5, B)
        return q_values.std(dim=0).mean()
