from __future__ import annotations

import torch
import torch.nn.functional as F

from evo_rlt.core.actor import ChunkActor
from evo_rlt.core.critic import TwinCritic
from evo_rlt.core.utils import compute_discount_vector, project_action_delta, unflatten_chunk


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


def _masked_candidate(
    base: torch.Tensor, alt: torch.Tensor, action_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Blend `alt` into `base`, restricted to the actor-controllable dims.

    `action_mask` mirrors ChunkActor.action_mask: 1 on dims the actor may
    move, 0 on dims pinned to the VLA reference (e.g. the frozen arm under
    actor_rl_arm="left"). Perturbing/randomizing/permuting a masked-out dim
    would build a candidate action the actor could never actually produce
    (its output on that dim is always exactly `base`), which only adds
    noise to the ranking signal below without changing what the actor can
    do about it. None means no masking (e.g. actor_rl_arm="both") -- `alt`
    is used as-is.
    """
    if action_mask is None:
        return alt
    return base + action_mask * (alt - base)


def _valid_action_mask(
    action_flat: torch.Tensor,
    actual_steps: torch.Tensor | None,
    chunk_length: int,
) -> torch.Tensor:
    """Return a (B, C*D) mask for physically executed chunk elements."""
    if actual_steps is None:
        return torch.ones_like(action_flat)
    action_dim = action_flat.shape[-1] // chunk_length
    steps = torch.arange(chunk_length, device=action_flat.device).unsqueeze(0)
    valid_steps = steps < actual_steps.long().unsqueeze(1)
    return valid_steps.unsqueeze(-1).expand(-1, -1, action_dim).reshape_as(action_flat).to(action_flat)


def _canonicalize_partial_action(
    action_flat: torch.Tensor,
    ref_flat: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Make never-executed suffixes identical for behavior and candidates."""
    return ref_flat + valid_mask * (action_flat - ref_flat)


def _apply_slew_rate_limit_flat(
    action_flat: torch.Tensor,
    ref_flat: torch.Tensor,
    chunk_length: int,
    slew_rate_limit: float | None,
    prev_residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Training-time equivalent of RLTActionModifier's sequential limiter.

    Rate-limits the ACTOR RESIDUAL (action - ref), not the absolute command --
    see RLTActionModifier._apply_slew_rate_limit for the full rationale. The
    VLA reference trajectory is trusted as-is; only the actor contribution
    (including any correctly aligned residual carried from the prior frame)
    is allowed to ramp.

    `prev_residual`, when available, must be the actor residual paired with
    the preceding frame's own VLA reference.  An absolute previous action is
    not a valid substitute: subtracting this chunk's first reference would
    rate-limit reference motion and can violate action_clip_delta.  Replay
    data currently does not persist the counterfactual actor residual, so
    training callers deliberately use the safe zero-residual anchor.
    """
    if slew_rate_limit is None:
        return action_flat
    chunk = unflatten_chunk(action_flat, chunk_length)
    ref_chunk = unflatten_chunk(ref_flat, chunk_length)
    residual = chunk - ref_chunk
    if prev_residual is None:
        prev = torch.zeros_like(residual[:, 0, :])
    else:
        prev = prev_residual.to(chunk)
    steps = []
    for t in range(chunk_length):
        prev = prev + (residual[:, t, :] - prev).clamp(-slew_rate_limit, slew_rate_limit)
        steps.append(prev)
    return (ref_chunk + torch.stack(steps, dim=1)).flatten(start_dim=-2)


def _random_action_like(action_flat: torch.Tensor) -> torch.Tensor:
    """A "random action" drawn from the batch's own per-dimension marginals.

    QUANTILES-normalized actions are not bounded to [-1, 1], so a fixed
    uniform sample is an easy out-of-distribution negative.  Shuffling each
    dimension independently across the batch preserves the observed
    marginals while destroying joint action structure.
    Degenerate for batch sizes below 2 (nothing to shuffle against), where
    the executed action is returned unchanged and the corresponding ranking
    pairs contribute a constant with no gradient direction.
    """
    if action_flat.shape[0] < 2:
        return action_flat
    shuffle = torch.argsort(
        torch.rand(action_flat.shape, device=action_flat.device), dim=0
    )
    return torch.gather(action_flat, 0, shuffle)


def _rankq_candidate_actions(
    action_flat: torch.Tensor,
    noise_scale: float,
    action_mask: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """Build RankQ's ordered candidate actions (shared by rankq_ranking_loss
    and q_action_sensitivity, so the diagnostic always measures the same
    construction the loss actually trains on)."""
    bs = action_flat.shape[0]
    eps = torch.randn_like(action_flat)
    perm = torch.roll(torch.arange(bs, device=action_flat.device), shifts=-1)
    return {
        "exec": action_flat,
        "noisy": _masked_candidate(action_flat, action_flat + eps * noise_scale, action_mask),
        "very_noisy": _masked_candidate(
            action_flat, action_flat + eps * (2.0 * noise_scale), action_mask
        ),
        "random": _masked_candidate(
            action_flat, _random_action_like(action_flat), action_mask
        ),
        "permuted": _masked_candidate(action_flat, action_flat[perm], action_mask),
    }


def rankq_ranking_loss(
    critic: TwinCritic,
    state_vec: torch.Tensor,
    action_flat: torch.Tensor,
    outcome: torch.Tensor,
    noise_scale: float = 0.15,
    alpha_success: float = 1.0,
    alpha_failure: float = 1.0,
    action_mask: torch.Tensor | None = None,
    margin: float = 0.0,
    margin_relative: bool = False,
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

    `action_mask`: see _masked_candidate -- restricts every perturbed
    candidate to the actor's controllable dims (e.g. left arm only under
    actor_rl_arm="left") so the ranking is over actions the actor could
    actually choose between, not actions differing only on a frozen arm.

    Deviation from the paper's raw Eq. A.5 (documented, not accidental):
    Eq. A.5 sums all pairwise softplus terms unweighted by count. At
    alpha=1.0 that gives ~6 success pairs + 1 failure pair, times 2 critic
    heads = up to 14 terms, each up to softplus(0)=ln(2)~=0.69 before the
    critic has learned to separate anything -- i.e. up to ~9.7 total,
    several times the plain 2-term TD MSE loss's typical scale (measured
    empirically on this codebase's chunk-flattened action space). Left as
    Eq. A.5 verbatim, this term can dominate critic training over accurately
    fitting returns, especially early on. Averaging each branch (success,
    failure) over its own pair count keeps `alpha_success`/`alpha_failure`
    meaningful as "how much to weight ranking vs. TD" regardless of how many
    pairs the two branches happen to define, instead of that weight
    silently scaling with an implementation detail.

    `margin` > 0 replaces Eq. A.5's softplus with a hard hinge,
    relu(q_neg - q_pos + margin).  Softplus keeps a nonzero incentive to
    widen already-correct separable pairs, while the hinge stops contributing
    once the requested gap is reached.  Set the margin relative to the reward
    scale. 0.0 retains the original softplus for compatibility.

    `margin_relative` reinterprets `margin` as a *fraction of the critic's
    own current Q scale* rather than an absolute gap: the hinge becomes
    relu(q_neg - q_pos + margin * mean|Q|), with mean|Q| taken over all
    candidate actions and detached (it sets the constraint's strength, it
    is not itself something to optimize).  An absolute margin silently stops
    constraining anything once the critic's Q scale drifts away from the
    reward scale it was tuned against -- observed on this codebase: with
    margin=0.1 against a Q that had drifted to ~4.8, the ranking term was
    2% of the signal it was ordering and contributed ~8% of the critic loss,
    leaving the ranking to a TD term that had itself overestimated by ~9x
    and had the sign backwards (Q of a human's successful takeover action
    ranked *below* the actor's own failing action).  A relative margin keeps
    the requested separation meaningful under exactly that drift. False
    (default) keeps the absolute-margin behavior.
    """
    if margin < 0:
        raise ValueError("margin must be >= 0")
    eligible_mask = (outcome == 1) | (outcome == 0)
    if not bool(eligible_mask.any()):
        return action_flat.new_zeros(())

    # Remove unresolved/excluded rows *before* constructing cross-batch
    # candidates.  Both the marginally shuffled "random" action and the
    # batch-rolled "permuted" action borrow values from other rows; masking
    # only the final loss would therefore still let offline demonstrations
    # shape the negative candidates used by eligible online transitions.
    if not bool(eligible_mask.all()):
        state_vec = state_vec[eligible_mask]
        action_flat = action_flat[eligible_mask]
        outcome = outcome[eligible_mask]
        if (
            action_mask is not None
            and action_mask.ndim > 1
            and action_mask.shape[0] == eligible_mask.shape[0]
        ):
            action_mask = action_mask[eligible_mask]

    success_mask = outcome == 1
    failure_mask = outcome == 0
    candidates = _rankq_candidate_actions(action_flat, noise_scale, action_mask)
    q1, q2 = {}, {}
    for name, action in candidates.items():
        q1[name], q2[name] = critic(state_vec, action)

    # One scale for every pair in this batch, not a per-pair one: the chain
    # constraints (exec > noisy > very_noisy > random) only compose into a
    # consistent ordering if they all ask for the same separation.
    effective_margin: float | torch.Tensor = margin
    if margin > 0 and margin_relative:
        with torch.no_grad():
            q_scale = torch.stack(
                [q.abs().mean() for q in (*q1.values(), *q2.values())]
            ).mean()
        effective_margin = margin * q_scale.clamp(min=1e-6)

    def _rank(q: dict[str, torch.Tensor], pos: str, neg: str, mask: torch.Tensor) -> torch.Tensor:
        if not bool(mask.any()):
            return action_flat.new_zeros(())
        gap = q[neg][mask] - q[pos][mask]
        if margin > 0:
            return F.relu(gap + effective_margin).mean()
        return F.softplus(gap).mean()

    # L^succ + L^chain (Eq. 4-5): executed action beats every suboptimal
    # variant, and the variants are themselves chained in quality order.
    success_pairs = [
        ("exec", "noisy"), ("exec", "very_noisy"), ("exec", "random"), ("exec", "permuted"),
        ("noisy", "very_noisy"), ("very_noisy", "random"),
    ]
    loss = action_flat.new_zeros(())
    for q in (q1, q2):
        success_terms = torch.stack([_rank(q, pos, neg, success_mask) for pos, neg in success_pairs])
        loss = loss + alpha_success * success_terms.mean()
        # L^fail (Eq. 6): only the weak "beats random" constraint -- already
        # a single term, nothing to average over.
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
    target_q_min: float | None = None,
    rankq_noise_scale: float = 0.15,
    rankq_alpha_success: float = 0.0,
    rankq_alpha_failure: float = 0.0,
    rankq_margin: float = 0.0,
    rankq_margin_relative: bool = False,
    action_mask: torch.Tensor | None = None,
    target_noise_std: float = 0.0,
    target_noise_clip: float = 0.3,
    action_clip_delta: float | None = None,
    slew_rate_limit: float | None = None,
    actor_deploy_scale: float = 1.0,
    info: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """TD3-style chunk-level TD loss with correct truncated-chunk handling.

    Uses actual_steps to compute the correct bootstrap exponent gamma^k
    instead of always using gamma^C.

    If `batch["rankq_outcome"]` (or the backward-compatible `"outcome"`
    fallback) is present and either rankq_alpha_* is > 0, adds
    RankQ's self-supervised ranking loss (see rankq_ranking_loss above) as
    an additional term. Fully backward compatible: with the default alphas
    (or no "outcome" key in batch) this is a no-op. `action_mask` (see
    _masked_candidate) is forwarded to it unchanged.

    `target_noise_std` > 0 enables TD3-style target policy smoothing:
    clipped Gaussian noise (std=target_noise_std, clipped to
    +/-target_noise_clip) is added to the target actor's action before
    evaluating target_critic on it. Without this, the critic is free to fit
    an arbitrarily sharp/discontinuous function of action right at whatever
    single point mu_next happens to be -- the actor's own subsequent
    gradient-ascent update then follows that same local sharpness, which
    reads as jittery real-robot output even though nothing about the
    physical actuators changed. Averaging the bootstrap target over a small
    neighborhood of actions (this is what the added+clipped noise
    approximates) forces the critic toward a locally smooth fit instead.
    Masked the same way as rankq_ranking_loss's candidates, since a frozen
    arm under actor_rl_arm="left" can't actually receive this noise either
    (mu_next is already pinned to ref there). Default 0.0 preserves the
    exact prior behavior (no smoothing) for existing callers.

    `action_clip_delta` (see project_action_delta) is applied to mu_next
    both before AND after the smoothing noise above, so target_noise_clip
    can never push the bootstrap action outside action_clip_delta of
    ref_next regardless of how the two are configured relative to each
    other -- the noised action is re-projected back into the same
    ref+/-action_clip_delta neighborhood the deployed actor can actually
    reach, instead of exploring a physically-unreachable region. None
    (default) preserves the exact prior behavior (mu_next only ever
    clamped to [-1,1], independent of any deployment delta bound).

    If `info` is passed, it is populated in place with the TD-only and
    RankQ-only components (`"loss_critic_td"`, `"loss_critic_rankq"`,
    both detached) so callers can log them separately -- the two terms can
    differ by several times in magnitude (RankQ's softplus floor is ~ln(2)
    per unsatisfied ranking pair regardless of how well the TD fit has
    converged), and the returned scalar alone can't be un-summed after the
    fact. Purely additive/optional: the returned loss is unchanged either way.

    `actor_deploy_scale` mirrors the runtime residual gate: 0 bootstraps at
    the exact VLA reference, values in (0, 1) use the same partial Actor
    residual currently allowed onto hardware, and 1 preserves the full
    policy behavior. This keeps TD targets aligned with physical rollout
    during the post-warmup deployment ramp.
    """
    if not 0.0 <= actor_deploy_scale <= 1.0:
        raise ValueError("actor_deploy_scale must be in [0, 1]")
    x = batch["state_vec"]
    ref = batch["ref_chunk_flat"]
    actual_steps = batch.get("actual_steps")
    valid_mask = _valid_action_mask(batch["exec_chunk_flat"], actual_steps, C)
    a = _canonicalize_partial_action(batch["exec_chunk_flat"], ref, valid_mask)
    x_next = batch["next_state_vec"]
    ref_next = batch["next_ref_flat"]
    reward_seq = batch["reward_seq"]
    done = batch["done"]

    with torch.no_grad():
        # Use deterministic mean for target action (TD3-style). Project onto
        # the same ref+/-action_clip_delta neighborhood the deployed actor is
        # bound to (see project_action_delta) -- matches what actor_loss does
        # to mu below, so the critic's TD target reflects the policy that
        # actually runs on hardware, not an unconstrained hypothetical one.
        mu_next, _ = actor.forward(x_next, ref_next)
        if actor_deploy_scale <= 0.0:
            mu_next = ref_next
        else:
            mu_next = ref_next + actor_deploy_scale * (mu_next - ref_next)
            mu_next = project_action_delta(mu_next, ref_next, action_clip_delta)
        if target_noise_std > 0:
            smoothing_noise = torch.randn_like(mu_next) * target_noise_std
            smoothing_noise = smoothing_noise.clamp(-target_noise_clip, target_noise_clip)
            mu_next = _masked_candidate(mu_next, mu_next + smoothing_noise, action_mask)
            # Re-bound after adding noise: target_noise_clip is tuned for
            # smoothing the Q-landscape and may be larger than
            # action_clip_delta (e.g. TD3's usual convention scales it to
            # the full action range) -- without this, the noised sample
            # could land outside the deployable neighborhood regardless of
            # how the two are configured relative to each other. A second
            # project_action_delta() call here (as before) is wrong:
            # that projection is a smooth *contraction* toward ref (tanh),
            # not a boundary-only clip, so applying it twice compounds
            # shrinkage instead of leaving already-in-bound points alone --
            # e.g. a raw delta of exactly action_clip_delta lands at ~0.76x
            # after one call and ~0.64x after two, systematically pulling
            # this *target* action closer to ref than the action
            # project_action_delta's own docstring promises actor_loss/
            # deployment ever produce for the same state. A hard clamp is
            # the right tool for this second step instead: identity for
            # anything already in bound (only reshapes genuine excursions
            # from the noise), and project_action_delta's smooth-gradient
            # property isn't needed here -- this whole function runs under
            # the enclosing no_grad(), so no gradient flows through this
            # re-bound either way.
            if action_clip_delta is not None:
                mu_next = (
                    ref_next if action_clip_delta <= 0
                    else ref_next + (mu_next - ref_next).clamp(-action_clip_delta, action_clip_delta)
                )
        if action_clip_delta is None and actor_deploy_scale > 0.0:
            mu_next = mu_next.clamp(-1.0, 1.0)
        # Replay stores physical behavior actions, not the counterfactual
        # target actor's preceding residual.  Start the target residual from
        # zero rather than inventing one by subtracting a mismatched reference.
        if actor_deploy_scale > 0.0:
            mu_next = _apply_slew_rate_limit_flat(
                mu_next, ref_next, C, slew_rate_limit
            )
        q_next = target_critic.min_q(x_next, mu_next)
        upper = target_q_clip if (target_q_clip is not None and target_q_clip > 0) else None
        # Default lower bound stays -target_q_clip so existing configs are
        # unaffected; target_q_min overrides it. Setting it to 0 for a
        # non-negative reward function is not a heuristic -- Q is a
        # discounted sum of rewards, so it is provably >= 0 there, and the
        # negative half is a region the bootstrap should never have been
        # allowed to enter. It matters because backup fits Q(s, a_data) but
        # bootstraps Q(s', pi(s')): when the actor is worse than the behavior
        # data (BC keeps it there by design), every backup subtracts that
        # gap, and over a long episode the sum walks Q well below zero with
        # no TD error to show for it -- each step stays locally consistent.
        lower = target_q_min if target_q_min is not None else (
            -upper if upper is not None else None
        )
        if lower is not None or upper is not None:
            q_next = q_next.clamp(min=lower, max=upper)
        r = discounted_chunk_return(reward_seq, gamma, actual_steps)

        # Bootstrap with gamma^k where k = actual steps executed
        if actual_steps is not None:
            bootstrap_exp = actual_steps.unsqueeze(-1).float()
        else:
            bootstrap_exp = torch.full_like(done.unsqueeze(-1), C, dtype=torch.float32)
        bootstrap = (gamma ** bootstrap_exp) * (1.0 - done.unsqueeze(-1)) * q_next
        target = r + bootstrap

    q1, q2 = critic(x, a)
    td_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
    loss = td_loss

    rankq_loss: torch.Tensor | None = None
    # ``outcome`` is also used by Actor demo BC, so online training supplies
    # a separate rankq_outcome that can exclude offline demonstrations without
    # discarding their successful-supervision label.
    outcome = batch.get("rankq_outcome", batch.get("outcome"))
    if outcome is not None and (rankq_alpha_success > 0 or rankq_alpha_failure > 0):
        rankq_loss = rankq_ranking_loss(
            critic, x, a, outcome,
            noise_scale=rankq_noise_scale,
            alpha_success=rankq_alpha_success,
            alpha_failure=rankq_alpha_failure,
            margin=rankq_margin,
            margin_relative=rankq_margin_relative,
            action_mask=(valid_mask if action_mask is None else valid_mask * action_mask),
        )
        loss = loss + rankq_loss

    if info is not None:
        info["loss_critic_td"] = td_loss.detach()
        info["loss_critic_rankq"] = (
            rankq_loss.detach() if rankq_loss is not None else td_loss.new_zeros(())
        )
    return loss


def _actor_bc_terms(
    actor: ChunkActor,
    batch: dict[str, torch.Tensor],
    mu: torch.Tensor,
    valid_mask: torch.Tensor,
    action_clip_delta: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return separate VLA-anchor and trusted-demonstration BC terms.

    Demonstrated elements are removed from the ordinary VLA anchor instead
    of receiving two conflicting targets.  ``demo_bc`` is therefore free to
    use a dedicated weight strong enough to remain meaningful when Critic Q
    gradients grow during online training.
    """
    ref = batch["ref_chunk_flat"]
    demo_mask = torch.zeros_like(mu)
    demo_target = ref
    intervention_mask = batch.get("intervention_mask_flat")
    if intervention_mask is not None and "exec_chunk_flat" in batch:
        demo_mask = intervention_mask.to(mu).clamp(0.0, 1.0) * valid_mask
        outcome = batch.get("outcome")
        if outcome is not None:
            successful_demo = (outcome.to(mu) >= 0.5).view(-1, 1)
            demo_mask = demo_mask * successful_demo
        actor_action_mask = getattr(actor, "action_mask", None)
        if actor_action_mask is not None:
            demo_mask = demo_mask * actor_action_mask.to(mu)
        demo_target = batch["exec_chunk_flat"].to(mu)
        if action_clip_delta is not None:
            if action_clip_delta <= 0:
                demo_target = ref
            else:
                # Invert project_action_delta so projecting the learned raw
                # output at deployment lands on the reachable demonstration.
                ratio = ((demo_target - ref) / action_clip_delta).clamp(-0.999, 0.999)
                demo_target = ref + action_clip_delta * torch.atanh(ratio)
        else:
            demo_target = demo_target.clamp(-1.0, 1.0)

    vla_mask = valid_mask * (1.0 - demo_mask)
    vla_bc = (((mu - ref) ** 2) * vla_mask).sum(dim=-1).mean()
    demo_bc = (((mu - demo_target) ** 2) * demo_mask).sum(dim=-1).mean()
    return vla_bc, demo_bc


def actor_behavior_cloning_loss(
    actor: ChunkActor,
    batch: dict[str, torch.Tensor],
    beta: float,
    demo_bc_weight: float | None = None,
    action_clip_delta: float | None = None,
    chunk_length: int | None = None,
    info: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """BC-only Actor update, safe to run before the Critic is trustworthy.

    This is used during online warmup/critic-only phases. It deliberately has
    no Q term: successful offline demonstrations and successful human
    corrections can train the Actor immediately without exposing it to a
    random or immature Critic.
    """
    x = batch["state_vec"]
    ref = batch["ref_chunk_flat"]
    mu, _ = actor.forward(x, ref, training=True)
    valid_mask = (
        torch.ones_like(mu)
        if chunk_length is None
        else _valid_action_mask(mu, batch.get("actual_steps"), chunk_length)
    )
    vla_bc, demo_bc = _actor_bc_terms(
        actor, batch, mu, valid_mask, action_clip_delta
    )
    effective_demo_weight = beta if demo_bc_weight is None else demo_bc_weight
    loss = beta * vla_bc + effective_demo_weight * demo_bc
    if info is not None:
        info["loss_actor_vla_bc"] = vla_bc.detach()
        info["loss_actor_demo_bc"] = demo_bc.detach()
        info["loss_actor_bc_only"] = loss.detach()
    return loss


def actor_loss(
    actor: ChunkActor,
    critic: TwinCritic,
    batch: dict[str, torch.Tensor],
    beta: float,
    demo_bc_weight: float | None = None,
    action_clip_delta: float | None = None,
    smoothness_weight: float = 0.0,
    chunk_length: int | None = None,
    slew_rate_limit: float | None = None,
    actor_deploy_scale: float = 1.0,
    info: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Q-maximization + BC regularization toward VLA reference.

    Uses deterministic mean (not noisy samples) for stable optimization.
    BC term is the per-sample squared distance summed across action dims, then
    averaged over the batch — matching the paper's β-scaling convention. This
    differs from mean-MSE by a factor of C*D_flat.

    `action_clip_delta` (see project_action_delta) projects `mu` into
    ref+/-action_clip_delta before querying the critic, so -q.mean()'s
    gradient reflects the action that actually gets deployed (see
    RLTActionModifier.compute_chunk, which uses the same projection), not an
    unconstrained hypothetical one the robot would never execute. BC
    regularization still regresses the *raw*, pre-projection mu -- it needs
    an unsaturating gradient to keep pulling a runaway residual back toward
    ref; regressing the already-bounded projected action would saturate
    near +/-action_clip_delta and lose that pull-back signal right where
    it's needed most. None (default) leaves mu unconstrained here, matching
    prior behavior.

    `smoothness_weight` > 0 adds a penalty on adjacent-timestep differences
    in the actor residual (mu - ref) within each chunk (requires
    `chunk_length` to unflatten it) -- a soft, training-time complement to
    the runtime residual limiter.  Trusted VLA reference motion is not
    penalized. 0.0 (default) is a no-op.

    `actor_deploy_scale` affects only the Q action, matching the residual
    fraction physically deployed during online ramp-up. BC still supervises
    the full raw Actor so it can learn demonstrations safely in the
    background even while deployment remains at 0.
    """
    if not 0.0 <= actor_deploy_scale <= 1.0:
        raise ValueError("actor_deploy_scale must be in [0, 1]")
    x = batch["state_vec"]
    ref = batch["ref_chunk_flat"]
    mu, _ = actor.forward(x, ref, training=True)
    if actor_deploy_scale <= 0.0:
        mu_safe = ref
    else:
        mu_deployed = ref + actor_deploy_scale * (mu - ref)
        mu_safe = project_action_delta(mu_deployed, ref, action_clip_delta)
    if action_clip_delta is None and actor_deploy_scale > 0.0:
        mu_safe = mu_safe.clamp(-1.0, 1.0)
    if chunk_length is None:
        if slew_rate_limit is not None:
            raise ValueError("slew-aware actor loss requires chunk_length")
        valid_mask = torch.ones_like(mu)
    else:
        valid_mask = _valid_action_mask(mu, batch.get("actual_steps"), chunk_length)
        if actor_deploy_scale > 0.0:
            mu_safe = _apply_slew_rate_limit_flat(
                mu_safe,
                ref,
                chunk_length,
                slew_rate_limit,
            )
    q_action = _canonicalize_partial_action(mu_safe, ref, valid_mask)
    q = critic.min_q(x, q_action)

    # Successful offline demos and successful human corrections get their
    # own BC coefficient; ordinary elements retain the VLA anchor coefficient
    # beta. Keeping these terms separate prevents a large Critic gradient
    # from silently drowning the known-correct demonstration signal.
    vla_bc, demo_bc = _actor_bc_terms(
        actor, batch, mu, valid_mask, action_clip_delta
    )
    effective_demo_weight = beta if demo_bc_weight is None else demo_bc_weight
    q_objective = -q.mean()
    loss = q_objective + beta * vla_bc + effective_demo_weight * demo_bc
    if smoothness_weight > 0:
        if chunk_length is None:
            raise ValueError("smoothness_weight > 0 requires chunk_length to unflatten mu")
        # Penalize the actor contribution, not VLA reference motion.  A
        # zero-residual actor following a rapidly changing trusted reference
        # is already the desired behavior and should have zero smoothness
        # cost.
        residual_chunk = unflatten_chunk(mu - ref, chunk_length)
        pair_mask = unflatten_chunk(valid_mask, chunk_length)[:, 1:, :]
        smoothness = (
            ((residual_chunk[:, 1:, :] - residual_chunk[:, :-1, :]) ** 2) * pair_mask
        ).sum(dim=(-1, -2)).mean()
        loss = loss + smoothness_weight * smoothness
    else:
        smoothness = loss.new_zeros(())
    if info is not None:
        info["loss_actor_q"] = q_objective.detach()
        info["loss_actor_vla_bc"] = vla_bc.detach()
        info["loss_actor_demo_bc"] = demo_bc.detach()
        info["loss_actor_smoothness"] = smoothness.detach()
    return loss


def q_action_sensitivity(
    critic: TwinCritic,
    state_vec: torch.Tensor,
    action_flat: torch.Tensor,
    noise_scale: float = 0.15,
    action_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Diagnostic: how much does critic.min_q(s, ·) actually vary across
    different actions at the same state?

    A critic trained under sparse, long-horizon reward can trivially
    minimize TD-MSE by learning a state-only baseline and ignoring the
    action input almost entirely -- functionally collapsing into a value
    function V(s) even though it's architecturally still Q(s,a) (see RankQ
    paper Fig.1's ∂Q/∂a visualization for the same failure mode via a true
    gradient; this is a cheaper no-autograd proxy for periodic logging).
    Reuses the same masked candidate-action construction as
    rankq_ranking_loss (see _masked_candidate) so the comparison stays
    apples-to-apples -- and so a frozen arm under actor_rl_arm="left"
    doesn't get credited as "action sensitivity" it doesn't actually have.

    Returns a scalar: mean over the batch of the per-sample std across 5
    candidate actions' (exec/noisy/very_noisy/random/permuted) Q values.
    Near 0 means Q is action-insensitive (collapsed toward V(s)); a healthy
    value should track Q's own scale and stay well above 0.
    """
    with torch.no_grad():
        candidates = _rankq_candidate_actions(action_flat, noise_scale, action_mask)
        q_values = torch.stack(
            [critic.min_q(state_vec, a).squeeze(-1) for a in candidates.values()], dim=0
        )  # (5, B)
        return q_values.std(dim=0).mean()
