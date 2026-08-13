"""Offline RL training on collected data, producing a deployable policy.

A peer of online training, not a preprocessing step for it. Both run the
same algorithm (TD3+BC + RankQ) through the same ChunkACPolicy losses and
the same OnlineRLTrainer sampling; they differ only in where transitions
come from and what drives the update schedule:

    online   robot rollouts, updates keyed to episodes as they arrive
    offline  fixed buffers, updates driven by a gradient-step budget

Output is a normal ChunkACPolicy checkpoint (config.json +
model.safetensors), deployed exactly like an online one:

    evo-rlt-record full --policy-path outputs/offline_rl/run1/final ...

Data sources, both opened read-only:
  --online-state        latest_online_state.pt from robot runs (repeatable)
  --offline-cache-path  the fixed VLA demonstration cache

The online runs stay resumable from their own snapshots afterwards; nothing
here writes into their directories.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Offline RL training on collected buffers; outputs a deployable policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_argument_group("data (read-only)")
    src.add_argument(
        "--online-state", action="append", default=[], metavar="PATH",
        help="latest_online_state.pt to take transitions from. Repeatable; "
        "episode ids are re-based per source so runs cannot collide.",
    )
    src.add_argument("--offline-cache-path", default=None, help="Fixed VLA demonstration cache directory.")
    src.add_argument(
        "--offline-batch-fraction", type=float, default=0.5,
        help="Fraction of each batch drawn from the demonstration cache.",
    )

    init = p.add_argument_group("policy")
    init.add_argument(
        "--policy-path", default=None,
        help="Checkpoint dir supplying config.json (and weights under --warm-start). "
        "Defaults to the newest step_*/ beside the first --online-state.",
    )
    init.add_argument("--vla-path", default=None, help="Override config.vla_pretrained_path.")
    init.add_argument("--rl-token-path", default=None, help="Override config.rl_token_pretrained_path.")
    init.add_argument("--tokenizer-path", default=None, help="Override config.tokenizer_path.")
    init.add_argument(
        "--warm-start", action="store_true",
        help="Initialize Actor/Critic from --policy-path instead of from scratch. "
        "Off by default so a run's result reflects its own hyperparameters "
        "rather than what the source checkpoint had already learned.",
    )

    tr = p.add_argument_group("training")
    tr.add_argument("--output-dir", required=True)
    tr.add_argument("--gradient-steps", type=int, default=100_000)
    tr.add_argument("--batch-size", type=int, default=256)
    tr.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument(
        "--critic-only-steps", type=int, default=2_000,
        help="Steps before the Actor starts updating, so it does not ascend a "
        "randomly-initialized Critic. Offline counterpart of the online "
        "critic_only_episodes window.",
    )
    tr.add_argument(
        "--actor-deploy-scale", type=float, default=1.0,
        help="Fraction of the Actor residual the losses treat as deployed. Online this "
        "is a hardware safety gate ramped per episode by OnlineRLTrainer.start_episode(); "
        "offline there is no robot, so the meaningful value is 1.0 -- the Critic then "
        "bootstraps on pi(s') and actor_loss's -Q term is a real function of mu. At 0.0 "
        "both collapse onto the VLA reference: the TD target evaluates the reference "
        "policy instead of the learned one, and -Q.mean() stops depending on the Actor "
        "at all, leaving it trained by the BC terms alone (i.e. plain behavior cloning, "
        "no policy improvement, whatever --gradient-steps says). Lower it only to "
        "reproduce a specific point on the online deployment ramp.",
    )

    hp = p.add_argument_group("hyperparameters (unset = inherit from the config.json)")
    for name in ("gamma", "beta", "demo-bc-weight", "tau", "target-q-clip", "target-q-min",
                 "rankq-margin", "rankq-alpha-success", "rankq-alpha-failure",
                 "rankq-noise-scale", "target-noise-std", "actor-action-clip-delta",
                 "actor-slew-rate-limit", "actor-smoothness-weight",
                 "lr-actor", "lr-critic"):
        hp.add_argument(f"--{name}", type=float, default=None)
    hp.add_argument("--actor-update-interval", type=int, default=None)
    hp.add_argument("--rankq-margin-relative", action=argparse.BooleanOptionalAction, default=None)

    io = p.add_argument_group("logging / evaluation / checkpoints")
    io.add_argument("--log-every", type=int, default=500)
    io.add_argument("--save-every", type=int, default=10_000)
    io.add_argument(
        "--eval-every", type=int, default=2_000,
        help="Evaluate on the cache's held-out val split every N steps. 0 disables.",
    )
    io.add_argument("--eval-batches", type=int, default=10)
    io.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    io.add_argument("--wandb-project", default="rlt-offline")
    io.add_argument("--wandb-entity", default=None)
    io.add_argument("--wandb-run-name", default=None)
    return p


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def _newest_step_dir(run_dir: Path) -> Path | None:
    dirs = sorted((p.parent for p in run_dir.glob("step_*/config.json")), key=lambda p: p.name)
    return dirs[-1] if dirs else None


def _resolve_policy_config(args):
    """ChunkACPolicyConfig from a saved checkpoint, plus that checkpoint's path.

    The architecture fields must match the weights that get loaded, and the
    hyperparameters are the baseline the CLI overrides are measured against.
    """
    from evo_rlt.adapters.lerobot.policies.configuration_rlt_ac import ChunkACPolicyConfig

    policy_path = args.policy_path
    if policy_path is None and args.online_state:
        found = _newest_step_dir(Path(args.online_state[0]).resolve().parent)
        if found is None:
            raise SystemExit(
                "No step_*/config.json found beside the first --online-state; "
                "pass --policy-path explicitly."
            )
        policy_path = str(found)
        logger.info("Policy config taken from %s", policy_path)
    if policy_path is None:
        raise SystemExit("Provide --policy-path, or --online-state to infer it from.")

    cfg = ChunkACPolicyConfig()
    stored = json.loads((Path(policy_path) / "config.json").read_text())
    # Assign only real dataclass fields; the dump also carries read-only
    # properties such as `type`, which have no setter.
    writable = {f.name for f in dataclasses.fields(cfg)}
    for key, value in stored.items():
        if key in writable:
            setattr(cfg, key, value)
    for attr, value in (
        ("vla_pretrained_path", args.vla_path),
        ("rl_token_pretrained_path", args.rl_token_path),
        ("tokenizer_path", args.tokenizer_path),
    ):
        if value is not None:
            setattr(cfg, attr, value)
    cfg.device = args.device
    return cfg, Path(policy_path)


def _apply_overrides(policy_cfg, online_cfg, args) -> dict[str, Any]:
    changed: dict[str, Any] = {}

    def put(cfg, attr, value):
        if value is None:
            return
        before = getattr(cfg, attr, None)
        if before != value:
            changed[attr] = {"from": before, "to": value}
        setattr(cfg, attr, value)

    put(policy_cfg, "gamma", args.gamma)
    put(policy_cfg, "beta", args.beta)
    put(policy_cfg, "actor_demo_bc_weight", args.demo_bc_weight)
    put(policy_cfg, "tau", args.tau)
    put(policy_cfg, "target_q_clip", args.target_q_clip)
    put(policy_cfg, "target_q_min", args.target_q_min)
    put(policy_cfg, "rankq_margin", args.rankq_margin)
    put(policy_cfg, "rankq_margin_relative", args.rankq_margin_relative)
    put(policy_cfg, "rankq_alpha_success", args.rankq_alpha_success)
    put(policy_cfg, "rankq_alpha_failure", args.rankq_alpha_failure)
    put(policy_cfg, "rankq_noise_scale", args.rankq_noise_scale)
    put(policy_cfg, "target_noise_std", args.target_noise_std)
    put(policy_cfg, "actor_action_clip_delta", args.actor_action_clip_delta)
    put(policy_cfg, "actor_slew_rate_limit", args.actor_slew_rate_limit)
    put(policy_cfg, "actor_smoothness_weight", args.actor_smoothness_weight)
    if args.actor_update_interval is not None:
        put(policy_cfg, "actor_update_interval", int(args.actor_update_interval))
    put(online_cfg, "lr_actor", args.lr_actor)
    put(online_cfg, "lr_critic", args.lr_critic)
    put(online_cfg, "batch_size", int(args.batch_size))
    put(online_cfg, "offline_batch_fraction", args.offline_batch_fraction)
    return changed


def _cache_reward_scale(cache_path: str | None) -> dict[str, float]:
    """The reward scale a transition cache was built under.

    OnlineRLTrainer refuses to mix a cache built at a different scale, and it
    is right to: the buffers' stored reward_seq bakes the scale in. Adopt the
    cache's own values rather than OnlineRLConfig's unrelated defaults.
    """
    if not cache_path:
        return {}
    meta = Path(cache_path)
    if meta.is_file():
        meta = meta.parent
    meta = meta / "cache_metadata.json"
    if not meta.is_file():
        return {}
    data = json.loads(meta.read_text())
    return {
        key: float(data[key])
        for key in ("milestone_reward", "terminal_reward", "time_decay")
        if key in data
    }


def _load_online_transitions(paths: list[str]) -> tuple[list, list[dict]]:
    """Concatenate snapshots' transitions, re-basing episode ids per source.

    Each snapshot numbers episodes from 0 independently, so merging as-is
    would fuse unrelated episodes under a shared id -- exactly what
    outcome_labels() and RankQ's success/failure grouping key on.
    """
    merged: list = []
    summary: list[dict] = []
    next_base = 0
    for path in paths:
        state = torch.load(path, map_location="cpu", weights_only=False)
        transitions = list(state["replay_buffer"])
        ids = {int(t.episode_id.item()) for t in transitions}
        offset = next_base
        for t in transitions:
            t.episode_id = t.episode_id + offset
        next_base = offset + (max(ids) + 1 if ids else 0)
        merged.extend(transitions)
        summary.append({
            "path": path, "transitions": len(transitions),
            "episodes": len(ids), "episode_id_offset": offset,
        })
        logger.info(
            "Loaded %d transitions / %d episodes from %s (episode ids +%d)",
            len(transitions), len(ids), path, offset,
        )
    return merged, summary


def _guard_output_dir(args, output_dir: Path) -> None:
    for path in args.online_state:
        source = Path(path).resolve().parent
        out = output_dir.resolve()
        if out == source or source in out.parents:
            raise SystemExit(
                f"--output-dir ({out}) is inside the online run at {source}.\n"
                "Use a separate directory so that run stays resumable."
            )


def _warm_start(policy, policy_path: Path, device: str) -> None:
    """Load Actor/Critic/target_critic out of the checkpoint's safetensors.

    Deliberately not ChunkACPolicy.from_pretrained(): that builds a *second*
    full policy -- including another frozen pi0.5 VLA and RL-token encoder,
    on the same device, since _resolve_policy_config() sets cfg.device --
    only to discard everything but three small MLPs. Two VLA copies do not
    fit in 16GB, which is why a no-warm-start run had room and this one does
    not. The three modules sit under plain `actor.` / `critic.` /
    `target_critic.` prefixes, so lifting them out directly costs one file
    read and no GPU memory.
    """
    from safetensors.torch import load_file

    weights = load_file(str(Path(policy_path) / "model.safetensors"), device="cpu")
    for name, module in (
        ("actor", policy.actor),
        ("critic", policy.critic),
        ("target_critic", policy.target_critic),
    ):
        prefix = f"{name}."
        sub = {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}
        if not sub:
            raise SystemExit(
                f"No {prefix}* tensors in {policy_path}/model.safetensors -- "
                "not a ChunkACPolicy checkpoint?"
            )
        module.load_state_dict(sub)
    policy.to(device)
    logger.info("Warm start: Actor/Critic loaded from %s", policy_path)


# ---------------------------------------------------------------------------
# Evaluation on the held-out split
# ---------------------------------------------------------------------------

def _load_val_buffer(cache_path: str | None, capacity: int):
    from evo_rlt.adapters.lerobot.offline_dataset import load_transition_cache

    if not cache_path:
        return None
    path = Path(cache_path)
    if not (path / "chunk_transitions_val.pt").is_file():
        logger.warning("No chunk_transitions_val.pt in %s -- evaluation disabled.", path)
        return None
    return load_transition_cache(str(path), "val", capacity=capacity)


@torch.no_grad()
def _evaluate(policy, val_buffer, batch_size: int, num_batches: int, device) -> dict[str, float]:
    """Metrics on episodes the run never trained on.

    q_gap is the one to watch: Q(actor's action) - Q(the demonstrated
    action). It should stay <= 0. Positive means the Critic prefers the
    Actor's own output over a successful demonstration on data it has never
    seen -- value hallucination, regardless of how low the training TD loss is.
    """
    from evo_rlt.core.losses import _valid_action_mask
    from evo_rlt.core.utils import project_action_delta

    was_training = policy.training
    policy.eval()
    clip = getattr(policy.config, "actor_action_clip_delta", None)
    chunk_length = policy.config.chunk_length
    totals = {"expert_action_mse": 0.0, "ref_action_mse": 0.0, "q_gap": 0.0,
              "q_policy": 0.0, "q_expert": 0.0}
    for _ in range(num_batches):
        raw = val_buffer.sample(batch_size)
        state = raw["state_vec"].to(device)
        ref = raw["ref_chunk_flat"].to(device)
        expert = raw["exec_chunk_flat"].to(device)
        steps = raw["actual_steps"].to(device)

        mu, _ = policy.actor.forward(state, ref, training=False)
        valid = _valid_action_mask(mu, steps, chunk_length)
        deployed = project_action_delta(mu, ref, clip)
        if clip is None:
            deployed = deployed.clamp(-1.0, 1.0)

        totals["expert_action_mse"] += (((mu - expert) ** 2) * valid).sum(-1).mean().item()
        totals["ref_action_mse"] += (((mu - ref) ** 2) * valid).sum(-1).mean().item()
        q_policy = policy.critic.min_q(state, deployed).mean().item()
        q_expert = policy.critic.min_q(state, expert).mean().item()
        totals["q_policy"] += q_policy
        totals["q_expert"] += q_expert
        totals["q_gap"] += q_policy - q_expert
    if was_training:
        policy.train()
    return {f"val/{k}": v / num_batches for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(asctime)s %(message)s", datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    if not args.online_state and not args.offline_cache_path:
        raise SystemExit("Nothing to train on: pass --online-state and/or --offline-cache-path.")

    output_dir = Path(args.output_dir)
    _guard_output_dir(args, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ChunkACPolicy resolves the RL-token checkpoint through LeRobot's config
    # registry, which only knows the rlt_* types after this call.
    from evo_rlt.adapters.lerobot.registry import register as register_rlt_policies

    register_rlt_policies()

    policy_cfg, policy_path = _resolve_policy_config(args)

    from evo_rlt.adapters.lerobot.policies.modeling_rlt_ac import ChunkACPolicy
    from evo_rlt.adapters.lerobot.record.backend import OnlineRLConfig
    from evo_rlt.adapters.lerobot.record.online_trainer import OnlineRLTrainer

    online_cfg = OnlineRLConfig()
    online_cfg.save_dir = str(output_dir)
    online_cfg.offline_cache_path = args.offline_cache_path
    online_cfg.replay_capacity = 10**9  # never evict; the buffers are fixed
    online_cfg.save_every_episodes = 10**9  # checkpointing is driven by _train
    online_cfg.wandb = bool(args.wandb)
    online_cfg.wandb_project = args.wandb_project
    online_cfg.wandb_entity = args.wandb_entity
    online_cfg.wandb_run_name = args.wandb_run_name
    online_cfg.wandb_run_id = None
    online_cfg.wandb_resume = "never"
    for key, value in _cache_reward_scale(args.offline_cache_path).items():
        setattr(online_cfg, key, value)
    changed = _apply_overrides(policy_cfg, online_cfg, args)

    logger.info("Building ChunkACPolicy (loads VLA + RL token; needed for a deployable checkpoint)")
    policy = ChunkACPolicy(policy_cfg).to(args.device)
    if args.warm_start:
        _warm_start(policy, policy_path, args.device)
    else:
        policy.target_critic.load_state_dict(policy.critic.state_dict())
        logger.info("Actor/Critic initialized from scratch")
    for p in policy.target_critic.parameters():
        p.requires_grad_(False)

    trainer = OnlineRLTrainer(policy, online_cfg, policy_cfg)
    # Must come after the constructor: it pins the scale to 0 for hardware
    # safety and relies on start_episode() to ramp it back up, which this
    # fixed-buffer loop never calls. Both copies matter -- the policy's drives
    # actor_loss/critic_loss, the trainer's drives _q_calibration_metrics'
    # deployed-action reconstruction.
    policy.set_actor_deploy_scale(args.actor_deploy_scale)
    trainer.actor_deploy_scale = args.actor_deploy_scale
    transitions, data_summary = _load_online_transitions(args.online_state)
    for t in transitions:
        trainer.replay_buffer.add(t)
    # Warmup gates on "enough robot episodes collected", which is already
    # settled here. The remaining question -- has the Critic had a head start
    # before the Actor ascends it -- is --critic-only-steps.
    trainer.warmup_completed_at_episode = 0

    val_buffer = None
    if args.eval_every > 0:
        val_buffer = _load_val_buffer(args.offline_cache_path, online_cfg.replay_capacity)

    n_offline = len(trainer.offline_buffer) if trainer.offline_buffer is not None else 0
    successes, failures = trainer.replay_buffer.count_outcomes()
    logger.info(
        "Data: %d online (%d success / %d failure episodes) + %d offline demos = %d transitions"
        "%s",
        len(trainer.replay_buffer), successes, failures, n_offline,
        len(trainer.replay_buffer) + n_offline,
        f" | val split: {len(val_buffer)}" if val_buffer is not None else "",
    )
    if changed:
        logger.info("Overrides: %s", json.dumps(changed, default=str))
    (output_dir / "run_config.json").write_text(json.dumps(
        {"args": vars(args), "overrides": changed, "online_sources": data_summary},
        indent=2, default=str,
    ))

    try:
        _train(trainer, policy, policy_cfg, online_cfg, args, output_dir, val_buffer, policy_path)
    finally:
        trainer.close()


def _train(trainer, policy, policy_cfg, online_cfg, args, output_dir, val_buffer, template) -> None:
    """Gradient-step loop over the fixed buffers.

    Mirrors OnlineRLTrainer.maybe_update()'s inner loop rather than calling
    it -- that method's contract is "train in proportion to newly collected
    data", which has no meaning against a fixed buffer. The per-step work
    (mixed-batch sampling, policy.forward, grad clipping at 1.0, soft target
    update after the optimizer step) is the same.
    """
    from evo_rlt.core.utils import soft_update, unflatten_chunk

    device = torch.device(args.device)
    chunk_length = policy_cfg.chunk_length
    real_tau = policy_cfg.tau
    real_actor_interval = policy_cfg.actor_update_interval
    start = time.perf_counter()
    policy.train()
    # forward() soft-updates the target *before* the optimizer step; disable
    # that and do it correctly afterwards, as the online path does.
    policy_cfg.tau = 0.0

    try:
        for step in range(1, args.gradient_steps + 1):
            policy_cfg.actor_update_interval = (
                10**9 if step <= args.critic_only_steps else real_actor_interval
            )
            raw, n_off, n_on = trainer._sample_training_batch()
            batch = {
                "state_vec": raw["state_vec"],
                "exec_chunk": unflatten_chunk(raw["exec_chunk_flat"], chunk_length),
                "ref_chunk": unflatten_chunk(raw["ref_chunk_flat"], chunk_length),
                "reward_seq": raw["reward_seq"],
                "next_state_vec": raw["next_state_vec"],
                "next_ref_chunk": unflatten_chunk(raw["next_ref_flat"], chunk_length),
                "done": raw["done"],
                "actual_steps": raw["actual_steps"],
                "outcome": raw["outcome"],
                "rankq_outcome": raw["rankq_outcome"],
                "intervention_mask_flat": raw["intervention_mask_flat"],
            }
            batch = {k: v.to(device) for k, v in batch.items()}

            loss, info = policy.forward(batch)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f"Non-finite loss at step {step}")
            trainer.critic_optimizer.zero_grad()
            trainer.actor_optimizer.zero_grad()
            loss.backward()
            actor_gn = torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), max_norm=1.0)
            critic_gn = torch.nn.utils.clip_grad_norm_(policy.critic.parameters(), max_norm=1.0)
            trainer.critic_optimizer.step()
            trainer.actor_optimizer.step()
            soft_update(policy.target_critic, policy.critic, real_tau)

            do_log = step % args.log_every == 0 or step == 1
            do_eval = val_buffer is not None and (step % args.eval_every == 0 or step == 1)
            if do_log or do_eval:
                metrics = {
                    f"offline_rl/{k}": (v.item() if torch.is_tensor(v) else v)
                    for k, v in info.items()
                }
                metrics["offline_rl/actor_grad_norm"] = actor_gn.item()
                metrics["offline_rl/critic_grad_norm"] = critic_gn.item()
                metrics["offline_rl/batch_offline"] = n_off
                metrics["offline_rl/batch_online"] = n_on
                metrics["offline_rl/steps_per_s"] = step / (time.perf_counter() - start)
                metrics.update({
                    k.replace("online_rl/", "offline_rl/"): v
                    for k, v in trainer._q_calibration_metrics(raw, device).items()
                })
                if do_eval:
                    metrics.update(_evaluate(
                        policy, val_buffer, online_cfg.batch_size, args.eval_batches, device,
                    ))
                trainer._log(metrics, step=step)
                logger.info(
                    "step %6d/%d | loss=%.4f critic=%.4f actor=%s | ratio=%s margin=%s%s | %.1f it/s",
                    step, args.gradient_steps, loss.item(),
                    metrics.get("offline_rl/loss_critic", float("nan")),
                    _fmt(metrics.get("offline_rl/loss_actor")),
                    _fmt(metrics.get("offline_rl/q_vs_return_ratio"), "%.2f"),
                    _fmt(metrics.get("offline_rl/q_rank_margin"), "%+.4f"),
                    f" | val q_gap={metrics['val/q_gap']:+.3f}" if "val/q_gap" in metrics else "",
                    metrics["offline_rl/steps_per_s"],
                )
            if args.save_every > 0 and step % args.save_every == 0:
                _save(policy, output_dir / f"step_{step:06d}")
    finally:
        policy_cfg.tau = real_tau
        policy_cfg.actor_update_interval = real_actor_interval
        policy.eval()

    _save(policy, output_dir / "final")
    logger.info("Finished in %.1f min", (time.perf_counter() - start) / 60)
    logger.info("Deploy with: evo-rlt-record full --policy-path %s ...", output_dir / "final")


def _fmt(value, spec: str = "%.4f") -> str:
    return "n/a" if value is None else spec % value


def _save(policy, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(path)
    logger.info("Checkpoint saved to %s", path)


if __name__ == "__main__":
    sys.exit(main())
