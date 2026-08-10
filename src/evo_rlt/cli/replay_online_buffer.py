"""Re-train Actor/Critic offline on an already-collected online replay buffer.

Purpose: try a hyperparameter change against the data you already have,
before spending robot hours finding out whether it helped. The critic
pathologies this exists to catch (value overestimation, inverted action
ranking) are visible from the buffer alone -- no rollout needed.

Runs the *same* update path as live training: it drives
OnlineRLTrainer.maybe_update() through a minimal policy object that borrows
ChunkACPolicy's own forward/loss methods, so nothing here can silently
diverge from what the robot does. The VLA and RL-token encoder are not
loaded (state_vec is already encoded in the buffer), which is why this
runs in minutes on CPU-sized batches.

Strictly read-only with respect to the source run: the state file is opened
once for reading, every write goes to --output-dir, and the script refuses
to start if that resolves inside the source run's directory. Resuming live
online training from the original --resume-from afterwards is unaffected.

Example:
    python -m evo_rlt.cli.replay_online_buffer \\
        --state-path outputs/online_rl/0807_online_rl/eval_online_rl_111713/latest_online_state.pt \\
        --output-dir outputs/offline_replay/gamma099 \\
        --cycles 60 --gamma 0.99 --target-q-clip 3.0 --rankq-margin-relative
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


class ReplayPolicy(nn.Module):
    """Minimal stand-in exposing exactly what OnlineRLTrainer touches.

    forward/_actor_loss_without_critic_grads/_coerce_batch are *bound from
    ChunkACPolicy itself* rather than reimplemented -- the point of this
    script is to test the real update, so a second copy of that logic here
    would defeat it the first time the two drifted apart.
    """

    def __init__(self, config, actor, critic, target_critic):
        super().__init__()
        self.config = config
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.actor_deploy_scale = 1.0
        self.register_buffer("_critic_step", torch.zeros((), dtype=torch.long))

    def set_actor_deploy_scale(self, scale: float) -> None:
        self.actor_deploy_scale = scale

    def save_pretrained(self, save_dir) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "target_critic_state_dict": self.target_critic.state_dict(),
            },
            save_dir / "actor_critic.pt",
        )


def _bind_chunk_ac_methods() -> None:
    from evo_rlt.adapters.lerobot.policies.modeling_rlt_ac import ChunkACPolicy

    for name in ("forward", "_actor_loss_without_critic_grads", "_coerce_batch"):
        setattr(ReplayPolicy, name, getattr(ChunkACPolicy, name))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Offline re-training on a collected online replay buffer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--state-path", required=True, help="latest_online_state.pt to read (never written).")
    p.add_argument("--output-dir", required=True, help="Where checkpoints/metrics go. Must be outside the source run.")
    p.add_argument("--cycles", type=int, default=60, help="Update cycles to run (one cycle ~ one episode's worth of updates).")
    p.add_argument(
        "--transitions-per-cycle", type=int, default=None,
        help="Transitions each cycle pretends arrived, which sets the update budget "
        "(x utd_ratio). Defaults to the source run's own mean transitions/episode.",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--warm-start", action="store_true",
        help="Continue from the snapshot's Actor/Critic weights instead of training "
        "fresh ones on its data. Off by default: weights carry the old run's "
        "hyperparameters baked in, so a config change measured on top of them shows "
        "whether it can repair that particular critic, not whether it would have "
        "prevented the problem. Use this only when repairability is the question.",
    )
    # Hyperparameters worth overriding. None = keep whatever the checkpoint's config had.
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--target-q-clip", type=float, default=None)
    p.add_argument("--rankq-margin", type=float, default=None)
    p.add_argument("--rankq-margin-relative", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--rankq-alpha-success", type=float, default=None)
    p.add_argument("--rankq-alpha-failure", type=float, default=None)
    p.add_argument("--actor-action-clip-delta", type=float, default=None)
    p.add_argument("--actor-slew-rate-limit", type=float, default=None)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--utd-ratio", type=int, default=None)
    p.add_argument("--lr-actor", type=float, default=None)
    p.add_argument("--lr-critic", type=float, default=None)
    p.add_argument("--offline-cache-path", default=None, help="Offline demo cache to mix in, as in live training.")
    p.add_argument("--offline-batch-fraction", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p


def _reset_module(module: nn.Module) -> None:
    """Re-randomize every layer that knows how to initialize itself."""
    for layer in module.modules():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()


def _newest_step_config(run_dir: Path) -> Path | None:
    """Newest step_NNNNNN/config.json in a run directory, if any."""
    candidates = sorted(
        (p for p in run_dir.glob("step_*/config.json")),
        key=lambda p: p.parent.name,
    )
    return candidates[-1] if candidates else None


def _guard_output_dir(state_path: Path, output_dir: Path) -> None:
    """Refuse to write anywhere the source run could be reading from."""
    source_dir = state_path.resolve().parent
    out = output_dir.resolve()
    if out == source_dir or source_dir in out.parents or out in source_dir.parents:
        raise SystemExit(
            f"--output-dir ({out}) overlaps the source run directory ({source_dir}).\n"
            "Pick a location outside it so live online training stays resumable."
        )


def _apply_overrides(policy_cfg, online_cfg, args) -> dict[str, Any]:
    """Push CLI overrides onto the checkpoint's configs; report what changed."""
    changed: dict[str, Any] = {}

    def put(cfg, attr, value):
        if value is None:
            return
        before = getattr(cfg, attr, None)
        if before != value:
            changed[attr] = {"from": before, "to": value}
        setattr(cfg, attr, value)

    put(policy_cfg, "gamma", args.gamma)
    put(policy_cfg, "target_q_clip", args.target_q_clip)
    put(policy_cfg, "rankq_margin", args.rankq_margin)
    put(policy_cfg, "rankq_margin_relative", args.rankq_margin_relative)
    put(policy_cfg, "rankq_alpha_success", args.rankq_alpha_success)
    put(policy_cfg, "rankq_alpha_failure", args.rankq_alpha_failure)
    put(policy_cfg, "actor_action_clip_delta", args.actor_action_clip_delta)
    put(policy_cfg, "actor_slew_rate_limit", args.actor_slew_rate_limit)
    put(policy_cfg, "beta", args.beta)
    put(policy_cfg, "utd_ratio", args.utd_ratio)
    put(online_cfg, "lr_actor", args.lr_actor)
    put(online_cfg, "lr_critic", args.lr_critic)
    put(online_cfg, "offline_cache_path", args.offline_cache_path)
    put(online_cfg, "offline_batch_fraction", args.offline_batch_fraction)
    put(online_cfg, "batch_size", args.batch_size)
    return changed


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(asctime)s %(message)s", datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)

    state_path = Path(args.state_path)
    output_dir = Path(args.output_dir)
    if not state_path.is_file():
        raise SystemExit(f"No such state file: {state_path}")
    _guard_output_dir(state_path, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _bind_chunk_ac_methods()
    from evo_rlt.adapters.lerobot.policies.configuration_rlt_ac import ChunkACPolicyConfig
    from evo_rlt.adapters.lerobot.record.online_trainer import OnlineRLTrainer
    from evo_rlt.core.actor import ChunkActor
    from evo_rlt.core.critic import TwinCritic

    logger.info("Reading %s (read-only)", state_path)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    # Read once here only to size the networks; the buffer the replay
    # actually trains on is restored through trainer.load_latest_state below.
    buffer = state["replay_buffer"]
    if not len(buffer):
        raise SystemExit("Replay buffer in the state file is empty; nothing to replay.")
    sample = buffer[0]
    state_dim = sample.state_vec.numel()
    chunk_dim = sample.ref_chunk.numel()

    # The state file holds weights and buffer but not the config they were
    # trained under; save_pretrained() writes that to the periodic
    # step_NNNNNN/ checkpoints instead. Take the newest one -- the
    # architecture fields have to match or load_state_dict below fails loudly,
    # and the hyperparameters are what the overrides are measured against.
    policy_cfg = ChunkACPolicyConfig()
    cfg_path = _newest_step_config(state_path.parent)
    if cfg_path is not None:
        stored = json.loads(cfg_path.read_text())
        # Assign only real dataclass fields: the config also exposes
        # read-only properties (`type`, ...) that are present in the dump but
        # have no setter.
        writable = {f.name for f in dataclasses.fields(policy_cfg)}
        for key, value in stored.items():
            if key in writable:
                setattr(policy_cfg, key, value)
        logger.info("Loaded policy config from %s", cfg_path)
    else:
        logger.warning(
            "No step_*/config.json under %s -- falling back to ChunkACPolicyConfig "
            "defaults. Architecture mismatches will surface as a load_state_dict "
            "error below; pass hyperparameters explicitly.",
            state_path.parent,
        )

    actor = ChunkActor(
        state_dim, chunk_dim,
        hidden_dim=policy_cfg.actor_hidden_dim, num_layers=policy_cfg.actor_num_layers,
        fixed_std=policy_cfg.actor_fixed_std, ref_dropout_p=policy_cfg.actor_ref_dropout_p,
        activation=policy_cfg.actor_activation, layer_norm=policy_cfg.actor_layer_norm,
        residual=policy_cfg.actor_residual, residual_to_ref=policy_cfg.actor_residual_to_ref,
    )
    critic = TwinCritic(
        state_dim, chunk_dim,
        hidden_dim=policy_cfg.critic_hidden_dim, num_layers=policy_cfg.critic_num_layers,
        activation=policy_cfg.critic_activation, layer_norm=policy_cfg.critic_layer_norm,
        residual=policy_cfg.critic_residual,
    )
    target_critic = copy.deepcopy(critic)
    for p in target_critic.parameters():
        p.requires_grad_(False)

    device = torch.device(args.device)
    policy = ReplayPolicy(policy_cfg, actor, critic, target_critic).to(device)

    online_cfg = _build_online_cfg(state, output_dir)
    changed = _apply_overrides(policy_cfg, online_cfg, args)
    trainer = OnlineRLTrainer(policy, online_cfg, policy_cfg)
    # Reuse the real resume path rather than restoring by hand: it also
    # carries critic_step, optimizer momentum, total_added and the warmup
    # anchor, and any of those getting out of sync would quietly change what
    # the replay measures.
    snapshot_episodes = trainer.load_latest_state(state_path)

    if args.warm_start:
        start_episode = snapshot_episodes
        if trainer.warmup_completed_at_episode is None:
            # A snapshot from before warmup completed would send the replay
            # into BC warm-start instead of the TD+RankQ path under test.
            trainer.warmup_completed_at_episode = 0
            logger.warning("Snapshot predates warmup completion; forcing warmup satisfied.")
        logger.info("Warm start: continuing the snapshot's weights (episode %d)", start_episode)
    else:
        # Default: keep the data, discard everything the previous run learned
        # from it. Re-runs the full curriculum (critic-only window, then the
        # actor unfreeze ramp) from episode 0 against a now-fixed buffer, so
        # the result speaks to the config rather than to this checkpoint.
        _reset_module(policy.actor)
        _reset_module(policy.critic)
        policy.target_critic.load_state_dict(policy.critic.state_dict())
        for p in policy.target_critic.parameters():
            p.requires_grad_(False)
        policy._critic_step = torch.zeros((), dtype=torch.long, device=device)
        trainer.actor_optimizer = torch.optim.Adam(
            policy.actor.parameters(), lr=online_cfg.lr_actor
        )
        trainer.critic_optimizer = torch.optim.Adam(
            policy.critic.parameters(), lr=online_cfg.lr_critic
        )
        # Let warmup resolve naturally so the curriculum that follows
        # (critic-only window, then the actor unfreeze ramp) runs in full.
        # Start the episode counter at warmup_episodes because that gate
        # counts *collected* episodes -- the buffer is already collected, and
        # from 0 the replay would spend its first cycles in BC warm-start
        # re-deciding a question the data has already answered.
        trainer.warmup_completed_at_episode = None
        start_episode = online_cfg.warmup_episodes
        logger.info("Fresh Actor/Critic on the snapshot's data only (no weights loaded)")

    per_cycle = args.transitions_per_cycle
    if per_cycle is None:
        episodes = len({int(t.episode_id.item()) for t in trainer.replay_buffer.buffer})
        per_cycle = max(1, round(len(trainer.replay_buffer) / max(episodes, 1)))
    buffer = trainer.replay_buffer

    logger.info(
        "Replay: %d transitions / %d episodes | %d cycles x %d transitions x utd %d "
        "= up to %d gradient steps | device=%s | weights=%s",
        len(buffer), snapshot_episodes, args.cycles, per_cycle, policy_cfg.utd_ratio,
        args.cycles * per_cycle * policy_cfg.utd_ratio, device,
        "warm-start" if args.warm_start else "fresh",
    )
    if changed:
        logger.info("Overrides: %s", json.dumps(changed, default=str))

    history = []
    for cycle in range(args.cycles):
        episode_id = start_episode + cycle
        trainer.actor_deploy_scale = trainer._actor_deploy_scale_for(episode_id)
        policy.set_actor_deploy_scale(trainer.actor_deploy_scale)
        # maybe_update() gates on total_added having grown; the buffer is
        # fixed here, so hand it a baseline that expresses the intended
        # update budget instead of adding fake transitions to the data.
        baseline = trainer.replay_buffer.total_added - per_cycle
        stats = trainer.maybe_update(episode_id, baseline)
        if stats is None:
            # maybe_update() also returns None while warmup is unresolved, in
            # which case it ran BC warm-start rather than nothing -- a cycle
            # worth continuing through, not an error.
            if trainer.warmup_completed_at_episode is None:
                logger.info("Cycle %d: warmup not yet resolved (BC warm-start).", cycle)
                continue
            logger.warning("Cycle %d ran no updates; stopping.", cycle)
            break
        row = {"cycle": cycle, "episode_id": episode_id, **_extract(stats)}
        # maybe_update() routes the calibration metrics to wandb/logging, not
        # into its return value, so recompute them here on a fresh batch.
        # Same helper the live run uses -- this is a read of the trainer's
        # state, not a second implementation of the measurement.
        raw, _, _ = trainer._sample_training_batch()
        row.update(
            {k.removeprefix("online_rl/"): v
             for k, v in trainer._q_calibration_metrics(raw, device).items()}
        )
        history.append(row)

    (output_dir / "replay_metrics.json").write_text(json.dumps(history, indent=2))
    policy.save_pretrained(output_dir / "final")
    _report(history, output_dir)


def _build_online_cfg(state: dict, output_dir: Path):
    """OnlineRLConfig pointed at output_dir, seeded from the source run."""
    from evo_rlt.adapters.lerobot.record.backend import OnlineRLConfig

    cfg = OnlineRLConfig()
    cfg.save_dir = str(output_dir)
    cfg.wandb = False
    cfg.replay_capacity = int(state.get("replay_capacity", 20000))
    cfg.offline_cache_path = state.get("offline_cache_path")
    frac = state.get("offline_batch_fraction")
    if frac is not None:
        cfg.offline_batch_fraction = float(frac)
    # The state file does not carry the reward scale, but the trainer refuses
    # to mix an offline cache built under a different one -- correctly, since
    # the buffer's stored reward_seq bakes it in. Adopt the cache's own scale
    # so the replay reproduces the run's reward semantics rather than
    # OnlineRLConfig's unrelated defaults.
    for key, value in _cache_reward_scale(cfg.offline_cache_path).items():
        setattr(cfg, key, value)
    # Never let the replay write a save the live run might pick up, and never
    # re-gate on warmup (already forced satisfied above).
    cfg.save_every_episodes = 10**9
    return cfg


def _cache_reward_scale(cache_path: str | None) -> dict[str, float]:
    """milestone/terminal/time_decay recorded in a transition cache's metadata."""
    if not cache_path:
        return {}
    meta_path = Path(cache_path)
    if meta_path.is_file():
        meta_path = meta_path.parent
    meta_path = meta_path / "cache_metadata.json"
    if not meta_path.is_file():
        return {}
    meta = json.loads(meta_path.read_text())
    scale = {
        key: float(meta[key])
        for key in ("milestone_reward", "terminal_reward", "time_decay")
        if key in meta
    }
    if scale:
        logger.info("Adopted reward scale from %s: %s", meta_path, scale)
    return scale


def _extract(stats: dict) -> dict:
    loss = stats.get("loss", {}) or {}
    keep = (
        "loss_critic", "loss_critic_td", "loss_critic_rankq", "loss_actor",
        "loss_actor_q", "loss_actor_vla_bc", "loss_actor_demo_bc",
        "q_action_sensitivity", "actor_grad_norm", "critic_grad_norm",
        "q_ref_mean", "empirical_return_mean", "q_vs_return_ratio",
        "q_rank_margin", "q_rank_correct_frac",
    )
    return {k: loss[k] for k in keep if k in loss}


def _report(history: list[dict], output_dir: Path) -> None:
    if not history:
        logger.warning("No cycles completed; nothing to report.")
        return
    first, last = history[0], history[-1]

    def fmt(row, key):
        return f"{row[key]:+.4f}" if key in row else "n/a"

    logger.info("=" * 72)
    logger.info("Replay finished: %d cycles -> %s", len(history), output_dir / "replay_metrics.json")
    logger.info("%-24s %12s %12s", "metric", "first cycle", "last cycle")
    for key in ("q_vs_return_ratio", "q_rank_margin", "q_rank_correct_frac",
                "q_ref_mean", "loss_critic_td", "loss_critic_rankq", "q_action_sensitivity"):
        if key in first or key in last:
            logger.info("%-24s %12s %12s", key, fmt(first, key), fmt(last, key))
    logger.info("-" * 72)
    logger.info("Targets: q_vs_return_ratio -> ~1 (Q calibrated to real returns)")
    logger.info("         q_rank_margin     -> > 0 (critic prefers the human takeover action)")
    logger.info("         q_rank_correct_frac -> > 0.5")
    logger.info("=" * 72)


if __name__ == "__main__":
    sys.exit(main())
