"""Build chunk-transition cache for ChunkACPolicy training.

Encodes each base frame in a LeRobotDataset through pi0.5 + the trained RL
Token encoder into a (state_vec, exec_chunk, ref_chunk, ...) tuple stored on
disk. The cache is consumed by ChunkTransitionDataset at AC training time.

This v2 replaces the legacy custom-load builder. It loads the preprocessor
directly from the SFT pi05 ckpt so the cache is byte-aligned with the deploy
normalization. Per-batch progress with elapsed time is written to stdout in
unbuffered mode so a hung run is visible immediately. Completed work is
checkpointed atomically every five episodes, so an interruption can forfeit
the current episode plus at most four completed episodes since the last save.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys
import time

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from evo_rlt.adapters.lerobot.policies.action_modifier import PrefixOutputCapture
from evo_rlt.adapters.lerobot.policies.configuration_rlt_token import RLTokenPolicyConfig
from evo_rlt.adapters.lerobot.policies.modeling_rlt_token import RLTokenPolicy
from evo_rlt.adapters.lerobot.policies.processor_rlt_token import make_rlt_token_pre_post_processors
from evo_rlt.adapters.lerobot.offline_dataset import build_overlap_frame_indices


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--demo-dataset-repo-id", required=True)
    p.add_argument("--demo-dataset-root", required=True)
    p.add_argument("--rl-token-policy-path", required=True)
    p.add_argument("--vla-pretrained-path", required=True,
                   help="SFT pi05 ckpt dir — preprocessor source. Must match deploy.")
    p.add_argument("--tokenizer-path", default=None,
                   help="PaliGemma tokenizer repo id or local snapshot path for the SFT preprocessor.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--task-instruction", default="screw")
    p.add_argument(
        "--default-episode-success",
        choices=("success", "failure", "error"),
        default="success",
        help="Outcome used when the VLA demonstration dataset has no "
        "episode_success metadata. VLA SFT demonstrations are normally successful. "
        "Only consulted for episodes not covered by --critical-segments-path.",
    )
    p.add_argument(
        "--critical-segments-path", default=None,
        help="Path to a critical_segments.json produced by "
        "diagnostics/critical_segment_labeler_cv.py (see that file's docstring for "
        "the schema). Defaults to <demo-dataset-root>/meta/critical_segments.json "
        "if that file exists. When present, every episode is built from ONLY its "
        "labeled [start, end] critical-phase span (with that span's own "
        "success/failure as episode_success) instead of the whole recorded "
        "episode -- matching what the online RL critical-phase reward actually "
        "sees (RLTOnlineCollector.flush_episode() rewards only the critical-phase "
        "attempt, not full-episode footage). Episodes with no entry, "
        "no_critical=true, or a null segment are skipped entirely, same as an "
        "unlabeled/discarded online attempt. Pass --no-critical-segments to opt "
        "out and fall back to whole-episode --default-episode-success semantics "
        "even if the file exists.",
    )
    p.add_argument(
        "--no-critical-segments", action="store_true",
        help="Ignore meta/critical_segments.json even if present; use whole-episode "
        "episode_success semantics (pre-critical-segment-crop behavior).",
    )
    p.add_argument(
        "--milestone-reward", type=float, default=0.3,
        help="One-time mid-phase shaping bonus applied to episodes whose critical-segment "
        "label has a milestone_frame, matching RLTOnlineCollector.mark_milestone(). Must "
        "equal the online run's --online_rl.milestone_reward, or offline and online "
        "transitions end up on different reward scales in the same replay buffer.",
    )
    p.add_argument(
        "--terminal-reward", type=float, default=1.0,
        help="Reward on a successful critical-phase segment's terminal chunk (failure is "
        "always 0). Must equal --online_rl.terminal_reward.",
    )
    p.add_argument(
        "--time-decay", type=float, default=0.995,
        help="Both rewards above are multiplied by time_decay ** (chunks closed so far in "
        "the segment), matching RLTOnlineCollector's chunk-resolution decay. Must equal "
        "--online_rl.time_decay. 1.0 reproduces the old fixed-reward-on-success behavior.",
    )
    p.add_argument("--chunk-length", type=int, default=10)
    p.add_argument(
        "--rl-action-arms", choices=("left", "both"), default="left",
        help="Build executed chunks for the deployed RL control authority. With "
        "'left', use demonstrated actions for the left half and the current VLA "
        "reference for the right half.",
    )
    p.add_argument("--frame-stride", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Cap on episodes to process (debug).")
    p.add_argument("--video-backend", default="pyav",
                   help="Video decoder backend passed to LeRobotDataset.")
    p.add_argument("--tolerance-s", type=float, default=0.04,
                   help="Timestamp tolerance passed to LeRobotDataset video decoding.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--empty-cache-every", type=int, default=4,
                   help="Call torch.cuda.empty_cache() every N batches.")
    return p.parse_args()


def _log(msg: str) -> None:
    """Unbuffered timestamped log line."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _get_episode_success(
    dataset: LeRobotDataset,
    episode_idx: int,
    default_outcome: str = "error",
) -> bool:
    """Per-episode success flag, matching RLTDemoDataset.get_episode_success semantics."""
    try:
        raw = dataset.meta.episodes["episode_success"][episode_idx]
    except (KeyError, TypeError):
        if default_outcome == "success":
            return True
        if default_outcome == "failure":
            return False
        raise ValueError(
            "Dataset has no episode_success metadata; pass "
            "--default-episode-success=success or failure explicitly"
        )
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized == "success":
            return True
        if normalized == "failure":
            return False
    elif isinstance(raw, bool):
        return raw
    elif isinstance(raw, (int, float)) and raw in (0, 1):
        return bool(raw)
    raise ValueError(f"Unrecognized episode_success value for episode {episode_idx}: {raw!r}")


def _load_critical_segments(path: pathlib.Path | None) -> dict:
    """Load a critical_segments.json (schema: see
    diagnostics/critical_segment_labeler_cv.py's module docstring), keyed by
    str(episode_index) -> {"segment": [start, end, "success"|"failure"] | None,
    "no_critical": bool, "milestone_frame": int | None, ...}. Returns {} if
    `path` is None or doesn't exist, which callers treat as "no episode is
    critical-phase-labeled"."""
    if path is None or not path.is_file():
        return {}
    return json.loads(path.read_text())


def _episode_critical_bounds(
    labels: dict,
    ep_id: int,
    ep_from: int,
) -> tuple[int, int, bool, int | None] | None:
    """Resolve one episode's critical-phase span from `labels` (as loaded by
    `_load_critical_segments`) into absolute dataset-row bounds, matching the
    online path's semantics: RLTOnlineCollector rewards only the critical-phase
    attempt (see RLTOnlineCollector.flush_episode()), never full-episode
    footage. Returns None if this episode has no usable label (missing entry,
    no_critical=true, or a null segment) -- callers must skip such episodes
    entirely, same as an unlabeled/discarded online attempt is never flushed
    into the replay buffer.

    `segment`'s [start, end] are LOCAL frame indices relative to the episode
    start (per the labeler's docstring) with `end` being the last critical
    frame (inclusive) -- converted here to the [ep_from+start, ep_from+end+1)
    absolute half-open range build_overlap_frame_indices()/episode_last_frame
    expect, matching dataset_from_index/dataset_to_index's own convention.

    The optional `milestone_frame` (same LOCAL-frame convention as `segment`)
    is resolved to an absolute frame index too, mirroring
    RLTOnlineCollector.mark_milestone(): a one-time mid-phase shaping bonus
    that can fire regardless of the segment's final success/failure outcome.
    Returns None in the 4th slot when the label has no milestone_frame (the
    demonstration never reached it, or the label predates milestone support).
    """
    entry = labels.get(str(ep_id))
    if entry is None or entry.get("no_critical") or not entry.get("segment"):
        return None
    local_start, local_end, outcome = entry["segment"]
    if outcome not in ("success", "failure"):
        raise ValueError(f"Episode {ep_id}: unrecognized segment outcome {outcome!r}")
    local_milestone = entry.get("milestone_frame")
    milestone_frame = ep_from + int(local_milestone) if local_milestone is not None else None
    return ep_from + int(local_start), ep_from + int(local_end) + 1, outcome == "success", milestone_frame


def _resolve_milestone_chunk(
    milestone_frame: int | None,
    critical_start_frame: int,
    episode_last_frame: int,
    chunk_length: int,
    frame_indices: list[int],
    ep_id: int,
) -> tuple[int | None, int | None]:
    """Map an absolute milestone_frame onto (milestone_actual_start_frame,
    milestone_chunks_closed): which encoded transition's reward_seq the
    one-time shaping bonus lands on, and the chunks-closed exponent to decay
    it by. Mirrors RLTOnlineCollector.mark_milestone()/_chunks_closed: the
    bonus decays as if the milestone landed at its chunk's start (chunks
    closed strictly BEFORE it), but is deposited at that chunk's own closing
    transition -- "applied by whichever _emit_transition() closes next".

    Returns (None, None) if there's no milestone, or no usable anchor exists
    to hang the bonus on (segment too short to produce any transition).
    """
    if milestone_frame is None:
        return None, None
    if not (critical_start_frame <= milestone_frame <= episode_last_frame):
        raise ValueError(
            f"Episode {ep_id}: milestone_frame {milestone_frame} outside its own "
            f"critical segment [{critical_start_frame}, {episode_last_frame}]"
        )
    C = chunk_length
    milestone_chunks_closed = (milestone_frame - critical_start_frame) // C
    milestone_target_start_frame = critical_start_frame + milestone_chunks_closed * C
    usable_starts = [f for f in frame_indices if f + C <= episode_last_frame]
    if not usable_starts:
        _log(f"  ep{ep_id}: no usable anchor to place the milestone bonus on -- skipping it")
        return None, None
    if milestone_target_start_frame in usable_starts:
        return milestone_target_start_frame, milestone_chunks_closed
    nearest = min(usable_starts, key=lambda f: abs(f - milestone_target_start_frame))
    _log(
        f"  ep{ep_id}: milestone target start_frame {milestone_target_start_frame} has no "
        "exact anchor (chunk_length not a multiple of frame_stride?) -- using nearest "
        f"anchor {nearest} instead"
    )
    return nearest, milestone_chunks_closed


def _compute_chunk_reward(
    start_frame: int,
    is_last: bool,
    episode_success: bool,
    terminal_chunks_closed: int,
    milestone_actual_start_frame: int | None,
    milestone_chunks_closed: int | None,
    chunk_length: int,
    milestone_reward: float,
    terminal_reward: float,
    time_decay: float,
) -> torch.Tensor:
    """Build one chunk's (C,) reward_seq, matching RLTOnlineCollector's
    combined milestone + terminal reward: both are undiscounted magnitudes
    scaled by time_decay ** (chunks closed), and both are added onto the same
    last-timestep slot when they land on the same chunk (e.g. the milestone
    is reached in the segment's final chunk)."""
    reward_seq = torch.zeros(chunk_length)
    if is_last and episode_success:
        reward_seq[chunk_length - 1] += terminal_reward * (time_decay ** terminal_chunks_closed)
    if milestone_actual_start_frame is not None and start_frame == milestone_actual_start_frame:
        reward_seq[chunk_length - 1] += milestone_reward * (time_decay ** milestone_chunks_closed)
    return reward_seq


def _compose_exec_chunk(
    demonstrated_chunk: Tensor,
    ref_chunk: Tensor,
    rl_action_arms: str,
) -> Tensor:
    """Keep the action that was truly demonstrated and received the label."""
    if demonstrated_chunk.shape != ref_chunk.shape:
        raise ValueError(
            f"Demonstrated action shape {tuple(demonstrated_chunk.shape)} does not match "
            f"VLA reference shape {tuple(ref_chunk.shape)}"
        )
    exec_chunk = demonstrated_chunk.clone()
    if rl_action_arms == "left":
        action_dim = exec_chunk.shape[-1]
        if action_dim % 2 != 0:
            raise ValueError("--rl-action-arms=left requires an even action_dim")
        exec_chunk[..., action_dim // 2 :] = ref_chunk[..., action_dim // 2 :]
    elif rl_action_arms != "both":
        raise ValueError(f"Unknown rl_action_arms={rl_action_arms!r}")
    return exec_chunk


def _demonstration_supervision_mask(
    exec_chunk: Tensor,
    rl_action_arms: str,
    episode_success: bool,
) -> Tensor:
    """Mark successful demonstrated action dimensions for direct Actor BC.

    ``intervention_mask`` is also used as the per-element Actor-supervision
    mask by ``actor_loss``.  Offline demonstrations are not online human
    interventions (their scalar ``intervention`` remains zero), but a
    successful demonstration is still a trusted action target.  Only mark
    dimensions the residual Actor is allowed to control.
    """
    mask = torch.zeros_like(exec_chunk)
    if not episode_success:
        return mask
    if rl_action_arms == "both":
        return torch.ones_like(exec_chunk)
    if rl_action_arms == "left":
        action_dim = exec_chunk.shape[-1]
        if action_dim % 2 != 0:
            raise ValueError("--rl-action-arms=left requires an even action_dim")
        mask[..., : action_dim // 2] = 1
        return mask
    raise ValueError(f"Unknown rl_action_arms={rl_action_arms!r}")


def _encode_episode(
    pi05,
    rl_token,
    preprocessor,
    capture: PrefixOutputCapture,
    dataset: LeRobotDataset,
    frame_indices: list[int],
    chunk_length: int,
    action_dim: int,
    proprio_dim: int,
    batch_size: int,
    num_workers: int,
    device: str,
    empty_cache_every: int,
    task_str: str,
    ep_id: int,
    episode_last_frame: int,
    episode_success: bool,
    rl_action_arms: str,
    critical_start_frame: int,
    milestone_frame: int | None,
    milestone_reward: float,
    terminal_reward: float,
    time_decay: float,
) -> list[dict[str, Tensor]]:
    """Encode anchors and build x_t -> x_{t+C} chunk transitions."""
    out: list[dict[str, Tensor]] = []
    if not frame_indices:
        return out

    loader = DataLoader(
        Subset(dataset, frame_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device == "cuda",
        persistent_workers=False,
    )

    state_vecs: list[Tensor] = []
    ref_chunks: list[Tensor] = []
    exec_chunks: list[Tensor] = []
    t_ep = time.time()
    for batch_i, batch in enumerate(loader):
        t_b = time.time()
        if "task" not in batch:
            batch["task"] = [task_str] * batch["observation.state"].shape[0]
        pre = preprocessor(batch)
        if "action" not in pre:
            raise KeyError(
                "The SFT preprocessor did not return normalized 'action'; offline "
                "exec_chunk must come from the demonstrated action, not the VLA reference."
            )
        with torch.no_grad():
            vla_chunk = pi05.predict_action_chunk(pre)
            prefix = capture.consume()
            z = rl_token.encode(prefix.to(torch.float32))
        if z.dim() == 3:
            z = z.mean(dim=1)
        proprio = pre["observation.state"][:, :proprio_dim].detach().to("cpu")
        state_vec = torch.cat([z.detach().to("cpu"), proprio], dim=-1)
        ref_chunk = vla_chunk[:, :chunk_length, :action_dim].detach().to("cpu")
        demonstrated_chunk = pre["action"][:, :chunk_length, :action_dim].detach().to("cpu")
        # Preserve the behavior action that actually produced the recorded
        # outcome. Fabricating a clipped action here while retaining the
        # original success label would teach the critic a false transition.
        exec_chunk = _compose_exec_chunk(demonstrated_chunk, ref_chunk, rl_action_arms)
        state_vecs.append(state_vec)
        ref_chunks.append(ref_chunk)
        exec_chunks.append(exec_chunk)
        del vla_chunk, prefix, z, pre
        if (batch_i + 1) % empty_cache_every == 0:
            torch.cuda.empty_cache()
        if batch_i == 0 or (batch_i + 1) % 4 == 0:
            elapsed = time.time() - t_b
            cum = time.time() - t_ep
            _log(f"    ep{ep_id} batch {batch_i+1}/{len(loader)} bs={batch_size} dt={elapsed:.2f}s cum={cum:.1f}s")

    state_vecs_t = torch.cat(state_vecs, dim=0)
    ref_chunks_t = torch.cat(ref_chunks, dim=0)
    exec_chunks_t = torch.cat(exec_chunks, dim=0)

    C = chunk_length
    frame_to_encoded_idx = {
        frame_idx: encoded_idx for encoded_idx, frame_idx in enumerate(frame_indices)
    }
    # Matches RLTOnlineCollector._chunks_closed at flush_episode() time: every
    # C frames of the critical-phase attempt closes one chunk, and the final
    # (possibly partial) chunk still counts as closed when the attempt ends.
    # ceil(), not floor()/(//): confirmed against a direct simulation of
    # RLTOnlineCollector's actual increment logic (a real attempt with N real
    # frames always ends at chunks_closed == ceil(N/C), because a leftover
    # partial chunk at flush_episode() still increments the counter by 1).
    # The offline anchor scheme can only ever *emit* floor(N/C) transitions
    # for such an attempt (the dataset's true last frame can only ever serve
    # as someone's bootstrap next_state, never start its own transition, so
    # there's no transition to represent that trailing partial chunk) -- but
    # that's a limitation of how many transitions get built, not of how much
    # time actually elapsed, so the terminal transition's decay exponent
    # still has to reflect the real ceil(N/C), not the smaller count of
    # transitions the pipeline managed to physically produce.
    critical_length = episode_last_frame + 1 - critical_start_frame
    terminal_chunks_closed = math.ceil(critical_length / C) if critical_length > 0 else 0
    milestone_actual_start_frame, milestone_chunks_closed = _resolve_milestone_chunk(
        milestone_frame, critical_start_frame, episode_last_frame, C, frame_indices, ep_id,
    )
    for i, start_frame in enumerate(frame_indices):
        next_frame = start_frame + C
        if next_frame > episode_last_frame:
            continue
        next_i = frame_to_encoded_idx.get(next_frame)
        if next_i is None:
            raise ValueError(
                f"Missing x_(t+C) anchor for episode {ep_id}: "
                f"{start_frame} -> {next_frame}"
            )
        is_last = next_frame == episode_last_frame
        reward_seq = _compute_chunk_reward(
            start_frame=start_frame,
            is_last=is_last,
            episode_success=episode_success,
            terminal_chunks_closed=terminal_chunks_closed,
            milestone_actual_start_frame=milestone_actual_start_frame,
            milestone_chunks_closed=milestone_chunks_closed,
            chunk_length=C,
            milestone_reward=milestone_reward,
            terminal_reward=terminal_reward,
            time_decay=time_decay,
        )
        out.append(
            {
                "state_vec": state_vecs_t[i],
                "exec_chunk": exec_chunks_t[i],
                "ref_chunk": ref_chunks_t[i],
                "reward_seq": reward_seq,
                "next_state_vec": state_vecs_t[next_i],
                "next_ref_chunk": ref_chunks_t[next_i],
                "done": torch.tensor(float(is_last)),
                "intervention": torch.tensor(0.0),
                "actual_steps": torch.tensor(C, dtype=torch.int64),
                "source": torch.tensor(0, dtype=torch.int64),
                "episode_id": torch.tensor(ep_id, dtype=torch.int64),
                "is_critical": torch.tensor(1.0),
                "outcome": torch.tensor(float(episode_success)),
                # Offline demos are not online interventions, so the scalar
                # intervention flag above stays zero.  This per-element mask
                # separately tells actor_loss which successful demonstrated
                # action dimensions are trusted direct-supervision targets.
                "intervention_mask": _demonstration_supervision_mask(
                    exec_chunks_t[i], rl_action_arms, episode_success
                ),
            }
        )
    return out


def _save_partial(out_dir: pathlib.Path, split: str, transitions: list, label: str) -> None:
    path = out_dir / f"chunk_transitions_{split}.pt"
    tmp = out_dir / f".chunk_transitions_{split}.tmp.pt"
    torch.save(transitions, tmp)
    tmp.replace(path)
    _log(f"  [{label}] checkpointed {len(transitions)} transitions -> {path.name}")


def _write_cache_metadata(out_dir: pathlib.Path, metadata: dict) -> None:
    """Atomically publish cache metadata, including its completion state."""
    path = out_dir / "cache_metadata.json"
    tmp = out_dir / ".cache_metadata.tmp.json"
    tmp.write_text(json.dumps(metadata, indent=2) + "\n")
    tmp.replace(path)


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    critical_segments_path = None
    if not args.no_critical_segments:
        critical_segments_path = (
            pathlib.Path(args.critical_segments_path) if args.critical_segments_path
            else pathlib.Path(args.demo_dataset_root) / "meta" / "critical_segments.json"
        )
    critical_segments = _load_critical_segments(critical_segments_path)
    if critical_segments:
        _log(
            f"using critical-phase labels from {critical_segments_path} "
            f"({len(critical_segments)} labeled episodes) -- each episode is cropped to "
            "ONLY its labeled critical span, matching the online RL reward's critical-phase "
            "scope. Episodes with no label / no_critical=true / null segment are skipped."
        )
    else:
        _log(
            f"no critical-phase labels found at {critical_segments_path} "
            "(or --no-critical-segments was passed) -- falling back to whole-episode "
            "episode_success semantics. This does NOT match what the online RL critical "
            "phase actually rewards; only use this for datasets that were never split into "
            "prefix/critical-phase segments."
        )

    metadata = {
        "format_version": 4,
        # Written false before any expensive work and changed to true only
        # after both split files finish. The online loader rejects a cache
        # left partial by an interrupted build.
        "build_complete": False,
        "exec_chunk_source": "demonstrated_action",
        # v1 means successful offline demonstrations carry a nonzero
        # per-element intervention_mask and therefore directly train the
        # Actor, rather than only supplying behavior actions to the Critic.
        # The loader rejects caches without this marker.
        "actor_supervision_schema_version": 1,
        "actor_supervision_source": "successful_demonstration",
        "rl_action_arms": args.rl_action_arms,
        "chunk_length": args.chunk_length,
        "task_instruction": args.task_instruction,
        "vla_pretrained_path": args.vla_pretrained_path,
        "rl_token_policy_path": args.rl_token_policy_path,
        "critical_segments_path": (
            str(critical_segments_path) if critical_segments else None
        ),
        "cropped_to_critical_segment": bool(critical_segments),
        "milestone_reward": args.milestone_reward,
        "terminal_reward": args.terminal_reward,
        "time_decay": args.time_decay,
        # Bump if the reward formula/placement semantics ever change again --
        # online_trainer.py's _load_offline_buffer() uses this to refuse
        # loading a cache whose reward scale can't be verified against the
        # current online_rl milestone/decay config.
        "reward_schema_version": 2,
    }
    _write_cache_metadata(out_dir, metadata)

    _log(f"args: {vars(args)}")
    _log(f"load RLTokenPolicy from {args.rl_token_policy_path}")
    RLTokenPolicyConfig.ensure_registered()
    policy = RLTokenPolicy.from_pretrained(args.rl_token_policy_path).to(args.device).eval()
    cfg = policy.config
    # Override the policy's recorded vla path so the preprocessor we load is the
    # SFT pi05's, even if the RL Token ckpt was trained against a different one.
    cfg.vla_pretrained_path = args.vla_pretrained_path
    if args.tokenizer_path is not None:
        cfg.tokenizer_path = args.tokenizer_path

    _log(f"load preprocessor from SFT pi05 dir {args.vla_pretrained_path}")
    preprocessor, _ = make_rlt_token_pre_post_processors(config=cfg)

    _log(f"load dataset {args.demo_dataset_repo_id} root={args.demo_dataset_root}")
    delta = {"action": [i / 30.0 for i in range(cfg.chunk_size)]}
    dataset = LeRobotDataset(
        repo_id=args.demo_dataset_repo_id,
        root=args.demo_dataset_root,
        delta_timestamps=delta,
        tolerance_s=args.tolerance_s,
        video_backend=args.video_backend,
    )
    n_episodes = dataset.num_episodes
    if args.max_episodes is not None:
        n_episodes = min(n_episodes, args.max_episodes)
    _log(f"episodes: {n_episodes} of {dataset.num_episodes}; batch_size={args.batch_size} num_workers={args.num_workers}")

    pi05 = policy._pi05
    rl_token = policy.rl_token

    capture = PrefixOutputCapture(
        token_pool_size=cfg.token_pool_size,
        image_only=cfg.image_only,
        num_image_tokens=policy._num_image_tokens,
    )
    capture.attach(policy._pi05)
    split_summaries: dict[str, dict[str, int]] = {}
    try:
        ep_indices = list(range(n_episodes))
        random.shuffle(ep_indices)
        n_train = int(args.train_ratio * n_episodes)
        train_eps = ep_indices[:n_train]
        val_eps = ep_indices[n_train:]
        _log(f"split: train={len(train_eps)} val={len(val_eps)}")

        t_start = time.time()
        for split_name, eps in (("train", train_eps), ("val", val_eps)):
            all_tx: list[dict[str, Tensor]] = []
            n_skipped = 0
            for k, ep_id in enumerate(eps):
                ep_meta = dataset.meta.episodes
                ep_from = int(ep_meta["dataset_from_index"][ep_id])
                ep_to = int(ep_meta["dataset_to_index"][ep_id])

                if critical_segments:
                    bounds = _episode_critical_bounds(critical_segments, ep_id, ep_from)
                    if bounds is None:
                        n_skipped += 1
                        _log(f"  [{split_name}] ep {k+1}/{len(eps)} id={ep_id} skipped (no critical-phase label)")
                        continue
                    crit_from, crit_to, episode_success, milestone_frame = bounds
                else:
                    crit_from, crit_to = ep_from, ep_to
                    episode_success = _get_episode_success(
                        dataset,
                        ep_id,
                        default_outcome=args.default_episode_success,
                    )
                    # No critical-segment label in this fallback mode, so
                    # there's no milestone_frame to resolve either.
                    milestone_frame = None

                frame_indices = build_overlap_frame_indices(
                    episode_start=crit_from,
                    episode_stop=crit_to,
                    chunk_length=args.chunk_length,
                    stride=args.frame_stride,
                )
                _log(f"  [{split_name}] ep {k+1}/{len(eps)} id={ep_id} frames={crit_to-crit_from} chunks={len(frame_indices)} success={episode_success} (total transitions={len(all_tx)}, wall={time.time()-t_start:.0f}s)")
                ep_tx = _encode_episode(
                    pi05=pi05,
                    rl_token=rl_token,
                    preprocessor=preprocessor,
                    capture=capture,
                    dataset=dataset,
                    frame_indices=frame_indices,
                    chunk_length=args.chunk_length,
                    action_dim=cfg.action_dim,
                    proprio_dim=cfg.proprio_dim,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    device=args.device,
                    empty_cache_every=args.empty_cache_every,
                    task_str=args.task_instruction,
                    ep_id=ep_id,
                    episode_last_frame=crit_to - 1,
                    episode_success=episode_success,
                    rl_action_arms=args.rl_action_arms,
                    critical_start_frame=crit_from,
                    milestone_frame=milestone_frame,
                    milestone_reward=args.milestone_reward,
                    terminal_reward=args.terminal_reward,
                    time_decay=args.time_decay,
                )
                all_tx.extend(ep_tx)
                if (k + 1) % 5 == 0:
                    _save_partial(out_dir, split_name, all_tx, f"{split_name} ep {k+1}/{len(eps)}")
            # Always publish the exact final split. In particular, the last
            # episode(s) may have been skipped before reaching the periodic
            # checkpoint above; without this write, build_complete=true could
            # otherwise describe a stale or truncated split file.
            _save_partial(out_dir, split_name, all_tx, f"{split_name} final")
            if critical_segments:
                _log(f"  [{split_name}] done: {len(eps) - n_skipped} episodes used, {n_skipped} skipped (unlabeled/no_critical), {len(all_tx)} transitions")
            split_summaries[split_name] = {
                "episodes_requested": len(eps),
                "episodes_used": len(eps) - n_skipped,
                "episodes_skipped": n_skipped,
                "transitions": len(all_tx),
            }
    finally:
        capture.detach()

    metadata["build_complete"] = True
    metadata["splits"] = split_summaries
    _write_cache_metadata(out_dir, metadata)
    _log(f"done, total wall {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
