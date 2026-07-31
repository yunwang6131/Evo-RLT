"""Build chunk-transition cache for ChunkACPolicy training.

Encodes each base frame in a LeRobotDataset through pi0.5 + the trained RL
Token encoder into a (state_vec, exec_chunk, ref_chunk, ...) tuple stored on
disk. The cache is consumed by ChunkTransitionDataset at AC training time.

This v2 replaces the legacy custom-load builder. It loads the preprocessor
directly from the SFT pi05 ckpt so the cache is byte-aligned with the deploy
normalization. Per-batch progress with elapsed time is written to stdout in
unbuffered mode so a hung run is visible immediately. Each completed episode
is checkpointed to a tmp file so a kill mid-run only forfeits the in-flight
episode.
"""
from __future__ import annotations

import argparse
import json
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
from evo_rlt.core.rewards import build_reward_seq


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
        "episode_success metadata. VLA SFT demonstrations are normally successful.",
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


def _compose_exec_chunk(
    demonstrated_chunk: Tensor,
    ref_chunk: Tensor,
    rl_action_arms: str,
) -> Tensor:
    """Project demonstrations onto the action support used during deployment."""
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
        # Online deployment only gives the actor authority over the selected
        # arm(s). Project demos onto that same action support.
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
        reward_seq = build_reward_seq(
            chunk_length=C,
            is_terminal_chunk=is_last,
            episode_success=episode_success,
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
            }
        )
    return out


def _save_partial(out_dir: pathlib.Path, split: str, transitions: list, label: str) -> None:
    path = out_dir / f"chunk_transitions_{split}.pt"
    tmp = out_dir / f".chunk_transitions_{split}.tmp.pt"
    torch.save(transitions, tmp)
    tmp.replace(path)
    _log(f"  [{label}] checkpointed {len(transitions)} transitions -> {path.name}")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cache_metadata.json").write_text(
        json.dumps(
            {
                "format_version": 2,
                "exec_chunk_source": "demonstrated_action",
                "rl_action_arms": args.rl_action_arms,
                "chunk_length": args.chunk_length,
                "task_instruction": args.task_instruction,
                "vla_pretrained_path": args.vla_pretrained_path,
                "rl_token_policy_path": args.rl_token_policy_path,
            },
            indent=2,
        )
        + "\n"
    )

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
            for k, ep_id in enumerate(eps):
                ep_meta = dataset.meta.episodes
                ep_from = int(ep_meta["dataset_from_index"][ep_id])
                ep_to = int(ep_meta["dataset_to_index"][ep_id])
                frame_indices = build_overlap_frame_indices(
                    episode_start=ep_from,
                    episode_stop=ep_to,
                    chunk_length=args.chunk_length,
                    stride=args.frame_stride,
                )
                episode_success = _get_episode_success(
                    dataset,
                    ep_id,
                    default_outcome=args.default_episode_success,
                )
                _log(f"  [{split_name}] ep {k+1}/{len(eps)} id={ep_id} frames={ep_to-ep_from} chunks={len(frame_indices)} success={episode_success} (total transitions={len(all_tx)}, wall={time.time()-t_start:.0f}s)")
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
                    episode_last_frame=ep_to - 1,
                    episode_success=episode_success,
                    rl_action_arms=args.rl_action_arms,
                )
                all_tx.extend(ep_tx)
                if (k + 1) % 5 == 0 or (k + 1) == len(eps):
                    _save_partial(out_dir, split_name, all_tx, f"{split_name} ep {k+1}/{len(eps)}")
    finally:
        capture.detach()

    _log(f"done, total wall {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
