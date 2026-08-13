"""CLI entrypoint for synchronous online RL training on real hardware.

Builds a fresh `rlt_ac` policy (frozen VLA + RL token, randomly-initialized
actor/critic with zero-init residual-to-VLA-reference), then reuses the
`evo-rlt-record` pipeline (backend.record()) with `online_rl.enable=true` so
each recorded critical-phase episode is followed by gradient updates scaled
to how much data it added (see backend.OnlineRLConfig), matching RLT's
Algorithm 1 but synchronous (rollout and training never run concurrently).

NOT a hardware E-stop. Physical supervision (hand near the leader arm, a
power cutoff within reach) is still required -- see README safety notes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from evo_rlt.adapters.lerobot.record.common import (
    build_dataset_argv,
    build_robot_argv,
    build_teleop_argv,
    configure_logging,
    load_robot_setup,
    preflight_motor_connections,
    remove_existing_dataset,
    resolve_run_paths,
    set_offline_env,
    stage_follower_calibrations,
    stage_leader_calibrations,
)
from evo_rlt.adapters.lerobot.record.runner import prepare_lerobot_runtime

DEFAULT_DATASET_TAG = "online_rl"
# lerobot's sanity_check_dataset_name() requires the dataset repo_id to start
# with "eval_" whenever a `policy` config is passed to record() (its
# convention: policy-driven recording == evaluating that policy) -- online RL
# training always passes `--policy.type=rlt_ac`, so the actual dataset leaf
# name (see resolve_run_paths()'s `dataset_prefix` param) must satisfy this,
# even though this data is used for training (via the replay buffer), not
# pure evaluation. DEFAULT_DATASET_TAG above still names the per-run output
# folder (via --dataset-tag) and is unaffected.
DEFAULT_DATASET_NAME_PREFIX = "eval_online_rl"
DEFAULT_TASK = "Insert the copper screw into the black sleeve."


def build_online_train_argv(args: argparse.Namespace, setup, paths, cal_dir: str, teleop_argv: list[str]) -> list[str]:
    argv = [
        "online_train",
        *build_robot_argv(setup.followers, setup.left_cameras, setup.right_cameras, cal_dir),
        *teleop_argv,
        # Fresh policy construction: no --policy.path, so make_policy() builds
        # a brand-new ChunkACPolicy (random actor/critic init) instead of
        # loading a checkpoint. VLA + RL token are frozen pretrained backbones.
        "--policy.type=rlt_ac",
        f"--policy.vla_pretrained_path={args.vla_path}",
        f"--policy.rl_token_pretrained_path={args.rl_token_path}",
        f"--policy.tokenizer_path={args.tokenizer_path}",
        "--policy.phase_mode=manual",
        f"--policy.chunk_exec_steps={args.chunk_exec_steps}",
        f"--policy.chunk_length={args.chunk_length}",
        f"--policy.action_dim={args.action_dim}",
        f"--policy.proprio_dim={args.proprio_dim}",
        "--policy.actor_residual_to_ref=true",
        f"--policy.actor_rl_arm={args.rl_action_arms}",
        # With residual_to_ref, mu = ref + delta unconditionally -- the output
        # always gets the true (undropped) ref added back as a bias, even on
        # a dropout-masked training sample. So input-side reference dropout
        # can no longer force "independent action generation" the way it does
        # for the paper's non-residual actor (mu = net(state, ref) directly,
        # no ref shortcut); it would just be extra training noise with none
        # of its intended effect. Disabled here rather than left at the
        # (non-residual-tuned) 0.5 default.
        "--policy.actor_ref_dropout_p=0.0",
        f"--policy.gamma={args.gamma}",
        f"--policy.beta={args.beta}",
        f"--policy.actor_demo_bc_weight={args.demo_bc_weight}",
        f"--policy.tau={args.tau}",
        f"--policy.utd_ratio={args.utd_ratio}",
        f"--policy.actor_update_interval={args.actor_update_interval}",
        f"--policy.actor_hidden_dim={args.actor_hidden_dim}",
        f"--policy.actor_num_layers={args.actor_num_layers}",
        f"--policy.actor_fixed_std={args.actor_fixed_std}",
        f"--policy.actor_activation={args.actor_activation}",
        f"--policy.actor_residual={'true' if args.actor_residual else 'false'}",
        f"--policy.actor_layer_norm={'true' if args.actor_layer_norm else 'false'}",
        f"--policy.critic_hidden_dim={args.critic_hidden_dim}",
        f"--policy.critic_num_layers={args.critic_num_layers}",
        f"--policy.critic_activation={args.critic_activation}",
        f"--policy.critic_residual={'true' if args.critic_residual else 'false'}",
        f"--policy.critic_layer_norm={'true' if args.critic_layer_norm else 'false'}",
        f"--policy.rankq_alpha_success={args.rankq_alpha_success}",
        f"--policy.rankq_alpha_failure={args.rankq_alpha_failure}",
        f"--policy.rankq_noise_scale={args.rankq_noise_scale}",
        f"--policy.rankq_margin={args.rankq_margin}",
        f"--policy.rankq_margin_relative={'true' if args.rankq_margin_relative else 'false'}",
        f"--policy.target_q_clip={args.target_q_clip}",
        f"--policy.target_q_min={args.target_q_min}",
        f"--policy.target_noise_std={args.target_noise_std}",
        f"--policy.target_noise_clip={args.target_noise_clip}",
        f"--policy.actor_smoothness_weight={args.actor_smoothness_weight}",
        "--policy.device=cuda",
        *build_dataset_argv(
            dataset_name=paths.dataset_name,
            dataset_root=paths.dataset_root,
            task=args.task,
            num_episodes=args.num_episodes,
            episode_time_s=args.episode_time_s,
            fps=args.fps,
            vcodec=args.vcodec,
        ),
        # Policyless (pure teleop, no VLA/RL) reset window between episodes --
        # NOT skipped for online training (unlike plain --only-critical data
        # collection): after a failed/aborted critical-phase attempt the robot
        # can be left in an out-of-distribution pose (e.g. pin half-inserted
        # at a bad angle), and letting the frozen VLA immediately resume
        # autonomous control from there, with no human in the loop yet, is
        # not something it was ever trained to recover from. This window
        # gives you `reset_time_s` seconds of pure leader-arm teleop to
        # physically reset the scene before the next episode's recording (and
        # possible autonomous critical-phase attempt) begins.
        f"--dataset.reset_time_s={args.reset_time_s}",
        # rlt_toggle_key starts the critical-phase attempt; the next press ends
        # it as success, while u ends it as failure immediately. Either key hands
        # control back to VLA, and flushes that reward into the online replay
        # buffer right there (see loop.py) -- it does NOT end the recorded
        # episode. The episode keeps recording (e.g. VLA autonomously
        # finishing a subsequent step like placing the object) until you
        # press the whole-episode outcome key (episode_success_key/
        # episode_failure_key, s/f by default) once that's done.
        "--rlt.enable=true",
        f"--rlt.rl_phase_key={args.rlt_toggle_key}",
        f"--rlt.milestone_key={args.milestone_key}",
        f"--rlt.intervention_action_blend_time_s={args.intervention_blend_time_s}",
        "--rlt.skip_prefix_recording=true",
        "--rlt.rl_phase_key_toggles_critical_phase=true",
        "--rlt.start_in_teleop=false",
        # v1 online training does not support the RTC runtime.
        "--rlt.rtc_enabled=false",
        "--enable_episode_outcome_labeling=true",
        "--require_episode_success_label=true",
        "--intervention_state_machine_enabled=true",
        f"--left_intervention_key={args.left_intervention_key}",
        f"--right_intervention_key={args.right_intervention_key}",
        f"--policy_sync_to_teleop={'true' if teleop_argv else 'false'}",
        f"--vla_ref={'true' if args.vla_ref else 'false'}",
        f"--play_sounds={'true' if args.play_sounds else 'false'}",
        # Online RL training loop (see backend.OnlineRLConfig).
        "--online_rl.enable=true",
        f"--online_rl.warmup_episodes={args.warmup_episodes}",
        f"--online_rl.critic_only_episodes={args.critic_only_episodes}",
        f"--online_rl.actor_unfreeze_ramp_episodes={args.actor_unfreeze_ramp_episodes}",
        f"--online_rl.min_warmup_transitions={args.min_warmup_transitions}",
        f"--online_rl.min_warmup_successes={args.min_warmup_successes}",
        f"--online_rl.min_warmup_failures={args.min_warmup_failures}",
        f"--online_rl.max_updates_per_episode={args.max_updates_per_episode}",
        f"--online_rl.use_stratified_sampling={'true' if args.stratified_sampling else 'false'}",
        f"--online_rl.replay_capacity={args.replay_capacity}",
        f"--online_rl.batch_size={args.batch_size}",
        f"--online_rl.offline_batch_fraction={args.offline_batch_fraction}",
        f"--online_rl.lr_actor={args.lr_actor}",
        f"--online_rl.lr_critic={args.lr_critic}",
        f"--online_rl.terminal_reward={args.terminal_reward}",
        f"--online_rl.milestone_reward={args.milestone_reward}",
        f"--online_rl.time_decay={args.time_decay}",
        f"--online_rl.save_dir={args.save_dir}",
        f"--online_rl.save_every_episodes={args.save_every_episodes}",
        f"--online_rl.go_home_time_s={args.go_home_time_s}",
        f"--online_rl.go_home_gripper_value={args.go_home_gripper_value}",
        f"--online_rl.wandb={'true' if args.wandb else 'false'}",
        f"--online_rl.wandb_project={args.wandb_project}",
    ]
    if args.actor_action_clip_delta is not None:
        argv.append(f"--policy.actor_action_clip_delta={args.actor_action_clip_delta}")
    if args.actor_slew_rate_limit is not None:
        argv.append(f"--policy.actor_slew_rate_limit={args.actor_slew_rate_limit}")
    if args.offline_cache_path is not None:
        argv.append(f"--online_rl.offline_cache_path={args.offline_cache_path}")
    if args.go_home_positions is not None:
        argv.append(f"--online_rl.go_home_positions={args.go_home_positions}")
    if args.wandb_entity is not None:
        argv.append(f"--online_rl.wandb_entity={args.wandb_entity}")
    if args.wandb_run_name is not None:
        argv.append(f"--online_rl.wandb_run_name={args.wandb_run_name}")
    if args.wandb_run_id is not None:
        argv.append(f"--online_rl.wandb_run_id={args.wandb_run_id}")
    if args.wandb_resume is not None:
        argv.append(f"--online_rl.wandb_resume={args.wandb_resume}")
    if args.resume_from is not None:
        argv.append(f"--online_rl.resume_from={args.resume_from}")
    return argv


def print_online_train_summary(args: argparse.Namespace, paths) -> None:
    print("\nOnline RL training (synchronous, real hardware)")
    print(f"Dataset (raw episodes, for record-keeping): {paths.dataset_name} -> {paths.dataset_root}")
    print(f"Checkpoints: {args.save_dir}")
    print(f"VLA: {args.vla_path}")
    print(f"RL token: {args.rl_token_path}")
    print(
        f"Replay: online + "
        f"{args.offline_cache_path or 'no offline cache'} "
        f"(offline_batch_fraction={args.offline_batch_fraction if args.offline_cache_path else 0.0})"
    )
    print(
        f"RL: warmup_episodes={args.warmup_episodes} (+min_transitions={args.min_warmup_transitions} "
        f"min_successes={args.min_warmup_successes} min_failures={args.min_warmup_failures}) "
        f"critic_only_episodes={args.critic_only_episodes} "
        f"actor_unfreeze_ramp_episodes={args.actor_unfreeze_ramp_episodes} batch_size={args.batch_size} "
        f"lr_actor={args.lr_actor} lr_critic={args.lr_critic} utd_ratio={args.utd_ratio} "
        f"max_updates_per_episode={args.max_updates_per_episode} "
        f"stratified_sampling={args.stratified_sampling} save_every={args.save_every_episodes}"
    )
    print(
        f"Actor: hidden_dim={args.actor_hidden_dim} num_layers={args.actor_num_layers} "
        f"layer_norm={args.actor_layer_norm} fixed_std={args.actor_fixed_std} | "
        f"Critic: hidden_dim={args.critic_hidden_dim} "
        f"num_layers={args.critic_num_layers} layer_norm={args.critic_layer_norm}"
    )
    print(
        f"RankQ: alpha_success={args.rankq_alpha_success} alpha_failure={args.rankq_alpha_failure} "
        f"noise_scale={args.rankq_noise_scale} margin={args.rankq_margin}"
        + (" (relative to mean|Q|)" if args.rankq_margin_relative else " (absolute)")
        + f" | target_q_clip={args.target_q_clip}"
    )
    print(
        f"Target policy smoothing: noise_std={args.target_noise_std} noise_clip={args.target_noise_clip}"
        + (" (disabled)" if args.target_noise_std <= 0 else "")
    )
    print(
        f"Safety: actor_action_clip_delta={args.actor_action_clip_delta} "
        f"slew_rate_limit={args.actor_slew_rate_limit}; "
        "stay near the leader arm / physical power cutoff"
    )
    print(
        f"Actor smoothness_weight={args.actor_smoothness_weight}"
        + (" (disabled)" if args.actor_smoothness_weight <= 0 else "")
    )
    print(
        f"Controls: {args.rlt_toggle_key}=start/end critical-phase attempt as success, "
        "u=end it as failure immediately (reward flushed immediately, "
        "recording continues under VLA afterward), s/f=end the whole recorded episode "
        "once VLA finishes, "
        f"{args.left_intervention_key}=left-arm intervention, "
        f"{args.right_intervention_key}=right-arm intervention (puppeteer the right arm out of the "
        "way; it is never RL-controlled under --rl-action-arms left, so this does not affect what's "
        "being learned -- but the frame is still tagged as an intervention, same as left), "
        f"{args.teleop_toggle_key}=both-arm intervention/release"
    )
    print(
        f"Go-home: {args.go_home_time_s}s ramp to the calibrated middle position "
        f"(gripper -> {args.go_home_gripper_value}, verify this means 'open' for your "
        "hardware) after each episode, before the reset window. 0 = disabled."
    )
    print(
        f"Reset window: {args.reset_time_s}s pure teleop after every episode "
        "(success or failure) before the next one starts -- use it to physically "
        "reposition task objects (go-home does not move them)."
    )
    if args.wandb:
        print(
            f"Wandb: project={args.wandb_project} entity={args.wandb_entity} "
            f"run_name={args.wandb_run_name} run_id={args.wandb_run_id} "
            f"resume={args.wandb_resume}"
        )
    if args.resume_from:
        print(f"Resuming online-RL state from: {args.resume_from}")


def run_online_train(args: argparse.Namespace) -> None:
    set_offline_env()
    if args.wandb_resume is not None and args.wandb_run_id is None:
        raise ValueError("--wandb-resume requires --wandb-run-id (the original W&B run ID).")
    keys = {
        args.teleop_toggle_key, args.left_intervention_key, args.right_intervention_key,
        args.rlt_toggle_key, "u",
    }
    if len(keys) != 5:
        raise ValueError(
            "--teleop-toggle-key, --left-intervention-key, --right-intervention-key, --rlt-toggle-key, "
            "and the fixed 'u' failure key must be distinct."
        )

    setup = load_robot_setup(args.setup_json)
    paths = resolve_run_paths(setup.setup, args.dataset_tag, DEFAULT_DATASET_NAME_PREFIX)
    if args.save_dir is None:
        if args.resume_from is not None:
            raise ValueError(
                "--resume-from requires --save-dir pointing at the same directory the "
                "resumed run used (so new checkpoints land alongside its history) -- pass "
                "it explicitly instead of relying on the auto-generated default."
            )
        # Mirrors the dataset run folder's own <MMDD>_<tag>/<prefix>_<HHMMSS> timestamp so
        # a session's checkpoints and its raw dataset are easy to correlate, and so
        # back-to-back fresh runs never collide/overwrite each other's checkpoints.
        args.save_dir = str(Path("outputs/online_rl") / paths.day_dir.name / paths.dataset_root.name)
    configure_logging(paths.log_file, args.log_level)
    remove_existing_dataset(paths.dataset_root)
    teleop_argv = build_teleop_argv(setup.leaders, no_teleop=False)
    if not teleop_argv:
        raise ValueError(
            "Online training requires leader teleop arms (mid-chunk human intervention "
            "is part of the training loop)."
        )

    leader_cal_dir = None
    with TemporaryDirectory(prefix="online-train-") as cal_dir:
        stage_follower_calibrations(setup.followers, cal_dir)
        leader_cal_dir = stage_leader_calibrations(setup.leaders, teleop_argv)
        if args.preflight:
            preflight_motor_connections(
                setup.followers, setup.leaders, cal_dir,
                leader_cal_dir.name if leader_cal_dir is not None else None,
            )
        sys.argv = build_online_train_argv(args, setup, paths, cal_dir, teleop_argv)
        print_online_train_summary(args, paths)
        if args.dry_run:
            print("\nDry run argv:")
            print(" ".join(sys.argv))
            return

        prepare_lerobot_runtime(
            intervention_toggle_key=args.teleop_toggle_key,
            # Deliberately NOT True here -- see the comment on
            # --dataset.reset_time_s in build_online_train_argv(). Plain
            # --only-critical data collection skips this because it's
            # replaying a fixed, already-trained checkpoint; online training
            # needs the human to get a guaranteed reset window between every
            # episode regardless of how the previous one ended.
            skip_policyless_reset_loop=False,
            background_episode_video_encoding=True,
        )
        from evo_rlt.adapters.lerobot.record.backend import record

        record()

    if leader_cal_dir is not None:
        leader_cal_dir.cleanup()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synchronous online RL training on real hardware")
    parser.add_argument("--vla-path", required=True, help="Frozen pi0.5 VLA checkpoint (from demo adaptation).")
    parser.add_argument("--rl-token-path", required=True, help="Frozen RL token checkpoint (from demo adaptation).")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--episode-time-s", type=int, default=3000)
    parser.add_argument(
        "--reset-time-s", type=int, default=15,
        help="Pure-teleop window between episodes to physically reset the scene "
        "(no VLA/RL action sent during this time). Runs regardless of whether the "
        "previous episode succeeded or failed.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--setup-json", default=None)
    parser.add_argument("--dataset-tag", default=DEFAULT_DATASET_TAG)
    parser.add_argument("--vcodec", default="h264")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--intervention-blend-time-s", type=float, default=0.3,
        help="Smooth both transitions across a human intervention -- takeover (space "
        "pressed, blends from the last policy action to teleop) and release (space "
        "released, blends from the last teleop position back to the freshly "
        "recomputed policy action) -- over this many seconds, instead of jumping "
        "instantly. 0 disables both blends.",
    )
    parser.add_argument("--vla-ref", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--play-sounds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rlt-toggle-key", default="r")
    parser.add_argument(
        "--milestone-key", default="m",
        help="Pressed once during an active critical-phase attempt to award the "
        "one-time mid-phase shaping bonus (--milestone-reward), e.g. for a "
        "sub-step like 'pin pulled out' ahead of the final r/u judgment. "
        "No-op outside an active critical phase or after the first press.",
    )
    parser.add_argument("--teleop-toggle-key", default="space")
    parser.add_argument("--left-intervention-key", default="i")
    parser.add_argument(
        "--right-intervention-key", default="o",
        help="Puppeteer only the right arm (Space releases it, same as left). The right arm is "
        "never RL-controlled under --rl-action-arms left, so this has no effect on what's being "
        "learned -- it just keeps a misbehaving right arm from ruining an otherwise-good "
        "left-arm attempt. Note: the frame is still tagged as an intervention while active (source "
        "is per-frame, not per-arm), same limitation as --left-intervention-key.",
    )
    parser.add_argument(
        "--preflight", action=argparse.BooleanOptionalAction, default=True,
        help="Connect/torque-check follower+leader arms before loading the policy.",
    )
    parser.add_argument("--dry-run", action="store_true", default=False)

    # Policy shape (must match the loaded VLA + RL token checkpoints).
    parser.add_argument("--chunk-length", type=int, default=10)
    parser.add_argument("--chunk-exec-steps", type=int, default=25)
    parser.add_argument("--action-dim", type=int, default=12)
    parser.add_argument("--proprio-dim", type=int, default=12)

    # TD3+BC hyperparameters.
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument(
        "--demo-bc-weight", type=float, default=1.0,
        help="Dedicated weight for direct Actor imitation of successful offline "
        "demonstrations and successful human corrections. Separate from --beta, "
        "which anchors non-demonstrated actions to the VLA reference.",
    )
    parser.add_argument("--tau", type=float, default=0.005)
    # Gradient updates per NEW transition this episode added, capped by
    # --max-updates-per-episode (not a fixed count per episode). Default 1
    # (vs. the paper/offline default 5) so a synchronous session doesn't
    # stall for a long training burst after every episode on modest
    # hardware; raise once the loop is confirmed stable.
    parser.add_argument("--utd-ratio", type=int, default=5)
    parser.add_argument("--max-updates-per-episode", type=int, default=1000)
    parser.add_argument("--actor-update-interval", type=int, default=2)

    # Network size. Defaults match src/evo_rlt/core/configs/ac_paper_screw.yaml
    # (the paper's complex-task tier: 3 layers, hidden_dim 512), appropriate
    # for multi-step, dexterous, contact-rich tasks like bimanual insertion.
    parser.add_argument("--actor-hidden-dim", type=int, default=512)
    parser.add_argument("--actor-num-layers", type=int, default=3)
    parser.add_argument("--actor-activation", default="relu")
    parser.add_argument("--actor-residual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--actor-layer-norm", action=argparse.BooleanOptionalAction, default=False,
        help="LayerNorm in Actor hidden blocks. Disabled by default; RLPD's key "
        "normalization recommendation applies to the Critic.",
    )
    parser.add_argument(
        "--rl-action-arms", choices=("left", "both"), default="left",
        help="Arm(s) whose action residual may be learned. 'left' keeps every "
        "right-arm action exactly equal to the frozen VLA reference.",
    )
    # NOTE: currently inert. RLTActionModifier.compute_chunk() calls
    # actor.forward() (deterministic mean) for real-robot execution, never
    # actor.sample(); and actor_loss()/critic_loss() also only ever use the
    # mean, discarding std. So this has zero effect on rollout OR training
    # right now -- kept at 0 rather than a nonzero value that would falsely
    # suggest exploration noise is happening. actor.sample()-based rollout
    # noise (with proper safety clamping) is a possible future addition, not
    # implemented yet.
    parser.add_argument("--actor-fixed-std", type=float, default=0.0)
    parser.add_argument("--critic-hidden-dim", type=int, default=512)
    parser.add_argument("--critic-num-layers", type=int, default=3)
    parser.add_argument("--critic-activation", default="relu")
    parser.add_argument("--critic-residual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--critic-layer-norm", action=argparse.BooleanOptionalAction, default=True,
        help="LayerNorm in both online and target Critic hidden blocks. Enabled by "
        "default for RLPD-style offline/online high-UTD stability.",
    )
    parser.add_argument(
        "--rankq-alpha-success", type=float, default=1.0,
        help="Weight of RankQ's ranking loss terms on successful-trajectory transitions "
        "(executed action must outrank noisy/very-noisy/random/permuted variants, chained "
        "in quality order). 0 disables it for success transitions.",
    )
    parser.add_argument(
        "--rankq-alpha-failure", type=float, default=1.0,
        help="Weight of RankQ's weak ranking constraint on failed-trajectory transitions "
        "(executed action must only outrank a random action). 0 disables it for failures.",
    )
    parser.add_argument(
        "--rankq-noise-scale", type=float, default=0.15,
        help="Std of the Gaussian perturbation RankQ uses to build 'noisy'/'very-noisy' "
        "negative actions from the executed action.",
    )
    parser.add_argument(
        "--rankq-margin", type=float, default=0.1,
        help="Hard-hinge margin for RankQ pairs. Positive values stop the ranking loss "
        "once a pair is separated by this amount, preventing unbounded Q-gap growth. "
        "Set to 0 only to restore the original softplus behavior. With "
        "--rankq-margin-relative this is a fraction of mean|Q| instead of an absolute gap.",
    )
    parser.add_argument(
        "--rankq-margin-relative", action=argparse.BooleanOptionalAction, default=True,
        help="Interpret --rankq-margin as a fraction of the Critic's own mean|Q| rather "
        "than an absolute gap. Enabled by default: an absolute margin stops constraining "
        "the action ordering once Q drifts off the reward scale it was tuned against "
        "(observed at margin=0.1 against a Q that had drifted to ~4.8, where the ranking "
        "term ordered 2% of its own signal and the Critic ended up scoring the human's "
        "successful takeover action BELOW the actor's failing one).",
    )
    parser.add_argument(
        "--target-q-min", type=float, default=0.0,
        help="Lower bound on the bootstrapped target Q. 0.0 is correct whenever every "
        "reward is non-negative (Q is then a discounted sum of non-negative terms and "
        "cannot be negative). Pass a negative value only if the reward function has "
        "negative terms. Without this bound the policy-vs-data Q gap -- backup fits "
        "Q(s, a_data) but bootstraps Q(s', pi(s')), and BC deliberately keeps the actor "
        "below the data -- accumulates over an episode: measured here at 0.086 per step, "
        "which walked Q to -2.7 over 100k steps while per-step TD error stayed at 0.05.",
    )
    parser.add_argument(
        "--target-q-clip", type=float, default=3.0,
        help="Clamp on the bootstrapped target Q, the coarse backstop against TD "
        "overestimation running away. Must be set near the largest episode return the "
        "reward config can actually produce -- the previous default of 100.0 against a "
        "best-observed return of 1.675 never once triggered while Q drifted to ~4.8. "
        "0 or negative disables it.",
    )
    parser.add_argument(
        "--target-noise-std", type=float, default=0.1,
        help="TD3-style target policy smoothing: std of clipped noise added to the target "
        "actor's action before evaluating target_critic on it, so the critic can't fit an "
        "arbitrarily sharp function right at one exact action (a known source of the actor "
        "chasing local Q noise into jittery real-robot output). 0 disables it.",
    )
    parser.add_argument(
        "--target-noise-clip", type=float, default=0.3,
        help="Clip range for --target-noise-std's injected noise.",
    )

    # Safety.
    parser.add_argument(
        "--actor-action-clip-delta", type=float, default=0.1,
        help="Bound the RL actor's per-step output deviation from the VLA reference "
        "chunk during critical phase to within this value (see project_action_delta -- "
        "a smooth ref+limit*tanh(...) projection, not a plain clamp, so the bound holds "
        "exactly even when the reference itself lies outside [-1,1]). Applied identically "
        "at deploy time and in actor_loss/critic_loss's target computation during "
        "training. Set to a large value or handle with care if disabling; there is no "
        "hardware E-stop.\n"
        "Size this against the human corrections actually present in the data, not just "
        "against a comfort level: any correction larger than this bound is one the actor "
        "is structurally incapable of ever reproducing, no matter how good the learning "
        "signal is (the tanh projection's range is exactly +/-limit, and the deviation "
        "does NOT accumulate across chunks -- the VLA reference returns toward its own "
        "trajectory each chunk rather than following where the arm actually got to). "
        "Measured on this project at limit=0.2: 34.7% of intervened action elements were "
        "unreachable and the actor sat saturated against the bound on 46% of the elements "
        "it was supposed to be learning from. Note this interacts with "
        "--actor-slew-rate-limit: whichever is tighter is the one actually constraining "
        "the robot, and a small limit here makes the slew limit dead code.",
    )
    parser.add_argument(
        "--actor-slew-rate-limit", type=float, default=None,
        help="Cap how much the RL actor residual (action minus VLA reference) may change "
        "per physical timestep. The trusted VLA trajectory itself is not rate-limited. "
        "None (default) disables it.\n"
        "This is the better of the two safety bounds to lean on: it limits abruptness "
        "(what actually damages hardware) rather than total travel, so it can stay strict "
        "while --actor-action-clip-delta is opened up enough for the actor to reproduce "
        "real human corrections. Compare the two as limit vs slew*chunk_length -- at "
        "slew=0.03 over 25 steps that is 0.75 of travel, so a clip_delta of 0.2 meant this "
        "limit never once bound anything.\n"
        "Size it against the residual slew humans actually produce during interventions, "
        "not by intuition -- that is motion the hardware has already survived. Measured "
        "here: p50=0.008 p75=0.015 p90=0.029 p95=0.044 p99=0.112. Keep "
        "slew*chunk_length above action_clip_delta, or this limit stops bounding "
        "abruptness and starts silently capping total travel below the clip instead "
        "(at 0.02 over 25 steps it capped travel at 0.5 against a clip of 0.7, cutting "
        "reproducible human corrections from 82% to 65%). For a stronger training-time "
        "smoothness constraint use --actor-smoothness-weight, which penalizes rather "
        "than truncates.",
    )
    parser.add_argument(
        "--actor-smoothness-weight", type=float, default=0.0,
        help="Training-time complement to --actor-slew-rate-limit: weight on a penalty "
        "for adjacent-timestep differences in the actor residual within a chunk (see "
        "losses.actor_loss), discouraging oscillation rather than only clipping it at "
        "deploy time. 0.0 (default) disables it.",
    )

    # Online RL loop.
    parser.add_argument("--warmup-episodes", type=int, default=5)
    parser.add_argument(
        "--critic-only-episodes", type=int, default=10,
        help="Episodes after warmup where Q-driven Actor updates remain frozen. Trusted "
        "demonstration BC may continue in the background, while actor_deploy_scale=0 "
        "keeps physical control exactly VLA-equivalent.",
    )
    parser.add_argument(
        "--actor-unfreeze-ramp-episodes", type=int, default=10,
        help="Instead of snapping actor_update_interval straight from frozen to its "
        "configured value the instant --critic-only-episodes elapses, ramp it there over "
        "this many additional episodes. A hard flip lets the actor immediately chase, at "
        "full lr_actor/utd_ratio, a critic that has only just started forming a "
        "non-random value estimate -- a direct contributor to actor jitter right when "
        "critic-only ends. The physical Actor residual is ramped from 0 to 1 over the "
        "same window; its first rollout remains at 0. 0 enables a hard flip.",
    )
    parser.add_argument(
        "--min-warmup-transitions", type=int, default=1000,
        help="Warmup also requires at least this many transitions in the buffer, "
        "not just --warmup-episodes worth of episodes.",
    )
    parser.add_argument(
        "--min-warmup-successes", type=int, default=3,
        help="Warmup also requires at least this many successful episodes in the buffer.",
    )
    parser.add_argument(
        "--min-warmup-failures", type=int, default=3,
        help="Warmup also requires at least this many failed episodes in the buffer "
        "(a critic that has only ever seen success, or only failure, can't discriminate).",
    )
    parser.add_argument(
        "--stratified-sampling", action=argparse.BooleanOptionalAction, default=True,
        help="Sample training batches stratified across success/failure/intervention/recent "
        "transitions instead of uniformly, so sparse positive-reward transitions aren't "
        "drowned out.",
    )
    parser.add_argument("--replay-capacity", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--offline-cache-path", default=None,
        help="Fixed demonstration transition cache: either a cache directory "
        "containing chunk_transitions_train.pt or that .pt file directly.",
    )
    parser.add_argument(
        "--offline-batch-fraction", type=float, default=0.5,
        help="Fraction of every gradient batch drawn from --offline-cache-path. "
        "The remainder is sampled from online replay; ignored without a cache.",
    )
    # Actor lr well below critic lr: the critic should adapt quickly, the
    # actor -- which directly drives the robot -- should not.
    parser.add_argument("--lr-actor", type=float, default=3e-5)
    parser.add_argument("--lr-critic", type=float, default=1e-4)
    parser.add_argument(
        "--terminal-reward", type=float, default=1.0,
        help="Reward on a successful critical-phase attempt (failure is always 0). "
        "Scaled by --time-decay before being written -- see that flag.",
    )
    parser.add_argument(
        "--milestone-reward", type=float, default=0.3,
        help="One-time mid-phase shaping bonus awarded by --milestone-key "
        "(0.0 disables it). Scaled by --time-decay before being written.",
    )
    parser.add_argument(
        "--time-decay", type=float, default=0.995,
        help="Both --milestone-reward and --terminal-reward are multiplied by "
        "time_decay ** (chunks CLOSED in the attempt so far, not raw frames) when "
        "awarded, so a faster attempt scores higher than a slower one reaching the "
        "same milestone/outcome. On by default: a typical critical-phase attempt "
        "is ~50-300 closed chunks, and 0.995 gives a real ~0.6-0.8x range there "
        "(0.995**50=0.78, 0.995**300=0.22). Pass 1.0 to disable this and reproduce "
        "the exact old fixed-magnitude-on-success behavior. Deliberately separate "
        "from --gamma: gamma is what lets the sparse terminal reward bootstrap "
        "back across a whole multi-hundred-chunk critical-phase attempt, so "
        "lowering it to add time pressure would undermine that long-horizon "
        "credit assignment instead of adding an independently-tunable speed "
        "incentive.",
    )
    parser.add_argument(
        "--save-dir", default=None,
        help="Where to write step_NNNNNN checkpoints, selectable "
        "step_NNNNNN/online_state.pt training snapshots, and latest_online_state.pt. If omitted, "
        "auto-generated under outputs/online_rl/<MMDD>_<dataset-tag>/<HHMMSS>/, timestamped "
        "the same way as the raw dataset folder so a fresh session never collides with a "
        "previous one. Required (not auto-generated) when --resume-from is set, so new "
        "checkpoints land alongside the resumed run's history instead of a disconnected "
        "new folder -- pass the same --save-dir the original run used.",
    )
    parser.add_argument("--save-every-episodes", type=int, default=5)
    parser.add_argument(
        "--go-home-time-s", type=float, default=3.0,
        help="After each episode ends (s/f), ramp the follower back to the calibrated "
        "middle position (all non-gripper joints = 0 degrees) over this many seconds, "
        "before the teleop reset window. 0 disables this step.",
    )
    parser.add_argument(
        "--go-home-gripper-value", type=float, default=100.0,
        help="Gripper target (0-100) during go-home. VERIFY which end means 'open' for "
        "your specific hardware (mounting-dependent) before relying on this.",
    )
    parser.add_argument(
        "--go-home-positions",
        default=(
            '{"left_shoulder_pan.pos": 2038, "left_shoulder_lift.pos": 2081, '
            '"left_elbow_flex.pos": 3034, "left_wrist_flex.pos": 1142, "left_gripper.pos": 2164, '
            '"right_shoulder_pan.pos": 2066, "right_shoulder_lift.pos": 2160, '
            '"right_elbow_flex.pos": 2880, "right_wrist_flex.pos": 1066, "right_gripper.pos": 2209}'
        ),
        help="Per-joint go-home targets as raw motor ticks, as a JSON object -- paste the "
        "POS column straight from lerobot-calibrate's \"recording positions\" screen, no "
        "manual conversion needed. Defaults to this rig's own calibrated reset pose (the "
        "one recorded in README_online.md) -- NOT portable to a different physical robot "
        "or a re-calibration; pass --go-home-positions explicitly to override, or "
        '--go-home-positions "{}" to fall back to the calibrated midpoint (0 degrees) for '
        "every joint. Joints not listed fall back to the calibrated midpoint; gripper "
        "joints listed here override --go-home-gripper-value.",
    )
    parser.add_argument(
        "--resume-from", default=None,
        help="Explicit path to a complete online_state.pt snapshot from a previous run "
        "(e.g. outputs/pin_insert_online_rl/step_000100/online_state.pt). "
        "latest_online_state.pt is also accepted for crash recovery, but is not required. "
        "Restores actor/critic/"
        "target_critic weights, optimizer momentum, the full replay buffer, and the "
        "warmup/critic-only anchor, then resumes the episode counter from there -- "
        "--num-episodes is a total target inclusive of the resumed count, not "
        "'N more episodes'. Recording still starts a fresh video dataset; only the "
        "online-RL training state is carried over. Omit to start a fresh session "
        "(default).",
    )
    parser.add_argument(
        "--wandb", action=argparse.BooleanOptionalAction, default=False,
        help="Log actor/critic loss and replay-buffer/warmup progress to Weights & Biases, "
        "one point per recorded episode. Requires `pip install evo-rlt[wandb]` and "
        "`wandb login` beforehand. No model weights/checkpoints are uploaded.",
    )
    parser.add_argument("--wandb-project", default="evo-rlt")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--wandb-run-id", default=None,
        help="Stable W&B run ID (not the display name). Reuse the original ID to append "
        "metrics to the same run.",
    )
    parser.add_argument(
        "--wandb-resume", choices=["allow", "must", "never", "auto"], default=None,
        help="W&B resume policy. For an intentional continuation, pass --wandb-run-id "
        "<original-id> --wandb-resume must.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_online_train(args)


if __name__ == "__main__":
    main()
