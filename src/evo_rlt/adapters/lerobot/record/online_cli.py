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
        f"--policy.tau={args.tau}",
        f"--policy.utd_ratio={args.utd_ratio}",
        f"--policy.actor_update_interval={args.actor_update_interval}",
        f"--policy.actor_hidden_dim={args.actor_hidden_dim}",
        f"--policy.actor_num_layers={args.actor_num_layers}",
        f"--policy.actor_fixed_std={args.actor_fixed_std}",
        f"--policy.actor_activation={args.actor_activation}",
        f"--policy.actor_residual={'true' if args.actor_residual else 'false'}",
        f"--policy.critic_hidden_dim={args.critic_hidden_dim}",
        f"--policy.critic_num_layers={args.critic_num_layers}",
        f"--policy.critic_activation={args.critic_activation}",
        f"--policy.critic_residual={'true' if args.critic_residual else 'false'}",
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
        # rlt_toggle_key starts the critical-phase attempt; the next press (or
        # double-tap) ends JUST the critical phase (success/failure), hands
        # control back to VLA, and flushes that reward into the online replay
        # buffer right there (see loop.py) -- it does NOT end the recorded
        # episode. The episode keeps recording (e.g. VLA autonomously
        # finishing a subsequent step like placing the object) until you
        # press the whole-episode outcome key (episode_success_key/
        # episode_failure_key, s/f by default) once that's done.
        "--rlt.enable=true",
        f"--rlt.rl_phase_key={args.rlt_toggle_key}",
        f"--rlt.rl_phase_double_tap_window_s={args.double_tap_window_s}",
        "--rlt.skip_prefix_recording=true",
        "--rlt.rl_phase_key_toggles_critical_phase=true",
        "--rlt.start_in_teleop=false",
        # v1 online training does not support the RTC runtime.
        "--rlt.rtc_enabled=false",
        "--enable_episode_outcome_labeling=true",
        "--require_episode_success_label=true",
        "--intervention_state_machine_enabled=true",
        f"--policy_sync_to_teleop={'true' if teleop_argv else 'false'}",
        f"--vla_ref={'true' if args.vla_ref else 'false'}",
        f"--play_sounds={'true' if args.play_sounds else 'false'}",
        f"--estop_key={args.estop_key}",
        # Online RL training loop (see backend.OnlineRLConfig).
        "--online_rl.enable=true",
        f"--online_rl.warmup_episodes={args.warmup_episodes}",
        f"--online_rl.critic_only_episodes={args.critic_only_episodes}",
        f"--online_rl.min_warmup_transitions={args.min_warmup_transitions}",
        f"--online_rl.min_warmup_successes={args.min_warmup_successes}",
        f"--online_rl.min_warmup_failures={args.min_warmup_failures}",
        f"--online_rl.max_updates_per_episode={args.max_updates_per_episode}",
        f"--online_rl.use_stratified_sampling={'true' if args.stratified_sampling else 'false'}",
        f"--online_rl.replay_capacity={args.replay_capacity}",
        f"--online_rl.batch_size={args.batch_size}",
        f"--online_rl.lr_actor={args.lr_actor}",
        f"--online_rl.lr_critic={args.lr_critic}",
        f"--online_rl.save_dir={args.save_dir}",
        f"--online_rl.save_every_episodes={args.save_every_episodes}",
        f"--online_rl.go_home_time_s={args.go_home_time_s}",
        f"--online_rl.go_home_gripper_value={args.go_home_gripper_value}",
    ]
    if args.actor_action_clip_delta is not None:
        argv.append(f"--policy.actor_action_clip_delta={args.actor_action_clip_delta}")
    if args.remote_server:
        argv.append(f"--online_rl.remote_server={args.remote_server}")
        if args.remote_token:
            argv.append(f"--online_rl.remote_token={args.remote_token}")
    return argv


def print_online_train_summary(args: argparse.Namespace, paths) -> None:
    print("\nOnline RL training (synchronous, real hardware)")
    print(f"Dataset (raw episodes, for record-keeping): {paths.dataset_name} -> {paths.dataset_root}")
    if args.remote_server:
        print(
            f"Mode: REMOTE -- inference + training run on {args.remote_server} "
            "(this machine only drives the robot/teleop/dataset/keyboard loop, no local GPU needed)"
        )
        print(
            "Checkpoints are saved on the remote host's --save-dir, NOT the --save-dir below "
            "(that value is unused in remote mode)."
        )
    else:
        print(f"Checkpoints: {args.save_dir}")
    print(f"VLA: {args.vla_path}")
    print(f"RL token: {args.rl_token_path}")
    print(
        f"RL: warmup_episodes={args.warmup_episodes} (+min_transitions={args.min_warmup_transitions} "
        f"min_successes={args.min_warmup_successes} min_failures={args.min_warmup_failures}) "
        f"critic_only_episodes={args.critic_only_episodes} batch_size={args.batch_size} "
        f"lr_actor={args.lr_actor} lr_critic={args.lr_critic} utd_ratio={args.utd_ratio} "
        f"max_updates_per_episode={args.max_updates_per_episode} "
        f"stratified_sampling={args.stratified_sampling} save_every={args.save_every_episodes}"
    )
    print(
        f"Actor: hidden_dim={args.actor_hidden_dim} num_layers={args.actor_num_layers} "
        f"fixed_std={args.actor_fixed_std} | Critic: hidden_dim={args.critic_hidden_dim} "
        f"num_layers={args.critic_num_layers}"
    )
    print(
        f"Safety: actor_action_clip_delta={args.actor_action_clip_delta} "
        f"estop_key={args.estop_key} (grabs manual control -- not a hardware E-stop; "
        "stay near the leader arm / power cutoff)"
    )
    print(
        f"Controls: {args.rlt_toggle_key}=start/end critical-phase attempt (double-tap "
        f"within {args.double_tap_window_s:.1f}s = failure; reward flushed immediately, "
        "recording continues under VLA afterward), s/f=end the whole recorded episode "
        "once VLA finishes, "
        f"{args.teleop_toggle_key} or {args.estop_key}=grab manual control"
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


def run_online_train(args: argparse.Namespace) -> None:
    set_offline_env()
    keys = {args.teleop_toggle_key, args.estop_key, args.rlt_toggle_key}
    if len(keys) != 3:
        raise ValueError("--teleop-toggle-key, --estop-key, and --rlt-toggle-key must be distinct.")

    setup = load_robot_setup(args.setup_json)
    paths = resolve_run_paths(setup.setup, args.dataset_tag, DEFAULT_DATASET_TAG)
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
    parser.add_argument(
        "--remote-server", default=None,
        help="URL of a running runing_service/rlt_ac/online_serve.py process "
        "(e.g. http://1.2.3.4:8600). When set, inference AND training run on that "
        "machine instead of this one -- this process only drives the robot/teleop/"
        "dataset/keyboard loop and needs no GPU. --vla-path/--rl-token-path/"
        "--tokenizer-path and the policy hyperparameter flags below are still "
        "required (for config validation) but are NOT sent to the server; they "
        "must independently match how the server was started, or training "
        "silently runs with different hyperparameters than this session's logs "
        "suggest.",
    )
    parser.add_argument(
        "--remote-token", default=None,
        help="Bearer token for --remote-server, if it was started with --auth-token. "
        "Put the server behind a firewall/security group restricting access to this "
        "robot host regardless -- this token is not a substitute for that.",
    )
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
    parser.add_argument("--double-tap-window-s", type=float, default=0.6)
    parser.add_argument("--vla-ref", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--play-sounds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rlt-toggle-key", default="r")
    parser.add_argument("--teleop-toggle-key", default="space")
    parser.add_argument("--estop-key", default="x")
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
    parser.add_argument("--tau", type=float, default=0.005)
    # Gradient updates per NEW transition this episode added, capped by
    # --max-updates-per-episode (not a fixed count per episode). Default 1
    # (vs. the paper/offline default 5) so a synchronous session doesn't
    # stall for a long training burst after every episode on modest
    # hardware; raise once the loop is confirmed stable.
    parser.add_argument("--utd-ratio", type=int, default=1)
    parser.add_argument("--max-updates-per-episode", type=int, default=200)
    parser.add_argument("--actor-update-interval", type=int, default=2)

    # Network size. Defaults match src/evo_rlt/core/configs/ac_paper_screw.yaml
    # (the paper's complex-task tier: 3 layers, hidden_dim 512), appropriate
    # for multi-step, dexterous, contact-rich tasks like bimanual insertion.
    parser.add_argument("--actor-hidden-dim", type=int, default=512)
    parser.add_argument("--actor-num-layers", type=int, default=3)
    parser.add_argument("--actor-activation", default="relu")
    parser.add_argument("--actor-residual", action=argparse.BooleanOptionalAction, default=True)
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

    # Safety.
    parser.add_argument(
        "--actor-action-clip-delta", type=float, default=0.1,
        help="Clamp the RL actor's per-step output deviation from the VLA reference "
        "chunk during critical phase. Set to a large value or handle with care if "
        "disabling; there is no hardware E-stop.",
    )

    # Online RL loop.
    parser.add_argument("--warmup-episodes", type=int, default=5)
    parser.add_argument(
        "--critic-only-episodes", type=int, default=10,
        help="Episodes after warmup where only the critic updates (actor frozen at its "
        "zero-init-residual, VLA-equivalent behavior) so the critic isn't acting on a "
        "random value estimate when the actor starts moving.",
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
    # Actor lr well below critic lr: the critic should adapt quickly, the
    # actor -- which directly drives the robot -- should not.
    parser.add_argument("--lr-actor", type=float, default=3e-5)
    parser.add_argument("--lr-critic", type=float, default=1e-4)
    parser.add_argument("--save-dir", required=True)
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
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_online_train(args)


if __name__ == "__main__":
    main()
