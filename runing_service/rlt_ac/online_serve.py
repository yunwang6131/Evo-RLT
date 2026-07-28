#!/usr/bin/env python3

"""云端/远程 GPU 上跑的 Evo-RLT 在线 RL 训练服务（推理 + replay buffer + TD3+BC 更新）。

配合 ``evo-rlt-online-train --remote-server http://<this-host>:<port>`` 使用：
机械臂本机只跑 `evo-rlt-online-train`（机器人 I/O + teleop + 键盘 + 数据集写入），
VLA + rlt_ac actor/critic 的推理和训练都在这台服务上完成，本机不需要 GPU。

    python runing_service/rlt_ac/online_serve.py \\
      --host 0.0.0.0 --port 8600 \\
      --auth-token "$(openssl rand -hex 16)" \\
      --vla-path pretrained/pi05_full_ft/pretrained_model \\
      --rl-token-path outputs/pin_insert_rl_token/checkpoints/last/pretrained_model \\
      --tokenizer-path /path/to/paligemma-3b-pt-224/snapshots/xxx \\
      --save-dir outputs/pin_insert_online_rl_remote

训练数学（replay buffer、TD3+BC 梯度更新、warmup/critic-only 窗口、checkpoint 保存）
复用 ``evo_rlt.adapters.lerobot.record.online_trainer.OnlineRLTrainer`` ——
和本机 ``evo-rlt-online-train`` 用的是同一个类，不是这里单独抄一份，所以以后改
``OnlineRLTrainer`` 会同时影响本机训练和这个云端服务，两边不会分叉。这个文件本身
只是把它包一层 HTTP 服务，并且只放在 ``runing_service`` 里，不进 evo_rlt 包、不注册
console script。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evo_rlt.adapters.lerobot.policies.configuration_rlt_ac import ChunkACPolicyConfig  # noqa: E402
from evo_rlt.adapters.lerobot.policies.modeling_rlt_ac import ChunkACPolicy  # noqa: E402
from evo_rlt.adapters.lerobot.record.backend import OnlineRLConfig  # noqa: E402
from evo_rlt.adapters.lerobot.record.common import load_dataset_stats_from_pretrained  # noqa: E402
from evo_rlt.adapters.lerobot.record.hil import (  # noqa: E402
    ACPInferenceConfig,
    _predict_policy_action_with_acp_inference,
)
from evo_rlt.adapters.lerobot.record.online_trainer import OnlineRLTrainer  # noqa: E402
from evo_rlt.adapters.lerobot.serve.codec import decode_observation_frame, decode_tensor, encode_tensor  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.utils.device_utils import get_safe_torch_device  # noqa: E402

logger = logging.getLogger(__name__)


class OnlineRLService:
    def __init__(self, *, policy_cfg: ChunkACPolicyConfig, online_rl_cfg: OnlineRLConfig, vla_ref: bool = True):
        self.policy_cfg = policy_cfg
        # This service has neither dataset metadata nor a simulation EnvConfig.
        # LeRobot's make_policy() requires exactly one of those, even though
        # ChunkACPolicyConfig already carries all state/action feature shapes.
        # Construct the policy directly so remote online inference does not
        # fail before startup with the factory's ds_meta/env_cfg validation.
        policy_cfg.validate_features()
        self.policy = ChunkACPolicy(policy_cfg).to(policy_cfg.device)
        self.policy.vla_ref = vla_ref
        if hasattr(self.policy, "set_rl_mode"):
            self.policy.set_rl_mode()
        # Fresh rlt_ac policy, no dataset at all here -- load the REAL
        # normalization the frozen VLA was actually trained with from its own
        # checkpoint (policy_preprocessor.json + *_normalizer_processor.
        # safetensors), instead of leaving state/action un-normalized. See
        # load_dataset_stats_from_pretrained()'s docstring: without this the
        # VLA reads as "acting randomly" even with zero RL/actor involvement.
        dataset_stats = load_dataset_stats_from_pretrained(policy_cfg.vla_pretrained_path)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=None,
            dataset_stats=dataset_stats,
            preprocessor_overrides={"device_processor": {"device": policy_cfg.device}},
        )
        self.trainer = OnlineRLTrainer(self.policy, online_rl_cfg, policy_cfg)
        self._device = get_safe_torch_device(policy_cfg.device)
        self._use_amp = getattr(policy_cfg, "use_amp", False)
        self._lock = threading.RLock()
        self._last_state_vec: torch.Tensor | None = None
        self._last_ref_chunk: torch.Tensor | None = None

    def metadata(self) -> dict[str, Any]:
        return {
            "action_dim": self.policy_cfg.action_dim,
            "proprio_dim": self.policy_cfg.proprio_dim,
            "chunk_length": self.policy_cfg.chunk_length,
            "chunk_exec_steps": self.policy_cfg.chunk_exec_steps,
        }

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        observation_frame = decode_observation_frame(payload["observation"])
        task = payload.get("task") or ""
        robot_type = payload.get("robot_type")
        with self._lock:
            action = _predict_policy_action_with_acp_inference(
                observation_frame=observation_frame,
                policy=self.policy,
                device=self._device,
                preprocessor=self.preprocessor,
                postprocessor=self.postprocessor,
                use_amp=self._use_amp,
                task=task,
                robot_type=robot_type,
                # CFG (if any) was already resolved into `task`'s text by the
                # client (see hil.py's remote branch) -- treat it as opaque here.
                acp_inference=ACPInferenceConfig(enable=False),
            )
            get_chunk_tensors = getattr(self.policy, "get_last_chunk_tensors", None)
            chunk_tensors = get_chunk_tensors() if callable(get_chunk_tensors) else None
            if chunk_tensors is not None:
                self._last_state_vec, self._last_ref_chunk = chunk_tensors
            pop_step_metadata = getattr(self.policy, "pop_step_metadata", None)
            step_metadata = pop_step_metadata() if callable(pop_step_metadata) else None
        return {
            "action": encode_tensor(action),
            "state_vec": encode_tensor(self._last_state_vec),
            "ref_chunk": encode_tensor(self._last_ref_chunk),
            "phase": step_metadata.phase if step_metadata is not None else None,
            "source_type": step_metadata.source_type if step_metadata is not None else None,
        }

    # Zero-arg, no-return ChunkACPolicy/RLTActionModifier methods the local
    # HIL loop (see loop.py) calls on `rlt = policy` at various key-press
    # transitions -- allowlisted (rather than a blind getattr(self.policy,
    # name)()) so this network-facing endpoint can't be used to invoke
    # arbitrary policy methods (e.g. save_pretrained, to, train) even if
    # network isolation is ever misconfigured. Add a name here if a future
    # evo_rlt version calls a new one on `rlt` and the client's generic
    # __getattr__ forward (see remote_client.py) starts hitting "unknown
    # method" for it.
    _ALLOWED_POLICY_METHODS = frozenset(
        {"set_rl_mode", "reset", "set_vla_mode", "trigger_critical_phase", "interrupt_chunk"}
    )

    def call_policy_method(self, payload: dict[str, Any]) -> dict[str, Any]:
        method = payload["method"]
        if method not in self._ALLOWED_POLICY_METHODS:
            raise ValueError(f"Policy method {method!r} is not allowlisted for remote calls.")
        with self._lock:
            if hasattr(self.policy, method):
                getattr(self.policy, method)()
        return {"ok": True}

    def episode_start(self, payload: dict[str, Any]) -> dict[str, Any]:
        episode_id = int(payload["episode_id"])
        with self._lock:
            baseline = self.trainer.start_episode(episode_id)
        return {"buffer_total_added_before": baseline}

    def episode_frame(self, payload: dict[str, Any]) -> dict[str, Any]:
        action = decode_tensor(payload["action"])
        source_type = float(payload["source_type"])
        is_critical = float(payload["is_critical"])
        with self._lock:
            # state_vec/ref_chunk are NOT sent by the client -- they're
            # exactly whatever this service returned from the most recent
            # /predict call (peek semantics, see RLTActionModifier.
            # get_last_chunk_tensors()), so re-sending them over the wire
            # would be redundant.
            self.trainer.collector.on_frame(
                action=action,
                state_vec=self._last_state_vec,
                ref_chunk=self._last_ref_chunk,
                source_type=source_type,
                is_critical=is_critical,
            )
        return {"ok": True}

    def episode_flush(self, payload: dict[str, Any]) -> dict[str, Any]:
        episode_success = bool(payload["episode_success"])
        with self._lock:
            self.trainer.collector.flush_episode(episode_success)
        return {"ok": True}

    def episode_update(self, payload: dict[str, Any]) -> dict[str, Any]:
        recorded_episodes = int(payload["recorded_episodes"])
        buffer_total_added_before = int(payload["buffer_total_added_before"])
        with self._lock:
            stats = self.trainer.maybe_update(recorded_episodes, buffer_total_added_before)
        return {"stats": stats}

    def episode_save_latest_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        completed_episodes = int(payload["completed_episodes"])
        with self._lock:
            self.trainer.save_latest_state(completed_episodes)
        return {"ok": True}


_POST_ROUTES = {
    "/predict": OnlineRLService.predict,
    "/policy/call": OnlineRLService.call_policy_method,
    "/episode/start": OnlineRLService.episode_start,
    "/episode/frame": OnlineRLService.episode_frame,
    "/episode/flush": OnlineRLService.episode_flush,
    "/episode/update": OnlineRLService.episode_update,
    "/episode/save_latest_state": OnlineRLService.episode_save_latest_state,
}


class OnlineRLRequestHandler(BaseHTTPRequestHandler):
    server: "OnlineRLHTTPServer"

    def _check_auth(self) -> bool:
        if not self.server.auth_token:
            return True
        header = self.headers.get("Authorization", "")
        if header == f"Bearer {self.server.auth_token}":
            return True
        self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "missing or invalid bearer token"})
        return False

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
        elif self.path == "/metadata":
            self._send_json(HTTPStatus.OK, self.server.service.metadata())
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        handler = _POST_ROUTES.get(self.path)
        if handler is None:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        if not self._check_auth():
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if not 0 <= content_length <= self.server.max_body_bytes:
                raise ValueError(
                    f"Invalid request size {content_length}; max={self.server.max_body_bytes} bytes"
                )
            payload = json.loads(self.rfile.read(content_length)) if content_length else {}
            self._send_json(HTTPStatus.OK, handler(self.server.service, payload))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:
            logger.exception("Request to %s failed", self.path)
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        logger.info("%s - %s", self.client_address[0], format % args)


class OnlineRLHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        address: tuple[str, int],
        service: OnlineRLService,
        *,
        max_body_bytes: int,
        auth_token: str | None,
    ) -> None:
        super().__init__(address, OnlineRLRequestHandler)
        self.service = service
        self.max_body_bytes = max_body_bytes
        self.auth_token = auth_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cloud/remote HTTP service for rlt_ac online RL training "
        "(inference + replay buffer + TD3+BC updates). Pair with "
        "`evo-rlt-online-train --remote-server http://<this-host>:<port>` on "
        "the machine driving the robot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8600)
    parser.add_argument(
        "--auth-token", default=None,
        help="If set, require `Authorization: Bearer <token>` on every POST. Not a "
        "substitute for a firewall/security group restricting access to the robot host.",
    )
    parser.add_argument("--max-body-mb", type=float, default=16.0)
    parser.add_argument("--vla-path", required=True)
    parser.add_argument("--rl-token-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-length", type=int, default=10)
    parser.add_argument("--chunk-exec-steps", type=int, default=25)
    parser.add_argument("--action-dim", type=int, default=12)
    parser.add_argument("--proprio-dim", type=int, default=12)
    parser.add_argument("--vla-ref", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--utd-ratio", type=int, default=1)
    parser.add_argument("--max-updates-per-episode", type=int, default=200)
    parser.add_argument("--actor-update-interval", type=int, default=2)
    parser.add_argument("--actor-hidden-dim", type=int, default=512)
    parser.add_argument("--actor-num-layers", type=int, default=3)
    parser.add_argument("--actor-activation", default="relu")
    parser.add_argument("--actor-residual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--actor-fixed-std", type=float, default=0.0)
    parser.add_argument("--critic-hidden-dim", type=int, default=512)
    parser.add_argument("--critic-num-layers", type=int, default=3)
    parser.add_argument("--critic-activation", default="relu")
    parser.add_argument("--critic-residual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--actor-action-clip-delta", type=float, default=0.1)

    parser.add_argument("--warmup-episodes", type=int, default=5)
    parser.add_argument("--critic-only-episodes", type=int, default=10)
    parser.add_argument("--min-warmup-transitions", type=int, default=1000)
    parser.add_argument("--min-warmup-successes", type=int, default=3)
    parser.add_argument("--min-warmup-failures", type=int, default=3)
    parser.add_argument("--stratified-sampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--replay-capacity", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr-actor", type=float, default=3e-5)
    parser.add_argument("--lr-critic", type=float, default=1e-4)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--save-every-episodes", type=int, default=5)
    return parser.parse_args()


def build_service(args: argparse.Namespace) -> OnlineRLService:
    policy_cfg = ChunkACPolicyConfig(
        vla_pretrained_path=args.vla_path,
        rl_token_pretrained_path=args.rl_token_path,
        tokenizer_path=args.tokenizer_path,
        phase_mode="manual",
        chunk_exec_steps=args.chunk_exec_steps,
        chunk_length=args.chunk_length,
        action_dim=args.action_dim,
        proprio_dim=args.proprio_dim,
        actor_residual_to_ref=True,
        # See online_cli.py's build_online_train_argv() comment: with
        # residual_to_ref, ref-dropout on the input side no longer forces
        # independent action generation, so it's disabled rather than left at
        # the (non-residual-tuned) 0.5 default.
        actor_ref_dropout_p=0.0,
        gamma=args.gamma,
        beta=args.beta,
        tau=args.tau,
        utd_ratio=args.utd_ratio,
        actor_update_interval=args.actor_update_interval,
        actor_hidden_dim=args.actor_hidden_dim,
        actor_num_layers=args.actor_num_layers,
        actor_fixed_std=args.actor_fixed_std,
        actor_activation=args.actor_activation,
        actor_residual=args.actor_residual,
        critic_hidden_dim=args.critic_hidden_dim,
        critic_num_layers=args.critic_num_layers,
        critic_activation=args.critic_activation,
        critic_residual=args.critic_residual,
        device=args.device,
        actor_action_clip_delta=args.actor_action_clip_delta,
    )
    online_rl_cfg = OnlineRLConfig(
        enable=True,
        warmup_episodes=args.warmup_episodes,
        critic_only_episodes=args.critic_only_episodes,
        replay_capacity=args.replay_capacity,
        batch_size=args.batch_size,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        max_updates_per_episode=args.max_updates_per_episode,
        min_warmup_transitions=args.min_warmup_transitions,
        min_warmup_successes=args.min_warmup_successes,
        min_warmup_failures=args.min_warmup_failures,
        use_stratified_sampling=args.stratified_sampling,
        save_dir=args.save_dir,
        save_every_episodes=args.save_every_episodes,
    )
    return OnlineRLService(policy_cfg=policy_cfg, online_rl_cfg=online_rl_cfg, vla_ref=args.vla_ref)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    service = build_service(args)
    server = OnlineRLHTTPServer(
        (args.host, args.port),
        service,
        max_body_bytes=int(args.max_body_mb * 1024 * 1024),
        auth_token=args.auth_token,
    )
    logger.info("Evo-RLT online-RL service listening on http://%s:%d", args.host, args.port)
    logger.info("metadata=%s", service.metadata())
    logger.info(
        "Checkpoints and latest_online_state.pt will be written to %s (same layout as local "
        "online training)",
        args.save_dir,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
