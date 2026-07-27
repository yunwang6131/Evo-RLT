#!/usr/bin/env python3

"""SO-101 双臂 PI05 HTTP 推理服务。

云端部署时，请通过安全组/防火墙只允许机器人主机访问端口。checkpoint
必须保存标准 SO-101 双臂的 ``action_feature_names``；服务会将关节顺序发布
到 metadata，执行端据此做严格校验，避免仅按向量维度误映射。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch

from pi05_http_server import PI05HTTPServer, PI05Service

logger = logging.getLogger(__name__)

SO101_JOINT_SUFFIXES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
SO101_DOUBLE_ACTION_KEYS = [
    f"{side}_{joint}" for side in ("left", "right") for joint in SO101_JOINT_SUFFIXES
]


class SO101DoublePI05Service(PI05Service):
    """PI05 service that validates and publishes the physical joint contract."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        action_keys = getattr(self.policy.config, "action_feature_names", None)
        if not action_keys:
            raise ValueError(
                "Checkpoint has no action_feature_names. Re-export it with dataset feature "
                "metadata; inferring joint order from dimensions is unsafe."
            )
        self.action_keys = list(action_keys)
        if self.action_keys != SO101_DOUBLE_ACTION_KEYS:
            raise ValueError(
                "Checkpoint action_feature_names do not match canonical SO-101 double-arm order."
                f"\nexpected={SO101_DOUBLE_ACTION_KEYS}\nactual={self.action_keys}"
            )
        if self.action_dim != len(self.action_keys):
            raise ValueError(
                f"Checkpoint action shape is {self.action_dim}, but action_feature_names has "
                f"{len(self.action_keys)} entries"
            )
        state_spec = self.input_features.get("observation.state")
        if state_spec is None or tuple(state_spec["shape"]) != (len(self.action_keys),):
            actual = None if state_spec is None else state_spec["shape"]
            raise ValueError(
                f"SO-101 double-arm observation.state must be [{len(self.action_keys)}], got {actual}"
            )

    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata()
        metadata.update(
            {
                "robot_type": "bi_so_follower",
                "state_keys": self.action_keys,
                "action_keys": self.action_keys,
            }
        )
        return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a bimanual SO-101 PI05 checkpoint over HTTP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--policy_path", type=Path, required=True)
    parser.add_argument("--task", required=True, help="Default language instruction.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_body_mb", type=float, default=16.0)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if not args.policy_path.is_dir():
        raise FileNotFoundError(f"PI05 pretrained_model directory not found: {args.policy_path}")
    if args.max_body_mb <= 0:
        raise ValueError("--max_body_mb must be positive")

    service = SO101DoublePI05Service(
        policy_path=args.policy_path,
        device=args.device,
        mock=False,
        chunk_size=50,
        latency_ms=0.0,
        default_task=args.task,
    )
    server = PI05HTTPServer(
        (args.host, args.port),
        service,
        max_body_bytes=int(args.max_body_mb * 1024 * 1024),
    )
    logger.info("PI05 SO-101 double-arm server: http://%s:%d", args.host, args.port)
    logger.info("metadata=%s", service.metadata())
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
