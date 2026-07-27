#!/usr/bin/env python3

"""Serve the repository's bimanual SO-101 PI0.5 checkpoint over HTTP.

python ./runing_service/pi05_double/pi05_double_server.py \
  --policy-path pretrained/pi05_full_ft/pretrained_model \
  --device cuda \
  --host 192.168.1.84 \
  --port 8000
  
  """

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
RUNNING_SERVICE_DIR = SCRIPT_DIR.parent
REPO_ROOT = RUNNING_SERVICE_DIR.parent
sys.path.insert(0, str(RUNNING_SERVICE_DIR))

from pi05_http_server import PI05HTTPServer, PI05Service  # noqa: E402

logger = logging.getLogger(__name__)

TASK = (
    "Pick up the black hexagonal part with the right arm, pull the gray pin out "
    "of the white platform with the left arm, align the gray pin with the hole "
    "in the side of the black hexagonal part, insert the gray pin into the hole, "
    "and place the assembled object in the red square area."
)
DEFAULT_POLICY_PATH = REPO_ROOT / "outputs" / "pretrained_model"
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
    """PI0.5 service with an explicit, validated bimanual joint contract."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        action_keys = list(getattr(self.policy.config, "action_feature_names", []) or [])
        if action_keys != SO101_DOUBLE_ACTION_KEYS:
            raise ValueError(
                "Checkpoint joint order is incompatible with bi_so_follower."
                f"\nexpected={SO101_DOUBLE_ACTION_KEYS}\nactual={action_keys}"
            )
        if self.action_dim != len(action_keys):
            raise ValueError(
                f"Checkpoint action dimension is {self.action_dim}, expected {len(action_keys)}"
            )
        state_spec = self.input_features.get("observation.state")
        if state_spec is None or tuple(state_spec["shape"]) != (len(action_keys),):
            actual = None if state_spec is None else state_spec["shape"]
            raise ValueError(f"Expected a 12-D observation.state, got {actual}")
        self.action_keys = action_keys

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
        description="Bimanual SO-101 PI0.5 HTTP inference server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-body-mb", type=float, default=16.0)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    policy_path = args.policy_path.expanduser().resolve()
    if not policy_path.is_dir():
        raise FileNotFoundError(f"PI0.5 pretrained_model directory not found: {policy_path}")
    if args.max_body_mb <= 0:
        raise ValueError("--max-body-mb must be positive")

    service = SO101DoublePI05Service(
        policy_path=policy_path,
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
    logger.info("Serving PI0.5 at http://%s:%d", args.host, args.port)
    logger.info("Policy: %s", policy_path)
    logger.info("Contract: %s", service.metadata())
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
