#!/usr/bin/env python3

"""Capture bimanual SO-101 observations and execute chunks from a PI0.5 server.
python runing_service/pi05_double/pi05_double_deploy.py \
  --server http://192.168.1.100:8000 \
  --calibration-dir /path/to/bimanual_calibration \
  --left-port /dev/ttyACM3 \
  --right-port /dev/ttyACM2 \
  --camera left_wrist=4 \
  --camera right_wrist=2 \
  --camera right_front=6 \
  --camera-fourcc YUYV \
  --control-fps 30 \
  --actions-per-chunk 50 \
  --execute \
  --yes
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from lerobot.cameras import make_cameras_from_configs
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.bi_so_follower import BiSOFollower, BiSOFollowerConfig
from lerobot.robots.so_follower import SOFollowerConfig

logger = logging.getLogger(__name__)

TASK = (
    "Pick up the black hexagonal part with the right arm, pull the gray pin out "
    "of the white platform with the left arm, align the gray pin with the hole "
    "in the side of the black hexagonal part, insert the gray pin into the hole, "
    "and place the assembled object in the red square area."
)
DEFAULT_CAMERAS = ("left_wrist=4", "right_wrist=2", "right_front=6")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bimanual SO-101 client for the separated PI0.5 service.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--robot-id", default="my_bimanual_so101")
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        required=True,
        help="Same directory previously passed as --robot.calibration_dir.",
    )
    parser.add_argument("--left-port", default="/dev/ttyACM3")
    parser.add_argument("--right-port", default="/dev/ttyACM2")
    parser.add_argument(
        "--camera",
        action="append",
        dest="cameras",
        metavar="FEATURE=INDEX_OR_PATH",
        help=(
            "Repeat to override the defaults: left_wrist=4, right_wrist=2, "
            "right_front=6. Names must match the checkpoint."
        ),
    )
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-fourcc", default="YUYV")
    parser.add_argument("--control-fps", type=float, default=30.0)
    parser.add_argument(
        "--actions-per-chunk",
        type=int,
        default=50,
        help="Use fewer actions for more frequent replanning; maximum is 50.",
    )
    parser.add_argument("--chunks", type=int, default=0, help="0 runs until Ctrl+C.")
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=5.0,
        help="Maximum joint change per command in degrees; <=0 disables clipping.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually drive the robot. Without this flag the client is a safe dry run.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required with --execute to acknowledge the hardware safety check.",
    )
    return parser.parse_args()


def request_json(
    method: str, url: str, timeout: float, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
    headers = {} if body is None else {"Content-Type": "application/json"}
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read())
    except HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"{method} {url}: HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"{method} {url}: {exc.reason}") from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"{method} {url}: expected a JSON object")
    return result


def parse_camera_args(values: list[str]) -> dict[str, int | str]:
    cameras: dict[str, int | str] = {}
    for value in values:
        name, separator, source_text = value.partition("=")
        name, source_text = name.strip(), source_text.strip()
        if not separator or not name or not source_text:
            raise ValueError(f"Invalid --camera {value!r}; expected FEATURE=INDEX_OR_PATH")
        if name in cameras:
            raise ValueError(f"Duplicate camera feature {name!r}")
        try:
            source: int | str = int(source_text)
        except ValueError:
            source = source_text
        cameras[name] = source
    return cameras


def encode_jpeg(frame: Any, quality: int) -> dict[str, str]:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    output = BytesIO()
    image.save(output, format="JPEG", quality=quality)
    return {"encoding": "jpeg", "data": base64.b64encode(output.getvalue()).decode("ascii")}


def build_wire_observation(
    raw_observation: dict[str, Any],
    camera_frames: dict[str, Any],
    input_features: dict[str, dict[str, Any]],
    state_keys: list[str],
    jpeg_quality: int,
) -> dict[str, Any]:
    observation: dict[str, Any] = {}
    for key, spec in input_features.items():
        shape = tuple(spec["shape"])
        if spec["type"] == "VISUAL":
            name = key.removeprefix("observation.images.")
            frame = np.asarray(camera_frames[name], dtype=np.uint8)
            expected_hwc = (shape[1], shape[2], shape[0])
            if frame.shape != expected_hwc:
                raise ValueError(f"Camera {name!r}: expected {expected_hwc}, got {frame.shape}")
            observation[key] = encode_jpeg(frame, jpeg_quality)
        elif key == "observation.state":
            state = [float(raw_observation[name]) for name in state_keys]
            if tuple(np.asarray(state).shape) != shape:
                raise ValueError(f"State expected {shape}, got {(len(state),)}")
            observation[key] = state
        else:
            raise ValueError(f"Cannot provide checkpoint feature {key!r}")
    return observation


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.execute and not args.yes:
        raise ValueError("--execute requires --yes after checking calibration, limits, and e-stop")
    if not 1 <= args.jpeg_quality <= 95:
        raise ValueError("--jpeg-quality must be in [1, 95]")
    if args.control_fps <= 0 or args.timeout <= 0 or args.chunks < 0:
        raise ValueError("--control-fps/--timeout must be positive and --chunks must be >= 0")
    calibration_dir = args.calibration_dir.expanduser().resolve()
    if not calibration_dir.is_dir():
        raise FileNotFoundError(f"Calibration directory not found: {calibration_dir}")

    base_url = args.server.rstrip("/")
    metadata = request_json("GET", f"{base_url}/metadata", args.timeout)
    if metadata.get("protocol_version") != 1 or metadata.get("policy") != "pi05":
        raise RuntimeError(f"Unexpected server metadata: {metadata}")
    if metadata.get("robot_type") != "bi_so_follower":
        raise RuntimeError("Server is not advertising the bi_so_follower contract")
    task = args.task.strip() or str(metadata.get("default_task", "")).strip()
    if not task:
        raise ValueError("PI0.5 requires a non-empty task")
    actions_per_chunk = args.actions_per_chunk
    if not 1 <= actions_per_chunk <= int(metadata["chunk_size"]):
        raise ValueError(f"--actions-per-chunk must be in [1, {metadata['chunk_size']}]")

    state_keys = list(metadata.get("state_keys", []))
    action_keys = list(metadata.get("action_keys", []))
    if not state_keys or not action_keys or state_keys != action_keys:
        raise RuntimeError("Server state/action joint contract is missing or inconsistent")

    visual_features = {
        key.removeprefix("observation.images."): spec
        for key, spec in metadata["input_features"].items()
        if spec["type"] == "VISUAL"
    }
    camera_sources = parse_camera_args(args.cameras or list(DEFAULT_CAMERAS))
    if set(camera_sources) != set(visual_features):
        raise ValueError(
            "Camera mapping does not match checkpoint: "
            f"missing={sorted(set(visual_features) - set(camera_sources))}, "
            f"unknown={sorted(set(camera_sources) - set(visual_features))}"
        )
    cameras = make_cameras_from_configs(
        {
            name: OpenCVCameraConfig(
                index_or_path=source,
                width=args.camera_width,
                height=args.camera_height,
                fps=args.camera_fps,
                fourcc=args.camera_fourcc,
            )
            for name, source in camera_sources.items()
        }
    )

    relative_limit = args.max_relative_target if args.max_relative_target > 0 else None
    robot = BiSOFollower(
        BiSOFollowerConfig(
            id=args.robot_id,
            calibration_dir=calibration_dir,
            left_arm_config=SOFollowerConfig(
                port=args.left_port,
                max_relative_target=relative_limit,
                use_degrees=True,
            ),
            right_arm_config=SOFollowerConfig(
                port=args.right_port,
                max_relative_target=relative_limit,
                use_degrees=True,
            ),
        )
    )
    robot_state_keys = {key for key, value in robot.observation_features.items() if value is float}
    robot_action_keys = set(robot.action_features)
    if set(state_keys) != robot_state_keys or set(action_keys) != robot_action_keys:
        raise ValueError(
            "Checkpoint and robot joint names differ; refusing unsafe execution. "
            f"state_delta={sorted(set(state_keys) ^ robot_state_keys)}, "
            f"action_delta={sorted(set(action_keys) ^ robot_action_keys)}"
        )

    logger.info(
        "Ready: server=%s task=%r cameras=%s chunk=%d fps=%.1f execute=%s",
        base_url,
        task,
        camera_sources,
        actions_per_chunk,
        args.control_fps,
        args.execute,
    )
    period = 1.0 / args.control_fps
    sequence_id = 0
    robot.connect()
    try:
        for camera in cameras.values():
            camera.connect()
        while args.chunks == 0 or sequence_id < args.chunks:
            request_started = time.perf_counter()
            raw_observation = robot.get_observation()
            camera_frames = {name: camera.read_latest() for name, camera in cameras.items()}
            payload = {
                "protocol_version": 1,
                "sequence_id": sequence_id,
                "task": task,
                "actions_per_chunk": actions_per_chunk,
                "observation": build_wire_observation(
                    raw_observation,
                    camera_frames,
                    metadata["input_features"],
                    state_keys,
                    args.jpeg_quality,
                ),
            }
            result = request_json("POST", f"{base_url}/predict", args.timeout, payload)
            if int(result.get("sequence_id", -1)) != sequence_id:
                raise ValueError("Response sequence_id does not match request")
            actions = np.asarray(result["actions"], dtype=np.float32)
            expected_shape = (actions_per_chunk, len(action_keys))
            if actions.shape != expected_shape or not np.isfinite(actions).all():
                raise ValueError(f"Invalid actions {actions.shape}; expected {expected_shape}")
            logger.info(
                "chunk=%d round_trip=%.1fms server=%.1fms actions=%s",
                sequence_id,
                (time.perf_counter() - request_started) * 1000,
                float(result.get("server_ms", -1)),
                actions.shape,
            )
            for row in actions:
                tick_started = time.perf_counter()
                action = dict(zip(action_keys, map(float, row), strict=True))
                if args.execute:
                    robot.send_action(action)
                else:
                    logger.info("dry-run action=%s", np.array2string(row, precision=3))
                time.sleep(max(0.0, period - (time.perf_counter() - tick_started)))
            sequence_id += 1
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        for camera in cameras.values():
            if camera.is_connected:
                camera.disconnect()
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
