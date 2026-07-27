#!/usr/bin/env python3

"""SO-101 双臂本地采集/执行端，可连接本机或云端 PI05 服务。

默认只打印动作，不驱动机器人。确认标定、关节顺序、单位和限位后才可添加
``--execute --yes``。每个 ``--camera`` 名称必须与 checkpoint 的
``observation.images.<name>`` 完全一致，例如::

    python examples/runing_service/pi05_double_local_executor.py \
      --server http://127.0.0.1:8000 --task "pick the cube" \
      --left_port COM5 --right_port COM6 \
      --camera base_0_rgb=0 --camera left_wrist_0_rgb=1 \
      --camera right_wrist_0_rgb=2
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import time
from io import BytesIO
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture SO-101 bimanual observations and execute PI05 action chunks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--server", required=True, help="For example http://127.0.0.1:8000")
    parser.add_argument("--task", default="", help="Overrides the server's default task.")
    parser.add_argument("--left_port", required=True, help="Left SO-101 motor-bus port.")
    parser.add_argument("--right_port", required=True, help="Right SO-101 motor-bus port.")
    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        metavar="FEATURE=INDEX",
        help="Repeat for each checkpoint image, e.g. --camera base_0_rgb=0.",
    )
    parser.add_argument("--camera_fps", type=int, default=30)
    parser.add_argument("--control_fps", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=0, help="0 runs until Ctrl+C.")
    parser.add_argument("--actions_per_chunk", type=int, default=0, help="0 uses server metadata.")
    parser.add_argument("--jpeg_quality", type=int, default=75)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--execute", action="store_true", help="Actually send received actions.")
    parser.add_argument("--yes", action="store_true", help="Required with --execute.")
    parser.add_argument("--robot_id", default="pi05_double")
    parser.add_argument(
        "--max_relative_target",
        type=float,
        default=5.0,
        help="Maximum per-command position change; <=0 disables clipping.",
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


def parse_camera_args(values: list[str]) -> dict[str, int]:
    cameras: dict[str, int] = {}
    for value in values:
        name, separator, index_text = value.partition("=")
        name = name.strip()
        if not separator or not name:
            raise ValueError(f"Invalid --camera {value!r}; expected FEATURE=INDEX")
        if name in cameras:
            raise ValueError(f"Duplicate --camera feature {name!r}")
        try:
            cameras[name] = int(index_text)
        except ValueError as exc:
            raise ValueError(f"Camera index must be an integer in {value!r}") from exc
    return cameras


def encode_jpeg(frame: Any, quality: int) -> dict[str, str]:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    output = BytesIO()
    image.save(output, format="JPEG", quality=quality)
    return {"encoding": "jpeg", "data": base64.b64encode(output.getvalue()).decode("ascii")}


def build_wire_observation(
    raw_state: dict[str, Any],
    camera_frames: dict[str, Any],
    input_features: dict[str, dict[str, Any]],
    state_keys: list[str],
    jpeg_quality: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, spec in input_features.items():
        shape = tuple(spec["shape"])
        if spec["type"] == "VISUAL":
            name = key.removeprefix("observation.images.")
            frame = np.asarray(camera_frames[name], dtype=np.uint8)
            expected = (shape[1], shape[2], shape[0])
            if frame.shape != expected:
                raise ValueError(f"Camera {name!r} expected HWC {expected}, got {frame.shape}")
            result[key] = encode_jpeg(frame, jpeg_quality)
        elif key == "observation.state":
            state = [float(raw_state[name]) for name in state_keys]
            if tuple(np.asarray(state).shape) != shape:
                raise ValueError(f"State expected {shape}, got {(len(state),)}")
            result[key] = state
        else:
            raise ValueError(
                f"SO-101 executor cannot provide checkpoint feature {key!r} ({spec['type']})"
            )
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.execute and not args.yes:
        raise ValueError("--execute requires --yes after checking calibration and safety limits")
    if not 1 <= args.jpeg_quality <= 95:
        raise ValueError("--jpeg_quality must be in [1, 95]")
    if args.control_fps <= 0 or args.timeout <= 0:
        raise ValueError("--control_fps and --timeout must be positive")

    base_url = args.server.rstrip("/")
    metadata = request_json("GET", f"{base_url}/metadata", args.timeout)
    if metadata.get("protocol_version") != 1 or metadata.get("policy") != "pi05":
        raise RuntimeError(f"Unexpected server metadata: {metadata}")
    if metadata.get("robot_type") != "bi_so_follower":
        raise RuntimeError("Server is not advertising the SO-101 double-arm contract")
    task = args.task.strip() or str(metadata.get("default_task", "")).strip()
    if not task:
        raise ValueError("PI05 needs --task or a server default task")
    actions_per_chunk = args.actions_per_chunk or int(metadata["actions_per_chunk"])
    if not 1 <= actions_per_chunk <= int(metadata["chunk_size"]):
        raise ValueError("--actions_per_chunk is outside the checkpoint chunk range")

    state_keys = list(metadata.get("state_keys", []))
    action_keys = list(metadata.get("action_keys", []))
    if not state_keys or not action_keys:
        raise RuntimeError("Server metadata is missing state_keys/action_keys")

    visual_features = {
        key.removeprefix("observation.images."): spec
        for key, spec in metadata["input_features"].items()
        if spec["type"] == "VISUAL"
    }
    camera_indices = parse_camera_args(args.camera)
    if set(camera_indices) != set(visual_features):
        raise ValueError(
            "Camera mapping does not match checkpoint: "
            f"missing={sorted(set(visual_features) - set(camera_indices))}, "
            f"unknown={sorted(set(camera_indices) - set(visual_features))}"
        )
    camera_configs = {}
    for name, index in camera_indices.items():
        shape = tuple(visual_features[name]["shape"])
        if len(shape) != 3 or shape[0] != 3:
            raise ValueError(f"Camera feature {name!r} is not RGB CHW: {shape}")
        camera_configs[name] = OpenCVCameraConfig(
            index_or_path=index,
            width=shape[2],
            height=shape[1],
            fps=args.camera_fps,
        )
    cameras = make_cameras_from_configs(camera_configs)

    relative_limit = args.max_relative_target if args.max_relative_target > 0 else None
    robot = BiSOFollower(
        BiSOFollowerConfig(
            id=args.robot_id,
            left_arm_config=SOFollowerConfig(
                port=args.left_port, max_relative_target=relative_limit, use_degrees=True
            ),
            right_arm_config=SOFollowerConfig(
                port=args.right_port, max_relative_target=relative_limit, use_degrees=True
            ),
        )
    )
    robot_state_keys = {k for k, value in robot.observation_features.items() if value is float}
    robot_action_keys = set(robot.action_features)
    if set(state_keys) != robot_state_keys or set(action_keys) != robot_action_keys:
        raise ValueError(
            "Checkpoint/SO-101 names differ; refusing unsafe mapping. "
            f"state_delta={sorted(set(state_keys) ^ robot_state_keys)}, "
            f"action_delta={sorted(set(action_keys) ^ robot_action_keys)}"
        )
    if int(metadata["action_dim"]) != len(action_keys):
        raise ValueError("Server action_dim does not match action_keys")

    logger.info(
        "Ready: task=%r state=%d action=%d cameras=%s chunk=%d execute=%s",
        task,
        len(state_keys),
        len(action_keys),
        sorted(cameras),
        actions_per_chunk,
        args.execute,
    )
    period = 1.0 / args.control_fps
    sequence_id = 0
    robot.connect()
    try:
        for camera in cameras.values():
            camera.connect()
        while args.steps == 0 or sequence_id < args.steps:
            request_started = time.perf_counter()
            raw_state = robot.get_observation()
            camera_frames = {name: camera.read_latest() for name, camera in cameras.items()}
            payload = {
                "protocol_version": 1,
                "sequence_id": sequence_id,
                "task": task,
                "actions_per_chunk": actions_per_chunk,
                "observation": build_wire_observation(
                    raw_state,
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
                raise ValueError(f"Invalid action array {actions.shape}; expected {expected_shape}")
            logger.info(
                "seq=%d round_trip=%.1fms server=%.1fms actions=%s",
                sequence_id,
                (time.perf_counter() - request_started) * 1000,
                float(result.get("server_ms", -1)),
                actions.shape,
            )
            for row in actions:
                step_started = time.perf_counter()
                action = dict(zip(action_keys, map(float, row), strict=True))
                if args.execute:
                    robot.send_action(action)
                else:
                    logger.info("action=%s", np.array2string(row, precision=4, suppress_small=True))
                remaining = period - (time.perf_counter() - step_started)
                if remaining > 0:
                    time.sleep(remaining)
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
