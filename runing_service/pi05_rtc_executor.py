#!/usr/bin/env python3

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CRP arm HTTP PI05 executor with execution-side Real-Time Chunking.

Pairs with ``pi05_http_server.py`` / ``pi05_local_http_server.py``. The control
loop runs at a fixed ``--fps`` and never blocks on HTTP. A background thread
requests action chunks; LeRobot ``ActionQueue`` + ``RTCConfig`` replace stale
tails and skip actions that elapsed while the request was in flight.

This is execution-side RTC only (same pattern as ``examples/act_rtc_executor.py``).
The PI05 HTTP server currently returns plain chunks and does not apply
model-side prefix conditioning.

Local example (pi0 conda)::

    # terminal 1
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python pi05_local_http_server.py \\
        --policy_path .../pretrained_model --task "..." --device cuda

    # terminal 2
    python pi05_rtc_executor.py \\
        --task "..." \\
        --robot_address 192.168.0.100 \\
        --top_camera 6 --wrist_camera 14 \\
        --fps 30 --steps 0
"""

from __future__ import annotations

import argparse
import base64
import logging
import os
import threading
import time
from io import BytesIO
from typing import Any

import numpy as np
import requests
import torch
from PIL import Image

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.policies.rtc import ActionQueue, RTCConfig
from lerobot.robots import make_robot_from_config
from lerobot.robots.crp_arm.config_crp_arm import CRPArmConfig

logger = logging.getLogger(__name__)

CRP_JOINT_KEYS = tuple(f"j{i}.pos" for i in range(1, 7))
CRP_ACTION_KEYS = (*CRP_JOINT_KEYS, "gripper.pos")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CRP arm HTTP PI05 executor using LeRobot's RTC action queue.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--server",
        default=os.environ.get("EAS_ENDPOINT", "http://127.0.0.1:8002"),
        help="PI05 base URL; defaults to EAS_ENDPOINT when set.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Authorization token; defaults to EAS_TOKEN without exposing it in --help.",
    )
    parser.add_argument(
        "--task",
        default="",
        help="Language instruction. Falls back to server metadata default_task when empty.",
    )
    parser.add_argument(
        "--robot_address",
        "--robot_port",
        dest="robot_address",
        default="192.168.0.100",
        help="CRP controller IPv4 address; --robot_port remains as a compatibility alias.",
    )
    parser.add_argument("--top_camera", type=int, default=6)
    parser.add_argument("--wrist_camera", type=int, default=14)
    parser.add_argument(
        "--speed_ratio",
        type=int,
        default=None,
        help="Set CRP speed ratio on connect; omitted keeps the controller's current setting.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=180,
        help="Control steps to execute; 0 runs continuously until Ctrl+C.",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--execution_horizon", type=int, default=10)
    parser.add_argument(
        "--rtc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replace stale action tails with RTC; --no-rtc appends complete chunks instead.",
    )
    parser.add_argument(
        "--print-action-chunks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print every complete action chunk returned by the server.",
    )
    parser.add_argument("--actions_per_chunk", type=int, default=0, help="0 uses server metadata.")
    parser.add_argument(
        "--request_timeout",
        type=float,
        default=60.0,
        help="HTTP timeout; PI05 inference is heavier than ACT.",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=70,
        help="JPEG quality for the two camera payloads (lower if the gateway returns HTTP 413).",
    )
    parser.add_argument("--max_consecutive_misses", type=int, default=10)
    parser.add_argument(
        "--allow_mock_server",
        action="store_true",
        help="Allow mock actions to reach the real arm. Unsafe except for a controlled bench test.",
    )
    parser.add_argument("--skip_confirm", action="store_true")
    args = parser.parse_args()
    if args.token is None:
        args.token = os.environ.get("EAS_TOKEN", "")
    return args


def encode_jpeg(image: np.ndarray, quality: int = 85) -> dict[str, str]:
    output = BytesIO()
    Image.fromarray(image).save(output, format="JPEG", quality=quality)
    return {"encoding": "jpeg", "data": base64.b64encode(output.getvalue()).decode()}


def to_hwc_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    return np.ascontiguousarray(image)


def crp_camera_size(input_features: dict[str, dict[str, Any]], name: str) -> tuple[int, int]:
    key = f"observation.images.{name}"
    spec = input_features.get(key)
    if spec is None or spec["type"] != "VISUAL":
        raise ValueError(f"PI05 checkpoint must contain visual feature {key!r}")
    shape = tuple(spec["shape"])
    if len(shape) != 3 or shape[0] != 3:
        raise ValueError(f"Expected RGB CHW feature for {key}, got {shape}")
    return shape[2], shape[1]


def make_crp_robot(args: argparse.Namespace, input_features: dict[str, dict[str, Any]]):
    top_width, top_height = crp_camera_size(input_features, "top")
    wrist_width, wrist_height = crp_camera_size(input_features, "wrist")
    config = CRPArmConfig(
        port=args.robot_address,
        use_gripper_feature=True,
        speed_ratio_on_connect=args.speed_ratio,
        cameras={
            "top": OpenCVCameraConfig(
                index_or_path=args.top_camera,
                width=top_width,
                height=top_height,
                fps=round(args.fps),
            ),
            "wrist": OpenCVCameraConfig(
                index_or_path=args.wrist_camera,
                width=wrist_width,
                height=wrist_height,
                fps=round(args.fps),
            ),
        },
    )
    return make_robot_from_config(config)


def build_crp_observation(
    raw: dict[str, Any], input_features: dict[str, dict[str, Any]]
) -> dict[str, np.ndarray]:
    state = np.asarray([float(raw[key]) for key in CRP_JOINT_KEYS], dtype=np.float32)
    observation: dict[str, np.ndarray] = {}
    for key, spec in input_features.items():
        shape = tuple(spec["shape"])
        if spec["type"] == "VISUAL":
            camera_name = key.rsplit(".", 1)[-1]
            if camera_name not in raw:
                raise KeyError(f"CRP observation has no camera {camera_name!r}; keys={sorted(raw)}")
            image = to_hwc_uint8(np.asarray(raw[camera_name]))
            expected = (shape[1], shape[2], shape[0])
            if image.shape != expected:
                raise ValueError(f"Camera {camera_name} expected {expected}, got {image.shape}")
            observation[key] = image
        elif spec["type"] == "STATE":
            if int(np.prod(shape)) != len(state):
                raise ValueError(f"CRP state is 6-D but PI05 feature {key} has shape {shape}")
            observation[key] = state.reshape(shape)
        else:
            raise ValueError(f"CRP PI05 executor cannot provide feature {key!r} of type {spec['type']}")
    return observation


def action_to_crp(action: torch.Tensor) -> dict[str, float]:
    values = action.detach().cpu().flatten().tolist()
    if len(values) != len(CRP_ACTION_KEYS):
        raise ValueError(f"CRP PI05 action must be 7-D (6 joints + gripper), got {len(values)}")
    return {key: float(value) for key, value in zip(CRP_ACTION_KEYS, values, strict=True)}


def hold_action_from_raw(raw: dict[str, Any]) -> torch.Tensor:
    values = [float(raw[key]) for key in CRP_ACTION_KEYS]
    return torch.tensor(values, dtype=torch.float32)


def require_success(response: requests.Response) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = response.text.strip().replace("\n", " ")[:500]
        raise RuntimeError(
            f"{response.request.method} {response.url} returned HTTP {response.status_code}: {body}"
        ) from exc


def get_metadata(session: requests.Session, base_url: str, timeout: float) -> dict[str, Any]:
    health_response = session.get(f"{base_url.rstrip('/')}/health", timeout=timeout)
    require_success(health_response)
    if health_response.json().get("status") != "ok":
        raise ValueError(f"Unexpected PI05 health response: {health_response.text[:500]}")

    response = session.get(f"{base_url.rstrip('/')}/metadata", timeout=timeout)
    require_success(response)
    metadata = response.json()
    if metadata.get("protocol_version") != 1 or metadata.get("policy") != "pi05":
        raise ValueError(f"Unsupported server metadata: {metadata}")
    return metadata


def resolve_task(args_task: str, metadata: dict[str, Any]) -> str:
    task = (args_task or "").strip() or str(metadata.get("default_task", "")).strip()
    if not task:
        raise ValueError(
            "PI05 requires a non-empty language task. Pass --task on the executor "
            "or start the server with --task."
        )
    return task


class RemotePI05RTC:
    """Background HTTP inference + execution-side RTC queue for PI05 chunks."""

    def __init__(
        self,
        *,
        server: str,
        token: str,
        task: str,
        fps: float,
        execution_horizon: int,
        actions_per_chunk: int,
        request_timeout: float,
        jpeg_quality: int,
        rtc_enabled: bool,
        print_action_chunks: bool,
    ) -> None:
        if fps <= 0 or execution_horizon <= 0:
            raise ValueError("fps and execution_horizon must be positive")
        if request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        if not 1 <= jpeg_quality <= 95:
            raise ValueError("jpeg_quality must be in [1, 95]")
        if not task.strip():
            raise ValueError("task must be non-empty")

        self.base_url = server.rstrip("/")
        self.task = task.strip()
        self.fps = fps
        self.execution_horizon = execution_horizon
        self.actions_per_chunk = actions_per_chunk
        self.request_timeout = request_timeout
        self.jpeg_quality = jpeg_quality
        self.rtc_enabled = rtc_enabled
        self.print_action_chunks = print_action_chunks
        self.queue = ActionQueue(RTCConfig(enabled=rtc_enabled, execution_horizon=execution_horizon))
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        if token:
            self.session.headers["Authorization"] = token

        self._observation: dict[str, np.ndarray] | None = None
        self._observation_lock = threading.Lock()
        self._observation_ready = threading.Event()
        self._first_chunk_ready = threading.Event()
        self._request_in_flight = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sequence_id = 0
        self.error: Exception | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._inference_loop, daemon=True, name="RemotePI05RTC")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._observation_ready.set()
        if self._thread is not None:
            self._thread.join(timeout=self.request_timeout + 1.0)
        self.session.close()

    def publish(self, observation: dict[str, np.ndarray]) -> None:
        with self._observation_lock:
            self._observation = {key: value.copy() for key, value in observation.items()}
        self._observation_ready.set()

    def wait_until_ready(self, timeout: float) -> None:
        if not self._first_chunk_ready.wait(timeout):
            if self.error is not None:
                raise RuntimeError("Initial PI05 request failed") from self.error
            raise TimeoutError(f"No initial PI05 chunk received within {timeout:.1f}s")

    def get_action(self) -> torch.Tensor | None:
        return self.queue.get()

    def needs_observation(self) -> bool:
        return (
            self.queue.qsize() <= self.execution_horizon
            and not self._observation_ready.is_set()
            and not self._request_in_flight.is_set()
        )

    def _snapshot(self) -> dict[str, np.ndarray] | None:
        with self._observation_lock:
            if self._observation is None:
                return None
            return {key: value.copy() for key, value in self._observation.items()}

    def _inference_loop(self) -> None:
        first_request = True
        while not self._stop.is_set():
            if self.queue.qsize() > self.execution_horizon:
                self._stop.wait(0.005)
                continue
            if not self._observation_ready.wait(0.1):
                continue
            self._observation_ready.clear()
            observation = self._snapshot()
            if observation is None:
                continue

            sequence_id = self._sequence_id
            self._sequence_id += 1
            index_before = self.queue.get_action_index()
            payload = {
                "protocol_version": 1,
                "sequence_id": sequence_id,
                "captured_at_ns": time.time_ns(),
                "actions_per_chunk": self.actions_per_chunk,
                "task": self.task,
                "observation": {
                    key: encode_jpeg(value, self.jpeg_quality) if value.ndim == 3 else value.tolist()
                    for key, value in observation.items()
                },
            }

            self._request_in_flight.set()
            try:
                start = time.perf_counter()
                response = self.session.post(
                    f"{self.base_url}/predict", json=payload, timeout=self.request_timeout
                )
                require_success(response)
                result = response.json()
                if result["sequence_id"] != sequence_id:
                    raise ValueError(
                        f"Response sequence mismatch: sent={sequence_id}, got={result['sequence_id']}"
                    )
                actions = torch.tensor(result["actions"], dtype=torch.float32)
                if actions.ndim != 2 or actions.shape[1] != result["action_dim"]:
                    raise ValueError(f"Invalid action chunk shape {tuple(actions.shape)}")

                round_trip_s = time.perf_counter() - start
                # Skip actions consumed by the control loop while this request was in flight.
                consumed_steps = max(0, self.queue.get_action_index() - index_before)
                delay_steps = 0 if first_request else consumed_steps
                self.queue.merge(actions, actions, delay_steps, index_before)
                first_request = False
                self.error = None
                self._first_chunk_ready.set()
                logger.info(
                    "seq=%d server_ms=%.1f round_trip_ms=%.1f delay_steps=%d queue=%d "
                    "task=%r predicted_subtask=%r",
                    sequence_id,
                    result["server_ms"],
                    round_trip_s * 1000.0,
                    delay_steps,
                    self.queue.qsize(),
                    result.get("task", self.task),
                    result.get("predicted_subtask"),
                )
                if self.print_action_chunks:
                    logger.info(
                        "ACTION_CHUNK seq=%d columns=%s raw_shape=%s execute_from_row=%d\n%s",
                        sequence_id,
                        CRP_ACTION_KEYS,
                        tuple(actions.shape),
                        delay_steps if self.rtc_enabled else 0,
                        np.array2string(
                            actions.cpu().numpy(),
                            precision=5,
                            suppress_small=False,
                            threshold=np.inf,
                            max_line_width=200,
                        ),
                    )
            except Exception as exc:
                self.error = exc
                logger.warning("PI05 request failed: %s", exc)
                stopped = self._stop.wait(0.25)
                if first_request and not stopped:
                    # Retry the latest observation after a transient failure before the loop starts.
                    self._observation_ready.set()
            finally:
                self._request_in_flight.clear()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    args = parse_args()

    if not args.server.startswith(("http://", "https://")):
        raise ValueError("--server/EAS_ENDPOINT must start with http:// or https://")
    if "/api/predict/" in args.server and not args.token:
        raise ValueError("EAS endpoint detected but EAS_TOKEN/--token is empty")
    if args.steps < 0:
        raise ValueError("--steps must be >= 0 (0 means run until Ctrl+C)")

    with requests.Session() as probe:
        probe.headers["Accept"] = "application/json"
        if args.token:
            probe.headers["Authorization"] = args.token
        metadata = get_metadata(probe, args.server, args.request_timeout)
    task = resolve_task(args.task, metadata)
    logger.info(
        "PI05 server ready: endpoint=%s action_dim=%s chunk=%s/%s task=%r",
        args.server,
        metadata["action_dim"],
        metadata["actions_per_chunk"],
        metadata["chunk_size"],
        task,
    )

    if metadata["mock"] and not args.allow_mock_server:
        raise ValueError(
            "Server reports mock=true; refusing to send mock actions to a real CRP arm. "
            "Use a real --policy_path, or pass --allow_mock_server only for a controlled bench test."
        )
    if metadata["action_dim"] != 7:
        raise ValueError(
            f"CRPArmConfig(use_gripper_feature=True) requires a 7-D PI05 action, got {metadata['action_dim']}"
        )

    actions_per_chunk = args.actions_per_chunk or metadata["actions_per_chunk"]
    if not 1 <= actions_per_chunk <= metadata["chunk_size"]:
        raise ValueError(f"actions_per_chunk must be in [1, {metadata['chunk_size']}]")
    if args.execution_horizon >= actions_per_chunk:
        raise ValueError("execution_horizon must be smaller than actions_per_chunk")

    robot = make_crp_robot(args, metadata["input_features"])
    rtc = RemotePI05RTC(
        server=args.server,
        token=args.token,
        task=task,
        fps=args.fps,
        execution_horizon=args.execution_horizon,
        actions_per_chunk=actions_per_chunk,
        request_timeout=args.request_timeout,
        jpeg_quality=args.jpeg_quality,
        rtc_enabled=args.rtc,
        print_action_chunks=args.print_action_chunks,
    )
    misses = 0
    consecutive_misses = 0
    last_action: torch.Tensor | None = None
    period = 1.0 / args.fps
    rtc_started = False
    completed_steps = 0

    if not args.skip_confirm:
        print(
            "\n=== PI05 RTC CRP deploy ===\n"
            f"  server:       {args.server}\n"
            f"  task:         {task}\n"
            f"  robot address:{args.robot_address}\n"
            f"  cameras:      top={args.top_camera}, wrist={args.wrist_camera}\n"
            f"  fps:          {args.fps}\n"
            f"  speed ratio:  {args.speed_ratio if args.speed_ratio is not None else 'unchanged'}\n"
            f"  jpeg quality:{args.jpeg_quality}\n"
            f"  chunk/horizon:{actions_per_chunk}/{args.execution_horizon}\n"
            f"  RTC:          {'enabled' if args.rtc else 'disabled (append chunks)'}\n"
            f"  print chunks: {args.print_action_chunks}\n"
            f"  steps:        {args.steps if args.steps else 'continuous until Ctrl+C'}\n"
            "Fixed-rate control loop + background HTTP; ActionQueue skips in-flight latency.\n"
            "Ensure the workspace is clear and the e-stop is reachable.\n"
        )
        input("Press Enter to connect and power the CRP arm (Ctrl+C to abort)... ")

    try:
        robot.connect()
        raw = robot.get_observation()
        last_action = hold_action_from_raw(raw)
        rtc.publish(build_crp_observation(raw, metadata["input_features"]))
        rtc.start()
        rtc_started = True
        rtc.wait_until_ready(args.request_timeout * 2.0 + 1.0)

        next_tick = time.perf_counter()
        step = 0
        while args.steps == 0 or step < args.steps:
            next_tick += period
            action = rtc.get_action()
            if action is None:
                misses += 1
                consecutive_misses += 1
                action = last_action
            else:
                last_action = action
                consecutive_misses = 0
            if consecutive_misses > args.max_consecutive_misses:
                raise RuntimeError(
                    f"RTC queue empty for {consecutive_misses} consecutive ticks; stopping the arm"
                )
            robot.send_action(action_to_crp(action))

            if rtc.needs_observation():
                raw = robot.get_observation()
                rtc.publish(build_crp_observation(raw, metadata["input_features"]))

            if step < 5 or (step + 1) % max(1, round(args.fps)) == 0:
                logger.info(
                    "step=%d queue=%d action_head=%s misses=%d",
                    step,
                    rtc.queue.qsize(),
                    np.round(last_action[:3].cpu().numpy(), 4).tolist(),
                    misses,
                )
            step += 1
            completed_steps = step
            time.sleep(max(0.0, next_tick - time.perf_counter()))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        try:
            if robot.is_connected:
                robot.disconnect()
        finally:
            if rtc_started:
                rtc.stop()

    if rtc.error is not None:
        raise RuntimeError("RTC inference thread stopped with an error") from rtc.error
    logger.info(
        "CRP PI05 RTC execution finished: steps=%d misses=%d final_queue=%d",
        completed_steps,
        misses,
        rtc.queue.qsize(),
    )


if __name__ == "__main__":
    main()
