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

"""CRP arm HTTP ACT executor without RTC.

This variant keeps ACT execution strictly sequential:

1. Read the latest robot observation
2. Send one HTTP /predict request
3. Receive one complete action chunk
4. Execute the chunk row-by-row in order
5. Repeat

ACT itself does not perform execution-side RTC in this script. The server simply
returns action chunks and the client executes them sequentially.
"""

import argparse
import base64
import json
import logging
import os
import time
from io import BytesIO
from typing import Any

import numpy as np
import requests
import torch
from PIL import Image

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots import make_robot_from_config
from lerobot.robots.crp_arm.config_crp_arm import CRPArmConfig

logger = logging.getLogger(__name__)

CRP_JOINT_KEYS = tuple(f"j{i}.pos" for i in range(1, 7))
CRP_ACTION_KEYS = (*CRP_JOINT_KEYS, "gripper.pos")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CRP arm HTTP ACT executor without RTC; execute returned chunks in order.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--server",
        default=os.environ.get("EAS_ENDPOINT", "http://127.0.0.1:8000"),
        help="ACT base URL; defaults to EAS_ENDPOINT when set.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Authorization token; defaults to EAS_TOKEN without exposing it in --help.",
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
        help="Total control steps to execute; 0 runs continuously until Ctrl+C.",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--print-action-chunks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print every complete action chunk returned by the server.",
    )
    parser.add_argument("--actions_per_chunk", type=int, default=0, help="0 uses server metadata.")
    parser.add_argument("--request_timeout", type=float, default=15.0)
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=70,
        help="JPEG quality for the two camera payloads (lower if the gateway returns HTTP 413).",
    )
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
        raise ValueError(f"ACT checkpoint must contain visual feature {key!r}")
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
                raise ValueError(f"CRP state is 6-D but ACT feature {key} has shape {shape}")
            observation[key] = state.reshape(shape)
        else:
            raise ValueError(f"CRP ACT executor cannot provide feature {key!r} of type {spec['type']}")
    return observation


def action_to_crp(action: torch.Tensor) -> dict[str, float]:
    values = action.detach().cpu().flatten().tolist()
    if len(values) != len(CRP_ACTION_KEYS):
        raise ValueError(f"CRP ACT action must be 7-D (6 joints + gripper), got {len(values)}")
    return {key: float(value) for key, value in zip(CRP_ACTION_KEYS, values, strict=True)}


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
        raise ValueError(f"Unexpected ACT health response: {health_response.text[:500]}")

    response = session.get(f"{base_url.rstrip('/')}/metadata", timeout=timeout)
    require_success(response)
    metadata = response.json()
    if metadata.get("protocol_version") != 1 or metadata.get("policy") != "act":
        raise ValueError(f"Unsupported server metadata: {metadata}")
    return metadata


def request_action_chunk(
    session: requests.Session,
    base_url: str,
    observation: dict[str, np.ndarray],
    *,
    sequence_id: int,
    actions_per_chunk: int,
    jpeg_quality: int,
    timeout: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    encode_start_ns = time.perf_counter_ns()
    encoded_observation = {
        key: encode_jpeg(value, jpeg_quality) if value.ndim == 3 else value.tolist()
        for key, value in observation.items()
    }
    encode_done_ns = time.perf_counter_ns()
    payload = {
        "protocol_version": 1,
        "sequence_id": sequence_id,
        "captured_at_ns": time.time_ns(),
        "actions_per_chunk": actions_per_chunk,
        "observation": encoded_observation,
    }
    request_body = json.dumps(payload, separators=(",", ":")).encode()
    serialized_ns = time.perf_counter_ns()

    start_ns = time.perf_counter_ns()
    response = session.post(
        f"{base_url.rstrip('/')}/predict",
        data=request_body,
        timeout=timeout,
        stream=True,
    )
    headers_received_ns = time.perf_counter_ns()
    response_body = response.content
    body_received_ns = time.perf_counter_ns()
    require_success(response)
    result = response.json()
    if result["sequence_id"] != sequence_id:
        raise ValueError(
            f"Response sequence mismatch: sent={sequence_id}, got={result['sequence_id']}"
        )
    actions = torch.tensor(result["actions"], dtype=torch.float32)
    if actions.ndim != 2 or actions.shape[1] != result["action_dim"]:
        raise ValueError(f"Invalid action chunk shape {tuple(actions.shape)}")

    parsed_ns = time.perf_counter_ns()
    encode_ms = (encode_done_ns - encode_start_ns) / 1e6
    serialize_ms = (serialized_ns - encode_done_ns) / 1e6
    upload_to_first_byte_ms = (headers_received_ns - start_ns) / 1e6
    download_ms = (body_received_ns - headers_received_ns) / 1e6
    parse_ms = (parsed_ns - body_received_ns) / 1e6
    total_ms = (parsed_ns - encode_start_ns) / 1e6
    server_ms = float(result["server_ms"])
    logger.info(
        "ACT_CHUNK_TRANSFER seq=%d chunk=%s upload_bytes=%d download_bytes=%d "
        "encode_ms=%.2f serialize_ms=%.2f upload_to_first_byte_ms=%.2f server_ms=%.2f "
        "transport_est_ms=%.2f download_ms=%.2f parse_ms=%.2f total_ms=%.2f",
        sequence_id,
        tuple(actions.shape),
        len(request_body),
        len(response_body),
        encode_ms,
        serialize_ms,
        upload_to_first_byte_ms,
        server_ms,
        upload_to_first_byte_ms - server_ms,
        download_ms,
        parse_ms,
        total_ms,
    )
    return actions, result


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
    logger.info(
        "ACT server ready: endpoint=%s action_dim=%s chunk=%s/%s",
        args.server,
        metadata["action_dim"],
        metadata["actions_per_chunk"],
        metadata["chunk_size"],
    )

    if metadata["mock"] and not args.allow_mock_server:
        raise ValueError(
            "Server reports mock=true; refusing to send mock actions to a real CRP arm. "
            "Use a real --policy_path, or pass --allow_mock_server only for a controlled bench test."
        )
    if metadata["action_dim"] != 7:
        raise ValueError(
            f"CRPArmConfig(use_gripper_feature=True) requires a 7-D ACT action, got {metadata['action_dim']}"
        )

    actions_per_chunk = args.actions_per_chunk or metadata["actions_per_chunk"]
    if not 1 <= actions_per_chunk <= metadata["chunk_size"]:
        raise ValueError(f"actions_per_chunk must be in [1, {metadata['chunk_size']}]")

    robot = make_crp_robot(args, metadata["input_features"])
    period = 1.0 / args.fps
    completed_steps = 0
    sequence_id = 0

    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    if args.token:
        session.headers["Authorization"] = args.token

    if not args.skip_confirm:
        print(
            "\n=== ACT CRP deploy (no RTC) ===\n"
            f"  server:       {args.server}\n"
            f"  robot address:{args.robot_address}\n"
            f"  cameras:      top={args.top_camera}, wrist={args.wrist_camera}\n"
            f"  fps:          {args.fps}\n"
            f"  speed ratio:  {args.speed_ratio if args.speed_ratio is not None else 'unchanged'}\n"
            f"  jpeg quality:{args.jpeg_quality}\n"
            f"  chunk size:   {actions_per_chunk}\n"
            f"  steps:        {args.steps if args.steps else 'continuous until Ctrl+C'}\n"
            "ACT here is plain chunk execution: upload observation, download one chunk, execute it in order.\n"
            "Ensure the workspace is clear and the e-stop is reachable.\n"
        )
        input("Press Enter to connect and power the CRP arm (Ctrl+C to abort)... ")

    try:
        robot.connect()
        while args.steps == 0 or completed_steps < args.steps:
            chunk_start_ns = time.perf_counter_ns()
            raw = robot.get_observation()
            captured_ns = time.perf_counter_ns()
            observation = build_crp_observation(raw, metadata["input_features"])
            observation_ready_ns = time.perf_counter_ns()
            actions, result = request_action_chunk(
                session,
                args.server,
                observation,
                sequence_id=sequence_id,
                actions_per_chunk=actions_per_chunk,
                jpeg_quality=args.jpeg_quality,
                timeout=args.request_timeout,
            )
            actions_ready_ns = time.perf_counter_ns()
            sequence_id += 1

            if args.print_action_chunks:
                logger.info(
                    "ACTION_CHUNK seq=%d columns=%s raw_shape=%s\n%s",
                    result["sequence_id"],
                    CRP_ACTION_KEYS,
                    tuple(actions.shape),
                    np.array2string(
                        actions.cpu().numpy(),
                        precision=5,
                        suppress_small=False,
                        threshold=np.inf,
                        max_line_width=200,
                    ),
                )

            first_action_ns = None
            execution_start_ns = None
            executed_rows = 0
            send_total_ms = 0.0
            send_max_ms = 0.0
            max_step_overrun_ms = 0.0
            for row_index, action in enumerate(actions):
                if args.steps and completed_steps >= args.steps:
                    break
                tick_start_ns = time.perf_counter_ns()
                if first_action_ns is None:
                    first_action_ns = tick_start_ns
                    execution_start_ns = tick_start_ns
                robot.send_action(action_to_crp(action))
                send_done_ns = time.perf_counter_ns()
                send_ms = (send_done_ns - tick_start_ns) / 1e6
                send_total_ms += send_ms
                send_max_ms = max(send_max_ms, send_ms)
                executed_rows += 1
                completed_steps += 1

                if completed_steps <= 5 or completed_steps % max(1, round(args.fps)) == 0:
                    logger.info(
                        "step=%d chunk_seq=%d row=%d/%d action_head=%s",
                        completed_steps - 1,
                        result["sequence_id"],
                        row_index,
                        len(actions) - 1,
                        np.round(action[:3].cpu().numpy(), 4).tolist(),
                    )

                tick_elapsed = (time.perf_counter_ns() - tick_start_ns) / 1e9
                max_step_overrun_ms = max(max_step_overrun_ms, (tick_elapsed - period) * 1e3)
                time.sleep(max(0.0, period - tick_elapsed))

            execution_done_ns = time.perf_counter_ns()
            if first_action_ns is not None and execution_start_ns is not None:
                logger.info(
                    "ACT_CHUNK_LATENCY seq=%d rows=%d observation_capture_ms=%.2f "
                    "observation_build_ms=%.2f observation_to_actions_ready_ms=%.2f "
                    "actions_ready_to_first_send_ms=%.2f execution_ms=%.2f expected_execution_ms=%.2f "
                    "send_action_avg_ms=%.2f send_action_max_ms=%.2f max_step_overrun_ms=%.2f",
                    result["sequence_id"],
                    executed_rows,
                    (captured_ns - chunk_start_ns) / 1e6,
                    (observation_ready_ns - captured_ns) / 1e6,
                    (actions_ready_ns - chunk_start_ns) / 1e6,
                    (first_action_ns - actions_ready_ns) / 1e6,
                    (execution_done_ns - execution_start_ns) / 1e6,
                    executed_rows * period * 1e3,
                    send_total_ms / executed_rows,
                    send_max_ms,
                    max(0.0, max_step_overrun_ms),
                )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        session.close()
        if robot.is_connected:
            robot.disconnect()

    logger.info("CRP execution finished without RTC: steps=%d", completed_steps)


if __name__ == "__main__":
    main()
