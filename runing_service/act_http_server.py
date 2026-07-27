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

"""Minimal HTTP action-chunk server for ACT.

Mock mode is self-contained and mirrors the CRP LingBot observation contract::

    uv run python examples/act_http_server.py --mock --latency_ms=80

Load a real LeRobot ACT checkpoint by pointing at its ``pretrained_model`` dir::

    uv run python examples/act_http_server.py \
        --policy_path outputs/train/checkpoints/last/pretrained_model \
        --device cuda

The API is intentionally small: ``GET /health``, ``GET /metadata`` and
``POST /predict``. Images are JPEG/base64; state-like features are JSON arrays.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTPolicy
from lerobot.policies.utils import prepare_observation_for_inference

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HTTP action-chunk server for real or simulated ACT inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--policy_path", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--mock", action="store_true", help="Run deterministic inference without a checkpoint."
    )
    parser.add_argument("--chunk_size", type=int, default=50, help="Mock ACT action horizon.")
    parser.add_argument("--latency_ms", type=float, default=0.0, help="Artificial latency for RTC testing.")
    parser.add_argument("--max_body_mb", type=float, default=8.0)
    return parser.parse_args()


def decode_observation(payload: dict[str, Any], features: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    wire_observation = payload.get("observation")
    if not isinstance(wire_observation, dict):
        raise ValueError("'observation' must be an object")

    unknown = set(wire_observation) - set(features)
    missing = set(features) - set(wire_observation)
    if unknown or missing:
        raise ValueError(f"Observation key mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}")

    observation: dict[str, np.ndarray] = {}
    for key, spec in features.items():
        value = wire_observation[key]
        if spec["type"] == "VISUAL":
            if not isinstance(value, dict) or value.get("encoding") != "jpeg":
                raise ValueError(f"{key} must be a JPEG object")
            raw = base64.b64decode(value["data"], validate=True)
            image = np.asarray(Image.open(BytesIO(raw)).convert("RGB"), dtype=np.uint8)
            expected = tuple(spec["shape"])
            if expected[0] != 3 or image.shape != (expected[1], expected[2], expected[0]):
                raise ValueError(
                    f"{key} shape mismatch: expected HWC={(expected[1], expected[2], expected[0])}, "
                    f"got {image.shape}"
                )
            observation[key] = image
        else:
            array = np.asarray(value, dtype=np.float32)
            expected = tuple(spec["shape"])
            if array.shape != expected:
                raise ValueError(f"{key} shape mismatch: expected {expected}, got {array.shape}")
            observation[key] = array
    return observation


class ACTService:
    def __init__(
        self,
        *,
        policy_path: Path | None,
        device: str,
        mock: bool,
        chunk_size: int,
        latency_ms: float,
    ) -> None:
        if mock == (policy_path is not None):
            raise ValueError("Choose exactly one of --mock or --policy_path")
        if chunk_size <= 0:
            raise ValueError("--chunk_size must be positive")

        self.device = torch.device(device)
        self.latency_s = max(0.0, latency_ms / 1000.0)
        self.lock = threading.Lock()
        self.mock = mock

        if mock:
            self.input_features = {
                "observation.images.top": {"type": "VISUAL", "shape": [3, 480, 640]},
                "observation.images.wrist": {"type": "VISUAL", "shape": [3, 480, 640]},
                "observation.state": {"type": "STATE", "shape": [6]},
            }
            self.action_dim = 7
            self.chunk_size = chunk_size
            self.actions_per_chunk = chunk_size
            self.policy = None
            self.preprocessor = None
            self.postprocessor = None
            return

        path = str(policy_path.resolve())
        self.policy = ACTPolicy.from_pretrained(path, local_files_only=True)
        self.policy.to(self.device)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=path,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        self.input_features = {
            key: {"type": feature.type.value, "shape": list(feature.shape)}
            for key, feature in self.policy.config.input_features.items()
        }
        action_features = list(self.policy.config.output_features.values())
        if len(action_features) != 1:
            raise ValueError(f"Expected one ACT output feature, got {len(action_features)}")
        self.action_dim = action_features[0].shape[-1]
        self.chunk_size = self.policy.config.chunk_size
        self.actions_per_chunk = self.policy.config.n_action_steps

    def metadata(self) -> dict[str, Any]:
        return {
            "protocol_version": 1,
            "policy": "act",
            "mock": self.mock,
            "input_features": self.input_features,
            "action_dim": self.action_dim,
            "chunk_size": self.chunk_size,
            "actions_per_chunk": self.actions_per_chunk,
        }

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("protocol_version") != 1:
            raise ValueError("Unsupported protocol_version; expected 1")
        sequence_id = int(payload["sequence_id"])
        observation = decode_observation(payload, self.input_features)
        requested = int(payload.get("actions_per_chunk", self.actions_per_chunk))
        if not 1 <= requested <= self.chunk_size:
            raise ValueError(f"actions_per_chunk must be in [1, {self.chunk_size}], got {requested}")

        start = time.perf_counter()
        with self.lock:
            if self.mock:
                actions = self._mock_actions(observation, sequence_id, requested)
            else:
                actions = self._act_actions(observation, requested)
            if self.latency_s:
                time.sleep(self.latency_s)

        return {
            "protocol_version": 1,
            "sequence_id": sequence_id,
            "actions": actions.tolist(),
            "action_dim": self.action_dim,
            "server_ms": (time.perf_counter() - start) * 1000.0,
        }

    def _mock_actions(
        self, observation: dict[str, np.ndarray], sequence_id: int, requested: int
    ) -> np.ndarray:
        state = observation["observation.state"]
        steps = np.arange(requested, dtype=np.float32)
        phase = sequence_id * 0.15 + steps * 0.04
        actions = np.zeros((requested, self.action_dim), dtype=np.float32)
        actions[:, : len(state)] = state + 0.03 * np.sin(phase)[:, None]
        actions[:, -1] = 0.5 + 0.2 * np.sin(phase)
        return actions

    def _act_actions(self, observation: dict[str, np.ndarray], requested: int) -> np.ndarray:
        with torch.inference_mode():
            batch = prepare_observation_for_inference(observation, self.device)
            batch = self.preprocessor(batch)
            chunk = self.policy.predict_action_chunk(batch)[:, :requested]
            processed = [self.postprocessor(chunk[:, i, :]) for i in range(chunk.shape[1])]
            return torch.stack(processed, dim=1).squeeze(0).detach().cpu().numpy()


class ACTRequestHandler(BaseHTTPRequestHandler):
    server: "ACTHTTPServer"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
        elif self.path == "/metadata":
            self._send_json(HTTPStatus.OK, self.server.service.metadata())
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/predict":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if not 0 < content_length <= self.server.max_body_bytes:
                raise ValueError(
                    f"Invalid request size {content_length}; max={self.server.max_body_bytes} bytes"
                )
            payload = json.loads(self.rfile.read(content_length))
            self._send_json(HTTPStatus.OK, self.server.service.predict(payload))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:
            logger.exception("Inference failed")
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        logger.info("%s - %s", self.client_address[0], format % args)


class ACTHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        address: tuple[str, int],
        service: ACTService,
        max_body_bytes: int,
    ) -> None:
        super().__init__(address, ACTRequestHandler)
        self.service = service
        self.max_body_bytes = max_body_bytes


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    service = ACTService(
        policy_path=args.policy_path,
        device=args.device,
        mock=args.mock,
        chunk_size=args.chunk_size,
        latency_ms=args.latency_ms,
    )
    server = ACTHTTPServer(
        (args.host, args.port),
        service,
        max_body_bytes=int(args.max_body_mb * 1024 * 1024),
    )
    logger.info("ACT HTTP server listening on http://%s:%d", args.host, args.port)
    logger.info("metadata=%s", service.metadata())
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
