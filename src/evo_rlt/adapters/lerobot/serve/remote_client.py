"""Local-side client for a cloud-hosted online-RL service
(`runing_service/rlt_ac/online_serve.py`). Used by `backend.record()` when
`online_rl.remote_server` is set (wired via `evo-rlt-online-train
--remote-server`).

`RemoteOnlineRLSession` is a single object that duck-types as THREE different
roles inside `backend.record()`/`record_loop()`, because remote mode assigns
the same instance to all three local variables (see backend.py):

  * `policy`           -- record_loop() calls `_predict_policy_action_with_
                           acp_inference(policy=policy, ...)`, which (see
                           hil.py's remote branch) calls this object's
                           `predict_from_observation_frame()` instead of
                           running a real forward pass. Also needs
                           `.set_rl_mode()`/`.reset()`/`.get_last_chunk_
                           tensors()`/`.pop_step_metadata()` -- these are
                           exactly the methods a real `ChunkACPolicy`
                           exposes for the same purposes (see
                           modeling_rlt_ac.py), so `loop.py`'s
                           `rlt = policy` / `hasattr(policy, "set_rl_mode")`
                           duck-typing picks this object up transparently.
  * `online_collector` -- record_loop() calls `.start_episode()`/`.on_frame()`/
                           `.flush_episode()`, exactly like a real
                           `RLTOnlineCollector`.
  * `online_trainer`   -- backend.record()'s per-episode loop calls
                           `.start_episode()`/`.maybe_update()`/
                           `.save_latest_state()`, exactly like a real
                           `OnlineRLTrainer`.

None of these local calls ever touch real tensors/gradients -- they all
become HTTP requests to the service, which holds the actual model, replay
buffer, and optimizers (built around the SAME `OnlineRLTrainer` class
`backend.record()` uses locally -- see online_trainer.py -- so the training
math has exactly one implementation, not a copy per deployment mode). This
machine never loads a policy or needs a GPU in remote mode.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from evo_rlt.adapters.lerobot.serve.codec import decode_tensor, encode_observation_frame, encode_tensor

logger = logging.getLogger(__name__)


@dataclass
class _RemoteStepMetadata:
    phase: float
    source_type: float


class RemoteOnlineRLSession:
    def __init__(
        self,
        *,
        base_url: str,
        auth_token: str | None = None,
        timeout_s: float = 30.0,
        jpeg_quality: int = 90,
    ):
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self.timeout_s = timeout_s
        self.jpeg_quality = jpeg_quality
        # Only `.device`/`.use_amp` are ever read locally (loop.py, to build a
        # torch.device passed into _predict_policy_action_with_acp_inference)
        # -- the remote branch there (see hil.py) returns before touching
        # either, so these values are never actually used for compute.
        self.config = SimpleNamespace(device="cpu", use_amp=False)
        self._last_chunk_tensors: tuple[torch.Tensor, torch.Tensor] | None = None
        self._last_step_metadata: _RemoteStepMetadata | None = None

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------
    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Remote online-RL service error on {path}: {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Cannot reach remote online-RL service at {url}: {exc}") from exc

    def get_metadata(self) -> dict[str, Any]:
        request = urllib.request.Request(f"{self.base_url}/metadata", method="GET")
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return json.loads(response.read())

    def check_health(self) -> bool:
        request = urllib.request.Request(f"{self.base_url}/health", method="GET")
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return json.loads(response.read()).get("status") == "ok"

    # ------------------------------------------------------------------
    # `policy` role
    # ------------------------------------------------------------------
    def predict_from_observation_frame(
        self, observation_frame: dict[str, np.ndarray], *, task: str | None, robot_type: str | None
    ) -> torch.Tensor:
        payload = {
            "observation": encode_observation_frame(observation_frame, jpeg_quality=self.jpeg_quality),
            "task": task,
            "robot_type": robot_type,
        }
        response = self._post("/predict", payload)
        action = decode_tensor(response["action"])
        state_vec = decode_tensor(response.get("state_vec"))
        ref_chunk = decode_tensor(response.get("ref_chunk"))
        if state_vec is not None and ref_chunk is not None:
            self._last_chunk_tensors = (state_vec, ref_chunk)
        phase = response.get("phase")
        source_type = response.get("source_type")
        self._last_step_metadata = (
            _RemoteStepMetadata(phase=phase, source_type=source_type) if phase is not None else None
        )
        return action

    def set_rl_mode(self) -> None:
        self._post("/policy/set_rl_mode", {})

    def reset(self) -> None:
        self._post("/policy/reset", {})

    def get_last_chunk_tensors(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Peek (non-consuming), matching RLTActionModifier.get_last_chunk_tensors():
        the service bundles these into every /predict response (see
        online_serve.py's OnlineRLService.predict), so this is just the last
        value received -- no extra round trip."""
        return self._last_chunk_tensors

    def pop_step_metadata(self) -> _RemoteStepMetadata | None:
        metadata = self._last_step_metadata
        self._last_step_metadata = None
        return metadata

    # ------------------------------------------------------------------
    # `online_collector` role (RLTOnlineCollector-compatible)
    # ------------------------------------------------------------------
    def start_episode(self, episode_id: int) -> int:
        response = self._post("/episode/start", {"episode_id": episode_id})
        return int(response["buffer_total_added_before"])

    def on_frame(
        self,
        *,
        action: torch.Tensor,
        state_vec: torch.Tensor | None,
        ref_chunk: torch.Tensor | None,
        source_type: float,
        is_critical: float,
    ) -> None:
        # state_vec/ref_chunk aren't sent -- the service already cached them
        # from the matching /predict call (see get_last_chunk_tensors()'s
        # docstring); re-uploading tensors it already has would be wasted
        # bandwidth for no behavior change.
        self._post(
            "/episode/frame",
            {
                "action": encode_tensor(action),
                "source_type": float(source_type),
                "is_critical": float(is_critical),
            },
        )
        return None

    def flush_episode(self, episode_success: bool) -> None:
        self._post("/episode/flush", {"episode_success": bool(episode_success)})
        return None

    # ------------------------------------------------------------------
    # `online_trainer` role (OnlineRLTrainer-compatible)
    # ------------------------------------------------------------------
    def maybe_update(self, recorded_episodes: int, buffer_total_added_before: int) -> dict | None:
        response = self._post(
            "/episode/update",
            {
                "recorded_episodes": recorded_episodes,
                "buffer_total_added_before": buffer_total_added_before,
            },
        )
        return response.get("stats")

    def save_latest_state(self, completed_episodes: int) -> None:
        self._post("/episode/save_latest_state", {"completed_episodes": completed_episodes})
