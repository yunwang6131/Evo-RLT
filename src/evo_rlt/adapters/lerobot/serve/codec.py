"""Wire-format helpers for the online-RL client/server split.

Shared by `remote_client.py` (here, imported by `backend.record()`) and
`runing_service/rlt_ac/online_serve.py` (the cloud server, a standalone
script that imports this module directly).

Mirrors `runing_service/pi05_http_server.py`'s proven convention: image-typed
observation entries travel as base64 JPEG, everything else as JSON float
arrays. A key is treated as an image iff `"image"` appears in it, matching
lerobot's own `prepare_observation_for_inference` (same substring check).

Tensors (state_vec/ref_chunk/action) round-trip via nested Python lists
(`tensor.tolist()` / `torch.tensor(nested_list)`) rather than a fixed dtype/
shape contract -- this deliberately avoids needing to know the exact shape
lerobot's internal policy/actor tensors use; whatever shape went in comes
back out unchanged.
"""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

import numpy as np
import torch
from PIL import Image


def _is_image_key(key: str) -> bool:
    return "image" in key


def encode_observation_frame(
    observation_frame: dict[str, np.ndarray], *, jpeg_quality: int = 90
) -> dict[str, Any]:
    wire: dict[str, Any] = {}
    for key, value in observation_frame.items():
        array = np.asarray(value)
        if _is_image_key(key):
            image = Image.fromarray(array.astype(np.uint8), mode="RGB")
            buf = BytesIO()
            image.save(buf, format="JPEG", quality=jpeg_quality)
            wire[key] = {
                "encoding": "jpeg",
                "data": base64.b64encode(buf.getvalue()).decode("ascii"),
            }
        else:
            wire[key] = np.asarray(array, dtype=np.float32).tolist()
    return wire


def decode_observation_frame(wire: dict[str, Any]) -> dict[str, np.ndarray]:
    observation: dict[str, np.ndarray] = {}
    for key, value in wire.items():
        if _is_image_key(key):
            if not isinstance(value, dict) or value.get("encoding") != "jpeg":
                raise ValueError(f"observation key '{key}' must be a JPEG object")
            raw = base64.b64decode(value["data"], validate=True)
            observation[key] = np.asarray(Image.open(BytesIO(raw)).convert("RGB"), dtype=np.uint8)
        else:
            observation[key] = np.asarray(value, dtype=np.float32)
    return observation


def encode_tensor(tensor: torch.Tensor | None) -> list | None:
    if tensor is None:
        return None
    return tensor.detach().cpu().tolist()


def decode_tensor(data: list | None, *, dtype: torch.dtype = torch.float32) -> torch.Tensor | None:
    if data is None:
        return None
    return torch.tensor(data, dtype=dtype)
