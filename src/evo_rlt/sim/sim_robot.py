"""A LeRobot `Robot` backed by the simulator process.

Drops into the existing record / HIL / online-RL loop in place of
`BiSOFollower`: the loop only ever calls `get_observation`, `send_action`,
`connect`, `disconnect`, `is_connected` and the two feature properties, and the
surrounding tooling reads `left_arm_config` / `right_arm_config` to enumerate
cameras. All of that is reproduced here, so teleop, human takeover, RTC, RLT and
dataset recording keep working untouched.

The calibration conversion lives on this side of the process boundary (see
`calib.py`): the loop hands over LeRobot's calibrated motor values, this class
turns them into radians, and the simulator only ever sees joint angles. That
placement is deliberate -- it means swapping MuJoCo for Isaac Lab cannot
reintroduce a calibration bug, and the simulator needs no notion of a Feetech
servo.

Camera frames arrive as raw uint8 RGB in extra ZMQ parts and are reshaped
without a copy of the JSON path.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot
from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig

from evo_rlt.sim.arms import FOLLOWER_CALIBRATION_DIR, arm as _arm
from evo_rlt.sim.calib import ARM_SIDES, MOTOR_NAMES, BimanualCalibration
from evo_rlt.sim.protocol import (
    CAMERA_KEYS,
    DEFAULT_ENDPOINT,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_TIMEOUT_S,
    JOINT_ORDER,
    Command,
    ProtocolError,
    Status,
    check_version,
)

logger = logging.getLogger(__name__)

#: 本项目自己的 follower 标定,由 ``diagnostics/calibration.py`` 写入。
#: 刻意不用 ``~/.cache/huggingface/lerobot/`` —— 那是全机器共享的,别的项目
#: 也读也写,标定会互相覆盖或被误用。
DEFAULT_CALIBRATION_DIR = str(FOLLOWER_CALIBRATION_DIR)


@RobotConfig.register_subclass("sim_bi_so_follower")
@dataclass
class SimRobotConfig(RobotConfig):
    """Configuration for the simulated bimanual SO-101.

    ``left_arm_config`` / ``right_arm_config`` exist because the recording
    backend reads them to enumerate cameras; their ``port`` is unused and the
    camera configs only carry the resolution the simulator should render.
    """

    endpoint: str = DEFAULT_ENDPOINT
    timeout_s: float = DEFAULT_TIMEOUT_S
    width: int = DEFAULT_IMAGE_WIDTH
    height: int = DEFAULT_IMAGE_HEIGHT
    fps: int = 30

    #: Follower calibration used to map motor values onto joint angles. Must be
    #: the *same* files the real followers use, or sim and real poses diverge.
    calibration_source_dir: str = DEFAULT_CALIBRATION_DIR
    left_calibration_id: str = ""
    right_calibration_id: str = ""

    #: Per-arm, per-joint direction corrections, e.g. ``{"left": {"elbow_flex": -1.0}}``.
    #: Determined by comparing sim and real at a known pose; see
    #: ``diagnostics/check_sim_calib.py``.
    joint_signs: dict[str, dict[str, float]] = field(default_factory=dict)

    left_arm_config: SOFollowerConfig | None = None
    right_arm_config: SOFollowerConfig | None = None

    def __post_init__(self) -> None:
        # Cameras are named the way the real rig records them: the wrist cams
        # are per-arm, and the fixed front view is carried by the right arm
        # (BiSOFollower re-adds the prefix, so it is stored unprefixed).
        per_arm = {
            "left": {"wrist": self._camera()},
            "right": {"wrist": self._camera(), "front": self._camera()},
        }
        if self.left_arm_config is None:
            self.left_arm_config = SOFollowerConfig(
                port="sim://left", use_degrees=True, cameras=per_arm["left"]
            )
        if self.right_arm_config is None:
            self.right_arm_config = SOFollowerConfig(
                port="sim://right", use_degrees=True, cameras=per_arm["right"]
            )
        super().__post_init__()

    def _camera(self) -> OpenCVCameraConfig:
        return OpenCVCameraConfig(
            index_or_path=-1, width=self.width, height=self.height, fps=self.fps
        )


class SimRobot(Robot):
    """Client half of the sim bridge, shaped like `BiSOFollower`."""

    config_class = SimRobotConfig
    name = "sim_bi_so_follower"

    def __init__(self, config: SimRobotConfig):
        super().__init__(config)
        self.config = config
        self.left_arm_config = config.left_arm_config
        self.right_arm_config = config.right_arm_config

        # 空 id 时按 configs/arms.json 解析,保证和标定工具写出的文件名一致。
        self.calibration_bridge = BimanualCalibration.from_dir(
            config.calibration_source_dir,
            left_id=config.left_calibration_id or _arm("left_follower").calibration_id,
            right_id=config.right_calibration_id or _arm("right_follower").calibration_id,
            signs=config.joint_signs,
        )

        # `BiSOFollower` exposes a merged camera dict, and the recording backend
        # sizes its image-writer thread pool from `len(robot.cameras)`; without
        # it, recording silently skips writing frames.
        self.cameras = {
            **{f"left_{k}": v for k, v in (self.left_arm_config.cameras or {}).items()},
            **{f"right_{k}": v for k, v in (self.right_arm_config.cameras or {}).items()},
        }

        self._socket = None
        self._context = None
        self._camera_keys: tuple[str, ...] = CAMERA_KEYS
        self._image_shape = (config.height, config.width, 3)
        # Last commanded angles, so a partial action leaves other joints held
        # rather than snapping them to zero.
        self._targets: dict[str, float] = {}

    # -- feature schema -----------------------------------------------------

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{side}_{motor}.pos": float for side in ARM_SIDES for motor in MOTOR_NAMES}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {key: self._image_shape for key in self._camera_keys}

    @property
    def observation_features(self) -> dict:
        return {**self._motors_ft, **self._cameras_ft}

    @property
    def action_features(self) -> dict:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self._socket is not None

    @property
    def is_calibrated(self) -> bool:
        """Calibration is a property of the files, which were loaded in __init__."""
        return True

    def calibrate(self) -> None:
        """No-op: the simulator inherits the real arms' calibration files."""

    def configure(self) -> None:
        """No-op: there are no servo registers to configure."""

    # -- transport ----------------------------------------------------------

    def connect(self, calibrate: bool = True) -> None:
        import zmq

        if self.is_connected:
            raise ConnectionError(f"{self} is already connected")

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        # Without these, a dead simulator makes recv() block forever and wedges
        # the record loop with no diagnostic.
        self._socket.setsockopt(zmq.RCVTIMEO, int(self.config.timeout_s * 1000))
        self._socket.setsockopt(zmq.SNDTIMEO, int(self.config.timeout_s * 1000))
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self.config.endpoint)

        try:
            reply, _ = self._request({"command": Command.HANDSHAKE})
        except Exception:
            self.disconnect()
            raise

        check_version(reply)
        self._verify_handshake(reply)
        self._camera_keys = tuple(reply["camera_keys"])
        self._image_shape = (reply["image_height"], reply["image_width"], 3)
        logger.info(
            "connected to simulator at %s: %d joints, cameras %s at %dx%d",
            self.config.endpoint,
            len(reply["joint_order"]),
            list(self._camera_keys),
            reply["image_width"],
            reply["image_height"],
        )
        self._targets = {}

    def _verify_handshake(self, reply: dict) -> None:
        """Reject a simulator whose joint ordering or image size disagrees.

        Joint order is checked because a mismatch would quietly drive the wrong
        joint; image size because the dataset's feature shapes are fixed at
        recording start and a late surprise corrupts the episode.
        """
        peer_order = tuple(reply.get("joint_order", ()))
        if peer_order != JOINT_ORDER:
            raise ProtocolError(
                f"simulator joint order does not match.\n"
                f"  simulator: {list(peer_order)}\n"
                f"  expected:  {list(JOINT_ORDER)}"
            )
        if (reply["image_width"], reply["image_height"]) != (
            self.config.width,
            self.config.height,
        ):
            raise ProtocolError(
                f"simulator renders {reply['image_width']}x{reply['image_height']}, "
                f"config expects {self.config.width}x{self.config.height}"
            )

    def disconnect(self) -> None:
        """Drop the connection, leaving the simulator running.

        Deliberately does *not* send CLOSE: an episode ending, or the record
        loop tearing down, must not kill a simulator that the next run wants to
        reuse. Use :meth:`shutdown_server` to actually stop it.
        """
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None

    def shutdown_server(self) -> None:
        """Ask the simulator process to exit, then disconnect."""
        if self._socket is not None:
            try:
                self._request({"command": Command.CLOSE})
            except Exception:
                pass  # already gone; tear down regardless
        self.disconnect()

    def _request(self, payload: dict) -> tuple[dict, list[bytes]]:
        import zmq

        if self._socket is None:
            raise ConnectionError(f"{self} is not connected")
        try:
            self._socket.send(json.dumps(payload).encode())
            parts = self._socket.recv_multipart()
        except zmq.Again as exc:
            # REQ sockets cannot recover from a half-finished exchange.
            self._socket.close(linger=0)
            self._socket = None
            raise ConnectionError(
                f"simulator at {self.config.endpoint} did not answer "
                f"{payload.get('command')!r} within {self.config.timeout_s}s"
            ) from exc

        reply = json.loads(parts[0])
        if reply.get("status") != Status.OK:
            raise ProtocolError(f"simulator error: {reply.get('error', reply)}")
        return reply, parts[1:]

    # -- observation / action ----------------------------------------------

    def _decode(self, reply: dict, frames: list[bytes]) -> dict:
        rads = dict(zip(JOINT_ORDER, reply["joint_positions"]))
        observation: dict = self.calibration_bridge.rad_to_observation(rads)

        if len(frames) != len(self._camera_keys):
            raise ProtocolError(
                f"expected {len(self._camera_keys)} frames, got {len(frames)}"
            )
        for key, buf in zip(self._camera_keys, frames):
            observation[key] = np.frombuffer(buf, dtype=np.uint8).reshape(self._image_shape)
        return observation

    def get_observation(self) -> dict:
        reply, frames = self._request({"command": Command.OBSERVE})
        return self._decode(reply, frames)

    def send_action(self, action: dict) -> dict:
        """Convert motor values to radians, step the simulator, echo what was sent.

        The return value mirrors `BiSOFollower`, which returns the action it
        actually applied -- the record loop logs that, not the request.
        """
        rads = self.calibration_bridge.action_to_rad(action)
        self._targets.update(rads)

        missing = [name for name in JOINT_ORDER if name not in self._targets]
        if missing:
            # First action of an episode may be partial; hold the measured pose
            # for anything not commanded yet instead of driving it to zero.
            reply, _ = self._request({"command": Command.OBSERVE})
            measured = dict(zip(JOINT_ORDER, reply["joint_positions"]))
            for name in missing:
                self._targets[name] = measured[name]

        targets = [self._targets[name] for name in JOINT_ORDER]
        self._request(
            {
                "command": Command.STEP,
                "joint_targets": targets,
                "duration_s": 1.0 / self.config.fps,
            }
        )
        return {
            key: value
            for key, value in self.calibration_bridge.rad_to_observation(self._targets).items()
            if key in action
        }

    def reset(self, qpos: list[float] | None = None) -> dict:
        """Reset the scene. Not part of `Robot`, but the loop calls it if present."""
        payload: dict = {"command": Command.RESET}
        if qpos is not None:
            payload["qpos"] = qpos
        reply, frames = self._request(payload)
        self._targets = {}
        return self._decode(reply, frames)

    def reset_objects(self, objects: list[str] | None = None) -> list[str]:
        """只把任务零件放回初始位姿,手臂保持不动。

        `reset()` 会把手臂一起弹回复位姿态 —— 遥操到一半这么来一下,手上的
        主臂和仿真里的从臂就对不上了。零件被碰歪时该用这个。
        `self._targets` 也刻意不清:手臂的指令还在生效。
        """
        payload: dict = {"command": Command.RESET_OBJECTS}
        if objects is not None:
            payload["objects"] = list(objects)
        reply, _ = self._request(payload)
        return reply.get("objects_reset", [])


def make_sim_robot(
    endpoint: str = DEFAULT_ENDPOINT,
    calibration_dir: str | Path = DEFAULT_CALIBRATION_DIR,
    **kwargs,
) -> SimRobot:
    """Convenience constructor mirroring the real robot's config loaders."""
    return SimRobot(
        SimRobotConfig(
            id="sim_bimanual",
            endpoint=endpoint,
            calibration_source_dir=str(calibration_dir),
            **kwargs,
        )
    )
