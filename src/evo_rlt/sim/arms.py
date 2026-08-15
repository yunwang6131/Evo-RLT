"""按 USB 序列号定位机械臂,并把标定隔离在本项目内。

两件事:

**认臂**。``/dev/ttyACM*`` 的序号按插入顺序分配,重启或换插口就会变,而四条
SO-101 的电机 ID 都是 1~6,通信层面无法区分。认错了不会有任何报错 —— 一条臂
的标定会被静默写进另一条的文件,之后所有数据都带着错误的姿态基准。序列号刻在
USB 转接板上,重插不变,所以用它认臂,``configs/arms.json`` 只需配一次。

**隔离标定**。标定默认写进 ``~/.cache/huggingface/lerobot/``,那是全机器共享
的,别的项目也读也写。本项目的标定改写到 ``configs/calibration/`` 下,并带
``evosim_`` 前缀,这样和其他项目既不互相覆盖,也不会误用对方的标定。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ARMS_CONFIG = REPO_ROOT / "configs" / "arms.json"

#: 标定落盘位置。LeRobot 按 name 分子目录,follower 和 leader 分开放。
CALIBRATION_ROOT = REPO_ROOT / "configs" / "calibration"
FOLLOWER_CALIBRATION_DIR = CALIBRATION_ROOT / "robots"
LEADER_CALIBRATION_DIR = CALIBRATION_ROOT / "teleoperators"

BY_ID_DIR = Path("/dev/serial/by-id")


class ArmResolveError(RuntimeError):
    """臂配置缺失,或对应的串口当前不在线。"""


@dataclass(frozen=True)
class Arm:
    alias: str
    serial: str
    kind: str  # "follower" | "leader"

    @property
    def calibration_id(self) -> str:
        """本项目专属的标定 id,带前缀避免和其他项目重名。"""
        return f"{_prefix()}_{self.alias}"

    @property
    def calibration_dir(self) -> Path:
        return FOLLOWER_CALIBRATION_DIR if self.kind == "follower" else LEADER_CALIBRATION_DIR

    @property
    def calibration_path(self) -> Path:
        return self.calibration_dir / f"{self.calibration_id}.json"

    @property
    def side(self) -> str:
        return "left" if self.alias.startswith("left") else "right"


def _config() -> dict:
    if not ARMS_CONFIG.is_file():
        raise ArmResolveError(f"缺少臂配置 {ARMS_CONFIG}")
    return json.loads(ARMS_CONFIG.read_text())


def _prefix() -> str:
    return _config().get("calibration_id_prefix", "evosim")


def load_arms() -> dict[str, Arm]:
    cfg = _config()
    return {
        alias: Arm(alias=alias, serial=entry["serial"], kind=entry["kind"])
        for alias, entry in cfg["arms"].items()
    }


def arm(alias: str) -> Arm:
    arms = load_arms()
    if alias not in arms:
        raise ArmResolveError(f"未知的臂 {alias!r},已配置的有 {sorted(arms)}")
    return arms[alias]


def by_id_map() -> dict[str, str]:
    """返回 {序列号: 设备路径}。"""
    found: dict[str, str] = {}
    if not BY_ID_DIR.is_dir():
        return found
    for link in BY_ID_DIR.iterdir():
        # 形如 usb-1a86_USB_Single_Serial_5AB9065103-if00
        name = link.name
        if "_" not in name:
            continue
        serial = name.rsplit("_", 1)[-1].split("-")[0]
        found[serial] = str(link.resolve())
    return found


def resolve_port(alias: str) -> str:
    """按序列号找出这条臂当前挂在哪个串口。"""
    target = arm(alias)
    devices = by_id_map()
    if target.serial not in devices:
        raise ArmResolveError(
            f"{alias}(序列号 {target.serial})不在线。"
            f"当前在线的序列号: {sorted(devices) or '无'}。"
            f"换过转接板就跑 python diagnostics/probe_arms.py --identify 重新认臂。"
        )
    return devices[target.serial]


def resolve_all(required: list[str] | None = None) -> dict[str, str]:
    """返回 {别名: 串口}。``required`` 里的臂缺失即报错。"""
    arms = load_arms()
    devices = by_id_map()
    out: dict[str, str] = {}
    missing: list[str] = []
    for alias, a in arms.items():
        if a.serial in devices:
            out[alias] = devices[a.serial]
        elif required is None or alias in required:
            missing.append(f"{alias}({a.serial})")
    if missing and required is not None:
        raise ArmResolveError(f"以下臂不在线: {missing}")
    return out


def build_device(alias: str, port: str | None = None):
    """按本项目的标定 id 和目录构造 LeRobot 设备。"""
    target = arm(alias)
    port = port or resolve_port(alias)
    target.calibration_dir.mkdir(parents=True, exist_ok=True)

    if target.kind == "follower":
        from lerobot.robots.so_follower import SOFollower
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

        return SOFollower(
            SOFollowerRobotConfig(
                id=target.calibration_id,
                calibration_dir=target.calibration_dir,
                port=port,
            )
        )

    from lerobot.teleoperators.so_leader import SOLeader
    from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig

    return SOLeader(
        SOLeaderTeleopConfig(
            id=target.calibration_id,
            calibration_dir=target.calibration_dir,
            port=port,
        )
    )


def calibration_status() -> dict[str, bool]:
    """每条臂是否已在本项目内标定过。"""
    return {alias: a.calibration_path.is_file() for alias, a in load_arms().items()}
