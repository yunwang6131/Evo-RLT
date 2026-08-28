#!/usr/bin/env python
"""MuJoCo simulator process for the dual SO-101 rig.

Runs standalone: it imports neither `evo_rlt` nor `lerobot` (the former pulls in
torch, and the whole point of the split is that these dependency sets do not
have to coexist). It speaks the REQ/REP protocol in ``protocol.py`` and thinks
purely in radians -- every LeRobot calibration concern is the client's job.

Run it with the NVIDIA GPU, always::

    python src/evo_rlt/sim/mj_server.py --build

On this Optimus laptop, GL defaults to the Intel iGPU, where a 640x480 frame
takes ~128 ms to render; the same frame on the discrete GPU takes ~0.4 ms. That
is the difference between 5 Hz and 500 Hz, so the offload environment variables
are set here rather than left to the caller to remember. Pass ``--no-gpu-offload``
to opt out (e.g. on a desktop with a single NVIDIA card).

Timing model: physics runs at ``--physics-hz`` and is advanced only inside a
``step`` request, by exactly the amount of wall time the client asked to elapse.
The simulator therefore never runs ahead of the controller, and a slow policy
step slows sim time instead of desynchronising it -- which is what makes
recorded episodes reproducible.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import traceback
from collections import deque
from pathlib import Path

import numpy as np

from collections.abc import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from protocol import (  # noqa: E402
    CAMERA_KEYS,
    DEFAULT_CONTROL_HZ,
    DEFAULT_ENDPOINT,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_ACTION_DELAY_STEPS,
    DEFAULT_IK_DAMPING,
    DEFAULT_IK_ITERS,
    DEFAULT_IK_ROTATION_WEIGHT,
    DEFAULT_PHYSICS_HZ,
    EE_BODIES,
    JOINT_ORDER,
    POSE_LEN,
    PROTOCOL_VERSION,
    Command,
    Status,
)


def enable_gpu_offload() -> None:
    """Force GL onto the discrete NVIDIA GPU before any GL context exists.

    Must run before mujoco imports its rendering backend, so this is called at
    module entry rather than inside the server.
    """
    os.environ.setdefault("MUJOCO_GL", "glfw")
    os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")


def _load_random_specs() -> dict[str, dict]:
    """读 configs/task_scene.json 里各零件的 ``reset_random``。

    直接读 JSON 而不是 import assets:这个进程刻意不依赖 ``evo_rlt``(见模块
    开头),而 assets 只在 ``--build`` 那条路上才导入。缺文件就不随机化。
    """
    path = Path(__file__).resolve().parents[3] / "configs" / "task_scene.json"
    if not path.is_file():
        return {}
    try:
        task = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{path} 不是合法 JSON: {exc}") from exc
    out = {}
    for name, spec in task.items():
        if not isinstance(spec, dict):
            continue
        rnd = spec.get("reset_random")
        if isinstance(rnd, dict) and float(rnd.get("radius", 0.0)) > 0:
            out[name] = rnd
    return out


class SimulatorState:
    """Owns the MuJoCo model, data and renderer."""

    def __init__(
        self,
        scene_path: Path,
        width: int,
        height: int,
        physics_hz: float,
        camera_keys: tuple[str, ...] = CAMERA_KEYS,
        action_delay_steps: int = 0,
        random_seed: int | None = None,
        penetration_warn_mm: float = 2.0,
    ) -> None:
        import mujoco

        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.width = width
        self.height = height
        self.camera_keys = camera_keys

        # 真机在指令发出后有一段时间完全不动(通信往返 + 舵机启动),实测约
        # 2 帧 @30Hz。这是纯延迟,不是慢响应:降低执行器增益只会让仿真"立刻
        # 开始、慢慢接近",而真机是"先不动、再动",曲线形状对不上。所以在这里
        # 排队缓冲,让指令晚 N 步才真正下发。
        self.action_delay_steps = action_delay_steps
        self._pending: deque[list[float]] = deque()
        self._camera_window = None

        self.model.opt.timestep = 1.0 / physics_hz
        self.physics_hz = physics_hz

        # 复位随机化。用**自己的** RNG 而不是全局 random:全局的会被其他代码
        # (以及策略推理)重新播种,那样同一个 --random-seed 复现不出同一批位姿。
        self._rng = random.Random(random_seed)
        self._random_specs = _load_random_specs()

        # 穿透监视。零件嵌进别的东西超过这个深度就打一行日志(见 _watch_penetration)。
        self._pen_threshold = penetration_warn_mm / 1000.0
        self._pen_seen: dict[tuple[int, int], float] = {}
        self._free_bodies = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, nm)
            for nm in self.free_objects()
        }

        self._scene_mtime = scene_path.stat().st_mtime
        self._scene_path = scene_path
        self._verify_scene()
        self.renderer = mujoco.Renderer(self.model, height, width)
        self.viewer = None
        self.reset()

    def open_viewer(self) -> None:
        """Open a live window on the scene.

        Passive viewer: it renders whatever the server has already stepped
        rather than driving the simulation itself, so opening it cannot change
        timing or physics. Used to eyeball joint directions during teleop.
        """
        import mujoco.viewer

        self.viewer = mujoco.viewer.launch_passive(
            self.model, self.data, show_left_ui=False, show_right_ui=False
        )

    def open_camera_window(self) -> None:
        """开一个窗口并排显示三路相机画面。

        放在仿真进程里而不是客户端:客户端跑在装了 lerobot 的环境,那边的
        opencv 是 headless 版(lerobot 的依赖),没有 GUI 支持;而且图像本来就在
        这边生成,传回去再画多一趟拷贝。
        """
        import cv2

        self._camera_window = "sim cameras"
        cv2.namedWindow(self._camera_window, cv2.WINDOW_NORMAL)

    def show_cameras(self, frames: list[bytes]) -> None:
        """把最近一次渲染的画面显示出来。"""
        if getattr(self, "_camera_window", None) is None or not frames:
            return
        import cv2
        import numpy as np

        tiles = []
        for key, buf in zip(self.camera_keys, frames):
            img = np.frombuffer(buf, dtype=np.uint8).reshape(self.height, self.width, 3)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
            cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
            cv2.putText(img, key, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            tiles.append(img)
        cv2.imshow(self._camera_window, np.hstack(tiles))
        cv2.waitKey(1)

    def sync_viewer(self) -> None:
        if self.viewer is None:
            return
        if self.viewer.is_running():
            self.viewer.sync()
        else:  # user closed the window; stop trying
            self.viewer = None

    def _verify_scene(self) -> None:
        """Fail at startup, not mid-episode, if the scene does not match the wire.

        A mismatch here is what silently sends the left arm's target to the
        right arm, so it is checked by name and order rather than by count.
        """
        mujoco = self._mujoco
        actuators = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]
        if tuple(actuators) != JOINT_ORDER:
            raise RuntimeError(
                f"scene actuators do not match the protocol.\n"
                f"  scene:    {actuators}\n"
                f"  expected: {list(JOINT_ORDER)}"
            )
        cameras = {
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            for i in range(self.model.ncam)
        }
        missing = [key for key in self.camera_keys if key not in cameras]
        if missing:
            raise RuntimeError(f"scene is missing cameras {missing}; has {sorted(cameras)}")

    # -- simulation ---------------------------------------------------------

    def reset(self, qpos: list[float] | None = None) -> None:
        # 优先用场景里的 home keyframe(build 时按标定零位写入)。全零是 URDF
        # 零位,和真机 value=0 的姿态差一个 wrist_roll 90 度,viewer 一开就错。
        if qpos is None and self.model.nkey > 0:
            self._mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            self._mujoco.mj_resetData(self.model, self.data)
        # 清空延迟队列:残留指令会在新回合开头下发,让复位后的姿态不可复现
        self._pending.clear()
        if qpos is not None:
            if len(qpos) != self.model.nq:
                raise ValueError(f"reset expects {self.model.nq} qpos, got {len(qpos)}")
            self.data.qpos[:] = qpos
        # Hold position at the reset pose so the arms do not fall on the first
        # step before the client has sent a target.
        self.data.ctrl[:] = self.data.qpos[: self.model.nu]
        self._mujoco.mj_forward(self.model, self.data)

    def free_objects(self) -> list[str]:
        """场景里所有自由刚体零件的名字(按 ``<name>_free`` 关节找)。"""
        names = []
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] != self._mujoco.mjtJoint.mjJNT_FREE:
                continue
            name = self._mujoco.mj_id2name(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, j)
            if name and name.endswith("_free"):
                names.append(name[: -len("_free")])
        return names

    def reset_objects(
        self, names: list[str] | None = None, poses: dict[str, list[float]] | None = None
    ) -> list[str]:
        """把指定零件放回 home keyframe 里的位姿,手臂和其他零件都不动。

        采数据或调试时零件常被碰歪,而整体 reset 会把手臂一起弹回复位姿态 ——
        遥操到一半被弹回去,手上的主臂和仿真就对不上了。所以单列一条命令。

        位姿取自 keyframe(build 时按 configs/task_scene.json 写入),和整体
        reset 用的是同一份基准,不会出现两种"初始位置"。

        ``poses`` 里点名的零件改为放到给定位姿(``[x,y,z,qw,qx,qy,qz]``,世界系),
        并**跳过随机化**。演示增广要按算出来的位姿摆零件 —— 那个位姿是先解 IK
        再反推出来的(见 ``evo_rlt.sim.augment``),用随机种子碰不出来。
        """
        if self.model.nkey == 0:
            raise RuntimeError("场景没有 home keyframe,无法复位零件;重建场景后再试")

        available = self.free_objects()
        poses = dict(poses or {})
        if names is None:
            names = list(available)
        unknown = [n for n in list(names) + list(poses) if n not in available]
        if unknown:
            raise ValueError(f"场景里没有零件 {sorted(set(unknown))};可用的是 {available}")
        for name in poses:
            if name not in names:
                names.append(name)

        key = self.model.key_qpos[0]
        for name in names:
            jid = self._mujoco.mj_name2id(
                self.model, self._mujoco.mjtObj.mjOBJ_JOINT, f"{name}_free"
            )
            adr = self.model.jnt_qposadr[jid]
            dadr = self.model.jnt_dofadr[jid]
            self.data.qvel[dadr : dadr + 6] = 0.0
            explicit = poses.get(name)
            if explicit is None:
                self.data.qpos[adr : adr + 7] = key[adr : adr + 7]
                self._randomize(name, adr)
                continue
            if len(explicit) != POSE_LEN:
                raise ValueError(f"{name} 的位姿要 {POSE_LEN} 个数,给了 {len(explicit)}")
            pose = np.asarray(explicit, dtype=float)
            norm = float(np.linalg.norm(pose[3:]))
            if norm < 1e-9:
                raise ValueError(f"{name} 的四元数是零向量")
            self.data.qpos[adr : adr + 3] = pose[:3]
            self.data.qpos[adr + 3 : adr + 7] = pose[3:] / norm
        self._mujoco.mj_forward(self.model, self.data)
        return names

    def _randomize(self, name: str, adr: int) -> None:
        """按 configs/task_scene.json 的 ``reset_random`` 打散这个零件的初始位姿。

        没配这一段的零件原样放回 keyframe。采 VLA 数据必须打散:每条 episode
        的起点都一样的话,策略学到的是"走到那个固定位置",不是"找到零件"。

        只动 xy 和 yaw,z 和倾角沿用 keyframe —— 那个 z 是 settle_objects.py 跑
        物理落稳得到的,凹槽底是平的,同一个 z 在整个可放区域都成立;而重新
        采样倾角会让零件初始就嵌进凹槽壁。
        """
        spec = self._random_specs.get(name)
        if not spec:
            return
        cx, cy = spec["center"]
        radius = float(spec.get("radius", 0.0))
        # 均匀采样圆盘要对半径开方,直接均匀采 r 会让点堆在圆心
        r = radius * math.sqrt(self._rng.random())
        theta = self._rng.uniform(0.0, 2.0 * math.pi)
        self.data.qpos[adr] = cx + r * math.cos(theta)
        self.data.qpos[adr + 1] = cy + r * math.sin(theta)

        yaw_deg = float(spec.get("yaw_deg", 0.0))
        if yaw_deg:
            yaw = math.radians(self._rng.uniform(-yaw_deg / 2.0, yaw_deg / 2.0))
            half = yaw / 2.0
            # 绕 z 转 yaw,叠加到 keyframe 里那个落稳姿态上(四元数左乘)
            qz = [math.cos(half), 0.0, 0.0, math.sin(half)]
            w, x, y, z = self.data.qpos[adr + 3 : adr + 7]
            self.data.qpos[adr + 3 : adr + 7] = [
                qz[0] * w - qz[3] * z,
                qz[0] * x - qz[3] * y,
                qz[0] * y + qz[3] * x,
                qz[0] * z + qz[3] * w,
            ]

    def step(self, targets: list[float], duration_s: float) -> int:
        """Apply joint targets and advance physics by ``duration_s``。

        指令先入队,实际下发的是 ``action_delay_steps`` 步之前那条 —— 队列没满
        之前保持当前位姿,对应真机上电后尚未收到有效指令的状态。
        """
        if len(targets) != self.model.nu:
            raise ValueError(f"step expects {self.model.nu} targets, got {len(targets)}")

        self._pending.append(list(targets))
        if len(self._pending) > self.action_delay_steps:
            self.data.ctrl[:] = self._pending.popleft()

        n = max(1, int(round(duration_s * self.physics_hz)))
        for _ in range(n):
            self._mujoco.mj_step(self.model, self.data)
            self._watch_penetration()
        return n

    def _watch_penetration(self) -> None:
        """零件嵌进别的东西太深就记一笔。遥操时不用人盯着看。

        为什么要这个:脚本能构造的失败场景我都试过了(自由落体、有界推力、
        定点合爪),全都过;而操作者在遥操里能做出脚本造不出的动作序列。与其
        继续猜他手怎么动的,不如让仿真自己在事发那一刻把证据留下来。

        只扫涉及自由零件的接触,开销可以忽略(接触总数里这类只占几十个)。
        """
        if self._pen_threshold <= 0:
            return
        mj = self._mujoco
        # 零件和台面都是几十上百个凸块,同一次事件会命中几十对几何。按**物体对**
        # 聚合成一行,否则一次穿透刷满屏幕,反而看不出发生了什么。
        worst: dict[tuple[int, int], tuple[float, float]] = {}
        force = np.zeros(6)
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            depth = -con.dist
            if depth < self._pen_threshold:
                continue
            b1 = int(self.model.geom_bodyid[con.geom1])
            b2 = int(self.model.geom_bodyid[con.geom2])
            if b1 not in self._free_bodies and b2 not in self._free_bodies:
                continue
            mj.mj_contactForce(self.model, self.data, c, force)
            key = (b1, b2) if b1 <= b2 else (b2, b1)
            prev = worst.get(key, (0.0, 0.0))
            # 力要**求和**不是取最大:两个物体都是几十个凸块,载荷分摊在几十个
            # 接触点上,单点的峰值只有合力的零头。报单点会让"其实压了 30N"
            # 看起来像"才 3N",把诊断带偏。
            worst[key] = (max(prev[0], depth), prev[1] + abs(float(force[0])))

        for key, (depth, fn) in worst.items():
            # 只在比这对物体历史最深还深 1mm 时才再报,免得持续嵌着刷屏
            if depth <= self._pen_seen.get(key, 0.0) + 0.001:
                continue
            self._pen_seen[key] = depth
            nm = lambda b: mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, b) or f"body{b}"
            print(f"[sim] 穿透 {depth*1000:6.2f}mm  t={self.data.time:7.3f}s  "
                  f"{nm(key[0])} × {nm(key[1])}  峰值法向力 {fn:.2f}N", flush=True)

    # -- observation --------------------------------------------------------

    def joint_positions(self) -> list[float]:
        """Measured joint angles, in `JOINT_ORDER`.

        Reads qpos rather than echoing ctrl, so the client sees where the arm
        actually is -- servo lag and contact are part of what the sim is for.
        """
        return [float(v) for v in self.data.qpos[: self.model.nu]]

    def joint_velocities(self) -> list[float]:
        return [float(v) for v in self.data.qvel[: self.model.nu]]

    # -- 位姿读出与运动学 ---------------------------------------------------

    def _body_pose(self, bid: int) -> list[float]:
        return [*(float(v) for v in self.data.xpos[bid]), *(float(v) for v in self.data.xquat[bid])]

    def object_poses(self) -> dict[str, list[float]]:
        """每个任务零件在世界系的位姿。

        零件坐标本来就在 ``data.qpos`` 里,只是从来没送到进程边界外面 —— 客户端
        看得见关节角和相机,却看不见零件在哪。自动成功判据和演示增广都要它,
        而且它必须和这一帧的图像来自同一次 ``mj_forward``,不能事后再问一遍。
        """
        return {
            name: self._body_pose(
                self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_BODY, name)
            )
            for name in self.free_objects()
        }

    def ee_poses(self) -> dict[str, list[float]]:
        """两只夹爪的 ``gripper_link`` 位姿,世界系。"""
        return {
            side: self._body_pose(
                self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_BODY, body)
            )
            for side, body in EE_BODIES.items()
        }

    def _arm_qpos_adr(self, side: str) -> "np.ndarray":
        """某条臂 5 个本体关节的 qpos 下标(不含夹爪)。

        夹爪排除在外:它不影响 ``gripper_link`` 的位姿,留在 IK 里只会给雅可比
        添一列零,让阻尼最小二乘的条件数变差。

        下标从 ``jnt_qposadr`` 查,不假设"关节顺序即 qpos 顺序" —— 场景是生成的,
        真出现错位时这里会直接找不到关节而报错,而不是悄悄驱动隔壁那个关节。
        """
        adrs = []
        for name in JOINT_ORDER:
            if not name.startswith(f"{side}_") or name.endswith("_gripper"):
                continue
            jid = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"场景里没有关节 {name}")
            adrs.append(int(self.model.jnt_qposadr[jid]))
        return np.array(adrs, dtype=int)

    def fk(self, qpos_batch: list[list[float]]) -> list[dict[str, list[float]]]:
        """一批 12 维关节角 -> 每帧两只夹爪的位姿。

        用一份独立的 ``MjData``,绝不碰 ``self.data`` —— 采集途中客户端问 FK 不该
        让正在跑的那条 episode 的状态动一下。
        """
        mujoco = self._mujoco
        bids = {
            side: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
            for side, body in EE_BODIES.items()
        }
        scratch = mujoco.MjData(self.model)
        out = []
        for qpos in qpos_batch:
            if len(qpos) != self.model.nu:
                raise ValueError(f"fk expects {self.model.nu} joint angles, got {len(qpos)}")
            scratch.qpos[: self.model.nu] = qpos
            # 只求运动学。mj_forward 会连碰撞一起算,而这个场景光凸包就有 280
            # 多块 —— 一条 700 帧的轨迹要解上万次,那部分开销是纯浪费。
            mujoco.mj_kinematics(self.model, scratch)
            out.append(
                {
                    side: [
                        *(float(v) for v in scratch.xpos[bid]),
                        *(float(v) for v in scratch.xquat[bid]),
                    ]
                    for side, bid in bids.items()
                }
            )
        return out

    def ik(
        self,
        side: str,
        targets: list[list[float]],
        seed: list[float],
        rotation_weight: float | Sequence[float] = DEFAULT_IK_ROTATION_WEIGHT,
        iters: int = DEFAULT_IK_ITERS,
        damping: float = DEFAULT_IK_DAMPING,
    ) -> dict[str, list]:
        """阻尼最小二乘 IK,一次解一整条轨迹。

        **SO-101 只有 5 个本体关节,够不到任意 6D 位姿。** 求解器因此是加权的:
        位置权重 1,姿态误差按世界坐标轴分别乘 ``rotation_weight``(标量则三轴
        同权)。默认 ``(1, 1, 0.02)`` —— 位置 3 维加倾角 2 维正好等于自由度数,
        只放开绕世界 z 的偏航,那正是实测缺的那一维。偏航残差随 ``rot_err``
        回给调用者(增广那边把零件的朝向按它反算回去,见 augment.py)。

        逐点用上一帧的解做种子:轨迹是连续的,这样既快又不会在相邻两帧跳到
        不同的 IK 分支上 —— 跳支在关节空间是一个大台阶,replay 时会甩臂。
        """
        if side not in EE_BODIES:
            raise ValueError(f"unknown arm side {side!r}, expected one of {sorted(EE_BODIES)}")
        if len(seed) != self.model.nu:
            raise ValueError(f"ik seed expects {self.model.nu} joint angles, got {len(seed)}")
        mujoco = self._mujoco
        dofs = self._arm_qpos_adr(side)
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, EE_BODIES[side])

        scratch = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, scratch, 0)
        scratch.qpos[: self.model.nu] = seed
        # mj_jacBody 要的是 xpos/xquat 和 cdof,mj_kinematics + mj_comPos 就够;
        # 走 mj_forward 会白算这个场景 280 多块凸包的碰撞,IK 的迭代次数一乘
        # 就是几个数量级的差别。
        q = np.array(seed, dtype=float)[dofs]
        lo = np.array([self.model.jnt_range[j][0] for j in dofs])
        hi = np.array([self.model.jnt_range[j][1] for j in dofs])

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        rot_w = np.broadcast_to(np.asarray(rotation_weight, dtype=float), (3,))
        weights = np.diag([1.0, 1.0, 1.0, *rot_w])
        qc = np.zeros(4)
        qe = np.zeros(4)
        rot_err_vec = np.zeros(3)

        solutions: list[list[float]] = []
        pos_errs: list[float] = []
        rot_errs: list[float] = []
        for target in targets:
            if len(target) != POSE_LEN:
                raise ValueError(f"ik target expects {POSE_LEN} numbers, got {len(target)}")
            tp = np.asarray(target[:3], dtype=float)
            tq = np.asarray(target[3:], dtype=float)
            tq = tq / max(float(np.linalg.norm(tq)), 1e-12)
            pos_err = rot_err = float("inf")
            for _ in range(iters):
                scratch.qpos[dofs] = q
                mujoco.mj_kinematics(self.model, scratch)
                mujoco.mj_comPos(self.model, scratch)
                e_pos = tp - scratch.xpos[bid]
                mujoco.mju_negQuat(qc, scratch.xquat[bid])
                mujoco.mju_mulQuat(qe, tq, qc)
                mujoco.mju_quat2Vel(rot_err_vec, qe, 1.0)
                pos_err = float(np.linalg.norm(e_pos))
                rot_err = float(np.linalg.norm(rot_err_vec))
                if pos_err < 5e-5 and rot_err < 1e-3:
                    break
                err = weights @ np.concatenate([e_pos, rot_err_vec])
                mujoco.mj_jacBody(self.model, scratch, jacp, jacr, bid)
                jac = weights @ np.vstack([jacp[:, dofs], jacr[:, dofs]])
                step = np.linalg.solve(
                    jac.T @ jac + damping * np.eye(len(dofs)), jac.T @ err
                )
                norm = float(np.linalg.norm(step))
                if norm > 0.05:  # 限步,避免near-singular 时一步甩到限位上
                    step *= 0.05 / norm
                q = np.clip(q + step, lo, hi)
            solutions.append([float(v) for v in q])
            pos_errs.append(pos_err)
            rot_errs.append(rot_err)
        return {"qpos": solutions, "pos_err": pos_errs, "rot_err": rot_errs, "qpos_adr": dofs.tolist()}

    def render_all(self) -> list[bytes]:
        frames = []
        for key in self.camera_keys:
            self.renderer.update_scene(self.data, camera=key)
            frames.append(self.renderer.render().tobytes())
        return frames

    def close(self) -> None:
        if getattr(self, "_camera_window", None) is not None:
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:  # pragma: no cover - best effort teardown
                pass
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:  # pragma: no cover - best effort teardown
                pass
        try:
            self.renderer.close()
        except Exception:  # pragma: no cover - best effort teardown
            pass


class SimServer:
    """REQ/REP loop.

    Camera frames ride as extra ZMQ message parts rather than inside the JSON,
    so they are never base64'd. Three 640x480 RGB frames are ~2.7 MB per step,
    which local TCP moves in about a millisecond -- shared memory would be
    faster but adds lifetime and cleanup failure modes, and the render itself
    (~1.8 ms for all three) is the larger cost anyway.
    """

    def __init__(self, sim: SimulatorState, endpoint: str, control_hz: float) -> None:
        import zmq

        self._zmq = zmq
        self.sim = sim
        self.control_hz = control_hz
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(endpoint)
        self.endpoint = endpoint
        self.steps = 0

    def _describe(self) -> dict:
        return {
            "status": Status.OK,
            "protocol_version": PROTOCOL_VERSION,
            "joint_order": list(JOINT_ORDER),
            "camera_keys": list(self.sim.camera_keys),
            "image_width": self.sim.width,
            "image_height": self.sim.height,
            "physics_hz": self.sim.physics_hz,
            "action_delay_steps": self.sim.action_delay_steps,
            "control_hz": self.control_hz,
            "nq": int(self.sim.model.nq),
            "free_objects": self.sim.free_objects(),
            "ee_bodies": dict(EE_BODIES),
        }

    def _observation(self) -> tuple[dict, list[bytes]]:
        """一帧观测。

        零件和夹爪位姿跟着每一帧一起回,不另开一条命令:成功判据要判的是"这一
        帧"的状态,分两次问会拿到两个不同时刻的场景(中间可能已经 step 过),
        而这种错位在数据里表现为偶发的误判,极难查。JSON 只多 ~30 个浮点数,
        相对同一条消息里 2.7 MB 的图像可以忽略。
        """
        return (
            {
                "status": Status.OK,
                "joint_positions": self.sim.joint_positions(),
                "joint_velocities": self.sim.joint_velocities(),
                "object_poses": self.sim.object_poses(),
                "ee_poses": self.sim.ee_poses(),
                "sim_time": float(self.sim.data.time),
                "steps": self.steps,
            },
            self.sim.render_all(),
        )

    def _warn_if_scene_stale(self) -> None:
        """磁盘上的 scene.xml 比本进程加载的新就喊一声。

        ``--build`` 只重写文件,**已经在跑的服务器不会重新加载** —— 场景是启动
        那一刻读进内存的。这个坑真实发生过好几次:改完接触参数、重建、然后对着
        一个旧场景调半天,而所有症状都指向"改动没生效"。
        客户端每次连上都会握手,所以这里查一次就够,不占步进的开销。
        """
        try:
            now = self.sim._scene_path.stat().st_mtime
        except OSError:
            return
        if now > self.sim._scene_mtime + 1.0:
            import datetime as _dt

            fmt = lambda t: _dt.datetime.fromtimestamp(t).strftime("%m-%d %H:%M:%S")
            print(f"\n[sim] ** 场景已过期 ** 本进程加载于 {fmt(self.sim._scene_mtime)},"
                  f"而 scene.xml 在 {fmt(now)} 被重建过。\n"
                  f"[sim]    你正在跑的是旧场景,重建的改动没有生效。"
                  f"Ctrl-C 重启本进程。\n", flush=True)

    def _handle(self, request: dict) -> tuple[dict, list[bytes]]:
        command = request.get("command")

        if command == Command.HANDSHAKE:
            self._warn_if_scene_stale()
            return self._describe(), []

        if command == Command.OBSERVE:
            return self._observation()

        if command == Command.STEP:
            targets = request["joint_targets"]
            duration = float(request.get("duration_s", 1.0 / self.control_hz))
            substeps = self.sim.step(targets, duration)
            self.steps += 1
            reply, frames = self._observation()
            reply["substeps"] = substeps
            return reply, frames

        if command == Command.RESET:
            self.sim.reset(request.get("qpos"))
            self.steps = 0
            return self._observation()

        if command == Command.RESET_OBJECTS:
            # 不清 self.steps:手臂没动,回合还是同一个
            done = self.sim.reset_objects(request.get("objects"), request.get("poses"))
            reply, frames = self._observation()
            reply["objects_reset"] = done
            return reply, frames

        if command == Command.FK:
            return {"status": Status.OK, "ee_poses": self.sim.fk(request["qpos"])}, []

        if command == Command.IK:
            solved = self.sim.ik(
                request["side"],
                request["targets"],
                request["seed"],
                rotation_weight=request.get(
                    "rotation_weight", DEFAULT_IK_ROTATION_WEIGHT
                ),
                iters=int(request.get("iters", DEFAULT_IK_ITERS)),
                damping=float(request.get("damping", DEFAULT_IK_DAMPING)),
            )
            return {"status": Status.OK, **solved}, []

        if command == Command.CLOSE:
            return {"status": Status.OK, "closing": True}, []

        raise ValueError(f"unknown command {command!r}")

    def serve_forever(self) -> None:
        import json

        print(f"[sim] listening on {self.endpoint}", flush=True)
        print(
            f"[sim] {self.sim.model.nu} joints, cameras {list(self.sim.camera_keys)} "
            f"at {self.sim.width}x{self.sim.height}, physics {self.sim.physics_hz:g} Hz, "
            f"action delay {self.sim.action_delay_steps} steps",
            flush=True,
        )
        while True:
            request = json.loads(self.socket.recv())
            try:
                reply, frames = self._handle(request)
            except Exception as exc:
                traceback.print_exc()
                reply, frames = (
                    {"status": Status.ERROR, "error": f"{type(exc).__name__}: {exc}"},
                    [],
                )
            self.socket.send_multipart([json.dumps(reply).encode(), *frames])
            self.sim.sync_viewer()
            self.sim.show_cameras(frames)
            if reply.get("closing"):
                break
        print("[sim] closing", flush=True)

    def close(self) -> None:
        self.socket.close(linger=0)
        self.context.term()
        self.sim.close()


def benchmark(sim: SimulatorState, control_hz: float, seconds: float = 3.0) -> None:
    """Report whether this machine can actually sustain the control rate."""
    budget_ms = 1000.0 / control_hz
    targets = sim.joint_positions()
    n = int(seconds * control_hz)

    t0 = time.perf_counter()
    for _ in range(n):
        sim.step(targets, 1.0 / control_hz)
    physics_ms = (time.perf_counter() - t0) / n * 1000

    t0 = time.perf_counter()
    for _ in range(n):
        sim.render_all()
    render_ms = (time.perf_counter() - t0) / n * 1000

    total = physics_ms + render_ms
    print(f"[sim] physics {physics_ms:6.2f} ms/step")
    print(f"[sim] render  {render_ms:6.2f} ms/step ({len(sim.camera_keys)} cameras)")
    print(f"[sim] total   {total:6.2f} ms vs {budget_ms:.1f} ms budget at {control_hz:g} Hz")
    if total > budget_ms:
        print(
            f"[sim] WARNING: cannot sustain {control_hz:g} Hz. If render dominates, "
            "check that GL is on the NVIDIA GPU (the iGPU is ~300x slower here)."
        )
    else:
        print(f"[sim] headroom {budget_ms / total:.1f}x")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=Path, default=None, help="scene.xml (default: cache dir)")
    parser.add_argument("--build", action="store_true", help="(re)build the scene from the URDF first")
    parser.add_argument("--urdf", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("~/.cache/evo_rlt/sim_assets"))
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--width", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument("--physics-hz", type=float, default=DEFAULT_PHYSICS_HZ)
    parser.add_argument("--action-delay-steps", type=int, default=DEFAULT_ACTION_DELAY_STEPS,
                        help="指令纯延迟步数,匹配真机通信+舵机启动时间")
    parser.add_argument("--control-hz", type=float, default=DEFAULT_CONTROL_HZ)
    parser.add_argument("--penetration-warn-mm", type=float, default=2.0,
                        help="零件嵌进别的东西超过这么深就打日志(0=关掉)。"
                             "查\"零件穿桌/穿钳口\"用,正常遥操时几乎不会触发")
    parser.add_argument("--random-seed", type=int, default=None,
                        help="复位随机化的种子。不给就每次不同(采数据要的就是这个);"
                             "给了同一个数就能复现同一批初始位姿,便于对照实验")
    parser.add_argument("--no-gpu-offload", action="store_true", help="do not force GL onto the NVIDIA GPU")
    parser.add_argument("--benchmark", action="store_true", help="report timing and exit")
    parser.add_argument("--viewer", action="store_true", help="开仿真场景窗口(看机械臂姿态)")
    parser.add_argument("--show-cameras", action="store_true",
                        help="开相机窗口(看策略实际拿到的三路图像)")
    args = parser.parse_args()

    if not args.no_gpu_offload:
        enable_gpu_offload()

    cache_dir = args.cache_dir.expanduser()
    scene_path = args.scene or (cache_dir / "scene.xml")
    if args.build or not scene_path.is_file():
        from assets import SceneConfig, build_scene

        cfg = SceneConfig(cache_dir=cache_dir)
        if args.urdf is not None:
            cfg.urdf_path = args.urdf
        scene_path = build_scene(cfg)
        print(f"[sim] built scene {scene_path}", flush=True)

    sim = SimulatorState(scene_path, args.width, args.height, args.physics_hz,
                         random_seed=args.random_seed,
                         penetration_warn_mm=args.penetration_warn_mm,
                         action_delay_steps=args.action_delay_steps)

    if args.benchmark:
        benchmark(sim, args.control_hz)
        sim.close()
        return 0

    # Bind before opening the viewer: if the port is taken (usually another
    # simulator still running) we should fail with that error, not leave a
    # half-initialised GL window behind.
    try:
        server = SimServer(sim, args.endpoint, args.control_hz)
    except Exception as exc:
        sim.close()
        print(f"[sim] cannot start: {exc}", flush=True)
        if "in use" in str(exc):
            print(f"[sim] another simulator is probably already on {args.endpoint}", flush=True)
        return 1

    if args.viewer:
        sim.open_viewer()
        print("[sim] viewer open", flush=True)
    if args.show_cameras:
        sim.open_camera_window()
        print("[sim] camera window open", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[sim] interrupted", flush=True)
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
