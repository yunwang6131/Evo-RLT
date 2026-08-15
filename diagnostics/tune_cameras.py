#!/usr/bin/env python
"""在仿真里实时调三路相机的位姿。纯仿真,不碰任何真机设备。

    ~/anaconda3/envs/rlt_sim/bin/python diagnostics/tune_cameras.py

窗口分四格:三路相机各自的画面,加一个全景 —— 全景里能看到相机装在哪、朝哪,
调 wrist 相机时尤其有用,否则相机埋进机身你只会看到一片机体颜色。

按键:
    Tab      切换要调的相机(标题里高亮)
    w / s    前 / 后        a / d    左 / 右        r / f    上 / 下
    i / k    俯 / 仰        j / l    左转 / 右转
    1-5      步长
    p        换机械臂姿态(默认是标定零位,即真机 value=0 的位置)
    0        当前相机恢复出厂默认
    9        当前相机回到上次保存的值(撤销未保存的改动)
    h        显示/隐藏按键帮助
    SPACE    保存到 configs/cameras.json
    ESC      退出(不保存)

保存后要 `mj_server.py --build` 重建场景才会生效。

wrist 相机的位姿是**相对夹爪**的,两只手共用同一套(它们来自同一个单臂模型);
right_front 是固定相机,调的是世界坐标。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "evo_rlt" / "sim"))

CAMERA_CONFIG = REPO_ROOT / "configs" / "cameras.json"
CAMERAS = ("left_wrist", "right_wrist", "right_front")
STEPS = [0.002, 0.005, 0.01, 0.02, 0.05]

#: 三路相机统一的垂直视场角(度)。刻意不做成可调:它决定的是镜头广角程度,
#: 不是安装位置,三路取同一值就够用,留着反而容易被误调出各路不一致的画面。
FOVY = 58.0

#: 手工编的参考姿态,只是让人看看不同角度的视野,和真机没有对应关系。
#: 真正该照着调的是 calib_zero —— 那才是标定时"摆到行程中间"的姿态。
SYNTHETIC_POSES = {
    "前伸(参考)": [0.0, -0.6, 1.0, 0.6, 0.0, 0.3],
    "下探(参考)": [0.0, -1.0, 1.4, 0.8, 0.0, 0.6],
    "侧摆(参考)": [0.6, -0.6, 1.0, 0.6, 0.8, 0.3],
}


def calibration_zero_pose() -> dict[str, list[float]] | None:
    """仿真的复位姿态,直接取场景里的 home keyframe。

    keyframe 由 ``assets.py`` 在 build 时按 ``configs/home_pose.json``(真机
    go-home 的 ticks)写入,所以这里读到的和 ``mj_server`` 复位后的姿态完全一致。
    调相机必须照着实际工作姿态,拿手编数值调出来的视野对不上真机。
    """
    import mujoco

    scene = Path("~/.cache/evo_rlt/sim_assets/scene.xml").expanduser()
    if not scene.is_file():
        print(f"场景还没构建: {scene}")
        return None
    model = mujoco.MjModel.from_xml_path(str(scene))
    if model.nkey == 0:
        print("场景里没有 home keyframe —— 重新 build 一次")
        return None
    q = model.key_qpos[0]
    return {"left": list(q[:6]), "right": list(q[6:12])}


def euler_to_xyaxes(yaw: float, pitch: float) -> tuple[float, ...]:
    """由 yaw/pitch 生成 MJCF 的 xyaxes(相机 x 轴与 y 轴)。

    MuJoCo 相机看向自身 -z,x 向右、y 向上。
    """
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    return (cy, sy, 0.0, -sy * sp, cy * sp, cp)


class CameraState:
    def __init__(self, name: str, pos, yaw: float, pitch: float, fovy: float):
        self.name = name
        self.pos = list(pos)
        self.yaw = yaw
        self.pitch = pitch
        self.fovy = fovy

    def to_dict(self) -> dict:
        return {
            "pos": [round(v, 5) for v in self.pos],
            "xyaxes": [round(v, 5) for v in euler_to_xyaxes(self.yaw, self.pitch)],
            "fovy": round(self.fovy, 2),
            "yaw": round(self.yaw, 5),
            "pitch": round(self.pitch, 5),
        }

    @classmethod
    def from_dict(cls, name: str, d: dict) -> CameraState:
        return cls(name, d["pos"], d.get("yaw", 0.0), d.get("pitch", 0.0), FOVY)


def xyaxes_to_quat(xyaxes) -> "np.ndarray":
    """把 MJCF 的 xyaxes 转成 MuJoCo 内部用的四元数。

    列向量分别是相机的 x/y/z 轴,z = x 叉 y。
    """
    import mujoco

    x = np.array(xyaxes[:3], dtype=float)
    y = np.array(xyaxes[3:], dtype=float)
    z = np.cross(x, y)
    mat = np.column_stack([x, y, z]).flatten()
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat)
    return quat


def apply_to_model(model, states: dict[str, CameraState]) -> None:
    """把当前外参直接写进模型。

    刻意不重建场景:每次按键都重建不仅慢(要重读 URDF、转 MJCF、写盘),更会
    新建一个 Renderer 而旧的 GL 上下文没释放,按几下就泄漏到渲染全黑。
    相机位姿本来就是模型里的可变字段,改完立即生效。
    """
    import mujoco

    for name, st in states.items():
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cam_id < 0:
            continue
        model.cam_pos[cam_id] = st.pos
        model.cam_quat[cam_id] = xyaxes_to_quat(euler_to_xyaxes(st.yaw, st.pitch))
        model.cam_fovy[cam_id] = st.fovy


def look_at(eye, target) -> tuple[float, float]:
    """求让相机从 eye 看向 target 的 yaw/pitch。

    ``euler_to_xyaxes`` 的朝向是 z_axis=(sy*cp, -cy*cp, sp),相机看向 -z。
    直接猜 yaw/pitch 很容易把相机指向虚空(渲染出全黑),所以默认值一律由
    几何反解得到,不靠手填。
    """
    dx, dy, dz = (t - e for t, e in zip(target, eye))
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    dx, dy, dz = dx / norm, dy / norm, dz / norm
    pitch = math.asin(-dz)
    cp = math.cos(pitch)
    return math.atan2(-dx / cp, dy / cp), pitch


def default_states() -> dict[str, CameraState]:
    # wrist 相机伸到夹爪前方一点,否则埋在机身里只看到一片机体颜色;
    # 朝向由 look_at 反解,保证一开始就能看到东西。
    wrist_pos = (0.02, -0.09, 0.05)
    wrist_yaw, wrist_pitch = look_at(wrist_pos, (0.02, -0.25, -0.10))
    front_pos = (0.55, -0.18, 0.35)
    front_yaw, front_pitch = look_at(front_pos, (0.10, 0.0, 0.12))
    return {
        "left_wrist": CameraState("left_wrist", wrist_pos, wrist_yaw, wrist_pitch, FOVY),
        "right_wrist": CameraState("right_wrist", wrist_pos, wrist_yaw, wrist_pitch, FOVY),
        "right_front": CameraState("right_front", front_pos, front_yaw, front_pitch, FOVY),
    }


def load_states() -> dict[str, CameraState]:
    if CAMERA_CONFIG.is_file():
        raw = json.loads(CAMERA_CONFIG.read_text())
        states = default_states()
        for name, entry in raw.items():
            if name in CAMERAS:
                states[name] = CameraState.from_dict(name, entry)
        return states
    return default_states()


def save_states(states: dict[str, CameraState]) -> None:
    CAMERA_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CAMERA_CONFIG.write_text(json.dumps({n: s.to_dict() for n, s in states.items()}, indent=2))


def build_scene_with(states: dict[str, CameraState], cache_dir: Path) -> Path:
    from assets import CameraPose, SceneConfig, build_scene

    lw, rf = states["left_wrist"], states["right_front"]
    # load_tuned_cameras=False:用正在调的值,而不是盘上那份
    cfg = SceneConfig(cache_dir=cache_dir, load_tuned_cameras=False)
    cfg.wrist_camera = CameraPose(tuple(lw.pos), euler_to_xyaxes(lw.yaw, lw.pitch), lw.fovy)
    cfg.front_camera = CameraPose(tuple(rf.pos), euler_to_xyaxes(rf.yaw, rf.pitch), rf.fovy)
    return build_scene(cfg)


#: 全景里标记相机位置的颜色。和左右臂的黄/绿呼应,一眼看出哪个点属于哪条臂。
MARKER_RGBA = {
    "left_wrist": (1.0, 0.85, 0.1, 1.0),    # 黄，配左臂
    "right_wrist": (0.25, 0.9, 0.35, 1.0),  # 绿，配右臂
    "right_front": (0.3, 0.6, 1.0, 1.0),    # 蓝，固定相机
}


def mark_cameras(renderer, model, data, active_name: str) -> None:
    """在全景场景里为每个相机加一个小球,并画一条朝向线。

    必须在 ``update_scene`` 之后、``render`` 之前调用:往 mjvScene 里追加几何体
    才能走正常的投影和深度测试,直接在图片上画点会穿透遮挡物。
    """
    import mujoco

    scene = renderer.scene
    for name, rgba in MARKER_RGBA.items():
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cam_id < 0 or scene.ngeom + 2 > scene.maxgeom:
            continue
        pos = data.cam_xpos[cam_id]
        mat = data.cam_xmat[cam_id].reshape(3, 3)
        is_active = name == active_name
        radius = 0.028 if is_active else 0.018

        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, 0, 0]), pos, np.eye(3).flatten(),
            np.array(rgba, dtype=np.float32),
        )
        scene.ngeom += 1

        # 朝向线:相机看向自身 -z
        tip = pos - mat[:, 2] * 0.13
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3), np.zeros(3), np.eye(3).flatten(),
            np.array(rgba, dtype=np.float32),
        )
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.006, pos, tip)
        scene.ngeom += 1


def label(img: np.ndarray, text: str, active: bool):
    import cv2

    color = (0, 255, 255) if active else (180, 180, 180)
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, 20), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    if active:
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (0, 255, 255), 2)
    return img


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", type=Path, default=Path("~/.cache/evo_rlt/sim_assets"))
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=360)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "glfw")
    os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

    import cv2
    import mujoco

    states = load_states()
    cache = args.cache_dir.expanduser()

    poses: dict[str, tuple[list[float], list[float]]] = {}
    zero = calibration_zero_pose()
    if zero is not None:
        poses["复位姿态"] = (zero["left"], zero["right"])
        print("已载入复位姿态(与 mj_server reset 一致)")
    else:
        print("警告: 读不到复位姿态,只能用手编姿态 —— 调出来的视野对不上实际工作姿态")
    for name, q in SYNTHETIC_POSES.items():
        poses[name] = (q, q)
    pose_names = list(poses)
    pose_idx = 0
    active = 0
    step_idx = 2
    show_help = True

    # 场景只建一次。之后所有调整都写进 model 的相机字段。
    scene = build_scene_with(states, cache)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, args.height, args.width)
    pose_dirty = True

    window = "sim cameras"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print(__doc__.split("按键:")[1].split("保存后")[0])

    try:
        while True:
            if pose_dirty:
                left_q, right_q = poses[pose_names[pose_idx]]
                data.qpos[: model.nu] = list(left_q) + list(right_q)
                pose_dirty = False
            apply_to_model(model, states)
            mujoco.mj_forward(model, data)

            frames = []
            for i, name in enumerate(CAMERAS):
                renderer.update_scene(data, camera=name)
                img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                frames.append(label(img.copy(), name, i == active))

            # 第四格:全景,能看出相机装在哪、朝哪
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam)
            # azimuth=45 时左臂显示在画面左边、右臂在右边,和实际方位一致。
            # 其它角度会左右颠倒,调相机时极易认错边。
            cam.distance, cam.azimuth, cam.elevation = 1.3, 45, -22
            cam.lookat[:] = [0.15, 0.0, 0.15]
            renderer.update_scene(data, camera=cam)
            mark_cameras(renderer, model, data, CAMERAS[active])
            overview = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            frames.append(label(overview, f"全景 姿态={pose_names[pose_idx]}  左臂黄/右臂绿  球=相机 线=朝向", False))

            top = np.hstack(frames[:2])
            bottom = np.hstack(frames[2:])
            canvas = np.vstack([top, bottom])

            st = states[CAMERAS[active]]
            info = (
                f"[{CAMERAS[active]}] pos=({st.pos[0]:+.3f},{st.pos[1]:+.3f},{st.pos[2]:+.3f})  "
                f"yaw={math.degrees(st.yaw):+.0f}  pitch={math.degrees(st.pitch):+.0f}  "
                f"step={STEPS[step_idx]:.3f}"
            )
            bar = np.zeros((26, canvas.shape[1], 3), np.uint8)
            cv2.putText(bar, info, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            frame = np.vstack([bar, canvas])
            if show_help:
                lines = [
                    "Tab cam   w/s x   a/d y   r/f z   i/k pitch   j/l yaw",
                    "1-5 步长   p 换姿态",
                    "0 出厂默认   9 回到已保存   SPACE 保存   h 隐藏帮助   ESC 退出",
                ]
                y0 = frame.shape[0] - 14 * len(lines) - 10
                cv2.rectangle(frame, (0, y0 - 8), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
                for i, line in enumerate(lines):
                    cv2.putText(frame, line, (8, y0 + 6 + i * 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
            cv2.imshow(window, frame)

            key = cv2.waitKey(30) & 0xFF
            s = STEPS[step_idx]
            st = states[CAMERAS[active]]
            if key == 27:
                break
            elif key == 9:
                active = (active + 1) % len(CAMERAS)
            elif key == ord(" "):
                save_states(states)
                print(f"已保存 {CAMERA_CONFIG} —— 跑 mj_server.py --build 生效")
            elif key == ord("p"):
                pose_idx = (pose_idx + 1) % len(pose_names)
                pose_dirty = True
            elif key == ord("0"):
                states[CAMERAS[active]] = default_states()[CAMERAS[active]]
                print(f"{CAMERAS[active]} 已恢复出厂默认")
            elif key == ord("9"):
                # 撤销本次未保存的改动,回到盘上那份 —— 调坏了不用从头再来
                saved = load_states()
                states[CAMERAS[active]] = saved[CAMERAS[active]]
                print(f"{CAMERAS[active]} 已回到上次保存的值")
            elif key == ord("h"):
                show_help = not show_help
            elif ord("1") <= key <= ord("5"):
                step_idx = key - ord("1")
            elif key in (ord("w"), ord("s")):
                st.pos[0] += s if key == ord("w") else -s
            elif key in (ord("a"), ord("d")):
                st.pos[1] += s if key == ord("a") else -s
            elif key in (ord("r"), ord("f")):
                st.pos[2] += s if key == ord("r") else -s
            elif key in (ord("i"), ord("k")):
                st.pitch += s if key == ord("i") else -s
            elif key in (ord("j"), ord("l")):
                st.yaw += s if key == ord("j") else -s
    finally:
        renderer.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
