#!/usr/bin/env python
"""验证夹爪能不能抓住零件,并扫出可用的接触参数。

    python diagnostics/grasp_test.py                    # 左臂抓螺栓
    python diagnostics/grasp_test.py --object socket    # 抓螺套
    python diagnostics/grasp_test.py --render           # 存一张夹持瞬间的图
    python diagnostics/grasp_test.py --sweep --apply    # 扫参并写回 configs/grasp.json

流程:把零件摆到钳口的夹持中心 -> 合爪 -> 抬升 -> 摇晃,量零件是否跟着走。

三段都过才算抓得住:
  合爪    接触求解不能把零件弹飞(轻零件被压进去再弹出是最常见的失败)
  抬升    抵抗重力,滑移要小
  摇晃    抵抗惯性力,这一段最能分辨"夹住了"和"恰好卡住了"

扫参不重建场景 —— condim/friction/solref/impratio 都能直接改内存里的模型,
而重建一次要处理 352 个凸块。这样一轮扫描是秒级而不是分钟级。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENE = Path("~/.cache/evo_rlt/sim_assets/scene.xml").expanduser()
GRASP_CONFIG = REPO_ROOT / "configs" / "grasp.json"

#: 夹爪行程端点(弧度),和 sim/calib.py 的 GRIPPER_* 一致
GRIPPER_CLOSED = math.radians(-13.5)
GRIPPER_OPEN = math.radians(-13.5 + 136.4)

#: 判定"飞了"的速度阈值(米/秒)。正常夹持里零件速度不会超过夹爪闭合的线速度。
FLY_SPEED = 0.5

#: 抬升测试抬多高(米),以及摇晃测试的幅度/频率
LIFT_HEIGHT = 0.08
SHAKE_AMPLITUDE = math.radians(12)
SHAKE_HZ = 2.0

#: 合爪角速度(弧度/秒)。取真机实测的**峰值**,即最苛刻的真实情形。
#: outputs/sign_check.json 209 秒配对遥操里,从臂夹爪实测峰值 2.81 rad/s、
#: p99.5 只有 1.24。原来这里写死"0.6 秒走完全行程",即 2.381/0.6 = 3.97 rad/s,
#: 比真机任何时候都快 —— 合爪冲击是测试造出来的,不是仿真的毛病,按它调接触
#: 参数会一路调偏。
GRIPPER_CLOSE_SPEED = 2.8


@dataclass
class Result:
    """一次抓取测试的结果。距离单位毫米。"""

    close_slip: float
    lift_slip: float
    shake_slip: float
    max_speed: float
    held: bool
    note: str = ""

    @property
    def verdict(self) -> str:
        if not self.held:
            return "飞了" if self.max_speed > FLY_SPEED else "掉了"
        return "抓住"


def _jaw_geoms(model, side: str):
    import mujoco

    name = lambda i: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
    gl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_gripper_link")
    hj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_moving_jaw_so101_v1_link")
    fixed = [i for i in range(model.ngeom)
             if "wrist_roll_follower" in name(i) and model.geom_bodyid[i] == gl]
    moving = [i for i in range(model.ngeom)
              if "moving_jaw" in name(i) and model.geom_bodyid[i] == hj]
    if not fixed or not moving:
        raise SystemExit(
            f"{side} 臂找不到钳口的凸块碰撞几何。先重建场景:\n"
            "  python src/evo_rlt/sim/mj_server.py --build"
        )
    return gl, hj, fixed, moving


def _part_geoms(model, obj: str) -> set[int]:
    """零件的所有碰撞 geom。

    零件做了凸分解,碰撞几何是几十个凸块而不是一个 geom —— 按 `<obj>_geom`
    单名去查只会拿到第一块,接触判定和受力统计都会漏。所以按 body 归属取。
    """
    import mujoco

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj)
    return {i for i in range(model.ngeom)
            if model.geom_bodyid[i] == bid and model.geom_contype[i]}


def pinch_center(model, data, side: str) -> tuple[np.ndarray, np.ndarray]:
    """闭合时两片钳口真正贴上的那点,以及合爪方向(单位向量,世界坐标)。

    不用 mj_geomDistance —— 它在网格对上会返回 0,给出的接触点不可信。直接
    拿凸块顶点算最近点对,结果可复核。铰链那圈顶点要排掉:两片钳口在那里
    本来就互相嵌着(所以它们的 body 对在 <exclude> 里),不是夹持面。
    """
    import mujoco

    gl, hj, fixed, moving = _jaw_geoms(model, side)
    adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_gripper")]
    saved = data.qpos[adr]
    data.qpos[adr] = GRIPPER_CLOSED
    mujoco.mj_forward(model, data)

    def verts(geoms):
        out = []
        for g in geoms:
            mid = model.geom_dataid[g]
            a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            v = model.mesh_vert[a:a + n]
            out.append(data.geom_xpos[g] + v @ data.geom_xmat[g].reshape(3, 3).T)
        return np.vstack(out)

    hinge = data.xpos[hj]
    F, M = verts(fixed), verts(moving)
    F = F[np.linalg.norm(F - hinge, axis=1) > 0.030]
    M = M[np.linalg.norm(M - hinge, axis=1) > 0.030]
    D = np.linalg.norm(F[:, None, :] - M[None, :, :], axis=2)
    i, j = np.unravel_index(D.argmin(), D.shape)
    centre = (F[i] + M[j]) / 2
    axis = M[j] - F[i]
    norm = np.linalg.norm(axis)
    axis = axis / norm if norm > 1e-9 else np.array([0.0, 1.0, 0.0])

    data.qpos[adr] = saved
    mujoco.mj_forward(model, data)
    return centre, axis


def grasp_pose(model, data, side: str, obj: str,
               depths=np.linspace(0.35, 1.05, 21)) -> tuple[np.ndarray, np.ndarray, float]:
    """在**当前(张开)**姿态下,沿钳口找一个真正放得下零件的位置。

    不能用闭合时的贴合点 —— 那是钳口尖端的一个点,18.6mm 的螺栓头摆上去会同时
    插进两片钳口,合爪瞬间的接触力直接把它弹飞(实测峰值 1.05 m/s)。

    做法:沿"铰链 -> 指尖"这条线扫候选位置,用 MuJoCo 自己的碰撞检测量间隙 ——
    临时把零件的 geom margin 放大,接触点就会在分离状态下也生成,`contact.dist`
    即真实间隙。取间隙最大的那点。

    返回 (位置, 四元数, 间隙毫米)。零件轴线对齐铰链轴,即钳口从径向合上,
    这是两个平面夹一个六角头最自然的抓法。
    """
    import mujoco

    gl, hj, fixed, moving = _jaw_geoms(model, side)
    jaw_geoms = set(fixed) | set(moving)
    ogeoms = _part_geoms(model, obj)
    ojoint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{obj}_free")
    adr = model.jnt_qposadr[ojoint]

    gadr = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_gripper")]
    open_angle = data.qpos[gadr]

    tip, _ = pinch_center(model, data, side)
    mujoco.mj_forward(model, data)          # pinch_center 会改 qpos,恢复现场
    hinge = data.xpos[hj]
    axis = data.xmat[hj].reshape(3, 3)[:, 2]        # 铰链转轴(世界)
    finger = tip - hinge
    finger -= axis * float(finger @ axis)            # 投到垂直于转轴的平面
    length = float(np.linalg.norm(finger))
    finger /= length
    perp = np.cross(axis, finger)
    perp /= np.linalg.norm(perp)                     # 合爪方向

    # 零件局部 z 轴转到世界的 axis 方向
    quat = np.zeros(4)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, axis)
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        quat[:] = [1, 0, 0, 0] if axis[2] > 0 else [0, 1, 0, 0]
    else:
        angle = math.atan2(s, float(z @ axis))
        mujoco.mju_axisAngle2Quat(quat, v / s, angle)

    saved_margin = {g: model.geom_margin[g] for g in ogeoms}
    saved_qpos = data.qpos[adr:adr + 7].copy()
    for g in ogeoms:
        model.geom_margin[g] = 0.02

    fixed_set, moving_set = set(fixed), set(moving)

    def gap_to(which: set[int]) -> float:
        out = 1e9
        for i in range(data.ncon):
            c = data.contact[i]
            if c.geom1 in ogeoms:
                other = c.geom2
            elif c.geom2 in ogeoms:
                other = c.geom1
            else:
                continue
            if other in which:
                out = min(out, c.dist)
        return out

    # 这个夹爪只有一片会动。零件必须先靠住**固定**钳口,活动钳口合上来才压得住;
    # 摆在 V 的正中间的话,活动钳口扫过去时零件被推向空处,固定那片自始至终碰不到
    # (实测:活动钳口 19 个接触点、固定钳口 0 个)。
    #
    # 深度同样关键,而且要尽量靠**指尖**。钳口是 V 形,越往喉部越宽:摆在喉部时
    # 合爪会像挤西瓜籽一样把零件往里推,推到某个位置钳口就能完全合拢(那里比零件
    # 宽),法向力从 59N 掉到 0.95N,等于没夹住。靠指尖则是两个面对压,力保得住。
    #
    # 判据:张开时贴着固定钳口(留 1mm 左右),合到底时活动钳口压得进来,
    # 在此前提下取最靠指尖的那个位置。
    best: tuple = (None, None, None)
    for frac in depths:
        for lateral in np.linspace(-0.030, 0.030, 25):
            pos = hinge + finger * (length * frac) + perp * lateral
            data.qpos[adr:adr + 3] = pos
            data.qpos[adr + 3:adr + 7] = quat

            data.qpos[gadr] = open_angle
            mujoco.mj_forward(model, data)
            gap_fixed, gap_moving = gap_to(fixed_set), gap_to(moving_set)
            if gap_fixed <= 0 or gap_moving <= 0:
                continue                     # 张开时就已经嵌进钳口

            data.qpos[gadr] = GRIPPER_CLOSED
            mujoco.mj_forward(model, data)
            if gap_to(moving_set) >= 0:
                continue                     # 合到底也压不到,夹不住

            # 同深度里取最贴住固定钳口的;深度本身留给 search_grasp 用物理去挑
            score = abs(gap_fixed - 0.001)
            if best[1] is None or score < best[0]:
                best = (score, pos.copy(), gap_fixed)

    for g, v in saved_margin.items():
        model.geom_margin[g] = v
    data.qpos[adr:adr + 7] = saved_qpos
    data.qpos[gadr] = open_angle
    mujoco.mj_forward(model, data)

    if best[1] is None:
        raise SystemExit(
            f"钳口里找不到能夹住 {obj} 的摆位。零件尺寸或钳口行程对不上,"
            "不是接触参数的问题。"
        )
    return best[1], quat, best[2] * 1000


def apply_contact(model, cfg: dict) -> None:
    """把接触参数直接写进已加载的模型,免去重建场景。"""
    import mujoco

    name = lambda i: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
    generic = [
        i for i in range(model.ngeom)
        if model.geom_contype[i]
        and ("wrist_roll_follower" in name(i) or "moving_jaw" in name(i))
    ]
    # 钳口用 friction(涩,夹得住),零件和台面用 part_friction(滑,杆能从孔里
    # 拔出来)。见 assets.GraspConfig.part_friction。这里若都写 cfg["friction"],
    # 扫参就会把零件那组悄悄覆盖掉,扫出来的结论和 build 出的场景对不上。
    parts = list(_part_geoms(model, "socket")) + list(_part_geoms(model, "bolt"))
    name_ = name
    parts += [i for i in range(model.ngeom) if name_(i).startswith("worktable_col")]
    for i in generic:
        model.geom_condim[i] = cfg["condim"]
        model.geom_friction[i] = cfg["friction"]
        model.geom_solref[i][:2] = cfg["solref"]
    for i in parts:
        model.geom_condim[i] = cfg["condim"]
        model.geom_friction[i] = cfg["part_friction"]
        model.geom_solref[i][:2] = cfg["solref"]
    model.opt.impratio = cfg["impratio"]

    for side in ("left", "right"):
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_gripper")
        model.jnt_actfrcrange[j] = [-cfg["gripper_force_limit"], cfg["gripper_force_limit"]]


#: 强行穿透多少毫米来测弹出。取浅穿透:实测最凶的弹射恰恰发生在这里
#: (2mm 穿透在 solref=0.004 下弹出 3.93 m/s,10mm 反而只有 1.36)。
EJECT_PROBE_MM = 2.0

#: 弹出速度上限(米/秒)。超过就算这组参数不可用 —— 遥操里夹爪被人手推进
#: 零件几毫米是常态,那时候不能炸。
EJECT_LIMIT = 0.5


def eject_speed(model, data, side: str, obj: str, pose=None) -> tuple[float, float]:
    """把零件强行摆进钳口内部,量它被弹出的峰值速度(米/秒)。

    这一项必须进扫描判据。遥操是位置控制,人手会把夹爪直接怼进零件里,接触
    是从"已经嵌进去几毫米"开始解算的 —— 接触太硬就把零件弹射出去,速度够高
    (>9 m/s)时还能一步跨过 18.5mm 的台面板直接穿过桌子。

    只看滑移量的扫描发现不了这个:发散的配置在夹持测试里可能恰好没炸,
    却被选中写进配置。曾经就这么选出过 solref=0.002(等于步长)。

    **测的时候必须关重力**,而且要用真实的抓取姿态。否则量到的是自由落体
    (0.8s 就有 7.8 m/s)和它落到桌面上的反弹,和接触刚度无关 —— 第一版就是
    这么写的,四组 solref 全报 1.14 m/s,一看就知道测的不是弹射。

    返回 (峰值速度 m/s, 实际穿透 mm)。穿透量一并返回,好核对探针真的嵌进去了。
    """
    import mujoco

    ga = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_gripper")
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{obj}_free")
    badr, bvadr = model.jnt_qposadr[jid], model.jnt_dofadr[jid]

    mujoco.mj_resetDataKeyframe(model, data, 0)
    for i in range(model.nu):
        data.ctrl[i] = data.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]]
    data.ctrl[ga] = GRIPPER_OPEN
    for _ in range(int(0.4 / model.opt.timestep)):
        mujoco.mj_step(model, data)

    centre, axis = pinch_center(model, data, side)
    mujoco.mj_forward(model, data)
    if pose is not None:
        centre, quat = pose[0], pose[1]
    else:
        quat = np.array([1.0, 0.0, 0.0, 0.0])

    ogeoms = _part_geoms(model, obj)
    _, _, fixed, _ = _jaw_geoms(model, side)
    fixed_set = set(fixed)

    def penetration() -> float:
        """零件与**固定**钳口的最深穿透(毫米,负值表示嵌入)。"""
        worst = 0.0
        for i in range(data.ncon):
            c = data.contact[i]
            if c.geom1 in ogeoms:
                other = c.geom2
            elif c.geom2 in ogeoms:
                other = c.geom1
            else:
                continue
            if other in fixed_set:
                worst = min(worst, c.dist)
        return worst * 1000

    # 推进方向取 MuJoCo 自己算的接触法向,不靠几何推算:`pinch_center` 的轴是
    # **闭合**姿态下算的,这里夹爪张着,方向对不上 —— 沿它推 12mm 也一次没嵌进去
    # (四组参数穿透全报 0.00)。先把 margin 放大让分离状态也生成接触点,
    # 从中取法向,再沿法向推进到目标穿透量。
    saved_margin = {g: model.geom_margin[g] for g in ogeoms}
    for g in ogeoms:
        model.geom_margin[g] = 0.03

    data.qpos[badr:badr + 3] = centre
    data.qpos[badr + 3:badr + 7] = quat
    mujoco.mj_forward(model, data)

    normal = None
    best = 1e9
    for i in range(data.ncon):
        c = data.contact[i]
        first = c.geom1 in ogeoms
        if not first and c.geom2 not in ogeoms:
            continue
        other = c.geom2 if first else c.geom1
        if other in fixed_set and c.dist < best:
            best = c.dist
            n = np.array(c.frame[:3])       # geom1 -> geom2
            normal = n if first else -n     # 统一成"零件 -> 钳口"

    pen = 0.0
    if normal is not None:
        for step_mm in np.arange(0.0, 15.0, 0.25):
            data.qpos[badr:badr + 3] = centre + normal * (step_mm / 1000.0)
            mujoco.mj_forward(model, data)
            pen = penetration()
            if pen <= -EJECT_PROBE_MM:
                break

    for g, v in saved_margin.items():
        model.geom_margin[g] = v
    data.qvel[bvadr:bvadr + 6] = 0
    mujoco.mj_forward(model, data)

    gravity = model.opt.gravity.copy()
    model.opt.gravity[:] = 0            # 量的是接触弹出,不是自由落体
    peak = 0.0
    try:
        for _ in range(int(0.5 / model.opt.timestep)):
            mujoco.mj_step(model, data)
            peak = max(peak, float(np.linalg.norm(data.cvel[bid][3:])))
    finally:
        model.opt.gravity[:] = gravity
    return peak, pen


def grip_force(model, data, obj: str) -> float:
    """零件身上所有接触的法向力之和(牛)。夹持牢不牢就看这个。"""
    import mujoco

    ogeoms = _part_geoms(model, obj)
    buf = np.zeros(6)
    total = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 in ogeoms or c.geom2 in ogeoms:
            mujoco.mj_contactForce(model, data, i, buf)
            total += abs(float(buf[0]))
    return total


def search_grasp(model, data, side: str, obj: str, verbose: bool = True):
    """逐个深度实际合一次爪,挑真正还留着夹持力的那个。

    分析式地挑摆位一直挑不对:靠喉部会被挤进去(法向力 59N -> 0.95N),靠指尖
    又会从尖端滑脱(合爪滑移 68.9mm)。钳口是 V 形加曲面,可用区间是哪一段,
    算不如试 —— 每个深度合一次爪、静置、量残余法向力,几十秒就有答案,而且
    这张表本身就是有用的信息:它说明策略必须把夹爪对到哪一段才抓得住。
    """
    profile = []
    if verbose:
        print(f"{'深度':>7}{'初始间隙mm':>12}{'合爪后力N':>11}{'静置后力N':>11}{'滑移mm':>9}")
    for frac in np.linspace(0.40, 1.00, 13):
        # 每个深度都要从"复位 + 张开"开始搜。上一轮 _close_on 把夹爪留在合拢
        # 状态,而 grasp_pose 拿 data.qpos[gripper] 当张开角 —— 不复位的话从第
        # 二个深度起每个候选都判成"张开时就嵌进钳口",整张表只剩第一行。
        _reset_open(model, data, side)
        try:
            pos, quat, gap = grasp_pose(model, data, side, obj, depths=[frac])
        except SystemExit:
            continue
        r = _close_on(model, data, side, obj, pos, quat)
        if r is None:
            continue
        f_close, f_settle, slip = r
        profile.append((f_settle, frac, pos, quat, gap))
        if verbose:
            print(f"{frac:>7.2f}{gap:>12.2f}{f_close:>11.2f}{f_settle:>11.2f}{slip:>9.1f}")
    if not profile:
        raise SystemExit(f"所有深度都夹不住 {obj} —— 是几何问题,不是接触参数问题")
    profile.sort(key=lambda t: -t[0])
    best = profile[0]
    if verbose:
        print(f"  -> 选深度 {best[1]:.2f}(残余夹持力 {best[0]:.2f} N)\n")
    return best[2], best[3]


def _reset_open(model, data, side: str) -> None:
    """复位到 home 并把夹爪张到底,停稳。

    ``grasp_pose`` 把 ``data.qpos[gripper]`` 当作"张开角"读进去,所以调它之前
    必须先真的张开。否则搜索是在夹爪当前(可能是合拢的)姿态下做的,每个候选
    都会判成"张开时就已经嵌进钳口"而被跳过。
    """
    import mujoco

    ga = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_gripper")
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for i in range(model.nu):
        data.ctrl[i] = data.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]]
    data.ctrl[ga] = GRIPPER_OPEN
    for _ in range(int(0.4 / model.opt.timestep)):
        mujoco.mj_step(model, data)


def _close_on(model, data, side: str, obj: str, pos, quat):
    """摆好零件后合爪,返回 (合爪后力, 静置后力, 滑移mm);没夹到返回 None。"""
    import mujoco

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj)
    ojoint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{obj}_free")
    badr, bvadr = model.jnt_qposadr[ojoint], model.jnt_dofadr[ojoint]

    _reset_open(model, data, side)

    data.qpos[badr:badr + 3] = pos
    data.qpos[badr + 3:badr + 7] = quat
    data.qvel[bvadr:bvadr + 6] = 0
    mujoco.mj_forward(model, data)
    start = data.xpos[bid].copy()

    ok = _ramp_closed(model, data, side, obj)
    if not ok:
        return None
    f_close = grip_force(model, data, obj)
    for _ in range(int(0.5 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    return f_close, grip_force(model, data, obj), float(np.linalg.norm(data.xpos[bid] - start) * 1000)


def _ramp_closed(model, data, side: str, obj: str) -> bool:
    """关重力合爪,两片钳口都接触上就恢复重力。返回是否夹到。"""
    import mujoco

    ga = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_gripper")
    ogeoms = _part_geoms(model, obj)
    _, _, fixed, moving = _jaw_geoms(model, side)
    fixed_set, moving_set = set(fixed), set(moving)

    def touching() -> bool:
        f = mv = False
        for i in range(data.ncon):
            c = data.contact[i]
            if c.dist > 5e-4:
                continue
            if c.geom1 in ogeoms:
                other = c.geom2
            elif c.geom2 in ogeoms:
                other = c.geom1
            else:
                continue
            f |= other in fixed_set
            mv |= other in moving_set
        return f and mv

    gravity = model.opt.gravity.copy()
    model.opt.gravity[:] = 0
    stroke = abs(GRIPPER_OPEN - GRIPPER_CLOSED)
    hold = int(stroke / GRIPPER_CLOSE_SPEED / model.opt.timestep)
    gripped = False
    try:
        for k in range(hold):
            data.ctrl[ga] = GRIPPER_OPEN + (GRIPPER_CLOSED - GRIPPER_OPEN) * (k + 1) / hold
            mujoco.mj_step(model, data)
            if not gripped and touching():
                gripped = True
                model.opt.gravity[:] = gravity
    finally:
        model.opt.gravity[:] = gravity
    return gripped


def run(model, data, side: str, obj: str, render_to: Path | None = None,
        pose=None) -> Result:
    import mujoco

    gj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_gripper")
    gadr = model.jnt_qposadr[gj]
    ga = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_gripper")
    la = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_shoulder_lift")
    wa = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_wrist_flex")
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj)
    badr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{obj}_free")]
    bvadr = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{obj}_free")]

    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.ctrl[:] = data.qpos[:model.nu] if model.nu <= model.nq else 0
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        data.ctrl[i] = data.qpos[model.jnt_qposadr[jid]]

    # 张爪,让钳口先到位再摆零件
    data.ctrl[ga] = GRIPPER_OPEN
    for _ in range(int(0.4 / model.opt.timestep)):
        mujoco.mj_step(model, data)

    centre, quat = pose if pose is not None else grasp_pose(model, data, side, obj)[:2]
    data.qpos[badr:badr + 3] = centre
    data.qpos[badr + 3:badr + 7] = quat
    data.qvel[bvadr:bvadr + 6] = 0
    mujoco.mj_forward(model, data)
    start = data.xpos[bid].copy()

    max_speed = 0.0

    def settle(steps: int) -> None:
        nonlocal max_speed
        for _ in range(steps):
            mujoco.mj_step(model, data)
            max_speed = max(max_speed, float(np.linalg.norm(data.cvel[bid][3:])))

    # --- 合爪 ---
    # 合爪期间零件先按住不放:钳口从张开合到夹住要 0.5s,松着的话零件早就
    # 自由落体掉到桌面上了(实测 0.1s 就落了 5cm),测的根本不是夹持。等两片
    # 钳口都接触上再松手,之后的运动才完全由接触力决定。
    _, _, fixed, moving = _jaw_geoms(model, side)
    fixed_set, moving_set = set(fixed), set(moving)
    ogeoms = _part_geoms(model, obj)

    def touching() -> tuple[bool, bool]:
        f = mv = False
        for i in range(data.ncon):
            c = data.contact[i]
            if c.dist > 5e-4:
                continue
            if c.geom1 in ogeoms:
                other = c.geom2
            elif c.geom2 in ogeoms:
                other = c.geom1
            else:
                continue
            f |= other in fixed_set
            mv |= other in moving_set
        return f, mv

    # 合爪期间先关重力:钳口从张开合到夹住要 0.6s,松着的话零件早自由落体掉到
    # 桌面上了(实测 0.1s 落 5cm),测的根本不是夹持。用瞬移按住则更糟 —— 每步
    # 重置 qpos 会把接触求解器的冲量丢掉,活动钳口能一路穿进零件 12mm。关重力
    # 不动接触求解,是这里唯一干净的做法。夹到之后立刻恢复,后面全靠接触力。
    gravity = model.opt.gravity.copy()
    model.opt.gravity[:] = 0

    # 合爪速度必须和 _ramp_closed 用同一个常量。这里曾经写死 0.6 秒走完全行程
    # (3.97 rad/s),比真机峰值还快 40% —— 接触调硬之后它会在两片钳口碰上之前
    # 就把零件弹开,于是整套测试报"摆位不对",而摆位其实是好的(搜索阶段同一个
    # 位姿量到 48.8 N 残余夹持力)。
    hold = int(abs(GRIPPER_OPEN - GRIPPER_CLOSED) / GRIPPER_CLOSE_SPEED / model.opt.timestep)
    gripped = False
    for k in range(hold):
        data.ctrl[ga] = GRIPPER_OPEN + (GRIPPER_CLOSED - GRIPPER_OPEN) * (k + 1) / hold
        mujoco.mj_step(model, data)
        if not gripped and all(touching()):
            gripped = True
            model.opt.gravity[:] = gravity
        if gripped:
            max_speed = max(max_speed, float(np.linalg.norm(data.cvel[bid][3:])))
    model.opt.gravity[:] = gravity
    if not gripped:
        return Result(0, 0, 0, 0, False,
                      note="合爪合到底两片钳口没同时接触 —— 摆位不对,不是接触参数问题")
    settle(int(0.5 / model.opt.timestep))
    grasped = data.xpos[bid].copy()
    close_slip = float(np.linalg.norm(grasped - start) * 1000)

    if render_to is not None:
        _render(model, data, centre, render_to)

    # --- 抬升:抬肩,零件应当跟着走 ---
    lift_start_obj = grasped.copy()
    lift_start_tcp = pinch_center(model, data, side)[0]
    target = data.ctrl[la] - 0.35
    ramp = int(1.0 / model.opt.timestep)
    for k in range(ramp):
        data.ctrl[la] = data.ctrl[la] + (target - data.ctrl[la]) / max(ramp - k, 1)
        mujoco.mj_step(model, data)
        max_speed = max(max_speed, float(np.linalg.norm(data.cvel[bid][3:])))
    settle(int(0.5 / model.opt.timestep))
    tcp_moved = pinch_center(model, data, side)[0] - lift_start_tcp
    lift_slip = float(np.linalg.norm((data.xpos[bid] - lift_start_obj) - tcp_moved) * 1000)

    # --- 摇晃:惯性力才能分辨"夹住"和"卡住" ---
    shake_start_obj = data.xpos[bid].copy()
    shake_start_tcp = pinch_center(model, data, side)[0]
    base = data.ctrl[wa]
    for k in range(int(2.0 / model.opt.timestep)):
        t = k * model.opt.timestep
        data.ctrl[wa] = base + SHAKE_AMPLITUDE * math.sin(2 * math.pi * SHAKE_HZ * t)
        mujoco.mj_step(model, data)
        max_speed = max(max_speed, float(np.linalg.norm(data.cvel[bid][3:])))
    data.ctrl[wa] = base
    settle(int(0.5 / model.opt.timestep))
    tcp_moved = pinch_center(model, data, side)[0] - shake_start_tcp
    shake_slip = float(np.linalg.norm((data.xpos[bid] - shake_start_obj) - tcp_moved) * 1000)

    # 还在钳口附近就算抓住 —— 掉了的话零件会落到桌面或地上
    final_gap = float(np.linalg.norm(data.xpos[bid] - pinch_center(model, data, side)[0]) * 1000)
    held = final_gap < 40.0
    return Result(close_slip, lift_slip, shake_slip, max_speed, held,
                  note=f"末态离钳口 {final_gap:.0f}mm")


def _render(model, data, lookat, path: Path) -> None:
    import mujoco

    try:
        import cv2
    except ImportError:
        print("  (装不上 cv2,跳过出图)")
        return
    r = mujoco.Renderer(model, 480, 640)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance = 0.18
    cam.elevation = -15
    tiles = []
    for az in (200, 290):
        cam.azimuth = az
        r.update_scene(data, cam)
        tiles.append(r.render())
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.hstack(tiles)[:, :, ::-1])
    print(f"  夹持瞬间已存到 {path}")


def load_config() -> dict:
    cfg = {"condim": 4, "friction": [1.5, 0.05, 0.0005],
           "part_friction": [0.5, 0.017, 0.0005], "solref": [0.01, 1.0],
           "impratio": 10.0, "gripper_force_limit": 0.981}
    if GRASP_CONFIG.is_file():
        cfg.update({k: v for k, v in json.loads(GRASP_CONFIG.read_text()).items()
                    if k in cfg})
    return cfg


#: 扫描网格。每项单独扫,不做全组合 —— 全组合上千次,且这些量的影响基本独立,
#: 逐项扫已经能看出各自的方向。
SWEEP = {
    "condim": [3, 4, 6],
    "friction": [[0.6, 0.005, 0.0001], [1.0, 0.02, 0.0002],
                 [1.5, 0.05, 0.0005], [2.5, 0.1, 0.001]],
    # part_friction 不扫:它由"杆能不能从孔里拔出来"定,而这个扫描只看夹持,
    # 完全测不到。抬高它还会让合爪时零件嵌进钳口(4.0/0.2 实测 -7.19mm),
    # 而且不单调 —— 2.0 穿、2.5 不穿、3.0 又穿。见 assets.GraspConfig.part_friction。
    # 时间常数下限是 2 倍步长(0.004)。曾经把 0.002 放进来过,扫描只看滑移量
    # 不看稳定性,那个发散的配置在当次测试里恰好没炸就被选中了 —— 结果是遥操
    # 里零件被弹飞、还能穿过桌子。坏值不该进网格,GraspConfig.validate 兜底。
    "solref": [[0.04, 1.0], [0.02, 1.0], [0.01, 1.0], [0.006, 1.0]],
    "impratio": [1.0, 5.0, 10.0, 20.0],
}
# gripper_force_limit 不扫:它不是"调到抓得住"的旋钮,而是仿真里代替真机柔性的
# 一个上限。真机钳口/连杆/塑料减速箱都会让,所以舵机堵转 2.9 N·m 时零件受到的
# 只有十几牛;仿真里钳口是刚体、伺服是刚性的,给多少力矩就全压在零件上。
# 曾经照抄堵转值填 3.0,夹持力算出来 86 N —— 20g 的零件受 86N 是 4300 m/s²,
# 一个 2ms 步长就到 8.6 m/s,于是"夹准了还被弹开",弹出去还能一步穿过台面板。
# 现值 0.6 由固定位姿实测定:17.4 N 夹持力,抬升和摇晃都过,0.4 就掉。


def main() -> int:
    import mujoco

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=["left", "right"], default="left")
    p.add_argument("--object", choices=["bolt", "socket"], default="bolt")
    p.add_argument("--sweep", action="store_true", help="逐项扫接触参数")
    p.add_argument("--apply", action="store_true", help="把扫出的最优写回 configs/grasp.json")
    p.add_argument("--render", action="store_true", help="存一张夹持瞬间的图")
    args = p.parse_args()

    if not SCENE.is_file():
        raise SystemExit(f"场景不存在 {SCENE},先跑 mj_server.py --build")

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    cfg = load_config()

    apply_contact(model, cfg)
    print(f"用物理搜 {args.arm} 臂抓 {args.object} 的摆位:")
    pose = search_grasp(model, data, args.arm, args.object)

    if not args.sweep:
        out = REPO_ROOT / "outputs" / f"grasp_{args.arm}_{args.object}.png" if args.render else None
        r = run(model, data, args.arm, args.object, out, pose=pose)
        print(f"{args.arm} 臂抓 {args.object}")
        print(f"  合爪滑移 {r.close_slip:6.1f} mm")
        print(f"  抬升滑移 {r.lift_slip:6.1f} mm")
        print(f"  摇晃滑移 {r.shake_slip:6.1f} mm")
        print(f"  峰值速度 {r.max_speed:6.2f} m/s")
        ej, pen = eject_speed(model, data, args.arm, args.object, pose=pose)
        print(f"  弹出速度 {ej:6.2f} m/s   (实际穿透 {pen:.2f}mm,上限 {EJECT_LIMIT:g} m/s)")
        print(f"  判定: {r.verdict}  ({r.note})")
        return 0 if r.held else 1

    print(f"逐项扫描({args.arm} 臂抓 {args.object}),基线 = 当前配置\n")
    best = dict(cfg)
    for key, values in SWEEP.items():
        print(f"--- {key} ---")
        print(f"{'取值':<26}{'合爪':>8}{'抬升':>8}{'摇晃':>8}{'弹出v':>8}{'穿透':>8}  判定")
        scored = []
        for v in values:
            trial = dict(best)
            trial[key] = v
            apply_contact(model, trial)
            r = run(model, data, args.arm, args.object, pose=pose)
            ej, pen = eject_speed(model, data, args.arm, args.object, pose=pose)
            # 排序:先剔掉会弹射的(遥操里夹爪怼进零件是常态,那时候不能炸),
            # 再剔掉抓不住的,最后才比滑移量。只比滑移量会选中发散的配置。
            scored.append(((ej > EJECT_LIMIT, not r.held, r.shake_slip + r.lift_slip), v, r))
            print(f"{str(v):<26}{r.close_slip:>8.1f}{r.lift_slip:>8.1f}"
                  f"{r.shake_slip:>8.1f}{ej:>8.2f}{pen:>8.2f}  {r.verdict}"
                  + ("  弹射!" if ej > EJECT_LIMIT else ""))
        scored.sort(key=lambda t: t[0])
        best[key] = scored[0][1]
        print(f"  -> 选 {scored[0][1]}\n")

    apply_contact(model, best)
    r = run(model, data, args.arm, args.object, pose=pose)
    ej, pen = eject_speed(model, data, args.arm, args.object, pose=pose)
    print("最终配置:")
    for k, v in best.items():
        print(f"  {k:<22}{v}")
    print(f"  合爪 {r.close_slip:.1f} / 抬升 {r.lift_slip:.1f} / 摇晃 {r.shake_slip:.1f} mm"
          f"   弹出 {ej:.2f} m/s (穿透 {pen:.2f}mm)   判定 {r.verdict}")
    if ej > EJECT_LIMIT:
        print(f"  警告: 弹出速度超过 {EJECT_LIMIT} m/s,遥操里怼进零件会被弹飞")

    if args.apply:
        GRASP_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        GRASP_CONFIG.write_text(json.dumps(best, indent=2, ensure_ascii=False))
        print(f"\n已写回 {GRASP_CONFIG}")
        print("重建场景生效: python src/evo_rlt/sim/mj_server.py --build")
    else:
        print("\n加 --apply 写回配置")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
