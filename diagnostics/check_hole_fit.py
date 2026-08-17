#!/usr/bin/env python
"""量出台面上各孔的实际通径,判断螺栓杆能否插入。

    python diagnostics/check_hole_fit.py

凸分解会把孔壁向内近似,通径必然小于原始几何(实测啃掉约 1.5mm)。杆比孔粗哪怕
0.3mm,复位瞬间就会嵌进孔壁,接触力把螺栓顶出来 —— 而现象看起来像"位置没放对",
很容易查错方向。所以分解完必须实测一次。

直接扫**已构建的场景**,不硬编码桌子位姿和孔位:那些值改过好几轮(桌子从
x=0.25 挪到 0.20、孔从 0.27 到 0.22),写死的话这个工具会一本正经地给出错误结论。

用射线自上而下扫台面:能穿到台面板底以下的点即孔的通路,其最大内接圆半径就是
实际通径。螺栓杆半径也从网格里量,不写死。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SCENE = Path("~/.cache/evo_rlt/sim_assets/scene.xml").expanduser()

#: 扫描区比台面碰撞体的包围盒外扩多少(米)。必须留出一圈,因为"孔"的判据是
#: 四周被台面围住(见 scan()),贴着扫描区边界的簇会被当成台面外侧丢掉 ——
#: 不外扩的话最外圈的孔会被误判。
SCAN_MARGIN = 0.02

#: 判定"够用"的最小单边间隙(毫米)
GOOD_CLEARANCE_MM = 0.3


def plate_extent(model, data) -> tuple[float, float]:
    """台面板的顶面和底面 z。取桌子碰撞凸块里最高的那层。"""
    import mujoco

    name = lambda i: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
    spans = []
    for g in range(model.ngeom):
        if not name(g).startswith("worktable_col"):
            continue
        mid = model.geom_dataid[g]
        a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        w = data.geom_xpos[g] + model.mesh_vert[a:a + n] @ data.geom_xmat[g].reshape(3, 3).T
        spans.append((w[:, 2].min(), w[:, 2].max()))
    if not spans:
        raise SystemExit("场景里没有桌子的碰撞凸块,先重建场景")
    top = max(hi for _, hi in spans)
    bottom = min(lo for lo, hi in spans if hi > top - 0.001)
    return bottom, top


def shaft_radius(model) -> float:
    """螺栓圆柱杆的半径(毫米)。取网格上半段(离头最远那 1/3)的最大径向距离。"""
    import mujoco

    mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, "task_bolt")
    a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    v = (model.mesh_vert[a:a + n] + model.mesh_pos[mid]) * 1000
    cut = v[:, 2].min() + (v[:, 2].max() - v[:, 2].min()) * 0.66
    return float(np.linalg.norm(v[v[:, 2] > cut][:, :2], axis=1).max())


def scan_area(model, data) -> tuple[tuple[float, float], tuple[float, float]]:
    """扫描范围,从台面碰撞体的实际包围盒推出来。

    曾经写死成 x(0.14,0.32)/y(-0.14,0.14)。桌子从 x=0.20 前移到 0.30 之后,
    孔跑到了扫描区外,工具报"台面上没扫到孔" —— 看起来像凸分解把孔填平了,
    实际只是没扫到。位姿在 configs/task_scene.json 里是可改的,所以不能写死。
    """
    import mujoco

    name = lambda i: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
    pts = []
    for g in range(model.ngeom):
        if not name(g).startswith("worktable_col"):
            continue
        mid = model.geom_dataid[g]
        a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        v = model.mesh_vert[a:a + n] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        pts.append(v[:, :2])
    if not pts:
        raise SystemExit("场景里没有 worktable_col* 碰撞几何,先跑 mj_server.py --build")
    xy = np.vstack(pts)
    lo, hi = xy.min(0) - SCAN_MARGIN, xy.max(0) + SCAN_MARGIN
    return (float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1]))


def scan(model, data, bottom: float, step: float) -> list[tuple[np.ndarray, float]]:
    """扫出台面上所有孔的 (中心, 通路半径毫米)。"""
    import mujoco

    scan_x, scan_y = scan_area(model, data)
    group = np.zeros(6, np.uint8)
    group[3] = 1
    geomid = np.zeros(1, np.int32)
    hits = []
    for x in np.arange(*scan_x, step):
        for y in np.arange(*scan_y, step):
            dist = mujoco.mj_ray(model, data, np.array([x, y, 0.5]),
                                 np.array([0.0, 0.0, -1.0]), group, 1, -1, geomid)
            z = 0.5 - dist if dist >= 0 else -1.0
            if z < bottom + 0.0005:      # 射线穿过了台面板 = 该点是通路
                hits.append((x, y))
    if not hits:
        return []

    pts = np.array(hits)
    used = np.zeros(len(pts), bool)
    holes = []
    for i in range(len(pts)):
        if used[i]:
            continue
        near = np.linalg.norm(pts - pts[i], axis=1) < 0.020
        used |= near
        cluster = pts[near]
        if len(cluster) < 4:
            continue
        # 台面**边缘**外侧的点也会被射线穿过,和孔无法凭"能穿过"区分。孔的定义
        # 是四周被台面围住,所以检查这一簇有没有贴到扫描区的边界:贴到了就是
        # 走出了台面,不是孔。
        lo, hi = cluster.min(0), cluster.max(0)
        if (lo[0] <= scan_x[0] + step or hi[0] >= scan_x[1] - 2 * step
                or lo[1] <= scan_y[0] + step or hi[1] >= scan_y[1] - 2 * step):
            continue
        centre = cluster.mean(0)
        holes.append((centre, float(np.linalg.norm(cluster - centre, axis=1).max() * 1000)))
    return holes


def main() -> int:
    import mujoco

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step-mm", type=float, default=0.4, help="射线扫描步长")
    args = p.parse_args()

    if not SCENE.is_file():
        raise SystemExit(f"场景不存在 {SCENE},先跑 mj_server.py --build --benchmark")

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    # 零件挪开再扫。螺栓初始就插在其中一个孔里,不挪的话射线打在它身上穿不过去,
    # 那个孔会凭空消失 —— 而这恰恰是最需要量的那个孔。
    for part in ("socket", "bolt"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{part}_free")
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid] + 2] += 5.0
    mujoco.mj_forward(model, data)

    bottom, top = plate_extent(model, data)
    radius = shaft_radius(model)
    print(f"台面板 z {bottom:.4f} ~ {top:.4f}  (厚 {(top-bottom)*1000:.1f} mm)")
    print(f"螺栓杆半径 {radius:.2f} mm —— 孔的通路半径必须大于它\n")

    holes = scan(model, data, bottom, args.step_mm / 1000.0)
    # 大圆凹槽不是通孔,扫不出来;这里剩下的都是小孔
    holes = [h for h in holes if h[1] < 15.0]
    if not holes:
        print("台面上没扫到孔 —— 凸分解可能把孔全填平了")
        return 1

    print(f"{'孔中心 (x, y)':>22}{'通路半径mm':>12}{'间隙mm':>9}{'判定':>8}")
    print("-" * 52)
    ok_any = False
    for centre, r in sorted(holes, key=lambda h: -h[1]):
        gap = r - radius
        verdict = "可以" if gap > GOOD_CLEARANCE_MM else ("勉强" if gap > 0 else "不行")
        ok_any |= gap > GOOD_CLEARANCE_MM
        print(f"  ({centre[0]:7.4f}, {centre[1]:7.4f}){r:>12.2f}{gap:>9.2f}{verdict:>8}")

    print()
    if ok_any:
        print("有孔的通径足够,螺栓可以插入。")
    else:
        print("所有孔都太窄。用 widen_holes.py --extra-mm 加大后重新分解 ——\n"
              "分解会把孔壁向内啃约 1.5mm,原始几何要多留出这个量。")
    return 0 if ok_any else 1


if __name__ == "__main__":
    raise SystemExit(main())
