#!/usr/bin/env python
"""把带孔洞的网格做凸分解,供 MuJoCo 用作碰撞几何。

    python diagnostics/decompose_mesh.py data/桌子_h119.STL

MuJoCo 的 mesh 碰撞取**凸包**,桌面上的孔和凹槽在物理层面会被填平 —— 螺栓插
不进去,反而被接触力顶出来。插销任务里孔就是任务本身,必须真实存在,所以把网格
拆成若干凸块,每块单独作为碰撞 geom,合起来才能围出凹陷。

视觉仍用原始网格(一个 geom,group 1),碰撞用这些凸块(group 3,不参与渲染),
两者互不干扰。

分解出的块数直接决定碰撞开销。``--threshold`` 越小越忠实、块越多:
0.02 左右能保住小孔,0.1 会把细节抹平。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def decompose(src: Path, out_dir: Path, threshold: float, max_hulls: int) -> list[Path]:
    import coacd
    import trimesh

    mesh = trimesh.load(src, force="mesh")
    print(f"原网格: {len(mesh.vertices)} 顶点, {len(mesh.faces)} 面")
    print(f"  水密(watertight): {mesh.is_watertight}")

    coacd.set_log_level("error")
    parts = coacd.run_coacd(
        coacd.Mesh(mesh.vertices, mesh.faces),
        threshold=threshold,
        max_convex_hull=max_hulls,
    )
    print(f"分解为 {len(parts)} 个凸块 (threshold={threshold})")

    out_dir.mkdir(parents=True, exist_ok=True)
    # 先清空旧产物。块数变少时(比如 threshold 调粗,64 -> 25)残留的
    # hull025..hull063 不会被覆盖,而 assets.py 是 glob("*.STL") —— 新旧混着
    # 读进场景,碰撞几何就是两次分解的叠加,而且看不出来。
    stale = sorted(out_dir.glob(f"{src.stem}_hull*.STL"))
    for path in stale:
        path.unlink()
    if stale:
        print(f"清掉上一轮的 {len(stale)} 个凸块")

    written = []
    for i, (verts, faces) in enumerate(parts):
        piece = trimesh.Trimesh(vertices=np.asarray(verts), faces=np.asarray(faces))
        path = out_dir / f"{src.stem}_hull{i:03d}.STL"
        piece.export(path)
        written.append(path)
    total = sum(p.stat().st_size for p in written)
    print(f"写入 {out_dir}/  共 {total/1024:.0f} KB")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="默认 <src 同级>/<stem>_hulls/")
    parser.add_argument("--threshold", type=float, default=0.02,
                        help="凹陷容差,越小越忠实、块越多(默认 0.02)")
    parser.add_argument("--max-hulls", type=int, default=64)
    args = parser.parse_args()

    if not args.src.is_file():
        raise SystemExit(f"找不到 {args.src}")
    out_dir = args.out_dir or args.src.parent / f"{args.src.stem}_hulls"
    decompose(args.src, out_dir, args.threshold, args.max_hulls)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
