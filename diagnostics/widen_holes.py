#!/usr/bin/env python
"""把桌面上的孔扩大,给插入留出间隙。

    python diagnostics/widen_holes.py --extra-mm 1.5

为什么需要:MuJoCo 的 mesh 碰撞只支持凸形状,带孔的桌面必须先做凸分解。分解出
的凸块会把孔壁向内近似,实测通径从 12.8mm 缩到 9.2mm,而螺栓杆是 9.5mm ——
复位瞬间杆就和孔壁嵌入 3~5mm,接触力直接把螺栓弹飞。

与其把分解精度一路调高(块数暴涨、碰撞变慢),不如在几何上留余量:孔壁顶点沿
径向外推,分解后仍有足够通径。

孔心坐标在原始 STL 坐标系下给出(毫米),默认是那两个小孔。
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

#: 原始 STL 坐标系下的孔心(毫米)。四角的安装孔不动 —— 它们不参与任务。
DEFAULT_HOLES = [(-85.0, 20.0), (-55.0, 20.0)]


def read_binary_stl(path: Path):
    raw = path.read_bytes()
    if raw[:5] == b"solid" and b"facet" in raw[:500]:
        raise SystemExit(f"{path} 是 ASCII STL")
    header, count = raw[:80], struct.unpack("<I", raw[80:84])[0]
    normals = np.empty((count, 3), np.float32)
    tris = np.empty((count, 3, 3), np.float32)
    for i in range(count):
        vals = struct.unpack("<12f", raw[84 + i * 50 : 84 + i * 50 + 48])
        normals[i] = vals[0:3]
        tris[i] = np.array(vals[3:12]).reshape(3, 3)
    return header, normals, tris


def write_binary_stl(path: Path, header: bytes, normals: np.ndarray, tris: np.ndarray) -> None:
    out = bytearray(header[:80].ljust(80, b"\0")) + struct.pack("<I", len(tris))
    for n, t in zip(normals, tris):
        out += struct.pack("<12fH", *n, *t.reshape(-1), 0)
    path.write_bytes(bytes(out))


def widen(tris: np.ndarray, holes, extra_mm: float, search_mm: float) -> int:
    """把孔壁顶点沿径向外推 extra_mm。返回移动的顶点数。"""
    flat = tris.reshape(-1, 3)
    moved = 0
    for cx, cy in holes:
        radial = flat[:, :2] - np.array([cx, cy])
        dist = np.linalg.norm(radial, axis=1)
        # 只动孔壁:落在搜索半径内、且不在孔心正上(避免除零)
        on_wall = (dist > 1e-6) & (dist < search_mm)
        if not on_wall.any():
            print(f"  警告: ({cx}, {cy}) 附近没找到孔壁顶点")
            continue
        direction = radial[on_wall] / dist[on_wall, None]
        flat[on_wall, :2] += direction * extra_mm
        moved += int(on_wall.sum())
    return moved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=Path("data/桌子_h119.STL"))
    parser.add_argument("--extra-mm", type=float, default=1.5,
                        help="孔半径外扩多少毫米(默认 1.5,即直径 +3)")
    parser.add_argument("--search-mm", type=float, default=9.0,
                        help="孔心多大范围内的顶点算孔壁")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--hole", action="append", metavar="X,Y", default=None,
                        help="孔心(毫米,原始 STL 坐标),可给多次。"
                             "不给就用桌子那两个小孔。螺套内孔在原点,用 --hole 0,0")
    args = parser.parse_args()

    holes = DEFAULT_HOLES
    if args.hole:
        try:
            holes = [tuple(float(v) for v in h.split(",")) for h in args.hole]
        except ValueError:
            raise SystemExit("--hole 要写成 X,Y,例如 --hole 0,0")
        if any(len(h) != 2 for h in holes):
            raise SystemExit("--hole 要写成 X,Y,例如 --hole 0,0")

    header, normals, tris = read_binary_stl(args.src)
    before = tris.copy()
    moved = widen(tris, holes, args.extra_mm, args.search_mm)

    out = args.out or args.src.with_name(f"{args.src.stem}_wide.STL")
    write_binary_stl(out, header, normals, tris)

    for cx, cy in holes:
        for label, arr in (("原", before), ("新", tris)):
            flat = arr.reshape(-1, 3)
            d = np.linalg.norm(flat[:, :2] - np.array([cx, cy]), axis=1)
            wall = d[(d > 1e-6) & (d < args.search_mm + args.extra_mm + 1)]
            if wall.size:
                print(f"  孔({cx:6.1f},{cy:5.1f}) {label}: 壁半径 {wall.min():.2f} ~ {wall.max():.2f} mm")
    print(f"\n移动 {moved} 个顶点,写入 {out}")
    print("接着重新凸分解:")
    print(f"  python diagnostics/decompose_mesh.py {out} --threshold 0.005 --max-hulls 256")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
