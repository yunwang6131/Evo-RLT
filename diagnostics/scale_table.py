#!/usr/bin/env python
"""把桌子 STL 的腿加长,生成一个新文件,不动原件。

    python diagnostics/scale_table.py --raise-mm 30

桌面板(z 0~15mm)保持原样,只把 z<0 的部分按比例拉伸 —— 横撑等结构会跟着
等比下移,而不是被平移压扁。法向量交给 MuJoCo 重算。
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

DEFAULT_SRC = Path("data/桌子.STL")


def read_binary_stl(path: Path) -> tuple[bytes, np.ndarray, np.ndarray]:
    """返回 (80 字节头, 法向量 Nx3, 顶点 Nx3x3)。"""
    raw = path.read_bytes()
    if raw[:5] == b"solid" and b"facet" in raw[:500]:
        raise SystemExit(f"{path} 是 ASCII STL,本工具只处理二进制 STL")
    header = raw[:80]
    count = struct.unpack("<I", raw[80:84])[0]
    normals = np.empty((count, 3), dtype=np.float32)
    tris = np.empty((count, 3, 3), dtype=np.float32)
    for i in range(count):
        off = 84 + i * 50
        vals = struct.unpack("<12f", raw[off : off + 48])
        normals[i] = vals[0:3]
        tris[i] = np.array(vals[3:12]).reshape(3, 3)
    return header, normals, tris


def write_binary_stl(path: Path, header: bytes, normals: np.ndarray, tris: np.ndarray) -> None:
    out = bytearray(header[:80].ljust(80, b"\0"))
    out += struct.pack("<I", len(tris))
    for n, t in zip(normals, tris):
        out += struct.pack("<12fH", *n, *t.reshape(-1), 0)
    path.write_bytes(bytes(out))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--raise-mm", type=float, default=30.0, help="加高多少毫米")
    parser.add_argument("--split-z", type=float, default=0.0,
                        help="这个 z 以下算桌腿(毫米),默认 0 = 桌面板底面")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    header, normals, tris = read_binary_stl(args.src)
    # 必须拷贝:tris[:, :, 2] 是视图,原地改 tris 后它也跟着变,
    # 打印出来的"原件尺寸"就会是改后的值。
    z = tris[:, :, 2].copy()
    leg_depth = -z.min()
    if leg_depth <= 0:
        raise SystemExit(f"{args.src} 没有 z<0 的部分,无法识别桌腿")

    factor = (leg_depth + args.raise_mm) / leg_depth
    below = z < args.split_z
    tris[:, :, 2] = np.where(below, z * factor, z)

    new_z = tris[:, :, 2]
    out = args.out or args.src.with_name(
        f"{args.src.stem}_h{int(new_z.max() - new_z.min())}.STL"
    )
    write_binary_stl(out, header, normals, tris)

    print(f"原件 {args.src.name}: z {z.min():.1f} ~ {z.max():.1f} mm  (高 {z.max()-z.min():.1f})")
    print(f"新件 {out.name}: z {new_z.min():.1f} ~ {new_z.max():.1f} mm  (高 {new_z.max()-new_z.min():.1f})")
    print(f"桌腿拉伸系数 {factor:.4f},桌面板({args.split_z:g}mm 以上)未改动")
    print(f"\n把 configs/task_scene.json 的 table.pos[2] 加 {args.raise_mm/1000:.3f},")
    print("并把 assets.py 的 TASK_MESHES['table'] 指向新文件。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
