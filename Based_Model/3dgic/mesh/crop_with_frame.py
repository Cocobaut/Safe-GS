"""
Cắt mesh GOF quanh mặt bàn bằng vùng cắt đã có sẵn (*_crop.json của crop_table.py).

3DGIC và InFusion dùng chung ảnh crop + COLMAP sparse/0 của LERF teatime -> cùng hệ toạ độ, nên mặt phẳng bàn
(plane_normal, plane_d), đường bao 2D (hull_2d trên basis_e1/e2) và khoảng chiều cao dùng lại được trực tiếp.
Nới thêm --extra_margin (ngang), --extra_up, --extra_down; giữ mọi tam giác có cả 3 đỉnh trong lăng trụ.

    python crop_with_frame.py --mesh mesh_gof.ply --frame_json mesh_gof_table_v2_crop.json --output_dir <thư mục mesh>
"""
import json
import os
import sys
from argparse import ArgumentParser

import numpy as np
import trimesh
from scipy.spatial import Delaunay

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gof.io_utils import rel_to, unique_path  # noqa: E402


def main():
    ap = ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--frame_json", required=True, help="*_crop.json chứa plane_normal, plane_d, hull_2d, basis_e1/e2, height_range")
    ap.add_argument("--extra_margin", type=float, default=0.08, help="Nới đường bao bàn theo chiều ngang (đơn vị scene)")
    ap.add_argument("--extra_up", type=float, default=0.08, help="Nới thêm phía trên vật cao nhất")
    ap.add_argument("--extra_down", type=float, default=0.08, help="Nới thêm phía dưới mặt bàn")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--name", default="mesh_gof_table.ply")
    args = ap.parse_args()

    fr = json.loads(open(args.frame_json).read())
    n, d = np.array(fr["plane_normal"]), float(fr["plane_d"])
    e1, e2 = np.array(fr["basis_e1"]), np.array(fr["basis_e2"])
    hull = np.array(fr["hull_2d"])
    dirs = hull - hull.mean(axis=0)
    hull = hull + dirs / np.linalg.norm(dirs, axis=1, keepdims=True) * args.extra_margin
    lo, hi = fr["height_range"][0] - args.extra_down, fr["height_range"][1] + args.extra_up
    tri = Delaunay(hull)
    ext = hull.max(axis=0) - hull.min(axis=0)
    print(f"[i] Vùng cắt: {ext[0]:.2f} x {ext[1]:.2f} (ngang), cao {lo:.3f} .. {hi:.3f} so với mặt bàn")

    mesh = trimesh.load(args.mesh, process=False)
    P, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    print(f"[i] Mesh gốc: {len(P)} đỉnh, {len(F)} mặt")
    h = P @ n + d
    keep_v = (h >= lo) & (h <= hi)
    cand = np.nonzero(keep_v)[0]
    keep_v[cand] = tri.find_simplex(np.stack([P[cand] @ e1, P[cand] @ e2], axis=1)) >= 0
    keep_f = keep_v[F].all(axis=1)
    out = mesh.submesh([np.nonzero(keep_f)[0]], append=True)

    os.makedirs(args.output_dir, exist_ok=True)
    path = unique_path(os.path.join(args.output_dir, args.name))
    out.export(path)
    meta = {"source": rel_to(args.mesh, args.output_dir), "frame_json": rel_to(args.frame_json, args.output_dir),
            "extra_margin": args.extra_margin, "height_range": [lo, hi], "hull_2d": hull.tolist(),
            "plane_normal": n.tolist(), "plane_d": d, "basis_e1": e1.tolist(), "basis_e2": e2.tolist(),
            "vertices": len(out.vertices), "faces": len(out.faces)}
    with open(os.path.splitext(path)[0] + "_crop.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[✓] {path}: {len(out.vertices)} đỉnh, {len(out.faces)} mặt, {os.path.getsize(path) / 1024**2:.0f} MB")


if __name__ == "__main__":
    main()
