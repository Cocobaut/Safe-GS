"""
Cắt mesh quanh vị trí của một vật (kèm một phần cảnh xung quanh) - dùng cho mesh sau khi đã xóa vật.

Vật đã bị xóa khỏi mesh mới nên vị trí của nó lấy từ mesh THAM CHIẾU (trước khi xóa):
  1. Đỉnh của mesh tham chiếu nhìn thấy (depth render tham chiếu) và rơi vào label SAM2 của vật (--ids)
     trên các view thấy vật rõ nhất -> điểm 3D của vật.
  2. Hộp bao theo hệ trục mặt bàn (e1, e2, n từ *_crop.json của crop_table.py; nếu không có thì theo trục toạ độ),
     lấy phân vị 1-99% cho chắc, nới thêm --margin theo ngang và --margin_up / --margin_down theo chiều đứng.
  3. Giữ mọi tam giác của mesh MỚI có cả 3 đỉnh trong hộp (không giảm lưới, không lọc mảnh rời).

    python crop_region.py --mesh <mesh mới> --ref_mesh <mesh trước khi xóa> --ref_render_dir <gs/train/ours_150 tham chiếu>
                          --labels_dir <mask/labels> --transforms <transforms.json> --ids 2
                          --frame_json <mesh_gof_table_v2_crop.json> --output_dir <thư mục>
"""
import json
import os
import sys
from argparse import ArgumentParser

import cv2
import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crop_table import crop_boxes, visible_labels  # noqa: E402
from gof.io_utils import rel_to, unique_path  # noqa: E402


def top_views(labels_dir, boxes, ids, k):
    """Các view có diện tích label (sau crop) của vật lớn nhất."""
    scores = []
    for f in sorted(os.listdir(labels_dir)):
        stem = os.path.splitext(f)[0]
        if stem not in boxes:
            continue
        x0, y0, x1, y1 = boxes[stem]
        lab = cv2.imread(os.path.join(labels_dir, f), cv2.IMREAD_UNCHANGED)[y0:y1, x0:x1]
        scores.append((np.isin(lab, ids).mean(), stem))
    return [s for a, s in sorted(scores, reverse=True)[:k] if a > 0]


def main():
    ap = ArgumentParser()
    ap.add_argument("--mesh", required=True, help="Mesh cần cắt (sau khi xóa vật)")
    ap.add_argument("--ref_mesh", required=True, help="Mesh tham chiếu còn vật (để định vị vật)")
    ap.add_argument("--ref_render_dir", required=True, help="gs/train/ours_<iter> của lần chạy tham chiếu")
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--transforms", required=True)
    ap.add_argument("--ids", required=True, help="ID SAM2 của vật, ví dụ 2")
    ap.add_argument("--views", default="", help="Mặc định: --n_views view thấy vật rõ nhất")
    ap.add_argument("--n_views", type=int, default=8)
    ap.add_argument("--frame_json", default="", help="*_crop.json của crop_table.py để lấy hệ trục mặt bàn")
    ap.add_argument("--margin", type=float, default=0.3, help="Nới hộp theo phương ngang (đơn vị scene)")
    ap.add_argument("--margin_up", type=float, default=0.15)
    ap.add_argument("--margin_down", type=float, default=0.3)
    ap.add_argument("--depth_tol", type=float, default=0.02)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--name", default="mesh_gof_region.ply")
    args = ap.parse_args()

    ids = [int(x) for x in args.ids.split(",") if x]
    boxes = crop_boxes(args.transforms)
    views = [v for v in args.views.split(",") if v] or top_views(args.labels_dir, boxes, ids, args.n_views)
    print(f"[i] View định vị vật {ids}: {views}")

    # 1. Điểm 3D của vật từ mesh tham chiếu
    ref = o3d.io.read_triangle_mesh(args.ref_mesh)
    R = np.asarray(ref.vertices)
    obj = np.zeros(len(R), dtype=bool)
    for v in views:
        lab = visible_labels(R, v, args.ref_render_dir, args.labels_dir, boxes[v], args.depth_tol)
        obj |= np.isin(lab, ids)
    obj_pts = R[obj]
    del ref, R
    print(f"[i] {len(obj_pts)} đỉnh của vật trong mesh tham chiếu")

    # 2. Hộp bao theo hệ trục mặt bàn
    if args.frame_json:
        fj = json.load(open(args.frame_json))
        e1, e2, n = (np.array(fj[k]) for k in ("basis_e1", "basis_e2", "plane_normal"))
    else:
        e1, e2, n = np.eye(3)
    B = np.stack([e1, e2, n], axis=0)  # world -> (e1, e2, n)
    q = obj_pts @ B.T
    lo, hi = np.percentile(q, 1, axis=0), np.percentile(q, 99, axis=0)
    lo = lo - np.array([args.margin, args.margin, args.margin_down])
    hi = hi + np.array([args.margin, args.margin, args.margin_up])
    print(f"[i] Hộp (e1, e2, n): {lo.round(3)} .. {hi.round(3)}")

    # 3. Cắt mesh mới
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    P, F = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    print(f"[i] Mesh cần cắt: {len(P)} đỉnh, {len(F)} mặt")
    qp = P @ B.T
    keep_v = np.all((qp >= lo) & (qp <= hi), axis=1)
    keep_f = keep_v[F].all(axis=1)
    used = np.unique(F[keep_f])
    remap = -np.ones(len(P), dtype=np.int64)
    remap[used] = np.arange(len(used))
    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(P[used])
    out.triangles = o3d.utility.Vector3iVector(remap[F[keep_f]])
    if mesh.has_vertex_colors():
        out.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors)[used])
    out.compute_vertex_normals()

    os.makedirs(args.output_dir, exist_ok=True)
    path = unique_path(os.path.join(args.output_dir, args.name))
    o3d.io.write_triangle_mesh(path, out, write_ascii=False, compressed=False)
    meta = {"mesh": rel_to(args.mesh, args.output_dir), "ref_mesh": rel_to(args.ref_mesh, args.output_dir), "ids": ids, "views": views,
            "basis_e1": e1.tolist(), "basis_e2": e2.tolist(), "plane_normal": n.tolist(),
            "box_min": lo.tolist(), "box_max": hi.tolist(), "object_points": int(len(obj_pts)),
            "vertices": int(len(used)), "faces": int(keep_f.sum())}
    with open(os.path.splitext(path)[0] + "_crop.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[✓] {path}: {len(used)} đỉnh, {int(keep_f.sum())} mặt, {os.path.getsize(path) / 1024**2:.0f} MB")


if __name__ == "__main__":
    main()
