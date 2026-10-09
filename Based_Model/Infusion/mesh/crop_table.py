"""
Cắt riêng mặt bàn + đồ vật trên bàn từ mesh GOF gốc (không giảm lưới, không lọc mảnh rời).

Vùng cắt được xác định từ dữ liệu, không chỉnh tay:
  1. Đỉnh mesh nhìn thấy (kiểm tra bằng depth render của InFusion) và rơi vào label "mặt bàn" của SAM2
     trên vài view -> RANSAC fit mặt phẳng bàn.
  2. Đường bao bàn = bao lồi 2D (trên mặt phẳng) của các đỉnh inlier, nới thêm --margin.
  3. Chiều cao tối đa = phân vị 99% độ cao các đỉnh thuộc các vật trên bàn (--object_ids) + --height_margin.
  4. Giữ mọi tam giác có cả 3 đỉnh nằm trong lăng trụ (đường bao x [-below, chiều cao tối đa]).

    python crop_table.py --mesh mesh_gof.ply --render_dir <gs/train/ours_150> --labels_dir <mask/labels>
                         --transforms <transforms.json> --table_id 1 --object_ids 15,22,29 --output_dir <mesh>
"""
import json
import math
import os
import sys
from argparse import ArgumentParser

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull, Delaunay

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gof.io_utils import unique_path  # noqa: E402


def crop_boxes(transforms_path):
    """Crop principal point về tâm, giống modal_lerf_prep.py::_frames (để label khớp ảnh/camera đã crop)."""
    boxes = {}
    for f in json.loads(open(transforms_path).read())["frames"]:
        w, h, cx, cy = int(f["w"]), int(f["h"]), float(f["cx"]), float(f["cy"])
        half_w, half_h = math.floor(min(cx, w - cx)), math.floor(min(cy, h - cy))
        x0 = min(max(int(round(cx - half_w)), 0), w - 2 * half_w)
        y0 = min(max(int(round(cy - half_h)), 0), h - 2 * half_h)
        boxes[os.path.splitext(os.path.basename(f["file_path"]))[0]] = (x0, y0, x0 + 2 * half_w, y0 + 2 * half_h)
    return boxes


def visible_labels(P, view, render_dir, labels_dir, box, depth_tol):
    """Label SAM2 của các đỉnh nhìn thấy trong view (0 nếu bị che hoặc ngoài ảnh)."""
    c2w = np.load(os.path.join(render_dir, "c2w", f"{view}.npy"))
    K = np.load(os.path.join(render_dir, "intri", f"{view}.npy"))
    depth = np.load(os.path.join(render_dir, "depth", f"{view}.npy"))
    x0, y0, x1, y1 = box
    label = cv2.imread(os.path.join(labels_dir, f"{view}.png"), cv2.IMREAD_UNCHANGED)[y0:y1, x0:x1]
    H, W = depth.shape
    label = cv2.resize(label, (W, H), interpolation=cv2.INTER_NEAREST) if label.shape != (H, W) else label

    w2c = np.linalg.inv(c2w)
    pc = P @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    out = np.zeros(len(P), dtype=np.int32)
    ok = z > 1e-3
    u = np.round(pc[ok, 0] / z[ok] * K[0, 0] + K[0, 2]).astype(np.int64)
    v = np.round(pc[ok, 1] / z[ok] * K[1, 1] + K[1, 2]).astype(np.int64)
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    idx = np.nonzero(ok)[0][inb]
    u, v = u[inb], v[inb]
    vis = np.abs(z[idx] - depth[v, u]) < depth_tol * depth[v, u]
    out[idx[vis]] = label[v[vis], u[vis]]
    return out


def main():
    ap = ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--render_dir", required=True, help="gs/train/ours_<iter> (c2w, intri, depth)")
    ap.add_argument("--labels_dir", required=True, help="mask/labels của SAM2 (ảnh gốc chưa crop)")
    ap.add_argument("--transforms", required=True)
    # Chỉ dùng view nhìn sát bàn: view rộng thấy cả dãy bàn ghép liền phía sau (SAM2 gán cùng ID) làm đường bao lan ra
    ap.add_argument("--views", default="frame_00127,frame_00050,frame_00100")
    ap.add_argument("--table_id", type=int, default=1)
    ap.add_argument("--object_ids", default="15,22,29", help="Các vật cao trên bàn để ước lượng chiều cao cắt")
    ap.add_argument("--margin", type=float, default=0.02, help="Nới đường bao bàn (đơn vị scene)")
    ap.add_argument("--below", type=float, default=0.03, help="Giữ thêm phần dưới mặt bàn (độ dày mép bàn)")
    ap.add_argument("--height_margin", type=float, default=0.03)
    ap.add_argument("--depth_tol", type=float, default=0.02, help="Sai lệch depth tương đối để coi là nhìn thấy")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--name", default="mesh_gof_table.ply")
    args = ap.parse_args()

    mesh = o3d.io.read_triangle_mesh(args.mesh)
    P = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    print(f"[i] Mesh gốc: {len(P)} đỉnh, {len(F)} mặt")

    boxes = crop_boxes(args.transforms)
    obj_ids = [int(x) for x in args.object_ids.split(",") if x]
    table_pts, obj_mask = [], np.zeros(len(P), dtype=bool)
    for view in args.views.split(","):
        lab = visible_labels(P, view, args.render_dir, args.labels_dir, boxes[view], args.depth_tol)
        table_pts.append(np.nonzero(lab == args.table_id)[0])
        obj_mask |= np.isin(lab, obj_ids)
        print(f"[i] {view}: {np.sum(lab == args.table_id)} đỉnh bàn, {np.isin(lab, obj_ids).sum()} đỉnh vật cao")
    table_idx = np.unique(np.concatenate(table_pts))

    # 1. Mặt phẳng bàn (RANSAC), pháp tuyến hướng về phía camera
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[table_idx]))
    (a, b, c, d), inliers = pcd.segment_plane(distance_threshold=0.005, ransac_n=3, num_iterations=2000)
    n = np.array([a, b, c]) / np.linalg.norm([a, b, c])
    d = d / np.linalg.norm([a, b, c])
    cam_center = np.load(os.path.join(args.render_dir, "c2w", f"{args.views.split(',')[0]}.npy"))[:3, 3]
    if cam_center @ n + d < 0:
        n, d = -n, -d
    inl = P[table_idx][inliers]
    print(f"[i] Mặt phẳng bàn: n={n.round(4)}, d={d:.4f}, {len(inliers)}/{len(table_idx)} inlier")

    # 2. Đường bao bàn trong toạ độ mặt phẳng
    e1 = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    to2d = lambda X: np.stack([X @ e1, X @ e2], axis=1)  # noqa: E731
    hull2d = to2d(inl)[ConvexHull(to2d(inl)).vertices]
    centroid = hull2d.mean(axis=0)
    dirs = hull2d - centroid
    hull2d = hull2d + dirs / np.linalg.norm(dirs, axis=1, keepdims=True) * args.margin
    tri = Delaunay(hull2d)

    # 3. Chiều cao tối đa từ các vật cao trên bàn
    h_all = P @ n + d
    obj_h = h_all[obj_mask & (tri.find_simplex(to2d(P)) >= 0)]
    h_max = (np.percentile(obj_h, 99) if len(obj_h) else 0.3) + args.height_margin
    print(f"[i] Chiều cao cắt: {-args.below:.3f} .. {h_max:.3f} (từ {len(obj_h)} đỉnh vật cao)")

    # 4. Giữ tam giác có cả 3 đỉnh trong lăng trụ
    keep_v = (h_all >= -args.below) & (h_all <= h_max)
    cand = np.nonzero(keep_v)[0]
    keep_v[cand] = tri.find_simplex(to2d(P[cand])) >= 0
    keep_f = keep_v[F].all(axis=1)
    out = o3d.geometry.TriangleMesh()
    used = np.unique(F[keep_f])
    remap = -np.ones(len(P), dtype=np.int64)
    remap[used] = np.arange(len(used))
    out.vertices = o3d.utility.Vector3dVector(P[used])
    out.triangles = o3d.utility.Vector3iVector(remap[F[keep_f]])
    if mesh.has_vertex_colors():
        out.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors)[used])
    out.compute_vertex_normals()

    os.makedirs(args.output_dir, exist_ok=True)
    path = unique_path(os.path.join(args.output_dir, args.name))
    o3d.io.write_triangle_mesh(path, out, write_ascii=False, compressed=False)
    meta = {"source": os.path.abspath(args.mesh), "plane_normal": n.tolist(), "plane_d": float(d),
            "hull_2d": hull2d.tolist(), "basis_e1": e1.tolist(), "basis_e2": e2.tolist(),
            "height_range": [-args.below, float(h_max)], "views": args.views, "table_id": args.table_id,
            "object_ids": obj_ids, "vertices": len(used), "faces": int(keep_f.sum())}
    with open(os.path.splitext(path)[0] + "_crop.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[✓] {path}: {len(used)} đỉnh, {int(keep_f.sum())} mặt, {os.path.getsize(path) / 1024**2:.0f} MB")


if __name__ == "__main__":
    main()
