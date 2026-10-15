"""
Tạo mask InFusion cho một vật khi SAM2 bị trôi ID (cùng vật mang ID khác ở vài ảnh, và ID đó lại là vật khác ở ảnh khác).

  1. Điểm 3D của vật = đỉnh mesh tham chiếu nhìn thấy (depth render) mang ID chính (--main_id) trên các view thấy vật rõ nhất.
  2. Mỗi ảnh: chiếu các điểm đó (kiểm tra depth) -> vùng vật trong ảnh.
  3. Mask = ID chính + các mảnh liền (connected component) của --extra_ids trùng vùng vật > --overlap.
Kết quả: <out_dir>/<frame>.png (0/255, đã crop, đã dilate) + preview overlay.

    python scripts/lerf_masks_3d_consistent.py --ref_mesh .../teatime/mesh/mesh_gof.ply --ref_render_dir .../teatime/gs/train/ours_150
        --labels_dir ".../mask/labels" --images_dir ".../images" --transforms ".../transforms.json"
        --main_id 2 --extra_ids 6,9 --out_dir <seg>
"""
import os
import sys
from argparse import ArgumentParser

import cv2
import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Based_Model", "Infusion", "mesh"))
from crop_region import top_views  # noqa: E402
from crop_table import crop_boxes, visible_labels  # noqa: E402


def project_visible(P, view, render_dir, depth_tol):
    c2w = np.load(os.path.join(render_dir, "c2w", f"{view}.npy"))
    K = np.load(os.path.join(render_dir, "intri", f"{view}.npy"))
    depth = np.load(os.path.join(render_dir, "depth", f"{view}.npy"))
    H, W = depth.shape
    w2c = np.linalg.inv(c2w)
    pc = P @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    ok = z > 1e-3
    u = np.round(pc[ok, 0] / z[ok] * K[0, 0] + K[0, 2]).astype(np.int64)
    v = np.round(pc[ok, 1] / z[ok] * K[1, 1] + K[1, 2]).astype(np.int64)
    zz = z[ok]
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, zz = u[inb], v[inb], zz[inb]
    vis = np.abs(zz - depth[v, u]) < depth_tol * depth[v, u]
    sil = np.zeros((H, W), np.uint8)
    sil[v[vis], u[vis]] = 255
    return sil


def main():
    ap = ArgumentParser()
    ap.add_argument("--ref_mesh", required=True)
    ap.add_argument("--ref_render_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--transforms", required=True)
    ap.add_argument("--main_id", type=int, required=True)
    ap.add_argument("--extra_ids", default="")
    ap.add_argument("--n_views", type=int, default=20, help="Số view dùng để lấy điểm 3D của vật")
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--depth_tol", type=float, default=0.03)
    ap.add_argument("--dilate", type=int, default=5)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--preview_dir", default="")
    args = ap.parse_args()

    extra = [int(x) for x in args.extra_ids.split(",") if x]
    boxes = crop_boxes(args.transforms)
    mesh = o3d.io.read_triangle_mesh(args.ref_mesh)
    P = np.asarray(mesh.vertices)
    del mesh

    views = top_views(args.labels_dir, boxes, [args.main_id], args.n_views)
    obj = np.zeros(len(P), bool)
    for v in views:
        obj |= visible_labels(P, v, args.ref_render_dir, args.labels_dir, boxes[v], args.depth_tol) == args.main_id
    Q = P[obj]
    del P
    print(f"[i] {len(Q)} điểm 3D của vật (ID {args.main_id}) từ {len(views)} view")

    os.makedirs(args.out_dir, exist_ok=True)
    if args.preview_dir:
        os.makedirs(args.preview_dir, exist_ok=True)
    kernel = np.ones((args.dilate, args.dilate), np.uint8) if args.dilate > 0 else None
    close = np.ones((41, 41), np.uint8)
    added, rejected, empty = 0, 0, 0
    for stem in sorted(boxes):
        x0, y0, x1, y1 = boxes[stem]
        label = cv2.imread(os.path.join(args.labels_dir, f"{stem}.png"), cv2.IMREAD_UNCHANGED)[y0:y1, x0:x1]
        sil = project_visible(Q, stem, args.ref_render_dir, args.depth_tol)
        if sil.shape != label.shape:
            sil = cv2.resize(sil, (label.shape[1], label.shape[0]), interpolation=cv2.INTER_NEAREST)
        sil = cv2.morphologyEx(cv2.dilate(sil, np.ones((5, 5), np.uint8)), cv2.MORPH_CLOSE, close)
        # Lấp lỗ được vật bao quanh (vd mõm/đầu gấu hiếm khi mang ID chính nên không có điểm 3D)
        flood = np.pad(sil, 1)
        cv2.floodFill(flood, None, (0, 0), 128)
        sil = (sil > 0) | (flood[1:-1, 1:-1] == 0)

        m = label == args.main_id
        for eid in extra:
            n, comp = cv2.connectedComponents((label == eid).astype(np.uint8))
            for c in range(1, n):
                part = comp == c
                if sil[part].mean() > args.overlap:
                    m |= part
                    added += 1
                else:
                    rejected += 1
        out = m.astype(np.uint8) * 255
        if kernel is not None:
            out = cv2.dilate(out, kernel)
        empty += not out.any()
        cv2.imwrite(os.path.join(args.out_dir, f"{stem}.png"), out)
        if args.preview_dir:
            img = cv2.imread(os.path.join(args.images_dir, f"{stem}.jpg"))[y0:y1, x0:x1]
            ov = img.copy()
            ov[out > 0] = (0.45 * ov[out > 0] + 0.55 * np.array([0, 255, 0])).astype(np.uint8)
            cv2.imwrite(os.path.join(args.preview_dir, f"{stem}.jpg"), ov)
    print(f"[✓] {len(boxes)} mask -> {args.out_dir} | thêm {added} mảnh ID {extra} trùng vật, loại {rejected} mảnh khác vật | {empty} ảnh trống")


if __name__ == "__main__":
    main()
