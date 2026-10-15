"""
Space carving: xóa Gaussian "rác" nằm trong khối của vật đã bị che khi train (Stage 1).

Vùng vật không được ảnh nào giám sát nên chứa Gaussian rác. Một Gaussian nằm TRONG khối vật sẽ rơi vào mask
ở hầu hết các view thấy nó; còn Gaussian của nền phía sau (sàn, ghế) sẽ lộ ra ngoài mask ở các view khác.
-> Xóa Gaussian có tỉ lệ (số view rơi vào mask / số view nhìn thấy trong khung) >= --ratio.

    python carve.py --ply <point_cloud.ply> --cameras <cameras.json> --seg_dir <data/seg> --out <point_cloud.ply mới>
"""
import json
import os
from argparse import ArgumentParser

import cv2
import numpy as np
from plyfile import PlyData, PlyElement


def main():
    ap = ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--cameras", required=True, help="cameras.json của InFusion")
    ap.add_argument("--seg_dir", required=True)
    ap.add_argument("--ratio", type=float, default=0.85)
    ap.add_argument("--min_views", type=int, default=5)
    ap.add_argument("--seg_dilate", type=int, default=9)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ply = PlyData.read(args.ply)
    v = ply["vertex"]
    X = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    inside = np.zeros(len(X), np.int32)
    seen = np.zeros(len(X), np.int32)
    kernel = np.ones((args.seg_dilate, args.seg_dilate), np.uint8)

    cams = json.load(open(args.cameras))
    for c in cams:
        seg_path = os.path.join(args.seg_dir, f"{c['img_name']}.png")
        if not os.path.exists(seg_path):
            continue
        m = cv2.dilate((cv2.imread(seg_path, 0) > 0).astype(np.uint8), kernel) > 0
        R = np.array(c["rotation"])  # C2W rotation
        pos = np.array(c["position"])
        W, H = int(c["width"]), int(c["height"])
        pc = (X - pos) @ R  # world -> camera (x_c = R^T (X - pos))
        z = pc[:, 2]
        ok = z > 0.05
        u = np.full(len(X), -1, np.int64)
        vv = np.full(len(X), -1, np.int64)
        u[ok] = np.round(pc[ok, 0] / z[ok] * c["fx"] + W / 2).astype(np.int64)
        vv[ok] = np.round(pc[ok, 1] / z[ok] * c["fy"] + H / 2).astype(np.int64)
        inb = ok & (u >= 0) & (u < m.shape[1]) & (vv >= 0) & (vv < m.shape[0])
        seen[inb] += 1
        inside[np.nonzero(inb)[0][m[vv[inb], u[inb]]]] += 1

    ratio = inside / np.maximum(seen, 1)
    carve = (seen >= args.min_views) & (ratio >= args.ratio)
    print(f"[i] {len(X)} Gaussians | xóa {carve.sum()} Gaussians trong khối vật (ratio >= {args.ratio})")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    PlyData([PlyElement.describe(v.data[~carve], "vertex")]).write(args.out)
    print(f"[✓] {args.out}: còn {(~carve).sum()} Gaussians")


if __name__ == "__main__":
    main()
