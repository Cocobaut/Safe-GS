"""
Mask inpaint cho một view trong chuỗi inpaint nhiều view:
  mask = (mask vật của view, nới --dilate)  TRỪ  phần đã được lấp bởi point cloud của các view trước.
Phần "đã lấp" = điểm của các *_mask.ply trước chiếu vào view (kiểm tra depth với depth render hiện tại).

    python coverage_mask.py --seg <seg/view.png> --render_dir <gs/train/ours_N> --view <frame> --prev_ply a.ply b.ply --out mask.png
"""
import os
from argparse import ArgumentParser

import cv2
import numpy as np
from plyfile import PlyData


def main():
    ap = ArgumentParser()
    ap.add_argument("--seg", required=True)
    ap.add_argument("--render_dir", required=True)
    ap.add_argument("--view", required=True)
    ap.add_argument("--prev_ply", nargs="*", default=[])
    ap.add_argument("--dilate", type=int, default=11)
    ap.add_argument("--depth_tol", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    c2w = np.load(os.path.join(args.render_dir, "c2w", f"{args.view}.npy"))
    K = np.load(os.path.join(args.render_dir, "intri", f"{args.view}.npy"))
    depth = np.load(os.path.join(args.render_dir, "depth", f"{args.view}.npy"))
    H, W = depth.shape
    seg = cv2.imread(args.seg, 0) > 0
    if seg.shape != (H, W):
        seg = cv2.resize(seg.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    mask = cv2.dilate(seg.astype(np.uint8), np.ones((args.dilate, args.dilate), np.uint8)) > 0

    covered = np.zeros((H, W), np.uint8)
    w2c = np.linalg.inv(c2w)
    for p in args.prev_ply:
        v = PlyData.read(p)["vertex"]
        X = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
        pc = X @ w2c[:3, :3].T + w2c[:3, 3]
        z = pc[:, 2]
        ok = z > 0.02
        u = np.round(pc[ok, 0] / z[ok] * K[0, 0] + K[0, 2]).astype(np.int64)
        vv = np.round(pc[ok, 1] / z[ok] * K[1, 1] + K[1, 2]).astype(np.int64)
        zz = z[ok]
        inb = (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
        u, vv, zz = u[inb], vv[inb], zz[inb]
        d = depth[vv, u]
        # Điểm của view trước nhìn thấy được (không bị nền/vật khác phía trước che)
        vis = ~np.isfinite(d) | (d <= 0) | (zz <= d * (1 + args.depth_tol))
        covered[vv[vis], u[vis]] = 255
    covered = cv2.morphologyEx(covered, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0

    new = mask & ~covered
    new = cv2.morphologyEx(new.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    cv2.imwrite(args.out, new.astype(np.uint8) * 255)
    print(f"[i] {args.view}: mask vật {mask.mean() * 100:.1f}% ảnh, đã lấp {(mask & covered).mean() * 100:.1f}%, "
          f"cần inpaint {new.mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
