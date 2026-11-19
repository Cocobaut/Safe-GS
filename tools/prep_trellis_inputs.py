"""
Chuẩn bị đầu vào cho bước hoàn thiện vật bằng TRELLIS (modal_trellis.py), cùng một quy tắc cho mọi vật.

Mỗi vật -> <out>/<vật>/
  grid.npz : lưới 64^3 trong hệ bàn (x, y = mặt bàn, z = pháp tuyến bàn, đúng quy ước z-lên của TRELLIS), chuẩn hoá về [-0.5, 0.5]^3:
             frame  = ô thuộc khung chặt (S3m: bề mặt thấy + ruột + khuôn),
             seen   = ô bề mặt camera thấy (lớp 1),
             unknown= ô ruột / khuôn chưa ai thấy (lớp 2, 3, 4),
             free   = ô camera đã nhìn xuyên (trong hull nhưng bị khoét) hoặc ngoài hull: CHẮC CHẮN trống,
             cùng center / size / R / p0 / mpu để đổi ngược về hệ cảnh.
  view_<k>.png : K ảnh RGBA (alpha = mask SAM 3 của vật), chọn ảnh mask lớn và hướng nhìn trải đều.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Based_Model" / "CoMe" / "scene"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from colmap_loader import read_extrinsics_binary, qvec2rotmat  # noqa: E402
from support_graph import read_ply_xyz  # noqa: E402
from mask_hull import OBJECTS_SAM3  # noqa: E402

RES = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--hulls", required=True)
    ap.add_argument("--support", required=True)
    ap.add_argument("--colmap", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", type=int, default=6)
    ap.add_argument("--pad", type=float, default=1.15)
    ap.add_argument("--objects", default="")
    a = ap.parse_args()
    out = Path(a.out)
    sup = json.loads(Path(a.support).read_text())
    n_t, d_t, mpu = np.array(sup["table"]["normal"]), float(sup["table"]["d"]), float(sup["meters_per_unit"])
    b0 = np.array([1.0, 0, 0]) if abs(n_t[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(n_t, b0); e1 /= np.linalg.norm(e1); e2 = np.cross(n_t, e1)
    R = np.stack([e1, e2, n_t], 0)
    p0 = -d_t * n_t
    ext = read_extrinsics_binary(str(Path(a.colmap) / "images.bin"))
    cams = {Path(e.name).stem: (-qvec2rotmat(e.qvec).T @ e.tvec) for e in ext.values()}
    labs = {f.stem: f for f in sorted(Path(a.labels).glob("frame_*.png"))}
    names = sorted(f.stem for f in Path(a.frames).glob("*.npz"))
    if a.objects:
        names = [n for n in names if n in a.objects.split(",")]
    summary = {}
    for n in names:
        z = np.load(Path(a.frames) / f"{n}.npz")
        F, cls, vox = z["xyz"].astype(np.float64), z["cls"], float(z["voxel"])
        hp = Path(a.hulls) / f"hull_{n}.ply"
        if len(F) < 30 or not hp.exists() or n not in OBJECTS_SAM3:
            continue
        H = read_ply_xyz(hp)
        Ft, Ht = (F - p0) @ R.T, (H - p0) @ R.T
        lo, hi = Ft.min(0), Ft.max(0)
        center = (lo + hi) / 2
        size = float((hi - lo).max() * a.pad)
        # tâm ô lưới 64^3 -> toạ độ hệ bàn -> tra voxel gần nhất (khung / hull)
        g = (np.arange(RES) + 0.5) / RES - 0.5
        X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
        C = np.stack([X, Y, Z], -1).reshape(-1, 3) * size + center
        dF, iF = cKDTree(Ft).query(C, k=1)
        dH, _ = cKDTree(Ht).query(C, k=1)
        tol = 0.87 * vox                                           # nửa đường chéo voxel
        in_frame = dF <= tol
        in_hull = dH <= tol
        c_ = cls[iF]
        frame = in_frame.reshape(RES, RES, RES)
        seen = (in_frame & (c_ == 1)).reshape(RES, RES, RES)
        unknown = (in_frame & (c_ != 1)).reshape(RES, RES, RES)
        free = (~in_frame).reshape(RES, RES, RES)                  # trong hull bị khoét hoặc ngoài hull
        carved = (in_hull & ~in_frame).reshape(RES, RES, RES)      # camera nhìn xuyên: CHẮC CHẮN trống
        # dưới mặt bàn chỉ chắc chắn trống nếu vật ĐẶT trên bàn (khung gần như không xuống dưới bàn quá 2 cm), vd gấu / cừu ngồi ghế thì không
        rests = float((Ft[:, 2] < -0.02 / mpu).mean()) < 0.05
        below = ((C[:, 2] < 0) & rests).reshape(RES, RES, RES)
        od = out / n
        od.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(od / "grid.npz", frame=frame, seen=seen, unknown=unknown, free=free, carved=carved, below=below,
                            center=center, size=size, R=R, p0=p0, mpu=mpu, voxel=vox)
        # chọn ảnh: mask lớn (>= 40 % lớn nhất), hướng nhìn trải đều (chọn xa nhất lần lượt)
        ids = OBJECTS_SAM3[n][0]
        areas = {}
        for st, f in labs.items():
            lab = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            m = np.isin(lab, ids)
            if m.sum() > 400:
                areas[st] = int(m.sum())
        if not areas:
            continue
        mx = max(areas.values())
        cand = [s for s, v in areas.items() if v >= 0.4 * mx and s in cams]
        cobj = (F.mean(0))
        dirs = {s: (cams[s] - cobj) / np.linalg.norm(cams[s] - cobj) for s in cand}
        pick = [max(cand, key=lambda s: areas[s])]
        while len(pick) < min(a.views, len(cand)):
            rest = [s for s in cand if s not in pick]
            pick.append(max(rest, key=lambda s: min(np.linalg.norm(dirs[s] - dirs[p]) for p in pick)))
        for k, st in enumerate(pick):
            img = cv2.imread(str(Path(a.images) / f"{st}.jpg"))
            lab = cv2.imread(str(labs[st]), cv2.IMREAD_UNCHANGED)
            if img is None or lab is None:
                continue
            m = np.isin(lab, ids).astype(np.uint8)
            if m.shape != img.shape[:2]:
                m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            ys, xs = np.nonzero(m)
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            s = int(max(y1 - y0, x1 - x0) * 1.3) + 4
            cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
            pad = s
            imgp = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
            mp = cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
            crop = imgp[cy - s // 2 + pad:cy + s // 2 + pad, cx - s // 2 + pad:cx + s // 2 + pad]
            mc = mp[cy - s // 2 + pad:cy + s // 2 + pad, cx - s // 2 + pad:cx + s // 2 + pad]
            rgba = np.dstack([crop, mc * 255]).astype(np.uint8)
            cv2.imwrite(str(od / f"view_{k}.png"), rgba)
        summary[n] = {"rests_on_table": rests, "frame_cells": int(frame.sum()), "seen_cells": int(seen.sum()), "unknown_cells": int(unknown.sum()),
                      "size_cm": size * mpu * 100, "cell_cm": size * mpu * 100 / RES, "views": pick}
        print(n, summary[n])
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
