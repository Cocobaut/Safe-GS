"""
S10 (ma sau khi bỏ vật, v1): alpha của RIÊNG các Gaussian lơ lửng trong khoảng trống vật để lại, trên nền S4b.

Cùng định nghĩa với tools/objectgs_eval_removal.py (ObjectGS) để so hai phương pháp:
  - vết chân = chiếu Gaussian của vật trên bàn (S3 objects_all.ply) xuống mặt bàn, lưới 1 cm, nới 1 ô;
  - cột khoảng trống = vết chân x độ cao từ hmin tới phân vị 98 % chiều cao vật;
  - Gaussian "ma" = Gaussian của nền S4b có tâm trong cột đó, cao hơn mặt bàn > hmin (1.5 cm cho số chính);
  - render RIÊNG các Gaussian ma (màu 1, nền đen) -> alpha; đo trung bình trên pixel vết chân của 23 view test,
    loại pixel bị gấu / cừu che (ID SAM2 2, 9, 4, 27, nới 15 px).
Xuất ghost.json.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from common import log, make_parser, load_scene, read_ply, write_ply, gaussians_from_ply
from arguments import get_combined_args
from gaussian_renderer import render
from utils.depth_utils import depths_to_points

OCCLUDER_IDS = [2, 9, 4, 27]


def main():
    p, model, pipe, ss, _, _ = make_parser("S10 ghost")
    for k in ("s3", "s4b", "labels"):
        p.add_argument(f"--{k}", type=str, required=True)
    p.add_argument("--occluder_ids", type=str, default="2,9,4,27", help="ID gấu / cừu trong thư mục --labels (SAM2: 2,9,4,27; SAM 3: 20,30)")
    args = get_combined_args(p)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    occluder_ids = [int(x) for x in args.occluder_ids.split(",")]
    S = load_scene(args, model, pipe, ss)
    sh = S["dataset"].sh_degree
    black = torch.zeros(3, device="cuda")
    sup = json.loads((Path(args.s3) / "support.json").read_text())
    n_t, d_t = np.array(sup["table"]["normal"], np.float64), float(sup["table"]["d"])
    mpu = float(sup["meters_per_unit"])
    ob = read_ply(Path(args.s3) / "gaussians" / "objects_all.ply")
    oxyz = np.stack([ob["x"], ob["y"], ob["z"]], 1).astype(np.float64)
    s_w = 1.0 if np.median(oxyz @ n_t + d_t) >= 0 else -1.0
    hmax_cm = float(np.percentile(s_w * (oxyz @ n_t + d_t), 98)) * mpu * 100

    a0 = np.array([1.0, 0, 0]) if abs(n_t[0]) < 0.9 else np.array([0, 1.0, 0])
    u_ = np.cross(n_t, a0); u_ /= np.linalg.norm(u_); v_ = np.cross(n_t, u_)
    p0 = -d_t * n_t
    cell = 0.01 / mpu
    oa, ob_ = np.floor((oxyz - p0) @ u_ / cell).astype(int), np.floor((oxyz - p0) @ v_ / cell).astype(int)
    amin, bmin = oa.min() - 2, ob_.min() - 2
    foot = np.zeros((ob_.max() - bmin + 3, oa.max() - amin + 3), np.uint8)
    foot[ob_ - bmin, oa - amin] = 1
    foot = cv2.dilate(foot, np.ones((3, 3), np.uint8)) > 0

    def cells(P):
        ia = np.floor((P - p0) @ u_ / cell).astype(int) - amin
        ib = np.floor((P - p0) @ v_ / cell).astype(int) - bmin
        ok = (ia >= 0) & (ia < foot.shape[1]) & (ib >= 0) & (ib < foot.shape[0])
        return ia, ib, ok

    def footprint_pixels(v):
        H, W = int(v.image_height), int(v.image_width)
        P1 = depths_to_points(v, torch.ones(1, H, W, device="cuda")).cpu().numpy()
        o = v.camera_center.cpu().numpy()
        dirs = P1 - o
        tt = -(o @ n_t + d_t) / (dirs @ n_t)
        ia, ib, ok = cells(o + tt[:, None] * dirs)
        ok &= tt > 0
        fp = np.zeros(H * W, bool)
        fp[ok] = foot[ib[ok], ia[ok]]
        return fp.reshape(H, W)

    el = read_ply(Path(args.s4b) / "background_finetuned.ply")
    P = np.stack([el["x"], el["y"], el["z"]], 1).astype(np.float64)
    h = s_w * (P @ n_t + d_t) * mpu * 100
    ia, ib, ok = cells(P)
    inside = ok.copy()
    inside[ok] = foot[ib[ok], ia[ok]]
    res = {"background_gaussians": int(len(P)), "object_height_p98_cm": hmax_cm,
           "footprint_cells_1cm": int(foot.sum())}
    for hmin in (0.5, 1.5, 3.0):
        res[f"ghost_gaussians_gt_{hmin}cm"] = int((inside & (h > hmin) & (h < hmax_cm)).sum())
    sel = inside & (h > 1.5) & (h < hmax_cm)
    log(f"Gaussian ma (> 1.5 cm): {int(sel.sum())} / {len(P)}")

    alphas, cover = [], []
    if sel.sum() > 0:
        tmp = out / "_ghost_subset.ply"
        write_ply(tmp, el[sel])
        gsub = gaussians_from_ply(tmp, sh, S["train"])
        ones = torch.ones(gsub.get_xyz.shape[0], 3, device="cuda")
        labels_dir = Path(args.labels)
        for v in S["scene"].getTestCameras():
            f = labels_dir / f"{Path(v.image_name).stem}.png"
            if not f.exists():
                continue
            fp = footprint_pixels(v)
            lab = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            occl = cv2.dilate(np.isin(lab, occluder_ids).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
            fp &= ~occl
            if fp.sum() < 50:
                continue
            with torch.no_grad():
                ga = render(v, gsub, S["pipe"], black, splat_args=S["splat_args"], override_color=ones)["render"][0]
            ga = ga.clamp(0, 1).cpu().numpy()
            alphas.append(float(ga[fp].mean())); cover.append(float((ga[fp] > 0.1).mean()))
        tmp.unlink(missing_ok=True)
    res["views_used"] = len(alphas)
    res["ghost_alpha_mean"] = float(np.mean(alphas)) if alphas else 0.0
    res["ghost_cover_pct_alpha_gt_0.1"] = 100 * float(np.mean(cover)) if cover else 0.0
    (out / "ghost.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    log(json.dumps(res))


if __name__ == "__main__":
    main()
