"""
Kiểm chứng phép tính tầm nhìn (tools/visibility_mold.py) bằng render: Gaussian của vật mà KHÔNG camera nào thấy thì xoá đi
ảnh render ở mọi view gần như không đổi.

So 3 cảnh trên mọi view COLMAP (train + test):
  full    : CoMe đầy đủ;
  -unseen : bỏ Gaussian vật có n_vis = 0 (khớp toạ độ với gauss_xyz_<vật>.npy);
  -ctrl   : bỏ cùng số Gaussian chọn ngẫu nhiên trong các Gaussian vật ĐƯỢC thấy (n_vis >= 3), cùng vật, cùng số lượng (đối chứng).
Đo: sai khác trung bình |full - x| và % pixel đổi > 2/255 trong vùng mask vật (SAM 3) và toàn ảnh; PSNR so ảnh thật trên view test.
Xuất: metrics.json, preview/<frame>.jpg [full | -unseen | -ctrl | sai khác x8]
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree

from common import log, make_parser, load_scene, view_name, read_ply, write_ply, gaussians_from_ply, render_rgbd, to_u8
from arguments import get_combined_args
from utils.image_utils import psnr

OBJ_IDS = [80, 70, 60, 100, 90, 50, 130, 110, 170, 171, 180, 181, 160, 161, 20, 30]
SHEET = ["frame_00017", "frame_00065", "frame_00113"]


def main():
    p, model, pipe, ss, _, _ = make_parser("S3v unseen check")
    p.add_argument("--vis_dir", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--min_vis", type=int, default=3)
    args = get_combined_args(p)
    out = Path(args.out)
    (out / "preview").mkdir(parents=True, exist_ok=True)
    S = load_scene(args, model, pipe, ss)
    g = S["g"]
    tmp = out / "_full.ply"
    g.save_ply(str(tmp))
    el = read_ply(tmp)
    xyz = np.stack([el["x"], el["y"], el["z"]], 1)
    tree = cKDTree(xyz)
    rng = np.random.default_rng(0)
    rm_u, rm_c, per = [], [], {}
    for f in sorted(Path(args.vis_dir).glob("gauss_xyz_*.npy")):
        n = f.stem[len("gauss_xyz_"):]
        P = np.load(f).astype(np.float64)
        nv = np.load(Path(args.vis_dir) / f"gauss_nvis_{n}.npy")
        d, idx = tree.query(P, k=1)
        ok = d < 1e-5
        u = idx[ok & (nv == 0)]
        seen = idx[ok & (nv >= args.min_vis)]
        c = rng.choice(seen, min(len(u), len(seen)), replace=False) if len(seen) else np.zeros(0, int)
        rm_u.append(u); rm_c.append(c)
        per[n] = {"matched": int(ok.sum()), "of": int(len(P)), "unseen_removed": int(len(u)), "ctrl_removed": int(len(c))}
    rm_u, rm_c = np.unique(np.concatenate(rm_u)), np.unique(np.concatenate(rm_c))
    log(f"bỏ unseen {len(rm_u)}, đối chứng {len(rm_c)} / {len(xyz)}; {per}")
    sh = S["dataset"].sh_degree
    models = {}
    for tag, rm in (("unseen", rm_u), ("ctrl", rm_c)):
        keep = np.ones(len(el), bool); keep[rm] = False
        pth = out / f"_{tag}.ply"
        write_ply(pth, el[keep])
        models[tag] = gaussians_from_ply(pth, sh, S["train"])
    obj_t = torch.tensor(OBJ_IDS, device="cuda")
    acc = {t: {"mad_obj": [], "chg_obj": [], "mad_all": [], "chg_all": [], "psnr_test": []} for t in ("unseen", "ctrl")}
    psnr_full = []
    test_names = {view_name(v) for v in S["scene"].getTestCameras()}
    with torch.no_grad():
        for v in S["views"]:
            nm = view_name(v)
            full = render_rgbd(v, g, S)[0].clamp(0, 1)
            f = Path(args.labels) / f"{nm}.png"
            M = None
            if f.exists():
                lab = torch.from_numpy(cv2.imread(str(f), cv2.IMREAD_UNCHANGED).astype(np.int64)).cuda()
                M = torch.isin(lab, obj_t)
            gt = v.original_image.cuda()[:3]
            if nm in test_names:
                psnr_full.append(float(psnr(full[None], gt[None]).mean()))
            imgs = {}
            for t, mdl in models.items():
                x = render_rgbd(v, mdl, S)[0].clamp(0, 1)
                imgs[t] = x
                dif = (full - x).abs().mean(0)
                acc[t]["mad_all"].append(float(dif.mean())); acc[t]["chg_all"].append(float((dif > 2 / 255).float().mean()))
                if M is not None and M.sum() > 100:
                    acc[t]["mad_obj"].append(float(dif[M].mean())); acc[t]["chg_obj"].append(float((dif[M] > 2 / 255).float().mean()))
                if nm in test_names:
                    acc[t]["psnr_test"].append(float(psnr(x[None], gt[None]).mean()))
            if nm in SHEET:
                sep = torch.ones(3, full.shape[1], 6, device="cuda")
                d8 = ((full - imgs["unseen"]).abs() * 8).clamp(0, 1)
                d8c = ((full - imgs["ctrl"]).abs() * 8).clamp(0, 1)
                row = torch.cat([full, sep, imgs["unseen"], sep, imgs["ctrl"], sep, d8, sep, d8c], 2)
                cv2.imwrite(str(out / "preview" / f"{nm}.jpg"), to_u8(row)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
    res = {"gaussians_total": int(len(xyz)), "removed_unseen": int(len(rm_u)), "removed_ctrl": int(len(rm_c)), "per_object": per,
           "psnr_test_full": float(np.mean(psnr_full)), "views": len(acc["unseen"]["mad_all"])}
    for t in acc:
        res[t] = {k: float(np.mean(v)) * (100 if k.startswith("chg") else 1) for k, v in acc[t].items() if v}
        res[t]["mad_obj_max_view"] = float(np.max(acc[t]["mad_obj"])) if acc[t]["mad_obj"] else None
    for pth in (tmp, out / "_unseen.ply", out / "_ctrl.ply"):
        pth.unlink(missing_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    log(json.dumps(res))


if __name__ == "__main__":
    main()
