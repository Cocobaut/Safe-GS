"""
S4d (thử nghiệm): cắt Gaussian nền trong vết chân vật, lấp mặt bàn (plane_fill của S4b), rồi tinh chỉnh theo ảnh inpaint (như S4b).

Đầu vào: kết quả S4c (nhãn SAM 3): --src <S4c/tag> có S3like/{support.json, gaussians/objects_all.ply}, S4like/background_finetuned.ply.
Cắt Gaussian nền (lưới 1 cm, vết chân từ Gaussian vật trên bàn) nếu:
  (a) tâm nằm trong vết chân nới --cut_dilate ô, độ cao > -1 cm (trên mặt bàn), hoặc
  (b) Gaussian lớn gần vết chân: khoảng cách từ tâm tới vết chân < --reach x scale lớn nhất và độ cao < --low_cm.
Bảo vệ (--protect 1): Gaussian nền thuộc con cừu / gấu (bỏ phiếu theo mask SAM 3 ID 20, 30, ngưỡng 0.5) không bị cắt.
Lấp: plane_fill (S4b) với dilate_cells = --fill_dilate, màu lấy từ vành ngoài.
Tinh chỉnh (--ft_iters > 0, cần --inpainted): như S4b (L1 + SSIM, không densify, lr mặc định x --lr_scale) trên view train.
Đo trên 23 view test, nền sau cắt + lấp (cf) và sau tinh chỉnh (ft):
  - % pixel trong mask vật còn giống ảnh thật (|sai khác| < 0.08) và % thủng (alpha < 0.5);
  - PSNR vùng KHÔNG-vật (ngoài mask vật nới 15 px) và vùng cừu / gấu (mask ID 20, 30) so với ảnh thật: đo phần cảnh phải giữ nguyên;
  - ảnh: ảnh thật | cf | ft.
Checkpoint: background_filled.ply sau lấp, ckpt/latest.ply mỗi 500 bước.
Xuất S3like / S4like (cho s10_ghost.py), metrics.json, cutfill.json, preview/*.jpg
"""
import json
import random
import shutil
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common import (log, make_parser, load_scene, view_name, read_ply, write_ply, gaussians_from_ply, render_rgbd, to_u8,
                    freeze)
from arguments import get_combined_args, OptimizationParams, MeshingParams
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from s4_background import plane_fill, plane_basis
from s3_label import accumulate
from s4c_decouple import OBJ_IDS, SHEET_VIEWS, unfreeze

OCC_IDS = [20, 30]      # con cừu / gấu bông (SAM 3)


def psnr_masked(a, b, m):
    mse = (((a - b) ** 2).mean(0))[m].mean().clamp(min=1e-10)
    return float(-10 * torch.log10(mse))


def main():
    p, model, pipe, ss, _, _ = make_parser("S4d cut+fill+ft")
    p.add_argument("--src", type=str, required=True)
    p.add_argument("--labels", type=str, required=True)
    p.add_argument("--inpainted", type=str, default="")
    p.add_argument("--cut_dilate", type=int, default=2)
    p.add_argument("--reach", type=float, default=2.0)
    p.add_argument("--low_cm", type=float, default=3.0)
    p.add_argument("--fill_dilate", type=int, default=4)
    p.add_argument("--fill_grid", type=float, default=0.01, help="cạnh ô lấp (mét); nhỏ hơn -> nhiều Gaussian dẹt nhỏ hơn")
    p.add_argument("--protect", type=int, default=1)
    p.add_argument("--ft_iters", type=int, default=0)
    p.add_argument("--freeze_region", type=int, default=0, help="1: khi tinh chỉnh, đóng băng Gaussian lấp và Gaussian có tâm trong "
                   "vết chân nới --freeze_dilate ô (chỉ cho phần bàn bên ngoài học thêm)")
    p.add_argument("--freeze_dilate", type=int, default=6)
    p.add_argument("--ft_mode", type=str, default="inpaint", help="inpaint: đích = ảnh ProPainter (S4a); masked: đích = ảnh thật, "
                   "KHÔNG giám sát trong mask vật nới 15 px (đích = chính ảnh render, gradient 0)")
    p.add_argument("--lr_scale", type=float, default=0.5)
    p.add_argument("--lam_dssim", type=float, default=0.2)
    p.add_argument("--band", type=int, default=3)
    args = get_combined_args(p)
    out, src = Path(args.out), Path(args.src)
    (out / "preview").mkdir(parents=True, exist_ok=True)
    (out / "ckpt").mkdir(exist_ok=True)
    s3 = src / "S3like"
    sup = json.loads((s3 / "support.json").read_text())
    n = np.array(sup["table"]["normal"], np.float64)
    d = float(sup["table"]["d"])
    mpu = float(sup["meters_per_unit"])
    labels_dir = Path(args.labels)
    S = load_scene(args, model, pipe, ss)
    sh = S["dataset"].sh_degree
    black = torch.zeros(3, device="cuda")
    obj_t = torch.tensor(OBJ_IDS, device="cuda")
    occ_t = torch.tensor(OCC_IDS, device="cuda")
    lab_cache = {}

    def lab_of(vw):
        nm = view_name(vw)
        if nm not in lab_cache:
            f = labels_dir / f"{nm}.png"
            lab_cache[nm] = torch.from_numpy(cv2.imread(str(f), cv2.IMREAD_UNCHANGED).astype(np.int64)).cuda() \
                if f.exists() else None
        return lab_cache[nm]

    filled_p = out / "background_filled.ply"
    info = {}
    if filled_p.exists():
        info = json.loads((out / "cutfill.json").read_text())
    else:
        ob = read_ply(s3 / "gaussians" / "objects_all.ply")
        oxyz = np.stack([ob["x"], ob["y"], ob["z"]], 1).astype(np.float64)
        s_w = 1.0 if np.median(oxyz @ n + d) >= 0 else -1.0
        el = read_ply(src / "S4like" / "background_finetuned.ply")
        P = np.stack([el["x"], el["y"], el["z"]], 1).astype(np.float64)
        h_cm = s_w * (P @ n + d) * mpu * 100
        u, v = plane_basis(n)
        p0 = -d * n
        cell = 0.01 / mpu
        oa = np.floor((oxyz - p0) @ u / cell).astype(int)
        ob_ = np.floor((oxyz - p0) @ v / cell).astype(int)
        amin, bmin = oa.min() - 30, ob_.min() - 30
        foot = np.zeros((ob_.max() - bmin + 31, oa.max() - amin + 31), np.uint8)
        foot[ob_ - bmin, oa - amin] = 1
        k3 = np.ones((3, 3), np.uint8)
        foot = cv2.dilate(foot, k3)
        footc = cv2.dilate(foot, k3, iterations=args.cut_dilate) > 0
        dist = cv2.distanceTransform((foot == 0).astype(np.uint8), cv2.DIST_L2, 5)
        ia = np.floor((P - p0) @ u / cell).astype(int) - amin
        ib = np.floor((P - p0) @ v / cell).astype(int) - bmin
        ok = (ia >= 0) & (ia < foot.shape[1]) & (ib >= 0) & (ib < foot.shape[0])
        in_c = np.zeros(len(P), bool)
        in_c[ok] = footc[ib[ok], ia[ok]]
        dd = np.full(len(P), 1e9)
        dd[ok] = dist[ib[ok], ia[ok]]
        smax = np.exp(np.stack([el["scale_0"], el["scale_1"], el["scale_2"]], 1)).max(1)
        a_ = in_c & (h_cm > -1.0)
        b_ = (dd * cell < args.reach * smax) & (h_cm < args.low_cm) & (h_cm > -1.0)
        cut = a_ | b_
        n_prot = 0
        if args.protect:
            tmp_ply = out / "_bg_src.ply"
            write_ply(tmp_ply, el)
            g_src = gaussians_from_ply(tmp_ply, sh, S["train"])

            def mask_fn(vw):
                lab = lab_of(vw)
                if lab is None:
                    return None
                m = torch.isin(lab, occ_t).float()
                return torch.stack([m, torch.ones_like(m)], 0)

            W = accumulate(S, g_src, S["train"], mask_fn, 2, out / "partial_vote.pt")
            ratio = (W[:, 0] / W[:, 1].clamp(min=1e-12)).cpu().numpy()
            tot = W[:, 1].cpu().numpy()
            prot = (ratio >= 0.5) & (tot > 1e-4)
            n_prot = int((cut & prot).sum())
            cut &= ~prot
            (out / "partial_vote.pt").unlink(missing_ok=True)
            tmp_ply.unlink(missing_ok=True)
            del g_src
            torch.cuda.empty_cache()
        log(f"cắt {int(cut.sum())} / {len(P)} (a {int(a_.sum())}, b {int(b_.sum())}, được bảo vệ {n_prot})")
        tmp = out / "_s3cut"
        (tmp / "gaussians").mkdir(parents=True, exist_ok=True)
        write_ply(tmp / "gaussians" / "background_with_support.ply", el[~cut])
        shutil.copy2(s3 / "gaussians" / "objects_all.ply", tmp / "gaussians" / "objects_all.ply")
        shutil.copy2(s3 / "support.json", tmp / "support.json")
        filled, info = plane_fill(tmp, {"grid": args.fill_grid, "dilate_cells": args.fill_dilate, "opacity": 0.9}, mpu)
        info.update({"cut": int(cut.sum()), "cut_a": int(a_.sum()), "cut_b": int(b_.sum()), "protected_not_cut": n_prot,
                     "kept": int((~cut).sum())})
        write_ply(filled_p, filled)
        footf = cv2.dilate(foot, k3, iterations=args.freeze_dilate) > 0
        kept_in = np.zeros(int((~cut).sum()), bool)
        ik, jk, okk = ia[~cut], ib[~cut], ok[~cut]
        kept_in[okk] = footf[jk[okk], ik[okk]]
        np.save(out / "freeze_region.npy", np.concatenate([kept_in, np.ones(len(filled) - len(kept_in), bool)]))
        (out / "cutfill.json").write_text(json.dumps(info, indent=2))
        log(f"lấp: {info}")
    (out / "S3like" / "gaussians").mkdir(parents=True, exist_ok=True)
    shutil.copy2(s3 / "gaussians" / "objects_all.ply", out / "S3like" / "gaussians" / "objects_all.ply")
    shutil.copy2(s3 / "support.json", out / "S3like" / "support.json")
    (out / "S4like").mkdir(exist_ok=True)

    def evaluate(ply, tag):
        gb = gaussians_from_ply(ply, sh, S["train"])
        ones = torch.ones(gb.get_xyz.shape[0], 3, device="cuda")
        same, hole, p_non, p_occ = [], [], [], []
        imgs = {}
        with torch.no_grad():
            for vw in S["scene"].getTestCameras():
                lab = lab_of(vw)
                if lab is None:
                    continue
                M = torch.isin(lab, obj_t).float()
                ero = (1 - F.max_pool2d((1 - M)[None, None], 2 * args.band + 1, 1, args.band)[0, 0]) > 0.5
                dil = F.max_pool2d(M[None, None], 31, 1, 15)[0, 0] > 0.5
                occ = torch.isin(lab, occ_t)
                rgb = render_rgbd(vw, gb, S)[0]
                gt = vw.original_image.cuda()[:3]
                if (~dil).sum() > 50:
                    p_non.append(psnr_masked(rgb, gt, ~dil))
                if occ.sum() > 50:
                    p_occ.append(psnr_masked(rgb, gt, occ))
                if ero.sum() >= 50:
                    al = render(vw, gb, S["pipe"], black, splat_args=S["splat_args"], override_color=ones)["render"][0]
                    diff = (rgb - gt).abs().mean(0)
                    same.append(float((diff[ero] < 0.08).float().mean()))
                    hole.append(float((al[ero] < 0.5).float().mean()))
                if view_name(vw) in SHEET_VIEWS:
                    imgs[view_name(vw)] = (gt, rgb)
        r = {"views": len(same), "remnant_looks_like_object_pct": 100 * float(np.mean(same)),
             "hole_pct": 100 * float(np.mean(hole)), "psnr_non_object": float(np.mean(p_non)),
             "psnr_sheep_bear": float(np.mean(p_occ)) if p_occ else None}
        log(f"[{tag}] {r}")
        return r, imgs

    res = {"cut": info, "args": {k: getattr(args, k) for k in ("cut_dilate", "reach", "low_cm", "fill_dilate", "protect",
                                                              "ft_iters", "lr_scale", "ft_mode", "freeze_region", "freeze_dilate")}}
    res["src"], _ = evaluate(src / "S4like" / "background_finetuned.ply", "src")      # đối chứng: nền S4c chưa cắt
    res["cf"], imgs_cf = evaluate(filled_p, "cf")
    final = filled_p
    imgs_ft = {}
    if args.ft_iters > 0:
        inp = Path(args.inpainted)
        targets = {}
        for vw in S["train"]:
            f = inp / f"{view_name(vw)}.png"
            if args.ft_mode == "masked":
                targets[view_name(vw)] = vw.original_image.cuda()[:3].contiguous()
            elif f.exists():
                im = cv2.imread(str(f))[..., ::-1].astype(np.float32) / 255.0
                im = cv2.resize(im, (int(vw.image_width), int(vw.image_height)), interpolation=cv2.INTER_AREA)
                targets[view_name(vw)] = torch.from_numpy(im).permute(2, 0, 1).contiguous().cuda()
        views = [vw for vw in S["train"] if view_name(vw) in targets]
        log(f"{len(views)} view có ảnh đích")
        start, ply = 0, filled_p
        if (out / "ckpt" / "latest.json").exists():
            start = json.loads((out / "ckpt" / "latest.json").read_text())["iteration"]
            ply = out / "ckpt" / "latest.ply"
            log(f"♻️ tiếp tục tinh chỉnh từ iter {start}")
        if start < args.ft_iters:
            S2 = load_scene(args, model, pipe, ss, ply_path=ply)
            g = S2["g"]
            pa = ArgumentParser()
            op_ = OptimizationParams(pa)
            mp_ = MeshingParams(pa)
            a2 = pa.parse_args([])
            opt, mesh = op_.extract(a2), mp_.extract(a2)
            g.spatial_lr_scale = S2["scene"].cameras_extent
            unfreeze(g)
            g.training_setup(opt, mesh, None)
            for pg in g.optimizer.param_groups:
                pg["lr"] *= args.lr_scale
            xyz_lr = g.xyz_scheduler_args(opt.iterations) * args.lr_scale
            frz = None
            if args.freeze_region and (out / "freeze_region.npy").exists():
                frz = torch.from_numpy(np.load(out / "freeze_region.npy")).cuda()
                log(f"đóng băng {int(frz.sum())} / {len(frz)} Gaussian (lấp + quanh vết chân)")
            rng = random.Random(start)
            logf = open(out / "loss.csv", "a")
            for it in range(start + 1, args.ft_iters + 1):
                for pg in g.optimizer.param_groups:
                    if pg["name"] == "xyz":
                        pg["lr"] = xyz_lr
                vw = rng.choice(views)
                gt = targets[view_name(vw)]
                img = render(vw, g, S2["pipe"], S2["bg"], splat_args=S2["splat_args"], gt_color=gt)["render"][:3]
                if args.ft_mode == "masked":
                    lab = lab_of(vw)
                    if lab is not None:
                        Mo = torch.isin(lab, obj_t).float()
                        keep = (F.max_pool2d(Mo[None, None], 31, 1, 15)[0, 0] < 0.5)[None]
                        gt = torch.where(keep, gt, img.detach())
                loss = (1 - args.lam_dssim) * l1_loss(img, gt) + args.lam_dssim * (1 - ssim(img, gt))
                loss.backward()
                if frz is not None:
                    for pg in g.optimizer.param_groups:
                        for q in pg["params"]:
                            if q.grad is not None and q.shape[0] == frz.shape[0]:
                                q.grad[frz] = 0
                g.optimizer.step()
                g.optimizer.zero_grad(set_to_none=True)
                if it % 50 == 0:
                    logf.write(f"{it},{loss.item():.6f}\n")
                    logf.flush()
                if it % 500 == 0 or it == args.ft_iters:
                    g.save_ply(str(out / "ckpt" / "latest_tmp.ply"))
                    (out / "ckpt" / "latest_tmp.ply").replace(out / "ckpt" / "latest.ply")
                    (out / "ckpt" / "latest.json").write_text(json.dumps({"iteration": it}))
                    log(f"tinh chỉnh {it}/{args.ft_iters} loss {loss.item():.4f} (checkpoint)")
            logf.close()
            freeze(g)
            del g, S2
            torch.cuda.empty_cache()
        final = out / "ckpt" / "latest.ply"
        res["ft"], imgs_ft = evaluate(final, "ft")
    shutil.copy2(final, out / "S4like" / "background_finetuned.ply")
    for nm, (gt, rgb) in imgs_cf.items():
        sep = torch.ones(3, gt.shape[1], 8, device="cuda")
        cols = [gt, sep, rgb] + ([sep, imgs_ft[nm][1]] if nm in imgs_ft else [])
        cv2.imwrite(str(out / "preview" / f"{nm}.jpg"), to_u8(torch.cat(cols, 2))[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
    (out / "metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
