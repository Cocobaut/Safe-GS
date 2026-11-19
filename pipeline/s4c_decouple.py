"""
S4c (thử nghiệm): tinh chỉnh CoMe với 2 nhóm Gaussian (vật xoá được / nền) và loss alpha theo nhóm.

Ý tưởng (xem RESULTS.md mục 6): vật dẹt bị "vẽ" lên chính Gaussian mặt bàn, nên cần phạt nền khi nó tô màu lên pixel của vật,
và phạt Gaussian của vật khi nó phủ ra ngoài mask của vật.
  1. Gán nhóm cho từng Gaussian bằng bỏ phiếu theo đóng góp render (như S3) với mask SAM 3: nhóm vật = hợp các ID vật xoá được.
     Gaussian chưa ai thấy (tổng đóng góp ~ 0) lấy nhóm theo láng giềng gần (kNN, như interior_knn của S3).
  2. Mỗi vòng huấn luyện: 2 lượt render. Lượt 1: RGB (ảnh ghép tất cả Gaussian). Lượt 2: override_color = [là-vật, là-nền, 1]
     nên kênh 0,1,2 là alpha của vật, alpha của nền, alpha tổng TRONG ẢNH GHÉP (có che bởi mọi Gaussian trước nó).
  3. Loss = (1-l) L1 + l (1-SSIM) trên RGB
          + lam_sil * [ BCE(alpha_vật, mask_vật) có trọng số 0 ở vành biên rộng 2*band+1 px + mean(alpha_vật^2 ngoài vành) ]
          + lam_leak * mean(alpha_nền^2 trong mask vật đã co lại band px)
     Vành biên bỏ qua vì mask SAM 3 có khe ~2 px không nhãn và biên sai số.
  4. Không densify / prune (cùng cách S4b); lr = lr mặc định CoMe x lr_scale. Chỉ view train (23 view test để đo held-out).
Đo sau khi xong (và cho iters = 0 để có đối chứng "v1 với nhãn SAM 3"):
  - PSNR / SSIM / LPIPS(vgg) trên 23 view test (ảnh ghép đầy đủ);
  - bỏ nhóm vật, render nhóm nền: % pixel trong mask vật mà ảnh nền vẫn giống ảnh thật (|sai khác| < 0.08, tức vật còn nguyên hình),
    % pixel thủng (alpha nền < 0.5); ảnh trước | sau;
  - xuất S3like/gaussians/objects_all.ply, S4like/background_finetuned.ply để s10_ghost.py đo alpha ma.
Checkpoint: nhãn lưu ngay sau khi bỏ phiếu; mỗi ckpt_every iter lưu ckpt/latest.ply + latest.json, chạy lại thì tiếp tục.
Xuất: metrics.json, labels_obj.npy, preview/*.jpg
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
from scipy.spatial import cKDTree

from common import (log, make_parser, load_scene, view_name, freeze, read_ply, write_ply, gaussians_from_ply, render_rgbd,
                    to_u8)
from arguments import OptimizationParams, MeshingParams, get_combined_args
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from lpipsPyTorch import LPIPSEval
from s3_label import accumulate

OBJ_IDS = [80, 70, 60, 100, 90, 50, 130, 110, 170, 171, 180, 181, 160, 161]      # ID SAM 3 (bản 3) của vật trên bàn xoá được
SHEET_VIEWS = ["frame_00017", "frame_00065", "frame_00113"]


def unfreeze(g):
    for t in (g._xyz, g._features_dc, g._features_rest, g._opacity, g._scaling, g._rotation, g._confidence):
        t.requires_grad_(True)


def main():
    p, model, pipe, ss, _, _ = make_parser("S4c decouple")
    p.add_argument("--labels", type=str, required=True)
    p.add_argument("--support", type=str, required=True)
    p.add_argument("--iters", type=int, default=8000)
    p.add_argument("--lam_sil", type=float, default=0.1)
    p.add_argument("--lam_leak", type=float, default=0.5)
    p.add_argument("--assign_ratio", type=float, default=0.5)
    p.add_argument("--lr_scale", type=float, default=0.5)
    p.add_argument("--lam_dssim", type=float, default=0.2)
    p.add_argument("--band", type=int, default=3)
    p.add_argument("--ckpt_every", type=int, default=1000)
    args = get_combined_args(p)
    out = Path(args.out); (out / "ckpt").mkdir(parents=True, exist_ok=True); (out / "preview").mkdir(exist_ok=True)
    labels_dir = Path(args.labels)
    black = torch.zeros(3, device="cuda")
    obj_ids_t = torch.tensor(OBJ_IDS, device="cuda")

    start, ply = 0, None
    if (out / "ckpt" / "latest.json").exists():
        start = json.loads((out / "ckpt" / "latest.json").read_text())["iteration"]
        ply = out / "ckpt" / "latest.ply"
        log(f"♻️ tiếp tục từ iter {start}")
    S = load_scene(args, model, pipe, ss, ply_path=ply)
    g = S["g"]
    lab_cache = {}

    def lab_of(view):
        nm = view_name(view)
        if nm not in lab_cache:
            f = labels_dir / f"{nm}.png"
            lab_cache[nm] = torch.from_numpy(cv2.imread(str(f), cv2.IMREAD_UNCHANGED).astype(np.int64)).cuda() \
                if f.exists() else None
        return lab_cache[nm]

    # ------------------------------------------------------------------ 1. nhóm Gaussian (vật / nền)
    lf = out / "labels_obj.npy"
    if lf.exists():
        obj_np = np.load(lf)
    else:
        freeze(g)

        def mask_fn(view):
            lab = lab_of(view)
            if lab is None:
                return None
            m = torch.isin(lab, obj_ids_t).float()
            return torch.stack([m, torch.ones_like(m)], 0)

        W = accumulate(S, g, S["train"], mask_fn, 2, out / "partial_vote.pt")
        ratio = (W[:, 0] / W[:, 1].clamp(min=1e-12)).cpu().numpy(); tot = W[:, 1].cpu().numpy()
        obj_np = (ratio >= args.assign_ratio) & (tot > 1e-4)
        seen = tot > 1e-4
        xyz = g.get_xyz.detach().cpu().numpy()
        n_unseen = int((~seen).sum())
        if n_unseen and seen.sum() > 8:                             # Gaussian chưa ai thấy: theo láng giềng gần
            tree = cKDTree(xyz[seen])
            idx = np.where(~seen)[0]
            d, nn = tree.query(xyz[idx], k=8)
            smax = g.get_scaling.max(1).values.detach().cpu().numpy()[idx]
            obj_np[idx] = (obj_np[seen][nn].mean(1) >= 0.5) & (d.mean(1) < 3 * smax)
        np.save(lf, obj_np)
        (out / "labels_info.json").write_text(json.dumps({
            "gaussians": int(len(obj_np)), "obj": int(obj_np.sum()), "unseen": n_unseen, "assign_ratio": args.assign_ratio}))
        (out / "partial_vote.pt").unlink(missing_ok=True)
        log(f"nhóm vật: {int(obj_np.sum())} / {len(obj_np)} Gaussian (chưa ai thấy: {n_unseen})")
    objT = torch.from_numpy(obj_np).cuda()
    ind = torch.stack([objT.float(), 1 - objT.float(), torch.ones_like(objT, dtype=torch.float32)], 1).contiguous()

    # ------------------------------------------------------------------ 2. tinh chỉnh
    if start < args.iters:
        pa = ArgumentParser(); op_ = OptimizationParams(pa); mp_ = MeshingParams(pa); a2 = pa.parse_args([])
        opt, mesh = op_.extract(a2), mp_.extract(a2)
        g.spatial_lr_scale = S["scene"].cameras_extent
        unfreeze(g)
        g.training_setup(opt, mesh, None)
        for pg in g.optimizer.param_groups:
            pg["lr"] *= args.lr_scale
        xyz_lr = g.xyz_scheduler_args(opt.iterations) * args.lr_scale
        rng = random.Random(start)
        views = [v for v in S["train"] if lab_of(v) is not None]
        k = 2 * args.band + 1
        logf = open(out / "loss.csv", "a")
        for it in range(start + 1, args.iters + 1):
            for pg in g.optimizer.param_groups:
                if pg["name"] == "xyz":
                    pg["lr"] = xyz_lr
            v = rng.choice(views)
            gt = v.original_image.cuda()[:3]
            M = torch.isin(lab_of(v), obj_ids_t).float()
            dil = F.max_pool2d(M[None, None], k, 1, k // 2)[0, 0]
            ero = 1 - F.max_pool2d((1 - M)[None, None], k, 1, k // 2)[0, 0]
            w_out, w_in = (dil < 0.5).float(), (ero > 0.5).float()
            w_sil = w_out + w_in                                    # 0 trong vành biên
            img = render(v, g, S["pipe"], S["bg"], splat_args=S["splat_args"], gt_color=gt)["render"][:3]
            l_rgb = (1 - args.lam_dssim) * l1_loss(img, gt) + args.lam_dssim * (1 - ssim(img, gt))
            a = render(v, g, S["pipe"], black, splat_args=S["splat_args"], override_color=ind)["render"][:3]
            a_obj, a_bg = a[0].clamp(1e-4, 1 - 1e-4), a[1]
            bce = -(M * torch.log(a_obj) + (1 - M) * torch.log(1 - a_obj))
            l_sil = (w_sil * bce).sum() / w_sil.sum().clamp(min=1) + (w_out * a_obj ** 2).sum() / w_out.sum().clamp(min=1)
            l_leak = (w_in * a_bg ** 2).sum() / w_in.sum().clamp(min=1)
            loss = l_rgb + args.lam_sil * l_sil + args.lam_leak * l_leak
            loss.backward()
            g.optimizer.step(); g.optimizer.zero_grad(set_to_none=True)
            if it % 50 == 0:
                logf.write(f"{it},{loss.item():.5f},{l_rgb.item():.5f},{l_sil.item():.5f},{l_leak.item():.5f}\n"); logf.flush()
            if it % args.ckpt_every == 0 or it == args.iters:
                g.save_ply(str(out / "ckpt" / "latest_tmp.ply"))
                (out / "ckpt" / "latest_tmp.ply").replace(out / "ckpt" / "latest.ply")
                (out / "ckpt" / "latest.json").write_text(json.dumps({"iteration": it}))
                log(f"tinh chỉnh {it}/{args.iters} rgb {l_rgb.item():.4f} sil {l_sil.item():.4f} leak {l_leak.item():.4f} (checkpoint)")
        logf.close()
        freeze(g)
        with torch.no_grad():
            g.compute_3D_filter(S["train"])

    # ------------------------------------------------------------------ 3. đo
    lp = LPIPSEval(device="cuda", net_type="vgg")
    test_views = list(S["scene"].getTestCameras())
    ps, ss_, lps = [], [], []
    with torch.no_grad():
        for v in test_views:
            img = render_rgbd(v, g, S)[0]
            gt = v.original_image.cuda()[:3]
            ps.append(float(psnr(img[None], gt[None]).mean())); ss_.append(float(ssim(img[None], gt[None])))
            lps.append(float(lp.criterion(img[None], gt[None])))
    res = {"iters": args.iters, "lam_sil": args.lam_sil, "lam_leak": args.lam_leak, "assign_ratio": args.assign_ratio,
           "obj_gaussians": int(obj_np.sum()), "gaussians": int(len(obj_np)),
           "nvs_test": {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss_)), "lpips": float(np.mean(lps))}}
    log(f"NVS test: {res['nvs_test']}")

    tmp = out / "full.ply"
    g.save_ply(str(tmp))
    el = read_ply(tmp)
    write_ply(out / "S3like" / "gaussians" / "objects_all.ply", el[obj_np])
    write_ply(out / "S4like" / "background_finetuned.ply", el[~obj_np])
    shutil.copy2(args.support, out / "S3like" / "support.json")
    tmp.unlink(missing_ok=True)
    sh = S["dataset"].sh_degree
    g_bg = gaussians_from_ply(out / "S4like" / "background_finetuned.ply", sh, S["train"])
    ones = torch.ones(g_bg.get_xyz.shape[0], 3, device="cuda")
    same, hole, n_pix = [], [], 0
    with torch.no_grad():
        for v in test_views:
            lab = lab_of(v)
            if lab is None:
                continue
            M = torch.isin(lab, obj_ids_t).float()
            ero = (1 - F.max_pool2d((1 - M)[None, None], 2 * args.band + 1, 1, args.band)[0, 0]) > 0.5
            if ero.sum() < 50:
                continue
            rgb = render_rgbd(v, g_bg, S)[0]
            al = render(v, g_bg, S["pipe"], black, splat_args=S["splat_args"], override_color=ones)["render"][0]
            gt = v.original_image.cuda()[:3]
            diff = (rgb - gt).abs().mean(0)
            same.append(float((diff[ero] < 0.08).float().mean())); hole.append(float((al[ero] < 0.5).float().mean()))
            if view_name(v) in SHEET_VIEWS:
                full = render_rgbd(v, g, S)[0]
                sep = torch.ones(3, full.shape[1], 8, device="cuda")
                cv2.imwrite(str(out / "preview" / f"{view_name(v)}.jpg"), to_u8(torch.cat([gt, sep, full, sep, rgb], 2))[..., ::-1],
                            [cv2.IMWRITE_JPEG_QUALITY, 88])
    res["removal"] = {"views": len(same), "remnant_looks_like_object_pct": 100 * float(np.mean(same)),
                      "hole_pct": 100 * float(np.mean(hole))}
    log(f"xoá nhóm vật: {res['removal']}")
    (out / "metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
