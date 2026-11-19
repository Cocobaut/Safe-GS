"""
Tinh chỉnh Gaussian bị giới hạn trong khung chặt của từng vật (pipeline/s3m_mold.py), cùng một quy tắc cho mọi vật.

  0. Bỏ Gaussian của vật không đóng góp render ở view train nào (contrib.npy < eps): "không thấy thì để trống"
     (đã kiểm chứng không đổi ảnh test).
  1. Mỗi vật: lưới khoảng cách tới khung (EDT trên lưới voxel của khung, nới --margin_cm), lấy mẫu tam tuyến (grid_sample) -> d_out (cm)
     = khoảng cách từ tâm Gaussian ra ngoài khung (0 nếu ở trong).
  2. Loss = (1-l) L1 + l (1-SSIM) trên ảnh thật (view train) + lam_bound * mean_i( opacity_i * d_out_i ) trên Gaussian vật.
     Gradient đẩy Gaussian ngoài khung về khung hoặc làm nó trong suốt. Không densify. lam_bound = 0 là đối chứng (chỉ tinh chỉnh).
  3. Sau tinh chỉnh: bỏ Gaussian vật có d_out > --prune_cm và opacity < --prune_op (đã bị ép trong suốt ngoài khung).
Đo (trước / sau, 23 view test):
  - PSNR / SSIM / LPIPS ảnh đầy đủ;
  - Gaussian vật ngoài khung: số lượng (d_out > 0.5 cm), tổng opacity;
  - render RIÊNG Gaussian vật (màu 1, nền đen): alpha trung bình NGOÀI mask vật nới 15 px (rò / mờ quanh vật) và alpha trong mask co 3 px (độ phủ).
Checkpoint: ckpt/latest.ply + latest.json mỗi 1000 bước. Xuất: metrics.json, preview/*.jpg, final.ply
"""
import json
import random
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

from common import log, make_parser, load_scene, view_name, read_ply, write_ply, gaussians_from_ply, render_rgbd, to_u8, freeze
from arguments import get_combined_args, OptimizationParams, MeshingParams
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from lpipsPyTorch import LPIPSEval
from s4c_decouple import unfreeze

SHEET = ["frame_00017", "frame_00065", "frame_00113"]   # anh preview neu co; canh khac lay 3 view test dau


class FrameDist:
    """Khoảng cách (đơn vị cảnh) từ điểm ra ngoài khung một vật: EDT trên lưới voxel + phần bị kẹp ra ngoài lưới."""

    def __init__(self, xyz, vox, margin):
        lo = xyz.min(0) - margin
        hi = xyz.max(0) + margin
        n = np.ceil((hi - lo) / vox).astype(int) + 1
        occ = np.zeros(n, bool)
        ijk = np.round((xyz - lo) / vox).astype(int)
        occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
        d = np.maximum(distance_transform_edt(~occ) * vox - 0.5 * vox, 0)
        self.grid = torch.tensor(d.transpose(2, 1, 0)[None, None], dtype=torch.float32, device="cuda")   # [1,1,nz,ny,nx]
        self.lo = torch.tensor(lo, dtype=torch.float32, device="cuda")
        self.hi = torch.tensor(lo + (n - 1) * vox, dtype=torch.float32, device="cuda")

    def __call__(self, p):
        pc = torch.max(torch.min(p, self.hi), self.lo)
        g = 2 * (pc - self.lo) / (self.hi - self.lo) - 1
        d = F.grid_sample(self.grid, g[None, :, None, None, :], align_corners=True, mode="bilinear").view(-1)
        return d + (p - pc).norm(dim=1)


def main():
    p, model, pipe, ss, _, _ = make_parser("S3c constrain")
    p.add_argument("--s3", required=True)
    p.add_argument("--mold", required=True, help="thư mục S3m/<tag> (contrib.npy, frame/<vật>.npz)")
    p.add_argument("--labels", required=True)
    p.add_argument("--iters", type=int, default=5000)
    p.add_argument("--lam_bound", type=float, default=1.0)
    p.add_argument("--lr_scale", type=float, default=0.5)
    p.add_argument("--lam_dssim", type=float, default=0.2)
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--margin_cm", type=float, default=10.0)
    p.add_argument("--prune_cm", type=float, default=1.0)
    p.add_argument("--prune_op", type=float, default=0.05)
    p.add_argument("--startup_cm", type=float, default=2.0,
                   help="giai đoạn 2 khi khởi động: Gaussian vật (khung đáng tin) ngoài khung > cm này và opacity < --startup_op bị bỏ trước khi train")
    p.add_argument("--startup_op", type=float, default=0.2)
    p.add_argument("--lam_leak", type=float, default=0.0,
                   help="opt-in: alpha RIENG cua 1 vat (ngau nhien, khung dang tin) ngoai mask SAM 3 no 15 px tren view train")
    p.add_argument("--lam_cover", type=float, default=0.0,
                   help="opt-in: (1 - alpha rieng cua vat) trong mask SAM 3 co 3 px tren view train (giu vat dac)")
    p.add_argument("--leak_norm", default="image", choices=["image", "mask"],
                   help="image: L_leak = trung binh alpha ngoai mask tren ca anh (cu, rat nho); mask: tong alpha ngoai mask / dien tich mask vat")
    p.add_argument("--aux_every", type=int, default=2, help="tinh L_leak / L_cover moi N buoc")
    p.add_argument("--max_out_pct", type=float, default=20.0,
                   help="khung của vật không đáng tin (hull cụt / rỗng) nếu > tỉ lệ này Gaussian quan sát được nằm ngoài khung > 0.5 cm: không ép")
    args = get_combined_args(p)
    out = Path(args.out)
    (out / "ckpt").mkdir(parents=True, exist_ok=True); (out / "preview").mkdir(exist_ok=True)
    s3, mold = Path(args.s3), Path(args.mold)
    sup = json.loads((s3 / "support.json").read_text())
    cm = 100 * float(sup["meters_per_unit"])
    groups = json.loads((s3 / "groups.json").read_text())["objects"]
    names = [n for n in groups if (mold / "frame" / f"{n}.npz").exists()]
    IDS = {n: [int(i) for i in groups[n]["ids"]] for n in groups}   # ID mask cua tung vat lay tu S3 (khong viet cung theo canh)

    # ---------------------------------------------------------------- 0. bỏ Gaussian vật không đóng góp
    S0 = load_scene(args, model, pipe, ss)
    lab = np.load(s3 / "labels.npy")
    tot = np.load(mold / "contrib.npy")
    tmp = out / "_full.ply"
    S0["g"].save_ply(str(tmp))
    el = read_ply(tmp)
    obj_lab = {groups[n]["label"]: i for i, n in enumerate(names)}
    oid = np.array([obj_lab.get(int(l), -1) for l in lab])
    drop = (oid >= 0) & (tot < args.eps)
    start_ply = out / "start.ply"
    if not start_ply.exists():
        write_ply(start_ply, el[~drop])
        np.save(out / "oid.npy", oid[~drop])
    oid = np.load(out / "oid.npy")
    tmp.unlink(missing_ok=True)
    log(f"bỏ {int(drop.sum())} Gaussian vật không đóng góp; còn {len(oid)}")
    del S0
    torch.cuda.empty_cache()

    fd = []
    for n in names:
        z = np.load(mold / "frame" / f"{n}.npz")
        fd.append(FrameDist(z["xyz"].astype(np.float64), float(z["voxel"]), args.margin_cm / cm))
    oid_t = torch.tensor(oid, device="cuda")

    def d_out_cm(xyz):
        d = torch.zeros(len(xyz), device="cuda")
        for i, f in enumerate(fd):
            m = oid_t == i
            if m.any():
                d[m] = f(xyz[m]) * cm
        return d

    # ---------------------------------------------------------------- khung nào đáng tin (cùng quy tắc mọi vật)
    S = load_scene(args, model, pipe, ss, ply_path=start_ply)
    with torch.no_grad():
        d0 = d_out_cm(S["g"].get_xyz)
    frame_check = {}
    reliable = np.zeros(len(names), bool)
    for i, n in enumerate(names):
        m = oid_t == i
        pct = 100 * float((d0[m] > 0.5).float().mean()) if m.any() else 0.0
        reliable[i] = pct <= args.max_out_pct
        frame_check[n] = {"gaussians": int(m.sum()), "outside_pct": pct, "constrained": bool(reliable[i])}
    log(f"kiểm khung: {frame_check}")
    rel_t = torch.tensor(reliable, device="cuda")
    cons = (oid_t >= 0) & rel_t[oid_t.clamp(min=0)]

    # ---------------------------------------------------------------- lọc khởi động giai đoạn 2
    # Giai đoạn 1 (trên): Gaussian không view train nào thấy đã bỏ, nên vùng không quan sát luôn trống.
    # Giai đoạn 2: Gaussian của vật có khung đáng tin, nằm ngoài khung > --startup_cm và opacity thấp, bị bỏ TRƯỚC khi train.
    # Lần chạy lại dùng lại start_f.ply / oid_f.npy đã ghi (không tính lại).
    startup_pruned = 0
    if (out / "start_f.ply").exists():
        start_ply = out / "start_f.ply"
        oid = np.load(out / "oid_f.npy")
    else:
        with torch.no_grad():
            d0s = d_out_cm(S["g"].get_xyz)
            op0 = S["g"].get_opacity.squeeze(-1)
            rm2 = (cons & (d0s > args.startup_cm) & (op0 < args.startup_op)).cpu().numpy()
        startup_pruned = int(rm2.sum())
        if startup_pruned:
            write_ply(out / "start_f.ply", read_ply(start_ply)[~rm2])
            oid = oid[~rm2]
            np.save(out / "oid_f.npy", oid)
            start_ply = out / "start_f.ply"
    oid_t = torch.tensor(oid, device="cuda")
    cons = (oid_t >= 0) & rel_t[oid_t.clamp(min=0)]
    log(f"giai đoạn 2 khi khởi động: bỏ {startup_pruned} Gaussian vật ngoài khung; còn {int(cons.sum())} Gaussian được ép")
    del S
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------- đo
    lp = LPIPSEval(device="cuda", net_type="vgg")
    black = torch.zeros(3, device="cuda")
    ids_all = torch.tensor(sorted({i for n in names for i in IDS.get(n, [])}), device="cuda")

    def evaluate(S, g, tag):
        tv = [view_name(v) for v in S["scene"].getTestCameras()]
        sheet = [n for n in SHEET if n in tv] or tv[:3]
        ps, ss_, lps, leak, cover = [], [], [], [], []
        per_obj = {}
        objm = (oid_t >= 0)
        col = torch.zeros(len(oid_t), 3, device="cuda"); col[objm] = 1
        imgs = {}
        with torch.no_grad():
            d = d_out_cm(g.get_xyz)
            op = g.get_opacity.squeeze(-1)
            outside = objm & (d > 0.5)
            for v in S["scene"].getTestCameras():
                nm = view_name(v)
                img = render_rgbd(v, g, S)[0]
                gt = v.original_image.cuda()[:3]
                ps.append(float(psnr(img[None], gt[None]).mean())); ss_.append(float(ssim(img[None], gt[None])))
                lps.append(float(lp.criterion(img[None], gt[None])))
                a = render(v, g, S["pipe"], black, splat_args=S["splat_args"], override_color=col)["render"][0]
                f = Path(args.labels) / f"{nm}.png"
                if f.exists():
                    M = torch.isin(torch.from_numpy(cv2.imread(str(f), cv2.IMREAD_UNCHANGED).astype(np.int64)).cuda(), ids_all).float()
                    dil = F.max_pool2d(M[None, None], 31, 1, 15)[0, 0] > 0.5
                    ero = (1 - F.max_pool2d((1 - M)[None, None], 7, 1, 3)[0, 0]) > 0.5
                    leak.append(float(a[~dil].mean())); cover.append(float(a[ero].mean()))
                if f.exists():
                    lab_v = torch.from_numpy(cv2.imread(str(f), cv2.IMREAD_UNCHANGED).astype(np.int64)).cuda()
                    for i, n in enumerate(names):
                        Mi = torch.isin(lab_v, torch.tensor(IDS.get(n, [-1]), device="cuda")).float()
                        if Mi.sum() < 200:
                            continue
                        ci = torch.zeros(len(oid_t), 3, device="cuda"); ci[oid_t == i] = 1
                        ai = render(v, g, S["pipe"], black, splat_args=S["splat_args"], override_color=ci)["render"][0]
                        eri = (1 - F.max_pool2d((1 - Mi)[None, None], 7, 1, 3)[0, 0]) > 0.5
                        dli = F.max_pool2d(Mi[None, None], 31, 1, 15)[0, 0] > 0.5
                        po = per_obj.setdefault(n, {"cover": [], "leak": []})
                        if eri.sum() > 50:
                            po["cover"].append(float(ai[eri].mean()))
                        po["leak"].append(float(ai[~dli].mean()))
                if nm in sheet:
                    imgs[nm] = (img, a)
        r = {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss_)), "lpips": float(np.mean(lps)),
             "obj_gaussians": int(objm.sum()), "outside_gaussians": int(outside.sum()),
             "outside_opacity_sum": float(op[outside].sum()), "outside_pct": 100 * float(outside.sum() / objm.sum().clamp(min=1)),
             "obj_alpha_outside_mask": float(np.mean(leak)), "obj_alpha_inside_mask": float(np.mean(cover)),
             "per_object": {}}
        for i, n in enumerate(names):
            m = oid_t == i
            po = per_obj.get(n, {"cover": [], "leak": []})
            r["per_object"][n] = {"gaussians": int(m.sum()), "outside_pct": 100 * float(outside[m].float().mean()) if m.any() else None,
                                  "cover": float(np.mean(po["cover"])) if po["cover"] else None,
                                  "leak": float(np.mean(po["leak"])) if po["leak"] else None}
        log(f"[{tag}] {r}")
        return r, imgs

    res = {"dropped_unobserved": int(drop.sum()), "startup_pruned": startup_pruned, "startup_cm": args.startup_cm,
           "startup_op": args.startup_op, "lam_bound": args.lam_bound, "iters": args.iters,
           "lam_leak": args.lam_leak, "lam_cover": args.lam_cover, "aux_every": args.aux_every, "leak_norm": args.leak_norm}
    S = load_scene(args, model, pipe, ss, ply_path=start_ply)
    if not (out / "metrics_before.json").exists():
        rb, imgs_b = evaluate(S, S["g"], "before")
        (out / "metrics_before.json").write_text(json.dumps(rb))
        torch.save({k: (a.cpu(), b.cpu()) for k, (a, b) in imgs_b.items()}, out / "imgs_before.pt")
    res["before"] = json.loads((out / "metrics_before.json").read_text())

    # ---------------------------------------------------------------- tinh chỉnh
    start, ply = 0, start_ply
    if (out / "ckpt" / "latest.json").exists():
        start = json.loads((out / "ckpt" / "latest.json").read_text())["iteration"]
        ply = out / "ckpt" / "latest.ply"
    if start < args.iters:
        S = load_scene(args, model, pipe, ss, ply_path=ply)
        g = S["g"]
        pa = ArgumentParser(); op_ = OptimizationParams(pa); mp_ = MeshingParams(pa); a2 = pa.parse_args([])
        opt, mesh = op_.extract(a2), mp_.extract(a2)
        g.spatial_lr_scale = S["scene"].cameras_extent
        unfreeze(g)
        g.training_setup(opt, mesh, None)
        for pg in g.optimizer.param_groups:
            pg["lr"] *= args.lr_scale
        xyz_lr = g.xyz_scheduler_args(opt.iterations) * args.lr_scale
        rng = random.Random(start)
        views = list(S["train"])
        objm = cons
        logf = open(out / "loss.csv", "a")
        use_aux = args.lam_leak > 0 or args.lam_cover > 0
        cons_ids = [i for i in range(len(names)) if reliable[i]]
        lab_cache = {}

        def view_labels(v):
            nm = view_name(v)
            if nm not in lab_cache:
                f = Path(args.labels) / f"{nm}.png"
                lab_cache[nm] = torch.from_numpy(cv2.imread(str(f), cv2.IMREAD_UNCHANGED).astype(np.int16)) if f.exists() else None
            return lab_cache[nm]

        lleak_v, lcov_v = 0.0, 0.0
        for it in range(start + 1, args.iters + 1):
            for pg in g.optimizer.param_groups:
                if pg["name"] == "xyz":
                    pg["lr"] = xyz_lr
            v = rng.choice(views)
            gt = v.original_image.cuda()[:3]
            img = render(v, g, S["pipe"], S["bg"], splat_args=S["splat_args"], gt_color=gt)["render"][:3]
            loss = (1 - args.lam_dssim) * l1_loss(img, gt) + args.lam_dssim * (1 - ssim(img, gt))
            lb = torch.zeros((), device="cuda")
            if args.lam_bound > 0:
                xyz = g.get_xyz[objm]
                d = torch.zeros(len(xyz), device="cuda")
                oo = oid_t[objm]
                for i, f in enumerate(fd):
                    m = oo == i
                    if m.any():
                        d[m] = f(xyz[m]) * cm
                lb = (g.get_opacity[objm].squeeze(-1) * d).mean()
                loss = loss + args.lam_bound * lb
            # L_leak / L_cover: alpha rieng vat i (Gaussian khac van che, mau 0) so voi mask SAM 3 cua vat i tren view nay.
            # Gradient cua phan nay chi giu cho Gaussian cua vat i (khong de vat khac che bot / lam trong suot de ha loss).
            if use_aux and cons_ids and it % args.aux_every == 0:
                lv = view_labels(v)
                cand = []
                if lv is not None:
                    lvg = lv.cuda().long()
                    for i in cons_ids:
                        Mi = torch.isin(lvg, torch.tensor(IDS[names[i]], device="cuda"))
                        if int(Mi.sum()) >= 200:
                            cand.append((i, Mi.float()))
                if cand:
                    i, Mi = cand[rng.randrange(len(cand))]
                    sel = oid_t == i
                    ci = torch.zeros(len(oid_t), 3, device="cuda"); ci[sel] = 1
                    ai = render(v, g, S["pipe"], black, splat_args=S["splat_args"], override_color=ci)["render"][0]
                    dli = F.max_pool2d(Mi[None, None], 31, 1, 15)[0, 0] > 0.5
                    eri = (1 - F.max_pool2d((1 - Mi)[None, None], 7, 1, 3)[0, 0]) > 0.5
                    l_leak = (ai[~dli].sum() / Mi.sum().clamp(min=1.0)) if args.leak_norm == "mask" else ai[~dli].mean()
                    l_cov = (1 - ai[eri]).clamp(min=0).mean() if int(eri.sum()) > 50 else torch.zeros((), device="cuda")
                    laux = args.lam_leak * l_leak + args.lam_cover * l_cov
                    laux.backward()
                    keep = sel.float()
                    for pg in g.optimizer.param_groups:
                        for t in pg["params"]:
                            if t.grad is not None and t.dim() > 0 and t.shape[0] == len(keep):
                                t.grad.mul_(keep.view(-1, *([1] * (t.grad.dim() - 1))))
                    lleak_v, lcov_v = float(l_leak), float(l_cov)
            loss.backward()
            g.optimizer.step(); g.optimizer.zero_grad(set_to_none=True)
            if it % 50 == 0:
                logf.write(f"{it},{loss.item():.5f},{lb.item():.5f},{lleak_v:.5f},{lcov_v:.5f}\n"); logf.flush()
            if it % 1000 == 0 or it == args.iters:
                g.save_ply(str(out / "ckpt" / "latest_tmp.ply"))
                (out / "ckpt" / "latest_tmp.ply").replace(out / "ckpt" / "latest.ply")
                (out / "ckpt" / "latest.json").write_text(json.dumps({"iteration": it}))
                log(f"tinh chỉnh {it}/{args.iters} loss {loss.item():.4f} bound {lb.item():.4f} leak {lleak_v:.4f} cover_miss {lcov_v:.4f} (checkpoint)")
        logf.close()
        freeze(g)
        del g, S
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- tỉa + đo sau
    S = load_scene(args, model, pipe, ss, ply_path=out / "ckpt" / "latest.ply")
    g = S["g"]
    with torch.no_grad():
        d = d_out_cm(g.get_xyz)
        op = g.get_opacity.squeeze(-1)
        pr = (cons & (d > args.prune_cm) & (op < args.prune_op)).cpu().numpy()
    tmp = out / "_after.ply"
    g.save_ply(str(tmp))
    el2 = read_ply(tmp)
    write_ply(out / "final.ply", el2[~pr])
    np.save(out / "oid_final.npy", oid[~pr])
    tmp.unlink(missing_ok=True)
    res["pruned_after"] = int(pr.sum())
    oid_t = torch.tensor(oid[~pr], device="cuda")
    res["frame_check"] = frame_check
    S = load_scene(args, model, pipe, ss, ply_path=out / "final.ply")
    res["after"], imgs_a = evaluate(S, S["g"], "after")
    ib = torch.load(out / "imgs_before.pt")
    for nm, (img, a) in imgs_a.items():
        if nm in ib:
            b_img, b_a = ib[nm][0].cuda(), ib[nm][1].cuda()
            sep = torch.ones(3, img.shape[1], 6, device="cuda")
            row = torch.cat([b_img, sep, img, sep, b_a[None].repeat(3, 1, 1), sep, a[None].repeat(3, 1, 1)], 2)
            cv2.imwrite(str(out / "preview" / f"{nm}.jpg"), to_u8(row)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
    (out / "metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
