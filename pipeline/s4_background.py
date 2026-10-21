"""
S4b: lấp nền sau khi bỏ vật.
  1. Nền = Gaussian nền + mặt đỡ từ S3 (gaussians/background_with_support.ply).
  2. plane_fill: chiếu Gaussian các vật xuống mặt bàn (support.json) -> lưới ô (cạnh grid mét), nới dilate_cells ô;
     mỗi ô thêm 1 Gaussian dẹt nằm trên mặt bàn, màu (f_dc) lấy từ Gaussian mặt bàn gần nhất ở vành ngoài vết vật.
  3. Tinh chỉnh toàn bộ Gaussian nền theo ảnh đã inpaint (S4a, ProPainter): loss (1-l)*L1 + l*(1-SSIM), không densify,
     lr = lr mặc định của CoMe x lr_scale, lr vị trí = giá trị cuối lịch (iter 30000).
Checkpoint: mỗi ckpt_every iter lưu ckpt/latest.ply + ckpt/latest.json; chạy lại thì nạp latest và tiếp.
Xuất: background_filled.ply (trước tinh chỉnh), background_finetuned.ply, preview/<frame>.jpg [ảnh inpaint | trước | sau],
      fill.json, loss.csv
"""
import json
import random
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree

from common import (log, make_parser, load_scene, view_name, read_ply, write_ply, gaussians_from_ply, render_rgbd,
                    to_u8, load_cfg)
from arguments import OptimizationParams, MeshingParams, get_combined_args
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim


def quat_from_R(R):
    """(w, x, y, z) từ ma trận quay 3x3 (cột = trục cục bộ trong toạ độ thế giới), đúng quy ước build_rotation."""
    m = R
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    i = int(np.argmax(np.diag(m)))
    if i == 0:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        return np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    if i == 1:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        return np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
    return np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])


def plane_basis(n):
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n, a); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def plane_fill(s3, cfg_fill, mpu):
    el = read_ply(s3 / "gaussians" / "background_with_support.ply")
    ob = read_ply(s3 / "gaussians" / "objects_all.ply")
    sup = json.loads((s3 / "support.json").read_text())
    n = np.array(sup["table"]["normal"]); d = sup["table"]["d"]
    u, v = plane_basis(n)
    p0 = -d * n
    cell = cfg_fill["grid"] / mpu                                   # mét -> đơn vị cảnh
    oxyz = np.stack([ob["x"], ob["y"], ob["z"]], 1)
    sd = oxyz @ n + d
    oxyz = oxyz[sd > -cell]                                         # bỏ phần (nếu có) nằm dưới mặt bàn
    a, b = (oxyz - p0) @ u, (oxyz - p0) @ v
    ia, ib = np.floor(a / cell).astype(int), np.floor(b / cell).astype(int)
    a0, b0 = ia.min() - 4 * cfg_fill["dilate_cells"], ib.min() - 4 * cfg_fill["dilate_cells"]
    H, W = ib.max() - b0 + 4 * cfg_fill["dilate_cells"] + 1, ia.max() - a0 + 4 * cfg_fill["dilate_cells"] + 1
    occ = np.zeros((H, W), np.uint8); occ[ib - b0, ia - a0] = 1
    k = np.ones((3, 3), np.uint8)
    foot = cv2.dilate(occ, k, iterations=cfg_fill["dilate_cells"]).astype(bool)
    ring = cv2.dilate(foot.astype(np.uint8), k, iterations=3).astype(bool) & ~foot
    # Gaussian nằm sát mặt bàn ở vành ngoài vết vật -> nguồn màu
    exyz = np.stack([el["x"], el["y"], el["z"]], 1)
    near_plane = np.abs(exyz @ n + d) < 2 * cell
    ea, eb = (exyz - p0) @ u, (exyz - p0) @ v
    ja, jb = np.floor(ea / cell).astype(int) - a0, np.floor(eb / cell).astype(int) - b0
    inside = (ja >= 0) & (ja < W) & (jb >= 0) & (jb < H)
    src = np.where(near_plane & inside)[0]
    src = src[ring[jb[src], ja[src]]]
    if len(src) < 10:
        src = np.where(near_plane)[0]
    tree = cKDTree(np.stack([ea[src], eb[src]], 1))
    yy, xx = np.nonzero(foot)
    ca, cb = (xx + a0 + 0.5) * cell, (yy + b0 + 0.5) * cell
    _, nn = tree.query(np.stack([ca, cb], 1), k=1)
    pts = p0 + ca[:, None] * u + cb[:, None] * v
    new = np.zeros(len(pts), dtype=el.dtype)
    new["x"], new["y"], new["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    for c in ("f_dc_0", "f_dc_1", "f_dc_2"):
        new[c] = el[c][src[nn]]
    op = cfg_fill["opacity"]
    new["opacity"] = np.log(op / (1 - op))
    new["scale_0"] = new["scale_1"] = np.log(cell * 0.75)
    new["scale_2"] = np.log(cell * 0.05)
    q = quat_from_R(np.stack([u, v, n], 1))
    for i in range(4):
        new[f"rot_{i}"] = q[i]
    filled = np.concatenate([el, new])
    info = {"cell_units": float(cell), "footprint_cells": int(foot.sum()), "added_gaussians": int(len(new)),
            "color_source_gaussians": int(len(src))}
    return filled, info


def main():
    p, model, pipe, ss, _, _ = make_parser("S4 background")
    p.add_argument("--s3", type=str, required=True)
    p.add_argument("--inpainted", type=str, required=True, help="thư mục frame_XXXXX.png đã inpaint (S4a)")
    args = get_combined_args(p)
    cfg_all = load_cfg(args.cfg)
    cfg = cfg_all["S4_background"]
    s3, out = Path(args.s3), Path(args.out)
    (out / "preview").mkdir(parents=True, exist_ok=True); (out / "ckpt").mkdir(exist_ok=True)
    mpu = json.loads((s3 / "support.json").read_text())["meters_per_unit"]

    filled_p = out / "background_filled.ply"
    if not filled_p.exists():
        if cfg["plane_fill"]["enabled"]:
            filled, info = plane_fill(s3, cfg["plane_fill"], mpu)
        else:
            filled, info = read_ply(s3 / "gaussians" / "background_with_support.ply"), {"added_gaussians": 0}
        write_ply(filled_p, filled)
        (out / "fill.json").write_text(json.dumps(info, indent=2))
        log(f"plane fill: {info}")

    start, ply = 0, filled_p
    if (out / "ckpt" / "latest.json").exists():
        st = json.loads((out / "ckpt" / "latest.json").read_text())
        start, ply = st["iteration"], out / "ckpt" / "latest.ply"
        log(f"♻️ tiếp tục tinh chỉnh từ iter {start}")
    S = load_scene(args, model, pipe, ss, ply_path=ply)
    g = S["g"]

    # ảnh đích (đã inpaint) cho các view đã đăng ký
    inp = Path(args.inpainted)
    targets = {}
    for v in S["views"]:
        f = inp / f"{view_name(v)}.png"
        if f.exists():
            im = cv2.imread(str(f))[..., ::-1].astype(np.float32) / 255.0
            im = cv2.resize(im, (int(v.image_width), int(v.image_height)), interpolation=cv2.INTER_AREA)
            targets[view_name(v)] = torch.from_numpy(im).permute(2, 0, 1).contiguous().cuda()
    views = [v for v in S["views"] if view_name(v) in targets]
    log(f"{len(views)} view có ảnh đích")

    ft = cfg["finetune"]
    before = {}
    if start == 0:
        for v in views:
            if view_name(v) in cfg_all["S3_label"]["preview_frames"]:
                before[view_name(v)] = render_rgbd(v, g, S)[0]
        torch.save({k: t.cpu() for k, t in before.items()}, out / "preview_before.pt")
    elif (out / "preview_before.pt").exists():
        before = {k: t.cuda() for k, t in torch.load(out / "preview_before.pt").items()}

    pa = ArgumentParser(); op_ = OptimizationParams(pa); mp_ = MeshingParams(pa); a2 = pa.parse_args([])
    opt, mesh = op_.extract(a2), mp_.extract(a2)
    g.spatial_lr_scale = S["scene"].cameras_extent
    g.training_setup(opt, mesh, None)
    for pg in g.optimizer.param_groups:
        pg["lr"] *= ft["lr_scale"]
    xyz_lr = g.xyz_scheduler_args(opt.iterations) * ft["lr_scale"]

    rng = random.Random(start)
    logf = open(out / "loss.csv", "a")
    for it in range(start + 1, ft["iterations"] + 1):
        for pg in g.optimizer.param_groups:
            if pg["name"] == "xyz":
                pg["lr"] = xyz_lr
        v = rng.choice(views)
        gt = targets[view_name(v)]
        img = render(v, g, S["pipe"], S["bg"], splat_args=S["splat_args"], gt_color=gt)["render"][:3]
        loss = (1 - ft["lambda_dssim"]) * l1_loss(img, gt) + ft["lambda_dssim"] * (1 - ssim(img, gt))
        loss.backward()
        g.optimizer.step(); g.optimizer.zero_grad(set_to_none=True)
        if it % 50 == 0:
            logf.write(f"{it},{loss.item():.6f}\n"); logf.flush()
        if it % ft["ckpt_every"] == 0 or it == ft["iterations"]:
            g.save_ply(str(out / "ckpt" / "latest_tmp.ply"))
            (out / "ckpt" / "latest_tmp.ply").replace(out / "ckpt" / "latest.ply")
            (out / "ckpt" / "latest.json").write_text(json.dumps({"iteration": it}))
            log(f"tinh chỉnh {it}/{ft['iterations']} loss {loss.item():.4f} (checkpoint)")
    logf.close()

    g.save_ply(str(out / "background_finetuned.ply"))
    with torch.no_grad():
        g.compute_3D_filter(S["train"])
    for v in views:
        nm = view_name(v)
        if nm in before:
            after = render_rgbd(v, g, S)[0]
            sep = torch.ones(3, after.shape[1], 8, device="cuda")
            img = to_u8(torch.cat([targets[nm], sep, before[nm], sep, after], 2))[..., ::-1]
            cv2.imwrite(str(out / "preview" / f"{nm}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    log(f"S4b xong -> {out}")


if __name__ == "__main__":
    main()
