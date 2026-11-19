"""
S4e: Support-Aware Non-Planar Surface Fill (3DGIC-inspired).

Khac voi s4d_cutfill (lap 1 mat phang ban ID 13), S4e lap be mat cong cuc bo cua BAT KY vat do nao (vd dia do banh_quy), cung
mot quy tac cho moi vat:
  1. Khung cuc bo (p0, u, v, n): mat phang PCA cua vat do (hoac mat phang ban trong support.json khi --support_name table).
     Vet chan = o luoi (--grid_mm) chua tam Gaussian cua cac vat bi nhac (--remove_names), chieu len (u, v).
  2. Be mat tham chieu: Gaussian vat do o VANH ngoai vet chan (ngoai vung cat, trong --ring_cells o), moi o giu diem cao nhat
     (mat tren, phia vat bi nhac) -> h(u, v) noi suy IDW kNN (khong ep phang).
  3. Cat (--cut_dilate o): Gaussian vat do co tam trong vet chan nong --cut_dilate o va cao hon be mat > -1 cm (phan con sot / bong
     cua vat bi nhac ngay tren be mat). Gaussian thap hon (mat duoi dia) giu nguyen.
  4. Lap (--fill_dilate o): Gaussian det tai h(u, v), xoay theo phap tuyen cuc bo cua h (gradient sai phan), scale trong mat
     = --tex_scale x o. Mau (3DGIC): chieu vao 154 view train, nhan pixel neu nhan SAM 3 thuoc vat do (co 2 px) VA do sau render
     cua vat do (truoc cat) khop |z - D| <= --depth_tol_cm; >= --min_views view -> trung vi. Vung lien thong khong thay: chep
     texture tu phep tinh tien ca vung tren mat cong (16 huong x 1/1.5/2/3 duong kinh, ty le thay >= --copy_frac).
  5. (--ft_iters > 0) Tinh chinh masked tren view train: chi tinh loss o pixel nhan vat do (co 2 px) NGOAI mask vat bi nhac nong
     15 px (anh that o do dung la vat do); --freeze_fill 1 dong bang Gaussian lap.
  6. Do tren 23 view test cho: truoc (vat do goc, vat bi nhac da bo), cat (chua lap), cf (cat + lap), ft:
       - hole_pct: % pixel alpha < 0.5 trong vung VET CHAN = (mask vat bi nhac, co --band px) giao (alpha > 0.5 cua rieng Gaussian lap
         render opacity 1, tuc la hinh chieu cua mat vet chan);
       - psnr_support_visible: PSNR o pixel nhan vat do ngoai mask vat bi nhac nong 15 px (phan phai giu nguyen);
       - psnr_footprint_visible: PSNR trong vet chan o pixel anh that thay vat do.
Ham dung chung (surface_fill_geometry, sample_depth_guided_colors, color_fill, fill_records) duoc s4d_cutfill.py --surface_fill 1
goi lai cho mat ban (cung cat / bao ve / tinh chinh / do nhu e8tex).
Checkpoint: <support>_filled.ply sau lap; ckpt/latest.ply moi 250 buoc tinh chinh.
"""
import json
import random
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, label as cc_label
from scipy.spatial import cKDTree

from common import (log, make_parser, load_scene, view_name, read_ply, write_ply,
                    gaussians_from_ply, render_rgbd, to_u8, freeze)
from arguments import get_combined_args, OptimizationParams, MeshingParams
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from s4_background import plane_basis, quat_from_R

C0 = 0.28209479177387814
SAM3_IDS = {
    "table": [13], "gau": [20], "cuu": [30], "tui_banh": [50], "dia": [60],
    "ly_tra": [70], "coc_su": [80], "tao": [90], "banh_quy": [100], "thia": [110],
    "khan_giay": [130], "coc_kem_sua": [160, 161], "vung_tra": [170, 171], "goi_keo": [180, 181],
}


def psnr_masked(a, b, m):
    mse = (((a - b) ** 2).mean(0))[m].mean().clamp(min=1e-10)
    return float(-10 * torch.log10(mse))


def _idw(uv_query, uv_ref, h_ref, k=16, power=2.0, tree=None):
    tree = tree or cKDTree(uv_ref)
    k_eff = min(k, len(uv_ref))
    dists, idxs = tree.query(uv_query, k=k_eff)
    if k_eff == 1:
        return h_ref[idxs]
    w = 1.0 / np.maximum(dists, 1e-9) ** power
    w /= w.sum(1, keepdims=True)
    return (w * h_ref[idxs]).sum(1)


class LocalSurface:
    """h(u, v) = da thuc bac 2 khop binh phuong toi thieu (2 vong, bo diem lech > 2.5 sigma) + IDW cua phan du.
    Da thuc ngoai suy duoc do cong vao giua vet chan (IDW thuan chi lay trung binh vanh), IDW giu chi tiet cuc bo."""

    def __init__(self, uv, h):
        self.uv, self.h = uv, h
        keep = np.ones(len(h), bool)
        for _ in range(2):
            A = self._A(uv[keep])
            self.c, *_ = np.linalg.lstsq(A, h[keep], rcond=None)
            r = h - self._A(uv) @ self.c
            sig = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep]))) + 1e-12
            keep = np.abs(r) <= 2.5 * sig
        self.res = h - self._A(uv) @ self.c
        self.res[~keep] = 0.0
        self.tree = cKDTree(uv)
        self.rms = float(np.sqrt(np.mean(self.res[keep] ** 2)))

    @staticmethod
    def _A(uv):
        a, b = uv[:, 0], uv[:, 1]
        return np.stack([np.ones_like(a), a, b, a * a, a * b, b * b], 1)

    def __call__(self, uv):
        return self._A(uv) @ self.c + _idw(uv, self.uv, self.res, tree=self.tree)


def surface_fill_geometry(sup_xyz, obj_xyz, cell, cut_dilate=2, fill_dilate=3, ring_cells=6, plane=None, band_cm_units=None,
                          cut_below=None, layer=0.0):
    """Hinh hoc lap tren be mat cong cuc bo. plane = (n, d) cho mat ban; None = PCA cua sup_xyz.
    band_cm_units: (1 cm, 2 cm) theo don vi canh, dung loc diem vanh. cut_below: Gaussian trong vung cat thap hon be mat
    qua nguong nay (don vi canh) duoc giu (mat duoi); None = 1 cm. Tra dict."""
    one_cm, two_cm = band_cm_units
    cut_below = one_cm if cut_below is None else cut_below
    if plane is None:
        p0 = sup_xyz.mean(0)
        _, _, vt = np.linalg.svd(sup_xyz - p0, full_matrices=False)
        n = vt[-1]
    else:
        n = np.asarray(plane[0], np.float64); n = n / np.linalg.norm(n)
        p0 = -float(plane[1]) * n
    for attempt in range(2):
        u, v = plane_basis(n)
        r = _surface_geo_frame(sup_xyz, obj_xyz, cell, p0, u, v, n, cut_dilate, fill_dilate, ring_cells, plane is not None,
                               one_cm, two_cm, cut_below, layer)
        if r is not None:
            return r
        n = -n                         # phap tuyen phai huong ve phia vat bi nhac (so voi vanh ngay canh vet chan)
    raise RuntimeError("khong xac dinh duoc chieu phap tuyen")


def _surface_geo_frame(sup_xyz, obj_xyz, cell, p0, u, v, n, cut_dilate, fill_dilate, ring_cells, has_plane, one_cm, two_cm,
                       cut_below, layer):
    to_uvh = lambda X: np.stack([(X - p0) @ u, (X - p0) @ v, (X - p0) @ n], 1)
    S_, O_ = to_uvh(sup_xyz), to_uvh(obj_xyz)
    pad = max(cut_dilate, fill_dilate) + ring_cells + 2
    ia, ib = np.floor(O_[:, 0] / cell).astype(int), np.floor(O_[:, 1] / cell).astype(int)
    a0, b0 = ia.min() - pad, ib.min() - pad
    Hh, Ww = ib.max() - b0 + pad + 1, ia.max() - a0 + pad + 1
    occ = np.zeros((Hh, Ww), bool); occ[ib - b0, ia - a0] = True
    k3 = np.ones((3, 3), bool)
    cut_g = binary_dilation(occ, k3, iterations=cut_dilate) if cut_dilate > 0 else occ
    fill_g = binary_dilation(occ, k3, iterations=fill_dilate) if fill_dilate > 0 else occ
    ring_g = binary_dilation(fill_g | cut_g, k3, iterations=ring_cells) & ~(fill_g | cut_g)
    ja, jb = np.floor(S_[:, 0] / cell).astype(int) - a0, np.floor(S_[:, 1] / cell).astype(int) - b0
    ins = (ja >= 0) & (ja < Ww) & (jb >= 0) & (jb < Hh)
    in_ring = np.zeros(len(S_), bool); in_ring[ins] = ring_g[jb[ins], ja[ins]]
    in_cut = np.zeros(len(S_), bool); in_cut[ins] = cut_g[jb[ins], ja[ins]]
    cand = np.where(in_ring)[0]
    if has_plane:
        cand = cand[np.abs(S_[cand, 2]) < two_cm]
    else:
        med = np.median(S_[cand, 2]) if len(cand) else 0.0
        cand = cand[np.abs(S_[cand, 2] - med) < two_cm]
    if len(cand) < 16:
        raise RuntimeError(f"vanh vet chan chi co {len(cand)} Gaussian vat do -> khong khop duoc be mat")
    if not has_plane and np.median(O_[:, 2]) < np.median(S_[cand, 2]):
        return None
    # moi o vanh: lop mat tren (diem cao hon max - layer) -> trung vi (layer <= 0: chi lay diem cao nhat)
    key = jb[cand] * Ww + ja[cand]
    order = np.lexsort((-S_[cand, 2], key))
    ks, hs = key[order], S_[cand[order], 2]
    first = np.r_[True, ks[1:] != ks[:-1]]
    ref = cand[order][first]
    grp = np.cumsum(first) - 1
    hmax = hs[first][grp]
    if layer > 0:
        top = hs >= hmax - layer
        ref_uv = np.array([S_[ref, 0], S_[ref, 1]]).T
        h_ref = np.array([np.median(x) for x in np.split(hs[top], np.flatnonzero(np.diff(grp[top])) + 1)])
    else:
        ref_uv, h_ref = S_[ref, :2], S_[ref, 2]
    surf = LocalSurface(ref_uv, h_ref)
    cut = np.zeros(len(S_), bool)
    ic = np.where(in_cut)[0]
    if len(ic):
        cut[ic] = S_[ic, 2] - surf(S_[ic, :2]) > -cut_below
    yy, xx = np.nonzero(fill_g)
    fuv = np.stack([(xx + a0 + 0.5) * cell, (yy + b0 + 0.5) * cell], 1)
    fh = surf(fuv)
    e = 0.5 * cell
    hu = (surf(fuv + [e, 0]) - surf(fuv - [e, 0])) / (2 * e)
    hv = (surf(fuv + [0, e]) - surf(fuv - [0, e])) / (2 * e)
    nrm = n[None] - hu[:, None] * u[None] - hv[:, None] * v[None]
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    pts = p0 + fuv[:, :1] * u + fuv[:, 1:] * v + fh[:, None] * n
    tilt = np.degrees(np.arccos(np.clip(nrm @ n, -1, 1)))
    return {"p0": p0, "u": u, "v": v, "n": n, "fill_pts": pts, "fill_uv": fuv, "fill_normals": nrm, "cut": cut,
            "surf": surf, "cell": cell,
            "info": {"footprint_cells": int(occ.sum()), "fill_cells": int(fill_g.sum()), "ring_ref_points": int(len(ref)),
                     "surface_fit_rms_cm": surf.rms / one_cm,
                     "cut_candidates_in_footprint": int(in_cut.sum()), "cut": int(cut.sum()),
                     "fill_h_vs_frame_plane_cm_p5_p50_p95": [float(x) for x in np.percentile(fh / one_cm, [5, 50, 95])],
                     "surface_h_range_cm": float(np.ptp(fh) / one_cm), "normal_tilt_deg_p50_p95":
                     [float(np.percentile(tilt, 50)), float(np.percentile(tilt, 95))]}}


def surf_points(geo, uv):
    h = geo["surf"](uv)
    return geo["p0"] + uv[:, :1] * geo["u"] + uv[:, 1:] * geo["v"] + h[:, None] * geo["n"]


@torch.no_grad()
def sample_depth_guided_colors(pts, views, g_depth, S, lab_of, support_ids, erode_px=2, depth_tol=0.05, cache=None):
    """3DGIC-style: diem 3D duoc view thay neu chieu vao trong anh (z > 0), nhan SAM 3 tai pixel thuoc support_ids (co erode_px)
    va |z_cam - D_render(g_depth)| <= depth_tol. Tra (trung vi mau [N,3], so view thay [N])."""
    P = torch.as_tensor(pts, dtype=torch.float32, device="cuda")
    N = len(P)
    Ph = torch.cat([P, torch.ones(N, 1, device="cuda")], 1)
    cols, cnt = [], torch.zeros(N, device="cuda")
    sid = torch.tensor(support_ids, device="cuda")
    for vw in views:
        lab = lab_of(vw)
        if lab is None:
            continue
        vn = view_name(vw)
        if cache is not None and vn in cache:
            sm, depth, img = cache[vn]
        else:
            H, W = lab.shape
            sm = torch.isin(lab, sid).float()
            if erode_px > 0:
                sm = 1 - F.max_pool2d((1 - sm)[None, None], 2 * erode_px + 1, 1, erode_px)[0, 0]
            sm = sm > 0.5
            depth = render_rgbd(vw, g_depth, S)[1]
            if depth.shape != (H, W):
                depth = F.interpolate(depth[None, None], (H, W), mode="nearest")[0, 0]
            img = vw.original_image.cuda()[:3]
            if img.shape[1:] != (H, W):
                img = F.interpolate(img[None], (H, W), mode="bilinear", align_corners=False)[0]
            depth, img = depth.half(), img.half()
            if cache is not None:
                cache[vn] = (sm, depth, img)
        H, W = sm.shape
        z = (Ph @ vw.world_view_transform.cuda())[:, 2]
        hom = Ph @ vw.full_proj_transform.cuda()
        w = hom[:, 3]
        xi = (((hom[:, 0] / w + 1) * W - 1) / 2).round().long()
        yi = (((hom[:, 1] / w + 1) * H - 1) / 2).round().long()
        ok = (w > 0) & (z > 0) & (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        okk = ok.clone()
        if ok.any():
            d = depth[yi[ok], xi[ok]].float()
            okk[ok] = sm[yi[ok], xi[ok]] & (d > 0) & ((z[ok] - d).abs() <= depth_tol)
        c = torch.full((N, 3), float("nan"), device="cuda")
        c[okk] = img[:, yi[okk], xi[okk]].T.float()
        cols.append(c)
        cnt += okk.float()
    if not cols:
        return np.zeros((N, 3)), np.zeros(N, int)
    med = torch.nanmedian(torch.stack(cols, 0), dim=0).values
    return med.cpu().numpy(), cnt.long().cpu().numpy()


def color_fill(geo, views, g_depth, S, lab_of, support_ids, depth_tol, min_views=2, copy_frac=0.9, log_fn=log):
    """Mau cho Gaussian lap: thay truc tiep -> trung vi; vung khong thay -> chep tinh tien tren mat cong; con lai -> gan nhat."""
    cache = {}
    pts, fuv, cell = geo["fill_pts"], geo["fill_uv"], geo["cell"]
    col, cnt = sample_depth_guided_colors(pts, views, g_depth, S, lab_of, support_ids, depth_tol=depth_tol, cache=cache)
    seen = cnt >= min_views
    rgb = np.full((len(pts), 3), np.nan)
    rgb[seen] = col[seen]
    ia = np.round((fuv[:, 0] - fuv[:, 0].min()) / cell).astype(int)
    ib = np.round((fuv[:, 1] - fuv[:, 1].min()) / cell).astype(int)
    grid = np.zeros((ib.max() + 1, ia.max() + 1), bool)
    grid[ib[~seen], ia[~seen]] = True
    lab_cc, ncc = cc_label(grid, structure=np.ones((3, 3)))
    comp = lab_cc[ib, ia]
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    copied, comps_ok = 0, 0
    for k in range(1, ncc + 1):
        idx = np.where((comp == k) & ~seen)[0]
        if len(idx) == 0:
            continue
        ext = max(np.ptp(fuv[idx, 0]), np.ptp(fuv[idx, 1])) + 2 * cell
        for dist in ext * np.array([1.0, 1.5, 2.0, 3.0]):
            shifts = np.stack([np.cos(angles), np.sin(angles)], 1) * dist
            cand = surf_points(geo, np.concatenate([fuv[idx] + s for s in shifts], 0))
            c2, n2 = sample_depth_guided_colors(cand, views, g_depth, S, lab_of, support_ids, depth_tol=depth_tol, cache=cache)
            c2 = c2.reshape(len(shifts), len(idx), 3); n2 = n2.reshape(len(shifts), len(idx))
            frac = (n2 >= min_views).mean(1)
            j = int(np.argmax(frac))
            if frac[j] >= copy_frac:
                ok = n2[j] >= min_views
                rgb[idx[ok]] = c2[j][ok]
                copied += int(ok.sum()); comps_ok += 1
                break
    have = ~np.isnan(rgb[:, 0])
    n_nn = int((~have).sum())
    if have.any() and n_nn:
        _, nn = cKDTree(fuv[have]).query(fuv[~have], k=1)
        rgb[~have] = rgb[have][nn]
    elif not have.any():
        rgb[:] = 0.5
    del cache
    torch.cuda.empty_cache()
    info = {"fill_points": int(len(pts)), "seen_direct": int(seen.sum()), "views_median": float(np.median(cnt)),
            "unseen_components": int(ncc), "components_copied": comps_ok, "copied_points": copied, "nearest_points": n_nn}
    log_fn(f"color_fill: {info}")
    return np.clip(rgb, 0, 1), info


def fill_records(dtype, geo, rgb, scale_frac=0.5, thick_frac=0.05, opacity=0.9):
    """Gaussian det: scale trong mat = scale_frac x o, day = thick_frac x o, truc nho nhat theo phap tuyen cuc bo."""
    pts, nrm, cell, u = geo["fill_pts"], geo["fill_normals"], geo["cell"], geo["u"]
    rec = np.zeros(len(pts), dtype=dtype)
    rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    for i, c in enumerate(("f_dc_0", "f_dc_1", "f_dc_2")):
        rec[c] = (rgb[:, i] - 0.5) / C0
    rec["opacity"] = np.log(opacity / (1.0 - opacity))
    rec["scale_0"] = rec["scale_1"] = np.log(cell * scale_frac)
    rec["scale_2"] = np.log(cell * thick_frac)
    t1 = u[None] - (nrm @ u)[:, None] * nrm
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(nrm, t1)
    q = np.stack([quat_from_R(np.stack([a, b, c], 1)) for a, b, c in zip(t1, t2, nrm)])
    for i in range(4):
        rec[f"rot_{i}"] = q[:, i]
    return rec                     # truong khac (f_rest_*, ...) = 0, nhu plane_fill cua S4b


def main():
    p, model, pipe, ss, _, _ = make_parser("S4e: Support-Aware Non-Planar Surface Fill (3DGIC-style)")
    p.add_argument("--src", required=True, help="Thu muc S4c (vd S4c/ctrl)")
    p.add_argument("--masks", required=True, help="Thu muc sam3_masks/labels")
    p.add_argument("--support_name", default="dia")
    p.add_argument("--remove_names", default="banh_quy", help="vat bi nhac, cach nhau dau phay")
    p.add_argument("--grid_mm", type=float, default=4.0)
    p.add_argument("--cut_dilate", type=int, default=2)
    p.add_argument("--fill_dilate", type=int, default=3)
    p.add_argument("--ring_cells", type=int, default=6)
    p.add_argument("--ring_layer_cm", type=float, default=0.0, help="> 0: be mat vanh = trung vi lop tren day nay (cm); 0 = diem cao nhat")
    p.add_argument("--cut_below_cm", type=float, default=0.5, help="trong vung cat, giu Gaussian thap hon be mat qua nguong nay")
    p.add_argument("--depth_tol_cm", type=float, default=2.5)
    p.add_argument("--min_views", type=int, default=2)
    p.add_argument("--copy_frac", type=float, default=0.9)
    p.add_argument("--tex_scale", type=float, default=0.5)
    p.add_argument("--ft_iters", type=int, default=0)
    p.add_argument("--lr_scale", type=float, default=0.5)
    p.add_argument("--lam_dssim", type=float, default=0.2)
    p.add_argument("--freeze_fill", type=int, default=0)
    p.add_argument("--band", type=int, default=3)
    args = get_combined_args(p)

    src, out = Path(args.src), Path(args.out)
    (out / "preview").mkdir(parents=True, exist_ok=True)
    (out / "ckpt").mkdir(exist_ok=True)
    S = load_scene(args, model, pipe, ss)
    sh = S["dataset"].sh_degree
    sup = json.loads((src / "S3like" / "support.json").read_text())
    mpu = float(sup["meters_per_unit"])
    cell = (args.grid_mm / 1000.0) / mpu
    depth_tol = (args.depth_tol_cm / 100.0) / mpu
    one_cm = 0.01 / mpu

    mask_dir = Path(args.masks)
    lab_cache = {}

    def lab_of(vw):
        vn = view_name(vw)
        if vn not in lab_cache:
            fp = mask_dir / f"{vn}.png"
            im = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED) if fp.exists() else None
            lab_cache[vn] = torch.from_numpy(im.astype(np.int64)).cuda() if im is not None else None
        return lab_cache[vn]

    # nguon Gaussian: S4c/<tag>/S3like/gaussians/<vat>.ply, neu khong co thi tach tu CoMe theo S3 labels.npy
    cfgp = Path(args.cfg)
    groups = {}
    if (cfgp.parent / "groups.json").exists():
        raw = json.loads((cfgp.parent / "groups.json").read_text())
        groups = {k: int(v["label"]) for k, v in raw.get("objects", raw).items() if isinstance(v, dict) and "label" in v}
    come_ply = Path(args.model_path) / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    _come = {}

    def object_ply(name):
        cand = src / "S3like" / "gaussians" / f"{name}.ply"
        if cand.exists():
            return cand
        tmp = out / f"_src_{name}.ply"
        if not tmp.exists():
            if name not in groups or not (cfgp.parent / "labels.npy").exists():
                raise FileNotFoundError(f"khong tim thay Gaussian cua {name}")
            if "arr" not in _come:
                _come["arr"], _come["lab"] = read_ply(come_ply), np.load(cfgp.parent / "labels.npy")
            write_ply(tmp, _come["arr"][_come["lab"] == groups[name]])
        return tmp

    xyz = lambda a: np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)
    remove = [x.strip() for x in args.remove_names.split(",") if x.strip()]
    rem_ids = sum((SAM3_IDS[r] for r in remove), [])
    if args.support_name == "table":
        sup_ply, support_ids = src / "S4like" / "background_finetuned.ply", [13]
        plane = (sup["table"]["normal"], sup["table"]["d"])
    else:
        sup_ply, support_ids, plane = object_ply(args.support_name), SAM3_IDS[args.support_name], None
    sup_arr = read_ply(sup_ply)
    obj_xyz = np.concatenate([xyz(read_ply(object_ply(r))) for r in remove], 0)
    geo = surface_fill_geometry(xyz(sup_arr), obj_xyz, cell, args.cut_dilate, args.fill_dilate, args.ring_cells,
                                plane=plane, band_cm_units=(one_cm, 2 * one_cm), cut_below=args.cut_below_cm * one_cm,
                                layer=args.ring_layer_cm * one_cm)
    log(f"hinh hoc: {geo['info']}")

    filled_p = out / f"{args.support_name}_filled.ply"
    if filled_p.exists() and (out / "fill.json").exists():
        info = json.loads((out / "fill.json").read_text())
    else:
        g_depth = gaussians_from_ply(sup_ply, sh, S["train"])
        rgb, cinfo = color_fill(geo, S["train"], g_depth, S, lab_of, support_ids, depth_tol, args.min_views, args.copy_frac)
        del g_depth
        torch.cuda.empty_cache()
        rec = fill_records(sup_arr.dtype, geo, rgb, args.tex_scale)
        kept = sup_arr[~geo["cut"]]
        write_ply(out / "_cut_only.ply", kept)
        write_ply(filled_p, np.concatenate([kept, rec]))
        write_ply(out / "_fill_only.ply", rec)
        np.save(out / "fill_mask.npy", np.r_[np.zeros(len(kept), bool), np.ones(len(rec), bool)])
        info = {**geo["info"], **cinfo, "kept": int(len(kept))}
        (out / "fill.json").write_text(json.dumps(info, indent=2))
    log(f"lap: {info}")

    final = filled_p
    if args.ft_iters > 0:
        start, ply = 0, filled_p
        if (out / "ckpt" / "latest.json").exists():
            start = json.loads((out / "ckpt" / "latest.json").read_text())["iteration"]
            ply = out / "ckpt" / "latest.ply"
            log(f"tiep tuc tinh chinh tu iter {start}")
        if start < args.ft_iters:
            from s4c_decouple import unfreeze
            S2 = load_scene(args, model, pipe, ss, ply_path=ply)
            g = S2["g"]
            pa = ArgumentParser()
            op_, mp_ = OptimizationParams(pa), MeshingParams(pa)
            a2 = pa.parse_args([])
            opt, mesh = op_.extract(a2), mp_.extract(a2)
            g.spatial_lr_scale = S2["scene"].cameras_extent
            unfreeze(g)
            g.training_setup(opt, mesh, None)
            for pg in g.optimizer.param_groups:
                pg["lr"] *= args.lr_scale
            xyz_lr = g.xyz_scheduler_args(opt.iterations) * args.lr_scale
            frz = torch.from_numpy(np.load(out / "fill_mask.npy")).cuda() if args.freeze_fill else None
            sid, rid = torch.tensor(support_ids, device="cuda"), torch.tensor(rem_ids, device="cuda")
            keeps = {}
            for vw in S["train"]:
                lab = lab_of(vw)
                if lab is None:
                    continue
                Ms = torch.isin(lab, sid).float()
                Ms = 1 - F.max_pool2d((1 - Ms)[None, None], 5, 1, 2)[0, 0]
                Mr = F.max_pool2d(torch.isin(lab, rid).float()[None, None], 31, 1, 15)[0, 0]
                k_ = (Ms > 0.5) & (Mr < 0.5)
                if k_.sum() > 200:
                    keeps[view_name(vw)] = k_[None]
            views = [vw for vw in S["train"] if view_name(vw) in keeps]
            log(f"tinh chinh masked tren {len(views)} view (pixel vat do ngoai vat bi nhac)")
            rng = random.Random(start)
            logf = open(out / "loss.csv", "a")
            for it in range(start + 1, args.ft_iters + 1):
                for pg in g.optimizer.param_groups:
                    if pg["name"] == "xyz":
                        pg["lr"] = xyz_lr
                vw = rng.choice(views)
                gt = vw.original_image.cuda()[:3]
                img = render(vw, g, S2["pipe"], S2["bg"], splat_args=S2["splat_args"], gt_color=gt)["render"][:3]
                keep = keeps[view_name(vw)]
                if keep.shape[1:] != img.shape[1:]:
                    keep = F.interpolate(keep[None].float(), img.shape[1:], mode="nearest")[0] > 0.5
                tgt = torch.where(keep, gt, img.detach())
                loss = (1 - args.lam_dssim) * l1_loss(img, tgt) + args.lam_dssim * (1 - ssim(img, tgt))
                loss.backward()
                if frz is not None:
                    for pg in g.optimizer.param_groups:
                        for q in pg["params"]:
                            if q.grad is not None and q.shape[0] == frz.shape[0]:
                                q.grad[frz] = 0
                g.optimizer.step()
                g.optimizer.zero_grad(set_to_none=True)
                if it % 50 == 0:
                    logf.write(f"{it},{loss.item():.6f}\n"); logf.flush()
                if it % 250 == 0 or it == args.ft_iters:
                    g.save_ply(str(out / "ckpt" / "latest_tmp.ply"))
                    (out / "ckpt" / "latest_tmp.ply").replace(out / "ckpt" / "latest.ply")
                    (out / "ckpt" / "latest.json").write_text(json.dumps({"iteration": it}))
                    log(f"tinh chinh {it}/{args.ft_iters} loss {loss.item():.4f} (checkpoint)")
            logf.close()
            freeze(g)
            del g, S2
            torch.cuda.empty_cache()
        final = out / "ckpt" / "latest.ply"

    # ------------------------------------------------------------------ do tren 23 view test
    black = torch.zeros(3, device="cuda")
    sid, rid = torch.tensor(support_ids, device="cuda"), torch.tensor(rem_ids, device="cuda")
    g_fp = gaussians_from_ply(out / "_fill_only.ply", sh, S["train"])
    with torch.no_grad():
        g_fp._opacity[:] = 10.0
    scenes = {"before": sup_ply, "cut": out / "_cut_only.ply", "cf": filled_p}
    if final != filled_p:
        scenes["ft"] = final
    G = {k: gaussians_from_ply(v, sh, S["train"]) for k, v in scenes.items()}
    acc = {k: {"hole": [], "psnr_sup": [], "psnr_fp": []} for k in G}
    region_px, sheets = [], []
    with torch.no_grad():
        for vw in S["scene"].getTestCameras():
            lab = lab_of(vw)
            if lab is None:
                continue
            gt = vw.original_image.cuda()[:3]
            Mr = torch.isin(lab, rid).float()
            ero = (1 - F.max_pool2d((1 - Mr)[None, None], 2 * args.band + 1, 1, args.band)[0, 0]) > 0.5
            dil = F.max_pool2d(Mr[None, None], 31, 1, 15)[0, 0] > 0.5
            fp = render(vw, g_fp, S["pipe"], black, splat_args=S["splat_args"],
                        override_color=torch.ones(g_fp.get_xyz.shape[0], 3, device="cuda"))["render"][0] > 0.5
            R = fp & ero
            vis_sup = torch.isin(lab, sid) & ~dil
            fp_sup = fp & torch.isin(lab, sid)
            row = [gt]
            for k, gk in G.items():
                rgb = render_rgbd(vw, gk, S)[0]
                row.append(rgb)
                if vis_sup.sum() > 50:
                    acc[k]["psnr_sup"].append(psnr_masked(rgb, gt, vis_sup))
                if fp_sup.sum() > 50:
                    acc[k]["psnr_fp"].append(psnr_masked(rgb, gt, fp_sup))
                if R.sum() >= 50:
                    al = render(vw, gk, S["pipe"], black, splat_args=S["splat_args"],
                                override_color=torch.ones(gk.get_xyz.shape[0], 3, device="cuda"))["render"][0]
                    acc[k]["hole"].append(float((al[R] < 0.5).float().mean()))
            if R.sum() >= 50:
                region_px.append(int(R.sum()))
                if len(sheets) < 4:
                    sep = torch.ones(3, gt.shape[1], 6, device="cuda")
                    cols = sum([[x, sep] for x in row], [])[:-1]
                    sheets.append((view_name(vw), torch.cat(cols, 2)))
    for nm, im in sheets:
        cv2.imwrite(str(out / "preview" / f"{nm}.jpg"), to_u8(im)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
    mean = lambda x: float(np.mean(x)) if x else None
    res = {"support_name": args.support_name, "remove_names": remove, "support_ids": support_ids, "fill": info,
           "args": {k: getattr(args, k) for k in ("grid_mm", "cut_dilate", "fill_dilate", "ring_cells", "ring_layer_cm", "cut_below_cm", "depth_tol_cm", "min_views",
                                                   "copy_frac", "tex_scale", "ft_iters", "lr_scale", "freeze_fill", "band")},
           "footprint_views": len(region_px), "footprint_px_median": float(np.median(region_px)) if region_px else 0,
           "preview_columns": ["gt"] + list(G), "preview_views": [n for n, _ in sheets]}
    for k in G:
        res[k] = {"hole_pct": 100 * mean(acc[k]["hole"]) if acc[k]["hole"] else None,
                  "psnr_support_visible": mean(acc[k]["psnr_sup"]), "psnr_footprint_visible": mean(acc[k]["psnr_fp"]),
                  "views_hole": len(acc[k]["hole"]), "views_fp": len(acc[k]["psnr_fp"])}
    (out / "metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    log(f"[S4e] xong: {json.dumps(res, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
