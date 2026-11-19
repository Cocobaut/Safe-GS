"""
Khung chặt + khuôn inpaint cho từng vật, đo bằng CHÍNH bộ render CoMe (không dùng vật cản giả định).

A. Gaussian được quan sát hay không: trọng số đóng góp render của từng Gaussian cộng qua mọi view train
   (s3_label.accumulate với mask = 1). Gaussian có tổng < --eps là "không view train nào thấy" (theo nguyên tắc: không thấy thì để trống).
   Kiểm chứng: bỏ các Gaussian đó (chỉ trong các vật) rồi render 23 view test (không dùng để tính) so với đối chứng
   bỏ cùng số Gaussian ngẫu nhiên trong nhóm được thấy của cùng vật.
B. Khoét hull theo ảnh độ sâu render (space carving): với mỗi view (train + test), mỗi voxel hull chiếu vào ảnh:
     alpha < 0.5 tại pixel đó hoặc z_voxel < depth - tol  -> camera nhìn XUYÊN qua voxel (bằng chứng trống)
     |z_voxel - depth| <= tol                              -> voxel nằm trên bề mặt thấy được
     z_voxel > depth + tol                                 -> bị che
   Trống nếu >= --min_vis view nhìn xuyên. Khung = hull - trống. Vỏ khung: "thấy" nếu >= min_vis view thấy bề mặt, còn lại = KHUÔN
   (đáy nếu cao <= max(1 cm, 15 % chiều cao), còn lại bên / sau).
Xuất (--out): contrib.npy, metrics.json, frame/<vật>.npz + .ply, preview/*.jpg
C. (--da3_dir) khoét thêm theo độ sâu Depth Anything 3 (pipeline/da3_carve.py).
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from common import log, make_parser, load_scene, view_name, read_ply, write_ply, gaussians_from_ply, render_rgbd, to_u8
from arguments import get_combined_args
from gaussian_renderer import render
from utils.image_utils import psnr
from s3_label import accumulate
from da3_carve import load_da3, da3_thru_votes, gauss_voxels

OBJ_IDS = [80, 70, 60, 100, 90, 50, 130, 110, 170, 171, 180, 181, 160, 161, 20, 30]
SHEET = ["frame_00017", "frame_00065", "frame_00113"]


def read_xyz(path):
    with open(path, "rb") as f:
        n = 0
        while True:
            line = f.readline()
            if line.startswith(b"element vertex"):
                n = int(line.split()[2])
            if line.startswith(b"end_header"):
                break
        dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
        a = np.frombuffer(f.read(n * dt.itemsize), dt)
    return np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)


def shell_of(V, vox):
    ijk = np.round((V - V.min(0)) / vox).astype(np.int64)
    s = 1 << 20
    key = lambda q: ((q[:, 0] + s) << 42) | ((q[:, 1] + s) << 21) | (q[:, 2] + s)
    keys = np.sort(key(ijk))
    inner = np.ones(len(V), bool)
    for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        k = key(ijk + np.array(d))
        i = np.minimum(np.searchsorted(keys, k), len(keys) - 1)
        inner &= keys[i] == k
    return ~inner


def write_rgb_ply(path, xyz, rgb):
    a = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    a["x"], a["y"], a["z"] = xyz.T
    a["red"], a["green"], a["blue"] = rgb.T
    head = ["ply", "format binary_little_endian 1.0", f"element vertex {len(a)}", "property float x", "property float y",
            "property float z", "property uchar red", "property uchar green", "property uchar blue", "end_header", ""]
    with open(path, "wb") as f:
        f.write("\n".join(head).encode())
        f.write(a.tobytes())


def main():
    p, model, pipe, ss, _, _ = make_parser("S3m mold")
    p.add_argument("--s3", required=True)
    p.add_argument("--hulls", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--tol_cm", type=float, default=1.0)
    p.add_argument("--min_vis", type=int, default=3)
    # Viewing-direction gating (CleanGS-style view quality): a view whose optical axis is nearly parallel to the table
    # gives unreliable surface hits (depth is smeared along a long ray), and such a false hit blocks carving of the empty
    # space above a flat object. Grazing views therefore do not count as surface evidence. cos_min 0 = legacy behaviour.
    p.add_argument("--cos_min", type=float, default=0.25, help="|cos(view axis, table normal)| below this: view gives no surface vote")
    # Thin-object protection: a voxel seen as surface by >= protect_vis non-grazing views is never carved. 0 = off (legacy).
    p.add_argument("--protect_vis", type=int, default=3)
    # Matter-gated protection: only protect voxels within 1 voxel of a centre of the object's OWN Gaussians (opacity > --protect_op).
    # Surface votes alone also land on empty voxels in front of a flat object (depth tolerance band), which bloats flat objects.
    p.add_argument("--protect_matter", type=int, default=0, help="1: protection needs object Gaussian matter nearby; 0 = legacy")
    p.add_argument("--protect_op", type=float, default=0.5)
    # Table-plane bottom clip for objects resting on the table (plane from S3 support.json, RANSAC).
    p.add_argument("--table_clip_cm", type=float, default=0.0, help="frame voxels below table plane + this (cm) are dropped for resting objects")
    # Free-space carving from Depth Anything 3 depth (pipeline/da3_carve.py, depth from tools/da3_depth.py). All views vote.
    p.add_argument("--da3_dir", default="", help="aligned DA3 depth dir (align_stats.json + aligned/); empty = off")
    p.add_argument("--da3_min_vis", type=int, default=3)
    p.add_argument("--da3_k_res", type=float, default=2.0, help="per-view tol = max(tol_cm, k * median alignment residual)")
    p.add_argument("--da3_edge", type=float, default=0.03, help="DA3 pixels with local depth range > this * depth are ignored")
    p.add_argument("--da3_op", type=float, default=0.5, help="voxels holding an object Gaussian with opacity > this are never DA3-carved")
    args = get_combined_args(p)
    out = Path(args.out)
    (out / "frame").mkdir(parents=True, exist_ok=True); (out / "preview").mkdir(exist_ok=True)
    s3 = Path(args.s3)
    sup = json.loads((s3 / "support.json").read_text())
    n_t, d_t = np.array(sup["table"]["normal"]), float(sup["table"]["d"])
    cm = 100 * float(sup["meters_per_unit"])
    groups = json.loads((s3 / "groups.json").read_text())["objects"]
    lab = np.load(s3 / "labels.npy")
    S = load_scene(args, model, pipe, ss)
    g = S["g"]
    N = g.get_xyz.shape[0]
    assert len(lab) == N, f"labels.npy {len(lab)} != {N} Gaussian"

    # ---------------------------------------------------------------- A. đóng góp render
    cf = out / "contrib.npy"
    if cf.exists():
        tot = np.load(cf)
    else:
        ones = lambda v: torch.ones(1, int(v.image_height), int(v.image_width), device="cuda")
        W = accumulate(S, g, S["train"], ones, 1, out / "partial_contrib.pt")
        tot = W[:, 0].cpu().numpy()
        np.save(cf, tot)
        (out / "partial_contrib.pt").unlink(missing_ok=True)
    per = {}
    rng = np.random.default_rng(0)
    rm_u, rm_c = [], []
    for name, gi in groups.items():
        idx = np.where(lab == gi["label"])[0]
        u = idx[tot[idx] < args.eps]
        s_ = idx[tot[idx] >= args.eps]
        c = rng.choice(s_, min(len(u), len(s_)), replace=False) if len(s_) else np.zeros(0, int)
        rm_u.append(u); rm_c.append(c)
        per[name] = {"gaussians": int(len(idx)), "unobserved": int(len(u)), "unobserved_pct": 100 * float(len(u) / max(len(idx), 1)),
                     "contrib_median": float(np.median(tot[idx])) if len(idx) else None}
    allu = int((tot < args.eps).sum())
    rm_u, rm_c = np.concatenate(rm_u), np.concatenate(rm_c)
    log(f"Gaussian không view train nào thấy: cả cảnh {allu}/{N}; trong vật {len(rm_u)}")
    tmp = out / "_full.ply"
    g.save_ply(str(tmp))
    el = read_ply(tmp)
    sh = S["dataset"].sh_degree
    models = {}
    for tag, rm in (("unobserved", rm_u), ("ctrl", rm_c)):
        keep = np.ones(N, bool); keep[rm] = False
        pth = out / f"_{tag}.ply"
        write_ply(pth, el[keep])
        models[tag] = gaussians_from_ply(pth, sh, S["train"])
        pth.unlink(missing_ok=True)
    tmp.unlink(missing_ok=True)
    obj_t = torch.tensor(OBJ_IDS, device="cuda")
    acc = {t: {"psnr": [], "mad_obj": [], "chg_obj": []} for t in models}
    pf = []
    with torch.no_grad():
        for v in S["scene"].getTestCameras():
            nm = view_name(v)
            full = render_rgbd(v, g, S)[0]
            gt = v.original_image.cuda()[:3]
            pf.append(float(psnr(full[None], gt[None]).mean()))
            f = Path(args.labels) / f"{nm}.png"
            M = torch.isin(torch.from_numpy(cv2.imread(str(f), cv2.IMREAD_UNCHANGED).astype(np.int64)).cuda(), obj_t) if f.exists() else None
            imgs = {}
            for t, mdl in models.items():
                x = render_rgbd(v, mdl, S)[0]
                imgs[t] = x
                acc[t]["psnr"].append(float(psnr(x[None], gt[None]).mean()))
                if M is not None and M.sum() > 100:
                    dif = (full - x).abs().mean(0)[M]
                    acc[t]["mad_obj"].append(float(dif.mean())); acc[t]["chg_obj"].append(100 * float((dif > 2 / 255).float().mean()))
            if nm in SHEET:
                sep = torch.ones(3, full.shape[1], 6, device="cuda")
                row = torch.cat([full, sep, imgs["unobserved"], sep, imgs["ctrl"], sep,
                                 ((full - imgs["unobserved"]).abs() * 8).clamp(0, 1), sep, ((full - imgs["ctrl"]).abs() * 8).clamp(0, 1)], 2)
                cv2.imwrite(str(out / "preview" / f"A_{nm}.jpg"), to_u8(row)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
    res = {"A_contrib": {"eps": args.eps, "scene_unobserved": allu, "scene_gaussians": N, "objects": per,
                         "removed_in_objects": int(len(rm_u)), "psnr_test_full": float(np.mean(pf)),
                         **{t: {k: float(np.mean(v)) for k, v in d.items() if v} for t, d in acc.items()}}}
    log(json.dumps(res["A_contrib"]))
    del models
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------- B. khoét hull theo độ sâu render
    tol = args.tol_cm / cm
    views = S["views"]
    # optical axis of each view in world coordinates: x_cam_z = X_world . W[:3,2] + const (row-vector convention)
    cos_tab = []
    for v in views:
        ax = v.world_view_transform[:3, 2].detach().cpu().numpy().astype(np.float64)
        cos_tab.append(abs(float(ax @ n_t) / np.linalg.norm(ax)))
    cos_tab = np.array(cos_tab)
    good_view = cos_tab >= args.cos_min
    log(f"view gating: {int(good_view.sum())}/{len(views)} views usable for surface votes (|cos| >= {args.cos_min}); "
        f"|cos| percentiles 10/50/90 = {np.percentile(cos_tab, [10, 50, 90]).round(3).tolist()}")
    depth, alpha = [], []
    ones3 = torch.ones(N, 3, device="cuda")
    with torch.no_grad():
        for v in views:
            r = render(v, g, S["pipe"], torch.zeros(3, device="cuda"), splat_args=S["splat_args"], override_color=ones3)["render"]
            alpha.append(r[0].half().cpu())
            depth.append(render_rgbd(v, g, S)[1].float().cpu())
    da3, cams = None, []
    if args.da3_dir:
        da3 = load_da3(args.da3_dir)
        for v in views:
            Wd, Hd = int(v.image_width), int(v.image_height)
            fx, fy = Wd / (2 * np.tan(v.FoVx / 2)), Hd / (2 * np.tan(v.FoVy / 2))
            cams.append({"name": view_name(v), "W": Wd, "H": Hd, "T": v.world_view_transform.detach().cpu().numpy().astype(np.float64).T,
                         "K": np.array([[fx, 0, Wd / 2], [0, fy, Hd / 2], [0, 0, 1]])})
        log(f"DA3: {len(da3)} aligned depth maps, {sum(c['name'] in da3 for c in cams)}/{len(cams)} views matched")
    summary = {}
    for name in groups:
        f = Path(args.hulls) / f"hull_{name}.ply"
        if not f.exists():
            continue
        V = read_xyz(f)
        hv = (V @ n_t + d_t) * cm
        # vật đặt trên bàn: Gaussian của nó gần như không có phần dưới mặt bàn (hull thì luôn lấn xuống vì bàn che, nên không dùng hull)
        gi_ = np.where(lab == groups[name]["label"])[0]
        hg_ = ((np.stack([el["x"][gi_], el["y"][gi_], el["z"][gi_]], 1).astype(np.float64) @ n_t) + d_t) * cm
        below5 = float((hg_ < -5.0).mean()) if len(gi_) else 0.0
        rests_on_table = below5 < 0.01
        if rests_on_table:
            V = V[hv >= -0.5]                                  # vật trên bàn không có phần dưới mặt bàn
        clip_info = {"rests_on_table": rests_on_table, "gauss_below_5cm_pct": 100 * below5,
                     "hull_below_2cm_pct": 100 * float((hv < -2.0).mean())}
        if len(V) == 0:
            continue
        # lattice spacing: smallest gap between distinct x columns, ignoring float jitter (gaps < half the median gap)
        dx = np.diff(np.unique(np.round(V[:, 0], 5)))
        vox = float(np.min(dx[dx > 0.5 * np.median(dx)]))
        Vt = torch.tensor(V, dtype=torch.float32, device="cuda")
        Vh = torch.cat([Vt, torch.ones(len(V), 1, device="cuda")], 1)
        thru = torch.zeros(len(V), dtype=torch.int32, device="cuda")
        surf = torch.zeros_like(thru)      # all views (diagnostic, legacy)
        surf_g = torch.zeros_like(thru)    # non-grazing views only: evidence used for carving protection and "seen"
        for v, D, A, gv in zip(views, depth, alpha, good_view):
            Xc = Vh @ v.world_view_transform                      # [N,4], hệ camera
            z = Xc[:, 2]
            Wd, Hd = int(v.image_width), int(v.image_height)
            fx, fy = Wd / (2 * np.tan(v.FoVx / 2)), Hd / (2 * np.tan(v.FoVy / 2))
            u = (fx * Xc[:, 0] / z.clamp(min=1e-6) + Wd / 2).long()
            w = (fy * Xc[:, 1] / z.clamp(min=1e-6) + Hd / 2).long()
            ok = (z > 1e-3) & (u >= 0) & (u < Wd) & (w >= 0) & (w < Hd)
            Dd, Ad = D.cuda(), A.cuda().float()
            d = torch.zeros_like(z); a_ = torch.zeros_like(z)
            d[ok], a_[ok] = Dd[w[ok], u[ok]], Ad[w[ok], u[ok]]
            thru += (ok & ((a_ < 0.5) | (z < d - tol))).int()
            on_surf = ok & (a_ >= 0.5) & ((z - d).abs() <= tol)
            surf += on_surf.int()
            if gv:
                surf_g += on_surf.int()
        thru, surf, surf_g = thru.cpu().numpy(), surf.cpu().numpy(), surf_g.cpu().numpy()
        carve = thru >= args.min_vis
        protected = (surf_g >= args.protect_vis) if args.protect_vis > 0 else np.zeros(len(V), bool)
        n_prot_raw = int((carve & protected).sum())
        if args.protect_vis > 0 and args.protect_matter:
            gp = gi_[1 / (1 + np.exp(-el["opacity"][gi_].astype(np.float64))) > args.protect_op]
            Gp = np.stack([el["x"][gp], el["y"][gp], el["z"][gp]], 1).astype(np.float64)
            if len(Gp):
                offs = np.stack(np.meshgrid(*[np.array([-1, 0, 1])] * 3, indexing="ij"), -1).reshape(-1, 3) * vox
                protected &= gauss_voxels(V, vox, (Gp[None] + offs[:, None]).reshape(-1, 3))
            else:
                protected[:] = False
        free = carve & ~protected
        da3_info = {}
        if da3 is not None:
            th_d, va_d = da3_thru_votes(V, cams, da3, tol, args.da3_k_res, args.da3_edge)
            gsel = gi_[1 / (1 + np.exp(-el["opacity"][gi_].astype(np.float64))) > args.da3_op]
            G = np.stack([el["x"][gsel], el["y"][gsel], el["z"][gsel]], 1).astype(np.float64)
            matter = gauss_voxels(V, vox, G) if len(G) else np.zeros(len(V), bool)
            free_d = (th_d >= args.da3_min_vis) & ~matter
            da3_info = {"da3_free_voxels": int(free_d.sum()), "da3_only_free_voxels": int((free_d & ~free).sum()),
                        "da3_blocked_by_gaussians": int(((th_d >= args.da3_min_vis) & matter).sum()),
                        "da3_valid_votes_median": float(np.median(va_d))}
            free = free | free_d
        keep = ~free
        if rests_on_table:
            # table-plane bottom clip on the frame: nothing below the RANSAC table plane (+ --table_clip_cm)
            keep &= (V @ n_t + d_t) * cm >= args.table_clip_cm
        n_rescued = int((carve & protected).sum())
        F = V[keep]
        thru_k, surf_k = thru[keep], surf[keep]
        seen = surf_g[keep] >= args.min_vis
        shl = shell_of(F, vox) if len(F) else np.zeros(0, bool)
        h = (F @ n_t + d_t) * cm
        hp98 = float(np.percentile(h, 98)) if len(h) else 0.0
        hlow = float(np.percentile(h, 2)) if len(h) else 0.0
        bottom = h <= hlow + max(1.0, 0.15 * (hp98 - hlow))      # đáy = phần thấp nhất của chính vật
        mold = shl & ~seen
        cls = np.where(mold & bottom, 3, np.where(mold, 4, np.where(seen, 1, 2)))
        pal = np.array([[0, 0, 0], [40, 200, 60], [240, 150, 40], [230, 40, 40], [150, 60, 220]], np.uint8)
        write_rgb_ply(out / "frame" / f"{name}.ply", F, pal[cls])
        np.savez_compressed(out / "frame" / f"{name}.npz", xyz=F.astype(np.float32), cls=cls.astype(np.uint8), voxel=vox,
                            surf=surf_k.astype(np.int16), thru=thru_k.astype(np.int16))
        idx = np.where(lab == groups[name]["label"])[0]
        gx = np.stack([el["x"][idx], el["y"][idx], el["z"][idx]], 1).astype(np.float64)
        hg = (gx @ n_t + d_t) * cm if len(idx) else np.zeros(1)
        hmin = float(np.percentile(h, 2)) if len(h) else 0.0
        r = {**clip_info, **da3_info, "hull_voxels": int(len(V)), "voxel_cm": vox * cm, "hull_cm3": float(len(V) * (vox * cm) ** 3),
             "frame_cm3": float(len(F) * (vox * cm) ** 3), "carved_pct": 100 * float((~keep).mean()),
             "protected_voxels": n_rescued, "protected_voxels_before_matter_gate": n_prot_raw, "frame_seen_p98_cm": float(np.percentile(h[seen], 98)) if seen.any() else 0.0,
             "height_p98_cm": {"hull": float(np.percentile((V @ n_t + d_t) * cm, 98)), "frame": hp98,
                               "gaussians": float(np.percentile(hg, 98))},
             "shell": int(shl.sum()), "shell_seen_pct": 100 * float((shl & seen).sum() / max(shl.sum(), 1)),
             "mold_bottom_pct": 100 * float((mold & bottom).sum() / max(shl.sum(), 1)),
             "mold_side_back_pct": 100 * float((mold & ~bottom).sum() / max(shl.sum(), 1)),
             "interior_voxels": int((~shl).sum())}
        summary[name] = r
        log(f"{name}: {json.dumps(r)}")
    res["B_carve"] = {"tol_cm": args.tol_cm, "min_vis": args.min_vis, "views": len(views), "protect_vis": args.protect_vis,
                      "protect_matter": args.protect_matter, "protect_op": args.protect_op,
                      "cos_min": args.cos_min, "da3_dir": args.da3_dir, "da3_k_res": args.da3_k_res, "da3_edge": args.da3_edge,
                      "objects": summary}
    (out / "metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
