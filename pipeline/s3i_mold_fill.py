"""
Bước 2 (bản cơ sở, không mô hình sinh): lấp KHUÔN của từng vật bằng Gaussian dẹt; cùng một quy tắc cho mọi vật.

Khuôn = voxel vỏ khung không camera nào thấy (pipeline/s3m_mold.py: lớp 3 đáy, lớp 4 bên / sau).
  - vị trí: tâm voxel khuôn; pháp tuyến: hướng ra ngoài khung (gradient của lưới chiếm chỗ khung đã làm mịn); đáy: -pháp tuyến mặt bàn;
  - hình dạng: đĩa dẹt nằm trong mặt tiếp tuyến (cạnh 0.75 voxel, dày 0.1 voxel), opacity 0.95;
  - màu (f_dc) và confidence: trung bình có trọng số 1/d của k Gaussian GẦN NHẤT của chính vật (phần thấy được), SH bậc cao = 0.
Vật có khung không đáng tin (> --max_out_pct Gaussian quan sát nằm ngoài khung) thì bỏ qua, ghi lại lý do.
Đo:
  (a) 23 view test, ảnh cả cảnh trước / sau lấp: khuôn là chỗ không camera nào thấy nên ảnh phải gần như không đổi;
  (b) mỗi vật đứng RIÊNG (như khi nhấc khỏi bàn), camera ảo quanh vật ở nhiều góc cao (kể cả từ dưới lên):
      % pixel "thủng" = pixel thuộc bóng của khung (chiếu voxel khung) mà alpha của vật < 0.5; trước / sau lấp.
Xuất: filled.ply (cả cảnh), objects/<vật>.ply, metrics.json, preview/<vật>.jpg (hàng trên trước, hàng dưới sau)
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, binary_closing, binary_fill_holes
from scipy.spatial import cKDTree

from common import log, make_parser, load_scene, view_name, read_ply, write_ply, gaussians_from_ply, render_rgbd, to_u8
from arguments import get_combined_args
from gaussian_renderer import render
from scene.cameras import MiniCam
from utils.graphics_utils import getProjectionMatrix
from utils.image_utils import psnr
from s4_background import quat_from_R


def look_at_cam(C, T, up, W=512, H=512, fov_deg=40.0, znear=0.01, zfar=100.0):
    z = T - C; z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)                                            # trục y camera hướng xuống (quy ước 3DGS)
    R = np.stack([x, y, z], 0)
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = -R @ C
    fov = math.radians(fov_deg)
    wv = torch.tensor(M, dtype=torch.float32, device="cuda").transpose(0, 1)
    proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fov, fovY=fov).transpose(0, 1).cuda()
    full = wv.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    return MiniCam(W, H, fov, fov, znear, zfar, wv, full), R, C


def project(P, R, C, W, H, fov_deg):
    X = (P - C) @ R.T
    f = W / (2 * math.tan(math.radians(fov_deg) / 2))
    z = X[:, 2]
    u = f * X[:, 0] / np.maximum(z, 1e-9) + W / 2
    v = f * X[:, 1] / np.maximum(z, 1e-9) + H / 2
    ok = (z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u[ok].astype(int), v[ok].astype(int)


def qmul(a, b):
    w1, x1, y1, z1 = a.T; w2, x2, y2, z2 = b.T
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], 1)


def trellis_fill(tg, tgrid, el, obj_xyz, obj_idx, knn, seen_dilate=0):
    """Gaussian TRELLIS -> hệ cảnh; chỉ giữ Gaussian nằm ở ô CHƯA quan sát (không phải bề mặt thấy và lân cận 1 ô, không phải ô trống
    đã biết). Màu / opacity / scale / hướng giữ của TRELLIS; confidence lấy từ Gaussian gần nhất của chính vật."""
    from scipy.ndimage import binary_dilation
    g = read_ply(tg)
    z = np.load(tgrid)
    R, p0, center, size = z["R"], z["p0"], z["center"], float(z["size"])
    res = z["seen"].shape[0]
    P = np.stack([g["x"], g["y"], g["z"]], 1).astype(np.float64)
    ijk = np.clip(np.floor((P + 0.5) * res).astype(int), 0, res - 1)
    seen_n = binary_dilation(z["seen"], iterations=seen_dilate) if seen_dilate > 0 else z["seen"]
    known_free = z["carved"] | z["below"] if "carved" in z.files else z["free"]
    unk = ~seen_n & ~known_free
    op = 1 / (1 + np.exp(-g["opacity"].astype(np.float64)))
    keep = unk[ijk[:, 0], ijk[:, 1], ijk[:, 2]] & (op > 0.05)
    g, P = g[keep], P[keep]
    Pw = (P * size + center) @ R + p0
    new = np.zeros(len(g), dtype=el.dtype)
    new["x"], new["y"], new["z"] = Pw[:, 0], Pw[:, 1], Pw[:, 2]
    for c in ("f_dc_0", "f_dc_1", "f_dc_2", "opacity"):
        new[c] = g[c]
    for k in range(3):
        new[f"scale_{k}"] = g[f"scale_{k}"] + np.log(size)
    q = np.stack([g[f"rot_{k}"] for k in range(4)], 1).astype(np.float64)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    qR = quat_from_R(R.T)                                       # hệ chuẩn hoá (bàn) -> hệ cảnh
    qs = qmul(np.repeat(qR[None], len(q), 0), q)
    for k in range(4):
        new[f"rot_{k}"] = qs[:, k]
    if "confidence" in el.dtype.names and len(new):
        _, nb = cKDTree(obj_xyz).query(Pw, k=1)
        new["confidence"] = el["confidence"][obj_idx[nb]]
    return new, {"trellis_gaussians": int(keep.size), "trellis_kept_unobserved": int(keep.sum())}


def orbit_cams(F, n_t, fov, elev, az, W, H):
    """Camera ao nhin vao tam khung vat F theo (goc cao, goc phuong vi), cung cach dung voi phan do (b)."""
    T = F.mean(0)
    rad = np.linalg.norm(F - T, axis=1).max()
    dist = rad / math.tan(math.radians(fov) / 2) * 1.3
    a0 = np.array([1.0, 0, 0]) if abs(n_t[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(n_t, a0); e1 /= np.linalg.norm(e1); e2 = np.cross(n_t, e1)
    res = []
    for el_ in elev:
        for az_ in az:
            dvec = math.cos(math.radians(el_)) * (math.cos(math.radians(az_)) * e1 + math.sin(math.radians(az_)) * e2) \
                + math.sin(math.radians(el_)) * n_t
            cam, R, C = look_at_cam(T + dist * dvec, T, n_t, W, H, fov)
            u, v = project(F, R, C, W, H, fov)
            sil = np.zeros((H, W), np.uint8); sil[v, u] = 1
            sil = cv2.morphologyEx(sil, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            sil = cv2.erode(sil, np.ones((5, 5), np.uint8)) > 0
            res.append((el_, az_, cam, sil))
    return res


def inpaint_refine(base, new, F, vox, n_t, S, sh, out, name, args):
    """Track B (kieu 3DGIC / High-Fidelity 3DGS Inpainting, ban khong mo hinh sinh): camera ao quanh vat (goc cao / phuong vi
    KHAC bo camera do (b): elev -60/-15/15/45, az lech 22.5 do), render vat TRUOC lap; lo = pixel trong bong khung co alpha < 0.5;
    anh dich = inpaint OpenCV Telea chi tu pixel vat thay duoc (ngoai bong khung cung coi la chua biet nen nen den khong loang vao);
    toi uu f_dc / opacity / scale / rotation cua RIENG Gaussian lap (Gaussian goc dong bang, vi tri giu nguyen o voxel khuon =>
    d_out = 0, scale <= 1 voxel) theo L1 mau trong bong + lam_alpha * (1 - alpha) trong bong.
    Anh that khong doi vi Gaussian lap da qua bo loc nhin thay (vi tri khong doi); do lai PSNR test o phan (a)."""
    import random as _r
    Wd = Hd = 384
    black = torch.zeros(3, device="cuda")
    tmp = out / f"_inp_{name}.ply"
    write_ply(tmp, base)
    m0 = gaussians_from_ply(tmp, sh, S["train"])
    write_ply(tmp, np.concatenate([base, new]))
    g = gaussians_from_ply(tmp, sh, S["train"])
    tmp.unlink(missing_ok=True)
    nb = len(base)
    cams = orbit_cams(F, n_t, args.fov, [-60, -15, 15, 45], [22.5 + 45 * k for k in range(8)], Wd, Hd)
    targets, hole0 = [], []
    with torch.no_grad():
        for el_, az_, cam, sil in cams:
            if sil.sum() < 50:
                continue
            rgb = render(cam, m0, S["pipe"], black, splat_args=S["splat_args"])["render"][:3].clamp(0, 1)
            al = render(cam, m0, S["pipe"], black, splat_args=S["splat_args"],
                        override_color=torch.ones(nb, 3, device="cuda"))["render"][0]
            al_np = al.cpu().numpy()
            hole = sil & (al_np < 0.5)
            hole0.append(float(hole.sum() / sil.sum()))
            u8 = (rgb / al.clamp(min=1e-3)).clamp(0, 1).permute(1, 2, 0).cpu().numpy()   # bo nhan alpha cho mau pixel thay
            u8 = (u8 * 255).astype(np.uint8)
            unknown = ((~sil) | hole).astype(np.uint8)
            if hole.any() and (~unknown.astype(bool)).sum() > 20:
                fill = cv2.inpaint(u8, unknown, 5, cv2.INPAINT_TELEA)
            else:
                fill = u8
            tgt = torch.tensor(fill, dtype=torch.float32, device="cuda").permute(2, 0, 1) / 255
            targets.append((cam, torch.tensor(sil, device="cuda"), tgt, torch.tensor(hole, device="cuda")))
    if not targets:
        return new, {"views": 0}
    params = [(g._features_dc, 0.0025), (g._opacity, 0.05), (g._scaling, 0.005), (g._rotation, 0.001)]
    for t, _ in params:
        t.requires_grad_(True)
    opt = torch.optim.Adam([{"params": [t], "lr": lr} for t, lr in params], eps=1e-15)
    keep = torch.zeros(len(g.get_xyz), device="cuda"); keep[nb:] = 1
    smax = math.log(vox)
    rng = _r.Random(0)
    first = last = None
    for it in range(args.inpaint_iters):
        cam, sil, tgt, hole = targets[rng.randrange(len(targets))]
        rgb = render(cam, g, S["pipe"], black, splat_args=S["splat_args"])["render"][:3]
        al = render(cam, g, S["pipe"], black, splat_args=S["splat_args"],
                    override_color=torch.ones(len(keep), 3, device="cuda"))["render"][0]
        # mau: so anh da bo nhan alpha voi dich; chi trong bong khung
        l_rgb = ((rgb - tgt * al[None]).abs().mean(0))[sil].mean()
        l_a = (1 - al)[sil].clamp(min=0).mean()
        loss = l_rgb + args.inpaint_lam_alpha * l_a
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for t, _ in params:
            if t.grad is not None:
                t.grad.mul_(keep.view(-1, *([1] * (t.grad.dim() - 1))))
        opt.step()
        with torch.no_grad():
            g._scaling.data[nb:] = g._scaling.data[nb:].clamp(max=smax)
        if it == 0:
            first = (float(l_rgb), float(l_a))
        last = (float(l_rgb), float(l_a))
    for t, _ in params:
        t.requires_grad_(False)
    tmp2 = out / f"_inp2_{name}.ply"
    g.save_ply(str(tmp2))
    el2 = read_ply(tmp2)
    tmp2.unlink(missing_ok=True)
    res = new.copy()
    for c in res.dtype.names:
        if c in el2.dtype.names and c not in ("x", "y", "z"):
            res[c] = el2[c][nb:]
    return res, {"views": len(targets), "iters": args.inpaint_iters, "hole_frac_train_views_before": float(np.mean(hole0)),
                 "loss_first": first, "loss_last": last, "inpaint": "opencv_telea"}


def main():
    p, model, pipe, ss, _, _ = make_parser("S3i mold fill")
    p.add_argument("--s3", required=True)
    p.add_argument("--mold", required=True)
    p.add_argument("--base", required=True, help="thư mục S3c/<tag> (final.ply, oid_final.npy, metrics.json)")
    p.add_argument("--knn", type=int, default=8)
    p.add_argument("--max_out_pct", type=float, default=20.0)
    p.add_argument("--fov", type=float, default=40.0)
    p.add_argument("--vis_tol_cm", type=float, default=0.3)
    p.add_argument("--trellis_dir", default="", help="trellis_out/<tag>: <vật>/gaussian.ply (hệ chuẩn hoá [-0.5, 0.5]^3, z lên)")
    p.add_argument("--trellis_seen_dilate", type=int, default=0, help="nới ô bề mặt thấy trước khi chọn Gaussian TRELLIS (bộ lọc nhìn thấy đã bảo vệ phần đã thấy)")
    p.add_argument("--trellis_grid", default="", help="trellis_in: <vật>/grid.npz (center, size, R, p0, seen, carved, below)")
    p.add_argument("--image_inpaint", type=int, default=0,
                   help="1: tinh chinh Gaussian lap theo anh camera ao da inpaint (OpenCV Telea tu phan vat thay duoc), vi tri giu o voxel khuon")
    p.add_argument("--inpaint_iters", type=int, default=600)
    p.add_argument("--inpaint_lam_alpha", type=float, default=0.5)
    p.add_argument("--support_graph", default="", help="graph.json của tools/support_graph.py: lấp mặt vật đỡ dưới vết chân vật nằm trên")
    args = get_combined_args(p)
    log("s3i_mold_fill phiên bản: lọc phần lấp nhìn thấy + đáy trong vết chân (filter3 + trellis)")
    out = Path(args.out)
    (out / "objects").mkdir(parents=True, exist_ok=True); (out / "preview").mkdir(exist_ok=True)
    s3, mold, base = Path(args.s3), Path(args.mold), Path(args.base)
    sup = json.loads((s3 / "support.json").read_text())
    n_t, d_t = np.array(sup["table"]["normal"]), float(sup["table"]["d"])
    cm = 100 * float(sup["meters_per_unit"])
    groups = json.loads((s3 / "groups.json").read_text())["objects"]
    names = [n for n in groups if (mold / "frame" / f"{n}.npz").exists()]
    check = json.loads((base / "metrics.json").read_text()).get("frame_check", {})
    el = read_ply(base / "final.ply")
    oid = np.load(base / "oid_final.npy")
    xyz = np.stack([el["x"], el["y"], el["z"]], 1).astype(np.float64)
    S = load_scene(args, model, pipe, ss, ply_path=base / "final.ply")
    sh = S["dataset"].sh_degree

    new_all, info, per_obj_new = [], {}, {}
    for i, n in enumerate(names):
        z = np.load(mold / "frame" / f"{n}.npz")
        F, cls, vox = z["xyz"].astype(np.float64), z["cls"], float(z["voxel"])
        ok_frame = check.get(n, {}).get("outside_pct", 0.0) <= args.max_out_pct
        M = np.isin(cls, [3, 4])
        info[n] = {"frame_voxels": int(len(F)), "mold_voxels": int(M.sum()), "mold_bottom": int((cls == 3).sum()),
                   "mold_side_back": int((cls == 4).sum()), "frame_reliable": bool(ok_frame)}
        obj_idx = np.where(oid == i)[0]
        tg = Path(args.trellis_dir) / n / "gaussian.ply" if args.trellis_dir else None
        tgrid = Path(args.trellis_grid) / n / "grid.npz" if args.trellis_grid else None
        if tg is not None and tg.exists() and tgrid.exists() and ok_frame and len(obj_idx) >= args.knn:
            new, tinfo = trellis_fill(tg, tgrid, el, xyz[obj_idx], obj_idx, args.knn, args.trellis_seen_dilate)
            new_all.append(new)
            per_obj_new[i] = new
            info[n].update(tinfo)
            info[n]["filled"] = int(len(new))
            info[n]["fill_source"] = "trellis"
            continue
        if not ok_frame or M.sum() == 0 or len(obj_idx) < args.knn:
            info[n]["filled"] = 0
            continue
        # pháp tuyến ra ngoài khung
        lo = F.min(0) - 3 * vox
        nn_ = np.ceil((F.max(0) + 3 * vox - lo) / vox).astype(int) + 1
        occ = np.zeros(nn_, np.float32)
        ijk = np.round((F - lo) / vox).astype(int)
        occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = 1
        sm = gaussian_filter(occ, 1.5)
        gx, gy, gz = np.gradient(sm)
        mi = ijk[M]
        g = -np.stack([gx[mi[:, 0], mi[:, 1], mi[:, 2]], gy[mi[:, 0], mi[:, 1], mi[:, 2]], gz[mi[:, 0], mi[:, 1], mi[:, 2]]], 1)
        nrm = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
        bot = cls[M] == 3
        nrm[bot] = -n_t
        P = F[M]
        # đáy chỉ trong vết chân của chính Gaussian vật trên mặt phẳng (đóng lỗ nhỏ, giữ mảnh rời là rời)
        b0 = np.array([1.0, 0, 0]) if abs(n_t[0]) < 0.9 else np.array([0, 1.0, 0])
        f1 = np.cross(n_t, b0); f1 /= np.linalg.norm(f1); f2 = np.cross(n_t, f1)
        uvG = np.floor(np.stack([xyz[obj_idx] @ f1, xyz[obj_idx] @ f2], 1) / vox).astype(int)
        uvP = np.floor(np.stack([P @ f1, P @ f2], 1) / vox).astype(int)
        o2 = np.minimum(uvG.min(0), uvP.min(0)) - 3
        sz = np.maximum(uvG.max(0), uvP.max(0)) - o2 + 4
        fp = np.zeros(sz, bool); fp[uvG[:, 0] - o2[0], uvG[:, 1] - o2[1]] = True
        fp = binary_fill_holes(binary_closing(fp, iterations=2))
        infp = fp[uvP[:, 0] - o2[0], uvP[:, 1] - o2[1]]
        keep_b = ~bot | infp
        info[n]["bottom_outside_footprint_dropped"] = int((~keep_b).sum())
        P, nrm, bot = P[keep_b], nrm[keep_b], bot[keep_b]
        # màu từ Gaussian gần nhất của chính vật
        tree = cKDTree(xyz[obj_idx])
        d, nb = tree.query(P, k=args.knn)
        w = 1 / np.maximum(d, 1e-6); w /= w.sum(1, keepdims=True)
        src = obj_idx[nb]
        new = np.zeros(len(P), dtype=el.dtype)
        new["x"], new["y"], new["z"] = P[:, 0], P[:, 1], P[:, 2]
        for c in ("f_dc_0", "f_dc_1", "f_dc_2", "confidence"):
            if c in el.dtype.names:
                new[c] = (el[c][src] * w).sum(1)
        new["opacity"] = np.log(0.95 / 0.05)
        new["scale_0"] = new["scale_1"] = np.log(0.75 * vox)
        new["scale_2"] = np.log(0.1 * vox)
        for k_ in range(len(P)):
            nz = nrm[k_]
            a0 = np.array([1.0, 0, 0]) if abs(nz[0]) < 0.9 else np.array([0, 1.0, 0])
            t1 = np.cross(nz, a0); t1 /= np.linalg.norm(t1); t2 = np.cross(nz, t1)
            q = quat_from_R(np.stack([t1, t2, nz], 1))
            for j in range(4):
                new[f"rot_{j}"][k_] = q[j]
        new_all.append(new)
        per_obj_new[i] = new
        info[n]["filled"] = int(len(P))
        info[n]["color_src_dist_cm_median"] = float(np.median(d[:, 0]) * cm)
    # ---------------------------------------------------------------- khuôn tiếp xúc: mặt vật đỡ nằm dưới vật bên trên
    if args.support_graph and Path(args.support_graph).exists():
        sg = json.loads(Path(args.support_graph).read_text())
        a0 = np.array([1.0, 0, 0]) if abs(n_t[0]) < 0.9 else np.array([0, 1.0, 0])
        e1 = np.cross(n_t, a0); e1 /= np.linalg.norm(e1); e2 = np.cross(n_t, e1)
        p0 = -d_t * n_t
        for lower, uppers in sg.get("support_children", {}).items():
            if lower not in names or not info.get(lower, {}).get("frame_reliable", False):
                continue
            iA = names.index(lower)
            A = np.where(oid == iA)[0]
            zA = np.load(mold / "frame" / f"{lower}.npz")
            cell = float(zA["voxel"]) * 1.5
            ab = lambda P: np.floor(np.stack([(P - p0) @ e1, (P - p0) @ e2], 1) / cell).astype(np.int64)
            hA = (xyz[A] @ n_t + d_t)
            cA = ab(xyz[A])
            cF = ab(zA["xyz"].astype(np.float64))
            for up in uppers:
                if up not in names:
                    continue
                B = np.where(oid == names.index(up))[0]
                if len(B) == 0:
                    continue
                cB = {tuple(c) for c in ab(xyz[B])}
                # nới vết chân B 1 ô
                cB = {(a + da, b + db) for a, b in cB for da in (-1, 0, 1) for db in (-1, 0, 1)}
                ring = {(a + da, b + db) for a, b in cB for da in range(-3, 4) for db in range(-3, 4)} - cB
                inring = np.array([tuple(c) in ring for c in cA])
                if inring.sum() < 10:
                    continue
                hB = None
                top = np.percentile(hA[inring], 50)        # độ cao mặt vật đỡ quanh vết chân; chỗ nào lộ ở view thật sẽ bị bộ lọc bỏ
                del hB
                topm = inring & (hA > top - 0.5 / cm)
                fcells = np.array(sorted(cB & {tuple(c) for c in cF}))
                if len(fcells) == 0:
                    continue
                ca = (fcells[:, 0] + 0.5) * cell; cb = (fcells[:, 1] + 0.5) * cell
                P = p0 + ca[:, None] * e1 + cb[:, None] * e2 + (top - d_t - p0 @ n_t) * 0 + (top) * n_t
                P = P - ((P @ n_t + d_t) - top)[:, None] * n_t            # đặt đúng độ cao mặt trên
                src_idx = A[topm]
                tree = cKDTree(xyz[src_idx])
                dd, nb = tree.query(P, k=min(args.knn, len(src_idx)))
                dd, nb = np.atleast_2d(dd), np.atleast_2d(nb)
                w = 1 / np.maximum(dd, 1e-6); w /= w.sum(1, keepdims=True)
                src = src_idx[nb]
                new = np.zeros(len(P), dtype=el.dtype)
                new["x"], new["y"], new["z"] = P[:, 0], P[:, 1], P[:, 2]
                for c in ("f_dc_0", "f_dc_1", "f_dc_2", "confidence"):
                    if c in el.dtype.names:
                        new[c] = (el[c][src] * w).sum(1)
                new["opacity"] = np.log(0.95 / 0.05)
                new["scale_0"] = new["scale_1"] = np.log(0.75 * cell)
                new["scale_2"] = np.log(0.1 * cell)
                q = quat_from_R(np.stack([e1, e2, n_t], 1))
                for j in range(4):
                    new[f"rot_{j}"] = q[j]
                new_all.append(new)
                per_obj_new[iA] = np.concatenate([per_obj_new[iA], new]) if iA in per_obj_new else new
                info[lower].setdefault("contact_fill", {})[up] = {"gaussians": int(len(P)), "top_height_cm": float(top * cm)}
                log(f"khuôn tiếp xúc {lower} dưới {up}: {len(P)} Gaussian ở cao {top * cm:.2f} cm")
    # ---------------------------------------------------------------- lọc: phần lấp KHÔNG được đổi pixel đã quan sát
    # giữ Gaussian lấp chỉ khi ở mọi view train nó rơi vào pixel có alpha >= 0.5 và nằm sau bề mặt render (z >= depth - tol)
    tolu = args.vis_tol_cm / cm
    ones3 = torch.ones(S["g"].get_xyz.shape[0], 3, device="cuda")
    DA = []
    with torch.no_grad():
        for v in S["train"]:
            r = render(v, S["g"], S["pipe"], torch.zeros(3, device="cuda"), splat_args=S["splat_args"], override_color=ones3)["render"]
            DA.append((v, render_rgbd(v, S["g"], S)[1].float(), r[0].float()))
    for i in list(per_obj_new):
        new = per_obj_new[i]
        Pn = torch.tensor(np.stack([new["x"], new["y"], new["z"]], 1), dtype=torch.float32, device="cuda")
        Ph = torch.cat([Pn, torch.ones(len(Pn), 1, device="cuda")], 1)
        bad = torch.zeros(len(Pn), dtype=torch.bool, device="cuda")
        for v, D, Al in DA:
            Xc = Ph @ v.world_view_transform
            z = Xc[:, 2]
            Wd, Hd = int(v.image_width), int(v.image_height)
            fx, fy = Wd / (2 * math.tan(v.FoVx / 2)), Hd / (2 * math.tan(v.FoVy / 2))
            u = (fx * Xc[:, 0] / z.clamp(min=1e-6) + Wd / 2).long(); w_ = (fy * Xc[:, 1] / z.clamp(min=1e-6) + Hd / 2).long()
            ok = (z > 1e-3) & (u >= 0) & (u < Wd) & (w_ >= 0) & (w_ < Hd)
            dd = torch.full_like(z, 1e9); aa = torch.zeros_like(z)
            dd[ok], aa[ok] = D[w_[ok], u[ok]], Al[w_[ok], u[ok]]
            bad |= ok & ((aa < 0.5) | (z < dd - tolu))
        keep = (~bad).cpu().numpy()
        n_ = names[i]
        info[n_]["fill_kept"] = int(keep.sum()); info[n_]["fill_dropped_visible"] = int((~keep).sum())
        per_obj_new[i] = new[keep]
    del DA
    torch.cuda.empty_cache()
    log("lọc phần lấp nhìn thấy được: " + json.dumps({names[i]: [info[names[i]]['fill_kept'], info[names[i]]['fill_dropped_visible']] for i in per_obj_new}))
    if args.image_inpaint:
        for i in list(per_obj_new):
            if len(per_obj_new[i]) == 0:
                continue
            n_ = names[i]
            zf = np.load(mold / "frame" / f"{n_}.npz")
            per_obj_new[i], info[n_]["image_inpaint"] = inpaint_refine(
                el[oid == i], per_obj_new[i], zf["xyz"].astype(np.float64), float(zf["voxel"]), n_t, S, sh, out, n_, args)
            log(f"{n_}: image inpaint {info[n_]['image_inpaint']}")
    filled = np.concatenate([el] + list(per_obj_new.values())) if per_obj_new else el
    oid_f = np.concatenate([oid] + [np.full(len(v), i) for i, v in per_obj_new.items()])
    write_ply(out / "filled.ply", filled)
    np.save(out / "oid_filled.npy", oid_f)
    log(f"lấp khuôn: {info}")

    # ---------------------------------------------------------------- (a) ảnh test cả cảnh
    g0 = S["g"]
    g1 = gaussians_from_ply(out / "filled.ply", sh, S["train"])
    pa, pb, mad = [], [], []
    with torch.no_grad():
        for v in S["scene"].getTestCameras():
            gt = v.original_image.cuda()[:3]
            a = render_rgbd(v, g0, S)[0]; b = render_rgbd(v, g1, S)[0]
            pa.append(float(psnr(a[None], gt[None]).mean())); pb.append(float(psnr(b[None], gt[None]).mean()))
            mad.append(float((a - b).abs().mean()))
    res = {"objects": info, "test_psnr_before": float(np.mean(pa)), "test_psnr_after": float(np.mean(pb)),
           "test_mean_abs_change": float(np.mean(mad))}
    log(f"test: {res['test_psnr_before']:.4f} -> {res['test_psnr_after']:.4f}, đổi trung bình {res['test_mean_abs_change']:.6f}")

    # ---------------------------------------------------------------- (b) vật đứng riêng, camera ảo
    black = torch.zeros(3, device="cuda")
    elev = [-75, -30, 0, 30, 60]
    az = list(range(0, 360, 45))
    Wd = Hd = 384
    for i, n in enumerate(names):
        if info[n].get("filled", 0) == 0 and "contact_fill" not in info[n]:
            continue
        z = np.load(mold / "frame" / f"{n}.npz")
        F = z["xyz"].astype(np.float64)
        T = F.mean(0)
        rad = np.linalg.norm(F - T, axis=1).max()
        dist = rad / math.tan(math.radians(args.fov) / 2) * 1.3
        sel_b = np.where(oid == i)[0]
        sel_a = np.where(oid_f == i)[0]
        models = {}
        for tag, sel, src in (("before", sel_b, el), ("after", sel_a, filled)):
            pth = out / f"_{tag}_{n}.ply"
            write_ply(pth, src[sel])
            models[tag] = gaussians_from_ply(pth, sh, S["train"])
            if tag == "after":
                (out / "objects" / f"{n}.ply").write_bytes(pth.read_bytes())
            pth.unlink(missing_ok=True)
        a0 = np.array([1.0, 0, 0]) if abs(n_t[0]) < 0.9 else np.array([0, 1.0, 0])
        e1 = np.cross(n_t, a0); e1 /= np.linalg.norm(e1); e2 = np.cross(n_t, e1)
        holes = {"before": {}, "after": {}}
        rows = {"before": [], "after": []}
        with torch.no_grad():
            for el_ in elev:
                for az_ in az:
                    dvec = math.cos(math.radians(el_)) * (math.cos(math.radians(az_)) * e1 + math.sin(math.radians(az_)) * e2) \
                        + math.sin(math.radians(el_)) * n_t
                    cam, R, C = look_at_cam(T + dist * dvec, T, n_t, Wd, Hd, args.fov)
                    u, v = project(F, R, C, Wd, Hd, args.fov)
                    sil = np.zeros((Hd, Wd), np.uint8); sil[v, u] = 1
                    sil = cv2.morphologyEx(sil, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
                    sil = cv2.erode(sil, np.ones((5, 5), np.uint8)) > 0             # bỏ viền bóng để không tính lệch biên
                    for tag, mdl in models.items():
                        ones = torch.ones(mdl.get_xyz.shape[0], 3, device="cuda")
                        al = render(cam, mdl, S["pipe"], black, splat_args=S["splat_args"], override_color=ones)["render"][0].cpu().numpy()
                        holes[tag].setdefault(el_, []).append(float((al[sil] < 0.5).mean()) if sil.sum() > 20 else np.nan)
                        if az_ in (0, 135, 270):
                            rgb = render(cam, mdl, S["pipe"], torch.ones(3, device="cuda"), splat_args=S["splat_args"])["render"][:3]
                            rows[tag].append(to_u8(rgb))
        info[n]["hole_pct_by_elevation"] = {t: {str(e): 100 * float(np.nanmean(v)) for e, v in h.items()} for t, h in holes.items()}
        info[n]["hole_pct_mean"] = {t: 100 * float(np.nanmean([x for v in h.values() for x in v])) for t, h in holes.items()}
        log(f"{n}: thủng % {info[n]['hole_pct_mean']}")
        img = np.vstack([np.hstack(rows["before"]), np.hstack(rows["after"])])
        cv2.imwrite(str(out / "preview" / f"{n}.jpg"), img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
        del models
        torch.cuda.empty_cache()
    res["objects"] = info
    res["virtual_views"] = {"elevations_deg": elev, "azimuths_deg": az, "fov_deg": args.fov,
                            "preview": "hang tren truoc, hang duoi sau; moi goc cao 3 goc phuong vi 0/135/270"}
    (out / "metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
