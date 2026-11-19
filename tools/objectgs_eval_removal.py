"""
ObjectGS train trên COLMAP của mình + mask SAM 3: đo (1) chất lượng ID render, (2) độ sạch khi xoá vật.
Chạy trong env objectgs, cwd = repo ObjectGS (xem modal_objectgs.py::_eval_removal).

    python /opt/objectgs_eval_removal.py -m outputs/safegs_sam3/sam3/teatime/<run> --support <S3_label>/support.json --out <dir>

Xoá vật = bỏ các anchor mang ID của vật khi render (cùng cách render_object.py của tác giả, đảo mask).
Kịch bản A: xoá vật trên bàn, gấu / cừu ở lại (pixel bị gấu / cừu che bị loại khỏi đo).
Kịch bản B: xoá cả gấu / cừu.
Vùng đo = "vết chân": chiếu anchor của vật trên bàn xuống mặt bàn (mặt phẳng từ S3, cùng hệ toạ độ COLMAP), lưới 1 cm, nới 1 ô.
Với pixel có tia nhìn cắt mặt bàn trong vết chân, lấy độ sâu render, đổi ra điểm 3D, đo độ cao so với mặt bàn:
  ghost  = nằm cao hơn mặt bàn > 1 cm (còn vật / floater lơ lửng)
  flat   = trong 1 cm quanh mặt bàn (bề mặt sạch)
  sunk   = thấp hơn mặt bàn > 1 cm (thủng, nhìn xuyên xuống thứ khác)
  empty  = độ phủ alpha < 0.3 (không có gì)
Thêm: số anchor còn lại lơ lửng trong khoảng trống (ID khác gấu / cừu), độ nét (phương sai Laplacian trong vết chân / vành quanh)
so với ảnh thật, và độ chính xác ID render so với mask SAM 3 trên view test.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

TABLE = 13
OCCLUDERS = [20, 30]                                              # gấu, cừu
REMOVE_TABLE_OBJECTS = [80, 70, 60, 100, 90, 50, 130, 110, 170, 171, 180, 181, 160, 161]
HULL_OF = {80: "coc_su", 70: "ly_tra", 60: "dia", 100: "banh_quy", 90: "tao", 50: "tui_banh", 130: "khan_giay",
           110: "thia", 170: "vung_tra", 171: "vung_tra", 180: "goi_keo", 181: "goi_keo", 160: "coc_kem_sua", 161: "coc_kem_sua"}


def read_hull(path):
    """Tâm voxel của hull_<vật>.ply (tools/mask_hull.py: float x y z + uchar r g b, little endian)."""
    raw = Path(path).read_bytes()
    i = raw.index(b"end_header\n") + len(b"end_header\n")
    n = int(raw.split(b"element vertex ")[1].split(b"\n")[0])
    d = np.frombuffer(raw[i:], dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("c", "u1", 3)], count=n)
    return np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_path", required=True)
    ap.add_argument("--support", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--remove_ids", default="", help="danh sách ID vật trên bàn cần xoá (mặc định: tất cả vật trên bàn); "
                    "v1 chỉ có cốc 80, ly 70, đĩa 60, bánh 100, táo 90, túi 50 -> dùng '80,70,60,100,90,50' để so")
    ap.add_argument("--hull_dir", default="", help="thư mục hull_<vật>.ply (tools/mask_hull.py): thêm kịch bản *_hullcut, xoá "
                    "thêm MỌI anchor (kể cả ID nền) nằm trong hull của vật đã xoá và cao hơn mặt bàn > 1.5 cm")
    ap.add_argument("--hull_margin_cm", type=float, default=1.5, help="nới hull (offset của Gaussian vượt ra ngoài anchor)")
    ap.add_argument("--hull_hmin_cm", type=float, default=1.5, help="chỉ cắt anchor cao hơn mặt bàn ngưỡng này; vật dẹt "
                    "(gói kẹo, khăn giấy, vành đĩa) nằm dưới 1.5 cm nên cần ngưỡng thấp hơn, đổi lại có thể cắt cả mặt bàn")
    ap.add_argument("--gauss_margin_cm", type=float, default=0.5, help="kịch bản *_gauss: nới hull khi lọc theo vị trí của "
                    "chính Gaussian sinh ra (không phải anchor)")
    ap.add_argument("--gauss_hmin_cm", type=float, default=0.3, help="kịch bản *_gauss: chỉ cắt Gaussian cao hơn mặt bàn ngưỡng này "
                    "(âm lớn = cắt cả mặt bàn trong hull, để thử giới hạn)")
    ap.add_argument("--foot_hmin_cm", type=float, default=-2.0, help="kịch bản *_foot: cắt mọi Gaussian trong cột trên vết chân "
                    "hull (XY) từ độ cao này trở lên, KỂ CẢ lớp mặt bàn (vật dẹt bị vẽ lên chính Gaussian mặt bàn)")
    ap.add_argument("--only", default="", help="chỉ chạy các kịch bản này (cách nhau bằng dấu phẩy); bỏ qua bước đo ID")
    ap.add_argument("--probe", action="store_true", help="chẩn đoán phần còn sót sau khi xoá: (a) hull có phủ mask vật không, "
                    "(b) alpha trong mask vật của từng nhóm Gaussian (độ cao x nhãn anchor); bỏ qua các kịch bản đo")
    ap.add_argument("--probe_views", default="test_frame_00017,test_frame_00065,test_frame_00113")
    a = ap.parse_args()
    global REMOVE_TABLE_OBJECTS
    if a.remove_ids:
        REMOVE_TABLE_OBJECTS = [int(x) for x in a.remove_ids.split(",")]
    sys.path.insert(0, os.getcwd())
    from scene import Scene
    from utils.general_utils import safe_state, parse_cfg
    import gaussian_renderer
    from gaussian_renderer.render import prefilter_voxel

    cfg = yaml.load(open(os.path.join(a.model_path, "config.yaml")), Loader=yaml.FullLoader)
    scene_name = a.model_path.rstrip("/").split("/")[-2]
    cfg["model_params"]["exp_name"] = os.path.join(cfg["model_params"]["exp_name"], scene_name)
    cfg["model_params"]["source_path"] = os.path.join(cfg["model_params"]["source_path"], scene_name)
    lp, op, pp = parse_cfg(cfg)
    lp.model_path = a.model_path
    safe_state(True)
    pp.add_prefilter = not (pp.no_prefilter_step > 0)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sup = json.loads(Path(a.support).read_text())
    n_t, d_t = np.array(sup["table"]["normal"], np.float64), float(sup["table"]["d"])
    mpu = float(sup["meters_per_unit"])

    import scene as scene_mod
    with torch.no_grad():
        g = getattr(scene_mod, lp.model_config["name"])(**lp.model_config["kwargs"])
        sc = Scene(lp, g, load_iteration=-1, shuffle=False)
        g.eval()
        bg = sc.background
        views = sorted(sc.getTestCameras(), key=lambda v: v.image_name)
        lab = g.label_ids.squeeze()
        anchors = g.get_anchor.detach()
        present = sorted(int(x) for x in torch.unique(lab).tolist())
        counts = {int(k): int((lab == k).sum()) for k in present}
        print("anchors:", len(lab), "labels:", counts, flush=True)

        def render(view, keep=None):
            g.set_anchor_mask(view.camera_center, view.resolution_scale)
            vis = prefilter_voxel(view, g).squeeze() if pp.add_prefilter else g._anchor_mask
            pkg = gaussian_renderer.render(view, g, pp, bg, visible_mask=vis, training=False, object_mask=keep)
            return pkg

        # ------------------------------------------------------------ 1. ID render so với mask SAM 3 (view test)
        acc_fg, per_lab = [], {}
        for v in ([] if a.only else views):
            ids = g.id_encoder.inverse_transform(render(v)["render_semantics"]).squeeze().cpu().numpy().astype(np.int32)
            gt = v.object_mask.cpu().numpy().astype(np.int32)
            if ids.shape != gt.shape:
                ids = cv2.resize(ids.astype(np.int16), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            fg = gt > 0
            acc_fg.append(float((ids[fg] == gt[fg]).mean()))
            for l in np.unique(gt):
                if l == 0 or (gt == l).sum() < 300:
                    continue
                i_ = np.logical_and(ids == l, gt == l).sum(); u_ = np.logical_or(ids == l, gt == l).sum()
                per_lab.setdefault(int(l), []).append(float(i_ / u_))
        seg = {"fg_pixel_accuracy": float(np.mean(acc_fg)),
               "per_label_iou": {k: round(float(np.mean(v)), 3) for k, v in sorted(per_lab.items())}} if acc_fg else {}
        print("seg:", json.dumps(seg), flush=True)

        # ------------------------------------------------------------ 2. vết chân và khoảng trống
        rm_ids = [i for i in REMOVE_TABLE_OBJECTS if i in present]
        rm_mask = torch.isin(lab, torch.tensor(rm_ids, device=lab.device))
        X = anchors[rm_mask].cpu().numpy().astype(np.float64)
        s_w = 1.0 if np.median(X @ n_t + d_t) >= 0 else -1.0
        hX = s_w * (X @ n_t + d_t)
        hmax = float(np.percentile(hX, 98))
        a0 = np.array([1.0, 0, 0]) if abs(n_t[0]) < 0.9 else np.array([0, 1.0, 0])
        u_ = np.cross(n_t, a0); u_ /= np.linalg.norm(u_); v_ = np.cross(n_t, u_)
        p0 = -d_t * n_t
        cell = 0.01 / mpu
        oa = np.floor((X - p0) @ u_ / cell).astype(int); ob = np.floor((X - p0) @ v_ / cell).astype(int)
        amin, bmin = oa.min() - 2, ob.min() - 2
        foot = np.zeros((ob.max() - bmin + 3, oa.max() - amin + 3), np.uint8)
        foot[ob - bmin, oa - amin] = 1
        foot = cv2.dilate(foot, np.ones((3, 3), np.uint8)) > 0

        def cells(P):
            ia = np.floor((P - p0) @ u_ / cell).astype(int) - amin
            ib = np.floor((P - p0) @ v_ / cell).astype(int) - bmin
            ok = (ia >= 0) & (ia < foot.shape[1]) & (ib >= 0) & (ib < foot.shape[0])
            return ia, ib, ok

        # ------------------------------------------------------------ 2b. cắt theo visual hull (chỉ từ mask + camera SfM)
        hullcut, hull_info = None, {}
        if a.hull_dir:
            from scipy.spatial import cKDTree
            names_ = sorted({HULL_OF[i] for i in rm_ids if i in HULL_OF})
            pts = [read_hull(Path(a.hull_dir) / f"hull_{n}.ply") for n in names_ if (Path(a.hull_dir) / f"hull_{n}.ply").exists()]
            pts = np.concatenate(pts)
            Pall = anchors.cpu().numpy().astype(np.float64)
            dist, _ = cKDTree(pts).query(Pall, distance_upper_bound=a.hull_margin_cm / (mpu * 100))
            hall = s_w * (Pall @ n_t + d_t) * mpu * 100
            cut = np.isfinite(dist) & (hall > a.hull_hmin_cm) & ~np.isin(lab.cpu().numpy(), OCCLUDERS)
            hullcut = torch.from_numpy(cut).to(lab.device)
            cl = lab.cpu().numpy()[cut]
            hull_info = {"hull_voxels": int(len(pts)), "cut_anchors": int(cut.sum()),
                         "cut_by_label": {int(k): int((cl == k).sum()) for k in np.unique(cl)}}
            print("hullcut:", json.dumps(hull_info), flush=True)

        # ------------------------------------------------------------ 2c. lọc theo vị trí CỦA CHÍNH Gaussian sinh ra
        # Mỗi anchor sinh ~10 Gaussian lệch khỏi anchor (offset x scale); cắt theo vị trí anchor có thể sót. Ở đây bọc
        # generate_neural_gaussians: Gaussian có tâm trong hull (lưới voxel 0.5 cm trên GPU) bị bỏ trước khi rasterize.
        st = {"hull": False, "ghost": False, "bin": None, "grp": None, "foot": False}
        class_idx = g.id_encoder.class_to_idx
        dev = lab.device
        nT = torch.tensor(n_t, dtype=torch.float32, device=dev); dT = float(d_t)
        uT = torch.tensor(u_, dtype=torch.float32, device=dev); vT = torch.tensor(v_, dtype=torch.float32, device=dev)
        p0T = torch.tensor(p0, dtype=torch.float32, device=dev)
        footT = torch.from_numpy(foot).to(dev)
        occl_cols = [g.id_encoder.class_to_idx[l] for l in OCCLUDERS if l in g.id_encoder.class_to_idx]
        lo = shape = occ = None
        cu = 0.005 / mpu                                                  # ô 0.5 cm
        if a.hull_dir:
            mu = a.gauss_margin_cm / (mpu * 100)
            ptsT = torch.from_numpy(pts).float().to(dev)
            lo = ptsT.min(0).values - mu - 2 * cu
            hi = ptsT.max(0).values + mu + 2 * cu
            shape = torch.ceil((hi - lo) / cu).long()
            occ = torch.zeros(tuple(shape.tolist()), dtype=torch.bool, device=dev)
            ii = ((ptsT - lo) / cu).floor().long()
            occ[ii[:, 0], ii[:, 1], ii[:, 2]] = True
            r = int(np.ceil(a.gauss_margin_cm / 0.5))
            occ = torch.nn.functional.max_pool3d(occ[None, None].float(), 2 * r + 1, 1, r)[0, 0] > 0

        footHT = None
        if a.hull_dir:                                    # vết chân XY của hull (lưới 1 cm, nới 1 ô) cho kịch bản *_foot
            ha = np.floor((pts - p0) @ u_ / cell).astype(int); hb = np.floor((pts - p0) @ v_ / cell).astype(int)
            amin_h, bmin_h = int(ha.min()) - 2, int(hb.min()) - 2
            foot_h = np.zeros((hb.max() - bmin_h + 3, ha.max() - amin_h + 3), np.uint8)
            foot_h[hb - bmin_h, ha - amin_h] = 1
            footHT = torch.from_numpy(cv2.dilate(foot_h, np.ones((3, 3), np.uint8)) > 0).to(dev)

        def in_foot(xyz, h):
            ia = torch.floor(((xyz - p0T) @ uT) / cell).long() - amin_h
            ib = torch.floor(((xyz - p0T) @ vT) / cell).long() - bmin_h
            ok = (ia >= 0) & (ia < footHT.shape[1]) & (ib >= 0) & (ib < footHT.shape[0])
            out_ = torch.zeros(len(xyz), dtype=torch.bool, device=xyz.device)
            out_[ok] = footHT[ib[ok], ia[ok]]
            return out_ & (h > a.foot_hmin_cm) & (h < hmax * mpu * 100)

        def in_hull(xyz):
            ii = ((xyz - lo) / cu).floor().long()
            inb = ((ii >= 0) & (ii < shape)).all(1)
            out_ = torch.zeros(len(xyz), dtype=torch.bool, device=xyz.device)
            j = ii[inb]
            out_[inb] = occ[j[:, 0], j[:, 1], j[:, 2]]
            return out_

        def in_column(xyz, h):
            ia = torch.floor(((xyz - p0T) @ uT) / cell).long() - int(amin)
            ib = torch.floor(((xyz - p0T) @ vT) / cell).long() - int(bmin)
            ok = (ia >= 0) & (ia < footT.shape[1]) & (ib >= 0) & (ib < footT.shape[0])
            out_ = torch.zeros(len(xyz), dtype=torch.bool, device=xyz.device)
            out_[ok] = footT[ib[ok], ia[ok]]
            return out_ & (h > 1.5) & (h < hmax * mpu * 100)

        orig_gen = g.generate_neural_gaussians

        def gen(viewpoint_camera, visible_mask=None, training=True):
            o_ = orig_gen(viewpoint_camera, visible_mask, training)
            if not (st["hull"] or st["ghost"] or st["foot"] or st["bin"] is not None or st["grp"] is not None):
                return o_
            xyz, offs, color, opacity, scaling, rot, shd, mask, sem = o_
            h = s_w * (xyz @ nT + dT) * (mpu * 100)
            is_occ = sem[:, occl_cols].sum(1) > 0.5 if occl_cols else torch.zeros(len(xyz), dtype=torch.bool, device=xyz.device)
            k_ = torch.ones(len(xyz), dtype=torch.bool, device=xyz.device)
            if st["bin"] is not None:                         # chẩn đoán: chỉ giữ Gaussian trong dải độ cao (cm) và nhóm nhãn
                k_ &= (h > st["bin"][0]) & (h <= st["bin"][1])
            if st["grp"] is not None:
                z_ = torch.zeros(len(xyz), dtype=torch.bool, device=xyz.device)
                g0 = sem[:, class_idx[0]] > 0.5 if 0 in class_idx else z_
                g13 = sem[:, class_idx[13]] > 0.5 if 13 in class_idx else z_
                k_ &= {"bg0": g0, "tab13": g13, "other": ~(g0 | g13)}[st["grp"]]
            if st["foot"]:
                k_ &= ~(in_foot(xyz, h) & ~is_occ)
            if st["hull"]:
                k_ &= ~(in_hull(xyz) & (h > a.gauss_hmin_cm) & ~is_occ)
            if st["ghost"]:
                k_ &= in_column(xyz, h) & ~is_occ
            return xyz[k_], offs[k_], color[k_], opacity[k_], scaling[k_], rot[k_], shd, mask, sem[k_]
        g.generate_neural_gaussians = gen

        # anchor còn lại lơ lửng trong cột khoảng trống (không tính gấu / cừu): mask bool theo anchor
        def ghost_sel(removed, hmin_cm, extra=None):
            cand = ~torch.isin(lab, torch.tensor(removed, device=lab.device))
            cand &= ~torch.isin(lab, torch.tensor(OCCLUDERS, device=lab.device))
            if extra is not None:
                cand &= ~extra
            idx = cand.nonzero().squeeze(1)
            P = anchors[idx].cpu().numpy().astype(np.float64)
            h = s_w * (P @ n_t + d_t) * mpu * 100
            ia, ib, ok = cells(P)
            inside = ok.copy()
            inside[ok] = foot[ib[ok], ia[ok]]
            m = inside & (h > hmin_cm) & (h < hmax * mpu * 100)
            sel = torch.zeros_like(cand)
            sel[idx[torch.from_numpy(m).to(idx.device)]] = True
            return sel

        # ------------------------------------------------------------ 3. kịch bản xoá
        scen = {"A_table_objects": rm_ids, "B_all_objects": rm_ids + [i for i in OCCLUDERS if i in present]}
        if hullcut is not None:
            scen["A_hullcut"] = scen["A_table_objects"]
            scen["B_hullcut"] = scen["B_all_objects"]
            scen["A_gauss"] = scen["A_table_objects"]
            scen["B_gauss"] = scen["B_all_objects"]
            scen["A_foot"] = scen["A_table_objects"]
            scen["B_foot"] = scen["B_all_objects"]
        removed_b = list(scen["B_all_objects"])
        if a.only:
            scen = {k: v for k, v in scen.items() if k in a.only.split(",")}
        if a.probe:
            scen = {}
        res = {"anchors": counts, "removed_table_ids": rm_ids, "hullcut": hull_info,
               "removed_anchors": int(rm_mask.sum()), "object_height_p98_cm": hmax * mpu * 100,
               "footprint_cells_1cm": int(foot.sum()), "seg": seg, "scenarios": {}}
        lap = lambda im: cv2.Laplacian(cv2.cvtColor((im * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY), cv2.CV_64F)
        sheets = []
        for name, removed in scen.items():
            extra = hullcut if name.endswith("_hullcut") else None
            st["hull"] = name.endswith("_gauss")
            st["foot"] = name.endswith("_foot")
            keep = ~torch.isin(lab, torch.tensor(removed, device=lab.device))
            if extra is not None:
                keep &= ~extra
            R = {"ghost": [], "flat": [], "sunk": [], "empty": [], "h_abs": [], "sharp": [], "sharp_gt": [],
                 "g_alpha": [], "g_cover": [], "gg_alpha": [], "gg_cover": []}
            gsel = ghost_sel(removed, 1.5, extra)    # chỉ các anchor ma (cao hơn mặt bàn > 1.5 cm trong cột trống)
            for k, v in enumerate(views):
                H, W = int(v.image_height), int(v.image_width)
                pkg = render(v, keep)
                rgb = torch.clamp(pkg["render"], 0, 1)
                dep = pkg["render_depth"][0].cpu().numpy()
                alp = pkg["render_alphas"][0].cpu().numpy()
                gt = v.original_image.cuda()[:3]
                gtl = v.object_mask.cpu().numpy().astype(np.int32)
                if gtl.shape != (H, W):
                    gtl = cv2.resize(gtl.astype(np.int16), (W, H), interpolation=cv2.INTER_NEAREST)
                c2w = v.c2w.cpu().numpy().astype(np.float64)
                o = c2w[:3, 3]
                uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
                dirs = np.stack([(uu - v.cx) / v.fx, (vv - v.cy) / v.fy, np.ones_like(uu)], -1).reshape(-1, 3) @ c2w[:3, :3].T
                tt = -(o @ n_t + d_t) / (dirs @ n_t)
                Xp = o + tt[:, None] * dirs
                ia, ib, ok = cells(Xp)
                fp = np.zeros(H * W, bool)
                fp[ok & (tt > 0)] = foot[ib[ok & (tt > 0)], ia[ok & (tt > 0)]]
                fp = fp.reshape(H, W)
                occl = np.isin(gtl, [i for i in OCCLUDERS if i not in removed])
                occl = cv2.dilate(occl.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
                fp &= ~occl
                if fp.sum() < 50:
                    continue
                P = o + dep[..., None] * dirs.reshape(H, W, 3)
                s_c = 1.0 if (o @ n_t + d_t) >= 0 else -1.0
                h = s_c * (P @ n_t + d_t) * mpu * 100
                empty = alp < 0.3
                valid = fp & ~empty
                n = fp.sum()
                # alpha của RIÊNG các anchor ma (không có anchor nào khác che): 0 = sạch, 1 = ma đặc
                ga = render(v, gsel)["render_alphas"][0].cpu().numpy() if bool(gsel.any()) else np.zeros((H, W))
                R["g_alpha"].append(float(ga[fp].mean())); R["g_cover"].append(float((ga[fp] > 0.1).mean()))
                # cùng ý nhưng ở cấp Gaussian sinh ra: chỉ giữ Gaussian trong cột khoảng trống, cao hơn mặt bàn > 1.5 cm
                st["ghost"] = True
                try:
                    gga = render(v, keep)["render_alphas"][0].cpu().numpy()
                except Exception as e:                       # không còn Gaussian nào để rasterize
                    print("ghost render:", type(e).__name__, str(e)[:80], flush=True)
                    gga = np.zeros((H, W))
                st["ghost"] = False
                R["gg_alpha"].append(float(gga[fp].mean())); R["gg_cover"].append(float((gga[fp] > 0.1).mean()))
                R["empty"].append(float((fp & empty).sum() / n))
                if valid.sum():
                    hv = h[valid]
                    R["ghost"].append(float((hv > 1).sum() / n)); R["sunk"].append(float((hv < -1).sum() / n))
                    R["flat"].append(float((np.abs(hv) <= 1).sum() / n)); R["h_abs"].append(float(np.median(np.abs(hv))))
                ring = (cv2.dilate(fp.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0) & ~fp & ~occl
                if ring.sum() > 100:
                    Lr = lap(rgb.permute(1, 2, 0).cpu().numpy()); Lg = lap(gt.permute(1, 2, 0).cpu().numpy())
                    R["sharp"].append(float(Lr[fp].var() / max(Lr[ring].var(), 1e-9)))
                    R["sharp_gt"].append(float(Lg[fp].var() / max(Lg[ring].var(), 1e-9)))
                if k in (2, 8, 14):
                    hh, hf = st["hull"], st["foot"]; st["hull"] = st["foot"] = False
                    base = (torch.clamp(render(v)["render"], 0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    st["hull"], st["foot"] = hh, hf
                    cur = (rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    tile = np.hstack([base, cur])
                    cv2.putText(tile, f"{name} {v.image_name}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    sheets.append((name, tile[..., ::-1]))
            mean = lambda x: float(np.mean(x)) if x else None
            res["scenarios"][name] = {
                "removed_ids": removed, "views_used": len(R["flat"]),
                "ghost_pct": 100 * mean(R["ghost"]), "flat_pct": 100 * mean(R["flat"]), "sunk_pct": 100 * mean(R["sunk"]),
                "empty_pct": 100 * mean(R["empty"]), "median_abs_height_cm": mean(R["h_abs"]),
                "sharpness_ratio_render": mean(R["sharp"]), "sharpness_ratio_gt": mean(R["sharp_gt"]),
                "ghost_alpha_mean": mean(R["g_alpha"]), "ghost_cover_pct_alpha_gt_0.1": 100 * mean(R["g_cover"]),
                "gauss_ghost_alpha_mean": mean(R["gg_alpha"]), "gauss_ghost_cover_pct_alpha_gt_0.1": 100 * mean(R["gg_cover"]),
                "ghost_anchors_gt_0.5cm": int(ghost_sel(removed, 0.5, extra).sum()),
                "ghost_anchors_gt_1.5cm": int(gsel.sum()),
                "ghost_anchors_gt_3cm": int(ghost_sel(removed, 3.0, extra).sum())}
            print(name, json.dumps(res["scenarios"][name]), flush=True)
        for i, (name, t) in enumerate(sheets):
            cv2.imwrite(str(out / f"{name}_{i:02d}.jpg"), t, [cv2.IMWRITE_JPEG_QUALITY, 85])

        # ------------------------------------------------------------ 4. chẩn đoán phần còn sót (kịch bản B: xoá hết theo ID)
        if a.probe:
            assert a.hull_dir, "--probe cần --hull_dir"
            hull_pts = {n: read_hull(Path(a.hull_dir) / f"hull_{n}.ply") for n in sorted(set(HULL_OF.values()))
                        if (Path(a.hull_dir) / f"hull_{n}.ply").exists()}
            keep_b = ~torch.isin(lab, torch.tensor(removed_b, device=lab.device))
            bins = [(-100.0, 0.3), (0.3, 1.5), (1.5, 5.0), (5.0, 1000.0)]
            bname = ["<0.3cm", "0.3-1.5cm", "1.5-5cm", ">5cm"]
            groups = ["bg0", "tab13", "other"]
            objs = {"goi_keo(180+181)": [180, 181], "dia(60)": [60], "ly_tra(70)": [70], "coc_su(80)": [80],
                    "tui_banh(50)": [50], "khan_giay(130)": [130], "banh_quy(100)": [100]}
            probe = {}
            for v in views:
                if v.image_name not in a.probe_views.split(","):
                    continue
                H, W = int(v.image_height), int(v.image_width)
                gtl = v.object_mask.cpu().numpy().astype(np.int32)
                if gtl.shape != (H, W):
                    gtl = cv2.resize(gtl.astype(np.int16), (W, H), interpolation=cv2.INTER_NEAREST)
                w2c = np.linalg.inv(v.c2w.cpu().numpy().astype(np.float64))
                cov = {}                                         # (a) hull có phủ mask vật của view này không
                for oname, ids in objs.items():
                    m = np.isin(gtl, ids)
                    hn = HULL_OF[ids[0]]
                    if m.sum() < 100 or hn not in hull_pts:
                        continue
                    Xc = hull_pts[hn] @ w2c[:3, :3].T + w2c[:3, 3]
                    okz = Xc[:, 2] > 1e-3
                    uu_ = (v.fx * Xc[okz, 0] / Xc[okz, 2] + v.cx).astype(int)
                    vv_ = (v.fy * Xc[okz, 1] / Xc[okz, 2] + v.cy).astype(int)
                    inb = (uu_ >= 0) & (uu_ < W) & (vv_ >= 0) & (vv_ < H)
                    hm = np.zeros((H, W), np.uint8); hm[vv_[inb], uu_[inb]] = 1
                    hm = cv2.dilate(hm, np.ones((9, 9), np.uint8)) > 0
                    cov[oname] = round(float((m & hm).sum() / m.sum()), 3)
                st["hull"] = False
                tiles, table = [], {}                           # (b) alpha trong mask vật theo (nhóm nhãn x dải độ cao)
                for gname in groups:
                    for (lo_, hi_), bn in zip(bins, bname):
                        st["bin"], st["grp"] = (lo_, hi_), gname
                        try:
                            pk = render(v, keep_b)
                            al = pk["render_alphas"][0].cpu().numpy()
                            tile = (torch.clamp(pk["render"], 0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        except Exception as e:                   # nhóm rỗng
                            al, tile = np.zeros((H, W)), np.zeros((H, W, 3), np.uint8)
                        for oname, ids in objs.items():
                            m = np.isin(gtl, ids)
                            if m.sum() >= 100:
                                table.setdefault(oname, {})[f"{gname}|{bn}"] = round(float(al[m].mean()), 3)
                        tile = cv2.resize(tile, (W // 2, H // 2))
                        cv2.putText(tile, f"{gname} h {bn}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        tiles.append(tile)
                st["bin"], st["grp"] = None, None
                grid = np.vstack([np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(3)])
                cv2.imwrite(str(out / f"probe_{v.image_name}.jpg"), grid[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
                probe[v.image_name] = {"hull_coverage_of_object_mask": cov, "alpha_in_object_mask": table}
                print("probe", v.image_name, json.dumps(probe[v.image_name]), flush=True)
            res["probe"] = probe
    (out / "removal_eval.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
