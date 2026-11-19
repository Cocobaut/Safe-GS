"""
Inspect Joint CoMe + Safe-GS object lift & ghost shell check
=============================================================
Nhấc từng vật (ví dụ gau, cuu) lên +25 cm hoặc ẩn hẳn để kiểm tra:
1. Ảnh render trực quan: [Gốc | Nhấc vật +25 cm | Ẩn vật]
2. Đo alpha "vỏ ma" (ghost shell) còn sót lại trong mask SAM 3 của vật khi vật đã bị nhấc/ẩn.
"""
import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData
from scipy.ndimage import distance_transform_edt

from arguments import ModelParams, PipelineParams, SplattingSettings
from gaussian_renderer import render
from scene import GaussianModel, Scene


class HullDist:
    def __init__(self, ply_path: Path, pad_vox: int = 8, max_dim: int = 128):
        self.valid = False
        if not ply_path.exists():
            return
        v = PlyData.read(str(ply_path))["vertex"]
        pts = np.stack([np.asarray(v["x"], np.float32),
                        np.asarray(v["y"], np.float32),
                        np.asarray(v["z"], np.float32)], 1)
        if len(pts) < 4:
            return
        sub = pts[::max(1, len(pts) // 1000)][:1000]
        dmat = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=-1)
        dmat[dmat == 0] = 1e9
        vs = float(np.median(dmat.min(1)))
        if vs <= 0 or not np.isfinite(vs):
            vs = float((pts.max(0) - pts.min(0)).max()) / 64.0
        lo = pts.min(0) - pad_vox * vs
        hi = pts.max(0) + pad_vox * vs
        span = np.maximum(hi - lo, vs * 4)
        dims = np.ceil(span / vs).astype(int) + 1
        if dims.max() > max_dim:
            vs = vs * (float(dims.max()) / max_dim)
            dims = np.ceil(span / vs).astype(int) + 1
        nx, ny, nz = [int(d) for d in dims]
        ijk = np.floor((pts - lo) / vs + 0.5).astype(int)
        ijk[:, 0] = np.clip(ijk[:, 0], 0, nx - 1)
        ijk[:, 1] = np.clip(ijk[:, 1], 0, ny - 1)
        ijk[:, 2] = np.clip(ijk[:, 2], 0, nz - 1)
        occ = np.zeros((nx, ny, nz), dtype=bool)
        occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
        edt = distance_transform_edt(~occ).astype(np.float32) * vs
        self.grid = torch.from_numpy(edt.transpose(2, 1, 0).copy())[None, None].cuda()
        self.lo = torch.tensor(lo, dtype=torch.float32, device="cuda")
        self.hi = torch.tensor(lo + (np.array([nx, ny, nz], np.float32) - 1.0) * vs,
                               dtype=torch.float32, device="cuda")
        self.valid = True

    def __call__(self, xyz: torch.Tensor) -> torch.Tensor:
        if not self.valid or len(xyz) == 0:
            return torch.full((len(xyz),), 1e6, dtype=torch.float32, device=xyz.device)
        clamped = torch.clamp(xyz, self.lo, self.hi)
        out_box = torch.norm(xyz - clamped, dim=-1)
        norm = 2.0 * (clamped - self.lo) / (self.hi - self.lo).clamp(min=1e-6) - 1.0
        d = F.grid_sample(self.grid, norm.view(1, 1, 1, -1, 3), mode="bilinear",
                          padding_mode="border", align_corners=True)
        return d.view(-1) + out_box


def main():
    ap = ArgumentParser()
    lp = ModelParams(ap)
    pp = PipelineParams(ap)
    ss = SplattingSettings(ap)
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--oid_npy", type=str, required=True)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--labels", type=str, required=True)
    ap.add_argument("--support_json", type=str, default="")
    ap.add_argument("--targets", nargs="+", default=["gau", "cuu"])
    ap.add_argument("--lift_cm", type=float, default=25.0)
    ap.add_argument("--hull_dir", type=str, default="")
    ap.add_argument("--hull_margin_cm", type=float, default=2.0)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args(sys.argv[1:])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    obj_names = list(cfg["S3_label"]["objects"].keys())
    obj_ids = cfg["S3_label"]["objects"]
    mpu = float(cfg.get("S6_mesh", {}).get("meters_per_unit", 0.247))
    cm = mpu * 100.0

    # Up-normal and plane offset from support.json
    n_up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    d_table = None
    if args.support_json and Path(args.support_json).exists():
        sup = json.loads(Path(args.support_json).read_text(encoding="utf-8"))
        if "table" in sup and "normal" in sup["table"]:
            n_up = np.array(sup["table"]["normal"], dtype=np.float32)
            n_up = n_up / max(np.linalg.norm(n_up), 1e-8)
            if "d" in sup["table"]:
                d_table = float(sup["table"]["d"])

    dataset = lp.extract(args)
    pipe = pp.extract(args)
    splat_args = ss.get_settings(args)
    gaussians = GaussianModel(dataset.sh_degree, use_SBs=pipe.convert_SBs_python)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    cams = scene.getTrainCameras()
    gaussians.compute_3D_filter(cams.copy())
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    black = torch.zeros(3, dtype=torch.float32, device="cuda")

    oid = np.load(args.oid_npy)
    assert len(oid) == len(gaussians._xyz), f"oid len {len(oid)} != xyz len {len(gaussians._xyz)}"
    oid_t = torch.from_numpy(oid).cuda()

    xyz_orig = gaussians._xyz.data.clone()
    op_orig = gaussians._opacity.data.clone()
    labels_dir = Path(args.labels)

    # support.json["table"]["normal"] is already oriented toward foreground objects by s3_label.py
    shift_vec = torch.from_numpy(n_up * (args.lift_cm / 100.0 / mpu)).cuda()
    results = {"lift_cm": args.lift_cm, "mpu": mpu, "n_up": n_up.tolist(), "targets": {}}

    with torch.no_grad():
        N = len(xyz_orig)
        ph = torch.cat([xyz_orig, torch.ones(N, 1, device="cuda")], 1)
        h_cm = None
        if d_table is not None:
            n_t = torch.from_numpy(n_up).cuda()
            h_cm = (xyz_orig @ n_t + d_table) * cm
        opa_eff = gaussians.get_opacity_with_3D_filter.squeeze(-1)

        def near_vox_mask(pts, base_mask, vox_step):
            key = lambda c: (c[:, 0] * 1_000_003 + c[:, 1]) * 1_000_033 + c[:, 2]
            b = torch.floor(pts / vox_step).long()
            ov = torch.unique(b[base_mask], dim=0)
            offs = torch.stack(torch.meshgrid(*[torch.arange(-1, 2, device=pts.device)] * 3, indexing="ij"), -1).reshape(-1, 3)
            dil = torch.unique(key((ov[:, None, :] + offs[None]).reshape(-1, 3)))
            return ~base_mask & torch.isin(key(b), dil)

        for name in args.targets:
            if name not in obj_names:
                continue
            k = obj_names.index(name)
            sel = oid_t == k
            gids_t = torch.tensor(obj_ids[name], dtype=torch.long, device="cuda")

            # Candidate non-other-object mask (allow unlabeled -1 and support >= len(obj_names), protect tabletop slab)
            cand = (oid_t < 0) | (oid_t >= len(obj_names))
            if h_cm is not None:
                cand = cand & ((h_cm > 1.2) | (h_cm < -1.5))

            # 1) Unoccluded multi-view silhouette projection (free_vote from come_remove_object.py: catches deep fur/inner layers/legs)
            inframe = torch.zeros(N, device="cuda")
            hit_free = torch.zeros(N, device="cuda")
            vis_cams = []
            for c in cams:
                lpth = labels_dir / f"{Path(c.image_name).stem}.png"
                if not lpth.exists():
                    continue
                lab_v = torch.from_numpy(cv2.imread(str(lpth), cv2.IMREAD_UNCHANGED).astype(np.int64)).cuda()
                m = torch.isin(lab_v, gids_t)
                if int(m.sum()) >= 1500:
                    vis_cams.append((c, m))
                H, W = int(c.image_height), int(c.image_width)
                z = (ph @ c.world_view_transform)[:, 2]
                pp_c = ph @ c.full_proj_transform
                ndc = pp_c[:, :2] / pp_c[:, 3:4].clamp(min=1e-8)
                px = (((ndc[:, 0] + 1) * W - 1) * 0.5).round().long()
                py = (((ndc[:, 1] + 1) * H - 1) * 0.5).round().long()
                inside = (z > 0.2) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
                idx_in = inside.nonzero(as_tuple=True)[0]
                inframe[idx_in] += 1
                hit_free[idx_in] += torch.isin(lab_v[py[idx_in], px[idx_in]], gids_t).float()

            # Free-vote interior/back/leg Gaussians within 12 cm of core object that project into target mask >= 35% of views
            rad_12cm = 0.12 / mpu
            near_free = near_vox_mask(xyz_orig, sel, rad_12cm) & cand
            free_hit = near_free & (inframe >= 3) & ((hit_free / inframe.clamp(min=1)) >= 0.35)
            sel = sel | free_hit

            # Faint boundary floaters (opa < 0.5) in immediate 2-voxel neighborhood of object that project into target mask >= 15% of views
            vox_med = float(gaussians.get_scaling[sel].max(1).values.median()) * 2.0
            near_faint = near_vox_mask(xyz_orig, sel, max(vox_med, 0.015 / mpu)) & cand
            faint_hit = near_faint & (opa_eff < 0.5) & ((hit_free / inframe.clamp(min=1)) >= 0.15)
            sel = sel | faint_hit

            # 2) Tight 3D surface shell (<= 2.0 cm from core object, above tabletop > 1.5 cm, never touching tabletop objects)
            #    Catches unsegmented attached accessories (e.g. cuu's blue lanyard labeled bg/ban/goi_keo) without grabbing table or nearby dishes.
            protected_fg = torch.zeros(N, dtype=torch.bool, device="cuda")
            for o_idx, o_nm in enumerate(obj_names):
                if o_idx != k and o_nm not in ("goi_keo",):
                    protected_fg |= (oid_t == o_idx)
            # Protect goi_keo on the table (h_cm <= 5 cm) while allowing blue-lanyard Gaussians high on cuu's neck (h_cm > 8 cm)
            if "goi_keo" in obj_names and h_cm is not None:
                gk_idx = obj_names.index("goi_keo")
                protected_fg |= ((oid_t == gk_idx) & (h_cm <= 8.0))

            near_2cm = near_vox_mask(xyz_orig, oid_t == k, 0.020 / mpu) & (~protected_fg)
            if h_cm is not None:
                above_table_shell = near_2cm & (h_cm > 1.5)
                below_table_leg = near_vox_mask(xyz_orig, oid_t == k, 0.060 / mpu) & (~protected_fg) & (h_cm < -2.0)
            else:
                above_table_shell = near_2cm
                below_table_leg = torch.zeros(N, dtype=torch.bool, device="cuda")

            # Use 1-pass rasterizer check on above_table_shell | below_table_leg so only Gaussians actually projecting into target mask are grouped
            gaussians._xyz.data.copy_(xyz_orig)
            gaussians._opacity.data.copy_(op_orig)
            gaussians._opacity.data[sel] = -20.0
            w_in = torch.zeros(N, device="cuda")
            w_out = torch.zeros(N, device="cuda")
            dummy = torch.zeros((N, 3), dtype=torch.float32, device="cuda", requires_grad=True)
            peel_cams = vis_cams[::max(1, len(vis_cams) // 16)][:16]
            with torch.enable_grad():
                for c_p, m_p in peel_cams:
                    mf = m_p.float()
                    dli = F.max_pool2d(mf[None, None], 21, 1, 10)[0, 0] > 0.5
                    chans = [dli.float(), 1.0 - dli.float()]
                    img_p = render(c_p, gaussians, pipe, black, splat_args=splat_args, override_color=dummy)["render"][:2]
                    loss_p = (img_p[0] * chans[0] + img_p[1] * chans[1]).sum()
                    if dummy.grad is not None:
                        dummy.grad.zero_()
                    loss_p.backward()
                    w_in += dummy.grad[:, 0].detach()
                    w_out += dummy.grad[:, 1].detach()
            in_share = w_in / (w_in + w_out).clamp(min=1e-12)
            peel_hit = (~sel) & (above_table_shell | below_table_leg) & ((w_in + w_out) > 1e-4) & (in_share >= 0.40)
            sel = sel | peel_hit

            print(f"[{name}] core={int((oid_t == k).sum())}, +free_vote={int(free_hit.sum())}, +faint={int(faint_hit.sum())}, +peel={int(peel_hit.sum())} -> total={int(sel.sum())}")
            step = max(1, len(vis_cams) // 4)
            chosen = vis_cams[::step][:4]

            # Measure ghost alpha across up to 20 visible cameras when target is hidden
            eval_step = max(1, len(vis_cams) // 20)
            eval_cams = vis_cams[::eval_step][:20]
            ghost_eroded_vals = []
            ghost_full_vals = []

            # Hide target Gaussians (opacity logit = -20 -> sigmoid ~ 0)
            gaussians._xyz.data.copy_(xyz_orig)
            gaussians._opacity.data.copy_(op_orig)
            gaussians._opacity.data[sel] = -20.0
            # Render remaining foreground+background opacity inside target mask
            fg_only = torch.zeros((len(oid_t), 3), device="cuda")
            fg_only[~sel] = 1.0
            for c, m in eval_cams:
                mf = m.float()
                er = (1.0 - F.max_pool2d((1.0 - mf)[None, None], 11, 1, 5)[0, 0]) > 0.5
                # Render scene without target
                rem_alpha = render(c, gaussians, pipe, black, splat_args=splat_args, override_color=fg_only)["render"][0]
                ghost_full_vals.append(float(rem_alpha[m].mean()))
                if er.sum() > 50:
                    ghost_eroded_vals.append(float(rem_alpha[er].mean()))

            # Render visual comparison [orig | lifted +25cm | removed] for chosen 4 views
            t_dir = out / name
            t_dir.mkdir(parents=True, exist_ok=True)
            for idx_v, (c, m) in enumerate(chosen):
                # 1. Orig
                gaussians._xyz.data.copy_(xyz_orig)
                gaussians._opacity.data.copy_(op_orig)
                img_orig = torch.clamp(render(c, gaussians, pipe, bg, splat_args=splat_args)["render"][:3], 0, 1)

                # 2. Lifted +25cm
                gaussians._xyz.data[sel] += shift_vec
                img_lift = torch.clamp(render(c, gaussians, pipe, bg, splat_args=splat_args)["render"][:3], 0, 1)

                # 3. Removed
                gaussians._xyz.data.copy_(xyz_orig)
                gaussians._opacity.data[sel] = -20.0
                img_rem = torch.clamp(render(c, gaussians, pipe, bg, splat_args=splat_args)["render"][:3], 0, 1)

                panel = torch.cat([img_orig, img_lift, img_rem], dim=2)
                arr = (panel.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]
                cv2.imwrite(str(t_dir / f"view_{idx_v}_{Path(c.image_name).stem}.jpg"), arr)

            results["targets"][name] = {
                "gaussians": int(sel.sum()),
                "eval_views": len(eval_cams),
                "remaining_alpha_full_mask_mean": float(np.mean(ghost_full_vals)) if ghost_full_vals else None,
                "remaining_alpha_eroded_mask_mean": float(np.mean(ghost_eroded_vals)) if ghost_eroded_vals else None,
            }
            print(f"[{name}] gaussians={int(sel.sum())}, ghost_alpha_eroded={results['targets'][name]['remaining_alpha_eroded_mask_mean']:.4f}, full={results['targets'][name]['remaining_alpha_full_mask_mean']:.4f}")

    (out / "lift_ghost_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
