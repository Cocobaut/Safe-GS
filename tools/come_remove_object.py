"""
Xoá Gaussian của 1 hoặc nhiều vật (theo ID mask SAM2) khỏi cảnh CoMe bằng bỏ phiếu đa view.

Chạy BÊN TRONG thư mục CoMe (cần CUDA), setup giống render.py:
    python come_remove_object.py -m <model> -s <colmap> --iteration 30000 --labels <dir labels> --ids 22 --out <dir>

Với mỗi camera (train + test, 177 view):
  - chiếu tâm Gaussian lên ảnh (world_view_transform / full_proj_transform như 3DGS, ndc2Pix = ((v+1)*S - 1)/2);
  - Gaussian "thấy được" ở view đó nếu nằm trong ảnh, trước camera, và không bị che: z <= depth_SDF * (1 + OCC_REL)
    + OCC_ABS (depth = kênh 6 của CoMe = độ sâu mặt opacity 0.5 tại pixel);
  - phiếu "thuộc vật" nếu label tại pixel thuộc --ids.
Gaussian bị xoá nếu số view thấy được >= MIN_VIEWS và tỉ lệ phiếu thuộc vật >= --ratio.
--cleanup (dọn thêm quanh vật đã xoá, theo ngưỡng 0.5 của GOF):
  - "vùng gần vật" = voxel có Gaussian của vật (cạnh voxel = trung vị trục lớn nhất của Gaussian vật x --vox_mult),
    nới thêm 1 voxel mỗi phía;
  - trong vùng đó xoá thêm Gaussian nếu tâm nằm BÊN TRONG mặt SDF (trường opacity GOF tại tâm >= 0.5, tính bằng
    gaussian_renderer.integrate qua các camera train, giống evaluate_alpha của extract_mesh_tets.py, trên cảnh đầy đủ)
    HOẶC opacity hiệu dụng của chính nó < 0.5 (floater mờ).
Xuất: <out>/scene_without.ply (cảnh đã xoá), <out>/object.ply (phần bị xoá), <out>/stats.json,
      <out>/preview/<frame>.jpg ([trước | sau] cho vài frame).
"""
import copy
import json
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch

from arguments import ModelParams, PipelineParams, SplattingSettings, get_combined_args
from gaussian_renderer import render, integrate, GaussianModel, GlobalSortOrder
from scene import Scene
from utils.general_utils import safe_state


@torch.no_grad()
def opacity_field(points, views, g, pipe, bg, splat_args):
    """Trường opacity GOF tại các điểm 3D: 1 - min_view(transmittance), như evaluate_alpha(exact_alpha_eval=True)
    và lần đánh giá đầu trong marching_tetrahedra_with_binary_search của extract_mesh_tets.py."""
    sa = copy.deepcopy(splat_args)
    sa.sort_settings.sort_order = GlobalSortOrder.MIN_Z_BOUNDING
    sa.meshing_settings.alpha_early_stop = True
    sa.meshing_settings.return_color = False
    P = points.shape[0]
    alpha = torch.ones(P, dtype=torch.float32, device="cuda")
    final = torch.ones(P, dtype=torch.float32, device="cuda")
    for view in views:
        integrate(points, alpha, view, g, pipe, bg, kernel_size=0.003, splat_args=sa)
        final = torch.min(final, alpha)
        alpha = torch.ones(P, dtype=torch.float32, device="cuda") * ~(alpha == 0)
    return 1 - final


def near_mask(xyz, obj, vox):
    """Gaussian (không thuộc vật) nằm trong voxel của vật hoặc voxel kề (26 hướng)."""
    key = lambda c: (c[:, 0] * 1_000_003 + c[:, 1]) * 1_000_033 + c[:, 2]
    base = torch.floor(xyz / vox).long()
    ov = torch.unique(base[obj], dim=0)
    offs = torch.stack(torch.meshgrid(*[torch.arange(-1, 2, device=xyz.device)] * 3, indexing="ij"), -1).reshape(-1, 3)
    dil = torch.unique(key((ov[:, None, :] + offs[None]).reshape(-1, 3)))
    return ~obj & torch.isin(key(base), dil)

OCC_REL, OCC_ABS = 0.03, 0.01
MIN_VIEWS = 3
PREVIEW = ["frame_00006", "frame_00041", "frame_00085", "frame_00108"]


def main():
    parser = ArgumentParser(description="remove object by mask voting")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    ss = SplattingSettings(parser, render=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--ids", type=int, nargs="+", required=True, help="-1 = mọi ID (mọi pixel label > 0)")
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--cleanup", action="store_true", help="dọn Gaussian bên trong mặt SDF + floater quanh vật")
    parser.add_argument("--vox_mult", type=float, default=1.0)
    parser.add_argument("--free_radius", type=float, default=0.0,
                        help="> 0: trong vùng cách vật đã xoá ~ bán kính này, xoá thêm Gaussian rơi vào mask vật ở >= --ratio "
                             "số view trong khung hình, KHÔNG kiểm tra che (bắt lớp lông sâu / mặt sau)")
    parser.add_argument("--diag_ids", type=int, nargs="*", default=[],
                        help="lưu <out>/diag.npz: số liệu từng Gaussian + phiếu riêng cho các ID này (vd 2 6 9 = gấu)")
    args = get_combined_args(parser)
    splat_args = ss.get_settings(args)
    safe_state(args.quiet)
    out = Path(args.out); (out / "preview").mkdir(parents=True, exist_ok=True)
    ids = torch.tensor(args.ids, device="cuda")

    dataset, pipe = model.extract(args), pipeline.extract(args)
    with torch.no_grad():
        g = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, g, load_iteration=args.iteration, shuffle=False)
        train = scene.getTrainCameras()
        g.compute_3D_filter(train.copy())                          # như render.py
        bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
        views = sorted(train + scene.getTestCameras(), key=lambda c: c.image_name)

        xyz = g.get_xyz
        N = xyz.shape[0]
        ph = torch.cat([xyz, torch.ones(N, 1, device="cuda")], 1)
        vis = torch.zeros(N, device="cuda")
        hit = torch.zeros(N, device="cuda")
        inframe = torch.zeros(N, device="cuda")       # nằm trong ảnh, trước camera
        hit_free = torch.zeros(N, device="cuda")      # phiếu thuộc vật ở mọi view trong khung hình (bỏ qua che)
        hit_diag = torch.zeros(N, device="cuda")      # phiếu cho --diag_ids (chỉ ở view thấy được)
        hit_diag_any = torch.zeros(N, device="cuda")  # phiếu cho --diag_ids kể cả view bị coi là che
        diag_ids = torch.tensor(args.diag_ids or [0], device="cuda")
        n_lab = 0
        for k, view in enumerate(views, 1):
            name = Path(view.image_name).stem
            lp = Path(args.labels) / f"{name}.png"
            if not lp.exists():
                continue
            lab = torch.from_numpy(cv2.imread(str(lp), cv2.IMREAD_UNCHANGED).astype(np.int64)).cuda()
            H, W = int(view.image_height), int(view.image_width)
            assert lab.shape == (H, W), f"{lp}: {tuple(lab.shape)} khác ảnh {H}x{W}"
            n_lab += 1
            depth = render(view, g, pipe, bg, splat_args=splat_args)["render"][6]
            z = (ph @ view.world_view_transform)[:, 2]
            pp = ph @ view.full_proj_transform
            ndc = pp[:, :2] / pp[:, 3:4].clamp(min=1e-8)
            px = (((ndc[:, 0] + 1) * W - 1) * 0.5).round().long()
            py = (((ndc[:, 1] + 1) * H - 1) * 0.5).round().long()
            inside = (z > 0.2) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
            idx = inside.nonzero(as_tuple=True)[0]
            inframe[idx] += 1
            lv_all = lab[py[idx], px[idx]]
            hit_free[idx] += ((lv_all > 0) if -1 in args.ids else torch.isin(lv_all, ids)).float()
            if args.diag_ids:
                hit_diag_any[idx] += torch.isin(lab[py[idx], px[idx]], diag_ids).float()
            d = depth[py[idx], px[idx]]
            ok = z[idx] <= d * (1 + OCC_REL) + OCC_ABS
            idx = idx[ok]
            vis[idx] += 1
            lv = lab[py[idx], px[idx]]
            hit[idx] += ((lv > 0) if -1 in args.ids else torch.isin(lv, ids)).float()
            if args.diag_ids:
                hit_diag[idx] += torch.isin(lv, diag_ids).float()
            if k % 30 == 0:
                print(f"[remove] {k}/{len(views)}", flush=True)

        ratio = hit / vis.clamp(min=1)
        obj = (vis >= MIN_VIEWS) & (ratio >= args.ratio)
        n_vote = int(obj.sum())
        print(f"[remove] {n_lab} view có label | Gaussian: {N:,} | xoá theo phiếu (ID {args.ids}): {n_vote:,} "
              f"| không view nào thấy: {int((vis == 0).sum()):,}", flush=True)

        clean = {}
        if args.cleanup and n_vote > 0:
            vox = float(g.get_scaling[obj].max(1).values.median()) * args.vox_mult
            near = near_mask(xyz, obj, vox)
            idx = near.nonzero(as_tuple=True)[0]
            field = torch.zeros(N, device="cuda")
            for s in range(0, len(idx), 2_000_000):
                part = idx[s:s + 2_000_000]
                field[part] = opacity_field(xyz[part].contiguous(), sorted(train, key=lambda c: c.image_name),
                                            g, pipe, bg, splat_args)
            opa = g.get_opacity_with_3D_filter.squeeze(-1)
            inside = near & (field >= 0.5)
            faint = near & ~inside & (opa < 0.5)
            obj = obj | inside | faint
            clean = {"voxel": vox, "near": int(near.sum()), "inside_sdf": int(inside.sum()),
                     "faint_floater": int(faint.sum()),
                     "inside_and_never_visible": int((inside & (vis == 0)).sum())}
            print(f"[remove] cleanup: voxel {vox:.4f} | gần vật {clean['near']:,} | bên trong SDF {clean['inside_sdf']:,} "
                  f"(trong đó không view nào thấy {clean['inside_and_never_visible']:,}) | floater < 0.5 "
                  f"{clean['faint_floater']:,}", flush=True)
        if args.free_radius > 0 and n_vote > 0:
            vote_obj = (vis >= MIN_VIEWS) & (ratio >= args.ratio)          # chỉ lấy vật xác định bằng phiếu làm mốc
            near_f = near_mask(xyz, vote_obj, args.free_radius) & ~obj
            free = near_f & (inframe >= MIN_VIEWS) & (hit_free / inframe.clamp(min=1) >= args.ratio)
            obj = obj | free
            clean["free_vote"] = {"radius": args.free_radius, "near": int(near_f.sum()), "removed": int(free.sum())}
            print(f"[remove] free vote: bán kính {args.free_radius} | gần vật {int(near_f.sum()):,} | xoá thêm "
                  f"{int(free.sum()):,}", flush=True)
        keep = ~obj
        print(f"[remove] tổng xoá {int(obj.sum()):,} | giữ {int(keep.sum()):,}", flush=True)

        if args.diag_ids:
            if not (args.cleanup and n_vote > 0):
                field = torch.full((N,), -1.0, device="cuda")
            c = lambda t: t.detach().cpu().numpy()
            np.savez_compressed(
                out / "diag.npz", xyz=c(xyz), vis=c(vis), hit=c(hit), inframe=c(inframe), hit_diag=c(hit_diag),
                hit_diag_any=c(hit_diag_any), removed=c(obj), field=c(field),
                opacity=c(g.get_opacity_with_3D_filter.squeeze(-1)), scale=c(g.get_scaling), diag_ids=np.array(args.diag_ids))
            print(f"[remove] lưu {out / 'diag.npz'}", flush=True)

        # trước khi prune: render preview "trước"
        before = {}
        for view in views:
            nm = Path(view.image_name).stem
            if nm in PREVIEW:
                before[nm] = render(view, g, pipe, bg, splat_args=splat_args)["render"][:3]

        full = out / "_full.ply"
        g.save_ply(str(full))
        from plyfile import PlyData, PlyElement
        el = PlyData.read(str(full)).elements[0].data
        keep_np, obj_np = keep.cpu().numpy(), obj.cpu().numpy()
        PlyData([PlyElement.describe(el[keep_np], "vertex")]).write(str(out / "scene_without.ply"))
        PlyData([PlyElement.describe(el[obj_np], "vertex")]).write(str(out / "object.ply"))
        full.unlink()

        # render "sau" bằng chính file scene_without.ply
        g2 = GaussianModel(dataset.sh_degree)
        g2.load_ply(str(out / "scene_without.ply"))
        for view in views:
            nm = Path(view.image_name).stem
            if nm not in before:
                continue
            after = render(view, g2, pipe, bg, splat_args=splat_args)["render"][:3]
            pair = torch.cat([before[nm], torch.ones(3, before[nm].shape[1], 8, device="cuda"), after], 2)
            img = (pair.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)[..., ::-1]
            cv2.imwrite(str(out / "preview" / f"{nm}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

        stats = {"ids": args.ids, "ratio": args.ratio, "min_views": MIN_VIEWS, "occ_rel": OCC_REL, "occ_abs": OCC_ABS,
                 "views_with_labels": n_lab, "gaussians": N, "removed_by_vote": n_vote, "removed": int(obj.sum()),
                 "kept": int(keep.sum()), "never_visible": int((vis == 0).sum()), "cleanup": clean}
        (out / "stats.json").write_text(json.dumps(stats, indent=2))
        print(f"[remove] xong -> {out}", flush=True)


if __name__ == "__main__":
    main()
