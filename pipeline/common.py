"""
Phần dùng chung cho các stage chạy BÊN TRONG thư mục CoMe (cwd = /opt/CoMe, cần CUDA).
Nạp cảnh giống render.py: ModelParams(sentinel) + PipelineParams + SplattingSettings(render=True) + get_combined_args.
"""
import hashlib
import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from arguments import ModelParams, PipelineParams, SplattingSettings, OptimizationParams, MeshingParams, get_combined_args
from gaussian_renderer import render, GaussianModel
from scene import Scene
from utils.general_utils import safe_state


def log(msg):
    print(f"[pipeline] {msg}", flush=True)


def make_parser(desc, with_train=False):
    p = ArgumentParser(description=desc)
    model = ModelParams(p, sentinel=True)
    pipe = PipelineParams(p)
    ss = SplattingSettings(p, render=True)
    opt = OptimizationParams(p) if with_train else None
    mp = MeshingParams(p) if with_train else None
    p.add_argument("--iteration", default=30000, type=int)
    p.add_argument("--cfg", type=str, required=True, help="config json của cả pipeline")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--quiet", action="store_true")
    return p, model, pipe, ss, opt, mp


def load_scene(args, model, pipe, ss, ply_path=None):
    """Trả về dict: g (GaussianModel, đã compute_3D_filter như render.py), scene, train, views (train+test, sắp theo tên),
    bg, pipe, splat_args, dataset. ply_path: nạp Gaussian từ file khác thay cho point_cloud/iteration_N."""
    splat_args = ss.get_settings(args)
    safe_state(args.quiet)
    dataset, pp = model.extract(args), pipe.extract(args)
    g = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, g, load_iteration=args.iteration, shuffle=False)
    if ply_path:
        g = GaussianModel(dataset.sh_degree)
        g.load_ply(str(ply_path))
    train = scene.getTrainCameras()
    with torch.no_grad():
        g.compute_3D_filter(train.copy())
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    views = sorted(train + scene.getTestCameras(), key=lambda c: c.image_name)
    return dict(g=g, scene=scene, train=train, views=views, bg=bg, pipe=pp, splat_args=splat_args, dataset=dataset)


def view_name(view):
    return Path(view.image_name).stem


def freeze(g):
    for t in (g._xyz, g._features_dc, g._features_rest, g._opacity, g._scaling, g._rotation, g._confidence):
        t.requires_grad_(False)


# ----------------------------------------------------------------------------------------------- ply
def read_ply(path):
    from plyfile import PlyData
    return PlyData.read(str(path)).elements[0].data


def write_ply(path, data):
    from plyfile import PlyData, PlyElement
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(data, "vertex")]).write(str(path))


def gaussians_from_ply(path, sh_degree, cams):
    g = GaussianModel(sh_degree)
    g.load_ply(str(path))
    with torch.no_grad():
        g.compute_3D_filter(cams)
    return g


# ----------------------------------------------------------------------------------------------- render
@torch.no_grad()
def render_rgbd(view, g, S, override_color=None):
    r = render(view, g, S["pipe"], S["bg"], splat_args=S["splat_args"], override_color=override_color)["render"]
    return r[:3].clamp(0, 1), r[6]


def to_u8(img_chw):
    return (img_chw.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


@torch.no_grad()
def opacity_field(points, views, g, S):
    """Trường opacity GOF tại điểm 3D: 1 - min_view(transmittance), như evaluate_alpha(exact_alpha_eval=True) trong
    extract_mesh_tets.py của CoMe (mặt SDF = 0 <=> giá trị 0.5)."""
    import copy
    from gaussian_renderer import integrate, GlobalSortOrder
    sa = copy.deepcopy(S["splat_args"])
    sa.sort_settings.sort_order = GlobalSortOrder.MIN_Z_BOUNDING
    sa.meshing_settings.alpha_early_stop = True
    sa.meshing_settings.return_color = False
    P = points.shape[0]
    alpha = torch.ones(P, dtype=torch.float32, device="cuda")
    final = torch.ones(P, dtype=torch.float32, device="cuda")
    ph = torch.cat([points, torch.ones(P, 1, device="cuda")], 1)
    for view in views:
        # rasterizer_impl.cu integrate(): identifyPixelRanges<<<(num_integrated+255)/256>>> lỗi khi num_integrated = 0,
        # tức không điểm nào vừa trong khung hình vừa còn alpha != 0 (điểm đã bị che hẳn ở view trước bị bỏ qua).
        # Bỏ view đó: không ảnh hưởng min theo view.
        pp = ph @ view.full_proj_transform
        z = (ph @ view.world_view_transform)[:, 2]
        ndc = pp[:, :2] / pp[:, 3:4].clamp(min=1e-8)
        if not bool(((z > 0.2) & (ndc.abs() <= 1).all(1) & (alpha != 0)).any()):
            continue
        integrate(points, alpha, view, g, S["pipe"], S["bg"], kernel_size=0.003, splat_args=sa)
        final = torch.min(final, alpha)
        alpha = torch.ones(P, dtype=torch.float32, device="cuda") * ~(alpha == 0)
    return 1 - final


# ----------------------------------------------------------------------------------------------- misc
def load_cfg(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ransac_plane(pts, iters=2000, thresh=0.01, normal_hint=None, min_cos=0.0, seed=0):
    """Mặt phẳng n.x + d = 0 khớp nhiều điểm nhất (|n| = 1). normal_hint + min_cos: chỉ nhận mặt gần song song."""
    rng = np.random.default_rng(seed)
    best, best_n = None, 0
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn_ = np.linalg.norm(n)
        if nn_ < 1e-12:
            continue
        n /= nn_
        if normal_hint is not None and abs(n @ normal_hint) < min_cos:
            continue
        d = -n @ a
        cnt = (np.abs(pts @ n + d) < thresh).sum()
        if cnt > best_n:
            best, best_n = (n, d), cnt
    n, d = best
    inl = np.abs(pts @ n + d) < thresh
    # tinh chỉnh bằng SVD trên inlier
    P = pts[inl]
    cen = P.mean(0)
    n2 = np.linalg.svd(P - cen)[2][-1]
    if n2 @ n < 0:
        n2 = -n2
    return n2, -n2 @ cen, inl
