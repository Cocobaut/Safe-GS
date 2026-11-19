"""
Joint CoMe + Safe-GS Training (Priority 2)
========================================
Huấn luyện CoMe 3DGS đồng thời với ràng buộc tách vật & khuôn 3D của Safe-GS:
- Giai đoạn 1 (1 .. safegs_from_iter - 1): CoMe chạy chuẩn để dựng hình học thô.
- Giai đoạn 2 (safegs_from_iter .. iterations): kích hoạt gán nhãn đa góc nhìn (fg_present),
  ép Gaussian về trong Visual Hull (L_bound), chống tràn alpha ra ngoài mask (L_leak)
  và giữ độ phủ trong mask (L_cover) ngay trong quá trình CoMe densify/prune/optimize.

Chạy trong /opt/CoMe (cùng môi trường CoMe trên Modal).
"""
import copy
import json
import os
import random
import sys
import time
import warnings
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path
from random import randint

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

from arguments import (MeshingParams, ModelParams, OptimizationParams,
                       PipelineParams, SplattingSettings)
from diff_gaussian_rasterization import ExtendedSettings
from gaussian_renderer import render
from scene import GaussianModel, Scene
from scene.appearance_network import (AppearanceEmbedding,
                                      SSIMDecoupledAppearanceEmbedding,
                                      VastGaussianAppearanceEmbedding)
from scene.densifier import (AbsGradDensifier, MCMCDensifier,
                             MSv2AbsGradDensifier)
from utils.depth_utils import central_diff, depth_to_normal
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim


class HullDist:
    """3D Euclidean distance field outside a voxel hull point cloud (differentiable via grid_sample)."""

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
            scale = float(dims.max()) / max_dim
            vs = vs * scale
            dims = np.ceil(span / vs).astype(int) + 1
        nx, ny, nz = [int(d) for d in dims]
        ijk = np.floor((pts - lo) / vs + 0.5).astype(int)
        ijk[:, 0] = np.clip(ijk[:, 0], 0, nx - 1)
        ijk[:, 1] = np.clip(ijk[:, 1], 0, ny - 1)
        ijk[:, 2] = np.clip(ijk[:, 2], 0, nz - 1)
        occ = np.zeros((nx, ny, nz), dtype=bool)
        occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
        edt = distance_transform_edt(~occ).astype(np.float32) * vs
        # grid_sample expects (1, 1, D=Z, H=Y, W=X)
        grid = torch.from_numpy(edt.transpose(2, 1, 0).copy())[None, None].cuda()
        self.grid = grid
        self.lo = torch.tensor(lo, dtype=torch.float32, device="cuda")
        self.hi = torch.tensor(lo + (np.array([nx, ny, nz], np.float32) - 1.0) * vs,
                               dtype=torch.float32, device="cuda")
        self.valid = True

    def __call__(self, xyz: torch.Tensor) -> torch.Tensor:
        if not self.valid or len(xyz) == 0:
            return torch.zeros(len(xyz), dtype=torch.float32, device=xyz.device)
        clamped = torch.clamp(xyz, self.lo, self.hi)
        out_box = torch.norm(xyz - clamped, dim=-1)
        norm = 2.0 * (clamped - self.lo) / (self.hi - self.lo).clamp(min=1e-6) - 1.0
        # grid_sample coords: (x, y, z) in [-1, 1]
        g = norm.view(1, 1, 1, -1, 3)
        d = F.grid_sample(self.grid, g, mode="bilinear", padding_mode="border", align_corners=True)
        return d.view(-1) + out_box


def view_stem(view) -> str:
    return Path(view.image_name).stem


@torch.no_grad()
def _zero_grad_params(gaussians):
    if gaussians.optimizer is not None:
        gaussians.optimizer.zero_grad(set_to_none=True)


def vote_assign_fg_present(
    gaussians: GaussianModel,
    train_cameras: list,
    pipe: PipelineParams,
    bg: torch.Tensor,
    splat_args: ExtendedSettings,
    lab_cache: dict,
    gid_t: list[torch.Tensor],
    n_obj: int,
    assign_ratio: float = 0.5,
    min_total_weight: float = 1e-4,
    fg_min_obj_share: float = 0.05,
    stride: int = 4,
) -> np.ndarray:
    """Fast gradient-contribution multi-view voting with fg_present normalization."""
    N = len(gaussians.get_xyz)
    K_ = len(gid_t)
    C = 2 * K_ + 1
    W = torch.zeros((N, C), dtype=torch.float32, device="cuda")
    dummy = torch.zeros((N, 3), dtype=torch.float32, device="cuda", requires_grad=True)
    black = torch.zeros(3, dtype=torch.float32, device="cuda")
    views = train_cameras[::max(1, stride)]

    # temporarily enable grad only on dummy, freeze gaussians params
    req_states = []
    for p in [gaussians._xyz, gaussians._features_dc, gaussians._features_rest,
              gaussians._opacity, gaussians._scaling, gaussians._rotation]:
        req_states.append(p.requires_grad)
        p.requires_grad_(False)

    try:
        with torch.enable_grad():
            for v in views:
                nm = view_stem(v)
                lv = lab_cache.get(nm)
                if lv is None:
                    continue
                lvg = lv.cuda().long()
                ms = [torch.isin(lvg, t).float() for t in gid_t]
                ps = [torch.ones_like(m) if float(m.sum()) >= 20 else torch.zeros_like(m) for m in ms]
                chans = ms + ps + [torch.ones_like(lvg, dtype=torch.float32)]
                for b in range(0, C, 3):
                    e = min(b + 3, C)
                    img = render(v, gaussians, pipe, black, splat_args=splat_args,
                                 override_color=dummy)["render"][:3]
                    loss = sum((img[j] * chans[b + j]).sum() for j in range(e - b))
                    if dummy.grad is not None:
                        dummy.grad.zero_()
                    loss.backward()
                    W[:, b:e] += dummy.grad[:, :e - b]
    finally:
        for p, st in zip([gaussians._xyz, gaussians._features_dc, gaussians._features_rest,
                          gaussians._opacity, gaussians._scaling, gaussians._rotation], req_states):
            p.requires_grad_(st)
        _zero_grad_params(gaussians)

    tot = W[:, -1]
    Wp = W[:, K_:2 * K_]
    fg_share = W[:, :n_obj].sum(1) / tot.clamp(min=1e-12)
    obj_share = W[:, :n_obj] / tot[:, None].clamp(min=1e-12)
    ratio_fg = torch.where(
        (fg_share[:, None] >= assign_ratio)
        & (Wp[:, :n_obj] >= 0.2 * tot[:, None])
        & (obj_share >= fg_min_obj_share),
        W[:, :n_obj] / Wp[:, :n_obj].clamp(min=1e-12),
        torch.zeros_like(Wp[:, :n_obj]),
    )
    if K_ > n_obj:
        ratio_sup = W[:, n_obj:K_] / tot[:, None].clamp(min=1e-12)
        ratio_t = torch.cat([ratio_fg, ratio_sup], 1)
    else:
        ratio_t = ratio_fg

    ratio = ratio_t.cpu().numpy()
    tot_np = tot.cpu().numpy()
    best = ratio.argmax(1)
    lab = np.where((ratio.max(1) >= assign_ratio) & (tot_np > min_total_weight), best, -1).astype(np.int16)
    fg_max_share = obj_share.max(1).values.detach()
    sup_share = (W[:, n_obj:K_].sum(1) / tot.clamp(min=1e-12)).detach() if K_ > n_obj else torch.zeros_like(tot)
    return lab, fg_max_share, sup_share


def training(
    dataset,
    opt,
    pipe: PipelineParams,
    mesh: MeshingParams,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    splat_args: ExtendedSettings,
    safegs_args,
):
    start_event = time.time()
    first_iter = 0

    os.makedirs(dataset.model_path, exist_ok=True)
    with open(os.path.join(dataset.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(dataset))))

    gaussians = GaussianModel(dataset.sh_degree, use_SBs=pipe.convert_SBs_python)
    scene = Scene(dataset, gaussians, MCMC_init=mesh.cap_max != -1)
    trainCameras = scene.getTrainCameras().copy()

    if mesh.use_vastgaussian_appearance:
        appearance_embedding = VastGaussianAppearanceEmbedding(num_views=len(trainCameras), lambda_ssim=opt.lambda_dssim)
    elif mesh.use_ssimdecoupled_appearance:
        appearance_embedding = SSIMDecoupledAppearanceEmbedding(num_views=len(trainCameras), lambda_ssim=opt.lambda_dssim)
    else:
        warnings.warn("Unknown appearance embedding, using default (No Appearance Embedding)")
        appearance_embedding = AppearanceEmbedding(num_views=len(trainCameras), lambda_ssim=opt.lambda_dssim)
    gaussians.training_setup(opt, mesh, appearance_embedding)
    if checkpoint:
        loaded = torch.load(checkpoint)
        model_params, first_iter = loaded[0], loaded[1]
        app_state = loaded[2]
        if isinstance(app_state, tuple):
            appearance_embedding.restore(*app_state)
        else:
            appearance_embedding.restore(app_state)
        gaussians.restore(model_params, opt, mesh, appearance_embedding)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    black = torch.zeros(3, dtype=torch.float32, device="cuda")

    if mesh.use_msv2_simplification:
        densifier = MSv2AbsGradDensifier(gaussians, opt, mesh, dataset, pipe)
    elif mesh.cap_max == -1:
        densifier = AbsGradDensifier(gaussians, opt, mesh, dataset, pipe)
    else:
        densifier = MCMCDensifier(gaussians, opt, mesh, dataset, pipe)

    for idx, camera in enumerate(scene.getTrainCameras() + scene.getTestCameras()):
        camera.idx = idx

    # Load Safe-GS config, hulls, and 2D masks
    cfg = json.loads(Path(safegs_args.safegs_config).read_text(encoding="utf-8"))
    s3_cfg = cfg["S3_label"]
    objects = {k: list(v) for k, v in s3_cfg["objects"].items()}
    support = {k: list(v) for k, v in s3_cfg["support"].items()}
    names = list(objects.keys())
    n_obj = len(names)
    all_groups = list(objects.values()) + list(support.values())
    gid_t = [torch.tensor(gids, dtype=torch.long, device="cuda") for gids in all_groups]

    mpu = safegs_args.safegs_mpu
    if mpu <= 0:
        mpu = float(cfg.get("S6_mesh", {}).get("meters_per_unit", 0.247))
    cm = mpu * 100.0

    hull_dir = Path(safegs_args.safegs_hull_dir)
    fds = [HullDist(hull_dir / f"hull_{n}.ply") for n in names]
    print(f"[Safe-GS Joint] Loaded {sum(f.valid for f in fds)}/{n_obj} object hulls, mpu={mpu:.4f}")

    labels_dir = Path(safegs_args.safegs_labels)
    lab_cache = {}
    for v in trainCameras:
        nm = view_stem(v)
        pth = labels_dir / f"{nm}.png"
        if pth.exists():
            lab_cache[nm] = torch.from_numpy(cv2.imread(str(pth), cv2.IMREAD_UNCHANGED).astype(np.int16))
    print(f"[Safe-GS Joint] Cached {len(lab_cache)}/{len(trainCameras)} train label maps")

    splat_args.render_opacity = False
    gaussians.compute_3D_filter(cameras=trainCameras, CUDA=not pipe.compute_filter3D_python)
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Joint Training")
    first_iter += 1

    oid_t = None
    reliable = np.zeros(n_obj, dtype=bool)
    cons_mask = None
    ghost_mask = None
    cons_ids = []
    rng = random.Random(42)

    def refresh_labels(it_num: int, stride: int = 4):
        nonlocal oid_t, reliable, cons_mask, ghost_mask, cons_ids
        lab, fg_max_share, sup_share = vote_assign_fg_present(
            gaussians, trainCameras, pipe, background, splat_args,
            lab_cache, gid_t, n_obj,
            assign_ratio=float(s3_cfg.get("assign_ratio", 0.5)),
            min_total_weight=float(s3_cfg.get("min_total_weight", 1e-4)),
            stride=stride,
        )
        oid_t = torch.from_numpy(lab).cuda()
        with torch.no_grad():
            xyz = gaussians.get_xyz
            reliable = np.zeros(n_obj, dtype=bool)
            out_pcts = []
            for k in range(n_obj):
                if not fds[k].valid:
                    continue
                mk = oid_t == k
                if not mk.any():
                    continue
                dk = fds[k](xyz[mk]) * cm
                pct = 100.0 * float((dk > 0.5).float().mean())
                reliable[k] = pct <= safegs_args.safegs_max_out_pct
                out_pcts.append(pct)
            rel_t = torch.tensor(reliable, device="cuda")
            fg_valid = (oid_t >= 0) & (oid_t < n_obj)
            cons_mask = fg_valid & rel_t[oid_t.clamp(0, max(0, n_obj - 1)).long()]
            cons_ids = [k for k in range(n_obj) if reliable[k]]
            # Boundary ghost candidates: unlabeled (-1), not table/support (sup_share < 0.35),
            # contributes >= 15% to a foreground object AND lies within 2.0 cm of a reliable object hull.
            g_cand = (oid_t == -1) & (fg_max_share >= 0.15) & (sup_share < 0.35)
            if safegs_args.safegs_lam_ghost > 0 and g_cand.any() and cons_ids:
                g_xyz = xyz[g_cand]
                min_d = torch.full((len(g_xyz),), 1e6, device="cuda")
                for k in cons_ids:
                    min_d = torch.minimum(min_d, fds[k](g_xyz) * cm)
                near_hull = min_d <= 2.0
                ghost_mask = torch.zeros_like(g_cand)
                ghost_mask[g_cand] = near_hull
            else:
                ghost_mask = torch.zeros_like(g_cand)
        mean_out = float(np.mean(out_pcts)) if out_pcts else 0.0
        print(f"\n[Safe-GS Joint @ iter {it_num}] N={len(oid_t)}, fg={int(fg_valid.sum())}, "
              f"reliable_objs={len(cons_ids)}/{n_obj}, mean_obj_out={mean_out:.2f}%, ghost_cand={int(ghost_mask.sum())}")

    for iteration in range(first_iter, opt.iterations + 1):
        xyz_lr = gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        if iteration > mesh.distortion_from_iter and mesh.lambda_opacity_field > 0.0:
            splat_args.render_opacity = True

        # Safe-GS joint label refresh
        safegs_active = iteration >= safegs_args.safegs_from_iter
        if safegs_active:
            need_refresh = (
                oid_t is None
                or len(oid_t) != len(gaussians._xyz)
                or iteration == safegs_args.safegs_from_iter
                or (iteration - safegs_args.safegs_from_iter) % safegs_args.safegs_relabel_every == 0
            )
            if need_refresh:
                stride = 8 if iteration <= opt.densify_until_iter else 4
                refresh_labels(iteration, stride=stride)

        gt_image = viewpoint_cam.original_image.cuda()
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, splat_args=splat_args, gt_color=gt_image.detach())
        rendering, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"], render_pkg["viewspace_points"],
            render_pkg["visibility_filter"], render_pkg["radii"]
        )
        image = rendering[:3, :, :]
        variance = rendering[11, :, :]
        normal_variance = rendering[12, :, :]

        if mesh.color_confidence and iteration >= mesh.color_confidence_from_iter:
            confidence = torch.clamp(rendering[10, :, :], min=1e-3, max=5.0).unsqueeze(0)
            pp_rgb_loss = appearance_embedding(image, gt_image, viewpoint_cam.idx)
            alpha = mesh.color_confidence_max
            rgb_loss = pp_rgb_loss * confidence - alpha * torch.log(confidence)
        else:
            rgb_loss = appearance_embedding(image, gt_image, viewpoint_cam.idx)

        distortion_loss = rendering[8, :, :].mean()
        depth = rendering[6, :, :]
        if depth.isnan().sum() > 0:
            depth[depth.isnan()] = 0.0
        depth_normal, _ = depth_to_normal(viewpoint_cam, depth[None, ...])
        depth_normal = depth_normal.permute(2, 0, 1)
        render_normal = torch.nn.functional.normalize(rendering[3:6, :, :], p=2, dim=0)
        c2w = viewpoint_cam.world_view_transform
        normal2 = c2w[:3, :3] @ render_normal.reshape(3, -1)
        render_normal_world = normal2.reshape(3, *render_normal.shape[1:])
        nabla_I = central_diff(viewpoint_cam.original_image.permute(1, 2, 0)).cuda()
        depth_normal_loss = (1 - (render_normal_world * depth_normal).sum(dim=0)).mean()

        lambda_distortion = mesh.lambda_distortion if iteration >= mesh.distortion_from_iter else 0.0
        lambda_depth_normal = mesh.lambda_depth_normal if iteration >= mesh.depth_normal_from_iter else 0.0
        normal_loss = (central_diff(render_normal_world.permute(1, 2, 0)) * torch.exp(-nabla_I)).mean()
        lambda_normal = mesh.lambda_smoothness if iteration >= mesh.depth_normal_from_iter else 0.0
        lambda_opacity_field = mesh.lambda_opacity_field if iteration >= mesh.distortion_from_iter else 0.0
        opa_loss = ((rendering[7] - 0.5) ** 2).mean()
        lambda_extent = mesh.lambda_extent if iteration >= mesh.distortion_from_iter else 0.0
        extent_loss = rendering[9].mean()
        rgb_loss_mean = rgb_loss.mean()
        lambda_variance = mesh.lambda_variance if iteration >= mesh.variance_from_iter else 0.0
        lambda_normal_variance = mesh.lambda_normal_variance if iteration >= mesh.normal_variance_from_iter else 0.0

        loss = (
            rgb_loss_mean
            + depth_normal_loss * lambda_depth_normal
            + distortion_loss * lambda_distortion
            + normal_loss * lambda_normal
            + opa_loss * lambda_opacity_field
            + extent_loss * lambda_extent
            + variance.mean() * lambda_variance
            + normal_variance.mean() * lambda_normal_variance
        )

        # Safe-GS L_bound (on reliable foreground objects)
        if safegs_active and safegs_args.safegs_lam_bound > 0 and cons_mask is not None and cons_mask.any():
            xyz_c = gaussians.get_xyz[cons_mask]
            oo = oid_t[cons_mask]
            d_c = torch.zeros(len(xyz_c), device="cuda")
            for k in cons_ids:
                mk = oo == k
                if mk.any():
                    d_c[mk] = fds[k](xyz_c[mk]) * cm
            lb = (gaussians.get_opacity[cons_mask].squeeze(-1) * d_c).mean()
            loss = loss + safegs_args.safegs_lam_bound * lb

        # Safe-GS L_ghost (suppress unlabeled boundary floaters near object hulls without touching table/other objects)
        if safegs_active and safegs_args.safegs_lam_ghost > 0 and ghost_mask is not None and ghost_mask.any():
            l_ghost = gaussians.get_opacity[ghost_mask].mean()
            loss = loss + safegs_args.safegs_lam_ghost * l_ghost

        # Safe-GS L_leak + L_cover (aux step, gradients masked to selected object k)
        if (
            safegs_active
            and (safegs_args.safegs_lam_leak > 0 or safegs_args.safegs_lam_cover > 0)
            and cons_ids
            and iteration % safegs_args.safegs_aux_every == 0
        ):
            lv = lab_cache.get(view_stem(viewpoint_cam))
            if lv is not None:
                lvg = lv.cuda().long()
                cand = []
                for k in cons_ids:
                    Mk = torch.isin(lvg, gid_t[k])
                    if int(Mk.sum()) >= 200:
                        cand.append((k, Mk.float()))
                if cand:
                    k, Mk = cand[rng.randrange(len(cand))]
                    sel = oid_t == k
                    ci = torch.zeros((len(oid_t), 3), device="cuda")
                    ci[sel] = 1.0
                    ak = render(viewpoint_cam, gaussians, pipe, black, splat_args=splat_args, override_color=ci)["render"][0]
                    dlk = F.max_pool2d(Mk[None, None], 31, 1, 15)[0, 0] > 0.5
                    erk = (1 - F.max_pool2d((1 - Mk)[None, None], 7, 1, 3)[0, 0]) > 0.5
                    l_leak = ak[~dlk].sum() / Mk.sum().clamp(min=1.0)
                    l_cov = (1.0 - ak[erk]).clamp(min=0.0).mean() if int(erk.sum()) > 50 else torch.zeros((), device="cuda")
                    laux = safegs_args.safegs_lam_leak * l_leak + safegs_args.safegs_lam_cover * l_cov
                    laux.backward()
                    keep = sel.float()
                    for pg in gaussians.optimizer.param_groups:
                        for t in pg["params"]:
                            if t.grad is not None and t.dim() > 0 and t.shape[0] == len(keep):
                                t.grad.mul_(keep.view(-1, *([1] * (t.grad.dim() - 1))))

        loss.backward()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}", "Size": f"{len(gaussians._xyz)}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration, appearance_embedding.capture())

            temp_splat_args = copy.deepcopy(splat_args)
            temp_splat_args.consider_max_weight = True
            render_simp = partial(render, pipe=pipe, bg_color=background, splat_args=temp_splat_args)
            densifier.densify(
                iteration=iteration,
                visibility_filter=visibility_filter,
                radii=radii,
                viewspace_point_tensor=viewspace_point_tensor,
                cameras_extent=scene.cameras_extent,
                trainCameras=trainCameras,
                render_simp=render_simp,
            )

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                densifier.postfix(xyz_lr=xyz_lr)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration, appearance_embedding.capture()),
                           scene.model_path + f"/chkpnt{iteration}.pth")

    # Final evaluation on test cameras + Safe-GS metrics
    if oid_t is not None and len(oid_t) == len(gaussians._xyz):
        np.save(os.path.join(dataset.model_path, "safegs_oid.npy"), oid_t.cpu().numpy())
    test_cams = scene.getTestCameras()
    with torch.no_grad():
        psnr_vals = []
        for tc in test_cams:
            gt = torch.clamp(tc.original_image.cuda()[:3], 0.0, 1.0)
            rd = torch.clamp(render(tc, gaussians, pipe, background, splat_args=splat_args)["render"][:3], 0.0, 1.0)
            psnr_vals.append(float(psnr(rd, gt).mean()))
        xyz = gaussians.get_xyz
        fg_mask = (oid_t >= 0) & (oid_t < n_obj) if oid_t is not None else torch.zeros(len(xyz), dtype=torch.bool, device="cuda")
        out_cnt = 0
        cons_cnt = 0
        cons_out_cnt = 0
        per_obj = {}
        for k, n in enumerate(names):
            if oid_t is None or not fds[k].valid:
                continue
            mk = oid_t == k
            if not mk.any():
                continue
            dk = fds[k](xyz[mk]) * cm
            oc = int((dk > 0.5).sum())
            out_cnt += oc
            if reliable[k]:
                cons_cnt += int(mk.sum())
                cons_out_cnt += oc
            per_obj[n] = {"gaussians": int(mk.sum()), "outside_pct": 100.0 * oc / int(mk.sum()), "reliable": bool(reliable[k])}
        summary = {
            "test_views": len(test_cams),
            "test_psnr": float(np.mean(psnr_vals)) if psnr_vals else None,
            "total_gaussians": int(len(xyz)),
            "fg_gaussians": int(fg_mask.sum()),
            "outside_pct_all_fg": 100.0 * out_cnt / max(int(fg_mask.sum()), 1),
            "outside_pct_constrained": 100.0 * cons_out_cnt / max(cons_cnt, 1),
            "elapsed_sec": time.time() - start_event,
            "per_object": per_obj,
        }
        Path(os.path.join(dataset.model_path, "safegs_joint_metrics.json")).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("[Safe-GS Joint] Summary:", json.dumps({k: v for k, v in summary.items() if k != "per_object"}, indent=2))


if __name__ == "__main__":
    parser = ArgumentParser(description="Joint CoMe + Safe-GS Training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    mp = MeshingParams(parser)
    ss = SplattingSettings(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)

    parser.add_argument("--safegs_config", type=str, required=True)
    parser.add_argument("--safegs_labels", type=str, required=True)
    parser.add_argument("--safegs_hull_dir", type=str, required=True)
    parser.add_argument("--safegs_from_iter", type=int, default=10_000)
    parser.add_argument("--safegs_relabel_every", type=int, default=2500)
    parser.add_argument("--safegs_aux_every", type=int, default=4)
    parser.add_argument("--safegs_lam_bound", type=float, default=0.05)
    parser.add_argument("--safegs_lam_leak", type=float, default=0.05)
    parser.add_argument("--safegs_lam_cover", type=float, default=0.02)
    parser.add_argument("--safegs_lam_ghost", type=float, default=0.0)
    parser.add_argument("--safegs_max_out_pct", type=float, default=20.0)
    parser.add_argument("--safegs_mpu", type=float, default=0.0)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    splat_args = ss.get_settings(args)
    training(
        lp.extract(args), op.extract(args), pp.extract(args), mp.extract(args),
        args.test_iterations, args.save_iterations,
        args.checkpoint_iterations, args.start_checkpoint,
        args.debug_from, splat_args, args,
    )
    print("\nJoint training complete.")
