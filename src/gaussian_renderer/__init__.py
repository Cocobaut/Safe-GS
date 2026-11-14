#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# Renderer switched to the CoMe rasterizer (StopThePop sorting + ray-based Gaussian evaluation + 3D filter),
# following CoMe gaussian_renderer.render. Signature and returned keys are kept as in Inpaint360GS; the 16-dim
# object feature ("render_object") is rendered by the same rasterizer through colors_precomp (3 channels per pass)
# and only when it is read.

import os
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer, ExtendedSettings, DebugVisualization
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

# Output channels of the CoMe rasterizer (cuda_rasterizer/auxiliary.h)
DEPTH_OFFSET = 6
ALPHA_OFFSET = 7
CONFIDENCE_OFFSET = 10

_splat_settings_cache = {}


def get_splat_settings() -> ExtendedSettings:
    """ExtendedSettings from the CoMe config.json given by COME_SPLATTING_CONFIG (cached)."""
    path = os.environ.get("COME_SPLATTING_CONFIG", "")
    if path not in _splat_settings_cache:
        if path:
            _splat_settings_cache[path] = ExtendedSettings.from_json(path)
        else:
            print("[Warning] COME_SPLATTING_CONFIG not set, using default CoMe ExtendedSettings")
            _splat_settings_cache[path] = ExtendedSettings()
    return _splat_settings_cache[path]


class RenderPackage(dict):
    """dict whose "render_object" entry is computed on first access."""

    def __init__(self, *args, object_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._object_fn = object_fn

    def _fill(self, key):
        if key == "render_object" and not dict.__contains__(self, key) and self._object_fn is not None:
            dict.__setitem__(self, key, self._object_fn())

    def __getitem__(self, key):
        self._fill(key)
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        self._fill(key)
        return dict.get(self, key, default)

    def __contains__(self, key):
        return dict.__contains__(self, key) or (key == "render_object" and self._object_fn is not None)


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, gt_color = None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    gt_color: ground-truth image [3, H, W] for the color-variance channel (as CoMe train.py); zeros if None.
    """
    pc.ensure_come_attributes()

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    def make_rasterizer(bg):
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            # recomputed every call: virtual cameras are edited in place by the Inpaint360GS tools
            inv_viewprojmatrix=torch.inverse(viewpoint_camera.full_proj_transform),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            settings=get_splat_settings(),
            debug_data=DebugVisualization(),
            debug=pipe.debug,
        )
        return GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    # Raw scale/opacity: the 3D filter is applied inside the CoMe rasterizer (compute_filter3D_python = False in CoMe)
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    confidences = torch.exp(pc.get_confidence)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered, radii, max_weights = make_rasterizer(bg_color)(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        confidences = confidences,
        cov3D_precomp = None,
        view2gaussian_precomp = None,
        filter_3d = pc.filter_3D,
        gt_color = gt_color)

    def render_objects():
        """Object feature [16, H, W]: blended like the colors, no background (as the Inpaint360GS rasterizer)."""
        feats = pc.get_objects[:, 0, :]                       # [N, 16]
        n_ch = feats.shape[1]
        pad = (-n_ch) % 3
        if pad:
            feats = torch.cat([feats, torch.zeros_like(feats[:, :pad])], dim=1)
        rasterizer = make_rasterizer(torch.zeros(3, dtype=torch.float32, device="cuda"))
        chunks = []
        for k in range(0, feats.shape[1], 3):
            out, _, _ = rasterizer(
                means3D = means3D,
                means2D = torch.zeros_like(means3D, requires_grad=True),
                shs = None,
                colors_precomp = feats[:, k:k + 3].contiguous(),
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                confidences = confidences,
                cov3D_precomp = None,
                view2gaussian_precomp = None,
                filter_3d = pc.filter_3D)
            chunks.append(out[0:3])
        return torch.cat(chunks, dim=0)[:n_ch]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return RenderPackage({"render": rendered[0:3],
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth_3dgs": rendered[DEPTH_OFFSET:DEPTH_OFFSET + 1],
            "alpha": rendered[ALPHA_OFFSET:ALPHA_OFFSET + 1],
            "confidence": rendered[CONFIDENCE_OFFSET:CONFIDENCE_OFFSET + 1],
            "rendered_all": rendered,
            "max_weights": max_weights},
            object_fn=render_objects)
