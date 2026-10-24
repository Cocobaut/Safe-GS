"""
integrate() của GOF (Based_Model/GOF/gaussian_renderer/__init__.py): tích phân opacity / màu
của các Gaussian dọc tia camera tới từng điểm 3D. Cần rasterizer của GOF (mesh/submodules/diff-gaussian-rasterization).
"""
import math

import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from .gaussian_model import GaussianModel


def integrate(points3D, viewpoint_camera, pc: GaussianModel, bg_color: torch.Tensor, kernel_size: float,
              scaling_modifier=1.0, subpixel_offset=None):
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if subpixel_offset is None:
        subpixel_offset = torch.zeros((int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2),
                                      dtype=torch.float32, device="cuda")

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    rendered_image, alpha_integrated, color_integrated, radii = rasterizer.integrate(
        points3D=points3D,
        means3D=pc.get_xyz,
        means2D=screenspace_points,
        shs=pc.get_features,
        colors_precomp=None,
        opacities=pc.get_opacity_with_3D_filter,
        scales=pc.get_scaling_with_3D_filter,
        rotations=pc.get_rotation,
        cov3D_precomp=None,
        view2gaussian_precomp=None)

    return {"render": rendered_image,
            "alpha_integrated": alpha_integrated,
            "color_integrated": color_integrated,
            "radii": radii}
