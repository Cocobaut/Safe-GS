"""Module 4 - GOF Meshing: Ray-Gaussian intersection & Min-Opacity field.

Cai dat lai thuat toan Gaussian Opacity Fields (Yu et al., 2024) tren
mot Composed-3DGS da huan luyen san (dau ra cua module3_object_composition).

Tham khao cai dat goc: Based_Model/Gaussian Opacity Fields (GOF)/
gaussian-opacity-fields/{scene/gaussian_model.py::get_tetra_points,
gaussian_renderer/__init__.py::integrate, extract_mesh.py::evaluage_alpha}

Hai doi tuong dau vao (duck-typed, khong rang buoc class cu the) can co:

- ``gaussians``: get_xyz (N,3), get_scaling_with_3D_filter (N,3),
  get_rotation (N,4) quaternion, get_opacity_with_3D_filter (N,1),
  get_features, active_sh_degree.
- ``views``: moi camera can world_view_transform (4,4), full_proj_transform
  (4,4), FoVx, FoVy, image_width, image_height, camera_center (3,),
  focal_x, focal_y.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

IntegrateFn = Callable[[torch.Tensor, object], Dict[str, torch.Tensor]]


@torch.no_grad()
def compute_frustum_mask(
    points: torch.Tensor,
    cameras: Sequence[object],
    near: float = 0.02,
    far: float = 1e6,
) -> torch.Tensor:
    """Giu lai cac diem duoc nhin thay (trong [near, far] va trong khung hinh)
    boi it nhat mot camera trong tap ``cameras``."""
    device = points.device
    H, W = cameras[0].image_height, cameras[0].image_width

    intrinsics = torch.stack(
        [
            torch.tensor(
                [[cam.focal_x, 0.0, W / 2.0], [0.0, cam.focal_y, H / 2.0], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
                device=device,
            )
            for cam in cameras
        ],
        dim=0,
    )  # (V, 3, 3)

    view_matrices = torch.stack(
        [cam.world_view_transform for cam in cameras], dim=0
    ).transpose(1, 2)  # (V, 4, 4), world_view_transform luu dang row-major transpose

    ones = torch.ones_like(points[:, :1])
    homo_points = torch.cat([points, ones], dim=-1)  # (N, 4)

    view_points = torch.einsum("vij,nj->vni", view_matrices, homo_points)[..., :3]  # (V, N, 3)
    uv_points = torch.einsum("vij,vnj->vni", intrinsics, view_points)  # (V, N, 3)

    z = view_points[..., 2]
    safe_z = z.clamp(min=1e-8)
    u = uv_points[..., 0] / safe_z
    v = uv_points[..., 1] / safe_z

    in_frustum = (z > near) & (z < far) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return in_frustum.any(dim=0)


def _build_rotation_matrices(quaternions: torch.Tensor) -> torch.Tensor:
    """Chuyen quaternion (N,4) [r,x,y,z] thanh ma tran xoay (N,3,3)."""
    norm = torch.sqrt((quaternions * quaternions).sum(dim=-1, keepdim=True))
    q = quaternions / norm
    r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.zeros((q.shape[0], 3, 3), dtype=q.dtype, device=q.device)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


@torch.no_grad()
def generate_tetra_candidate_points(
    gaussians: object,
    views: Sequence[object],
    near: float = 0.02,
    far: float = 1e6,
    box_scale: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sinh diem ung vien cho luoi tu dien (tetrahedral grid).

    Voi moi Gaussian, lay 8 dinh cua hop bao huong (oriented box) theo
    scale/rotation cua no cong voi tam Gaussian, roi loc bo cac diem
    nam ngoai frustum cua toan bo camera huan luyen.

    Tra ve:
        points: (M, 3) toa do cac diem ung vien.
        point_scales: (M, 1) scale lon nhat cua Gaussian sinh ra diem do,
            dung cho buoc loc mesh sau nay (module5_postprocessing).
    """
    device = gaussians.get_xyz.device
    # 8 dinh cua khoi lap phuong don vi [-1, 1]^3
    unit_corners = torch.tensor(
        [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ],
        dtype=torch.float32,
        device=device,
    )

    xyz = gaussians.get_xyz  # (N, 3)
    scale = gaussians.get_scaling_with_3D_filter * box_scale  # (N, 3)
    rots = _build_rotation_matrices(gaussians.get_rotation)  # (N, 3, 3)

    n = xyz.shape[0]
    corners = unit_corners.unsqueeze(0).expand(n, -1, -1)  # (N, 8, 3)
    corners = corners * scale.unsqueeze(1)  # scale theo tung truc
    corners = torch.bmm(corners, rots.transpose(1, 2)) + xyz.unsqueeze(1)  # xoay + dich
    corners = corners.reshape(-1, 3)

    all_points = torch.cat([corners, xyz], dim=0)

    max_scale = scale.max(dim=-1, keepdim=True)[0]  # (N, 1)
    corner_scales = max_scale.repeat(1, 8).reshape(-1, 1)
    all_scales = torch.cat([corner_scales, max_scale], dim=0)

    mask = compute_frustum_mask(all_points, views, near, far)
    return all_points[mask], all_scales[mask]


@torch.no_grad()
def evaluate_min_opacity_field(
    points: torch.Tensor,
    views: Sequence[object],
    integrate_fn: IntegrateFn,
    return_color: bool = False,
    desc: str = "Danh gia opacity field",
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """Tinh Min-Opacity field tai cac diem ``points``.

    Voi moi camera, tich luy alpha cua tia di qua diem (ray-Gaussian
    intersection). Opacity field cua GOF la (1 - min_alpha) tren toan bo
    view: mot diem chi "dac" (opacity cao) neu no bi che khuat boi Gaussian
    o MOI goc nhin, tranh sinh be mat gia (floaters) chi xuat hien o mot view.
    """
    device = points.device
    final_min_alpha = torch.ones(points.shape[0], dtype=torch.float32, device=device)
    final_color: Optional[torch.Tensor] = None
    if return_color:
        final_color = torch.ones((points.shape[0], 3), dtype=torch.float32, device=device)

    for view in tqdm(views, desc=desc):
        result = integrate_fn(points, view)
        alpha_integrated = result["alpha_integrated"]
        if return_color:
            color_integrated = result["color_integrated"]
            take_new = (alpha_integrated < final_min_alpha).unsqueeze(-1)
            final_color = torch.where(take_new, color_integrated, final_color)
        final_min_alpha = torch.minimum(final_min_alpha, alpha_integrated)

    opacity_field = 1.0 - final_min_alpha
    if return_color:
        return opacity_field, final_color
    return opacity_field


def make_diff_rasterizer_integrator(
    gaussians: object,
    background: torch.Tensor,
    kernel_size: float,
    scaling_modifier: float = 1.0,
    compute_debug: bool = False,
) -> IntegrateFn:
    """Tao ``integrate_fn(points, view) -> dict`` dung CUDA rasterizer
    trong ``submodules/diff_gaussian_rasterization``.

    Luu y: day PHAI la ban rasterizer da duoc mo rong voi method
    ``integrate()`` (ban cua GOF), khong phai rasterizer 3DGS goc chi co
    ``forward()``/``backward()``. Cai dat tham khao: Based_Model/
    Gaussian Opacity Fields (GOF)/gaussian-opacity-fields/
    gaussian_renderer/__init__.py::integrate.
    """
    try:
        from diff_gaussian_rasterization import (  # type: ignore
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )
    except ImportError as exc:  # pragma: no cover - phu thuoc moi truong
        raise ImportError(
            "Khong import duoc 'diff_gaussian_rasterization'. Submodule "
            "submodules/diff_gaussian_rasterization can duoc build voi "
            "method integrate() (ban mo rong cua GOF) truoc khi chay "
            "module4_gof_meshing.opacity_field. Tham khao ban da build san "
            "trong Based_Model/Gaussian Opacity Fields (GOF)/"
            "gaussian-opacity-fields/submodules/diff-gaussian-rasterization."
        ) from exc

    def integrate_fn(points: torch.Tensor, view: object) -> Dict[str, torch.Tensor]:
        screenspace_points = torch.zeros_like(gaussians.get_xyz, requires_grad=False)
        subpixel_offset = torch.zeros(
            (int(view.image_height), int(view.image_width), 2),
            dtype=torch.float32,
            device=points.device,
        )
        raster_settings = GaussianRasterizationSettings(
            image_height=int(view.image_height),
            image_width=int(view.image_width),
            tanfovx=math.tan(view.FoVx * 0.5),
            tanfovy=math.tan(view.FoVy * 0.5),
            kernel_size=kernel_size,
            subpixel_offset=subpixel_offset,
            bg=background,
            scale_modifier=scaling_modifier,
            viewmatrix=view.world_view_transform,
            projmatrix=view.full_proj_transform,
            sh_degree=gaussians.active_sh_degree,
            campos=view.camera_center,
            prefiltered=False,
            debug=compute_debug,
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        _, alpha_integrated, color_integrated, _ = rasterizer.integrate(
            points3D=points,
            means3D=gaussians.get_xyz,
            means2D=screenspace_points,
            shs=gaussians.get_features,
            colors_precomp=None,
            opacities=gaussians.get_opacity_with_3D_filter,
            scales=gaussians.get_scaling_with_3D_filter,
            rotations=gaussians.get_rotation,
            cov3D_precomp=None,
        )
        return {"alpha_integrated": alpha_integrated, "color_integrated": color_integrated}

    return integrate_fn


def make_diff_rasterizer_integrator_from_config(
    gaussians: object,
    background: torch.Tensor,
    config_path: str,
) -> IntegrateFn:
    """Nhu ``make_diff_rasterizer_integrator`` nhung doc ``kernel_size``,
    ``scaling_modifier``, ``compute_debug`` tu ``configs/gof_meshing.toml``
    thay vi truyen tay.
    """
    from src.common.config_loader import load_toml_config

    cfg = load_toml_config(config_path)
    if "kernel_size" not in cfg:
        raise KeyError(
            f"Thieu key bat buoc 'kernel_size' trong file config: {config_path}"
        )
    return make_diff_rasterizer_integrator(
        gaussians,
        background,
        kernel_size=cfg["kernel_size"],
        scaling_modifier=cfg.get("scaling_modifier", 1.0),
        compute_debug=cfg.get("compute_debug", False),
    )
