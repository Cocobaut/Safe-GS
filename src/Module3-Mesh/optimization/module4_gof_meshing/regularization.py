"""Module 4 - GOF Meshing: Depth Distortion Loss & Normal Consistency Loss.

Hai regularizer nay duoc GOF dung trong luc huan luyen Composed-3DGS de ep
cac Gaussian "dep mong" lai theo huong be mat that, giup opacity field
(module4_gof_meshing.opacity_field) sinh dang diem 0.5 sac net hon khi
trich mesh bang Marching Tetrahedra.

Tham khao cai dat goc: Based_Model/Gaussian Opacity Fields (GOF)/
gaussian-opacity-fields/{train.py, utils/depth_utils.py}.

Yeu cau render_pkg (dau ra cua rasterizer, giong 2DGS/GOF) tra ve mot
tensor ``rendering`` (C, H, W) trong do:
    - kenh [3:6] la normal duoc render (khong gian camera),
    - kenh [6]   la depth (median hoac expected depth),
    - kenh [8]   la distortion map.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


def depths_to_points(view: object, depth: torch.Tensor) -> torch.Tensor:
    """Unproject depth map (1, H, W) ve diem 3D trong world space."""
    device = depth.device
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height
    fx = W / (2 * math.tan(view.FoVx / 2.0))
    fy = H / (2 * math.tan(view.FoVy / 2.0))
    intrins = torch.tensor(
        [[fx, 0.0, W / 2.0], [0.0, fy, H / 2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    grid_x, grid_y = torch.meshgrid(
        torch.arange(W, device=device, dtype=torch.float32) + 0.5,
        torch.arange(H, device=device, dtype=torch.float32) + 0.5,
        indexing="xy",
    )
    pixels = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = pixels @ intrins.inverse().T @ c2w[:3, :3].T
    rays_o = c2w[:3, 3]
    points = depth.reshape(-1, 1) * rays_d + rays_o
    return points


def depth_to_normal(view: object, depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Uoc luong normal map tu depth map bang finite-difference giua cac
    diem 3D lan can (pseudo-normal, dung lam "ground-truth" cho
    normal_consistency_loss)."""
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = points[2:, 1:-1] - points[:-2, 1:-1]
    dy = points[1:-1, 2:] - points[1:-1, :-2]
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output, points


def edge_aware_weight_map(gt_image: torch.Tensor) -> torch.Tensor:
    """Trong so exp(-gradient anh) dung de giam distortion loss tai vien
    canh sac net (noi depth that su khong lien tuc)."""
    grad_left = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 1:-1, :-2]), dim=0)
    grad_right = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 1:-1, 2:]), dim=0)
    grad_top = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, :-2, 1:-1]), dim=0)
    grad_bottom = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 2:, 1:-1]), dim=0)
    max_grad = torch.max(torch.stack([grad_left, grad_right, grad_top, grad_bottom], dim=-1), dim=-1)[0]
    weight = torch.exp(-max_grad)
    weight = torch.nn.functional.pad(weight, (1, 1, 1, 1), mode="constant", value=0)
    return weight


def depth_distortion_loss(
    distortion_map: torch.Tensor,
    gt_image: torch.Tensor | None = None,
    edge_aware: bool = False,
) -> torch.Tensor:
    """Loss ep depth cua tia lai gan nhau (giam hien tuong nhieu lop
    Gaussian mong chong len nhau tao floaters)."""
    if edge_aware:
        if gt_image is None:
            raise ValueError("gt_image la bat buoc khi edge_aware=True")
        return (distortion_map * edge_aware_weight_map(gt_image)).mean()
    return distortion_map.mean()


def normal_consistency_loss(
    render_normal: torch.Tensor,
    depth: torch.Tensor,
    view: object,
) -> torch.Tensor:
    """Loss ep normal duoc render khop voi normal suy ra tu depth map,
    theo dung dinh nghia cua GOF/2DGS.

    Args:
        render_normal: (3, H, W) normal duoc rasterizer render, khong gian camera.
        depth: (1, H, W) depth map duoc rasterizer render.
        view: camera hien tai (can world_view_transform de doi world <-> camera).
    """
    depth_normal, _ = depth_to_normal(view, depth)
    depth_normal = depth_normal.permute(2, 0, 1)  # (3, H, W), world space

    render_normal = torch.nn.functional.normalize(render_normal, p=2, dim=0)
    c2w = (view.world_view_transform.T).inverse()
    render_normal_world = (c2w[:3, :3] @ render_normal.reshape(3, -1)).reshape(*render_normal.shape)

    normal_error = 1.0 - (render_normal_world * depth_normal).sum(dim=0)
    return normal_error.mean()


def compute_gof_regularization(
    rendering: torch.Tensor,
    view: object,
    iteration: int | None = None,
    config: dict | None = None,
    gt_image: torch.Tensor | None = None,
    edge_aware_distortion: bool = False,
) -> dict:
    """Goi tat ca regularizer tu mot ``rendering`` tensor theo dung layout
    cua GOF: kenh [3:6]=normal, [6]=depth, [8]=distortion.

    Mac dinh (khong truyen ``iteration``/``config``) tra ve raw losses,
    khong nhan trong so — giu nguyen hanh vi cu de tuong thich nguoc.

    Neu truyen ca ``iteration`` va ``config`` (dict tu
    ``configs/gof_meshing.toml``), tinh them ``"weighted_loss"`` theo dung
    lambda-schedule cua GOF: ``lambda_distortion`` chi ap dung tu
    ``distortion_from_iter``, ``lambda_depth_normal`` chi ap dung tu
    ``depth_normal_from_iter`` (xem train.py cua GOF).
    """
    distortion_map = rendering[8, :, :]
    depth = rendering[6, :, :][None, ...]
    render_normal = rendering[3:6, :, :]

    result = {
        "distortion_loss": depth_distortion_loss(distortion_map, gt_image, edge_aware_distortion),
        "normal_consistency_loss": normal_consistency_loss(render_normal, depth, view),
    }

    if iteration is not None and config is not None:
        lambda_distortion = config.get("lambda_distortion", 0.0) if iteration >= config.get("distortion_from_iter", 0) else 0.0
        lambda_depth_normal = config.get("lambda_depth_normal", 0.0) if iteration >= config.get("depth_normal_from_iter", 0) else 0.0
        result["weighted_loss"] = (
            result["distortion_loss"] * lambda_distortion
            + result["normal_consistency_loss"] * lambda_depth_normal
        )

    return result
