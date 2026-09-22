"""Module 4 - GOF Meshing: Binary Search tim isosurface 0.5.

Sau khi Marching Tetrahedra (marching_tetrahedra.py) cho vi tri xap xi cua
giao diem tren tung canh tu dien, buoc nay lam min dan vi tri do bang tim
kiem nhi phan 8 buoc doc theo canh, danh gia lai opacity field
(opacity_field.evaluate_min_opacity_field) tai moi buoc — dung Y het thuat
toan cua GOF (Yu et al., 2024).

Nguon tham khao: Based_Model/Gaussian Opacity Fields (GOF)/
gaussian-opacity-fields/extract_mesh.py::marching_tetrahedra_with_binary_search
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import trimesh

from .marching_tetrahedra import delaunay_triangulate, marching_tetrahedra, sdf_from_alpha
from .opacity_field import IntegrateFn, evaluate_min_opacity_field, generate_tetra_candidate_points

N_BINARY_STEPS = 8
ISO_LEVEL = 0.5


@torch.no_grad()
def binary_search_level_set(
    left_points: torch.Tensor,
    right_points: torch.Tensor,
    left_sdf: torch.Tensor,
    right_sdf: torch.Tensor,
    views: Sequence[object],
    integrate_fn: IntegrateFn,
    n_steps: int = N_BINARY_STEPS,
    texture_at_last_step: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Tim nhi phan vi tri isosurface tren tung canh [left_points, right_points].

    Moi buoc: lay diem giua, danh gia lai opacity field tai do, roi thu
    hep canh ve phia co dau SDF khac voi diem giua (giu bat bien: left
    luon o phia am, right luon o phia duong theo dinh nghia SDF = alpha -
    0.5). Sau ``n_steps`` buoc, sai so vi tri con lai giam theo cap so
    nhan 2^-n_steps so voi canh tu dien ban dau.

    Tra ve:
        points: (M, 3) vi tri dinh mesh cuoi cung.
        vertex_colors: (M, 3) uint8 mau noi suy tai buoc cuoi (None neu
            ``texture_at_last_step=False``).
    """
    left_points = left_points.clone()
    right_points = right_points.clone()
    left_sdf = left_sdf.clone()
    right_sdf = right_sdf.clone()

    points = (left_points + right_points) / 2.0
    vertex_colors = None

    for step in range(n_steps):
        mid_points = (left_points + right_points) / 2.0
        alpha = evaluate_min_opacity_field(
            mid_points, views, integrate_fn, desc=f"Binary search buoc {step + 1}/{n_steps}"
        )
        mid_sdf = sdf_from_alpha(alpha).unsqueeze(-1)

        keep_left_side = ((mid_sdf < 0) & (left_sdf < 0)) | ((mid_sdf > 0) & (left_sdf > 0))

        left_sdf[keep_left_side] = mid_sdf[keep_left_side]
        right_sdf[~keep_left_side] = mid_sdf[~keep_left_side]
        left_points[keep_left_side.flatten()] = mid_points[keep_left_side.flatten()]
        right_points[~keep_left_side.flatten()] = mid_points[~keep_left_side.flatten()]

        points = (left_points + right_points) / 2.0

        if texture_at_last_step and step == n_steps - 1:
            _, color = evaluate_min_opacity_field(points, views, integrate_fn, return_color=True)
            vertex_colors = (color.cpu().numpy() * 255).astype(np.uint8)

    return points, vertex_colors


def build_mesh(
    points: torch.Tensor,
    faces: np.ndarray,
    vertex_colors: Optional[np.ndarray] = None,
    distance: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    filter_mesh: bool = True,
) -> trimesh.Trimesh:
    """Dung Trimesh tu dinh + mat, loc bo canh tu dien qua dai so voi
    scale cua Gaussian sinh ra no (dau hieu cua vung khong duoc quan sat
    ky, de lai cho watertight_sewer.py o module5 xu ly tiep)."""
    mesh = trimesh.Trimesh(
        vertices=points.detach().cpu().numpy(),
        faces=faces,
        vertex_colors=vertex_colors,
        process=False,
    )

    if filter_mesh and distance is not None and scale is not None:
        mask = (distance <= scale).cpu().numpy()
        face_mask = mask[faces].all(axis=1)
        mesh.update_vertices(mask)
        mesh.update_faces(face_mask)

    return mesh


def extract_mesh_gof(
    gaussians: object,
    views: Sequence[object],
    integrate_fn: IntegrateFn,
    near: float = 0.02,
    far: float = 1e6,
    box_scale: float = 3.0,
    iso_level: float = ISO_LEVEL,
    chunk_size: int = 32 * 1024 * 1024,
    n_binary_steps: int = N_BINARY_STEPS,
    filter_mesh: bool = True,
    texture_mesh: bool = False,
) -> trimesh.Trimesh:
    """Pipeline day du cua Module 4: Composed-3DGS -> raw_gof mesh.

    1) generate_tetra_candidate_points  (opacity_field.py)
    2) delaunay_triangulate             (marching_tetrahedra.py)
    3) evaluate_min_opacity_field       (opacity_field.py)
    4) marching_tetrahedra              (marching_tetrahedra.py)
    5) binary_search_level_set          (file nay)
    6) build_mesh                       (file nay)

    Day la mesh THO (raw_gof.obj) — decimation/watertight sewing thuoc
    ve module5_postprocessing, KHONG xu ly o day.
    """
    points, point_scales = generate_tetra_candidate_points(gaussians, views, near, far, box_scale)
    cells = delaunay_triangulate(points)

    alpha = evaluate_min_opacity_field(points, views, integrate_fn, desc="Danh gia opacity field ban dau")
    sdf = sdf_from_alpha(alpha, iso_level)

    torch.cuda.empty_cache()
    verts_list, scale_list, faces_list, _ = marching_tetrahedra(
        points[None], cells, sdf[None], point_scales[None], chunk_size
    )
    torch.cuda.empty_cache()

    end_points, end_sdf = verts_list[0]
    end_scales = scale_list[0]
    faces = faces_list[0].cpu().numpy()

    left_points, right_points = end_points[:, 0, :], end_points[:, 1, :]
    left_sdf, right_sdf = end_sdf[:, 0, :], end_sdf[:, 1, :]
    left_scale, right_scale = end_scales[:, 0, 0], end_scales[:, 1, 0]

    distance = torch.norm(left_points - right_points, dim=-1)
    edge_scale = left_scale + right_scale

    final_points, vertex_colors = binary_search_level_set(
        left_points, right_points, left_sdf, right_sdf, views, integrate_fn,
        n_steps=n_binary_steps, texture_at_last_step=texture_mesh,
    )

    return build_mesh(final_points, faces, vertex_colors, distance, edge_scale, filter_mesh)


def extract_mesh_gof_from_config(
    gaussians: object,
    views: Sequence[object],
    integrate_fn: IntegrateFn,
    config_path: str,
) -> trimesh.Trimesh:
    """Nhu ``extract_mesh_gof`` nhung doc toan bo tham so tu
    ``configs/gof_meshing.toml`` thay vi truyen tay tung gia tri.

    Day la entry point du kien cho scripts/04_run_gof_meshing.py.
    """
    from src.common.config_loader import load_toml_config

    cfg = load_toml_config(config_path)
    return extract_mesh_gof(
        gaussians,
        views,
        integrate_fn,
        near=cfg.get("near", 0.02),
        far=cfg.get("far", 1e6),
        box_scale=cfg.get("box_scale", 3.0),
        iso_level=cfg.get("iso_level", ISO_LEVEL),
        chunk_size=cfg.get("chunk_size", 32 * 1024 * 1024),
        n_binary_steps=cfg.get("n_binary_steps", N_BINARY_STEPS),
        filter_mesh=cfg.get("filter_mesh", True),
        texture_mesh=cfg.get("texture_mesh", False),
    )
