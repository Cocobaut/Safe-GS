"""Module 4: Trich xuat mesh tho tu Composed-3DGS bang Gaussian Opacity Fields."""

from .level_set_solver import (
    binary_search_level_set,
    build_mesh,
    extract_mesh_gof,
    extract_mesh_gof_from_config,
)
from .marching_tetrahedra import delaunay_triangulate, marching_tetrahedra, sdf_from_alpha
from .opacity_field import (
    compute_frustum_mask,
    evaluate_min_opacity_field,
    generate_tetra_candidate_points,
    make_diff_rasterizer_integrator,
    make_diff_rasterizer_integrator_from_config,
)
from .regularization import (
    compute_gof_regularization,
    depth_distortion_loss,
    depth_to_normal,
    normal_consistency_loss,
)

__all__ = [
    "extract_mesh_gof",
    "extract_mesh_gof_from_config",
    "binary_search_level_set",
    "build_mesh",
    "delaunay_triangulate",
    "marching_tetrahedra",
    "sdf_from_alpha",
    "compute_frustum_mask",
    "evaluate_min_opacity_field",
    "generate_tetra_candidate_points",
    "make_diff_rasterizer_integrator",
    "make_diff_rasterizer_integrator_from_config",
    "compute_gof_regularization",
    "depth_distortion_loss",
    "depth_to_normal",
    "normal_consistency_loss",
]
