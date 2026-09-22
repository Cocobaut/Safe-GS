"""Module 4 - GOF Meshing: Marching Tetrahedra tren luoi Delaunay.

Gom 2 buoc:
  1. ``delaunay_triangulate``: tam giac phan hoa (tetrahedralize) tap diem
     ung vien (opacity_field.generate_tetra_candidate_points) thanh cac
     tu dien (tetrahedra).
  2. ``marching_tetrahedra``: thuat toan Marching Tetrahedra thuan-torch,
     khu vi phan duoc, chuyen SDF roi rac tren luoi tu dien thanh mesh
     tam giac. Ham nay vendor tu kaolin (Apache-2.0), duoc GOF/2DGS/
     DMTet dung lai nguyen ban.

Nguon tham khao: Based_Model/Gaussian Opacity Fields (GOF)/
gaussian-opacity-fields/{extract_mesh.py, utils/tetmesh.py}, gan voi
kaolin.ops.conversions.tetmesh (https://github.com/NVIDIAGameWorks/kaolin).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

__all__ = [
    "delaunay_triangulate",
    "marching_tetrahedra",
    "sdf_from_alpha",
]

# ---------------------------------------------------------------------------
# 1. Delaunay tetrahedralization
# ---------------------------------------------------------------------------


def delaunay_triangulate(points: torch.Tensor) -> torch.Tensor:
    """Tam giac phan hoa tap diem 3D thanh cac tu dien Delaunay.

    Uu tien dung extension CUDA da bien dich trong
    ``submodules/gof_cuda`` (dua tren CGAL, giong ``tetranerf.utils.
    extension.cpp.triangulate`` cua GOF) vi no xu ly duoc hang trieu diem.
    Neu chua build, fallback ve ``scipy.spatial.Delaunay`` tren CPU —
    CHI phu hop de test nhanh tren mot ROI/object nho (vai nghin - vai
    chuc nghin diem), khong dung cho toan bo scene lon.
    """
    try:
        from gof_cuda import cpp as _gof_cpp  # type: ignore

        cells = _gof_cpp.triangulate(points)
        return cells.to(points.device).long()
    except ImportError:
        pass

    try:
        from tetranerf.utils.extension import cpp as _tetranerf_cpp  # type: ignore

        cells = _tetranerf_cpp.triangulate(points)
        return cells.to(points.device).long()
    except ImportError:
        pass

    try:
        from scipy.spatial import Delaunay as _ScipyDelaunay
    except ImportError as exc:  # pragma: no cover - phu thuoc moi truong
        raise ImportError(
            "Khong tim thay extension tam giac phan hoa nao. Can build "
            "submodules/gof_cuda (hoac vendor tetra-triangulation tu "
            "Based_Model/Gaussian Opacity Fields (GOF)/"
            "gaussian-opacity-fields/submodules/tetra-triangulation), "
            "hoac cai `scipy` de dung fallback CPU khi test nhanh."
        ) from exc

    tri = _ScipyDelaunay(points.detach().cpu().numpy())
    cells = torch.from_numpy(tri.simplices.astype(np.int64))
    return cells.to(points.device)


# ---------------------------------------------------------------------------
# 2. SDF tu opacity field
# ---------------------------------------------------------------------------


def sdf_from_alpha(alpha: torch.Tensor, iso_level: float = 0.5) -> torch.Tensor:
    """Chuyen opacity field alpha in [0, 1] thanh SDF quanh isosurface
    ``iso_level`` (mac dinh 0.5 theo dinh nghia cua GOF)."""
    return alpha - iso_level


# ---------------------------------------------------------------------------
# 3. Marching Tetrahedra (vendor tu kaolin, Apache-2.0)
# ---------------------------------------------------------------------------

_TRIANGLE_TABLE = torch.tensor(
    [
        [-1, -1, -1, -1, -1, -1],
        [1, 0, 2, -1, -1, -1],
        [4, 0, 3, -1, -1, -1],
        [1, 4, 2, 1, 3, 4],
        [3, 1, 5, -1, -1, -1],
        [2, 3, 0, 2, 5, 3],
        [1, 4, 0, 1, 5, 4],
        [4, 2, 5, -1, -1, -1],
        [4, 5, 2, -1, -1, -1],
        [4, 1, 0, 4, 5, 1],
        [3, 2, 0, 3, 5, 2],
        [1, 3, 5, -1, -1, -1],
        [4, 1, 2, 4, 3, 1],
        [3, 0, 4, -1, -1, -1],
        [2, 0, 1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1],
    ],
    dtype=torch.long,
)
_NUM_TRIANGLES_TABLE = torch.tensor([0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0], dtype=torch.long)
_BASE_TET_EDGES = torch.tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long)
_V_ID = torch.pow(2, torch.arange(4, dtype=torch.long))


def _unbatched_marching_tetrahedra(
    vertices: torch.Tensor,
    tets: torch.Tensor,
    sdf: torch.Tensor,
    scales: torch.Tensor,
    chunk_size: int = 32 * 1024 * 1024,
):
    device = vertices.device

    if tets.shape[0] > chunk_size:
        merged_verts = None
        merged_scales = None
        merged_faces = None
        merged_verts_ids = None
        for tet_chunk in torch.chunk(tets, tets.shape[0] // chunk_size + 1):
            torch.cuda.empty_cache()
            verts, verts_scales, faces, verts_ids = _unbatched_marching_tetrahedra(
                vertices, tet_chunk, sdf, scales, chunk_size
            )
            if merged_verts is None:
                merged_verts, merged_scales, merged_faces, merged_verts_ids = (
                    verts,
                    verts_scales,
                    faces,
                    verts_ids,
                )
            else:
                all_edges = torch.cat([merged_verts_ids, verts_ids], dim=0)
                unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)

                unique_verts_0 = torch.zeros((unique_edges.shape[0], 2, 3), dtype=torch.float, device=device)
                unique_verts_1 = torch.zeros((unique_edges.shape[0], 2, 1), dtype=torch.float, device=device)
                unique_verts_0[idx_map[: merged_verts[0].shape[0]]] = merged_verts[0]
                unique_verts_0[idx_map[merged_verts[0].shape[0] :]] = verts[0]
                unique_verts_1[idx_map[: merged_verts[0].shape[0]]] = merged_verts[1]
                unique_verts_1[idx_map[merged_verts[0].shape[0] :]] = verts[1]

                unique_scales = torch.zeros((unique_edges.shape[0], 2, 1), dtype=torch.float, device=device)
                unique_scales[idx_map[: merged_verts[0].shape[0]]] = merged_scales
                unique_scales[idx_map[merged_verts[0].shape[0] :]] = verts_scales

                unique_faces_0 = idx_map[merged_faces.reshape(-1)].reshape(-1, 3)
                unique_faces_1 = idx_map[faces.reshape(-1) + merged_verts[0].shape[0]].reshape(-1, 3)

                merged_faces = torch.cat([unique_faces_0, unique_faces_1], dim=0)
                merged_verts = (unique_verts_0, unique_verts_1)
                merged_scales = unique_scales
                merged_verts_ids = unique_edges
                torch.cuda.empty_cache()

        return merged_verts, merged_scales, merged_faces, merged_verts_ids

    with torch.no_grad():
        occ_n = sdf > 0
        occ_fx4 = occ_n[tets.reshape(-1)].reshape(-1, 4)
        occ_sum = torch.sum(occ_fx4, -1)

        valid_tets = (occ_sum > 0) & (occ_sum < 4)

        all_edges = tets[valid_tets][:, _BASE_TET_EDGES.to(device)].reshape(-1, 2)

        order = (all_edges[:, 0] > all_edges[:, 1]).bool()
        all_edges[order] = all_edges[order][:, [1, 0]]

        unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)
        unique_edges = unique_edges.long()

        mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1
        mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device=device) * -1
        mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long, device=device)
        idx_map = mapping[idx_map]

        interp_v = unique_edges[mask_edges]

    edges_to_interp = vertices[interp_v.reshape(-1)].reshape(-1, 2, 3)
    edges_to_interp_sdf = sdf[interp_v.reshape(-1)].reshape(-1, 2, 1)
    verts_scales = scales[interp_v.reshape(-1)].reshape(-1, 2, 1)

    verts = (edges_to_interp, edges_to_interp_sdf)
    idx_map = idx_map.reshape(-1, 6)

    tetindex = (occ_fx4[valid_tets] * _V_ID.to(device).unsqueeze(0)).sum(-1)
    num_triangles = _NUM_TRIANGLES_TABLE.to(device)[tetindex]
    triangle_table_device = _TRIANGLE_TABLE.to(device)

    faces = torch.cat(
        (
            torch.gather(
                input=idx_map[num_triangles == 1],
                dim=1,
                index=triangle_table_device[tetindex[num_triangles == 1]][:, :3],
            ).reshape(-1, 3),
            torch.gather(
                input=idx_map[num_triangles == 2],
                dim=1,
                index=triangle_table_device[tetindex[num_triangles == 2]][:, :6],
            ).reshape(-1, 3),
        ),
        dim=0,
    )

    return verts, verts_scales, faces, interp_v


def marching_tetrahedra(
    vertices: torch.Tensor,
    tets: torch.Tensor,
    sdf: torch.Tensor,
    scales: torch.Tensor,
    chunk_size: int = 32 * 1024 * 1024,
) -> Tuple[List, List, List, List]:
    """Chuyen SDF roi rac tren luoi tu dien thanh mesh tam giac (khu vi
    phan duoc theo vi tri dinh va gia tri SDF).

    Args:
        vertices: (B, N, 3) toa do dinh cua luoi tu dien, theo batch.
        tets: (T, 4) chi so 4 dinh cua moi tu dien (khong batch).
        sdf: (B, N) gia tri SDF tai moi dinh.
        scales: (B, N) scale cua Gaussian gan nhat voi moi dinh (dung de
            loc mesh o buoc hau xu ly).
        chunk_size: so tu dien xu ly toi da moi lan de tranh OOM tren GPU
            (configs/gof_meshing.toml::chunk_size).

    Returns:
        verts_list, scale_list, faces_list, edge_ids_list — moi phan tu
        ung voi mot item trong batch.
    """
    outputs = [
        _unbatched_marching_tetrahedra(vertices[b], tets, sdf[b], scales[b], chunk_size)
        for b in range(vertices.shape[0])
    ]
    return list(zip(*outputs))
