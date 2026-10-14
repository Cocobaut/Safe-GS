"""
Trích xuất mesh từ Gaussians (InFusion / 3DGIC) bằng logic của GOF (Based_Model/GOF/extract_mesh.py):
  1. Sinh điểm tứ diện từ các Gaussian (tâm + 8 góc box 3-sigma), lọc theo frustum camera
  2. Tam giác hóa Delaunay (tetra-triangulation / CGAL)
  3. Tính opacity tại mỗi đỉnh = 1 - min_view(transmittance) qua integrate() -> sdf = alpha - 0.5
  4. Marching tetrahedra + binary search 8 bước để đặt đỉnh đúng mặt mức 0.5

    python extract_mesh.py -m <model_path InFusion> --iteration 150 --output_dir <thư mục mesh> --texture_mesh
    3DGIC (ply inpaint nằm ngoài point_cloud/, cameras.json ở model 3DGS gốc):
    python extract_mesh.py --ply_path <.../3dgic_iteration_9999/point_cloud.ply> --cameras_json <.../3dgs/cameras.json>         --output_dir <thư mục mesh> --texture_mesh
"""
import os
import random
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import trimesh
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gof.cameras import load_cameras_json  # noqa: E402
from gof.gaussian_model import GaussianModel  # noqa: E402
from gof.io_utils import unique_path  # noqa: E402
from gof.renderer import integrate  # noqa: E402
from gof.tetmesh import marching_tetrahedra  # noqa: E402
from tetranerf.utils.extension import cpp  # noqa: E402


@torch.no_grad()
def evaluage_alpha(points, views, gaussians, background, kernel_size, return_color=False):
    final_alpha = torch.ones((points.shape[0]), dtype=torch.float32, device="cuda")
    if return_color:
        final_color = torch.ones((points.shape[0], 3), dtype=torch.float32, device="cuda")

    for view in tqdm(views, desc="Rendering progress"):
        ret = integrate(points, view, gaussians, background, kernel_size=kernel_size)
        alpha_integrated = ret["alpha_integrated"]
        if return_color:
            color_integrated = ret["color_integrated"]
            final_color = torch.where((alpha_integrated < final_alpha).reshape(-1, 1), color_integrated, final_color)
        final_alpha = torch.min(final_alpha, alpha_integrated)

    alpha = 1 - final_alpha
    if return_color:
        return alpha, final_color
    return alpha


@torch.no_grad()
def marching_tetrahedra_with_binary_search(out_dir, views, gaussians, background, kernel_size, filter_mesh, texture_mesh,
                                           near, far, opacity_threshold, mesh_name):
    os.makedirs(out_dir, exist_ok=True)

    points, points_scale = gaussians.get_tetra_points(views, near, far, opacity_threshold)
    print(f"[i] {points.shape[0]} điểm tứ diện")
    cells_path = os.path.join(out_dir, "cells.pt")
    if os.path.exists(cells_path):
        print("load existing cells")
        cells = torch.load(cells_path)
    else:
        print("create cells and save")
        cells = cpp.triangulate(points)
        torch.save(cells, cells_path)

    alpha = evaluage_alpha(points, views, gaussians, background, kernel_size)

    vertices = points.cuda()[None]
    tets = cells.cuda().long()
    print(vertices.shape, tets.shape, alpha.shape)

    def alpha_to_sdf(alpha):
        sdf = alpha - 0.5
        return sdf[None]

    sdf = alpha_to_sdf(alpha)

    torch.cuda.empty_cache()
    verts_list, scale_list, faces_list, _ = marching_tetrahedra(vertices, tets, sdf, points_scale[None])
    torch.cuda.empty_cache()

    end_points, end_sdf = verts_list[0]
    end_scales = scale_list[0]

    faces = faces_list[0].cpu().numpy()

    left_points = end_points[:, 0, :]
    right_points = end_points[:, 1, :]
    left_sdf = end_sdf[:, 0, :]
    right_sdf = end_sdf[:, 1, :]
    left_scale = end_scales[:, 0, 0]
    right_scale = end_scales[:, 1, 0]
    distance = torch.norm(left_points - right_points, dim=-1)
    scale = left_scale + right_scale

    n_binary_steps = 8
    mesh_path = None
    for step in range(n_binary_steps):
        print("binary search in step {}".format(step))
        mid_points = (left_points + right_points) / 2
        alpha = evaluage_alpha(mid_points, views, gaussians, background, kernel_size)
        mid_sdf = alpha_to_sdf(alpha).squeeze().unsqueeze(-1)

        ind_low = ((mid_sdf < 0) & (left_sdf < 0)) | ((mid_sdf > 0) & (left_sdf > 0))

        left_sdf[ind_low] = mid_sdf[ind_low]
        right_sdf[~ind_low] = mid_sdf[~ind_low]
        left_points[ind_low.flatten()] = mid_points[ind_low.flatten()]
        right_points[~ind_low.flatten()] = mid_points[~ind_low.flatten()]

        points = (left_points + right_points) / 2
        if step != n_binary_steps - 1:
            continue

        if texture_mesh:
            _, color = evaluage_alpha(points, views, gaussians, background, kernel_size, return_color=True)
            vertex_colors = (color.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        else:
            vertex_colors = None
        mesh = trimesh.Trimesh(vertices=points.cpu().numpy(), faces=faces, vertex_colors=vertex_colors, process=False)

        if filter_mesh:
            mask = (distance <= scale).cpu().numpy()
            face_mask = mask[faces].all(axis=1)
            mesh.update_vertices(mask)
            mesh.update_faces(face_mask)

        mesh_path = unique_path(os.path.join(out_dir, mesh_name))
        mesh.export(mesh_path)
        print(f"[✓] Mesh: {mesh_path} ({len(mesh.vertices)} đỉnh, {len(mesh.faces)} mặt)")
    return mesh_path


def main():
    parser = ArgumentParser(description="Trích xuất mesh GOF từ Gaussians của InFusion / 3DGIC")
    parser.add_argument("-m", "--model_path", default="", help="Thư mục model InFusion (chứa point_cloud/, cameras.json)")
    parser.add_argument("--ply_path", default="", help="Đường dẫn ply trực tiếp (thay cho -m/--iteration), ví dụ ply 3DGIC")
    parser.add_argument("--cameras_json", default="", help="cameras.json (mặc định <model_path>/cameras.json)")
    parser.add_argument("--iteration", default=150, type=int, help="Iteration của point cloud (InFusion cuối cùng: 150)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mesh_name", default="mesh_gof.ply")
    parser.add_argument("--filter_mesh", action="store_true", help="Bỏ mặt có cạnh dài hơn scale Gaussian (như GOF)")
    parser.add_argument("--texture_mesh", action="store_true", help="Gán màu đỉnh từ Gaussians")
    parser.add_argument("--near", default=0.02, type=float)
    parser.add_argument("--far", default=1e6, type=float)
    parser.add_argument("--kernel_size", default=0.0, type=float, help="Bộ lọc 2D của rasterizer GOF (GOF mặc định 0.0)")
    parser.add_argument("--use_3d_filter", action="store_true",
                        help="Tính 3D filter của GOF. InFusion train không có 3D filter nên mặc định tắt (filter_3D = 0)")
    parser.add_argument("--opacity_threshold", default=0.1, type=float, help="Bỏ Gaussian có opacity thấp hơn (GOF: 0.1)")
    parser.add_argument("--view_stride", default=1, type=int, help="Dùng 1/N camera để tăng tốc")
    parser.add_argument("--white_background", action="store_true")
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

    cameras_json = args.cameras_json or os.path.join(args.model_path, "cameras.json")
    ply_path = args.ply_path or os.path.join(args.model_path, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply")
    views = load_cameras_json(cameras_json, args.view_stride)
    print(f"[i] {len(views)} camera | ply {ply_path}")
    gaussians = GaussianModel()
    gaussians.load_ply(ply_path)
    print(f"[i] {gaussians.get_xyz.shape[0]} Gaussians, sh_degree {gaussians.max_sh_degree}")
    if args.use_3d_filter:
        gaussians.compute_3D_filter(views)

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    marching_tetrahedra_with_binary_search(args.output_dir, views, gaussians, background, args.kernel_size,
                                           args.filter_mesh, args.texture_mesh, args.near, args.far,
                                           args.opacity_threshold, args.mesh_name)


if __name__ == "__main__":
    main()
