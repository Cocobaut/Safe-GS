"""
Phần GaussianModel của GOF (Based_Model/GOF/scene/gaussian_model.py) cần cho trích xuất mesh:
thuộc tính scale/opacity có 3D filter, compute_3D_filter, get_tetra_points, get_frustum_mask.
Đọc ply 3DGS chuẩn (InFusion: sh_degree 0, 3DGIC: sh_degree 3; thuộc tính thừa như obj_dc_* bị bỏ qua).
"""
from typing import List

import numpy as np
import torch
import trimesh
from einops import einsum
from plyfile import PlyData

from .cameras import MeshCamera


def build_rotation(r):
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

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
def get_frustum_mask(points: torch.Tensor, cameras: List[MeshCamera], near: float = 0.02, far: float = 1e6,
                     chunk_size: int = 4):
    # Xử lý theo lô camera rồi OR lại (giống bản GOF trong repo) để tránh cấp phát tensor (n_view, N, 3) quá lớn
    H, W = cameras[0].image_height, cameras[0].image_width
    mask = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)

    ones = torch.ones_like(points[:, 0]).unsqueeze(-1)
    homo_points = torch.cat([points, ones], dim=-1)

    for start in range(0, len(cameras), chunk_size):
        chunk_cams = cameras[start:start + chunk_size]

        intrinsics = torch.stack(
            [
                torch.Tensor(
                    [[cam.focal_x, 0, W / 2],
                     [0, cam.focal_y, H / 2],
                     [0, 0, 1]]
                ) for cam in chunk_cams
            ],
            dim=0
        ).to(points.device)

        view_matrices = torch.stack(
            [cam.world_view_transform for cam in chunk_cams], dim=0
        ).transpose(1, 2)

        view_points = einsum(view_matrices, homo_points, "n_view b c, N c -> n_view N b")
        view_points = view_points[:, :, :3]

        uv_points = einsum(intrinsics, view_points, "n_view b c, n_view N c -> n_view N b")

        z = uv_points[:, :, -1:]
        uv_points = uv_points[:, :, :2] / z
        u, v = uv_points[:, :, 0], uv_points[:, :, 1]

        depth = view_points[:, :, -1]
        cull_near_fars = (depth >= near) & (depth <= far)

        chunk_mask = torch.any(cull_near_fars & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1), dim=0)
        mask = mask | chunk_mask
        del intrinsics, view_matrices, view_points, uv_points, z, u, v, depth, cull_near_fars, chunk_mask
        torch.cuda.empty_cache()

    return mask


class GaussianModel:
    def __init__(self, sh_degree: int = 0):
        self.max_sh_degree = sh_degree
        self.active_sh_degree = sh_degree

    # --- activations như 3DGS / GOF ---
    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_scaling_with_3D_filter(self):
        scales = self.get_scaling
        scales = torch.square(scales) + torch.square(self.filter_3D)
        return torch.sqrt(scales)

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    @property
    def get_opacity_with_3D_filter(self):
        opacity = torch.sigmoid(self._opacity)
        scales_square = torch.square(self.get_scaling)
        det1 = scales_square.prod(dim=1)
        scales_after_square = scales_square + torch.square(self.filter_3D)
        det2 = scales_after_square.prod(dim=1)
        coef = torch.sqrt(det1 / det2)
        return opacity * coef[..., None]

    def load_ply(self, path):
        plydata = PlyData.read(path)
        v = plydata.elements[0]
        xyz = np.stack((np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])), axis=1)
        opacities = np.asarray(v["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        for i in range(3):
            features_dc[:, i, 0] = np.asarray(v[f"f_dc_{i}"])

        extra_f_names = sorted([p.name for p in v.properties if p.name.startswith("f_rest_")], key=lambda x: int(x.split('_')[-1]))
        # Suy ra bậc SH từ số hệ số f_rest (InFusion dùng sh_degree 0 -> không có f_rest)
        n_coeffs = len(extra_f_names) // 3 + 1
        self.max_sh_degree = self.active_sh_degree = int(round(n_coeffs ** 0.5)) - 1
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(v[attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, n_coeffs - 1))

        scale_names = sorted([p.name for p in v.properties if p.name.startswith("scale_")], key=lambda x: int(x.split('_')[-1]))
        scales = np.stack([np.asarray(v[n]) for n in scale_names], axis=1)
        rot_names = sorted([p.name for p in v.properties if p.name.startswith("rot")], key=lambda x: int(x.split('_')[-1]))
        rots = np.stack([np.asarray(v[n]) for n in rot_names], axis=1)

        names = {p.name for p in v.properties}
        filter_3D = np.asarray(v["filter_3D"])[..., np.newaxis] if "filter_3D" in names else np.zeros((xyz.shape[0], 1))

        self._xyz = torch.tensor(xyz, dtype=torch.float, device="cuda")
        self._features_dc = torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
        self._features_rest = torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
        self._opacity = torch.tensor(opacities, dtype=torch.float, device="cuda")
        self._scaling = torch.tensor(scales, dtype=torch.float, device="cuda")
        self._rotation = torch.tensor(rots, dtype=torch.float, device="cuda")
        self.filter_3D = torch.tensor(filter_3D, dtype=torch.float, device="cuda")

    @torch.no_grad()
    def compute_3D_filter(self, cameras):
        # Copy từ GOF: bán kính lọc 3D theo khoảng cách gần nhất tới camera nhìn thấy Gaussian
        print("Computing 3D filter")
        xyz = self.get_xyz
        distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
        valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)

        focal_length = 0.
        for camera in cameras:
            R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float32)
            T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float32)
            xyz_cam = xyz @ R + T[None, :]

            valid_depth = xyz_cam[:, 2] > 0.2

            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z = torch.clamp(z, min=0.001)

            x = x / z * camera.focal_x + camera.image_width / 2.0
            y = y / z * camera.focal_y + camera.image_height / 2.0

            in_screen = torch.logical_and(torch.logical_and(x >= -0.15 * camera.image_width, x <= camera.image_width * 1.15),
                                          torch.logical_and(y >= -0.15 * camera.image_height, y <= 1.15 * camera.image_height))

            valid = torch.logical_and(valid_depth, in_screen)
            distance[valid] = torch.min(distance[valid], z[valid])
            valid_points = torch.logical_or(valid_points, valid)
            if focal_length < camera.focal_x:
                focal_length = camera.focal_x

        distance[~valid_points] = distance[valid_points].max()
        filter_3D = distance / focal_length * (0.2 ** 0.5)
        self.filter_3D = filter_3D[..., None]

    @torch.no_grad()
    def get_tetra_points(self, views: List[MeshCamera], near: float = 0.02, far: float = 1e6, opacity_threshold: float = 0.1):
        M = trimesh.creation.box()
        M.vertices *= 2

        rots = build_rotation(self._rotation)
        xyz = self.get_xyz
        scale = self.get_scaling_with_3D_filter * 3.

        # Bỏ Gaussian gần trong suốt (GOF dùng 0.1) để giảm số điểm Delaunay
        opacity = self.get_opacity_with_3D_filter
        mask = (opacity > opacity_threshold).squeeze(-1)
        xyz = xyz[mask]
        scale = scale[mask]
        rots = rots[mask]

        vertices = M.vertices.T
        vertices = torch.from_numpy(vertices).float().cuda().unsqueeze(0).repeat(xyz.shape[0], 1, 1)
        vertices = vertices * scale.unsqueeze(-1)
        vertices = torch.bmm(rots, vertices).squeeze(-1) + xyz.unsqueeze(-1)
        vertices = vertices.permute(0, 2, 1).reshape(-1, 3).contiguous()
        vertices = torch.cat([vertices, xyz], dim=0)

        scale = scale.max(dim=-1, keepdim=True)[0]
        scale_corner = scale.repeat(1, 8).reshape(-1, 1)
        vertices_scale = torch.cat([scale_corner, scale], dim=0)

        vertex_mask = get_frustum_mask(vertices, views, near, far)
        return vertices[vertex_mask], vertices_scale[vertex_mask]
