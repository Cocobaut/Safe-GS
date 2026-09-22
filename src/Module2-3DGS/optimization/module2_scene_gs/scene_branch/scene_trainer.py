import os
import sys
import math
import random
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from plyfile import PlyData, PlyElement
from typing import Tuple, Union

try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    from simple_knn._C import distCUDA2
except ImportError as e:
    raise ImportError(
        "Chưa cài diff-gaussian-rasterization hoặc simple-knn trong env hiện tại. "
        "Cài bằng: pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git "
        "và pip install git+https://gitlab.inria.fr/bkerbl/simple-knn.git (cần --no-build-isolation)."
    ) from e

from src.common.config_loader import load_toml_config
from src.module2_scene_gs.scene_branch.camera_scheduler import CameraScheduler
from src.module2_scene_gs.scene_branch.camera_utils import load_training_cameras
from src.module2_scene_gs.scene_branch.loss_utils import l1_loss, ssim


def get_expon_lr_func(lr_init, lr_final, max_steps):
    """Tính learning rate suy giảm hàm mũ theo số vòng lặp."""
    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        t = np.clip(step / max_steps, 0.0, 1.0)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return log_lerp
    return helper


def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Chuyển quaternion (N,4) dạng [w,x,y,z] sang ma trận xoay (N,3,3)."""
    q = torch.nn.functional.normalize(q, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    N = q.shape[0]
    R = torch.zeros((N, 3, 3), device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


class GaussianSceneModel:
    """
    Quản lý các thuộc tính hình học và tối ưu của mây hạt 3D Gaussian.
    """
    def __init__(self, sh_degree: int = 3):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.optimizer = None

        # Trạng thái cho Adaptive Density Control (densify/prune)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.percent_dense = 0.01
        self.spatial_lr_scale = 1.0
        self.filter_3D = torch.empty(0)

    def create_from_pcd(self, xyz: np.ndarray, rgb: np.ndarray, spatial_lr_scale: float = 1.0):
        """Khởi tạo các hạt Gaussian từ mây điểm thưa SfM."""
        device = "cuda"
        fused_point_cloud = torch.tensor(xyz, dtype=torch.float, device=device)
        fused_color = torch.tensor(rgb / 255.0, dtype=torch.float, device=device)

        # Chuyển đổi RGB sang Spherical Harmonics bậc 0 (DC)
        C0 = 0.28209479177387814
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2), device=device)
        features[:, :3, 0] = (fused_color - 0.5) / C0

        # Khoảng cách lân cận qua KNN để khởi tạo tỉ lệ hạt (Scale)
        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

        # Quaternions khởi tạo quay ngẫu nhiên dạng đơn vị [1, 0, 0, 0]
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=device)
        rots[:, 0] = 1

        # Độ đục ban đầu khởi tạo qua inverse sigmoid (khoảng 0.1)
        opacities = torch.logit(0.1 * torch.ones((fused_point_cloud.shape[0], 1), device=device))

        self._xyz = torch.nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = torch.nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = torch.nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = torch.nn.Parameter(scales.requires_grad_(True))
        self._rotation = torch.nn.Parameter(rots.requires_grad_(True))
        self._opacity = torch.nn.Parameter(opacities.requires_grad_(True))

        self.spatial_lr_scale = spatial_lr_scale
        n = fused_point_cloud.shape[0]
        self.xyz_gradient_accum = torch.zeros((n, 1), device=device)
        self.denom = torch.zeros((n, 1), device=device)
        self.max_radii2D = torch.zeros(n, device=device)

        self.setup_optimizer(spatial_lr_scale)

    def setup_optimizer(self, spatial_lr_scale: float):
        l = [
            {'params': [self._xyz], 'lr': 0.00016 * spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': 0.0025, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': 0.0025 / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': 0.05, "name": "opacity"},
            {'params': [self._scaling], 'lr': 0.005, "name": "scaling"},
            {'params': [self._rotation], 'lr': 0.001, "name": "rotation"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    @property
    def get_scaling_with_3D_filter(self):
        """Scale sau khi cong them filter_3D (chuan GOF - Mip-Splatting) - can goi compute_3D_filter() truoc."""
        scales = self.get_scaling
        scales = torch.square(scales) + torch.square(self.filter_3D)
        return torch.sqrt(scales)

    @property
    def get_opacity_with_3D_filter(self):
        """Opacity da bu tru theo he so filter_3D de giu do sang khong doi khi lam mo hat qua nho."""
        opacity = self.get_opacity
        scales = self.get_scaling
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        scales_after_square = scales_square + torch.square(self.filter_3D)
        det2 = scales_after_square.prod(dim=1)
        coef = torch.sqrt(det1 / det2)
        return opacity * coef[..., None]

    @torch.no_grad()
    def compute_3D_filter(self, cameras):
        """Tinh filter_3D tu vi tri Gaussian + pose/intrinsics camera da dung de train (khong can train
        lai) - port dung cong thuc goc GOF: filter = khoang_cach_gan_nhat_tu_camera_nhin_thay / focal * sqrt(0.2)."""
        xyz = self.get_xyz
        distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
        valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)

        focal_length = 0.0
        for camera in cameras:
            R_wc = camera.world_view_transform[:3, :3].T
            t_wc = camera.world_view_transform[3, :3]
            xyz_cam = xyz @ R_wc + t_wc[None, :]

            valid_depth = xyz_cam[:, 2] > 0.2

            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z_clamped = torch.clamp(z, min=0.001)

            tanfovx = math.tan(camera.fovx * 0.5)
            tanfovy = math.tan(camera.fovy * 0.5)
            focal_x = camera.image_width / (2.0 * tanfovx)
            focal_y = camera.image_height / (2.0 * tanfovy)

            x_screen = x / z_clamped * focal_x + camera.image_width / 2.0
            y_screen = y / z_clamped * focal_y + camera.image_height / 2.0

            in_screen = torch.logical_and(
                torch.logical_and(x_screen >= -0.15 * camera.image_width, x_screen <= camera.image_width * 1.15),
                torch.logical_and(y_screen >= -0.15 * camera.image_height, y_screen <= 1.15 * camera.image_height),
            )

            valid = torch.logical_and(valid_depth, in_screen)
            distance[valid] = torch.min(distance[valid], z[valid])
            valid_points = torch.logical_or(valid_points, valid)
            if focal_length < focal_x:
                focal_length = focal_x

        distance[~valid_points] = distance[valid_points].max()
        filter_3D = distance / focal_length * (0.2 ** 0.5)
        self.filter_3D = filter_3D[..., None]
        print(f"[Loading] Đã tính filter_3D cho {xyz.shape[0]} hạt Gaussian (dùng {len(cameras)} camera).")

    def _replace_tensor_in_optimizer(self, tensor, name):
        for group in self.optimizer.param_groups:
            if group["name"] != name:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            stored_state["exp_avg"] = torch.zeros_like(tensor)
            stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
            del self.optimizer.state[group['params'][0]]
            group["params"][0] = torch.nn.Parameter(tensor.requires_grad_(True))
            self.optimizer.state[group['params'][0]] = stored_state
            return group["params"][0]

    def reset_opacity(self):
        """Đặt lại toàn bộ opacity về giá trị thấp để loại bỏ 'hạt ma' (floaters)."""
        opacities_new = torch.logit(torch.clamp(self.get_opacity, max=0.01))
        new_param = self._replace_tensor_in_optimizer(opacities_new, "opacity")
        self._opacity = new_param

    def _prune_optimizer(self, mask):
        optimizable = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = torch.nn.Parameter((group["params"][0][mask]).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
            else:
                group["params"][0] = torch.nn.Parameter(group["params"][0][mask].requires_grad_(True))
            optimizable[group["name"]] = group["params"][0]
        return optimizable

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable["xyz"]
        self._features_dc = optimizable["f_dc"]
        self._features_rest = optimizable["f_rest"]
        self._opacity = optimizable["opacity"]
        self._scaling = optimizable["scaling"]
        self._rotation = optimizable["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def _cat_tensors_to_optimizer(self, tensors_dict):
        optimizable = {}
        for group in self.optimizer.param_groups:
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = torch.nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
            else:
                group["params"][0] = torch.nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
            optimizable[group["name"]] = group["params"][0]
        return optimizable

    def _densification_postfix(self, new_xyz, new_f_dc, new_f_rest, new_opacities, new_scaling, new_rotation):
        tensors_dict = {
            "xyz": new_xyz, "f_dc": new_f_dc, "f_rest": new_f_rest,
            "opacity": new_opacities, "scaling": new_scaling, "rotation": new_rotation,
        }
        optimizable = self._cat_tensors_to_optimizer(tensors_dict)
        self._xyz = optimizable["xyz"]
        self._features_dc = optimizable["f_dc"]
        self._features_rest = optimizable["f_rest"]
        self._opacity = optimizable["opacity"]
        self._scaling = optimizable["scaling"]
        self._rotation = optimizable["rotation"]

        n = self._xyz.shape[0]
        device = self._xyz.device
        self.xyz_gradient_accum = torch.zeros((n, 1), device=device)
        self.denom = torch.zeros((n, 1), device=device)
        self.max_radii2D = torch.zeros(n, device=device)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, region_mask=None):
        """region_mask (bool, N): neu co, chi cho phep clone hat nam trong vung nay (vd ROI cua 1 object) - dung cho Object-GS confined densification."""
        selected_pts_mask = torch.norm(grads, dim=-1) >= grad_threshold
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent,
        )
        if region_mask is not None:
            selected_pts_mask = torch.logical_and(selected_pts_mask, region_mask)
        new_xyz = self._xyz[selected_pts_mask]
        new_f_dc = self._features_dc[selected_pts_mask]
        new_f_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        self._densification_postfix(new_xyz, new_f_dc, new_f_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, region_mask=None):
        """region_mask (bool, N): neu co, chi cho phep split hat nam trong vung nay - dung cho Object-GS confined densification."""
        n_init_points = self._xyz.shape[0]
        padded_grad = torch.zeros(n_init_points, device=self._xyz.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = padded_grad >= grad_threshold
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent,
        )
        if region_mask is not None:
            selected_pts_mask = torch.logical_and(selected_pts_mask, region_mask)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=self._xyz.device)
        samples = torch.normal(mean=means, std=stds)
        rots = _quat_to_rotmat(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = torch.log(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_f_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_f_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self._densification_postfix(new_xyz, new_f_dc, new_f_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((
            selected_pts_mask,
            torch.zeros(N * selected_pts_mask.sum(), device=self._xyz.device, dtype=bool),
        ))
        self.prune_points(prune_filter)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, region_mask_fn=None):
        """region_mask_fn (callable: xyz -> bool mask N) neu co, densify (clone/split ra hat MOI)
        chi xay ra trong vung nay - dung cho Object-GS: cai thien 1 object trong scene ma khong dung
        vao phan con lai. La CALLABLE (khong phai tensor san) vi so luong hat thay doi ngay giua
        clone va split (clone them hat truoc) nen phai tinh lai mask theo dung kich thuoc hien tai
        moi lan goi. Prune (xoa hat opacity thap) van ap dung toan cuc nhu binh thuong."""
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        region_mask = region_mask_fn(self.get_xyz) if region_mask_fn is not None else None
        self.densify_and_clone(grads, max_grad, extent, region_mask=region_mask)

        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        region_mask = region_mask_fn(self.get_xyz) if region_mask_fn is not None else None
        self.densify_and_split(grads, max_grad, extent, region_mask=region_mask)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = torch.max(self.get_scaling, dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def save_ply(self, path: Union[str, Path]):
        """Xuất mây hạt Gaussian ra định dạng PLY tiêu chuẩn."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(str(path))
        print(f"[✓] Đã lưu Scene-GS checkpoint ({xyz.shape[0]} Gaussians) tại: {path}")

    @staticmethod
    def load_ply(path: Union[str, Path], sh_degree: int = 3, device: str = "cuda") -> "GaussianSceneModel":
        """Load lại checkpoint .ply đã lưu bởi save_ply (dùng cho render sau khi train xong)."""
        plydata = PlyData.read(str(path))
        v = plydata["vertex"]
        prop_names = [p.name for p in v.properties]

        xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
        opacities = np.asarray(v["opacity"], dtype=np.float32)[..., None]

        f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)

        n_rest = sum(1 for n in prop_names if n.startswith("f_rest_"))
        f_rest = np.stack([v[f"f_rest_{i}"] for i in range(n_rest)], axis=1).astype(np.float32)
        # save_ply ghi theo thu tu kenh-mau-truoc (transpose(1,2).flatten): 15 he so kenh R, roi 15 kenh G, roi 15 kenh B.
        # Phai reshape (-1, 3, 15) roi transpose lai (-1, 15, 3) moi dung thu tu (coef, kenh) ban dau.
        f_rest = f_rest.reshape((-1, 3, n_rest // 3)).transpose(0, 2, 1)

        n_scale = sum(1 for n in prop_names if n.startswith("scale_"))
        scales = np.stack([v[f"scale_{i}"] for i in range(n_scale)], axis=1).astype(np.float32)

        n_rot = sum(1 for n in prop_names if n.startswith("rot_"))
        rots = np.stack([v[f"rot_{i}"] for i in range(n_rot)], axis=1).astype(np.float32)

        model = GaussianSceneModel(sh_degree=sh_degree)
        model.active_sh_degree = sh_degree
        model._xyz = torch.tensor(xyz, device=device)
        model._features_dc = torch.tensor(f_dc, device=device).unsqueeze(1)
        model._features_rest = torch.tensor(f_rest, device=device)
        model._opacity = torch.tensor(opacities, device=device)
        model._scaling = torch.tensor(scales, device=device)
        model._rotation = torch.tensor(rots, device=device)
        print(f"[Loading] Đã load {xyz.shape[0]} Gaussians từ checkpoint: {path}")
        return model

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append(f'f_dc_{i}')
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append(f'f_rest_{i}')
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append(f'scale_{i}')
        for i in range(self._rotation.shape[1]):
            l.append(f'rot_{i}')
        return l


def render_gaussian_model(model: "GaussianSceneModel", camera, bg_color: torch.Tensor):
    """Render 1 ảnh từ 1 camera + 1 GaussianSceneModel bất kỳ (dùng chung cho train và render sau train)."""
    screenspace_points = torch.zeros_like(model.get_xyz, dtype=model.get_xyz.dtype,
                                           requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    tanfovx = math.tan(camera.fovx * 0.5)
    tanfovy = math.tan(camera.fovy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height),
        image_width=int(camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        sh_degree=model.active_sh_degree,
        campos=camera.camera_center,
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    shs = torch.cat((model._features_dc, model._features_rest), dim=1)

    rendered_image, radii = rasterizer(
        means3D=model.get_xyz,
        means2D=screenspace_points,
        shs=shs,
        colors_precomp=None,
        opacities=model.get_opacity,
        scales=model.get_scaling,
        rotations=model.get_rotation,
        cov3D_precomp=None,
    )
    visibility_filter = radii > 0
    return rendered_image, screenspace_points, radii, visibility_filter


class SceneGSTrainer:
    """
    Bộ điều khiển huấn luyện toàn cảnh Scene-GS (20k iterations).
    """
    def __init__(self, config_path: str = "configs/base_scene.toml", overrides: dict = None):
        self.cfg = load_toml_config(config_path)
        if overrides:
            self.cfg.update(overrides)

        self.iterations = self.cfg.get("iterations", 20000)
        self.output_dir = Path(self.cfg.get("workspace_dir", "data/workspace"))
        if "checkpoints_dir" in self.cfg:
            self.checkpoints_dir = Path(self.cfg["checkpoints_dir"])
        elif "3dgs" in self.output_dir.parts or self.output_dir.name == "checkpoints":
            self.checkpoints_dir = self.output_dir
        else:
            self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_tag = self.cfg.get("checkpoint_tag", "")


        self.model = GaussianSceneModel(sh_degree=self.cfg.get("sh_degree", 3))

    def load_sparse_points(self, sfm_sparse_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Đọc mây điểm thưa từ points3D.bin hoặc points3D.txt của COLMAP."""
        points3d_file = sfm_sparse_dir / "points3D.bin"
        if points3d_file.exists():
            import pycolmap
            recon = pycolmap.Reconstruction(sfm_sparse_dir)
            xyz = np.array([p.xyz for p in recon.points3D.values()])
            rgb = np.array([p.color for p in recon.points3D.values()])
            return xyz, rgb
        else:
            print("[Error] Cảnh báo: Không tìm thấy points3D.bin. Đang dùng mây điểm ngẫu nhiên để test...")
            xyz = np.random.uniform(-1.0, 1.0, (2000, 3))
            rgb = np.random.randint(0, 255, (2000, 3))
            return xyz, rgb

    def render(self, camera, bg_color: torch.Tensor):
        """Render 1 ảnh từ 1 camera bằng CUDA rasterizer. Trả về (ảnh render, viewspace_points, radii, visibility_filter)."""
        return render_gaussian_model(self.model, camera, bg_color)

    def train(self):
        """Vòng lặp huấn luyện chính Scene-GS: render thật + loss L1/D-SSIM + densify/prune + checkpoint 7k/30k."""
        sfm_dir = Path(self.cfg.get("sfm_dir", "outputs/workspace/dtu_scan24/sfm/sparse/0"))
        mask_meta = Path(self.cfg.get("mask_meta", "outputs/workspace/dtu_scan24/segmentation/segmentation_meta.json"))
        images_dir = Path(self.cfg.get("raw_image_dir", ""))

        save_iterations = set(self.cfg.get("save_iterations", [7000, 30000]))
        densify_from_iter = self.cfg.get("densify_from_iter", 500)
        densify_until_iter = self.cfg.get("densify_until_iter", 15000)
        densification_interval = self.cfg.get("densification_interval", 100)
        opacity_reset_interval = self.cfg.get("opacity_reset_interval", 3000)
        densify_grad_threshold = self.cfg.get("densify_grad_threshold", 0.0002)
        lambda_dssim = self.cfg.get("lambda_dssim", 0.2)

        # 1. Chọn camera theo chiến lược của CameraScheduler
        scheduler = CameraScheduler(
            sfm_dir=sfm_dir,
            mask_meta_path=mask_meta,
            close_up_ratio_threshold=self.cfg.get("close_up_ratio_threshold", 0.08)
        )
        selected_names = scheduler.schedule_scene_cameras(
            roi_sample_rate=self.cfg.get("roi_sample_rate", 0.5),
            output_file=self.output_dir / "scene_camera_schedule.json"
        )

        # 2. Khởi tạo mây hạt từ sparse points
        print("[Loading] Đang khởi tạo mô hình Gaussian từ mây điểm SfM...")
        import pycolmap
        rec = pycolmap.Reconstruction(sfm_dir)
        xyz, rgb = self.load_sparse_points(sfm_dir)
        scene_extent = float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)))
        self.model.create_from_pcd(xyz, rgb, spatial_lr_scale=self.cfg.get("spatial_lr_scale", 1.0))

        # 3. Load ảnh gốc + pose thật cho từng camera đã chọn
        print("[Loading] Đang load ảnh gốc + pose camera cho render...")
        viewpoint_cameras = load_training_cameras(rec, images_dir, selected_names, device="cuda")
        if len(viewpoint_cameras) == 0:
            raise RuntimeError(
                f"Không load được camera/ảnh nào để train. Kiểm tra lại raw_image_dir='{images_dir}' "
                f"có đúng chứa các ảnh trùng tên với ảnh trong '{sfm_dir}' không."
            )
        print(f"[Loading] Số camera dùng để train: {len(viewpoint_cameras)}")

        bg_color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")

        # 4. Quá trình tối ưu (Optimization Loop)
        print(f"[Loading] Bắt đầu huấn luyện Scene-GS ({self.iterations} iterations)...")
        xyz_lr_scheduler = get_expon_lr_func(
            lr_init=0.00016 * self.model.spatial_lr_scale,
            lr_final=0.0000016 * self.model.spatial_lr_scale,
            max_steps=self.iterations
        )

        viewpoint_stack = []
        progress_bar = tqdm(range(1, self.iterations + 1), desc="Training Scene-GS")
        ema_loss = 0.0
        for iteration in progress_bar:
            current_xyz_lr = xyz_lr_scheduler(iteration)
            for param_group in self.model.optimizer.param_groups:
                if param_group["name"] == "xyz":
                    param_group["lr"] = current_xyz_lr

            if iteration % 1000 == 0 and self.model.active_sh_degree < self.model.max_sh_degree:
                self.model.active_sh_degree += 1

            if not viewpoint_stack:
                viewpoint_stack = list(viewpoint_cameras)
                random.shuffle(viewpoint_stack)
            camera = viewpoint_stack.pop()

            rendered_image, viewspace_points, radii, visibility_filter = self.render(camera, bg_color)
            gt_image = camera.get_image(device="cuda")

            loss_l1 = l1_loss(rendered_image, gt_image)
            loss_ssim = ssim(rendered_image, gt_image)
            loss = (1.0 - lambda_dssim) * loss_l1 + lambda_dssim * (1.0 - loss_ssim)

            self.model.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            with torch.no_grad():
                ema_loss = 0.4 * loss.item() + 0.6 * ema_loss if iteration > 1 else loss.item()

                if iteration < densify_until_iter:
                    self.model.max_radii2D[visibility_filter] = torch.max(
                        self.model.max_radii2D[visibility_filter], radii[visibility_filter].float())
                    self.model.add_densification_stats(viewspace_points, visibility_filter)

                    if iteration > densify_from_iter and iteration % densification_interval == 0:
                        size_threshold = 20 if iteration > opacity_reset_interval else None
                        self.model.densify_and_prune(
                            densify_grad_threshold, 0.005, scene_extent, size_threshold)

                    if iteration % opacity_reset_interval == 0:
                        self.model.reset_opacity()

                self.model.optimizer.step()

            if iteration % 100 == 0:
                progress_bar.set_postfix({
                    "loss": f"{ema_loss:.4f}",
                    "n_gaussians": self.model.get_xyz.shape[0],
                })

            if iteration in save_iterations:
                save_path = self.checkpoints_dir / f"scene_gs_{iteration}{self.checkpoint_tag}.ply"
                self.model.save_ply(save_path)

        # 5. Checkpoint cuối cùng cho Module 3
        save_path = self.checkpoints_dir / f"scene_gs{self.checkpoint_tag}.ply"
        self.model.save_ply(save_path)
        print(f"[Done] Hoàn thành huấn luyện Scene-GS. Checkpoint sẵn sàng cho Module 3 tại: {save_path}")


if __name__ == "__main__":
    trainer = SceneGSTrainer()
    trainer.train()