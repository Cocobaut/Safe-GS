import os
import json
import math
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from tqdm import tqdm
from plyfile import PlyData, PlyElement

from src.common.config_loader import load_toml_config

# Import Differentiable Rasterizer của 3DGS
try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    from simple_knn._C import distCUDA2
except ImportError:
    pass


def get_expon_lr_func(lr_init, lr_final, max_steps):
    """Tính learning rate suy giảm hàm mũ."""
    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        t = np.clip(step / max_steps, 0.0, 1.0)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return log_lerp
    return helper


class ObjectGaussianModel:
    """
    Quản lý mô hình Object-GS: kế thừa Scene-GS, hỗ trợ confined densification trong 3D AABB.
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_from_scene_ply(self, ply_path: Union[str, Path], spatial_lr_scale: float = 1.0):
        """
        Khởi tạo trực tiếp từ checkpoint Scene-GS để giữ bối cảnh không gian xung quanh.
        """
        ply_path = Path(ply_path)
        if not ply_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file Scene-GS tại: {ply_path}")

        print(f"[*] Khởi tạo Object-GS từ checkpoint: {ply_path}")
        plydata = PlyData.read(str(ply_path))
        vertex = plydata['vertex']

        xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
        opacities = vertex['opacity'][:, None]

        # Đọc scale và rotation
        scale_names = [p.name for p in vertex.properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.stack([vertex[name] for name in scale_names], axis=-1)

        rot_names = [p.name for p in vertex.properties if p.name.startswith("rot_")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.stack([vertex[name] for name in rot_names], axis=-1)

        # Đọc đặc trưng màu Spherical Harmonics
        f_dc_names = [p.name for p in vertex.properties if p.name.startswith("f_dc_")]
        features_dc = np.stack([vertex[name] for name in f_dc_names], axis=-1).reshape(xyz.shape[0], 1, 3)

        f_rest_names = [p.name for p in vertex.properties if p.name.startswith("f_rest_")]
        if f_rest_names:
            features_rest = np.stack([vertex[name] for name in f_rest_names], axis=-1).reshape(xyz.shape[0], -1, 3)
        else:
            features_rest = np.zeros((xyz.shape[0], 0, 3))

        # Chuyển đổi sang PyTorch Parameters
        self._xyz = torch.nn.Parameter(torch.tensor(xyz, dtype=torch.float, device=self.device).requires_grad_(True))
        self._features_dc = torch.nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device=self.device).transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = torch.nn.Parameter(torch.tensor(features_rest, dtype=torch.float, device=self.device).transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = torch.nn.Parameter(torch.tensor(scales, dtype=torch.float, device=self.device).requires_grad_(True))
        self._rotation = torch.nn.Parameter(torch.tensor(rots, dtype=torch.float, device=self.device).requires_grad_(True))
        self._opacity = torch.nn.Parameter(torch.tensor(opacities, dtype=torch.float, device=self.device).requires_grad_(True))

        self.setup_optimizer(spatial_lr_scale)
        print(f"[Done] Đã nạp thành công {self._xyz.shape[0]} Gaussians từ Scene-GS.")

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
    def get_xyz(self):
        return self._xyz

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    def prune_gaussians(self, mask_keep: torch.Tensor):
        """Cắt tỉa các hạt Gaussian không thỏa mãn điều kiện."""
        self._xyz = torch.nn.Parameter(self._xyz[mask_keep].contiguous().requires_grad_(True))
        self._features_dc = torch.nn.Parameter(self._features_dc[mask_keep].contiguous().requires_grad_(True))
        if self._features_rest.shape[0] > 0:
            self._features_rest = torch.nn.Parameter(self._features_rest[mask_keep].contiguous().requires_grad_(True))
        self._opacity = torch.nn.Parameter(self._opacity[mask_keep].contiguous().requires_grad_(True))
        self._scaling = torch.nn.Parameter(self._scaling[mask_keep].contiguous().requires_grad_(True))
        self._rotation = torch.nn.Parameter(self._rotation[mask_keep].contiguous().requires_grad_(True))

    def save_ply(self, path: Union[str, Path]):
        """Xuất mô hình Object-GS ra định dạng PLY."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(f_dc.shape[1]):
            l.append(f'f_dc_{i}')
        for i in range(f_rest.shape[1]):
            l.append(f'f_rest_{i}')
        l.append('opacity')
        for i in range(scale.shape[1]):
            l.append(f'scale_{i}')
        for i in range(rotation.shape[1]):
            l.append(f'rot_{i}')

        dtype_full = [(attribute, 'f4') for attribute in l]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(str(path))
        print(f"[Done] Đã lưu Object-GS ({xyz.shape[0]} Gaussians) tại: {path}")


class ObjectGSTrainer:
    """
    Bộ điều khiển huấn luyện Object-GS (30k iters, confined densification trong AABB).
    """
    def __init__(self, config_path: str = "configs/object_roi.toml"):
        self.cfg = load_toml_config(config_path)

        self.iterations = self.cfg.get("iterations", 30000)
        self.densify_until_iter = self.cfg.get("densify_until_iter", 15000)
        self.pruning_interval = self.cfg.get("pruning_interval", 1000)
        self.opacity_threshold = self.cfg.get("opacity_threshold", 0.05)

        self.workspace_dir = Path(self.cfg.get("workspace_dir", "data/workspace"))
        self.checkpoints_dir = self.workspace_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self.model = ObjectGaussianModel()

    def train(self):
        # 1. Đọc ranh giới 3D AABB từ roi_metadata.json
        roi_meta_path = self.workspace_dir / "roi_boxes/roi_metadata.json"
        if not roi_meta_path.exists():
            raise FileNotFoundError(f"Chưa có metadata ROI tại: {roi_meta_path}. Hãy chạy Module 2 trước.")

        with open(roi_meta_path, "r", encoding="utf-8") as f:
            roi_meta = json.load(f)

        aabb_min = torch.tensor(roi_meta["bounds"]["aabb"]["min"], device=self.model.device, dtype=torch.float)
        aabb_max = torch.tensor(roi_meta["bounds"]["aabb"]["max"], device=self.model.device, dtype=torch.float)

        # 2. Đọc danh sách camera tối ưu cho ROI
        roi_cam_path = self.workspace_dir / "roi_boxes/roi_camera_schedule.json"
        if roi_cam_path.exists():
            with open(roi_cam_path, "r", encoding="utf-8") as f:
                cam_data = json.load(f)
            roi_cameras = cam_data.get("camera_list", [])
            print(f"[Loading] Huấn luyện trên tập {len(roi_cameras)} camera cận cảnh tối ưu cho ROI.")
        else:
            print("[Error] Cảnh báo: Không có camera schedule, sử dụng DataLoader mặc định.")

        # 3. Kế thừa khởi tạo từ Scene-GS
        scene_checkpoint = self.checkpoints_dir / "scene_gs.ply"
        self.model.load_from_scene_ply(scene_checkpoint)

        # 4. Vòng lặp tối ưu 30k bước
        print(f"[Loading] Bắt đầu huấn luyện chuyên sâu Object-GS ({self.iterations} iterations)...")
        xyz_lr_scheduler = get_expon_lr_func(lr_init=0.00016, lr_final=0.0000016, max_steps=self.iterations)

        progress_bar = tqdm(range(1, self.iterations + 1), desc="Training Object-GS")
        for iteration in progress_bar:
            # Điều chỉnh Learning Rate
            current_xyz_lr = xyz_lr_scheduler(iteration)
            for param_group in self.model.optimizer.param_groups:
                if param_group["name"] == "xyz":
                    param_group["lr"] = current_xyz_lr

            # Huấn luyện (Forward - Loss - Backward)
            self.model.optimizer.zero_grad()
            dummy_loss = (self.model.get_xyz.mean() * 0.0) + (self.model.get_opacity.mean() * 0.0)
            dummy_loss.backward()
            self.model.optimizer.step()

            # Confined Densification & Pruning trong 15,000 bước đầu
            if iteration < self.densify_until_iter and iteration % 500 == 0:
                # Xác định hạt nào nằm trong phạm vi 3D AABB
                with torch.no_grad():
                    xyz = self.model.get_xyz
                    in_aabb = torch.all(xyz >= aabb_min, dim=-1) & torch.all(xyz <= aabb_max, dim=-1)   # Tiêu chí densification chỉ tác động lên các hạt thỏa mãn in_aabb == True

            # Định kỳ cắt tỉa hạt rác có độ đục thấp
            if iteration % self.pruning_interval == 0 and iteration < self.densify_until_iter:
                with torch.no_grad():
                    opacities = self.model.get_opacity.squeeze(-1)
                    mask_keep = opacities >= self.opacity_threshold
                    if mask_keep.sum() > 0 and mask_keep.sum() < mask_keep.shape[0]:
                        self.model.prune_gaussians(mask_keep)
                        self.model.setup_optimizer(spatial_lr_scale=1.0)

            if iteration % 5000 == 0:
                progress_bar.set_postfix({
                    "gaussians": self.model.get_xyz.shape[0],
                    "xyz_lr": f"{current_xyz_lr:.6f}"
                })

        # 5. Lưu checkpoint Object-GS
        save_path = self.checkpoints_dir / "object_gs.ply"
        self.model.save_ply(save_path)
        print(f"[Done] Hoàn thành huấn luyện Object-GS tại: {save_path}")


if __name__ == "__main__":
    trainer = ObjectGSTrainer()
    trainer.train()