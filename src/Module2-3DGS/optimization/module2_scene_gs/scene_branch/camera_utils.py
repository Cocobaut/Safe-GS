"""
Tiện ích dựng ma trận camera (world-to-view, projection) và load ảnh gốc theo đúng
quy ước chuẩn của 3D Gaussian Splatting (Kerbl et al.), dựa trên pose COLMAP (pycolmap).
"""
import math
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from PIL import Image


def focal2fov(focal: float, pixels: int) -> float:
    return 2 * math.atan(pixels / (2 * focal))


def get_projection_matrix(znear: float, zfar: float, fovx: float, fovy: float) -> torch.Tensor:
    tan_half_fovy = math.tan(fovy / 2)
    tan_half_fovx = math.tan(fovx / 2)
    top = tan_half_fovy * znear
    bottom = -top
    right = tan_half_fovx * znear
    left = -right

    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class TrainingCamera:
    """Gói đủ thông tin 1 camera cần cho 1 lần render trong vòng lặp train.

    Ảnh gốc được giải mã (decode JPEG/PNG) đúng 1 lần, cache lại dạng uint8 trên RAM CPU
    (không phải GPU, không phải float32) — tránh 2 vấn đề cùng lúc:
      - Load lại + decode ảnh từ đĩa mỗi iteration (chậm, khiến GPU rảnh chờ CPU - nguyên nhân
        chính khiến training chỉ đạt ~15 it/s dù GPU chỉ dùng 64% công suất).
      - Giữ hết ảnh dạng float32 trên GPU cùng lúc (OOM với scene nhiều ảnh như Replica 2000 ảnh).
    uint8 trên CPU RAM chỉ tốn ~1/4 dung lượng so với float32, và với Replica (2000 ảnh
    1200x680) chỉ khoảng 4.8GB RAM - rất nhẹ so với 41GB RAM trống trên máy.
    """

    def __init__(self, image_path, image_height, image_width, fovx, fovy,
                 world_view_transform, full_proj_transform, camera_center, image_name):
        self.image_path = image_path
        self.image_height = image_height
        self.image_width = image_width
        self.fovx = fovx
        self.fovy = fovy
        self.world_view_transform = world_view_transform  # (4,4) trên GPU
        self.full_proj_transform = full_proj_transform     # (4,4) trên GPU
        self.camera_center = camera_center                 # (3,) trên GPU
        self.image_name = image_name
        self._cpu_uint8 = None  # cache (H,W,3) uint8 trên RAM, load 1 lần lúc dùng đầu tiên

    def get_image(self, device: str = "cuda") -> torch.Tensor:
        """Trả về ảnh gốc (3,H,W) giá trị [0,1] trên device. Giải mã từ đĩa lần đầu, sau đó dùng cache RAM."""
        if self._cpu_uint8 is None:
            pil_img = Image.open(self.image_path).convert("RGB")
            if pil_img.size != (self.image_width, self.image_height):
                pil_img = pil_img.resize((self.image_width, self.image_height), Image.BILINEAR)
            self._cpu_uint8 = np.array(pil_img, dtype=np.uint8)  # (H,W,3), cache trong RAM

        image_tensor = torch.from_numpy(self._cpu_uint8).to(device, non_blocking=True).float() / 255.0
        return image_tensor.permute(2, 0, 1).contiguous()


def load_training_cameras(
    rec,
    images_dir: Union[str, Path],
    selected_names: List[str],
    device: str = "cuda",
    znear: float = 0.01,
    zfar: float = 100.0,
) -> List[TrainingCamera]:
    """
    Dựng danh sách TrainingCamera từ pose COLMAP (pycolmap.Reconstruction) + ảnh gốc.
    Chỉ giữ lại các ảnh vừa có trong reconstruction vừa nằm trong selected_names.
    """
    images_dir = Path(images_dir)
    name_to_image = {img.name: img for img in rec.images.values()}
    selected_set = set(selected_names)

    cameras = []
    for name, colmap_img in name_to_image.items():
        if selected_set and name not in selected_set:
            continue

        image_path = images_dir / name
        if not image_path.exists():
            continue

        cam = rec.cameras[colmap_img.camera_id]
        # Model PINHOLE: params = [fx, fy, cx, cy]
        fx, fy = cam.params[0], cam.params[1]
        width, height = cam.width, cam.height
        fovx = focal2fov(fx, width)
        fovy = focal2fov(fy, height)

        cfw = colmap_img.cam_from_world()
        R_wc = cfw.rotation.matrix()          # world -> camera
        t_wc = np.array(cfw.translation)

        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R_wc
        w2c[:3, 3] = t_wc

        world_view_transform = torch.tensor(w2c, dtype=torch.float32, device=device).transpose(0, 1)
        projection_matrix = get_projection_matrix(znear, zfar, fovx, fovy).to(device).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        cameras.append(TrainingCamera(
            image_path=image_path,
            image_height=height,
            image_width=width,
            fovx=fovx,
            fovy=fovy,
            world_view_transform=world_view_transform,
            full_proj_transform=full_proj_transform,
            camera_center=camera_center,
            image_name=name,
        ))

    return cameras
