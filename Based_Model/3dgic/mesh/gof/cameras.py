"""
Camera tối thiểu có đủ thuộc tính mà logic GOF cần (theo GOF scene/cameras.py), dựng từ cameras.json
mà InFusion ghi ra khi train (gaussian_splatting/utils/camera_utils.py::camera_to_JSON).
"""
import json
import math

import numpy as np
import torch

from .graphics import focal2fov, getProjectionMatrix, getWorld2View2


class MeshCamera:
    def __init__(self, uid, image_name, R, T, FoVx, FoVy, width, height, znear=0.01, zfar=100.0):
        self.uid = uid
        self.image_name = image_name
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_width = width
        self.image_height = height
        self.znear = znear
        self.zfar = zfar

        tan_fovx = math.tan(self.FoVx * 0.5)
        tan_fovy = math.tan(self.FoVy * 0.5)
        self.focal_y = self.image_height / (2.0 * tan_fovy)
        self.focal_x = self.image_width / (2.0 * tan_fovx)

        self.world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def load_cameras_json(path, view_stride=1):
    """
    cameras.json của InFusion: 'rotation' = R (C2W rotation, lưu chuyển vị như 3DGS), 'position' = tâm camera.
    Suy ra T = -R^T @ position (world-to-view translation) và FoV từ fx, fy.
    """
    entries = sorted(json.loads(open(path).read()), key=lambda c: c["img_name"])
    cams = []
    for c in entries[::view_stride]:
        R = np.array(c["rotation"], dtype=np.float64)
        pos = np.array(c["position"], dtype=np.float64)
        T = -R.T @ pos
        w, h = int(c["width"]), int(c["height"])
        cams.append(MeshCamera(c["id"], c["img_name"], R, T, focal2fov(c["fx"], w), focal2fov(c["fy"], h), w, h))
    return cams
