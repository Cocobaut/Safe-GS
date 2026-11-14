"""
Reconstruction of `dreamgaussian/gs_renderer_2D.py`, which dream_gaussian_2D.py imports but which is missing
from the published OMEGAS code (the upstream repo CrystalWlz/OMEGAS is gone and the public mirror lacks it too).

It is NOT the authors' file: it re-implements the interface dream_gaussian_2D.py uses, on top of the OMEGAS
2DGS model (gaussian_splatting.scene_2D.GaussianModel, which carries the object features `_objects`) and the 2DGS
surfel renderer (gaussian_splatting.gaussian_renderer_2D.render):

    Renderer(sh_degree)                              .gaussians  (2DGS GaussianModel)
    Renderer.initialize(input=ply | None, num_pts, depth_ratio)
    Renderer.render(cam, scaling_modifier=1, bg_color=None) -> {"image", "alpha", "depth", "render_object",
                                                                "viewspace_points", "visibility_filter", "radii"}
    MiniCam(c2w, width, height, fovy, fovx, znear, zfar)   (c2w as returned by cam_utils.orbit_camera)

MiniCam follows dreamgaussian/gs_renderer.py of this repo (w2c = inv(c2w), no OpenGL rectification, because
dream_gaussian_2D.py builds its poses with orbit_camera(..., opengl=False)).
omegas_launch.py registers this module as `dreamgaussian.gs_renderer_2D` before running the authors' script.
"""
from types import SimpleNamespace

import numpy as np
import torch

from gaussian_splatting.gaussian_renderer_2D import render as render_2d
from gaussian_splatting.scene_2D.gaussian_model import GaussianModel
from gaussian_splatting.utils.graphics_utils import BasicPointCloud, getProjectionMatrix
from gaussian_splatting.utils.sh_utils import SH2RGB


class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar):
        self.image_width = int(width)
        self.image_height = int(height)
        self.FoVy = float(fovy)
        self.FoVx = float(fovx)
        self.znear = float(znear)
        self.zfar = float(zfar)

        c2w = np.asarray(c2w, dtype=np.float64)
        w2c = np.linalg.inv(c2w)
        self.world_view_transform = torch.tensor(w2c, dtype=torch.float32).transpose(0, 1).cuda()
        self.projection_matrix = (
            getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy)
            .transpose(0, 1).float().cuda()
        )
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = torch.tensor(c2w[:3, 3], dtype=torch.float32).cuda()


class Renderer:
    def __init__(self, sh_degree=3, white_background=True, radius=1):
        self.sh_degree = sh_degree
        self.white_background = white_background
        self.radius = radius
        self.gaussians = GaussianModel(sh_degree)
        self.bg_color = torch.tensor([1, 1, 1] if white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
        # PipelineParams of the 2DGS renderer (defaults of gaussian_splatting/arguments, depth_ratio from the config)
        self.pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, depth_ratio=1.0, debug=False)

    def initialize(self, input=None, num_pts=5000, radius=0.5, depth_ratio=None):
        if depth_ratio is not None:
            self.pipe.depth_ratio = float(depth_ratio)
        if input is None:
            # random ball, as in dreamgaussian/gs_renderer.py
            phis = np.random.random((num_pts,)) * 2 * np.pi
            costheta = np.random.random((num_pts,)) * 2 - 1
            thetas = np.arccos(costheta)
            mu = np.random.random((num_pts,))
            r = radius * np.cbrt(mu)
            xyz = np.stack((r * np.sin(thetas) * np.cos(phis), r * np.sin(thetas) * np.sin(phis), r * np.cos(thetas)), axis=1)
            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
            self.gaussians.create_from_pcd(pcd, 10)
        elif isinstance(input, BasicPointCloud):
            self.gaussians.create_from_pcd(input, 1)
        else:
            self.gaussians.load_ply(input)
            # load_ply leaves spatial_lr_scale = 0 (xyz frozen); use 1 like create_from_pcd(input, 1) above
            self.gaussians.spatial_lr_scale = 1.0

    def render(self, viewpoint_camera, scaling_modifier=1.0, bg_color=None, override_color=None, **_):
        bg = self.bg_color if bg_color is None else bg_color
        pkg = render_2d(viewpoint_camera, self.gaussians, self.pipe, bg, scaling_modifier=scaling_modifier,
                        override_color=override_color)
        return {
            "image": pkg["render"],
            "alpha": pkg["rend_alpha"],
            "depth": pkg["surf_depth"],
            "normal": pkg["rend_normal"],
            "render_object": pkg["render_object"],
            "viewspace_points": pkg["viewspace_points"],
            "visibility_filter": pkg["visibility_filter"],
            "radii": pkg["radii"],
        }
