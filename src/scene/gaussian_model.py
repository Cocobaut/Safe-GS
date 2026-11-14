#
# CoMe-aware GaussianModel for Inpaint360GS.
# The original Inpaint360GS model lives unchanged in scene/gaussian_model_inpaint360.py; this subclass adds the two
# CoMe per-Gaussian attributes so the CoMe rasterizer can be used and the ply stays readable by CoMe:
#   confidence : blended into the confidence channel (CoMe color confidence), kept in sync with the points
#   filter_3D  : Mip-Splatting 3D filter, recomputed from the CoMe training cameras whenever the points change
#

import os

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from scene.gaussian_model_inpaint360 import GaussianModel as GaussianModelInpaint360
from utils.system_utils import mkdir_p
from utils.graphics_utils import BasicPointCloud  # noqa: F401  (scene/dataset_readers.py imports it from here)
from diff_gaussian_rasterization._C import compute_filter_3d  # CoMe rasterizer


class GaussianModel(GaussianModelInpaint360):

    def __init__(self, sh_degree: int):
        super().__init__(sh_degree)
        self._confidence = torch.empty(0)
        self.filter_3D = torch.empty(0)
        # cameras of the CoMe training split (set by Scene); used to recompute filter_3D
        self.filter_cameras = None

    @property
    def get_confidence(self):
        return self._confidence

    def _sync_confidence(self):
        """New points without a confidence (e.g. created by the original code) start at 0, as in CoMe."""
        n = self._xyz.shape[0]
        if self._confidence.shape[0] != n:
            if self._confidence.shape[0] > 0:
                print(f"[CoMe] confidence size {self._confidence.shape[0]} != {n} points -> reset to 0")
            self._confidence = torch.zeros((n, 1), dtype=torch.float, device="cuda")

    @torch.no_grad()
    def compute_3D_filter(self, cameras=None):
        """Same as CoMe GaussianModel.compute_3D_filter (CUDA path)."""
        cameras = cameras if cameras is not None else self.filter_cameras
        viewmatrices = torch.stack([c.world_view_transform for c in cameras]).cuda()
        filter_3d = torch.ones_like(self._opacity) * -1
        compute_filter_3d(self.get_xyz.detach().contiguous(), viewmatrices, cameras[0].image_width,
                          cameras[0].image_height, float(cameras[0].focal_x), float(cameras[0].focal_y), filter_3d)
        # never observed -> largest filter
        filter_3d[filter_3d < -0.2] = filter_3d.max()
        self.filter_3D = filter_3d

    def ensure_come_attributes(self):
        """Called by the renderer: keep confidence / filter_3D aligned with the current points."""
        self._sync_confidence()
        if self.filter_3D.shape[0] != self._xyz.shape[0]:
            if self.filter_cameras is None:
                raise RuntimeError("filter_3D out of date and no filter cameras attached (create a Scene first)")
            self.compute_3D_filter()

    # ---------------------------------------------------------------------------------------------- ply
    def construct_list_of_attributes(self):
        return super().construct_list_of_attributes() + ['confidence', 'filter_3D']

    def save_ply(self, path):
        self.ensure_come_attributes()
        mkdir_p(os.path.dirname(path))
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        obj_dc = self._objects_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        confidence = self._confidence.detach().cpu().numpy()
        filter_3D = self.filter_3D.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, obj_dc, confidence,
                                     filter_3D), axis=1)
        elements[:] = list(map(tuple, attributes))
        PlyData([PlyElement.describe(elements, 'vertex')]).write(path)

    def load_ply(self, path):
        super().load_ply(path)
        v = PlyData.read(path).elements[0]
        names = {p.name for p in v.properties}
        n = self._xyz.shape[0]
        zeros = np.zeros((n, 1), dtype=np.float32)
        confidence = np.asarray(v["confidence"])[..., None] if "confidence" in names else zeros
        filter_3D = np.asarray(v["filter_3D"])[..., None] if "filter_3D" in names else zeros
        self._confidence = torch.tensor(confidence, dtype=torch.float, device="cuda")
        # the stored filter is replaced by Scene (recomputed from the CoMe training cameras, as CoMe render.py does)
        self.filter_3D = torch.tensor(filter_3D, dtype=torch.float, device="cuda")

    # ---------------------------------------------------------------------------------------------- point changes
    def prune_points(self, mask):
        super().prune_points(mask)
        valid = ~mask
        if self._confidence.shape[0] == valid.shape[0]:
            self._confidence = self._confidence[valid]
        if self.filter_3D.shape[0] == valid.shape[0]:
            self.filter_3D = self.filter_3D[valid]

    def densification_postfix(self, new_xyz, *args, **kwargs):
        super().densification_postfix(new_xyz, *args, **kwargs)
        n_new = new_xyz.shape[0]
        self._confidence = torch.cat((self._confidence, torch.zeros((n_new, 1), device="cuda")), dim=0)
        # filter_3D is recomputed by the renderer (size changed)

    @staticmethod
    def _union_mask(masks_per_obj):
        union = None
        for masks in masks_per_obj.values():
            m = masks["mask3d"].bool().squeeze()
            union = m if union is None else (union | m)
        return union

    def removal_setup(self, masks_per_obj):
        self._sync_confidence()
        conf, filt = self._confidence, self.filter_3D
        objects_model = super().removal_setup(masks_per_obj)
        for obj_id, sub in objects_model.items():
            # sub-models are plain Inpaint360GS models: give them the CoMe attributes
            m = masks_per_obj[obj_id]["mask3d"].bool().squeeze()
            sub.__class__ = GaussianModel
            sub._confidence, sub.filter_3D, sub.filter_cameras = conf[m].clone(), filt[m].clone(), self.filter_cameras
        keep = ~self._union_mask(masks_per_obj)
        self._confidence = conf[keep]
        self.filter_3D = filt[keep]
        return objects_model

    def inpaint_setup(self, args, training_args, masks_per_obj):
        self._sync_confidence()
        conf = self._confidence
        keep = ~self._union_mask(masks_per_obj)
        super().inpaint_setup(args, training_args, masks_per_obj)
        n_new = self._xyz.shape[0] - int(keep.sum())
        self._confidence = torch.cat((conf[keep], torch.zeros((n_new, 1), device="cuda")), dim=0)
        self.filter_3D = torch.empty(0)  # recomputed by the renderer
