#
# CoMe geometric regularization (Based_Model/CoMe/train.py, loss block after the RGB term), with the mesh_args of the
# CoMe training of teatime (CoMe_Infusion Step 2). Used by edit_object_inpaint.py when fine-tuning the inpainted region.
# The appearance embedding / color-confidence RGB term is not used: the virtual views have no per-view embedding and
# the RGB term of Inpaint360GS is kept.
#

import os

import torch

from utils.come_depth_utils import depth_to_normal, central_diff

# rendered_all channels of the CoMe rasterizer (cuda_rasterizer/auxiliary.h)
NORMAL, DEPTH, OPACITY, DISTORTION, EXTENT, COLOR_VARIANCE, NORMAL_VARIANCE = slice(3, 6), 6, 7, 8, 9, 11, 12

COME_LAMBDAS = {
    "depth_normal": 0.05,
    "distortion": 100.0,
    "smoothness": 0.01,
    "opacity_field": 0.004,
    "extent": 0.1,
    "variance": 0.5,
    "normal_variance": 0.005,
}


def come_loss_enabled():
    return os.environ.get("COME_LOSS", "1") != "0"


def come_regularization(rendering, viewpoint_cam, lambdas=COME_LAMBDAS):
    """rendering: the 13-channel output of the CoMe rasterizer. Returns (weighted sum, dict of unweighted terms)."""
    depth = rendering[DEPTH]
    if depth.isnan().any():
        depth = torch.nan_to_num(depth, nan=0.0)
    depth_normal, _ = depth_to_normal(viewpoint_cam, depth[None, ...])
    depth_normal = depth_normal.permute(2, 0, 1)

    render_normal = torch.nn.functional.normalize(rendering[NORMAL], p=2, dim=0)
    c2w = viewpoint_cam.world_view_transform
    render_normal_world = (c2w[:3, :3] @ render_normal.reshape(3, -1)).reshape(3, *render_normal.shape[1:])

    nabla_I = central_diff(viewpoint_cam.original_image.permute(1, 2, 0)).cuda()

    terms = {
        "depth_normal": (1 - (render_normal_world * depth_normal).sum(dim=0)).mean(),
        "distortion": rendering[DISTORTION].mean(),
        "smoothness": (central_diff(render_normal_world.permute(1, 2, 0)) * torch.exp(-nabla_I)).mean(),
        "opacity_field": ((rendering[OPACITY] - 0.5) ** 2).mean(),
        "extent": rendering[EXTENT].mean(),
        "variance": rendering[COLOR_VARIANCE].mean(),
        "normal_variance": rendering[NORMAL_VARIANCE].mean(),
    }
    total = sum(terms[k] * lambdas[k] for k in terms)
    return total, terms
