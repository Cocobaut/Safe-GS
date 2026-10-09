import os
from argparse import ArgumentParser
from tqdm import tqdm
import math
import json

import numpy as np
import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state


def metric(dataset, pipe, checkpoint_iterations=None):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=checkpoint_iterations, shuffle=False)
    viewpoint_cam_list = scene.getTrainCameras()

    kernel_size = dataset.kernel_size

    bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    depth_name = "expected_depth" if dataset.depth_ratio < 0.5 else "median_depth"

    metric_dict = {"depth_mae": [], "depth_rmse": [], "normal": []}
    os.makedirs(os.path.join(dataset.model_path, "train", "image"), exist_ok=True)

    for idx, viewpoint_cam in enumerate(tqdm(viewpoint_cam_list)):
        # Rendering offscreen from that camera
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, kernel_size)
        mask = viewpoint_cam.gt_mask
        gt_depth = viewpoint_cam.depth
        gt_normal = viewpoint_cam.normal
        recon_depth = render_pkg[depth_name]
        recon_normal = render_pkg["normal"]
        if gt_depth is not None:
            gt_depth = recon_depth.new_tensor(gt_depth)
        if gt_normal is not None:
            gt_normal = recon_normal.new_tensor(gt_normal[..., :3])

        if gt_depth is None:
            depth_rmse = 0
        else:
            depth_mask = (mask.squeeze() > 0.9) * (gt_depth < 100)
            depth_rmse = math.sqrt(((recon_depth.squeeze() - gt_depth)[depth_mask] ** 2).mean())
            depth_mae = ((recon_depth.squeeze() - gt_depth)[depth_mask].abs()).mean()

        if gt_normal is None:
            angle_metric = 0
        else:
            output_normal = torch.nn.functional.normalize(recon_normal.permute(1, 2, 0), dim=-1).detach()
            normal_mask = (mask.squeeze() > 0.9) & (gt_normal[..., -1] < 0)
            angle_metric = torch.acos((output_normal * gt_normal).sum(dim=-1).clip(-1, 1))[normal_mask].mean() / (np.pi) * 180
        metric_dict["depth_rmse"].append(float(depth_rmse))
        metric_dict["depth_mae"].append(float(depth_mae))
        metric_dict["normal"].append(float(angle_metric))

    mean_metric_dict = {}
    mean_metric_dict["depth_rmse"] = np.mean(metric_dict["depth_rmse"])
    mean_metric_dict["depth_mae"] = np.mean(metric_dict["depth_mae"])
    mean_metric_dict["normal"] = np.mean(metric_dict["normal"])
    print(mean_metric_dict)
    with open(dataset.model_path + "/results.json", "w") as fp:
        json.dump(mean_metric_dict, fp, indent=True)
    with open(dataset.model_path + "/results_all.json", "w") as fp:
        json.dump(metric_dict, fp, indent=True)


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    with torch.no_grad():
        metric(
            model.extract(args),
            pipeline.extract(args),
            args.iteration,
        )
