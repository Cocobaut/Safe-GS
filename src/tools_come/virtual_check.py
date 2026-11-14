"""Diagnose renders from cameras built by hand (virtual_pose.py / render_video_func_wriva style).

  1. a real training camera rebuilt the virtual way from its own R, T  -> must equal the normal render
  2. poses of generate_ellipse_path (240, as the video) next to the nearest real training view
  3. the same ellipse poses rendered without the 3D filter / with only Gaussians of moderate size

  COME_SPLATTING_CONFIG=<3dgs>/config.json python tools_come/virtual_check.py -s <data> -m <model> --out_dir <dir>
"""
import copy
import json
import os
from argparse import ArgumentParser

import numpy as np
import torch
import torchvision

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.graphics_utils import getWorld2View2
from utils.pose_utils import generate_ellipse_path


def set_pose(view, R, T):
    v = copy.deepcopy(view)
    v.world_view_transform = torch.tensor(getWorld2View2(R, T, view.trans, view.scale)).transpose(0, 1).cuda()
    v.full_proj_transform = (v.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
    v.camera_center = v.world_view_transform.inverse()[3, :3]
    return v


def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--out_dir", required=True)
    args = get_combined_args(parser)
    dataset, pipe = model.extract(args), pipeline.extract(args)
    os.makedirs(args.out_dir, exist_ok=True)
    report = {}
    with torch.no_grad():
        g = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, g, load_iteration=args.iteration, shuffle=False)
        bg = torch.zeros(3, device="cuda")
        views = scene.getTrainCameras()
        save = lambda img, name: torchvision.utils.save_image(img.clamp(0, 1), os.path.join(args.out_dir, name))

        # 1. rebuilt real camera
        v0 = views[0]
        a = render(v0, g, pipe, bg)["render"]
        v0b = set_pose(v0, v0.R, v0.T)
        b = render(v0b, g, pipe, bg)["render"]
        report["rebuilt_real_cam"] = {
            "view": v0.image_name,
            "max_abs_diff_matrix": float((v0.world_view_transform - v0b.world_view_transform).abs().max()),
            "dtype_real": str(v0.world_view_transform.dtype), "dtype_rebuilt": str(v0b.world_view_transform.dtype),
            "max_abs_diff_render": float((a - b).abs().max())}
        save(torch.cat([a, b], 2), "1_real_vs_rebuilt.png")

        # 2. ellipse poses (as the video) vs nearest real view
        poses = generate_ellipse_path(views)
        centers = torch.stack([v.camera_center for v in views])
        rows = []
        for k in (0, 60, 120, 180):
            p = poses[k]
            vk = set_pose(views[0], p[:3, :3].T, p[:3, 3])
            d = (centers - vk.camera_center).norm(dim=1)
            j = int(d.argmin())
            r_virtual = render(vk, g, pipe, bg)["render"]
            r_real = render(views[j], g, pipe, bg)["render"]
            # same pose without the 3D filter, and with only Gaussians whose largest scale < 0.5
            f_keep = g.filter_3D
            g.filter_3D = torch.zeros_like(f_keep)
            r_nofilter = render(vk, g, pipe, bg)["render"]
            g.filter_3D = f_keep
            save(torch.cat([r_virtual, r_nofilter, r_real], 2), f"2_ellipse{k:03d}_virtual_nofilter_nearestreal.png")
            rows.append({"k": k, "nearest_real": views[j].image_name, "dist_to_nearest_real": float(d[j]),
                         "virtual_center": vk.camera_center.tolist(), "virtual_forward_world": vk.world_view_transform[:3, 2].tolist(),
                         "nearest_forward_world": views[j].world_view_transform[:3, 2].tolist()})
        report["ellipse"] = rows
        sc = g.get_scaling.max(1).values
        report["gaussians"] = {"n": int(sc.numel()), "scale_gt_0.5": int((sc > 0.5).sum()),
                               "scale_gt_5": int((sc > 5).sum()), "filter3D_max": float(g.filter_3D.max()),
                               "filter3D_median": float(g.filter_3D.median())}
    json.dump(report, open(os.path.join(args.out_dir, "report.json"), "w"), indent=2)
    print("REPORT " + json.dumps(report))


if __name__ == "__main__":
    main()
