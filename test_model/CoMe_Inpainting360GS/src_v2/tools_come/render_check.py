"""Step 4: render the CoMe 3DGS with the Inpaint360GS renderer (switched to the CoMe rasterizer) on all views and
compare with the CoMe renders (step2_reuse_come_renders).

  COME_SPLATTING_CONFIG=<model>/config.json python tools_come/render_check.py -s <data> -m <model> \
      --resolution 1 --eval --iteration 30000 --come_renders <dir> --out_dir <dir>
"""
import json
import os
import time
from argparse import ArgumentParser

import numpy as np
import torch
import torchvision
from PIL import Image
from plyfile import PlyData

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import Scene, GaussianModel

PASS = {"psnr_mean": 35.0, "psnr_min": 30.0}


def quantize(img):
    """As torchvision.utils.save_image: clamp [0,1] -> uint8 levels."""
    return (img.clamp(0, 1) * 255 + 0.5).floor()


def psnr(a, b):
    mse = ((a - b) ** 2).mean().item()
    return 10 * np.log10(255 ** 2 / max(mse, 1e-10))


def main():
    parser = ArgumentParser()
    model = ModelParams(parser)  # defaults on: the CoMe cfg_args lacks the Inpaint360GS keys
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--come_renders", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--save_views", nargs="+", default=["frame_00001", "frame_00064", "frame_00127", "frame_00140"])
    args = get_combined_args(parser)
    dataset, pipe = model.extract(args), pipeline.extract(args)
    for d in ["1_come_render", "2_new_renderer"]:
        os.makedirs(os.path.join(args.out_dir, d), exist_ok=True)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        ply = os.path.join(dataset.model_path, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply")
        filter_ply = torch.tensor(np.asarray(PlyData.read(ply).elements[0]["filter_3D"]), device="cuda")[:, None]
        filter_diff = float((gaussians.filter_3D - filter_ply).abs().max())
        bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

        rows, t0 = [], time.time()
        cams = [("train", c) for c in scene.getTrainCameras()] + [("test", c) for c in scene.getTestCameras()]
        for split, cam in sorted(cams, key=lambda x: x[1].image_name):
            pkg = render(cam, gaussians, pipe, bg)
            rgb = pkg["render"]
            r = quantize(rgb)
            gt = quantize(cam.original_image[0:3].cuda())
            ref_path = os.path.join(args.come_renders, f"{cam.image_name}.png")
            ref = torch.tensor(np.array(Image.open(ref_path).convert("RGB")), device="cuda").permute(2, 0, 1).float()
            row = {"view": cam.image_name, "split": split,
                   "psnr_vs_come_render": round(psnr(r, ref), 3),
                   "max_abs_diff_vs_come_render": int((r - ref).abs().max().item()),
                   "frac_pixels_diff_vs_come_render": round(float(((r - ref).abs().max(0)[0] > 0).float().mean()), 6),
                   "psnr_vs_gt_image": round(psnr(r, gt), 3)}
            rows.append(row)
            if cam.image_name in args.save_views:
                torchvision.utils.save_image(rgb, os.path.join(args.out_dir, "2_new_renderer", f"{cam.image_name}.png"))
                Image.open(ref_path).save(os.path.join(args.out_dir, "1_come_render", f"{cam.image_name}.png"))
                if cam.image_name == args.save_views[0]:
                    obj = pkg["render_object"]  # exercise the object-feature path once
                    row["render_object_shape"] = list(obj.shape)
                    row["render_object_finite"] = bool(torch.isfinite(obj).all())
            print(f"[{split}] {row}", flush=True)
        seconds = time.time() - t0

    def stats(key, split=None):
        v = [r[key] for r in rows if split is None or r["split"] == split]
        return {"n": len(v), "mean": round(float(np.mean(v)), 3), "min": round(float(np.min(v)), 3),
                "max": round(float(np.max(v)), 3)} if v else None

    p = stats("psnr_vs_come_render")
    summary = {
        "n_views": len(rows), "n_gaussians": int(gaussians.get_xyz.shape[0]),
        "n_filter_cameras": len(gaussians.filter_cameras),
        "come_splatting_config": os.environ.get("COME_SPLATTING_CONFIG", ""),
        "filter_3D_recomputed_max_abs_diff_vs_ply": filter_diff,
        "psnr_vs_come_render": p,
        "max_abs_diff_vs_come_render": max(r["max_abs_diff_vs_come_render"] for r in rows),
        "psnr_vs_gt_image_train": stats("psnr_vs_gt_image", "train"),
        "psnr_vs_gt_image_test": stats("psnr_vs_gt_image", "test"),
        "seconds_per_view": round(seconds / max(len(rows), 1), 3),
        "pass_threshold": PASS,
        "pass": bool(p and p["mean"] >= PASS["psnr_mean"] and p["min"] >= PASS["psnr_min"]),
    }
    json.dump(rows, open(os.path.join(args.out_dir, "per_view_psnr.json"), "w"), indent=2)
    json.dump(summary, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2)
    print("SUMMARY " + json.dumps(summary), flush=True)
    if not summary["pass"]:
        raise SystemExit("renderer check FAILED")


if __name__ == "__main__":
    main()
