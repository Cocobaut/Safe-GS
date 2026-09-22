"""
Object-GS dung y chang paper ROI-GS (Bui et al. 2025):
  - Khoi tao TU TOAN BO checkpoint Scene-GS (khong tach rieng) - giu ngu canh xung quanh.
  - Loss tren TOAN BO anh (khong crop, khong dong bang gradient ngoai ROI) - Gaussian
    quanh box + nen van duoc optimize binh thuong, CHI de tranh floating artifacts
    (dung y nguyen tinh than paper: "Optimization is applied to the entire image,
    updating Gaussians even in the vicinity of the object box and the background").
  - CHI gioi han DENSIFY (sinh hat moi qua clone/split) trong ROI, trong nua dau qua
    trinh train (paper: 15K/30K).
  - Prune van ap dung toan cuc nhu binh thuong (loai hat khong dong gop).

Ket qua: vung ROI duoc "nang cap LOD" (nhieu Gaussian hon, chi tiet hon) trong khi phan
con lai cua scene giu nguyen so luong/chat luong nhu Scene-GS goc.

Chay 1 object:
    /home/ml4u/conda_envs/safe-gs/bin/python tmp/refine_object_in_scene.py --instance_id 4 --category chair
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pycolmap
import torch
from tqdm import tqdm

sys.path.insert(0, "/media/ml4u/Extreme SSD/Safe-GS")

from src.module2_scene_gs.scene_branch.scene_trainer import GaussianSceneModel, render_gaussian_model
from src.module2_scene_gs.scene_branch.camera_utils import load_training_cameras
from src.module2_scene_gs.scene_branch.loss_utils import l1_loss, ssim

SCENE_PLY = "/media/ml4u/Extreme SSD/Safe-GS/outputs/Replica/3dgs/office0/point_cloud/iteration_30000/point_cloud.ply"
BOXES_JSON = "/media/ml4u/Extreme SSD/Safe-GS/outputs/Replica/bounding_box_3D_and_view_selection/office0/office_0_3d_boxes.json"
VIEWS_JSON = "/media/ml4u/Extreme SSD/Safe-GS/outputs/Replica/bounding_box_3D_and_view_selection/office0/office_0_views.json"
SPARSE_DIR = "/media/ml4u/Extreme SSD/Safe-GS/outputs/Replica/sfm/office0/sparse/0"
IMAGES_DIR = "/media/ml4u/Extreme SSD/Replica 8 Scene/Replica/office0/results/image"
OUTPUT_DIR = Path("/media/ml4u/Extreme SSD/Safe-GS/tmp/roi_gs_objects")


def init_model_from_scene_ply(ply_path, sh_degree=3, device="cuda"):
    model = GaussianSceneModel.load_ply(ply_path, sh_degree=sh_degree, device=device)
    model.active_sh_degree = model.max_sh_degree
    n = model._xyz.shape[0]
    model._xyz = torch.nn.Parameter(model._xyz.requires_grad_(True))
    model._features_dc = torch.nn.Parameter(model._features_dc.requires_grad_(True))
    model._features_rest = torch.nn.Parameter(model._features_rest.requires_grad_(True))
    model._scaling = torch.nn.Parameter(model._scaling.requires_grad_(True))
    model._rotation = torch.nn.Parameter(model._rotation.requires_grad_(True))
    model._opacity = torch.nn.Parameter(model._opacity.requires_grad_(True))
    model.xyz_gradient_accum = torch.zeros((n, 1), device=device)
    model.denom = torch.zeros((n, 1), device=device)
    model.max_radii2D = torch.zeros(n, device=device)
    model.setup_optimizer(spatial_lr_scale=1.0)
    return model


def compute_roi_mask(xyz: torch.Tensor, obb: dict) -> torch.Tensor:
    center = torch.tensor(obb["center"], dtype=torch.float32, device=xyz.device)
    sizes = torch.tensor(obb["sizes"], dtype=torch.float32, device=xyz.device)
    R = torch.tensor(obb["rotation_matrix"], dtype=torch.float32, device=xyz.device)
    local = (xyz - center[None, :]) @ R.T
    half = sizes / 2.0
    return torch.all(torch.abs(local) <= half[None, :], dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_id", required=True, type=str)
    parser.add_argument("--category", default="object")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--densify_until_iter", type=int, default=15000)
    parser.add_argument("--densify_from_iter", type=int, default=500)
    parser.add_argument("--densification_interval", type=int, default=100)
    parser.add_argument("--opacity_reset_interval", type=int, default=3000)
    parser.add_argument("--densify_grad_threshold", type=float, default=0.0002)
    parser.add_argument("--save_every", type=int, default=5000)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_ply = OUTPUT_DIR / f"office0_{args.instance_id}_{args.category}_roigs.ply"

    print(f"[{args.instance_id}:{args.category}] Khởi tạo từ TOÀN BỘ checkpoint Scene-GS...")
    model = init_model_from_scene_ply(SCENE_PLY)
    n_total = model.get_xyz.shape[0]

    views = json.loads(Path(VIEWS_JSON).read_text())[args.instance_id]
    boxes = json.loads(Path(BOXES_JSON).read_text())
    obb = next(b["arbitrary_3d_obb"] for b in boxes if str(b["instance_id"]) == args.instance_id)

    with torch.no_grad():
        roi_mask0 = compute_roi_mask(model.get_xyz, obb)
    print(f"[{args.instance_id}:{args.category}] Gaussian trong ROI ban đầu: {int(roi_mask0.sum())}/{n_total}")
    print(f"[{args.instance_id}:{args.category}] Số camera liên quan: {len(views)}")

    rec = pycolmap.Reconstruction(SPARSE_DIR)
    cameras = load_training_cameras(rec, IMAGES_DIR, views, device="cuda")

    scene_extent = float(np.linalg.norm(
        model.get_xyz.detach().cpu().numpy().max(axis=0) - model.get_xyz.detach().cpu().numpy().min(axis=0)))
    bg_color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")

    t0 = time.time()
    viewpoint_stack = []
    progress_bar = tqdm(range(1, args.iterations + 1), desc=f"ROI-GS obj {args.instance_id}:{args.category}")
    for iteration in progress_bar:
        if not viewpoint_stack:
            viewpoint_stack = list(cameras)
            random.shuffle(viewpoint_stack)
        cam = viewpoint_stack.pop()

        rendered_image, viewspace_points, radii, visibility_filter = render_gaussian_model(model, cam, bg_color)
        gt_image = cam.get_image(device="cuda")

        loss_l1 = l1_loss(rendered_image, gt_image)
        loss_ssim = ssim(rendered_image, gt_image)
        loss = 0.8 * loss_l1 + 0.2 * (1.0 - loss_ssim)

        model.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        with torch.no_grad():
            if iteration < args.densify_until_iter:
                model.max_radii2D[visibility_filter] = torch.max(
                    model.max_radii2D[visibility_filter], radii[visibility_filter].float())
                model.add_densification_stats(viewspace_points, visibility_filter)

                if iteration > args.densify_from_iter and iteration % args.densification_interval == 0:
                    size_threshold = 20 if iteration > args.opacity_reset_interval else None
                    model.densify_and_prune(args.densify_grad_threshold, 0.005, scene_extent, size_threshold,
                                             region_mask_fn=lambda xyz: compute_roi_mask(xyz, obb))

                if iteration % args.opacity_reset_interval == 0:
                    model.reset_opacity()

            model.optimizer.step()

        if iteration % 200 == 0:
            with torch.no_grad():
                roi_now = int(compute_roi_mask(model.get_xyz, obb).sum())
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "n_total": model.get_xyz.shape[0], "n_roi": roi_now})

        if iteration % args.save_every == 0:
            model.save_ply(str(output_ply).replace(".ply", f"_{iteration}.ply"))

    elapsed = time.time() - t0
    model.save_ply(str(output_ply))
    with torch.no_grad():
        roi_final = int(compute_roi_mask(model.get_xyz, obb).sum())
    print(f"[{args.instance_id}:{args.category}] Xong trong {elapsed/60:.1f} phút. "
          f"Gaussian: {n_total}->{model.get_xyz.shape[0]}, ROI: {int(roi_mask0.sum())}->{roi_final}")
    print(f"[{args.instance_id}:{args.category}] Checkpoint: {output_ply}")


if __name__ == "__main__":
    main()
