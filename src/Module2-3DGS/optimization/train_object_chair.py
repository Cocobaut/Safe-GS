"""
Train "kĩ" chỉ trên 1 object (ghế) thay vì train lại full scene:
  - Khởi tạo Gaussian TỪ file đã tách theo bbox (Replica_office0_3dgs_object_4_chair.ply)
    thay vì từ sparse points của cả scene.
  - Chỉ dùng các camera thật sự nhìn thấy ghế (lấy từ office_0_views.json, instance 4 - 50
    view thay vì toàn bộ 2000 view của scene).
  - Loss chỉ tính trong vùng ảnh 2D chứa ghế (project 8 góc của arbitrary_3d_obb qua từng
    camera để lấy bbox 2D), tránh bị nhiễu bởi nền/vật thể khác trong ảnh gốc.

Muc dich: so sanh toc do/chat luong voi train full scene (scene_trainer.py, ~958k
Gaussian, 30k iterations, ~30-60 phut) khi chi tap trung vao ~23k Gaussian cua ghe.

Chay:
    /home/ml4u/conda_envs/safe-gs/bin/python tmp/train_object_chair.py
"""
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pycolmap
import torch

sys.path.insert(0, "/media/ml4u/Extreme SSD/Safe-GS")

from src.module2_scene_gs.scene_branch.scene_trainer import GaussianSceneModel, render_gaussian_model
from src.module2_scene_gs.scene_branch.camera_utils import load_training_cameras
from src.module2_scene_gs.scene_branch.loss_utils import l1_loss, ssim

CHAIR_PLY = "/media/ml4u/Extreme SSD/Safe-GS/tmp/Replica_office0_3dgs_object_4_chair.ply"
BOXES_JSON = "/media/ml4u/Extreme SSD/Safe-GS/outputs/Replica/bounding_box_3D_and_view_selection/office0/office_0_3d_boxes.json"
VIEWS_JSON = "/media/ml4u/Extreme SSD/Safe-GS/outputs/Replica/bounding_box_3D_and_view_selection/office0/office_0_views.json"
SPARSE_DIR = "/media/ml4u/Extreme SSD/Safe-GS/outputs/Replica/sfm/office0/sparse/0"
IMAGES_DIR = "/media/ml4u/Extreme SSD/Replica 8 Scene/Replica/office0/results/image"
OUTPUT_PLY = "/media/ml4u/Extreme SSD/Safe-GS/tmp/object_gs_chair_trained.ply"

INSTANCE_ID = "4"
ITERATIONS = 30000
SAVE_EVERY = 1000
PADDING_PIXELS = 20


def init_model_from_cropped_ply(ply_path, sh_degree=3, device="cuda"):
    """Load Gaussian đã cắt sẵn, thiết lập lại optimizer + buffer densify để train tiếp."""
    model = GaussianSceneModel.load_ply(ply_path, sh_degree=sh_degree, device=device)
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


def obb_corners_world(obb):
    center = np.array(obb["center"])
    half = np.array(obb["sizes"]) / 2.0
    R = np.array(obb["rotation_matrix"])
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    corners_local = signs * half[None, :]
    corners_world = center[None, :] + corners_local @ R
    return corners_world


def compute_2d_crop_boxes(cameras, corners_world, padding=20):
    """Voi moi camera, project 8 goc OBB -> pixel, lay bbox 2D (co padding, clamp trong anh)."""
    corners_t = torch.tensor(corners_world, dtype=torch.float32, device="cuda")
    crops = {}
    for cam in cameras:
        w2c = cam.world_view_transform.T
        R_wc, t_wc = w2c[:3, :3], w2c[:3, 3]
        p_cam = corners_t @ R_wc.T + t_wc[None, :]
        fx = cam.image_width / (2.0 * math.tan(cam.fovx * 0.5))
        fy = cam.image_height / (2.0 * math.tan(cam.fovy * 0.5))
        cx, cy = cam.image_width / 2.0, cam.image_height / 2.0
        z = torch.clamp(p_cam[:, 2], min=1e-3)
        u = p_cam[:, 0] / z * fx + cx
        v = p_cam[:, 1] / z * fy + cy
        x0 = int(torch.clamp(u.min() - padding, 0, cam.image_width - 1).item())
        x1 = int(torch.clamp(u.max() + padding, 0, cam.image_width - 1).item())
        y0 = int(torch.clamp(v.min() - padding, 0, cam.image_height - 1).item())
        y1 = int(torch.clamp(v.max() + padding, 0, cam.image_height - 1).item())
        if x1 > x0 and y1 > y0:
            crops[cam.image_name] = (x0, y0, x1, y1)
    return crops


def main():
    print("[Loading] Khởi tạo Object-GS từ file đã cắt theo bbox ghế...")
    model = init_model_from_cropped_ply(CHAIR_PLY)
    n_gaussians_init = model.get_xyz.shape[0]
    print(f"[Loading] Số Gaussian ban đầu (chỉ ghế): {n_gaussians_init}")

    views = json.loads(Path(VIEWS_JSON).read_text())[INSTANCE_ID]
    boxes = json.loads(Path(BOXES_JSON).read_text())
    obb = next(b["arbitrary_3d_obb"] for b in boxes if str(b["instance_id"]) == INSTANCE_ID)

    print(f"[Loading] Số camera thấy ghế: {len(views)}")
    rec = pycolmap.Reconstruction(SPARSE_DIR)
    cameras = load_training_cameras(rec, IMAGES_DIR, views, device="cuda")
    print(f"[Loading] Số camera load được pose: {len(cameras)}")

    corners_world = obb_corners_world(obb)
    crop_boxes = compute_2d_crop_boxes(cameras, corners_world, padding=PADDING_PIXELS)
    cameras = [c for c in cameras if c.image_name in crop_boxes]
    print(f"[Loading] Số camera có vùng crop 2D hợp lệ: {len(cameras)}")

    bg_color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")

    print(f"[Loading] Bắt đầu train Object-GS (ghế) {ITERATIONS} iterations...")
    t0 = time.time()
    viewpoint_stack = []
    from tqdm import tqdm
    progress_bar = tqdm(range(1, ITERATIONS + 1), desc="Training Object-GS (chair)")
    for iteration in progress_bar:
        if iteration % 1000 == 0 and model.active_sh_degree < model.max_sh_degree:
            model.active_sh_degree += 1

        if not viewpoint_stack:
            viewpoint_stack = list(cameras)
            random.shuffle(viewpoint_stack)
        cam = viewpoint_stack.pop()
        x0, y0, x1, y1 = crop_boxes[cam.image_name]

        rendered_image, viewspace_points, radii, visibility_filter = render_gaussian_model(model, cam, bg_color)
        gt_image = cam.get_image(device="cuda")

        render_crop = rendered_image[:, y0:y1, x0:x1]
        gt_crop = gt_image[:, y0:y1, x0:x1]

        loss_l1 = l1_loss(render_crop, gt_crop)
        loss_ssim = ssim(render_crop, gt_crop)
        loss = 0.8 * loss_l1 + 0.2 * (1.0 - loss_ssim)

        model.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        with torch.no_grad():
            model.max_radii2D[visibility_filter] = torch.max(
                model.max_radii2D[visibility_filter], radii[visibility_filter].float())
            model.add_densification_stats(viewspace_points, visibility_filter)

            if 200 < iteration < ITERATIONS - 500 and iteration % 100 == 0:
                model.densify_and_prune(0.0002, 0.005, 1.0, 20)

            if iteration % 1500 == 0:
                model.reset_opacity()

            model.optimizer.step()

        if iteration % 100 == 0:
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "n_gaussians": model.get_xyz.shape[0]})

        if iteration % SAVE_EVERY == 0:
            model.save_ply(OUTPUT_PLY.replace(".ply", f"_{iteration}.ply"))

    elapsed = time.time() - t0
    model.save_ply(OUTPUT_PLY)
    print(f"\n[Done] Train xong Object-GS (ghế) trong {elapsed:.1f}s ({elapsed/60:.1f} phút).")
    print(f"[Done] Số Gaussian: {n_gaussians_init} -> {model.get_xyz.shape[0]}")
    print(f"[Done] Checkpoint: {OUTPUT_PLY}")


if __name__ == "__main__":
    main()
