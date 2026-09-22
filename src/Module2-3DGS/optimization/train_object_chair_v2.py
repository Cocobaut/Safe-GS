"""
Ban 2: Van train RIENG object (nhanh) nhu train_object_chair.py, nhung FIX van de thieu
chi tiet mong (chan ghe) bang cach BO SUNG diem khoi tao day dac hon CHI trong vung ghe
truoc khi train - vi nguyen nhan thieu chan ghe la do point cloud khoi tao ca scene lay
mau depth cach 8 pixel (point_cloud_stride_pixels=8 trong replica_to_colmap.py), qua thua
cho cau truc mong nhu chan ghe.

Cach lam:
  1. Lay 22,847 diem/Gaussian da tach san (Replica_office0_3dgs_object_4_chair.ply).
  2. Voi 1 so view lien quan, back-project depth TUNG PIXEL (khong stride) trong vung 2D
     chieu tu 3D OBB, giu lai diem nam trong OBB that (khong chi 2D).
  3. Loc theo luoi voxel nho (5mm) - CHI giu diem moi nam o o voxel CHUA co diem cu -> chi
     vao lap dung cho fix thieu vao thoi, khong lam phinh to vo ich.
  4. Khoi tao Gaussian moi cho cac diem vua them (mau tu RGB, scale nho co dinh, opacity
     0.1), noi vao 22,847 Gaussian cu (giu nguyen tham so da hoc), roi train tiep y het
     nhu train_object_chair.py (nhanh, camera + loss khoanh vung 2D quanh ghe).

Chay:
    /home/ml4u/conda_envs/safe-gs/bin/python tmp/train_object_chair_v2.py
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
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "/media/ml4u/Extreme SSD/Safe-GS")

from src.module2_scene_gs.scene_branch.scene_trainer import GaussianSceneModel, render_gaussian_model
from src.module2_scene_gs.scene_branch.camera_utils import load_training_cameras
from src.module2_scene_gs.scene_branch.loss_utils import l1_loss, ssim

CHAIR_PLY = "/media/ml4u/Extreme SSD/Safe-GS/tmp/Replica_office0_3dgs_object_4_chair.ply"
BOXES_JSON = "/media/ml4u/Extreme SSD/Safe-GS/outputs/Replica/bounding_box_3D_and_view_selection/office0/office_0_3d_boxes.json"
VIEWS_JSON = "/media/ml4u/Extreme SSD/Safe-GS/outputs/Replica/bounding_box_3D_and_view_selection/office0/office_0_views.json"
SPARSE_DIR = "/media/ml4u/Extreme SSD/Safe-GS/outputs/Replica/sfm/office0/sparse/0"
IMAGES_DIR = "/media/ml4u/Extreme SSD/Replica 8 Scene/Replica/office0/results/image"
DEPTH_DIR = "/media/ml4u/Extreme SSD/Replica 8 Scene/Replica/office0/results/depth_image"
CAM_PARAMS_JSON = "/media/ml4u/Extreme SSD/Replica 8 Scene/Replica/cam_params.json"
OUTPUT_PLY = "/media/ml4u/Extreme SSD/Safe-GS/tmp/object_gs_chair_trained_v2.ply"

INSTANCE_ID = "4"
ITERATIONS = 7000
SAVE_EVERY = 1000
PADDING_PIXELS = 20
DENSE_INIT_NUM_VIEWS = 12   # so view dung de lay mau day dac (trong 50 view lien quan)
DENSE_INIT_STRIDE = 1       # 1 = tung pixel
VOXEL_SIZE = 0.005          # 5mm - kich thuoc o luoi de xac dinh "da co diem hay chua"


def obb_corners_world(obb):
    center = np.array(obb["center"])
    half = np.array(obb["sizes"]) / 2.0
    R = np.array(obb["rotation_matrix"])
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    corners_local = signs * half[None, :]
    return center[None, :] + corners_local @ R


def points_in_obb(xyz, obb):
    center = np.array(obb["center"])
    half = np.array(obb["sizes"]) / 2.0
    R = np.array(obb["rotation_matrix"])
    local = (xyz - center[None, :]) @ R.T
    return np.all(np.abs(local) <= half[None, :], axis=1)


def build_dense_gap_fill_points(existing_xyz, obb, views, num_views, stride, voxel_size):
    """Back-project depth tung pixel trong vung OBB tu 1 so view, chi giu diem vao cho
    trong (voxel chua co diem cu nao)."""
    cam = json.loads(Path(CAM_PARAMS_JSON).read_text())["camera"]
    fx, fy, cx, cy, depth_scale = cam["fx"], cam["fy"], cam["cx"], cam["cy"], cam["scale"]
    width, height = cam["w"], cam["h"]

    traj = np.loadtxt("/media/ml4u/Extreme SSD/Replica 8 Scene/Replica/office0/traj.txt").reshape(-1, 4, 4)
    corners_world = obb_corners_world(obb)

    existing_voxels = set(map(tuple, np.floor(existing_xyz / voxel_size).astype(np.int64)))

    chosen_views = random.sample(views, min(num_views, len(views)))
    new_xyz, new_rgb = [], []

    for frame_name in chosen_views:
        idx = int(Path(frame_name).stem.replace("frame", ""))
        c2w = traj[idx]
        w2c = np.linalg.inv(c2w)
        R_wc, t_wc = w2c[:3, :3], w2c[:3, 3]

        corners_cam = corners_world @ R_wc.T + t_wc[None, :]
        z = np.clip(corners_cam[:, 2], 1e-3, None)
        u = corners_cam[:, 0] / z * fx + cx
        v = corners_cam[:, 1] / z * fy + cy
        x0, x1 = int(max(u.min(), 0)), int(min(u.max(), width - 1))
        y0, y1 = int(max(v.min(), 0)), int(min(v.max(), height - 1))
        if x1 <= x0 or y1 <= y0:
            continue

        depth_path = Path(DEPTH_DIR) / f"depth{idx:06d}.png"
        image_path = Path(IMAGES_DIR) / frame_name
        if not depth_path.exists() or not image_path.exists():
            continue
        depth = np.array(Image.open(depth_path)).astype(np.float32) / depth_scale
        rgb_img = np.array(Image.open(image_path).convert("RGB"))

        ys, xs = np.meshgrid(np.arange(y0, y1, stride), np.arange(x0, x1, stride), indexing="ij")
        zs = depth[ys, xs]
        valid = zs > 0
        xs_v, ys_v, zs_v = xs[valid], ys[valid], zs[valid]
        if len(xs_v) == 0:
            continue

        X = (xs_v - cx) * zs_v / fx
        Y = (ys_v - cy) * zs_v / fy
        pts_cam = np.stack([X, Y, zs_v, np.ones_like(X)], axis=1)
        pts_world = (c2w @ pts_cam.T).T[:, :3]

        inside = points_in_obb(pts_world, obb)
        pts_world, colors = pts_world[inside], rgb_img[ys_v, xs_v][inside]
        if len(pts_world) == 0:
            continue

        voxels = np.floor(pts_world / voxel_size).astype(np.int64)
        keep = np.array([tuple(v) not in existing_voxels for v in voxels])
        pts_world, colors, voxels = pts_world[keep], colors[keep], voxels[keep]

        for p, c, vx in zip(pts_world, colors, voxels):
            key = tuple(vx)
            if key in existing_voxels:
                continue
            existing_voxels.add(key)
            new_xyz.append(p)
            new_rgb.append(c)

    if not new_xyz:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
    return np.array(new_xyz), np.array(new_rgb, dtype=np.uint8)


def init_model_with_gap_fill(chair_ply, obb, views, sh_degree=3, device="cuda"):
    model = GaussianSceneModel.load_ply(chair_ply, sh_degree=sh_degree, device=device)
    existing_xyz = model.get_xyz.detach().cpu().numpy()
    n_existing = existing_xyz.shape[0]

    print("[Loading] Đang bổ sung điểm khởi tạo dày hơn cho vùng ghế (vá lỗ chân ghế)...")
    new_xyz, new_rgb = build_dense_gap_fill_points(
        existing_xyz, obb, views, DENSE_INIT_NUM_VIEWS, DENSE_INIT_STRIDE, VOXEL_SIZE)
    print(f"[Loading] Số điểm mới bổ sung (vào chỗ trống): {len(new_xyz)}")

    if len(new_xyz) > 0:
        C0 = 0.28209479177387814
        new_xyz_t = torch.tensor(new_xyz, dtype=torch.float32, device=device)
        new_rgb_t = torch.tensor(new_rgb / 255.0, dtype=torch.float32, device=device)
        n_new = new_xyz_t.shape[0]

        new_features = torch.zeros((n_new, 3, (sh_degree + 1) ** 2), device=device)
        new_features[:, :3, 0] = (new_rgb_t - 0.5) / C0
        new_f_dc = new_features[:, :, 0:1].transpose(1, 2).contiguous()
        new_f_rest = new_features[:, :, 1:].transpose(1, 2).contiguous()

        new_scaling = torch.log(torch.full((n_new, 3), VOXEL_SIZE, device=device))
        new_rotation = torch.zeros((n_new, 4), device=device)
        new_rotation[:, 0] = 1.0
        new_opacity = torch.logit(0.1 * torch.ones((n_new, 1), device=device))

        model._xyz = torch.cat([model._xyz.detach(), new_xyz_t], dim=0)
        model._features_dc = torch.cat([model._features_dc.detach(), new_f_dc], dim=0)
        model._features_rest = torch.cat([model._features_rest.detach(), new_f_rest], dim=0)
        model._scaling = torch.cat([model._scaling.detach(), new_scaling], dim=0)
        model._rotation = torch.cat([model._rotation.detach(), new_rotation], dim=0)
        model._opacity = torch.cat([model._opacity.detach(), new_opacity], dim=0)

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
    print(f"[Loading] Tổng Gaussian sau khi vá: {n_existing} -> {n}")
    return model


def compute_2d_crop_boxes(cameras, corners_world, padding=20):
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
    views = json.loads(Path(VIEWS_JSON).read_text())[INSTANCE_ID]
    boxes = json.loads(Path(BOXES_JSON).read_text())
    obb = next(b["arbitrary_3d_obb"] for b in boxes if str(b["instance_id"]) == INSTANCE_ID)

    model = init_model_with_gap_fill(CHAIR_PLY, obb, views)

    print(f"[Loading] Số camera thấy ghế: {len(views)}")
    rec = pycolmap.Reconstruction(SPARSE_DIR)
    cameras = load_training_cameras(rec, IMAGES_DIR, views, device="cuda")

    corners_world = obb_corners_world(obb)
    crop_boxes = compute_2d_crop_boxes(cameras, corners_world, padding=PADDING_PIXELS)
    cameras = [c for c in cameras if c.image_name in crop_boxes]
    print(f"[Loading] Số camera có vùng crop 2D hợp lệ: {len(cameras)}")

    bg_color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")

    print(f"[Loading] Bắt đầu train Object-GS (ghế, đã vá) {ITERATIONS} iterations...")
    t0 = time.time()
    viewpoint_stack = []
    progress_bar = tqdm(range(1, ITERATIONS + 1), desc="Training Object-GS v2 (chair)")
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
    print(f"\n[Done] Train xong trong {elapsed:.1f}s ({elapsed/60:.1f} phút).")
    print(f"[Done] Checkpoint: {OUTPUT_PLY}")


if __name__ == "__main__":
    main()
