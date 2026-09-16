"""
Render ảnh từ 1 checkpoint Gaussian Splatting (.ply) theo đúng pose camera thật đã dùng để train.

Chạy:
    export PYTHONNOUSERSITE=1
    export DISPLAY=:0
    /home/ml4u/conda_envs/safe-gs/bin/python scripts/render_gs.py \
        --checkpoint outputs/DTU/3dgs/dtu_scan24/scene_gs_30000.ply \
        --output_dir outputs/DTU/3dgs/dtu_scan24/renders_30000

Muốn render checkpoint khác chỉ cần đổi --checkpoint (vd scene_gs_7000.ply) và --output_dir.

Xem chất lượng Gaussian trực quan hơn (video xoay 360° quanh vật thể):
    /home/ml4u/conda_envs/safe-gs/bin/python scripts/render_gs.py \
        --checkpoint outputs/DTU/3dgs/dtu_scan24/scene_gs_30000.ply \
        --output_dir outputs/DTU/3dgs/dtu_scan24/orbit_30000 --orbit --num_frames 90

"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import pycolmap
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.common.config_loader import load_toml_config
from src.module2_scene_gs.scene_branch.scene_trainer import GaussianSceneModel, render_gaussian_model
from src.module2_scene_gs.scene_branch.camera_utils import (
    load_training_cameras, TrainingCamera, get_projection_matrix,
)


def isolate_single_gaussian(model, idx: int, scale_boost: float = 1.0):
    """Tách đúng 1 hạt Gaussian ra thành 1 model riêng để quan sát cận cảnh (phóng to scale_boost lần)."""
    import math as _math

    iso = GaussianSceneModel(sh_degree=model.max_sh_degree)
    iso.active_sh_degree = model.active_sh_degree
    iso._xyz = model._xyz[idx:idx + 1].clone()
    iso._features_dc = model._features_dc[idx:idx + 1].clone()
    iso._features_rest = model._features_rest[idx:idx + 1].clone()
    iso._opacity = model._opacity[idx:idx + 1].clone()
    iso._scaling = model._scaling[idx:idx + 1].clone() + _math.log(max(scale_boost, 1e-6))
    iso._rotation = model._rotation[idx:idx + 1].clone()
    return iso


def pick_best_gaussian_idx(model) -> int:
    """Tự chọn 1 hạt Gaussian to + rõ (opacity cao, scale lớn) để làm ví dụ quan sát."""
    score = model.get_scaling.max(dim=1).values * model.get_opacity.squeeze(-1)
    return int(torch.argmax(score).item())


def build_orbit_cameras(model, real_cameras, num_frames: int, device: str = "cuda",
                         target_override=None, radius_override=None):
    """Sinh N camera ảo xoay đều 360° quanh vật thể, dùng lại intrinsics/FoV của camera thật đầu tiên."""
    ref_cam = real_cameras[0]
    target = target_override if target_override is not None else model._xyz.detach().cpu().numpy().mean(axis=0)

    centers = np.stack([c.camera_center.detach().cpu().numpy() for c in real_cameras])
    offsets = centers - target

    # Truc xoay = eigenvector ung voi phuong sai nho nhat (mat phang cac camera nam gan nhu tren 1 duong tron)
    cov = offsets.T @ offsets
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, 0]
    axis = axis / (np.linalg.norm(axis) + 1e-8)

    height_offset = 0.0 if radius_override is not None else float(np.mean(offsets @ axis))
    radial = offsets - np.outer(offsets @ axis, axis)
    radius = radius_override if radius_override is not None else float(np.mean(np.linalg.norm(radial, axis=1)))

    # 2 vector truc giao voi axis de dung mat phang quy dao
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, arbitrary)
    u = u / (np.linalg.norm(u) + 1e-8)
    v = np.cross(axis, u)

    projection_matrix = get_projection_matrix(0.01, 100.0, ref_cam.fovx, ref_cam.fovy).to(device).transpose(0, 1)

    orbit_cameras = []
    for i in range(num_frames):
        angle = 2 * np.pi * i / num_frames
        cam_pos = target + radius * (np.cos(angle) * u + np.sin(angle) * v) + height_offset * axis

        forward = target - cam_pos
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        right = np.cross(forward, axis)
        right = right / (np.linalg.norm(right) + 1e-8)
        down = np.cross(forward, right)

        R_wc = np.stack([right, down, forward], axis=0)  # world -> camera
        t_wc = -R_wc @ cam_pos

        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R_wc
        w2c[:3, 3] = t_wc
        world_view_transform = torch.tensor(w2c, dtype=torch.float32, device=device).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        orbit_cameras.append(TrainingCamera(
            image_path=None,
            image_height=ref_cam.image_height,
            image_width=ref_cam.image_width,
            fovx=ref_cam.fovx,
            fovy=ref_cam.fovy,
            world_view_transform=world_view_transform,
            full_proj_transform=full_proj_transform,
            camera_center=camera_center,
            image_name=f"orbit_{i:04d}",
        ))
    return orbit_cameras


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_scene.toml", help="Config để lấy sfm_dir/raw_image_dir mặc định")
    parser.add_argument("--checkpoint", required=True, help="Đường dẫn checkpoint .ply muốn render (vd scene_gs_7000.ply, scene_gs_30000.ply)")
    parser.add_argument("--sfm_dir", default=None, help="Mặc định lấy từ config")
    parser.add_argument("--images_dir", default=None, help="Mặc định lấy từ config (raw_image_dir)")
    parser.add_argument("--raw_image_dir", default=None, help="Alias của --images_dir")
    parser.add_argument("--output_dir", required=True, help="Thư mục lưu ảnh render ra")
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--num_views", type=int, default=8, help="Số góc camera muốn render thử (mặc định 8, lấy rải đều)")
    parser.add_argument("--save_compare", action="store_true", help="Lưu thêm ảnh ghép [render | ảnh gốc] để so sánh")
    parser.add_argument("--orbit", action="store_true", help="Render 1 quỹ đạo xoay 360° quanh vật thể thay vì render lại đúng các ảnh training")
    parser.add_argument("--num_frames", type=int, default=90, help="Số khung hình cho quỹ đạo xoay (--orbit)")
    parser.add_argument("--fps", type=int, default=30, help="FPS của video xuất ra (--orbit)")
    parser.add_argument("--isolate_gaussian", action="store_true",
                         help="Chỉ xoay quanh 1 hạt Gaussian duy nhất để soi kỹ (kết hợp với --orbit)")
    parser.add_argument("--gaussian_idx", type=int, default=-1,
                         help="Chỉ số hạt Gaussian muốn soi (--isolate_gaussian). Để -1 thì tự chọn hạt to+rõ nhất")
    parser.add_argument("--scale_boost", type=float, default=15.0,
                         help="Phóng to hạt Gaussian được chọn bao nhiêu lần để dễ nhìn (--isolate_gaussian)")
    args = parser.parse_args()

    cfg = load_toml_config(args.config)
    sfm_dir = Path(args.sfm_dir or cfg.get("sfm_dir"))
    raw_img = args.images_dir or args.raw_image_dir or cfg.get("raw_image_dir")
    images_dir = Path(raw_img)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = GaussianSceneModel.load_ply(args.checkpoint, sh_degree=args.sh_degree, device="cuda")

    rec = pycolmap.Reconstruction(sfm_dir)
    all_names = [img.name for img in rec.images.values()]
    cameras = load_training_cameras(rec, images_dir, all_names, device="cuda")
    print(f"[Loading] Tổng số camera có pose: {len(cameras)}")

    bg_color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")

    if args.isolate_gaussian:
        idx = args.gaussian_idx if args.gaussian_idx >= 0 else pick_best_gaussian_idx(model)
        print(f"[Loading] Đang soi hạt Gaussian số {idx} (phóng to x{args.scale_boost})...")
        iso_model = isolate_single_gaussian(model, idx, scale_boost=args.scale_boost)
        target = iso_model._xyz[0].detach().cpu().numpy()
        mean_scale = float(iso_model.get_scaling.mean().item())
        radius = mean_scale * 25.0
        orbit_cameras = build_orbit_cameras(
            model, cameras, args.num_frames, device="cuda",
            target_override=target, radius_override=radius,
        )
        with torch.no_grad():
            for i, cam in enumerate(orbit_cameras):
                rendered_image, _, _, _ = render_gaussian_model(iso_model, cam, bg_color)
                rendered_np = (rendered_image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                out_path = output_dir / f"frame_{i:04d}.png"
                Image.fromarray(rendered_np).save(out_path)
                if (i + 1) % 10 == 0 or i == len(orbit_cameras) - 1:
                    print(f"[✓] Đã render {i + 1}/{len(orbit_cameras)} khung hình")

        video_path = output_dir / "isolate_gaussian.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(args.fps),
            "-i", str(output_dir / "frame_%04d.png"),
            "-pix_fmt", "yuv420p", str(video_path),
        ], check=True)
        print(f"[Done] Đã ghép video soi hạt Gaussian số {idx}: {video_path}")
        return

    if args.orbit:
        orbit_cameras = build_orbit_cameras(model, cameras, args.num_frames, device="cuda")
        with torch.no_grad():
            for i, cam in enumerate(orbit_cameras):
                rendered_image, _, _, _ = render_gaussian_model(model, cam, bg_color)
                rendered_np = (rendered_image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                out_path = output_dir / f"frame_{i:04d}.png"
                Image.fromarray(rendered_np).save(out_path)
                if (i + 1) % 10 == 0 or i == len(orbit_cameras) - 1:
                    print(f"[✓] Đã render {i + 1}/{len(orbit_cameras)} khung hình")

        video_path = output_dir / "orbit.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(args.fps),
            "-i", str(output_dir / "frame_%04d.png"),
            "-pix_fmt", "yuv420p", str(video_path),
        ], check=True)
        print(f"[Done] Đã ghép video xoay 360°: {video_path}")
        return

    step = max(1, len(cameras) // args.num_views)
    selected = cameras[::step][: args.num_views]

    with torch.no_grad():
        for cam in selected:
            rendered_image, _, _, _ = render_gaussian_model(model, cam, bg_color)
            rendered_np = (rendered_image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            out_path = output_dir / f"render_{Path(cam.image_name).stem}.png"

            if args.save_compare:
                gt_np = (cam.get_image(device="cuda").clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                combined = np.concatenate([rendered_np, gt_np], axis=1)
                Image.fromarray(combined).save(out_path)
            else:
                Image.fromarray(rendered_np).save(out_path)

            print(f"[✓] Đã render: {out_path}")

    print(f"[Done] Đã render {len(selected)} ảnh vào: {output_dir}")


if __name__ == "__main__":
    main()
