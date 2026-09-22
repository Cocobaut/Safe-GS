"""
Chuyển pose ground-truth của Replica (traj.txt + cam_params.json) thành 1 COLMAP sparse
model giả (cameras.bin/images.bin/points3D.bin) — để GOF's train.py đọc được mà KHÔNG
cần chạy COLMAP thật (vốn rất chậm với 2000 ảnh/scene, xem HUONG_DAN_CHAY.md).

Point cloud khởi tạo được back-project từ depth map ground-truth (không phải SfM).

Chạy:
    /home/ml4u/conda_envs/safe-gs/bin/python scripts/replica_to_colmap.py --scene office0

Sinh ra: outputs/Replica/sfm/<scene>/sparse/0/{cameras,images,points3D}.bin
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pycolmap
from PIL import Image


def build_colmap_model(scene: str, replica_root: Path, output_dir: Path,
                        point_cloud_stride_frames: int = 40, point_cloud_stride_pixels: int = 8,
                        camera_stride: int = 1):
    scene_dir = replica_root / scene
    cam = json.loads((replica_root / "cam_params.json").read_text())["camera"]
    fx, fy, cx, cy, depth_scale = cam["fx"], cam["fy"], cam["cx"], cam["cy"], cam["scale"]
    width, height = cam["w"], cam["h"]

    traj = np.loadtxt(scene_dir / "traj.txt").reshape(-1, 4, 4)
    num_frames = traj.shape[0]
    print(f"[{scene}] Số frame trong traj.txt: {num_frames}")

    rec = pycolmap.Reconstruction()
    camera = pycolmap.Camera.create_from_model_name(1, "PINHOLE", fx, width, height)
    camera.params = np.array([fx, fy, cx, cy])
    rec.add_camera(camera)

    rig = pycolmap.Rig()
    rig.rig_id = 1
    rig.add_ref_sensor(pycolmap.sensor_t(pycolmap.SensorType.CAMERA, 1))
    rec.add_rig(rig)

    camera_frame_indices = list(range(0, num_frames, camera_stride))
    for i in camera_frame_indices:
        c2w = traj[i]
        w2c = np.linalg.inv(c2w)
        cam_from_world = pycolmap.Rigid3d(w2c[:3, :4])

        image_name = f"frame{i:06d}.jpg"
        img = pycolmap.Image(name=image_name, camera_id=1, image_id=i + 1)
        rec.add_image_with_trivial_frame(img, cam_from_world)

    print(f"[{scene}] Đã thêm {len(camera_frame_indices)}/{num_frames} camera pose "
          f"(ground-truth, camera_stride={camera_stride}, không qua COLMAP).")

    # Point cloud khoi tao: back-project depth GT tu 1 so frame rai deu
    xyz_all, rgb_all = [], []
    frame_indices = range(0, num_frames, point_cloud_stride_frames)
    for i in frame_indices:
        depth_path = scene_dir / "results" / "depth_image" / f"depth{i:06d}.png"
        image_path = scene_dir / "results" / "image" / f"frame{i:06d}.jpg"
        if not depth_path.exists() or not image_path.exists():
            continue

        depth = np.array(Image.open(depth_path)).astype(np.float32) / depth_scale
        rgb = np.array(Image.open(image_path).convert("RGB"))

        ys, xs = np.meshgrid(
            np.arange(0, height, point_cloud_stride_pixels),
            np.arange(0, width, point_cloud_stride_pixels),
            indexing="ij",
        )
        zs = depth[ys, xs]
        valid = zs > 0
        xs_v, ys_v, zs_v = xs[valid], ys[valid], zs[valid]

        X = (xs_v - cx) * zs_v / fx
        Y = (ys_v - cy) * zs_v / fy
        Z = zs_v
        pts_cam = np.stack([X, Y, Z, np.ones_like(X)], axis=1)

        c2w = traj[i]
        pts_world = (c2w @ pts_cam.T).T[:, :3]

        xyz_all.append(pts_world)
        rgb_all.append(rgb[ys_v, xs_v])

    xyz_all = np.concatenate(xyz_all, axis=0)
    rgb_all = np.concatenate(rgb_all, axis=0)
    print(f"[{scene}] Point cloud khởi tạo từ depth GT: {len(xyz_all)} điểm (từ {len(list(frame_indices))} frame).")

    empty_track = pycolmap.Track()
    for xyz, rgb in zip(xyz_all, rgb_all):
        rec.add_point3D(xyz.astype(np.float64), empty_track, rgb.astype(np.uint8))

    sparse_dir = output_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    rec.write(str(sparse_dir))
    print(f"[{scene}] Đã ghi COLMAP sparse model giả tại: {sparse_dir}")
    return sparse_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, help="Tên scene Replica (vd office0, room1)")
    parser.add_argument("--replica_root", default="../Replica 8 Scene/Replica")
    parser.add_argument("--output_dir", default=None, help="Mặc định outputs/Replica/sfm/<scene>")
    parser.add_argument("--point_cloud_stride_frames", type=int, default=40)
    parser.add_argument("--point_cloud_stride_pixels", type=int, default=8)
    parser.add_argument("--camera_stride", type=int, default=1,
                         help="Chi them 1/N camera vao sparse model (vd 8 -> 250/2000 anh) - giam RAM khi extract_mesh.py load het anh training")
    args = parser.parse_args()

    output_dir = Path(args.output_dir or f"outputs/Replica/sfm/{args.scene}")
    build_colmap_model(

        args.scene, Path(args.replica_root), output_dir,
        args.point_cloud_stride_frames, args.point_cloud_stride_pixels,
        args.camera_stride,
    )


if __name__ == "__main__":
    main()
