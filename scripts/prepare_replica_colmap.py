"""
prepare_replica_colmap.py
=========================
Chuẩn bị dữ liệu Pha 0 cho Replica (Safe-GS):
Chuyển đổi trực tiếp từ render Replica (NICE-SLAM: cam_params.json + traj.txt + mesh.ply)
sang định dạng COLMAP (sparse/0/cameras.*, images.*, points3D.*) và lấy mẫu 250 frame ảnh.

Tránh hoàn toàn rủi ro COLMAP feature matching thất bại do bề mặt render phẳng, ít texture,
đồng thời giữ 100% camera được đăng ký đúng tuyệt đối và giữ nguyên thang đo mét (metric scale).

Cách dùng:
    python scripts/prepare_replica_colmap.py --scenes office0 room0
    python scripts/prepare_replica_colmap.py --scenes office0 --num-frames 250 --num-points 150000
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

# Tự động nạp đường dẫn PROJECT_ROOT nếu có
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from common.paths import DATA_DIR
    DEFAULT_DATA_DIR = Path(DATA_DIR)
except Exception:
    DEFAULT_DATA_DIR = Path("F:/")


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    """Chuyển ma trận quay 3x3 sang quaternion (qw, qx, qy, qz) theo chuẩn COLMAP."""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0.0, 0.0, 0.0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0.0, 0.0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0.0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]
    ]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def write_cameras_text(path: Path, width: int, height: int, fx: float, fy: float, cx: float, cy: float):
    """Ghi cameras.txt chuẩn PINHOLE."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {width} {height} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")


def write_cameras_binary(path: Path, width: int, height: int, fx: float, fy: float, cx: float, cy: float):
    """Ghi cameras.bin chuẩn PINHOLE (model_id=1)."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 1))  # 1 camera
        # camera_id (i), model_id (i), width (Q), height (Q)
        f.write(struct.pack("<iiQQ", 1, 1, width, height))
        # fx, fy, cx, cy (4 x double)
        f.write(struct.pack("<4d", fx, fy, cx, cy))


def write_images_text(path: Path, image_entries: list[dict]):
    """Ghi images.txt chuẩn COLMAP."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for e in image_entries:
            q = e["qvec"]
            t = e["tvec"]
            f.write(f"{e['id']} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} "
                    f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {e['camera_id']} {e['name']}\n\n")


def write_images_binary(path: Path, image_entries: list[dict]):
    """Ghi images.bin chuẩn COLMAP."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(image_entries)))
        for e in image_entries:
            q = e["qvec"]
            t = e["tvec"]
            # image_id (i), qw, qx, qy, qz (4d), tx, ty, tz (3d), camera_id (i)
            f.write(struct.pack("<idddddddi", e["id"], q[0], q[1], q[2], q[3], t[0], t[1], t[2], e["camera_id"]))
            f.write(e["name"].encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", 0))  # 0 points2D


def write_points3D_text(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    """Ghi points3D.txt chuẩn COLMAP."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for i in range(len(xyz)):
            x, y, z = xyz[i]
            r, g, b = rgb[i]
            f.write(f"{i + 1} {x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)} 0.0\n")


def write_points3D_binary(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    """Ghi points3D.bin chuẩn COLMAP."""
    with open(path, "wb") as f:
        n = len(xyz)
        f.write(struct.pack("<Q", n))
        for i in range(n):
            x, y, z = xyz[i]
            r, g, b = rgb[i]
            # point_id (Q), x, y, z (3d), r, g, b (3B), error (d)
            f.write(struct.pack("<QdddBBBd", i + 1, x, y, z, int(r), int(g), int(b), 0.0))
            f.write(struct.pack("<Q", 0))  # track_length = 0


def write_points3D_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    """Ghi points3D.ply chuẩn 3DGS."""
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    elements = np.empty(len(xyz), dtype=dtype)
    elements["x"] = xyz[:, 0]
    elements["y"] = xyz[:, 1]
    elements["z"] = xyz[:, 2]
    elements["nx"] = 0.0
    elements["ny"] = 0.0
    elements["nz"] = 0.0
    elements["red"] = rgb[:, 0].astype(np.uint8)
    elements["green"] = rgb[:, 1].astype(np.uint8)
    elements["blue"] = rgb[:, 2].astype(np.uint8)

    vertex_el = PlyElement.describe(elements, "vertex")
    PlyData([vertex_el], byte_order="<").write(str(path))


def process_scene(scene: str, raw_dir: Path, output_dir: Path, num_frames: int, num_points: int, seed: int = 42):
    print(f"\n=======================================================")
    print(f"[*] Đang xử lý scene: {scene}")
    print(f"=======================================================")

    scene_raw = raw_dir / scene
    if not scene_raw.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục scene raw: {scene_raw}")

    # 1. Đọc intrinsics
    cam_json_path = raw_dir / "cam_params.json"
    if not cam_json_path.exists():
        cam_json_path = scene_raw / "cam_params.json"
    if not cam_json_path.exists():
        raise FileNotFoundError(f"Không tìm thấy cam_params.json tại {raw_dir} hoặc {scene_raw}")

    with open(cam_json_path, "r", encoding="utf-8") as f:
        cam_params = json.load(f)["camera"]

    width = int(cam_params["w"])
    height = int(cam_params["h"])
    fx = float(cam_params["fx"])
    fy = float(cam_params["fy"])
    cx = float(cam_params["cx"])
    cy = float(cam_params["cy"])
    print(f"  - Camera intrinsics: {width}x{height}, fx={fx}, fy={fy}, cx={cx}, cy={cy}")

    # 2. Đọc trajectory (C2W ground-truth)
    traj_path = scene_raw / "traj.txt"
    if not traj_path.exists():
        raise FileNotFoundError(f"Không tìm thấy traj.txt tại {traj_path}")

    with open(traj_path, "r", encoding="utf-8") as f:
        lines = [line.strip().split() for line in f if line.strip()]

    total_poses = len(lines)
    all_c2w = np.array(lines, dtype=np.float64).reshape(-1, 4, 4)
    print(f"  - Tổng số poses trong traj.txt: {total_poses}")

    # 3. Lấy danh sách ảnh
    img_dir = scene_raw / "results" / "image"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {img_dir}")

    all_images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    if len(all_images) != total_poses:
        print(f"  [!] Cảnh báo: Số lượng ảnh ({len(all_images)}) khác số pose ({total_poses}). Dùng min.")
        total_count = min(len(all_images), total_poses)
    else:
        total_count = total_poses

    # 4. Lấy mẫu đều num_frames
    sample_indices = np.linspace(0, total_count - 1, min(num_frames, total_count), dtype=int)
    print(f"  - Đã chọn lấy mẫu {len(sample_indices)} frames (từ {total_count} frames)")

    # 5. Tạo cấu trúc thư mục đích
    scene_out = output_dir / scene
    images_out = scene_out / "images"
    sparse_out = scene_out / "sparse" / "0"
    gt_mesh_out = scene_out / "gt_mesh"

    images_out.mkdir(parents=True, exist_ok=True)
    sparse_out.mkdir(parents=True, exist_ok=True)
    gt_mesh_out.mkdir(parents=True, exist_ok=True)

    # 6. Copy ảnh và lập danh sách extrinsic COLMAP (W2C)
    image_entries = []
    sampled_c2w_lines = []

    print(f"  - Đang sao chép {len(sample_indices)} ảnh và chuyển đổi pose...")
    for new_id, idx in enumerate(sample_indices, start=1):
        img_name = all_images[idx]
        src_img = img_dir / img_name
        dst_img = images_out / img_name

        if not dst_img.exists() or dst_img.stat().st_size != src_img.stat().st_size:
            shutil.copy2(src_img, dst_img)

        c2w = all_c2w[idx]
        sampled_c2w_lines.append(" ".join(f"{v:.18e}" for v in c2w.flatten()))

        # c2w -> w2c
        R_c2w = c2w[:3, :3]
        t_c2w = c2w[:3, 3]
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ t_c2w
        qvec = rotmat2qvec(R_w2c)

        image_entries.append({
            "id": new_id,
            "qvec": qvec,
            "tvec": t_w2c,
            "camera_id": 1,
            "name": img_name
        })

    # Lưu traj_gt.txt của đúng các frame đã chọn
    with open(scene_out / "traj_gt.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(sampled_c2w_lines) + "\n")
    print(f"  - Đã lưu: {scene_out / 'traj_gt.txt'}")

    # Ghi cameras.txt và cameras.bin
    write_cameras_text(sparse_out / "cameras.txt", width, height, fx, fy, cx, cy)
    write_cameras_binary(sparse_out / "cameras.bin", width, height, fx, fy, cx, cy)
    print(f"  - Đã lưu cameras.txt & cameras.bin")

    # Ghi images.txt và images.bin
    write_images_text(sparse_out / "images.txt", image_entries)
    write_images_binary(sparse_out / "images.bin", image_entries)
    print(f"  - Đã lưu images.txt & images.bin")

    # 7. Đọc mesh và lấy mẫu 3D points
    mesh_candidates = [
        raw_dir / f"{scene}_mesh.ply",
        scene_raw / "mesh.ply",
        raw_dir / f"{scene}.ply",
    ]
    mesh_path = None
    for cand in mesh_candidates:
        if cand.exists():
            mesh_path = cand
            break

    if mesh_path is None:
        raise FileNotFoundError(f"Không tìm thấy mesh của scene {scene} tại các vị trí: {mesh_candidates}")

    print(f"  - Đang đọc mesh: {mesh_path.name}")
    ply_data = PlyData.read(str(mesh_path))
    vertices = ply_data["vertex"]
    xyz_all = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=-1).astype(np.float32)

    has_rgb = all(prop in vertices for prop in ["red", "green", "blue"])
    if has_rgb:
        rgb_all = np.stack([vertices["red"], vertices["green"], vertices["blue"]], axis=-1).astype(np.uint8)
    else:
        print("  [!] Mesh không chứa thuộc tính RGB, dùng màu xám mặc định (128, 128, 128)")
        rgb_all = np.full((len(xyz_all), 3), 128, dtype=np.uint8)

    # Lấy mẫu ngẫu nhiên có kiểm soát seed
    rng = np.random.default_rng(seed)
    n_sample = min(num_points, len(xyz_all))
    sample_pt_indices = rng.choice(len(xyz_all), size=n_sample, replace=False)
    xyz_sampled = xyz_all[sample_pt_indices]
    rgb_sampled = rgb_all[sample_pt_indices]

    print(f"  - Đã lấy mẫu {n_sample} điểm 3D từ mesh tổng cộng {len(xyz_all)} đỉnh")

    # Ghi points3D.txt, points3D.bin và points3D.ply
    write_points3D_text(sparse_out / "points3D.txt", xyz_sampled, rgb_sampled)
    write_points3D_binary(sparse_out / "points3D.bin", xyz_sampled, rgb_sampled)
    write_points3D_ply(sparse_out / "points3D.ply", xyz_sampled, rgb_sampled)
    print(f"  - Đã lưu points3D.txt, points3D.bin, points3D.ply")

    # 8. Sao chép mesh gốc sang gt_mesh/scene_gt.ply
    dst_scene_gt = gt_mesh_out / "scene_gt.ply"
    if not dst_scene_gt.exists() or dst_scene_gt.stat().st_size != mesh_path.stat().st_size:
        shutil.copy2(mesh_path, dst_scene_gt)
        print(f"  - Đã sao chép mesh GT sang: {dst_scene_gt}")

    # 9. Ghi file T_colmap_to_gt.json (Identity, vì hệ tọa độ COLMAP trùng khớp 100% với GT)
    t_gt = {
        "scale": 1.0,
        "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "translation": [0.0, 0.0, 0.0],
        "sim3": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ],
        "note": "COLMAP coordinate system is built directly from GT camera poses and GT mesh, so Sim(3) is Identity."
    }
    with open(scene_out / "T_colmap_to_gt.json", "w", encoding="utf-8") as f:
        json.dump(t_gt, f, indent=2)
    print(f"  - Đã tạo: {scene_out / 'T_colmap_to_gt.json'} (Identity Sim(3))")

    print(f"[V] Hoàn tất chuẩn bị cho scene: {scene} -> {scene_out}")


def main():
    parser = argparse.ArgumentParser(description="Chuyển đổi Replica NICE-SLAM sang định dạng COLMAP cho Safe-GS.")
    parser.add_argument("--scenes", nargs="+", default=["office0", "room0"],
                        help="Danh sách scene cần xử lý (mặc định: office0 room0)")
    parser.add_argument("--raw-dir", type=str, default="F:/Replica 8 Scene/Replica",
                        help="Thư mục raw render Replica")
    parser.add_argument("--output-dir", type=str, default="F:/Replica 8 Scene/Replica_safegs",
                        help="Thư mục đích lưu dataset chuẩn bị")
    parser.add_argument("--num-frames", type=int, default=250,
                        help="Số lượng frame lấy mẫu đều từ chuỗi ~2000 frame (mặc định: 250)")
    parser.add_argument("--num-points", type=int, default=150000,
                        help="Số điểm 3D lấy mẫu từ mesh làm điểm khởi tạo (mặc định: 150000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed cho lấy mẫu điểm (mặc định: 42)")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    print(f"=== Chuẩn bị dữ liệu Pha 0 (Replica -> COLMAP) ===")
    print(f"Raw dir:    {raw_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Scenes:     {args.scenes}")
    print(f"Frames:     {args.num_frames} frames/scene")
    print(f"Points:     {args.num_points} points/scene")

    for sc in args.scenes:
        process_scene(
            scene=sc,
            raw_dir=raw_dir,
            output_dir=output_dir,
            num_frames=args.num_frames,
            num_points=args.num_points,
            seed=args.seed
        )

    print("\n[V] TẤT CẢ CÁC SCENE ĐÃ ĐƯỢC CHUẨN BỊ XONG!")


if __name__ == "__main__":
    main()

