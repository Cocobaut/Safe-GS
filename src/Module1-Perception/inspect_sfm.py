"""
Script kiểm tra/trực quan hóa output SfM (COLMAP) sau khi chạy scripts/01_run_perception.py.

Chạy:
    export DISPLAY=:0   # cần màn hình thật để mở cửa sổ Open3D
    /home/ml4u/conda_envs/safe-gs/bin/python scripts/inspect_sfm.py \
        --sparse_dir outputs/DTU/sfm/dtu_scan24/sparse/0 \
        --visibility_json outputs/DTU/sfm/dtu_scan24/visibility_graph.json


Thêm --no_viz nếu không muốn mở cửa sổ Open3D (chỉ in số liệu).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pycolmap


def print_camera_intrinsics(rec):
    print("\n===== 1. Camera intrinsics (cameras.bin) =====")
    for cam_id, cam in rec.cameras.items():
        print(f"Camera {cam_id} - model {cam.model.name}")
        print(f"  Kích thước ảnh : {cam.width} x {cam.height}")
        print(f"  Tham số (fx, fy, cx, cy) : {cam.params}")


def print_image_poses(rec):
    print("\n===== 2. Pose từng ảnh (images.bin) =====")
    for img_id, img in sorted(rec.images.items(), key=lambda x: x[1].name):
        center = img.projection_center()
        print(f"{img.name} - vị trí camera (x,y,z): {center}")


def print_points_stats(rec):
    print("\n===== 3. Thống kê point cloud (points3D.bin) =====")
    errors = np.array([p.error for p in rec.points3D.values()])
    colors = np.array([p.color for p in rec.points3D.values()])
    print(f"Số điểm 3D       : {len(rec.points3D)}")
    print(f"Reprojection error - mean: {errors.mean():.4f}  max: {errors.max():.4f}")
    print(f"Màu RGB - min: {colors.min(axis=0)}  max: {colors.max(axis=0)}")


def print_visibility_stats(visibility_json_path):
    print("\n===== 4. Visibility graph — mỗi điểm được bao nhiêu ảnh thấy =====")
    with open(visibility_json_path) as f:
        vis = json.load(f)
    counts = np.array([len(v["visible_in_images"]) for v in vis["point_to_images"].values()])
    print(f"Số điểm trong visibility graph : {len(counts)}")
    print(f"Số ảnh thấy 1 điểm - min: {counts.min()}  mean: {counts.mean():.2f}  max: {counts.max()}")

    weak = int((counts == counts.min()).sum())
    p90 = np.percentile(counts, 90)
    strong = int((counts >= p90).sum())
    print(f"Điểm chỉ được {counts.min()} ảnh thấy (yếu nhất, dễ bị nhiễu) : {weak} điểm ({100*weak/len(counts):.1f}%)")
    print(f"Điểm thuộc top 10% được thấy nhiều nhất (>= {p90:.0f} ảnh)   : {strong} điểm")

    img_point_counts = np.array([len(v) for v in vis["image_to_points"].values()])
    print(f"Mỗi ảnh nhìn thấy trung bình {img_point_counts.mean():.1f} điểm (min {img_point_counts.min()}, max {img_point_counts.max()})")


def umeyama_alignment(src, dst):
    """Tìm scale s, xoay R, tịnh tiến t sao cho s*R@src + t ~ dst (Umeyama, có scale)."""
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = (dst_c.T @ src_c) / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_src = (src_c ** 2).sum() / len(src)
    scale = np.trace(np.diag(D) @ S) / var_src
    t = mu_dst - scale * (R @ mu_src)
    return scale, R, t


def compare_with_dtu(rec, dtu_cameras_npz):
    import cv2

    print("\n===== 6. So sánh sparse cloud/pose COLMAP với ground-truth DTU =====")
    data = np.load(dtu_cameras_npz)

    colmap_centers, dtu_centers, colmap_Rs, dtu_Rs = [], [], [], []
    for img in rec.images.values():
        idx = int(Path(img.name).stem)
        key = f"world_mat_{idx}"
        if key not in data:
            continue
        P = data[key][:3, :4]
        K, R_dtu_wc, t_h, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
        dtu_center = (t_h[:3] / t_h[3]).flatten()
        dtu_R_cam_to_world = R_dtu_wc.T

        cfw = img.cam_from_world()
        colmap_R_cam_to_world = cfw.rotation.matrix().T

        colmap_centers.append(img.projection_center())
        dtu_centers.append(dtu_center)
        colmap_Rs.append(colmap_R_cam_to_world)
        dtu_Rs.append(dtu_R_cam_to_world)

    colmap_centers = np.array(colmap_centers)
    dtu_centers = np.array(dtu_centers)

    scale, R_align, t_align = umeyama_alignment(colmap_centers, dtu_centers)
    aligned = scale * (R_align @ colmap_centers.T).T + t_align
    pos_errors = np.linalg.norm(aligned - dtu_centers, axis=1)

    rot_errors = []
    for R_c, R_d in zip(colmap_Rs, dtu_Rs):
        R_c_aligned = R_align @ R_c
        cos_angle = np.clip((np.trace(R_c_aligned.T @ R_d) - 1) / 2, -1, 1)
        rot_errors.append(np.degrees(np.arccos(cos_angle)))
    rot_errors = np.array(rot_errors)

    print(f"Số ảnh so khớp được với ground-truth DTU : {len(colmap_centers)}/{len(rec.images)}")
    print(f"Hệ số scale căn chỉnh (COLMAP -> DTU)     : {scale:.4f}")
    print(f"Sai số vị trí camera (đơn vị DTU) - mean: {pos_errors.mean():.4f}  max: {pos_errors.max():.4f}")
    print(f"Sai số góc xoay (độ)              - mean: {rot_errors.mean():.4f}  max: {rot_errors.max():.4f}")


def export_ply(rec, output_path):
    import open3d as o3d

    print(f"\n===== Xuất point cloud ra .ply: {output_path} =====")
    xyz = np.array([p.xyz for p in rec.points3D.values()])
    rgb = np.array([p.color for p in rec.points3D.values()]) / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), pcd)
    print(f"Đã lưu {len(rec.points3D)} điểm vào {output_path}")


def visualize_open3d(rec, color_by_density=False):
    import open3d as o3d

    print("\n===== 5. Mở cửa sổ Open3D (kéo chuột để xoay, cuộn để zoom) =====")
    xyz = np.array([p.xyz for p in rec.points3D.values()])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if color_by_density:
        print("Tô màu theo mật độ: XANH DƯƠNG = thưa/rỗng, ĐỎ = dày đặc")
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        radius = 0.05 * np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0))
        density = np.array([
            kdtree.search_radius_vector_3d(pt, radius)[0] for pt in xyz
        ], dtype=float)
        density_norm = (density - density.min()) / (density.max() - density.min() + 1e-8)
        colors = np.zeros((len(xyz), 3))
        colors[:, 0] = density_norm       # đỏ tăng theo mật độ
        colors[:, 2] = 1.0 - density_norm  # xanh dương tăng khi thưa
    else:
        colors = np.array([p.color for p in rec.points3D.values()]) / 255.0

    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])


def _is_gaussian_ply(prop_names):
    return "f_dc_0" in prop_names and "opacity" in prop_names


def view_ply(ply_path):
    """Mở trực tiếp 1 file .ply bất kỳ để xem bằng Open3D.

    Nhận diện riêng file checkpoint Gaussian Splatting (có f_dc_0/1/2, opacity, scale, rot) —
    dạng này không có sẵn màu RGB chuẩn nên phải tự chuyển từ hệ số Spherical Harmonics bậc 0
    sang màu thật (công thức nghịch đảo với lúc train: color = f_dc * C0 + 0.5).
    """
    import open3d as o3d
    from plyfile import PlyData

    print(f"\n===== Mở file .ply: {ply_path} =====")
    plydata = PlyData.read(str(ply_path))
    v = plydata["vertex"]
    prop_names = [p.name for p in v.properties]

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if _is_gaussian_ply(prop_names):
        print("Phát hiện file checkpoint Gaussian Splatting — tự chuyển SH DC sang màu RGB.")
        C0 = 0.28209479177387814
        f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float64)
        colors = np.clip(f_dc * C0 + 0.5, 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        print(f"Số hạt Gaussian: {len(xyz)}")
    else:
        if "red" in prop_names:
            colors = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float64) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
        print(f"Số điểm: {len(xyz)}")

    o3d.visualization.draw_geometries([pcd])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse_dir", default="outputs/DTU/sfm/dtu_scan24/sparse/0")
    parser.add_argument("--visibility_json", default="outputs/DTU/sfm/dtu_scan24/visibility_graph.json")
    parser.add_argument("--no_viz", action="store_true", help="Bỏ qua bước mở cửa sổ Open3D")
    parser.add_argument("--export_ply", default=None, help="Đường dẫn file .ply muốn xuất ra (vd outputs/DTU/sfm/dtu_scan24/sparse_cloud.ply)")
    parser.add_argument("--color_by_density", action="store_true", help="Tô màu point cloud theo mật độ (đỏ=dày, xanh dương=thưa) thay vì màu RGB gốc")
    parser.add_argument("--dtu_cameras_npz", default=None, help="Đường dẫn cameras.npz gốc của DTU để so sánh pose (vd '../DTU Preprocess/DTU/scan24/cameras.npz')")
    parser.add_argument("--view_ply", default=None, help="Chỉ mở xem 1 file .ply bất kỳ (vd file ground-truth STL của DTU), bỏ qua toàn bộ phần đọc COLMAP")
    args = parser.parse_args()


    if args.view_ply:
        view_ply(args.view_ply)
        return

    rec = pycolmap.Reconstruction(args.sparse_dir)

    print_camera_intrinsics(rec)
    print_image_poses(rec)
    print_points_stats(rec)
    print_visibility_stats(args.visibility_json)

    if args.dtu_cameras_npz:
        compare_with_dtu(rec, args.dtu_cameras_npz)

    if args.export_ply:
        export_ply(rec, args.export_ply)

    if not args.no_viz:
        visualize_open3d(rec, color_by_density=args.color_by_density)


if __name__ == "__main__":
    main()
