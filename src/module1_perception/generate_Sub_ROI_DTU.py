import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import open3d as o3d

# Đảm bảo hiển thị UTF-8 trên Windows Console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Thêm repo root vào sys.path để import an toàn module cấu hình dù script chạy từ bất kỳ thư mục nào
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.common.config_loader import load_toml_config
    from src.common.dataset_config import load_dataset_paths
except ImportError:
    load_toml_config = None
    load_dataset_paths = None

# ==============================================================================
# CẤU HÌNH ĐƯỜNG DẪN MẶC ĐỊNH (RELATIVE CONFIG PATHS)
# ==============================================================================
# 1. DEFAULT_DTU_IMAGE_DIR:
#    Thư mục chứa preprocessed DTU data (cameras.npz, mask/, images/, points.ply).
#    Mặc định: "../DTU Preprocess/DTU" (ngang hàng với repo Safe-GS).
#    Ví dụ scan24: "../DTU Preprocess/DTU/scan24/mask"
DEFAULT_DTU_IMAGE_DIR = REPO_ROOT.parent / "DTU Preprocess" / "DTU"

# 2. DEFAULT_STL_DIR:
#    Thư mục chứa ground-truth STL dense point clouds (stlXXX_total.ply).
#    Mặc định: "../DTU Dataset/Points/Points/stl" (ngang hàng với repo Safe-GS).
#    Lưu ý: Parameter này ĐƯỢC SỬ DỤNG để đọc dense point cloud gốc (>5 triệu điểm/scan)
#    nhằm tính toán Bounding Box chính xác và phân chia 16 sub-ROIs.
#    Hệ thống cũng tự động hỗ trợ fallback sang SfM points (points.ply) nếu file STL không có sẵn.
DEFAULT_STL_DIR = REPO_ROOT.parent / "DTU Dataset" / "Points" / "Points" / "stl"

# 3. DEFAULT_OUTPUT_DIR:
#    Thư mục lưu các file JSON Bounding Box và 16 Sub-ROIs.
#    Mặc định: "outputs/DTU/sub_bounding_box_3D" (tương đối theo Safe-GS repo root).
#    Ví dụ scan24: "outputs/DTU/sub_bounding_box_3D/scan24"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "DTU" / "sub_bounding_box_3D"

# Ngưỡng đồng thuận đa góc nhìn (Multi-view consensus threshold)
DEFAULT_VISIBILITY_THRESHOLD = 0.85

# Tỉ lệ nới rộng an toàn (3%)
DEFAULT_EXPANSION_RATIO = 0.03

# Phân bố ô chia nhỏ trong không gian 3D (4 x 2 x 2 = 16 sub-ROIs)
DEFAULT_GRID_SPLITS = (4, 2, 2)


def resolve_path(p: Union[str, Path], base_dir: Path = REPO_ROOT) -> Path:
    """Chuyển đổi đường dẫn tương đối thành Path tuyệt đối chuẩn hóa dựa trên base_dir."""
    path_obj = Path(p)
    if path_obj.is_absolute():
        return path_obj
    return (base_dir / path_obj).resolve()


def extract_scan_number(folder_name: str) -> int:
    """Trích xuất số nguyên từ tên thư mục (ví dụ: 'scan24' -> 24, 'scan105' -> 105)."""
    match = re.search(r"\d+", folder_name)
    return int(match.group(0)) if match else -1


def load_cameras_and_masks(cameras_npz_path: Union[str, Path], mask_dir: Union[str, Path]):
    """
    Nạp ma trận camera từ cameras.npz và các ảnh mask tương ứng.
    Quy ước DTU / IDR: world_mat_i là ma trận chiếu trực tiếp từ
    hệ tọa độ mm thực tế sang pixel 2D (1200x1600).
    """
    cameras_npz_path = str(cameras_npz_path)
    mask_dir = str(mask_dir)

    if not os.path.exists(cameras_npz_path):
        return [], []
    if not os.path.exists(mask_dir):
        return [], []

    camera_dict = np.load(cameras_npz_path)
    mask_files = sorted(
        glob.glob(os.path.join(mask_dir, "*.png")) + 
        glob.glob(os.path.join(mask_dir, "*.jpg"))
    )

    cameras = []
    masks = []

    for mask_path in mask_files:
        filename = os.path.basename(mask_path)
        # Bỏ qua các file ẩn/tạm (ví dụ resource fork của macOS: ._000.png)
        if filename.startswith("."):
            continue

        basename = os.path.splitext(filename)[0]
        digits = "".join(filter(str.isdigit, basename))
        if not digits:
            continue
        idx = int(digits)

        world_mat_key = f"world_mat_{idx}"
        if world_mat_key not in camera_dict:
            continue

        P = camera_dict[world_mat_key][:3, :4].astype(np.float64)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            continue
        mask_bin = (mask_img > 128).astype(np.uint8)

        cameras.append(P)
        masks.append(mask_bin)

    return cameras, masks


def project_points_to_masks(points_3d, cameras, masks, threshold=0.85):
    """
    Chiếu tập điểm 3D lên toàn bộ các camera và lọc các điểm thực sự thuộc vật thể.
    """
    num_pts = len(points_3d)
    if num_pts == 0:
        return np.zeros(0, dtype=bool)

    pts_homo = np.hstack([points_3d, np.ones((num_pts, 1), dtype=np.float64)]).T  # 4 x N

    in_mask_counts = np.zeros(num_pts, dtype=np.int32)
    valid_camera_counts = np.zeros(num_pts, dtype=np.int32)

    for P, mask in zip(cameras, masks):
        H, W = mask.shape
        proj = P @ pts_homo  # 3 x N
        depth = proj[2, :]

        front_mask = depth > 1e-4
        if not np.any(front_mask):
            continue

        u = np.full(num_pts, -1.0, dtype=np.float64)
        v = np.full(num_pts, -1.0, dtype=np.float64)
        u[front_mask] = proj[0, front_mask] / depth[front_mask]
        v[front_mask] = proj[1, front_mask] / depth[front_mask]

        valid_pixels = front_mask & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
        valid_indices = np.where(valid_pixels)[0]
        if len(valid_indices) == 0:
            continue

        valid_camera_counts[valid_indices] += 1

        u_idx = np.clip(np.round(u[valid_pixels]).astype(np.int32), 0, W - 1)
        v_idx = np.clip(np.round(v[valid_pixels]).astype(np.int32), 0, H - 1)
        hit_mask = mask[v_idx, u_idx] == 1
        in_mask_counts[valid_indices[hit_mask]] += 1

    visibility_ratio = np.divide(
        in_mask_counts,
        np.maximum(valid_camera_counts, 1),
        dtype=np.float64
    )
    # Phải có ít nhất 4 camera quan sát và tỉ lệ rơi vào mask >= threshold
    is_object = (visibility_ratio >= threshold) & (valid_camera_counts >= 4)
    return is_object


def compute_bbox_info(pts, name, category, instance_id, expansion_ratio=0.03):
    """Tính toàn bộ AABB, Gravity-aligned OBB và Arbitrary OBB chuẩn xác cho cụm điểm."""
    scale = 1.0 + expansion_ratio

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # 1. AABB
    aabb_min = np.min(pts, axis=0)
    aabb_max = np.max(pts, axis=0)
    aabb_center = (aabb_min + aabb_max) / 2.0
    aabb_sizes = (aabb_max - aabb_min) * scale
    aabb_min_scaled = aabb_center - aabb_sizes / 2.0
    aabb_max_scaled = aabb_center + aabb_sizes / 2.0

    # 2. Arbitrary 3D OBB (PCA 3D)
    obb_3d = pcd.get_oriented_bounding_box()
    obb_center = np.array(obb_3d.center)
    obb_extent = np.array(obb_3d.extent) * scale
    obb_rot = np.array(obb_3d.R)

    # 3. Gravity-aligned OBB (Xoay quanh Z)
    pts_xy = pts[:, :2]
    xy_mean = np.mean(pts_xy, axis=0)
    cov = np.cov(pts_xy - xy_mean, rowvar=False)
    _, eig_vecs = np.linalg.eigh(cov)

    pts_proj = (pts_xy - xy_mean) @ eig_vecs
    min_proj = np.min(pts_proj, axis=0)
    max_proj = np.max(pts_proj, axis=0)
    xy_sizes = (max_proj - min_proj) * scale
    xy_center_proj = (min_proj + max_proj) / 2.0
    xy_center = xy_mean + (xy_center_proj @ eig_vecs.T)
    yaw_rad = float(np.arctan2(eig_vecs[1, 0], eig_vecs[0, 0]))

    return {
        "instance_id": instance_id,
        "name": name,
        "category": category,
        "num_points": int(len(pts)),
        "aabb": {
            "center": aabb_center.tolist(),
            "sizes": aabb_sizes.tolist(),
            "min": aabb_min_scaled.tolist(),
            "max": aabb_max_scaled.tolist(),
        },
        "gravity_aligned_obb": {
            "center": [float(xy_center[0]), float(xy_center[1]), float(aabb_center[2])],
            "sizes": [float(xy_sizes[0]), float(xy_sizes[1]), float(aabb_sizes[2])],
            "yaw_rad": yaw_rad,
            "yaw_deg": float(np.degrees(yaw_rad)),
        },
        "arbitrary_3d_obb": {
            "center": obb_center.tolist(),
            "sizes": obb_extent.tolist(),
            "rotation_matrix": obb_rot.tolist(),
        },
    }


def subdivide_obb_into_sub_rois(parent_bbox_info, pts=None, grid_splits=(4, 2, 2)):
    """
    Chia 1 Bounding Box cha (OBB) thành 16 ô nhỏ đều nhau (4 x 2 x 2 = 16 sub-ROIs).
    Lưu thông tin AABB, Gravity OBB, Arbitrary OBB và số lượng điểm của từng sub-box.
    """
    nx, ny, nz = grid_splits
    obb_info = parent_bbox_info["arbitrary_3d_obb"]
    parent_center = np.array(obb_info["center"], dtype=np.float64)
    parent_sizes = np.array(obb_info["sizes"], dtype=np.float64)
    R = np.array(obb_info["rotation_matrix"], dtype=np.float64)

    sub_sizes = parent_sizes / np.array([nx, ny, nz], dtype=np.float64)

    # Chuyển điểm thực sang tọa độ cục bộ của OBB để đếm số điểm trong từng sub-ROI
    local_pts = None
    if pts is not None and len(pts) > 0:
        local_pts = (pts - parent_center) @ R

    sub_rois_dict = {}

    sub_idx = 1
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                name = f"sub_roi_{sub_idx}"

                # Giới hạn cục bộ của sub-box
                loc_min_x = -parent_sizes[0] / 2.0 + i * sub_sizes[0]
                loc_max_x = loc_min_x + sub_sizes[0]
                loc_min_y = -parent_sizes[1] / 2.0 + j * sub_sizes[1]
                loc_max_y = loc_min_y + sub_sizes[1]
                loc_min_z = -parent_sizes[2] / 2.0 + k * sub_sizes[2]
                loc_max_z = loc_min_z + sub_sizes[2]

                local_center = np.array([
                    (loc_min_x + loc_max_x) / 2.0,
                    (loc_min_y + loc_max_y) / 2.0,
                    (loc_min_z + loc_max_z) / 2.0
                ])

                # Tâm thực tế trong không gian 3D
                sub_center = parent_center + (R @ local_center)

                # 8 đỉnh của sub-box
                dx, dy, dz = sub_sizes / 2.0
                local_corners = np.array([
                    [-dx, -dy, -dz],
                    [ dx, -dy, -dz],
                    [ dx,  dy, -dz],
                    [-dx,  dy, -dz],
                    [-dx, -dy,  dz],
                    [ dx, -dy,  dz],
                    [ dx,  dy,  dz],
                    [-dx,  dy,  dz],
                ])
                global_corners = sub_center + (local_corners @ R.T)

                # AABB của sub-box
                sub_aabb_min = np.min(global_corners, axis=0)
                sub_aabb_max = np.max(global_corners, axis=0)
                sub_aabb_center = (sub_aabb_min + sub_aabb_max) / 2.0
                sub_aabb_sizes = sub_aabb_max - sub_aabb_min

                # Đếm số điểm thực nằm trong sub-box
                num_pts_sub = 0
                if local_pts is not None:
                    in_box = (
                        (local_pts[:, 0] >= loc_min_x) & (local_pts[:, 0] <= loc_max_x) &
                        (local_pts[:, 1] >= loc_min_y) & (local_pts[:, 1] <= loc_max_y) &
                        (local_pts[:, 2] >= loc_min_z) & (local_pts[:, 2] <= loc_max_z)
                    )
                    num_pts_sub = int(np.sum(in_box))

                sub_item = {
                    "instance_id": sub_idx,
                    "name": name,
                    "category": "sub_roi",
                    "grid_index": [int(i), int(j), int(k)],
                    "num_points": num_pts_sub,
                    "aabb": {
                        "center": sub_aabb_center.tolist(),
                        "sizes": sub_aabb_sizes.tolist(),
                        "min": sub_aabb_min.tolist(),
                        "max": sub_aabb_max.tolist(),
                    },
                    "gravity_aligned_obb": {
                        "center": sub_center.tolist(),
                        "sizes": sub_sizes.tolist(),
                        "yaw_rad": float(parent_bbox_info["gravity_aligned_obb"]["yaw_rad"]),
                        "yaw_deg": float(parent_bbox_info["gravity_aligned_obb"]["yaw_deg"]),
                    },
                    "arbitrary_3d_obb": {
                        "center": sub_center.tolist(),
                        "sizes": sub_sizes.tolist(),
                        "rotation_matrix": R.tolist(),
                    }
                }
                sub_rois_dict[name] = sub_item
                sub_idx += 1

    return sub_rois_dict


def process_single_scan(
    scan_id: int,
    image_dir: Union[str, Path],
    stl_dir: Optional[Union[str, Path]],
    output_dir: Union[str, Path],
    threshold: float = 0.85,
    expansion_ratio: float = 0.03,
    grid_splits: tuple = (4, 2, 2)
) -> bool:
    """Xử lý tạo BBox và 16 Sub-ROIs cho một scan DTU cụ thể."""
    scan_name = f"scan{scan_id}"
    image_dir_path = resolve_path(image_dir)
    stl_dir_path = resolve_path(stl_dir) if stl_dir else None
    out_dir_path = resolve_path(output_dir)

    # 1. Xác định cameras_npz và mask_dir linh hoạt
    # Hỗ trợ cả khi image_dir là:
    # - DTU Preprocess/DTU
    # - DTU Preprocess/DTU/scan24
    # - DTU Preprocess/DTU/scan24/mask
    if image_dir_path.name == "mask" and (image_dir_path.parent.name == scan_name or str(scan_id) in image_dir_path.parent.name):
        scan_dir = image_dir_path.parent
        mask_dir = image_dir_path
        cameras_npz = scan_dir / "cameras.npz"
    elif image_dir_path.name == scan_name or str(scan_id) in image_dir_path.name:
        scan_dir = image_dir_path
        mask_dir = scan_dir / "mask"
        cameras_npz = scan_dir / "cameras.npz"
    else:
        scan_dir = image_dir_path / scan_name
        mask_dir = scan_dir / "mask"
        cameras_npz = scan_dir / "cameras.npz"

    # Kiểm tra Dataset Configs nếu cameras_npz chưa tìm thấy
    if not cameras_npz.is_file() and load_dataset_paths:
        try:
            dp = load_dataset_paths("dtu", scan=scan_name)
            if "cameras_npz" in dp and os.path.exists(dp["cameras_npz"]):
                cameras_npz = Path(dp["cameras_npz"])
            if "scan_dir" in dp and not mask_dir.is_dir():
                cand_mask = Path(dp["scan_dir"]) / "mask"
                if cand_mask.is_dir():
                    mask_dir = cand_mask
        except Exception:
            pass

    # 2. Xác định file Point Cloud (Dense STL hoặc SfM points fallback)
    stl_filename = f"stl{scan_id:03d}_total.ply"
    point_cloud_path = None
    points_source_name = "STL dense point cloud"

    # 2.1. Thử trong stl_dir
    if stl_dir_path:
        cand_stl = stl_dir_path / stl_filename
        if cand_stl.is_file():
            point_cloud_path = cand_stl

    # 2.2. Thử qua Dataset Configs (dtu.yaml)
    if point_cloud_path is None and load_dataset_paths:
        try:
            dp = load_dataset_paths("dtu", scan=scan_name)
            if "gt_points_ply" in dp and os.path.exists(dp["gt_points_ply"]):
                point_cloud_path = Path(dp["gt_points_ply"])
        except Exception:
            pass

    # 2.3. Thử các candidate paths phổ biến
    if point_cloud_path is None:
        candidates = [
            REPO_ROOT.parent / "DTU Dataset" / "Points" / "Points" / "stl" / stl_filename,
            REPO_ROOT.parent / "DTU Dataset" / "Points" / "stl" / stl_filename,
            REPO_ROOT / "Data" / "DTU" / "Points" / "stl" / stl_filename,
        ]
        for c in candidates:
            if c.is_file():
                point_cloud_path = c
                break

    # 2.4. Fallback sang SfM points (points.ply) nếu không có STL dense
    if point_cloud_path is None:
        sfm_ply = scan_dir / "points.ply"
        if sfm_ply.is_file():
            point_cloud_path = sfm_ply
            points_source_name = "SfM points fallback"

    # Kiểm tra tồn tại dữ liệu
    if point_cloud_path is None or not point_cloud_path.is_file():
        print(f"[-] Bỏ qua {scan_name}: Không tìm thấy point cloud (cả STL {stl_filename} lẫn SfM points.ply)")
        return False
    if not cameras_npz.is_file():
        print(f"[-] Bỏ qua {scan_name}: Không tìm thấy cameras.npz: {cameras_npz}")
        return False
    if not mask_dir.is_dir():
        print(f"[-] Bỏ qua {scan_name}: Không tìm thấy thư mục mask: {mask_dir}")
        return False

    print(f"\n[+] Đang xử lý: {scan_name.upper()} ({point_cloud_path.name})")
    print(f"    - Nguồn điểm: {points_source_name} ({point_cloud_path})")
    print(f"    - Cameras:    {cameras_npz}")
    print(f"    - Masks:      {mask_dir}")

    # 3. Đọc đám mây điểm
    pcd = o3d.io.read_point_cloud(str(point_cloud_path))
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        print(f"    [!] File {point_cloud_path.name} rỗng.")
        return False
    print(f"    - Điểm nguồn gốc: {len(pts):,} điểm")

    # 4. Nạp camera parameters và mask images
    cameras, masks = load_cameras_and_masks(cameras_npz, mask_dir)
    if len(cameras) == 0:
        print(f"    [!] Không nạp được cặp camera/mask nào cho {scan_name}.")
        return False
    print(f"    - Nạp thành công: {len(cameras)} cameras & masks")

    # 5. Chiếu điểm đa góc nhìn và lọc vật thể
    is_object = project_points_to_masks(pts, cameras, masks, threshold=threshold)
    object_points = pts[is_object]

    # Nếu ngưỡng 0.85 quá cao cho một số scan bị che khuất, fallback nhẹ về 0.70
    if len(object_points) < 1000:
        print(f"    [!] Ngưỡng {threshold} quá gắt ({len(object_points)} điểm), thử với ngưỡng 0.70...")
        is_object = project_points_to_masks(pts, cameras, masks, threshold=0.70)
        object_points = pts[is_object]

    if len(object_points) == 0:
        print(f"    [!] Không trích xuất được điểm vật thể nào cho {scan_name}.")
        return False

    print(f"    - Điểm vật thể xác nhận: {len(object_points):,} ({len(object_points)/len(pts)*100:.1f}%)")

    # 6. Lọc nhẹ outlier
    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(object_points)
    obj_pcd, _ = obj_pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
    filtered_object_pts = np.asarray(obj_pcd.points)

    # 7. Tính toán Bounding Box vật thể chính
    obj_bbox_info = compute_bbox_info(
        filtered_object_pts,
        name=f"scan{scan_id}_object",
        category="foreground_object",
        instance_id=0,
        expansion_ratio=expansion_ratio
    )

    # 8. Chia thành 16 Sub-ROIs bên trong
    sub_rois_dict = subdivide_obb_into_sub_rois(
        obj_bbox_info, pts=filtered_object_pts, grid_splits=grid_splits
    )
    obj_bbox_info["sub_rois"] = sub_rois_dict

    # 9. Xác định thư mục lưu kết quả (hỗ trợ phân cấp theo scan)
    if out_dir_path.name in (scan_name, f"dtu_{scan_name}", f"scan_{scan_id}"):
        target_out_dir = out_dir_path
    else:
        target_out_dir = out_dir_path / scan_name

    target_out_dir.mkdir(parents=True, exist_ok=True)
    out_filename = f"scan{scan_id}_object_base_bboxes.json"
    out_path = target_out_dir / out_filename

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([obj_bbox_info], f, indent=2)

    obb_sizes = [round(float(s), 1) for s in obj_bbox_info['arbitrary_3d_obb']['sizes']]
    print(f"    [✓] Đã xuất thành công: {out_path}")
    print(f"        * Kích thước OBB: {obb_sizes} mm")
    print(f"        * Số Sub-ROIs: {len(sub_rois_dict)} ô (sub_roi_1 -> sub_roi_{len(sub_rois_dict)})")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Tạo Bounding Box và 16 Sub-ROIs cho các scan DTU Dataset bằng relative paths."
    )
    parser.add_argument(
        "--config", type=str, default="configs/object_roi.toml",
        help="Đường dẫn file config TOML (mặc định: configs/object_roi.toml)"
    )
    parser.add_argument(
        "--dataset_config", type=str, default="dtu",
        help="Tên dataset config trong 'Dataset Configs/' (mặc định: dtu)"
    )
    parser.add_argument(
        "--image_dir", type=str, default=str(DEFAULT_DTU_IMAGE_DIR),
        help=f"Thư mục chứa scan ảnh & mask hoặc thư mục scan cụ thể (mặc định: {DEFAULT_DTU_IMAGE_DIR})"
    )
    parser.add_argument(
        "--stl_dir", type=str, default=str(DEFAULT_STL_DIR),
        help=f"Thư mục chứa ground-truth STL point clouds (mặc định: {DEFAULT_STL_DIR})"
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help=f"Thư mục lưu JSON kết quả (mặc định: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--scans", type=str, nargs="*", default=None,
        help="Danh sách scan ID cần chạy (vd: --scans 24 hoặc --scans 24 37 40 hoặc 'all'). Mặc định lấy từ config hoặc quét image_dir."
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_VISIBILITY_THRESHOLD,
        help="Ngưỡng đồng thuận đa góc nhìn (mặc định: 0.85)"
    )
    parser.add_argument(
        "--expansion_ratio", type=float, default=DEFAULT_EXPANSION_RATIO,
        help="Tỉ lệ nới rộng BBox (mặc định: 0.03 tương đương 3%%)"
    )

    args = parser.parse_args()

    # Phân giải đường dẫn tương đối thành Path chuẩn
    resolved_image_dir = resolve_path(args.image_dir)
    resolved_stl_dir = resolve_path(args.stl_dir) if args.stl_dir else None
    resolved_output_dir = resolve_path(args.output_dir)

    # Đọc thêm cấu hình từ TOML nếu có
    cfg_target_scan = None
    if args.config:
        cfg_path = resolve_path(args.config)
        if cfg_path.is_file() and load_toml_config:
            try:
                cfg = load_toml_config(cfg_path)
                ds_cfg = cfg.get("dataset", {})
                if isinstance(ds_cfg, dict) and ds_cfg.get("name") == "dtu":
                    cfg_target_scan = ds_cfg.get("scan") or ds_cfg.get("scene")
            except Exception as e:
                print(f"[!] Warning reading TOML config {cfg_path}: {e}")

    # Xác định danh sách scan cần xử lý
    target_scan_ids = []
    if args.scans is not None and len(args.scans) > 0:
        if len(args.scans) == 1 and args.scans[0].lower() in ("all", "full", "benchmark"):
            # Quét toàn bộ scan trong image_dir
            args_scans_all = True
        else:
            args_scans_all = False
            for s in args.scans:
                sid = extract_scan_number(str(s))
                if sid > 0:
                    target_scan_ids.append(sid)
    else:
        args_scans_all = False
        if cfg_target_scan:
            sid = extract_scan_number(str(cfg_target_scan))
            if sid > 0:
                target_scan_ids.append(sid)

    # Nếu chưa xác định từ --scans hoặc config, kiểm tra xem resolved_image_dir có phải là 1 scan đơn lẻ không
    if not target_scan_ids and not args_scans_all:
        single_sid = extract_scan_number(resolved_image_dir.name)
        if single_sid <= 0 and resolved_image_dir.parent:
            single_sid = extract_scan_number(resolved_image_dir.parent.name)
        if single_sid > 0:
            target_scan_ids = [single_sid]

    # Nếu vẫn chưa có scan nào, quét toàn bộ thư mục image_dir
    if not target_scan_ids:
        if resolved_image_dir.is_dir():
            all_entries = os.listdir(resolved_image_dir)
            for entry in all_entries:
                entry_path = resolved_image_dir / entry
                if entry_path.is_dir() and "scan" in entry.lower():
                    sid = extract_scan_number(entry)
                    if sid > 0:
                        target_scan_ids.append(sid)
            target_scan_ids.sort()

    print("=" * 80)
    print("        TIẾN TRÌNH TẠO BOUNDING BOX & 16 SUB-ROIS CHO CÁC SCAN DTU")
    print("=" * 80)
    print(f"[*] Thư mục Image/Mask nguồn: {resolved_image_dir}")
    print(f"[*] Thư mục STL nguồn:        {resolved_stl_dir}")
    print(f"[*] Thư mục JSON xuất ra:     {resolved_output_dir}")
    print(f"[*] Ngưỡng đồng thuận:       {args.threshold}")
    print(f"[*] Cấu hình chia Sub-ROI:   {DEFAULT_GRID_SPLITS[0]}x{DEFAULT_GRID_SPLITS[1]}x{DEFAULT_GRID_SPLITS[2]} = 16 ô")
    print("-" * 80)
    print(f"[*] Danh sách scan cần xử lý: {target_scan_ids}")

    success_count = 0
    fail_count = 0

    for sid in target_scan_ids:
        ok = process_single_scan(
            scan_id=sid,
            image_dir=resolved_image_dir,
            stl_dir=resolved_stl_dir,
            output_dir=resolved_output_dir,
            threshold=args.threshold,
            expansion_ratio=args.expansion_ratio,
            grid_splits=DEFAULT_GRID_SPLITS
        )
        if ok:
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "=" * 80)
    print(f"[✓] HOÀN TẤT TIẾN TRÌNH SUB-ROIS CHO DTU DATASET!")
    print(f"    - Thành công: {success_count} / {len(target_scan_ids)} scan")
    print(f"    - Bỏ qua / Lỗi: {fail_count} scan")
    print(f"    - Thư mục lưu kết quả: {resolved_output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()