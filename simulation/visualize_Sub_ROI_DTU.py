import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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
REPO_ROOT = Path(__file__).resolve().parents[1]
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
# 1. DEFAULT_SUB_JSON_DIR:
#    Thư mục chứa các file JSON Sub-ROIs từ Module 1.
#    Mặc định: "outputs/DTU/sub_bounding_box_3D" (tương đối theo repo root).
#    Ví dụ scan24: "outputs/DTU/sub_bounding_box_3D/scan24"
DEFAULT_SUB_JSON_DIR = REPO_ROOT / "outputs" / "DTU" / "sub_bounding_box_3D"

# 2. DEFAULT_PARENT_JSON_DIR:
#    Thư mục chứa các file JSON BBox ban đầu / cha (Module 1 bounding_box_3D).
#    Mặc định: "outputs/DTU/bounding_box_3D" (tương đối theo repo root).
#    Ví dụ scan24: "outputs/DTU/bounding_box_3D/dtu_scan24"
DEFAULT_PARENT_JSON_DIR = REPO_ROOT / "outputs" / "DTU" / "bounding_box_3D"

# 3. DEFAULT_STL_DIR:
#    Thư mục chứa dense point cloud ground-truth (stlXXX_total.ply).
#    Mặc định: "../DTU Dataset/Points/Points/stl" (ngang hàng repo Safe-GS).
DEFAULT_STL_DIR = REPO_ROOT.parent / "DTU Dataset" / "Points" / "Points" / "stl"

# 4. DEFAULT_OUTPUT_DIR:
#    Thư mục lưu các file PLY kết quả trực quan hóa.
#    Mặc định: "simulation/result_simulation" (trong repo Safe-GS).
DEFAULT_OUTPUT_DIR = REPO_ROOT / "simulation" / "result_simulation"

# Tùy chọn hiển thị mặc định:
DEFAULT_BBOX_TYPE = "arbitrary_3d_obb"  # 'arbitrary_3d_obb', 'gravity_aligned_obb', 'aabb'
DRAW_PARENT_BBOX = True                 # Vẽ BBox ban đầu từ bounding_box_3D (Khung Trắng)
DRAW_OBJECT_BBOX = True                 # Vẽ BBox vật thể lớn (Khung Đỏ)
DRAW_SUB_ROIS = True                    # Vẽ 16 ô Sub-ROIs nhỏ bên trong (Đa màu)

# Số điểm lấy mẫu trên mỗi cạnh khung (độ đậm nét)
POINTS_PER_EDGE_PARENT = 250  # Nét viền cho BBox ban đầu / cha
POINTS_PER_EDGE_OBJECT = 200  # Nét viền cho BBox vật thể lớn
POINTS_PER_EDGE_SUB = 100     # Nét viền cho từng ô sub-ROI con

# Màu sắc:
PARENT_BBOX_COLOR = [1.0, 1.0, 1.0]  # Trắng (White) cho BBox ban đầu
OBJECT_BBOX_COLOR = [1.0, 0.0, 0.0]  # Đỏ (Red) cho BBox vật thể lớn

# Bảng 16 màu phân biệt cho 16 sub-ROIs bên trong
SUB_ROI_COLORS = [
    [0.0, 1.0, 0.0],   # 1. Xanh lá sáng (Lime)
    [0.0, 0.4, 1.0],   # 2. Xanh dương (Blue)
    [1.0, 0.85, 0.0],  # 3. Vàng (Yellow)
    [1.0, 0.0, 1.0],   # 4. Hồng tím (Magenta)
    [0.0, 1.0, 1.0],   # 5. Xanh ngọc (Cyan)
    [1.0, 0.5, 0.0],   # 6. Cam (Orange)
    [0.6, 0.0, 1.0],   # 7. Tím đậm (Purple)
    [0.0, 0.8, 0.4],   # 8. Xanh mạ (Spring Green)
    [1.0, 0.2, 0.6],   # 9. Hồng dâu (Deep Pink)
    [0.2, 0.6, 1.0],   # 10. Xanh da trời (Sky Blue)
    [0.8, 1.0, 0.0],   # 11. Xanh nõn chuối (Chartreuse)
    [1.0, 0.4, 0.4],   # 12. Đỏ san hô (Coral)
    [0.4, 1.0, 0.8],   # 13. Xanh bạc hà (Aquamarine)
    [0.9, 0.6, 0.2],   # 14. Nâu cam (Amber)
    [0.5, 0.5, 1.0],   # 15. Tím oải hương (Lavender)
    [0.0, 0.7, 0.7],   # 16. Xanh mòng két (Teal)
]


def resolve_path(p: Union[str, Path], base_dir: Path = REPO_ROOT) -> Path:
    """Chuyển đổi đường dẫn tương đối thành Path tuyệt đối chuẩn hóa dựa trên base_dir."""
    path_obj = Path(p)
    if path_obj.is_absolute():
        return path_obj
    return (base_dir / path_obj).resolve()


def extract_scan_number(text: str) -> int:
    """Trích xuất ID scan từ chuỗi (ví dụ: scan24_object_base_bboxes.json -> 24)."""
    match = re.search(r"scan_?(\d+)", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    digits = re.search(r"\d+", text)
    return int(digits.group(0)) if digits else -1


def find_sub_json_file(sub_json_dir: Path, scan_id: int) -> Optional[Path]:
    """Tìm file JSON sub-ROI cho một scan cụ thể trong thư mục sub_json_dir."""
    scan_name = f"scan{scan_id}"
    candidates = [
        sub_json_dir / scan_name / f"{scan_name}_object_base_bboxes.json",
        sub_json_dir / f"dtu_{scan_name}" / f"{scan_name}_object_base_bboxes.json",
        sub_json_dir / f"{scan_name}_object_base_bboxes.json",
        sub_json_dir / f"{scan_name}_sub_rois.json",
    ]
    for c in candidates:
        if c.is_file():
            return c

    # Nếu sub_json_dir trỏ thẳng vào thư mục scan cụ thể (vd: .../sub_bounding_box_3D/scan24)
    if sub_json_dir.is_dir():
        for f in sub_json_dir.glob("*.json"):
            if str(scan_id) in f.name:
                return f
        # Quét đệ quy
        for f in sub_json_dir.glob(f"**/*{scan_name}*.json"):
            if f.is_file():
                return f
    return None


def find_parent_json_file(parent_json_dir: Path, scan_id: int) -> Optional[Path]:
    """Tìm file JSON BBox ban đầu (cha) cho một scan cụ thể."""
    scan_name = f"scan{scan_id}"
    candidates = [
        parent_json_dir / f"dtu_{scan_name}" / f"{scan_name}_3d_boxes.json",
        parent_json_dir / scan_name / f"{scan_name}_3d_boxes.json",
        parent_json_dir / f"{scan_name}_3d_boxes.json",
        parent_json_dir / f"{scan_name}_bbox.json",
        parent_json_dir / f"dtu_{scan_name}" / f"{scan_name}_bbox.json",
    ]
    for c in candidates:
        if c.is_file():
            return c

    # Nếu parent_json_dir trỏ thẳng vào folder của scan (vd: outputs/DTU/bounding_box_3D/dtu_scan24)
    if parent_json_dir.is_dir():
        for f in parent_json_dir.glob("*.json"):
            if str(scan_id) in f.name:
                return f
        for f in parent_json_dir.glob(f"**/*{scan_name}*.json"):
            if f.is_file():
                return f
    return None


def find_stl_file(stl_dir: Optional[Path], scan_id: int) -> Tuple[Optional[Path], str]:
    """Tìm file Point Cloud gốc (ưu tiên Dense STL ground truth, fallback sang SfM points)."""
    scan_name = f"scan{scan_id}"
    stl_filename = f"stl{scan_id:03d}_total.ply"

    # 1. Thử trong stl_dir
    if stl_dir:
        cand = stl_dir / stl_filename
        if cand.is_file():
            return cand, "STL dense point cloud"

    # 2. Thử qua Dataset Configs (dtu.yaml)
    if load_dataset_paths:
        try:
            dp = load_dataset_paths("dtu", scan=scan_name)
            if "gt_points_ply" in dp and os.path.exists(dp["gt_points_ply"]):
                return Path(dp["gt_points_ply"]), "STL dense point cloud (dataset_config)"
        except Exception:
            pass

    # 3. Thử các candidate relative paths phổ biến
    candidates = [
        REPO_ROOT.parent / "DTU Dataset" / "Points" / "Points" / "stl" / stl_filename,
        REPO_ROOT.parent / "DTU Dataset" / "Points" / "stl" / stl_filename,
        REPO_ROOT / "Data" / "DTU" / "Points" / "stl" / stl_filename,
    ]
    for c in candidates:
        if c.is_file():
            return c, "STL dense point cloud (relative candidate)"

    # 4. Fallback sang SfM points nếu không có STL
    sfm_cand = REPO_ROOT.parent / "DTU Preprocess" / "DTU" / scan_name / "points.ply"
    if sfm_cand.is_file():
        return sfm_cand, "SfM points fallback"

    return None, "Not found"


def sample_line_points(pt1, pt2, num_samples=100):
    """Lấy mẫu các điểm phân bố đều trên đoạn thẳng nối 2 đỉnh pt1 và pt2."""
    alphas = np.linspace(0.0, 1.0, num_samples)[:, None]
    return (1.0 - alphas) * pt1 + alphas * pt2


def get_obb_corners(center, sizes, R):
    """Tính tọa độ 8 đỉnh trong không gian thế giới từ center, sizes và ma trận xoay R."""
    dx, dy, dz = sizes[0] / 2.0, sizes[1] / 2.0, sizes[2] / 2.0
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
    return center + (local_corners @ R.T)


def compute_corners(item, bbox_type="arbitrary_3d_obb"):
    """Trích xuất 8 đỉnh của hộp chữ nhật theo loại BBox từ dict JSON."""
    if bbox_type == "aabb" or (bbox_type not in item and "aabb" in item):
        aabb = item["aabb"]
        return get_obb_corners(np.array(aabb["center"]), np.array(aabb["sizes"]), np.eye(3))

    elif bbox_type == "gravity_aligned_obb" and "gravity_aligned_obb" in item:
        g_obb = item["gravity_aligned_obb"]
        center = np.array(g_obb["center"])
        sizes = np.array(g_obb["sizes"])
        yaw = g_obb.get("yaw_rad", 0.0)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        R = np.array([
            [cos_y, -sin_y, 0.0],
            [sin_y,  cos_y, 0.0],
            [  0.0,    0.0, 1.0]
        ])
        return get_obb_corners(center, sizes, R)

    elif bbox_type == "arbitrary_3d_obb" and "arbitrary_3d_obb" in item:
        obb = item["arbitrary_3d_obb"]
        return get_obb_corners(
            np.array(obb["center"]),
            np.array(obb["sizes"]),
            np.array(obb["rotation_matrix"])
        )
    return None


def create_wireframe_points(corners, color, points_per_edge=100):
    """Nối 12 cạnh của 8 đỉnh để tạo khung hộp 3D dạng chuỗi điểm đặc."""
    edges = [
        # Mặt đáy (Z-)
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Mặt nắp (Z+)
        (4, 5), (5, 6), (6, 7), (7, 4),
        # 4 trụ đứng
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    wireframe_pts = []
    for i, j in edges:
        wireframe_pts.append(sample_line_points(corners[i], corners[j], points_per_edge))
    wireframe_pts = np.vstack(wireframe_pts)
    wireframe_colors = np.tile(color, (len(wireframe_pts), 1))
    return wireframe_pts, wireframe_colors


def extract_target_item(json_path: Union[str, Path]):
    """Đọc file JSON và lấy item bounding box đại diện."""
    json_path = Path(json_path)
    if not json_path.is_file():
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        return None
    for item in data:
        if item.get("category") == "foreground_object" or "object" in item.get("name", "").lower():
            return item
    return data[0]


def process_single_scan_visualization(
    scan_id: int,
    sub_json_path: Union[str, Path],
    stl_dir: Optional[Union[str, Path]],
    parent_json_dir: Optional[Union[str, Path]],
    output_dir: Union[str, Path],
    bbox_type: str = "arbitrary_3d_obb",
    draw_parent: bool = True,
    draw_object: bool = True,
    draw_sub_rois: bool = True
) -> bool:
    """Dựng khung viền 3D cho 1 scan DTU và xuất file PLY duy nhất vào output_dir."""
    sub_json_path = Path(sub_json_path)
    stl_dir_path = resolve_path(stl_dir) if stl_dir else None
    parent_json_dir_path = resolve_path(parent_json_dir) if parent_json_dir else None
    out_dir_path = resolve_path(output_dir)

    # 1. Xác định file point cloud
    point_cloud_path, point_source = find_stl_file(stl_dir_path, scan_id)
    if point_cloud_path is None or not point_cloud_path.is_file():
        print(f"[-] Bỏ qua scan {scan_id}: Không tìm thấy point cloud STL stl{scan_id:03d}_total.ply hoặc SfM points")
        return False

    # 2. Kiểm tra file JSON Sub-ROIs
    if not sub_json_path.is_file():
        cand = find_sub_json_file(sub_json_path.parent if sub_json_path.suffix else sub_json_path, scan_id)
        if cand and cand.is_file():
            sub_json_path = cand
        else:
            print(f"[-] Bỏ qua scan {scan_id}: Không tìm thấy file JSON Sub-ROI: {sub_json_path}")
            return False

    out_filename = f"scan{scan_id}_with_sub_rois.ply"
    out_path = out_dir_path / out_filename

    print(f"\n[+] Đang dựng 3D cho: SCAN {scan_id} ({point_cloud_path.name})")
    print(f"    - Nguồn điểm:  {point_source} ({point_cloud_path})")
    print(f"    - Sub-ROI JSON: {sub_json_path}")

    # 3. Đọc file Point Cloud gốc
    pcd = o3d.io.read_point_cloud(str(point_cloud_path))
    dense_pts = np.asarray(pcd.points)
    if len(dense_pts) == 0:
        print(f"    [!] File {point_cloud_path.name} rỗng.")
        return False

    if pcd.has_colors():
        dense_colors = np.asarray(pcd.colors)
    else:
        dense_colors = np.full((len(dense_pts), 3), 0.65)

    all_bbox_pts = []
    all_bbox_colors = []

    # 4. Đọc và dựng Bounding Box Ban Đầu / Cha (Màu Trắng) nếu được bật
    parent_loaded = False
    if draw_parent and parent_json_dir_path:
        parent_json_path = find_parent_json_file(parent_json_dir_path, scan_id)
        if parent_json_path and parent_json_path.is_file():
            parent_item = extract_target_item(parent_json_path)
            if parent_item is not None:
                corners_parent = compute_corners(parent_item, bbox_type=bbox_type)
                if corners_parent is not None:
                    pts_parent, clrs_parent = create_wireframe_points(
                        corners_parent, color=PARENT_BBOX_COLOR, points_per_edge=POINTS_PER_EDGE_PARENT
                    )
                    all_bbox_pts.append(pts_parent)
                    all_bbox_colors.append(clrs_parent)
                    parent_loaded = True
                    print(f"    - BBox Cha:    {parent_json_path.name} (Khung Trắng)")

    if not parent_loaded and draw_parent:
        print(f"    [*] Thông báo: Không tìm thấy file JSON BBox cha (bỏ qua khung trắng)")

    # 5. Đọc và dựng Bounding Box Vật Thể (Đỏ) + 16 Sub-ROIs (Đa màu)
    object_item = extract_target_item(sub_json_path)
    num_sub_added = 0
    if object_item is not None:
        if draw_object:
            corners_object = compute_corners(object_item, bbox_type=bbox_type)
            if corners_object is not None:
                pts_object, clrs_object = create_wireframe_points(
                    corners_object, color=OBJECT_BBOX_COLOR, points_per_edge=POINTS_PER_EDGE_OBJECT
                )
                all_bbox_pts.append(pts_object)
                all_bbox_colors.append(clrs_object)

        if draw_sub_rois and "sub_rois" in object_item:
            sub_rois_dict = object_item["sub_rois"]
            for idx, (sub_name, sub_info) in enumerate(sub_rois_dict.items()):
                corners_sub = compute_corners(sub_info, bbox_type=bbox_type)
                if corners_sub is not None:
                    color = SUB_ROI_COLORS[idx % len(SUB_ROI_COLORS)]
                    pts_sub, clrs_sub = create_wireframe_points(
                        corners_sub, color=color, points_per_edge=POINTS_PER_EDGE_SUB
                    )
                    all_bbox_pts.append(pts_sub)
                    all_bbox_colors.append(clrs_sub)
                    num_sub_added += 1

    # 6. Gộp điểm dense point cloud và các khung viền box
    if all_bbox_pts:
        all_bbox_pts = np.vstack(all_bbox_pts)
        all_bbox_colors = np.vstack(all_bbox_colors)
        merged_pts = np.vstack([dense_pts, all_bbox_pts])
        merged_colors = np.vstack([dense_colors, all_bbox_colors])
    else:
        merged_pts = dense_pts
        merged_colors = dense_colors

    # 7. Lưu ra file PLY duy nhất vào output_dir
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(merged_pts)
    combined_pcd.colors = o3d.utility.Vector3dVector(merged_colors)

    out_dir_path.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_path), combined_pcd)

    print(f"    [✓] Đã xuất thành công: {out_path}")
    print(f"        * Tổng số điểm: {len(merged_pts):,} điểm")
    print(f"        * Đã bao gồm: 1 BBox vật thể + {num_sub_added} ô Sub-ROIs đa màu")
    return True


def discover_all_sub_json_files(sub_json_dir: Path) -> List[Tuple[int, Path]]:
    """Tự động quét tất cả file JSON sub-ROI trong thư mục hoặc các thư mục con."""
    found = []
    # 1. Các file json nằm ngay trong sub_json_dir
    for f in sub_json_dir.glob("*.json"):
        sid = extract_scan_number(f.name)
        if sid > 0:
            found.append((sid, f))
    # 2. Các file json nằm trong subfolder (vd: scan24/scan24_object_base_bboxes.json)
    for f in sub_json_dir.glob("*/*.json"):
        sid = extract_scan_number(f.name)
        if sid <= 0 and f.parent:
            sid = extract_scan_number(f.parent.name)
        if sid > 0:
            found.append((sid, f))

    # Loại bỏ trùng lặp và sắp xếp theo ID scan
    unique = {}
    for sid, path in found:
        if sid not in unique:
            unique[sid] = path
    return sorted(unique.items(), key=lambda x: x[0])


def main():
    parser = argparse.ArgumentParser(
        description="Trực quan hóa hàng loạt Bounding Box và 16 Sub-ROIs trên Dense Point Cloud DTU bằng relative paths."
    )
    parser.add_argument(
        "--config", type=str, default="configs/simulation.toml",
        help="Đường dẫn file config TOML (mặc định: configs/simulation.toml)"
    )
    parser.add_argument(
        "--dataset_config", type=str, default="dtu",
        help="Tên dataset config trong 'Dataset Configs/' (mặc định: dtu)"
    )
    parser.add_argument(
        "--sub_json_dir", type=str, default=str(DEFAULT_SUB_JSON_DIR),
        help=f"Thư mục chứa các file JSON sub-ROI (mặc định: {DEFAULT_SUB_JSON_DIR})"
    )
    parser.add_argument(
        "--parent_json_dir", type=str, default=str(DEFAULT_PARENT_JSON_DIR),
        help=f"Thư mục chứa các file JSON BBox cha ban đầu (mặc định: {DEFAULT_PARENT_JSON_DIR})"
    )
    parser.add_argument(
        "--stl_dir", type=str, default=str(DEFAULT_STL_DIR),
        help=f"Thư mục chứa các file dense point cloud stlXXX_total.ply (mặc định: {DEFAULT_STL_DIR})"
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help=f"Thư mục lưu các file PLY kết quả (mặc định: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--scans", type=str, nargs="*", default=None,
        help="Danh sách scan ID cần vẽ (vd: --scans 24 hoặc --scans 24 37 40 hoặc 'all'). Mặc định quét tự động."
    )
    parser.add_argument(
        "--box_type", type=str, choices=["arbitrary_3d_obb", "gravity_aligned_obb", "aabb"],
        default=DEFAULT_BBOX_TYPE,
        help="Loại Bounding Box cần vẽ (mặc định: arbitrary_3d_obb)"
    )
    parser.add_argument(
        "--no_parent", action="store_true",
        help="Không vẽ BBox ban đầu màu trắng"
    )
    parser.add_argument(
        "--no_sub_rois", action="store_true",
        help="Không vẽ 16 Sub-ROIs bên trong"
    )

    args = parser.parse_args()

    # Phân giải đường dẫn tương đối
    resolved_sub_json_dir = resolve_path(args.sub_json_dir)
    resolved_parent_json_dir = resolve_path(args.parent_json_dir) if args.parent_json_dir else None
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
    scan_tasks: List[Tuple[int, Path]] = []

    # 1. Nếu người dùng chỉ định scan cụ thể qua --scans
    if args.scans is not None and len(args.scans) > 0:
        if len(args.scans) == 1 and args.scans[0].lower() in ("all", "full", "benchmark"):
            scan_tasks = discover_all_sub_json_files(resolved_sub_json_dir)
        else:
            for s in args.scans:
                sid = extract_scan_number(str(s))
                if sid > 0:
                    sub_file = find_sub_json_file(resolved_sub_json_dir, sid)
                    if sub_file:
                        scan_tasks.append((sid, sub_file))
                    else:
                        print(f"[-] Cảnh báo: Không tìm thấy file JSON Sub-ROI cho scan {sid} trong {resolved_sub_json_dir}")

    # 2. Nếu lấy từ file config TOML
    elif cfg_target_scan:
        sid = extract_scan_number(str(cfg_target_scan))
        if sid > 0:
            sub_file = find_sub_json_file(resolved_sub_json_dir, sid)
            if sub_file:
                scan_tasks.append((sid, sub_file))

    # 3. Nếu resolved_sub_json_dir trỏ thẳng vào 1 scan cụ thể (vd: outputs/.../scan24)
    elif extract_scan_number(resolved_sub_json_dir.name) > 0:
        sid = extract_scan_number(resolved_sub_json_dir.name)
        sub_file = find_sub_json_file(resolved_sub_json_dir, sid)
        if sub_file:
            scan_tasks.append((sid, sub_file))

    # 4. Quét tự động tất cả các file JSON tìm thấy
    if not scan_tasks:
        if resolved_sub_json_dir.is_dir():
            scan_tasks = discover_all_sub_json_files(resolved_sub_json_dir)

    print("=" * 80)
    print("      TIẾN TRÌNH TRỰC QUAN HÓA BOUNDING BOX & SUB-ROIS CHO DTU POINT CLOUD")
    print("=" * 80)
    print(f"[*] Thư mục Sub-ROI JSON nguồn: {resolved_sub_json_dir}")
    print(f"[*] Thư mục Parent JSON nguồn:  {resolved_parent_json_dir}")
    print(f"[*] Thư mục STL nguồn:           {resolved_stl_dir}")
    print(f"[*] Thư mục PLY xuất ra:         {resolved_output_dir}")
    print(f"[*] Loại BBox hiển thị:          {args.box_type}")
    print(f"[*] Vẽ BBox cha (Trắng):         {'TẮT' if args.no_parent else 'BẬT'}")
    print(f"[*] Vẽ 16 Sub-ROIs (Đa màu):     {'TẮT' if args.no_sub_rois else 'BẬT'}")
    print("-" * 80)
    print(f"[*] Tìm thấy {len(scan_tasks)} scan cần xử lý: {[s[0] for s in scan_tasks]}")

    success_count = 0
    fail_count = 0

    for sid, sub_json_path in scan_tasks:
        ok = process_single_scan_visualization(
            scan_id=sid,
            sub_json_path=sub_json_path,
            stl_dir=resolved_stl_dir,
            parent_json_dir=resolved_parent_json_dir,
            output_dir=resolved_output_dir,
            bbox_type=args.box_type,
            draw_parent=(not args.no_parent),
            draw_object=True,
            draw_sub_rois=(not args.no_sub_rois)
        )
        if ok:
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "=" * 80)
    print(f"[✓] HOÀN TẤT TIẾN TRÌNH TRỰC QUAN HÓA DTU POINT CLOUD!")
    print(f"    - Thành công: {success_count} / {len(scan_tasks)} scan")
    print(f"    - Bỏ qua / Lỗi: {fail_count} scan")
    print(f"    - Toàn bộ file PLY kết quả được lưu tại: {resolved_output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()