"""Module Simulation: Visualize 3D ROI Bounding Boxes on DTU Point Cloud.

Trực quan hóa các 3D Bounding Boxes (Arbitrary 3D OBB, Gravity-aligned OBB, AABB)
được xuất từ Module 1 Perception chồng lấn lên Dense Point Cloud (STL) của DTU Dataset.

Đường dẫn input và output được quản lý tự động thông qua relative paths kết hợp giữa:
    1. Safe-GS configs (configs/simulation.toml hoặc configs/object_roi.toml)
    2. Dataset Configs (Dataset Configs/dtu.yaml qua src.common.dataset_config)
    3. Input Bboxes:  outputs/DTU/bounding_box_3D (tương đối theo repo root)
    4. Input STL:     DTU Dataset/Points/Points/stl (tự động qua dataset_config)
    5. Mặc định output: simulation/result_simulation (lưu trực tiếp chung thư mục)

Cách chạy:
    1. Trực quan hóa 1 scan cụ thể (ví dụ: scan24 hoặc 24):
    
        python simulation/visualize_origin_ROI_DTU.py --scan scan24
    
    2. Trực quan hóa toàn bộ 15 benchmark scans của DTU Dataset:
    
        python simulation/visualize_origin_ROI_DTU.py --scan all
    
    3. Tùy chỉnh tham số hoặc đường dẫn:

        python simulation/visualize_origin_ROI_DTU.py \
            --json-dir outputs/DTU/bounding_box_3D \
            --output-dir simulation/result_simulation \
            --box-type arbitrary_3d_obb \
            --samples-per-edge 300
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import open3d as o3d

# Đảm bảo hiển thị UTF-8 không bị lỗi font trên Windows Console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Thêm repo root vào sys.path để import an toàn dù script chạy từ bất kỳ thư mục nào
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.common.config_loader import load_toml_config
    from src.common.dataset_config import load_dataset_paths
except ImportError:
    load_toml_config = None
    load_dataset_paths = None

# Danh sách 15 scan chuẩn benchmark DTU
DEFAULT_BENCHMARK_SCANS = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]


def parse_scan_id(scan_str: Union[str, int]) -> int:
    """Trích xuất ID scan dạng số: 'scan24' -> 24, '24' -> 24, 'stl024' -> 24."""
    if isinstance(scan_str, int):
        return scan_str
    match = re.search(r"(\d+)", str(scan_str))
    if match:
        return int(match.group(1))
    raise ValueError(f"Không thể phân tích scan ID từ: '{scan_str}'")


def denormalize_box(
    box_center: np.ndarray,
    box_sizes: np.ndarray,
    pcd_points: np.ndarray,
    norm_mode: str = "auto",
) -> Tuple[np.ndarray, np.ndarray]:
    """Đưa tâm (center) và kích thước (sizes) của bounding box từ không gian chuẩn hóa
    về lại đúng thang đo (scale) và vị trí của Đám mây điểm thực (PCD).
    """
    pcd_min = pcd_points.min(axis=0)
    pcd_max = pcd_points.max(axis=0)
    pcd_center = (pcd_min + pcd_max) / 2.0
    pcd_extent = pcd_max - pcd_min  # [dx, dy, dz] của PCD thực

    # Nếu bounding box đã ở thang đo lớn (> 50 đơn vị), không cần denormalize
    if np.max(box_sizes) > 50.0:
        return box_center, box_sizes

    if norm_mode == "auto":
        scale_ratio = np.max(pcd_extent) / np.max(box_sizes)
        real_sizes = box_sizes * scale_ratio
        real_center = pcd_center + (box_center * scale_ratio)
        return real_center, real_sizes

    elif norm_mode == "scale_only":
        scale_ratio = np.mean(pcd_extent / box_sizes)
        return box_center * scale_ratio, box_sizes * scale_ratio

    return box_center, box_sizes


def create_bbox_wireframe_points(
    center: list,
    sizes: list,
    rot_mat: Optional[list] = None,
    num_samples_per_edge: int = 250,
    color: list = [255, 0, 0],
) -> Tuple[np.ndarray, np.ndarray]:
    """Tạo các điểm dày đặc chạy dọc theo 12 cạnh của 3D Bounding Box."""
    dx, dy, dz = [float(s) / 2.0 for s in sizes]

    corners = np.array([
        [-dx, -dy, -dz],
        [ dx, -dy, -dz],
        [ dx,  dy, -dz],
        [-dx,  dy, -dz],
        [-dx, -dy,  dz],
        [ dx, -dy,  dz],
        [ dx,  dy,  dz],
        [-dx,  dy,  dz],
    ], dtype=np.float64)

    if rot_mat is not None:
        corners = corners @ np.array(rot_mat, dtype=np.float64).T

    corners += np.array(center, dtype=np.float64)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Mặt đáy
        (4, 5), (5, 6), (6, 7), (7, 4),  # Mặt trên
        (0, 4), (1, 5), (2, 6), (3, 7),  # 4 cạnh bên
    ]

    edge_points = []
    t = np.linspace(0, 1, num_samples_per_edge)[:, np.newaxis]
    for start_idx, end_idx in edges:
        p_start = corners[start_idx]
        p_end = corners[end_idx]
        pts = p_start + t * (p_end - p_start)
        edge_points.append(pts)

    edge_points = np.vstack(edge_points)
    edge_colors = np.tile(np.array(color, dtype=np.uint8), (edge_points.shape[0], 1))

    return edge_points, edge_colors


def resolve_dtu_json_file(json_dir: Path, scan_id: int) -> Optional[Path]:
    """Tìm file JSON bounding box của scan_id trong json_dir."""
    candidates = [
        json_dir / f"dtu_scan{scan_id}" / f"scan{scan_id}_3d_boxes.json",
        json_dir / f"dtu_scan{scan_id}" / f"scan{scan_id}_bbox.json",
        json_dir / f"dtu_scan{scan_id}" / "roi_bounding_box.json",
        json_dir / f"scan{scan_id}_3d_boxes.json",
        json_dir / f"scan{scan_id}_bbox.json",
        json_dir / f"scan_{scan_id}_3d_boxes.json",
        json_dir / f"scan_{scan_id}_bbox.json",
    ]

    for c in candidates:
        if c.is_file():
            return c
    return None


def resolve_dtu_stl_file(stl_dir: Optional[Path], scan_id: int) -> Optional[Path]:
    """Tìm file STL point cloud gốc của scan_id."""
    filename = f"stl{scan_id:03d}_total.ply"

    if stl_dir and (stl_dir / filename).is_file():
        return stl_dir / filename

    if load_dataset_paths:
        try:
            p = load_dataset_paths("dtu", scan=f"scan{scan_id}")
            if "gt_points_ply" in p and Path(p["gt_points_ply"]).is_file():
                return Path(p["gt_points_ply"])
        except Exception:
            pass

    candidates = [
        REPO_ROOT.parent / "DTU Dataset" / "Points" / "Points" / "stl" / filename,
        REPO_ROOT.parent / "DTU Dataset" / "Points" / "stl" / filename,
        REPO_ROOT / "Data" / "DTU" / "Points" / "stl" / filename,
    ]
    for c in candidates:
        if c.is_file():
            return c

    return None


def visualize_single_dtu_scan(
    json_path: Path,
    stl_path: Path,
    output_ply_path: Path,
    scan_id: int,
    box_type: str = "arbitrary_3d_obb",
    samples_per_edge: int = 300,
    box_color: List[int] = [255, 0, 0],
    unnormalize: bool = True,
) -> bool:
    """Tạo 1 file PLY kết hợp Point Cloud và 3D Bounding Box."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            boxes_data = json.load(f)

        if not boxes_data:
            print(f"    [!] File {json_path.name} rỗng, bỏ qua.")
            return False

        pcd_orig = o3d.io.read_point_cloud(str(stl_path))
        orig_points = np.asarray(pcd_orig.points)

        if len(orig_points) == 0:
            print(f"    [!] File {stl_path.name} không có điểm nào.")
            return False

        if len(pcd_orig.colors) == 0:
            orig_colors = np.full((orig_points.shape[0], 3), 190, dtype=np.uint8)
        else:
            orig_colors = (np.asarray(pcd_orig.colors) * 255).astype(np.uint8)

        all_box_points = []
        all_box_colors = []

        for box in boxes_data:
            target_box = box.get(box_type) or box.get("arbitrary_3d_obb") or box.get("gravity_aligned_obb")

            if target_box is not None:
                center = np.array(target_box["center"], dtype=np.float64)
                sizes = np.array(target_box["sizes"], dtype=np.float64)
                rot_mat = target_box.get("rotation_matrix", np.eye(3).tolist())
            elif "aabb" in box:
                aabb_info = box["aabb"]
                if "center" in aabb_info and "sizes" in aabb_info:
                    center = np.array(aabb_info["center"], dtype=np.float64)
                    sizes = np.array(aabb_info["sizes"], dtype=np.float64)
                else:
                    b_min = np.array(aabb_info["min"], dtype=np.float64)
                    max_b = np.array(aabb_info["max"], dtype=np.float64)
                    center = (b_min + max_b) / 2.0
                    sizes = max_b - b_min
                rot_mat = np.eye(3).tolist()
            else:
                continue

            if unnormalize:
                center, sizes = denormalize_box(
                    box_center=center,
                    box_sizes=sizes,
                    pcd_points=orig_points,
                    norm_mode="auto",
                )

            pts, cols = create_bbox_wireframe_points(
                center=center.tolist(),
                sizes=sizes.tolist(),
                rot_mat=rot_mat,
                num_samples_per_edge=samples_per_edge,
                color=box_color,
            )
            all_box_points.append(pts)
            all_box_colors.append(cols)

        if not all_box_points:
            print(f"    [!] Không trích xuất được thông số box nào từ {json_path.name}.")
            return False

        combined_points = np.vstack([orig_points] + all_box_points)
        combined_colors = np.vstack([orig_colors] + all_box_colors).astype(np.float64) / 255.0

        out_pcd = o3d.geometry.PointCloud()
        out_pcd.points = o3d.utility.Vector3dVector(combined_points)
        out_pcd.colors = o3d.utility.Vector3dVector(combined_colors)

        output_ply_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(output_ply_path), out_pcd)

        print(f"    [✓] Đã xuất thành công: {output_ply_path.name} ({len(combined_points):,} điểm)")
        return True

    except Exception as e:
        print(f"    [!] Lỗi khi xử lý scan {scan_id}: {e}")
        return False


def run_dtu_batch_visualization(
    json_dir: Path,
    stl_dir: Optional[Path],
    output_dir: Path,
    target_scan: Optional[Union[str, int]] = None,
    box_type: str = "arbitrary_3d_obb",
    samples_per_edge: int = 300,
    unnormalize: bool = True,
) -> None:
    """Thực thi visualization hàng loạt cho các scan DTU."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if target_scan is not None and str(target_scan).lower() not in ("all", "benchmark", "default"):
        if str(target_scan).lower() == "full":
            if stl_dir and stl_dir.exists():
                found = []
                for f in stl_dir.glob("stl*_total.ply"):
                    try:
                        found.append(parse_scan_id(f.name))
                    except ValueError:
                        pass
                scans_to_process = sorted(list(set(found))) if found else DEFAULT_BENCHMARK_SCANS
            else:
                scans_to_process = DEFAULT_BENCHMARK_SCANS
        elif "," in str(target_scan):
            scans_to_process = [parse_scan_id(s) for s in str(target_scan).split(",")]
        else:
            scans_to_process = [parse_scan_id(target_scan)]
    else:
        scans_to_process = DEFAULT_BENCHMARK_SCANS

    print(f"[*] Input JSON Directory: {json_dir}")
    print(f"[*] Input STL Directory:  {stl_dir if stl_dir else '(Auto-resolve via Dataset Configs)'}")
    print(f"[*] Output Directory:     {output_dir}")
    print(f"[*] Box Type:             {box_type}")
    print(f"[*] Un-normalize:         {'BẬT' if unnormalize else 'TẮT'}")
    print(f"[*] Target Scan(s):       {scans_to_process}")
    print("=" * 75)

    success_count = 0
    skipped_count = 0

    for idx, scan_id in enumerate(scans_to_process, 1):
        print(f"[{idx}/{len(scans_to_process)}] Visualizing Scan {scan_id}")

        json_file = resolve_dtu_json_file(json_dir, scan_id)
        if json_file is None:
            print(f"    [-] Bỏ qua: Không tìm thấy file JSON cho Scan {scan_id} trong {json_dir}")
            skipped_count += 1
            print("-" * 75)
            continue

        stl_file = resolve_dtu_stl_file(stl_dir, scan_id)
        if stl_file is None:
            print(f"    [-] Bỏ qua: Không tìm thấy file STL point cloud cho Scan {scan_id}")
            skipped_count += 1
            print("-" * 75)
            continue

        print(f"    * Đọc JSON: {json_file.name}  <--->  STL: {stl_file.name}")

        out_ply_file = output_dir / f"scan{scan_id}_with_bbox.ply"

        ok = visualize_single_dtu_scan(
            json_path=json_file,
            stl_path=stl_file,
            output_ply_path=out_ply_file,
            scan_id=scan_id,
            box_type=box_type,
            samples_per_edge=samples_per_edge,
            box_color=[255, 0, 0],
            unnormalize=unnormalize,
        )

        if ok:
            success_count += 1
        else:
            skipped_count += 1

        print("-" * 75)

    print(f"\n[✓] Hoàn tất quá trình visualize DTU!")
    print(f"    - Thành công: {success_count} scan")
    print(f"    - Bỏ qua / Thất bại: {skipped_count} scan")


def resolve_dtu_viz_paths(
    config_file: Optional[Union[str, Path]] = None,
    custom_json_dir: Optional[Union[str, Path]] = None,
    custom_stl_dir: Optional[Union[str, Path]] = None,
    custom_output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Optional[Path], Path]:
    """Tự động resolve đường dẫn tương đối từ configs cho DTU visualization."""
    # 1. Output Dir (Mặc định: simulation/result_simulation)
    default_output_dir = REPO_ROOT / "simulation" / "result_simulation"
    resolved_output_dir = default_output_dir

    # 2. JSON Dir (Mặc định: outputs/DTU/bounding_box_3D)
    default_json_dir = REPO_ROOT / "outputs" / "DTU" / "bounding_box_3D"
    resolved_json_dir = default_json_dir

    # 3. STL Dir (Mặc định auto resolve qua Dataset Configs)
    resolved_stl_dir = None
    if load_dataset_paths:
        try:
            p = load_dataset_paths("dtu", scan="scan24")
            if "gt_points_ply" in p:
                cand = Path(p["gt_points_ply"]).parent
                if cand.exists():
                    resolved_stl_dir = cand
        except Exception:
            pass

    if resolved_stl_dir is None:
        cands = [
            REPO_ROOT.parent / "DTU Dataset" / "Points" / "Points" / "stl",
            REPO_ROOT.parent / "DTU Dataset" / "Points" / "stl",
            REPO_ROOT / "Data" / "DTU" / "Points" / "stl",
        ]
        for c in cands:
            if c.exists():
                resolved_stl_dir = c
                break

    if config_file:
        cfg_path = REPO_ROOT / config_file if not Path(config_file).is_absolute() else Path(config_file)
        if cfg_path.exists() and load_toml_config:
            try:
                cfg = load_toml_config(cfg_path)
                if "simulation_dir" in cfg:
                    p = Path(cfg["simulation_dir"])
                    resolved_output_dir = p if p.is_absolute() else REPO_ROOT / p
                if "bounding_box_dir" in cfg:
                    p = Path(cfg["bounding_box_dir"])
                    resolved_json_dir = p if p.is_absolute() else REPO_ROOT / p
            except Exception as e:
                print(f"[!] Warning reading TOML config {cfg_path}: {e}")

    if custom_json_dir:
        p = Path(custom_json_dir)
        resolved_json_dir = p if p.is_absolute() else REPO_ROOT / p

    if custom_stl_dir:
        p = Path(custom_stl_dir)
        resolved_stl_dir = p if p.is_absolute() else REPO_ROOT / p

    if custom_output_dir:
        p = Path(custom_output_dir)
        resolved_output_dir = p if p.is_absolute() else REPO_ROOT / p

    return resolved_json_dir, resolved_stl_dir, resolved_output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tạo file PLY kết hợp Dense Point Cloud và Bounding Box cho DTU (lưu vào result_simulation)."
    )
    parser.add_argument(
        "--config",
        default="configs/simulation.toml",
        help="Đường dẫn file config TOML (mặc định: configs/simulation.toml).",
    )
    parser.add_argument(
        "--scan",
        default=None,
        help="Scan cụ thể cần trực quan hóa (vd: scan24, 24, 24,37, hoặc 'all' cho 15 benchmark scans; mặc định: all).",
    )
    parser.add_argument(
        "--json-dir",
        default=None,
        help="Thư mục chứa các file JSON 3D box (mặc định: outputs/DTU/bounding_box_3D).",
    )
    parser.add_argument(
        "--stl-dir",
        default=None,
        help="Thư mục Points/stl chứa các file stlXXX_total.ply gốc (mặc định: tự động resolve).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Thư mục lưu file PLY kết quả (mặc định: simulation/result_simulation).",
    )
    parser.add_argument(
        "--box-type",
        type=str,
        choices=["arbitrary_3d_obb", "gravity_aligned_obb", "aabb"],
        default="arbitrary_3d_obb",
        help="Loại bounding box cần vẽ (mặc định: arbitrary_3d_obb).",
    )
    parser.add_argument(
        "--samples-per-edge",
        type=int,
        default=300,
        help="Số điểm lấy mẫu trên mỗi cạnh khung viền (mặc định: 300).",
    )
    parser.add_argument(
        "--no-unnormalize",
        action="store_true",
        help="Tắt chế độ tự động khôi phục tỉ lệ / vị trí bbox.",
    )

    args = parser.parse_args()

    json_dir, stl_dir, output_dir = resolve_dtu_viz_paths(
        config_file=args.config,
        custom_json_dir=args.json_dir,
        custom_stl_dir=args.stl_dir,
        custom_output_dir=args.output_dir,
    )

    target_scan = args.scan
    if target_scan is None and args.config:
        cfg_path = REPO_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
        if cfg_path.exists() and load_toml_config:
            cfg = load_toml_config(cfg_path)
            ds_cfg = cfg.get("dataset", {})
            if isinstance(ds_cfg, dict) and ds_cfg.get("name") == "dtu":
                target_scan = ds_cfg.get("scan") or ds_cfg.get("scene")

    run_dtu_batch_visualization(
        json_dir=json_dir,
        stl_dir=stl_dir,
        output_dir=output_dir,
        target_scan=target_scan or "all",
        box_type=args.box_type,
        samples_per_edge=args.samples_per_edge,
        unnormalize=(not args.no_unnormalize),
    )


if __name__ == "__main__":
    main()