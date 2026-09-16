"""Module 1 Perception: Generate 3D ROI Bounding Boxes for DTU Dataset.

Tự động trích xuất các hộp bao 3D (AABB, Gravity-aligned OBB, Arbitrary 3D OBB)
từ ground truth STL point clouds hoặc preprocessed SfM points của DTU Dataset.

Đường dẫn input và output được quản lý tự động thông qua relative paths kết hợp giữa:
    1. Safe-GS configs (configs/object_roi.toml hoặc configs/base_scene.toml)
    2. Dataset Configs (Dataset Configs/dtu.yaml qua src.common.dataset_config)
    3. Mặc định output: outputs/DTU/bounding_box_3D (tương đối theo repo root)

Cách chạy:
    1. Chạy trích xuất cho 1 scan cụ thể (ví dụ: scan24 hoặc 24):
    
        python src/module1_perception/generate_ROI_DTU.py --scan scan24
    
    2. Chạy hàng loạt (Batch) cho 15 benchmark scans của DTU Dataset:
    
        python src/module1_perception/generate_ROI_DTU.py --scan all
    
    3. Tùy chỉnh tham số hoặc đường dẫn khác nếu cần:

        python src/module1_perception/generate_ROI_DTU.py \
            --config configs/object_roi.toml \
            --output-dir outputs/DTU/bounding_box_3D \
            --expansion-ratio 0.05
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

# Thêm repo root vào sys.path để import an toàn dù script chạy từ bất kỳ thư mục nào
REPO_ROOT = Path(__file__).resolve().parents[2]
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


def compute_bounding_boxes(
    ply_path: Union[str, Path],
    scan_id: int,
    expansion_ratio: float = 0.05
) -> Optional[List[dict]]:
    """Tính toán AABB, Arbitrary 3D OBB và Gravity-aligned OBB từ file point cloud."""
    ply_path = Path(ply_path)
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(pcd.points)

    if len(pts) == 0:
        print(f"[!] Warning: File {ply_path} không có điểm nào.")
        return None

    # Tỉ lệ mở rộng biên (expansion margin)
    scale = 1.0 + expansion_ratio

    # 1. Axis-Aligned Bounding Box (AABB)
    aabb_min = np.min(pts, axis=0)
    aabb_max = np.max(pts, axis=0)
    aabb_center = (aabb_min + aabb_max) / 2.0
    aabb_sizes = (aabb_max - aabb_min) * scale
    aabb_min_scaled = aabb_center - aabb_sizes / 2.0
    aabb_max_scaled = aabb_center + aabb_sizes / 2.0

    # 2. Arbitrary 3D Oriented Bounding Box (3D OBB qua PCA / Open3D)
    obb_3d = pcd.get_oriented_bounding_box()
    obb_3d_center = np.array(obb_3d.center)
    obb_3d_sizes = np.array(obb_3d.extent) * scale
    obb_3d_rot = np.array(obb_3d.R)

    # 3. Gravity-aligned OBB (xoay quanh trục Z song song mặt sàn)
    pts_xy = pts[:, :2]
    xy_mean = np.mean(pts_xy, axis=0)
    xy_centered = pts_xy - xy_mean
    cov = np.cov(xy_centered, rowvar=False)
    _, eig_vecs = np.linalg.eigh(cov)

    pts_proj = xy_centered @ eig_vecs
    min_proj = np.min(pts_proj, axis=0)
    max_proj = np.max(pts_proj, axis=0)
    xy_sizes = (max_proj - min_proj) * scale
    xy_center_proj = (min_proj + max_proj) / 2.0
    xy_center = xy_mean + (xy_center_proj @ eig_vecs.T)

    yaw_rad = np.arctan2(eig_vecs[1, 0], eig_vecs[0, 0])
    yaw_deg = float(np.degrees(yaw_rad))

    grav_center = [float(xy_center[0]), float(xy_center[1]), float(aabb_center[2])]
    grav_sizes = [float(xy_sizes[0]), float(xy_sizes[1]), float(aabb_sizes[2])]

    data = [
        {
            "instance_id": 0,
            "name": f"scan{scan_id}_object",
            "category": "foreground_object",
            "num_points": int(len(pts)),
            "aabb": {
                "center": aabb_center.tolist(),
                "sizes": aabb_sizes.tolist(),
                "min": aabb_min_scaled.tolist(),
                "max": aabb_max_scaled.tolist(),
            },
            "gravity_aligned_obb": {
                "center": grav_center,
                "sizes": grav_sizes,
                "yaw_rad": float(yaw_rad),
                "yaw_deg": yaw_deg,
            },
            "arbitrary_3d_obb": {
                "center": obb_3d_center.tolist(),
                "sizes": obb_3d_sizes.tolist(),
                "rotation_matrix": obb_3d_rot.tolist(),
            },
        }
    ]

    return data


def parse_scan_id(scan_str: Union[str, int]) -> int:
    """Trích xuất ID scan dạng số: 'scan24' -> 24, '24' -> 24, 'stl024' -> 24."""
    if isinstance(scan_str, int):
        return scan_str
    match = re.search(r"(\d+)", str(scan_str))
    if match:
        return int(match.group(1))
    raise ValueError(f"Không thể phân tích scan ID từ: '{scan_str}'")


def resolve_dtu_io_paths(
    config_file: Optional[Union[str, Path]] = None,
    dataset_name: str = "dtu",
    custom_input_dir: Optional[Union[str, Path]] = None,
    custom_output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Path]:
    """Tự động resolve input_dir (chứa STL point clouds) và output_dir
    dựa trên configs TOML và Dataset Configs YAML với đường dẫn tương đối.
    """
    # 1. Resolve Output Directory (Mặc định: outputs/DTU/bounding_box_3D relative to repo)
    default_rel_output = Path("outputs/DTU/bounding_box_3D")
    resolved_output_dir = REPO_ROOT / default_rel_output

    if config_file:
        cfg_path = REPO_ROOT / config_file if not Path(config_file).is_absolute() else Path(config_file)
        if cfg_path.exists() and load_toml_config:
            try:
                cfg = load_toml_config(cfg_path)
                if "bounding_box_dir" in cfg:
                    p = Path(cfg["bounding_box_dir"])
                    resolved_output_dir = p if p.is_absolute() else REPO_ROOT / p
            except Exception as e:
                print(f"[!] Warning reading TOML config {cfg_path}: {e}")

    if custom_output_dir:
        p = Path(custom_output_dir)
        resolved_output_dir = p if p.is_absolute() else REPO_ROOT / p

    # 2. Resolve Input Directory (Thư mục chứa STL files)
    resolved_input_dir = None
    if load_dataset_paths:
        try:
            paths = load_dataset_paths(dataset_name, scan="scan24")
            if "gt_points_ply" in paths:
                stl_cand = Path(paths["gt_points_ply"]).parent
                if stl_cand.exists():
                    resolved_input_dir = stl_cand
        except Exception as e:
            print(f"[!] Warning reading dataset config '{dataset_name}': {e}")

    if resolved_input_dir is None:
        # Fallback 1: DTU Dataset/Points/Points/stl hoặc DTU Dataset/Points/stl ngang hàng repo
        candidates = [
            REPO_ROOT.parent / "DTU Dataset" / "Points" / "Points" / "stl",
            REPO_ROOT.parent / "DTU Dataset" / "Points" / "stl",
            REPO_ROOT / "Data" / "DTU" / "Points" / "stl",
        ]
        for cand in candidates:
            if cand.exists():
                resolved_input_dir = cand
                break

    if custom_input_dir:
        p = Path(custom_input_dir)
        resolved_input_dir = p if p.is_absolute() else REPO_ROOT / p

    if resolved_input_dir is None:
        resolved_input_dir = REPO_ROOT.parent / "DTU Dataset" / "Points" / "Points" / "stl"

    return resolved_input_dir, resolved_output_dir


def process_dtu_scans(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    target_scan: Optional[Union[str, int]] = None,
    expansion_ratio: float = 0.05,
) -> None:
    """Xử lý tạo 3D bounding boxes cho các scan DTU."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if target_scan is not None and str(target_scan).lower() not in ("all", "benchmark", "default"):
        if str(target_scan).lower() == "full":
            # Chạy toàn bộ 124 scan có trong thư mục dataset
            if input_dir.exists():
                found_scans = []
                for f in input_dir.glob("stl*_total.ply"):
                    try:
                        found_scans.append(parse_scan_id(f.name))
                    except ValueError:
                        pass
                scans_to_process = sorted(list(set(found_scans))) if found_scans else DEFAULT_BENCHMARK_SCANS
            else:
                scans_to_process = DEFAULT_BENCHMARK_SCANS
        elif "," in str(target_scan):
            scans_to_process = [parse_scan_id(s) for s in str(target_scan).split(",")]
        else:
            scan_id = parse_scan_id(target_scan)
            scans_to_process = [scan_id]
    else:
        # Mặc định khi chạy 'all' hoặc không truyền: Chạy 15 scan chuẩn benchmark
        scans_to_process = DEFAULT_BENCHMARK_SCANS


    print(f"[*] Input STL Directory: {input_dir}")
    print(f"[*] Output Directory:    {output_dir}")
    print(f"[*] Expansion Ratio:     {expansion_ratio * 100:.1f}%")
    print(f"[*] Target Scan(s):      {scans_to_process}")
    print("=" * 70)

    success_count = 0
    skipped_count = 0

    for idx, scan_id in enumerate(scans_to_process, 1):
        filename = f"stl{scan_id:03d}_total.ply"
        ply_path = input_dir / filename

        print(f"[{idx}/{len(scans_to_process)}] Processing Scan {scan_id}: {filename}")

        if not ply_path.is_file():
            # Thử fallback qua sfm points trong preprocessed DTU nếu có
            fallback_ply = None
            if load_dataset_paths:
                try:
                    p = load_dataset_paths("dtu", scan=f"scan{scan_id}")
                    if "sfm_points_ply" in p and Path(p["sfm_points_ply"]).is_file():
                        fallback_ply = Path(p["sfm_points_ply"])
                except Exception:
                    pass

            if fallback_ply and fallback_ply.is_file():
                print(f"    [*] STL not found, using SfM points fallback: {fallback_ply}")
                ply_path = fallback_ply
            else:
                print(f"    [-] Warning: File {ply_path} không tồn tại. Bỏ qua.")
                skipped_count += 1
                print("-" * 70)
                continue

        try:
            boxes_data = compute_bounding_boxes(
                ply_path=ply_path,
                scan_id=scan_id,
                expansion_ratio=expansion_ratio,
            )

            if boxes_data is not None:
                # Nếu output_dir đã là thư mục riêng của scan đó, lưu trực tiếp
                if output_dir.name in (f"dtu_scan{scan_id}", f"scan{scan_id}", f"scan_{scan_id}"):
                    out_file = output_dir / f"scan{scan_id}_3d_boxes.json"
                else:
                    # Ngược lại, lưu vào subfolder dtu_scan{id} bên trong output_dir
                    sub_dir = output_dir / f"dtu_scan{scan_id}"
                    sub_dir.mkdir(parents=True, exist_ok=True)
                    out_file = sub_dir / f"scan{scan_id}_3d_boxes.json"

                out_file.parent.mkdir(parents=True, exist_ok=True)
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(boxes_data, f, indent=2)

                print(f"    [✓] Đã lưu: {out_file}")
                success_count += 1

            else:
                skipped_count += 1
        except Exception as e:
            print(f"    [!] Lỗi khi xử lý scan {scan_id}: {e}")
            skipped_count += 1

        print("-" * 70)

    print(
        f"[✓] Finished! Successfully processed: {success_count} scan(s) | "
        f"Skipped/Failed: {skipped_count} scan(s)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract 3D ROI Bounding Boxes for DTU scans using relative configs."
    )
    parser.add_argument(
        "--config",
        default="configs/object_roi.toml",
        help="Đường dẫn file config TOML (mặc định: configs/object_roi.toml).",
    )
    parser.add_argument(
        "--dataset-config",
        default="dtu",
        help="Tên dataset config trong Dataset Configs/ (mặc định: dtu).",
    )
    parser.add_argument(
        "--scan",
        default=None,
        help="Scan cụ thể cần chạy (vd: scan24, 24, hoặc 'all' để chạy tất cả 15 benchmark scans; mặc định lấy từ config hoặc all).",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Ghi đè đường dẫn thư mục chứa STL point clouds.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Ghi đè đường dẫn thư mục output (mặc định: outputs/DTU/bounding_box_3D).",
    )
    parser.add_argument(
        "--expansion-ratio",
        type=float,
        default=0.05,
        help="Tỉ lệ nới rộng bounding box (mặc định: 0.05 tương ứng 5%%).",
    )

    args = parser.parse_args()

    input_dir, output_dir = resolve_dtu_io_paths(
        config_file=args.config,
        dataset_name=args.dataset_config,
        custom_input_dir=args.input_dir,
        custom_output_dir=args.output_dir,
    )

    # Đọc scan từ config nếu người dùng không truyền vào CLI
    target_scan = args.scan
    if target_scan is None and args.config:
        cfg_path = REPO_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
        if cfg_path.exists() and load_toml_config:
            cfg = load_toml_config(cfg_path)
            ds_cfg = cfg.get("dataset", {})
            if isinstance(ds_cfg, dict) and ds_cfg.get("name") == "dtu":
                target_scan = ds_cfg.get("scan") or ds_cfg.get("scene")

    process_dtu_scans(
        input_dir=input_dir,
        output_dir=output_dir,
        target_scan=target_scan or "all",
        expansion_ratio=args.expansion_ratio,
    )


if __name__ == "__main__":
    main()