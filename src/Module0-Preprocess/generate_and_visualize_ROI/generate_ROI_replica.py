"""Module 1 Perception: Generate 3D ROI Bounding Boxes for Replica Dataset.

Tự động trích xuất các hộp bao 3D (AABB, Gravity-aligned OBB, Arbitrary 3D OBB) từ semantic mesh và metadata của Replica Dataset.

Đường dẫn input và output được quản lý tự động thông qua relative paths kết hợp giữa:
    1. Safe-GS configs (configs/object_roi.toml hoặc configs/base_scene.toml)
    2. Dataset Configs (Dataset Configs/replica.yaml qua src.common.dataset_config)
    3. Mặc định output: outputs/Replica/bounding_box_3D (tương đối theo repo root)

Cách chạy:
    1. Chạy trích xuất cho 1 scene cụ thể (ví dụ: office0): 
    
        python src/module1_perception/generate_ROI_replica.py --scene office0
    
    2. Chạy hàng loạt (Batch) cho toàn bộ các scene trong Replica Dataset: 
    
        python src/module1_perception/generate_ROI_replica.py --scene all
    
    3. Tùy chỉnh tham số hoặc đường dẫn khác nếu cần:

        python src/module1_perception/generate_ROI_replica.py \
        --config configs/object_roi.toml \
        --output-dir outputs/Replica/bounding_box_3D \
        --up-axis y \
        --min-points 30
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
from plyfile import PlyData
from scipy.spatial import ConvexHull

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


def minimum_bounding_rectangle_2d(points_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Tính hình chữ nhật bao quanh nhỏ nhất (Minimum Area Bounding Box)
    trên mặt phẳng 2D sử dụng Rotating Calipers trên Convex Hull.
    """
    if len(points_2d) < 3:
        min_xy = points_2d.min(axis=0)
        max_xy = points_2d.max(axis=0)
        center = (min_xy + max_xy) / 2.0
        dims = max_xy - min_xy
        return center, dims, 0.0

    hull = ConvexHull(points_2d)
    hull_points = points_2d[hull.vertices]

    edges = np.diff(hull_points, axis=0, append=hull_points[:1])
    edge_angles = np.arctan2(edges[:, 1], edges[:, 0])
    edge_angles = np.unique(np.mod(edge_angles, np.pi / 2))

    rotations = np.vstack([
        np.cos(edge_angles), -np.sin(edge_angles),
        np.sin(edge_angles), np.cos(edge_angles)
    ]).T.reshape(-1, 2, 2)

    rot_points = np.matmul(rotations, hull_points.T)
    min_x = rot_points[:, 0, :].min(axis=1)
    max_x = rot_points[:, 0, :].max(axis=1)
    min_y = rot_points[:, 1, :].min(axis=1)
    max_y = rot_points[:, 1, :].max(axis=1)

    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    best_angle = edge_angles[best_idx]
    rot_mat = rotations[best_idx]

    c_x = (max_x[best_idx] + min_x[best_idx]) / 2.0
    c_y = (max_y[best_idx] + min_y[best_idx]) / 2.0
    center = np.linalg.inv(rot_mat) @ np.array([c_x, c_y])

    dim_x = max_x[best_idx] - min_x[best_idx]
    dim_y = max_y[best_idx] - min_y[best_idx]

    return center, np.array([dim_x, dim_y]), float(best_angle)


def compute_gravity_aligned_obb(points: np.ndarray, up_axis: str = "y") -> dict:
    """Tính 3D OBB căn theo trục đứng, chỉ xoay trên mặt phẳng sàn."""
    if up_axis.lower() == "y":
        ground_pts = points[:, [0, 2]]  # X, Z
        up_pts = points[:, 1]           # Y
    else:
        ground_pts = points[:, [0, 1]]  # X, Y
        up_pts = points[:, 2]           # Z

    center_2d, dims_2d, yaw = minimum_bounding_rectangle_2d(ground_pts)

    min_up = float(np.min(up_pts))
    max_up = float(np.max(up_pts))
    height = max_up - min_up
    center_up = (min_up + max_up) / 2.0

    if up_axis.lower() == "y":
        center = [float(center_2d[0]), float(center_up), float(center_2d[1])]
        sizes = [float(dims_2d[0]), float(height), float(dims_2d[1])]
    else:
        center = [float(center_2d[0]), float(center_2d[1]), float(center_up)]
        sizes = [float(dims_2d[0]), float(dims_2d[1]), float(height)]

    return {
        "center": center,
        "sizes": sizes,
        "yaw_rad": float(yaw),
        "yaw_deg": float(np.degrees(yaw)),
    }


def compute_full_3d_obb_pca(points: np.ndarray) -> dict:
    """Tính OBB 3D tự do (3 chiều xoay tự do) bằng PCA."""
    center = np.mean(points, axis=0)
    centered = points - center
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, -1] *= -1

    proj = centered @ eigenvectors
    min_bounds = proj.min(axis=0)
    max_bounds = proj.max(axis=0)

    obb_center = center + eigenvectors @ ((min_bounds + max_bounds) / 2.0)
    obb_sizes = max_bounds - min_bounds

    return {
        "center": obb_center.tolist(),
        "sizes": obb_sizes.tolist(),
        "rotation_matrix": eigenvectors.tolist(),
    }


def extract_replica_3d_boxes(
    ply_path: Union[str, Path],
    json_path: Union[str, Path],
    output_path: Union[str, Path] = "replica_3d_boxes.json",
    up_axis: str = "y",
    min_points: int = 30,
) -> List[dict]:
    """Hàm xử lý và trích xuất 3D OBB cho một scene."""
    ply_path = Path(ply_path)
    json_path = Path(json_path)
    output_path = Path(output_path)

    print(f"[*] Loading metadata from: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        info_semantic = json.load(f)

    id_to_meta = {}
    if "classes" in info_semantic and "objects" in info_semantic:
        classes = {c["id"]: c["name"] for c in info_semantic["classes"]}
        for obj in info_semantic["objects"]:
            obj_id = obj.get("id")
            class_id = obj.get("class_id")
            name = obj.get("name", classes.get(class_id, "unknown"))
            id_to_meta[obj_id] = {
                "instance_name": name,
                "category": classes.get(class_id, "unknown"),
            }
    elif "objects" in info_semantic:
        for obj in info_semantic["objects"]:
            id_to_meta[obj.get("id")] = {
                "instance_name": obj.get("name", "unknown"),
                "category": obj.get("class_name", "unknown"),
            }

    print(f"[*] Reading PLY mesh geometry: {ply_path}")
    plydata = PlyData.read(str(ply_path))
    vertex_data = plydata["vertex"]

    x = np.asarray(vertex_data["x"])
    y = np.asarray(vertex_data["y"])
    z = np.asarray(vertex_data["z"])
    coords = np.stack([x, y, z], axis=1)

    id_prop_name = None
    for cand in ["object_id", "objectId", "label", "category"]:
        if cand in vertex_data.data.dtype.names:
            id_prop_name = cand
            break

    if id_prop_name is not None:
        object_ids = np.asarray(vertex_data[id_prop_name])
    else:
        face_data = plydata["face"]
        face_obj_ids = np.asarray(face_data["object_id"])
        face_indices = face_data["vertex_indices"]

        object_ids = np.full(coords.shape[0], fill_value=-1, dtype=np.int32)
        for f_idx, v_list in enumerate(face_indices):
            obj_id = face_obj_ids[f_idx]
            object_ids[v_list] = obj_id

    unique_ids = np.unique(object_ids)
    print(f"[*] Found {len(unique_ids)} instances. Computing bounding boxes...")

    results = []
    for uid in unique_ids:
        if uid < 0:
            continue

        mask = (object_ids == uid)
        instance_pts = coords[mask]

        if len(instance_pts) < min_points:
            continue

        centroid = np.median(instance_pts, axis=0)
        dist = np.linalg.norm(instance_pts - centroid, axis=1)
        inlier_mask = dist < np.percentile(dist, 98)
        clean_pts = instance_pts[inlier_mask]

        if len(clean_pts) < min_points:
            clean_pts = instance_pts

        meta = id_to_meta.get(uid, {"instance_name": f"instance_{uid}", "category": "unknown"})
        gravity_obb = compute_gravity_aligned_obb(clean_pts, up_axis=up_axis)
        pca_obb = compute_full_3d_obb_pca(clean_pts)

        aabb_min = clean_pts.min(axis=0).tolist()
        aabb_max = clean_pts.max(axis=0).tolist()

        record = {
            "instance_id": int(uid),
            "name": meta["instance_name"],
            "category": meta["category"],
            "num_points": int(len(clean_pts)),
            "aabb": {
                "min": aabb_min,
                "max": aabb_max,
            },
            "gravity_aligned_obb": gravity_obb,
            "arbitrary_3d_obb": pca_obb,
        }
        results.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[✓] Exported {len(results)} instances to: {output_path}\n")
    return results


def normalize_scene_name(scene_name: str) -> str:
    """Chuẩn hóa tên scene: 'office_0' -> 'office0', 'room_1' -> 'room1'."""
    return re.sub(r"_(\d+)$", r"\1", scene_name.strip())


def resolve_paths_from_configs(
    config_file: Optional[Union[str, Path]] = None,
    dataset_name: str = "replica",
    custom_input_dir: Optional[Union[str, Path]] = None,
    custom_output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Path]:
    """Tự động resolve input_dir (chứa raw Replica-Dataset) và output_dir
    dựa trên configs TOML và Dataset Configs YAML với đường dẫn tương đối.
    """
    # 1. Resolve Output Directory (Mặc định: outputs/Replica/bounding_box_3D relative to repo)
    default_rel_output = Path("outputs/Replica/bounding_box_3D")
    resolved_output_dir = REPO_ROOT / default_rel_output

    if config_file:
        cfg_path = REPO_ROOT / config_file if not Path(config_file).is_absolute() else Path(config_file)
        if cfg_path.exists() and load_toml_config:
            try:
                cfg = load_toml_config(cfg_path)
                ds_cfg = cfg.get("dataset", {})
                is_replica_cfg = (isinstance(ds_cfg, dict) and ds_cfg.get("name") == "replica") or "replica" in str(cfg_path).lower()

                if "bounding_box_dir" in cfg:
                    p = Path(cfg["bounding_box_dir"])
                    resolved_output_dir = p if p.is_absolute() else REPO_ROOT / p
                elif is_replica_cfg and "workspace_dir" in cfg and "bounding_box_3D" in cfg["workspace_dir"]:
                    p = Path(cfg["workspace_dir"])
                    resolved_output_dir = p if p.is_absolute() else REPO_ROOT / p
            except Exception as e:
                print(f"[!] Warning reading TOML config {cfg_path}: {e}")

    if custom_output_dir:
        p = Path(custom_output_dir)
        resolved_output_dir = p if p.is_absolute() else REPO_ROOT / p


    # 2. Resolve Input Directory (Raw Replica-Dataset)
    resolved_input_dir = None
    if load_dataset_paths:
        try:
            paths = load_dataset_paths(dataset_name)
            if "raw_dataset_dir" in paths and Path(paths["raw_dataset_dir"]).exists():
                resolved_input_dir = Path(paths["raw_dataset_dir"])
        except Exception as e:
            print(f"[!] Warning reading dataset config '{dataset_name}': {e}")

    if resolved_input_dir is None:
        # Fallback 1: Thư mục Replica/Replica-Dataset ngang hàng với Safe-GS
        fallback = REPO_ROOT.parent / "Replica" / "Replica-Dataset"
        if fallback.exists():
            resolved_input_dir = fallback
        else:
            # Fallback 2: Thư mục Data/Replica trong repo
            resolved_input_dir = REPO_ROOT / "Data" / "Replica"

    if custom_input_dir:
        p = Path(custom_input_dir)
        resolved_input_dir = p if p.is_absolute() else REPO_ROOT / p

    return resolved_input_dir, resolved_output_dir


# Danh sách 8 scene chuẩn benchmark của Replica Dataset
DEFAULT_BENCHMARK_SCENES = ["office0", "office1", "office2", "office3", "office4", "room0", "room1", "room2"]


def process_all_replica_scenes(
    dataset_root: Union[str, Path],
    output_dir: Union[str, Path],
    target_scene: Optional[str] = None,
    up_axis: str = "y",
    min_points: int = 30,
) -> None:
    """Duyệt qua các scene trong Replica-Dataset và trích xuất 3D OBBs."""
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Root dataset directory does not exist: {dataset_root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Lấy danh sách thư mục scene có trong dataset
    all_scenes = sorted([
        d.name for d in dataset_root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    if target_scene is not None and str(target_scene).lower() not in ("all", "benchmark", "default"):
        if str(target_scene).lower() == "full":
            # Chạy toàn bộ tất cả scene có trong dataset (ví dụ 18 scene)
            scenes_to_process = all_scenes
        elif "," in str(target_scene):
            # Chạy danh sách scene chỉ định (ví dụ: office0,room1)
            target_list = [s.strip() for s in str(target_scene).split(",")]
            target_norms = [normalize_scene_name(s).lower() for s in target_list]
            scenes_to_process = [
                s for s in all_scenes
                if s.lower() in [t.lower() for t in target_list] or normalize_scene_name(s).lower() in target_norms
            ]
        else:
            target_norm = normalize_scene_name(target_scene).lower()
            matched_scenes = [
                s for s in all_scenes
                if s.lower() == target_scene.lower() or normalize_scene_name(s).lower() == target_norm
            ]
            if not matched_scenes:
                raise ValueError(
                    f"Scene '{target_scene}' không tìm thấy trong {dataset_root}.\n"
                    f"Danh sách scene có sẵn: {all_scenes}"
                )
            scenes_to_process = matched_scenes
    else:
        # Mặc định khi chạy 'all' hoặc không truyền: Chạy 8 benchmark scenes
        benchmark_norms = [normalize_scene_name(s).lower() for s in DEFAULT_BENCHMARK_SCENES]
        scenes_to_process = [
            s for s in all_scenes
            if normalize_scene_name(s).lower() in benchmark_norms
        ]
        if not scenes_to_process:
            scenes_to_process = all_scenes


    print(f"[*] Input Dataset Root: {dataset_root}")
    print(f"[*] Output Directory:   {output_dir}")
    print(f"[*] Found {len(scenes_to_process)} scene(s) to process: {scenes_to_process}")
    print("=" * 70)

    success_count = 0
    skipped_count = 0

    for idx, scene_name in enumerate(scenes_to_process, 1):
        scene_dir = dataset_root / scene_name
        ply_path = scene_dir / "habitat" / "mesh_semantic.ply"
        json_path = scene_dir / "habitat" / "info_semantic.json"

        norm_name = normalize_scene_name(scene_name)
        if output_dir.name.lower() in (scene_name.lower(), norm_name.lower()):
            scene_out_path = output_dir / f"{scene_name}_3d_boxes.json"
        else:
            sub_dir = output_dir / norm_name
            sub_dir.mkdir(parents=True, exist_ok=True)
            scene_out_path = sub_dir / f"{scene_name}_3d_boxes.json"

        print(f"[{idx}/{len(scenes_to_process)}] Processing Scene: {scene_name}")

        if not ply_path.is_file() or not json_path.is_file():
            print(f"[-] Warning: Missing PLY or JSON in '{scene_name}/habitat/'. Skipping.")
            skipped_count += 1
            print("-" * 70)
            continue

        try:
            results = extract_replica_3d_boxes(
                ply_path=ply_path,
                json_path=json_path,
                output_path=scene_out_path,
                up_axis=up_axis,
                min_points=min_points,
            )
            success_count += 1

        except Exception as e:
            print(f"[!] Error processing {scene_name}: {e}")
            skipped_count += 1

        print("-" * 70)

    print(
        f"[✓] Finished! Successfully processed: {success_count} scene(s) | "
        f"Skipped/Failed: {skipped_count} scene(s)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract 3D ROI Bounding Boxes for Replica scenes using relative configs."
    )
    parser.add_argument(
        "--config",
        default="configs/object_roi.toml",
        help="Đường dẫn file config TOML (mặc định: configs/object_roi.toml).",
    )
    parser.add_argument(
        "--dataset-config",
        default="replica",
        help="Tên dataset config trong Dataset Configs/ (mặc định: replica).",
    )
    parser.add_argument(
        "--scene",
        default=None,
        help="Scene cụ thể cần chạy (vd: office0, office_0, room0, hoặc 'all' để chạy tất cả; mặc định lấy từ config hoặc all).",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Ghi đè đường dẫn thư mục raw dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Ghi đè đường dẫn thư mục output (mặc định: outputs/Replica/bounding_box_3D).",
    )
    parser.add_argument(
        "--up-axis",
        default="y",
        choices=["y", "z"],
        help="Trục thẳng đứng ('y' hoặc 'z', mặc định: 'y').",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=30,
        help="Số lượng điểm tối thiểu của 1 instance để tạo bounding box (mặc định: 30).",
    )

    args = parser.parse_args()

    # Tự động tìm kiếm input và output paths từ các configs
    input_dir, output_dir = resolve_paths_from_configs(
        config_file=args.config,
        dataset_name=args.dataset_config,
        custom_input_dir=args.input_dir,
        custom_output_dir=args.output_dir,
    )

    # Đọc scene từ config nếu người dùng không truyền vào CLI
    target_scene = args.scene
    if target_scene is None and args.config:
        cfg_path = REPO_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
        if cfg_path.exists() and load_toml_config:
            cfg = load_toml_config(cfg_path)
            dataset_cfg = cfg.get("dataset", {})
            if isinstance(dataset_cfg, dict) and dataset_cfg.get("name") == "replica":
                target_scene = dataset_cfg.get("scene")

    process_all_replica_scenes(
        dataset_root=input_dir,
        output_dir=output_dir,
        target_scene=target_scene or "all",
        up_axis=args.up_axis,
        min_points=args.min_points,
    )


if __name__ == "__main__":
    main()