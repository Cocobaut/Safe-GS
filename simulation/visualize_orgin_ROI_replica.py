"""Module Simulation: Visualize 3D ROI Bounding Boxes on Replica Ground Truth Mesh.

Trực quan hóa các 3D Bounding Boxes (Arbitrary OBB, Gravity-aligned OBB, AABB)
được xuất từ Module 1 Perception chồng lấn lên Ground Truth Mesh của Replica Dataset.

Đường dẫn input và output được quản lý tự động thông qua relative paths kết hợp giữa:
    1. Safe-GS configs (configs/simulation.toml hoặc configs/object_roi.toml)
    2. Dataset Configs (Dataset Configs/replica.yaml qua src.common.dataset_config)
    3. Input Bboxes:  outputs/Replica/bounding_box_3D (tương đối theo repo root)
    4. Mặc định output: simulation/result_simulation (tương đối theo repo root)

Cách chạy:
    1. Trực quan hóa 1 scene cụ thể (ví dụ: office0):
    
        python simulation/visualize_orgin_ROI_replica.py --scene office0
    
    2. Trực quan hóa toàn bộ 8 benchmark scenes:
    
        python simulation/visualize_orgin_ROI_replica.py --scene all
    
    3. Tùy chỉnh tham số hoặc đường dẫn:

        python simulation/visualize_orgin_ROI_replica.py \
            --json-dir outputs/Replica/bounding_box_3D \
            --output-dir simulation/result_simulation \
            --box-mode arbitrary_obb \
            --box-line-radius 0.003
"""

from __future__ import annotations

import argparse
import colorsys
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import open3d as o3d
import trimesh

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

# Danh sách 8 scene chuẩn benchmark của Replica Dataset
DEFAULT_BENCHMARK_SCENES = ["office0", "office1", "office2", "office3", "office4", "room0", "room1", "room2"]


def normalize_scene_name(scene_name: str) -> str:
    """Chuẩn hóa tên scene: 'office_0' -> 'office0', 'room_1' -> 'room1'."""
    return re.sub(r"_(\d+)$", r"\1", scene_name.strip())


def generate_distinct_colors(num_colors: int) -> List[List[float]]:
    """Sinh danh sách N màu sắc tách biệt hoàn toàn trên vòng tròn màu HSV."""
    colors = []
    golden_ratio_conjugate = 0.618033988749895
    h = 0.1

    for _ in range(num_colors):
        h = (h + golden_ratio_conjugate) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        colors.append([float(r), float(g), float(b)])
    return colors


def create_obb_cylinder_mesh_from_corners(
    corners: np.ndarray,
    color: List[float],
    radius: float = 0.003
) -> o3d.geometry.TriangleMesh:
    """Tạo TriangleMesh khung dây hình trụ cho OBB từ 8 tọa độ đỉnh chuẩn Open3D."""
    corners = np.asarray(corners, dtype=np.float64)
    if corners.shape != (8, 3):
        raise ValueError(f"Tập đỉnh OBB phải có kích thước (8, 3), nhận được {corners.shape}")

    lines = [
        (0, 1), (1, 7), (7, 2), (2, 0),  # Mặt đáy 1
        (3, 6), (6, 4), (4, 5), (5, 3),  # Mặt đỉnh 2
        (0, 3), (1, 6), (7, 4), (2, 5),  # Cạnh nối cột
    ]

    bbox_mesh = o3d.geometry.TriangleMesh()

    for start_idx, end_idx in lines:
        p_start = corners[start_idx]
        p_end = corners[end_idx]
        line_vec = p_end - p_start
        line_len = np.linalg.norm(line_vec)

        if line_len < 1e-6:
            continue

        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius, height=line_len, resolution=8
        )
        cylinder.paint_uniform_color(color)

        z_axis = np.array([0.0, 0.0, 1.0])
        dir_norm = line_vec / line_len
        rot_axis = np.cross(z_axis, dir_norm)
        rot_axis_len = np.linalg.norm(rot_axis)

        if rot_axis_len > 1e-6:
            rot_axis = rot_axis / rot_axis_len
            angle = np.arccos(np.clip(np.dot(z_axis, dir_norm), -1.0, 1.0))
            K = np.array([
                [0.0, -rot_axis[2], rot_axis[1]],
                [rot_axis[2], 0.0, -rot_axis[0]],
                [-rot_axis[1], rot_axis[0], 0.0],
            ])
            R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
            cylinder.rotate(R, center=np.array([0.0, 0.0, 0.0]))

        cylinder.translate((p_start + p_end) / 2.0)
        bbox_mesh += cylinder

    return bbox_mesh


def extract_corners_from_item(item: dict, preferred_mode: str = "arbitrary_obb") -> Optional[np.ndarray]:
    """Trích xuất 8 đỉnh từ 1 phần tử JSON (hỗ trợ arbitrary_3d_obb, gravity_aligned_obb, aabb, obb_tight)."""
    # 1. Nếu là định dạng arbitrary_3d_obb
    if preferred_mode == "arbitrary_obb" and "arbitrary_3d_obb" in item:
        data = item["arbitrary_3d_obb"]
        center = np.array(data["center"], dtype=np.float64)
        extent = np.array(data["sizes"], dtype=np.float64)
        R = np.array(data["rotation_matrix"], dtype=np.float64)
        obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
        return np.asarray(obb.get_box_points())

    # 2. Nếu là định dạng gravity_aligned_obb
    if (preferred_mode == "gravity_aligned_obb" or "arbitrary_3d_obb" not in item) and "gravity_aligned_obb" in item:
        data = item["gravity_aligned_obb"]
        center = np.array(data["center"], dtype=np.float64)
        extent = np.array(data["sizes"], dtype=np.float64)
        yaw = float(data.get("yaw_rad", 0.0))
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        R_z = np.array([
            [cos_y, -sin_y, 0.0],
            [sin_y,  cos_y, 0.0],
            [0.0,    0.0,   1.0]
        ], dtype=np.float64)
        obb = o3d.geometry.OrientedBoundingBox(center, R_z, extent)
        return np.asarray(obb.get_box_points())

    # 3. Nếu là định dạng aabb
    if "aabb" in item:
        aabb_data = item["aabb"]
        if "min" in aabb_data and "max" in aabb_data:
            min_b = np.array(aabb_data["min"], dtype=np.float64)
            max_b = np.array(aabb_data["max"], dtype=np.float64)
        elif "center" in aabb_data and "sizes" in aabb_data:
            c = np.array(aabb_data["center"], dtype=np.float64)
            s = np.array(aabb_data["sizes"], dtype=np.float64)
            min_b = c - s / 2.0
            max_b = c + s / 2.0
        else:
            return None
        aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_b, max_bound=max_b)
        return np.asarray(aabb.get_box_points())

    # 4. Định dạng pipeline cũ: obb_padded_roi hoặc obb_tight
    target_dict = item.get("obb_padded_roi") or item.get("obb_tight") or item
    if "corners" in target_dict:
        return np.array(target_dict["corners"], dtype=np.float64)

    if "center" in target_dict and "extent" in target_dict and "rotation_matrix" in target_dict:
        center = np.array(target_dict["center"], dtype=np.float64)
        extent = np.array(target_dict["extent"], dtype=np.float64)
        R = np.array(target_dict["rotation_matrix"], dtype=np.float64)
        obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
        return np.asarray(obb.get_box_points())

    return None


def resolve_json_file(json_dir: Union[str, Path], scene_name: str) -> Optional[Path]:
    """Tự động tìm kiếm file JSON tương ứng của scene trong json_dir."""
    json_dir = Path(json_dir)
    norm_name = normalize_scene_name(scene_name)
    alt_name = scene_name.replace("office", "office_").replace("room", "room_")

    candidates = [
        json_dir / norm_name / f"{scene_name}_3d_boxes.json",
        json_dir / norm_name / f"{alt_name}_3d_boxes.json",
        json_dir / norm_name / f"{norm_name}_3d_boxes.json",
        json_dir / alt_name / f"{scene_name}_3d_boxes.json",
        json_dir / alt_name / f"{alt_name}_3d_boxes.json",
        json_dir / f"{scene_name}_3d_boxes.json",
        json_dir / f"{alt_name}_3d_boxes.json",
        json_dir / f"{norm_name}_3d_boxes.json",
        json_dir / f"roi_bounding_box_{scene_name}.json",
        json_dir / f"roi_bounding_box_{alt_name}.json",
    ]

    for c in candidates:
        if c.is_file():
            return c
    return None


def resolve_gt_mesh_file(
    gt_dir: Optional[Union[str, Path]],
    scene_name: str,
    dataset_name: str = "replica"
) -> Optional[Path]:
    """Tìm file Ground Truth Mesh .ply cho scene."""
    norm_name = normalize_scene_name(scene_name)
    alt_name = scene_name.replace("office", "office_").replace("room", "room_")

    # 1. Nếu có gt_dir chỉ định
    if gt_dir:
        gt_path = Path(gt_dir)
        cands = [
            gt_path / f"{norm_name}_mesh.ply",
            gt_path / f"{scene_name}_mesh.ply",
            gt_path / f"{alt_name}_mesh.ply",
            gt_path / norm_name / "mesh.ply",
            gt_path / alt_name / "mesh.ply",
            gt_path / norm_name / "habitat" / "mesh_semantic.ply",
            gt_path / alt_name / "habitat" / "mesh_semantic.ply",
        ]
        for c in cands:
            if c.is_file():
                return c

    # 2. Sử dụng Dataset Configs (replica.yaml)
    if load_dataset_paths:
        try:
            paths = load_dataset_paths(dataset_name, scene=norm_name)
            if "gt_mesh_ply" in paths and Path(paths["gt_mesh_ply"]).is_file():
                return Path(paths["gt_mesh_ply"])
        except Exception:
            pass

    # 3. Fallback tìm kiếm trong các thư mục mặc định
    fallback_candidates = [
        REPO_ROOT.parent / "Replica 8 Scene" / "Replica" / f"{norm_name}_mesh.ply",
        REPO_ROOT.parent / "Replica 8 Scene" / "Replica" / f"{scene_name}_mesh.ply",
        REPO_ROOT / "Data" / "Replica" / f"{norm_name}_mesh.ply",
        REPO_ROOT / "Data" / "Replica" / f"{scene_name}_mesh.ply",
    ]
    for c in fallback_candidates:
        if c.is_file():
            return c

    return None


def export_single_scene_visualization(
    json_path: Path,
    gt_geometry_path: Path,
    output_ply_path: Path,
    scene_name: str,
    box_mode: str = "arbitrary_obb",
    line_radius: float = 0.003,
) -> bool:
    """Đọc JSON boxes và GT mesh, tạo wireframe cylinder boxes và lưu file PLY kết hợp."""
    if not json_path.is_file():
        print(f"[-] Warning: Không tìm thấy file JSON: {json_path}")
        return False

    if not gt_geometry_path.is_file():
        print(f"[-] Warning: Không tìm thấy file GT Mesh: {gt_geometry_path}")
        return False

    print(f"      * Đọc JSON: {json_path.name}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        target_objects = data
    elif isinstance(data, dict) and "objects" in data and isinstance(data["objects"], list):
        target_objects = data["objects"]
    else:
        target_objects = [data]

    num_objects = len(target_objects)
    palette = generate_distinct_colors(num_objects)

    # Đọc Ground Truth Mesh
    is_mesh = False
    try:
        loaded_geom = trimesh.load(str(gt_geometry_path), force="mesh", process=False)
        gt_geom = o3d.geometry.TriangleMesh()
        gt_geom.vertices = o3d.utility.Vector3dVector(loaded_geom.vertices)
        gt_geom.triangles = o3d.utility.Vector3iVector(loaded_geom.faces)

        if len(gt_geom.triangles) > 0:
            is_mesh = True
            gt_geom.compute_vertex_normals()
            gt_geom.paint_uniform_color([0.75, 0.75, 0.75])
    except Exception:
        is_mesh = False

    if not is_mesh:
        gt_geom = o3d.io.read_point_cloud(str(gt_geometry_path))
        if len(gt_geom.points) == 0:
            print(f"[!] Error: Không thể đọc dữ liệu điểm/mesh từ: {gt_geometry_path}")
            return False
        if not gt_geom.has_colors():
            gt_geom.paint_uniform_color([0.7, 0.7, 0.7])

    # Tạo TriangleMesh khung viền cho các Bounding Box
    combined_boxes_mesh = o3d.geometry.TriangleMesh()
    valid_box_count = 0

    for idx, obj in enumerate(target_objects):
        corners = extract_corners_from_item(obj, preferred_mode=box_mode)
        if corners is not None and corners.shape == (8, 3):
            box_mesh = create_obb_cylinder_mesh_from_corners(
                corners, color=palette[idx], radius=line_radius
            )
            combined_boxes_mesh += box_mesh
            valid_box_count += 1

    # Xuất file PLY kết hợp
    output_ply_path.parent.mkdir(parents=True, exist_ok=True)

    if is_mesh:
        final_mesh = gt_geom + combined_boxes_mesh
        o3d.io.write_triangle_mesh(str(output_ply_path), final_mesh, write_ascii=False)
    else:
        sample_count = 80000 * max(1, valid_box_count)
        box_pcd = combined_boxes_mesh.sample_points_uniformly(number_of_points=sample_count)
        final_pcd = gt_geom + box_pcd
        o3d.io.write_point_cloud(str(output_ply_path), final_pcd, write_ascii=False)

    print(f"      [✓] Hoàn tất xuất file: {output_ply_path.name} ({valid_box_count}/{num_objects} boxes)")
    return True


def run_batch_visualization(
    json_dir: Union[str, Path],
    gt_dir: Optional[Union[str, Path]],
    output_dir: Union[str, Path],
    target_scene: Optional[str] = None,
    box_mode: str = "arbitrary_obb",
    line_radius: float = 0.003,
) -> None:
    """Thực thi visualization hàng loạt cho các scene Replica."""
    json_dir = Path(json_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if target_scene and str(target_scene).lower() not in ("all", "benchmark", "default"):
        if str(target_scene).lower() == "full":
            scenes_to_process = DEFAULT_BENCHMARK_SCENES
        elif "," in str(target_scene):
            scenes_to_process = [s.strip() for s in str(target_scene).split(",")]
        else:
            scenes_to_process = [target_scene]
    else:
        scenes_to_process = DEFAULT_BENCHMARK_SCENES

    print(f"[*] Input JSON Directory: {json_dir}")
    print(f"[*] Output Directory:     {output_dir}")
    print(f"[*] Box Visualization:    {box_mode}")
    print(f"[*] Target Scene(s):      {scenes_to_process}")
    print("=" * 70)

    success_count = 0
    skipped_count = 0

    for idx, scene_name in enumerate(scenes_to_process, 1):
        norm_name = normalize_scene_name(scene_name)
        print(f"[{idx}/{len(scenes_to_process)}] Visualizing Scene: {scene_name}")

        json_file = resolve_json_file(json_dir, scene_name)
        if json_file is None:
            print(f"    [-] Bỏ qua: Không tìm thấy file JSON cho Scene '{scene_name}' trong {json_dir}")
            skipped_count += 1
            print("-" * 70)
            continue

        gt_mesh_file = resolve_gt_mesh_file(gt_dir, scene_name)
        if gt_mesh_file is None:
            print(f"    [-] Bỏ qua: Không tìm thấy file GT Mesh cho Scene '{scene_name}'")
            skipped_count += 1
            print("-" * 70)
            continue

        out_ply_path = output_dir / f"visualization_roi_combined_{norm_name}.ply"

        try:
            ok = export_single_scene_visualization(
                json_path=json_file,
                gt_geometry_path=gt_mesh_file,
                output_ply_path=out_ply_path,
                scene_name=scene_name,
                box_mode=box_mode,
                line_radius=line_radius,
            )
            if ok:
                success_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            print(f"    [!] Error visualizing {scene_name}: {e}")
            skipped_count += 1

        print("-" * 70)

    print(
        f"[✓] Finished Visualization! Successfully processed: {success_count} scene(s) | "
        f"Skipped/Failed: {skipped_count} scene(s)."
    )


def resolve_io_paths_from_configs(
    config_file: Optional[Union[str, Path]] = None,
    custom_json_dir: Optional[Union[str, Path]] = None,
    custom_gt_dir: Optional[Union[str, Path]] = None,
    custom_output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Optional[Path], Path]:
    """Tự động resolve đường dẫn tương đối từ configs."""
    # 1. Output Dir (Mặc định: simulation/result_simulation)
    default_output_dir = REPO_ROOT / "simulation" / "result_simulation"
    resolved_output_dir = default_output_dir

    # 2. JSON Dir (Mặc định: outputs/Replica/bounding_box_3D)
    default_json_dir = REPO_ROOT / "outputs" / "Replica" / "bounding_box_3D"
    resolved_json_dir = default_json_dir

    # 3. GT Dir
    resolved_gt_dir = None

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

    if custom_gt_dir:
        p = Path(custom_gt_dir)
        resolved_gt_dir = p if p.is_absolute() else REPO_ROOT / p

    if custom_output_dir:
        p = Path(custom_output_dir)
        resolved_output_dir = p if p.is_absolute() else REPO_ROOT / p

    return resolved_json_dir, resolved_gt_dir, resolved_output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize 3D ROI Bounding Boxes on Replica Ground Truth Mesh."
    )
    parser.add_argument(
        "--config",
        default="configs/simulation.toml",
        help="Đường dẫn file config TOML (mặc định: configs/simulation.toml).",
    )
    parser.add_argument(
        "--scene",
        default=None,
        help="Scene cụ thể cần trực quan hóa (vd: office0, room0, office0,office1, hoặc 'all'; mặc định: all).",
    )
    parser.add_argument(
        "--json-dir",
        default=None,
        help="Thư mục chứa các file JSON 3D box (mặc định: outputs/Replica/bounding_box_3D).",
    )
    parser.add_argument(
        "--gt-dir",
        default=None,
        help="Thư mục chứa các file Ground Truth mesh .ply (mặc định tự động resolve qua Dataset Configs).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Thư mục lưu file PLY kết quả (mặc định: simulation/result_simulation).",
    )
    parser.add_argument(
        "--box-mode",
        default="arbitrary_obb",
        choices=["arbitrary_obb", "gravity_aligned_obb", "aabb"],
        help="Chế độ hiển thị box: arbitrary_obb, gravity_aligned_obb, aabb (mặc định: arbitrary_obb).",
    )
    parser.add_argument(
        "--box-line-radius",
        type=float,
        default=0.003,
        help="Bán kính cạnh hình trụ của bounding box (đơn vị: mét, mặc định: 0.003).",
    )

    args = parser.parse_args()

    json_dir, gt_dir, output_dir = resolve_io_paths_from_configs(
        config_file=args.config,
        custom_json_dir=args.json_dir,
        custom_gt_dir=args.gt_dir,
        custom_output_dir=args.output_dir,
    )

    target_scene = args.scene
    if target_scene is None and args.config:
        cfg_path = REPO_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
        if cfg_path.exists() and load_toml_config:
            cfg = load_toml_config(cfg_path)
            ds_cfg = cfg.get("dataset", {})
            if isinstance(ds_cfg, dict) and ds_cfg.get("name") == "replica":
                target_scene = ds_cfg.get("scene")

    run_batch_visualization(
        json_dir=json_dir,
        gt_dir=gt_dir,
        output_dir=output_dir,
        target_scene=target_scene or "all",
        box_mode=args.box_mode,
        line_radius=args.box_line_radius,
    )


if __name__ == "__main__":
    main()