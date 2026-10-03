"""Module 1 Perception: Generate 3D Instance Segmentation Mesh Masks for Replica Dataset.

Tự động trích xuất 3D Mesh Masks (Vertices, Faces, Edges) từ semantic mesh
và metadata của Replica Dataset và lưu ra định dạng JSON.
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

# Tự động tính REPO_ROOT an toàn dù script nằm ở bất kỳ cấp nào
current_file = Path(__file__).resolve()
REPO_ROOT = current_file.parents[1]
for parent in current_file.parents:
    if (parent / "configs").exists() or (parent / "src").exists() or (parent / "outputs").exists():
        REPO_ROOT = parent
        break

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.common.config_loader import load_toml_config
    from src.common.dataset_config import load_dataset_paths
except ImportError:
    load_toml_config = None
    load_dataset_paths = None

# Đường dẫn mặc định tuyệt đối theo hệ thống của bạn
DEFAULT_OUTPUT_DIR = Path("/media/ml4u/Extreme SSD/Safe-GS/outputs/Replica/mask_3D")
DEFAULT_RAW_DATASET_DIR = Path("/media/ml4u/Extreme SSD/Replica/Replica-Dataset")

DEFAULT_BENCHMARK_SCENES = [
    "office0", "office1", "office2", "office3", 
    "office4", "room0", "room1", "room2"
]


def normalize_scene_name(scene_name: str) -> str:
    """Chuẩn hóa tên scene: 'office_0' -> 'office0', 'room_1' -> 'room1'."""
    return re.sub(r"_(\d+)$", r"\1", scene_name.strip())


def extract_unique_edges_from_faces(faces: np.ndarray) -> List[List[int]]:
    """Trích xuất tập hợp các cạnh không trùng lặp (Edges) từ danh sách mặt tam giác (Faces)."""
    if len(faces) == 0:
        return []
    
    # 3 cạnh của từng tam giác: (v0, v1), (v1, v2), (v2, v0)
    e1 = faces[:, [0, 1]]
    e2 = faces[:, [1, 2]]
    e3 = faces[:, [2, 0]]
    
    all_edges = np.vstack([e1, e2, e3])
    # Sắp xếp min-max trên từng cạnh để loại trừ trùng hướng: (v1, v2) tương đương (v2, v1)
    all_edges = np.sort(all_edges, axis=1)
    unique_edges = np.unique(all_edges, axis=0)
    return unique_edges.tolist()


def extract_replica_3d_masks(
    ply_path: Union[str, Path],
    json_path: Union[str, Path],
    output_path: Union[str, Path],
    min_points: int = 30,
) -> List[dict]:
    """Trích xuất 3D mesh mask (Vertices, Faces, Edges) dưới dạng Dict key-value theo index."""
    ply_path = Path(ply_path)
    json_path = Path(json_path)
    output_path = Path(output_path)

    print(f"[*] Loading metadata from: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        info_semantic = json.load(f)

    # 1. Map ID -> Instance metadata
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

    # 2. Đọc Mesh PLY gốc
    print(f"[*] Reading PLY mesh geometry: {ply_path}")
    plydata = PlyData.read(str(ply_path))
    vertex_data = plydata["vertex"]
    face_data = plydata["face"]

    x = np.asarray(vertex_data["x"])
    y = np.asarray(vertex_data["y"])
    z = np.asarray(vertex_data["z"])
    coords = np.stack([x, y, z], axis=1)

    raw_faces = face_data["vertex_indices"]
    faces_all = np.vstack(raw_faces).astype(np.int64)

    # 3. Lấy object_id cho từng vertex
    id_prop_name = None
    for cand in ["object_id", "objectId", "label", "category"]:
        if cand in vertex_data.data.dtype.names:
            id_prop_name = cand
            break

    if id_prop_name is not None:
        vertex_object_ids = np.asarray(vertex_data[id_prop_name], dtype=np.int32)
    else:
        face_obj_ids = np.asarray(face_data["object_id"], dtype=np.int32)
        vertex_object_ids = np.full(coords.shape[0], fill_value=-1, dtype=np.int32)
        for f_idx, v_list in enumerate(faces_all):
            vertex_object_ids[v_list] = face_obj_ids[f_idx]

    unique_ids = np.unique(vertex_object_ids)
    print(f"[*] Found {len(unique_ids)} instances. Extracting Vertices, Faces, and Edges...")

    results = []
    total_vertices_num = coords.shape[0]

    for uid in unique_ids:
        if uid < 0:
            continue

        global_v_indices = np.where(vertex_object_ids == uid)[0]
        if len(global_v_indices) < min_points:
            continue

        instance_pts = coords[global_v_indices]

        # Khử outlier (loại 2% điểm xa median)
        centroid_temp = np.median(instance_pts, axis=0)
        dist = np.linalg.norm(instance_pts - centroid_temp, axis=1)
        inlier_mask = dist < np.percentile(dist, 98)

        valid_global_v_indices = global_v_indices[inlier_mask]
        if len(valid_global_v_indices) < min_points:
            valid_global_v_indices = global_v_indices

        clean_pts = coords[valid_global_v_indices]

        # 4. Ánh xạ Global Index sang Local Index an toàn bằng NumPy
        global_to_local_map = np.full(total_vertices_num, fill_value=-1, dtype=np.int64)
        global_to_local_map[valid_global_v_indices] = np.arange(len(valid_global_v_indices), dtype=np.int64)

        # 5. Lọc các mặt tam giác (cả 3 đỉnh phải cùng thuộc instance)
        f0 = faces_all[:, 0]
        f1 = faces_all[:, 1]
        f2 = faces_all[:, 2]

        loc_f0 = global_to_local_map[f0]
        loc_f1 = global_to_local_map[f1]
        loc_f2 = global_to_local_map[f2]

        valid_face_mask = (loc_f0 >= 0) & (loc_f1 >= 0) & (loc_f2 >= 0)

        if np.any(valid_face_mask):
            local_faces = np.stack([
                loc_f0[valid_face_mask],
                loc_f1[valid_face_mask],
                loc_f2[valid_face_mask]
            ], axis=1)
        else:
            local_faces = np.empty((0, 3), dtype=np.int64)

        # 6. Trích xuất Edges duy nhất
        local_edges = extract_unique_edges_from_faces(local_faces)

        # 7. Định dạng Dict key-value theo index
        vertices_dict = {str(i): pt.tolist() for i, pt in enumerate(np.round(clean_pts, 4))}
        edges_dict = {str(i): edge for i, edge in enumerate(local_edges)}
        faces_dict = {str(i): face.tolist() for i, face in enumerate(local_faces)}

        meta = id_to_meta.get(uid, {"instance_name": f"instance_{uid}", "category": "unknown"})
        
        record = {
            "instance_id": int(uid),
            "name": meta["instance_name"],
            "category": meta["category"],
            "num_vertices": int(len(clean_pts)),
            "num_faces": int(len(local_faces)),
            "num_edges": int(len(local_edges)),
            "centroid": clean_pts.mean(axis=0).tolist(),
            "vertices": vertices_dict,
            "edges": edges_dict,
            "faces": faces_dict,
        }
        results.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[✓] Exported {len(results)} meshes to: {output_path}\n")
    return results


def resolve_input_directory(
    config_file: Optional[Union[str, Path]] = None,
    dataset_name: str = "replica",
    custom_input_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Tự động tìm kiếm thư mục raw Replica dataset."""
    if custom_input_dir:
        return Path(custom_input_dir)

    # Ưu tiên trực tiếp đường dẫn thực tế trên ổ đĩa
    if DEFAULT_RAW_DATASET_DIR.exists():
        return DEFAULT_RAW_DATASET_DIR

    resolved_input_dir = None
    if load_dataset_paths:
        try:
            paths = load_dataset_paths(dataset_name)
            if "raw_dataset_dir" in paths and Path(paths["raw_dataset_dir"]).exists():
                resolved_input_dir = Path(paths["raw_dataset_dir"])
        except Exception as e:
            print(f"[!] Warning reading dataset config '{dataset_name}': {e}")

    if resolved_input_dir is None:
        fallbacks = [
            REPO_ROOT.parent / "Replica" / "Replica-Dataset",
            REPO_ROOT / "Data" / "Replica",
            Path("/media/ml4u/Extreme SSD/Replica/Replica-Dataset"),
            Path("/media/ml4u/Extreme SSD/Safe-GS/Data/Replica"),
        ]
        for fb in fallbacks:
            if fb.exists():
                resolved_input_dir = fb
                break

    return resolved_input_dir or DEFAULT_RAW_DATASET_DIR


def process_all_replica_scenes(
    dataset_root: Union[str, Path],
    output_dir: Union[str, Path],
    target_scene: Optional[str] = None,
    min_points: int = 30,
) -> None:
    """Duyệt qua các scene và xuất file 3D mask JSON."""
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Root dataset directory does not exist: {dataset_root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    all_scenes = sorted([
        d.name for d in dataset_root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    if target_scene is not None and str(target_scene).lower() not in ("all", "benchmark", "default"):
        if str(target_scene).lower() == "full":
            scenes_to_process = all_scenes
        elif "," in str(target_scene):
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
        
        # Tìm file mesh
        ply_cands = [
            scene_dir / "habitat" / "mesh_semantic.ply",
            scene_dir / "mesh_semantic.ply",
            scene_dir / "mesh.ply",
        ]
        ply_path = next((p for p in ply_cands if p.is_file()), None)

        # Tìm file metadata json
        json_cands = [
            scene_dir / "habitat" / "info_semantic.json",
            scene_dir / "info_semantic.json",
        ]
        json_path = next((p for p in json_cands if p.is_file()), None)

        norm_name = normalize_scene_name(scene_name)
        sub_dir = output_dir / norm_name
        sub_dir.mkdir(parents=True, exist_ok=True)
        scene_out_path = sub_dir / f"{scene_name}_3d_masks.json"

        print(f"[{idx}/{len(scenes_to_process)}] Processing Scene: {scene_name}")

        if ply_path is None or json_path is None:
            print(f"[-] Warning: Missing PLY or JSON in '{scene_name}'. Skipping.")
            skipped_count += 1
            print("-" * 70)
            continue

        try:
            extract_replica_3d_masks(
                ply_path=ply_path,
                json_path=json_path,
                output_path=scene_out_path,
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
        description="Extract 3D Instance Segmentation Mesh Masks (Vertices, Faces, Edges) for Replica scenes."
    )
    parser.add_argument(
        "--config",
        default="configs/object_roi.toml",
        help="Đường dẫn file config TOML.",
    )
    parser.add_argument(
        "--dataset-config",
        default="replica",
        help="Tên dataset config trong Dataset Configs/ (mặc định: replica).",
    )
    parser.add_argument(
        "--scene",
        default=None,
        help="Scene cụ thể cần chạy (vd: office0, room0, hoặc 'all').",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Ghi đè đường dẫn thư mục raw dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Đường dẫn output (Mặc định: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=30,
        help="Số đỉnh tối thiểu của một vật thể (mặc định: 30).",
    )

    args = parser.parse_args()

    input_dir = resolve_input_directory(
        config_file=args.config,
        dataset_name=args.dataset_config,
        custom_input_dir=args.input_dir,
    )
    output_dir = Path(args.output_dir)

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
        min_points=args.min_points,
    )


if __name__ == "__main__":
    main()