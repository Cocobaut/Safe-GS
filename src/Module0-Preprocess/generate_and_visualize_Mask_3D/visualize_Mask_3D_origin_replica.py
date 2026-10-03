"""Module Simulation: Visualize 3D Instance Segmentation Mesh on Replica Dataset.

Đọc trực tiếp cấu trúc hình học (Vertices + Faces) của Mesh gốc Replica,
tô màu theo từng instance trong file JSON 3D Mask và xuất ra file PLY Triangle Mesh hoàn chỉnh.
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
from plyfile import PlyData, PlyElement

# Tự động tính REPO_ROOT an toàn
current_file = Path(__file__).resolve()
REPO_ROOT = current_file.parents[1]
for parent in current_file.parents:
    if (parent / "configs").exists() or (parent / "src").exists() or (parent / "outputs").exists():
        REPO_ROOT = parent
        break

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Đường dẫn mặc định
DEFAULT_MASK_DIR = Path("/media/ml4u/Extreme SSD/Safe-GS/outputs/Replica/mask_3D")
DEFAULT_GT_DIR = Path("/media/ml4u/Extreme SSD/Replica/Replica-Dataset")

DEFAULT_BENCHMARK_SCENES = [
    "office0", "office1", "office2", "office3", 
    "office4", "room0", "room1", "room2"
]


def normalize_scene_name(scene_name: str) -> str:
    """Chuẩn hóa tên scene: 'office_0' -> 'office0'."""
    return re.sub(r"_(\d+)$", r"\1", scene_name.strip())


def get_scene_aliases(scene_name: str) -> List[str]:
    """Sinh ra các biến thể tên gọi: office0 và office_0."""
    norm = normalize_scene_name(scene_name)
    alt = re.sub(r"(\D+)(\d+)$", r"\1_\2", norm)
    return list(dict.fromkeys([norm, alt, scene_name]))


def generate_distinct_colors(num_colors: int) -> List[Tuple[int, int, int]]:
    """Sinh danh sách N màu sắc RGB (0-255) tách biệt hoàn toàn trên HSV."""
    colors = []
    golden_ratio_conjugate = 0.618033988749895
    h = 0.1
    for _ in range(num_colors):
        h = (h + golden_ratio_conjugate) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def resolve_mask_json_file(mask_dir: Path, scene_name: str) -> Optional[Path]:
    """Tìm file JSON mask chính xác theo cấu trúc thư mục scene."""
    aliases = get_scene_aliases(scene_name)
    
    for folder_name in aliases:
        folder_path = mask_dir / folder_name
        if folder_path.is_dir():
            for file_alias in aliases:
                cands = [
                    folder_path / f"{file_alias}_3d_masks.json",
                    folder_path / f"{file_alias}_masks.json",
                    folder_path / f"{file_alias}.json",
                ]
                for c in cands:
                    if c.is_file():
                        return c
            
            any_jsons = sorted(list(folder_path.glob("*.json")))
            if any_jsons:
                return any_jsons[0]

    for file_alias in aliases:
        cand = mask_dir / f"{file_alias}_3d_masks.json"
        if cand.is_file():
            return cand

    return None


def resolve_semantic_mesh_file(
    gt_dir: Optional[Union[str, Path]], 
    scene_name: str
) -> Optional[Path]:
    """Tìm file Mesh gốc có chứa thông tin semantic/object ID của Replica."""
    aliases = get_scene_aliases(scene_name)

    search_roots = []
    if gt_dir:
        search_roots.append(Path(gt_dir))
    
    search_roots.extend([
        DEFAULT_GT_DIR,
        Path("/media/ml4u/Extreme SSD/Replica"),
        REPO_ROOT.parent / "Replica" / "Replica-Dataset",
        REPO_ROOT / "outputs" / "Replica" / "mesh",
        Path("/media/ml4u/Extreme SSD/Safe-GS/outputs/Replica/mesh"),
        REPO_ROOT / "Data" / "Replica",
        Path("/media/ml4u/Extreme SSD/Safe-GS/Data/Replica"),
    ])

    for root in search_roots:
        if not root.exists():
            continue
        for a in aliases:
            # Ưu tiên các file có chứa object_id semantic
            cands = [
                root / a / "habitat" / "mesh_semantic.ply",
                root / a / "mesh_semantic.ply",
                root / a / "mesh.ply",
                root / a / f"{a}_mesh.ply",
                root / f"{a}_mesh.ply",
            ]
            for c in cands:
                if c.is_file():
                    return c

    return None


def export_colored_instance_mesh(
    json_path: Path,
    semantic_mesh_path: Path,
    output_ply_path: Path,
) -> bool:
    """Đọc mesh gốc, gán màu theo từng instance trong file JSON và xuất ra Triangle Mesh."""
    if not json_path.is_file() or not semantic_mesh_path.is_file():
        return False

    print(f"      * Đọc JSON Mask:     {json_path.name}")
    print(f"      * Đọc Semantic Mesh: {semantic_mesh_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        instances = json.load(f)

    if not isinstance(instances, list):
        print(f"[!] Warning: File JSON không đúng định dạng danh sách: {json_path}")
        return False

    # Đọc cấu trúc mesh bằng plyfile để giữ nguyên toàn bộ vertex & face
    plydata = PlyData.read(str(semantic_mesh_path))
    vertex_element = plydata["vertex"]
    face_element = plydata["face"]

    num_vertices = len(vertex_element.data)
    num_faces = len(face_element.data)

    # 1. Xác định object_id của từng vertex
    id_prop_name = None
    for cand in ["object_id", "objectId", "label", "category"]:
        if cand in vertex_element.data.dtype.names:
            id_prop_name = cand
            break

    if id_prop_name is not None:
        vertex_object_ids = np.asarray(vertex_element[id_prop_name], dtype=np.int32)
    else:
        # Nếu ID nằm ở Face, ánh xạ sang Vertex
        face_obj_ids = np.asarray(face_element["object_id"], dtype=np.int32)
        face_indices = face_element["vertex_indices"]
        vertex_object_ids = np.full(num_vertices, fill_value=-1, dtype=np.int32)
        for f_idx, v_list in enumerate(face_indices):
            vertex_object_ids[v_list] = face_obj_ids[f_idx]

    # 2. Khởi tạo mảng màu cho toàn bộ đỉnh (Mặc định: màu xám nền [210, 210, 210])
    vertex_colors = np.full((num_vertices, 3), fill_value=210, dtype=np.uint8)

    palette = generate_distinct_colors(len(instances))
    colored_instances_count = 0

    for idx, inst in enumerate(instances):
        inst_id = inst.get("instance_id")
        if inst_id is None or inst_id < 0:
            continue

        mask = (vertex_object_ids == inst_id)
        if np.any(mask):
            color = palette[idx]
            vertex_colors[mask] = color
            colored_instances_count += 1

    # 3. Tạo cấu trúc Vertex Element mới chứa trường màu rgb
    x = vertex_element["x"]
    y = vertex_element["y"]
    z = vertex_element["z"]
    red = vertex_colors[:, 0]
    green = vertex_colors[:, 1]
    blue = vertex_colors[:, 2]

    # Tạo structured array cho vertex mới
    vertex_dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1")
    ]
    new_vertices = np.empty(num_vertices, dtype=vertex_dtype)
    new_vertices["x"] = x
    new_vertices["y"] = y
    new_vertices["z"] = z
    new_vertices["red"] = red
    new_vertices["green"] = green
    new_vertices["blue"] = blue

    new_vertex_element = PlyElement.describe(new_vertices, "vertex")

    # 4. Ghi trực tiếp ra file PLY dạng Triangle Mesh nhị phân
    output_ply_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([new_vertex_element, face_element], text=False).write(str(output_ply_path))

    print(
        f"      [✓] Hoàn tất: {output_ply_path.name} "
        f"({colored_instances_count}/{len(instances)} instances | "
        f"Vertices: {num_vertices:,} | Faces: {num_faces:,})"
    )
    return True


def run_batch_mask_visualization(
    mask_dir: Union[str, Path],
    gt_dir: Optional[Union[str, Path]],
    target_scene: Optional[str] = None,
) -> None:
    """Quét các scene và xuất file PLY Mesh có màu sắc instance."""
    mask_dir = Path(mask_dir)
    if not mask_dir.exists():
        raise FileNotFoundError(f"Thư mục mask_3D không tồn tại: {mask_dir}")

    existing_subdirs = sorted([d.name for d in mask_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])

    if target_scene and str(target_scene).lower() not in ("all", "benchmark", "default"):
        if "," in str(target_scene):
            scenes_to_process = [s.strip() for s in str(target_scene).split(",")]
        else:
            scenes_to_process = [target_scene]
    else:
        scenes_to_process = existing_subdirs if existing_subdirs else DEFAULT_BENCHMARK_SCENES

    print(f"[*] Root Mask & Output Directory: {mask_dir}")
    print(f"[*] Root Mesh Directory:          {gt_dir or DEFAULT_GT_DIR}")
    print(f"[*] Target scene(s):              {scenes_to_process}")
    print("=" * 70)

    success_count = 0
    skipped_count = 0

    for idx, scene_name in enumerate(scenes_to_process, 1):
        norm_name = normalize_scene_name(scene_name)
        print(f"[{idx}/{len(scenes_to_process)}] Processing Mesh Scene: {scene_name}")

        json_file = resolve_mask_json_file(mask_dir, scene_name)
        if json_file is None:
            print(f"    [-] Bỏ qua: Không tìm thấy file JSON mask cho Scene '{scene_name}'")
            skipped_count += 1
            print("-" * 70)
            continue

        gt_mesh_file = resolve_semantic_mesh_file(gt_dir, scene_name)
        if gt_mesh_file is None:
            print(f"    [-] Bỏ qua: Không tìm thấy file GT Mesh gốc cho Scene '{scene_name}'")
            skipped_count += 1
            print("-" * 70)
            continue

        # Thêm prefix "origin" vào tên file kết quả
        out_ply_path = json_file.parent / f"visualization_3d_mask_origin_{norm_name}.ply"

        try:
            ok = export_colored_instance_mesh(
                json_path=json_file,
                semantic_mesh_path=gt_mesh_file,
                output_ply_path=out_ply_path,
            )
            if ok:
                success_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            print(f"    [!] Lỗi khi xuất mesh {scene_name}: {e}")
            skipped_count += 1

        print("-" * 70)

    print(
        f"[✓] Xong! Thành công: {success_count} scene(s) | Bỏ qua/Lỗi: {skipped_count} scene(s)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Xuất Triangle Mesh 3D tô màu theo Instance Segmentation từ Replica Mesh gốc."
    )
    parser.add_argument(
        "--mask-dir",
        default=str(DEFAULT_MASK_DIR),
        help=f"Thư mục chứa các file JSON mask (mặc định: {DEFAULT_MASK_DIR}).",
    )
    parser.add_argument(
        "--gt-dir",
        default=str(DEFAULT_GT_DIR),
        help=f"Thư mục chứa mesh gốc Replica (mặc định: {DEFAULT_GT_DIR}).",
    )
    parser.add_argument(
        "--scene",
        default="all",
        help="Scene cụ thể cần trực quan hóa (ví dụ: office0, room0, hoặc 'all').",
    )

    args = parser.parse_args()

    run_batch_mask_visualization(
        mask_dir=args.mask_dir,
        gt_dir=args.gt_dir,
        target_scene=args.scene,
    )


if __name__ == "__main__":
    main()