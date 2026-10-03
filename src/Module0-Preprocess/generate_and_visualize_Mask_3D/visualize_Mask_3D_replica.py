"""Module Simulation: Visualize 3D Instance Segmentation Mesh on Replica Dataset.

Đọc trực tiếp Vertices và Faces từ file JSON 3D Mask kết hợp với Mesh gốc,
tạo file PLY Triangle Mesh hoàn chỉnh với màu sắc riêng biệt cho từng instance.
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

# Tự động tính REPO_ROOT
current_file = Path(__file__).resolve()
REPO_ROOT = current_file.parents[1]
for parent in current_file.parents:
    if (parent / "configs").exists() or (parent / "src").exists() or (parent / "outputs").exists():
        REPO_ROOT = parent
        break

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Đường dẫn mặc định tuyệt đối theo hệ thống của bạn
DEFAULT_MASK_DIR = Path("/media/ml4u/Extreme SSD/Safe-GS/outputs/Replica/mask_3D")
DEFAULT_GT_DIR = Path("/media/ml4u/Extreme SSD/Replica/Replica-Dataset")

DEFAULT_BENCHMARK_SCENES = [
    "office0", "office1", "office2", "office3", 
    "office4", "room0", "room1", "room2"
]


def normalize_scene_name(scene_name: str) -> str:
    """Chuẩn hóa tên scene: 'office_0' -> 'office0', 'room_1' -> 'room1'."""
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


def resolve_mesh_file(
    gt_dir: Optional[Union[str, Path]], 
    scene_name: str
) -> Optional[Path]:
    """Tìm file Mesh gốc của Replica để làm nền."""
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
            cands = [
                root / a / "mesh.ply",
                root / a / "habitat" / "mesh_semantic.ply",
                root / a / "mesh_semantic.ply",
                root / a / f"{a}_mesh.ply",
                root / f"{a}_mesh.ply",
            ]
            for c in cands:
                if c.is_file():
                    return c

    return None


def parse_dict_or_list_elements(data: Union[Dict, List]) -> np.ndarray:
    """Hỗ trợ nạp mảng từ cả định dạng Dictionary {'0': [...], ...} và List [[...], ...]."""
    if isinstance(data, dict):
        sorted_keys = sorted(data.keys(), key=lambda k: int(k) if k.isdigit() else k)
        return np.array([data[k] for k in sorted_keys])
    return np.array(data)


def triangulate_faces(raw_faces: Union[np.ndarray, list]) -> np.ndarray:
    """Chuẩn hóa mọi kiểu mặt (tam giác, tứ giác, đa giác) về ma trận tam giác (M, 3)."""
    triangles = []
    for face in raw_faces:
        f = list(face)
        if len(f) == 3:
            triangles.append(f)
        elif len(f) == 4:
            # Tách tứ giác thành 2 tam giác
            triangles.append([f[0], f[1], f[2]])
            triangles.append([f[0], f[2], f[3]])
        elif len(f) > 4:
            # Fan triangulation cho đa giác nhiều đỉnh
            for i in range(1, len(f) - 1):
                triangles.append([f[0], f[i], f[i + 1]])
    
    if len(triangles) == 0:
        return np.empty((0, 3), dtype=np.int64)
    return np.asarray(triangles, dtype=np.int64)


def export_colored_mask_mesh(
    json_path: Path,
    bg_mesh_path: Path,
    output_ply_path: Path,
    include_background: bool = True,
) -> bool:
    """Đọc Vertices và Faces từ file JSON, ghép với mesh nền và xuất ra Triangle Mesh PLY."""
    if not json_path.is_file() or not bg_mesh_path.is_file():
        return False

    print(f"      * Đọc JSON Mask: {json_path.name}")
    print(f"      * Đọc Mesh gốc:  {bg_mesh_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        instances = json.load(f)

    if not isinstance(instances, list):
        print(f"[!] Warning: File JSON không chứa danh sách instances: {json_path}")
        return False

    all_vertices_coords = []
    all_vertices_colors = []
    all_faces_indices = []
    current_vertex_offset = 0

    # 1. Đọc Mesh gốc làm nền phòng (Background) tô màu xám nhạt
    if include_background:
        try:
            plydata_bg = PlyData.read(str(bg_mesh_path))
            v_bg = plydata_bg["vertex"]
            f_bg = plydata_bg["face"]

            bg_coords = np.stack([v_bg["x"], v_bg["y"], v_bg["z"]], axis=1).astype(np.float32)
            num_bg_vertices = bg_coords.shape[0]

            # Tô màu xám nhạt cho nền phòng
            bg_colors = np.full((num_bg_vertices, 3), fill_value=210, dtype=np.uint8)
            
            # Chuẩn hóa mặt nền về (N, 3) tránh lỗi tứ giác size 4
            raw_bg_faces = f_bg["vertex_indices"]
            bg_faces = triangulate_faces(raw_bg_faces)

            all_vertices_coords.append(bg_coords)
            all_vertices_colors.append(bg_colors)
            all_faces_indices.append(bg_faces)
            current_vertex_offset += num_bg_vertices
        except Exception as e:
            print(f"[-] Không thể đọc mesh nền: {e}. Tiếp tục chỉ xuất mask instance.")

    # 2. Đọc Vertices và Faces của từng instance từ JSON và gán màu riêng biệt
    palette = generate_distinct_colors(len(instances))
    valid_instance_count = 0

    for idx, inst in enumerate(instances):
        v_raw = inst.get("vertices")
        f_raw = inst.get("faces")

        if not v_raw:
            continue

        pts = parse_dict_or_list_elements(v_raw).astype(np.float32)
        if len(pts) == 0:
            continue

        # Lấy faces từ JSON và chuẩn hóa về (M, 3)
        if f_raw:
            raw_faces_inst = parse_dict_or_list_elements(f_raw)
            faces = triangulate_faces(raw_faces_inst)
        else:
            faces = np.empty((0, 3), dtype=np.int64)

        color = palette[idx]
        num_v = pts.shape[0]
        inst_colors = np.tile(color, (num_v, 1)).astype(np.uint8)

        # Offset lại index của Faces theo tổng số vertex hiện tại
        if len(faces) > 0:
            offset_faces = faces + current_vertex_offset
            all_faces_indices.append(offset_faces)

        all_vertices_coords.append(pts)
        all_vertices_colors.append(inst_colors)
        current_vertex_offset += num_v
        valid_instance_count += 1

    if not all_vertices_coords:
        print(f"[!] Warning: Không có đỉnh nào được trích xuất cho scene: {json_path.name}")
        return False

    # 3. Gộp toàn bộ mảng Vertices và Faces
    merged_coords = np.vstack(all_vertices_coords).astype(np.float32)
    merged_colors = np.vstack(all_vertices_colors).astype(np.uint8)

    if all_faces_indices:
        merged_faces = np.vstack(all_faces_indices).astype(np.int32)
    else:
        merged_faces = np.empty((0, 3), dtype=np.int32)

    total_vertices = merged_coords.shape[0]
    total_faces = merged_faces.shape[0]

    # 4. Tạo cấu trúc PlyData
    vertex_dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1")
    ]
    new_vertices = np.empty(total_vertices, dtype=vertex_dtype)
    new_vertices["x"] = merged_coords[:, 0]
    new_vertices["y"] = merged_coords[:, 1]
    new_vertices["z"] = merged_coords[:, 2]
    new_vertices["red"] = merged_colors[:, 0]
    new_vertices["green"] = merged_colors[:, 1]
    new_vertices["blue"] = merged_colors[:, 2]
    new_vertex_element = PlyElement.describe(new_vertices, "vertex")

    face_dtype = [("vertex_indices", "i4", (3,))]
    new_faces = np.empty(total_faces, dtype=face_dtype)
    new_faces["vertex_indices"] = merged_faces
    new_face_element = PlyElement.describe(new_faces, "face")

    # 5. Ghi ra file PLY
    output_ply_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([new_vertex_element, new_face_element], text=False).write(str(output_ply_path))

    print(
        f"      [✓] Hoàn tất: {output_ply_path.name} "
        f"({valid_instance_count}/{len(instances)} instances | "
        f"Vertices: {total_vertices:,} | Faces: {total_faces:,})"
    )
    return True


def run_batch_mask_visualization(
    mask_dir: Union[str, Path],
    gt_dir: Optional[Union[str, Path]],
    target_scene: Optional[str] = None,
    no_background: bool = False,
) -> None:
    """Quét các scene và xuất file PLY Mesh kết hợp."""
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
        print(f"[{idx}/{len(scenes_to_process)}] Visualizing Mesh Scene: {scene_name}")

        json_file = resolve_mask_json_file(mask_dir, scene_name)
        if json_file is None:
            print(f"    [-] Bỏ qua: Không tìm thấy file JSON mask cho Scene '{scene_name}'")
            skipped_count += 1
            print("-" * 70)
            continue

        gt_mesh_file = resolve_mesh_file(gt_dir, scene_name)
        if gt_mesh_file is None:
            print(f"    [-] Bỏ qua: Không tìm thấy file GT Mesh gốc cho Scene '{scene_name}'")
            skipped_count += 1
            print("-" * 70)
            continue

        out_ply_path = json_file.parent / f"visualization_3d_mask_{norm_name}.ply"

        try:
            ok = export_colored_mask_mesh(
                json_path=json_file,
                bg_mesh_path=gt_mesh_file,
                output_ply_path=out_ply_path,
                include_background=not no_background,
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
        description="Xuất Triangle Mesh 3D đọc trực tiếp Vertices và Faces từ file JSON mask kết hợp Mesh gốc."
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
    parser.add_argument(
        "--no-background",
        action="store_true",
        help="Nếu bật cờ này, file xuất ra chỉ chứa mesh các vật thể trong JSON mà không ghép mesh phòng nền.",
    )

    args = parser.parse_args()

    run_batch_mask_visualization(
        mask_dir=args.mask_dir,
        gt_dir=args.gt_dir,
        target_scene=args.scene,
        no_background=args.no_background,
    )


if __name__ == "__main__":
    main()