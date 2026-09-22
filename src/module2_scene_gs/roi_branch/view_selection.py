"""Module 2 Scene-GS / ROI Branch: View Selection for 3D Bounding Boxes.

Thuật toán chọn góc nhìn tập trung vào vật thể (Object-Focused Camera Selection)
dựa trên ROI-GS (Bui et al., 2025).

Nhận đầu vào là file JSON chứa danh sách các bounding boxes 3D (ví dụ:
outputs/Replica/bounding_box_3D_and_view_selection/office0/office_0_3d_boxes.json)
và mô hình SfM COLMAP của scene đó.
Đầu ra là một file JSON được lưu cùng thư mục với file JSON đầu vào, chứa
mapping giữa ID bounding box và danh sách Top K tên ảnh quan sát tốt nhất:
{
    "<instance_id>": ["frame000687.jpg", "frame000688.jpg", ...]
}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pycolmap

# Đảm bảo hiển thị UTF-8 trên Windows Console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Thêm repo root vào sys.path để import an toàn module cấu hình dù script chạy từ bất kỳ thư mục nào
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.common.config_loader import load_toml_config
    from src.common.dataset_config import load_dataset_paths
except ImportError:
    load_toml_config = None
    load_dataset_paths = None


def resolve_path(p: Union[str, Path], base_dir: Path = REPO_ROOT) -> Path:
    """Chuyển đổi đường dẫn tương đối thành Path tuyệt đối chuẩn hóa dựa trên base_dir."""
    path_obj = Path(p)
    if path_obj.is_absolute():
        return path_obj
    return (base_dir / path_obj).resolve()


def get_cam_pose(image: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trích xuất R (3x3), t (3,), và cam_center (3,) từ đối tượng Image của pycolmap.

    Tương thích đa phiên bản pycolmap (Rigid3d, rotation matrix, qvec/tvec).
    """
    # 1. pycolmap bản mới: cam_from_world (Rigid3d)
    if hasattr(image, "cam_from_world"):
        cfw = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
        rot = cfw.rotation() if callable(cfw.rotation) else cfw.rotation
        R = rot.matrix() if hasattr(rot, "matrix") else np.array(rot)
        t = np.array(cfw.translation() if callable(cfw.translation) else cfw.translation).flatten()
        if hasattr(image, "projection_center"):
            center = np.array(image.projection_center())
        else:
            center = -R.T @ t
        return R, t, center

    # 2. pycolmap có image.rotation
    if hasattr(image, "rotation"):
        rot = image.rotation() if callable(image.rotation) else image.rotation
        R = rot.matrix() if hasattr(rot, "matrix") else np.array(rot)
        trans = image.translation() if callable(image.translation) else image.translation
        t = np.array(trans).flatten()
        center = -R.T @ t
        return R, t, center

    # 3. pycolmap bản cũ dùng qvec và tvec chuẩn COLMAP
    if hasattr(image, "qvec") and hasattr(image, "tvec"):
        qvec = image.qvec
        w, x, y, z = qvec[0], qvec[1], qvec[2], qvec[3]
        R = np.array([
            [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2]
        ])
        t = np.array(image.tvec).flatten()
        center = -R.T @ t
        return R, t, center

    raise AttributeError("Không thể đọc pose từ đối tượng Image của pycolmap.")


def get_obb_corners(center: np.ndarray, sizes: np.ndarray, R: np.ndarray) -> np.ndarray:
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


def extract_corners_from_box(item: Dict, bbox_type: str = "arbitrary_3d_obb") -> Tuple[np.ndarray, np.ndarray]:
    """Trích xuất 8 đỉnh (corners) và tâm (center) của hộp bao 3D từ dict JSON.

    Hỗ trợ arbitrary_3d_obb, gravity_aligned_obb và aabb. Tự động fallback sang AABB nếu OBB không tồn tại.
    """
    # 1. Arbitrary 3D OBB
    if bbox_type == "arbitrary_3d_obb" and "arbitrary_3d_obb" in item:
        obb = item["arbitrary_3d_obb"]
        center = np.array(obb["center"], dtype=np.float64)
        sizes = np.array(obb["sizes"], dtype=np.float64)
        R = np.array(obb["rotation_matrix"], dtype=np.float64)
        return get_obb_corners(center, sizes, R), center

    # 2. Gravity-aligned OBB
    if (bbox_type == "gravity_aligned_obb" or "arbitrary_3d_obb" not in item) and "gravity_aligned_obb" in item:
        g_obb = item["gravity_aligned_obb"]
        center = np.array(g_obb["center"], dtype=np.float64)
        sizes = np.array(g_obb["sizes"], dtype=np.float64)
        yaw = float(g_obb.get("yaw_rad", 0.0))
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        R = np.array([
            [cos_y, -sin_y, 0.0],
            [sin_y,  cos_y, 0.0],
            [  0.0,    0.0, 1.0]
        ], dtype=np.float64)
        return get_obb_corners(center, sizes, R), center

    # 3. Axis-Aligned Bounding Box (AABB)
    if "aabb" in item:
        aabb = item["aabb"]
        min_pt = np.array(aabb["min"], dtype=np.float64)
        max_pt = np.array(aabb["max"], dtype=np.float64)
        if "center" in aabb:
            center = np.array(aabb["center"], dtype=np.float64)
        else:
            center = (min_pt + max_pt) / 2.0
        corners = []
        for x in [min_pt[0], max_pt[0]]:
            for y in [min_pt[1], max_pt[1]]:
                for z in [min_pt[2], max_pt[2]]:
                    corners.append([x, y, z])
        return np.array(corners, dtype=np.float64), center

    # Fallback nếu cấu trúc dạng bounds.center / bounds.aabb (chuẩn cũ)
    if "bounds" in item:
        b = item["bounds"]
        center = np.array(b["center"], dtype=np.float64)
        min_pt = np.array(b["aabb"]["min"], dtype=np.float64)
        max_pt = np.array(b["aabb"]["max"], dtype=np.float64)
        corners = []
        for x in [min_pt[0], max_pt[0]]:
            for y in [min_pt[1], max_pt[1]]:
                for z in [min_pt[2], max_pt[2]]:
                    corners.append([x, y, z])
        return np.array(corners, dtype=np.float64), center

    raise ValueError(f"Không thể đọc thông tin bounding box từ item: {item.keys()}")


def extract_scene_name(path: Union[str, Path]) -> str:
    """Trích xuất tên scene từ đường dẫn file hoặc tên file (ví dụ: office0, room1)."""
    p = Path(path)
    # 1. Thử từ tên thư mục cha (ví dụ: .../office0/office_0_3d_boxes.json)
    dir_name = p.parent.name.lower()
    if any(k in dir_name for k in ["office", "room", "scan"]):
        return dir_name

    # 2. Thử regex trên tên file (ví dụ: office_0_3d_boxes.json -> office0)
    stem = p.stem.lower()
    m = re.search(r"(office|room|scan)_?(\d+)", stem)
    if m:
        return f"{m.group(1)}{m.group(2)}"

    return "office0"


class ROIViewSelector:
    """Thuật toán chọn góc nhìn tập trung vào vật thể (Object-Focused Camera Selection)

    dựa trên ROI-GS (Bui et al., 2025). Hỗ trợ nạp trước toàn bộ camera của Reconstruction
    để xử lý hàng loạt nhiều bounding box cực nhanh.
    """

    def __init__(
        self,
        sfm_sparse_dir: Union[str, Path],
        min_projected_area_ratio: float = 0.005,
    ):
        self.sfm_sparse_dir = Path(sfm_sparse_dir)
        self.min_projected_area_ratio = min_projected_area_ratio

        if not self.sfm_sparse_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục SfM COLMAP: {self.sfm_sparse_dir}")

        print(f"[*] Nạp mô hình COLMAP Reconstruction từ: {self.sfm_sparse_dir}")
        self.recon = pycolmap.Reconstruction(str(self.sfm_sparse_dir))
        print(f"    - Tổng số ảnh:       {len(self.recon.images):,} cameras")
        print(f"    - Tổng điểm 3D SfM:  {len(self.recon.points3D):,} points")

        # Cache trước toàn bộ thông số máy ảnh dạng NumPy để tính toán siêu tốc
        self._cache_camera_parameters()

    def _cache_camera_parameters(self) -> None:
        """Trích xuất và lưu mảng NumPy thông số nội tại và ngoại tại của toàn bộ camera."""
        self.cam_names = []
        self.cam_R = []
        self.cam_t = []
        self.cam_centers = []
        self.cam_forwards = []
        self.cam_intrinsics = []  # (fx, fy, cx, cy, width, height)

        for img_id, img in self.recon.images.items():
            if not img.has_pose:
                continue

            R, t, center = get_cam_pose(img)
            cam = self.recon.cameras[img.camera_id]

            # Intrinsics: PINHOLE / OPENCV
            params = cam.params
            if len(params) >= 4:
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            elif len(params) == 3:
                fx, fy, cx, cy = params[0], params[0], params[1], params[2]
            else:
                fx = fy = params[0]
                cx, cy = cam.width / 2.0, cam.height / 2.0

            self.cam_names.append(img.name)
            self.cam_R.append(R)
            self.cam_t.append(t)
            self.cam_centers.append(center)
            self.cam_forwards.append(R[2, :])  # Hướng nhìn chính (optical axis) của camera
            self.cam_intrinsics.append((fx, fy, cx, cy, cam.width, cam.height))

        self.cam_names = np.array(self.cam_names)
        self.cam_R = np.array(self.cam_R)                # (N, 3, 3)
        self.cam_t = np.array(self.cam_t)                # (N, 3)
        self.cam_centers = np.array(self.cam_centers)    # (N, 3)
        self.cam_forwards = np.array(self.cam_forwards)  # (N, 3)

    def select_cameras_for_box(
        self,
        corners: np.ndarray,
        center: np.ndarray,
        top_k: int = 50,
        min_cos_angle: float = 0.05
    ) -> List[str]:
        """Chọn Top K góc nhìn tốt nhất cho một cụm 8 đỉnh bounding box 3D cụ thể."""
        num_cams = len(self.cam_names)
        if num_cams == 0:
            return []

        scored_views: List[Tuple[str, float]] = []

        # Vector từ camera tới tâm vật thể và khoảng cách
        diffs = center - self.cam_centers  # (N, 3)
        distances = np.linalg.norm(diffs, axis=1)  # (N,)
        view_dirs = diffs / (distances[:, None] + 1e-7)  # (N, 3)

        # Góc giữa hướng nhìn camera và hướng tới vật thể (cos theta)
        cos_angles = np.sum(view_dirs * self.cam_forwards, axis=1)  # (N,)

        for i in range(num_cams):
            # 1. Bỏ qua camera quay lưng lại với vật thể
            if cos_angles[i] < min_cos_angle:
                continue

            R = self.cam_R[i]
            t = self.cam_t[i]
            fx, fy, cx, cy, w, h = self.cam_intrinsics[i]

            # 2. Chiếu 8 đỉnh sang Camera Space
            p_c = (R @ corners.T + t[:, None]).T  # (8, 3)

            # Phải có ít nhất 2 đỉnh nằm trước mặt phẳng camera (Z > 0.05)
            in_front = p_c[:, 2] > 0.05
            if np.sum(in_front) < 2:
                continue

            valid_p_c = p_c[in_front]

            # 3. Chiếu sang Pixel Coordinate 2D
            u = (valid_p_c[:, 0] / valid_p_c[:, 2]) * fx + cx
            v = (valid_p_c[:, 1] / valid_p_c[:, 2]) * fy + cy

            box_u_min = max(0.0, float(np.min(u)))
            box_v_min = max(0.0, float(np.min(v)))
            box_u_max = min(float(w), float(np.max(u)))
            box_v_max = min(float(h), float(np.max(v)))

            # 4. Kiểm tra giao cắt với khung nhìn ảnh (Frustum intersection)
            if box_u_max <= box_u_min or box_v_max <= box_v_min:
                continue

            # 5. Tỉ lệ diện tích chiếm trên ảnh
            projected_area = (box_u_max - box_u_min) * (box_v_max - box_v_min)
            area_ratio = projected_area / (w * h)

            if area_ratio < self.min_projected_area_ratio:
                continue

            dist = distances[i]
            cos_a = cos_angles[i]

            # Hàm chấm điểm ưu tiên:
            # - Diện tích vật thể trên ảnh càng lớn càng tốt (area_ratio * 10.0)
            # - Hướng nhìn càng chính diện tâm vật thể càng tốt (1.5 * cos_a)
            # - Camera càng gần vật thể càng tốt (-0.05 * dist)
            score = (area_ratio * 10.0) + (1.5 * cos_a) - (0.05 * dist)

            scored_views.append((self.cam_names[i], float(score)))

        # Sắp xếp theo điểm số giảm dần
        scored_views.sort(key=lambda x: x[1], reverse=True)

        selected_names = [name for name, _ in scored_views[:top_k]]
        return selected_names

    def process_boxes_json(
        self,
        boxes_json_path: Union[str, Path],
        output_name: Optional[str] = None,
        top_k: int = 50,
        bbox_type: str = "arbitrary_3d_obb"
    ) -> Tuple[Path, Dict[str, List[str]]]:
        """Xử lý toàn bộ các bounding boxes trong file JSON đầu vào và lưu file kết quả

        vào CÙNG THƯ MỤC với file JSON đầu vào.
        """
        boxes_json_path = resolve_path(boxes_json_path)
        if not boxes_json_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file JSON bounding boxes: {boxes_json_path}")

        with open(boxes_json_path, "r", encoding="utf-8") as f:
            boxes_data = json.load(f)

        if not isinstance(boxes_data, list):
            raise ValueError(f"Dữ liệu JSON trong {boxes_json_path} phải là một danh sách các bounding boxes.")

        print(f"\n[+] Đang xử lý {len(boxes_data)} bounding boxes từ: {boxes_json_path.name}")
        print(f"    - Loại BBox ưu tiên: {bbox_type}")
        print(f"    - Số góc nhìn Top K: {top_k}")

        views_per_box: Dict[str, List[str]] = {}

        for idx, box in enumerate(boxes_data):
            # Xác định instance_id
            inst_id = str(box.get("instance_id", idx))

            try:
                corners, center = extract_corners_from_box(box, bbox_type=bbox_type)
            except Exception as e:
                print(f"    [!] Bỏ qua box ID {inst_id}: Lỗi trích xuất tọa độ ({e})")
                views_per_box[inst_id] = []
                continue

            selected_cams = self.select_cameras_for_box(corners=corners, center=center, top_k=top_k)
            views_per_box[inst_id] = selected_cams

        # Xác định file output lưu cùng thư mục với file JSON đầu vào
        input_dir = boxes_json_path.parent
        if output_name is None:
            # Ví dụ: office_0_3d_boxes.json -> office_0_views.json
            base_stem = boxes_json_path.stem
            if "_3d_boxes" in base_stem:
                clean_stem = base_stem.replace("_3d_boxes", "")
                out_filename = f"{clean_stem}_views.json"
            else:
                out_filename = f"{base_stem}_views.json"
        else:
            out_filename = output_name if output_name.endswith(".json") else f"{output_name}.json"

        output_path = input_dir / out_filename

        # Lưu file JSON kết quả
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(views_per_box, f, indent=2)

        # Thống kê kết quả
        total_views_found = sum(len(v) for v in views_per_box.values())
        non_empty_boxes = sum(1 for v in views_per_box.values() if len(v) > 0)
        avg_views = total_views_found / max(1, len(views_per_box))

        print(f"    [✓] Đã xuất thành công: {output_path}")
        print(f"        * Số bounding boxes:   {len(views_per_box)}")
        print(f"        * Số box tìm thấy ảnh: {non_empty_boxes}/{len(views_per_box)}")
        print(f"        * Trung bình ảnh/box:  {avg_views:.1f} ảnh (tối đa {top_k})")

        return output_path, views_per_box


def resolve_sfm_directory(
    scene_name: str,
    custom_sfm_dir: Optional[Union[str, Path]] = None,
    dataset_name: str = "replica"
) -> Optional[Path]:
    """Tự động tìm đường dẫn thư mục SfM COLMAP (sparse/0) cho một scene."""
    if custom_sfm_dir:
        p = resolve_path(custom_sfm_dir)
        if p.is_dir():
            if (p / "0").is_dir():
                return p / "0"
            return p

    # 1. Thử từ Dataset Configs nếu có
    if load_dataset_paths:
        try:
            dp = load_dataset_paths(dataset_name, scene=scene_name)
            if "sfm_dir" in dp and os.path.exists(dp["sfm_dir"]):
                return Path(dp["sfm_dir"])
        except Exception:
            pass

    # 2. Thử các đường dẫn relative chuẩn trong repo Safe-GS
    candidates = [
        REPO_ROOT / "outputs" / "Replica" / "sfm" / scene_name / "sparse" / "0",
        REPO_ROOT / "outputs" / "Replica" / "sfm" / scene_name / "sparse",
        REPO_ROOT / "outputs" / "Replica" / "sfm" / scene_name,
        REPO_ROOT / "outputs" / "sfm" / scene_name / "sparse" / "0",
        REPO_ROOT / "Data" / "Replica" / scene_name / "sfm" / "sparse" / "0",
    ]
    for c in candidates:
        if c.is_dir() and ((c / "cameras.bin").is_file() or (c / "cameras.txt").is_file()):
            return c

    # Kiểm tra cả trường hợp candidate là thư mục cha chứa "0"
    for c in candidates:
        if (c / "0").is_dir() and ((c / "0" / "cameras.bin").is_file() or (c / "0" / "cameras.txt").is_file()):
            return c / "0"

    return candidates[0]


def main():
    parser = argparse.ArgumentParser(
        description="Chọn lọc Top K góc nhìn tốt nhất cho từng bounding box 3D trong file JSON."
    )
    parser.add_argument(
        "--boxes_json", "-i", type=str,
        default="outputs/Replica/bounding_box_3D_and_view_selection/office0/office_0_3d_boxes.json",
        help="Đường dẫn file JSON chứa các bounding boxes (mặc định: outputs/Replica/bounding_box_3D_and_view_selection/office0/office_0_3d_boxes.json)"
    )
    parser.add_argument(
        "--sfm_dir", type=str, default=None,
        help="Đường dẫn thư mục COLMAP reconstruction sparse/0 (mặc định: tự động tìm theo scene trong outputs/Replica/sfm/<scene>/sparse/0)"
    )
    parser.add_argument(
        "--top_k", "-k", type=int, default=50,
        help="Số lượng góc nhìn (ảnh) tối ưu lấy cho mỗi bounding box (mặc định: 50)"
    )
    parser.add_argument(
        "--output_name", "-o", type=str, default=None,
        help="Tên file JSON kết quả lưu trong cùng thư mục với file input (mặc định: <input_stem>_views.json)"
    )
    parser.add_argument(
        "--box_type", type=str, choices=["arbitrary_3d_obb", "gravity_aligned_obb", "aabb"],
        default="arbitrary_3d_obb",
        help="Loại bounding box sử dụng để tính toán góc nhìn (mặc định: arbitrary_3d_obb)"
    )
    parser.add_argument(
        "--all_scenes", action="store_true",
        help="Xử lý hàng loạt cho tất cả các scene có trong outputs/Replica/bounding_box_3D_and_view_selection"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("      TIẾN TRÌNH VIEW SELECTION CHO 3D BOUNDING BOXES (REPLICA DATASET)")
    print("=" * 80)

    # Nếu chạy hàng loạt toàn bộ scene
    if args.all_scenes:
        base_dir = REPO_ROOT / "outputs" / "Replica" / "bounding_box_3D_and_view_selection"
        if not base_dir.is_dir():
            print(f"[Lỗi] Thư mục không tồn tại: {base_dir}")
            return

        json_targets = sorted(list(base_dir.glob("*/*_3d_boxes.json")))
        if not json_targets:
            print(f"[!] Không tìm thấy file *_3d_boxes.json nào trong {base_dir}")
            return

        print(f"[*] Tìm thấy {len(json_targets)} files JSON scene cần xử lý.")

        for target_json in json_targets:
            scene = extract_scene_name(target_json)
            sfm_path = resolve_sfm_directory(scene, custom_sfm_dir=args.sfm_dir)
            if not sfm_path or not sfm_path.exists():
                print(f"[-] Bỏ qua {scene}: Không tìm thấy SfM sparse tại {sfm_path}")
                continue

            try:
                selector = ROIViewSelector(sfm_sparse_dir=sfm_path)
                selector.process_boxes_json(
                    boxes_json_path=target_json,
                    output_name=args.output_name,
                    top_k=args.top_k,
                    bbox_type=args.box_type
                )
            except Exception as e:
                print(f"[!] Lỗi khi xử lý {target_json}: {e}")

    # Chạy cho một file JSON cụ thể
    else:
        target_json = resolve_path(args.boxes_json)
        if not target_json.is_file():
            print(f"[Lỗi] Không tìm thấy file JSON đầu vào: {target_json}")
            return

        scene = extract_scene_name(target_json)
        sfm_path = resolve_sfm_directory(scene, custom_sfm_dir=args.sfm_dir)

        if not sfm_path or not sfm_path.exists():
            print(f"[Lỗi] Không tìm thấy SfM sparse directory tại: {sfm_path}")
            return

        print(f"[*] File JSON đầu vào: {target_json}")
        print(f"[*] Scene xác định:     {scene}")
        print(f"[*] SfM sparse nguồn:   {sfm_path}")
        print(f"[*] Top K góc nhìn:     {args.top_k}")
        print("-" * 80)

        selector = ROIViewSelector(sfm_sparse_dir=sfm_path)
        out_path, _ = selector.process_boxes_json(
            boxes_json_path=target_json,
            output_name=args.output_name,
            top_k=args.top_k,
            bbox_type=args.box_type
        )

    print("\n" + "=" * 80)
    print(f"[✓] HOÀN TẤT TIẾN TRÌNH VIEW SELECTION!")
    print("=" * 80)


if __name__ == "__main__":
    main()