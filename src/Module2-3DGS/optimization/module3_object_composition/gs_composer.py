import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Union, List, Optional
from plyfile import PlyData, PlyElement


class GSComposer:
    """
    Module hợp nhất mô hình 3DGS toàn cảnh và mô hình cục bộ:
    Thay thế các hạt Gaussian thô của Scene-GS bên trong AABB bằng các hạt siêu nét của Object-GS.
    """
    def __init__(
        self,
        workspace_dir: Union[str, Path] = "data/workspace",
        roi_metadata_path: Optional[Union[str, Path]] = None,
        blend_margin_ratio: float = 0.0  # Tùy chọn nới rộng/thu hẹp biên hoán đổi hạt
    ):
        self.workspace_dir = Path(workspace_dir)
        self.checkpoints_dir = self.workspace_dir / "checkpoints"
        
        if roi_metadata_path is None:
            self.roi_metadata_path = self.workspace_dir / "roi_boxes/roi_metadata.json"
        else:
            self.roi_metadata_path = Path(roi_metadata_path)

        self.blend_margin_ratio = blend_margin_ratio
        self._load_roi_bounds()

    def _load_roi_bounds(self):
        """Đọc tọa độ AABB từ file metadata của Module 2."""
        if not self.roi_metadata_path.exists():
            raise FileNotFoundError(f"Không tìm thấy metadata ROI tại: {self.roi_metadata_path}")

        with open(self.roi_metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        aabb_info = meta["bounds"]["aabb"]
        self.aabb_min = np.array(aabb_info["min"], dtype=np.float32)
        self.aabb_max = np.array(aabb_info["max"], dtype=np.float32)

        if self.blend_margin_ratio != 0.0:
            extent = self.aabb_max - self.aabb_min
            margin = extent * self.blend_margin_ratio
            self.aabb_min -= margin
            self.aabb_max += margin

        print(f"[*] Phạm vi hoán đổi hạt 3D AABB:")
        print(f"    - Min: {self.aabb_min}")
        print(f"    - Max: {self.aabb_max}")

    def _is_inside_aabb(self, xyz: np.ndarray) -> np.ndarray:
        """Kiểm tra điều kiện tọa độ điểm nằm trong hộp bao AABB."""
        return np.all((xyz >= self.aabb_min) & (xyz <= self.aabb_max), axis=-1)

    def _read_ply_elements(self, ply_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Đọc toàn bộ thuộc tính và tên thuộc tính từ file PLY của 3DGS."""
        if not ply_path.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint PLY: {ply_path}")

        plydata = PlyData.read(str(ply_path))
        vertex_data = plydata['vertex'].data
        property_names = [prop.name for prop in plydata['vertex'].properties]

        # Trích xuất tọa độ XYZ để lọc không gian
        xyz = np.stack([vertex_data['x'], vertex_data['y'], vertex_data['z']], axis=-1)
        return vertex_data, xyz, property_names

    def compose(
        self,
        scene_ply_path: Optional[Union[str, Path]] = None,
        object_ply_path: Optional[Union[str, Path]] = None,
        output_ply_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Thực hiện hoán đổi và ghép hạt, sau đó lưu ra file composed_3dgs.ply.
        """
        scene_path = Path(scene_ply_path) if scene_ply_path else self.checkpoints_dir / "scene_gs.ply"
        object_path = Path(object_ply_path) if object_ply_path else self.checkpoints_dir / "object_gs.ply"
        output_path = Path(output_ply_path) if output_ply_path else self.checkpoints_dir / "composed_3dgs.ply"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[*] Bắt đầu quá trình ghép nối (Scene-Objects Composition)...")
        print(f"    - Scene-GS input:  {scene_path}")
        print(f"    - Object-GS input: {object_path}")

        # 1. Đọc dữ liệu Scene-GS và lọc bỏ vùng AABB
        scene_vertex, scene_xyz, scene_props = self._read_ply_elements(scene_path)
        scene_inside_mask = self._is_inside_aabb(scene_xyz)
        scene_keep_mask = ~scene_inside_mask  # Chỉ giữ lại hạt NẰM NGOÀI AABB

        filtered_scene_vertex = scene_vertex[scene_keep_mask]
        num_scene_orig = len(scene_vertex)
        num_scene_kept = len(filtered_scene_vertex)
        print(f"[✓] Scene-GS: Loại bỏ {num_scene_orig - num_scene_kept} hạt thô trong ROI. Giữ lại {num_scene_kept} hạt nền.")

        # 2. Đọc dữ liệu Object-GS và chỉ lấy vùng AABB
        object_vertex, object_xyz, object_props = self._read_ply_elements(object_path)
        object_inside_mask = self._is_inside_aabb(object_xyz)  # Chỉ giữ lại hạt NẰM TRONG AABB

        filtered_object_vertex = object_vertex[object_inside_mask]
        num_object_orig = len(object_vertex)
        num_object_kept = len(filtered_object_vertex)
        print(f"[✓] Object-GS: Lấy {num_object_kept}/{num_object_orig} hạt chi tiết cao bên trong ROI.")

        # 3. Kiểm tra tính đồng bộ của danh sách thuộc tính
        if scene_props != object_props:
            raise ValueError("Định dạng thuộc tính của Scene-GS và Object-GS không đồng nhất!")

        # 4. Hợp nhất hai mảng hạt
        composed_vertex = np.concatenate([filtered_scene_vertex, filtered_object_vertex], axis=0)
        total_gaussians = len(composed_vertex)

        # 5. Ghi ra file PLY hoàn chỉnh
        composed_element = PlyElement.describe(composed_vertex, 'vertex')
        PlyData([composed_element]).write(str(output_path))

        print(f"[✓] Ghép nối thành công!")
        print(f"    -> Tổng số lượng hạt của Composed-3DGS: {total_gaussians}")
        print(f"    -> File đã lưu tại: {output_path}")

        # Xuất file báo cáo thông số ghép nối
        meta_summary = {
            "scene_gaussians_original": int(num_scene_orig),
            "scene_gaussians_kept_background": int(num_scene_kept),
            "object_gaussians_original": int(num_object_orig),
            "object_gaussians_kept_foreground": int(num_object_kept),
            "total_composed_gaussians": int(total_gaussians),
            "aabb_min": self.aabb_min.tolist(),
            "aabb_max": self.aabb_max.tolist()
        }
        summary_path = self.workspace_dir / "checkpoints/composition_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(meta_summary, f, indent=4)

        return output_path


if __name__ == "__main__":
    # Doc duong dan tu configs/object_roi.toml thay vi hard-code (thay doi
    # duong dan khi chuyen may chi can sua file config, khong sua code).
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    with open("configs/object_roi.toml", "rb") as f:
        _cfg = tomllib.load(f)

    composer = GSComposer(workspace_dir=_cfg.get("workspace_dir", "data/workspace"))
    composer.compose()