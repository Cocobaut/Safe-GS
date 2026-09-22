import os
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional


class CameraScheduler:
    """
    Phân loại và điều phối danh sách Camera cho Scene-GS:
    - Nhận diện góc nhìn toàn cảnh (Coarse Views) và cận cảnh (Close-up ROI Views).
    - Lấy mẫu kết hợp (Coarse + tỷ lệ ROI) theo chiến lược đề xuất từ ROI-GS.
    """
    def __init__(
        self,
        sfm_dir: Union[str, Path],
        mask_meta_path: Optional[Union[str, Path]] = None,
        close_up_ratio_threshold: float = 0.08, # Ngưỡng diện tích mask/ảnh để coi là cận cảnh
        seed: int = 42
    ):
        self.sfm_dir = Path(sfm_dir)
        self.mask_meta_path = Path(mask_meta_path) if mask_meta_path else None
        self.close_up_ratio_threshold = close_up_ratio_threshold
        random.seed(seed)
        np.random.seed(seed)

    def classify_views(self) -> Tuple[List[str], List[str]]:
        """
        Phân tách ảnh thành 2 nhóm:
        - coarse_views: Ảnh toàn cảnh hoặc góc rộng.
        - closeup_views: Ảnh cận cảnh tập trung vào vật thể (ROI).
        """
        coarse_views = []
        closeup_views = []

        # Cách 1: Phân loại dựa trên diện tích Bounding Box / 2D Mask từ segmentation_meta.json
        if self.mask_meta_path and self.mask_meta_path.exists():
            with open(self.mask_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            for img_name, data in meta.items():
                img_area = data["width"] * data["height"]
                max_obj_area = 0

                for det in data.get("detections", []):
                    bbox = det["bbox_xyxy"] # [x1, y1, x2, y2]
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if area > max_obj_area:
                        max_obj_area = area

                coverage = max_obj_area / (img_area + 1e-6)
                if coverage >= self.close_up_ratio_threshold:
                    closeup_views.append(img_name)
                else:
                    coarse_views.append(img_name)

        # Cách 2: Fallback nếu chưa có file mask metadata (dựa vào số lượng SfM points quan sát được)
        else:
            # sfm_dir thuong la ".../sfm/sparse/0" (chua truc tiep points3D.bin),
            # trong khi visibility_graph.json duoc sfm_pipeline.py ghi o ".../sfm/"
            # (2 cap tren, chinh la ong-noi cua sparse/0) -> thu ca 2 vi tri de
            # tuong thich voi ca truong hop sfm_dir tro thang vao "sfm/".
            candidates = [
                self.sfm_dir.parent.parent / "visibility_graph.json",
                self.sfm_dir.parent / "visibility_graph.json",
            ]
            vis_graph_path = next((p for p in candidates if p.exists()), candidates[0])
            if vis_graph_path.exists():
                with open(vis_graph_path, "r", encoding="utf-8") as f:
                    vis_data = json.load(f)
                img_to_pts = vis_data.get("image_to_points", {})
                
                sorted_imgs = sorted(img_to_pts.items(), key=lambda x: len(x[1]))
                split_idx = int(len(sorted_imgs) * 0.7)
                coarse_views = [k for k, _ in sorted_imgs[:split_idx]]
                closeup_views = [k for k, _ in sorted_imgs[split_idx:]]
            else:
                raise FileNotFoundError(f"Không tìm thấy file metadata tại {self.mask_meta_path} hoặc {vis_graph_path}")

        print(f"[Loading] Phân loại Camera: {len(coarse_views)} ảnh Coarse | {len(closeup_views)} ảnh Close-up")
        return coarse_views, closeup_views

    def schedule_scene_cameras(
        self,
        roi_sample_rate: float = 0.5,
        output_file: Optional[Union[str, Path]] = None
    ) -> List[str]:
        """
        Tạo danh sách tập Camera tối ưu để train Scene-GS:
        100% Coarse Views + (roi_sample_rate * 100)% Close-up Views ngẫu nhiên đều.
        """
        coarse_views, closeup_views = self.classify_views()

        # Lấy mẫu một phần ảnh cận cảnh theo tỷ lệ của ROI-GS (50%)
        num_roi_samples = int(len(closeup_views) * roi_sample_rate)
        sampled_closeup = random.sample(closeup_views, num_roi_samples) if closeup_views else []

        final_scene_cameras = sorted(list(set(coarse_views + sampled_closeup)))

        schedule_meta = {
            "num_total_cameras": len(final_scene_cameras),
            "num_coarse_views": len(coarse_views),
            "num_sampled_closeup": len(sampled_closeup),
            "roi_sample_rate": roi_sample_rate,
            "scene_camera_list": final_scene_cameras
        }

        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(schedule_meta, f, indent=4)
            print(f"[Done] Đã xuất lịch trình Camera cho Scene-GS tại: {output_file}")

        return final_scene_cameras


if __name__ == "__main__":
    # Doc duong dan tu configs/base_scene.toml thay vi hard-code (thay doi
    # duong dan khi chuyen may chi can sua file config, khong sua code).
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    with open("configs/base_scene.toml", "rb") as f:
        _cfg = tomllib.load(f)
    _workspace_dir = Path(_cfg.get("workspace_dir", "data/workspace"))

    scheduler = CameraScheduler(
        sfm_dir=_cfg.get("sfm_dir", "data/sfm/sparse/0"),
        mask_meta_path=_cfg.get("mask_meta", "data/segmentation/segmentation_meta.json"),
        close_up_ratio_threshold=_cfg.get("close_up_ratio_threshold", 0.08)
    )
    selected_cams = scheduler.schedule_scene_cameras(
        roi_sample_rate=_cfg.get("roi_sample_rate", 0.5),
        output_file=_workspace_dir / "scene_camera_schedule.json",
    )
    print(f"Tổng số góc máy đưa vào huấn luyện Scene-GS: {len(selected_cams)}")