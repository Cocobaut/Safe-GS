import os
import sys
import json
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np


class SfMPipeline:
    """
    Pipeline tự động hóa Structure-from-Motion (SfM) qua COLMAP CLI (hỗ trợ CUDA GPU):
    Trích xuất Camera Poses, Intrinsics, Mây điểm thưa và Visibility Graph.
    Tương thích COLMAP 4.x (--device CUDA).
    """
    def __init__(
        self,
        images_dir: Union[str, Path],
        output_dir: Union[str, Path],
        camera_model: str = "PINHOLE",
        single_camera: bool = True,
        use_gpu: bool = True,
        matching_method: str = "sequential"  # "sequential" (khuyên dùng cho Replica sequence) hoặc "exhaustive"
    ):
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.camera_model = camera_model
        self.single_camera = single_camera
        self.use_gpu = use_gpu
        self.matching_method = matching_method

        self.database_path = self.output_dir / "database.db"
        self.sparse_dir = self.output_dir / "sparse"
        self.sparse_model_dir = self.sparse_dir / "0"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sparse_dir.mkdir(parents=True, exist_ok=True)

    def _check_pycolmap_available(self) -> bool:
        try:
            import pycolmap
            return True
        except ImportError:
            return False

    def _find_colmap_binary(self) -> str:
        """
        Tìm chính xác đường dẫn tới colmap.exe để tránh xung đột với colmap.bat khi đường dẫn có khoảng trắng.
        """
        colmap_bin = shutil.which("colmap.exe")
        if colmap_bin is None:
            colmap_bin = shutil.which("colmap")

        # Nếu nhận diện file .bat, trỏ sang file .exe trong thư mục bin
        if colmap_bin and colmap_bin.lower().endswith(".bat"):
            exe_cand = Path(colmap_bin).parent / "bin" / "colmap.exe"
            if exe_cand.exists():
                colmap_bin = str(exe_cand)

        if colmap_bin is None:
            raise RuntimeError(
                "Không tìm thấy binary 'colmap' trong biến môi trường PATH.\n"
                "Hãy kiểm tra lại đường dẫn C:\\colmap\\bin hoặc E:\\Download folder\\...\\bin trong System PATH."
            )
        return colmap_bin

    def run_colmap_cli(self) -> bool:
        """
        Thực thi SfM thông qua COLMAP CLI (tận dụng CUDA trên GPU sẵn có).
        """
        colmap_bin = self._find_colmap_binary()

        device_arg = "CUDA" if self.use_gpu else "CPU"
        device_label = device_arg
        if self.use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    device_label = f"CUDA ({torch.cuda.get_device_name(0)})"
            except ImportError:
                pass
        print(f"[Loading] Sử dụng COLMAP binary: {colmap_bin}")
        print(f"[Loading] Chế độ thiết bị: {device_label}")
        print(f"[Loading] Chế độ Matching: {self.matching_method}")

        if self.database_path.exists():
            self.database_path.unlink()

        # Dọn biến môi trường Qt bị các thư viện khác (vd opencv-python) ghi đè -
        # opencv tự trỏ QT_QPA_PLATFORM_PLUGIN_PATH vào Qt plugin rieng cua no khi import,
        # khien COLMAP con (dung Qt/OpenGL de khoi tao SiftGPU) bi SIGABRT vi load nham
        # plugin xcb khong tuong thich. Xoa bien nay khoi env truyen cho subprocess de
        # COLMAP tu tim plugin Qt he thong.
        clean_env = os.environ.copy()
        clean_env.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
        clean_env.pop("QT_PLUGIN_PATH", None)

        # 1. Feature Extractor (SIFT trên GPU với COLMAP 4.x)
        cmd_extract = [
            colmap_bin, "feature_extractor",
            "--database_path", str(self.database_path),
            "--image_path", str(self.images_dir),
            "--ImageReader.camera_model", self.camera_model,
            "--ImageReader.single_camera", "1" if self.single_camera else "0",
        ]
        print("\n[Loading] [1/3] CLI: Đang trích xuất đặc trưng SIFT (Feature Extractor)...")
        subprocess.run(cmd_extract, check=True, env=clean_env)

        # 2. Feature Matching (Sequential hoặc Exhaustive)
        if self.matching_method == "sequential":
            matcher_cmd_name = "sequential_matcher"
            cmd_match = [
                colmap_bin, matcher_cmd_name,
                "--database_path", str(self.database_path),
                "--SequentialMatching.overlap", "15",
            ]
        else:
            matcher_cmd_name = "exhaustive_matcher"
            cmd_match = [
                colmap_bin, matcher_cmd_name,
                "--database_path", str(self.database_path),
            ]

        print(f"\n[Loading] [2/3] CLI: Đang so khớp đặc trưng ({matcher_cmd_name})...")
        subprocess.run(cmd_match, check=True, env=clean_env)

        # 3. Incremental Mapper
        self.sparse_model_dir.mkdir(parents=True, exist_ok=True)
        cmd_map = [
            colmap_bin, "mapper",
            "--database_path", str(self.database_path),
            "--image_path", str(self.images_dir),
            "--output_path", str(self.sparse_dir)
        ]
        print("\n[Loading] [3/3] CLI: Đang tái tạo 3D gia tăng (Incremental Mapper)...")
        subprocess.run(cmd_map, check=True, env=clean_env)

        if not (self.sparse_model_dir / "cameras.bin").exists() and not (self.sparse_model_dir / "cameras.txt").exists():
            subdirs = [d for d in self.sparse_dir.iterdir() if d.is_dir() and d.name != "0"]
            if subdirs:
                chosen_dir = subdirs[0]
                for f in chosen_dir.glob("*"):
                    shutil.move(str(f), str(self.sparse_model_dir))

        print("\n[Done] Hoàn thành COLMAP CLI mapping.")
        self._parse_and_export_visibility_from_disk()
        return True

    def _export_visibility_graph(self, reconstruction) -> None:
        visibility = {
            "num_points": len(reconstruction.points3D),
            "num_images": len(reconstruction.images),
            "point_to_images": {},
            "image_to_points": {}
        }

        for img_id, img in reconstruction.images.items():
            visibility["image_to_points"][img.name] = []

        for p3d_id, pt in reconstruction.points3D.items():
            visible_image_names = []
            for track_el in pt.track.elements:
                img = reconstruction.images.get(track_el.image_id)
                if img is not None:
                    visible_image_names.append(img.name)
                    visibility["image_to_points"][img.name].append(int(p3d_id))

            visibility["point_to_images"][int(p3d_id)] = {
                "xyz": pt.xyz.tolist(),
                "rgb": pt.color.tolist(),
                "error": float(pt.error),
                "visible_in_images": visible_image_names
            }

        out_file = self.output_dir / "visibility_graph.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(visibility, f, indent=2)
        print(f"[Done] Đã xuất Visibility Graph tại: {out_file}")

    def _read_points3d_binary(self, path_to_model_file: Path) -> Dict:
        points3D = {}
        with open(path_to_model_file, "rb") as fid:
            num_points = struct.unpack("<Q", fid.read(8))[0]
            for _ in range(num_points):
                # SỬA DÒNG NÀY: Đổi 40 thành 43 bytes để khớp với format "<QdddBBBd"
                # (Q=8 + 3*d=24 + 3*B=3 + d=8 -> Tổng cộng = 43 bytes)
                binary_point_header = fid.read(43)
                
                point_id, x, y, z, r, g, b, error = struct.unpack("<QdddBBBd", binary_point_header)
                track_length = struct.unpack("<Q", fid.read(8))[0]
                image_ids = []
                for _ in range(track_length):
                    image_id, point2d_idx = struct.unpack("<ii", fid.read(8))
                    image_ids.append(image_id)
                points3D[point_id] = {
                    "xyz": [x, y, z],
                    "rgb": [r, g, b],
                    "error": error,
                    "image_ids": image_ids
                }
        return points3D

    def _read_images_binary(self, path_to_model_file: Path) -> Dict[int, str]:
        images = {}
        with open(path_to_model_file, "rb") as fid:
            num_reg_images = struct.unpack("<Q", fid.read(8))[0]
            for _ in range(num_reg_images):
                binary_image_header = fid.read(64)
                image_id = struct.unpack("<i", binary_image_header[:4])[0]
                
                name_chars = []
                while True:
                    ch = fid.read(1)
                    if ch == b"\x00":
                        break
                    name_chars.append(ch.decode("utf-8", errors="ignore"))
                image_name = "".join(name_chars)
                
                num_points2D = struct.unpack("<Q", fid.read(8))[0]
                fid.seek(num_points2D * 24, os.SEEK_CUR)
                images[image_id] = image_name
        return images

    def _parse_and_export_visibility_from_disk(self) -> None:
        if self._check_pycolmap_available():
            try:
                import pycolmap
                recon = pycolmap.Reconstruction(self.sparse_model_dir)
                self._export_visibility_graph(recon)
                return
            except Exception:
                pass

        bin_points = self.sparse_model_dir / "points3D.bin"
        bin_images = self.sparse_model_dir / "images.bin"

        if not (bin_points.exists() and bin_images.exists()):
            print("[Error] Không tìm thấy points3D.bin hoặc images.bin để xuất visibility graph.")
            return

        id_to_name = self._read_images_binary(bin_images)
        raw_points = self._read_points3d_binary(bin_points)

        visibility = {
            "num_points": len(raw_points),
            "num_images": len(id_to_name),
            "point_to_images": {},
            "image_to_points": {name: [] for name in id_to_name.values()}
        }

        for p3d_id, data in raw_points.items():
            vis_names = [id_to_name[i_id] for i_id in data["image_ids"] if i_id in id_to_name]
            for name in vis_names:
                visibility["image_to_points"][name].append(int(p3d_id))

            visibility["point_to_images"][int(p3d_id)] = {
                "xyz": data["xyz"],
                "rgb": data["rgb"],
                "error": data["error"],
                "visible_in_images": vis_names
            }

        out_file = self.output_dir / "visibility_graph.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(visibility, f, indent=2)
        print(f"[Done] Đã xuất Visibility Graph từ disk binary tại: {out_file}")

    def run(self) -> bool:
        return self.run_colmap_cli()


if __name__ == "__main__":
    # Doc duong dan tu configs/base_scene.toml thay vi hard-code (thay doi
    # duong dan khi chuyen may chi can sua file config, khong sua code).
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    with open("configs/base_scene.toml", "rb") as f:
        _cfg = tomllib.load(f)

    from src.common.config_loader import resolve_raw_image_dir, resolve_run_tag

    raw_images = resolve_raw_image_dir(_cfg)
    # Ghep them run_tag (vd "replica_office0") de output SfM cua tung
    # dataset/scene khac nhau khong bi ghi de len nhau.
    run_tag = resolve_run_tag(_cfg)
    sfm_output = str(Path(_cfg.get("workspace_dir", "data/workspace")) / "sfm" / run_tag)

    if Path(raw_images).exists() and any(Path(raw_images).iterdir()):
        sfm = SfMPipeline(
            images_dir=raw_images,
            output_dir=sfm_output,
            camera_model="PINHOLE",
            single_camera=True,
            use_gpu=True,                     # Tận dụng CUDA trên GPU sẵn có
            matching_method="sequential"      # Sequential Matching cực nhanh cho chuỗi ảnh Replica
        )
        success = sfm.run()
        if success:
            print(f"\n[Done] Hoàn thành Module 1 SfM. Dữ liệu sẵn sàng tại: {sfm_output}/sparse/0/")
        else:
            print("\n[Error] Chạy SfM thất bại.")
    else:
        print(f"Vui lòng đặt ảnh vào thư mục: {raw_images} trước khi chạy thử.")