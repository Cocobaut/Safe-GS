import os
import sys
import time
import argparse
from pathlib import Path

# Thêm thư mục gốc vào PYTHONPATH để import được các modules trong src/
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.common.config_loader import load_toml_config
from src.module1_perception.sfm_pipeline import SfMPipeline
from src.module1_perception.segmenter import RMBG2Segmenter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Module 1: Chạy SfM (COLMAP) và 2D Segmentation (RMBG-2.0) trên tập ảnh RGB."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Đường dẫn tới file cấu hình dataset (VD: configs/datasets/dtu_scan24.toml). Nếu không có, dùng mặc định."
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="data/raw",
        help="Thư mục chứa ảnh RGB đầu vào (mặc định: data/raw)"
    )
    parser.add_argument(
        "--workspace_dir",
        type=str,
        default="data/workspace",
        help="Thư mục chứa kết quả trung gian workspace (mặc định: data/workspace)"
    )
    parser.add_argument(
        "--sfm_output_dir",
        type=str,
        default=None,
        help="Chỉ định trực tiếp thư mục xuất SfM (vd outputs/DTU/sfm/dtu_scan24)"
    )
    parser.add_argument(
        "--seg_output_dir",
        type=str,
        default=None,
        help="Chỉ định trực tiếp thư mục xuất Segmentation"
    )
    parser.add_argument(
        "--skip_sfm",
        action="store_true",
        help="Bỏ qua bước SfM nếu đã có sẵn dữ liệu COLMAP (poses, intrinsics, points3D)"
    )
    parser.add_argument(
        "--skip_seg",
        action="store_true",
        help="Bỏ qua bước Segmentation nếu đã tạo xong 2D Masks"
    )
    parser.add_argument(
        "--save_vis",
        action="store_true",
        help="Lưu ảnh visualization đè mask màu xanh để kiểm tra chất lượng cắt biên"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Thiết lập đường dẫn từ file TOML hoặc từ tham số dòng lệnh
    if args.config and Path(args.config).exists():
        cfg = load_toml_config(args.config)
        images_dir = Path(cfg.get("raw_image_dir", args.images_dir))
        workspace_dir = Path(cfg.get("workspace_dir", args.workspace_dir))
        camera_model = cfg.get("camera_model", "PINHOLE")
        single_camera = cfg.get("single_camera", True)
    else:
        images_dir = Path(args.images_dir)
        workspace_dir = Path(args.workspace_dir)
        camera_model = "PINHOLE"
        single_camera = True

    # Định nghĩa cấu trúc thư mục output
    if args.sfm_output_dir:
        sfm_output_dir = Path(args.sfm_output_dir)
    elif "sfm" in workspace_dir.parts or workspace_dir.name == "sfm":
        sfm_output_dir = workspace_dir
    else:
        sfm_output_dir = workspace_dir / "sfm"

    sparse_model_dir = sfm_output_dir / "sparse" / "0"
    
    if args.seg_output_dir:
        seg_output_dir = Path(args.seg_output_dir)
    else:
        seg_output_dir = workspace_dir / "segmentation"
        
    vis_output_dir = seg_output_dir / "vis" if args.save_vis else None


    print("=" * 70)
    print(">>> BẮT ĐẦU CHẠY MODULE 1: PERCEPTION (SfM + 2D SEGMENTATION) <<<")
    print("=" * 70)
    print(f"[*] Thư mục ảnh đầu vào  : {images_dir}")
    print(f"[*] Thư mục Workspace    : {workspace_dir}")
    print(f"[*] Output SfM           : {sfm_output_dir}")
    print(f"[*] Output 2D Masks      : {seg_output_dir}")
    print("-" * 70)

    # Kiểm tra sự tồn tại của thư mục ảnh
    if not images_dir.exists():
        print(f"[!] Lỗi: Không tìm thấy thư mục ảnh đầu vào tại: {images_dir}")
        sys.exit(1)

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [p for p in images_dir.iterdir() if p.suffix.lower() in valid_extensions]
    if not image_files:
        print(f"[!] Lỗi: Không có ảnh hợp lệ nào trong: {images_dir}")
        sys.exit(1)

    print(f"[*] Tìm thấy {len(image_files)} ảnh hợp lệ cần xử lý.")
    start_total_time = time.time()

    # =========================================================================
    # GIAI ĐOẠN 1: TÁI TẠO HÌNH HỌC VÀ CAMERA POSES QUA COLMAP (SfM)
    # =========================================================================
    has_sparse_bin = (sparse_model_dir / "cameras.bin").exists() or (sparse_model_dir / "cameras.txt").exists()

    if args.skip_sfm and has_sparse_bin:
        print(f"\n[BƯỚC 1/2] Bỏ qua SfM (Đã có sẵn mô hình sparse tại: {sparse_model_dir})")
    else:
        print("\n[BƯỚC 1/2] Khởi chạy Structure-from-Motion (COLMAP)...")
        sfm_start = time.time()

        sfm_pipeline = SfMPipeline(
            images_dir=images_dir,
            output_dir=sfm_output_dir,
            camera_model=camera_model,
            single_camera=single_camera,
            use_gpu=True
        )
        sfm_success = sfm_pipeline.run()

        if not sfm_success:
            print("[!] Lỗi: Chạy SfM thất bại. Vui lòng kiểm tra lại chất lượng ảnh hoặc cài đặt COLMAP.")
            sys.exit(1)

        sfm_elapsed = time.time() - sfm_start
        print(f"[✓] Hoàn thành SfM trong {int(sfm_elapsed // 60)}m {int(sfm_elapsed % 60)}s.")

    # =========================================================================
    # GIAI ĐOẠN 2: PHÂN ĐOẠN ĐỐI TƯỢNG TIỀN CẢNH BẰNG RMBG-2.0
    # =========================================================================
    meta_json_path = seg_output_dir / "segmentation_meta.json"

    if args.skip_seg:
        print(f"\n[BƯỚC 2/2] Bỏ qua 2D Segmentation theo yêu cầu (--skip_seg).")
    else:
        print("\n[BƯỚC 2/2] Khởi chạy 2D Segmentation bằng RMBG-2.0...")
        seg_start = time.time()

        segmenter = RMBG2Segmenter(
            model_name_or_path="briaai/RMBG-2.0",
            threshold=0.5
        )

        segmenter.process_directory(
            images_dir=images_dir,
            output_mask_dir=seg_output_dir,
            save_visualizations=args.save_vis,
            vis_output_dir=vis_output_dir
        )

        seg_elapsed = time.time() - seg_start
        print(f"[✓] Hoàn thành Segmentation trong {int(seg_elapsed // 60)}m {int(seg_elapsed % 60)}s.")

    # =========================================================================
    # KIỂM TRA TÍNH TOÀN VẸN TRƯỚC KHI CHUYỂN QUA MODULE 2
    # =========================================================================
    total_elapsed = time.time() - start_total_time
    print("\n" + "=" * 70)
    print(">>> HOÀN THÀNH TIẾN TRÌNH MODULE 1 THÀNH CÔNG <<<")
    print(f"[*] Tổng thời gian thực thi : {int(total_elapsed // 60)}m {int(total_elapsed % 60)}s")
    print(f"[*] Dữ liệu SfM sẵn sàng   : {sparse_model_dir}")
    print(f"[*] Mặt nạ nhị phân 2D     : {seg_output_dir}")
    print(f"[*] Sẵn sàng chạy tiếp     :")
    print(f"    - Nhánh Scene-GS: python scripts/02_run_scene_gs.py")
    print(f"    - Nhánh ROI 3D  : python scripts/02_run_roi_3d.py")
    print("=" * 70)


if __name__ == "__main__":
    main()