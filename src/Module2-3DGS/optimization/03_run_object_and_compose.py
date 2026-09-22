import os
import sys
import time
import argparse
from pathlib import Path

# Thêm thư mục gốc vào PYTHONPATH để import module từ src/
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.common.config_loader import load_toml_config
from src.module3_object_composition.object_trainer import ObjectGSTrainer
from src.module3_object_composition.gs_composer import GSComposer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Module 3: Huấn luyện Object-GS cục bộ và Hợp nhất thành Composed-3DGS."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/object_roi.toml",
        help="Đường dẫn tới file cấu hình (VD: configs/object_roi.toml hoặc configs/datasets/dtu_scan24.toml)"
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Bỏ qua bước huấn luyện Object-GS nếu đã có sẵn checkpoint object_gs.ply (chỉ chạy bước ghép Composer)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"[!] Lỗi: Không tìm thấy file cấu hình tại: {config_path}")
        sys.exit(1)

    cfg = load_toml_config(config_path)

    # Xác định thư mục workspace
    workspace_dir = Path(cfg.get("workspace_dir", "data/workspace"))
    checkpoints_dir = workspace_dir / "checkpoints"
    roi_boxes_dir = workspace_dir / "roi_boxes"

    # Kiểm tra các điều kiện tiên quyết từ Module 2
    scene_ply_path = checkpoints_dir / "scene_gs.ply"
    roi_metadata_path = roi_boxes_dir / "roi_metadata.json"

    print("=" * 70)
    print(">>> BẮT ĐẦU CHẠY MODULE 3: OBJECT-GS TRAINING & 3DGS COMPOSITION <<<")
    print("=" * 70)
    print(f"[*] Cấu hình thực thi : {config_path}")
    print(f"[*] Workspace          : {workspace_dir}")
    print(f"[*] Scene-GS input     : {scene_ply_path}")
    print(f"[*] ROI Metadata input : {roi_metadata_path}")
    print("-" * 70)

    if not scene_ply_path.exists():
        print(f"[!] Lỗi: Chưa có checkpoint Scene-GS tại '{scene_ply_path}'.")
        print("    Vui lòng chạy Module 2 (scripts/02_run_scene_gs.py) trước!")
        sys.exit(1)

    if not roi_metadata_path.exists():
        print(f"[!] Lỗi: Chưa có metadata ROI tại '{roi_metadata_path}'.")
        print("    Vui lòng chạy nhánh ROI 3D (scripts/02_run_roi_3d.py) trước!")
        sys.exit(1)

    start_time = time.time()

    # =========================================================================
    # GIAI ĐOẠN 1: HUẤN LUYỆN CHUYÊN SÂU OBJECT-GS (TARGETED ROI TRAINING)
    # =========================================================================
    object_ply_path = checkpoints_dir / "object_gs.ply"

    if args.skip_training and object_ply_path.exists():
        print(f"\n[BƯỚC 1/2] Bỏ qua huấn luyện Object-GS (Đã có sẵn: {object_ply_path})")
    else:
        print("\n[BƯỚC 1/2] Bắt đầu huấn luyện nhánh Object-GS (Confined Densification trong AABB)...")
        trainer = ObjectGSTrainer(config_path=str(config_path))
        trainer.train()

        if not object_ply_path.exists():
            print(f"[!] Lỗi: Quá trình huấn luyện không sinh ra file checkpoint '{object_ply_path}'.")
            sys.exit(1)

    # =========================================================================
    # GIAI ĐOẠN 2: HỢP NHẤT KHÔNG GIAN (SCENE-OBJECTS COMPOSITION)
    # =========================================================================
    print("\n[BƯỚC 2/2] Khởi chạy GSComposer: Hoán đổi hạt và sinh Composed-3DGS...")
    composed_ply_path = checkpoints_dir / "composed_3dgs.ply"

    composer = GSComposer(
        workspace_dir=workspace_dir,
        roi_metadata_path=roi_metadata_path,
        blend_margin_ratio=cfg.get("blend_margin_ratio", 0.0)
    )

    final_ply = composer.compose(
        scene_ply_path=scene_ply_path,
        object_ply_path=object_ply_path,
        output_ply_path=composed_ply_path
    )

    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    print("\n" + "=" * 70)
    print(">>> HOÀN THÀNH TOÀN BỘ TIẾN TRÌNH MODULE 3 THÀNH CÔNG <<<")
    print(f"[*] Tổng thời gian thực thi : {minutes} phút {seconds} giây")
    print(f"[*] File 3DGS hoàn chỉnh    : {final_ply}")
    print(f"[*] Sẵn sàng cho Module 4   : scripts/04_run_gof_meshing.py")
    print("=" * 70)


if __name__ == "__main__":
    main()