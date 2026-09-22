import argparse
import sys
from pathlib import Path

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.module2_scene_gs.scene_branch.scene_trainer import SceneGSTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_scene.toml")
    parser.add_argument("--sh_degree", type=int, default=None,
                         help="Ghi đè sh_degree trong config (vd 0 để tắt hẳn Spherical Harmonics bậc cao, chỉ giữ màu DC)")
    parser.add_argument("--checkpoint_tag", default="",
                         help="Hậu tố thêm vào tên file checkpoint (vd '_sh0' -> scene_gs_7000_sh0.ply)")
    parser.add_argument("--sfm_dir", default=None, help="Ghi đè sfm_dir trong config (vd để train scene khác)")
    parser.add_argument("--raw_image_dir", default=None, help="Ghi đè raw_image_dir trong config")
    parser.add_argument("--workspace_dir", default=None, help="Ghi đè workspace_dir trong config")
    parser.add_argument("--iterations", type=int, default=None, help="Ghi đè số iterations trong config")
    args = parser.parse_args()

    print("=" * 60)
    print(">>> BẮT ĐẦU CHẠY MODULE 2: NHÁNH SCENE-GS TRAINING <<<")
    print("=" * 60)

    overrides = {}
    if args.sh_degree is not None:
        overrides["sh_degree"] = args.sh_degree
    if args.checkpoint_tag:
        overrides["checkpoint_tag"] = args.checkpoint_tag
    if args.sfm_dir is not None:
        overrides["sfm_dir"] = args.sfm_dir
    if args.raw_image_dir is not None:
        overrides["raw_image_dir"] = args.raw_image_dir
    if args.workspace_dir is not None:
        overrides["workspace_dir"] = args.workspace_dir
    if args.iterations is not None:
        overrides["iterations"] = args.iterations

    trainer = SceneGSTrainer(config_path=args.config, overrides=overrides)
    trainer.train()

    print("=" * 60)
    print(">>> HOÀN THÀNH HUẤN LUYỆN SCENE-GS THÀNH CÔNG <<<")
    print("=" * 60)


if __name__ == "__main__":
    main()