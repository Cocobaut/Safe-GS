"""
Modal experiment subset cho Safe-GS (modal_experiment.py)
=========================================================
Tạo Volume 'safe-gs-experiment' chứa tập con nhỏ của Replica để thí nghiệm nhanh:
  1. make_subset : lọc scene (mặc định office0) từ Volume 'safe-gs-dataset', giữ 1/stride ảnh RGB+depth.
  2. build_sfm   : chuyển pose ground-truth -> COLMAP sparse model (không chạy SfM thật).
  3. show        : liệt kê nội dung Volume.

Tất cả chạy bằng CPU (không cần GPU / payment method).

Layout trong Volume 'safe-gs-experiment':
  Replica/cam_params.json
  Replica/<scene>/traj.txt                  (đủ 2000 dòng pose, giữ nguyên chỉ số frame)
  Replica/<scene>/results/image/frameXXXXXX.jpg   (chỉ các frame i % stride == 0)
  Replica/<scene>/results/depth_image/depthXXXXXX.png
  sfm/<scene>/sparse/0/{cameras,images,points3D}.bin
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import modal

app = modal.App("safe-gs-experiment")

dataset_vol = modal.Volume.from_name("safe-gs-dataset", create_if_missing=True)
exp_vol = modal.Volume.from_name("safe-gs-experiment", create_if_missing=True)

repo_dir = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("numpy<2", "pillow", "pycolmap>=0.6.0")
    .add_local_file(
        repo_dir / "src" / "Module1-Perception" / "replica_to_colmap.py",
        "/workspace/replica_to_colmap.py",
    )
)

SRC_ROOT = Path("/data/Replica 8 Scene/Replica")
EXP_ROOT = Path("/exp")


@app.function(image=image, volumes={"/data": dataset_vol, "/exp": exp_vol}, timeout=1800)
def make_subset(scene: str = "office0", stride: int = 8):
    """Giữ frame i % stride == 0 (stride=8 -> ~250/2000 ảnh)."""
    dataset_vol.reload()
    src = SRC_ROOT / scene
    dst = EXP_ROOT / "Replica" / scene
    (dst / "results" / "image").mkdir(parents=True, exist_ok=True)
    (dst / "results" / "depth_image").mkdir(parents=True, exist_ok=True)

    shutil.copy2(SRC_ROOT / "cam_params.json", EXP_ROOT / "Replica" / "cam_params.json")
    shutil.copy2(src / "traj.txt", dst / "traj.txt")

    imgs = sorted((src / "results" / "image").glob("frame*.jpg"))
    print(f"[{scene}] Tổng {len(imgs)} ảnh gốc, giữ 1/{stride}")
    kept = 0
    for p in imgs:
        i = int(p.stem.replace("frame", ""))
        if i % stride:
            continue
        d = src / "results" / "depth_image" / f"depth{i:06d}.png"
        if not d.exists():
            print(f"  [WARN] thiếu depth cho frame {i}, bỏ qua")
            continue
        shutil.copy2(p, dst / "results" / "image" / p.name)
        shutil.copy2(d, dst / "results" / "depth_image" / d.name)
        kept += 1
    exp_vol.commit()
    print(f"[SUCCESS] Đã giữ {kept} cặp RGB+depth tại {dst}")


@app.function(image=image, volumes={"/exp": exp_vol}, timeout=1800)
def build_sfm(scene: str = "office0", stride: int = 8):
    """Dựng COLMAP model từ pose GT cho đúng các frame đã lọc."""
    import subprocess

    exp_vol.reload()
    out = EXP_ROOT / "sfm" / scene
    cmd = [
        sys.executable, "/workspace/replica_to_colmap.py",
        "--scene", scene,
        "--replica_root", str(EXP_ROOT / "Replica"),
        "--output_dir", str(out),
        "--camera_stride", str(stride),
        "--point_cloud_stride_frames", str(stride * 5),
    ]
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    exp_vol.commit()
    print(f"[SUCCESS] COLMAP model tại {out}/sparse/0")


@app.function(image=image, volumes={"/exp": exp_vol}, timeout=300)
def show():
    exp_vol.reload()
    for root, dirs, files in os.walk(EXP_ROOT):
        rel = os.path.relpath(root, EXP_ROOT)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > 3:
            continue
        print(f"{'  ' * depth}[D] {os.path.basename(root) if rel != '.' else '/'}  ({len(files)} files)")
        for f in sorted(files)[:5] if depth >= 3 else sorted(files)[:10]:
            print(f"{'  ' * (depth + 1)}- {f} ({os.path.getsize(os.path.join(root, f)) / 1024:.1f} kb)")
