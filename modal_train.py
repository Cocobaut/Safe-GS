"""
Modal GPU Training Runner cho Safe-GS (modal_train.py)
======================================================
File chuyên trách thực thi các tác vụ huấn luyện trên Cloud GPU (A10G, 24GB VRAM):
  1. train_3dgs_sample: Huấn luyện 3D Gaussian Splatting mẫu cho Scene 1 (office0).
  2. run_gof_sample: Huấn luyện Gaussian Opacity Fields (GOF) và trích xuất Mesh tam giác cho Scene 1.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Đảm bảo Windows console in tiếng Việt UTF-8 không bị lỗi charmap
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import modal

# -----------------------------------------------------------------------------
# 1. Khởi tạo App & Kết nối Persistent Volumes
# -----------------------------------------------------------------------------
APP_NAME = "safe-gs"
app = modal.App(APP_NAME)

dataset_vol = modal.Volume.from_name("safe-gs-dataset", create_if_missing=True)
outputs_vol = modal.Volume.from_name("safe-gs-outputs", create_if_missing=True)

# -----------------------------------------------------------------------------
# 2. Xây dựng Môi trường Docker GPU (CUDA 12.1 Devel + 3DGS & GOF)
# -----------------------------------------------------------------------------
cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install(
        "git",
        "cmake",
        "build-essential",
        "ninja-build",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "ffmpeg",
        "colmap",
    )
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        "torchaudio==2.1.2",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "numpy<2",
        "scipy>=1.11.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "opencv-python-headless>=4.8.0",
        "pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "open3d>=0.18.0",
        "trimesh>=4.0.0",
        "plyfile>=1.0.0",
        "mujoco>=3.0.0",
        "imageio>=2.31.0",
        "imageio-ffmpeg>=0.4.9",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "rich>=13.0.0",
        "tomli>=2.0.1",
        "pycolmap>=0.6.0",
        "kornia>=0.7.1",
        "einops>=0.7.0",
        "transformers>=4.36.0",
        "timm>=0.9.0",
    )
    .run_commands(
        "pip install --no-build-isolation git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git",
        "pip install --no-build-isolation git+https://gitlab.inria.fr/bkerbl/simple-knn.git",
    )
    .env({
        "PYTHONPATH": "/workspace/Safe-GS:/workspace/Safe-GS/Based_Model/GOF",
        "DATASET_ROOT": "/data",
        "DATASET_CONFIGS_DIR": "/workspace/Dataset Configs",
        "PYTHONNOUSERSITE": "1",
        "CUDA_HOME": "/usr/local/cuda",
        "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0",
    })
)

# -----------------------------------------------------------------------------
# 3. Mounts & Ánh xạ Volume
# -----------------------------------------------------------------------------
repo_dir = Path(__file__).resolve().parent

code_mount = modal.Mount.from_local_dir(
    repo_dir,
    remote_path="/workspace/Safe-GS",
    condition=lambda p: not any(
        x in p for x in [".venv", ".git", "outputs", "tmp", "__pycache__", ".ipynb_checkpoints"]
    ),
)

dataset_configs_dir = repo_dir.parent / "Dataset Configs"
mount_list = [code_mount]
if dataset_configs_dir.exists():
    mount_list.append(
        modal.Mount.from_local_dir(
            dataset_configs_dir,
            remote_path="/workspace/Dataset Configs",
        )
    )

VOLUMES_MAPPING = {
    "/data": dataset_vol,
    "/workspace/Safe-GS/outputs": outputs_vol,
}


def _run_shell(cmd: str, cwd: str = "/workspace/Safe-GS") -> int:
    """Chạy lệnh shell và in log thời gian thực."""
    print(f"\n[Modal GPU] CWD: {cwd}")
    print(f"[Modal GPU] RUN: {cmd}\n" + "-" * 60)
    process = subprocess.Popen(
        cmd,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if process.stdout:
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
    process.wait()
    print("-" * 60 + f"\n[Modal GPU] EXIT CODE: {process.returncode}\n")
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {cmd}")
    return process.returncode


# -----------------------------------------------------------------------------
# 4. Các Hàm Huấn Luyện trên GPU
# -----------------------------------------------------------------------------

@app.function(
    image=cuda_image,
    gpu="A10G",
    volumes=VOLUMES_MAPPING,
    mounts=mount_list,
    timeout=300,
)
def verify_training_paths(scene: str = "office0"):
    """
    Hàm kiểm tra toàn diện môi trường GPU, đường dẫn dữ liệu và quyền truy cập trước khi train.
    Chạy trong 10-20 giây để phát hiện sớm bất kỳ lỗi thiếu file hoặc sai đường dẫn nào.
    """
    import torch
    dataset_vol.reload()
    outputs_vol.reload()

    print("=" * 60)
    print(f">>> KIỂM TRA MÔI TRƯỜNG & ĐƯỜNG DẪN DỮ LIỆU (SCENE: {scene}) <<<")
    print("=" * 60)

    # 1. Kiểm tra GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[✓] GPU: {gpu_name} ({vram_gb:.2f} GB VRAM) - CUDA {torch.version.cuda}")
    else:
        print("[!] Không tìm thấy GPU CUDA!")

    # 2. Kiểm tra đường dẫn Dataset Volume (/data)
    replica_root = Path("/data/Replica 8 Scene/Replica")
    print(f"\n[1] Kiểm tra Dataset Volume: {replica_root}")
    if not replica_root.exists():
        print(f"    [X] Thư mục gốc không tồn tại: {replica_root}")
    else:
        print(f"    [✓] Thư mục gốc tồn tại.")

    cam_params = replica_root / "cam_params.json"
    print(f"    {'[✓]' if cam_params.exists() else '[X]'} cam_params.json: {cam_params}")

    scene_dir = replica_root / scene
    print(f"    {'[✓]' if scene_dir.exists() else '[X]'} Thư mục scene '{scene}': {scene_dir}")

    traj_file = scene_dir / "traj.txt"
    print(f"    {'[✓]' if traj_file.exists() else '[X]'} File trajectory: {traj_file}")

    img_dir = scene_dir / "results" / "image"
    depth_dir = scene_dir / "results" / "depth_image"
    n_imgs = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
    n_depths = len(list(depth_dir.glob("*.png"))) if depth_dir.exists() else 0
    print(f"    {'[✓]' if n_imgs > 0 else '[X]'} Ảnh RGB: {n_imgs} file ({img_dir})")
    print(f"    {'[✓]' if n_depths > 0 else '[X]'} Ảnh Depth: {n_depths} file ({depth_dir})")

    # 3. Kiểm tra Outputs Volume (/workspace/Safe-GS/outputs)
    outputs_path = Path("/workspace/Safe-GS/outputs")
    print(f"\n[2] Kiểm tra Outputs Volume: {outputs_path}")
    print(f"    {'[✓]' if outputs_path.exists() else '[X]'} Outputs path tồn tại.")
    if outputs_path.exists():
        sub_items = [p.name for p in outputs_path.iterdir()]
        print(f"    [✓] Nội dung hiện tại trong outputs: {sub_items}")

    # 4. Kiểm tra mã nguồn và các thư viện cần thiết
    print("\n[3] Kiểm tra Import Modules và C++ CUDA Extensions:")
    try:
        import diff_gaussian_rasterization
        print("    [✓] diff_gaussian_rasterization: OK")
    except Exception as e:
        print(f"    [X] diff_gaussian_rasterization: Lỗi ({e})")

    try:
        import simple_knn
        print("    [✓] simple_knn: OK")
    except Exception as e:
        print(f"    [X] simple_knn: Lỗi ({e})")

    try:
        from src.module2_scene_gs.scene_branch.scene_trainer import SceneGSTrainer
        print("    [✓] Safe-GS SceneGSTrainer import: OK")
    except Exception as e:
        print(f"    [X] Safe-GS SceneGSTrainer import: Lỗi ({e})")

    # 5. Kiểm tra Based_Model/GOF
    gof_train = Path("/workspace/Safe-GS/Based_Model/GOF/train.py")
    gof_extract = Path("/workspace/Safe-GS/Based_Model/GOF/extract_mesh.py")
    print(f"\n[4] Kiểm tra thư mục Based_Model/GOF:")
    print(f"    {'[✓]' if gof_train.exists() else '[X]'} GOF train.py: {gof_train}")
    print(f"    {'[✓]' if gof_extract.exists() else '[X]'} GOF extract_mesh.py: {gof_extract}")

    print("\n" + "=" * 60)
    print("[SUCCESS] Hoàn tất kiểm tra đường dẫn và môi trường!")
    print("=" * 60 + "\n")


@app.function(
    image=cuda_image,
    gpu="A10G",
    volumes=VOLUMES_MAPPING,
    mounts=mount_list,
    timeout=3600 * 3,  # Tối đa 3 tiếng
)
def train_3dgs_sample(scene: str = "office0", iterations: int = 30000):
    """
    Huấn luyện 3D Gaussian Splatting mẫu cho Scene 1 (mặc định: office0).
    Quy trình:
      1. Đọc ảnh từ /data/Replica 8 Scene/Replica/<scene>
      2. Tạo COLMAP SfM model giả nếu chưa có (outputs/Replica/sfm/<scene>/sparse/0)
      3. Huấn luyện 3DGS 30,000 iterations -> lưu checkpoint vào outputs/Replica/3dgs/<scene>/
      4. Tự động render video xoay 360 orbit để kiểm tra chất lượng
    """
    dataset_vol.reload()
    outputs_vol.reload()

    replica_root = "/data/Replica 8 Scene/Replica"
    sfm_dir = f"outputs/Replica/sfm/{scene}"
    sparse_dir = f"{sfm_dir}/sparse/0"
    raw_images = f"{replica_root}/{scene}/results/image"
    model_dir = f"outputs/Replica/3dgs/{scene}"

    print(f"\n{'=' * 60}\n>>> BẮT ĐẦU TRAIN 3DGS MẪU CHO SCENE 1: {scene} <<<\n{'=' * 60}\n")

    # Bước 1: Chuẩn bị SfM nếu chưa có sẵn trong outputs
    if not (Path("/workspace/Safe-GS") / sparse_dir / "points3D.bin").exists():
        print(f"[*] Chưa có SfM trong outputs. Đang sinh COLMAP sparse model cho {scene}...")
        _run_shell(
            f"python src/Module1-Perception/replica_to_colmap.py "
            f"--scene {scene} --replica_root \"{replica_root}\" --output_dir \"{sfm_dir}\""
        )
        outputs_vol.commit()

    # Bước 2: Huấn luyện 3DGS 30k iterations
    print(f"[*] Huấn luyện 3DGS ({iterations} iterations)...")
    _run_shell(
        f"python src/Module2-3DGS/optimization/02_run_scene_gs.py "
        f"--sfm_dir \"{sparse_dir}\" "
        f"--raw_image_dir \"{raw_images}\" "
        f"--workspace_dir \"{model_dir}\" "
        f"--iterations {iterations}"
    )
    outputs_vol.commit()

    # Bước 3: Render video xoay 360 orbit
    ckpt_path = f"{model_dir}/scene_gs_30000.ply"
    orbit_dir = f"{model_dir}/orbit_30000"
    if (Path("/workspace/Safe-GS") / ckpt_path).exists():
        print(f"[*] Render video orbit 360...")
        _run_shell(
            f"python src/Module2-3DGS/render_gs.py "
            f"--checkpoint \"{ckpt_path}\" "
            f"--sfm_dir \"{sparse_dir}\" "
            f"--images_dir \"{raw_images}\" "
            f"--output_dir \"{orbit_dir}\" "
            f"--orbit --num_frames 90"
        )
        outputs_vol.commit()

    print(f"\n[✓] Hoàn thành Train 3DGS mẫu cho {scene}! Kết quả đã lưu an toàn trong outputs_vol.")


@app.function(
    image=cuda_image,
    gpu="A10G",
    volumes=VOLUMES_MAPPING,
    mounts=mount_list,
    timeout=3600 * 4,  # Tối đa 4 tiếng
)
def run_gof_sample(scene: str = "office0", iterations: int = 30000):
    """
    Chạy pipeline GOF (Gaussian Opacity Fields) đầy đủ cho Scene 1 (office0):
      1. Sinh SfM mật độ cao (stride_frames=15, stride_pixels=5) vào outputs/Replica/sfm/<scene>_gof
      2. Huấn luyện GOF bằng train.py của Based_Model/GOF (30,000 iterations)
      3. Trích xuất Mesh bằng extract_mesh.py của Based_Model/GOF (Marching Tetrahedra)
      4. Lưu Mesh kết quả vào outputs/Replica/mesh/<scene>_gof/
    """
    dataset_vol.reload()
    outputs_vol.reload()

    # Cài đặt tetra-triangulation nếu chưa có (cần cho Marching Tetrahedra của GOF)
    try:
        import tetranerf
    except ImportError:
        print("[*] Đang cài đặt GOF tetra-triangulation extension...")
        _run_shell("pip install -e Based_Model/GOF/submodules/tetra-triangulation")

    replica_root = "/data/Replica 8 Scene/Replica"
    sfm_dir = f"outputs/Replica/sfm/{scene}_gof"
    sparse_dir = f"{sfm_dir}/sparse/0"
    raw_images = f"{replica_root}/{scene}/results/image"
    model_dir = f"outputs/Replica/3dgs/{scene}_gof"
    mesh_dir = f"outputs/Replica/mesh/{scene}_gof"

    os.makedirs(f"/workspace/Safe-GS/{mesh_dir}", exist_ok=True)

    print(f"\n{'=' * 60}\n>>> BẮT ĐẦU CHẠY GOF (TRAIN + TRÍCH MESH) CHO SCENE 1: {scene} <<<\n{'=' * 60}\n")

    # Bước 1: Sinh SfM mật độ cao nếu chưa có
    if not (Path("/workspace/Safe-GS") / sparse_dir / "points3D.bin").exists():
        print(f"[*] Sinh SfM mật độ cao cho GOF...")
        _run_shell(
            f"python src/Module1-Perception/replica_to_colmap.py "
            f"--scene {scene} --replica_root \"{replica_root}\" --output_dir \"{sfm_dir}\" "
            f"--point_cloud_stride_frames 15 --point_cloud_stride_pixels 5"
        )
        outputs_vol.commit()

    # Bước 2: Huấn luyện GOF Gaussian Opacity Fields
    ckpt_30k = f"{model_dir}/point_cloud/iteration_{iterations}/point_cloud.ply"
    if not (Path("/workspace/Safe-GS") / ckpt_30k).exists():
        print(f"[*] Huấn luyện GOF 3DGS ({iterations} iterations)...")
        _run_shell(
            f"python -u train.py "
            f"-s /workspace/Safe-GS/{sfm_dir} "
            f"-i \"{raw_images}\" "
            f"-m /workspace/Safe-GS/{model_dir} "
            f"--data_device cpu",
            cwd="/workspace/Safe-GS/Based_Model/GOF",
        )
        outputs_vol.commit()
    else:
        print(f"[SKIP] Đã có checkpoint GOF {iterations} iters, bỏ qua bước train.")

    # Bước 3: Trích xuất Mesh bằng Marching Tetrahedra & Level Set Solver
    print(f"[*] Trích xuất Mesh bằng Marching Tetrahedra...")
    _run_shell(
        f"python -u extract_mesh.py "
        f"-s /workspace/Safe-GS/{sfm_dir} "
        f"-i \"{raw_images}\" "
        f"-m /workspace/Safe-GS/{model_dir} "
        f"--data_device cpu",
        cwd="/workspace/Safe-GS/Based_Model/GOF",
    )

    # Bước 4: Chuyển file mesh sinh ra sang thư mục outputs chuẩn
    fusion_dir = Path(f"/workspace/Safe-GS/{model_dir}/test/ours_{iterations}/fusion")
    dest_mesh_dir = Path(f"/workspace/Safe-GS/{mesh_dir}")
    if fusion_dir.exists():
        for mesh_file in fusion_dir.glob("mesh_binary_search_*.ply"):
            import shutil
            shutil.copy(mesh_file, dest_mesh_dir / mesh_file.name)
            print(f"[✓] Đã lưu mesh tại: {dest_mesh_dir / mesh_file.name}")

    outputs_vol.commit()
    print(f"\n[✓] Hoàn thành pipeline GOF cho Scene 1 ({scene})! Mesh đã được lưu trong outputs_vol.")


# -----------------------------------------------------------------------------
# 5. CLI Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== MODAL GPU TRAINING RUNNER CHO SAFE-GS ===")
    print("Vui lòng sử dụng lệnh 'python -m modal run modal_train.py::<function_name>' để chạy trên GPU:")
    print("  python -m modal run modal_train.py::verify_training_paths # Kiểm tra nhanh toàn bộ đường dẫn trên GPU")
    print("  python -m modal run modal_train.py::train_3dgs_sample     # Train 3DGS mẫu cho Scene 1 (office0)")
    print("  python -m modal run modal_train.py::run_gof_sample       # Train + Trích mesh GOF cho Scene 1 (office0)\n")
