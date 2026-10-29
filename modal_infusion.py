"""
Modal GPU Runner cho InFusion (modal_infusion.py)
=================================================
Chạy InFusion (Inpainting 3D Gaussians via Depth Completion from Diffusion Prior) trên Modal GPU,
chỉ inference: model depth-inpainting dùng checkpoint pretrained Johanan0528/Infusion (không train lại).

Pipeline gốc của tác giả gồm 3 stage chạy tay; file này nối lại thành pipeline tự động:
  0. prepare  : chuyển scene SPIn-NeRF (/data/SPIn-NeRF/{scene}) sang cấu trúc InFusion
                (images = images_4, seg = label 0/255, sparse/0) trong outputs/infusion/{scene}/data
  1. stage1   : train Gaussians "thiếu" (mask_training, bỏ vùng object) + render depth/c2w/intri
  2. stage2   : chọn 1 view, tạo mask (label dilate), inpaint ảnh 2D bằng SDXL-inpainting
                (hoặc ảnh bạn tự upload), rồi chạy depth inpainting -> point cloud vùng bị thiếu
  3. stage3   : compose Gaussians + fine-tune 150 iters trên ảnh đã inpaint + render kết quả

Lệnh:
  python -m modal run modal_infusion.py::verify_infusion
  python -m modal run modal_infusion.py::run_infusion --scene 1
  python -m modal run modal_infusion.py::infusion_stage3 --scene 1 --threshold 0.5   # chỉnh lại floaters
"""

from __future__ import annotations

import os
import shutil
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
# 1. App & Volumes (dùng chung volume với modal_train.py)
# -----------------------------------------------------------------------------
APP_NAME = "safe-gs-infusion"
app = modal.App(APP_NAME)

dataset_vol = modal.Volume.from_name("safe-gs-dataset", create_if_missing=True)
outputs_vol = modal.Volume.from_name("safe-gs-outputs", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("safe-gs-hf-cache", create_if_missing=True)

VOLUMES_MAPPING = {
    "/data": dataset_vol,
    "/workspace/Safe-GS/outputs": outputs_vol,
    "/cache": hf_cache_vol,
}

INFUSION_REMOTE = "/workspace/Infusion"
OUTPUTS_ROOT = "/workspace/Safe-GS/outputs/infusion"
CKPT_DIR = "/cache/infusion/checkpoint"
INFUSION_HF_REPO = "Johanan0528/Infusion"
SDXL_INPAINT_REPO = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

repo_dir = Path(__file__).resolve().parent
infusion_local = repo_dir / "Based_Model" / "Infusion"
submodules_local = infusion_local / "gaussian_splatting" / "submodules"

# -----------------------------------------------------------------------------
# 2. Image: CUDA 12.1 + torch 2.1.2 + diffusers 0.25.1 (đúng version lưu trong checkpoint unet/config.json)
#    Rasterizer phải là bản "confidence" của repo (trả về depth), KHÔNG dùng bản graphdeco gốc.
# -----------------------------------------------------------------------------
infusion_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "build-essential", "ninja-build", "libgl1", "libglib2.0-0", "libgomp1")
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "numpy==1.26.4",
        "plyfile==1.0.3",
        "opencv-python-headless==4.8.1.78",
        "matplotlib==3.7.5",
        "imageio==2.31.2",
        "open3d==0.18.0",
        "scikit-learn==1.3.2",
        "tqdm",
        "pillow==10.4.0",
        "diffusers==0.25.1",
        "transformers==4.36.2",
        "huggingface_hub==0.20.3",  # diffusers 0.25 cần cached_download (đã bị xóa từ hub 0.26)
        "accelerate==0.25.0",
        "safetensors==0.4.5",
        "wheel",
        "setuptools<70",
    )
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0",
        "MAX_JOBS": "4",
        "CC": "gcc",
        "CXX": "g++",
        "HF_HOME": "/cache/huggingface",
        "PYTHONUNBUFFERED": "1",
    })
    .add_local_dir(
        submodules_local,
        remote_path="/opt/infusion_submodules",
        copy=True,
        ignore=lambda p: any(x in str(p).replace("\\", "/") for x in ["/build/", ".egg-info", "glm/doc", "glm/test"]),
    )
    .run_commands(
        "pip install --no-build-isolation /opt/infusion_submodules/diff-gaussian-rasterization-confidence",
        "pip install --no-build-isolation /opt/infusion_submodules/simple-knn",
    )
    .add_local_dir(
        infusion_local,
        remote_path=INFUSION_REMOTE,
        copy=False,
        ignore=lambda p: any(
            x in str(p).replace("\\", "/")
            for x in ["checkpoint", "assets", "submodules", "__pycache__", ".git"]
        ) or "output" in Path(p).parts,  # kết quả tải về local, không upload lại
    )
)


def _run_shell(cmd: str, cwd: str = INFUSION_REMOTE) -> int:
    """Chạy lệnh shell và in log thời gian thực."""
    print(f"\n[Modal GPU] CWD: {cwd}")
    print(f"[Modal GPU] RUN: {cmd}\n" + "-" * 60)
    process = subprocess.Popen(
        cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
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


def _ensure_checkpoint() -> str:
    """Tải checkpoint InFusion về volume cache (chỉ safetensors, ~5.2 GB) nếu chưa có."""
    if (Path(CKPT_DIR) / "unet" / "diffusion_pytorch_model.safetensors").exists():
        return CKPT_DIR
    from huggingface_hub import snapshot_download

    print(f"[*] Tải checkpoint {INFUSION_HF_REPO} -> {CKPT_DIR} ...")
    snapshot_download(
        INFUSION_HF_REPO,
        local_dir=CKPT_DIR,
        allow_patterns=[
            "scheduler/*", "tokenizer/*", "unet/*",
            "text_encoder/config.json", "text_encoder/model.safetensors",
            "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
        ],
    )
    hf_cache_vol.commit()
    return CKPT_DIR


def _paths(scene: str) -> dict:
    root = Path(OUTPUTS_ROOT) / scene
    return {
        "root": root,
        "data": root / "data",
        "gs": root / "gs",
        "inpaint": root / "inpaint",
    }


def _find_spinnerf_scene(scene: str) -> Path:
    for root in ["/data/SPIn-NeRF", "/data/SPIn-NeRF/spinnerf-dataset"]:
        if (Path(root) / scene / "sparse").exists():
            return Path(root) / scene
    raise FileNotFoundError(
        f"Chưa có dữ liệu InFusion cho scene '{scene}'. SPIn-NeRF: python modal_sync.py upload-spinnerf --scene {scene}; "
        f"LERF: chạy modal_lerf_prep.py::lerf_colmap và ::lerf_sam2_masks trước."
    )


# -----------------------------------------------------------------------------
# 3. Các bước của pipeline (hàm thường, được gọi bởi các Modal function bên dưới)
# -----------------------------------------------------------------------------
def _prepare_spinnerf(scene: str) -> None:
    """SPIn-NeRF -> cấu trúc InFusion: images (images_4), seg (label 0/1 -> 0/255), sparse/0."""
    import numpy as np
    from PIL import Image

    p = _paths(scene)
    if (p["data"] / "seg").exists() and any((p["data"] / "seg").iterdir()):
        # Dữ liệu đã có sẵn (SPIn-NeRF đã chuẩn bị, hoặc LERF từ modal_lerf_prep.py)
        if not (p["data"] / "sparse" / "0").exists():
            raise FileNotFoundError(f"Thiếu {p['data']}/sparse/0. Với LERF chạy: python -m modal run modal_lerf_prep.py::lerf_colmap --scene {scene}")
        print(f"[SKIP] Đã có dữ liệu InFusion tại {p['data']}")
        return

    src = _find_spinnerf_scene(scene)
    print(f"[*] Chuẩn bị dữ liệu InFusion từ {src}")
    (p["data"] / "images").mkdir(parents=True, exist_ok=True)
    (p["data"] / "seg").mkdir(parents=True, exist_ok=True)
    shutil.copytree(src / "sparse", p["data"] / "sparse", dirs_exist_ok=True)

    for img in sorted((src / "images_4").glob("*.png")):
        shutil.copy(img, p["data"] / "images" / img.name)

    # Chỉ 60 view train có label (có object); 40 view test (không object) bị reader bỏ qua vì không có seg
    labels = sorted((src / "images_4" / "label").glob("*.png"))
    for lbl in labels:
        m = np.array(Image.open(lbl).convert("L"))
        m = np.where(m > 0, 255, 0).astype(np.uint8)
        Image.fromarray(m).save(p["data"] / "seg" / lbl.name)
    print(f"[✓] {len(list((p['data'] / 'images').iterdir()))} ảnh, {len(labels)} mask -> {p['data']}")


def _stage1(scene: str, iterations: int, color_aug: bool) -> None:
    p = _paths(scene)
    ckpt = p["gs"] / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    if not ckpt.exists():
        _run_shell(
            f"python -u train.py -s {p['data']} -m {p['gs']} -u nothing --mask_training "
            f"--iterations {iterations} {'--color_aug' if color_aug else ''}",
            cwd=f"{INFUSION_REMOTE}/gaussian_splatting",
        )
        outputs_vol.commit()
    else:
        print(f"[SKIP] Đã có Gaussians {iterations} iters: {ckpt}")

    render_dir = p["gs"] / "train" / f"ours_{iterations}" / "depth_dis"
    if not render_dir.exists() or not any(render_dir.iterdir()):
        _run_shell(
            f"python -u render.py -s {p['data']} -m {p['gs']} -u nothing --iteration {iterations} --skip_test",
            cwd=f"{INFUSION_REMOTE}/gaussian_splatting",
        )
        outputs_vol.commit()


def _default_view(scene: str) -> str:
    # LERF: view chọn trong scripts/lerf_select_object.py (ghi bởi modal_lerf_prep.py)
    selected = _paths(scene)["data"] / "selected_view.txt"
    if selected.exists() and selected.read_text().strip():
        return selected.read_text().strip()
    segs = sorted((_paths(scene)["data"] / "seg").glob("*"))
    if not segs:
        raise FileNotFoundError("Không có mask trong data/seg, hãy chạy stage1 trước.")
    return segs[0].stem


def _sdxl_inpaint(src_png: Path, mask_png: Path, out_png: Path, prompt: str, negative_prompt: str, seed: int) -> None:
    """Inpaint 1 ảnh 2D bằng SDXL-inpainting, giữ nguyên pixel ngoài mask."""
    import numpy as np
    import torch
    from diffusers import StableDiffusionXLInpaintPipeline
    from PIL import Image

    src = Image.open(src_png).convert("RGB")
    mask = Image.open(mask_png).convert("L")
    W, H = src.size
    scale = 1024 / max(W, H)
    w8, h8 = int(round(W * scale / 8) * 8), int(round(H * scale / 8) * 8)

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        SDXL_INPAINT_REPO, torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
    gen = torch.Generator("cuda").manual_seed(seed)
    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=src.resize((w8, h8), Image.BICUBIC),
        mask_image=mask.resize((w8, h8), Image.NEAREST),
        height=h8,
        width=w8,
        strength=0.99,
        guidance_scale=8.0,
        num_inference_steps=30,
        generator=gen,
    ).images[0].resize((W, H), Image.BICUBIC)
    del pipe
    torch.cuda.empty_cache()
    hf_cache_vol.commit()

    m = (np.array(mask) > 127)[..., None]
    composed = np.where(m, np.array(out), np.array(src)).astype(np.uint8)
    Image.fromarray(composed).save(out_png)
    print(f"[✓] Ảnh đã inpaint (SDXL): {out_png}")


def _stage2(
    scene: str, iterations: int, view: str, inpainted_rgb: str, prompt: str, negative_prompt: str,
    mask_dilate: int, denoise_steps: int, depth_loops: int, seed: int,
) -> str:
    import cv2
    import numpy as np

    p = _paths(scene)
    view = view or _default_view(scene)
    ours = p["gs"] / "train" / f"ours_{iterations}"
    render_png = ours / "renders" / f"{view}.png"
    if not render_png.exists():
        raise FileNotFoundError(f"Không thấy render của view '{view}': {render_png} (chạy stage1 trước)")
    p["inpaint"].mkdir(parents=True, exist_ok=True)
    print(f"[i] View được chọn để inpaint: {view}")

    # Mask: label của view, dilate để phủ hết vùng bị thiếu (README: không cần chính xác nhưng phải phủ hết)
    seg = next((p["data"] / "seg").glob(f"{view}.*"))
    render = cv2.imread(str(render_png))
    m = cv2.imread(str(seg), cv2.IMREAD_GRAYSCALE)
    m = cv2.resize(m, (render.shape[1], render.shape[0]), interpolation=cv2.INTER_NEAREST)
    m = (m > 127).astype(np.uint8) * 255
    if mask_dilate > 0:
        m = cv2.dilate(m, np.ones((mask_dilate, mask_dilate), np.uint8), iterations=1)
    mask_png = p["inpaint"] / "mask.png"
    cv2.imwrite(str(mask_png), m)

    # Ảnh 2D đã inpaint: tên file phải là <view>.png để output là <view>_mask.ply
    rgb_png = p["inpaint"] / f"{view}.png"
    if inpainted_rgb:
        src = Path(inpainted_rgb)
        if not src.is_absolute():
            src = Path("/workspace/Safe-GS/outputs") / inpainted_rgb
        img = cv2.imread(str(src))
        if img.shape[:2] != render.shape[:2]:
            img = cv2.resize(img, (render.shape[1], render.shape[0]), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(rgb_png), img)
        print(f"[✓] Dùng ảnh inpaint do bạn cung cấp: {src}")
    else:
        _sdxl_inpaint(render_png, mask_png, rgb_png, prompt, negative_prompt, seed)
    outputs_vol.commit()

    ckpt = _ensure_checkpoint()
    depth_in = ours / "depth_dis" / f"{view}.npy"
    for i in range(max(1, depth_loops)):
        print(f"[*] Depth inpainting lần {i + 1}/{depth_loops} (input depth: {depth_in})")
        _run_shell(
            f"python -u run_inference_inpainting.py "
            f"--input_rgb_path {rgb_png} --input_mask {mask_png} --input_depth_path {depth_in} "
            f"--model_path {ckpt} --output_dir {p['inpaint']} --denoise_steps {denoise_steps} "
            f"--intri {ours / 'intri' / f'{view}.npy'} --c2w {ours / 'c2w' / f'{view}.npy'} "
            f"--use_mask --blend --seed {seed}",
            cwd=f"{INFUSION_REMOTE}/depth_inpainting/run",
        )
        # README tip: dùng lại depth vừa inpaint làm input để tinh chỉnh thêm
        looped = p["inpaint"] / f"{view}_depth_dis_loop{i + 1}.npy"
        shutil.copy(p["inpaint"] / f"{view}_depth_dis.npy", looped)
        depth_in = looped
    outputs_vol.commit()
    print(f"[✓] Point cloud vùng inpaint: {p['inpaint'] / f'{view}_mask.ply'}")
    return view


def _stage3(
    scene: str, iterations: int, view: str, nb_points: int, threshold: float, radius: float, finetune_iters: int,
) -> dict:
    p = _paths(scene)
    view = view or _default_view(scene)
    supp_ply = p["inpaint"] / f"{view}_mask.ply"
    rgb_png = p["inpaint"] / f"{view}.png"
    if not supp_ply.exists():
        raise FileNotFoundError(f"Thiếu {supp_ply} (chạy stage2 trước)")

    composed_iter = iterations + 1
    _run_shell(
        f"python -u compose.py "
        f"--original_ply {p['gs']}/point_cloud/iteration_{iterations}/point_cloud.ply "
        f"--supp_ply {supp_ply} "
        f"--save_ply {p['gs']}/point_cloud/iteration_{composed_iter}/point_cloud.ply "
        f"--temp_ply {p['inpaint']}/temp.ply "
        f"--nb_points {nb_points} --threshold {threshold} --radius {radius}",
    )

    # -u phải khớp tên file ảnh của view trong data/images (images_4 là .png)
    image_file = next((p["data"] / "images").glob(f"{view}.*")).name
    _run_shell(
        f"python -u train.py -s {p['data']} -m {p['gs']} -u {image_file} -n {rgb_png} "
        f"--load_iteration {composed_iter} --iterations {finetune_iters}",
        cwd=f"{INFUSION_REMOTE}/gaussian_splatting",
    )
    _run_shell(
        f"python -u render.py -s {p['data']} -m {p['gs']} -u nothing --iteration {finetune_iters} --skip_test",
        cwd=f"{INFUSION_REMOTE}/gaussian_splatting",
    )
    outputs_vol.commit()
    result = {
        "final_ply": f"{p['gs']}/point_cloud/iteration_{finetune_iters}/point_cloud.ply",
        "renders": f"{p['gs']}/train/ours_{finetune_iters}/renders",
    }
    print(f"[✓] Hoàn tất InFusion cho scene {scene}: {result}")
    return result


# -----------------------------------------------------------------------------
# 4. Modal functions
# -----------------------------------------------------------------------------
_SD_PROMPT = "empty background, seamless natural continuation of the surrounding scene, photorealistic"
_SD_NEG = "object, person, animal, text, watermark, blurry, artifacts"


@app.function(image=infusion_image, gpu="L4", volumes=VOLUMES_MAPPING, timeout=1800)
def verify_infusion(scene: str = "1"):
    """Kiểm tra GPU, CUDA extensions, checkpoint và dữ liệu scene (~1-5 phút lần đầu do tải checkpoint)."""
    import torch

    print(f"[✓] GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | CUDA {torch.version.cuda}")
    from diff_gaussian_rasterization import GaussianRasterizationSettings  # noqa: F401
    from simple_knn._C import distCUDA2  # noqa: F401
    import diffusers

    print(f"[✓] diff-gaussian-rasterization-confidence + simple_knn OK | diffusers {diffusers.__version__}")

    ckpt = _ensure_checkpoint()
    from diffusers import UNet2DConditionModel

    unet = UNet2DConditionModel.from_pretrained(ckpt, subfolder="unet", torch_dtype=torch.float16)
    print(f"[✓] Checkpoint OK: unet in_channels={unet.config.in_channels} ({ckpt})")

    dataset_vol.reload()
    outputs_vol.reload()
    data = _paths(scene)["data"]
    if (data / "seg").exists() and (data / "sparse" / "0").exists():
        n_img = len(list((data / "images").glob("*")))
        n_seg = len(list((data / "seg").glob("*")))
        print(f"[✓] Dữ liệu InFusion đã sẵn sàng: {data} ({n_img} ảnh, {n_seg} mask, view: {_default_view(scene)})")
        return
    try:
        print(f"[✓] Dataset scene: {_find_spinnerf_scene(scene)}")
    except FileNotFoundError as e:
        print(f"[X] {e}")


@app.function(image=infusion_image, gpu="L4", volumes=VOLUMES_MAPPING, timeout=3600 * 2)
def infusion_stage1(scene: str = "1", iterations: int = 30000, color_aug: bool = False):
    """Stage 0+1: chuẩn bị dữ liệu, train Gaussians thiếu (mask_training) và render depth/c2w/intri."""
    dataset_vol.reload()
    outputs_vol.reload()
    _prepare_spinnerf(scene)
    outputs_vol.commit()
    _stage1(scene, iterations, color_aug)


@app.function(image=infusion_image, gpu="L4", volumes=VOLUMES_MAPPING, timeout=3600)
def infusion_stage2(
    scene: str = "1",
    iterations: int = 30000,
    view: str = "",
    inpainted_rgb: str = "",
    prompt: str = _SD_PROMPT,
    negative_prompt: str = _SD_NEG,
    mask_dilate: int = 25,
    denoise_steps: int = 20,
    depth_loops: int = 1,
    seed: int = 0,
):
    """Stage 2: mask + inpaint ảnh 2D (SDXL hoặc ảnh tự cung cấp) + depth inpainting -> <view>_mask.ply."""
    outputs_vol.reload()
    return _stage2(scene, iterations, view, inpainted_rgb, prompt, negative_prompt,
                   mask_dilate, denoise_steps, depth_loops, seed)


@app.function(image=infusion_image, gpu="L4", volumes=VOLUMES_MAPPING, timeout=3600)
def infusion_stage3(
    scene: str = "1",
    iterations: int = 30000,
    view: str = "",
    nb_points: int = 100,
    threshold: float = 1.0,
    radius: float = 0.1,
    finetune_iters: int = 150,
):
    """Stage 3: compose + fine-tune trên ảnh đã inpaint + render. Chạy lại được để chỉnh nb_points/threshold."""
    outputs_vol.reload()
    return _stage3(scene, iterations, view, nb_points, threshold, radius, finetune_iters)


@app.function(image=infusion_image, gpu="L4", volumes=VOLUMES_MAPPING, timeout=3600 * 3)
def run_infusion(
    scene: str = "1",
    iterations: int = 30000,
    color_aug: bool = False,
    view: str = "",
    inpainted_rgb: str = "",
    prompt: str = _SD_PROMPT,
    mask_dilate: int = 25,
    depth_loops: int = 1,
    nb_points: int = 100,
    threshold: float = 1.0,
    finetune_iters: int = 150,
    seed: int = 0,
):
    """Chạy toàn bộ pipeline InFusion (stage 0 -> 3) trong 1 container."""
    dataset_vol.reload()
    outputs_vol.reload()
    print("=" * 60 + f"\n>>> INFUSION TRÊN MODAL GPU (SCENE: {scene}) <<<\n" + "=" * 60)
    _prepare_spinnerf(scene)
    outputs_vol.commit()
    _stage1(scene, iterations, color_aug)
    view = _stage2(scene, iterations, view, inpainted_rgb, prompt, _SD_NEG, mask_dilate, 20, depth_loops, seed)
    return _stage3(scene, iterations, view, nb_points, threshold, 0.1, finetune_iters)


# -----------------------------------------------------------------------------
# 5. Trích xuất mesh từ Gaussians InFusion bằng logic GOF (Based_Model/Infusion/mesh)
#    Image riêng: rasterizer của GOF (có integrate) trùng tên package với rasterizer "confidence" của InFusion,
#    và cần CGAL để build tetra-triangulation.
# -----------------------------------------------------------------------------
mesh_local = infusion_local / "mesh"
MESH_REMOTE = f"{INFUSION_REMOTE}/mesh"

mesh_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "build-essential", "cmake", "ninja-build", "libcgal-dev", "libgmp-dev", "libmpfr-dev",
                 "libboost-dev", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.1.2", "torchvision==0.16.2", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("numpy==1.26.4", "plyfile==1.0.3", "trimesh==4.4.9", "einops==0.8.0", "tqdm", "wheel", "setuptools<70")
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0",
        "MAX_JOBS": "4",
        "CC": "gcc",
        "CXX": "g++",
        "PYTHONUNBUFFERED": "1",
    })
    .add_local_dir(mesh_local / "submodules", remote_path="/opt/gof_submodules", copy=True,
                   ignore=lambda p: any(x in str(p).replace("\\", "/") for x in ["/build/", ".egg-info", "__pycache__"]))
    .run_commands(
        "pip install --no-build-isolation /opt/gof_submodules/diff-gaussian-rasterization",
        "cd /opt/gof_submodules/tetra-triangulation && export CPLUS_INCLUDE_PATH=/usr/local/cuda/include && "
        "cmake . -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)') && "
        "make -j8 && pip install -e .",
        "python -c \"from tetranerf.utils.extension import cpp; import diff_gaussian_rasterization; print('GOF rasterizer + tetra-triangulation OK')\"",
    )
    .add_local_dir(mesh_local, remote_path=MESH_REMOTE, copy=False,
                   ignore=lambda p: "submodules" in Path(p).parts or "__pycache__" in Path(p).parts)
)


@app.function(image=mesh_image, gpu="L4", cpu=8, memory=65536, volumes=VOLUMES_MAPPING, timeout=3600 * 4)
def infusion_extract_mesh(
    scene: str = "teatime",
    iteration: int = 150,
    texture_mesh: bool = True,
    filter_mesh: bool = False,
    view_stride: int = 1,
    opacity_threshold: float = 0.1,
    use_3d_filter: bool = False,
):
    """Trích xuất mesh (GOF marching tetrahedra + binary search) từ Gaussians InFusion cuối cùng."""
    outputs_vol.reload()
    p = _paths(scene)
    ply = p["gs"] / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not ply.exists():
        raise FileNotFoundError(f"Không thấy {ply} (chạy infusion_stage3 trước)")
    mesh_dir = p["root"] / "mesh"
    _run_shell(
        f"python -u extract_mesh.py -m {p['gs']} --iteration {iteration} --output_dir {mesh_dir} "
        f"--view_stride {view_stride} --opacity_threshold {opacity_threshold}"
        f"{' --texture_mesh' if texture_mesh else ''}{' --filter_mesh' if filter_mesh else ''}"
        f"{' --use_3d_filter' if use_3d_filter else ''}",
        cwd=MESH_REMOTE,
    )
    outputs_vol.commit()
    # extract_mesh.py không ghi đè mesh cũ (mesh_gof.ply -> mesh_gof_v2.ply ...): trả về file mới nhất
    mesh = max(mesh_dir.glob("mesh_gof*.ply"), key=lambda f: f.stat().st_mtime)
    return {"mesh": str(mesh), "size_mb": round(mesh.stat().st_size / 1024**2, 1)}


if __name__ == "__main__":
    print("\n=== MODAL GPU RUNNER CHO INFUSION ===")
    print("  python -m modal run modal_infusion.py::verify_infusion --scene 1")
    print("  python -m modal run modal_infusion.py::run_infusion --scene 1")
    print("  python -m modal run modal_infusion.py::infusion_stage1 --scene 1")
    print("  python -m modal run modal_infusion.py::infusion_stage2 --scene 1 [--view <name>] [--inpainted-rgb <path>]")
    print("  python -m modal run modal_infusion.py::infusion_stage3 --scene 1 [--threshold 0.5 --nb-points 50]\n")
