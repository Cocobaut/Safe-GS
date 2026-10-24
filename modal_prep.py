"""
Modal runner: chuẩn bị đầu vào cho Split&Splat trên LERF  (COLMAP -> Murre depth -> 3DGS dày)  -> <scene>_prep.zip
==================================================================================================================
Dùng lại ĐÚNG code các hàm trong notebooks/split_and_splat_prep_colmap_murre_3dgs.ipynb (exec các cell hàm), nên kết quả
giống bản Colab. Dữ liệu/checkpoint nằm trong Volume 'safe-gs-prep' (tiếp tục được nếu bị ngắt).

Các bước (chạy từ thư mục Safe-GS):
    python -m modal run modal_prep.py::fetch_assets          # CPU, rẻ: tải Murre-ckpt, checkpoint cũ, zip LERF vào Volume
    python -m modal run modal_prep.py::check_env             # CPU, rẻ: kiểm tra môi trường trước khi tốn GPU
    python -m modal run --detach modal_prep.py::prep_scene --scene figurines     # GPU L4 (trần chi phí theo timeout)
    python -m modal volume get safe-gs-prep out/figurines_prep.zip E:\\Safe-GS\\Data\\      # lấy kết quả về máy

Hàm kết thúc là container tắt (không tính tiền thêm).
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import modal

app = modal.App("safe-gs-prep")
vol = modal.Volume.from_name("safe-gs-prep", create_if_missing=True)
VOL = "/vol"
PREP = f"{VOL}/prep"

REPO_DIR = Path(__file__).resolve().parent
NOTEBOOK = REPO_DIR / "notebooks" / "split_and_splat_prep_colmap_murre_3dgs.ipynb"

SS_URL, SS_REF = "https://github.com/LTTM/Split_and_Splat.git", "9b7ba906511f75b87c438b4cf30ae3d1d5302f61"
MURRE_URL, MURRE_REF = "https://github.com/zju3dv/Murre.git", "1c759de92e9f323e4565c7506121df47c1d457ce"

# file zip bạn đã đặt chế độ "Anyone with the link" trên Drive của bạn + 2 zip LERF của tác giả
MURRE_CKPT_ID = "11hyjtfZhDcVK2l0tAf9OhxH_Uu1JuvFY"
SEED_ID = "1cPFOvfkLvqKY09yo098ndka_PFfnhYzq"
LERF_IDS = ["1t08_1xV1h5pCdU6-oKRP49veUjxUSM_8", "1LqmUsixkDkw7GIWQX0ZmJxXjFdMMlPjc"]

# ---------------------------------------------------------------------------------------------
# Image nhẹ (CPU) để tải dữ liệu
# ---------------------------------------------------------------------------------------------
fetch_image = modal.Image.debian_slim(python_version="3.12").pip_install("gdown")

# ---------------------------------------------------------------------------------------------
# Image đầy đủ: giống Colab (Ubuntu 24.04, CUDA 12.6, venv py3.12 cho 3DGS, venv py3.10 cho Murre)
# ---------------------------------------------------------------------------------------------
image_core = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "wget", "unzip", "build-essential", "ninja-build", "libgl1", "libglib2.0-0", "libgomp1")
    .pip_install("uv", "gdown")
    .env({
        "CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "8.9", "FORCE_CUDA": "1", "MAX_JOBS": "4",
        "CC": "gcc", "CXX": "g++", "LDSHARED": "g++ -shared",
        "PATH": "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
    })
    # --- mã nguồn ---
    .run_commands(
        f"git clone --recursive {SS_URL} /opt/Split_and_Splat && git -C /opt/Split_and_Splat checkout -q {SS_REF} "
        f"&& git -C /opt/Split_and_Splat submodule update --init --recursive -q",
        f"git clone {MURRE_URL} /opt/Murre && git -C /opt/Murre checkout -q {MURRE_REF}",
    )
    # --- venv 3.12: COLMAP tools + 3DGS (giống ô 'Môi trường' trong notebook) ---
    .run_commands(
        "uv venv --python 3.12 /opt/venv312",
        "uv pip install --python /opt/venv312/bin/python torch==2.5.1 torchvision==0.20.1",
        "uv pip install --python /opt/venv312/bin/python numpy==2.1.3 pillow==11.1.0 scipy==1.15.2 scikit-learn==1.6.1 "
        "matplotlib==3.10.1 opencv-python-headless==4.11.0.86 open3d==0.19.0 pycolmap==3.11.1 joblib==1.4.2 pandas==2.2.3 "
        "plyfile tqdm",
        "uv pip install --python /opt/venv312/bin/python setuptools wheel ninja",
    )
    .run_commands(
        "for s in diff-gaussian-rasterization simple-knn fused-ssim; do "
        "uv pip install --python /opt/venv312/bin/python /opt/Split_and_Splat/submodules/$s --no-build-isolation || exit 1; done",
    )
    # --- venv 3.10: Murre (README Murre: torch 2.0.1 + cu118) ---
    .run_commands(
        "uv venv --python 3.10 /opt/venv310",
        "uv pip install --python /opt/venv310/bin/python 'torch==2.0.1+cu118' 'torchvision==0.15.2+cu118' "
        "--extra-index-url https://download.pytorch.org/whl/cu118 --index-strategy unsafe-best-match",
        "uv pip install --python /opt/venv310/bin/python numpy==1.26.0 opencv-python-headless==4.8.0.76 tqdm==4.66.1 trimesh==4.0.2 "
        "pillow==11.1.0 matplotlib==3.8.3 scikit-learn==1.4.1.post1 diffusers==0.27.2 transformers==4.39.1 scipy==1.11.2 "
        "typing-extensions==4.10.0 huggingface_hub==0.25.0 open3d==0.18.0",
    )
    .pip_install("numpy")                                  # step_murre của notebook dùng numpy ngay trong tiến trình chính
)
image = image_core.add_local_file(NOTEBOOK, "/opt/prep_notebook.ipynb")

# Image cho bước lan truyền mask: thêm SAM 2 (đúng commit notebook), point_projection (README mục 5), checkpoint SAM 2
SAM2_GIT_REF = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
NOTEBOOK_PROP = REPO_DIR / "notebooks" / "split_and_splat_stage1_step2-4_propagation.ipynb"
prop_image = (
    image_core
    .run_commands(
        "uv pip install --python /opt/venv312/bin/python hydra-core==1.3.2 iopath==0.1.10 omegaconf==2.3.0",
        f"SAM2_BUILD_CUDA=0 uv pip install --python /opt/venv312/bin/python "
        f"'git+https://github.com/facebookresearch/sam2.git@{SAM2_GIT_REF}' --no-build-isolation",
        "uv pip install --python /opt/venv312/bin/python -e /opt/Split_and_Splat/point_projection --no-build-isolation",
        "mkdir -p /opt/Split_and_Splat/checkpoints && wget -q -O /opt/Split_and_Splat/checkpoints/sam2.1_hiera_large.pt "
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
    )
    .add_local_file(NOTEBOOK, "/opt/prep_notebook.ipynb")
    .add_local_file(NOTEBOOK_PROP, "/opt/prop_notebook.ipynb")
)


def _commit_loop(stop: threading.Event):
    """Commit Volume định kỳ để lỡ bị ngắt vẫn còn checkpoint."""
    while not stop.wait(120):
        try:
            vol.commit()
        except Exception as e:
            print(f"[commit] {e}", flush=True)


# ---------------------------------------------------------------------------------------------
# 1) Tải dữ liệu vào Volume (CPU)
# ---------------------------------------------------------------------------------------------
@app.function(image=fetch_image, volumes={VOL: vol}, timeout=2 * 3600, cpu=2, memory=8192)
def fetch_assets(force: bool = False):
    import gdown

    p = Path(PREP); p.mkdir(parents=True, exist_ok=True)

    def get(fid: str, dst: Path):
        if dst.exists() and dst.stat().st_size > 1_000_000 and not force:
            print(f"đã có {dst} ({dst.stat().st_size / 1e6:.0f} MB)", flush=True); return
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"tải {dst.name} từ Drive ...", flush=True)
        out = gdown.download(id=fid, output=str(dst), quiet=True)
        if not out or not dst.exists():
            raise RuntimeError(f"gdown lỗi cho {dst.name} (id {fid}): kiểm tra chế độ chia sẻ 'Anyone with the link' / quota")
        print(f"  xong {dst.name}: {dst.stat().st_size / 1e6:.0f} MB", flush=True)

    # Murre checkpoint
    ck = p / "murre_ckpt"
    if not any(ck.rglob("model_index.json")) or force:
        z = p / "downloads" / "Murre-ckpt.zip"
        get(MURRE_CKPT_ID, z)
        ck.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(ck)
        z.unlink()                                                    # đỡ chiếm chỗ trong Volume
        vol.commit()
    print("Murre ckpt:", [str(x.parent) for x in ck.rglob("model_index.json")][:1], flush=True)

    # checkpoint cũ (150 depth + colmap.zip của figurines): giải nén vào seed/ckpt/<scene>/...
    seed = p / "seed"
    if not any(seed.rglob("colmap.zip")) or force:
        z = p / "downloads" / "ckpt_seed.zip"
        get(SEED_ID, z)
        seed.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            assert zf.testzip() is None, "ckpt seed hỏng"
            zf.extractall(seed)
        z.unlink()
        vol.commit()
    print("seed:", sorted(str(x.relative_to(seed)) for x in seed.rglob("*") if x.is_dir())[:6],
          "| depth:", len(list(seed.rglob("*_pred.npy"))), flush=True)

    # zip LERF
    for i, fid in enumerate(LERF_IDS):
        get(fid, p / "lerf" / f"lerf_{i}.zip")
    vol.commit()
    print("XONG fetch_assets", flush=True)


# ---------------------------------------------------------------------------------------------
# 2) Kiểm tra môi trường (CPU, vài cent)
# ---------------------------------------------------------------------------------------------
@app.function(image=image, volumes={VOL: vol}, timeout=1200, cpu=2, memory=8192)
def check_env():
    def run(cmd):
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable="/bin/bash")
        return (r.stdout + r.stderr).strip()

    print(run("nvcc --version | tail -2"))
    print("venv312:", run("/opt/venv312/bin/python -c \"import torch; print('torch', torch.__version__, torch.version.cuda)\""))
    print("3DGS ext:", run("cd /opt/Split_and_Splat && /opt/venv312/bin/python -c \"import diff_gaussian_rasterization, simple_knn, fused_ssim; print('OK')\""))
    print("venv310:", run("/opt/venv310/bin/python -c \"import torch, diffusers; print('torch', torch.__version__, 'diffusers', diffusers.__version__)\""))
    print("murre repo:", run("git -C /opt/Murre rev-parse HEAD"))
    print("notebook:", Path("/opt/prep_notebook.ipynb").exists())
    vol.reload()
    p = Path(PREP)
    print("Volume murre_ckpt:", [str(x.parent) for x in p.rglob("model_index.json")][:1])
    print("Volume seed depth:", len(list((p / "seed").rglob("*_pred.npy"))), "| colmap.zip:", [str(x) for x in (p / "seed").rglob("colmap.zip")])
    print("Volume lerf zips:", [(x.name, round(x.stat().st_size / 1e6)) for x in (p / "lerf").glob("*.zip")])
    print("colmap:", run("which colmap || echo 'chưa có COLMAP (scene có checkpoint COLMAP thì không cần)'"))
    # nạp thử các cell hàm của notebook (không chạy bước nặng)
    ns = _load_notebook_namespace()
    print("hàm notebook:", all(k in ns for k in ("step_colmap", "step_murre", "step_gs", "step_finalize", "zip_prep", "check_depth", "check_sfm")))


# ---------------------------------------------------------------------------------------------
# 3) Chạy pipeline cho 1 scene (GPU)
# ---------------------------------------------------------------------------------------------
def _load_notebook_namespace() -> dict:
    import numpy as np  # noqa: F401  (cell zip_prep dùng np)

    nb = json.loads(Path("/opt/prep_notebook.ipynb").read_text(encoding="utf8"))
    cells = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]

    def pick(marker: str) -> str:
        m = [s for s in cells if marker in s]
        assert m, f"không thấy cell chứa {marker!r}"
        return m[0]

    ns: dict = dict(
        os=os, re=re, sys=sys, json=json, shutil=shutil, subprocess=subprocess, time=time, zipfile=zipfile, Path=Path,
        glob=__import__("glob"),
        REPO=Path("/opt/Split_and_Splat"), MURRE=Path("/opt/Murre"),
        VENV=Path("/opt/venv312"), VPY=Path("/opt/venv312/bin/python"), VPY_M=Path("/opt/venv310/bin/python"),
        CUDA_HOME="/usr/local/cuda", COMPUTE_CAP="8.9",
        DRIVE_DIR="", CKPT_SEED_GDRIVE_ID="", ZIP_PARTS=1, MURRE_CKPT_EVERY=50,
        CAMERA_MODEL="PINHOLE", COLMAP_GPU="auto",
        MURRE_PROCESSING_RES=0, MURRE_MAX_DEPTH=10.0, MURRE_DENOISE_STEPS=10, MURRE_ENSEMBLE_SIZE=5,
        MURRE_ERR_THR="", MURRE_NVIEWS_THR="", GS_ITERATIONS=10000, DOWNLOAD_WHEN="never",
    )
    # cell 1: sh, ensure_drive, try_download ... (chỉ phần trước các hàm phát hiện CUDA, vì ở đây CUDA có sẵn)
    first = pick("def ensure_drive")
    exec(first[: first.index("def _nvcc_version")], ns)
    mk = pick("def make_env")
    exec(mk[: mk.index('sh("apt-get')], ns)                     # chỉ lấy hàm make_env, bỏ đoạn cài môi trường của notebook
    exec(pick("# ===== Checkpoint"), ns)                  # hàm checkpoint
    # trỏ checkpoint/seed vào Volume
    ns["CKPT_LOCAL_DIR"] = Path(f"{PREP}/ckpt")
    ns["SEED_DIR"] = Path(f"{PREP}/seed")
    ns["SEED_ZIPS_DIR"] = Path(f"{PREP}/seed_zips")
    for marker in ("def data_dir(scene)", "def _link(src, dst)", "INVDEPTH_PY = r", "def zip_prep"):
        exec(pick(marker), ns)
    return ns


@app.function(
    image=image, gpu="L4", volumes={VOL: vol},
    timeout=int(2.5 * 3600), cpu=4, memory=16384,        # trần chi phí: 2.5 giờ x ~$0.80 = ~$2
)
def prep_scene(scene: str = "figurines", murre_ensemble: int = 5):
    t0 = time.time()
    vol.reload()
    stop = threading.Event()
    threading.Thread(target=_commit_loop, args=(stop,), daemon=True).start()
    os.chdir("/tmp")
    try:
        print(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True,
                             capture_output=True, text=True).stdout.strip(), flush=True)
        ns = _load_notebook_namespace()
        ns["MURRE_ENSEMBLE_SIZE"] = murre_ensemble
        P = Path(PREP)

        # Murre checkpoint
        mk = next(P.rglob("model_index.json"), None)
        assert mk is not None, "Chưa có Murre ckpt trong Volume: chạy fetch_assets trước"
        ns["MURRE_CKPT"] = mk.parent

        # COLMAP (nếu có thì dùng; scene có checkpoint COLMAP thì không cần)
        has_colmap_bin = shutil.which("colmap") is not None
        ns["USE_COLMAP_GPU"] = False
        ns["ENV"] = ns["make_env"](ns["VENV"], extra_pythonpath=str(ns["REPO"]))

        # dữ liệu LERF: giải nén zip trong Volume, tìm thư mục images của scene
        raw = Path("/tmp/lerf_raw"); raw.mkdir(exist_ok=True)
        for z in sorted((P / "lerf").glob("*.zip")):
            dest = raw / z.stem
            if not dest.exists():
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(dest, [m for m in zf.namelist() if "__MACOSX" not in m and not m.endswith(".DS_Store")])
        found = {}
        for d in sorted(raw.rglob("images")):
            if d.is_dir() and any(p.suffix.lower() in (".jpg", ".jpeg", ".png") for p in d.iterdir()):
                found.setdefault(d.parent.name, d)
        assert scene in found, f"không thấy scene {scene} trong dữ liệu LERF (có: {list(found)})"
        ns["LERF_SCENES"] = {scene: found[scene]}
        print(f"scene {scene}: {sum(1 for _ in found[scene].iterdir())} ảnh ← {found[scene]}", flush=True)

        def stage_images(sc, src_dir):
            dst = ns["REPO"] / "data" / sc / "input"; dst.mkdir(parents=True, exist_ok=True); n = 0
            for p in sorted(Path(src_dir).iterdir()):
                if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    shutil.copy2(p, dst / p.name.replace("frame_", "")); n += 1
            return n
        ns["stage_images"] = stage_images
        if not has_colmap_bin:
            print("(container này chưa có COLMAP: chỉ dùng được khi có checkpoint COLMAP của scene trong Volume)", flush=True)

        # chạy các bước đúng thứ tự notebook
        ns["step_colmap"](scene)
        sfm = ns["check_sfm"](scene)
        ns["step_murre"](scene)
        dep = ns["check_depth"](scene)
        ply = ns["step_gs"](scene)
        ns["step_finalize"](scene, ply)

        # đóng gói kết quả vào Volume
        out = Path(PREP) / "out"; out.mkdir(parents=True, exist_ok=True)
        for z in ns["zip_prep"](scene, 1):
            shutil.copy2(z, out / Path(z).name)
            print(f"📦 {out / Path(z).name}: {Path(z).stat().st_size / 1e6:.0f} MB", flush=True)
        # gói checkpoint đầy đủ (dùng làm seed lần sau)
        roots = ns["ckpt_roots"](scene)
        depth = {}
        for r in reversed(roots):
            for p in (r / "depth").glob("*_pred.npy"):
                depth[p.name] = p
        with zipfile.ZipFile(out / f"ckpt_{scene}.zip", "w", zipfile.ZIP_STORED) as zf:
            for nm in ("colmap.zip", "gs_point_cloud.ply", "gs_depth_params.json"):
                for r in roots:
                    if (r / nm).exists():
                        zf.write(r / nm, f"ckpt/{scene}/{nm}"); break
            for nm, p in sorted(depth.items()):
                zf.write(p, f"ckpt/{scene}/depth/{nm}")
        vol.commit()
        mins = (time.time() - t0) / 60
        print(f"\n[XONG] {scene}: đăng ký {sfm['registered']}/{sfm['input_images']} ảnh, {sfm['points3D']} điểm SfM, "
              f"depth [{dep['depth_min']:.3f}, {dep['depth_max']:.3f}] | {mins:.1f} phút (~${mins / 60 * 0.80:.2f} GPU L4)", flush=True)
        print(f"Lấy về: python -m modal volume get safe-gs-prep out/{scene}_prep.zip E:\\Safe-GS\\Data\\", flush=True)
    finally:
        stop.set()
        vol.commit()


# ---------------------------------------------------------------------------------------------
# 4) Lan truyền mask (README Stage 1, bước 2-4): mask_propagation.py -> mv -> prepare_folder.sh
# ---------------------------------------------------------------------------------------------
MASK_RE = re.compile(r"^([^/]+)/03_final/masks/([^/]+)/(\d+\.png)$")


def _strip_frame_prefix(stem: str) -> str:
    return stem.replace("frame_", "")          # quy tắc của sam2/auto_seg.py khi đổi tên ảnh


def _install_stage1_masks(zp: Path, repo: Path):
    """Giải nén mask Stage 1 vào <repo>/output/<scene>_autoseg_mask/<tên>/<k>.png (giống notebook)."""
    scene, frames, n = None, set(), 0
    with zipfile.ZipFile(zp) as zf:
        for name in zf.namelist():
            m = MASK_RE.match(name)
            if not m:
                continue
            scene = m[1]
            stem = _strip_frame_prefix(m[2])
            out = repo / "output" / f"{scene}_autoseg_mask" / stem / m[3]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(zf.read(name))
            frames.add(stem); n += 1
        for name in zf.namelist():
            m2 = re.match(r"^([^/]+)/03_final/meta/([^/]+)\.json$", name)
            if m2:
                scene = m2[1]
                (repo / "output" / f"{scene}_autoseg_mask" / _strip_frame_prefix(m2[2])).mkdir(parents=True, exist_ok=True)
                frames.add(_strip_frame_prefix(m2[2]))
    return scene, len(frames), n


VIZ_PY = r'''
import sys, os, json, csv
import numpy as np
from PIL import Image, ImageDraw
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
scene, outdir = sys.argv[1], sys.argv[2]
base = f"data/{scene}"
def color(i):
    rng = np.random.default_rng(int(i) * 7919 + 12345); return np.array(rng.integers(40, 256, size=3), dtype=np.uint8)
kept = sorted((d for d in os.listdir(f"{base}/masks") if os.path.isdir(f"{base}/masks/{d}")), key=lambda x: int(x) if x.isdigit() else 10**9)
disc = sorted(os.listdir(f"{base}/discard")) if os.path.isdir(f"{base}/discard") else []
def ply_n(p):
    try:
        import re
        head = open(p, "rb").read(2000).decode("latin1"); m = re.search(r"element vertex (\d+)", head); return int(m[1]) if m else None
    except Exception: return None
rows = []
for status, ids, root in (("kept", kept, "masks"), ("discarded", disc, "discard")):
    for i in ids:
        d = f"{base}/{root}/{i}"
        pngs = sorted(os.listdir(f"{d}/masks")) if os.path.isdir(f"{d}/masks") else []
        areas = []
        for p in pngs:
            a = int((np.array(Image.open(f"{d}/masks/{p}").convert("L")) > 0).sum()); areas.append(a)
        rows.append({"id": i, "status": status, "n_frames_with_mask": len(pngs), "mean_mask_px": int(np.mean(areas)) if areas else 0,
                     "max_mask_px": max(areas) if areas else 0, "ply_points": ply_n(f"{d}/points3d.ply")})
os.makedirs(outdir, exist_ok=True)
with open(f"{outdir}/instances.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["id"]); w.writeheader(); w.writerows(rows)
# 1) overlay tất cả vật trên 4 frame
frames = sorted({os.path.splitext(p)[0] for i in kept for p in os.listdir(f"{base}/masks/{i}/masks")})
pick = [frames[k] for k in np.linspace(0, len(frames) - 1, min(4, len(frames))).astype(int)] if frames else []
if pick:
    fig, axes = plt.subplots(1, len(pick), figsize=(6 * len(pick), 5), squeeze=False)
    for ax, fr in zip(axes[0], pick):
        img = np.array(Image.open(f"{base}/images/{fr}.JPEG").convert("RGB")).astype(np.float32); out = img * 0.45
        for i in kept:
            mp = f"{base}/masks/{i}/masks/{fr}.png"
            if os.path.exists(mp):
                m = np.array(Image.open(mp).convert("L")) > 0; out[m] += color(i) * 0.55
        ax.imshow(np.clip(out, 0, 255).astype(np.uint8)); ax.set_title(f"{scene} / {fr}"); ax.axis("off")
    plt.tight_layout(); plt.savefig(f"{outdir}/overlay_{scene}.png", dpi=110); plt.close()
# 2) ảnh cắt từng vật (frame có mask lớn nhất) -> lưới, để xem từng ID là vật gì
tiles = []
for i in kept:
    d = f"{base}/masks/{i}/masks"; best, ba = None, -1
    for p in sorted(os.listdir(d)):
        a = int((np.array(Image.open(f"{d}/{p}").convert("L")) > 0).sum())
        if a > ba: best, ba = p, a
    if best is None: continue
    m = np.array(Image.open(f"{d}/{best}").convert("L")) > 0
    ys, xs = np.where(m)
    if len(ys) == 0: continue
    img = Image.open(f"{base}/images/{os.path.splitext(best)[0]}.JPEG").convert("RGB")
    pad = 12; box = (max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad, img.width), min(ys.max() + pad, img.height))
    arr = np.array(img).astype(np.float32); mm = m[..., None]; arr = np.where(mm, arr, arr * 0.35).astype(np.uint8)
    t = Image.fromarray(arr).crop(box); t.thumbnail((200, 200)); canvas = Image.new("RGB", (200, 216), (20, 20, 20)); canvas.paste(t, (0, 16))
    ImageDraw.Draw(canvas).text((4, 2), f"id {i} ({ba}px)", fill=(255, 255, 255)); tiles.append(canvas)
if tiles:
    cols = 8; rws = (len(tiles) + cols - 1) // cols; grid = Image.new("RGB", (cols * 200, rws * 216), (0, 0, 0))
    for k, t in enumerate(tiles): grid.paste(t, ((k % cols) * 200, (k // cols) * 216))
    grid.save(f"{outdir}/instances_grid.png")
print(json.dumps({"kept": len(kept), "discarded": len(disc), "tiles": len(tiles)}))
'''


@app.function(image=prop_image, volumes={VOL: vol}, timeout=1200, cpu=2, memory=8192)
def check_env_prop():
    def run(cmd):
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable="/bin/bash")
        return (r.stdout + r.stderr).strip()[-600:]
    ld = "$(/opt/venv312/bin/python -c \"import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'lib'))\")"
    print("sam2 + point_projection:", run(f"cd /opt/Split_and_Splat && LD_LIBRARY_PATH={ld}:/usr/local/cuda/lib64 /opt/venv312/bin/python -c "
          "\"import sam2, point_projection_cuda, pycolmap, open3d, hydra; print('OK')\""))
    print("sam2 ckpt:", run("ls -la /opt/Split_and_Splat/checkpoints/ | tail -2"))
    vol.reload()
    P = Path(PREP)
    print("prep zip:", [(x.name, round(x.stat().st_size / 1e6)) for x in (P / "out").glob("*_prep.zip")])
    print("mask zips:", [(x.name, round(x.stat().st_size / 1e6)) for x in (P / "inputs").glob("*.zip")])
    print("prop notebook:", Path("/opt/prop_notebook.ipynb").exists())


def _snapshot_loop(stop: threading.Event, src: Path, dst: Path):
    """Chép output đang dở lên Volume mỗi 5 phút: lỡ hết timeout/OOM vẫn còn dữ liệu để xem."""
    while not stop.wait(300):
        try:
            if src.exists():
                shutil.copytree(src, dst, dirs_exist_ok=True)
                vol.commit()
        except Exception as e:
            print(f"[snapshot] {e}", flush=True)


# timeout 6h (bước 2 thực tế mất ~3h vì chậm dần theo số frame: 12 -> 25 s/frame ở pha 1, 5 -> 30 s/frame ở pha 3);
# RAM 32 GB vì tiến trình đã vượt 18 GB ở bản 16 GB.
@app.function(image=prop_image, gpu="L4", volumes={VOL: vol}, timeout=8 * 3600, cpu=4, memory=32768)
def propagate_scene(scene: str = "figurines", masks_zip: str = "figurines_sam2_masks.zip"):
    """Bước 2-4 của README (đúng lệnh gốc) + lưu nhiều dữ liệu để xem: mask & point cloud từng vật, lưới ảnh từng vật, thống kê."""
    t0 = time.time()
    vol.reload()
    os.chdir("/tmp")
    ns = _load_notebook_namespace()
    sh, REPO, VPY = ns["sh"], ns["REPO"], ns["VPY"]
    ENV = ns["make_env"](ns["VENV"])
    ENV["MPLBACKEND"] = "Agg"                     # container không có màn hình; tkinter của uv-python bị lỗi nếu matplotlib tự chọn backend
    P = Path(PREP)
    res_dir = P / "out" / f"{scene}_propagation"
    print(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True,
                         capture_output=True, text=True).stdout.strip(), flush=True)

    # đầu vào chuẩn bị -> data/<scene>/
    prep = P / "out" / f"{scene}_prep.zip"
    assert prep.exists(), f"chưa có {prep}: chạy prep_scene trước"
    with zipfile.ZipFile(prep) as zf:
        zf.extractall(REPO / "data", [n for n in zf.namelist() if n.split("/")[0] == scene])
    print(f"[{scene}] giải nén prep → data/{scene}", flush=True)

    # mask Stage 1 -> output/<scene>_autoseg_mask
    mz = P / "inputs" / masks_zip
    assert mz.exists(), f"chưa có {mz}: modal volume put safe-gs-prep <zip> prep/inputs/{masks_zip}"
    sc, nf, nm = _install_stage1_masks(mz, REPO)
    assert sc == scene, f"zip mask là của scene {sc!r}, không phải {scene!r}"
    print(f"[{scene}] mask Stage 1: {nf} frame, {nm} mask", flush=True)

    # kiểm tra đầu vào (đúng CHECK_PY của notebook)
    prop_nb = json.loads(Path("/opt/prop_notebook.ipynb").read_text(encoding="utf8"))
    src = next("".join(c["source"]) for c in prop_nb["cells"] if c["cell_type"] == "code" and 'CHECK_PY = r"""' in "".join(c["source"]))
    check_py = re.search(r'CHECK_PY = r"""(.*?)"""', src, re.S).group(1)
    (REPO / "check_inputs_tmp.py").write_text(check_py, encoding="utf-8")
    out = subprocess.run([str(VPY), "check_inputs_tmp.py", scene], cwd=REPO, env=ENV, capture_output=True, text=True)
    assert out.returncode == 0, f"kiểm tra đầu vào lỗi:\n{out.stderr[-1500:]}"
    chk = json.loads(out.stdout.strip().splitlines()[-1])
    (REPO / "check_inputs_tmp.py").unlink()
    print(f"[{scene}] COLMAP {chk['colmap_images']} ảnh, ply {chk.get('ply_points')} điểm, camera {chk['camera_models']}", flush=True)
    for p in chk["problems"]:
        print("  ❌", p, flush=True)
    assert not chk["problems"], "đầu vào chưa khớp"

    stop_snap = threading.Event()
    threading.Thread(target=_snapshot_loop, daemon=True, args=(
        stop_snap, REPO / "output" / f"{scene}_masks", P / "out" / f"{scene}_propagation_partial" / "masks_so_far")).start()
    # code gốc ép matplotlib.use("TkAgg") (tác giả chạy trên máy có màn hình) -> container không có màn hình sẽ lỗi.
    # Chỉ vá BẢN COPY trong container sang backend Agg (hình chỉ dùng khi --verbose); không đụng repo của bạn.
    sh("sed -i 's/matplotlib.use(\"TkAgg\")/matplotlib.use(\"Agg\")/' ./sam2/mask_propagation.py && grep -n 'matplotlib.use' ./sam2/mask_propagation.py", cwd=str(REPO))
    # bước 2 (nặng nhất, ~2,5 giờ): LƯU NGAY kết quả lên Volume; lần sau có rồi thì bỏ qua bước này
    out_dir = REPO / "output" / f"{scene}_masks"
    step2_vol = P / "out" / f"{scene}_propagation_step2" / f"{scene}_masks"
    log2 = "/tmp/step2.log"
    if step2_vol.exists() and any(step2_vol.iterdir()):
        shutil.copytree(step2_vol, out_dir, dirs_exist_ok=True)
        print(f"[{scene}] đã có kết quả bước 2 trong Volume → bỏ qua, nạp lại {sum(1 for p in out_dir.iterdir() if p.is_dir())} vật", flush=True)
        Path(log2).write_text("(bước 2 nạp lại từ Volume)\n")
    else:
        t = time.time()
        sh(f"set -o pipefail; {VPY} ./sam2/mask_propagation.py -dataset {scene} 2>&1 | tee {log2}", cwd=str(REPO), env=ENV)
        n_ids = sum(1 for p in out_dir.iterdir() if p.is_dir()) if out_dir.exists() else 0
        print(f"[{scene}] bước 2 xong {time.time() - t:.0f}s → {n_ids} vật (ID)", flush=True)
        assert n_ids > 0, "bước 2 không tạo ra vật nào"
        step2_vol.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(out_dir, step2_vol, dirs_exist_ok=True)
        shutil.copy2(log2, step2_vol.parent / "mask_propagation.log")
        vol.commit()
        print(f"💾 đã lưu kết quả bước 2 vào Volume: {step2_vol}", flush=True)
    # bước 3
    sh(f"mv ./output/{scene}_masks ./data/{scene}/masks", cwd=str(REPO))
    # bước 4: prepare_folder.sh bật `set -e` rồi chạy `mv "$SUB"/*.png ... 2>/dev/null`; vật không có PNG nào làm mv lỗi
    # và script thoát mã 1 (trong khi ý tác giả là chuyển vật < 2 mask vào discard). Chỉ vá BẢN COPY trong container.
    sh(r"""sed -i '/mv "\$SUB"\/\*.png/ s/2>\/dev\/null/2>\/dev\/null || true/' ./bash_dir_utils/prepare_folder.sh && grep -n 'png' ./bash_dir_utils/prepare_folder.sh""", cwd=str(REPO))
    # script chép cả sparse/ (có points3D.ply 3,1 GB) vào TỪNG vật -> 55 vật ~170 GB, dễ đầy ổ. Tạm cất file lớn ra,
    # chạy script, rồi hard link lại vào từng vật: cùng nội dung như script gốc làm, nhưng không tốn thêm chỗ.
    big = REPO / "data" / scene / "sparse" / "0" / "points3D.ply"
    big_tmp = REPO / "data" / f"_{scene}_points3D.ply"
    if big.exists():
        big.rename(big_tmp)
    sh(f"chmod +x ./bash_dir_utils/prepare_folder.sh && ./bash_dir_utils/prepare_folder.sh {scene}", cwd=str(REPO), quiet=True)
    base = REPO / "data" / scene
    if big_tmp.exists():
        n_link = 0
        for sub in ("masks", "discard"):
            for d in ((base / sub).iterdir() if (base / sub).exists() else []):
                if (d / "sparse" / "0").is_dir():
                    os.link(big_tmp, d / "sparse" / "0" / "points3D.ply"); n_link += 1
        big_tmp.rename(big)
        print(f"[{scene}] hard link points3D.ply vào {n_link} vật", flush=True)
    kept = sorted(p.name for p in (base / "masks").iterdir() if p.is_dir())
    disc = sorted(p.name for p in (base / "discard").iterdir() if p.is_dir()) if (base / "discard").exists() else []
    print(f"[{scene}] giữ {len(kept)} vật, loại {len(disc)} vật (dưới 2 mask)", flush=True)

    # lưu dữ liệu để xem
    res_dir.mkdir(parents=True, exist_ok=True)
    (REPO / "viz_tmp.py").write_text(VIZ_PY, encoding="utf-8")
    v = subprocess.run([str(VPY), "viz_tmp.py", scene, str(res_dir)], cwd=REPO, env=ENV, capture_output=True, text=True)
    print("viz:", (v.stdout or v.stderr)[-500:].strip(), flush=True)
    shutil.copy2(log2, res_dir / "mask_propagation.log")

    files = []
    for sub in ("masks", "discard"):
        if (base / sub).exists():
            for d in (base / sub).iterdir():
                if d.is_dir():
                    files += list((d / "masks").glob("*.png"))
                    if (d / "points3d.ply").exists():
                        files.append(d / "points3d.ply")
    zp = res_dir / f"{scene}_split_instances_light.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for f in sorted(files):
            zf.write(f, str(Path(scene) / f.relative_to(base)))
    vol.commit()
    mins = (time.time() - t0) / 60
    print(f"\n[XONG] {scene}: {len(kept)} vật giữ, {len(disc)} loại | {zp.name}: {zp.stat().st_size / 1e6:.0f} MB | "
          f"{mins:.1f} phút (~${mins / 60 * 1.25:.2f}: L4 + 4 CPU + 32 GB RAM)", flush=True)
    print(f"Lấy về: python -m modal volume get safe-gs-prep prep/out/{scene}_propagation <thư mục>", flush=True)


# ---------------------------------------------------------------------------------------------
# 5) Stage 2a — train Gaussian từng vật (README: ./run_all.sh <scene>), có checkpoint TỪNG VẬT
# ---------------------------------------------------------------------------------------------
def _build_instance_folders(ns: dict, scene: str) -> tuple[list[str], list[str]]:
    """Dựng lại data/<scene>/masks/<id>/ từ kết quả bước 2 đã lưu (bước 3-4 y như propagate_scene)."""
    sh, REPO = ns["sh"], ns["REPO"]
    P = Path(PREP)
    with zipfile.ZipFile(P / "out" / f"{scene}_prep.zip") as zf:
        zf.extractall(REPO / "data", [n for n in zf.namelist() if n.split("/")[0] == scene])
    step2_vol = P / "out" / f"{scene}_propagation_step2" / f"{scene}_masks"
    assert step2_vol.exists(), f"chưa có kết quả bước 2 ({step2_vol}): chạy propagate_scene trước"
    shutil.copytree(step2_vol, REPO / "data" / scene / "masks")                         # bước 3 (mv)
    # vá BẢN COPY trong container: (1) `mv *.png` không làm script thoát khi vật không có PNG (set -e);
    # (2) chép images/ và sparse/ bằng hard link (cp -rl) thay vì copy: cùng nội dung, không tốn 54 x (ảnh + ply 3,1 GB)
    sh(r"""sed -i -e '/mv "\$SUB"\/\*.png/ s/2>\/dev\/null/2>\/dev\/null || true/' """
       r"""-e 's/cp -r "\$SOURCE_IMAGES"/cp -rl "$SOURCE_IMAGES"/' -e 's/cp -r "\$SOURCE_SPARSE"/cp -rl "$SOURCE_SPARSE"/' """
       r"""./bash_dir_utils/prepare_folder.sh && grep -n 'cp -r\|png' ./bash_dir_utils/prepare_folder.sh""", cwd=str(REPO))
    sh(f"chmod +x ./bash_dir_utils/prepare_folder.sh && ./bash_dir_utils/prepare_folder.sh {scene}", cwd=str(REPO), quiet=True)
    base = REPO / "data" / scene
    # images.bin (COLMAP) ghi tên '00001.jpg' nhưng ảnh đã đổi thành '00001.JPEG' (cho mask_propagation.py) -> train.py
    # không mở được ảnh và bỏ qua MỌI camera (FileNotFoundError). Thêm tên .jpg trỏ cùng file (hard link) cho từng vật.
    n_lnk = 0
    for sub in ("masks", "discard"):
        for d in ((base / sub).iterdir() if (base / sub).exists() else []):
            for f in ((d / "images").glob("*.JPEG") if (d / "images").is_dir() else []):
                j = f.with_suffix(".jpg")
                if not j.exists():
                    os.link(f, j); n_lnk += 1
    print(f"[{scene}] thêm {n_lnk} tên ảnh .jpg (hard link) cho khớp images.bin", flush=True)
    kept = sorted(p.name for p in (base / "masks").iterdir() if p.is_dir())
    disc = sorted(p.name for p in (base / "discard").iterdir() if p.is_dir()) if (base / "discard").exists() else []
    return kept, disc


@app.function(image=prop_image, gpu="L4", volumes={VOL: vol}, timeout=8 * 3600, cpu=4, memory=32768)
def train_instances_lerf(scene: str = "figurines", ids: str = "", limit: int = 0):
    """Đúng lệnh train.py trong run_all.sh của tác giả, chạy từng vật; xong vật nào lưu ngay vật đó lên Volume,
    chạy lại thì bỏ qua vật đã xong. ids="1,8" hoặc limit=N để chạy thử ít vật."""
    t0 = time.time()
    vol.reload()
    os.chdir("/tmp")
    ns = _load_notebook_namespace()
    REPO, VPY = ns["REPO"], ns["VPY"]
    ENV = ns["make_env"](ns["VENV"], extra_pythonpath=str(REPO))
    ENV["MPLBACKEND"] = "Agg"
    print(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True,
                         capture_output=True, text=True).stdout.strip(), flush=True)

    kept, disc = _build_instance_folders(ns, scene)
    print(f"[{scene}] thư mục vật: giữ {len(kept)}, loại {len(disc)}", flush=True)

    # run_all.sh duyệt "./data/<scene>/masks"/*/ (thứ tự glob) và chỉ nhận thư mục có transforms_train.json (ScanNet).
    # Dữ liệu LERF có sparse/ (COLMAP) và train.py đọc sparse/ trước -> nhận thư mục có json HOẶC sparse/.
    # Lệnh train.py và tham số giữ NGUYÊN như run_all.sh.
    base = REPO / "data" / scene / "masks"
    todo = [d for d in kept if (base / d / "transforms_train.json").exists() or (base / d / "sparse").is_dir()]
    if ids:
        want = {s.strip() for s in ids.split(",") if s.strip()}
        todo = [d for d in todo if d in want]
    if limit:
        todo = todo[:limit]

    out_vol = Path(PREP) / "out" / f"{scene}_stage2"
    (out_vol / "logs").mkdir(parents=True, exist_ok=True)
    prog_csv = out_vol / "progress_2a.csv"
    rows = []
    if prog_csv.exists():
        rows = list(csv.DictReader(open(prog_csv, encoding="utf8")))

    def is_done(i):
        return (out_vol / "raw" / i / "point_cloud" / "iteration_1000" / "point_cloud.ply").exists()

    n_skip = sum(is_done(i) for i in todo)
    print(f"[{scene}] sẽ train {len(todo) - n_skip} vật, bỏ qua {n_skip} vật đã xong trong Volume", flush=True)
    for k, i in enumerate(todo, 1):
        if is_done(i):
            continue
        ti = time.time()
        log_i = out_vol / "logs" / f"train_2a_{i}.log"
        cmd = (f"{VPY} -u train.py -s ./data/{scene}/masks/{i} -m ./output/{scene}/raw/{i} "
               f"--iterations 1000 --is_instance --init_rec --test_iterations 500 1000 --save_iteration 500 1000")
        print(f"\n==> [{k}/{len(todo)}] vật {i}: {cmd}", flush=True)
        with open("/tmp/train_i.log", "w") as lf:
            p = subprocess.Popen(cmd, shell=True, cwd=REPO, env=ENV, executable="/bin/bash",
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in p.stdout:
                lf.write(line)
                s = line.strip()
                if re.search(r"(\d+)/(\d+) \[", s):
                    m = re.search(r"(\d+)/(\d+) \[", s)
                    c, tt = int(m[1]), int(m[2])
                    if tt > 1 and (c == tt or c % max(tt // 4, 1) == 0):
                        print(f"    [{i}] {s[:150]}", flush=True)
                elif s and ("ITER" in s or "Error" in s or "Traceback" in s or "Instance point" in s or "NO POINTS" in s):
                    print(f"    [{i}] {s[:200]}", flush=True)
            rc = p.wait()
        src = REPO / "output" / scene / "raw" / i
        ok = rc == 0 and (src / "point_cloud" / "iteration_1000" / "point_cloud.ply").exists()
        if src.exists():                                                   # lưu NGAY vật này (kể cả khi lỗi, để xem)
            shutil.copytree(src, out_vol / "raw" / i, dirs_exist_ok=True)
        shutil.copy2("/tmp/train_i.log", log_i)
        sec = time.time() - ti
        rows = [r for r in rows if r["id"] != i] + [{"id": i, "seconds": round(sec, 1), "status": "ok" if ok else f"FAILED(rc={rc})"}]
        with open(prog_csv, "w", newline="", encoding="utf8") as f:
            w = csv.DictWriter(f, fieldnames=["id", "seconds", "status"]); w.writeheader(); w.writerows(rows)
        vol.commit()
        n_new = sum(1 for r in rows if r["id"] in todo)
        print(f"==> vật {i}: {'ok' if ok else 'LỖI rc=' + str(rc)} {sec:.0f}s | đã lưu Volume | "
              f"tổng {(time.time() - t0) / 60:.1f} phút (~${(time.time() - t0) / 3600 * 1.25:.2f})", flush=True)
        if not ok:
            print(open('/tmp/train_i.log').read()[-2500:], flush=True)

    vol.commit()
    ok_all = [i for i in todo if is_done(i)]
    print(f"\n[XONG 2a] {len(ok_all)}/{len(todo)} vật có Gaussian | {(time.time() - t0) / 60:.1f} phút "
          f"(~${(time.time() - t0) / 3600 * 1.25:.2f}) | Volume: prep/out/{scene}_stage2/raw/<id>", flush=True)
