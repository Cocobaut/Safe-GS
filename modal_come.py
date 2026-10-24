"""
Modal runner cho pipeline CoMe trên teatime (modal_come.py)
===========================================================
Bước hiện có: A1 = COLMAP trên đúng 180 ảnh của team (Data/teatime/images) -> định dạng 3DGS (images/ + sparse/0)
để CoMe train (`train.py -s <thư mục này>`).

COLMAP chạy ĐÚNG như convert.py của 3DGS (bản trong Based_Model/Split_and_Splat/convert.py, giống notebook prep của team):
    feature_extractor (single_camera, PINHOLE, GPU) -> exhaustive_matcher (GPU) -> mapper (ba_global_function_tolerance=1e-6)
    -> image_undistorter (gọi chính convert.py --skip_matching)
cùng bản COLMAP 3.11.1 build CUDA như notebook. Camera PINHOLE để ảnh sau undistort giữ nguyên 994x738 -> khớp mask SAM2.
Tên ảnh giữ nguyên frame_XXXXX.jpg (trùng tên mask).

Checkpoint (Modal hay thu hồi máy): mọi thứ lưu vào Volume 'safe-gs-come', chạy lại là tiếp tục:
    - sau feature_extractor : lưu database.db
    - trong exhaustive_matcher : chụp database.db mỗi SNAPSHOT_SEC giây (sqlite backup, nhất quán);
      COLMAP tự bỏ qua cặp ảnh đã match (src/colmap/controllers/feature_matching_utils.cc, 3.11.1)
    - sau mapper : lưu distorted/sparse
    - sau undistort : lưu kết quả cuối teatime/colmap/{images,sparse/0}
    Feature extraction / mapper bị ngắt giữa chừng thì làm lại riêng bước đó (vài phút), không resume giữa bước,
    để kết quả giống hệt một lần chạy liền.

Lệnh (chạy từ thư mục Safe-GS, tài khoản Modal đang active):
    python -m modal run modal_come.py::upload_teatime                 # đưa 180 ảnh lên Volume (1 lần)
    python -m modal run modal_come.py::check_colmap                   # CPU, rẻ: kiểm tra build COLMAP
    python -m modal run --detach modal_come.py::colmap_teatime        # GPU L4, có retries + checkpoint
    python -m modal run modal_come.py::status                         # xem tiến độ / checkpoint trong Volume
    python -m modal run modal_come.py::inspect_models                 # số ảnh từng model mapper tạo ra
    python -m modal run modal_come.py::reset_stage --stage mapped     # lùi checkpoint để chạy lại các bước sau
    python -m modal volume get safe-gs-come teatime/colmap E:\\Safe-GS\\Data\\teatime\\    # lấy kết quả về máy

Bước A2 = train CoMe (Based_Model/CoMe, commit caf62f6) trên teatime/colmap, cờ ĐÚNG như scripts/run_tnt.py, run_scn.py
của tác giả (STD_ARGS + config "CoMe"). Môi trường theo environment.yml (python 3.10, torch>=2.1 + CUDA 12.1).
    python -m modal run modal_come.py::check_come                     # CPU, rẻ: kiểm tra build extension CoMe
    python -m modal run --detach modal_come.py::train_teatime         # GPU L4, retries + checkpoint mỗi CKPT_EVERY iter
    python -m modal run modal_come.py::train_status                   # iteration, checkpoint mới nhất, log
    python -m modal volume get safe-gs-come teatime/come E:\\Safe-GS\\Data\\teatime\\      # lấy kết quả về máy
Checkpoint CoMe: train.py tự lưu chkpnt<iter>.pth (--checkpoint_iterations); một thread chép bản mới nhất lên Volume
(ghi file tạm rồi đổi tên, commit), xoá bản cũ. Chạy lại -> nạp checkpoint mới nhất qua --start_checkpoint.
Sửa 1 lỗi của tác giả (chỉ áp trong container, repo local giữ nguyên): train.py nạp checkpoint bằng
`(model_params, first_iter, (_appearance_embedding, _appearance_net)) = torch.load(...)` trong khi lúc lưu là
`appearance_embedding.capture()` (1 dict) và `AppearanceEmbedding.restore(capture)` chỉ nhận 1 tham số -> resume luôn lỗi.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import modal

app = modal.App("safe-gs-come")
vol = modal.Volume.from_name("safe-gs-come", create_if_missing=True)
VOL = "/vol"
SCENE = "teatime"

REPO_DIR = Path(__file__).resolve().parent
LOCAL_IMAGES = REPO_DIR / "Data" / SCENE / "images"
CONVERT_PY = REPO_DIR / "Based_Model" / "Split_and_Splat" / "convert.py"

COLMAP_TAG = "3.11.1"          # giống notebook split_and_splat_prep_colmap_murre_3dgs.ipynb
CUDA_ARCH = "89"               # L4
CAMERA = "PINHOLE"
SNAPSHOT_SEC = 120

# ---------------------------------------------------------------------------------------------
# Image: CUDA 12.6 + COLMAP 3.11.1 build CUDA (cùng lệnh apt/cmake với notebook prep)
# ---------------------------------------------------------------------------------------------
colmap_image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu24.04", add_python="3.12")
    .apt_install(
        "git", "cmake", "ninja-build", "build-essential", "libboost-program-options-dev", "libboost-graph-dev",
        "libboost-system-dev", "libeigen3-dev", "libfreeimage-dev", "libmetis-dev", "libgoogle-glog-dev", "libgtest-dev",
        "libgmock-dev", "libsqlite3-dev", "libglew-dev", "libcgal-dev", "libceres-dev", "libcurl4-openssl-dev",
        "libssl-dev", "libflann-dev", "libglapi-mesa", "libopengl-dev", "libgl1-mesa-dev",
    )
    .run_commands(
        f"git clone --depth 1 --branch {COLMAP_TAG} https://github.com/colmap/colmap.git /opt/colmap_src",
        f"cd /opt/colmap_src && cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release -DGUI_ENABLED=OFF "
        f"-DCUDA_ENABLED=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES={CUDA_ARCH} "
        f"-DTESTS_ENABLED=OFF "
        f"-DCMAKE_EXE_LINKER_FLAGS='-Wl,--no-as-needed /usr/lib/x86_64-linux-gnu/libglapi.so.0 -Wl,--allow-shlib-undefined' "
        f"&& cmake --build build -j$(nproc) && cmake --install build && rm -rf /opt/colmap_src/build",
    )
    .pip_install("pycolmap==3.11.1", "pillow==11.1.0", "numpy==2.1.3")
    .env({"QT_QPA_PLATFORM": "offscreen", "PYTHONUNBUFFERED": "1"})
    .add_local_file(CONVERT_PY, "/opt/convert.py")
)


# ---------------------------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------------------------
def _log_path() -> Path:
    p = Path(VOL) / SCENE / "logs" / "colmap.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(_log_path(), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _sh(cmd: str, cwd: str | None = None):
    """Chạy lệnh, ghi toàn bộ output vào log trên Volume, chỉ in dòng quan trọng ra console."""
    _log(f"$ {cmd}")
    p = subprocess.Popen(cmd, shell=True, cwd=cwd, executable="/bin/bash",
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail: list[str] = []
    n = 0
    with open(_log_path(), "a", encoding="utf-8") as f:
        for line in p.stdout:
            f.write(line)
            tail = (tail + [line])[-30:]
            n += 1
            noisy = any(k in line for k in ("Name:", "Dimensions:", "Camera:", "Focal Length:", "Features:", "Processed file"))
            if not noisy or n % 50 == 0:
                print(line, end="", flush=True)
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"Lệnh lỗi (mã {rc}): {cmd}\n" + "".join(tail))


def _backup_db(src: Path, dst: Path):
    """Chụp sqlite nhất quán kể cả khi COLMAP đang ghi (chờ khoá tối đa 120s)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".db.tmp")
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True, timeout=120)
    d = sqlite3.connect(tmp)
    try:
        s.backup(d)
    finally:
        d.close(); s.close()
    os.replace(tmp, dst)


def _match_progress(db: Path, n_images: int) -> str:
    try:
        c = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30)
        done = c.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0]
        c.close()
        total = n_images * (n_images - 1) // 2
        return f"{done}/{total} cặp ({100 * done / max(total, 1):.1f}%)"
    except Exception as e:
        return f"? ({e})"


def _state(ck: Path) -> dict:
    f = ck / "state.json"
    return json.loads(f.read_text()) if f.exists() else {"stage": "none"}


def _set_state(ck: Path, stage: str, **extra):
    ck.mkdir(parents=True, exist_ok=True)
    st = {"stage": stage, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}
    (ck / "state.json").write_text(json.dumps(st, indent=2))
    vol.commit()
    _log(f"[checkpoint] stage = {stage}")


STAGES = ["none", "extracted", "matched", "mapped", "done"]


def _reached(st: dict, stage: str) -> bool:
    return STAGES.index(st["stage"]) >= STAGES.index(stage)


def _largest_model_to_0(sp: Path):
    """convert.py luôn dùng distorted/sparse/0, nhưng COLMAP không đánh số model theo kích thước.
    Đưa model có nhiều ảnh nhất về '0', các model khác chuyển sang distorted/sparse_other/."""
    import pycolmap
    models = sorted(x for x in sp.iterdir() if x.is_dir())
    sizes = {m.name: len(pycolmap.Reconstruction(str(m)).images) for m in models}
    _log(f"mapper models (số ảnh): {sizes}")
    best = max(sizes, key=sizes.get)
    if best == "0" and len(models) == 1:
        return sizes
    other = sp.parent / "sparse_other"
    shutil.rmtree(other, ignore_errors=True); other.mkdir()
    for m in models:
        shutil.move(str(m), str(other / m.name))
    shutil.move(str(other / best), str(sp / "0"))
    _log(f"dùng model '{best}' ({sizes[best]} ảnh) làm sparse/0; model khác để ở {other}")
    return sizes


CHECK_SFM = r"""
import sys, json, os
from PIL import Image
import pycolmap
root = sys.argv[1]
rec = pycolmap.Reconstruction(f"{root}/sparse/0")
n_in = len(os.listdir(f"{root}/input"))
models = sorted({c.model.name for c in rec.cameras.values()})
bad = []
for img in rec.images.values():
    iw, ih = Image.open(f"{root}/images/{img.name}").size
    ow, oh = Image.open(f"{root}/input/{img.name}").size
    if (iw, ih) != (img.camera.width, img.camera.height) or (iw, ih) != (ow, oh):
        bad.append((img.name, (iw, ih), (ow, oh), (img.camera.width, img.camera.height)))
names = sorted(img.name for img in rec.images.values())
missing = sorted(set(os.listdir(f"{root}/input")) - set(names))
print(json.dumps({"input_images": n_in, "registered": len(rec.images), "points3D": len(rec.points3D),
                  "models": models, "n_mismatch": len(bad), "size_mismatch": bad[:5], "not_registered": missing[:20]}))
"""


# ---------------------------------------------------------------------------------------------
# 1) Upload ảnh teatime lên Volume (chạy ở máy, không tốn container)
# ---------------------------------------------------------------------------------------------
@app.local_entrypoint()
def upload_teatime():
    imgs = sorted(LOCAL_IMAGES.glob("frame_*.jpg"))
    assert len(imgs) == 180, f"cần 180 ảnh trong {LOCAL_IMAGES}, đang có {len(imgs)}"
    with vol.batch_upload(force=True) as batch:
        batch.put_directory(str(LOCAL_IMAGES), f"/{SCENE}/input")
    print(f"Đã upload {len(imgs)} ảnh -> Volume safe-gs-come:/{SCENE}/input")


# ---------------------------------------------------------------------------------------------
# 2) Kiểm tra build COLMAP (CPU, rẻ)
# ---------------------------------------------------------------------------------------------
@app.function(image=colmap_image, volumes={VOL: vol}, timeout=600, cpu=2, memory=4096)
def check_colmap():
    r = subprocess.run("colmap -h | head -3", shell=True, capture_output=True, text=True)
    head = (r.stdout + r.stderr).strip()
    print("COLMAP:", head)
    print("CUDA build:", "with CUDA" in head and "without CUDA" not in head)
    print("pycolmap:", subprocess.run([sys.executable, "-c", "import pycolmap; print(pycolmap.__version__)"],
                                      capture_output=True, text=True).stdout.strip())
    print("convert.py:", Path("/opt/convert.py").exists())
    vol.reload()
    inp = Path(VOL) / SCENE / "input"
    print("Volume input:", len(list(inp.glob("*.jpg"))) if inp.exists() else 0, "ảnh")
    print("Checkpoint state:", _state(Path(VOL) / SCENE / "colmap_ckpt"))


# ---------------------------------------------------------------------------------------------
# 3) COLMAP (GPU L4), resume được
# ---------------------------------------------------------------------------------------------
@app.function(
    image=colmap_image, gpu="L4", volumes={VOL: vol},
    timeout=2 * 3600, cpu=8, memory=32768,
    retries=modal.Retries(max_retries=3, initial_delay=10.0, backoff_coefficient=1.0),
)
def colmap_teatime():
    t0 = time.time()
    vol.reload()
    V = Path(VOL) / SCENE
    ck = V / "colmap_ckpt"
    final = V / "colmap"
    D = Path("/tmp") / SCENE                     # làm việc trên đĩa local (sqlite không nên chạy trên Volume)
    db = D / "distorted" / "database.db"

    _log(f"===== colmap_teatime bắt đầu (lần chạy này), state = {_state(ck)}")
    _log(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True,
                        capture_output=True, text=True).stdout.strip())
    head = subprocess.run("colmap -h | head -3", shell=True, capture_output=True, text=True)
    head = head.stdout + head.stderr
    assert "with CUDA" in head and "without CUDA" not in head, f"COLMAP không có CUDA:\n{head}"

    # ảnh: Volume -> local, giữ nguyên tên
    src = V / "input"
    imgs = sorted(src.glob("frame_*.jpg"))
    assert len(imgs) == 180, f"Volume có {len(imgs)} ảnh trong {src}, cần 180: chạy upload_teatime trước"
    (D / "input").mkdir(parents=True, exist_ok=True)
    for p in imgs:
        if not (D / "input" / p.name).exists():
            shutil.copy2(p, D / "input" / p.name)
    (D / "distorted" / "sparse").mkdir(parents=True, exist_ok=True)
    n = len(imgs)

    st = _state(ck)
    if _reached(st, "done") and (final / "sparse" / "0" / "points3D.bin").exists():
        _log("Đã xong từ trước -> chỉ kiểm tra lại kết quả")
    else:
        # khôi phục database
        if _reached(st, "extracted") or (ck / "database.db").exists():
            if (ck / "database.db").exists():
                shutil.copy2(ck / "database.db", db)
                _log(f"♻️ khôi phục database.db từ checkpoint ({_match_progress(db, n)} đã match)")

        # --- 1. feature extraction (lệnh giống convert.py) ---
        if not _reached(st, "extracted"):
            if db.exists():
                db.unlink()                       # extraction dở dang -> làm lại từ đầu cho sạch
            _sh(f"colmap feature_extractor --database_path {db} --image_path {D}/input "
                f"--ImageReader.single_camera 1 --ImageReader.camera_model {CAMERA} --SiftExtraction.use_gpu 1")
            _backup_db(db, ck / "database.db")
            _set_state(ck, "extracted")
            st = _state(ck)

        # --- 2. exhaustive matching (lệnh giống convert.py), chụp db định kỳ ---
        if not _reached(st, "matched"):
            stop = threading.Event()

            def snap():
                while not stop.wait(SNAPSHOT_SEC):
                    try:
                        _backup_db(db, ck / "database.db")
                        vol.commit()
                        _log(f"[checkpoint] matching {_match_progress(db, n)}")
                    except Exception as e:
                        _log(f"[checkpoint] lỗi chụp db: {e}")

            th = threading.Thread(target=snap, daemon=True)
            th.start()
            try:
                _sh(f"colmap exhaustive_matcher --database_path {db} --SiftMatching.use_gpu 1")
            finally:
                stop.set(); th.join(timeout=5)
            _backup_db(db, ck / "database.db")
            _set_state(ck, "matched", matches=_match_progress(db, n))
            st = _state(ck)

        # --- 3. mapper (lệnh giống convert.py) ---
        sp = D / "distorted" / "sparse"
        if not _reached(st, "mapped"):
            shutil.rmtree(sp, ignore_errors=True); sp.mkdir(parents=True)
            _sh(f"colmap mapper --database_path {db} --image_path {D}/input --output_path {sp} "
                f"--Mapper.ba_global_function_tolerance=0.000001")
            models = sorted(x.name for x in sp.iterdir() if x.is_dir())
            assert (sp / "0" / "points3D.bin").exists(), f"mapper không tạo sparse/0 (có: {models})"
            shutil.rmtree(ck / "distorted_sparse", ignore_errors=True)
            shutil.copytree(sp, ck / "distorted_sparse")          # checkpoint giữ nguyên output gốc của mapper
            _set_state(ck, "mapped", models=models)
        else:
            shutil.rmtree(sp, ignore_errors=True)
            shutil.copytree(ck / "distorted_sparse", sp)
            _log("♻️ khôi phục distorted/sparse từ checkpoint")
        sizes = _largest_model_to_0(sp)
        assert max(sizes.values()) >= 0.9 * n, f"model lớn nhất chỉ có {max(sizes.values())}/{n} ảnh: {sizes}"

        # --- 4. undistort + sắp sparse/0: gọi chính convert.py của 3DGS ---
        shutil.rmtree(D / "images", ignore_errors=True); shutil.rmtree(D / "sparse", ignore_errors=True)
        shutil.rmtree(D / "stereo", ignore_errors=True)
        _sh(f"{sys.executable} /opt/convert.py -s {D} --camera {CAMERA} --skip_matching")
        assert (D / "sparse" / "0" / "points3D.bin").exists(), "convert.py không tạo sparse/0"

        # lưu kết quả cuối
        shutil.rmtree(final, ignore_errors=True); final.mkdir(parents=True)
        for sub in ("images", "sparse", "input"):
            shutil.copytree(D / sub, final / sub)
        shutil.copy2(db, final / "database.db")
        _set_state(ck, "done")

    # kiểm tra
    chk = Path("/tmp/check_sfm.py"); chk.write_text(CHECK_SFM)
    r = subprocess.run([sys.executable, str(chk), str(final)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"check_sfm lỗi:\n{r.stderr[-1500:]}")
    res = json.loads(r.stdout.strip().splitlines()[-1])
    (V / "colmap_check.json").write_text(json.dumps(res, indent=2))
    vol.commit()
    _log(f"KẾT QUẢ: đăng ký {res['registered']}/{res['input_images']} ảnh, {res['points3D']} điểm 3D, camera {res['models']}, "
         f"lệch kích thước {res['n_mismatch']} ảnh")
    if res["not_registered"]:
        _log(f"⚠️ ảnh không đăng ký được: {res['not_registered']}")
    assert res["models"] == [CAMERA], f"camera phải là {CAMERA}"
    assert res["n_mismatch"] == 0, f"ảnh bị đổi kích thước -> mask không khớp: {res['size_mismatch']}"
    _log(f"Xong sau {(time.time() - t0) / 60:.1f} phút (lần chạy này)")


@app.function(image=modal.Image.debian_slim(python_version="3.12"), volumes={VOL: vol}, timeout=300)
def reset_stage(stage: str):
    """Lùi checkpoint về một stage (giữ database.db / distorted_sparse) để chạy lại các bước sau nó."""
    assert stage in STAGES, f"stage phải thuộc {STAGES}"
    vol.reload()
    ck = Path(VOL) / SCENE / "colmap_ckpt"
    print("trước:", _state(ck))
    st = {"stage": stage, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "note": "reset thủ công"}
    (ck / "state.json").write_text(json.dumps(st, indent=2))
    vol.commit()
    print("sau:", st)


@app.function(image=colmap_image, volumes={VOL: vol}, timeout=600, cpu=2, memory=4096)
def inspect_models():
    """In số ảnh / điểm của từng model mapper tạo ra (checkpoint distorted_sparse)."""
    import pycolmap
    vol.reload()
    sp = Path(VOL) / SCENE / "colmap_ckpt" / "distorted_sparse"
    for m in sorted(x for x in sp.iterdir() if x.is_dir()):
        rec = pycolmap.Reconstruction(str(m))
        names = sorted(i.name for i in rec.images.values())
        print(f"model {m.name}: {len(names)} ảnh, {len(rec.points3D)} điểm 3D, "
              f"ảnh đầu/cuối: {names[:3]} ... {names[-3:]}")


# =============================================================================================
# A2: train CoMe
# =============================================================================================
COME_DIR = REPO_DIR / "Based_Model" / "CoMe"
ITERATIONS = 30000
CKPT_EVERY = 1000
# scripts/run_tnt.py, run_scn.py: STD_ARGS + configs["CoMe"]
COME_ARGS = (
    f"--iterations {ITERATIONS} --lambda_distortion 100 --eval --far_plane 100. --detach_alpha False "
    "--splatting_config configs/hierarchical.json --use_ssimdecoupled_appearance "
    "--color_confidence --color_confidence_max 0.075 --color_confidence_from_iter 500 "
    "--lambda_variance 0.5 --lambda_normal_variance 0.005"
)

come_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "build-essential", "ninja-build", "cmake", "libgl1", "libglib2.0-0", "libgomp1",
                 "libgmp-dev", "libcgal-dev", "libegl1", "libx11-6")
    .env({"CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "8.9", "CUDA_ARCHITECTURES": "89",
          "FORCE_CUDA": "1", "MAX_JOBS": "8",
          # python của Modal (python-build-standalone) mặc định gọi clang++ khi build extension: dùng gcc như modal_prep.py
          "CC": "gcc", "CXX": "g++", "LDSHARED": "g++ -shared"})
    .pip_install("torch==2.4.1", "torchvision==0.19.1", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("numpy<2", "plyfile", "tqdm", "dacite", "open3d", "ninja", "GPUtil", "opencv-python", "trimesh",
                 "einops", "tensorboard", "setuptools", "wheel",
                 # không có trong environment.yml nhưng code import (conda cài kèm qua gói khác)
                 "matplotlib", "imageio", "scipy", "scikit-image==0.24", "scikit-learn")
    .add_local_dir(COME_DIR, "/opt/CoMe", copy=True,
                   ignore=["assets", "splatviz", ".git", "**/build", "**/*.egg-info", "**/__pycache__", "**/*.so"])
    # README CoMe, mục Setup
    .run_commands(
        "cd /opt/CoMe && pip install submodules/simple-knn/ --no-build-isolation",
        "cd /opt/CoMe && pip install submodules/diff-gaussian-rasterization/ --no-build-isolation",
        "cd /opt/CoMe && pip install submodules/decoupled-fused-ssim/ --no-build-isolation",
        "pip install git+https://github.com/rahul-goel/fused-ssim/ --no-build-isolation",
    )
    # thư viện hệ thống open3d cần lúc import (layer riêng sau phần compile để sửa không phải build lại CUDA)
    .apt_install("libusb-1.0-0")
    # import thử các module không cần GPU (fused-ssim/decoupled cần CUDA lúc import nên không thử ở đây)
    .run_commands(
        "cd /opt/CoMe && python -c 'import open3d, cv2, matplotlib, trimesh, plyfile, einops, imageio, skimage, "
        "utils.vis_utils, utils.depth_utils, utils.tetmesh, arguments; print(\"imports OK\")'",
    )
    .env({"PYTHONUNBUFFERED": "1"})
)

RESUME_BUG = "(model_params, first_iter, (_appearance_embedding, _appearance_net)) = torch.load(checkpoint)\n" \
             "        appearance_embedding.restore(_appearance_embedding, _appearance_net)"
RESUME_FIX = "(model_params, first_iter, _appearance_capture) = torch.load(checkpoint)\n" \
             "        appearance_embedding.restore(_appearance_capture)"


def _patch_resume():
    f = Path("/opt/CoMe/train.py")
    s = f.read_text()
    if RESUME_FIX in s:
        return
    assert RESUME_BUG in s, "train.py khác bản đã kiểm tra (commit caf62f6): không áp được bản sửa resume"
    f.write_text(s.replace(RESUME_BUG, RESUME_FIX))


def _ckpt_iter(p: Path) -> int:
    return int(p.stem.replace("chkpnt", ""))


def _tlog(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    p = Path(VOL) / SCENE / "logs" / "come_train.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(line + "\n")


SMALL_FILES = ("cfg_args", "config.json", "mesh_args", "rem_args", "cameras.json")


class _CkptSync:
    """Chép checkpoint mới nhất + file nhỏ từ thư mục output local lên Volume."""

    def __init__(self, local: Path, remote: Path):
        self.local, self.remote = local, remote
        self.sizes: dict[str, int] = {}
        self.lock = threading.Lock()

    def once(self, final: bool = False):
        with self.lock:
            self.remote.mkdir(parents=True, exist_ok=True)
            for name in SMALL_FILES:
                if (self.local / name).exists():
                    shutil.copy2(self.local / name, self.remote / name)
            if (self.local / "tensorboard").exists():
                shutil.copytree(self.local / "tensorboard", self.remote / "tensorboard", dirs_exist_ok=True)
            cks = sorted(self.local.glob("chkpnt*.pth"), key=_ckpt_iter)
            uploaded = {_ckpt_iter(p) for p in self.remote.glob("chkpnt*.pth")}
            for p in reversed(cks):                    # bản mới nhất đã ghi xong (kích thước ổn định giữa 2 lần xem)
                size = p.stat().st_size
                stable = final or self.sizes.get(p.name) == size
                self.sizes[p.name] = size
                if not stable:
                    continue
                it = _ckpt_iter(p)
                if it not in uploaded:
                    tmp = self.remote / (p.name + ".tmp")
                    shutil.copy2(p, tmp)
                    os.replace(tmp, self.remote / p.name)
                    (self.remote / "latest.json").write_text(json.dumps({"iteration": it, "time": time.strftime("%H:%M:%S")}))
                    vol.commit()
                    _tlog(f"[checkpoint] iter {it} -> Volume ({size / 1e9:.2f} GB)")
                for q in self.remote.glob("chkpnt*.pth"):           # giữ 1 bản trên Volume
                    if _ckpt_iter(q) < it:
                        q.unlink()
                for q in cks:                                       # và xoá bản cũ ở local
                    if _ckpt_iter(q) < it:
                        q.unlink(missing_ok=True)
                break
            vol.commit()


def _run_train(cmd: str, cwd: str, logfile: Path, every: int = 60):
    """Chạy train.py, ghi toàn bộ output (cả thanh tqdm dùng \\r) vào log, in tiến độ mỗi `every` giây."""
    _tlog(f"$ {cmd}")
    p = subprocess.Popen(cmd, shell=True, cwd=cwd, executable="/bin/bash",
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    buf, last, tail = b"", 0.0, []
    with open(logfile, "ab") as f:
        while True:
            chunk = p.stdout.read1(65536) if hasattr(p.stdout, "read1") else p.stdout.read(4096)
            if not chunk:
                break
            f.write(chunk); f.flush()
            buf += chunk
            parts = buf.replace(b"\r", b"\n").split(b"\n")
            buf = parts[-1]
            for raw in parts[:-1]:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                tail = (tail + [line])[-30:]
                if line.startswith("Training progress"):
                    if time.time() - last > every:
                        last = time.time(); _tlog(line[:160])
                else:
                    print(line[:300], flush=True)
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"train.py lỗi (mã {rc}):\n" + "\n".join(tail))


@app.function(image=come_image, volumes={VOL: vol}, timeout=900, cpu=2, memory=8192)
def check_come():
    def run(c):
        r = subprocess.run(c, shell=True, capture_output=True, text=True, cwd="/opt/CoMe")
        return (r.stdout + r.stderr).strip()[-1500:]
    print(run("python -c \"import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)\""))
    print(run("python -c \"import diff_gaussian_rasterization, simple_knn._C, decoupled_fused_ssim, fused_ssim; "
              "from diff_gaussian_rasterization import ExtendedSettings; print('extensions OK')\""))
    print(run("python -c \"import scene, gaussian_renderer, arguments; print('CoMe modules OK')\""))
    print(run("pip freeze | grep -i -E 'fused|gaussian|knn|open3d|numpy|trimesh'"))
    _patch_resume()
    print("patch resume:", RESUME_FIX in Path("/opt/CoMe/train.py").read_text())
    vol.reload()
    c = Path(VOL) / SCENE / "colmap"
    print("Volume colmap:", len(list((c / "images").glob("*.jpg"))), "ảnh,", (c / "sparse" / "0" / "points3D.bin").exists())


@app.function(image=come_image, gpu="L4", volumes={VOL: vol}, timeout=1800, cpu=8, memory=32768)
def smoke_come():
    """GPU, vài phút: train 700 iter (qua densify + color confidence), checkpoint 600, rồi resume từ 600 -> kiểm tra resume."""
    _patch_resume()
    vol.reload()
    data = Path("/tmp/teatime_colmap")
    for sub in ("images", "sparse"):
        shutil.copytree(Path(VOL) / SCENE / "colmap" / sub, data / sub, dirs_exist_ok=True)
    out = Path("/tmp/smoke"); shutil.rmtree(out, ignore_errors=True)
    log = Path("/tmp/smoke.log")
    base = f"python train.py -s {data} -m {out} {COME_ARGS} --iterations 700 --save_iterations 700"
    t = time.time()
    _run_train(f"{base} --checkpoint_iterations 600", "/opt/CoMe", log, every=20)
    print(f"lần 1: {time.time() - t:.0f}s, checkpoint: {sorted(p.name for p in out.glob('chkpnt*.pth'))}", flush=True)
    t = time.time()
    _run_train(f"{base} --start_checkpoint {out / 'chkpnt600.pth'}", "/opt/CoMe", log, every=20)
    print(f"resume: {time.time() - t:.0f}s, ply: {(out / 'point_cloud' / 'iteration_700' / 'point_cloud.ply').exists()}", flush=True)


@app.function(
    image=come_image, gpu="L4", volumes={VOL: vol},
    timeout=4 * 3600, cpu=8, memory=32768,
    retries=modal.Retries(max_retries=6, initial_delay=10.0, backoff_coefficient=1.0),
)
def train_teatime(ckpt_every: int = CKPT_EVERY):
    t0 = time.time()
    vol.reload()
    V = Path(VOL) / SCENE
    remote = V / "come"
    final_ply = remote / "point_cloud" / f"iteration_{ITERATIONS}" / "point_cloud.ply"
    if final_ply.exists():
        _tlog(f"Đã train xong từ trước: {final_ply}")
        return
    _patch_resume()
    _tlog("===== train_teatime bắt đầu (lần chạy này)")
    _tlog(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True,
                         capture_output=True, text=True).stdout.strip())

    # dữ liệu: Volume -> local
    data = Path("/tmp/teatime_colmap")
    if not data.exists():
        for sub in ("images", "sparse"):
            shutil.copytree(V / "colmap" / sub, data / sub)
    assert (data / "sparse" / "0" / "points3D.bin").exists(), "chưa có teatime/colmap trên Volume"

    # output local + khôi phục checkpoint
    local = Path("/tmp/come_out/teatime"); local.mkdir(parents=True, exist_ok=True)
    resume = ""
    cks = sorted(remote.glob("chkpnt*.pth"), key=_ckpt_iter) if remote.exists() else []
    if cks:
        for name in SMALL_FILES:
            if (remote / name).exists() and name != "cameras.json":
                shutil.copy2(remote / name, local / name)
        if (remote / "cameras.json").exists():
            shutil.copy2(remote / "cameras.json", local / "cameras.json")
        if (remote / "tensorboard").exists():
            shutil.copytree(remote / "tensorboard", local / "tensorboard", dirs_exist_ok=True)
        shutil.copy2(cks[-1], local / cks[-1].name)
        resume = f"--start_checkpoint {local / cks[-1].name}"
        _tlog(f"♻️ tiếp tục từ checkpoint iter {_ckpt_iter(cks[-1])}")

    ck_iters = " ".join(str(i) for i in range(ckpt_every, ITERATIONS, ckpt_every))
    cmd = (f"python train.py -s {data} -m {local} {COME_ARGS} "
           f"--checkpoint_iterations {ck_iters} {resume}")

    sync = _CkptSync(local, remote)
    stop = threading.Event()

    def loop():
        while not stop.wait(20):
            try:
                sync.once()
            except Exception as e:
                _tlog(f"[checkpoint] lỗi đồng bộ: {e}")

    th = threading.Thread(target=loop, daemon=True); th.start()
    try:
        _run_train(cmd, "/opt/CoMe", V / "logs" / "come_train_raw.log")
    finally:
        stop.set(); th.join(timeout=30)
        sync.once(final=True)

    # kết quả cuối
    shutil.copytree(local / "point_cloud", remote / "point_cloud", dirs_exist_ok=True)
    sync.once(final=True)
    vol.commit()
    assert final_ply.exists(), "train xong nhưng không thấy point_cloud cuối"
    _tlog(f"XONG train: {final_ply} | lần chạy này {(time.time() - t0) / 60:.1f} phút")


@app.function(
    image=come_image, gpu="L4", volumes={VOL: vol}, timeout=3600, cpu=8, memory=32768,
    retries=modal.Retries(max_retries=4, initial_delay=10.0, backoff_coefficient=1.0),
)
def render_teatime():
    """render.py (train 154 + test 23 ảnh) + metrics.py (test) của tác giả, rồi đổi tên 00000.png -> frame_XXXXX
    và ghép [render | ảnh gốc] để xem. Kết quả: teatime/come/renders/{train,test}/, metrics.json.
    Bước nhanh (vài phút) nên bị ngắt thì làm lại từ đầu; phần đã xong lưu lên Volume theo từng split."""
    import numpy as np
    from PIL import Image
    vol.reload()
    V = Path(VOL) / SCENE
    remote = V / "come"
    out_dir = remote / "renders"
    data = Path("/tmp/teatime_colmap")
    for sub in ("images", "sparse"):
        shutil.copytree(V / "colmap" / sub, data / sub, dirs_exist_ok=True)
    local = Path("/tmp/come_out/teatime"); local.mkdir(parents=True, exist_ok=True)   # cùng đường dẫn lúc train
    for name in SMALL_FILES:
        shutil.copy2(remote / name, local / name)
    shutil.copytree(remote / "point_cloud", local / "point_cloud", dirs_exist_ok=True)

    log = V / "logs" / "come_render.log"
    if not (out_dir / "test" / "_done").exists() or not (out_dir / "train" / "_done").exists():
        _run_train(f"python render.py -m {local} -s {data} --iteration {ITERATIONS}", "/opt/CoMe", log)
        try:
            _run_train(f"python metrics.py -m {local}", "/opt/CoMe", log)
            if (local / "results.json").exists():
                shutil.copy2(local / "results.json", remote / "metrics.json")
                shutil.copy2(local / "per_view.json", remote / "metrics_per_view.json")
        except Exception as e:
            _tlog(f"metrics.py lỗi (bỏ qua, ảnh render vẫn dùng được): {str(e)[-500:]}")

        # thứ tự camera giống dataset_readers.readColmapSceneInfo: sắp theo tên, test = idx % 8 == 0
        names = sorted(p.stem for p in (data / "images").glob("*.jpg"))
        split = {"train": [n for i, n in enumerate(names) if i % 8 != 0],
                 "test": [n for i, n in enumerate(names) if i % 8 == 0]}
        for sp, nm in split.items():
            src = local / sp / f"ours_{ITERATIONS}"
            rs = sorted((src / "renders").glob("*.png"))
            assert len(rs) == len(nm), f"{sp}: {len(rs)} ảnh render nhưng {len(nm)} camera"
            d = out_dir / sp; shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
            for r, n in zip(rs, nm):
                ren = Image.open(r).convert("RGB")
                gt = Image.open(src / "gt" / r.name).convert("RGB")
                ren.save(d / f"{n}.png")
                both = np.concatenate([np.asarray(ren), np.full((ren.height, 8, 3), 255, np.uint8), np.asarray(gt)], 1)
                Image.fromarray(both).save(d / f"{n}__render_vs_gt.jpg", quality=90)
            (d / "_done").write_text(str(len(nm)))
            vol.commit()
            _tlog(f"render {sp}: {len(nm)} ảnh -> {d}")
    if (remote / "metrics.json").exists():
        _tlog(f"metrics (test): {(remote / 'metrics.json').read_text()}")


@app.function(
    image=come_image.add_local_file(REPO_DIR / "tools" / "come_sdf_vis.py", "/opt/come_sdf_vis.py"),
    gpu="L4", volumes={VOL: vol}, timeout=3600, cpu=8, memory=32768,
    retries=modal.Retries(max_retries=4, initial_delay=10.0, backoff_coefficient=1.0),
)
def sdf_vis_teatime(dump: str = "", fresh: bool = False):
    """Render mặt SDF = 0 (opacity 0.5) về 177 góc camera: tools/come_sdf_vis.py -> teatime/come/sdf_vis/.
    Ghi thẳng từng frame vào Volume, commit mỗi 30 giây; chạy lại thì bỏ qua frame đã có.
    --dump frame_00108: chỉ lưu 13 kênh render thô của 1 frame (debug)."""
    vol.reload()
    V = Path(VOL) / SCENE
    remote = V / "come"
    data = Path("/tmp/teatime_colmap")
    for sub in ("images", "sparse"):
        shutil.copytree(V / "colmap" / sub, data / sub, dirs_exist_ok=True)
    local = Path("/tmp/come_out/teatime"); local.mkdir(parents=True, exist_ok=True)   # cùng đường dẫn lúc train
    for name in SMALL_FILES:
        shutil.copy2(remote / name, local / name)
    shutil.copytree(remote / "point_cloud", local / "point_cloud", dirs_exist_ok=True)
    shutil.copy2("/opt/come_sdf_vis.py", "/opt/CoMe/come_sdf_vis.py")
    if fresh:                                    # xoá kết quả cũ (vd sau khi sửa script), giữ debug/
        for sub in ("normal", "contour"):
            shutil.rmtree(remote / "sdf_vis" / sub, ignore_errors=True)
        vol.commit()

    stop = threading.Event()

    def loop():
        while not stop.wait(30):
            try:
                vol.commit()
            except Exception as e:
                _tlog(f"[commit] {e}")

    th = threading.Thread(target=loop, daemon=True); th.start()
    try:
        _run_train(f"python come_sdf_vis.py -m {local} -s {data} --iteration {ITERATIONS} --out {remote / 'sdf_vis'}"
                   + (f" --dump {dump}" if dump else ""),
                   "/opt/CoMe", V / "logs" / "come_sdf_vis.log")
    finally:
        stop.set(); th.join(timeout=10)
        vol.commit()
    _tlog(f"sdf_vis xong: {len(list((remote / 'sdf_vis' / 'contour').glob('*.png')))} frame")


@app.local_entrypoint()
def upload_labels():
    """Đưa Data/teatime/sam2_masks/labels (180 PNG ID) lên Volume: teatime/sam2_masks/labels."""
    src = REPO_DIR / "Data" / SCENE / "sam2_masks" / "labels"
    n = len(list(src.glob("frame_*.png")))
    assert n == 180, f"cần 180 label trong {src}, có {n}"
    with vol.batch_upload(force=True) as batch:
        batch.put_directory(str(src), f"/{SCENE}/sam2_masks/labels")
    print(f"Đã upload {n} label -> Volume safe-gs-come:/{SCENE}/sam2_masks/labels")


@app.function(
    image=come_image.add_local_file(REPO_DIR / "tools" / "come_remove_object.py", "/opt/come_remove_object.py"),
    gpu="L4", volumes={VOL: vol}, timeout=3600, cpu=8, memory=32768,
    retries=modal.Retries(max_retries=4, initial_delay=10.0, backoff_coefficient=1.0),
)
def remove_object(ids: str = "22", ratio: float = 0.5, cleanup: bool = False, vox_mult: float = 1.0,
                  diag_ids: str = "", free_radius: float = 0.0):
    """Xoá Gaussian của vật (ID mask, cách nhau bởi dấu phẩy) -> teatime/come/edit/remove_id<ids>/.
    --cleanup: dọn thêm Gaussian bên trong mặt SDF (opacity field >= 0.5) + floater < 0.5 quanh vật -> thư mục *_v2_sdf.
    Bước nhanh (vài phút); bị ngắt thì chạy lại từ đầu."""
    vol.reload()
    V = Path(VOL) / SCENE
    remote = V / "come"
    if ids.strip().lower() == "all":              # mọi ID (label > 0)
        id_list, out = [-1], remote / "edit" / "delete_all_object"
    else:
        id_list = [int(x) for x in ids.split(",") if x.strip()]
        out = remote / "edit" / ("remove_id" + "_".join(map(str, id_list)))
    if free_radius > 0:
        out = out.with_name(out.name + "_v3_freevote")
    elif cleanup:
        out = out.with_name(out.name + "_v2_sdf")
    data = Path("/tmp/teatime_colmap")
    for sub in ("images", "sparse"):
        shutil.copytree(V / "colmap" / sub, data / sub, dirs_exist_ok=True)
    local = Path("/tmp/come_out/teatime"); local.mkdir(parents=True, exist_ok=True)
    for name in SMALL_FILES:
        shutil.copy2(remote / name, local / name)
    shutil.copytree(remote / "point_cloud", local / "point_cloud", dirs_exist_ok=True)
    labels = V / "sam2_masks" / "labels"
    assert len(list(labels.glob("*.png"))) == 180, "chưa có label trên Volume: chạy upload_labels trước"
    shutil.copy2("/opt/come_remove_object.py", "/opt/CoMe/come_remove_object.py")
    tmp_out = Path("/tmp/remove_out")
    _run_train(f"python come_remove_object.py -m {local} -s {data} --iteration {ITERATIONS} --labels {labels} "
               f"--ids {' '.join(map(str, id_list))} --ratio {ratio} --out {tmp_out}"
               + (f" --cleanup --vox_mult {vox_mult}" if cleanup else "")
               + (f" --diag_ids {' '.join(diag_ids.split(','))}" if diag_ids else "")
               + (f" --free_radius {free_radius}" if free_radius > 0 else ""),
               "/opt/CoMe", V / "logs" / "come_remove.log")
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(tmp_out, out)
    vol.commit()
    _tlog(f"remove_object xong -> {out}: {(out / 'stats.json').read_text()}")


@app.function(image=modal.Image.debian_slim(python_version="3.12"), volumes={VOL: vol}, timeout=300)
def train_status():
    vol.reload()
    remote = Path(VOL) / SCENE / "come"
    if (remote / "latest.json").exists():
        print("checkpoint mới nhất:", (remote / "latest.json").read_text())
    print("checkpoint trên Volume:", [(p.name, round(p.stat().st_size / 1e9, 2)) for p in remote.glob("chkpnt*.pth")])
    pc = remote / "point_cloud"
    print("point_cloud:", sorted(x.name for x in pc.iterdir()) if pc.exists() else [])
    log = Path(VOL) / SCENE / "logs" / "come_train.log"
    if log.exists():
        print("--- log:"); print("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]))


# ---------------------------------------------------------------------------------------------
# 4) Xem tiến độ
# ---------------------------------------------------------------------------------------------
@app.function(image=modal.Image.debian_slim(python_version="3.12"), volumes={VOL: vol}, timeout=300)
def status():
    vol.reload()
    V = Path(VOL) / SCENE
    print("input:", len(list((V / "input").glob("*.jpg"))) if (V / "input").exists() else 0, "ảnh")
    print("state:", _state(V / "colmap_ckpt"))
    db = V / "colmap_ckpt" / "database.db"
    if db.exists():
        print("database.db:", round(db.stat().st_size / 1e6, 1), "MB |", _match_progress(db, 180))
    if (V / "colmap_check.json").exists():
        print("check:", (V / "colmap_check.json").read_text())
    log = V / "logs" / "colmap.log"
    if log.exists():
        lines = [l for l in log.read_text(encoding="utf-8", errors="replace").splitlines() if l.startswith("[")]
        print("--- log (dòng chính cuối):"); print("\n".join(lines[-15:]))
