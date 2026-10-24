"""
ObjectGS (ICCV 2025, github.com/RuijieZhu94/ObjectGS @ 0f6842b) chạy nguyên code tác giả trên LERF-Mask teatime.

    python -m modal run --detach modal_objectgs.py::train       # ply_preprocessing.py (corr) + train.py (3D, config lerf_mask)
    python -m modal run modal_objectgs.py::status

Dữ liệu = bộ LERF-Mask teatime của Gaussian Grouping (images, sparse/0, object_mask DEVA), đúng nguồn README chỉ.
Khác README (ghi rõ để báo cáo):
  1. Env conda tạo đúng environment.yml (python 3.8, torch 2.1.2, CUDA 11.8, gsplat-object) trong image CUDA 11.8 devel.
  2. ply_preprocessing.py chạy với --output_ply_name points3D.ply (cờ có sẵn) vì train.py đọc sparse/0/points3D.ply;
     file points3D.ply gốc (không nhãn) giữ lại thành points3D_orig.ply.
  3. train.py thêm cờ có sẵn --checkpoint_iterations để chạy tiếp khi Modal ngắt.
Output trên Volume safe-gs-come: teatime/objectgs/ (datasets, outputs, _done, log).
"""
from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path

import modal

import modal_come as mc

app = modal.App("safe-gs-objectgs")
vol, VOL = mc.vol, mc.VOL
REPO = Path(__file__).resolve().parent / "Based_Model" / "ObjectGS"
R = "/opt/ObjectGS"
ROOT = Path(VOL) / "teatime" / "objectgs"
LOG = Path(VOL) / "teatime" / "logs"
CONDA = "/opt/conda/bin/conda"
PY = f"{CONDA} run --no-capture-output -n objectgs python"

img = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "wget", "build-essential", "ninja-build", "libglm-dev", "libgl1", "libglib2.0-0")
    .run_commands("wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_24.7.1-0-Linux-x86_64.sh -O /tmp/mc.sh "
                  "&& bash /tmp/mc.sh -b -p /opt/conda && rm /tmp/mc.sh")
    .env({"CONDA_PLUGINS_AUTO_ACCEPT_TOS": "yes", "CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "8.9",
          "MAX_JOBS": "8"})
    .add_local_file(REPO / "environment.yml", "/tmp/environment.yml", copy=True)
    .run_commands(f"{CONDA} env create --file /tmp/environment.yml")          # README: conda env create
    .apt_install("libtiff5")                                                    # pillow của env cần libtiff.so.5
    .add_local_dir(REPO, R, copy=True, ignore=[".git", "assets"])
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("modal_come")
)

GPU = dict(gpu="L4", volumes={VOL: vol}, timeout=8 * 3600, cpu=8, memory=49152,
           retries=modal.Retries(max_retries=4, initial_delay=10.0, backoff_coefficient=1.0))


def _link(src: Path, dst: str):
    src.mkdir(parents=True, exist_ok=True)
    d = Path(dst)
    if d.is_symlink() or d.is_file():
        d.unlink()
    elif d.exists():
        shutil.rmtree(d)
    d.parent.mkdir(parents=True, exist_ok=True)
    d.symlink_to(src, target_is_directory=True)


def _done(step):
    return (ROOT / "_done" / step).exists()


def _mark(step, info=None):
    f = ROOT / "_done" / step
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **(info or {})}, ensure_ascii=False))
    vol.commit()


def _sh(step, cmd, cwd=R):
    if _done(step):
        mc._tlog(f"[objectgs] skip {step}")
        return
    LOG.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()

    def loop():                                                   # commit Volume mỗi 60 s: checkpoint còn khi bị ngắt
        while not stop.wait(60):
            try:
                vol.commit()
            except Exception as e:
                print(f"[commit] {e}", flush=True)
    th = threading.Thread(target=loop, daemon=True); th.start()
    try:
        mc._run_train(f"set -e; {cmd}", cwd, LOG / f"objectgs_{step}.log")
    finally:
        stop.set(); th.join(timeout=10); vol.commit()
    _mark(step, {"cmd": cmd})


@app.function(image=img, **GPU)
def _train():
    vol.reload()
    _link(ROOT / "datasets", f"{R}/datasets")
    _link(ROOT / "outputs", f"{R}/outputs")
    d = ROOT / "datasets" / "lerf_mask" / "teatime"
    if not _done("prep"):
        src = Path(VOL) / "teatime" / "benchmarks" / "lerf_mask" / "teatime"
        for sub in ("images", "object_mask", "sparse"):
            shutil.copytree(src / sub, d / sub, dirs_exist_ok=True)
        if (d / "sparse/0/points3D.ply").exists():
            shutil.move(d / "sparse/0/points3D.ply", d / "sparse/0/points3D_orig.ply")
        _mark("prep", {"images": len(list((d / "images").iterdir())), "masks": len(list((d / "object_mask").iterdir()))})
    _sh("ply_corr", f"{PY} ply_preprocessing.py --dataset_path datasets/lerf_mask --algorithm corr "
                    f"--output_ply_name points3D.ply")
    if not _done("train"):
        ck = sorted((ROOT / "outputs").rglob("chkpnt*.pth"), key=lambda p: int(p.stem[6:]))
        resume = f"--start_checkpoint {ck[-1]}" if ck else ""
        _sh("train", f"{PY} train.py --config config/objectgs/3d/lerf_mask/config.yaml --scene_name teatime "
                     f"--checkpoint_iterations {' '.join(str(i) for i in range(3000, 30000, 3000))} {resume}")
    runs = sorted((ROOT / "outputs" / "lerf_mask" / "rebuttal" / "teatime").iterdir())
    return {"run": str(runs[-1]), "files": sorted(p.name for p in runs[-1].iterdir())}


@app.function(image=modal.Image.debian_slim().add_local_python_source("modal_come"), volumes={VOL: vol}, timeout=300)
def _status():
    vol.reload()
    d = ROOT / "_done"
    return "\n".join(f"{f.name:10s} {f.read_text()[:200]}" for f in sorted(d.iterdir(), key=lambda p: p.stat().st_mtime)) \
        if d.exists() else "chưa có gì"


@app.local_entrypoint()
def train():
    print(_train.remote())


@app.local_entrypoint()
def status():
    print(_status.remote())
