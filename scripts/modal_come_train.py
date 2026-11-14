"""
Step 0-2 of Safe-GS on Modal: COLMAP -> train CoMe 3DGS -> render (PLAN_REBUILD_BASE_FRAMEWORK.md, Phase 4)
==========================================================================================================
From origin/come-setup:modal_come.py (teatime only), parameterised by --dataset / --scene. Volume `safe-gs-come`,
one folder per scene: <key> = <dataset lower>/<scene> (e.g. lerf/teatime), or --volume-key (the teatime scene
trained on come-setup lives at key `teatime`).
    <key>/input/            images uploaded from <DATA_DIR>/<data_subdir>/<scene>/images (or --images)
    <key>/colmap_ckpt/      COLMAP checkpoints (database.db, distorted_sparse, state.json)
    <key>/colmap/           images/ + sparse/0 (3DGS format, PINHOLE, sizes unchanged)        = step 0
    <key>/come/             CoMe model folder (-m): point_cloud/iteration_N, cfg_args, config.json, cameras.json,
                            chkpnt<iter>.pth (latest only), latest.json                       = step 1
    <key>/come/renders/     {train,test}/<image>.png (+ __render_vs_gt.jpg), metrics.json     = step 2
    <key>/logs/

COLMAP exactly as 3DGS convert.py (src/come/tools/convert_colmap.py = Based_Model/Split_and_Splat/convert.py):
feature_extractor (single camera, PINHOLE, GPU) -> exhaustive_matcher (GPU) -> mapper -> image_undistorter.
CoMe training flags = scripts/run_tnt.py / run_scn.py of the authors (configs "CoMe"); code = src/come (with the
checkpoint-resume fix of train.py). GPU L4 / L40S (extensions built for sm_89).

  $env:PYTHONIOENCODING="utf-8"                                        # Windows PowerShell
  python -m modal run scripts/modal_come_train.py::upload_images --dataset LERF --scene teatime
  python -m modal run scripts/modal_come_train.py::check_come                       # CPU, cheap: build + imports
  python -m modal run --detach scripts/modal_come_train.py::colmap --dataset LERF --scene teatime
  python -m modal run --detach scripts/modal_come_train.py::train --dataset LERF --scene teatime
  python -m modal run --detach scripts/modal_come_train.py::render --dataset LERF --scene teatime
  python -m modal run scripts/modal_come_train.py::status --dataset LERF --scene teatime
  python -m modal run scripts/modal_come_train.py::download --dataset LERF --scene teatime --run teatime_napkin
      -> outputs/LERF/teatime_napkin/{step0_colmap, step1_come_3dgs, step2_come_renders} (+ summary.json each)
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

import modal

import modal_common as mc

mc.utf8_console()

app = modal.App("safe-gs-come-train")
vol = modal.Volume.from_name("safe-gs-come", create_if_missing=True)
VOL = mc.VOL
IMG_EXT = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

CONVERT_PY = mc.COME_DIR / "tools" / "convert_colmap.py"
COLMAP_TAG = "3.11.1"
CUDA_ARCH = "89"               # L4 / L40S
CAMERA = "PINHOLE"
SNAPSHOT_SEC = 120
ITERATIONS = 30000
CKPT_EVERY = 1000
# scripts/run_tnt.py, run_scn.py of CoMe: STD_ARGS + configs["CoMe"] (= HUONG_DAN.txt of the teatime run)
COME_FLAGS = (
    "--lambda_distortion 100 --eval --far_plane 100. --detach_alpha False "
    "--splatting_config configs/hierarchical.json --use_ssimdecoupled_appearance "
    "--color_confidence --color_confidence_max 0.075 --color_confidence_from_iter 500 "
    "--lambda_variance 0.5 --lambda_normal_variance 0.005"
)
COME = "/opt/CoMe"
SMALL_FILES = ("cfg_args", "config.json", "mesh_args", "rem_args", "cameras.json")
RETRY = dict(retries=modal.Retries(max_retries=4, initial_delay=10.0, backoff_coefficient=1.0))

# ---------------------------------------------------------------------------------------------------- images
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
    .add_local_python_source("modal_common")
)
come_image = mc.come_image(with_tetra=False)
light_image = modal.Image.debian_slim(python_version="3.10").add_local_python_source("modal_common")


# ---------------------------------------------------------------------------------------------------- helpers
def _key(dataset: str, scene: str, volume_key: str = "") -> str:
    return volume_key.strip("/") or mc.dataset_key(dataset, scene)


def _images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix in IMG_EXT) if folder.exists() else []


class _Log:
    def __init__(self, key: str, name: str):
        self.path = Path(VOL) / key / "logs" / name

    def __call__(self, msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _sh(cmd: str, log: _Log, cwd: str | None = None):
    """Run a command, full output to the log on the Volume, only the important lines on the console."""
    log(f"$ {cmd}")
    p = subprocess.Popen(cmd, shell=True, cwd=cwd, executable="/bin/bash",
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail: list[str] = []
    n = 0
    with open(log.path, "a", encoding="utf-8") as f:
        for line in p.stdout:
            f.write(line)
            tail = (tail + [line])[-30:]
            n += 1
            noisy = any(k in line for k in ("Name:", "Dimensions:", "Camera:", "Focal Length:", "Features:", "Processed file"))
            if not noisy or n % 50 == 0:
                print(line, end="", flush=True)
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"command failed ({rc}): {cmd}\n" + "".join(tail))


def _run_train(cmd: str, cwd: str, logfile: Path, log: _Log, every: int = 60):
    """Run train/render, full output (incl. tqdm \\r) to logfile, progress every `every` s."""
    log(f"$ {cmd}")
    logfile.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen(cmd, shell=True, cwd=cwd, executable="/bin/bash", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
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
                        last = time.time(); log(line[:160])
                else:
                    print(line[:300], flush=True)
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"command failed ({rc}):\n" + "\n".join(tail))


def _backup_db(src: Path, dst: Path):
    """Consistent sqlite snapshot even while COLMAP writes (waits up to 120 s for the lock)."""
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
        return f"{done}/{total} pairs ({100 * done / max(total, 1):.1f}%)"
    except Exception as e:  # noqa: BLE001
        return f"? ({e})"


STAGES = ["none", "extracted", "matched", "mapped", "done"]


def _state(ck: Path) -> dict:
    f = ck / "state.json"
    return json.loads(f.read_text()) if f.exists() else {"stage": "none"}


def _set_state(ck: Path, stage: str, log: _Log, **extra):
    ck.mkdir(parents=True, exist_ok=True)
    (ck / "state.json").write_text(json.dumps({"stage": stage, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **extra},
                                              indent=2))
    vol.commit()
    log(f"[checkpoint] stage = {stage}")


def _reached(st: dict, stage: str) -> bool:
    return STAGES.index(st["stage"]) >= STAGES.index(stage)


def _largest_model_to_0(sp: Path, log: _Log):
    """convert.py always uses distorted/sparse/0, COLMAP does not number models by size: move the largest to 0."""
    import pycolmap
    models = sorted(x for x in sp.iterdir() if x.is_dir())
    sizes = {m.name: len(pycolmap.Reconstruction(str(m)).images) for m in models}
    log(f"mapper models (images): {sizes}")
    best = max(sizes, key=sizes.get)
    if best == "0" and len(models) == 1:
        return sizes
    other = sp.parent / "sparse_other"
    shutil.rmtree(other, ignore_errors=True); other.mkdir()
    for m in models:
        shutil.move(str(m), str(other / m.name))
    shutil.move(str(other / best), str(sp / "0"))
    log(f"model '{best}' ({sizes[best]} images) -> sparse/0; others in {other}")
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
                  "models": models, "n_mismatch": len(bad), "size_mismatch": bad[:5], "not_registered": missing[:50]}))
"""


def _ckpt_iter(p: Path) -> int:
    return int(p.stem.replace("chkpnt", ""))


RESUME_BUG = "(model_params, first_iter, (_appearance_embedding, _appearance_net)) = torch.load(checkpoint)"
RESUME_FIX = "(model_params, first_iter, _appearance_capture) = torch.load(checkpoint)"


def _assert_resume_fix():
    s = Path(COME, "train.py").read_text()
    if RESUME_FIX not in s:
        raise RuntimeError("src/come/train.py lacks the checkpoint-resume fix (PLAN Phase 1.5): "
                           + ("original bug present" if RESUME_BUG in s else "unknown train.py"))


def _stage_colmap(V: Path, data: Path):
    for sub in ("images", "sparse"):
        shutil.copytree(V / "colmap" / sub, data / sub, dirs_exist_ok=True)
    assert (data / "sparse" / "0" / "points3D.bin").exists(), f"no COLMAP in {V / 'colmap'}: run ::colmap first"


def _stage_model(remote: Path, local: Path):
    local.mkdir(parents=True, exist_ok=True)
    for name in SMALL_FILES:
        if (remote / name).exists():
            shutil.copy2(remote / name, local / name)
    shutil.copytree(remote / "point_cloud", local / "point_cloud", dirs_exist_ok=True)


class _CkptSync:
    """Copy the newest finished checkpoint + small files from the local output folder to the Volume."""

    def __init__(self, local: Path, remote: Path, log: _Log):
        self.local, self.remote, self.log = local, remote, log
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
            for p in reversed(cks):                    # newest finished one (same size between two looks)
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
                    self.log(f"[checkpoint] iter {it} -> Volume ({size / 1e9:.2f} GB)")
                for q in self.remote.glob("chkpnt*.pth"):           # keep one on the Volume
                    if _ckpt_iter(q) < it:
                        q.unlink()
                for q in cks:                                       # and delete older local ones
                    if _ckpt_iter(q) < it:
                        q.unlink(missing_ok=True)
                break
            vol.commit()


# ---------------------------------------------------------------------------------------------------- step 0
@app.function(image=colmap_image, volumes={VOL: vol}, timeout=600, cpu=2, memory=4096)
def check_colmap(key: str = "") -> str:
    r = subprocess.run("colmap -h | head -3", shell=True, capture_output=True, text=True)
    head = (r.stdout + r.stderr).strip()
    out = [f"COLMAP: {head}", f"CUDA build: {'with CUDA' in head and 'without CUDA' not in head}",
           f"convert.py: {Path('/opt/convert.py').exists()}"]
    if key:
        vol.reload()
        out += [f"Volume input: {len(_images(Path(VOL) / key / 'input'))} images",
                f"state: {_state(Path(VOL) / key / 'colmap_ckpt')}"]
    return "\n".join(out)


@app.function(image=colmap_image, gpu="L4", volumes={VOL: vol}, timeout=3 * 3600, cpu=8, memory=32768,
              retries=modal.Retries(max_retries=3, initial_delay=10.0, backoff_coefficient=1.0))
def colmap_scene(key: str, min_registered: float = 0.9) -> dict:
    t0 = time.time()
    vol.reload()
    V = Path(VOL) / key
    log = _Log(key, "colmap.log")
    ck, final = V / "colmap_ckpt", V / "colmap"
    D = Path("/tmp") / key.replace("/", "_")             # sqlite on local disk, not on the Volume
    db = D / "distorted" / "database.db"
    log(f"===== colmap {key}, state = {_state(ck)}")
    head = subprocess.run("colmap -h | head -3", shell=True, capture_output=True, text=True)
    head = head.stdout + head.stderr
    assert "with CUDA" in head and "without CUDA" not in head, f"COLMAP without CUDA:\n{head}"

    imgs = _images(V / "input")
    assert imgs, f"no images in {V / 'input'}: run ::upload_images first"
    (D / "input").mkdir(parents=True, exist_ok=True)
    for p in imgs:
        if not (D / "input" / p.name).exists():
            shutil.copy2(p, D / "input" / p.name)
    (D / "distorted" / "sparse").mkdir(parents=True, exist_ok=True)
    n = len(imgs)

    st = _state(ck)
    if _reached(st, "done") and (final / "sparse" / "0" / "points3D.bin").exists():
        log("already done -> only re-checking the result")
    else:
        if (ck / "database.db").exists():
            shutil.copy2(ck / "database.db", db)
            log(f"restored database.db from checkpoint ({_match_progress(db, n)} matched)")
        if not _reached(st, "extracted"):
            if db.exists():
                db.unlink()
            _sh(f"colmap feature_extractor --database_path {db} --image_path {D}/input "
                f"--ImageReader.single_camera 1 --ImageReader.camera_model {CAMERA} --SiftExtraction.use_gpu 1", log)
            _backup_db(db, ck / "database.db")
            _set_state(ck, "extracted", log)
            st = _state(ck)
        if not _reached(st, "matched"):
            stop = threading.Event()

            def snap():
                while not stop.wait(SNAPSHOT_SEC):
                    try:
                        _backup_db(db, ck / "database.db")
                        vol.commit()
                        log(f"[checkpoint] matching {_match_progress(db, n)}")
                    except Exception as e:  # noqa: BLE001
                        log(f"[checkpoint] db snapshot failed: {e}")

            th = threading.Thread(target=snap, daemon=True); th.start()
            try:
                _sh(f"colmap exhaustive_matcher --database_path {db} --SiftMatching.use_gpu 1", log)
            finally:
                stop.set(); th.join(timeout=5)
            _backup_db(db, ck / "database.db")
            _set_state(ck, "matched", log, matches=_match_progress(db, n))
            st = _state(ck)
        sp = D / "distorted" / "sparse"
        if not _reached(st, "mapped"):
            shutil.rmtree(sp, ignore_errors=True); sp.mkdir(parents=True)
            _sh(f"colmap mapper --database_path {db} --image_path {D}/input --output_path {sp} "
                f"--Mapper.ba_global_function_tolerance=0.000001", log)
            models = sorted(x.name for x in sp.iterdir() if x.is_dir())
            assert (sp / "0" / "points3D.bin").exists(), f"mapper made no sparse/0 (have: {models})"
            shutil.rmtree(ck / "distorted_sparse", ignore_errors=True)
            shutil.copytree(sp, ck / "distorted_sparse")
            _set_state(ck, "mapped", log, models=models)
        else:
            shutil.rmtree(sp, ignore_errors=True)
            shutil.copytree(ck / "distorted_sparse", sp)
            log("restored distorted/sparse from checkpoint")
        sizes = _largest_model_to_0(sp, log)
        assert max(sizes.values()) >= min_registered * n, f"largest model has {max(sizes.values())}/{n} images: {sizes}"
        for sub in ("images", "sparse", "stereo"):
            shutil.rmtree(D / sub, ignore_errors=True)
        _sh(f"{sys.executable} /opt/convert.py -s {D} --camera {CAMERA} --skip_matching", log)
        assert (D / "sparse" / "0" / "points3D.bin").exists(), "convert.py made no sparse/0"
        shutil.rmtree(final, ignore_errors=True); final.mkdir(parents=True)
        for sub in ("images", "sparse", "input"):
            shutil.copytree(D / sub, final / sub)
        shutil.copy2(db, final / "database.db")
        _set_state(ck, "done", log)

    chk = Path("/tmp/check_sfm.py"); chk.write_text(CHECK_SFM)
    r = subprocess.run([sys.executable, str(chk), str(final)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"check_sfm failed:\n{r.stderr[-1500:]}")
    res = json.loads(r.stdout.strip().splitlines()[-1])
    (V / "colmap_check.json").write_text(json.dumps(res, indent=2))
    vol.commit()
    log(f"RESULT: {res['registered']}/{res['input_images']} registered, {res['points3D']} points, cameras {res['models']}, "
        f"size mismatch {res['n_mismatch']} | {(time.time() - t0) / 60:.1f} min")
    assert res["models"] == [CAMERA], f"camera must be {CAMERA}"
    assert res["n_mismatch"] == 0, f"images were resized -> masks would not match: {res['size_mismatch']}"
    return res


@app.function(image=light_image, volumes={VOL: vol}, timeout=300)
def reset_colmap_stage(key: str, stage: str):
    """Move the COLMAP checkpoint back to a stage (keeps database.db / distorted_sparse) to redo the later steps."""
    assert stage in STAGES, f"stage must be in {STAGES}"
    vol.reload()
    ck = Path(VOL) / key / "colmap_ckpt"
    (ck / "state.json").write_text(json.dumps({"stage": stage, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                               "note": "manual reset"}, indent=2))
    vol.commit()


# ---------------------------------------------------------------------------------------------------- step 1
@app.function(image=come_image, volumes={VOL: vol}, timeout=900, cpu=2, memory=8192)
def check_come_fn(key: str = "") -> str:
    r = subprocess.run(mc.COME_CHECK, shell=True, capture_output=True, text=True, cwd=COME)
    out = [(r.stdout + r.stderr).strip()[-2000:]]
    try:
        _assert_resume_fix(); out.append("train.py resume fix: OK")
    except RuntimeError as e:
        out.append(str(e))
    if key:
        vol.reload()
        c = Path(VOL) / key / "colmap"
        out.append(f"Volume {key}/colmap: {len(_images(c / 'images'))} images, "
                   f"sparse/0 {(c / 'sparse' / '0' / 'points3D.bin').exists()}")
    return "\n".join(out)


@app.function(image=come_image, gpu=mc.GPU_COME, volumes={VOL: vol}, timeout=6 * 3600, cpu=8, memory=32768,
              retries=modal.Retries(max_retries=6, initial_delay=10.0, backoff_coefficient=1.0))
def train_scene(key: str, iterations: int = ITERATIONS, ckpt_every: int = CKPT_EVERY, extra_flags: str = "") -> dict:
    t0 = time.time()
    vol.reload()
    V = Path(VOL) / key
    log = _Log(key, "come_train.log")
    remote = V / "come"
    final_ply = remote / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    if final_ply.exists():
        log(f"already trained: {final_ply}")
        return {"ply": str(final_ply)}
    _assert_resume_fix()
    log(f"===== train {key} ({subprocess.run('nvidia-smi --query-gpu=name --format=csv,noheader', shell=True, capture_output=True, text=True).stdout.strip()})")
    data = Path("/tmp/come_data") / key.replace("/", "_")
    if not data.exists():
        _stage_colmap(V, data)
    local = Path("/tmp/come_out") / key.replace("/", "_")
    local.mkdir(parents=True, exist_ok=True)
    resume = ""
    cks = sorted(remote.glob("chkpnt*.pth"), key=_ckpt_iter) if remote.exists() else []
    if cks:
        for name in SMALL_FILES:
            if (remote / name).exists():
                shutil.copy2(remote / name, local / name)
        if (remote / "tensorboard").exists():
            shutil.copytree(remote / "tensorboard", local / "tensorboard", dirs_exist_ok=True)
        shutil.copy2(cks[-1], local / cks[-1].name)
        resume = f"--start_checkpoint {local / cks[-1].name}"
        log(f"resuming from checkpoint iter {_ckpt_iter(cks[-1])}")
    ck_iters = " ".join(str(i) for i in range(ckpt_every, iterations, ckpt_every))
    cmd = (f"python train.py -s {data} -m {local} --iterations {iterations} {COME_FLAGS} {extra_flags} "
           f"--checkpoint_iterations {ck_iters} {resume}")
    sync = _CkptSync(local, remote, log)
    stop = threading.Event()

    def loop():
        while not stop.wait(20):
            try:
                sync.once()
            except Exception as e:  # noqa: BLE001
                log(f"[checkpoint] sync failed: {e}")

    th = threading.Thread(target=loop, daemon=True); th.start()
    try:
        _run_train(cmd, COME, V / "logs" / "come_train_raw.log", log)
    finally:
        stop.set(); th.join(timeout=30)
        sync.once(final=True)
    shutil.copytree(local / "point_cloud", remote / "point_cloud", dirs_exist_ok=True)
    sync.once(final=True)
    vol.commit()
    assert final_ply.exists(), "training finished but no final point_cloud"
    log(f"DONE train: {final_ply} | {(time.time() - t0) / 60:.1f} min (this run)")
    return {"ply": str(final_ply), "minutes": round((time.time() - t0) / 60, 1)}


# ---------------------------------------------------------------------------------------------------- step 2
@app.function(image=come_image, gpu=mc.GPU_COME, volumes={VOL: vol}, timeout=3600, cpu=8, memory=32768, **RETRY)
def render_scene(key: str, iterations: int = ITERATIONS) -> dict:
    """render.py (train + test) + metrics.py (test) of CoMe, renders renamed to the image names and paired with the
    photo: <key>/come/renders/{train,test}/<name>.png, <name>__render_vs_gt.jpg, metrics.json."""
    import numpy as np
    from PIL import Image
    vol.reload()
    V = Path(VOL) / key
    log = _Log(key, "come_render.log")
    remote = V / "come"
    out_dir = remote / "renders"
    data = Path("/tmp/come_data") / key.replace("/", "_")
    _stage_colmap(V, data)
    local = Path("/tmp/come_out") / key.replace("/", "_")
    _stage_model(remote, local)
    if not (out_dir / "test" / "_done").exists() or not (out_dir / "train" / "_done").exists():
        _run_train(f"python render.py -m {local} -s {data} --iteration {iterations}", COME, V / "logs" / "come_render_raw.log", log)
        try:
            _run_train(f"python metrics.py -m {local}", COME, V / "logs" / "come_render_raw.log", log)
            if (local / "results.json").exists():
                shutil.copy2(local / "results.json", remote / "metrics.json")
                shutil.copy2(local / "per_view.json", remote / "metrics_per_view.json")
        except Exception as e:  # noqa: BLE001
            log(f"metrics.py failed (renders still usable): {str(e)[-500:]}")
        # camera order of dataset_readers.readColmapSceneInfo: sorted by name, test = idx % 8 == 0
        names = sorted(p.stem for p in _images(data / "images"))
        if "inpaint360" in str(data):                 # object-free GT views are not cameras of CoMe (dataset_readers)
            names = [n for n in names if "test" not in n]
        split = {"train": [n for i, n in enumerate(names) if i % 8 != 0],
                 "test": [n for i, n in enumerate(names) if i % 8 == 0]}
        for sp, nm in split.items():
            src = local / sp / f"ours_{iterations}"
            rs = sorted((src / "renders").glob("*.png"))
            assert len(rs) == len(nm), f"{sp}: {len(rs)} renders but {len(nm)} cameras"
            d = out_dir / sp
            shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
            for r, n in zip(rs, nm):
                ren = Image.open(r).convert("RGB")
                gt = Image.open(src / "gt" / r.name).convert("RGB")
                ren.save(d / f"{n}.png")
                both = np.concatenate([np.asarray(ren), np.full((ren.height, 8, 3), 255, np.uint8), np.asarray(gt)], 1)
                Image.fromarray(both).save(d / f"{n}__render_vs_gt.jpg", quality=90)
            (d / "_done").write_text(str(len(nm)))
            vol.commit()
            log(f"render {sp}: {len(nm)} images -> {d}")
    metrics = json.loads((remote / "metrics.json").read_text()) if (remote / "metrics.json").exists() else {}
    log(f"metrics (test): {metrics}")
    return {"renders": str(out_dir), "metrics": metrics}


# ---------------------------------------------------------------------------------------------------- extras (debug)
@app.function(image=come_image, gpu=mc.GPU_COME, volumes={VOL: vol}, timeout=3600, cpu=8, memory=32768, **RETRY)
def sdf_vis_scene(key: str, iterations: int = ITERATIONS, dump: str = "") -> str:
    """SDF = 0 surface (opacity 0.5) rendered at every camera: src/come/tools/come_sdf_vis.py -> <key>/come/sdf_vis/."""
    vol.reload()
    V = Path(VOL) / key
    log = _Log(key, "come_sdf_vis.log")
    data = Path("/tmp/come_data") / key.replace("/", "_")
    _stage_colmap(V, data)
    local = Path("/tmp/come_out") / key.replace("/", "_")
    _stage_model(V / "come", local)
    with mc.PeriodicCommit(mc.VolumeCommitter(vol), every=30):
        _run_train(f"python tools/come_sdf_vis.py -m {local} -s {data} --iteration {iterations} "
                   f"--out {V / 'come' / 'sdf_vis'}" + (f" --dump {dump}" if dump else ""), COME,
                   V / "logs" / "come_sdf_vis_raw.log", log)
    return str(V / "come" / "sdf_vis")


@app.function(image=light_image, volumes={VOL: vol}, timeout=300)
def status_fn(key: str) -> str:
    vol.reload()
    V = Path(VOL) / key
    out = [f"== {key}", f"input: {len(_images(V / 'input'))} images", f"colmap state: {_state(V / 'colmap_ckpt')}"]
    if (V / "colmap_check.json").exists():
        out.append(f"colmap check: {(V / 'colmap_check.json').read_text()}")
    remote = V / "come"
    if (remote / "latest.json").exists():
        out.append(f"latest checkpoint: {(remote / 'latest.json').read_text()}")
    pc = remote / "point_cloud"
    out.append(f"point_cloud: {sorted(x.name for x in pc.iterdir()) if pc.exists() else []}")
    out.append(f"renders: train {(remote / 'renders' / 'train' / '_done').exists()}, "
               f"test {(remote / 'renders' / 'test' / '_done').exists()}")
    for name in ("colmap.log", "come_train.log", "come_render.log"):
        log = V / "logs" / name
        if log.exists():
            out += [f"--- {name}"] + log.read_text(encoding="utf-8", errors="replace").splitlines()[-8:]
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------------- local entrypoints
def _local_key(dataset, scene, volume_key):
    mc.check_profile()
    k = _key(dataset, scene, volume_key)
    print(f"[safe-gs-come] volume key: {k}")
    return k


@app.local_entrypoint()
def upload_images(dataset: str = "LERF", scene: str = "teatime", volume_key: str = "", images: str = ""):
    """<DATA_DIR>/<data_subdir>/<scene>/images (configs/datasets/*.yaml) or --images -> <key>/input."""
    k = _local_key(dataset, scene, volume_key)
    src = Path(images) if images else Path(mc.run_config(dataset, scene, scene)["data_dir"]) / "images"
    imgs = _images(src)
    if not imgs:
        raise SystemExit(f"no images in {src} (set SAFE_GS_DATA_DIR or pass --images)")
    mc.upload_dir(vol, src, f"{k}/input")
    print(f"{len(imgs)} images -> safe-gs-come:{k}/input")


@app.local_entrypoint()
def upload_colmap(dataset: str = "Replica", scene: str = "office0", volume_key: str = ""):
    """Upload pre-built COLMAP data (<data_dir>/sparse/0 and <data_dir>/images) directly to Volume
    safe-gs-come:<key>/colmap and <key>/input. Used for Replica where COLMAP is prepared offline."""
    k = _local_key(dataset, scene, volume_key)
    cfg = mc.run_config(dataset, scene, scene)
    data_dir = Path(cfg["data_dir"])
    sparse_dir = data_dir / "sparse" / "0"
    img_dir = data_dir / "images"
    if not sparse_dir.is_dir() or not img_dir.is_dir():
        raise SystemExit(f"Missing sparse/0 or images in {data_dir}. Run prepare_replica_colmap.py first!")
    mc.upload_dir(vol, img_dir, f"{k}/input")
    mc.upload_dir(vol, img_dir, f"{k}/colmap/images")
    mc.upload_dir(vol, sparse_dir, f"{k}/colmap/sparse/0")
    print(f"[upload_colmap] Uploaded pre-built COLMAP for {k} to safe-gs-come. Ready for ::train!")


@app.local_entrypoint()
def check_come(dataset: str = "", scene: str = "", volume_key: str = ""):
    k = _key(dataset, scene, volume_key) if (dataset and scene) or volume_key else ""
    print(check_colmap.remote(k))
    print(check_come_fn.remote(k))


@app.local_entrypoint()
def colmap(dataset: str = "LERF", scene: str = "teatime", volume_key: str = "", min_registered: float = 0.9):
    print(json.dumps(colmap_scene.remote(_local_key(dataset, scene, volume_key), min_registered), indent=2))


@app.local_entrypoint()
def train(dataset: str = "LERF", scene: str = "teatime", volume_key: str = "", iterations: int = 0,
          ckpt_every: int = CKPT_EVERY):
    k = _local_key(dataset, scene, volume_key)
    it = iterations or int(mc.run_config(dataset, scene, scene).get("come", {}).get("iterations", ITERATIONS))
    print(train_scene.remote(k, it, ckpt_every))


@app.local_entrypoint()
def render(dataset: str = "LERF", scene: str = "teatime", volume_key: str = "", iterations: int = 0):
    k = _local_key(dataset, scene, volume_key)
    it = iterations or int(mc.run_config(dataset, scene, scene).get("come", {}).get("iterations", ITERATIONS))
    print(json.dumps(render_scene.remote(k, it), indent=2))


@app.local_entrypoint()
def all_steps(dataset: str = "LERF", scene: str = "teatime", volume_key: str = ""):
    """colmap -> train -> render, one after the other (use with --detach)."""
    k = _local_key(dataset, scene, volume_key)
    it = int(mc.run_config(dataset, scene, scene).get("come", {}).get("iterations", ITERATIONS))
    print(colmap_scene.remote(k)); print(train_scene.remote(k, it)); print(render_scene.remote(k, it))


@app.local_entrypoint()
def sdf_vis(dataset: str = "LERF", scene: str = "teatime", volume_key: str = "", dump: str = ""):
    print(sdf_vis_scene.remote(_local_key(dataset, scene, volume_key), ITERATIONS, dump))


@app.local_entrypoint()
def status(dataset: str = "LERF", scene: str = "teatime", volume_key: str = ""):
    print(status_fn.remote(_local_key(dataset, scene, volume_key)))


@app.local_entrypoint()
def download(dataset: str = "LERF", scene: str = "teatime", run: str = "", volume_key: str = "", iterations: int = 0,
             overwrite: bool = False):
    """<key>/colmap, come, come/renders -> outputs/<Dataset>/<run>/step0_colmap, step1_come_3dgs, step2_come_renders.
    Checkpoints (chkpnt*.pth), the COLMAP database and input/ are not downloaded."""
    k = _local_key(dataset, scene, volume_key)
    cfg = mc.run_config(dataset, run or scene, scene)
    it = iterations or int(cfg.get("come", {}).get("iterations", ITERATIONS))
    run_dir = Path(cfg["run_dir"])
    s0, s1, s2 = run_dir / "step0_colmap", run_dir / "step1_come_3dgs", run_dir / "step2_come_renders"
    mc.download_dir(vol, f"{k}/colmap", s0, overwrite, skip=lambda r: r.startswith("input/") or r.endswith(".db"))
    mc.download_dir(vol, f"{k}/come", s1, overwrite,
                    skip=lambda r: r.startswith(("renders/", "sdf_vis/", "edit/", "tensorboard/")) or r.endswith(".pth")
                    and not r.endswith("appearance_embedding.pth"))
    mc.download_dir(vol, f"{k}/come/renders", s2 / "renders", overwrite)
    try:
        mc.download_dir(vol, f"{k}/logs", run_dir / "logs" / "step0-2_come_train", overwrite)
    except Exception as e:  # noqa: BLE001
        print(f"[download] no logs: {e}")
    _write_summaries(s0, s1, s2, it, k)


def _write_summaries(s0: Path, s1: Path, s2: Path, it: int, key: str):
    """summary.json of steps 0-2 (same keys as the reference run outputs/LERF/teatime_napkin)."""
    sys.path.insert(0, str(mc.SRC / "tools"))
    from read_write_model import read_model
    from PIL import Image
    cams, imgs, _ = read_model(str(s0 / "sparse" / "0"), ext=".bin")
    names = sorted(p.name for p in _images(s0 / "images"))
    sizes = sorted({Image.open(s0 / "images" / n).size for n in names})
    reg = sorted(im.name for im in imgs.values())
    (s0 / "registered_images.txt").write_text("\n".join(reg) + "\n")
    (s0 / "summary.json").write_text(json.dumps({
        "source": f"safe-gs-come:{key}/colmap", "images": len(names), "registered": len(reg),
        "image_sizes": [list(s) for s in sizes],
        "cameras": {str(c.id): {"model": c.model, "width": c.width, "height": c.height, "params": list(map(float, c.params))}
                    for c in cams.values()},
        "check_ok": len(reg) == len(names) and all(c.model == CAMERA for c in cams.values())}, indent=2))
    ply = s1 / "point_cloud" / f"iteration_{it}" / "point_cloud.ply"
    props, n = [], 0
    if ply.exists():
        with open(ply, "rb") as f:
            for line in f:
                t = line.decode(errors="replace").split()
                if t[:2] == ["element", "vertex"]:
                    n = int(t[2])
                elif t[:1] == ["property"]:
                    props.append(t[-1])
                if t[:1] == ["end_header"]:
                    break
    (s1 / "summary.json").write_text(json.dumps({
        "source": f"safe-gs-come:{key}/come", "excluded": ["chkpnt*.pth"], "iteration": it, "num_gaussians": n,
        "ply_properties": props, "confidence": "confidence" in props, "filter_3D": "filter_3D" in props,
        "check_ok": n > 0}, indent=2))
    rn = {sp: sorted(p.stem for p in (s2 / "renders" / sp).glob("*.png")) for sp in ("train", "test")}
    allr = sorted(rn["train"] + rn["test"])
    (s2 / "summary.json").write_text(json.dumps({
        "source": f"safe-gs-come:{key}/come/renders", "renders": len(allr),
        "names_match_registered": allr == sorted(Path(x).stem for x in reg), "check_ok": len(allr) == len(reg)}, indent=2))
    print(f"[summary] step0 {len(reg)}/{len(names)} registered, step1 {n} Gaussians, step2 {len(allr)} renders")
