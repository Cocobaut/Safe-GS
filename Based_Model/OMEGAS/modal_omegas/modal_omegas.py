"""
OMEGAS on Modal GPU - LERF teatime, object ID 11 (white paper napkin).   Plan: ../plan.md

Every original OMEGAS script runs unchanged; this file only prepares inputs, runs the commands, checks the results,
commits the volume periodically (preemption safety) and records the state of each step.

Local (venv F:\\Safe-GS\\.venv active, profile nguyenkhacd10), from F:\\Safe-GS\\Based_Model\\OMEGAS:
  python -m modal run modal_omegas/modal_omegas.py::upload                         # raw teatime -> volume (once)
  python -m modal run --detach modal_omegas/modal_omegas.py::start --steps 0,1     # CPU-only dry run
  python -m modal run --detach modal_omegas/modal_omegas.py::start                 # v1: steps 0,1,2,3,4,6 (mesh from Step 4)
  python -m modal run --detach modal_omegas/modal_omegas.py::start --steps 5,6 --mesh-source step5   # v2 (Step 5)
  python -m modal run modal_omegas/modal_omegas.py::status
  python -m modal run modal_omegas/modal_omegas.py::download --step 6              # + sha256 verification
  python -m modal run modal_omegas/modal_omegas.py::verify --folder outputs/teatime_napkin/step6_mesh_extraction
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

# --------------------------------------------------------------------------------------------------
# 1. Settings
# --------------------------------------------------------------------------------------------------
APP_NAME = "omegas-teatime-napkin"
WORKSPACE_PROFILE = "nguyenkhacd10"
GPU = ["L4", "A10G"]          # fallback list: first one with free capacity
OBJ_ID = 11
RUN_NAME = "teatime_napkin"
COMMIT_EVERY_S = 300
CKPT_KEEP = 2
SAVE_ITERS = "5000 10000 15000 20000 25000 30000"
SNAPSHOT_ITER = 15000

LOCAL_ROOT = Path(__file__).resolve().parents[1]           # F:\Safe-GS\Based_Model\OMEGAS
LOCAL_RAW = Path(r"F:\LERF Datasets\teatime\teatime")
LOCAL_OUT = LOCAL_ROOT / "outputs" / RUN_NAME

VOL = "/vol"
REPO = "/root/OMEGAS"
RAW = f"{VOL}/raw/teatime"
D = f"{VOL}/data/teatime"
O = f"{VOL}/outputs/{RUN_NAME}"
STATE = f"{O}/state"
TOOLS = f"{REPO}/modal_omegas"
CFG = f"{TOOLS}/teatime_obj11.json"

STEP_DIRS = {0: "step0_data", 1: "step1_sam_masks", 2: "step2_2dgs_scene", 3: "step3_id_check",
             4: "step4_object_extraction", 5: "step5_gaussian_completion", 6: "step6_mesh_extraction"}

app = modal.App(APP_NAME)
vol = modal.Volume.from_name("omegas", create_if_missing=True)
VOLUMES = {VOL: vol}
RETRIES = modal.Retries(max_retries=2, backoff_coefficient=1.0, initial_delay=30.0)

BASE_ENV = {"PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg", "QT_QPA_PLATFORM": "offscreen",
            "HF_HOME": f"{VOL}/hf_cache", "HF_HUB_DISABLE_TELEMETRY": "1"}

# --------------------------------------------------------------------------------------------------
# 2. Images
# --------------------------------------------------------------------------------------------------
REPO_IGNORE = [
    "outputs", "outputs/**", "media", "media/**", "SuGaR", "SuGaR/**",
    "Tracking-Anything-with-DEVA", "Tracking-Anything-with-DEVA/**",
    "Grounded-Segment-Anything", "Grounded-Segment-Anything/**", "submodules", "submodules/**",
    "gaussian_splatting/SIBR_viewers", "gaussian_splatting/SIBR_viewers/**",
    "gaussian_splatting/submodules", "gaussian_splatting/submodules/**",
    "dreamgaussian/diff-gaussian-rasterization-dm", "dreamgaussian/diff-gaussian-rasterization-dm/**",
    "dreamgaussian/simple-knn", "dreamgaussian/simple-knn/**",
    "**/__pycache__", "**/__pycache__/**", "**/*.pyc", "**/*.pth", "**/*.ckpt", "**/.git", "**/.git/**", "plan.md",
]
SUBMODULES = LOCAL_ROOT / "gaussian_splatting" / "submodules"
EXT = "/opt/ext"

# CPU image: COLMAP + light python (Step 0/1 and the orchestrator)
prep_image = (
    modal.Image.from_registry("ubuntu:22.04", add_python="3.10")
    .apt_install("colmap")
    .pip_install("numpy==1.26.4", "pillow==10.4.0", "scipy==1.13.1", "plyfile==1.0.3")
    .env(BASE_ENV)
    .add_local_dir(str(LOCAL_ROOT / "modal_omegas"), TOOLS, ignore=["**/__pycache__/**"])
)

# GPU image: README of the authors (python 3.10 - Modal no longer offers 3.9 -, CUDA 11.8, torch 2.0.1) + 2DGS extensions + pytorch3d
main_image = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "build-essential", "ninja-build", "wget", "ffmpeg",
                 "libgl1", "libglib2.0-0", "libsm6", "libxext6", "libgomp1")
    .env({**BASE_ENV, "CUDA_HOME": "/usr/local/cuda", "FORCE_CUDA": "1", "CC": "gcc", "CXX": "g++",
          "TORCH_CUDA_ARCH_LIST": "8.6;8.9"})          # A10G = 8.6, L4 = 8.9 (no GPU during the build)
    .pip_install("wheel", "setuptools==69.5.1", "ninja")
    # extra_index_url (not index_url): torch's dependency `lit` is an sdist whose build needs `wheel` from PyPI
    .pip_install("torch==2.0.1+cu118", "torchvision==0.15.2+cu118", "numpy==1.23.1",
                 extra_index_url="https://download.pytorch.org/whl/cu118")
    .add_local_file(str(LOCAL_ROOT / "requirements.txt"), "/opt/req/requirements.txt", copy=True)
    .add_local_file(str(LOCAL_ROOT / "modal_omegas" / "constraints.txt"), "/opt/req/constraints.txt", copy=True)
    .run_commands(
        # paddleseg is not imported anywhere in OMEGAS and drags in the paddle stack -> left out
        "grep -v -i '^paddleseg' /opt/req/requirements.txt > /opt/req/requirements_modal.txt",
        "pip install -r /opt/req/requirements_modal.txt -c /opt/req/constraints.txt",
    )
    .run_commands(
        "pip install fvcore iopath -c /opt/req/constraints.txt",      # pytorch3d deps (PyPI)
        "pip install --no-index --no-deps --no-cache-dir pytorch3d "
        "-f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt201/download.html",
    )
    .add_local_dir(str(SUBMODULES / "diff-surfel-rasterization"), f"{EXT}/diff-surfel-rasterization", copy=True,
                   ignore=["build", "build/**", "*.egg-info", "*.egg-info/**", "**/__pycache__/**"])
    # diff-surfel-rasterization ships without third_party/glm: use the glm headers of diff-gaussian-rasterization
    .add_local_dir(str(SUBMODULES / "diff-gaussian-rasterization" / "third_party" / "glm"),
                   f"{EXT}/diff-surfel-rasterization/third_party/glm", copy=True,
                   ignore=["doc", "doc/**", "test", "test/**", "**/.git/**"])
    .add_local_dir(str(SUBMODULES / "simple-knn"), f"{EXT}/simple-knn", copy=True,
                   ignore=["build", "build/**", "*.egg-info", "*.egg-info/**", "**/__pycache__/**"])
    .run_commands(
        f"pip install --no-build-isolation {EXT}/diff-surfel-rasterization",
        f"pip install --no-build-isolation {EXT}/simple-knn",
        "python -c \"import torch, diff_surfel_rasterization, simple_knn._C, pytorch3d, open3d, diffusers, "
        "transformers, rembg, plyfile, cv2, omegaconf; print('imports OK', torch.__version__)\"",
    )
    # render.py (imported by render_obj.py) needs the 3DGS rasterizer too (README step 3 installs it as well)
    .add_local_dir(str(SUBMODULES / "diff-gaussian-rasterization"), f"{EXT}/diff-gaussian-rasterization", copy=True,
                   ignore=["build", "build/**", "*.egg-info", "*.egg-info/**", "**/__pycache__/**",
                           "third_party/glm/doc", "third_party/glm/doc/**", "third_party/glm/test",
                           "third_party/glm/test/**", "**/.git/**"])
    .run_commands(
        f"pip install --no-build-isolation {EXT}/diff-gaussian-rasterization",
        # gaussian_splatting/utils/mesh_utils.py (mesh_2D.py) imports pyrender, missing from requirements.txt
        "pip install pyrender -c /opt/req/constraints.txt",
        "python -c \"import diff_gaussian_rasterization, pyrender, mediapy, trimesh; print('render deps OK')\"",
    )
    # the import check above created HF_HOME=/vol/hf_cache inside the image; the volume needs an empty mount point
    .run_commands("rm -rf /vol")
    .add_local_dir(str(LOCAL_ROOT), REPO, ignore=REPO_IGNORE)   # code mounted at start (not baked in)
)

# --------------------------------------------------------------------------------------------------
# 3. Helpers (run inside containers)
# --------------------------------------------------------------------------------------------------
def _commit(reason: str = "") -> None:
    try:
        vol.commit()
        print(f"[volume] committed {reason}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[volume] commit failed ({reason}): {e}", flush=True)


def _prune_checkpoints(folder: str) -> None:
    ck = sorted(Path(folder).glob("chkpnt*.pth"), key=lambda p: int(p.stem.replace("chkpnt", "") or 0))
    for p in ck[:-CKPT_KEEP]:
        p.unlink(missing_ok=True)


class PeriodicCommit:
    """Commit every COMMIT_EVERY_S seconds, on SIGTERM (preemption) and at the end of the block."""

    def __init__(self, prune_dir: str | None = None):
        self.prune_dir = prune_dir
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self.stop.wait(COMMIT_EVERY_S):
            if self.prune_dir:
                _prune_checkpoints(self.prune_dir)
            _commit("periodic")

    def __enter__(self):
        def on_term(signum, frame):
            _commit(f"signal {signum}")
            os._exit(128 + signum)
        try:
            signal.signal(signal.SIGTERM, on_term)
        except ValueError:
            pass
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        if self.prune_dir:
            _prune_checkpoints(self.prune_dir)
        _commit("end of block")
        return False


def _state_path(step: int) -> Path:
    return Path(STATE) / f"step{step}.json"


def _load_state(step: int) -> dict | None:
    p = _state_path(step)
    return json.loads(p.read_text()) if p.exists() else None


def _save_state(step: int, st: dict) -> None:
    Path(STATE).mkdir(parents=True, exist_ok=True)
    _state_path(step).write_text(json.dumps(st, indent=2, ensure_ascii=False))


def _next_dir(step: int) -> str:
    base = STEP_DIRS[step]
    if not (Path(O) / base).exists():
        return base
    k = 2
    while (Path(O) / f"{base}_v{k}").exists():
        k += 1
    return f"{base}_v{k}"


def _begin(step: int, force: bool, keep_partial: bool = False, **meta) -> dict | None:
    """New run (fresh _vK folder if the step was done), or retry of an unfinished run (folder cleaned unless
    keep_partial). Returns None when the step is done and not forced."""
    st = _load_state(step)
    if st and st["status"] == "done" and not force:
        print(f"[step{step}] already done -> {st['dir']} (force for a new version)", flush=True)
        return None
    if st and st["status"] in ("running", "failed") and not force:
        st["attempts"] += 1
        st["status"] = "running"
        print(f"[step{step}] retry {st['dir']} (attempt {st['attempts']})", flush=True)
        if not keep_partial:
            shutil.rmtree(Path(O) / st["dir"], ignore_errors=True)
    else:
        st = {"step": step, "dir": _next_dir(step), "status": "running", "attempts": 1, "parts": [],
              "started": time.strftime("%Y-%m-%d %H:%M:%S"), **meta}
        print(f"[step{step}] new run -> {st['dir']}", flush=True)
    (Path(O) / st["dir"] / "logs").mkdir(parents=True, exist_ok=True)
    _save_state(step, st)
    _commit(f"step{step} begin")
    return st


def _finish(step: int, st: dict, status: str = "done", **extra) -> dict:
    st.update(extra, status=status, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    _save_state(step, st)
    (Path(O) / st["dir"] / "state.json").write_text(json.dumps(st, indent=2, ensure_ascii=False))
    _commit(f"step{step} {status}")
    return st


def _fail(step: int, st: dict, err: Exception):
    _finish(step, st, status="failed", error=f"{type(err).__name__}: {err}"[-3000:])
    raise err


def _require(step: int) -> dict:
    st = _load_state(step)
    if not st or st["status"] != "done":
        raise RuntimeError(f"step{step} is not done (state: {st['status'] if st else 'none'})")
    return st


def _sh(cmd: str, st: dict, cwd: str = REPO, env: dict | None = None, log_name: str = "run.log") -> None:
    """Run one command with bash, stream it (progress bars throttled) and append it to <dir>/logs/<log_name>."""
    log = Path(O) / st["dir"] / "logs" / log_name
    full_env = {**os.environ, **(env or {})}
    last_bar = 0.0
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n===== [attempt {st['attempts']}] {time.strftime('%H:%M:%S')} $ {cmd}\n")
        f.flush()
        print(f"$ {cmd}", flush=True)
        proc = subprocess.Popen(["bash", "-c", f"set -eo pipefail; {cmd}"], cwd=cwd, env=full_env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            if "it/s]" in line or "s/it]" in line:      # tqdm: keep one line every 30 s
                if time.time() - last_bar < 30:
                    continue
                last_bar = time.time()
            sys.stdout.write(line)
            f.write(line)
        rc = proc.wait()
        f.write(f"===== exit {rc}\n")
    if rc != 0:
        raise RuntimeError(f"command failed ({rc}): {cmd}")


def _copy_model(src_model: str, dst_model: str, iteration: int, cfg_from: str | None = None,
                cameras_from: str | None = None) -> None:
    """Minimal 2DGS model folder (cfg_args, cameras.json, point_cloud/iteration_N) so that a script writing into
    its -m folder (render_obj / extract_object / mesh_2D) never touches the folder of an earlier step."""
    dst = Path(dst_model)
    (dst / "point_cloud").mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(cfg_from or src_model) / "cfg_args", dst / "cfg_args")
    cams = Path(cameras_from or src_model) / "cameras.json"
    if cams.exists():
        shutil.copy2(cams, dst / "cameras.json")
    it = dst / "point_cloud" / f"iteration_{iteration}"
    shutil.rmtree(it, ignore_errors=True)
    shutil.copytree(Path(src_model) / "point_cloud" / f"iteration_{iteration}", it)


def _tools():
    sys.path.insert(0, TOOLS)
    import object_tools  # noqa: E402
    return object_tools


def _mesh(st: dict, src_model: str, src_data: str, iteration: int, out_dir: str, cfg_from: str,
          cameras_json: str, log_name: str, ref_ply: str | None = None) -> dict:
    """mesh_2D.py on a copy of the model (raw fuse.ply / fuse_post.ply stay in <out_dir>/model/train/ours_N, not
    downloaded) -> cleaned object mesh <out_dir>/object_11_mesh.{ply,obj} + mesh_info.json.
    ref_ply: Gaussians whose bbox delimits the object (default: the meshed model itself)."""
    ot = _tools()
    out = Path(out_dir)
    model = out / "model"
    _copy_model(src_model, str(model), iteration, cfg_from=cfg_from)
    ply = model / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    pruned = ot.prune_gaussians(ply, ref_ply) if ref_ply else None     # on the copy only
    if pruned:
        print(f"[mesh] Gaussians inside the object bbox: {pruned['kept']}/{pruned['before']}", flush=True)
    p = ot.mesh_params(str(ref_ply or ply), cameras_json, voxels_across=200)
    _sh(f"python {TOOLS}/omegas_launch.py mesh_2D.py -s {src_data} -m {model} --iteration {iteration} --depth_ratio 1.0 "
        f"--voxel_size {p['voxel_size']} --sdf_trunc {p['sdf_trunc']} --depth_trunc {p['depth_trunc']} --skip_test",
        st, log_name=log_name)
    tdir = model / "train" / f"ours_{iteration}"
    clean = ot.clean_mesh(tdir / "fuse.ply", str(ref_ply or ply), out / f"object_{OBJ_ID}_mesh.ply",
                          out / f"object_{OBJ_ID}_mesh.obj")
    info = {"iteration": iteration, "source_model": src_model, "source_data": src_data, "tsdf": p,
            "raw_meshes": str(tdir), "pruned_gaussians": pruned, "mesh": clean}
    (out / "mesh_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2), flush=True)
    if clean["clean"]["triangles"] < 1000:
        raise RuntimeError(f"mesh too small ({clean['clean']['triangles']} triangles): see {out}/mesh_info.json")
    return info


# --------------------------------------------------------------------------------------------------
# 4. Steps
# --------------------------------------------------------------------------------------------------
@app.function(image=prep_image, cpu=16, memory=16384, volumes=VOLUMES, timeout=3 * 3600, retries=RETRIES)
def step0_prepare_data(force: bool = False):
    vol.reload()
    if not Path(f"{RAW}/transforms.json").exists():
        raise RuntimeError(f"{RAW}/transforms.json missing: run ::upload first")
    st = _begin(0, force)
    if st is None:
        return
    try:
        if force:   # keep the COLMAP feature/match cache, rebuild everything else
            for sub in ("images", "object_mask", "sparse"):
                shutil.rmtree(Path(D) / sub, ignore_errors=True)
        with PeriodicCommit():
            _sh(f"python {TOOLS}/prepare_teatime.py --raw {RAW} --out {D} "
                f"--report {O}/{st['dir']}/report.json --threads 16", st, cwd=TOOLS)
        _finish(0, st, data=D)
    except Exception as e:  # noqa: BLE001
        _fail(0, st, e)


@app.function(image=prep_image, cpu=4, memory=8192, volumes=VOLUMES, timeout=3600, retries=RETRIES)
def step1_check_masks(force: bool = False):
    vol.reload()
    _require(0)
    st = _begin(1, force)
    if st is None:
        return
    try:
        _sh(f"python {TOOLS}/check_masks.py --data {D} --id {OBJ_ID} --out {O}/{st['dir']}", st, cwd=TOOLS)
        rep = json.loads((Path(O) / st["dir"] / "mask_validation_report.json").read_text())["summary"]
        _finish(1, st, summary=rep)
    except Exception as e:  # noqa: BLE001
        _fail(1, st, e)


@app.function(image=main_image, gpu=GPU, cpu=8, memory=32768, volumes=VOLUMES, timeout=8 * 3600, retries=RETRIES)
def step2_train_scene(force: bool = False):
    vol.reload()
    _require(1)
    st = _begin(2, force)
    if st is None:
        return
    m = f"{O}/{st['dir']}"
    try:
        # no --load_checkpoint: restore() is commented out in run_gaussian_2D.py, a retry trains from 0
        with PeriodicCommit(prune_dir=m):
            _sh(f"python run_gaussian_2D.py -s {D} -m {m} --config_file configs/gaussian_dataset/train.json "
                f"--depth_ratio 1.0 --iterations 30000 --save_iterations {SAVE_ITERS} "
                f"--checkpoint_iterations {SAVE_ITERS}", st)
        it = Path(m) / "point_cloud" / "iteration_30000"
        for need in (it / "point_cloud.ply", it / "classifier.pth", Path(m) / "cameras.json", Path(m) / "cfg_args"):
            if not need.exists():
                raise RuntimeError(f"missing {need}")
        _finish(2, st, model=m, gaussians=_tools().ply_stats(str(it / "point_cloud.ply"))["n_gaussians"])
    except Exception as e:  # noqa: BLE001
        _fail(2, st, e)


@app.function(image=main_image, gpu=GPU, cpu=4, memory=32768, volumes=VOLUMES, timeout=2 * 3600, retries=RETRIES)
def step3_id_check(force: bool = False, min_iou: float = 0.5):
    vol.reload()
    s2 = _require(2)
    st = _begin(3, force)
    if st is None:
        return
    try:
        model = f"{O}/{st['dir']}/model"
        _copy_model(s2["model"], model, 30000)
        # The authors' scripts disagree on the mask format: run_gaussian_2D.py reads object masks as H x W x 3
        # (gt_obj[:, :, 0]) while render_obj.py::visualize_gt needs H x W. render_obj.py gets its own copy of the
        # data with 1-channel masks (same images / poses / IDs); training data stays as it is.
        d1 = Path(f"{VOL}/data/teatime_1ch")
        if not (d1 / "object_mask").exists() or len(list((d1 / "object_mask").glob("*.png"))) != len(list((Path(D) / "object_mask").glob("*.png"))):
            import numpy as np
            from PIL import Image
            shutil.rmtree(d1, ignore_errors=True)
            shutil.copytree(Path(D) / "images", d1 / "images")
            shutil.copytree(Path(D) / "sparse", d1 / "sparse")
            (d1 / "object_mask").mkdir(parents=True)
            for p in (Path(D) / "object_mask").glob("*.png"):
                a = np.asarray(Image.open(p))
                Image.fromarray((a[..., 0] if a.ndim == 3 else a).astype(np.uint8)).save(d1 / "object_mask" / p.name)
            _commit("data/teatime_1ch")
        with PeriodicCommit():
            _sh(f"python {TOOLS}/omegas_launch.py render_obj.py -m {model} -s {d1} --iteration 30000 --config_file {CFG}", st)
        rep = _tools().id_iou(f"{model}/objects_pred", f"{D}/object_mask", OBJ_ID, f"{O}/{st['dir']}/iou_report.json")
        if rep["mean_iou"] < min_iou:
            raise RuntimeError(f"mean IoU {rep['mean_iou']} < {min_iou} for ID {OBJ_ID} (see iou_report.json); "
                               f"rerun with --min-iou 0 to continue anyway")
        _finish(3, st, iou=rep)
    except Exception as e:  # noqa: BLE001
        _fail(3, st, e)


@app.function(image=main_image, gpu=GPU, cpu=4, memory=32768, volumes=VOLUMES, timeout=2 * 3600, retries=RETRIES)
def step4_extract_object(force: bool = False):
    vol.reload()
    s2 = _require(2)
    st = _begin(4, force)
    if st is None:
        return
    try:
        model = f"{O}/{st['dir']}/model"
        _copy_model(s2["model"], model, 30000)
        with PeriodicCommit():
            _sh(f"python {TOOLS}/omegas_launch.py extract_object.py -m {model} -s {D} --iteration 30000 --config_file {CFG}", st)
        obj = Path(model) / f"object_{OBJ_ID}"
        it = obj / "point_cloud" / "iteration_30000"
        # dream_gaussian_2D.py (Step 5) reads these two files from the object folder; extract_object saves only the ply
        shutil.copy2(Path(s2["model"]) / "cameras.json", obj / "cameras.json")
        shutil.copy2(Path(s2["model"]) / "point_cloud" / "iteration_30000" / "classifier.pth", it / "classifier.pth")
        stats = _tools().ply_stats(str(it / "point_cloud.ply"))
        print(json.dumps(stats, indent=2), flush=True)
        if not 200 <= stats["n_gaussians"] <= 3_000_000:
            raise RuntimeError(f"implausible object size: {stats['n_gaussians']} Gaussians")
        _finish(4, st, object_dir=str(obj), object_stats=stats)
    except Exception as e:  # noqa: BLE001
        _fail(4, st, e)


@app.function(image=main_image, gpu=["A100-40GB", "L40S", "A100-80GB"], cpu=8, memory=32768, volumes=VOLUMES, timeout=2 * 3600,
              retries=RETRIES)
def dreambooth_train(object_images: str, out_dir: str, log_path: str, steps: int = 600):
    """Paper TR, Eq. 8: DreamBooth personalization of SD 2.1 on the segmented target views (UNet + text encoder,
    full fine-tune -> A100-40GB; L4 24 GB is too small without 8-bit Adam)."""
    vol.reload()
    st = {"dir": str(Path(log_path).parent.parent.relative_to(O)), "attempts": 1}
    with PeriodicCommit():
        _sh(f"python {TOOLS}/tr_dreambooth.py --object_images {object_images} --out {out_dir} --steps {steps}",
            st, log_name=Path(log_path).name)
    if not (Path(out_dir) / "unet" / "config.json").exists():
        raise RuntimeError("DreamBooth produced no unet/")
    return json.loads((Path(out_dir) / "info.json").read_text())


@app.function(image=main_image, gpu=GPU, cpu=8, memory=32768, volumes=VOLUMES, timeout=8 * 3600, retries=RETRIES)
def step5_completion(force: bool = False, mask_seeds: int = 10, dreambooth_steps: int = 600):
    """Target Replenishment (paper): 5P DreamBooth personalization of SD 2.1 on the target views ->
    5A n random novel views of the target 2DGS, mask from the personalized SD noise residual at t=991 averaged over
    10 seeds, SD-2 inpainting of the masked areas (dream_gaussian_2D.py::random_render_mask) ->
    5B further 2DGS training with the original target views + the inpainted views.
    Uses the reconstructed dreamgaussian/gs_renderer_2D (see modal_omegas/gs_renderer_2D.py)."""
    vol.reload()
    s4 = _require(4)
    st = _begin(5, force, keep_partial=True)
    if st is None:
        return
    ot = _tools()
    base = Path(O) / st["dir"]
    obj = base / f"object_{OBJ_ID}"
    refined = obj / "refined"
    r2 = base / "refined_2dgs"
    dbdir = base / "dreambooth" / "refined_sd_finetuned"
    try:
        if "5P" not in st["parts"]:
            _sh(f"python {TOOLS}/omegas_launch.py --preflight", st, log_name="5P.log")   # base SD + inpainting cache
            _commit("hf cache")
            st["dreambooth"] = dreambooth_train.remote(str(Path(s4["object_dir"]) / "images"), str(dbdir),
                                                       str(base / "logs" / "5P_dreambooth.log"), dreambooth_steps)
            vol.reload()
            st["parts"].append("5P")
            _save_state(5, st)
            _commit("5P done")

        if "5A" not in st["parts"]:
            shutil.rmtree(obj, ignore_errors=True)
            shutil.copytree(s4["object_dir"], obj, ignore=shutil.ignore_patterns("refined"))
            ov = ot.dream_overrides(str(obj / "point_cloud" / "iteration_30000" / "point_cloud.ply"), str(obj / "cameras.json"))
            st["dream_overrides"] = ov
            _save_state(5, st)
            env = {"OMEGAS_DREAMBOOTH_DIR": str(dbdir), "OMEGAS_MASK_SEEDS": str(mask_seeds)}
            with PeriodicCommit():
                _sh(f"python {TOOLS}/omegas_launch.py --preflight", st, env=env, log_name="5A.log")
                cli = " ".join(f"{k}={ov[k]}" for k in ("W", "H", "_W", "_H", "fovx", "fovy", "radius"))
                _sh(f"python {TOOLS}/omegas_launch.py dream_gaussian_2D.py --config {TOOLS}/dream_teatime_obj11.yaml "
                    f"load={obj}/ {cli}", st, env=env, log_name="5A.log")
            # the original views of the object (extract_object.py: object_11/images/<frame>.jpg) join the random views
            imgs = refined / "images"
            n_orig = 0
            for p in (obj / "images").glob("*.jpg"):
                if not (imgs / p.name).exists():
                    shutil.copy2(p, imgs / p.name)
                    n_orig += 1
            n_seed = ot.seed_points(str(obj / "point_cloud" / "iteration_30000" / "point_cloud.ply"), str(refined / "points3D.ply"))
            cams = json.loads((refined / "cameras.json").read_text())
            missing = [c["img_name"] for c in cams if not (imgs / f"{c['img_name']}.jpg").exists()]
            if missing:
                raise RuntimeError(f"{len(missing)} cameras without image in refined/images, e.g. {missing[:5]}")
            st["parts"].append("5A")
            st["refined"] = {"cameras": len(cams), "random_views": sum(1 for c in cams if c["img_name"].startswith("random_")),
                             "original_views": n_orig, "seed_points": n_seed}
            _save_state(5, st)
            _commit("5A done")

        if "5B" not in st["parts"]:
            shutil.rmtree(r2, ignore_errors=True)
            snap = {"done": False}

            def watcher(stop: threading.Event):
                ply = r2 / "point_cloud" / f"iteration_{SNAPSHOT_ITER}" / "point_cloud.ply"
                last = -1
                while not stop.wait(20):
                    if not ply.exists():
                        continue
                    size = ply.stat().st_size
                    if size != last:
                        last = size
                        continue
                    try:
                        _commit(f"before snapshot {SNAPSHOT_ITER}")
                        snap["info"] = _mesh(st, str(r2), str(refined), SNAPSHOT_ITER,
                                             str(base / f"mesh_snapshot_{SNAPSHOT_ITER}"), str(r2),
                                             str(r2 / "cameras.json"), "mesh_snapshot.log")
                        snap["done"] = True
                        _commit(f"mesh snapshot {SNAPSHOT_ITER}")
                    except Exception as e:  # noqa: BLE001  (a failed snapshot must not stop training)
                        snap["error"] = str(e)
                        print(f"[snapshot] failed: {e}", flush=True)
                    return

            stop = threading.Event()
            th = threading.Thread(target=watcher, args=(stop,), daemon=True)
            th.start()
            with PeriodicCommit(prune_dir=str(r2)):
                _sh(f"python run_gaussian_2D.py -s {refined} -m {r2} --no_obj_loss --depth_ratio 1.0 --lambda_dist 1000 "
                    f"--iterations 30000 --save_iterations {SAVE_ITERS} --checkpoint_iterations {SAVE_ITERS}",
                    st, log_name="5B.log")
                th.join(timeout=1800)
                stop.set()
            if not (r2 / "point_cloud" / "iteration_30000" / "point_cloud.ply").exists():
                raise RuntimeError("5B finished without point_cloud/iteration_30000")
            st["parts"].append("5B")
            st["snapshot"] = snap
        _finish(5, st, refined_dir=str(refined), refined_model=str(r2))
    except Exception as e:  # noqa: BLE001
        _fail(5, st, e)


@app.function(image=main_image, gpu=GPU, cpu=8, memory=32768, volumes=VOLUMES, timeout=3 * 3600, retries=RETRIES)
def step6_mesh(force: bool = False, source: str = "step4"):
    """TSDF mesh (mesh_2D.py) of the object. source=step4: extracted object Gaussians (v1, no completion);
    source=step5: refined 2DGS model of Step 5 (v2)."""
    vol.reload()
    s2 = _require(2)
    prev = _load_state(6)
    if prev and prev.get("source") != source:
        force = True                      # a different source is always a new version
    st = _begin(6, force, source=source)
    if st is None:
        return
    base = Path(O) / st["dir"]
    try:
        with PeriodicCommit():
            if source == "step4":
                s4 = _require(4)
                info = _mesh(st, s4["object_dir"], D, 30000, str(base / "final" / "mesh_iter_30000"),
                             s2["model"], f"{s2['model']}/cameras.json", "run.log")
            elif source == "step5":
                s5 = _require(5)
                s4 = _require(4)
                ref = f"{s4['object_dir']}/point_cloud/iteration_30000/point_cloud.ply"   # object bbox (Step 4)
                cams = f"{s5['refined_model']}/cameras.json"
                info = {}
                for it, sub in ((SNAPSHOT_ITER, "intermediate"), (30000, "final")):
                    info[sub] = _mesh(st, s5["refined_model"], s5["refined_dir"], it, str(base / sub / f"mesh_iter_{it}"),
                                      s5["refined_model"], cams, f"mesh_{it}.log", ref_ply=ref)
                    _commit(f"mesh {it}")
            else:
                raise ValueError(f"unknown source {source}")
        n = _tools().manifest(str(base), exclude_dirs=("model",))
        _finish(6, st, mesh=info, manifest_files=n)
    except Exception as e:  # noqa: BLE001
        _fail(6, st, e)


STEP_FUNCS = {0: step0_prepare_data, 1: step1_check_masks, 2: step2_train_scene, 3: step3_id_check,
              4: step4_extract_object, 5: step5_completion, 6: step6_mesh}


@app.function(image=prep_image, cpu=1, memory=1024, volumes=VOLUMES, timeout=24 * 3600)
def run_all(steps: str = "0,1,2,3,4", force: str = "", mesh_source: str = "step4", min_iou: float = 0.5):
    """Orchestrator ON Modal: the chain continues after the laptop disconnects (modal run --detach)."""
    todo = [int(s) for s in steps.split(",") if s.strip()]
    forced = {int(s) for s in force.split(",") if s.strip()}
    for i in todo:
        kw = {"force": i in forced}
        if i == 3:
            kw["min_iou"] = min_iou
        if i == 6:
            kw["source"] = mesh_source
        print(f"\n######## step{i} {STEP_DIRS[i]} {kw} ########", flush=True)
        t0 = time.time()
        STEP_FUNCS[i].remote(**kw)
        print(f"######## step{i} finished in {(time.time() - t0) / 60:.1f} min", flush=True)
    vol.reload()
    print(json.dumps({i: (_load_state(i) or {}).get("status") for i in range(7)}, indent=1), flush=True)


# --------------------------------------------------------------------------------------------------
# 5. Local entrypoints
# --------------------------------------------------------------------------------------------------
def _assert_workspace() -> None:
    r = subprocess.run([sys.executable, "-m", "modal", "profile", "current"], capture_output=True, text=True)
    current = r.stdout.strip()
    if current != WORKSPACE_PROFILE:
        raise SystemExit(f"Modal profile is '{current}', expected '{WORKSPACE_PROFILE}': "
                         f"python -m modal profile activate {WORKSPACE_PROFILE}")


def _remote_json(path: str):
    try:
        return json.loads(b"".join(vol.read_file(path)))
    except Exception:  # noqa: BLE001
        return None


@app.local_entrypoint()
def upload(force: bool = False):
    """Raw teatime (images, mask/labels, objects.json, transforms.json) -> volume:/raw/teatime."""
    _assert_workspace()
    if not force and _remote_json("raw/teatime/transforms.json") is not None:
        print("raw/teatime already on the volume (use --force to upload again)")
        return
    with vol.batch_upload(force=True) as b:
        b.put_directory(str(LOCAL_RAW / "images"), "raw/teatime/images")
        b.put_directory(str(LOCAL_RAW / "mask" / "labels"), "raw/teatime/mask/labels")
        b.put_file(str(LOCAL_RAW / "mask" / "objects.json"), "raw/teatime/mask/objects.json")
        b.put_file(str(LOCAL_RAW / "transforms.json"), "raw/teatime/transforms.json")
    n_img = sum(1 for e in vol.listdir("raw/teatime/images") if e.path.endswith(".jpg"))
    n_msk = sum(1 for e in vol.listdir("raw/teatime/mask/labels") if e.path.endswith(".png"))
    print(f"uploaded: {n_img} images, {n_msk} masks")
    if n_img != 180 or n_msk != 180:
        raise SystemExit("expected 180 images and 180 masks")


@app.local_entrypoint()
def start(steps: str = "0,1,2,3,4", force: str = "", mesh_source: str = "step4", min_iou: float = 0.5):
    _assert_workspace()
    print(f"[start] steps={steps} force={force} mesh_source={mesh_source} on {WORKSPACE_PROFILE}/{GPU}")
    run_all.remote(steps=steps, force=force, mesh_source=mesh_source, min_iou=min_iou)


@app.local_entrypoint()
def status():
    for i in range(7):
        st = _remote_json(f"outputs/{RUN_NAME}/state/step{i}.json")
        if st:
            short = {k: st.get(k) for k in ("dir", "status", "attempts", "parts", "started", "finished", "error") if st.get(k)}
            print(f"step{i}: {json.dumps(short, ensure_ascii=False)}")
        else:
            print(f"step{i}: -")


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_local(folder: Path) -> list[str]:
    problems = []
    for mf in folder.rglob("manifest.json"):
        for rel, info in json.loads(mf.read_text()).items():
            p = mf.parent / rel
            if not p.exists():
                problems.append(f"missing {p}")
            elif p.stat().st_size != info["bytes"] or _sha256(p) != info["sha256"]:
                problems.append(f"corrupt {p}")
    return problems


@app.local_entrypoint()
def download(step: int = 6, include_models: bool = False):
    """Copy outputs/<run>/<stepN dir> to the local outputs folder (never overwrites: _dl2, _dl3 ...) and verify."""
    st = _remote_json(f"outputs/{RUN_NAME}/state/step{step}.json")
    if not st:
        raise SystemExit(f"step{step} has no state on the volume")
    remote = f"outputs/{RUN_NAME}/{st['dir']}"
    dest = LOCAL_OUT / st["dir"]
    k = 2
    while dest.exists():
        dest = LOCAL_OUT / f"{st['dir']}_dl{k}"
        k += 1
    n = 0
    for e in vol.listdir(remote, recursive=True):
        if e.type != modal.volume.FileEntryType.FILE:
            continue
        rel = e.path[len(remote) + 1:]
        if not include_models and ("/model/" in f"/{rel}" or rel.endswith(".pth")):
            continue
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            for chunk in vol.read_file(e.path):
                f.write(chunk)
        n += 1
    print(f"downloaded {n} files -> {dest}")
    probs = _verify_local(dest)
    print("\n".join(probs) if probs else "verify: OK (sha256 matches manifest.json)" if any(dest.rglob("manifest.json"))
          else "verify: no manifest.json in this step (only Step 6 writes one)")


@app.local_entrypoint()
def verify(folder: str):
    probs = _verify_local(Path(folder))
    print("\n".join(probs) if probs else "OK: every file matches its manifest.json")
