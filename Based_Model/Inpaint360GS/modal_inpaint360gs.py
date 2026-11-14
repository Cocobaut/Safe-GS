"""
Inpaint360GS on LERF teatime, removing the napkin, on Modal (workspace nguyenkhacd10) -- see plan.md
====================================================================================================
Runs the original scripts (run_seg.sh -> run_remove.sh -> run_inpaint.sh, scripts/run_data_prepare.sh) unchanged,
one Modal function per step. Each step is copied to /vol/export/teatime_napkin/stepN_<name>[_vK] and downloaded to
outputs/teatime_napkin/.

Preemption / GPU loss (Modal can stop a container at any time):
  * every step function commits the volume every COMMIT_EVERY_S seconds and on SIGTERM;
  * 3DGS training writes official checkpoints (--checkpoint_iterations, every CKPT_EVERY iterations) and resumes
    from the newest loadable one with --start_checkpoint;
  * step state lives in /vol/state/stepN.json: a retried (preempted) input resumes the same run and export folder,
    a finished step is skipped unless --force (which creates the next _vK export).

Commands (python -m modal ...; profile must be nguyenkhacd10; use `run --detach` so a closed laptop does not stop GPU work):
  modal run modal_inpaint360gs.py::upload                    # LERF images + existing SAM masks -> volume
  modal run modal_inpaint360gs.py::upload_weight --path <CropFormer_hornet_3x_03823a.pth>   # gated HF weight
  modal run --detach modal_inpaint360gs.py::run_step --step 0   # COLMAP (convert.py)         [CPU]
  modal run modal_inpaint360gs.py::accept_step0              # keep the images COLMAP registered (user decision)
  modal run --detach modal_inpaint360gs.py::run_step --step 1   # train 3DGS                  [L4]
  modal run --detach modal_inpaint360gs.py::run_step --step 2   # Mask2D (check SAM / HQ-SAM)  [L4]
  modal run --detach modal_inpaint360gs.py::run_step --step 3   # Sync mask + images_2_num     [L4]
  modal run --detach modal_inpaint360gs.py::run_step --step 4   # ID distillation + render     [L4]
  modal run --detach modal_inpaint360gs.py::run_step --step 5 --target-id <id> --surrounding-id None
  modal run modal_inpaint360gs.py::samt                      # interactive SAM-Track (Gradio)  [L4]
  modal run modal_inpaint360gs.py::samt_stop                 # (other terminal) save masks and stop the app
  modal run --detach modal_inpaint360gs.py::run_step --step 6   # LaMa color + depth          [L4]
  modal run --detach modal_inpaint360gs.py::run_step --step 7   # plyfusion + 3D inpaint      [L4]
  modal run modal_inpaint360gs.py::run_step --step 8         # collect renders                [CPU]
  modal run modal_inpaint360gs.py::download [--step N]       # -> outputs/teatime_napkin (never overwrites)
  modal run modal_inpaint360gs.py::status
"""

from __future__ import annotations

import fnmatch
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import modal

# --------------------------------------------------------------------------------------------------
# 1. Configuration
# --------------------------------------------------------------------------------------------------
APP_NAME = "inpaint360gs"
WORKSPACE_PROFILE = "nguyenkhacd10"
GPU = "L4"

DATASET = "others"          # teatime has no test_ images -> "others" branch of the original scripts
SCENE = "teatime"
OBJECT = "napkin"
RESOLUTION = 2              # raw_mask_sam.py / add_label_num_hqsam.py read images_{resolution}; 1 is not supported
RUN_NAME = f"{SCENE}_{OBJECT}"

COMMIT_EVERY_S = 300        # volume commit interval (preemption safety)
CKPT_EVERY = 5000           # 3DGS checkpoint interval (iterations)
CKPT_KEEP = 2               # keep the newest N checkpoints (the newest may be half-written when preempted)

LOCAL_ROOT = Path(__file__).resolve().parent
LERF_SCENE = Path(r"F:\LERF Datasets\teatime\teatime")
LOCAL_OUT = LOCAL_ROOT / "outputs" / RUN_NAME

VOL = "/vol"
REPO = "/opt/inpaint360gs"
D = f"{VOL}/data/{DATASET}/{SCENE}"                       # source path (-s)
M = f"{VOL}/output/{DATASET}/{SCENE}"                     # model path (-m)
D_REL = f"data/{DATASET}/{SCENE}"
M_REL = f"output/{DATASET}/{SCENE}"
LERF_MASK = f"{VOL}/lerf_mask"
EXPORT = f"{VOL}/export/{RUN_NAME}"
STATE = f"{VOL}/state"
SAMT = f"{REPO}/Segment-and-Track-Anything"
LAMA_DATA_NAME = f"360_{SCENE}_virtual"
CROPFORMER_WEIGHT = "CropFormer_hornet_3x_03823a.pth"

STEP_NAMES = {
    0: "data", 1: "3dgs", 2: "mask2d", 3: "sync_mask", 4: "id_3dgs",
    5: "virtual_cam", 6: "inpaint2d", 7: "inpaint_3dgs", 8: "render",
}

# repo path (inside the image) -> volume path: everything the original scripts write lands on the volume
VOLUME_LINKS = {
    "data": "data",
    "output": "output",
    "LaMa/data": "lama/data",
    "LaMa/output": "lama/output",
    "Segment-and-Track-Anything/assets": "samt/assets",
    "Segment-and-Track-Anything/tracking_results": "samt/tracking_results",
    f"config/object_removal/{DATASET}": f"config/object_removal/{DATASET}",
    f"config/object_inpaint/{DATASET}": f"config/object_inpaint/{DATASET}",
}

app = modal.App(APP_NAME)
vol = modal.Volume.from_name("inpaint360gs", create_if_missing=True)
ctl = modal.Dict.from_name("inpaint360gs-ctl", create_if_missing=True)
RETRIES = modal.Retries(max_retries=2, backoff_coefficient=1.0, initial_delay=30.0)

BASE_ENV = {
    "PYTHONUNBUFFERED": "1",
    "QT_QPA_PLATFORM": "offscreen",
    "MPLBACKEND": "Agg",
    "WANDB_MODE": "disabled",
}

# --------------------------------------------------------------------------------------------------
# 2. Images
# --------------------------------------------------------------------------------------------------
REPO_IGNORE = [
    "outputs", "outputs/**", "data", "data/**", "output", "output/**", "LaMa", "LaMa/**",
    "Paper", "Paper/**", "media", "media/**", "plan.md", "modal_inpaint360gs.py",
    "tools_safe/label_color_ids.py",
    "gaussian_splatting/SIBR_viewers", "gaussian_splatting/SIBR_viewers/**",
    "gaussian_splatting/assets", "gaussian_splatting/assets/**",
    "**/third_party/glm/doc", "**/third_party/glm/doc/**",
    "**/third_party/glm/test", "**/third_party/glm/test/**",
    "seg/detectron2/docs", "seg/detectron2/docs/**", "seg/detectron2/tests", "seg/detectron2/tests/**",
    "seg/Entity", "seg/Entity/**", "logs_local", "logs_local/**", "**/__pycache__", "**/__pycache__/**",
    "**/*.pyc", "**/.git", "**/.git/**",
]

CROPFORMER = f"{REPO}/seg/detectron2/detectron2/projects/CropFormer"

# env inpaint360gs (docs/install.md + install.sh, same order); colmap/imagemagick for convert.py
main_image = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "wget", "curl", "unzip", "build-essential", "ninja-build", "cmake",
                 "colmap", "imagemagick", "ffmpeg", "libgl1", "libglib2.0-0", "libsm6", "libxext6")
    .env({**BASE_ENV, "CUDA_HOME": "/usr/local/cuda", "FORCE_CUDA": "1", "CC": "gcc", "CXX": "g++",
          "TORCH_CUDA_ARCH_LIST": "8.6;8.9", "MMCV_WITH_OPS": "0"})
    .pip_install("torch==2.0.0+cu118", "torchvision==0.15.1+cu118", "torchaudio==2.0.1+cu118",
                 extra_index_url="https://download.pytorch.org/whl/cu118")
    .pip_install("wheel", "ninja", "cython", "gdown", "ipython", "tensorboard")
    .add_local_dir(str(LOCAL_ROOT), REPO, copy=True, ignore=REPO_IGNORE)
    .run_commands(
        # install.sh steps 2-4
        f"cd {REPO} && pip install gaussian_splatting/submodules/diff-gaussian-rasterization --no-build-isolation",
        f"cd {REPO} && pip install gaussian_splatting/submodules/simple-knn --no-build-isolation",
        f"cd {REPO} && pip install submodules/diff-gaussian-rasterization --no-build-isolation",
        f"cd {REPO} && pip install . --no-build-isolation",
        gpu=GPU,  # CUDA extensions check torch.cuda.is_available() at build time
    )
    .run_commands(  # separate layers: a change in one group does not rebuild the others
        # install.sh step 5 (SAM, detectron2, CropFormer ops, fvcore patch)
        "pip install git+https://github.com/facebookresearch/segment-anything.git",
        "pip install setuptools==75.8.0",
        f"cd {REPO}/seg && python -m pip install -e detectron2 --no-build-isolation",
        f"cd {CROPFORMER}/entity_api/PythonAPI && make",
        f"cd {CROPFORMER}/mask2former/modeling/pixel_decoder/ops && sh make.sh",
        f"cd {REPO} && python seg/patch_fvcore.py",
        # CropFormer weight is gated on HuggingFace -> not baked in; uploaded to /vol/weights (::upload_weight)
        f"mkdir -p {REPO}/seg/weight",
        gpu=GPU,
    )
    .run_commands(
        # install.sh step 6 (Segment-and-Track-Anything: script/install.sh + script/download_ckpt.sh)
        f"cd {SAMT} && bash script/install.sh",
        f"mkdir -p {SAMT}/ckpt {SAMT}/ast_master/pretrained_models",
        f"cd {SAMT} && gdown 1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ -O ./ckpt/R50_DeAOTL_PRE_YTB_DAV.pth",
        f"cd {SAMT} && wget -q -P ./ckpt https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        f"cd {SAMT} && wget -q -P ./ckpt https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
        f"cd {SAMT} && (wget -q -O ./ast_master/pretrained_models/audio_mdl.pth "
        "'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1' || echo 'audio ckpt skipped (audio unused)')",
        gpu=GPU,
    )
    # SAM-Track install.sh pins websockets==10.4 for gradio 3.39, but pip pulls the newest uvicorn, whose websockets
    # backend needs websockets>=13 (ImportError: ServerProtocol); uvicorn 0.23.2 is the gradio-3.39-era release
    # and the newest starlette changed TemplateResponse's signature (TypeError: unhashable type 'dict'); fastapi 0.104.1
    # pulls the matching starlette 0.27
    .run_commands('pip install "uvicorn==0.23.2" "fastapi==0.104.1" "pydantic==2.5.3"')
)

# env lama (LaMa/conda_env.yml + requirements.txt). The original cudatoolkit 10.2 cannot run on L4 (sm_89),
# so torch 1.12.1+cu113 on Python 3.10 (oldest Python Modal supports) is used with the same python-level packages
# (environment change only; scikit-learn 1.0.2 and opencv 4.5.5.64: the pinned versions have no cp310 wheel).
lama_image = (
    modal.Image.from_registry("nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04", add_python="3.10")
    .apt_install("wget", "unzip", "libgl1", "libglib2.0-0", "libsm6", "libxext6")
    .env(BASE_ENV)
    .pip_install("torch==1.12.1+cu113", "torchvision==0.13.1+cu113",
                 extra_index_url="https://download.pytorch.org/whl/cu113")
    .pip_install(
        "numpy==1.21.6", "scipy==1.7.3", "pyyaml", "tqdm", "easydict==1.9", "scikit-image==0.19.3",
        "scikit-learn==1.0.2", "joblib", "matplotlib==3.5.3", "pandas==1.3.5", "albumentations==0.5.2",
        "imgaug==0.4.0", "shapely==1.8.5", "hydra-core==1.1.0", "omegaconf==2.1.1",
        "pytorch-lightning==1.2.9", "torchmetrics==0.2.0", "kornia==0.5.0", "tabulate", "webdataset==0.1.76",
        "braceexpand==0.1.7", "packaging", "opencv-python-headless==4.5.5.64", "protobuf==3.20.3",
        "tensorboard", "future",
    )
    .run_commands(
        "mkdir -p /opt/weights && cd /opt/weights && "
        "wget -q https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip && unzip -q big-lama.zip && rm big-lama.zip"
    )
    .add_local_dir(str(LOCAL_ROOT / "LaMa"), f"{REPO}/LaMa", copy=True,
                   ignore=["data", "data/**", "output", "output/**", "big-lama", "big-lama/**", "**/__pycache__/**"])
    # the released repo lacks LaMa/saicinpainting/training/data/ (imported by bin/predict_*.py and trainers); restore
    # the 4 upstream files (advimman/lama, pinned). The authors' depth (.npy) loader lives in evaluation/data.py
    # (present, unchanged); none of these files calls the modified load_image.
    .run_commands(
        f"mkdir -p {REPO}/LaMa/saicinpainting/training/data && cd {REPO}/LaMa/saicinpainting/training/data && "
        "for f in __init__.py aug.py datasets.py masks.py; do wget -q -O $f "
        "https://raw.githubusercontent.com/advimman/lama/786f5936b27fb3dacd2b1ad799e4de968ea697e7/"
        "saicinpainting/training/data/$f; done"
    )
)

VOLUMES = {VOL: vol}

# --------------------------------------------------------------------------------------------------
# 3. Container helpers
# --------------------------------------------------------------------------------------------------

def _link_volume(links: dict[str, str]) -> None:
    """Replace repo folders by symlinks to the volume so every output of the original code persists."""
    for repo_rel, vol_rel in links.items():
        target = Path(VOL) / vol_rel
        target.mkdir(parents=True, exist_ok=True)
        link = Path(REPO) / repo_rel
        if link.is_symlink():
            continue
        if link.is_dir():
            for p in link.iterdir():
                if not (target / p.name).exists():
                    shutil.move(str(p), str(target / p.name))
            shutil.rmtree(link)
        elif link.exists():
            link.unlink()
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target, target_is_directory=True)


def _commit(reason: str = "") -> None:
    try:
        vol.commit()
        print(f"[volume] committed {reason}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[volume] commit failed ({reason}): {e}", flush=True)


def _prune_checkpoints(folder: str) -> None:
    ckpts = sorted(Path(folder).glob("chkpnt*.pth"), key=lambda p: int(p.stem.replace("chkpnt", "")))
    for p in ckpts[:-CKPT_KEEP]:
        p.unlink(missing_ok=True)


class PeriodicCommit:
    """Commit the volume every COMMIT_EVERY_S seconds and on SIGTERM, so a preempted container loses little."""

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
        except ValueError:  # not in the main thread: periodic commits still run
            pass
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        _commit("end of step")
        return False


def _state_path(step) -> Path:
    return Path(STATE) / f"step{step}.json"


def _load_state(step) -> dict | None:
    p = _state_path(step)
    return json.loads(p.read_text()) if p.exists() else None


def _save_state(step, st: dict) -> None:
    Path(STATE).mkdir(parents=True, exist_ok=True)
    _state_path(step).write_text(json.dumps(st, indent=2, ensure_ascii=False))


def _next_export_name(step: int) -> str:
    base = f"step{step}_{STEP_NAMES[step]}"
    if not (Path(EXPORT) / base).exists():
        return base
    k = 2
    while (Path(EXPORT) / f"{base}_v{k}").exists():
        k += 1
    return f"{base}_v{k}"


def _begin(step: int, force: bool) -> dict | None:
    """Start or resume a step. Returns None if the step is already done (and not forced)."""
    st = _load_state(step)
    if st and st["status"] == "done" and not force:
        print(f"[step {step}] already done -> {st['export']} (use --force for a new version)")
        return None
    if st and st["status"] == "running":
        st["attempts"] += 1
        print(f"[step {step}] resuming {st['export']} (attempt {st['attempts']})")
    else:
        st = {"step": step, "status": "running", "t0": time.time(), "export": _next_export_name(step),
              "attempts": 1, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        print(f"[step {step}] new run -> {st['export']}")
    (Path(EXPORT) / st["export"]).mkdir(parents=True, exist_ok=True)
    _save_state(step, st)
    _commit(f"step {step} begin")
    return st


def _finish(step: int, st: dict, status: str = "done", **extra) -> dict:
    st.update(extra, status=status, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    _save_state(step, st)
    (Path(EXPORT) / st["export"] / "state.json").write_text(json.dumps(st, indent=2, ensure_ascii=False))
    _commit(f"step {step} {status}")
    return st


def _sh(cmd: str, st: dict, cwd: str = REPO, env: dict | None = None) -> None:
    """Run one original command (bash), stream it and append it to logs/<export>.log."""
    log = Path(EXPORT) / "logs" / f"{st['export']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    full_env = {**os.environ, **(env or {})}
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n===== [attempt {st['attempts']}] {time.strftime('%H:%M:%S')} $ {cmd}\n")
        f.flush()
        print(f"$ {cmd}", flush=True)
        proc = subprocess.Popen(["bash", "-c", f"set -eo pipefail; {cmd}"], cwd=cwd, env=full_env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        rc = proc.wait()
        f.write(f"===== exit {rc}\n")
    if rc != 0:
        raise RuntimeError(f"command failed ({rc}): {cmd}")


def _export(st: dict, sources: list[tuple[str, str, list[str] | None, list[str]]], sub: str = "",
            t0: float | None = None) -> int:
    """Copy files created/modified since the step started. sources = [(root, dst_sub, include, exclude)]."""
    t0 = (t0 if t0 is not None else st["t0"]) - 2
    dst_root = Path(EXPORT) / st["export"] / sub
    n = 0
    for root, dst_sub, include, exclude in sources:
        if not os.path.isdir(root):
            continue
        for dirpath, _, files in os.walk(root):
            for name in files:
                src = os.path.join(dirpath, name)
                rel = os.path.relpath(src, root).replace(os.sep, "/")
                if include and not any(fnmatch.fnmatch(rel, p) for p in include):
                    continue
                if any(fnmatch.fnmatch(rel, p) for p in exclude):
                    continue
                if os.path.getmtime(src) < t0:
                    continue
                dst = dst_root / dst_sub / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                n += 1
    print(f"[export] {n} files -> {dst_root}")
    return n


def _container_setup() -> None:
    vol.reload()
    _link_volume(VOLUME_LINKS)
    weight, link = Path(VOL) / "weights" / CROPFORMER_WEIGHT, Path(REPO) / "seg" / "weight" / CROPFORMER_WEIGHT
    if weight.exists() and not link.exists():
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(weight)


# run_seg.sh: export PYTHONPATH=$(pwd):$(pwd)/seg/detectron2; CropFormer modules also import the top-level
# package `mask2former` (criterion_view.py), so its project folder is on the path too (no code change)
SEG_PYTHONPATH = f"{REPO}:{REPO}/seg/detectron2:{CROPFORMER}"


def _seg_env() -> dict:
    return {"PYTHONPATH": SEG_PYTHONPATH + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")}


def _repo_env() -> dict:
    """run_remove.sh / run_inpaint.sh: tools/virtual_pose.py imports the repo-root module `render`, so the repo root
    must be on PYTHONPATH (as in the authors' shell after run_seg.sh); environment only, no code change."""
    return {"PYTHONPATH": REPO + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")}

# --------------------------------------------------------------------------------------------------
# 4. Steps
# --------------------------------------------------------------------------------------------------

@app.function(image=main_image, cpu=8, memory=16384, volumes=VOLUMES, timeout=3 * 3600, retries=RETRIES)
def step0_prepare_data(force: bool = False):
    """README 'Run on your own dataset': convert.py (COLMAP) on <scene>/input. No test_ images -> no split."""
    _container_setup()
    if not os.path.isdir(f"{D}/input"):
        raise RuntimeError("no input images on the volume: run ::upload first")
    st = _begin(0, force)
    if st is None:
        return
    with PeriodicCommit():
        n_input = len([f for f in os.listdir(f"{D}/input") if f.lower().endswith((".jpg", ".png"))])
        # COLMAP is not resumable mid-way: start each attempt from a clean slate (input/ is kept)
        for sub in ("distorted", "sparse", "images", "images_2", "images_4", "images_8", "stereo"):
            shutil.rmtree(f"{D}/{sub}", ignore_errors=True)
        for f in Path(D).glob("run-colmap-*.sh"):
            f.unlink()
        # --no_gpu: the apt COLMAP build has no CUDA SIFT (same SIFT algorithm on CPU)
        _sh(f"python convert.py -s {D_REL} --resize --magick_executable convert --no_gpu", st)
        out = subprocess.run(
            ["python", "-c", "import sys; from tools.read_write_model import read_model; "
             f"c,i,p=read_model('{D}/sparse/0', '.bin'); print(len(i), len(p))"],
            cwd=REPO, capture_output=True, text=True, check=True).stdout.split()
        n_reg, n_pts = int(out[0]), int(out[1])
        sizes = {}
        for sub in ("images", "images_2"):
            first = sorted(os.listdir(f"{D}/{sub}"))[0]
            r = subprocess.run(["identify", "-format", "%wx%h", f"{D}/{sub}/{first}"], capture_output=True, text=True)
            sizes[sub] = r.stdout.strip()
        info = {"input_images": n_input, "registered_images": n_reg, "points3D": n_pts, "image_sizes": sizes}
        print(json.dumps(info, indent=2))
        _export(st, [(D, "", ["sparse/*"], [])], t0=0)  # image_undistorter keeps old mtimes for some files
        (Path(EXPORT) / st["export"] / "colmap_info.json").write_text(json.dumps(info, indent=2))
        (Path(EXPORT) / st["export"] / "images_list.txt").write_text("\n".join(sorted(os.listdir(f"{D}/images"))))
    if n_reg != n_input:
        _finish(0, st, status="failed", info=info,
                error=f"COLMAP registered {n_reg}/{n_input} images; plan.md: stop and report, do not fall back")
        return
    _finish(0, st, info=info)


@app.function(image=main_image, cpu=1, volumes=VOLUMES, timeout=600)
def step0_accept(note: str):
    """User decision: continue with the images COLMAP registered (convert.py drops the others)."""
    _container_setup()
    st = _load_state(0)
    if not st or st["status"] != "failed":
        raise RuntimeError(f"step 0 state is {st and st['status']}, nothing to accept")
    _export(st, [(D, "", ["sparse/*"], [])], t0=0)
    registered = sorted(os.listdir(f"{D}/images"))
    missing = sorted(set(os.listdir(f"{D}/input")) - set(registered))
    _finish(0, st, status="done", accepted=note, missing_images=missing, error=None)
    print(f"step 0 accepted: {len(registered)} images, missing {missing}")


def _latest_loadable_checkpoint(folder: str) -> str | None:
    import torch
    ckpts = sorted(Path(folder).glob("chkpnt*.pth"), key=lambda p: int(p.stem.replace("chkpnt", "")), reverse=True)
    for p in ckpts:
        try:
            torch.load(p, map_location="cpu")
            return str(p)
        except Exception as e:  # noqa: BLE001  (half-written by a preempted container)
            print(f"[resume] {p.name} unreadable ({e}); trying older")
    return None


@app.function(image=main_image, gpu=GPU, volumes=VOLUMES, timeout=8 * 3600, retries=RETRIES)
def step1_train_3dgs(force: bool = False):
    """run_seg.sh 1: vanilla 3DGS. Checkpoints only add saves (no change in training); resume via --start_checkpoint."""
    _container_setup()
    st = _begin(1, force)
    if st is None:
        return
    out_dir = f"{M}/3dgs_output"
    if st["attempts"] == 1 and os.path.isdir(out_dir):
        shutil.rmtree(out_dir)  # fresh run (forced): the original code writes to a fixed path
    ckpt_iters = " ".join(str(i) for i in range(CKPT_EVERY, 30000, CKPT_EVERY))
    resume = _latest_loadable_checkpoint(out_dir) if os.path.isdir(out_dir) else None
    cmd = (f"python gaussian_splatting/train.py -s {D_REL} -m {M_REL}/3dgs_output --init_mode sparse --eval "
           f"--resolution {RESOLUTION} --checkpoint_iterations {ckpt_iters}")
    if resume:
        cmd += f" --start_checkpoint {resume}"
    with PeriodicCommit(prune_dir=out_dir):
        _sh(cmd, st)
        _export(st, [(out_dir, "", None, ["chkpnt*.pth"])], t0=0)
    _finish(1, st, resumed_from=resume)


@app.function(image=main_image, gpu=GPU, volumes=VOLUMES, timeout=3 * 3600, retries=RETRIES)
def step2_mask2d(force: bool = False):
    """run_seg.sh 2: raw 2D masks. Existing LERF SAM masks are used only if they satisfy the HQ-SAM contract."""
    _container_setup()
    if not (Path(VOL) / "weights" / CROPFORMER_WEIGHT).exists():
        raise RuntimeError(f"{CROPFORMER_WEIGHT} missing on the volume: run ::upload_weight --path <file>")
    st = _begin(2, force)
    if st is None:
        return
    report = Path(EXPORT) / st["export"] / "check_report.json"
    with PeriodicCommit():
        usable = False
        if os.path.isdir(LERF_MASK):
            r = subprocess.run(["python", "tools_safe/check_sam_masks.py", "--mask_dir", LERF_MASK,
                                "--images_dir", f"{D}/images_{RESOLUTION}", "--report", str(report)],
                               cwd=REPO, text=True)
            usable = r.returncode == 0
        else:
            report.write_text(json.dumps({"usable": False, "decision": "no LERF masks uploaded"}, indent=2))
        raw = f"{D}/raw_hqsam"
        shutil.rmtree(raw, ignore_errors=True)
        shutil.rmtree(raw + "_color", ignore_errors=True)
        if usable:
            labels = f"{LERF_MASK}/labels" if os.path.isdir(f"{LERF_MASK}/labels") else LERF_MASK
            shutil.copytree(labels, raw)
            _sh(f"python -c \"from tools.vis_obj_color import vis_mask_images; "
                f"vis_mask_images('{raw}', '{raw}_color')\"", st, env=_seg_env())
            source = "LERF copy"
        else:
            _sh(f"python seg/raw_mask_sam.py --dataset_path data/{DATASET}/ --scene_name {SCENE} "
                f"--image_folder images_{RESOLUTION} --method hqsam", st, env=_seg_env())
            source = "rerun hqsam"
        rep = json.loads(report.read_text())
        rep["source"] = source
        report.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
        _export(st, [(D, "", ["raw_hqsam/*", "raw_hqsam_color/*"], [])], t0=0)
    _finish(2, st, mask_source=source)


@app.function(image=main_image, gpu=GPU, volumes=VOLUMES, timeout=3 * 3600, retries=RETRIES)
def step3_sync_mask(force: bool = False):
    """run_seg.sh 3-4: Key Object Management association + label numbers (to pick the napkin id)."""
    _container_setup()
    st = _begin(3, force)
    if st is None:
        return
    with PeriodicCommit():
        _sh(f"python seg/mask_associate.py --source_path {D_REL} --model_path {M_REL}/3dgs_output "
            f"--resolution {RESOLUTION} --mask_generator hqsam --eval", st, env=_seg_env())
        _sh(f"python tools/add_label_num_hqsam.py --source_path {D_REL} --resolution {RESOLUTION} "
            f"--mask_generator hqsam", st, env=_seg_env())
        _export(st, [(D, "", ["associated_hqsam/*", "associated_hqsam_color/*", f"images_{RESOLUTION}_num/*"], [])])
    _finish(3, st)


@app.function(image=main_image, gpu=GPU, volumes=VOLUMES, timeout=4 * 3600, retries=RETRIES)
def step4_distill_id(force: bool = False):
    """run_seg.sh 5-6: object ID distillation into the Gaussians + render (2000 iterations, short: step-level resume)."""
    _container_setup()
    st = _begin(4, force)
    if st is None:
        return
    with PeriodicCommit():
        _sh(f"python seg/distillation.py --source_path {D_REL} --model_path {M_REL} "
            f"--vanilla_3dgs_path {M_REL}/3dgs_output --resolution {RESOLUTION} --object_path associated_hqsam --eval",
            st, env=_seg_env())
        _sh(f"python render.py -m {M_REL} --render_video", st, env=_seg_env())
        _export(st, [(M, "", None, ["3dgs_output/*", "*chkpnt*.pth"])])
    _finish(4, st)


@app.function(image=main_image, gpu=GPU, volumes=VOLUMES, timeout=4 * 3600, retries=RETRIES)
def step5_removal_virtual_cam(target_id: str, surrounding_id: str = "None", force: bool = False,
                              circle_radius: float = -1.0):
    """run_remove.sh 1-3: config, object removal, virtual camera trajectory (renders zipped for SAM-Track)."""
    _container_setup()
    st = _begin(5, force)
    if st is None:
        return
    cfg_removal = f"config/object_removal/{DATASET}/{SCENE}.json"
    with PeriodicCommit():
        _sh(f"python tools/init_configs.py --dataset_name {DATASET} --scene {SCENE} --target_id {target_id} "
            f"--target_surronding_id {surrounding_id}", st, env=_repo_env())
        _sh(f"python edit_object_removal.py --source_path {D_REL} -m {M_REL} --config_file {cfg_removal} --render_video",
            st, env=_repo_env())
        # --circle_radius: official option of tools/virtual_pose.py (run_remove.sh: "# --circle_radius 0.5");
        # -1 = the code's automatic radius
        radius = f" --circle_radius {circle_radius}" if circle_radius != -1 else ""
        _sh(f"python tools/virtual_pose.py -s {D_REL} -m {M_REL} --config_file {cfg_removal}{radius}", st, env=_repo_env())
        _export(st, [(M, "", None, ["3dgs_output/*", "*chkpnt*.pth"]),
                     (f"{VOL}/config", "config", None, []),
                     (f"{VOL}/samt/assets", "samt_assets", ["images.zip"], [])])
    _finish(5, st, target_id=target_id, surrounding_id=surrounding_id, samt="pending", circle_radius=circle_radius)


@app.function(image=main_image, gpu=GPU, volumes=VOLUMES, timeout=4 * 3600)
def samt_app():
    """run_remove.sh 4: interactive Segment-and-Track-Anything on the virtual renders (paper Eq. 6, M_i = SAMT(C_i))."""
    _container_setup()
    st5 = _load_state(5)
    if not st5 or st5["status"] != "done":
        raise RuntimeError("step 5 is not done")
    ctl["samt_stop"] = False
    # app.py lists the upstream demo video assets/blackswan.mp4 as a gr.Examples entry (not used here) and fails
    # without it: fetch the upstream file, or write a 1 s placeholder (app.py itself is not changed)
    demo = Path(SAMT) / "assets" / "blackswan.mp4"
    if not demo.exists():
        subprocess.run(f"wget -q -O {demo} https://github.com/z-x-yang/Segment-and-Track-Anything/raw/main/assets/blackswan.mp4"
                       f" || ffmpeg -loglevel error -y -f lavfi -i color=c=black:s=64x64:d=1 {demo}", shell=True)
        if demo.exists() and demo.stat().st_size == 0:
            subprocess.run(f"ffmpeg -loglevel error -y -f lavfi -i color=c=black:s=64x64:d=1 {demo}", shell=True)
    env = {**os.environ, "GRADIO_SERVER_NAME": "0.0.0.0", "GRADIO_SERVER_PORT": "7860"}
    masks_dir = f"{SAMT}/tracking_results/images/images_masks"
    with PeriodicCommit(), modal.forward(7860) as tunnel:
        print("=" * 80)
        print(f"SAM-Track UI: {tunnel.url}")
        print("1) choose the example 'images.zip' (virtual renders after removal)  2) click the napkin hole on frame 0")
        print("3) Start Tracking  4) when done run:  modal run modal_inpaint360gs.py::samt_stop")
        print("=" * 80, flush=True)
        n_virtual = max((len(list(p.glob("*.png"))) for p in (Path(M) / "virtual").rglob("renders")), default=0)
        # app.py's img_seq_type_input_tracking deletes images_masks/ before every "Start Tracking"; a second click
        # after a finished run wipes the good masks and then fails. Keep a copy of the last complete set.
        backup = Path(VOL) / "samt" / "masks_complete"

        def n_frame_masks(d):
            return len([f for f in os.listdir(d) if f.endswith(".png") and not f.endswith("_new.png")]) \
                if os.path.isdir(d) else 0

        last_saved = None
        proc = subprocess.Popen(["python", "app.py"], cwd=SAMT, env=env)
        while proc.poll() is None:
            if ctl.get("samt_stop", False):
                proc.terminate()
                proc.wait(timeout=60)
                break
            if n_virtual and n_frame_masks(masks_dir) >= n_virtual:
                stamp = max(os.path.getmtime(os.path.join(masks_dir, f)) for f in os.listdir(masks_dir))
                if stamp != last_saved:
                    time.sleep(5)  # let the last frames finish writing
                    shutil.rmtree(backup, ignore_errors=True)
                    shutil.copytree(masks_dir, backup)
                    last_saved = stamp
                    print(f"[samt] complete mask set backed up ({n_virtual} frames)", flush=True)
                    _commit("samt masks backup")
            time.sleep(10)
    if n_frame_masks(masks_dir) < n_virtual and n_frame_masks(backup) >= n_virtual:
        print("[samt] images_masks incomplete (re-clicked Start Tracking?): restoring the last complete set")
        shutil.rmtree(masks_dir, ignore_errors=True)
        shutil.copytree(backup, masks_dir)
    n_masks = n_frame_masks(masks_dir)
    print(f"[samt] masks: {n_masks}, virtual renders: {n_virtual}")
    if n_masks == 0 or n_masks < n_virtual:
        st5["samt"] = "pending"
        _save_state(5, st5)
        _commit("samt no masks")
        raise RuntimeError(f"only {n_masks}/{n_virtual} SAM-Track masks saved; run samt again")
    dst = Path(EXPORT) / st5["export"] / "samt_masks"
    k = 2
    while dst.exists():
        dst = Path(EXPORT) / st5["export"] / f"samt_masks_v{k}"
        k += 1
    shutil.copytree(masks_dir, dst)
    st5.update(samt="done", samt_masks=n_masks, virtual_views=n_virtual)
    _save_state(5, st5)
    (Path(EXPORT) / st5["export"] / "state.json").write_text(json.dumps(st5, indent=2, ensure_ascii=False))
    _commit("samt done")


@app.function(image=main_image, gpu=GPU, volumes=VOLUMES, timeout=3600, retries=RETRIES)
def samt_auto(target_id: int):
    """run_remove.sh 4 without the browser: the same SAM-Track calls app.py makes (init_SegTracker -> seg_acc_click with
    one Positive point -> img_seq_type_input_tracking, UI default args), with the click placed automatically at the
    centroid of the removed object (`target_id`) in the first virtual view (objects_pred of the full-scene render)."""
    import cv2
    import numpy as np
    import torch
    _container_setup()
    st5 = _load_state(5)
    if not st5 or st5["status"] != "done":
        raise RuntimeError("step 5 is not done")
    os.chdir(SAMT)
    sys.path.insert(0, SAMT)
    from model_args import aot_args, sam_args, segtracker_args
    from SegTracker import SegTracker
    from seg_track_anything import aot_model2ckpt, img_seq_type_input_tracking

    with PeriodicCommit():
        # get_meta_from_img_seq: unzip assets/images.zip into ./assets/images, first sorted image = first frame
        img_dir = Path(SAMT) / "assets" / "images"
        shutil.rmtree(img_dir, ignore_errors=True)
        subprocess.run(["unzip", "-q", str(Path(SAMT) / "assets" / "images.zip"), "-d", str(img_dir)], check=True)
        imgs_path = sorted(str(p) for p in img_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        first = cv2.cvtColor(cv2.imread(imgs_path[0]), cv2.COLOR_BGR2RGB)

        # automatic click: centroid of the largest component of target_id in the first virtual view
        ids = cv2.imread(str(Path(M) / "virtual" / "ours_2000" / "objects_pred" / Path(imgs_path[0]).name), -1)
        obj = (ids == target_id).astype(np.uint8)
        n, lab, stats, cents = cv2.connectedComponentsWithStats(obj)
        if n < 2:
            raise RuntimeError(f"id {target_id} not visible in the first virtual view")
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y = int(round(cents[k][0])), int(round(cents[k][1]))
        if lab[y, x] != k:  # centroid outside a concave blob: nearest pixel of the blob
            ys, xs = np.nonzero(lab == k)
            j = int(np.argmin((xs - x) ** 2 + (ys - y) ** 2))
            x, y = int(xs[j]), int(ys[j])
        print(f"[samt-auto] click (x={x}, y={y}) on {Path(imgs_path[0]).name}, id {target_id} area {stats[k][4]} px")

        # init_SegTracker with the UI defaults (aot r50_deaotl, mem 9999/9999, sam_gap 1000, max_obj 255, pps 16)
        aot_args["model"] = "r50_deaotl"
        aot_args["model_path"] = aot_model2ckpt["r50_deaotl"]
        aot_args["long_term_mem_gap"] = 9999
        aot_args["max_len_long_term"] = 9999
        segtracker_args["sam_gap"] = 1000
        segtracker_args["max_obj_num"] = 255
        sam_args["generator_args"]["points_per_side"] = 16
        tracker = SegTracker(segtracker_args, sam_args, aot_args)
        tracker.restart_tracker()
        # sam_click -> seg_acc_click -> SegTracker_add_first_frame
        predicted_mask, masked_frame = tracker.seg_acc_click(origin_frame=first, coords=np.array([[x, y]]),
                                                             modes=np.array([1]), multimask="True")
        with torch.cuda.amp.autocast():
            tracker.restart_tracker()
            tracker.add_reference(first, predicted_mask, 0)
            tracker.first_frame_mask = predicted_mask

        # tracking_objects_in_video (img-seq branch), fps slider default 8
        video_name = "images"
        result_dir = f"{SAMT}/tracking_results/{video_name}"
        os.makedirs(result_dir, exist_ok=True)
        io_args = {
            "tracking_result_dir": result_dir,
            "output_mask_dir": f"{result_dir}/{video_name}_masks",
            "output_masked_frame_dir": f"{result_dir}/{video_name}_masked_frames",
            "output_video": f"{result_dir}/{video_name}_seg.mp4",
            "output_gif": f"{result_dir}/{video_name}_seg.gif",
        }
        img_seq_type_input_tracking(tracker, io_args, video_name, imgs_path, 8, 0)

        masks_dir = io_args["output_mask_dir"]
        raw_copy = Path(SAMT) / "tracking_results" / f"{video_name}_masks_raw"
        shutil.rmtree(raw_copy, ignore_errors=True)
        shutil.copytree(masks_dir, raw_copy)

        # what a user does in the UI after a bad track: "roll back" to the first frame where the hole is visible but
        # missed, click it again and "Start tracking to refine" (app.py roll_back_sam_click + tracking frame_num=f)
        def obj_mask(i):
            return cv2.imread(str(Path(M) / "virtual" / "ours_2000" / "objects_pred" / Path(imgs_path[i]).name), -1) == target_id

        def track_mask(i):
            return np.array(Image.open(os.path.join(masks_dir, Path(imgs_path[i]).stem + ".png")).convert("P"))

        from PIL import Image
        min_px = 200
        refined = []
        for _ in range(3):
            bad = None
            for i in range(1, len(imgs_path)):
                o = obj_mask(i)
                if o.sum() >= min_px and ((track_mask(i) > 0) & o).sum() < 0.3 * o.sum():
                    bad = i
                    break
            if bad is None:
                break
            o = obj_mask(bad).astype(np.uint8)
            n, lab, stats, cents = cv2.connectedComponentsWithStats(o)
            k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            ys, xs = np.nonzero(lab == k)
            j = int(np.argmin((xs - cents[k][0]) ** 2 + (ys - cents[k][1]) ** 2))
            cx, cy = int(xs[j]), int(ys[j])
            frame = cv2.cvtColor(cv2.imread(imgs_path[bad]), cv2.COLOR_BGR2RGB)
            curr = track_mask(bad)
            tracker.curr_idx = 1
            pm, _ = tracker.seg_acc_click(origin_frame=frame, coords=np.array([[cx, cy]]), modes=np.array([1]),
                                          multimask="True")
            curr[curr == 1] = 0
            curr[pm != 0] = 1
            with torch.cuda.amp.autocast():
                tracker.restart_tracker()
                tracker.add_reference(frame, curr, 0)
                tracker.first_frame_mask = curr
            print(f"[samt-auto] refine at frame {bad}: click (x={cx}, y={cy})")
            img_seq_type_input_tracking(tracker, io_args, video_name, imgs_path, 8, bad)
            refined.append([bad, cx, cy])

        # where the removed object is entirely out of view the hole cannot be visible: drop mask components that do
        # not touch the (dilated) object footprint; tracker drift elsewhere (e.g. onto the sheep) would be inpainted
        from seg_track_anything import save_prediction
        kernel = np.ones((31, 31), np.uint8)
        gated = []
        for i, p in enumerate(imgs_path):
            m = track_mask(i)
            near = cv2.dilate(obj_mask(i).astype(np.uint8), kernel) > 0
            n, lab = cv2.connectedComponents((m > 0).astype(np.uint8))
            keep = np.zeros_like(m, dtype=bool)
            for c in range(1, n):
                if (near & (lab == c)).any():
                    keep |= lab == c
            if (m > 0).sum() != keep.sum():
                gated.append([i, int((m > 0).sum() - keep.sum())])
                m = np.where(keep, m, 0).astype(m.dtype)
                save_prediction(m, masks_dir, Path(p).stem + ".png")
        print(f"[samt-auto] refined at {refined}; removed off-object pixels in frames {gated}")

        n_masks = len([f for f in os.listdir(masks_dir) if f.endswith(".png") and not f.endswith("_new.png")])
        areas = [float((cv2.imread(os.path.join(masks_dir, Path(p).stem + ".png"), 0) > 0).mean()) for p in imgs_path]
        print(f"[samt-auto] masks {n_masks}/{len(imgs_path)}; mask area % min {100 * min(areas):.2f} "
              f"mean {100 * np.mean(areas):.2f} max {100 * max(areas):.2f}")
        if n_masks < len(imgs_path):
            raise RuntimeError(f"only {n_masks}/{len(imgs_path)} masks")
        dst = Path(EXPORT) / st5["export"] / "samt_masks"
        k = 2
        while dst.exists():
            dst = Path(EXPORT) / st5["export"] / f"samt_masks_v{k}"
            k += 1
        from seg_track_anything import draw_mask
        overlay = dst.parent / (dst.name + "_overlay")
        overlay.mkdir(parents=True)
        for i, p in enumerate(imgs_path):  # overlay of the final masks (red: removed object footprint)
            frame = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            vis = cv2.cvtColor(draw_mask(frame, track_mask(i)), cv2.COLOR_RGB2BGR)
            cs, _ = cv2.findContours(obj_mask(i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cs, -1, (0, 0, 255), 1)
            cv2.imwrite(str(overlay / Path(p).name), vis)
        shutil.copytree(masks_dir, dst)
        shutil.copytree(raw_copy, dst.parent / (dst.name + "_raw_track"))
        st5.update(samt="done", samt_mode="auto: SAM-Track click at target-id centroid + auto refine + off-object gating",
                   samt_click=[x, y], samt_refine=refined, samt_gated_frames=gated, samt_masks=n_masks,
                   virtual_views=len(imgs_path), samt_export=dst.name,
                   samt_area_pct=[round(100 * min(areas), 2), round(100 * max(areas), 2)])
        _save_state(5, st5)
        (Path(EXPORT) / st5["export"] / "state.json").write_text(json.dumps(st5, indent=2, ensure_ascii=False))


@app.local_entrypoint()
def samt_auto_run(target_id: int = 44):
    _assert_workspace()
    _require(5)
    samt_auto.remote(target_id)
    print(json.dumps(_remote_state(5), indent=2, ensure_ascii=False))


@app.function(image=main_image, cpu=4, memory=16384, volumes=VOLUMES, timeout=900)
def samt_selftest():
    """CPU-only check that the SAM-Track Gradio page serves (no GPU spent on dependency fixes). Read-only on the volume."""
    import urllib.request
    _container_setup()  # never committed here, so nothing written below persists
    env ={**os.environ, "GRADIO_SERVER_NAME": "0.0.0.0", "GRADIO_SERVER_PORT": "7860"}
    demo = Path(SAMT) / "assets" / "blackswan.mp4"
    demo.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(f"ffmpeg -loglevel error -y -f lavfi -i color=c=black:s=64x64:d=1 {demo}", shell=True)
    log = open("/tmp/app.log", "w")
    proc = subprocess.Popen(["python", "app.py"], cwd=SAMT, env=env, stdout=log, stderr=subprocess.STDOUT)
    result = "no response"
    for _ in range(90):
        time.sleep(2)
        try:
            with urllib.request.urlopen("http://127.0.0.1:7860/", timeout=10) as r:
                result = f"HTTP {r.status}, {len(r.read())} bytes"
            break
        except Exception as e:  # noqa: BLE001
            result = f"error: {e}"
            if proc.poll() is not None:
                break
    time.sleep(2)
    proc.terminate()
    log.close()
    tail = Path("/tmp/app.log").read_text(errors="replace").splitlines()[-25:]
    print("\n".join(tail))
    print(f"[selftest] GET / -> {result}")
    return result


@app.local_entrypoint()
def samt_check():
    print(samt_selftest.remote())


@app.function(image=main_image, cpu=2, volumes=VOLUMES, timeout=3600, retries=RETRIES)
def step6_lama_prepare(force: bool = False):
    """run_inpaint.sh 1a: prepare_lama_data.py --inpaint2lama (mask enlarge + copy into LaMa/data)."""
    _container_setup()
    st5 = _load_state(5)
    if not st5 or st5.get("samt") != "done":
        raise RuntimeError("SAM-Track masks missing: run samt first")
    st = _begin(6, force)
    if st is None:
        return False
    if "prepare" not in st.get("parts", []):
        with PeriodicCommit():
            _sh(f"python tools/prepare_lama_data.py -s {D_REL} -m {M_REL} -r {RESOLUTION} --inpaint2lama", st,
                env=_repo_env())
        st.setdefault("parts", []).append("prepare")
        _save_state(6, st)
        _commit("lama prepare")
    return True


@app.function(image=lama_image, gpu=GPU, volumes=VOLUMES, timeout=4 * 3600, retries=RETRIES)
def step6_lama_predict():
    """run_inpaint.sh 1b: LaMa color + depth (env lama)."""
    vol.reload()
    _link_volume({"LaMa/data": "lama/data", "LaMa/output": "lama/output"})
    big = Path(f"{REPO}/LaMa/big-lama")
    if not big.exists():
        big.symlink_to("/opt/weights/big-lama", target_is_directory=True)
    st = _load_state(6)
    lama = f"{REPO}/LaMa"
    env = {"TORCH_HOME": lama, "PYTHONPATH": lama}
    with PeriodicCommit():
        for kind in ("color", "depth"):
            if kind in st.get("parts", []):
                continue
            _sh(f"python bin/predict_{kind}.py --data_name {LAMA_DATA_NAME}", st, cwd=lama, env=env)
            st.setdefault("parts", []).append(kind)
            _save_state(6, st)
            _commit(f"lama {kind}")


@app.function(image=main_image, cpu=2, volumes=VOLUMES, timeout=3600, retries=RETRIES)
def step6_lama_collect():
    """run_inpaint.sh 1c: prepare_lama_data.py (copy LaMa results back)."""
    _container_setup()
    st = _load_state(6)
    with PeriodicCommit():
        _sh(f"python tools/prepare_lama_data.py -s {D_REL} -m {M_REL} -r {RESOLUTION}", st, env=_repo_env())
        _export(st, [(D, "", ["images_inpaint_unseen_virtual/*", "inpaint_2d_unseen_mask_virtual/*"], []),
                     (f"{VOL}/lama", "lama", None, []),
                     (M, "", ["virtual/*depth_completed/*"], [])])
    _finish(6, st)


# User decision (2026-10-26): keep the code's virtual trajectory (cameras look at the training-camera focus point, auto
# circle_radius); virtual views where the removed object is out of view / occluded have an empty inpainting mask and
# crash finetune_inpaint (mask_to_bbox on an empty mask). This wrapper runs edit_object_inpaint.py unchanged except
# that finetune_inpaint only receives the views whose mask (objects > 128) is non-empty.
SKIP_EMPTY_WRAPPER = r'''
import ast, json, os, sys
sys.path.insert(0, os.getcwd())
import edit_object_inpaint as eoi

_orig = eoi.finetune_inpaint

def finetune_inpaint(args, opt, model_path, iteration, views, *rest, **kw):
    keep = [v for v in views if bool((v.objects > 128).any())]
    skipped = [v.image_name for v in views if not bool((v.objects > 128).any())]
    print(f"[skip-empty] training on {len(keep)}/{len(views)} virtual views; skipped (empty mask): {skipped}", flush=True)
    with open(os.path.join(model_path, "skipped_virtual_views.json"), "w") as f:
        json.dump({"used": [v.image_name for v in keep], "skipped": skipped}, f, indent=2)
    return _orig(args, opt, model_path, iteration, keep, *rest, **kw)

eoi.finetune_inpaint = finetune_inpaint
src = open(eoi.__file__, encoding="utf-8").read()
main = [n for n in ast.parse(src).body if isinstance(n, ast.If) and "__main__" in ast.unparse(n.test)][0]
eoi.__dict__["__name__"] = "__main__"
exec(compile(ast.Module(body=main.body, type_ignores=[]), eoi.__file__, "exec"), eoi.__dict__)
'''


@app.function(image=main_image, gpu=GPU, volumes=VOLUMES, timeout=6 * 3600, retries=RETRIES)
def step7_inpaint3d(force: bool = False, skip_empty_views: bool = True):
    """run_inpaint.sh 2-3: colorful point cloud fusion + missing-area optimization (renders -> step 8)."""
    _container_setup()
    st = _begin(7, force)
    if st is None:
        return
    with PeriodicCommit():
        _sh(f"python edit_object_removal_plyfusion.py -s {D_REL} -m {M_REL} "
            f"--config_file config/object_removal/{DATASET}/{SCENE}.json", st, env=_repo_env())
        script = "edit_object_inpaint.py"
        if skip_empty_views:
            wrapper = Path(EXPORT) / st["export"] / "edit_object_inpaint_skip_empty.py"
            wrapper.write_text(SKIP_EMPTY_WRAPPER)
            script = str(wrapper)
        _sh(f"python {script} -s {D_REL} -m {M_REL} --config_file config/object_inpaint/{DATASET}/{SCENE}.json "
            f"--resolution {RESOLUTION} --render_video", st, env=_repo_env())
        _export(st, [(M, "", None, ["3dgs_output/*", "train/*", "test/*", "video/*", "*chkpnt*.pth"])])
    _finish(7, st, skip_empty_views=skip_empty_views)


@app.function(image=main_image, cpu=1, volumes=VOLUMES, timeout=12 * 3600)
def chain_5_to_8(target_id: int = 44, circle_radius: float = -1.0):
    """Run step 5 (force, new version) -> automatic SAM-Track masks -> step 6 (force) -> step 7 -> step 8 entirely on
    Modal, so the local machine can be switched off (launch with `modal run --detach ...::chain`)."""
    print("[chain] step 5")
    step5_removal_virtual_cam.remote(str(target_id), "None", True, circle_radius)
    print("[chain] SAM-Track auto masks")
    samt_auto.remote(target_id)
    print("[chain] step 6")
    if step6_lama_prepare.remote(True):
        step6_lama_predict.remote()
        step6_lama_collect.remote()
    print("[chain] step 7")
    step7_inpaint3d.remote(False, True)
    print("[chain] step 8")
    step8_render.remote(False)
    vol.reload()
    for s in (5, 6, 7, 8):
        st = _load_state(s)
        print(f"[chain] step {s}: {st and st['status']} {st and st['export']}")


@app.local_entrypoint()
def chain(target_id: int = 44, circle_radius: float = -1.0):
    _assert_workspace()
    chain_5_to_8.remote(target_id, circle_radius)


@app.function(image=main_image, cpu=2, volumes=VOLUMES, timeout=3600, retries=RETRIES)
def step8_render(force: bool = False):
    """Renders of the inpainted scene written by edit_object_inpaint.py --render_video (no metrics: no GT for teatime)."""
    _container_setup()
    st7 = _load_state(7)
    if not st7 or st7["status"] != "done":
        raise RuntimeError("step 7 is not done")
    st = _begin(8, force)
    if st is None:
        return
    _export(st, [(M, "", ["train/*", "test/*", "video/*"], [])], t0=st7["t0"])
    _finish(8, st, from_step7=st7["export"])

# --------------------------------------------------------------------------------------------------
# 5. Local entrypoints
# --------------------------------------------------------------------------------------------------

def _assert_workspace() -> None:
    r = subprocess.run([sys.executable, "-m", "modal", "profile", "current"], capture_output=True, text=True)
    current = r.stdout.strip()
    if current != WORKSPACE_PROFILE:
        raise SystemExit(f"Modal profile is '{current}', expected '{WORKSPACE_PROFILE}': "
                         f"python -m modal profile activate {WORKSPACE_PROFILE}")


def _remote_state(step: int) -> dict | None:
    try:
        return json.loads(b"".join(vol.read_file(f"state/step{step}.json")))
    except Exception:  # noqa: BLE001
        return None


def _require(step: int) -> None:
    st = _remote_state(step)
    if not st or st["status"] != "done":
        raise SystemExit(f"step {step} is not done (state: {st['status'] if st else 'none'})")


@app.local_entrypoint()
def upload():
    """LERF teatime images -> data/others/teatime/input, existing SAM masks -> lerf_mask."""
    _assert_workspace()
    with vol.batch_upload(force=True) as b:
        for f in sorted((LERF_SCENE / "images").glob("*.jpg")):
            b.put_file(str(f), f"/{D_REL}/input/{f.name}")
        mask = LERF_SCENE / "mask"
        b.put_directory(str(mask / "labels"), "/lerf_mask/labels")
        for name in ("sam2_config.json", "objects.json"):
            if (mask / name).exists():
                b.put_file(str(mask / name), f"/lerf_mask/{name}")
    print("uploaded images + LERF masks")


@app.local_entrypoint()
def accept_step0(note: str = "user: continue with COLMAP-registered images"):
    _assert_workspace()
    step0_accept.remote(note)


@app.local_entrypoint()
def upload_weight(path: str):
    """Gated HQ-SAM/CropFormer weight (HuggingFace qqlu1992/Adobe_EntitySeg) -> /vol/weights."""
    _assert_workspace()
    src = Path(path)
    if src.name != CROPFORMER_WEIGHT or src.stat().st_size < 800_000_000:
        raise SystemExit(f"expected {CROPFORMER_WEIGHT} (~889 MB), got {src} ({src.stat().st_size} bytes)")
    with vol.batch_upload(force=True) as b:
        b.put_file(str(src), f"/weights/{CROPFORMER_WEIGHT}")
    print("uploaded", CROPFORMER_WEIGHT)


@app.local_entrypoint()
def run_step(step: int, force: bool = False, target_id: str = "", surrounding_id: str = "None",
             circle_radius: float = -1.0):
    _assert_workspace()
    if step > 0:
        _require(step - 1)
    if step == 0:
        step0_prepare_data.remote(force)
    elif step == 1:
        step1_train_3dgs.remote(force)
    elif step == 2:
        step2_mask2d.remote(force)
    elif step == 3:
        step3_sync_mask.remote(force)
        print("Next: download --step 3, look at images_2_num/ and pick the napkin id (+ occluders).")
    elif step == 4:
        step4_distill_id.remote(force)
    elif step == 5:
        if not target_id:
            raise SystemExit("--target-id is required (napkin id from step3 images_2_num)")
        step5_removal_virtual_cam.remote(target_id, surrounding_id, force, circle_radius)
        print("Next: modal run modal_inpaint360gs.py::samt")
    elif step == 6:
        if step6_lama_prepare.remote(force):
            step6_lama_predict.remote()
            step6_lama_collect.remote()
    elif step == 7:
        step7_inpaint3d.remote(force)
    elif step == 8:
        step8_render.remote(force)
    print(json.dumps(_remote_state(step), indent=2, ensure_ascii=False))


@app.local_entrypoint()
def samt():
    _assert_workspace()
    _require(5)
    samt_app.remote()


@app.local_entrypoint()
def samt_stop():
    ctl["samt_stop"] = True
    print("stop requested; the SAM-Track container saves the masks and exits")


@app.local_entrypoint()
def status():
    for s in STEP_NAMES:
        st = _remote_state(s)
        if st:
            print(f"step {s}: {st['status']:8s} {st['export']:24s} attempts={st['attempts']} "
                  f"{ {k: v for k, v in st.items() if k in ('mask_source', 'target_id', 'samt', 'error')} }")
        else:
            print(f"step {s}: -")


@app.local_entrypoint()
def download(step: int = -1):
    """export/<run>/stepN_* (+ logs) -> outputs/<run>/ via parallel `modal volume get`; local files are never overwritten."""
    _assert_workspace()
    prefix = f"export/{RUN_NAME}"
    tops = [e.path.rstrip("/").split("/")[-1] for e in vol.listdir(prefix)]
    wanted = [t for t in tops if t.startswith("step") and (step < 0 or t.startswith(f"step{step}_"))]
    n_new = 0
    for top in wanted + ["logs"]:
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable, "-m", "modal", "volume", "get", "inpaint360gs", f"{prefix}/{top}", tmp],
                           check=True)
            src_root = next(Path(tmp).rglob(top)) if not (Path(tmp) / top).exists() else Path(tmp) / top
            for src in src_root.rglob("*"):
                if not src.is_file():
                    continue
                if top == "logs" and step >= 0 and not src.name.startswith(f"step{step}_"):
                    continue
                dst = LOCAL_OUT / top / src.relative_to(src_root)
                if dst.exists() and dst.stat().st_size == src.stat().st_size:
                    continue
                if dst.exists():  # a partial file from an interrupted download: complete it
                    print(f"[download] replacing incomplete {dst}")
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                n_new += 1
    _write_readme()
    print(f"downloaded {n_new} new files -> {LOCAL_OUT}")


def _write_readme() -> None:
    lines = [f"# {RUN_NAME}: Inpaint360GS on LERF {SCENE}, removing {OBJECT}", "",
             f"Modal workspace `{WORKSPACE_PROFILE}`, GPU {GPU}, dataset branch `{DATASET}`, resolution {RESOLUTION}. "
             "Pipeline: original run_seg.sh / run_remove.sh / run_inpaint.sh (see ../../plan.md).", "",
             "| step | folder | status | attempts | notes |", "|---|---|---|---|---|"]
    for s in STEP_NAMES:
        st = _remote_state(s)
        if not st:
            continue
        notes = {k: st[k] for k in ("mask_source", "target_id", "surrounding_id", "samt", "resumed_from", "accepted",
                                    "missing_images", "error", "info") if k in st}
        lines.append(f"| {s} | {st['export']} | {st['status']} | {st['attempts']} | `{json.dumps(notes, ensure_ascii=False)}` |")
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    (LOCAL_OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
