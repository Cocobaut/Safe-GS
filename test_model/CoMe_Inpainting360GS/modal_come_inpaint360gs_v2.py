"""
v2 (option C) of Inpaint360GS + CoMe on LERF teatime (napkin): re-run only steps 10-12 with src_v2/
=================================================================================================
src_v2/ = src/ + (1) LaMa mask dilated 20 px instead of 10 (tools/prepare_lama_data.py),
               (2) the new Gaussians are initialized from the virtual view with the cleanest LaMa fill
                   (tools_come/select_init_view.py -> INPAINT360_INIT_VIEW) instead of the fixed view 00004,
               (3) render.py renders the video on a copy of views[0] (v1 overwrote the pose of frame_00002).
Steps 0-9 are reused from v1 (state copied, read from the v1 folders of the volume). Everything v2 writes goes to
/vol/come360/{data,output}/teatime_v2, lama/*/360_teatime_v2_virtual, state_v2/, export/teatime_napkin_v2/;
v1 data on the volume is only read. Results are downloaded to output_v2/teatime_napkin/.

  python -m modal run --detach modal_come_inpaint360gs_v2.py::run          # setup + step 10, 11, 12
  python -m modal run modal_come_inpaint360gs_v2.py::status
  python -m modal run modal_come_inpaint360gs_v2.py::download              # export -> output_v2 (never overwrites)
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
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
APP_NAME = "come-inpaint360gs-v2"
WORKSPACE_PROFILE = "nguyenkhacd10"
GPU = "L4"

SCENE = "teatime"
RUN_NAME = "teatime_napkin"
NAPKIN_REF_ID = 11            # CoMe_Infusion/Step_1_Sync_ID label of the napkin (reference only)
COMMIT_EVERY_S = 300

HERE = Path(__file__).resolve().parent
SRC = HERE / "src_v2"
LOCAL_OUT = HERE / "output_v2" / RUN_NAME
# main_image (original Inpaint360GS code); fixed path: inside the container this file sits in /root
LOCAL_ROOT = Path(r"F:\Safe-GS\Based_Model\Inpaint360GS")

VOL = "/vol"
ROOT = f"{VOL}/come360"
D1 = f"{ROOT}/data/{SCENE}"                     # v1 data (read only)
D2VOL = f"{ROOT}/data/{SCENE}_v2"               # what v2 writes into the dataset folder
D = f"/tmp/v2tree/data/{SCENE}_v2"              # -s: per-container tree of links (_v2_tree)
M = f"{ROOT}/output/{SCENE}/3dgs_output"        # -m (CoMe 3DGS, iteration 30000)
REF = f"{ROOT}/ref"                             # come_renders/, napkin_mask_id11/
STATE = f"{ROOT}/state_v2"
STATE1 = f"{ROOT}/state"
EXPORT = f"{ROOT}/export/{RUN_NAME}_v2"
ROOT_REL = "come360"

REPO = "/opt/inpaint360gs"                      # original code in main_image
CODE = "/opt/come360"                           # src/ in come_image
SAMT = f"{REPO}/Segment-and-Track-Anything"
CROPFORMER_WEIGHT = "CropFormer_hornet_3x_03823a.pth"

STEPS = {
    4: "step4_renderer_check",
    5: "step5_mask2d_hqsam",
    6: "step6_sync_mask",
    7: "step7_id_distill",
    8: "step8_removal_virtual_cam",
    9: "step9_inpaint_masks_samt",
    10: "step10_inpaint2d_lama",
    11: "step11_inpaint_3dgs_come_loss",
    12: "step12_render",
}

# step 7-11 (original Inpaint360GS layout: -m holds the distilled model, 3dgs_output the CoMe 3DGS)
DATASET = "others"                              # config/object_{removal,inpaint}/others/teatime.json
MI1 = f"{ROOT}/output/{SCENE}"                  # v1 model folder (read only)
MI2VOL = f"{ROOT}/output/{SCENE}_v2"            # what v2 writes into the model folder
MI = f"/tmp/v2tree/output/{SCENE}_v2"           # -m: per-container tree of links (_v2_tree)
ITER_ID = 2000                                  # distillation iterations (config/object_distill/train_distill.json)
INPAINT_ITER = 5000                             # config/object_inpaint/common.json finetune_iteration
CFG_REMOVAL = f"config/object_removal/{DATASET}/{SCENE}.json"
CFG_INPAINT = f"config/object_inpaint/{DATASET}/{SCENE}.json"
WORK = "/tmp/come360_work_v2"                   # writable copy of src_v2/ (come_image)
SAMT_VOL = f"{ROOT}/samt"                       # Segment-and-Track-Anything assets/ + tracking_results/
LAMA_VOL = f"{ROOT}/lama"                       # LaMa data/ + output/
LAMA_DATA_NAME = f"360_{SCENE}_v2_virtual"   # 360_<name of -s>_virtual

app = modal.App(APP_NAME)
vol = modal.Volume.from_name("inpaint360gs", create_if_missing=True)
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
_GLM_IGNORE = ["third_party/glm/doc", "third_party/glm/doc/**", "third_party/glm/test", "third_party/glm/test/**",
               "build", "build/**", "*.egg-info", "*.egg-info/**", "**/__pycache__", "**/__pycache__/**"]

come_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "build-essential", "ninja-build", "ffmpeg", "libgl1", "libglib2.0-0", "libsm6", "libxext6",
                 "libgomp1")
    .pip_install("torch==2.4.1", "torchvision==0.19.1", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("numpy==1.26.4", "plyfile", "tqdm", "scipy", "opencv-python-headless", "scikit-learn", "lpips",
                 "torchmetrics", "imageio", "imageio-ffmpeg", "open3d", "wandb", "pytorch_fid", "pillow",
                 "matplotlib", "wheel", "setuptools<70")
    .env({**BASE_ENV, "CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "8.9", "MAX_JOBS": "8",
          "CC": "gcc", "CXX": "g++"})
    .add_local_dir(str(SRC / "submodules" / "simple-knn-come"), "/opt/come_sub/simple-knn-come", copy=True,
                   ignore=_GLM_IGNORE)
    .add_local_dir(str(SRC / "submodules" / "diff-gaussian-rasterization-come"),
                   "/opt/come_sub/diff-gaussian-rasterization-come", copy=True, ignore=_GLM_IGNORE)
    .run_commands(
        "pip install --no-build-isolation /opt/come_sub/simple-knn-come",
        "pip install --no-build-isolation /opt/come_sub/diff-gaussian-rasterization-come",
    )
    .pip_install("dacite")  # imported by the CoMe rasterizer package (ExtendedSettings)
    .apt_install("libegl1", "libgl1-mesa-dri")  # open3d (imported by the Inpaint360GS GaussianModel)
    # code last and not copied: editing src/ does not rebuild the CUDA extensions
    .add_local_dir(str(SRC), CODE, copy=False, ignore=[
        "LaMa", "LaMa/**", "Segment-and-Track-Anything", "Segment-and-Track-Anything/**", "seg/detectron2",
        "seg/detectron2/**", "submodules", "submodules/**", "gaussian_splatting", "gaussian_splatting/**",
        "docs", "docs/**", "media", "media/**", "plan.md", "modal_inpaint360gs.py", "**/__pycache__",
        "**/__pycache__/**", "**/*.pyc"])
)

# ---- main_image: verbatim copy of Based_Model/Inpaint360GS/modal_inpaint360gs.py (lines 122-185) ----
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


# ---- lama_image: verbatim copy of Based_Model/Inpaint360GS/modal_inpaint360gs.py (lines 186-218) ----
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


light_image = modal.Image.debian_slim(python_version="3.10")

VOLUMES = {VOL: vol}

# --------------------------------------------------------------------------------------------------
# 3. Container helpers
# --------------------------------------------------------------------------------------------------

def _commit(reason: str = "") -> None:
    try:
        vol.commit()
        print(f"[volume] committed {reason}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[volume] commit failed ({reason}): {e}", flush=True)


class PeriodicCommit:
    """Commit the volume every COMMIT_EVERY_S seconds and on SIGTERM, so a preempted container loses little."""

    def __init__(self):
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self.stop.wait(COMMIT_EVERY_S):
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
        _commit("end of step")
        return False


def _load_state(step) -> dict | None:
    p = Path(STATE) / f"step{step}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _save_state(step, st: dict) -> None:
    Path(STATE).mkdir(parents=True, exist_ok=True)
    (Path(STATE) / f"step{step}.json").write_text(json.dumps(st, indent=2, ensure_ascii=False))


def _next_export_name(step: int) -> str:
    base = STEPS[step]
    if not (Path(EXPORT) / base).exists():
        return base
    k = 2
    while (Path(EXPORT) / f"{base}_v{k}").exists():
        k += 1
    return f"{base}_v{k}"


def _begin(step: int, force: bool) -> dict | None:
    """Start or resume a step. Returns None if the step is already done (and not forced)."""
    vol.reload()
    st = _load_state(step)
    if st and st["status"] == "done" and not force:
        print(f"[step {step}] already done -> {st['export']} (use --force for a new version)")
        return None
    if st and st["status"] == "failed" and not any(p.is_file() for p in (Path(EXPORT) / st["export"]).rglob("*")):
        st.update(status="running", attempts=st["attempts"] + 1)   # failed before writing anything: same folder
        print(f"[step {step}] retrying {st['export']} (previous attempt produced nothing)")
    elif st and st["status"] == "running":
        st["attempts"] += 1
        print(f"[step {step}] resuming {st['export']} (attempt {st['attempts']})")
    else:
        st = {"step": step, "status": "running", "export": _next_export_name(step), "attempts": 1,
              "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        print(f"[step {step}] new run -> {st['export']}")
    (Path(EXPORT) / st["export"]).mkdir(parents=True, exist_ok=True)
    _save_state(step, st)
    _commit(f"step {step} begin")
    return st


def _finish(step: int, st: dict, status: str = "done", **extra) -> dict:
    st.update(extra, status=status, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    _save_state(step, st)
    _commit(f"step {step} {status}")
    return st


def _sh(cmd: str, st: dict, cwd: str, env: dict | None = None) -> None:
    """Run one command (bash), stream it and append it to logs/<export>.log."""
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


def _copy_dir(src: str, dst: Path, exclude: tuple = ()) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(Path(src).iterdir()):
        if p.is_file() and p.name not in exclude:
            shutil.copy2(p, dst / p.name)
            n += 1
    return n


# run_seg.sh: export PYTHONPATH=$(pwd):$(pwd)/seg/detectron2; CropFormer modules also import the top-level
# package `mask2former` (criterion_view.py), so its project folder is on the path too (no code change)
SEG_PYTHONPATH = f"{REPO}:{REPO}/seg/detectron2:{CROPFORMER}"


def _seg_env() -> dict:
    return {"PYTHONPATH": SEG_PYTHONPATH + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")}


def _come_env(code: str = CODE, **extra) -> dict:
    return {"PYTHONPATH": code, "COME_SPLATTING_CONFIG": f"{M}/config.json", **extra}


def _link(link: Path, target: str) -> None:
    Path(target).mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        return
    if link.exists():
        shutil.rmtree(link)
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)


def _v2_tree() -> None:
    """-s / -m of v2 as local folders of links: inputs point to the v1 folders (read only), every folder the
    scripts write into points to the v2 folders of the volume."""
    links = {
        f"{D}/images": f"{D1}/images", f"{D}/sparse": f"{D1}/sparse", f"{D}/associated_hqsam": f"{D1}/associated_hqsam",
        f"{D}/inpaint_2d_unseen_mask_virtual": f"{D2VOL}/inpaint_2d_unseen_mask_virtual",
        f"{D}/images_inpaint_unseen_virtual": f"{D2VOL}/images_inpaint_unseen_virtual",
        f"{MI}/point_cloud": f"{MI1}/point_cloud",
        f"{MI}/point_cloud_object_removal": f"{MI1}/point_cloud_object_removal",
        f"{MI}/virtual": f"{MI2VOL}/virtual",
        f"{MI}/point_cloud_object_inpaint_virtual": f"{MI2VOL}/point_cloud_object_inpaint_virtual",
        f"{MI}/video/ours_ours_object_removal_{ITER_ID}": f"{MI1}/video/ours_ours_object_removal_{ITER_ID}",
        f"{MI}/video/ours__object_inpaint_virtual": f"{MI2VOL}/video/ours__object_inpaint_virtual",
    }
    for split in ("train", "test"):
        links[f"{MI}/{split}/ours_object_removal"] = f"{MI1}/{split}/ours_object_removal"
        links[f"{MI}/{split}/ours_object_inpaint_virtual"] = f"{MI2VOL}/{split}/ours_object_inpaint_virtual"
    for link, target in links.items():
        if not Path(link).is_symlink():
            Path(target).mkdir(parents=True, exist_ok=True)
            Path(link).parent.mkdir(parents=True, exist_ok=True)
            Path(link).symlink_to(target, target_is_directory=True)
    for f in ("cfg_args", "object_center.json"):
        if not (Path(MI) / f).exists():
            shutil.copy2(Path(MI1) / f, Path(MI) / f)


def _clear_dir(path: str) -> None:
    """Empty the folder a link points to (the link stays)."""
    real = Path(os.path.realpath(path))
    shutil.rmtree(real, ignore_errors=True)
    real.mkdir(parents=True, exist_ok=True)


def _come_workdir() -> str:
    """Writable copy of src/; the folders the original scripts write next to the code (config/<dataset>,
    Segment-and-Track-Anything, LaMa) are links to the volume, so they persist and are shared between images."""
    vol.reload()
    _v2_tree()
    if not Path(WORK).exists():
        shutil.copytree(CODE, WORK, symlinks=True)
    for rel, target in {f"config/object_removal/{DATASET}": f"{ROOT}/config/object_removal/{DATASET}",
                        f"config/object_inpaint/{DATASET}": f"{ROOT}/config/object_inpaint/{DATASET}",
                        "Segment-and-Track-Anything": SAMT_VOL, "LaMa": LAMA_VOL}.items():
        _link(Path(WORK) / rel, target)
    return WORK


def _require_done(step: int) -> dict:
    vol.reload()
    st = _load_state(step)
    if not st or st["status"] != "done":
        raise RuntimeError(f"step {step} is not done")
    return st


def _object_center() -> str:
    return _require_done(8)["object_center_world"]


def _link_weight() -> None:
    weight, link = Path(VOL) / "weights" / CROPFORMER_WEIGHT, Path(REPO) / "seg" / "weight" / CROPFORMER_WEIGHT
    if not weight.exists():
        raise RuntimeError(f"{weight} missing on the volume")
    if not link.exists():
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(weight)


def _draw_ids(image_bgr, ann):
    """Same drawing as tools/add_label_num_hqsam.py (label number at each contour centroid)."""
    import cv2
    import numpy as np
    for label in np.unique(ann):
        if label == 0:
            continue
        contours, _ = cv2.findContours((ann == label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            m = cv2.moments(c)
            if m["m00"] != 0:
                cv2.putText(image_bgr, str(label), (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return image_bgr

# --------------------------------------------------------------------------------------------------
# 4. Steps
# --------------------------------------------------------------------------------------------------

@app.function(image=come_image, gpu=GPU, memory=32768, volumes=VOLUMES, timeout=2 * 3600, retries=RETRIES)
def step4_renderer_check(force: bool = False) -> dict:
    """Render the CoMe 3DGS with the new renderer on all 177 views; PSNR vs the CoMe renders >= 35 dB."""
    st = _begin(4, force)
    if st is None:
        return _load_state(4)
    out = Path(EXPORT) / st["export"]
    with PeriodicCommit():
        try:
            _sh(f"python tools_come/render_check.py -s {D} -m {M} --resolution 1 --eval --iteration 30000 "
                f"--come_renders {REF}/come_renders --out_dir {out}", st, cwd=CODE, env=_come_env())
        except RuntimeError as e:
            print(f"[step 4] {e}")
    s = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    brief = {k: s.get(k) for k in ["pass", "psnr_vs_come_render", "max_abs_diff_vs_come_render",
                                   "filter_3D_recomputed_max_abs_diff_vs_ply", "n_filter_cameras", "seconds_per_view"]}
    return _finish(4, st, "done" if s.get("pass") else "failed", check=brief)


@app.function(image=main_image, gpu=GPU, memory=32768, volumes=VOLUMES, timeout=3 * 3600, retries=RETRIES)
def step5_mask2d_hqsam(force: bool = False) -> dict:
    """run_seg.sh 2: HQ-SAM (CropFormer) masks on the CoMe images (994x738), original code and defaults."""
    import numpy as np
    from PIL import Image

    _link_weight()
    st = _begin(5, force)
    if st is None:
        return _load_state(5)
    out = Path(EXPORT) / st["export"]
    raw = f"{D}/raw_hqsam"
    with PeriodicCommit():
        shutil.rmtree(raw, ignore_errors=True)
        shutil.rmtree(raw + "_color", ignore_errors=True)
        _sh(f"python seg/raw_mask_sam.py --dataset_path {ROOT}/data/ --scene_name {SCENE} "
            f"--image_folder images --method hqsam", st, cwd=REPO, env=_seg_env())
        per_frame, gaps = {}, {}
        for p in sorted(Path(raw).glob("*.png")):
            m = np.array(Image.open(p))
            ids = [int(i) for i in np.unique(m) if i != 0]
            missing = sorted(set(range(1, max(ids, default=0) + 1)) - set(ids))
            per_frame[p.stem] = {"n_ids": len(ids), "max_id": max(ids, default=0), "shape": list(m.shape)}
            if missing:
                gaps[p.stem] = missing
        n_img = len(list(Path(f"{D}/images").glob("*")))
        summary = {
            "n_images": n_img, "n_masks": len(per_frame),
            "mask_shapes": sorted({tuple(v["shape"]) for v in per_frame.values()}),
            "ids_per_frame": {"mean": round(float(np.mean([v["n_ids"] for v in per_frame.values()])), 2),
                              "min": min(v["n_ids"] for v in per_frame.values()),
                              "max": max(v["n_ids"] for v in per_frame.values())},
            "frames_with_id_gaps": len(gaps),
            "id_gaps_note": "raw_mask_sam.py paints higher-score masks over lower ones, so a fully covered mask "
                            "leaves its id unused; mask_associate.py handles empty ids",
            "id_gaps": gaps, "per_frame": per_frame,
        }
        summary["check_ok"] = summary["n_masks"] == n_img == 177 and summary["mask_shapes"] == [(738, 994)]
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=list))
        _copy_dir(raw, out / "1_masks")
        _copy_dir(raw + "_color", out / "2_masks_color")
    return _finish(5, st, "done" if summary["check_ok"] else "failed",
                   check={k: summary[k] for k in ["n_masks", "mask_shapes", "ids_per_frame", "frames_with_id_gaps",
                                                  "check_ok"]})


def _pick_napkin_id(assoc_dir: str, ref_dir: str) -> dict:
    """Napkin id = associated id with the largest IoU (summed over frames) with the verified napkin mask."""
    import numpy as np
    from PIL import Image

    inter, area, frames = {}, {}, []
    ref_total = 0
    for rp in sorted(Path(ref_dir).glob("*.png")):
        r = np.array(Image.open(rp)) > 0
        a = np.array(Image.open(Path(assoc_dir) / rp.name))
        ref_total += int(r.sum())
        for i, c in zip(*np.unique(a[r], return_counts=True)):
            if i:
                inter[int(i)] = inter.get(int(i), 0) + int(c)
        for i, c in zip(*np.unique(a, return_counts=True)):
            if i:
                area[int(i)] = area.get(int(i), 0) + int(c)
        frames.append((rp.stem, r, a))
    iou = {i: inter[i] / (area[i] + ref_total - inter[i]) for i in inter}
    ranked = sorted(iou, key=iou.get, reverse=True)
    best = ranked[0]
    per_frame = {}
    for stem, r, a in frames:
        b = a == best
        i, u = int((b & r).sum()), int((b | r).sum())
        if u:
            per_frame[stem] = {"iou": round(i / u, 4), "precision": round(i / max(int(b.sum()), 1), 4),
                               "recall": round(i / max(int(r.sum()), 1), 4)}
    vis = [k for k, v in per_frame.items() if v["recall"] > 0 or v["precision"] > 0]
    return {
        "napkin_id": best,
        "iou_total": round(iou[best], 4),
        "precision_total": round(inter[best] / area[best], 4),
        "recall_total": round(inter[best] / ref_total, 4),
        "candidates_top5": [{"id": i, "iou_total": round(iou[i], 4), "inter_px": inter[i], "id_area_px": area[i]}
                            for i in ranked[:5]],
        "reference": f"CoMe_Infusion/Step_1_Sync_ID labels == {NAPKIN_REF_ID} (step3_reuse_napkin_reference)",
        "frames_ref_visible": sum(1 for _, r, _ in frames if r.any()),
        "frames_id_present": sum(1 for _, _, a in frames if (a == best).any()),
        "per_frame": per_frame,
        "note": "low precision_total would mean the id also covers other objects (e.g. uint8 id overflow)",
        "frames_listed": len(vis),
    }


@app.function(image=main_image, gpu=GPU, memory=32768, volumes=VOLUMES, timeout=8 * 3600, retries=RETRIES)
def step6_sync_mask(force: bool = False) -> dict:
    """run_seg.sh 3-4: Key Object Management association on the CoMe 3DGS + label numbers; pick the napkin id."""
    import cv2

    st = _begin(6, force)
    if st is None:
        return _load_state(6)
    out = Path(EXPORT) / st["export"]
    assoc = f"{D}/associated_hqsam"
    with PeriodicCommit():
        shutil.rmtree(assoc, ignore_errors=True)
        shutil.rmtree(assoc + "_color", ignore_errors=True)
        _sh(f"python seg/mask_associate.py --source_path {D} --model_path {M} --resolution 1 "
            f"--mask_generator hqsam --eval", st, cwd=REPO, env=_seg_env())
        # add_label_num_hqsam.py reads <source>/images_<resolution>: images_1 -> images (environment only)
        tmp = Path("/tmp/label_num")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        (tmp / "images_1").symlink_to(f"{D}/images", target_is_directory=True)
        (tmp / "associated_hqsam").symlink_to(assoc, target_is_directory=True)
        _sh(f"python tools/add_label_num_hqsam.py --source_path {tmp} --resolution 1 --mask_generator hqsam",
            st, cwd=REPO, env=_seg_env())

        scene_info = json.loads((Path(assoc) / "scene.json").read_text())
        _copy_dir(assoc, out / "1_masks_synced", exclude=("scene.json",))
        _copy_dir(assoc + "_color", out / "2_masks_synced_color")
        _copy_dir(str(tmp / "images_1_num"), out / "3_ids_on_images")
        (out / "4_ids_on_masks_color").mkdir(parents=True, exist_ok=True)
        for p in sorted(Path(assoc + "_color").glob("*.png")):
            ann = cv2.imread(str(Path(assoc) / p.name), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(str(out / "4_ids_on_masks_color" / p.name), _draw_ids(cv2.imread(str(p)), ann))
        (out / "scene.json").write_text(json.dumps(scene_info, indent=2))

        pick = _pick_napkin_id(assoc, f"{REF}/napkin_mask_id{NAPKIN_REF_ID}")
        pick["num_classes"] = scene_info["num_classes"]
        pick["hit_256_cap"] = scene_info["num_classes"] >= 256
        (out / "napkin_id.json").write_text(json.dumps(pick, indent=2))
    n = len(list((out / "1_masks_synced").glob("*.png")))
    ok = n == 177 and pick["iou_total"] > 0.3
    return _finish(6, st, "done" if ok else "failed",
                   check={"n_masks": n, "num_classes": pick["num_classes"], "hit_256_cap": pick["hit_256_cap"],
                          "napkin_id": pick["napkin_id"], "iou_total": pick["iou_total"],
                          "precision_total": pick["precision_total"], "recall_total": pick["recall_total"],
                          "candidates_top5": pick["candidates_top5"]})


def _id_iou(pred_dirs: list[Path], gt_dir: str, obj_id: int) -> dict:
    """IoU of `obj_id` between rendered objects_pred and the synced HQ-SAM masks, summed over all views."""
    import numpy as np
    from PIL import Image

    inter = union = 0
    per_view = {}
    for d in pred_dirs:
        for p in sorted(d.glob("*.png")):
            g = Path(gt_dir) / p.name
            if not g.exists():
                continue
            a = np.array(Image.open(p)) == obj_id
            b = np.array(Image.open(g)) == obj_id
            i, u = int((a & b).sum()), int((a | b).sum())
            inter, union = inter + i, union + u
            if u:
                per_view[p.stem] = round(i / u, 4)
    return {"iou_total": round(inter / max(union, 1), 4), "views_with_id": len(per_view), "per_view": per_view}


def _ids_on_color(pred_dir: Path, color_dir: Path, dst: Path) -> int:
    import cv2
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(color_dir.glob("*.png")):
        ann = cv2.imread(str(pred_dir / p.name), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(str(dst / p.name), _draw_ids(cv2.imread(str(p)), ann))
        n += 1
    return n


@app.function(image=come_image, gpu=GPU, memory=65536, volumes=VOLUMES, timeout=6 * 3600, retries=RETRIES)
def step7_id_distill(force: bool = False) -> dict:
    """run_seg.sh 5-6: object-ID distillation into the CoMe Gaussians (only obj_dc trained, 2000 it.) + render."""
    st6 = _require_done(6)
    napkin = int(st6["check"]["napkin_id"])
    st = _begin(7, force)
    if st is None:
        return _load_state(7)
    work = _come_workdir()
    env = _come_env(work)
    out = Path(EXPORT) / st["export"]
    it_dir = Path(MI) / "point_cloud" / f"iteration_{ITER_ID}"
    with PeriodicCommit():
        if not (it_dir / "classifier.pth").exists():
            _sh(f"python seg/distillation.py --source_path {D} --model_path {MI} --vanilla_3dgs_path {M} "
                f"--resolution 1 --object_path associated_hqsam --eval", st, cwd=work, env=env)
        _commit("distillation")
        _sh(f"python render.py -m {MI} --render_video", st, cwd=work, env=env)

        shutil.copy2(it_dir / "point_cloud.ply", out / f"point_cloud_id_iter{ITER_ID}.ply")
        shutil.copy2(it_dir / "classifier.pth", out / "classifier.pth")
        for split in ("train", "test"):
            r = Path(MI) / split / f"ours_{ITER_ID}"
            _copy_dir(str(r / "renders"), out / "1_renders_rgb" / split)
            _ids_on_color(r / "objects_pred", r / "objects_pred_color", out / "2_renders_id_color_num" / split)
        videos = sorted((Path(MI) / "video" / f"ours_{ITER_ID}").glob("*.mp4"))
        if videos:
            shutil.copy2(videos[0], out / "video_rgb_id.mp4")
        iou = _id_iou([Path(MI) / s / f"ours_{ITER_ID}" / "objects_pred" for s in ("train", "test")],
                      f"{D}/associated_hqsam", napkin)
        check = {"napkin_id": napkin, "napkin_iou_render_vs_synced_mask": iou["iou_total"],
                 "views_with_napkin": iou["views_with_id"],
                 "n_renders": len(list((out / "1_renders_rgb").rglob("*.png")))}
        (out / "summary.json").write_text(json.dumps({**check, "per_view_iou": iou["per_view"]}, indent=2))
    ok = check["n_renders"] == 177 and check["napkin_iou_render_vs_synced_mask"] >= 0.5
    return _finish(7, st, "done" if ok else "failed", check=check)


@app.function(image=come_image, gpu=GPU, memory=65536, volumes=VOLUMES, timeout=4 * 3600, retries=RETRIES)
def step8_removal_virtual_cam(force: bool = False) -> dict:
    """run_remove.sh 1-3: config, object removal, 30 virtual cameras orbiting the removed object (paper Sec. 3.4)."""
    import numpy as np
    from PIL import Image
    from plyfile import PlyData

    _require_done(7)
    napkin = int(_load_state(6)["check"]["napkin_id"])
    st = _begin(8, force)
    if st is None:
        return _load_state(8)
    work = _come_workdir()
    env = _come_env(work)
    out = Path(EXPORT) / st["export"]
    rem = Path(MI) / "point_cloud_object_removal" / f"iteration_{ITER_ID}"
    with PeriodicCommit():
        _sh(f"python tools/init_configs.py --dataset_name {DATASET} --scene {SCENE} --target_id {napkin} "
            f"--target_surronding_id None", st, cwd=work, env=env)
        _sh(f"python edit_object_removal.py --source_path {D} -m {MI} --config_file {CFG_REMOVAL} --render_video",
            st, cwd=work, env=env)

        # center of the removed object: median of its Gaussians (robust to stray points of the 3D mask)
        v = PlyData.read(str(rem / f"point_cloud_{napkin}.ply")).elements[0]
        xyz = np.stack([np.asarray(v[a]) for a in "xyz"], axis=1)
        med = np.median(xyz, axis=0)
        center = {"object_id": napkin, "n_gaussians": int(len(xyz)), "median_world": med.round(5).tolist(),
                  "mean_world": xyz.mean(0).round(5).tolist(),
                  "used": "median_world (INPAINT360_OBJECT_CENTER for virtual_pose, plyfusion, edit_object_inpaint)"}
        center_str = ",".join(f"{c:.6f}" for c in med)
        (Path(MI) / "object_center.json").write_text(json.dumps(center, indent=2))

        _sh(f"python tools/virtual_pose.py -s {D} -m {MI} --config_file {CFG_REMOVAL}", st, cwd=work,
            env={**env, "INPAINT360_OBJECT_CENTER": center_str})

        cfg = out / "1_config"
        cfg.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f"{ROOT}/{CFG_REMOVAL}", cfg / "object_removal.json")
        shutil.copy2(f"{ROOT}/{CFG_INPAINT}", cfg / "object_inpaint.json")
        o2 = out / "2_object_removal"
        o2.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rem / "point_cloud.ply", o2 / "point_cloud_removed.ply")
        shutil.copy2(rem / f"point_cloud_{napkin}.ply", o2 / f"removed_object_id{napkin}.ply")
        (o2 / "object_center.json").write_text(json.dumps(center, indent=2))
        for vdir, name in [(f"ours_ours_object_removal_{ITER_ID}", "video_removed_scene.mp4"),
                           (f"ours_ours_removed_obj_{napkin}", "video_removed_object.mp4")]:
            mp4 = sorted((Path(MI) / "video" / vdir).glob("*.mp4"))
            if mp4:
                shutil.copy2(mp4[0], o2 / name)
        full = Path(MI) / "virtual" / f"ours_{ITER_ID}"
        removed = Path(MI) / "virtual" / "ours_object_removal" / f"iteration_{ITER_ID}"
        _copy_dir(str(full / "renders"), out / "3_virtual_views_original")
        _copy_dir(str(removed / "renders"), out / "4_virtual_views_removed")
        _ids_on_color(full / "objects_pred", full / "objects_pred_color", out / "5_virtual_views_ids")

        px = {p.stem: int((np.array(Image.open(p)) == napkin).sum()) for p in sorted((full / "objects_pred").glob("*.png"))}
        visible = [k for k, n in px.items() if n > 0]
        check = {"napkin_id": napkin, "object_center_world": center_str, "n_virtual_views": len(px),
                 "views_with_object": len(visible), "views_without_object": [k for k in px if px[k] == 0],
                 "first_view_object_px": px.get("00000", 0),
                 "zip_for_samt": os.path.exists(f"{SAMT_VOL}/assets/images.zip"),
                 "circle_radius": json.loads(Path(f"{ROOT}/{CFG_REMOVAL}").read_text()).get("circle_radius"),
                 "target_object_radius": json.loads(Path(f"{ROOT}/{CFG_REMOVAL}").read_text()).get("target_object_radius")}
        (out / "summary.json").write_text(json.dumps({**check, "object_px_per_view": px}, indent=2))
    ok = check["n_virtual_views"] == 30 and check["first_view_object_px"] > 0 and check["views_with_object"] >= 24 \
        and check["zip_for_samt"]
    return _finish(8, st, "done" if ok else "failed", check=check, object_center_world=center_str)


@app.function(image=main_image, gpu=GPU, memory=32768, volumes=VOLUMES, timeout=3600, retries=RETRIES)
def step9_inpaint_masks_samt(force: bool = False) -> dict:
    """run_remove.sh 4 without the browser (same as Based_Model/Inpaint360GS samt_auto): the SAM-Track calls app.py
    makes, one positive click at the centroid of the removed object in virtual view 0, automatic re-click where the
    track is lost, and mask parts that do not touch the (dilated) object footprint are dropped."""
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    _require_done(8)
    target_id = int(_load_state(6)["check"]["napkin_id"])
    st = _begin(9, force)
    if st is None:
        return _load_state(9)
    out = Path(EXPORT) / st["export"]
    _link(Path(SAMT) / "tracking_results", f"{SAMT_VOL}/tracking_results")
    os.chdir(SAMT)
    sys.path.insert(0, SAMT)
    from model_args import aot_args, sam_args, segtracker_args
    from SegTracker import SegTracker
    from seg_track_anything import aot_model2ckpt, img_seq_type_input_tracking, save_prediction, draw_mask

    objects_pred = Path(MI) / "virtual" / f"ours_{ITER_ID}" / "objects_pred"
    with PeriodicCommit():
        # get_meta_from_img_seq: unzip assets/images.zip into ./assets/images, first sorted image = first frame
        img_dir = Path(SAMT) / "assets" / "images"
        shutil.rmtree(img_dir, ignore_errors=True)
        # (the unzip binary of main_image exits 2 on this archive although it tests fine; same extraction in Python)
        import zipfile
        img_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(f"{SAMT_VOL}/assets/images.zip") as z:
            z.extractall(img_dir)
        imgs_path = sorted(str(p) for p in img_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        first = cv2.cvtColor(cv2.imread(imgs_path[0]), cv2.COLOR_BGR2RGB)

        def obj_mask(i):
            return cv2.imread(str(objects_pred / Path(imgs_path[i]).name), -1) == target_id

        def blob_click(o):
            n, lab, stats, cents = cv2.connectedComponentsWithStats(o.astype(np.uint8))
            k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            x, y = int(round(cents[k][0])), int(round(cents[k][1]))
            if lab[y, x] != k:  # centroid outside a concave blob: nearest pixel of the blob
                ys, xs = np.nonzero(lab == k)
                j = int(np.argmin((xs - x) ** 2 + (ys - y) ** 2))
                x, y = int(xs[j]), int(ys[j])
            return x, y, int(stats[k][4])

        if not obj_mask(0).any():
            raise RuntimeError(f"id {target_id} not visible in the first virtual view")
        x, y, area = blob_click(obj_mask(0))
        print(f"[samt-auto] click (x={x}, y={y}) on {Path(imgs_path[0]).name}, id {target_id} area {area} px")

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
        predicted_mask, _ = tracker.seg_acc_click(origin_frame=first, coords=np.array([[x, y]]),
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
        shutil.rmtree(io_args["output_mask_dir"], ignore_errors=True)
        img_seq_type_input_tracking(tracker, io_args, video_name, imgs_path, 8, 0)
        masks_dir = io_args["output_mask_dir"]
        raw_copy = Path(SAMT) / "tracking_results" / f"{video_name}_masks_raw"
        shutil.rmtree(raw_copy, ignore_errors=True)
        shutil.copytree(masks_dir, raw_copy)

        def track_mask(i):
            return np.array(Image.open(os.path.join(masks_dir, Path(imgs_path[i]).stem + ".png")).convert("P"))

        # "roll back" to the first frame where the object is visible but the track missed it, click again and
        # "Start tracking to refine" (app.py roll_back_sam_click + tracking frame_num=f)
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
            cx, cy, _ = blob_click(obj_mask(bad))
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

        # drop mask components that do not touch the (dilated) object footprint (tracker drift elsewhere)
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
                save_prediction(np.where(keep, m, 0).astype(m.dtype), masks_dir, Path(p).stem + ".png")
        print(f"[samt-auto] refined at {refined}; removed off-object pixels in frames {gated}")

        n_masks = len([f for f in os.listdir(masks_dir) if f.endswith(".png") and not f.endswith("_new.png")])
        areas = {Path(p).stem: round(100 * float((cv2.imread(os.path.join(masks_dir, Path(p).stem + ".png"), 0) > 0).mean()), 3)
                 for p in imgs_path}
        _copy_dir(masks_dir, out / "1_masks")
        (out / "2_masks_overlay").mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(imgs_path):  # final mask + red contour of the removed object footprint
            frame = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            vis = cv2.cvtColor(draw_mask(frame, track_mask(i)), cv2.COLOR_RGB2BGR)
            cs, _ = cv2.findContours(obj_mask(i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cs, -1, (0, 0, 255), 1)
            cv2.imwrite(str(out / "2_masks_overlay" / Path(p).name), vis)
        check = {"n_masks": n_masks, "n_virtual_views": len(imgs_path), "click": [x, y], "refine": refined,
                 "gated_frames": gated, "empty_masks": [k for k, a in areas.items() if a == 0],
                 "area_pct_min": min(areas.values()), "area_pct_max": max(areas.values())}
        (out / "summary.json").write_text(json.dumps({**check, "area_pct_per_view": areas}, indent=2))
    ok = n_masks == len(imgs_path) == 30 and check["area_pct_max"] <= 50
    return _finish(9, st, "done" if ok else "failed", check=check)


@app.function(image=come_image, cpu=4, memory=16384, volumes=VOLUMES, timeout=3600, retries=RETRIES)
def step10_lama_io(st: dict, to_lama: bool) -> dict | None:
    """tools/prepare_lama_data.py: --inpaint2lama (mask enlarge + copy into LaMa/data), or copy the LaMa results back
    (and export step 10)."""
    import matplotlib
    import numpy as np
    from PIL import Image

    work = _come_workdir()
    if to_lama:
        for d in [f"{LAMA_VOL}/data/color/{LAMA_DATA_NAME}", f"{LAMA_VOL}/data/depth/{LAMA_DATA_NAME}",
                  f"{LAMA_VOL}/output/color/{LAMA_DATA_NAME}", f"{LAMA_VOL}/output/depth/{LAMA_DATA_NAME}",
                  f"{D}/inpaint_2d_unseen_mask_virtual", f"{D}/images_inpaint_unseen_virtual"]:
            _clear_dir(d)
    with PeriodicCommit():
        _sh(f"python tools/prepare_lama_data.py -s {D} -m {MI} -r 1" + (" --inpaint2lama" if to_lama else ""),
            st, cwd=work, env=_come_env(work))
        if to_lama:
            return None
        out = Path(EXPORT) / st["export"]
        depth_done = Path(MI) / "virtual" / "ours_object_removal" / f"iteration_{ITER_ID}" / "depth_completed"
        n1 = _copy_dir(f"{D}/inpaint_2d_unseen_mask_virtual", out / "1_masks_dilated")
        n2 = _copy_dir(f"{D}/images_inpaint_unseen_virtual", out / "2_inpainted_rgb")
        n3 = _copy_dir(str(depth_done), out / "3_inpainted_depth_npy")
        vis = out / "4_inpainted_depth_vis"
        vis.mkdir(parents=True, exist_ok=True)
        for p in sorted(depth_done.glob("*.npy")):  # min-max normalised, jet (as render.py)
            d = np.load(p).squeeze()
            d = (d - d.min()) / max(float(d.max() - d.min()), 1e-8)
            Image.fromarray((matplotlib.colormaps["jet"](d)[..., :3] * 255).astype(np.uint8)).save(vis / (p.stem + ".png"))
        check = {"n_masks_dilated": n1, "n_inpainted_rgb": n2, "n_inpainted_depth": n3}
        (out / "summary.json").write_text(json.dumps(check, indent=2))
    return check


@app.function(image=lama_image, gpu=GPU, memory=32768, volumes=VOLUMES, timeout=3 * 3600, retries=RETRIES)
def step10_lama_predict(st: dict) -> None:
    """run_inpaint.sh 1b: LaMa color + depth (env lama, original code)."""
    vol.reload()
    lama = f"{REPO}/LaMa"
    _link(Path(lama) / "data", f"{LAMA_VOL}/data")
    _link(Path(lama) / "output", f"{LAMA_VOL}/output")
    big = Path(lama) / "big-lama"
    if not big.exists():
        big.symlink_to("/opt/weights/big-lama", target_is_directory=True)
    with PeriodicCommit():
        for kind in ("color", "depth"):
            _sh(f"python bin/predict_{kind}.py --data_name {LAMA_DATA_NAME}", st, cwd=lama,
                env={"TORCH_HOME": lama, "PYTHONPATH": lama})


@app.function(image=light_image, cpu=1, volumes=VOLUMES, timeout=8 * 3600)
def step10_inpaint2d_lama(force: bool = False) -> dict:
    """run_inpaint.sh 1: SAM-Track masks -> LaMa (color + depth) -> back into the dataset / virtual depth."""
    _require_done(9)
    st = _begin(10, force)
    if st is None:
        return _load_state(10)
    step10_lama_io.remote(st, True)
    step10_lama_predict.remote(st)
    check = step10_lama_io.remote(st, False)
    ok = check["n_masks_dilated"] == 30 and check["n_inpainted_rgb"] >= 30 and check["n_inpainted_depth"] == 30
    return _finish(10, st, "done" if ok else "failed", check=check)


@app.function(image=come_image, gpu=GPU, memory=65536, volumes=VOLUMES, timeout=8 * 3600, retries=RETRIES)
def step11_inpaint_3dgs_come_loss(force: bool = False) -> dict:
    """run_inpaint.sh 2-3: colored point-cloud fusion + fine-tuning of the inpainted region with the Inpaint360GS
    loss + CoMe geometric regularization (COME_LOSS=1), on the virtual views orbiting the removed object."""
    from plyfile import PlyData

    _require_done(10)
    center = _object_center()
    st = _begin(11, force)
    if st is None:
        return _load_state(11)
    work = _come_workdir()
    env = _come_env(work, INPAINT360_OBJECT_CENTER=center, COME_LOSS="1")
    out = Path(EXPORT) / st["export"]
    with PeriodicCommit():
        for d in ("point_cloud_object_inpaint_virtual", "train/ours_object_inpaint_virtual",
                  "test/ours_object_inpaint_virtual", "video/ours__object_inpaint_virtual"):
            _clear_dir(f"{MI}/{d}")
        _sh(f"python edit_object_removal_plyfusion.py -s {D} -m {MI} --config_file {CFG_REMOVAL}",
            st, cwd=work, env=env)
        _sh(f"python tools_come/select_init_view.py -s {D} --out {out}/init_view_selection.json", st, cwd=work, env=env)
        init_view = json.loads((out / "init_view_selection.json").read_text())["selected"]
        env["INPAINT360_INIT_VIEW"] = init_view
        _sh(f"python edit_object_inpaint.py -s {D} -m {MI} --config_file {CFG_INPAINT} --resolution 1 --render_video",
            st, cwd=work, env=env)
        ply = Path(MI) / "point_cloud_object_inpaint_virtual" / f"iteration_{INPAINT_ITER}" / "point_cloud.ply"
        shutil.copy2(ply, out / f"point_cloud_inpainted_iter{INPAINT_ITER}.ply")
        for f in ("loss_log.json", "used_virtual_views.json"):
            if (Path(MI) / f).exists():
                shutil.copy2(Path(MI) / f, out / f)
        v = PlyData.read(str(ply)).elements[0]
        names = [p.name for p in v.properties]
        log = json.loads((Path(MI) / "loss_log.json").read_text()) if (Path(MI) / "loss_log.json").exists() else []
        used = json.loads((Path(MI) / "used_virtual_views.json").read_text())
        check = {"n_gaussians": int(v.count), "n_properties": len(names),
                 "has_confidence": "confidence" in names, "has_filter_3D": "filter_3D" in names,
                 "virtual_views_used": len(used["used"]), "virtual_views_skipped": used["skipped"],
                 "object_center_world": center, "init_view": init_view,
                 "loss_first": log[0] if log else None, "loss_last": log[-1] if log else None}
        (out / "summary.json").write_text(json.dumps(check, indent=2))
    ok = check["has_confidence"] and check["has_filter_3D"] and bool(log)
    return _finish(11, st, "done" if ok else "failed", check=check)



@app.function(image=come_image, cpu=4, memory=16384, volumes=VOLUMES, timeout=3600, retries=RETRIES)
def step12_render(force: bool = False) -> dict:
    """Renders written by edit_object_removal.py (step 8) and edit_object_inpaint.py --render_video (step 11):
    train/test {0_gt, 1_removed, 2_inpainted}, a gt|removed|inpainted strip per view, and the two videos."""
    import numpy as np
    from PIL import Image

    _require_done(11)
    st = _begin(12, force)
    if st is None:
        return _load_state(12)
    vol.reload()
    _v2_tree()
    out = Path(EXPORT) / st["export"]
    removed_it = f"ours_object_removal/iteration_{ITER_ID}"
    inpaint_it = f"ours_object_inpaint_virtual/iteration_{INPAINT_ITER}"
    counts = {}
    with PeriodicCommit():
        for split in ("train", "test"):
            n_gt = _copy_dir(f"{MI}/{split}/{inpaint_it}/gt", out / split / "0_gt")
            n_rm = _copy_dir(f"{MI}/{split}/{removed_it}/renders", out / split / "1_removed")
            n_in = _copy_dir(f"{MI}/{split}/{inpaint_it}/renders", out / split / "2_inpainted")
            strip = out / split / "3_gt_removed_inpainted"
            strip.mkdir(parents=True, exist_ok=True)
            for p in sorted((out / split / "2_inpainted").glob("*.png")):
                ims = [Image.open(out / split / d / p.name).convert("RGB") for d in ("0_gt", "1_removed", "2_inpainted")]
                w, h = ims[0].size
                row = Image.new("RGB", (3 * w // 2, h // 2))
                for i, im in enumerate(ims):
                    row.paste(im.resize((w // 2, h // 2)), (i * w // 2, 0))
                row.save(strip / (p.stem + ".jpg"), quality=90)
            counts[split] = {"gt": n_gt, "removed": n_rm, "inpainted": n_in}
        videos = {}
        for vdir, name in [(f"ours_ours_object_removal_{ITER_ID}", "video_1_removed.mp4"),
                           (f"ours__object_inpaint_virtual/iteration_{INPAINT_ITER}", "video_2_inpainted.mp4")]:
            mp4 = sorted((Path(MI) / "video" / vdir).glob("*.mp4"))
            if mp4:
                shutil.copy2(mp4[0], out / name)
            videos[name] = bool(mp4)
        check = {"counts": counts, "videos": videos, "from_step11": _load_state(11)["export"],
                 "note": "LERF teatime has no ground truth without the napkin: no metrics, visual check only"}
        (out / "summary.json").write_text(json.dumps(check, indent=2))
    ok = counts["train"]["inpainted"] == 154 and counts["test"]["inpainted"] == 23 and all(videos.values())
    return _finish(12, st, "done" if ok else "failed", check=check)


@app.function(image=come_image, gpu=GPU, memory=65536, volumes=VOLUMES, timeout=3600)
def diag_virtual_check() -> dict:
    """Diagnostic (no step state): renders from hand-built cameras vs real cameras (tools_come/virtual_check.py)."""
    vol.reload()
    work = _come_workdir()
    out = Path(EXPORT) / "diag_virtual_check"
    k = 2
    while out.exists():
        out = Path(EXPORT) / f"diag_virtual_check_v{k}"
        k += 1
    st = {"export": out.name, "attempts": 1}
    _sh(f"python tools_come/virtual_check.py -s {D} -m {MI} --out_dir {out}", st, cwd=work, env=_come_env(work))
    _commit("diag")
    return {"export": out.name, **json.loads((out / "report.json").read_text())}

@app.function(image=come_image, cpu=2, memory=8192, volumes=VOLUMES, timeout=3600)
def setup_v2() -> dict:
    """State of v1 steps 4-9 -> state_v2 (marked reused); virtual renders/depth of v1 step 8 -> output/teatime_v2."""
    vol.reload()
    copied = []
    for s in range(4, 10):
        p1, p2 = Path(STATE1) / f"step{s}.json", Path(STATE) / f"step{s}.json"
        if p2.exists():
            continue
        st = json.loads(p1.read_text())
        if st["status"] != "done":
            raise RuntimeError(f"v1 step {s} is not done")
        st["reused_from_v1"] = str(p1)
        _save_state(s, st)
        copied.append(s)
    v1, v2 = Path(MI1) / "virtual", Path(MI2VOL) / "virtual"
    done = []
    for rel in [f"ours_{ITER_ID}/renders", f"ours_{ITER_ID}/depth", f"ours_{ITER_ID}/objects_pred",
                f"ours_object_removal/iteration_{ITER_ID}/renders", f"ours_object_removal/iteration_{ITER_ID}/depth"]:
        if not (v2 / rel).exists():
            shutil.copytree(v1 / rel, v2 / rel)
        done.append(f"{rel}: {len(list((v2 / rel).iterdir()))} files")
    _commit("setup v2")
    return {"states_copied": copied, "virtual": done}

STEP_FNS = {10: step10_inpaint2d_lama, 11: step11_inpaint_3dgs_come_loss, 12: step12_render}


@app.function(image=light_image, cpu=1, volumes=VOLUMES, timeout=24 * 3600)
def chain(steps: list[int], force: bool = False) -> list[dict]:
    """Run the steps in order on Modal (survives the local machine going offline); stop at the first failure."""
    results = []
    for s in steps:
        print(f"===== step {s}", flush=True)
        r = STEP_FNS[s].remote(force=force)
        print(json.dumps(r, indent=2, ensure_ascii=False), flush=True)
        results.append(r)
        if not r or r.get("status") != "done":
            print(f"step {s} not done -> chain stopped", flush=True)
            break
    return results

# --------------------------------------------------------------------------------------------------
# 5. Local entrypoints
# --------------------------------------------------------------------------------------------------

def _assert_workspace() -> None:
    r = subprocess.run([sys.executable, "-m", "modal", "profile", "current"], capture_output=True, text=True)
    current = r.stdout.strip()
    if current != WORKSPACE_PROFILE:
        raise SystemExit(f"Modal profile is '{current}', expected '{WORKSPACE_PROFILE}': "
                         f"python -m modal profile activate {WORKSPACE_PROFILE}")


def _remote_files(rel: str) -> dict[str, int]:
    try:
        return {e.path.lstrip("/"): e.size for e in vol.listdir(rel, recursive=True)
                if e.type == modal.volume.FileEntryType.FILE}
    except Exception:  # noqa: BLE001
        return {}


@app.local_entrypoint()
def run(steps: str = "10,11,12", force: bool = False):
    _assert_workspace()
    print(json.dumps(setup_v2.remote(), indent=2))
    for r in chain.remote([int(s) for s in steps.split(",")], force=force):
        print(json.dumps(r, indent=2, ensure_ascii=False))


@app.local_entrypoint()
def status():
    for s in STEP_FNS:
        try:
            st = json.loads(b"".join(vol.read_file(f"{ROOT_REL}/state_v2/step{s}.json")))
        except Exception:  # noqa: BLE001
            st = None
        print(f"step {s}: " + (json.dumps(st, indent=2, ensure_ascii=False) if st else "not started"))


@app.local_entrypoint()
def download(steps: str = "10,11,12"):
    """export/<step> -> outputs/teatime_napkin/<step> (an existing local folder gets _v2, _v3 ...)."""
    _assert_workspace()
    for s in [int(x) for x in steps.split(",")]:
        st = json.loads(b"".join(vol.read_file(f"{ROOT_REL}/state_v2/step{s}.json")))
        name = st["export"]
        rel = f"{ROOT_REL}/export/{RUN_NAME}_v2/{name}"
        dst = LOCAL_OUT / name
        k = 2
        while dst.exists():
            dst = LOCAL_OUT / f"{name}_v{k}"
            k += 1
        files = _remote_files(rel)
        log_rel = f"{ROOT_REL}/export/{RUN_NAME}_v2/logs/{name}.log"
        for path, size in list(files.items()) + [(log_rel, None)]:
            out = (LOCAL_OUT / "logs" / f"{dst.name}.log") if path == log_rel else dst / path[len(rel) + 1:]
            out.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(out, "wb") as f:
                    for chunk in vol.read_file(path):
                        f.write(chunk)
            except Exception as e:  # noqa: BLE001
                print(f"[download] {path}: {e}")
                continue
            if size is not None and out.stat().st_size != size:
                raise SystemExit(f"size mismatch: {path}")
        print(f"[download] step {s}: {len(files)} files -> {dst}")


@app.local_entrypoint()
def diag():
    _assert_workspace()
    print(json.dumps(diag_virtual_check.remote(), indent=2))
