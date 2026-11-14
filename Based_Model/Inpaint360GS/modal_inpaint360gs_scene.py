"""
Inpaint360GS (original code, standard 3DGS rasterizer) on an Inpaint360GS-dataset scene, full pipeline on Modal
===============================================================================================================
Baseline "Inpaint360GS (gốc)" of plan.md (Table 2). Runs the authors' scripts unchanged, in order:
    run_seg.sh     -> step1 3DGS, step2 HQ-SAM masks, step3 mask association + label numbers, step4 ID distillation
    run_remove.sh  -> step5 init_configs + edit_object_removal + virtual_pose, SAM-Track (automatic click, as
                      modal_inpaint360gs_object.py::samt_auto)
    run_inpaint.sh -> step6 LaMa colour/depth, step7 plyfusion + edit_object_inpaint, step8 renders +
                      tools/metrics_fid_masked.py (inpaint360 has object-free GT: test_* views + unseen_mask)

Step 0 (COLMAP) is NOT recomputed: it is reused from outputs/Inpaint360GS/ours/<run>/step0_colmap, which is the
authors' released sparse/0 (byte-identical, checked by md5 in ::upload) -> same coordinate frame and test cameras as
the other methods (plan.md R33). images_2 (read by raw_mask_sam.py / add_label_num_hqsam.py) and unseen_mask (read
by metrics_fid_masked.py) come from the released dataset F:/inpaint360/inpaint360/<scene>.

Volume `inpaint360gs-baseline` (current Modal profile): runs/<run>/{data/inpaint360/<scene>, output/inpaint360/<scene>,
lama, samt, config, export/stepN_*, state}; CropFormer weight in /vol/weights (fetched with the `huggingface` secret).
The data path contains "inpaint360" -> dataset_readers.py inpaint360 branch (test_* = object-free GT).

  $env:PYTHONIOENCODING="utf-8"                                                         # from F:\\Safe-GS
  python -m modal run Based_Model/Inpaint360GS/modal_inpaint360gs_scene.py::upload --run redbull_main --scene redbull
  python -m modal run --detach Based_Model/Inpaint360GS/modal_inpaint360gs_scene.py::run --run redbull_main --scene redbull
  python -m modal run Based_Model/Inpaint360GS/modal_inpaint360gs_scene.py::status --run redbull_main
  python -m modal run Based_Model/Inpaint360GS/modal_inpaint360gs_scene.py::download --run redbull_main
      -> outputs/Inpaint360GS/Inpaint360GS/<run>/stepN_* (+ logs)

Target id: Inpaint360GS's own association IDs differ from ours, so after step 3 the id is picked automatically at the
same click as ours (configs/datasets/inpaint360gs.yaml 6id: IMG_2760 (1154, 830) full-res = (577, 415) in images_2)
unless --target-id > 0. redbull: ID 7 = wire bull (checked on IMG_2760/2790/2830, ~1-3 % of the frame).
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

if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace")

import modal

# --------------------------------------------------------------------------------------------------- configuration
GPU = "L4"                                # image build GPU (keeps the cached build)
RUN_GPU = ["L4", "A10G", "L40S"]          # runtime fallback; only sm_86/sm_89 GPUs (TORCH_CUDA_ARCH_LIST of the image)
BIG_GPU = "L40S"   # step 7-8: render_video_func_wriva OOMs on 24 GB; NOT A100/H100: CUDA ext. built for sm_86/89 only
DATASET = "inpaint360"
RES = 2                                   # run_seg.sh / run_remove.sh / run_inpaint.sh: resolution=2 for inpaint360
LOCAL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = LOCAL_ROOT.parents[1] if len(LOCAL_ROOT.parents) > 1 else LOCAL_ROOT
OUT_ROOT = PROJECT_ROOT / "outputs" / "Inpaint360GS" / "Inpaint360GS"
OURS_ROOT = PROJECT_ROOT / "outputs" / "Inpaint360GS" / "ours"
DATA_ROOT = Path("F:/inpaint360/inpaint360")
# ours 6id click, converted to images_2 pixels: configs/datasets/inpaint360gs.yaml gives (1154, 830) in FULL-resolution
# pixels (2962x1912); at (1154, 830) in images_2 lies the lawn (ID 8, 43-70 % of the frame), the bull is at (577, 415)
DEFAULT_CLICK = {"redbull": ("IMG_2760", 577, 415)}
MAX_TARGET_AREA = 0.25      # an ID covering more of a frame on average is background, not a removable object

VOL = "/vol"
REPO = "/opt/inpaint360gs"
SAMT = f"{REPO}/Segment-and-Track-Anything"
CROPFORMER = f"{REPO}/seg/detectron2/detectron2/projects/CropFormer"
CROPFORMER_WEIGHT = "CropFormer_hornet_3x_03823a.pth"
CROPFORMER_URL = ("https://huggingface.co/datasets/qqlu1992/Adobe_EntitySeg/resolve/main/CropFormer_model/"
                  "Entity_Segmentation/CropFormer_hornet_3x/CropFormer_hornet_3x_03823a.pth")
STEP_NAMES = {0: "colmap", 1: "3dgs", 2: "mask2d_hqsam", 3: "sync_mask", 4: "id_3dgs", 5: "removal_virtual_cam",
              6: "inpaint2d_lama", 7: "inpaint_3dgs", 8: "render_metrics"}
COMMIT_EVERY_S = 300
CKPT_EVERY = 5000
CKPT_KEEP = 2

app = modal.App("inpaint360gs-scene")
vol = modal.Volume.from_name("inpaint360gs-baseline", create_if_missing=True)
VOLUMES = {VOL: vol}
RETRIES = modal.Retries(max_retries=2, backoff_coefficient=1.0, initial_delay=30.0)
BASE_ENV = {"PYTHONUNBUFFERED": "1", "QT_QPA_PLATFORM": "offscreen", "MPLBACKEND": "Agg", "WANDB_MODE": "disabled"}

# --------------------------------------------------------------------------------------------------- images
# verbatim from modal_inpaint360gs_object.py (+ this file ignored, so the copied repo is identical -> cached build)
REPO_IGNORE = [
    "outputs", "outputs/**", "data", "data/**", "output", "output/**", "LaMa", "LaMa/**",
    "Paper", "Paper/**", "media", "media/**", "plan.md", "modal_inpaint360gs.py", "modal_inpaint360gs_object.py",
    "modal_inpaint360gs_scene.py",
    "tools_safe/label_color_ids.py",
    "gaussian_splatting/SIBR_viewers", "gaussian_splatting/SIBR_viewers/**",
    "gaussian_splatting/assets", "gaussian_splatting/assets/**",
    "**/third_party/glm/doc", "**/third_party/glm/doc/**",
    "**/third_party/glm/test", "**/third_party/glm/test/**",
    "seg/detectron2/docs", "seg/detectron2/docs/**", "seg/detectron2/tests", "seg/detectron2/tests/**",
    "seg/Entity", "seg/Entity/**", "logs_local", "logs_local/**", "**/__pycache__", "**/__pycache__/**",
    "**/*.pyc", "**/.git", "**/.git/**",
]

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
        f"cd {REPO} && pip install gaussian_splatting/submodules/diff-gaussian-rasterization --no-build-isolation",
        f"cd {REPO} && pip install gaussian_splatting/submodules/simple-knn --no-build-isolation",
        f"cd {REPO} && pip install submodules/diff-gaussian-rasterization --no-build-isolation",
        f"cd {REPO} && pip install . --no-build-isolation",
        gpu=GPU,
    )
    .run_commands(
        "pip install git+https://github.com/facebookresearch/segment-anything.git",
        "pip install setuptools==75.8.0",
        f"cd {REPO}/seg && python -m pip install -e detectron2 --no-build-isolation",
        f"cd {CROPFORMER}/entity_api/PythonAPI && make",
        f"cd {CROPFORMER}/mask2former/modeling/pixel_decoder/ops && sh make.sh",
        f"cd {REPO} && python seg/patch_fvcore.py",
        f"mkdir -p {REPO}/seg/weight",
        gpu=GPU,
    )
    .run_commands(
        f"cd {SAMT} && bash script/install.sh",
        f"mkdir -p {SAMT}/ckpt {SAMT}/ast_master/pretrained_models",
        f"cd {SAMT} && gdown 1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ -O ./ckpt/R50_DeAOTL_PRE_YTB_DAV.pth",
        f"cd {SAMT} && (wget -q --tries=3 -O ./ckpt/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth "
        "|| wget -q --tries=3 -O ./ckpt/sam_vit_b_01ec64.pth "
        "https://huggingface.co/ybelkada/segment-anything/resolve/main/checkpoints/sam_vit_b_01ec64.pth)",
        f"test $(stat -c %s {SAMT}/ckpt/sam_vit_b_01ec64.pth) -gt 300000000",
        f"cd {SAMT} && wget -q -P ./ckpt https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
        f"cd {SAMT} && (wget -q -O ./ast_master/pretrained_models/audio_mdl.pth "
        "'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1' || echo 'audio ckpt skipped (audio unused)')",
        gpu=GPU,
    )
    .run_commands('pip install "uvicorn==0.23.2" "fastapi==0.104.1" "pydantic==2.5.3"')
)

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
    .run_commands(
        f"mkdir -p {REPO}/LaMa/saicinpainting/training/data && cd {REPO}/LaMa/saicinpainting/training/data && "
        "for f in __init__.py aug.py datasets.py masks.py; do wget -q -O $f "
        "https://raw.githubusercontent.com/advimman/lama/786f5936b27fb3dacd2b1ad799e4de968ea697e7/"
        "saicinpainting/training/data/$f; done"
    )
)


# --------------------------------------------------------------------------------------------------- run-scoped paths
class P:
    """Volume paths of one run (set by _bind in every container)."""
    run = scene = R = D = M = D_REL = M_REL = EXPORT = STATE = ""


def _bind(run: str, scene: str) -> None:
    P.run, P.scene = run, scene
    P.R = f"{VOL}/runs/{run}"
    P.D_REL = f"data/{DATASET}/{scene}"
    P.M_REL = f"output/{DATASET}/{scene}"
    P.D = f"{P.R}/{P.D_REL}"
    P.M = f"{P.R}/{P.M_REL}"
    P.EXPORT = f"{P.R}/export"
    P.STATE = f"{P.R}/state"


def _links() -> dict[str, str]:
    r = f"runs/{P.run}"
    return {"data": f"{r}/data", "output": f"{r}/output", "LaMa/data": f"{r}/lama/data",
            "LaMa/output": f"{r}/lama/output", "Segment-and-Track-Anything/assets": f"{r}/samt/assets",
            "Segment-and-Track-Anything/tracking_results": f"{r}/samt/tracking_results",
            f"config/object_removal/{DATASET}": f"{r}/config/object_removal/{DATASET}",
            f"config/object_inpaint/{DATASET}": f"{r}/config/object_inpaint/{DATASET}"}


def _link_volume(links: dict[str, str]) -> None:
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


def _setup(run: str, scene: str) -> None:
    _bind(run, scene)
    vol.reload()
    _link_volume(_links())
    weight, link = Path(VOL) / "weights" / CROPFORMER_WEIGHT, Path(REPO) / "seg" / "weight" / CROPFORMER_WEIGHT
    if weight.exists() and not link.exists():
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(weight)


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
    """Commit the volume every COMMIT_EVERY_S seconds and on SIGTERM (preemption)."""

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
        _commit("end of step")
        return False


def _state_path(step) -> Path:
    return Path(P.STATE) / f"step{step}.json"


def _load_state(step) -> dict | None:
    p = _state_path(step)
    return json.loads(p.read_text()) if p.exists() else None


def _save_state(step, st: dict) -> None:
    Path(P.STATE).mkdir(parents=True, exist_ok=True)
    _state_path(step).write_text(json.dumps(st, indent=2, ensure_ascii=False))


def _begin(step: int, force: bool) -> dict | None:
    st = _load_state(step)
    if st and st["status"] == "done" and not force:
        print(f"[step {step}] already done -> {st['export']}")
        return None
    if st and st["status"] == "running":
        st["attempts"] += 1
    else:
        base, k = f"step{step}_{STEP_NAMES[step]}", 2
        name = base
        while (Path(P.EXPORT) / name).exists():
            name, k = f"{base}_v{k}", k + 1
        st = {"step": step, "status": "running", "t0": time.time(), "export": name, "attempts": 1,
              "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(f"[step {step}] -> {st['export']} (attempt {st['attempts']})")
    (Path(P.EXPORT) / st["export"]).mkdir(parents=True, exist_ok=True)
    _save_state(step, st)
    _commit(f"step {step} begin")
    return st


def _finish(step: int, st: dict, **extra) -> dict:
    st.update(extra, status="done", finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    _save_state(step, st)
    (Path(P.EXPORT) / st["export"] / "state.json").write_text(json.dumps(st, indent=2, ensure_ascii=False))
    _commit(f"step {step} done")
    return st


def _sh(cmd: str, st: dict, cwd: str = REPO, env: dict | None = None) -> None:
    log = Path(P.EXPORT) / "logs" / f"{st['export']}.log"
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


def _export(st: dict, sources: list[tuple[str, str, list[str] | None, list[str]]], t0: float | None = None) -> int:
    """Copy files created/modified since the step started. sources = [(root, dst_sub, include, exclude)]."""
    t0 = (t0 if t0 is not None else st["t0"]) - 2
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
                if any(fnmatch.fnmatch(rel, p) for p in exclude) or os.path.getmtime(src) < t0:
                    continue
                dst = Path(P.EXPORT) / st["export"] / dst_sub / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                n += 1
    print(f"[export] {n} files -> {st['export']}")
    return n


# run_seg.sh: PYTHONPATH=$(pwd):$(pwd)/seg/detectron2 (+ CropFormer project: criterion_view.py imports mask2former)
def _seg_env() -> dict:
    extra = (":" + os.environ["PYTHONPATH"]) if os.environ.get("PYTHONPATH") else ""
    return {"PYTHONPATH": f"{REPO}:{REPO}/seg/detectron2:{CROPFORMER}" + extra}


def _repo_env() -> dict:
    return {"PYTHONPATH": REPO + ((":" + os.environ["PYTHONPATH"]) if os.environ.get("PYTHONPATH") else "")}


# --------------------------------------------------------------------------------------------------- weight
@app.function(image=main_image, cpu=2, volumes=VOLUMES, timeout=3600,
              secrets=[modal.Secret.from_name("huggingface")])
def fetch_weight():
    """CropFormer (HQ-SAM entity) weight from the gated HF dataset -> /vol/weights (once for all runs)."""
    vol.reload()
    dst = Path(VOL) / "weights" / CROPFORMER_WEIGHT
    if dst.exists() and dst.stat().st_size > 800_000_000:
        return f"present ({dst.stat().st_size} bytes)"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    subprocess.run(["wget", "-q", "--header", f"Authorization: Bearer {os.environ['HF_TOKEN']}", "-O", str(tmp),
                    CROPFORMER_URL], check=True)
    if tmp.stat().st_size < 800_000_000:
        raise RuntimeError(f"CropFormer download too small: {tmp.stat().st_size} bytes")
    tmp.rename(dst)
    vol.commit()
    return f"downloaded ({dst.stat().st_size} bytes)"


# --------------------------------------------------------------------------------------------------- steps
def _latest_loadable_checkpoint(folder: str) -> str | None:
    import torch
    ckpts = sorted(Path(folder).glob("chkpnt*.pth"), key=lambda p: int(p.stem.replace("chkpnt", "")), reverse=True)
    for p in ckpts:
        try:
            torch.load(p, map_location="cpu")
            return str(p)
        except Exception as e:  # noqa: BLE001
            print(f"[resume] {p.name} unreadable ({e}); trying older")
    return None


@app.function(image=main_image, gpu=RUN_GPU, memory=32768, volumes=VOLUMES, timeout=8 * 3600, retries=RETRIES)
def step1_train_3dgs(run: str, scene: str, force: bool = False):
    """run_seg.sh 1: vanilla 3DGS (checkpoints only add saves; resume via --start_checkpoint)."""
    _setup(run, scene)
    st = _begin(1, force)
    if st is None:
        return
    out_dir = f"{P.M}/3dgs_output"
    if st["attempts"] == 1 and os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    ckpt_iters = " ".join(str(i) for i in range(CKPT_EVERY, 30000, CKPT_EVERY))
    resume = _latest_loadable_checkpoint(out_dir) if os.path.isdir(out_dir) else None
    cmd = (f"python gaussian_splatting/train.py -s {P.D_REL} -m {P.M_REL}/3dgs_output --init_mode sparse --eval "
           f"--resolution {RES} --checkpoint_iterations {ckpt_iters}")
    if resume:
        cmd += f" --start_checkpoint {resume}"
    with PeriodicCommit(prune_dir=out_dir):
        _sh(cmd, st)
        _export(st, [(out_dir, "", None, ["chkpnt*.pth"])], t0=0)
    _finish(1, st, resumed_from=resume)


@app.function(image=main_image, gpu=RUN_GPU, memory=32768, volumes=VOLUMES, timeout=4 * 3600, retries=RETRIES)
def step2_mask2d(run: str, scene: str, force: bool = False):
    """run_seg.sh 2: raw HQ-SAM (CropFormer) masks on images_2."""
    _setup(run, scene)
    if not (Path(VOL) / "weights" / CROPFORMER_WEIGHT).exists():
        raise RuntimeError("CropFormer weight missing: run fetch_weight")
    st = _begin(2, force)
    if st is None:
        return
    with PeriodicCommit():
        shutil.rmtree(f"{P.D}/raw_hqsam", ignore_errors=True)
        shutil.rmtree(f"{P.D}/raw_hqsam_color", ignore_errors=True)
        _sh(f"python seg/raw_mask_sam.py --dataset_path data/{DATASET}/ --scene_name {scene} "
            f"--image_folder images_{RES} --method hqsam", st, env=_seg_env())
        _export(st, [(P.D, "", ["raw_hqsam/*", "raw_hqsam_color/*"], [])], t0=0)
    _finish(2, st)


@app.function(image=main_image, gpu=RUN_GPU, memory=32768, volumes=VOLUMES, timeout=4 * 3600, retries=RETRIES)
def step3_sync_mask(run: str, scene: str, force: bool = False):
    """run_seg.sh 3-4: 3D mask association + label numbers."""
    _setup(run, scene)
    st = _begin(3, force)
    if st is None:
        return
    with PeriodicCommit():
        _sh(f"python seg/mask_associate.py --source_path {P.D_REL} --model_path {P.M_REL}/3dgs_output "
            f"--resolution {RES} --mask_generator hqsam --eval", st, env=_seg_env())
        _sh(f"python tools/add_label_num_hqsam.py --source_path {P.D_REL} --resolution {RES} --mask_generator hqsam",
            st, env=_seg_env())
        _export(st, [(P.D, "", ["associated_hqsam/*", "associated_hqsam_color/*", f"images_{RES}_num/*"], [])])
    _finish(3, st)


@app.function(image=main_image, gpu=RUN_GPU, memory=32768, volumes=VOLUMES, timeout=4 * 3600, retries=RETRIES)
def step4_distill_id(run: str, scene: str, force: bool = False):
    """run_seg.sh 5-6: ID distillation (2000 it) + render.py --render_video."""
    _setup(run, scene)
    st = _begin(4, force)
    if st is None:
        return
    with PeriodicCommit():
        _sh(f"python seg/distillation.py --source_path {P.D_REL} --model_path {P.M_REL} "
            f"--vanilla_3dgs_path {P.M_REL}/3dgs_output --resolution {RES} --object_path associated_hqsam --eval",
            st, env=_seg_env())
        _sh(f"python render.py -m {P.M_REL} --render_video", st, env=_seg_env())
        _export(st, [(P.M, "", None, ["3dgs_output/*", "*chkpnt*.pth", "*/ours_*/depth/*"])])
    _finish(4, st)


@app.function(image=main_image, cpu=2, volumes=VOLUMES, timeout=1800)
def pick_target_id(run: str, scene: str, frame: str, x: int, y: int) -> dict:
    """Inpaint360GS association ID under the click (5x5 majority, ignoring 0) in associated_hqsam/<frame>.png."""
    import numpy as np
    from PIL import Image
    _setup(run, scene)
    m = np.array(Image.open(f"{P.D}/associated_hqsam/{frame}.png"))
    win = m[max(0, y - 2):y + 3, max(0, x - 2):x + 3].ravel()
    win = win[win > 0]
    if win.size == 0:
        raise RuntimeError(f"no object under click ({x},{y}) on {frame}")
    tid = int(np.bincount(win).argmax())
    frames = sorted(Path(f"{P.D}/associated_hqsam").glob("*.png"))
    areas = [float((np.array(Image.open(f)) == tid).mean()) for f in frames]
    vis = [a for a in areas if a > 0]
    info = {"target_id": tid, "source": "click", "click": [frame, x, y], "mask_shape": list(m.shape),
            "frames_with_id": len(vis), "frames_total": len(frames),
            "mean_area_pct_visible": round(100 * float(np.mean(vis)), 2) if vis else 0.0}
    Path(P.STATE).mkdir(parents=True, exist_ok=True)
    (Path(P.STATE) / "target_id.json").write_text(json.dumps(info, indent=2))
    vol.commit()
    print(f"[id] {info}")
    if vis and np.mean(vis) > MAX_TARGET_AREA:
        raise RuntimeError(f"id {tid} covers {100 * np.mean(vis):.1f}% of the frames: background, check the click")
    return info


@app.function(image=main_image, cpu=1, volumes=VOLUMES, timeout=600)
def record_target_id(run: str, scene: str, target_id: int) -> None:
    _bind(run, scene)
    vol.reload()
    Path(P.STATE).mkdir(parents=True, exist_ok=True)
    (Path(P.STATE) / "target_id.json").write_text(json.dumps({"target_id": target_id, "source": "--target-id"}, indent=2))
    vol.commit()


@app.function(image=main_image, gpu=RUN_GPU, memory=32768, volumes=VOLUMES, timeout=4 * 3600, retries=RETRIES)
def step5_removal_virtual_cam(run: str, scene: str, target_id: int, surrounding_id: str = "None",
                              force: bool = False):
    """run_remove.sh 1-3: config, object removal, virtual camera trajectory (automatic circle_radius)."""
    _setup(run, scene)
    st = _begin(5, force)
    if st is None:
        return
    cfg = f"config/object_removal/{DATASET}/{scene}.json"
    with PeriodicCommit():
        _sh(f"python tools/init_configs.py --dataset_name {DATASET} --scene {scene} --target_id {target_id} "
            f"--target_surronding_id {surrounding_id}", st, env=_repo_env())
        _sh(f"python edit_object_removal.py --source_path {P.D_REL} -m {P.M_REL} --config_file {cfg} --render_video",
            st, env=_repo_env())
        _sh(f"python tools/virtual_pose.py -s {P.D_REL} -m {P.M_REL} --config_file {cfg}", st, env=_repo_env())
        _export(st, [(P.M, "", None, ["*chkpnt*.pth", "3dgs_output/*"]), (f"{P.R}/config", "config", None, []),
                     (f"{P.R}/samt/assets", "samt_assets", ["images.zip"], [])])
    _finish(5, st, target_id=target_id, surrounding_id=surrounding_id, samt="pending", circle_radius="auto")


@app.function(image=main_image, gpu=RUN_GPU, memory=32768, volumes=VOLUMES, timeout=3600, retries=RETRIES)
def samt_auto(run: str, scene: str, target_id: int):
    """run_remove.sh 4 without the browser (= modal_inpaint360gs_object.py::samt_auto): SAM-Track click at the
    target-id centroid of the first view where it is visible + automatic refine + off-object gating."""
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    _setup(run, scene)
    st5 = _load_state(5)
    if not st5 or st5["status"] != "done":
        raise RuntimeError("step 5 is not done")
    os.chdir(SAMT)
    sys.path.insert(0, SAMT)
    from model_args import aot_args, sam_args, segtracker_args
    from SegTracker import SegTracker
    from seg_track_anything import aot_model2ckpt, draw_mask, img_seq_type_input_tracking, save_prediction

    with PeriodicCommit():
        img_dir = Path(SAMT) / "assets" / "images"
        shutil.rmtree(img_dir, ignore_errors=True)
        subprocess.run(["unzip", "-q", str(Path(SAMT) / "assets" / "images.zip"), "-d", str(img_dir)], check=True)
        imgs_path = sorted(str(p) for p in img_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        objects_pred = Path(P.M) / "virtual" / "ours_2000" / "objects_pred"

        def obj_mask(i):
            return cv2.imread(str(objects_pred / Path(imgs_path[i]).name), -1) == target_id

        start = next((i for i in range(len(imgs_path)) if obj_mask(i).sum() >= 50), None)
        if start is None:
            raise RuntimeError(f"id {target_id} not visible in any virtual view")
        first = cv2.cvtColor(cv2.imread(imgs_path[start]), cv2.COLOR_BGR2RGB)
        n, lab, stats, cents = cv2.connectedComponentsWithStats(obj_mask(start).astype(np.uint8))
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y = int(round(cents[k][0])), int(round(cents[k][1]))
        if lab[y, x] != k:
            ys, xs = np.nonzero(lab == k)
            j = int(np.argmin((xs - x) ** 2 + (ys - y) ** 2))
            x, y = int(xs[j]), int(ys[j])
        print(f"[samt-auto] click (x={x}, y={y}) on view {start}, id {target_id} area {stats[k][4]} px")

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

        video_name = "images"
        result_dir = f"{SAMT}/tracking_results/{video_name}"
        os.makedirs(result_dir, exist_ok=True)
        io_args = {"tracking_result_dir": result_dir, "output_mask_dir": f"{result_dir}/{video_name}_masks",
                   "output_masked_frame_dir": f"{result_dir}/{video_name}_masked_frames",
                   "output_video": f"{result_dir}/{video_name}_seg.mp4", "output_gif": f"{result_dir}/{video_name}_seg.gif"}
        masks_dir = io_args["output_mask_dir"]
        if start > 0:
            for d in (masks_dir, io_args["output_masked_frame_dir"]):
                shutil.rmtree(d, ignore_errors=True)
                os.makedirs(d)
            h, w = first.shape[:2]
            for i in range(start):
                save_prediction(np.zeros((h, w), np.uint8), masks_dir, Path(imgs_path[i]).stem + ".png")
                shutil.copy2(imgs_path[i], os.path.join(io_args["output_masked_frame_dir"], Path(imgs_path[i]).name))
        img_seq_type_input_tracking(tracker, io_args, video_name, imgs_path, 8, start)
        raw_copy = Path(SAMT) / "tracking_results" / f"{video_name}_masks_raw"
        shutil.rmtree(raw_copy, ignore_errors=True)
        shutil.copytree(masks_dir, raw_copy)

        def track_mask(i):
            return np.array(Image.open(os.path.join(masks_dir, Path(imgs_path[i]).stem + ".png")).convert("P"))

        refined = []
        for _ in range(3):
            bad = None
            for i in range(start + 1, len(imgs_path)):
                o = obj_mask(i)
                if o.sum() >= 200 and ((track_mask(i) > 0) & o).sum() < 0.3 * o.sum():
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

        kernel = np.ones((31, 31), np.uint8)
        gated, clamped, fallback = [], [], []
        for i, p in enumerate(imgs_path):
            m = track_mask(i)
            o = obj_mask(i)
            near = cv2.dilate(o.astype(np.uint8), kernel) > 0
            n, lab = cv2.connectedComponents((m > 0).astype(np.uint8))
            keep = np.zeros_like(m, dtype=bool)
            for c in range(1, n):
                if (near & (lab == c)).any():
                    keep |= lab == c
            if (keep & ~near).any():
                clamped.append([i, int((keep & ~near).sum())])
                keep &= near
            if o.sum() >= 200 and (keep & o).sum() < 0.3 * o.sum():
                fallback.append(i)
                keep = cv2.dilate(o.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
                m = np.where(keep, 1, 0).astype(m.dtype)
            if (m > 0).sum() != keep.sum() or i in fallback:
                gated.append([i, int((m > 0).sum() - keep.sum())])
                save_prediction(np.where(keep, m, 0).astype(m.dtype), masks_dir, Path(p).stem + ".png")
        print(f"[samt-auto] refined at {refined}; gated {gated}; clamped {clamped}; fallback {fallback}")

        n_masks = len([f for f in os.listdir(masks_dir) if f.endswith(".png") and not f.endswith("_new.png")])
        areas = [float((cv2.imread(os.path.join(masks_dir, Path(p).stem + ".png"), 0) > 0).mean()) for p in imgs_path]
        print(f"[samt-auto] masks {n_masks}/{len(imgs_path)}; area % min {100 * min(areas):.2f} "
              f"mean {100 * np.mean(areas):.2f} max {100 * max(areas):.2f}")
        if n_masks < len(imgs_path):
            raise RuntimeError(f"only {n_masks}/{len(imgs_path)} masks")
        dst = Path(P.EXPORT) / st5["export"] / "samt_masks"
        overlay = dst.parent / "samt_masks_overlay"
        shutil.rmtree(dst, ignore_errors=True)
        shutil.rmtree(overlay, ignore_errors=True)
        overlay.mkdir(parents=True)
        for i, p in enumerate(imgs_path):
            frame = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            vis = cv2.cvtColor(draw_mask(frame, track_mask(i)), cv2.COLOR_RGB2BGR)
            cs, _ = cv2.findContours(obj_mask(i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cs, -1, (0, 0, 255), 1)
            cv2.imwrite(str(overlay / Path(p).name), vis)
        shutil.copytree(masks_dir, dst)
        raw_dst = dst.parent / "samt_masks_raw_track"
        shutil.rmtree(raw_dst, ignore_errors=True)
        shutil.copytree(raw_copy, raw_dst)
        st5.update(samt="done", samt_mode="auto: click at target-id centroid + auto refine + off-object gating + clamp",
                   samt_start_view=start, samt_click=[x, y], samt_refine=refined, samt_gated_frames=gated,
                   samt_clamped_frames=clamped, samt_fallback_frames=fallback,
                   samt_masks=n_masks, virtual_views=len(imgs_path),
                   samt_area_pct=[round(100 * min(areas), 2), round(100 * max(areas), 2)])
        _save_state(5, st5)
        (Path(P.EXPORT) / st5["export"] / "state.json").write_text(json.dumps(st5, indent=2, ensure_ascii=False))


@app.function(image=main_image, cpu=4, memory=16384, volumes=VOLUMES, timeout=3600, retries=RETRIES)
def step6_lama_prepare(run: str, scene: str, force: bool = False):
    _setup(run, scene)
    st5 = _load_state(5)
    if not st5 or st5.get("samt") != "done":
        raise RuntimeError("SAM-Track masks missing")
    st = _begin(6, force)
    if st is None:
        return False
    if "prepare" not in st.get("parts", []):
        with PeriodicCommit():
            _sh(f"python tools/prepare_lama_data.py -s {P.D_REL} -m {P.M_REL} -r {RES} --inpaint2lama", st,
                env=_repo_env())
        st.setdefault("parts", []).append("prepare")
        _save_state(6, st)
        _commit("lama prepare")
    return True


@app.function(image=lama_image, gpu=RUN_GPU, volumes=VOLUMES, timeout=4 * 3600, retries=RETRIES)
def step6_lama_predict(run: str, scene: str):
    _bind(run, scene)
    vol.reload()
    r = f"runs/{run}"
    _link_volume({"LaMa/data": f"{r}/lama/data", "LaMa/output": f"{r}/lama/output"})
    big = Path(f"{REPO}/LaMa/big-lama")
    if not big.exists():
        big.symlink_to("/opt/weights/big-lama", target_is_directory=True)
    st = _load_state(6)
    lama = f"{REPO}/LaMa"
    with PeriodicCommit():
        for kind in ("color", "depth"):
            if kind in st.get("parts", []):
                continue
            _sh(f"python bin/predict_{kind}.py --data_name 360_{scene}_virtual", st, cwd=lama,
                env={"TORCH_HOME": lama, "PYTHONPATH": lama})
            st.setdefault("parts", []).append(kind)
            _save_state(6, st)
            _commit(f"lama {kind}")


@app.function(image=main_image, cpu=4, memory=16384, volumes=VOLUMES, timeout=3600, retries=RETRIES)
def step6_lama_collect(run: str, scene: str):
    _setup(run, scene)
    st = _load_state(6)
    with PeriodicCommit():
        _sh(f"python tools/prepare_lama_data.py -s {P.D_REL} -m {P.M_REL} -r {RES}", st, env=_repo_env())
        _export(st, [(P.D, "", ["images_inpaint_unseen_virtual/*", "inpaint_2d_unseen_mask_virtual/*"], []),
                     (f"{P.R}/lama", "lama", None, []),
                     (P.M, "", ["virtual/*depth_completed/*"], [])])
    _finish(6, st)


# = modal_inpaint360gs_object.py SKIP_EMPTY_WRAPPER (user decision 2026-10-26): finetune_inpaint only on the virtual
# views whose inpainting mask is non-empty; init cloud falls back to the used view with the most points.
SKIP_EMPTY_WRAPPER = r'''
import ast, json, os, sys
sys.path.insert(0, os.getcwd())
import edit_object_inpaint as eoi

_orig = eoi.finetune_inpaint

def finetune_inpaint(args, opt, model_path, iteration, views, *rest, **kw):
    keep = [v for v in views if bool((v.objects > 128).any())]
    skipped = [v.image_name for v in views if not bool((v.objects > 128).any())]
    print(f"[skip-empty] training on {len(keep)}/{len(views)} virtual views; skipped (empty mask): {skipped}", flush=True)
    def _nv(path):
        with open(path, "rb") as fh:
            for line in fh:
                if line.startswith(b"element vertex"):
                    return int(line.split()[-1])
                if line.startswith(b"end_header"):
                    return 0
        return 0
    fdir = os.path.dirname(args.supp_ply)
    counts = {v.image_name: _nv(os.path.join(fdir, v.image_name + ".ply")) for v in keep
              if os.path.exists(os.path.join(fdir, v.image_name + ".ply"))}
    default_n = _nv(args.supp_ply) if os.path.exists(args.supp_ply) else 0
    if default_n < 1000 and counts:
        best = max(counts, key=counts.get)
        print(f"[skip-empty] init cloud {os.path.basename(args.supp_ply)} has {default_n} points -> using {best}.ply "
              f"({counts[best]} points)", flush=True)
        args.supp_ply = os.path.join(fdir, best + ".ply")
    with open(os.path.join(model_path, "skipped_virtual_views.json"), "w") as f:
        json.dump({"used": [v.image_name for v in keep], "skipped": skipped, "init_ply": os.path.basename(args.supp_ply),
                   "default_init_points": default_n, "points_per_view": counts}, f, indent=2)
    return _orig(args, opt, model_path, iteration, keep, *rest, **kw)

eoi.finetune_inpaint = finetune_inpaint
src = open(eoi.__file__, encoding="utf-8").read()
main = [n for n in ast.parse(src).body if isinstance(n, ast.If) and "__main__" in ast.unparse(n.test)][0]
eoi.__dict__["__name__"] = "__main__"
exec(compile(ast.Module(body=main.body, type_ignores=[]), eoi.__file__, "exec"), eoi.__dict__)
'''


@app.function(image=main_image, gpu=BIG_GPU, memory=32768, volumes=VOLUMES, timeout=6 * 3600, retries=RETRIES)
def step7_inpaint3d(run: str, scene: str, force: bool = False):
    """run_inpaint.sh 2-3: colourful point cloud fusion + missing-area optimization (--render_video)."""
    _setup(run, scene)
    st = _begin(7, force)
    if st is None:
        return
    with PeriodicCommit():
        _sh(f"python edit_object_removal_plyfusion.py -s {P.D_REL} -m {P.M_REL} "
            f"--config_file config/object_removal/{DATASET}/{scene}.json", st, env=_repo_env())
        wrapper = Path(P.EXPORT) / st["export"] / "edit_object_inpaint_skip_empty.py"
        wrapper.write_text(SKIP_EMPTY_WRAPPER)
        _sh(f"python {wrapper} -s {P.D_REL} -m {P.M_REL} --config_file config/object_inpaint/{DATASET}/{scene}.json "
            f"--resolution {RES} --render_video", st, env=_repo_env())
        _export(st, [(P.M, "", None, ["train/*", "test/*", "inpaint/*", "video/*", "*chkpnt*.pth", "3dgs_output/*"])])
    _finish(7, st, skip_empty_views=True)


@app.function(image=main_image, gpu=BIG_GPU, memory=32768, volumes=VOLUMES, timeout=2 * 3600, retries=RETRIES)
def step8_render_metrics(run: str, scene: str, force: bool = False):
    """run_inpaint.sh 4: renders of step 7 + tools/metrics_fid_masked.py (inpaint views vs object-free GT)."""
    _setup(run, scene)
    st7 = _load_state(7)
    if not st7 or st7["status"] != "done":
        raise RuntimeError("step 7 is not done")
    st = _begin(8, force)
    if st is None:
        return
    with PeriodicCommit():
        _sh(f"python tools/metrics_fid_masked.py -m {P.M_REL}", st, env=_repo_env())
        _export(st, [(P.M, "", ["train/*", "test/*", "inpaint/*", "video/*"], [])], t0=st7["t0"])
        _export(st, [(P.M, "", ["inpaint/*.json", "inpaint/**/*.json"], []),
                     (f"{P.R}/output/{DATASET}", "", ["all_scene_evaluation_results.json"], [])])
    res = Path(P.R) / "output" / DATASET / "all_scene_evaluation_results.json"
    _finish(8, st, from_step7=st7["export"], metrics=json.loads(res.read_text()) if res.exists() else None)


@app.function(image=main_image, cpu=1, volumes=VOLUMES, timeout=24 * 3600)
def chain(run: str, scene: str, target_id: int = 0, surrounding_id: str = "None", click: list | None = None):
    """steps 1 -> 8 entirely on Modal (local machine may sleep). Finished steps are skipped (resume)."""
    _bind(run, scene)
    print(fetch_weight.remote())
    vol.reload()

    def todo(step):  # skip finished steps without asking for a GPU container
        st = _load_state(step)
        if st and st["status"] == "done":
            print(f"[chain] step {step} already done -> {st['export']}")
            return False
        return True

    for s, fn in ((1, step1_train_3dgs), (2, step2_mask2d), (3, step3_sync_mask)):
        if todo(s):
            fn.remote(run, scene)
    if target_id <= 0:
        frame, x, y = click
        target_id = pick_target_id.remote(run, scene, frame, int(x), int(y))["target_id"]
    else:
        record_target_id.remote(run, scene, target_id)
    print(f"[chain] target_id = {target_id}")
    if todo(4):
        step4_distill_id.remote(run, scene)
    if todo(5):
        step5_removal_virtual_cam.remote(run, scene, target_id, surrounding_id)
    vol.reload()
    if (_load_state(5) or {}).get("samt") != "done":
        samt_auto.remote(run, scene, target_id)
    if step6_lama_prepare.remote(run, scene):
        step6_lama_predict.remote(run, scene)
        step6_lama_collect.remote(run, scene)
    if todo(7):
        step7_inpaint3d.remote(run, scene)
    if todo(8):
        step8_render_metrics.remote(run, scene)
    vol.reload()
    return {s: (_load_state(s) or {}).get("status") for s in STEP_NAMES}


# --------------------------------------------------------------------------------------------------- local entrypoints
def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


@app.local_entrypoint()
def upload(run: str = "redbull_main", scene: str = "redbull"):
    """Step 0 reuse: ours/<run>/step0_colmap (images + sparse/0) + released images_2 / unseen_mask -> volume."""
    s0 = OURS_ROOT / run / "step0_colmap"
    rel = DATA_ROOT / scene
    imgs = sorted(p for p in (s0 / "images").iterdir() if p.is_file())
    imgs2 = sorted(p for p in (rel / f"images_{RES}").iterdir() if p.is_file())
    if {p.name for p in imgs} != {p.name for p in imgs2}:
        raise SystemExit("step0_colmap/images and released images_2 have different file names")
    check = {f.name: _md5(f) == _md5(rel / "sparse" / "0" / f.name) for f in sorted((s0 / "sparse" / "0").iterdir())}
    summary = json.loads((s0 / "summary.json").read_text())
    if not summary.get("check_ok") or summary["cameras"]["1"]["model"] != "PINHOLE":
        raise SystemExit(f"step0 summary not usable: {summary}")
    r = f"/runs/{run}/data/{DATASET}/{scene}"
    with vol.batch_upload(force=True) as b:
        b.put_directory(str(s0 / "images"), f"{r}/images")
        b.put_directory(str(s0 / "sparse" / "0"), f"{r}/sparse/0")
        b.put_directory(str(rel / f"images_{RES}"), f"{r}/images_{RES}")
        b.put_directory(str(rel / "unseen_mask"), f"{r}/unseen_mask")
    info = {"reused_from": str(s0), "same_as_released_sparse_md5": check, "colmap_summary": summary,
            "images": len(imgs), "test_images": sum(p.name.startswith("test_") for p in imgs),
            f"images_{RES}_from": str(rel / f"images_{RES}"), "unseen_mask_from": str(rel / "unseen_mask"),
            "volume": f"inpaint360gs-baseline:{r}", "uploaded": time.strftime("%Y-%m-%d %H:%M:%S")}
    local = OUT_ROOT / run / "step0_colmap"
    local.mkdir(parents=True, exist_ok=True)
    (local / "reused_from.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


@app.local_entrypoint()
def run(run: str = "redbull_main", scene: str = "redbull", target_id: int = 0, surrounding_id: str = "None",
        click: str = ""):
    """Full chain (steps 1-8). click = "FRAME,X,Y" in images_2 pixels (default: ours 6id click of the scene)."""
    c = click.split(",") if click else list(DEFAULT_CLICK.get(scene, ()))
    if target_id <= 0 and len(c) != 3:
        raise SystemExit("give --target-id or --click FRAME,X,Y")
    print(json.dumps(chain.remote(run, scene, target_id, surrounding_id, c), indent=2))


@app.local_entrypoint()
def status(run: str = "redbull_main"):
    try:
        print("target_id:", json.loads(b"".join(vol.read_file(f"runs/{run}/state/target_id.json"))))
    except Exception:  # noqa: BLE001
        print("target_id: -")
    for s in STEP_NAMES:
        try:
            st = json.loads(b"".join(vol.read_file(f"runs/{run}/state/step{s}.json")))
            print(f"step {s}: {st['status']:8s} {st['export']:26s} attempts={st['attempts']} "
                  f"{ {k: v for k, v in st.items() if k in ('target_id', 'samt', 'samt_area_pct', 'metrics')} }")
        except Exception:  # noqa: BLE001
            print(f"step {s}: -")


@app.local_entrypoint()
def download(run: str = "redbull_main", overwrite: bool = False):
    """runs/<run>/export (stepN_* + logs) + state -> outputs/Inpaint360GS/Inpaint360GS/<run>/ (skip existing files)."""
    local_root = OUT_ROOT / run
    n = 0
    for remote, local in ((f"runs/{run}/export", local_root), (f"runs/{run}/state", local_root / "state")):
        for e in vol.listdir(remote, recursive=True):
            if e.type != modal.volume.FileEntryType.FILE:
                continue
            rel = e.path.strip("/")[len(remote) + 1:]
            dst = local / rel
            if dst.exists() and not overwrite:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".part")
            with open(tmp, "wb") as fh:
                for chunk in vol.read_file(e.path):
                    fh.write(chunk)
            os.replace(tmp, dst)
            n += 1
    print(f"[download] {n} files -> {local_root}")
