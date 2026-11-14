"""
Inpaint360GS on LERF teatime for ANOTHER object (steps 5-8), reusing the scene-level steps 0-4 of teatime_napkin
=================================================================================================================
Steps 0-4 (COLMAP, 3DGS, HQ-SAM masks, ID sync, ID distillation) depend only on the scene, so they are copied from
outputs/LERF/Inpaint360GS/teatime_napkin. Only steps 5-8 (removal + virtual cameras -> SAM-Track masks -> LaMa ->
3D inpaint -> renders) are run here, with exactly the same original commands as modal_inpaint360gs.py (run_remove.sh,
run_inpaint.sh, automatic SAM-Track `samt_auto`, skip-empty-view wrapper of step 7).

Why a rebuild of the Modal data: the napkin intermediates on workspace nguyenkhacd10 are gone (volume recreated
2026-11-09), so the scene data/model are rebuilt from the local napkin outputs (pixel-exact):
    data/others/teatime/images            <- step8_render/{train,test}/0_gt  (177 PNG 494x365 = the images_2 tensors
                                             the original run loaded: 988x730 undistorted, resolution 2, ceil(988/2))
    data/others/teatime/associated_hqsam  <- step3_sync_mask/masks_synced (+ scene.json)
    data/others/teatime/sparse/0          <- step0_colmap/sparse/0
    output/others/teatime/point_cloud/iteration_2000/{point_cloud.ply, classifier.pth} <- step4_id_3dgs
    output/others/teatime/cfg_args        <- step4_id_3dgs/cfg_args with resolution=1 (images are already 1/2 size)
and every `--resolution/-r 2` of the original scripts becomes 1 (same tensors: ceil(494/1) = ceil(988/2)).

Volume `inpaint360gs-baseline` (current Modal profile), one folder per run: runs/<run>/{data, output, lama, samt,
config, export/stepN_*, state}. Images = the ones of modal_inpaint360gs.py (copied verbatim, so the build matches).

  $env:PYTHONIOENCODING="utf-8"                                         # from F:\\Safe-GS
  python -m modal run --detach Based_Model/Inpaint360GS/modal_inpaint360gs_object.py::run --run teatime_cafe_box --target-id 22
  python -m modal run Based_Model/Inpaint360GS/modal_inpaint360gs_object.py::status --run teatime_cafe_box
  python -m modal run Based_Model/Inpaint360GS/modal_inpaint360gs_object.py::download --run teatime_cafe_box
      -> outputs/LERF/Inpaint360GS/<run>/_modal_export/ (raw step exports + logs)
"""
from __future__ import annotations

import fnmatch
import io
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
GPU = "L4"
DATASET = "others"
SCENE = "teatime"
RES_ORIG = 2                 # resolution of the original napkin run
RES = 1                      # images uploaded at 1/2 size already
LOCAL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = LOCAL_ROOT.parents[1] if len(LOCAL_ROOT.parents) > 1 else LOCAL_ROOT
OUT_ROOT = PROJECT_ROOT / "outputs" / "LERF" / "Inpaint360GS"
REUSE_RUN = "teatime_napkin"

VOL = "/vol"
REPO = "/opt/inpaint360gs"
SAMT = f"{REPO}/Segment-and-Track-Anything"
CROPFORMER = f"{REPO}/seg/detectron2/detectron2/projects/CropFormer"
D_REL = f"data/{DATASET}/{SCENE}"
M_REL = f"output/{DATASET}/{SCENE}"
LAMA_DATA_NAME = f"360_{SCENE}_virtual"
STEP_NAMES = {5: "virtual_cam", 6: "inpaint2d", 7: "inpaint_3dgs", 8: "render"}
COMMIT_EVERY_S = 300

app = modal.App("inpaint360gs-object")
vol = modal.Volume.from_name("inpaint360gs-baseline", create_if_missing=True)
VOLUMES = {VOL: vol}
RETRIES = modal.Retries(max_retries=2, backoff_coefficient=1.0, initial_delay=30.0)
BASE_ENV = {"PYTHONUNBUFFERED": "1", "QT_QPA_PLATFORM": "offscreen", "MPLBACKEND": "Agg", "WANDB_MODE": "disabled"}

# --------------------------------------------------------------------------------------------------- images
# verbatim from modal_inpaint360gs.py (+ this file in the ignore list so editing it never rebuilds the image)
REPO_IGNORE = [
    "outputs", "outputs/**", "data", "data/**", "output", "output/**", "LaMa", "LaMa/**",
    "Paper", "Paper/**", "media", "media/**", "plan.md", "modal_inpaint360gs.py", "modal_inpaint360gs_object.py",
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
    run = R = D = M = EXPORT = STATE = ""


def _bind(run: str) -> None:
    P.run = run
    P.R = f"{VOL}/runs/{run}"
    P.D = f"{P.R}/{D_REL}"
    P.M = f"{P.R}/{M_REL}"
    P.EXPORT = f"{P.R}/export"
    P.STATE = f"{P.R}/state"


def _links() -> dict[str, str]:
    """repo path -> volume path (relative to /vol): everything the original code writes lands on the volume."""
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


def _setup(run: str) -> None:
    _bind(run)
    vol.reload()
    _link_volume(_links())


def _commit(reason: str = "") -> None:
    try:
        vol.commit()
        print(f"[volume] committed {reason}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[volume] commit failed ({reason}): {e}", flush=True)


class PeriodicCommit:
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


def _repo_env() -> dict:
    return {"PYTHONPATH": REPO + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")}


# --------------------------------------------------------------------------------------------------- steps 5-8
@app.function(image=main_image, gpu=GPU, volumes=VOLUMES, timeout=4 * 3600, retries=RETRIES)
def step5_removal_virtual_cam(run: str, target_id: int, force: bool = False):
    """run_remove.sh 1-3: config, object removal, virtual camera trajectory (code's automatic circle_radius)."""
    _setup(run)
    st = _begin(5, force)
    if st is None:
        return
    cfg = f"config/object_removal/{DATASET}/{SCENE}.json"
    with PeriodicCommit():
        _sh(f"python tools/init_configs.py --dataset_name {DATASET} --scene {SCENE} --target_id {target_id} "
            f"--target_surronding_id None", st, env=_repo_env())
        # edit_object_removal.py reads <model>/train|test/ours_<it>/depth/*.npy written by step 4's render.py
        # (run_seg.sh 6); only the step-4 images were kept locally, so re-render the original model once here.
        if not any(Path(P.M).glob("train/ours_*/depth/*.npy")):
            _sh(f"python render.py -s {D_REL} -m {M_REL}", st, env=_repo_env())
        _sh(f"python edit_object_removal.py --source_path {D_REL} -m {M_REL} --config_file {cfg} --render_video",
            st, env=_repo_env())
        _sh(f"python tools/virtual_pose.py -s {D_REL} -m {M_REL} --config_file {cfg}", st, env=_repo_env())
        _export(st, [(P.M, "", None, ["*chkpnt*.pth"]), (f"{P.R}/config", "config", None, []),
                     (f"{P.R}/samt/assets", "samt_assets", ["images.zip"], [])])
    _finish(5, st, target_id=target_id, surrounding_id="None", samt="pending", circle_radius="auto")


@app.function(image=main_image, gpu=GPU, volumes=VOLUMES, timeout=3600, retries=RETRIES)
def samt_auto(run: str, target_id: int):
    """run_remove.sh 4 without the browser (same as modal_inpaint360gs.py::samt_auto): SAM-Track click at the
    target-id centroid of virtual view 0 + automatic refine + removal of off-object mask fragments."""
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    _setup(run)
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

        # first view where the object is visible (napkin: view 0)
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
        # app.py tracks from frame 0; when the object is hidden in the first views, start where it is visible and
        # write empty masks before it (nothing to inpaint there)
        # (img_seq_type_input_tracking with frame_num > 0 reads the masks / masked frames of the earlier frames)
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
            # a drifted track can be ONE component spreading over the table that still touches the object
            # (cafe_box views 18-23: 48 % of the frame) -> also clamp pixel-wise to the dilated target-id region
            if (keep & ~near).any():
                clamped.append([i, int((keep & ~near).sum())])
                keep &= near
            # track lost the object entirely -> use the (dilated) rendered target-id region itself
            if o.sum() >= 200 and (keep & o).sum() < 0.3 * o.sum():
                fallback.append(i)
                keep = cv2.dilate(o.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
                m = np.where(keep, 1, 0).astype(m.dtype)
            if (m > 0).sum() != keep.sum() or i in fallback:
                gated.append([i, int((m > 0).sum() - keep.sum())])
                save_prediction(np.where(keep, m, 0).astype(m.dtype), masks_dir, Path(p).stem + ".png")
        print(f"[samt-auto] refined at {refined}; removed off-object pixels in frames {gated}; "
              f"clamped to object region {clamped}; object-region fallback {fallback}")

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
        st5.update(samt="done", samt_mode="auto: click at target-id centroid + auto refine + off-object gating + clamp to dilated target region",
                   samt_start_view=start, samt_click=[x, y], samt_refine=refined, samt_gated_frames=gated,
                   samt_clamped_frames=clamped, samt_fallback_frames=fallback,
                   samt_masks=n_masks, virtual_views=len(imgs_path),
                   samt_area_pct=[round(100 * min(areas), 2), round(100 * max(areas), 2)])
        _save_state(5, st5)
        (Path(P.EXPORT) / st5["export"] / "state.json").write_text(json.dumps(st5, indent=2, ensure_ascii=False))


@app.function(image=main_image, cpu=2, volumes=VOLUMES, timeout=3600, retries=RETRIES)
def step6_lama_prepare(run: str, force: bool = False):
    _setup(run)
    st5 = _load_state(5)
    if not st5 or st5.get("samt") != "done":
        raise RuntimeError("SAM-Track masks missing")
    st = _begin(6, force)
    if st is None:
        return False
    if "prepare" not in st.get("parts", []):
        with PeriodicCommit():
            _sh(f"python tools/prepare_lama_data.py -s {D_REL} -m {M_REL} -r {RES} --inpaint2lama", st, env=_repo_env())
        st.setdefault("parts", []).append("prepare")
        _save_state(6, st)
        _commit("lama prepare")
    return True


@app.function(image=lama_image, gpu=GPU, volumes=VOLUMES, timeout=4 * 3600, retries=RETRIES)
def step6_lama_predict(run: str):
    _bind(run)
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
            _sh(f"python bin/predict_{kind}.py --data_name {LAMA_DATA_NAME}", st, cwd=lama,
                env={"TORCH_HOME": lama, "PYTHONPATH": lama})
            st.setdefault("parts", []).append(kind)
            _save_state(6, st)
            _commit(f"lama {kind}")


@app.function(image=main_image, cpu=2, volumes=VOLUMES, timeout=3600, retries=RETRIES)
def step6_lama_collect(run: str):
    _setup(run)
    st = _load_state(6)
    with PeriodicCommit():
        _sh(f"python tools/prepare_lama_data.py -s {D_REL} -m {M_REL} -r {RES}", st, env=_repo_env())
        _export(st, [(P.D, "", ["images_inpaint_unseen_virtual/*", "inpaint_2d_unseen_mask_virtual/*"], []),
                     (f"{P.R}/lama", "lama", None, []),
                     (P.M, "", ["virtual/*depth_completed/*"], [])])
    _finish(6, st)


# modal_inpaint360gs.py SKIP_EMPTY_WRAPPER (user decision 2026-10-26): finetune_inpaint only on non-empty-mask views
SKIP_EMPTY_WRAPPER = r'''
import ast, json, os, sys
sys.path.insert(0, os.getcwd())
import edit_object_inpaint as eoi

_orig = eoi.finetune_inpaint

def finetune_inpaint(args, opt, model_path, iteration, views, *rest, **kw):
    keep = [v for v in views if bool((v.objects > 128).any())]
    skipped = [v.image_name for v in views if not bool((v.objects > 128).any())]
    print(f"[skip-empty] training on {len(keep)}/{len(views)} virtual views; skipped (empty mask): {skipped}", flush=True)
    # the authors hard-code the init cloud to virtual view 00004 ("a better initialization could be selected ...
    # manually"); if that view's mask is empty (object not visible there) use the used view with the most points.
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


@app.function(image=main_image, gpu=GPU, volumes=VOLUMES, timeout=6 * 3600, retries=RETRIES)
def step7_inpaint3d(run: str, force: bool = False):
    _setup(run)
    st = _begin(7, force)
    if st is None:
        return
    with PeriodicCommit():
        _sh(f"python edit_object_removal_plyfusion.py -s {D_REL} -m {M_REL} "
            f"--config_file config/object_removal/{DATASET}/{SCENE}.json", st, env=_repo_env())
        wrapper = Path(P.EXPORT) / st["export"] / "edit_object_inpaint_skip_empty.py"
        wrapper.write_text(SKIP_EMPTY_WRAPPER)
        _sh(f"python {wrapper} -s {D_REL} -m {M_REL} --config_file config/object_inpaint/{DATASET}/{SCENE}.json "
            f"--resolution {RES} --render_video", st, env=_repo_env())
        _export(st, [(P.M, "", None, ["train/*", "test/*", "video/*", "*chkpnt*.pth"])])
    _finish(7, st, skip_empty_views=True)


@app.function(image=main_image, cpu=2, volumes=VOLUMES, timeout=3600, retries=RETRIES)
def step8_render(run: str, force: bool = False):
    _setup(run)
    st7 = _load_state(7)
    if not st7 or st7["status"] != "done":
        raise RuntimeError("step 7 is not done")
    st = _begin(8, force)
    if st is None:
        return
    _export(st, [(P.M, "", ["train/*", "test/*", "video/*"], [])], t0=st7["t0"])
    _finish(8, st, from_step7=st7["export"])


@app.function(image=main_image, cpu=1, volumes=VOLUMES, timeout=12 * 3600)
def chain(run: str, target_id: int, redo_samt: bool = False):
    """step 5 -> automatic SAM-Track masks -> step 6 -> step 7 -> step 8, entirely on Modal (local machine may sleep).
    redo_samt: keep step 5, recompute the SAM-Track masks and redo steps 6-8."""
    _bind(run)
    if redo_samt:
        vol.reload()
        st5 = _load_state(5)
        if st5:
            st5["samt"] = "pending"
            _save_state(5, st5)
        for s in (6, 7, 8):
            _state_path(s).unlink(missing_ok=True)
        _commit("redo samt: reset steps 6-8")
    step5_removal_virtual_cam.remote(run, target_id)
    vol.reload()
    if (_load_state(5) or {}).get("samt") != "done":
        samt_auto.remote(run, target_id)
    if step6_lama_prepare.remote(run):
        step6_lama_predict.remote(run)
        step6_lama_collect.remote(run)
    step7_inpaint3d.remote(run)
    step8_render.remote(run)
    vol.reload()
    return {s: (_load_state(s) or {}).get("status") for s in STEP_NAMES}


# --------------------------------------------------------------------------------------------------- local entrypoints
def _cfg_args_res1(text: str) -> str:
    if f"resolution={RES_ORIG}" not in text:
        raise SystemExit(f"cfg_args has no resolution={RES_ORIG}: {text}")
    return text.replace(f"resolution={RES_ORIG}", f"resolution={RES}")


@app.local_entrypoint()
def upload(run: str = "teatime_cafe_box"):
    _upload(run)


def _upload(run: str) -> None:
    """Rebuild data/ and output/ of the scene on the volume from the local napkin steps 0-4 (see module docstring)."""
    src = OUT_ROOT / REUSE_RUN
    gts = sorted((src / "step8_render").glob("*/0_gt/*.png"))
    masks = src / "step3_sync_mask" / "masks_synced"
    if len(gts) != 177 or not (masks / "scene.json").exists():
        raise SystemExit(f"expected 177 GT images and scene.json under {src} (got {len(gts)})")
    r = f"/runs/{run}"
    with vol.batch_upload(force=True) as b:
        for g in gts:
            b.put_file(str(g), f"{r}/{D_REL}/images/{g.name}")
        b.put_directory(str(masks), f"{r}/{D_REL}/associated_hqsam")
        b.put_directory(str(src / "step0_colmap" / "sparse" / "0"), f"{r}/{D_REL}/sparse/0")
        it = f"{r}/{M_REL}/point_cloud/iteration_2000"
        b.put_file(str(src / "step4_id_3dgs" / "point_cloud_iter2000.ply"), f"{it}/point_cloud.ply")
        b.put_file(str(src / "step4_id_3dgs" / "classifier.pth"), f"{it}/classifier.pth")
        cfg = _cfg_args_res1((src / "step4_id_3dgs" / "cfg_args").read_text())
        b.put_file(io.BytesIO(cfg.encode()), f"{r}/{M_REL}/cfg_args")
        b.put_file(str(src / "step1_3dgs" / "cameras.json"), f"{r}/{M_REL}/cameras.json")
    print(f"[upload] {len(gts)} images + masks + sparse + ID-3DGS (iteration_2000) -> inpaint360gs-baseline:{r}")


@app.local_entrypoint()
def run(run: str = "teatime_cafe_box", target_id: int = 22, skip_upload: bool = False, redo_samt: bool = False):
    if not skip_upload:
        _upload(run)
    print(json.dumps(chain.remote(run, target_id, redo_samt), indent=2))


@app.local_entrypoint()
def status(run: str = "teatime_cafe_box"):
    for s in STEP_NAMES:
        try:
            st = json.loads(b"".join(vol.read_file(f"runs/{run}/state/step{s}.json")))
            print(f"step {s}: {st['status']:8s} {st['export']:22s} attempts={st['attempts']} "
                  f"{ {k: v for k, v in st.items() if k in ('target_id', 'samt', 'samt_click', 'samt_area_pct')} }")
        except Exception:  # noqa: BLE001
            print(f"step {s}: -")


@app.local_entrypoint()
def download(run: str = "teatime_cafe_box", overwrite: bool = False):
    """runs/<run>/export (stepN_* + logs) -> outputs/LERF/Inpaint360GS/<run>/_modal_export/ (raw, never renamed)."""
    remote = f"runs/{run}/export"
    local = OUT_ROOT / run / "_modal_export"
    n = 0
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
    print(f"[download] {n} files {remote} -> {local}")
