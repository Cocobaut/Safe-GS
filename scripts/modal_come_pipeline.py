"""
Step 4-12 of Safe-GS on Modal: Inpaint360GS with the CoMe rasterizer (PLAN_REBUILD_BASE_FRAMEWORK.md, Phase 4)
==============================================================================================================
Rebuilt from the commands recorded in outputs/LERF/teatime_napkin/logs/*.log (the original runner was not kept).
Commands / exports / Volume layout: scripts/pipeline_steps.py. Code = src/ (copied into the image at /opt/safegs).

Parts (each resumable: state.json on the Volume, retries, Volume commit every 300 s and on preemption):
    prep  data + CoMe model of the scene on Volume `inpaint360gs` (from Volume `safe-gs-come` or local outputs)
    4     renderer check CoMe vs Inpaint360GS renderer                                L4/L40S
    5     HQ-SAM (CropFormer) 2D masks                                                L4/L40S
    6     mask association (3D-consistent IDs) + numbered images                     L4/L40S
    7     ID distillation (classifier) + render                                       L4/L40S
    8     per-run copy, init_configs (target ids), removal, virtual cameras         L4/L40S
    9     inpainting masks on the virtual views (SAM-Track click / fallback)          L4/L40S
    10a/l/b  LaMa data -> LaMa colour + depth (T4, conda env of the authors) -> back L4 / T4
    11    init view selection, ply fusion, 3DGS inpainting with CoMe loss            L4/L40S
    12    train/test renders + videos of the removed / inpainted scene               CPU (in GPU image)

  $env:PYTHONIOENCODING="utf-8"
  python -m modal run scripts/modal_come_pipeline.py::check                                   # build + imports (CPU)
  python -m modal run scripts/modal_come_pipeline.py::plan --dataset LERF --run teatime_napkin  # dry run (no GPU)
  python -m modal run scripts/modal_come_pipeline.py::upload_local --dataset LERF --run teatime_napkin
      # prep source = outputs/<Dataset>/<run>/step0*, step1*, step2* (or: --source volume, from safe-gs-come)
  python -m modal run --detach scripts/modal_come_pipeline.py::run --dataset LERF --run teatime_napkin --steps 4-12
  python -m modal run scripts/modal_come_pipeline.py::status --dataset LERF --run teatime_napkin
  python -m modal run scripts/modal_come_pipeline.py::download --dataset LERF --run teatime_napkin
      # come360/export/<Dataset>/<run>/ -> outputs/<Dataset>/<run>/ (existing files kept unless --overwrite)
Secrets: `huggingface` (HF_TOKEN, CropFormer weights are in a gated HF dataset, needed by Step 5 only) - name
overridable with SAFE_GS_HF_SECRET. Reusing the step 5-6 masks of another run (::upload_local --with-masks) does not
need it: set SAFE_GS_SKIP_CROPFORMER=1 (the image is then built without the CropFormer weights).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import modal

import modal_common as mc
import pipeline_steps as ps

mc.utf8_console()

app = modal.App("safe-gs-come-pipeline")
vol = modal.Volume.from_name("inpaint360gs", create_if_missing=True)
come_vol = modal.Volume.from_name("safe-gs-come", create_if_missing=True)
COME_VOL = "/vol_come"
VOLUMES = {ps.VOL: vol, COME_VOL: come_vol}
CODE, LAMA_CODE = ps.CODE, ps.LAMA_CODE
SRC = mc.SRC
HF_SECRET = os.environ.get("SAFE_GS_HF_SECRET", "huggingface")
CROPFORMER = ("https://huggingface.co/datasets/qqlu1992/Adobe_EntitySeg/resolve/main/CropFormer_model/"
              "Entity_Segmentation/CropFormer_hornet_3x/CropFormer_hornet_3x_03823a.pth")
BIG_LAMA = "https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip"
LAMA_COMMIT = "786f5936b27fb3dacd2b1ad799e4de968ea697e7"
SAM_VIT_B = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
DEAOT_GDRIVE = "1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ"           # R50_DeAOTL_PRE_YTB_DAV.pth (script/download_ckpt.sh)
RETRY = modal.Retries(max_retries=2, backoff_coefficient=1.0, initial_delay=30.0)
SAMT = f"{CODE}/Segment-and-Track-Anything"

_CODE_IGNORE = ["come", "come/**", "LaMa", "LaMa/**", "submodules", "submodules/**", "gaussian_splatting",
                "gaussian_splatting/**", "instantmesh", "instantmesh/**", "seg/detectron2", "seg/detectron2/**",
                "Segment-and-Track-Anything/sam", "Segment-and-Track-Anything/sam/**",
                "Segment-and-Track-Anything/Pytorch-Correlation-extension",
                "Segment-and-Track-Anything/Pytorch-Correlation-extension/**", "docs", "docs/**", "media", "media/**",
                "**/__pycache__", "**/__pycache__/**", "**/*.pyc", "**/*.so"]

# ---------------------------------------------------------------------------------------------------- images
_i360_base = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "build-essential", "ninja-build", "cmake", "wget", "unzip", "ffmpeg", "libgl1",
                 "libglib2.0-0", "libsm6", "libxext6", "libgomp1", "libegl1", "libx11-6", "libusb-1.0-0")
    .env({"CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "8.9", "FORCE_CUDA": "1", "MAX_JOBS": "8",
          "CC": "gcc", "CXX": "g++", "LDSHARED": "g++ -shared"})
    .pip_install("torch==2.4.1", "torchvision==0.19.1", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("numpy==1.26.4", "plyfile", "tqdm", "scipy", "opencv-python-headless", "scikit-learn", "lpips",
                 "torchmetrics", "imageio", "imageio-ffmpeg", "open3d==0.18.0", "wandb", "pytorch_fid", "pillow",
                 "matplotlib", "timm", "dacite", "pycocotools", "scikit-image", "gdown", "einops", "trimesh",
                 "wheel", "setuptools<70", "ninja")
    # CoMe rasterizer + simple-knn of the CoMe branch (src/submodules/*-come), cached layers
    .add_local_dir(str(SRC / "submodules" / "simple-knn-come"), "/opt/sub/simple-knn-come", copy=True, ignore=mc.JUNK)
    .add_local_dir(str(SRC / "submodules" / "diff-gaussian-rasterization-come"),
                   "/opt/sub/diff-gaussian-rasterization-come", copy=True, ignore=mc.JUNK)
    .run_commands("pip install --no-build-isolation /opt/sub/simple-knn-come",
                  "pip install --no-build-isolation /opt/sub/diff-gaussian-rasterization-come")
    # segmentation (src/install.sh step 5): SAM, detectron2 + CropFormer ops, fvcore patch, CropFormer weights
    .run_commands("pip install git+https://github.com/facebookresearch/segment-anything.git", "pip install setuptools==75.8.0")
    .add_local_dir(str(SRC / "seg" / "detectron2"), f"{CODE}/seg/detectron2", copy=True, ignore=mc.JUNK)
    .add_local_file(str(SRC / "seg" / "patch_fvcore.py"), f"{CODE}/seg/patch_fvcore.py", copy=True)
    .run_commands(
        f"cd {CODE}/seg && python -m pip install -e detectron2 --no-build-isolation",
        f"cd {CODE}/seg/detectron2/detectron2/projects/CropFormer/entity_api/PythonAPI && make",
        f"cd {CODE}/seg/detectron2/detectron2/projects/CropFormer/mask2former/modeling/pixel_decoder/ops && sh make.sh",
        f"cd {CODE} && python seg/patch_fvcore.py",
    )
    # CropFormer weights (gated HF dataset, Step 5 only) are added below unless SAFE_GS_SKIP_CROPFORMER=1
)
if not os.environ.get("SAFE_GS_SKIP_CROPFORMER"):
    _i360_base = _i360_base.run_commands(
        f'mkdir -p {CODE}/seg/weight && wget -q --header="Authorization: Bearer $HF_TOKEN" '
        f"-O {CODE}/seg/weight/CropFormer_hornet_3x_03823a.pth {CROPFORMER}",
        secrets=[modal.Secret.from_name(HF_SECRET)])
i360_image = (
    _i360_base
    # Segment-and-Track-Anything (script/install.sh without GroundingDINO / audio, not used headless)
    .add_local_dir(str(SRC / "Segment-and-Track-Anything" / "sam"), f"{SAMT}/sam", copy=True, ignore=mc.JUNK)
    .add_local_dir(str(SRC / "Segment-and-Track-Anything" / "Pytorch-Correlation-extension"),
                   "/opt/sub/Pytorch-Correlation-extension", copy=True, ignore=mc.JUNK)
    .run_commands(f"cd {SAMT}/sam && pip install -e . --no-build-isolation",
                  "cd /opt/sub/Pytorch-Correlation-extension && pip install . --no-build-isolation",
                  f"mkdir -p {SAMT}/ckpt && wget -q -P {SAMT}/ckpt {SAM_VIT_B}",
                  f"gdown {DEAOT_GDRIVE} -O {SAMT}/ckpt/R50_DeAOTL_PRE_YTB_DAV.pth && ls -la {SAMT}/ckpt")
    .env({**mc.BASE_ENV, "PYTHONPATH": CODE})
    .add_local_dir(str(SRC), CODE, copy=True, ignore=_CODE_IGNORE)        # code last: edits rebuild only this layer
    .add_local_python_source("modal_common", "pipeline_steps")
)

lama_image = (                                         # docs/install.md section 2 (authors' conda env), T4
    modal.Image.from_registry("continuumio/miniconda3:24.7.1-0")
    .apt_install("unzip", "curl", "git", "libgl1", "libglib2.0-0")
    .env({"CONDA_PLUGINS_AUTO_ACCEPT_TOS": "yes"})
    .add_local_file(str(SRC / "LaMa" / "conda_env.yml"), "/opt/conda_env.yml", copy=True)
    .run_commands("conda env create -f /opt/conda_env.yml",
                  "conda install -n lama pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y",
                  "conda run -n lama pip install pytorch-lightning==1.2.9")
    .run_commands(f"mkdir -p {LAMA_CODE} && cd {LAMA_CODE} && curl -sLJO {BIG_LAMA} && unzip -q big-lama.zip "
                  f"&& rm big-lama.zip")
    .add_local_dir(str(SRC / "LaMa"), LAMA_CODE, copy=True, ignore=["data", "data/**", "output", "output/**"] + mc.JUNK)
    # files missing from some copies of the authors' LaMa (cp -n = only add what is missing)
    .run_commands(f"git clone -q https://github.com/advimman/lama.git /tmp/lama && git -C /tmp/lama checkout -q "
                  f"{LAMA_COMMIT} && cp -rn /tmp/lama/saicinpainting/. {LAMA_CODE}/saicinpainting/ "
                  f"&& cp -rn /tmp/lama/configs/. {LAMA_CODE}/configs/ && rm -rf /tmp/lama")
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("modal_common", "pipeline_steps")
)
light_image = modal.Image.debian_slim(python_version="3.10").add_local_python_source("modal_common", "pipeline_steps")


# ---------------------------------------------------------------------------------------------------- container helpers
def _link(dst: str, src: str):
    """dst (in the code tree) -> src (on the Volume)."""
    s, d = Path(src), Path(dst)
    s.mkdir(parents=True, exist_ok=True)
    if d.is_symlink() or d.is_file():
        d.unlink()
    elif d.exists():
        for f in d.iterdir():                 # keep template files that ship with the code (e.g. none expected)
            t = s / f.name
            if not t.exists():
                shutil.move(str(f), str(t))
        shutil.rmtree(d)
    d.parent.mkdir(parents=True, exist_ok=True)
    d.symlink_to(s, target_is_directory=True)


def _copy(src: str, dst: Path) -> int:
    """Copy a file / folder ('/' suffix) / newest glob match to dst. Returns number of files."""
    if any(c in src for c in "*?["):
        cands = sorted(Path("/").glob(src.lstrip("/")), key=lambda p: p.stat().st_mtime)
        if not cands:
            return 0
        src = str(cands[-1]) + ("/" if cands[-1].is_dir() else "")
    s = Path(src.rstrip("/"))
    if not s.exists():
        print(f"[export] missing {s}", flush=True)
        return 0
    if s.is_dir():
        shutil.copytree(s, dst, dirs_exist_ok=True)
        return sum(1 for f in dst.rglob("*") if f.is_file())
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(s, dst)
    return 1


def _export(spec: dict, part: str) -> dict:
    ex = ps.exports(spec).get(part)
    if not ex:
        return {}
    step = ps.STEP_DIRS[part.rstrip("abl")]
    if part == "8" and spec.get("step8_dir"):
        step = spec["step8_dir"]
    out = Path(ps.paths(spec)["EXPORT"]) / step
    counts = {dst: _copy(src, out / dst) for src, dst in ex}
    summ = out / "summary.json"
    old = json.loads(summ.read_text()) if summ.exists() else {}
    summ.write_text(json.dumps({**old, "exported": counts, "run": spec["run"], "dataset": spec["dataset"]}, indent=2))
    return {"exported": counts}


def _sh_part(spec: dict, part: str, attempt: int, cwd: str = CODE, env: dict | None = None):
    p = ps.paths(spec)
    log = Path(p["EXPORT"]) / "logs" / f"{ps.LOG_NAMES[part]}.log"
    env = {"COME_SPLATTING_CONFIG": f"{p['M']}/3dgs_output/config.json", "PYTHONPATH": CODE,
           "HF_HUB_DISABLE_TELEMETRY": "1", **(env or {})}
    for cmd in ps.commands(spec)[part]:
        mc.run_logged(cmd, log, cwd, env, attempt)


def _prep(spec: dict, source: str):
    """Scene data + CoMe model on Volume inpaint360gs: S/{images,sparse/0,images_<r>}, M/3dgs_output, REF."""
    from PIL import Image
    p = ps.paths(spec)
    S, M, REF = Path(p["S"]), Path(p["M"]), Path(p["REF"])
    if source == "volume":                                    # Volume safe-gs-come:<key>/{colmap,come}
        come_vol.reload()
        key = spec.get("come_key") or spec["dk"]
        C = Path(COME_VOL) / key
        for sub in ("images", "sparse"):
            shutil.copytree(C / "colmap" / sub, S / sub, dirs_exist_ok=True)
        (M / "3dgs_output").mkdir(parents=True, exist_ok=True)
        for f in (C / "come").iterdir():
            if f.name in ("renders", "sdf_vis", "edit", "tensorboard") or f.name.startswith("chkpnt"):
                continue
            (shutil.copytree if f.is_dir() else shutil.copy2)(f, M / "3dgs_output" / f.name,
                                                              **({"dirs_exist_ok": True} if f.is_dir() else {}))
        rend = C / "come" / "renders"
    else:                                                     # uploaded by ::upload_local
        rend = Path(ps.ROOT) / "ref_upload" / spec["dk"]
    assert (S / "sparse" / "0").exists() and (M / "3dgs_output" / "cfg_args").exists(), "scene data / CoMe model missing"
    REF.mkdir(parents=True, exist_ok=True)
    n_ref = 0
    for f in rend.rglob("*.png"):                            # flat <image>.png, as render_check.py expects
        if "__" not in f.stem:
            shutil.copy2(f, REF / f.name); n_ref += 1
    r = spec["resolution"]
    imgs = sorted(f for f in (S / "images").iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    dst = S / f"images_{r}"
    dst.mkdir(exist_ok=True)
    for f in imgs:
        if (dst / f.name).exists():
            continue
        if r == 1:
            shutil.copy2(f, dst / f.name)
        else:
            im = Image.open(f)
            im.resize((round(im.width / r), round(im.height / r)), Image.LANCZOS).save(dst / f.name, quality=95)
    return {"images": len(imgs), f"images_{r}": len(list(dst.iterdir())), "come_renders": n_ref}


def _prepare_run(spec: dict):
    """Per-run copies RS (data) / RM (model) and the code-tree links (config, LaMa, tracking_results, assets)."""
    p = ps.paths(spec)
    for src, dst in ((p["S"], p["RS"]), (p["M"], p["RM"])):
        if not Path(dst).exists():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("chkpnt*.pth"))
    for d, s in ps.run_links(spec).items():
        if d.startswith(CODE):
            _link(d, s)


def _distill_iteration() -> int:
    try:
        return int(json.loads(Path(CODE, "config/object_distill/train_distill.json").read_text())["iterations"])
    except Exception:  # noqa: BLE001
        return 2000


# ---------------------------------------------------------------------------------------------------- functions
@app.function(image=i360_image, gpu=mc.GPU_COME, cpu=8, memory=65536, volumes=VOLUMES, timeout=8 * 3600, retries=RETRY)
def main_parts(spec: dict, parts: list[str], force: list[str] | None = None, source: str = "volume") -> list[dict]:
    """Every part of `parts` that runs in the Step 4-12 image, in order."""
    c = mc.VolumeCommitter(vol)
    p = ps.paths(spec)
    spec = {**spec, "distill_iteration": _distill_iteration()}
    scene_state = mc.StepState(c, p["SCENE_STATE"])
    run_state = mc.StepState(c, p["RUN_STATE"])
    out = []
    for part in parts:
        st = scene_state if part in ps.SCENE_STEPS else run_state
        if part in ps.RUN_STEPS:
            _prepare_run(spec)

        def body(attempt, part=part):
            if part == "prep":
                return _prep(spec, source)
            _sh_part(spec, part, attempt)
            return _export(spec, part)
        out.append({part: st.run(part, body, force=part in (force or []))})
    return out


@app.function(image=lama_image, gpu="T4", cpu=4, memory=32768, volumes=VOLUMES, timeout=4 * 3600, retries=RETRY)
def lama_part(spec: dict, force: bool = False) -> dict:
    c = mc.VolumeCommitter(vol)
    for d, s in ps.run_links(spec).items():
        if d.startswith(LAMA_CODE):
            _link(d, s)
    run = f"export TORCH_HOME={LAMA_CODE} PYTHONPATH={LAMA_CODE} LAMA_PX_BUDGET=1000000; conda run --no-capture-output -n lama "

    def body(attempt):
        p = ps.paths(spec)
        log = Path(p["EXPORT"]) / "logs" / f"{ps.LOG_NAMES['10l']}.log"
        for cmd in ps.commands(spec)["10l"]:
            mc.run_logged(run + cmd, log, LAMA_CODE, None, attempt)
        return {}
    return mc.StepState(c, ps.paths(spec)["RUN_STATE"]).run("10l", body, force=force)


@app.function(image=light_image, volumes=VOLUMES, timeout=24 * 3600)
def chain(spec: dict, parts: list[str], force: list[str], source: str = "volume") -> list:
    """Server-side orchestration (survives a closed laptop with --detach): groups of main parts, LaMa on T4."""
    res, group = [], []

    def flush():
        if group:
            res.extend(main_parts.remote(spec, list(group), list(force), source))
            group.clear()
    for part in parts:
        if part == "10l":
            flush()
            res.append({"10l": lama_part.remote(spec, "10l" in force)})
        else:
            group.append(part)
    flush()
    return res


@app.function(image=i360_image, cpu=4, memory=16384, timeout=1800)
def check_env() -> str:
    """CPU: build the images' main layers and import what the steps import (CUDA kernels need a GPU to run)."""
    import subprocess
    cmd = ("import torch, cv2, open3d, detectron2, segment_anything, simple_knn._C, diff_gaussian_rasterization; "
           "from diff_gaussian_rasterization import ExtendedSettings; "
           "import sys; sys.path[:0]=['Segment-and-Track-Anything','Segment-and-Track-Anything/aot','Segment-and-Track-Anything/sam']; "
           "import spatial_correlation_sampler; "
           "print('imports OK', torch.__version__)")
    r = subprocess.run([sys.executable, "-c", cmd], cwd=CODE, capture_output=True, text=True)
    ck = [str(p) for p in Path(CODE).glob("**/ckpt/*.pth")] + [str(p) for p in Path(CODE, "seg/weight").glob("*.pth")]
    return (r.stdout + r.stderr)[-3000:] + f"\ncheckpoints: {ck}"


@app.function(image=light_image, volumes=VOLUMES, timeout=600)
def status_fn(spec: dict) -> str:
    vol.reload()
    p = ps.paths(spec)
    out = [f"== {spec['dataset']}/{spec['run']} ({spec['dk']})"]
    for name, d in (("scene", p["SCENE_STATE"]), ("run", p["RUN_STATE"])):
        f = Path(d) / "state.json"
        st = json.loads(f.read_text()) if f.exists() else {"parts": {}}
        for k, v in st["parts"].items():
            out.append(f"  [{name}] {k:5s} {v.get('status'):8s} attempts={v.get('attempts')} "
                       f"min={v.get('minutes', '')} {v.get('error', '')[:200]}")
    e = Path(p["EXPORT"])
    out.append(f"export: {sorted(x.name for x in e.iterdir()) if e.exists() else []}")
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------------- local entrypoints
def _spec(dataset: str, run: str, target_id: int = 0, click: str = "", step8_dir: str = "", come_key: str = "") -> dict:
    cfg = mc.run_config(dataset, run)
    if target_id:
        cfg["target_id"] = target_id
    if click:
        cfg["samtrack_click"] = [int(v) for v in click.split(",")]
    spec = ps.make_spec(cfg)
    if step8_dir:
        spec["step8_dir"] = step8_dir
    if come_key:
        spec["come_key"] = come_key
    return spec


@app.local_entrypoint()
def plan(dataset: str = "LERF", run: str = "teatime_napkin", steps: str = "4-12", target_id: int = 0, click: str = ""):
    print(ps.describe(_spec(dataset, run, target_id, click), ps.expand_steps(steps)))


@app.local_entrypoint()
def check():
    mc.check_profile()
    print(check_env.remote())


@app.local_entrypoint()
def upload_local(dataset: str = "LERF", run: str = "teatime_napkin", scene_from: str = "", with_masks: bool = False):
    """outputs/<Dataset>/<run>/step0*/{images,sparse}, step1* (no checkpoints), step2*/renders -> Volume (prep source).
    --scene-from <other run>: take these folders from another run of the same scene (e.g. teatime_napkin).
    --with-masks: also step5 1_masks -> S/raw_hqsam and step6 1_masks_synced (+ scene.json, colour, numbered images)
    -> S/associated_hqsam*: the 3D-consistent ID space of that run is reused (its target ids stay valid), so steps 5-6
    are skipped (run --steps prep,4,7,8-12)."""
    mc.check_profile()
    sys.path.insert(0, str(SRC))
    from common.paths import step_dir
    spec = _spec(dataset, run)
    rd = Path(mc.run_config(dataset, scene_from or run)["run_dir"])
    s0, s1, s2 = step_dir(rd, 0), step_dir(rd, 1), step_dir(rd, 2)
    dk = spec["dk"]
    S = f"/{ps.ROOT_REL}/data/{dk}"
    with vol.batch_upload(force=True) as b:
        for sub in ("images", "sparse"):
            b.put_directory(str(s0 / sub), f"{S}/{sub}")
        for f in s1.rglob("*"):
            if f.is_file() and not f.name.startswith("chkpnt") and f.name != "summary.json":
                b.put_file(str(f), f"/{ps.ROOT_REL}/output/{dk}/3dgs_output/{f.relative_to(s1).as_posix()}")
        b.put_directory(str(s2 / "renders"), f"/{ps.ROOT_REL}/ref_upload/{dk}")
        if with_masks:
            s5, s6 = step_dir(rd, 5), step_dir(rd, 6)
            r = spec["resolution"]
            b.put_directory(str(s5 / "1_masks"), f"{S}/raw_hqsam")
            b.put_directory(str(s6 / "1_masks_synced"), f"{S}/associated_hqsam")
            b.put_file(str(s6 / "scene.json"), f"{S}/associated_hqsam/scene.json")
            if (s6 / "2_masks_synced_color").exists():
                b.put_directory(str(s6 / "2_masks_synced_color"), f"{S}/associated_hqsam_color")
            if (s6 / "3_ids_on_images").exists():
                b.put_directory(str(s6 / "3_ids_on_images"), f"{S}/images_{r}_num")
    print(f"[upload] {rd.name}: {s0.name}, {s1.name}, {s2.name}" + (", step5/6 masks" if with_masks else "")
          + f" -> inpaint360gs:{ps.ROOT_REL}/{{data,output,ref_upload}}/{dk}")


@app.local_entrypoint()
def run(dataset: str = "LERF", run: str = "teatime_napkin", steps: str = "4-12", force: str = "",
        source: str = "volume", target_id: int = 0, click: str = "", step8_dir: str = "", come_key: str = ""):
    """--steps prep,4-12 | 8-12 | 10 ... ; --source volume (safe-gs-come) | local (after ::upload_local);
    --force 9,11 redoes parts already done; --step8-dir step8_removal_virtual_cam_v2 to export step 8 under that name."""
    mc.check_profile()
    spec = _spec(dataset, run, target_id, click, step8_dir, come_key)
    parts = ps.expand_steps(steps)
    if spec["target_id"] is None and any(x in ps.RUN_STEPS for x in parts):
        raise SystemExit(f"no target_id for {dataset}/{run}: add it to configs/datasets or pass --target-id")
    print(ps.describe(spec, parts))
    forced = ps.expand_steps(force) if force else []
    for r in chain.remote(spec, parts, forced, source):
        print(json.dumps(r, indent=1, default=str)[:2000])


@app.local_entrypoint()
def status(dataset: str = "LERF", run: str = "teatime_napkin"):
    mc.check_profile()
    print(status_fn.remote(_spec(dataset, run)))


@app.local_entrypoint()
def download(dataset: str = "LERF", run: str = "teatime_napkin", overwrite: bool = False):
    mc.check_profile()
    spec = _spec(dataset, run)
    rd = Path(mc.run_config(dataset, run)["run_dir"])
    mc.download_dir(vol, ps.paths(spec)["EXPORT_REL"], rd, overwrite)
