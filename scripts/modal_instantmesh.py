"""
Step 14.2-14.5 of Safe-GS on Modal: object mesh with InstantMesh from a 360 virtual video (PLAN Phase 4)
=======================================================================================================
From test_model/CoMe_Inpainting360GS/example_modal_come_instantmesh.py (plan_mesh_completion_instantMesh.md).
Code = src/tools_instantmesh + src/instantmesh. Volume `instantmesh` (HF cache + nvdiffrast plugin + runs):
    <Dataset>/<run>/inputs_14/            = outputs/<Dataset>/<run>/step14_instantmesh/1_object_gaussians (14.1, local)
    <Dataset>/<run>/step14_instantmesh/{2_orbit_video, 3_view_selection, 4_instantmesh, 5_final_mesh}

  python scripts/run_pipeline.py --dataset LERF --run teatime_napkin --step 14.1 --object-id 44     # local CPU
  python -m modal run --detach scripts/modal_instantmesh.py::start --dataset LERF --run-name teatime_napkin
  python -m modal run scripts/modal_instantmesh.py::status --dataset LERF --run-name teatime_napkin
  python -m modal run scripts/modal_instantmesh.py::download --dataset LERF --run-name teatime_napkin  # sha256-verified
  python scripts/run_pipeline.py --dataset LERF --run teatime_napkin --step 15                       # local CPU
Modal profile: --profile or env MODAL_PROFILE (optional; default = the active profile).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import modal

import modal_common as mc

mc.utf8_console()

APP_NAME = "safe-gs-instantmesh"
GPU = ["L4", "A10G"]                     # no CoMe extension here: A10G is fine
VOL = "/vol"
CODE = "/root/safegs"
PARTS = {"14.2": "2_orbit_video", "14.3": "3_view_selection", "14.4": "4_instantmesh", "14.5": "5_final_mesh"}
STEP = "step14_instantmesh"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name("instantmesh", create_if_missing=True)
RETRIES = modal.Retries(max_retries=3, backoff_coefficient=1.0, initial_delay=30.0)

# same recipe as Based_Model/InstantMesh -> cached layers
ENV = {"PYTHONUNBUFFERED": "1", "HF_HOME": f"{VOL}/hf_cache", "HF_HUB_DISABLE_TELEMETRY": "1",
       "HF_HUB_ENABLE_HF_TRANSFER": "1", "TORCH_EXTENSIONS_DIR": f"{VOL}/torch_extensions",
       "TORCH_CUDA_ARCH_LIST": "8.6;8.9", "CUDA_HOME": "/usr/local/cuda"}
CONSTRAINTS = "\\n".join(["numpy==1.26.4", "torch==2.1.0", "torchvision==0.16.0", "huggingface-hub==0.17.3",
                          "transformers==4.34.1", "diffusers==0.20.2", "pytorch-lightning==2.1.2",
                          "accelerate==0.24.1", "onnxruntime==1.17.3", "rembg==2.0.56",
                          "opencv-python-headless==4.9.0.80"])
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "build-essential", "ninja-build", "ffmpeg", "libgl1", "libglib2.0-0")
    .env(ENV)
    .pip_install("wheel", "setuptools==69.5.1", "ninja")
    .pip_install("torch==2.1.0", "torchvision==0.16.0", extra_index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("xformers==0.0.22.post7", "numpy==1.26.4", extra_index_url="https://download.pytorch.org/whl/cu121")
    .run_commands(
        f"printf '{CONSTRAINTS}\\n' > /opt/constraints.txt",
        "pip install -c /opt/constraints.txt pytorch-lightning huggingface-hub hf_transfer einops omegaconf "
        "torchmetrics webdataset accelerate tensorboard PyMCubes trimesh rembg onnxruntime transformers "
        "diffusers 'imageio[ffmpeg]' xatlas plyfile opencv-python-headless scipy networkx",
        "pip install -c /opt/constraints.txt git+https://github.com/NVlabs/nvdiffrast.git@v0.3.3",
    )
    .pip_install("matplotlib==3.8.4", "numpy==1.26.4")
    .add_local_dir(str(mc.SRC / "tools_instantmesh"), f"{CODE}/tools_instantmesh", ignore=["**/__pycache__/**"])
    .add_local_dir(str(mc.SRC / "instantmesh"), f"{CODE}/instantmesh", ignore=["**/__pycache__/**"])
    .add_local_python_source("modal_common")
)


def _rel(dataset: str, run_name: str) -> str:
    return f"{dataset}/{run_name}"


@app.function(image=image, gpu=GPU, volumes={VOL: vol}, timeout=3 * 3600, retries=RETRIES, cpu=4, memory=32768)
def run_step14(rel: str, parts: list[str], force: list[str], basic_mesh: str = ""):
    sys.path.insert(0, f"{CODE}/tools_instantmesh")
    c = mc.VolumeCommitter(vol)
    c.reload()
    import step14_gpu as s14
    inputs = f"{VOL}/{rel}/inputs_14"
    if not Path(inputs, "frame.json").exists():
        raise RuntimeError(f"{inputs}/frame.json missing: run ::start (uploads 14.1) after extract_object.py")
    d = {k: Path(VOL, rel, STEP, v) for k, v in PARTS.items()}
    for k in force:
        if d[k].exists():
            shutil.rmtree(d[k]); print(f"[force] removed {d[k]}", flush=True)

    def manifest(folder: Path):
        man = {k: {"bytes": v["size"], "sha256": v["sha256"]} for k, v in mc.manifest(folder).items()
               if k != "manifest.json"}
        (folder / "manifest.json").write_text(json.dumps(man, indent=1))

    with mc.PeriodicCommit(c, every=120):
        if "14.2" in parts and not (d["14.2"] / "done.json").exists():
            s14.step14_2_orbit(inputs, str(d["14.2"]), commit=c.commit)
            manifest(d["14.2"]); c.commit("14.2 manifest")
        if "14.3" in parts:
            s14.step14_3_select(inputs, str(d["14.2"]), str(d["14.3"]))
            manifest(d["14.3"]); c.commit("14.3 done")
        if "14.4" in parts:
            s14.step14_4_5_recon_eval(inputs, str(d["14.2"]), str(d["14.3"]), str(d["14.4"]), f"{CODE}/instantmesh",
                                      basic_mesh=basic_mesh or None, commit=c.commit)
            manifest(d["14.4"]); manifest(d["14.5"]); c.commit("14.4/14.5 done")
    print("[run_step14] finished", flush=True)


def _run_local(dataset: str, run_name: str) -> Path:
    return mc.OUTPUTS / dataset / "ours" / run_name


@app.local_entrypoint()
def start(dataset: str = "LERF", run_name: str = "teatime_napkin", parts: str = "14.2,14.3,14.4", force: str = "",
          profile: str = "", basic_mesh: str = ""):
    """Upload 14.1 (1_object_gaussians, without renderer_check/) then run 14.2-14.5. Use with --detach.
    --basic-mesh: optional reference mesh (local path) compared in 14.5 (e.g. mesh_napkin_closed.ply of the basic run)."""
    mc.check_profile(profile or None)
    run_local = _run_local(dataset, run_name)
    s141 = run_local / STEP / "1_object_gaussians"
    if not (s141 / "frame.json").exists():
        raise SystemExit(f"{s141}/frame.json missing: run 14.1 first (scripts/run_pipeline.py --step 14.1)")
    rel = _rel(dataset, run_name)
    with vol.batch_upload(force=True) as b:
        for f in s141.rglob("*"):
            if f.is_file() and "renderer_check" not in f.parts:
                b.put_file(str(f), f"/{rel}/inputs_14/{f.relative_to(s141).as_posix()}")
        bm = ""
        if basic_mesh:
            b.put_file(basic_mesh, f"/{rel}/ref/basic_mesh.ply")
            bm = f"{VOL}/{rel}/ref/basic_mesh.ply"
    print(f"[upload] {s141} -> instantmesh:{rel}/inputs_14")
    run_step14.remote(rel, [p for p in parts.split(",") if p], [x for x in force.split(",") if x], bm)


def _remote_json(path):
    try:
        return json.loads(b"".join(vol.read_file(path)))
    except Exception:  # noqa: BLE001
        return None


@app.local_entrypoint()
def status(dataset: str = "LERF", run_name: str = "teatime_napkin", profile: str = ""):
    mc.check_profile(profile or None)
    base = f"{_rel(dataset, run_name)}/{STEP}"
    print("14.2:", _remote_json(f"{base}/2_orbit_video/done.json"))
    s = _remote_json(f"{base}/3_view_selection/selected_views.json")
    print("14.3:", s and {k: v for k, v in s.items() if k != "chosen"})
    print("14.5:", _remote_json(f"{base}/5_final_mesh/summary.json"))


def _verify(folder: Path):
    bad = []
    for mf in folder.rglob("manifest.json"):
        for rel, info in json.loads(mf.read_text()).items():
            p = mf.parent / rel
            if not p.exists() or p.stat().st_size != info["bytes"] or mc.sha256(p) != info["sha256"]:
                bad.append(str(p))
    return bad


@app.local_entrypoint()
def download(dataset: str = "LERF", run_name: str = "teatime_napkin", profile: str = ""):
    """Volume step-14 parts -> outputs/<Dataset>/<run>/step14_instantmesh/ (never overwrites: <part>_dl2, _dl3 ...)."""
    mc.check_profile(profile or None)
    s14_local = _run_local(dataset, run_name) / STEP
    for name in PARTS.values():
        remote = f"{_rel(dataset, run_name)}/{STEP}/{name}"
        try:
            vol.listdir(remote)
        except Exception:  # noqa: BLE001
            print(f"{name}: not on the volume"); continue
        dest, j = s14_local / name, 2
        while dest.exists():
            dest = s14_local / f"{name}_dl{j}"; j += 1
        n = mc.download_dir(vol, remote, dest, overwrite=True)
        bad = _verify(dest)
        if bad:                                           # one retry of the corrupted files
            mc.download_dir(vol, remote, dest, overwrite=True, skip=lambda r: str(dest / r) not in bad)
            bad = _verify(dest)
        print(f"{name}: {n} files -> {dest}  verify: {'OK' if not bad else bad}")
