"""
Step 2 (Modal GPU): original InstantMesh (run.py unchanged: Zero123++ 6 views -> LRM FlexiCubes mesh) on the Replica
objects room0_obj1, room0_obj2, office0_obj1, office0_obj2, from the RGBA crops of prepare_replica.py (step 1).

Cost choices: ONE GPU container for all objects/frames (one cold start), cheapest 24 GB GPU first (L4, then A10G),
weights downloaded by a CPU-only function (no GPU billed while downloading ~10 GB), mesh only (no --save_video,
no --export_texmap), inputs sent as bytes (no mounted folder), finished frames skipped on retry (done.json).

From F:\\Safe-GS\\Based_Model\\InstantMesh (venv F:\\Safe-GS\\.venv, $env:PYTHONIOENCODING="utf-8"):
  python modal_instantmesh/prepare_replica.py                                    # step 1 (local CPU)
  python -m modal run modal_instantmesh/modal_instantmesh_replica.py::cache      # weights -> Volume (CPU, once)
  python -m modal run --detach modal_instantmesh/modal_instantmesh_replica.py::start [--runs room0_obj1]
  python -m modal run modal_instantmesh/modal_instantmesh_replica.py::download   # -> step9_* (sha256 verified)
  python modal_instantmesh/finalize_replica.py                                   # step 3 (local) + copy ours 0-7
Volume `instantmesh`: hf_cache/, torch_extensions/, outputs/Replica/<run>/step9_instantmesh/<frame>/
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

APP_NAME = "instantmesh-replica"
WORKSPACE_PROFILE = "quangkhaisaigonhcm"
GPU = ["L4", "A10G"]
CONFIG = "configs/instant-mesh-large.yaml"
SEED = 42
DIFFUSION_STEPS = 75
COMMIT_EVERY_S = 120
RUNS = ["room0_obj1", "room0_obj2", "office0_obj1", "office0_obj2"]

_HERE = Path(__file__).resolve()                                     # in the container: /root/<this file>
LOCAL_ROOT = _HERE.parents[1] if len(_HERE.parents) > 3 else Path("/root")   # Based_Model/InstantMesh (local)
OUT_ROOT = (LOCAL_ROOT.parents[1] if len(LOCAL_ROOT.parents) > 1 else LOCAL_ROOT) \
    / "outputs" / "Replica" / "Object_mesh_SAM_InstantMesh"
STEP1, STEP2 = "step8_instantmesh_input_sam", "step9_instantmesh_zero123_lrm"

VOL = "/vol"
REPO = "/root/InstantMesh"
REMOTE = "outputs/Replica"                                          # <run>/step9_instantmesh/<frame>/

app = modal.App(APP_NAME)
vol = modal.Volume.from_name("instantmesh", create_if_missing=True)
RETRIES = modal.Retries(max_retries=3, backoff_coefficient=1.0, initial_delay=30.0)
ENV = {"PYTHONUNBUFFERED": "1", "HF_HOME": f"{VOL}/hf_cache", "HF_HUB_DISABLE_TELEMETRY": "1",
       "HF_HUB_ENABLE_HF_TRANSFER": "1", "TORCH_EXTENSIONS_DIR": f"{VOL}/torch_extensions",
       "TORCH_CUDA_ARCH_LIST": "8.6;8.9", "CUDA_HOME": "/usr/local/cuda"}

# same environment as modal_instantmesh.py (README of the authors + pins that keep diffusers 0.20.2 working)
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
    .add_local_dir(str(LOCAL_ROOT), REPO, ignore=["outputs", "outputs/**", "output_v2", "output_v2/**", "ckpts",
                                                  "ckpts/**", "**/.git/**", "**/__pycache__/**", "*.md",
                                                  "examples/**", "assets/**", "src_v2/**", "modal_instantmesh/**"])
)


def _commit(reason: str = "") -> None:
    try:
        vol.commit()
        print(f"[volume] committed {reason}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[volume] commit failed ({reason}): {e}", flush=True)


class PeriodicCommit:
    """Commit every COMMIT_EVERY_S s, on SIGTERM (preemption) and at the end of the block."""

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
        _commit("end of block")
        return False


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(folder: Path) -> None:
    man = {str(p.relative_to(folder)).replace("\\", "/"): {"bytes": p.stat().st_size, "sha256": _sha256(p)}
           for p in sorted(folder.rglob("*")) if p.is_file() and p.name not in ("manifest.json", "done.json")}
    (folder / "manifest.json").write_text(json.dumps(man, indent=1))


def _mesh_stats(obj: Path) -> dict:
    import trimesh
    m = trimesh.load(obj, process=False, force="mesh")
    parts = m.split(only_watertight=False)
    big = max(parts, key=lambda p: len(p.faces)) if len(parts) else m
    return {"vertices": int(len(m.vertices)), "faces": int(len(m.faces)), "components": int(len(parts)),
            "largest_component_face_share": round(len(big.faces) / max(len(m.faces), 1), 4),
            "watertight": bool(m.is_watertight), "largest_watertight": bool(big.is_watertight),
            "extents": [round(float(x), 4) for x in m.extents]}


@app.function(image=image, volumes={VOL: vol}, cpu=4, memory=8192, timeout=3600, retries=RETRIES)
def cache_weights() -> str:
    """CPU only: every checkpoint once into HF_HOME on the Volume (resumable)."""
    from huggingface_hub import hf_hub_download, snapshot_download
    vol.reload()
    marker = Path(VOL) / "hf_cache" / "instantmesh_weights.ok"
    if marker.exists():
        return "weights already cached"
    t = time.time()
    with PeriodicCommit():
        snapshot_download("sudo-ai/zero123plus-v1.2")
        hf_hub_download("TencentARC/InstantMesh", "diffusion_pytorch_model.bin", repo_type="model")
        hf_hub_download("TencentARC/InstantMesh", f"{Path(CONFIG).stem.replace('-', '_')}.ckpt", repo_type="model")
        snapshot_download("facebook/dino-vitb16")
        marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    return f"weights cached in {time.time() - t:.0f}s"


@app.function(image=image, gpu=GPU, volumes={VOL: vol}, timeout=3 * 3600, retries=RETRIES, cpu=4, memory=32768)
def run_all(jobs: list[dict]):
    """jobs: [{run, frame, png: bytes, frames_json: str}], all objects in this one container."""
    vol.reload()
    if not (Path(VOL) / "hf_cache" / "instantmesh_weights.ok").exists():
        raise RuntimeError("weights not cached: run ::cache first (CPU)")
    gpu_name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
    print(f"[run_all] gpu={gpu_name} jobs={[(j['run'], j['frame']) for j in jobs]}", flush=True)
    with PeriodicCommit():
        for j in jobs:
            run_dir = Path(VOL) / REMOTE / j["run"] / "step9_instantmesh"
            out = run_dir / j["frame"]
            if (out / "done.json").exists():
                print(f"[skip] {j['run']}/{j['frame']} already done", flush=True)
                continue
            out.mkdir(parents=True, exist_ok=True)
            (run_dir / "step1_frames.json").write_text(j["frames_json"])
            inp = Path("/tmp/inputs") / j["run"] / f"{j['frame']}.png"
            inp.parent.mkdir(parents=True, exist_ok=True)
            inp.write_bytes(j["png"])
            cmd = ["python", "run.py", CONFIG, str(inp), "--output_path", str(out), "--no_rembg",
                   "--seed", str(SEED), "--diffusion_steps", str(DIFFUSION_STEPS)]
            print(f"[run] {j['run']}/{j['frame']}: {' '.join(cmd)}", flush=True)
            t = time.time()
            with open(out / "run.log", "w") as log:
                rc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT).returncode
            if rc != 0:
                print(Path(out, "run.log").read_text()[-3000:], flush=True)
                _commit(f"{j['run']}/{j['frame']} FAILED")
                raise RuntimeError(f"run.py failed for {j['run']}/{j['frame']} (rc={rc})")
            obj = next((out / Path(CONFIG).stem / "meshes").glob("*.obj"))
            info = {"run": j["run"], "frame": j["frame"], "seconds": round(time.time() - t, 1), "gpu": gpu_name,
                    "config": CONFIG, "seed": SEED, "diffusion_steps": DIFFUSION_STEPS,
                    "mesh": str(obj.relative_to(out)), "stats": _mesh_stats(obj)}
            _write_manifest(out)
            (out / "done.json").write_text(json.dumps(info, indent=2))
            print(f"[done] {json.dumps(info)}", flush=True)
            _commit(f"{j['run']}/{j['frame']} done")
    print("[run_all] finished", flush=True)


def _assert_workspace():
    cur = subprocess.run([sys.executable, "-m", "modal", "profile", "current"], capture_output=True, text=True).stdout.strip()
    if cur != WORKSPACE_PROFILE:
        raise SystemExit(f"active Modal profile is '{cur}', expected '{WORKSPACE_PROFILE}'")


def _step1(run: str) -> Path:
    scene, obj = run.split("_")
    return OUT_ROOT / scene / obj / STEP1


@app.local_entrypoint()
def cache():
    _assert_workspace()
    print(cache_weights.remote())


@app.local_entrypoint()
def start(runs: str = ",".join(RUNS), frames: str = ""):
    """python -m modal run --detach modal_instantmesh/modal_instantmesh_replica.py::start"""
    _assert_workspace()
    only = {f for f in frames.split(",") if f}
    jobs = []
    for run in [r for r in runs.split(",") if r]:
        d = _step1(run)
        if not (d / "frames.json").exists():
            raise SystemExit(f"{d} missing: run modal_instantmesh/prepare_replica.py first")
        fj = (d / "frames.json").read_text()
        for r in json.loads(fj)["frames"]:
            if not only or r["frame"] in only:
                jobs.append({"run": run, "frame": r["frame"], "png": (d / "rgba" / f"{r['frame']}.png").read_bytes(),
                             "frames_json": fj})
    print(f"[start] {len(jobs)} jobs in one container")
    run_all.remote(jobs)


@app.local_entrypoint()
def status(runs: str = ",".join(RUNS)):
    for run in runs.split(","):
        try:
            entries = vol.listdir(f"{REMOTE}/{run}/step9_instantmesh")
        except Exception:  # noqa: BLE001
            print(f"{run}: -")
            continue
        for e in entries:
            if e.type == modal.volume.FileEntryType.DIRECTORY:
                try:
                    d = json.loads(b"".join(vol.read_file(f"{e.path}/done.json")))
                    print(f"{run}/{Path(e.path).name}: {json.dumps({'s': d['seconds'], **d['stats']})}")
                except Exception:  # noqa: BLE001
                    print(f"{run}/{Path(e.path).name}: running")


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
def download(runs: str = ",".join(RUNS), overwrite: bool = False):
    """Volume <run>/step9_instantmesh -> outputs/Replica/Object_mesh_SAM_InstantMesh/<scene>/<obj>/step9_*,
    sha256-verified against manifest.json, corrupt files fetched again once."""
    _assert_workspace()
    for run in runs.split(","):
        remote = f"{REMOTE}/{run}/step9_instantmesh"
        dest = _step1(run).parent / STEP2
        if dest.exists() and not overwrite:
            print(f"[skip] {dest} exists (--overwrite)")
            continue

        def fetch(path: str):
            out = dest / path[len(remote) + 1:]
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "wb") as f:
                for chunk in vol.read_file(path):
                    f.write(chunk)

        files = [e.path for e in vol.listdir(remote, recursive=True) if e.type == modal.volume.FileEntryType.FILE]
        for p in files:
            fetch(p)
        probs = _verify_local(dest)
        if probs:
            for p in files:
                if any(str(dest / p[len(remote) + 1:]) in q for q in probs):
                    fetch(p)
            probs = _verify_local(dest)
        print(f"{run}: {len(files)} files -> {dest}; " + ("verify OK" if not probs else "PROBLEMS:\n" + "\n".join(probs)))
