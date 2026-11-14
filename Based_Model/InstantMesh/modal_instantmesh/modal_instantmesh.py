"""
InstantMesh on Modal GPU - LERF teatime, object ID 11 (white paper napkin).   Guide: ../RUN_TEATIME_NAPKIN.md

The authors' run.py runs unchanged (one subprocess per input frame). This file only builds the environment,
caches the weights on a volume, runs the frames, commits the volume periodically / after every frame
(preemption safety: a restarted container skips every frame that already has done.json) and verifies downloads.

Local (venv F:\\Safe-GS\\.venv active, profile nguyenkhacd10), from F:\\Safe-GS\\Based_Model\\InstantMesh:
  python modal_instantmesh/prepare_teatime_napkin.py                                   # step 1 (local CPU)
  python -m modal run --detach modal_instantmesh/modal_instantmesh.py::start           # step 2 (GPU)
  python -m modal run modal_instantmesh/modal_instantmesh.py::status
  python -m modal run modal_instantmesh/modal_instantmesh.py::download                 # + sha256 verification
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import modal

# --------------------------------------------------------------------------------------------------
# 1. Settings
# --------------------------------------------------------------------------------------------------
APP_NAME = "instantmesh-teatime-napkin"
WORKSPACE_PROFILE = "nguyenkhacd10"
GPU = ["L4", "A10G"]                 # first one with free capacity (both 24 GB)
RUN_NAME = "teatime_napkin"
CONFIG = "configs/instant-mesh-large.yaml"
SEED = 42
DIFFUSION_STEPS = 75
COMMIT_EVERY_S = 120
VARIANTS = {                          # name -> extra run.py flags (vertex colours first: cheapest, most important)
    "vertex_color": ["--save_video"],
    "texmap": ["--export_texmap"],
}

LOCAL_ROOT = Path(__file__).resolve().parents[1]           # F:\Safe-GS\Based_Model\InstantMesh
LOCAL_OUT = LOCAL_ROOT / "outputs" / RUN_NAME
LOCAL_INPUTS = LOCAL_OUT / "step1_inputs"

VOL = "/vol"
REPO = "/root/InstantMesh"
INPUTS = "/root/inputs"                                    # step1_inputs mounted read-only
STEP2 = "step2_instantmesh"
O = f"{VOL}/outputs/{RUN_NAME}/{STEP2}"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name("instantmesh", create_if_missing=True)
RETRIES = modal.Retries(max_retries=3, backoff_coefficient=1.0, initial_delay=30.0)

ENV = {"PYTHONUNBUFFERED": "1", "HF_HOME": f"{VOL}/hf_cache", "HF_HUB_DISABLE_TELEMETRY": "1",
       "HF_HUB_ENABLE_HF_TRANSFER": "1",
       "TORCH_EXTENSIONS_DIR": f"{VOL}/torch_extensions",   # nvdiffrast JIT plugin built once, reused
       "TORCH_CUDA_ARCH_LIST": "8.6;8.9",                   # A10G = 8.6, L4 = 8.9
       "CUDA_HOME": "/usr/local/cuda"}

# --------------------------------------------------------------------------------------------------
# 2. Image (README of the authors: python 3.10, torch 2.1.0 cu121, xformers 0.0.22.post7, requirements.txt)
#    gradio is left out (only app.py uses it); versions that later broke diffusers 0.20.2 / torch 2.1 are pinned.
# --------------------------------------------------------------------------------------------------
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
        # v0.3.3 = JIT plugin (built at first use on the GPU, cached in TORCH_EXTENSIONS_DIR)
        "pip install -c /opt/constraints.txt git+https://github.com/NVlabs/nvdiffrast.git@v0.3.3",
    )
    .add_local_dir(str(LOCAL_ROOT), REPO, ignore=["outputs", "outputs/**", "ckpts", "ckpts/**", "**/.git/**",
                                                  "**/__pycache__/**", "*.md", "examples/**", "assets/**"])
    .add_local_dir(str(LOCAL_INPUTS), INPUTS)
)


# --------------------------------------------------------------------------------------------------
# 3. Helpers (container side)
# --------------------------------------------------------------------------------------------------
def _commit(reason: str = "") -> None:
    try:
        vol.commit()
        print(f"[volume] committed {reason}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[volume] commit failed ({reason}): {e}", flush=True)


class PeriodicCommit:
    """Commit every COMMIT_EVERY_S seconds, on SIGTERM (Modal preemption) and at the end of the block."""

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
            "extents": [round(float(x), 4) for x in m.extents], "area": round(float(m.area), 4)}


def _prefetch_weights() -> None:
    """Download every checkpoint once into HF_HOME on the volume (resumable), then commit."""
    from huggingface_hub import hf_hub_download, snapshot_download
    marker = Path(VOL) / "hf_cache" / "instantmesh_weights.ok"
    if marker.exists():
        return
    t = time.time()
    snapshot_download("sudo-ai/zero123plus-v1.2")
    hf_hub_download("TencentARC/InstantMesh", "diffusion_pytorch_model.bin", repo_type="model")
    hf_hub_download("TencentARC/InstantMesh", f"{Path(CONFIG).stem.replace('-', '_')}.ckpt", repo_type="model")
    snapshot_download("facebook/dino-vitb16")
    marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    _commit(f"weights cached ({time.time() - t:.0f}s)")


# --------------------------------------------------------------------------------------------------
# 4. GPU job
# --------------------------------------------------------------------------------------------------
@app.function(image=image, gpu=GPU, volumes={VOL: vol}, timeout=3 * 3600, retries=RETRIES, cpu=4, memory=32768)
def run_all(frames: list[str] | None = None, variants: list[str] | None = None):
    vol.reload()
    cfg = json.loads(Path(INPUTS, "frames.json").read_text())
    frames = frames or [r["frame"] for r in cfg["frames"]]
    variants = variants or list(VARIANTS)
    Path(O).mkdir(parents=True, exist_ok=True)
    Path(O, "step1_frames.json").write_text(json.dumps(cfg, indent=2))
    gpu_name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
    print(f"[run_all] gpu={gpu_name} frames={frames} variants={variants}", flush=True)

    with PeriodicCommit():
        _prefetch_weights()
        for variant in variants:                        # all frames of variant 1 before variant 2
            for frame in frames:
                out = Path(O) / frame / variant
                if (out / "done.json").exists():
                    print(f"[skip] {frame}/{variant} already done", flush=True)
                    continue
                out.mkdir(parents=True, exist_ok=True)
                cmd = ["python", "run.py", CONFIG, f"{INPUTS}/rgba/{frame}.png", "--output_path", str(out),
                       "--no_rembg", "--seed", str(SEED), "--diffusion_steps", str(DIFFUSION_STEPS),
                       *VARIANTS[variant]]
                print(f"[run] {frame}/{variant}: {' '.join(cmd)}", flush=True)
                t = time.time()
                with open(out / "run.log", "w") as log:
                    rc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT).returncode
                if rc != 0:
                    print(Path(out, "run.log").read_text()[-3000:], flush=True)
                    _commit(f"{frame}/{variant} FAILED")
                    raise RuntimeError(f"run.py failed for {frame}/{variant} (rc={rc})")
                obj = next((out / Path(CONFIG).stem / "meshes").glob("*.obj"))
                info = {"frame": frame, "variant": variant, "seconds": round(time.time() - t, 1), "gpu": gpu_name,
                        "config": CONFIG, "seed": SEED, "diffusion_steps": DIFFUSION_STEPS,
                        "mesh": str(obj.relative_to(out)), "stats": _mesh_stats(obj)}
                _write_manifest(out)
                (out / "done.json").write_text(json.dumps(info, indent=2))
                print(f"[done] {json.dumps(info)}", flush=True)
                _commit(f"{frame}/{variant} done")          # = per-frame checkpoint
    print("[run_all] finished", flush=True)


# --------------------------------------------------------------------------------------------------
# 5. Local entrypoints
# --------------------------------------------------------------------------------------------------
def _assert_workspace():
    cur = subprocess.run(["python", "-m", "modal", "profile", "current"], capture_output=True, text=True).stdout.strip()
    if cur != WORKSPACE_PROFILE:
        raise SystemExit(f"active Modal profile is '{cur}', expected '{WORKSPACE_PROFILE}'")
    if not (LOCAL_INPUTS / "frames.json").exists():
        raise SystemExit("run modal_instantmesh/prepare_teatime_napkin.py first")


@app.local_entrypoint()
def start(frames: str = "", variants: str = ""):
    """Always launch with: python -m modal run --detach modal_instantmesh/modal_instantmesh.py::start"""
    _assert_workspace()
    run_all.remote(frames=[f for f in frames.split(",") if f] or None,
                   variants=[v for v in variants.split(",") if v] or None)


def _remote_json(path: str):
    try:
        return json.loads(b"".join(vol.read_file(path)))
    except Exception:  # noqa: BLE001
        return None


@app.local_entrypoint()
def status():
    cfg = json.loads((LOCAL_INPUTS / "frames.json").read_text())
    for r in cfg["frames"]:
        for v in VARIANTS:
            d = _remote_json(f"outputs/{RUN_NAME}/{STEP2}/{r['frame']}/{v}/done.json")
            print(f"{r['frame']}/{v}: " + (json.dumps({"s": d["seconds"], **d["stats"]}) if d else "-"))


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
def download():
    """outputs/<run>/step2_instantmesh -> local (never overwrites: _dl2, _dl3 ...), sha256-verified, retried once."""
    remote = f"outputs/{RUN_NAME}/{STEP2}"
    dest = LOCAL_OUT / STEP2
    k = 2
    while dest.exists():
        dest = LOCAL_OUT / f"{STEP2}_dl{k}"
        k += 1

    def fetch(entry_path: str):
        out = dest / entry_path[len(remote) + 1:]
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            for chunk in vol.read_file(entry_path):
                f.write(chunk)

    files = [e.path for e in vol.listdir(remote, recursive=True) if e.type == modal.volume.FileEntryType.FILE]
    for p in files:
        fetch(p)
    print(f"downloaded {len(files)} files -> {dest}")
    probs = _verify_local(dest)
    if probs:                                   # memory: modal downloads can be silently corrupted -> retry once
        print("re-downloading:", *probs, sep="\n  ")
        for p in files:
            local = dest / p[len(remote) + 1:]
            if any(str(local) in q for q in probs):
                fetch(p)
        probs = _verify_local(dest)
    print("\n".join(probs) if probs else "verify: OK (sha256 matches manifest.json)")
