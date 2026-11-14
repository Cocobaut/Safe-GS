"""
Shared pieces of the Safe-GS Modal runners (scripts/modal_*.py). PLAN_REBUILD_BASE_FRAMEWORK.md, Phase 4.

Local side : PROJECT_ROOT / SRC / COME_DIR (relative to this file), run config from configs/datasets/*.yaml,
             Modal profile check (env MODAL_PROFILE, never a hard-coded account), Volume upload / download.
Container  : logged shell commands (log format of the reference run: "===== [attempt k] HH:MM:SS $ cmd" ...
             "===== exit rc"), periodic Volume commit, per-step state with retries.
Images     : come_image() = CUDA 12.1 + torch 2.4.1 + the CoMe extensions of src/come/submodules (Step 1, 2, 13).

Every image that runs code of a runner must end with `.add_local_python_source("modal_common")`.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
COME_DIR = SRC / "come"
OUTPUTS = Path(os.environ.get("SAFE_GS_OUTPUTS_DIR", PROJECT_ROOT / "outputs"))

VOL = "/vol"
BASE_ENV = {"PYTHONUNBUFFERED": "1", "QT_QPA_PLATFORM": "offscreen", "MPLBACKEND": "Agg", "WANDB_MODE": "disabled"}
GPU_COME = ["L4", "L40S"]                  # CoMe extensions built for sm_89 only (never A10G)
JUNK = ["build", "build/**", "**/build/**", "*.egg-info", "*.egg-info/**", "**/*.egg-info/**", "**/__pycache__",
        "**/__pycache__/**", "**/*.pyc", "**/*.so", "third_party/glm/doc", "third_party/glm/doc/**",
        "third_party/glm/test", "third_party/glm/test/**", "CMakeCache.txt", "CMakeFiles", "CMakeFiles/**"]


def utf8_console() -> None:
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ==================================================================================================== local side
def check_profile(expected: str | None = None) -> str:
    """Print the active Modal profile; fail only if one is requested (arg or env MODAL_PROFILE) and differs."""
    expected = expected or os.environ.get("MODAL_PROFILE") or None
    r = subprocess.run([sys.executable, "-m", "modal", "profile", "current"], capture_output=True, text=True)
    cur = r.stdout.strip()
    if expected and cur != expected:
        raise SystemExit(f"active Modal profile is '{cur}', expected '{expected}' "
                         f"(modal profile activate {expected}, or unset MODAL_PROFILE)")
    print(f"[modal] profile: {cur or '?'}")
    return cur


def run_config(dataset: str, run: str, scene: str | None = None) -> dict:
    """Resolved configs/datasets/<dataset>.yaml entry of a run (local only, needs pyyaml)."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from common.datasets import resolve_run
    return resolve_run(dataset, run, scene)


def dataset_key(dataset: str, scene: str) -> str:
    """Volume key of a scene: lerf/teatime, replica/office0, inpaint360gs/doppelherz.
    (src/scene/dataset_readers.py switches on 'inpaint360' in the path: only the Inpaint360GS dataset contains it.)"""
    return f"{dataset.lower()}/{scene}"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(folder: Path) -> dict:
    return {str(p.relative_to(folder)).replace("\\", "/"): {"size": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(folder.rglob("*")) if p.is_file()}


def upload_dir(vol: modal.Volume, local: Path, remote: str, force: bool = True) -> int:
    files = [p for p in Path(local).rglob("*") if p.is_file()]
    with vol.batch_upload(force=force) as b:
        b.put_directory(str(local), "/" + remote.strip("/"))
    print(f"[upload] {len(files)} files {local} -> {remote}")
    return len(files)


def download_dir(vol: modal.Volume, remote: str, local: Path, overwrite: bool = False,
                 skip=lambda rel: False) -> int:
    """Copy a Volume folder to `local` (keeps the relative layout). Existing local files are kept unless overwrite."""
    remote = remote.strip("/")
    n = 0
    for e in vol.listdir(remote, recursive=True):
        if e.type != modal.volume.FileEntryType.FILE:
            continue
        rel = e.path.strip("/")[len(remote) + 1:]
        if skip(rel):
            continue
        dst = Path(local) / rel
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
    return n


# ==================================================================================================== container side
class VolumeCommitter:
    def __init__(self, vol: modal.Volume):
        self.vol = vol

    def commit(self, reason: str = "") -> None:
        try:
            self.vol.commit()
            print(f"[volume] committed {reason}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[volume] commit failed ({reason}): {e}", flush=True)

    def reload(self) -> None:
        try:
            self.vol.reload()
        except RuntimeError as e:   # retry in a reused container: files of the preempted attempt still open
            print(f"[volume] reload skipped: {e}"[:300], flush=True)


class PeriodicCommit:
    """Commit the Volume every `every` s and on SIGTERM (preemption), so a retry resumes from the last commit."""

    def __init__(self, committer: VolumeCommitter, every: int = 300):
        self.c, self.every = committer, every
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self.stop.wait(self.every):
            self.c.commit("periodic")

    def __enter__(self):
        def on_term(signum, frame):
            self.c.commit(f"signal {signum}")
            os._exit(128 + signum)
        try:
            signal.signal(signal.SIGTERM, on_term)
        except ValueError:
            pass
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.c.commit("end of part")
        return False


def run_logged(cmd: str, log: Path, cwd: str, env: dict | None = None, attempt: int = 1) -> None:
    """bash -c 'set -eo pipefail; cmd' with stdout+stderr streamed to the console and appended to `log`."""
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    full_env = {**os.environ, **(env or {})}
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n===== [attempt {attempt}] {time.strftime('%H:%M:%S')} $ {cmd}\n")
        f.flush()
        print(f"$ {cmd}", flush=True)
        proc = subprocess.Popen(["bash", "-c", f"set -eo pipefail; {cmd}"], cwd=cwd, env=full_env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                errors="replace")
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        rc = proc.wait()
        f.write(f"===== exit {rc}\n")
    if rc != 0:
        raise RuntimeError(f"command failed ({rc}): {cmd}")


class StepState:
    """<state_dir>/state.json: {part: {status, attempts, minutes, ...}}. run() skips done parts, checks needs,
    commits the Volume around the part (PeriodicCommit) and records failures."""

    def __init__(self, committer: VolumeCommitter, state_dir: str | Path, every: int = 300):
        self.c, self.dir, self.every = committer, Path(state_dir), every
        self.file = self.dir / "state.json"

    def load(self) -> dict:
        return json.loads(self.file.read_text()) if self.file.exists() else {"parts": {}}

    def save(self, st: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file.write_text(json.dumps(st, indent=2, ensure_ascii=False))

    def done(self, part: str) -> bool:
        return self.load()["parts"].get(part, {}).get("status") == "done"

    def run(self, part: str, body, force: bool = False, needs: list[str] | None = None) -> dict:
        self.c.reload()
        st = self.load()
        if st["parts"].get(part, {}).get("status") == "done" and not force:
            print(f"[{part}] already done", flush=True)
            return st["parts"][part]
        for need in needs or []:
            if st["parts"].get(need, {}).get("status") != "done":
                raise RuntimeError(f"part {need} not done (needed by {part})")
        rec = st["parts"].get(part, {})
        rec.update(status="running", attempts=rec.get("attempts", 0) + 1, started=time.strftime("%Y-%m-%d %H:%M:%S"))
        rec.pop("error", None)
        st["parts"][part] = rec
        self.save(st)
        self.c.commit(f"{part} begin")
        t0 = time.time()
        try:
            with PeriodicCommit(self.c, self.every):
                extra = body(rec["attempts"]) or {}
            rec.update(status="done", minutes=round((time.time() - t0) / 60, 1),
                       finished=time.strftime("%Y-%m-%d %H:%M:%S"), **extra)
        except Exception as e:  # noqa: BLE001
            rec.update(status="failed", error=f"{type(e).__name__}: {e}"[-2000:])
            st = self.load(); st["parts"][part] = rec; self.save(st)
            self.c.commit(f"{part} failed")
            raise
        st = self.load(); st["parts"][part] = rec; self.save(st)
        self.c.commit(f"{part} done")
        return rec


# ==================================================================================================== images
def come_image(with_tetra: bool = True, extra_sources: tuple[str, ...] = ()) -> modal.Image:
    """CoMe (src/come, r4dl/CoMe @ caf62f6): extensions built from src/come/submodules (cached layers), code mounted
    at /opt/CoMe without copy (editing src/come does not rebuild CUDA). CUDA 12.1 + torch 2.4.1, sm_89."""
    sub = COME_DIR / "submodules"
    img = (
        modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
        .apt_install("git", "build-essential", "ninja-build", "cmake", "libgl1", "libglib2.0-0", "libgomp1",
                     "libgmp-dev", "libmpfr-dev", "libboost-dev", "libcgal-dev", "libegl1", "libx11-6", "libusb-1.0-0",
                     "ffmpeg")
        .env({"CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "8.9", "CUDA_ARCHITECTURES": "89",
              "FORCE_CUDA": "1", "MAX_JOBS": "8",
              # Modal's python (python-build-standalone) defaults to clang++ for extensions: use gcc
              "CC": "gcc", "CXX": "g++", "LDSHARED": "g++ -shared"})
        .pip_install("torch==2.4.1", "torchvision==0.19.1", index_url="https://download.pytorch.org/whl/cu121")
        .pip_install("numpy==1.26.4", "plyfile", "tqdm", "dacite", "open3d==0.18.0", "ninja", "GPUtil",
                     "opencv-python-headless", "trimesh", "einops", "tensorboard", "setuptools<70", "wheel",
                     "matplotlib", "imageio", "imageio-ffmpeg", "scipy", "scikit-image==0.24", "scikit-learn==1.3.2",
                     "pillow", "pyyaml", "configargparse", "lpips")
        .add_local_dir(str(sub / "simple-knn"), "/opt/come_sub/simple-knn", copy=True, ignore=JUNK)
        .add_local_dir(str(sub / "diff-gaussian-rasterization"), "/opt/come_sub/diff-gaussian-rasterization",
                       copy=True, ignore=JUNK)
        .add_local_dir(str(sub / "decoupled-fused-ssim"), "/opt/come_sub/decoupled-fused-ssim", copy=True,
                       ignore=JUNK)
        .run_commands(                                                 # CoMe README, Setup
            "pip install --no-build-isolation /opt/come_sub/simple-knn",
            "pip install --no-build-isolation /opt/come_sub/diff-gaussian-rasterization",
            "pip install --no-build-isolation /opt/come_sub/decoupled-fused-ssim",
            "pip install --no-build-isolation git+https://github.com/rahul-goel/fused-ssim/",
        )
    )
    if with_tetra:                                                     # Step 13: extract_mesh_tets.py
        img = (img
               .add_local_dir(str(sub / "tetra-triangulation"), "/opt/come_sub/tetra-triangulation", copy=True,
                              ignore=JUNK)
               .run_commands(
                   "cd /opt/come_sub/tetra-triangulation && export CPLUS_INCLUDE_PATH=/usr/local/cuda/include && "
                   "cmake . -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release "
                   "-DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)') && make -j8 && "
                   "pip install -e . --no-build-isolation",
               ))
    return (img
            .env(BASE_ENV)
            .add_local_dir(str(COME_DIR), "/opt/CoMe", copy=False,
                           ignore=["submodules", "submodules/**", "assets", "assets/**", "splatviz", "splatviz/**",
                                   "**/__pycache__", "**/__pycache__/**", "**/*.pyc"])
            .add_local_python_source("modal_common", *extra_sources))


COME_CHECK = ("python -c \"import sys; sys.argv=['x']; import torch, trimesh, open3d; "
              "import diff_gaussian_rasterization, simple_knn._C; "   # fused_ssim needs a GPU at import
              "from diff_gaussian_rasterization import ExtendedSettings; "
              # scene / gaussian_renderer create CUDA tensors at import (utils/reloc_utils.py): GPU-only, not here
              "import arguments; from utils.tetmesh import marching_tetrahedra; "
              "print('CoMe imports OK, torch', torch.__version__, 'cuda', torch.version.cuda)\"")
