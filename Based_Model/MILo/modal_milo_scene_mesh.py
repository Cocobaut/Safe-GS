"""
MILo baseline (scene-level mesh) on Modal: train.py (MILo, Mesh-In-the-Loop GS) -> mesh_extract_sdf.py
=======================================================================================================
One result per scene (no object removal), same layout as the CoMe baseline (Based_Model/CoMe/modal_come_scene_mesh.py):
    outputs/<Dataset>/Scene_Mesh_MILo/<scene>/step0_colmap   images + sparse/0 (reused COLMAP, not re-run) + summary.json
    outputs/<Dataset>/Scene_Mesh_MILo/<scene>/step1_3DGS     MILo model (point_cloud/iteration_<it>, cfg_args, cameras.json)
    outputs/<Dataset>/Scene_Mesh_MILo/<scene>/step2_Mesh     mesh_milo_learnable_sdf.ply + done.json + logs/

COLMAP is reused from --colmap-root/<scene> (images/ + sparse/0), default env SAFE_GS_REPLICA_COLMAP or
"F:/Replica 8 Scene/Replica_safegs". Volume `safe-gs-milo`: <dataset lower>/<scene>/{data, model, out, state}.
Recipe (MILo README, indoor):  train.py --imp_metric indoor --rasterizer radegs [--eval]   (18k it, configs/fast)
                               mesh_extract_sdf.py --rasterizer radegs --config default [--eval]
--eval (every 8th view held out) is kept on by default, like the CoMe baseline / Ours (plan.md R30).
Only change to MILo code (in the image copy, local repo untouched): nvdiffrast RasterizeGLContext -> RasterizeCudaContext
(no EGL/OpenGL on Modal workers).

  $env:PYTHONIOENCODING="utf-8"; $env:MODAL_PROFILE="<profile of the workspace>"      # from F:\\Safe-GS
  python -m modal run --detach Based_Model/MILo/modal_milo_scene_mesh.py::run --dataset Replica --scenes room0,office0
  python -m modal run Based_Model/MILo/modal_milo_scene_mesh.py::download --dataset Replica --scenes room0,office0
Variant (own output folder Scene_Mesh_MILo_<variant> and Volume key <scene>_<variant>), e.g. dense, all views:
  ... ::run --scenes room0,office0 --variant dense --no-eval --train-flags "--imp_metric indoor --rasterizer radegs --dense_gaussians"
  ... ::download --scenes room0,office0 --variant dense
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
if len(_HERE.parents) > 2 and (_HERE.parents[2] / "scripts" / "modal_common.py").exists():   # local: <root>/scripts
    sys.path.insert(0, str(_HERE.parents[2] / "scripts"))

import modal  # noqa: E402

import modal_common as mc  # noqa: E402

mc.utf8_console()

MILO_DIR = _HERE.parent                                   # Based_Model/MILo (local only)
COLMAP_ROOT = os.environ.get("SAFE_GS_REPLICA_COLMAP", "F:/Replica 8 Scene/Replica_safegs")
METHOD_DIR = "Scene_Mesh_MILo"
ITER = 18000                                              # configs/fast: iterations=18_000
GPU = "L40S"
TRAIN_FLAGS = "--imp_metric indoor --rasterizer radegs"
MESH_FLAGS = "--rasterizer radegs --config default"
# local build artefacts (CMakeCache, _deps, .so of a py3.9 build, ...) must not reach the image
BUILD_JUNK = mc.JUNK + ["CMakeCache.txt", "CMakeFiles", "CMakeFiles/**", "Makefile", "cmake_install.cmake",
                        "_deps", "_deps/**", "**/pybind11-*", "**/pybind11-*/**", "**/*.so", "**/*.o", "bin",
                        "bin/**", "lib", "lib/**", "checkpoints/**"]


def milo_image() -> modal.Image:
    sub = MILO_DIR / "submodules"
    img = (
        modal.Image.from_registry("nvidia/cuda:11.8.0-devel-ubuntu22.04", add_python="3.10")
        .apt_install("git", "build-essential", "ninja-build", "cmake", "libgl1", "libglib2.0-0", "libgomp1",
                     "libgmp-dev", "libmpfr-dev", "libboost-dev", "libcgal-dev", "libegl1", "libx11-6")
        .env({"CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "8.9", "CMAKE_CUDA_ARCHITECTURES": "89",
              "FORCE_CUDA": "1", "MAX_JOBS": "8", "CC": "gcc", "CXX": "g++", "LDSHARED": "g++ -shared"})
        .pip_install("torch==2.3.1", "torchvision==0.18.1", index_url="https://download.pytorch.org/whl/cu118")
        .pip_install("numpy==1.26.4", "open3d==0.19.0", "trimesh==4.6.8", "scikit-image==0.24.0",
                     "opencv-python-headless==4.11.0.86", "plyfile==1.1", "tqdm==4.67.1", "pyyaml", "ninja",
                     "setuptools<70", "wheel", "matplotlib", "scipy", "pillow", "lpips")
    )
    for name in ("diff-gaussian-rasterization_ms", "diff-gaussian-rasterization", "diff-gaussian-rasterization_gof",
                 "simple-knn", "fused-ssim", "nvdiffrast", "tetra_triangulation"):
        img = img.add_local_dir(str(sub / name), f"/opt/MILo/submodules/{name}", copy=True, ignore=BUILD_JUNK)
    img = img.run_commands(                                            # MILo install.py, in the same order
        "cd /opt/MILo/submodules && for d in diff-gaussian-rasterization_ms diff-gaussian-rasterization "
        "diff-gaussian-rasterization_gof simple-knn fused-ssim; do pip install --no-build-isolation ./$d || exit 1; done",
        "cd /opt/MILo/submodules/tetra_triangulation && export CPLUS_INCLUDE_PATH=/usr/local/cuda/include && "
        "cmake . -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 "
        "-DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)') && make -j8 && "
        "pip install -e . --no-build-isolation",
        "cd /opt/MILo/submodules/nvdiffrast && pip install --no-build-isolation .",
    )
    return (img
            .env({**mc.BASE_ENV, "TORCH_EXTENSIONS_DIR": "/root/.cache/torch_extensions"})
            .add_local_dir(str(MILO_DIR / "milo"), "/opt/MILo/milo", copy=True,
                           ignore=["**/__pycache__", "**/__pycache__/**", "**/*.pyc", "data", "data/**", "output",
                                   "output/**", "train.py.new", "train.py.orig", "blender", "blender/**"])
            .run_commands("sed -i 's/use_opengl=True/use_opengl=False/' /opt/MILo/milo/scene/mesh.py && "
                          "grep -n 'use_opengl=False' /opt/MILo/milo/scene/mesh.py")
            .add_local_python_source("modal_common"))


app = modal.App("safe-gs-milo-scene-mesh")
vol = modal.Volume.from_name("safe-gs-milo", create_if_missing=True)
image = milo_image()

MILO_CHECK = ("python -c \"import torch, trimesh, open3d, nvdiffrast.torch, diff_gaussian_rasterization, "
              "diff_gaussian_rasterization_ms, diff_gaussian_rasterization_gof, simple_knn._C; "
              "from tetranerf.utils.extension import cpp; print('MILo imports OK', torch.__version__, torch.version.cuda)\"")


def _local(dataset: str, scene: str, variant: str = "") -> Path:
    return mc.OUTPUTS / dataset / (METHOD_DIR + (f"_{variant}" if variant else "")) / scene


def _remote(dataset: str, scene: str, variant: str = "") -> str:
    return mc.dataset_key(dataset, scene) + (f"_{variant}" if variant else "")


# ==================================================================================================== GPU
@app.function(image=image, gpu=GPU, cpu=8, memory=65536, volumes={mc.VOL: vol}, timeout=8 * 3600, retries=1)
def milo_scene(rel: str, it: int = ITER, eval_split: bool = True, train_flags: str = TRAIN_FLAGS,
               mesh_flags: str = MESH_FLAGS, force: bool = False) -> dict:
    root = Path(mc.VOL) / rel
    data, model, out = root / "data", root / "model", root / "out"
    committer = mc.VolumeCommitter(vol)
    committer.reload()
    if (out / "done.json").exists() and not force:
        return json.loads((out / "done.json").read_text())
    if not (data / "sparse" / "0").exists():
        raise RuntimeError(f"{data}/sparse/0 missing (run the local entrypoint: it uploads step0_colmap)")
    state = mc.StepState(committer, root / "state")
    ev = " --eval" if eval_split else ""
    cwd = "/opt/MILo/milo"
    logs = out / "logs"
    mc.run_logged(MILO_CHECK, logs / "env_check.log", cwd)

    def train(attempt):
        if model.exists():                                   # restart cleanly (no MILo resume for the mesh state)
            shutil.rmtree(model)
        t0 = time.time()
        mc.run_logged(f"python -u train.py -s {data} -m {model} {train_flags}{ev}",
                      logs / "step1_train.log", cwd, attempt=attempt)
        ply = model / "point_cloud" / f"iteration_{it}" / "point_cloud.ply"
        if not ply.exists():
            raise RuntimeError(f"{ply} missing after training")
        return {"train_minutes": round((time.time() - t0) / 60, 1)}

    def mesh(attempt):
        mc.run_logged(f"python -u mesh_extract_sdf.py -s {data} -m {model} --iteration {it} {mesh_flags}{ev}",
                      logs / "step2_mesh.log", cwd, attempt=attempt)
        src = model / "mesh_learnable_sdf.ply"
        if not src.exists():
            raise RuntimeError(f"{src} missing after mesh extraction")
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out / "mesh_milo_learnable_sdf.ply")
        return {}

    state.run("train", train, force=force)
    state.run("mesh", mesh, force=force, needs=["train"])
    import trimesh
    m = trimesh.load(str(out / "mesh_milo_learnable_sdf.ply"), process=False)
    st = state.load()["parts"]
    info = {"method": "MILo (original, scene-level mesh, no object removal)", "key": rel, "iteration": it,
            "output": "mesh_milo_learnable_sdf.ply", "n_vertices": int(len(m.vertices)), "n_faces": int(len(m.faces)),
            "n_gaussians": _count_gaussians(model / "point_cloud" / f"iteration_{it}" / "point_cloud.ply"),
            "train_minutes": st["train"].get("train_minutes"), "mesh_minutes": st["mesh"].get("minutes"),
            "gpu": GPU, "eval_split": eval_split,
            "command": [f"train.py -s <step0_colmap> -m <model> {train_flags}{ev}",
                        f"mesh_extract_sdf.py -s <step0_colmap> -m <model> --iteration {it} {mesh_flags}{ev}"]}
    (out / "done.json").write_text(json.dumps(info, indent=2))
    committer.commit("done")
    return info


def _count_gaussians(ply: Path) -> int:
    from plyfile import PlyData
    return int(PlyData.read(str(ply))["vertex"].count)


# ==================================================================================================== local
def _prepare_step0(scene: str, dataset: str, colmap_root: str, variant: str = "") -> Path:
    """Reuse the existing COLMAP (images + sparse/0): copy into outputs/.../step0_colmap and write summary.json."""
    src = Path(colmap_root) / scene
    dst = _local(dataset, scene, variant) / "step0_colmap"
    if not (src / "sparse" / "0" / "cameras.bin").exists() or not (src / "images").is_dir():
        raise SystemExit(f"{src} must contain images/ and sparse/0/{{cameras,images,points3D}}.bin")
    for sub in ("images", "sparse"):
        if not (dst / sub).exists():
            shutil.copytree(src / sub, dst / sub)
    imgs = sorted(p.name for p in (dst / "images").iterdir() if p.is_file())
    cams = (dst / "sparse" / "0" / "cameras.txt")
    summary = {"source": str(src), "reused_colmap": True, "images": len(imgs),
               "cameras_txt": cams.read_text().strip().splitlines()[-1] if cams.exists() else None}
    (dst / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[step0] {src} -> {dst} ({len(imgs)} images)")
    return dst


@app.local_entrypoint()
def run(dataset: str = "Replica", scenes: str = "room0,office0", iteration: int = ITER, no_eval: bool = False,
        train_flags: str = TRAIN_FLAGS, mesh_flags: str = MESH_FLAGS, colmap_root: str = COLMAP_ROOT,
        force: bool = False, skip_upload: bool = False, variant: str = ""):
    mc.check_profile()
    names = [s.strip() for s in scenes.split(",") if s.strip()]
    for scene in names:
        s0 = _prepare_step0(scene, dataset, colmap_root, variant)
        if not skip_upload:
            rel = _remote(dataset, scene, variant)
            with vol.batch_upload(force=True) as b:
                b.put_directory(str(s0 / "images"), f"/{rel}/data/images")
                b.put_directory(str(s0 / "sparse"), f"/{rel}/data/sparse")
            print(f"[upload] {s0} -> safe-gs-milo:{rel}/data")
    calls = [milo_scene.spawn(_remote(dataset, s, variant), iteration, not no_eval, train_flags, mesh_flags, force)
             for s in names]
    for scene, call in zip(names, calls):
        try:
            print(f"[{scene}]", json.dumps(call.get(), indent=2))
        except Exception as e:  # noqa: BLE001
            print(f"[{scene}] FAILED: {e}")


@app.local_entrypoint()
def download(dataset: str = "Replica", scenes: str = "room0,office0", overwrite: bool = False,
             with_checkpoint: bool = False, variant: str = ""):
    mc.check_profile()
    for scene in (s.strip() for s in scenes.split(",") if s.strip()):
        rel, loc = _remote(dataset, scene, variant), _local(dataset, scene, variant)
        mc.download_dir(vol, f"{rel}/out", loc / "step2_Mesh", overwrite)
        mc.download_dir(vol, f"{rel}/model", loc / "step1_3DGS", overwrite,
                        skip=lambda r: (r.endswith(".pth") and not with_checkpoint) or r == "mesh_learnable_sdf.ply")
        if vol.listdir(f"{rel}/state"):
            mc.download_dir(vol, f"{rel}/state", loc / "step2_Mesh" / "state", True)
