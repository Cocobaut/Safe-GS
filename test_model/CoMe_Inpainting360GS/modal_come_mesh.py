"""
Step 13: mesh of the final 3DGS of v1 and v2 with the original CoMe code (extract_mesh_tets.py), on Modal
=========================================================================================================
Image: CUDA 12.1 + torch 2.4.1 (as the CoMe training), submodules of Based_Model/CoMe built as in the CoMe README
(diff-gaussian-rasterization, simple-knn, decoupled-fused-ssim, tetra-triangulation + CGAL); Based_Model/CoMe mounted
unchanged at /workspace/CoMe. Each model folder is CoMe-style: point_cloud/iteration_5000/point_cloud.ply (the step 11
ply; CoMe load_ply reads confidence/filter_3D by name and ignores the Inpaint360GS obj_dc_*) + cfg_args, config.json,
cameras.json, mesh_args, rem_args of the CoMe training (3dgs_output). -s = the CoMe COLMAP (data/teatime).

  python -m modal run --detach modal_come_mesh.py::run --variants v1,v2
  python -m modal run modal_come_mesh.py::download --variants v1,v2   # -> outputs/ (v1), output_v2/ (v2), never overwrites
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import modal

APP_NAME = "come-inpaint360gs-mesh"
WORKSPACE_PROFILE = "nguyenkhacd10"
HERE = Path(__file__).resolve().parent
COME_CODE = Path(r"F:\Safe-GS\Based_Model\CoMe")

VOL = "/vol"
ROOT = f"{VOL}/come360"
ROOT_REL = "come360"
D = f"{ROOT}/data/teatime"                          # CoMe COLMAP (images + sparse)
M = f"{ROOT}/output/teatime/3dgs_output"            # CoMe training: cfg_args, config.json, mesh_args, ...
STEP = "step13_come_mesh"
SAFEGS = "/vol_safegs"                             # volume safe-gs-outputs (CoMe_Infusion results)
VARIANTS = {   # name -> source ply, iteration, export folder of the run, local output folder
    "v1": {"ply": f"{ROOT}/export/teatime_napkin/step11_inpaint_3dgs_come_loss/point_cloud_inpainted_iter5000.ply",
           "iter": 5000, "run": "teatime_napkin", "local": HERE / "outputs" / "teatime_napkin"},
    "v2": {"ply": f"{ROOT}/export/teatime_napkin_v2/step11_inpaint_3dgs_come_loss/point_cloud_inpainted_iter5000.ply",
           "iter": 5000, "run": "teatime_napkin_v2", "local": HERE / "output_v2" / "teatime_napkin"},
    # InFusion (CoMe_Infusion phase 6, napkin not added back): same extractor, for a comparison of the inpainting only
    "infusion": {"ply": f"{SAFEGS}/come_infusion/phase6_train_3dgs/gs/point_cloud/iteration_150/point_cloud.ply",
                 "iter": 150, "run": "infusion_phase6",
                 "local": Path(r"F:\Safe-GS\test_model\CoMe_Infusion\Step_6_Mesh_GOF")},
    # CoMe_Infusion_Inpainting (v3: one SDXL image warped with the table plane, depth = plane)
    "v3": {"ply": f"{ROOT}/export/teatime_napkin_v3/step9_inpaint_3dgs_come_loss/point_cloud_inpainted_iter5000.ply",
           "iter": 5000, "run": "teatime_napkin_v3", "step": "step11_come_mesh",
           "local": Path(r"F:\Safe-GS\test_model\CoMe_Infusion_Inpainting\outputs\teatime_napkin")},
    # step 15 (plan.md 4b): the napkin alone, closed with the predicted underside of step 14 (v2 support). Upload first:
    #   python -m modal run modal_come_mesh.py::upload_input --variant napkin_v2 --path <step14>/point_cloud_44_closed.ply
    # Only the cameras that see the whole napkin (step14 napkin_views.json): with the object alone, a camera that does not
    # see it has 0 points to integrate and the CoMe CUDA code launches an empty grid (no guard for 0 in integrate)
    "napkin_v2": {"ply": f"{ROOT}/mesh_input/napkin_v2/point_cloud_44_closed.ply", "iter": 1,
                  "source": f"{ROOT}/data/teatime_napkin_views",
                  "run": "teatime_napkin_v2", "step": "step15_object_mesh_merge",
                  "local": HERE / "output_v2" / "teatime_napkin"},
}

app = modal.App(APP_NAME)
vol = modal.Volume.from_name("inpaint360gs", create_if_missing=True)
safegs_vol = modal.Volume.from_name("safe-gs-outputs")

_junk = ["build", "build/**", "*.egg-info", "*.egg-info/**", "**/__pycache__", "**/__pycache__/**",
         "third_party/glm/doc", "third_party/glm/doc/**", "third_party/glm/test", "third_party/glm/test/**",
         "CMakeCache.txt", "CMakeFiles", "CMakeFiles/**"]
SUB = COME_CODE / "submodules"
mesh_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "build-essential", "ninja-build", "cmake", "libgl1", "libglib2.0-0", "libgomp1",
                 "libcgal-dev", "libgmp-dev", "libmpfr-dev", "libboost-dev")
    .pip_install("torch==2.4.1", "torchvision==0.19.1", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("numpy==1.26.4", "plyfile", "tqdm", "dacite", "opencv-python-headless", "trimesh", "einops",
                 "pillow", "scipy", "pyyaml", "matplotlib", "imageio", "open3d==0.18.0", "scikit-learn==1.3.2",
                 "configargparse", "lpips", "wheel", "setuptools<70")
    .env({"CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "8.9", "MAX_JOBS": "8", "CC": "gcc", "CXX": "g++",
          "PYTHONUNBUFFERED": "1"})
    .add_local_dir(str(SUB / "diff-gaussian-rasterization"), "/opt/come_sub/diff-gaussian-rasterization", copy=True,
                   ignore=_junk)
    .add_local_dir(str(SUB / "simple-knn"), "/opt/come_sub/simple-knn", copy=True, ignore=_junk)
    .add_local_dir(str(SUB / "decoupled-fused-ssim"), "/opt/come_sub/decoupled-fused-ssim", copy=True, ignore=_junk)
    .run_commands(
        "pip install --no-build-isolation /opt/come_sub/simple-knn",
        "pip install --no-build-isolation /opt/come_sub/diff-gaussian-rasterization",
        "pip install --no-build-isolation /opt/come_sub/decoupled-fused-ssim",
    )
    .add_local_dir(str(SUB / "tetra-triangulation"), "/opt/come_sub/tetra-triangulation", copy=True, ignore=_junk)
    .run_commands(  # CoMe README / CoMe_Infusion mesh image
        "cd /opt/come_sub/tetra-triangulation && export CPLUS_INCLUDE_PATH=/usr/local/cuda/include && "
        "cmake . -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release "
        "-DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)') && make -j8 && "
        "pip install -e . --no-build-isolation",
        "python -c \"from tetranerf.utils.extension import cpp; print('tetra-triangulation OK')\"",
    )
    .add_local_dir(str(COME_CODE), "/workspace/CoMe", copy=False,
                   ignore=["submodules", "submodules/**", "splatviz", "splatviz/**", "assets", "assets/**",
                           "**/__pycache__", "**/__pycache__/**"])
)
light_image = modal.Image.debian_slim(python_version="3.10")
VOLUMES = {VOL: vol, SAFEGS: safegs_vol}


def _sh(cmd: str, log: Path, cwd: str) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n===== {time.strftime('%H:%M:%S')} $ {cmd}\n")
        print(f"$ {cmd}", flush=True)
        proc = subprocess.Popen(["bash", "-c", f"set -eo pipefail; {cmd}"], cwd=cwd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        rc = proc.wait()
        f.write(f"===== exit {rc}\n")
    if rc != 0:
        raise RuntimeError(f"command failed ({rc}): {cmd}")


def _export_dir(run: str, step: str = STEP) -> Path:
    """export/<run>/step13_come_mesh; an unfinished one (no done.json) is resumed, a finished one gets _v2, _v3 ..."""
    folder, k = Path(f"{ROOT}/export/{run}/{step}"), 2
    while (folder / "done.json").exists():
        folder = Path(f"{ROOT}/export/{run}/{step}_v{k}")
        k += 1
    return folder


@app.function(image=mesh_image, gpu="L40S", cpu=16, memory=131072, volumes=VOLUMES, timeout=5 * 3600, retries=2)
def mesh_variant(name: str, debug: bool = False) -> dict:
    """CoMe extract_mesh_tets.py (defaults of the CoMe code) on the final ply of one variant."""
    vol.reload()
    var = VARIANTS[name]
    run, it, ply = var["run"], var["iter"], Path(var["ply"])
    src = var.get("source", D)
    if not ply.exists():
        raise RuntimeError(f"{ply} missing")
    step = var.get("step", STEP)
    out = _export_dir(run, step)
    model = Path(f"{ROOT}/mesh/{name}")
    it_dir = model / "point_cloud" / f"iteration_{it}"
    it_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ply, it_dir / "point_cloud.ply")    # always: an input replaced with the same size must not be skipped
    for f in ["cfg_args", "config.json", "cameras.json", "mesh_args", "rem_args"]:
        if Path(M, f).exists():
            shutil.copy2(Path(M, f), model / f)
    vol.commit()
    log = out / "logs" / f"{step}_{name}.log"
    res_dir = model / "test" / f"ours_{it}"
    found = sorted(res_dir.glob("mesh_faster_binary_search_*.ply")) if res_dir.exists() else []
    t0 = time.time()
    if not found:
        blocking = "CUDA_LAUNCH_BLOCKING=1 " if debug else ""      # exact failing kernel in the traceback
        _sh(f"{blocking}python -u extract_mesh_tets.py -m {model} -s {src} --iteration {it}", log, "/workspace/CoMe")
        found = sorted(res_dir.glob("mesh_faster_binary_search_*.ply"))
    mesh = max(found, key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(mesh, out / f"mesh_come_{name}_iter{it}.ply")
    import trimesh
    m = trimesh.load(str(mesh), process=False)
    info = {"variant": name, "source_ply": str(ply), "mesh": mesh.name, "n_vertices": int(len(m.vertices)),
            "n_faces": int(len(m.faces)), "minutes": round((time.time() - t0) / 60, 1),
            "command": f"extract_mesh_tets.py -m <model> -s {src} --iteration {it} (CoMe defaults)"}
    (out / "done.json").write_text(json.dumps(info, indent=2))
    vol.commit()
    print(json.dumps(info), flush=True)
    return {**info, "export": str(out)}


@app.function(image=mesh_image, cpu=2, memory=8192, timeout=1200)
def check_env() -> str:
    """CPU only: build the image and import what extract_mesh_tets.py imports."""
    r = subprocess.run(["python", "-c", "import sys; sys.argv=['x']; import torch, trimesh, open3d; "
                        "from tetranerf.utils.extension import cpp; import diff_gaussian_rasterization, simple_knn; "
                        "from scene import Scene; from gaussian_renderer import render, integrate; "
                        "from utils.tetmesh import marching_tetrahedra; print('imports OK', torch.__version__)"],
                       cwd="/workspace/CoMe", capture_output=True, text=True)
    return r.stdout + r.stderr[-3000:]


@app.function(image=light_image, cpu=1, volumes=VOLUMES, timeout=12 * 3600)
def mesh_all(names: list[str], debug: bool = False) -> list[dict]:
    return [mesh_variant.remote(n, debug) for n in names]


def _assert_workspace() -> None:
    r = subprocess.run([sys.executable, "-m", "modal", "profile", "current"], capture_output=True, text=True)
    if r.stdout.strip() != WORKSPACE_PROFILE:
        raise SystemExit(f"Modal profile is '{r.stdout.strip()}', expected '{WORKSPACE_PROFILE}'")


@app.local_entrypoint()
def run(variants: str = "v1,v2", debug: bool = False):
    _assert_workspace()
    for r in mesh_all.remote(variants.split(","), debug):
        print(json.dumps(r, indent=2))


@app.local_entrypoint()
def download(variants: str = "v1,v2"):
    """export/<run>/step13_come_mesh* -> <local output>/step13_come_mesh (an existing folder gets _v2, _v3 ...)."""
    _assert_workspace()
    for name in variants.split(","):
        run, local = VARIANTS[name]["run"], VARIANTS[name]["local"]
        step = VARIANTS[name].get("step", STEP)
        entries = [e for e in vol.listdir(f"{ROOT_REL}/export/{run}") if e.path.rsplit("/", 1)[-1].startswith(step)]
        for e in entries:
            folder = e.path.rsplit("/", 1)[-1]
            dst = local / folder
            k = 2
            while dst.exists():
                dst = local / f"{folder}_v{k}"
                k += 1
            files = [f for f in vol.listdir(e.path, recursive=True) if f.type == modal.volume.FileEntryType.FILE]
            for f in files:
                out = dst / f.path[len(e.path.lstrip("/")) + 1:]
                out.parent.mkdir(parents=True, exist_ok=True)
                with open(out, "wb") as fh:
                    for chunk in vol.read_file(f.path):
                        fh.write(chunk)
            print(f"[download] {name}: {len(files)} files -> {dst}")


@app.local_entrypoint()
def check():
    print(check_env.remote())


@app.local_entrypoint()
def upload_input(variant: str, path: str):
    """Put a local ply at the volume path of a variant (e.g. the closed napkin of step 14)."""
    _assert_workspace()
    dst = VARIANTS[variant]["ply"][len(VOL):]
    with vol.batch_upload(force=True) as b:
        b.put_file(path, dst)
    size = {e.path.lstrip("/"): e.size for e in vol.listdir(str(Path(dst).parent).replace("\\", "/"))}
    print(f"[upload] {path} -> {dst} ({Path(path).stat().st_size} bytes local; volume: {size})")


@app.function(image=mesh_image, gpu="L40S", memory=65536, volumes=VOLUMES, timeout=3600)
def debug_integrate(name: str, script: str) -> str:
    """Run tools_mesh/debug_come_integrate.py (passed as text) on the model folder of a variant (prepared by mesh_variant)."""
    vol.reload()
    Path("/tmp/debug_come_integrate.py").write_text(script)
    it = VARIANTS[name]["iter"]
    r = subprocess.run(f"CUDA_LAUNCH_BLOCKING=1 python -u /tmp/debug_come_integrate.py -m {ROOT}/mesh/{name} -s {D} "
                       f"--iteration {it}", shell=True, cwd="/workspace/CoMe", capture_output=True, text=True)
    lines = [l for l in (r.stdout + r.stderr).splitlines() if not l.startswith("Reading camera")]
    return "\n".join(lines)[-6000:]


@app.local_entrypoint()
def debug(variant: str = "napkin_v2"):
    _assert_workspace()
    print(debug_integrate.remote(variant, (HERE / "src_v2" / "tools_mesh" / "debug_come_integrate.py").read_text()))


@app.function(image=mesh_image, cpu=2, memory=8192, volumes=VOLUMES, timeout=1800)
def make_view_subset(dst: str, names: list[str], rw_model: str) -> dict:
    """COLMAP dataset with only some images: cameras.bin and points3D.bin unchanged, images.bin filtered, the kept
    images copied. `rw_model` = text of tools/read_write_model.py (COLMAP reader/writer)."""
    import types
    vol.reload()
    rw = types.ModuleType("read_write_model")
    exec(rw_model, rw.__dict__)
    cams, imgs, pts = rw.read_model(f"{D}/sparse/0", ext=".bin")
    keep = {k: im for k, im in imgs.items() if Path(im.name).stem in set(names)}
    out = Path(dst)
    (out / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rw.write_model(cams, keep, pts, str(out / "sparse" / "0"), ext=".bin")
    for im in keep.values():
        shutil.copy2(Path(D) / "images" / im.name, out / "images" / im.name)
    vol.commit()
    return {"dst": dst, "images": len(keep), "of": len(imgs)}


@app.local_entrypoint()
def view_subset(variant: str = "napkin_v2", views_json: str = ""):
    """Build the image subset of a variant (`source`) from a json with a "kept" list of image names."""
    _assert_workspace()
    names = json.loads(Path(views_json).read_text())["kept"]
    rw = (HERE / "src_v2" / "tools" / "read_write_model.py").read_text()
    print(make_view_subset.remote(VARIANTS[variant]["source"], names, rw))
