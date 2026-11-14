"""
Step 13 of Safe-GS on Modal: mesh of the inpainted 3DGS with CoMe extract_mesh_tets.py (PLAN Phase 4)
=====================================================================================================
From test_model/CoMe_Inpainting360GS/example_modal_come_mesh.py. Code = src/come (no Based_Model, no absolute path).
The model folder is CoMe-style: point_cloud/iteration_<it>/point_cloud.ply (= step 11 ply; CoMe load_ply reads
confidence / filter_3D by name and ignores the Inpaint360GS obj_dc_*) + cfg_args, config.json, cameras.json,
mesh_args, rem_args of the CoMe training (scene model M/3dgs_output). -s = scene data S (COLMAP).

Volume `inpaint360gs` (layout of scripts/pipeline_steps.py):
    input ply  come360/export/<Dataset>/<run>/step11_inpaint_3dgs_come_loss/point_cloud_inpainted_iter<it>.ply
               (written by modal_come_pipeline.py; or --source local: uploaded from outputs/<Dataset>/<run>/)
    scene      come360/data/<dk>, come360/output/<dk>/3dgs_output   (modal_come_pipeline.py prep / upload_local)
    output     come360/export/<Dataset>/<run>/step13_come_mesh/mesh_come_iter<it>.ply + done.json (+ logs/)
Then locally: python scripts/run_pipeline.py --dataset <D> --run <run> --step 13b  (crop to the table prism).

  python -m modal run scripts/modal_come_mesh.py::check                                     # CPU: build + imports
  python -m modal run --detach scripts/modal_come_mesh.py::run --dataset LERF --run teatime_napkin [--source local]
  python -m modal run scripts/modal_come_mesh.py::download --dataset LERF --run teatime_napkin
GPU L40S (128 GB RAM: the tetrahedralisation of ~1.9 M Gaussians needs it).
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import modal

import modal_common as mc
import pipeline_steps as ps

mc.utf8_console()

app = modal.App("safe-gs-come-mesh")
vol = modal.Volume.from_name("inpaint360gs", create_if_missing=True)
VOLUMES = {ps.VOL: vol}
STEP = "step13_come_mesh"
mesh_image = mc.come_image(with_tetra=True, extra_sources=("pipeline_steps",))
light_image = modal.Image.debian_slim(python_version="3.10").add_local_python_source("modal_common", "pipeline_steps")


OBJ_STEP = "step13_come_mesh_object"     # mesh of the object Gaussians alone (step 14.1 ply), not the inpainted scene


def _layout(spec: dict, it: int, kind: str = "scene") -> dict:
    p = ps.paths(spec)
    if kind == "object":   # object Gaussians selected by ID from step 7 (src/tools_instantmesh/extract_object.py)
        return {"ply": f"{p['EXPORT']}/{OBJ_STEP}/input/point_cloud_{spec['target_id']}_object.ply",
                "S": p["S"], "M3": f"{p['M']}/3dgs_output",
                "model": f"{ps.ROOT}/mesh_object/{spec['dataset']}/{spec['run']}",
                "out": f"{p['EXPORT']}/{OBJ_STEP}", "out_rel": f"{p['EXPORT_REL']}/{OBJ_STEP}"}
    return {"ply": f"{p['EXPORT']}/step11_inpaint_3dgs_come_loss/point_cloud_inpainted_iter{it}.ply",
            "S": p["S"], "M3": f"{p['M']}/3dgs_output", "model": f"{ps.ROOT}/mesh/{spec['dataset']}/{spec['run']}",
            "out": f"{p['EXPORT']}/{STEP}", "out_rel": f"{p['EXPORT_REL']}/{STEP}"}


@app.function(image=mesh_image, gpu="L40S", cpu=16, memory=131072, volumes=VOLUMES, timeout=5 * 3600, retries=2)
def mesh_run(spec: dict, it: int = 5000, debug: bool = False, force: bool = False, kind: str = "scene") -> dict:
    vol.reload()
    L = _layout(spec, it, kind)
    ply, out, model = Path(L["ply"]), Path(L["out"]), Path(L["model"])
    if (out / "done.json").exists() and not force:
        return json.loads((out / "done.json").read_text())
    if not ply.exists():
        raise RuntimeError(f"{ply} missing (run step 11, or ::run --source local)")
    it_dir = model / "point_cloud" / f"iteration_{it}"
    it_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ply, it_dir / "point_cloud.ply")
    for f in ("cfg_args", "config.json", "cameras.json", "mesh_args", "rem_args"):
        if Path(L["M3"], f).exists():
            shutil.copy2(Path(L["M3"], f), model / f)
    vol.commit()
    res_dir = model / "test" / f"ours_{it}"
    found = sorted(res_dir.glob("mesh_faster_binary_search_*.ply")) if res_dir.exists() else []
    t0 = time.time()
    if not found or force:
        blocking = "CUDA_LAUNCH_BLOCKING=1 " if debug else ""
        with mc.PeriodicCommit(mc.VolumeCommitter(vol)):
            mc.run_logged(f"{blocking}python -u extract_mesh_tets.py -m {model} -s {L['S']} --iteration {it}",
                          out / "logs" / f"{STEP}.log", "/opt/CoMe")
        found = sorted(res_dir.glob("mesh_faster_binary_search_*.ply"))
    mesh = max(found, key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"mesh_come_iter{it}.ply"
    shutil.copy2(mesh, dst)
    import trimesh
    m = trimesh.load(str(mesh), process=False)
    info = {"run": spec["run"], "dataset": spec["dataset"], "source_ply": str(ply), "mesh": mesh.name,
            "output": dst.name, "n_vertices": int(len(m.vertices)), "n_faces": int(len(m.faces)),
            "minutes": round((time.time() - t0) / 60, 1),
            "command": f"extract_mesh_tets.py -m <model> -s {L['S']} --iteration {it} (CoMe defaults)"}
    (out / "done.json").write_text(json.dumps(info, indent=2))
    vol.commit()
    return info


@app.function(image=mesh_image, cpu=2, memory=8192, timeout=1200)
def check_env() -> str:
    import subprocess
    r = subprocess.run(mc.COME_CHECK, shell=True, cwd="/opt/CoMe", capture_output=True, text=True)
    r2 = subprocess.run([sys.executable, "-c", "from tetranerf.utils.extension import cpp; print('tetra OK')"],
                        capture_output=True, text=True)
    return (r.stdout + r.stderr)[-3000:] + (r2.stdout + r2.stderr)[-500:]


def _spec(dataset: str, run: str) -> dict:
    return ps.make_spec(mc.run_config(dataset, run))


@app.local_entrypoint()
def check():
    mc.check_profile()
    print(check_env.remote())


@app.local_entrypoint()
def run(dataset: str = "LERF", run: str = "teatime_napkin", iteration: int = 0, source: str = "export",
        debug: bool = False, force: bool = False):
    """--source export (step 11 already on the Volume) | local (upload outputs/<D>/<run>/step11*/...iter<it>.ply)."""
    mc.check_profile()
    cfg = mc.run_config(dataset, run)
    spec = ps.make_spec(cfg)
    it = iteration or int(cfg.get("mesh_iteration", 5000))
    if source == "local":
        ply = Path(cfg["run_dir"]) / "step11_inpaint_3dgs_come_loss" / f"point_cloud_inpainted_iter{it}.ply"
        if not ply.exists():
            raise SystemExit(f"{ply} missing")
        rel = _layout(spec, it)["ply"][len(ps.VOL):]
        with vol.batch_upload(force=True) as b:
            b.put_file(str(ply), rel)
        print(f"[upload] {ply} -> inpaint360gs:{rel}")
    print(json.dumps(mesh_run.remote(spec, it, debug, force), indent=2))


@app.function(image=light_image, volumes=VOLUMES, timeout=600)
def vol_has(paths: list) -> list:
    vol.reload()
    return [Path(p).exists() for p in paths]


@app.local_entrypoint()
def run_object(dataset: str = "Replica", run: str = "room0_obj1", iteration: int = 30000, debug: bool = False,
               force: bool = False):
    """CoMe extract_mesh_tets.py on the object Gaussians only (outputs/<D>/<run>/step14*/1_object_gaussians/
    point_cloud_<target_id>_object.ply = step 7 Gaussians whose ID == target_id). Uploads it, plus the scene data S
    (step0 images, sparse) and the CoMe configs of step 1 when they are not on the Volume yet.
    --iteration only names the CoMe iteration folder (the ply carries the trained step-7 Gaussians)."""
    mc.check_profile()
    sys.path.insert(0, str(mc.SRC))
    from common.paths import step_dir
    cfg = mc.run_config(dataset, run)
    spec = ps.make_spec(cfg)
    rd = Path(cfg["run_dir"])
    L = _layout(spec, iteration, "object")
    src = step_dir(rd, 14) / "1_object_gaussians" / f"point_cloud_{spec['target_id']}_object.ply"
    if not src.exists():
        raise SystemExit(f"{src} missing (run step 14.1: src/tools_instantmesh/extract_object.py)")
    s0, s1 = step_dir(rd, 0), step_dir(rd, 1)
    has_s, has_m = vol_has.remote([f"{L['S']}/sparse", f"{L['M3']}/cfg_args"])
    rel = lambda p: p[len(ps.VOL):]
    with vol.batch_upload(force=True) as b:
        b.put_file(str(src), rel(L["ply"]))
        if not has_s:
            for sub in ("images", "sparse"):
                b.put_directory(str(s0 / sub), f"{rel(L['S'])}/{sub}")
        if not has_m:
            for f in ("cfg_args", "config.json", "cameras.json", "mesh_args", "rem_args"):
                if (s1 / f).exists():
                    b.put_file(str(s1 / f), f"{rel(L['M3'])}/{f}")
    print(f"[upload] {src.name} (+ scene data: {not has_s}, configs: {not has_m}) -> inpaint360gs:{rel(L['out'])}")
    print(json.dumps(mesh_run.remote(spec, iteration, debug, force, "object"), indent=2))


@app.local_entrypoint()
def download(dataset: str = "LERF", run: str = "teatime_napkin", overwrite: bool = False, kind: str = "scene"):
    mc.check_profile()
    cfg = mc.run_config(dataset, run)
    spec = ps.make_spec(cfg)
    mc.download_dir(vol, _layout(spec, 0, kind)["out_rel"], Path(cfg["run_dir"]) / (OBJ_STEP if kind == "object" else STEP),
                    overwrite)
