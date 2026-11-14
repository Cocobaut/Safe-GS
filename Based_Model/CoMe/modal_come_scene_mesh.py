"""
CoMe baseline (scene-level mesh) on Modal: extract_mesh_tets.py on the CoMe 3DGS of step 1 (no object removal)
================================================================================================================
Original CoMe only extracts a scene-level mesh, so one result per scene (not per object run):
    outputs/<Dataset>/CoMe/<scene>/step0_colmap   COLMAP (images + sparse/0)          = step 0 of the pipeline
    outputs/<Dataset>/CoMe/<scene>/step1_3DGS     CoMe model (point_cloud/iteration_30000, cfg_args, ...) + renders/
    outputs/<Dataset>/CoMe/<scene>/step2_Mesh     mesh_come_iter<it>.ply + done.json + logs/   <- this script
Volume `safe-gs-come`: <key>/scene_mesh/{data, model, out}, <key> = <dataset lower>/<scene> (e.g. replica/room0).
step0/step1 are uploaded from the local folders above (office0 is no longer on any Volume).
Image / helpers = scripts/modal_common.py (CoMe code mounted from src/come; extract_mesh_tets.py there is identical
to Based_Model/CoMe/extract_mesh_tets.py = r4dl/CoMe @ caf62f6).

  $env:PYTHONIOENCODING="utf-8"                                  # from F:\\Safe-GS (any cwd works)
  python -m modal run --detach Based_Model/CoMe/modal_come_scene_mesh.py::run --dataset Replica --scene room0
  python -m modal run Based_Model/CoMe/modal_come_scene_mesh.py::download --dataset Replica --scene room0
"""
from __future__ import annotations

import json
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

app = modal.App("safe-gs-come-scene-mesh")
vol = modal.Volume.from_name("safe-gs-come", create_if_missing=True)
mesh_image = mc.come_image(with_tetra=True)
STEP = "step2_Mesh"
SMALL_FILES = ("cfg_args", "config.json", "cameras.json", "mesh_args", "rem_args")


def _local(dataset: str, scene: str) -> Path:
    return mc.OUTPUTS / dataset / "CoMe" / scene


def _remote(dataset: str, scene: str) -> str:
    return f"{mc.dataset_key(dataset, scene)}/scene_mesh"


@app.function(image=mesh_image, gpu="L40S", cpu=16, memory=131072, volumes={mc.VOL: vol}, timeout=5 * 3600,
              retries=1)
def mesh_scene(rel: str, it: int, force: bool = False) -> dict:
    vol.reload()
    root = Path(mc.VOL) / rel
    data, model, out = root / "data", root / "model", root / "out"
    if (out / "done.json").exists() and not force:
        return json.loads((out / "done.json").read_text())
    ply = model / "point_cloud" / f"iteration_{it}" / "point_cloud.ply"
    if not ply.exists():
        raise RuntimeError(f"{ply} missing (upload step1_3DGS first)")
    res_dir = model / "test" / f"ours_{it}"
    found = sorted(res_dir.glob("mesh_faster_binary_search_*.ply")) if res_dir.exists() else []
    t0 = time.time()
    if not found or force:
        with mc.PeriodicCommit(mc.VolumeCommitter(vol)):
            mc.run_logged(f"python -u extract_mesh_tets.py -m {model} -s {data} --iteration {it}",
                          out / "logs" / f"{STEP}.log", "/opt/CoMe")
        found = sorted(res_dir.glob("mesh_faster_binary_search_*.ply"))
    mesh = max(found, key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"mesh_come_iter{it}.ply"
    shutil.copy2(mesh, dst)
    import trimesh
    m = trimesh.load(str(mesh), process=False)
    info = {"method": "CoMe (original, scene-level mesh, no object removal)", "key": rel, "iteration": it,
            "mesh": mesh.name, "output": dst.name, "n_vertices": int(len(m.vertices)), "n_faces": int(len(m.faces)),
            "minutes": round((time.time() - t0) / 60, 1),
            "command": f"extract_mesh_tets.py -m <model> -s <step0_colmap> --iteration {it} (CoMe defaults)"}
    (out / "done.json").write_text(json.dumps(info, indent=2))
    vol.commit()
    return info


@app.local_entrypoint()
def run(dataset: str = "Replica", scene: str = "room0", iteration: int = 30000, force: bool = False,
        skip_upload: bool = False):
    mc.check_profile()
    loc, rel = _local(dataset, scene), _remote(dataset, scene)
    s0, s1 = loc / "step0_colmap", loc / "step1_3DGS"
    ply = s1 / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not (s0 / "sparse" / "0").exists() or not ply.exists():
        raise SystemExit(f"missing {s0}/sparse/0 or {ply}")
    if not skip_upload:
        with vol.batch_upload(force=True) as b:
            b.put_directory(str(s0 / "images"), f"/{rel}/data/images")
            b.put_directory(str(s0 / "sparse"), f"/{rel}/data/sparse")
            b.put_directory(str(ply.parent), f"/{rel}/model/point_cloud/iteration_{iteration}")
            for f in SMALL_FILES:
                if (s1 / f).exists():
                    b.put_file(str(s1 / f), f"/{rel}/model/{f}")
        print(f"[upload] {s0} + {s1} -> safe-gs-come:{rel}/{{data,model}}")
    print(json.dumps(mesh_scene.remote(rel, iteration, force), indent=2))


@app.local_entrypoint()
def download(dataset: str = "Replica", scene: str = "room0", overwrite: bool = False):
    mc.check_profile()
    mc.download_dir(vol, f"{_remote(dataset, scene)}/out", _local(dataset, scene) / STEP, overwrite)
