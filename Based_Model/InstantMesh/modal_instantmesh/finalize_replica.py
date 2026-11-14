"""
Step 3 (local CPU): choose the final InstantMesh mesh of each Replica object + copy the reused Safe-GS steps.

For each outputs/Replica/Object_mesh_SAM_InstantMesh/<scene>/<obj>/:
  * step9_instantmesh_zero123_lrm/<frame>/done.json (one InstantMesh run per input frame) -> choose the frame whose
    mesh is cleanest: largest share of faces in the biggest connected component, then fewest components, then the
    largest input mask (frames.json order). No GT is used for the choice (fair baseline).
  * step10_instantmesh_final_mesh/: <obj>_instantmesh.obj (largest component, vertex colours, InstantMesh's normalised
    frame: no scene scale/pose), multiview.png (Zero123++ 6 views of that frame), selection.json
  * step0_colmap .. step7_id_distill copied from outputs/Replica/ours/<run>/ (the masks of step 8 = ours step 5/6 SAM);
    files already present are not copied again.

    python modal_instantmesh/finalize_replica.py [--runs room0_obj1] [--no-copy-steps]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import trimesh

SAFE_GS = Path(__file__).resolve().parents[3]
OURS = SAFE_GS / "outputs" / "Replica" / "ours"
OUT_ROOT = SAFE_GS / "outputs" / "Replica" / "Object_mesh_SAM_InstantMesh"
RUNS = ["room0_obj1", "room0_obj2", "office0_obj1", "office0_obj2"]
STEP1, STEP2, STEP3 = "step8_instantmesh_input_sam", "step9_instantmesh_zero123_lrm", "step10_instantmesh_final_mesh"
REUSED = ["step0_colmap", "step1_come_3dgs", "step2_come_renders", "step3_object_reference", "step4_renderer_check",
          "step5_mask2d_hqsam", "step6_sync_mask", "step7_id_distill"]


def copy_steps(run: str, dst: Path) -> int:
    n = 0
    for s in REUSED:
        sd = OURS / run / s
        if not sd.exists():
            print(f"[warn] {run}: {s} missing in ours")
            continue
        for f in sd.rglob("*"):
            t = dst / s / f.relative_to(sd)
            if f.is_file() and not t.exists():
                t.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, t)
                n += 1
    return n


def finalize(run: str, copy: bool) -> None:
    scene, obj = run.split("_")
    base = OUT_ROOT / scene / obj
    order = [r["frame"] for r in json.loads((base / STEP1 / "frames.json").read_text())["frames"]]
    done = {d.parent.name: json.loads(d.read_text()) for d in (base / STEP2).glob("*/done.json")}
    if not done:
        raise SystemExit(f"{run}: no InstantMesh result in {base / STEP2} (run ::download)")
    rank = sorted(done, key=lambda f: (-done[f]["stats"]["largest_component_face_share"],
                                       done[f]["stats"]["components"], order.index(f) if f in order else 99))
    best = rank[0]
    run_dir = base / STEP2 / best
    m = trimesh.load(run_dir / done[best]["mesh"], process=False, force="mesh")
    parts = m.split(only_watertight=False)
    big = max(parts, key=lambda p: len(p.faces)) if len(parts) else m
    out = base / STEP3
    out.mkdir(parents=True, exist_ok=True)
    big.export(out / f"{obj}_instantmesh.obj")
    mv = next((run_dir / "instant-mesh-large" / "images").glob("*.png"), None)
    if mv:
        shutil.copy2(mv, out / "multiview.png")
    sel = {"run": run, "chosen_frame": best, "rule": "max largest_component_face_share, then min components, "
           "then largest input mask", "kept": "largest connected component",
           "final": {"vertices": int(len(big.vertices)), "faces": int(len(big.faces)),
                     "watertight": bool(big.is_watertight), "extents": [round(float(x), 4) for x in big.extents]},
           "candidates": {f: {"seconds": done[f]["seconds"], "gpu": done[f]["gpu"], **done[f]["stats"]} for f in rank},
           "frame": "InstantMesh normalised object frame (no Replica scale/pose)"}
    (out / "selection.json").write_text(json.dumps(sel, indent=2))
    n = copy_steps(run, base) if copy else 0
    print(f"{run}: frame {best} -> {out / f'{obj}_instantmesh.obj'} ({sel['final']['vertices']} v, "
          f"watertight {sel['final']['watertight']}); ours steps 0-7: {n} files copied")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=",".join(RUNS))
    ap.add_argument("--no-copy-steps", action="store_true")
    a = ap.parse_args()
    for run in a.runs.split(","):
        finalize(run, not a.no_copy_steps)


if __name__ == "__main__":
    main()
