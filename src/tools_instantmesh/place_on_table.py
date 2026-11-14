"""Step 15 (local CPU): put the InstantMesh object (step 14) on the inpainted table (step 13). Nothing is cut or stitched.

1. table  = step13_come_mesh/*_table.ply (--table), unchanged
   object = step14_instantmesh/5_final_mesh/mesh_object_instantmesh.ply (legacy name mesh_napkin_instantmesh.ply is
            still read), unchanged except one rigid translation
2. for every object vertex: its height above the table top straight under it (ray from below the table along +n,
   open3d BVH; robust to the small floating fragments of the step-13 mesh)
3. one translation along the table normal n so that the object rests on the table: lowest clearance = --gap
   (lifts it if part of the predicted underside went into the table, lowers it if it floated)
4. mesh_object_placed.ply = the translated object
   scene_scene_demo_object.ply = table + object as two separate pieces in one file

  python tools_instantmesh/place_on_table.py --run <outputs/LERF/teatime_napkin> [--plane table_plane.json]
"""
from __future__ import annotations

import json
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import open3d as o3d

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools_mesh"))
import mesh_io as mio  # noqa: E402

TABLE_COLOUR = [0.72, 0.62, 0.48]          # table mesh has no vertex colours (same colour as basic step 15)

# output / input names (object-agnostic). Step-14 results made before the rename (still on the Modal Volume
# `instantmesh`, re-downloaded with ::download) carry the napkin name -> still read as a fallback.
OBJECT_MESH_14 = "mesh_object_instantmesh.ply"
OBJECT_MESH_14_LEGACY = "mesh_napkin_instantmesh.ply"
OBJECT_PLACED = "mesh_object_placed.ply"
SCENE_MERGED = "scene_scene_demo_object.ply"


def main():
    ap = ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--gap", type=float, default=0.001, help="clearance between object and table (scene units)")
    ap.add_argument("--plane", default=None,
                    help="table_plane.json (default: <run>/table_plane.json, <run>/step13_come_mesh/, tools_mesh/)")
    ap.add_argument("--force", action="store_true", help="redo an existing step15_object_mesh_merge")
    ap.add_argument("--contact_frac", type=float, default=0.15,
                    help="only object vertices in the lowest contact_frac x object height (and support surfaces at most "
                         "that high) decide the translation; a fragment above the object can no longer lift it")
    ap.add_argument("--contact_pct", type=float, default=5.0,
                    help="percentile of the contact clearances put at --gap (0 = strict minimum, old behaviour)")
    ap.add_argument("--table", default=None,
                    help="table mesh (default: step13_come_mesh/mesh_come_v2_table.ply, else newest *_table.ply)")
    args = ap.parse_args()
    run = Path(args.run)
    out = run / "step15_object_mesh_merge"
    if (out / "done.json").exists():
        if not args.force:
            raise SystemExit(f"{out} already finished - not overwriting (--force to redo)")
        (out / "done.json").unlink()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    plane = mio.load_plane(args.plane, run_dir=run)
    n = plane["n"]                                                     # table normal (up)

    s13 = run / "step13_come_mesh"
    table = Path(args.table) if args.table else (s13 / "mesh_come_v2_table.ply" if (s13 / "mesh_come_v2_table.ply").exists()
                                                 else max(s13.glob("*_table.ply"), key=lambda q: q.stat().st_mtime))
    TV, TF, _ = mio.read_mesh(table)
    fin = run / "step14_instantmesh" / "5_final_mesh"
    obj14 = fin / OBJECT_MESH_14 if (fin / OBJECT_MESH_14).exists() else fin / OBJECT_MESH_14_LEGACY
    NV, NF, NC = mio.read_mesh(obj14)
    if NC is None:
        NC = np.full((len(NV), 3), 0.9)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(TV.astype(np.float32)), o3d.core.Tensor(TF.astype(np.uint32)))

    ext = float(np.ptp(mio.to_uvh(plane, NV)[:, :2], 0).max())

    def clearance(V):
        """height of each vertex above the table surface straight under it, measured with a ray from BELOW the table
        going up (first hit = table top). Rays from the vertex itself would also hit the small floating fragments of
        the step-13 mesh above the table and count the vertex as 'inside'."""
        # start every ray 0.5 x extent below the table PLANE (not below the vertex): for a tall object (bag, bottle) a
        # start point 0.5 x extent below a high vertex would still be above the table and never hit it.
        # For a flat object (napkin) this gives the same first hit as before.
        below = (V @ n + plane["d"]) + 0.5 * ext
        start = V - below[:, None] * n
        r = np.concatenate([start, np.tile(n, (len(V), 1))], 1).astype(np.float32)
        t = scene.cast_rays(o3d.core.Tensor(r))["t_hit"].numpy()
        return below - t, np.isfinite(t)

    c0, hit0 = clearance(NV)
    # contact = the lowest part of the object only, and only support surfaces near it: through a hole in the step-13
    # surface the first hit can be a leftover fragment far above (Replica office0 bin: rim remnant 37 cm up -> the bin
    # was lifted 10-14 cm). Flat objects (e.g. napkin): the band covers the whole object, same result as before.
    hN = NV @ n + plane["d"]
    hgt = float(np.ptp(hN))
    band = max(args.contact_frac * hgt, 0.02 * ext)
    support_h = hN - c0                                                # height of the hit surface above the plane
    contact = hit0 & (hN <= hN.min() + band) & (support_h <= hN.min() + band)
    use = contact if contact.any() else hit0
    # robust minimum: a few contact vertices may sink into fuzzy support (rug fibres, mesh noise) instead of the single
    # highest fragment deciding the lift (Replica office0: rug mesh 1.4 cm median, 3.9 cm max under the bin)
    shift = float(args.gap - np.percentile(c0[use], args.contact_pct))  # move along n by this much
    V2 = NV + shift * n
    c1, hit1 = clearance(V2)

    mio.write_mesh(out / OBJECT_PLACED, V2, NF, NC)
    scene_V = np.concatenate([TV, V2])
    scene_F = np.concatenate([TF, NF + len(TV)])
    scene_C = np.concatenate([np.tile(TABLE_COLOUR, (len(TV), 1)), NC])
    mio.write_mesh(out / SCENE_MERGED, scene_V, scene_F, scene_C)

    pct = lambda c: [round(float(np.percentile(c, q)), 5) for q in (0, 5, 50)]
    res = {"table": {"file": table.relative_to(run).as_posix() if table.is_relative_to(run) else str(table), "vertices": int(len(TV)), "faces": int(len(TF)),
                     "changed": False},
           "object": {"file": obj14.relative_to(run).as_posix(), "vertices": int(len(NV)),
                      "faces": int(len(NF)), "edges": mio.edge_stats(NF), "changed": "rigid translation along n only"},
           "translation_along_n": round(shift, 5),
           "contact": {"contact_frac": args.contact_frac, "contact_pct": args.contact_pct, "band": round(band, 5), "object_height": round(hgt, 5),
                       "contact_vertices": int(contact.sum()), "fallback_all_vertices": bool(not contact.any())},
           "translation_relative_to_object_width": round(shift / ext, 4),
           "before": {"vertices_inside_table": int(((c0 < 0) & hit0).sum()), "clearance_min_p05_p50": pct(c0[hit0])},
           "after": {"vertices_inside_table": int(((c1 < 0) & hit1).sum()), "clearance_min_p05_p50": pct(c1[hit1]),
                     "vertices_with_table_below": round(float(hit1.mean()), 4)},
           "gap": args.gap, "seconds": round(time.time() - t0, 1),
           "outputs": [OBJECT_PLACED, f"{SCENE_MERGED} (table + object, two separate pieces)"]}
    (out / "summary.json").write_text(json.dumps(res, indent=2))
    sunk = (c1 < 0) & hit1 & contact
    res["after"]["contact_vertices_inside_table"] = int(sunk.sum())
    res["after"]["contact_sunk_frac"] = round(float(sunk.sum() / max(int(contact.sum()), 1)), 4)
    (out / "summary.json").write_text(json.dumps(res, indent=2))
    (out / "done.json").write_text(json.dumps({"ok": res["after"]["contact_sunk_frac"] <= args.contact_pct / 100 + 0.01,
                                               "scene": SCENE_MERGED}, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
