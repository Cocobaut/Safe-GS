"""Step 13b (local CPU): crop the full CoMe mesh of the inpainted scene to the table (prism of table_plane.json).

The prism = the 2D convex hull `hull_2d` of the table in the plane basis (e1, e2) x the height band `height_range` above
the plane n.X + d = 0. A face is kept when all three vertices are inside the prism; unused vertices are dropped.
Rebuilds step13_come_mesh/mesh_come_v2_table.ply + mesh_come_v2_table_crop.json of the reference run (the original
crop script was not kept; this follows the json it wrote: source, prism, plane, height_range, vertices, faces).

  python tools_mesh/crop_table.py --run <outputs/LERF/teatime_napkin> [--mesh <full mesh>] [--plane table_plane.json]
"""
from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mesh_io as mio  # noqa: E402


def inside_hull(P2, hull):
    """Points (N,2) inside the convex polygon `hull` (K,2) (either orientation), boundary included."""
    H = np.asarray(hull, float)
    if np.allclose(H[0], H[-1]):
        H = H[:-1]
    a, b = H, np.roll(H, -1, 0)
    cross = (b[:, 0] - a[:, 0])[None] * (P2[:, 1:2] - a[None, :, 1]) - \
            (b[:, 1] - a[:, 1])[None] * (P2[:, 0:1] - a[None, :, 0])
    return (cross >= -1e-12).all(1) | (cross <= 1e-12).all(1)


def main():
    ap = ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--mesh", default=None, help="full mesh (default: newest step13_come_mesh/mesh_come_*_iter*.ply)")
    ap.add_argument("--plane", default=None, help="table_plane.json with hull_2d + height_range")
    ap.add_argument("--out", default=None, help="default: <mesh dir>/<mesh stem without _iterN>_table.ply")
    ap.add_argument("--force", action="store_true", help="overwrite an existing table mesh")
    ap.add_argument("--max_edge_k", type=float, default=0.0,
                    help="optional: drop faces whose longest edge > k x median longest edge (0 = keep all, default). "
                         "Off by default: on Replica office0 k=10 cut 6-11%% holes into flat seen surfaces (sofa, walls), "
                         "where CoMe uses larger triangles; the step-13 mesh itself has no holes in the camera views")
    ap.add_argument("--clean_full", action="store_true",
                    help="also write the full mesh without long faces: <mesh dir>/mesh_clean_<mesh name> (for viewing)")
    args = ap.parse_args()
    run = Path(args.run)
    s13 = run / "step13_come_mesh"
    cands = [q for q in s13.glob("mesh_come_*iter*.ply") if not q.stem.endswith("_table")]
    if not args.mesh and not cands:
        raise SystemExit(f"no full mesh mesh_come_*iter*.ply in {s13}: run step 13 (and its download) first")
    mesh = Path(args.mesh) if args.mesh else max(cands, key=lambda p: p.stat().st_mtime)
    plane_path = Path(args.plane) if args.plane else mio.find_plane(run)
    cfg = json.loads(plane_path.read_text(encoding="utf-8"))
    plane = mio.load_plane(plane_path)
    stem = mesh.stem.rsplit("_iter", 1)[0]
    out = Path(args.out) if args.out else mesh.with_name(f"{stem}_table.ply")
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists - not overwriting (--force to redo)")

    V, F, C = mio.read_mesh(mesh)
    long_info = None
    if args.max_edge_k > 0:
        e = np.max(np.stack([np.linalg.norm(V[F[:, i]] - V[F[:, (i + 1) % 3]], axis=1) for i in range(3)], 1), 1)
        med = float(np.median(e))
        short = e <= args.max_edge_k * med
        long_info = {"max_edge_k": args.max_edge_k, "median_longest_edge": med,
                     "long_faces_removed_full_mesh": int((~short).sum()), "long_faces_frac": float((~short).mean())}
        F = F[short]
        if args.clean_full:
            res_full = mio.compact(V, F, C)
            full_out = mesh.with_name(f"mesh_clean_{mesh.name}")
            mio.write_mesh(full_out, res_full[0], res_full[1], res_full[2] if C is not None else None)
            long_info["clean_full_mesh"] = str(full_out)
    uvh = mio.to_uvh(plane, V)
    lo, hi = cfg["height_range"]
    ok = inside_hull(uvh[:, :2], cfg["hull_2d"]) & (uvh[:, 2] >= lo) & (uvh[:, 2] <= hi)
    Fk = F[ok[F].all(1)]
    res = mio.compact(V, Fk, C)
    mio.write_mesh(out, res[0], res[1], res[2] if C is not None else None)
    info = {"source": str(mesh), "prism": str(plane_path), "plane_normal": cfg["plane_normal"], "plane_d": cfg["plane_d"],
            "height_range": cfg["height_range"], "vertices": int(len(res[0])), "faces": int(len(res[1])),
            "long_face_filter": long_info}
    out.with_name(out.stem + "_crop.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
