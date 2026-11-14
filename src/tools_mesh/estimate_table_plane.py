"""Step 13p (local CPU): table / support plane of a run -> <run>/table_plane.json (read by crop_table.py (13b),
extract_object.py (14.1), place_on_table.py (15), evaluate_all_metrics.py through mesh_io.find_plane).

Automatic, from data of the run only (no manual click):
  1. object  = Gaussians of the removed object (step 8: 2_object_removal/removed_object_id<id>.ply, or --object_ply):
     robust centre c and extent (max range of the 2-98 % box)
  2. up hint = mean camera up of step1 cameras.json (OpenCV convention: up = -R[:, 1])
  3. RANSAC on the vertices of the full step-13 mesh (inpainted scene, object removed) within 3 x extent of c:
     plane with most inliers (|dist| < tol) whose normal is within 35 deg of the up hint and which lies below c
     (0 < height of c <= 1.5 x extent); least-squares refinement; normal oriented towards the cameras
  4. support region = 2D grid (cell = extent / 20) of the inliers within --region_radius x extent, connected component that
     contains (or is nearest to) the object footprint -> convex hull `hull_2d` (+ margin)
  5. height_range = [-below, max(extent, object top + 5 % extent)]
Output keys = those of the teatime reference src/tools_mesh/table_plane.json: plane_normal, plane_d, basis_e1,
basis_e2, hull_2d, height_range (+ method, object, inliers, ...).

  python tools_mesh/estimate_table_plane.py --run <outputs/<Dataset>/<run>> [--object_id 44] [--mesh <full mesh>]
"""
from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mesh_io as mio  # noqa: E402


def _xyz_ply(path: Path) -> np.ndarray:
    from plyfile import PlyData
    v = PlyData.read(str(path)).elements[0]
    return np.stack([np.asarray(v[k], np.float64) for k in ("x", "y", "z")], 1)


def _find(run: Path, pattern: str):
    c = sorted(run.glob(pattern), key=lambda p: p.stat().st_mtime)
    return c[-1] if c else None


def _cameras(run: Path):
    for p in sorted(run.glob("step1*/cameras.json")):
        cams = json.loads(p.read_text())
        C = np.array([c["position"] for c in cams], float)
        up = np.array([-np.array(c["rotation"])[:, 1] for c in cams], float).mean(0)
        return C, up / np.linalg.norm(up)
    raise SystemExit(f"no step1*/cameras.json in {run}")


def _basis(n):
    a = np.eye(3)[np.argmin(np.abs(n))]
    e1 = np.cross(n, a); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return e1, e2


def main():
    ap = ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--object_id", "--object-id", type=int, default=None,
                    help="id of removed_object_id<id>.ply (default: any removed_object_id*.ply of step 8)")
    ap.add_argument("--object_ply", default=None, help="explicit object Gaussians (xyz used)")
    ap.add_argument("--mesh", default=None, help="full step-13 mesh (default: newest step13*/mesh_come_*iter*.ply)")
    ap.add_argument("--tol", type=float, default=0.015, help="inlier distance, fraction of the object extent")
    ap.add_argument("--max_tilt_deg", type=float, default=35.0)
    ap.add_argument("--margin", type=float, default=0.02, help="hull margin, fraction of the object extent")
    ap.add_argument("--below", type=float, default=0.035, help="crop depth below the plane, fraction of extent")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--region_radius", type=float, default=6.0, help="search radius of the support region (x extent)")
    ap.add_argument("--closing", type=int, default=1, help="grid closing iterations (bridges small holes)")
    ap.add_argument("--out", default=None, help="default: <run>/table_plane.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="overwrite an existing table_plane.json")
    args = ap.parse_args()
    run = Path(args.run)
    dst = Path(args.out) if args.out else run / "table_plane.json"
    if dst.exists() and not args.force:
        raise SystemExit(f"{dst} exists - not overwriting (--force to redo)")
    rng = np.random.default_rng(args.seed)

    obj = Path(args.object_ply) if args.object_ply else _find(
        run, f"step8*/2_object_removal/removed_object_id{args.object_id if args.object_id is not None else '*'}.ply")
    if obj is None:
        raise SystemExit("no object Gaussians: run step 8 (export removed_object_id<id>.ply) or pass --object_ply")
    mesh = Path(args.mesh) if args.mesh else _find(run, "step13*/mesh_come_*iter*.ply")
    if mesh is None:
        raise SystemExit("no full step-13 mesh (step13_come_mesh/mesh_come_*iter*.ply): run step 13 or pass --mesh")

    O = _xyz_ply(obj)
    lo, hi = np.percentile(O, 2, 0), np.percentile(O, 98, 0)
    O = O[np.all((O >= lo) & (O <= hi), 1)]
    c = np.median(O, 0)
    ext = float(np.max(hi - lo))
    tol = args.tol * ext
    C, up = _cameras(run)
    V, _, _ = mio.read_mesh(mesh)
    near = V[np.linalg.norm(V - c, axis=1) < 3 * ext]
    if len(near) < 100:
        raise SystemExit(f"only {len(near)} mesh vertices near the object")
    sample = near[rng.choice(len(near), min(len(near), 400_000), replace=False)]

    best, best_n = None, -1
    cos_tilt = np.cos(np.radians(args.max_tilt_deg))
    for _ in range(args.iters):
        p = sample[rng.choice(len(sample), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n /= nn
        if n @ up < 0:
            n = -n
        if n @ up < cos_tilt:
            continue
        d = -n @ p[0]
        hc = n @ c + d                                   # height of the object centre above the plane
        if not (0 < hc <= 1.5 * ext):
            continue
        k = int((np.abs(sample @ n + d) < tol).sum())
        if k > best_n:
            best, best_n = (n, d), k
    if best is None:
        raise SystemExit("RANSAC found no plane below the object (check --max_tilt_deg / the object ply)")
    n, d = best
    inl = near[np.abs(near @ n + d) < tol]
    for _ in range(3):                                   # least-squares refinement
        m = inl.mean(0)
        _, _, vt = np.linalg.svd(inl - m, full_matrices=False)
        n = vt[2] if vt[2] @ up >= 0 else -vt[2]
        d = -n @ m
        inl = near[np.abs(near @ n + d) < tol]
    if np.mean((C - (-d * n)) @ n) < 0:                  # towards the cameras
        n, d = -n, -d
    e1, e2 = _basis(n)
    plane = {"n": n, "d": float(d), "e1": e1, "e2": e2}

    # support region: connected component of the inlier grid at the object footprint
    from scipy import ndimage
    from scipy.spatial import ConvexHull
    allv = V[np.linalg.norm(V - c, axis=1) < args.region_radius * ext]
    uvh = mio.to_uvh(plane, allv)
    sup = uvh[np.abs(uvh[:, 2]) < tol][:, :2]
    cell = ext / 20
    o0 = sup.min(0)
    ij = np.floor((sup - o0) / cell).astype(int)
    grid = np.zeros(ij.max(0) + 1, bool)
    grid[ij[:, 0], ij[:, 1]] = True
    grid = ndimage.binary_closing(grid, iterations=args.closing) if args.closing else grid
    lab, _ = ndimage.label(grid)
    cuv = mio.to_uvh(plane, c[None])[0, :2]
    ci = np.floor((cuv - o0) / cell).astype(int)
    occ = np.argwhere(lab > 0)
    tgt = occ[np.argmin(np.linalg.norm(occ - ci, axis=1))]
    comp = lab[tgt[0], tgt[1]]
    pts = sup[lab[ij[:, 0], ij[:, 1]] == comp]
    hull = pts[ConvexHull(pts).vertices]
    hull = hull + args.margin * ext * (hull - hull.mean(0)) / np.linalg.norm(hull - hull.mean(0), axis=1, keepdims=True)
    top = float(np.max(mio.to_uvh(plane, O)[:, 2]))
    out = {"source": mesh.name, "method": "estimate_table_plane.py (RANSAC below the removed object)",
           "object": obj.name, "object_extent": round(ext, 5), "tol": round(tol, 5),
           "plane_normal": n.tolist(), "plane_d": float(d), "basis_e1": e1.tolist(), "basis_e2": e2.tolist(),
           "hull_2d": hull.tolist(), "height_range": [-args.below * ext, max(ext, top + 0.05 * ext)],
           "object_centre_height": round(float(n @ c + d), 5), "inliers_near_object": int(len(inl)),
           "support_region_points": int(len(pts))}
    dst.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "hull_2d"}, indent=1))
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
