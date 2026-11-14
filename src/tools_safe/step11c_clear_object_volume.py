"""Step 11c (local CPU): delete the Gaussians that step 11 (3D inpainting) put INSIDE the volume of the removed object.

Why: Inpaint360GS initialises new Gaussians from the inpainted depth of virtual views. Seen from the cameras they are
coloured like the background, so the renders of step 12 look clean, but many of them float where the object used to be
(Replica office0 bin: ~5.4k new Gaussians 3-45 cm above the floor). Step 13 (CoMe tetra mesh) turns them into a bump on
the support surface and step 15 then puts the object mesh on top of that bump (bin 14.5 cm above the floor).

Volume to clear (data of the run only, no GT):
  object  = step 8 removed_object_id<id>.ply (Gaussians with opacity > --min_opacity)
  plane   = <run>/table_plane.json (step 13p) if present, else a RANSAC fit on the opaque Gaussians of the
            step-8 scene (object removed) around the object, normal within --max_tilt_deg of the mean camera up
  prism   = convex hull of the object footprint on the plane, grown by --margin (fraction of the object extent),
            heights (--keep_above, object top + --top_margin * extent]
Gaussians of the step-11 ply whose centre is inside the prism are dropped. The step-11 ply is rewritten in place;
the original is kept as point_cloud_inpainted_iter<it>_raw.ply (re-running starts again from _raw).

  python tools_safe/step11c_clear_object_volume.py --run <outputs/<Dataset>/<run>> --object_id 19
Then step 13 must read the cleaned ply:  modal_come_mesh.py::run --source local --force
"""
from __future__ import annotations

import json
import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools_mesh"))
import mesh_io as mio  # noqa: E402
from crop_table import inside_hull  # noqa: E402


def _xyz_op(path: Path):
    v = PlyData.read(str(path))["vertex"]
    xyz = np.stack([np.asarray(v[k], np.float64) for k in ("x", "y", "z")], 1)
    op = 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float64)))
    return xyz, op


def _camera_up(run: Path):
    for p in sorted(run.glob("step1*/cameras.json")):
        cams = json.loads(p.read_text())
        up = np.array([-np.array(c["rotation"])[:, 1] for c in cams], float).mean(0)
        return up / np.linalg.norm(up)
    raise SystemExit(f"no step1*/cameras.json in {run}")


def _basis(n):
    a = np.eye(3)[np.argmin(np.abs(n))]
    e1 = np.cross(n, a); e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def fit_plane(scene_xyz, obj_xyz, up, ext, max_tilt_deg, iters=3000, seed=0):
    """Support plane below the object from the scene Gaussians around it (normal towards `up`)."""
    c = np.median(obj_xyz, 0)
    near = scene_xyz[np.linalg.norm(scene_xyz - c, axis=1) < 3 * ext]
    lowest = np.percentile(obj_xyz @ up, 2)
    near = near[np.abs(near @ up - lowest) < 0.5 * ext]           # around the object's base
    if len(near) < 50:
        raise SystemExit("too few Gaussians around the object to fit a plane: run 13p first or pass --plane")
    rng = np.random.default_rng(seed)
    tol, cos_max = 0.01 * ext + 0.003, np.cos(np.radians(max_tilt_deg))
    best, best_n = None, -1
    for _ in range(iters):
        p = near[rng.choice(len(near), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        n = n if n @ up > 0 else -n
        if n @ up < cos_max:
            continue
        inl = np.abs(near @ n - p[0] @ n) < tol
        if inl.sum() > best_n:
            best_n, best = inl.sum(), inl
    P = near[best]
    ctr = P.mean(0)
    n = np.linalg.svd(P - ctr)[2][-1]
    n = n if n @ up > 0 else -n
    e1, e2 = _basis(n)
    return {"n": n, "d": float(-n @ ctr), "e1": e1, "e2": e2, "inliers": int(best_n)}


def main():
    ap = ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--object_id", "--object-id", type=int, required=True)
    ap.add_argument("--iteration", type=int, default=5000)
    ap.add_argument("--plane", default=None, help="table_plane.json (default: <run>/table_plane.json, else RANSAC fit)")
    ap.add_argument("--min_opacity", type=float, default=0.1, help="object Gaussians that define the footprint")
    ap.add_argument("--margin", type=float, default=0.05, help="footprint growth, fraction of the object extent")
    ap.add_argument("--keep_above", type=float, default=0.02, help="keep everything below this height (m) = surface")
    ap.add_argument("--top_margin", type=float, default=0.10, help="clear up to object top + this x extent")
    ap.add_argument("--max_tilt_deg", type=float, default=35.0)
    args = ap.parse_args()

    run = Path(args.run)
    s11 = run / "step11_inpaint_3dgs_come_loss"
    ply = s11 / f"point_cloud_inpainted_iter{args.iteration}.ply"
    raw = s11 / f"point_cloud_inpainted_iter{args.iteration}_raw.ply"
    if not raw.exists():
        if not ply.exists():
            raise SystemExit(f"{ply} missing: download step 11 first")
        shutil.copy2(ply, raw)
    obj_ply = run / "step8_removal_virtual_cam" / "2_object_removal" / f"removed_object_id{args.object_id}.ply"
    if not obj_ply.exists():
        raise SystemExit(f"{obj_ply} missing: download step 8 first")

    oxyz, oop = _xyz_op(obj_ply)
    oxyz = oxyz[oop > args.min_opacity]
    lo, hi = np.percentile(oxyz, 2, 0), np.percentile(oxyz, 98, 0)
    ext = float(np.max(hi - lo))

    plane_path = Path(args.plane) if args.plane else run / "table_plane.json"
    if plane_path.is_file():
        plane, plane_src = mio.load_plane(plane_path), str(plane_path)
    else:
        sxyz, sop = _xyz_op(run / "step8_removal_virtual_cam" / "2_object_removal" / "point_cloud_removed.ply")
        plane = fit_plane(sxyz[sop > 0.5], oxyz, _camera_up(run), ext, args.max_tilt_deg)
        plane_src = f"RANSAC on step-8 scene Gaussians ({plane['inliers']} inliers)"

    ouvh = mio.to_uvh(plane, oxyz)
    keep_obj = (ouvh[:, 2] > np.percentile(ouvh[:, 2], 1)) & (ouvh[:, 2] < np.percentile(ouvh[:, 2], 99))
    uv = ouvh[keep_obj, :2]
    hull = uv[ConvexHull(uv).vertices]
    ctr = hull.mean(0)
    r = np.linalg.norm(hull - ctr, axis=1, keepdims=True)
    hull = ctr + (hull - ctr) * (r + args.margin * ext) / np.maximum(r, 1e-9)          # grow by margin
    top = float(np.percentile(ouvh[:, 2], 99)) + args.top_margin * ext

    data = PlyData.read(str(raw))
    v = data["vertex"].data
    xyz = np.stack([np.asarray(v[k], np.float64) for k in ("x", "y", "z")], 1)
    op = 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float64)))
    uvh = mio.to_uvh(plane, xyz)
    drop = inside_hull(uvh[:, :2], hull) & (uvh[:, 2] > args.keep_above) & (uvh[:, 2] <= top)
    PlyData([PlyElement.describe(v[~drop], "vertex")], text=False).write(str(ply))

    info = {"step": "11c clear object volume", "object_id": args.object_id, "source": str(raw), "output": str(ply),
            "plane": plane_src, "plane_normal": plane["n"].tolist(), "plane_d": plane["d"],
            "object_extent_m": ext, "footprint_hull_2d": hull.tolist(),
            "height_band_m": [args.keep_above, top], "gaussians_before": int(len(v)),
            "dropped": int(drop.sum()), "dropped_opaque_gt_0.5": int((drop & (op > 0.5)).sum()),
            "gaussians_after": int((~drop).sum())}
    (s11 / "step11c_clear_object_volume.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({k: info[k] for k in ("plane", "object_extent_m", "height_band_m", "gaussians_before",
                                           "dropped", "dropped_opaque_gt_0.5", "gaussians_after")}, indent=1))


if __name__ == "__main__":
    main()
