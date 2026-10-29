"""Step 13 check: is the table where the napkin was a clean collision surface in the CoMe meshes?

Footprint = napkin Gaussians of step 8 (removed_object_id44.ply) projected on the table plane (plane of
CoMe_Infusion surface_adherence_eval.json, same COLMAP frame). Ring = table around the footprint (real table, baseline).
Per mesh, inside the footprint column (|height| < 0.5 napkin extent):
  height of the vertices above the table, share above / below 2 % of the napkin extent,
  table coverage = footprint cells with a vertex within +-2 % extent of the plane (1 = no hole),
  floating parts = connected components of the column that never come within 2 % extent of the table,
plus a top-down map of the highest surface per cell.

  python tools_local/eval_mesh_table.py --mesh v1=<ply> --mesh v2=<ply> --out <dir>
"""
import json
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from plyfile import PlyData

HERE = Path(__file__).resolve().parents[1]
NAPKIN = HERE / "outputs/teatime_napkin/step8_removal_virtual_cam_v2/2_object_removal/removed_object_id44.ply"
PLANE = Path(r"F:/Safe-GS/test_model/CoMe_Infusion/Step_4_Output_Inpainting/phase6_train_3dgs/surface_adherence_eval.json")


def main():
    ap = ArgumentParser()
    ap.add_argument("--mesh", action="append", required=True, help="name=path")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    plane = json.loads(PLANE.read_text(encoding="utf-8"))["plane_n_d"]
    n, d = np.array(plane[:3], float), float(plane[3])
    v = PlyData.read(str(NAPKIN)).elements[0]
    nap = np.stack([np.asarray(v[a]) for a in "xyz"], 1)
    if np.median(nap @ n + d) < 0:          # normal points to the napkin side (up)
        n, d = -n, -d
    e1 = np.cross(n, [1, 0, 0] if abs(n[0]) < 0.9 else [0, 1, 0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    core = nap[np.abs(nap @ n + d) < 0.3]
    uv_core = np.stack([core @ e1, core @ e2], 1)
    ext = float(np.ptp(uv_core, 0).max())
    cell = ext / 40
    pad = 12 * cell
    umin = uv_core.min(0) - pad
    G = np.ceil((uv_core.max(0) + pad - umin) / cell).astype(int)
    foot = np.zeros(G[::-1], np.uint8)
    ij = ((uv_core - umin) / cell).astype(int)
    foot[ij[:, 1], ij[:, 0]] = 1
    foot = cv2.morphologyEx(foot, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    foot = cv2.dilate(foot, np.ones((3, 3), np.uint8)) > 0
    ring = (cv2.dilate(foot.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0) & ~foot
    tol = 0.02 * ext

    res = {"napkin_extent": round(ext, 4), "cell": round(cell, 5), "tolerance_2pct": round(tol, 4),
           "footprint_cells": int(foot.sum()), "ring_cells": int(ring.sum()), "plane_n_d": [*n.round(6).tolist(), d],
           "meshes": {}}
    maps = []
    for spec in args.mesh:
        name, path = spec.split("=", 1)
        m = o3d.io.read_triangle_mesh(path)
        V, F = np.asarray(m.vertices), np.asarray(m.triangles)
        h = V @ n + d
        uv = np.stack([V @ e1, V @ e2], 1)
        cij = np.floor((uv - umin) / cell).astype(int)
        inb = (cij[:, 0] >= 0) & (cij[:, 0] < G[0]) & (cij[:, 1] >= 0) & (cij[:, 1] < G[1]) & (np.abs(h) < 0.5 * ext)
        in_foot = np.zeros(len(V), bool)
        in_ring = np.zeros(len(V), bool)
        in_foot[inb] = foot[cij[inb, 1], cij[inb, 0]]
        in_ring[inb] = ring[cij[inb, 1], cij[inb, 0]]

        def region(mask, cells):
            hh = h[mask]
            cov = np.zeros_like(cells)
            sel = cij[mask & (np.abs(h) < tol)]
            cov[sel[:, 1], sel[:, 0]] = True
            return {"n_vertices": int(mask.sum()),
                    "height_p05_p50_p95": [round(float(np.percentile(hh, q)), 4) for q in (5, 50, 95)] if len(hh) else None,
                    "height_max": round(float(hh.max()), 4) if len(hh) else None,
                    "frac_above_2pct": round(float((hh > tol).mean()), 4) if len(hh) else None,
                    "frac_below_2pct": round(float((hh < -tol).mean()), 4) if len(hh) else None,
                    "table_coverage": round(float(cov[cells].mean()), 4)}

        # floating parts in the footprint column: components of the faces whose vertices are all in the column
        col_faces = F[in_foot[F].all(1)]
        floating = []
        if len(col_faces):
            sub = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(col_faces))
            labels = np.asarray(sub.cluster_connected_triangles()[0])
            fh = h[col_faces]                       # [n_faces, 3] heights
            for c in np.unique(labels):
                ch = fh[labels == c]
                if np.abs(ch).min() > tol:
                    floating.append({"n_faces": int((labels == c).sum()), "min_height": round(float(ch.min()), 4),
                                     "max_height": round(float(ch.max()), 4)})
        res["meshes"][name] = {
            "file": path, "n_vertices": int(len(V)), "n_faces": int(len(F)),
            "is_watertight": bool(m.is_watertight()),
            "footprint": region(in_foot, foot), "ring_real_table": region(in_ring, ring),
            "floating_parts_in_footprint": {"count": len(floating), "faces": int(sum(f["n_faces"] for f in floating)),
                                            "largest": sorted(floating, key=lambda f: -f["n_faces"])[:5]},
        }
        top = np.full(foot.shape, np.nan)
        for (i, j), z in zip(cij[inb], h[inb]):
            if np.isnan(top[j, i]) or z > top[j, i]:
                top[j, i] = z
        img = np.nan_to_num(np.clip(top / (0.1 * ext), -1, 1), nan=-1.5)
        col = cv2.applyColorMap(((img + 1.5) / 2.5 * 255).astype(np.uint8), cv2.COLORMAP_JET)
        col[~(foot | ring)] //= 3
        cs, _ = cv2.findContours(foot.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        col = cv2.resize(col, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
        cv2.drawContours(col, [c * 6 for c in cs], -1, (255, 255, 255), 1)
        cv2.putText(col, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        maps.append(col)
    cv2.imwrite(str(out / "table_height_maps.png"), np.hstack(maps))
    (out / "mesh_table_check.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
