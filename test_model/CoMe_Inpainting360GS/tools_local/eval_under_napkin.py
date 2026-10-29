"""Table structure under the napkin, compared between table meshes (same COLMAP frame).

Region: napkin footprint = napkin Gaussians (step 8, removed_object_id44.ply) projected on the table plane
(plane of CoMe_Infusion/Step_6_Mesh_GOF/mesh_gof_table_crop.json), cell = napkin extent / 40.
In every cell the "table layer" is the LOWEST surface above -0.5*extent: for a mesh that still has the napkin
(GOF of CoMe_Infusion) that is the table under the napkin, for the inpainted meshes it is the table itself.
  coverage   : cells whose table layer is within +-tol of the plane
  dent       : table layer below -tol            raised : table layer above +tol (no table at plane level)
  empty      : no vertex at all in the cell (hole)
  roughness  : std of the table-layer height over the covered cells; tilt of the vertex normals in the band
  above      : share of cells with surfaces above the table layer (the napkin for GOF; lumps for inpainted)
  fragments  : connected parts lying entirely in the footprint column that never touch the table band
"""
import json
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from plyfile import PlyData

PRISM = Path(r"F:/Safe-GS/test_model/CoMe_Infusion/Step_6_Mesh_GOF/mesh_gof_table_crop.json")
NAPKIN = Path(r"F:/Safe-GS/test_model/CoMe_Inpainting360GS/outputs/teatime_napkin/step8_removal_virtual_cam_v2/"
              r"2_object_removal/removed_object_id44.ply")


def main():
    ap = ArgumentParser()
    ap.add_argument("--mesh", action="append", required=True, help="name=path")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    c = json.loads(PRISM.read_text(encoding="utf-8"))
    n, d = np.array(c["plane_normal"]), float(c["plane_d"])
    e1, e2 = np.array(c["basis_e1"]), np.array(c["basis_e2"])
    v = PlyData.read(str(NAPKIN)).elements[0]
    nap = np.stack([np.asarray(v[a]) for a in "xyz"], 1)
    nap = nap[np.abs(nap @ n + d) < 0.3]
    uv = np.stack([nap @ e1, nap @ e2], 1)
    ext = float(np.ptp(uv, 0).max())
    cell, tol = ext / 40, 0.02 * ext
    umin = uv.min(0) - 6 * cell
    G = np.ceil((uv.max(0) + 6 * cell - umin) / cell).astype(int)
    foot = np.zeros(G[::-1], np.uint8)
    ij = ((uv - umin) / cell).astype(int)
    foot[ij[:, 1], ij[:, 0]] = 1
    foot = cv2.morphologyEx(foot, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    foot = cv2.dilate(foot, np.ones((3, 3), np.uint8)) > 0
    ncell = G[0] * G[1]

    res = {"napkin_extent": round(ext, 4), "cell": round(cell, 4), "tol": round(tol, 4),
           "footprint_cells": int(foot.sum()), "meshes": {}}
    panels = []
    for spec in args.mesh:
        name, path = spec.split("=", 1)
        m = o3d.io.read_triangle_mesh(path)
        m.compute_vertex_normals()
        V, F, N = np.asarray(m.vertices), np.asarray(m.triangles), np.asarray(m.vertex_normals)
        h = V @ n + d
        cij = np.floor((np.stack([V @ e1, V @ e2], 1) - umin) / cell).astype(int)
        ok = (cij[:, 0] >= 0) & (cij[:, 0] < G[0]) & (cij[:, 1] >= 0) & (cij[:, 1] < G[1]) & (np.abs(h) < 0.5 * ext)
        col = np.zeros(len(V), bool)
        col[ok] = foot[cij[ok, 1], cij[ok, 0]]
        lin = cij[col, 1] * G[0] + cij[col, 0]
        hh = h[col]
        low = np.full(ncell, np.inf)
        np.minimum.at(low, lin, hh)
        high = np.full(ncell, -np.inf)
        np.maximum.at(high, lin, hh)
        low, high = low.reshape(G[1], G[0]), high.reshape(G[1], G[0])
        has = np.isfinite(low)
        layer = low[foot & has]
        covered = foot & has & (np.abs(low) <= tol)
        band = col & (np.abs(h) <= tol)
        tilt = np.abs(N[band] @ n)
        # fragments: components of the footprint-column faces that never come within tol of the plane
        fc = F[col[F].all(1)]
        frags = []
        if len(fc):
            sub = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(fc))
            lab = np.asarray(sub.cluster_connected_triangles()[0])
            fh = h[fc]
            for k in np.unique(lab):
                hk = fh[lab == k]
                if np.abs(hk).min() > tol:
                    frags.append({"faces": int((lab == k).sum()), "height_min": round(float(hk.min()), 4),
                                  "height_max": round(float(hk.max()), 4)})
        res["meshes"][name] = {
            "file": path, "vertices_in_footprint_column": int(col.sum()),
            "coverage_table_layer_within_tol": round(float(covered.sum() / foot.sum()), 4),
            "empty_cells": round(float((foot & ~has).sum() / foot.sum()), 4),
            "dent_cells_below_tol": round(float((foot & has & (low < -tol)).sum() / foot.sum()), 4),
            "raised_cells_above_tol": round(float((foot & has & (low > tol)).sum() / foot.sum()), 4),
            "table_layer_height_p05_p50_p95": [round(float(np.percentile(layer, q)), 4) for q in (5, 50, 95)],
            "table_layer_roughness_std_covered": round(float(np.std(low[covered])), 5),
            "band_normals_mean_abs_dot": round(float(tilt.mean()), 4) if len(tilt) else None,
            "band_normals_tilted_gt_20deg": round(float((tilt < np.cos(np.radians(20))).mean()), 4) if len(tilt) else None,
            "cells_with_surface_above_layer_gt_tol": round(float((foot & has & (high - low > tol)).sum() / foot.sum()), 4),
            "fragments_off_table": {"count": len(frags), "faces": int(sum(f["faces"] for f in frags)),
                                    "largest": sorted(frags, key=lambda f: -f["faces"])[:3]},
        }
        img = np.where(has, np.clip(low / (3 * tol), -1, 1), np.nan)
        col_img = cv2.applyColorMap(((np.nan_to_num(img, nan=-1) + 1) / 2 * 255).astype(np.uint8), cv2.COLORMAP_JET)
        col_img[~has] = 0
        col_img[~foot] //= 4
        col_img = cv2.resize(col_img, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
        cv2.putText(col_img, name, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        panels.append(col_img)
        print(name, json.dumps(res["meshes"][name]), flush=True)
    bar = np.zeros((30, sum(p.shape[1] for p in panels), 3), np.uint8)
    cv2.putText(bar, f"table layer height: blue = -{3 * tol:.3f}, green = 0 (plane), red = +{3 * tol:.3f}; black = empty",
                (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(str(out / "under_napkin_table_layer.png"), np.vstack([np.hstack(panels), bar]))
    (out / "under_napkin.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
