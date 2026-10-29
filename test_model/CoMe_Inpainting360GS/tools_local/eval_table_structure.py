"""Compare table meshes (same table prism, same COLMAP frame) on the structure of the table top.

Frame: table plane + 2D table outline of CoMe_Infusion/Step_6_Mesh_GOF/mesh_gof_table_crop.json.
Grid of cells (--cell) over the outline. Per cell, the "top surface" = highest vertex with |height| < --obj_h
(objects such as cups / plates / the bear are taller than --obj_h and are reported apart).
  coverage   : outline cells that have table surface (|height| < tol)
  holes      : outline cells with no vertex at all (no table and no object)
  bumps      : cells whose top surface is between tol and --obj_h above the plane (small lumps, not objects)
  dents      : cells that only have surface below -tol
  roughness  : spread of the top-surface height over the table cells
  normals    : mean |n_vertex . n_plane| of the table-surface vertices (1 = perfectly flat shading)
  components : connected parts of the mesh; parts < --small faces = loose fragments
  napkin     : the same numbers restricted to the napkin footprint (napkin Gaussians of step 8)
"""
import json
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from plyfile import PlyData
from scipy.spatial import Delaunay

PRISM = Path(r"F:/Safe-GS/test_model/CoMe_Infusion/Step_6_Mesh_GOF/mesh_gof_table_crop.json")
NAPKIN = Path(r"F:/Safe-GS/test_model/CoMe_Inpainting360GS/outputs/teatime_napkin/step8_removal_virtual_cam_v2/"
              r"2_object_removal/removed_object_id44.ply")


def main():
    ap = ArgumentParser()
    ap.add_argument("--mesh", action="append", required=True, help="name=path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", type=float, default=0.02)
    ap.add_argument("--tol", type=float, default=0.017, help="table surface band (2 %% of the napkin extent)")
    ap.add_argument("--obj_h", type=float, default=0.06, help="taller than this = object on the table")
    ap.add_argument("--small", type=int, default=500)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    c = json.loads(PRISM.read_text(encoding="utf-8"))
    n, d = np.array(c["plane_normal"]), float(c["plane_d"])
    e1, e2 = np.array(c["basis_e1"]), np.array(c["basis_e2"])
    hull = np.array(c["hull_2d"])
    tri = Delaunay(hull)
    umin = hull.min(0)
    G = np.ceil((hull.max(0) - umin) / args.cell).astype(int) + 1
    gx, gy = np.meshgrid(np.arange(G[0]) + 0.5, np.arange(G[1]) + 0.5)
    inside = (tri.find_simplex(np.stack([gx.ravel(), gy.ravel()], 1) * args.cell + umin) >= 0).reshape(G[1], G[0])

    v = PlyData.read(str(NAPKIN)).elements[0]
    nap = np.stack([np.asarray(v[a]) for a in "xyz"], 1)
    nap = nap[np.abs(nap @ n + d) < 0.3]
    nij = np.floor((np.stack([nap @ e1, nap @ e2], 1) - umin) / args.cell).astype(int)
    foot = np.zeros(inside.shape, np.uint8)
    foot[nij[:, 1], nij[:, 0]] = 1
    foot = (cv2.morphologyEx(foot, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0) & inside

    res = {"frame": str(PRISM), "cell": args.cell, "tol": args.tol, "obj_h": args.obj_h,
           "outline_cells": int(inside.sum()), "napkin_cells": int(foot.sum()), "meshes": {}}
    panels = []
    for spec in args.mesh:
        name, path = spec.split("=", 1)
        m = o3d.io.read_triangle_mesh(path)
        m.compute_vertex_normals()
        V, F = np.asarray(m.vertices), np.asarray(m.triangles)
        N = np.asarray(m.vertex_normals)
        h = V @ n + d
        ij = np.floor((np.stack([V @ e1, V @ e2], 1) - umin) / args.cell).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < G[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < G[1])
        ij, hh, NN = ij[ok], h[ok], N[ok]
        lin = ij[:, 1] * G[0] + ij[:, 0]
        ncell = G[0] * G[1]

        any_v = np.bincount(lin, minlength=ncell).reshape(G[1], G[0]) > 0
        surf = np.abs(hh) < args.tol
        has_surf = np.bincount(lin[surf], minlength=ncell).reshape(G[1], G[0]) > 0
        low = hh < args.obj_h
        top = np.full(ncell, -np.inf)
        np.maximum.at(top, lin[low], hh[low])
        top = top.reshape(G[1], G[0])
        obj = np.zeros(ncell, bool)
        obj[np.unique(lin[hh >= args.obj_h])] = True
        obj = obj.reshape(G[1], G[0])
        has_low = np.isfinite(top)

        def stats(region):
            tbl = region & has_low
            t = top[tbl]
            return {
                "cells": int(region.sum()),
                "coverage_table_surface": round(float(has_surf[region].mean()), 4),
                "holes_no_vertex": round(float((~any_v)[region].mean()), 4),
                "cells_with_object_above": round(float(obj[region].mean()), 4),
                "bumps_tol_to_objh": round(float(((top > args.tol) & has_low)[region].mean()), 4),
                "dents_only_below": round(float((has_low & (top < -args.tol))[region].mean()), 4),
                "top_height_p05_p50_p95": [round(float(np.percentile(t, q)), 4) for q in (5, 50, 95)] if len(t) else None,
                "roughness_std_of_top_on_flat_cells": round(float(np.std(t[np.abs(t) < args.tol])), 5)
                if (np.abs(t) < args.tol).any() else None,
            }

        table_region = inside & ~obj
        flat_n = np.abs(NN[surf] @ n)
        labels, counts, _ = m.cluster_connected_triangles()
        counts = np.asarray(counts)
        small = counts < args.small
        nm_edges = len(np.asarray(m.get_non_manifold_edges(allow_boundary_edges=True)))
        area = np.asarray([0.0])
        tris = V[F]
        area = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
        res["meshes"][name] = {
            "file": path, "n_vertices": int(len(V)), "n_faces": int(len(F)),
            "file_mb": round(Path(path).stat().st_size / 2 ** 20, 1), "has_colors": bool(m.has_vertex_colors()),
            "whole_table_without_objects": stats(table_region),
            "napkin_footprint": stats(foot),
            "napkin_footprint_has_object_above_obj_h": round(float(obj[foot].mean()), 4),
            "table_normals_mean_abs_dot": round(float(flat_n.mean()), 4),
            "table_normals_frac_tilted_gt_20deg": round(float((flat_n < np.cos(np.radians(20))).mean()), 4),
            "components": {"count": int(len(counts)), "largest_share_faces": round(float(counts.max() / counts.sum()), 4),
                           f"small_lt_{args.small}_faces": int(small.sum()),
                           "faces_in_small": int(counts[small].sum())},
            "non_manifold_edges": int(nm_edges),
            "degenerate_faces_area_lt_1e-10": int((area < 1e-10).sum()),
            "edge_length_median": round(float(np.median(np.linalg.norm(tris[:, 1] - tris[:, 0], axis=1))), 5),
        }
        # top-down maps: top-surface height (objects grey) and surface tilt
        hmap = np.where(has_low, np.clip(top / (3 * args.tol), -1, 1), np.nan)
        img = cv2.applyColorMap(((np.nan_to_num(hmap, nan=-1) + 1) / 2 * 255).astype(np.uint8), cv2.COLORMAP_JET)
        img[obj] = (110, 110, 110)
        img[~any_v] = (0, 0, 0)
        img[~inside] //= 4
        cs, _ = cv2.findContours(foot.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        cv2.drawContours(img, [cc * 3 for cc in cs], -1, (255, 255, 255), 1)
        cv2.putText(img, name, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        panels.append(img)
        print(name, json.dumps(res["meshes"][name]), flush=True)
    cv2.imwrite(str(out / "table_top_height_maps.png"), np.hstack(panels))
    (out / "table_structure.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
