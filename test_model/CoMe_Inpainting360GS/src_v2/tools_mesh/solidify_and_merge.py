"""Step 15 (local part): closed object mesh from the CoMe mesh of the closed 3DGS + merge with the inpainted table.

Object (plan.md 4b, step 15.2-15.3). The raw CoMe mesh of the step-14 Gaussians leaks BELOW the support (the space
hidden behind the bottom layer counts as inside), so it is cut at the support; instead of capping the irregular
cross-section loops, the solid is rebuilt as a height field over the support (the fallback of the plan, made the main
path because it is always watertight):
  top    = first hit of rays cast along -n onto the largest component of the CoMe mesh (cell = --cell_frac x extent)
  bottom = support S(u,v) + eps (support grid of step 14: the inpainted table mesh under the object)
  walls  = vertical quads on the border of the solid cells
Cells: step-14 footprint where a ray hits the mesh, closed, holes filled, diagonal-only contacts removed, largest region.
Overhangs of the real object (folds) are filled: fine for a rigid collider, not for a deformable one.

Table (step 15.4): components smaller than --min_table_faces removed (floating fragments, e.g. above the napkin
footprint), quadric decimation to --table_faces.

The support under the object is re-measured on the collision table itself (rays along -n, 3x3 maximum) and the bottom
corners take the maximum of their cells, so the object never dips into the table it will rest on. The collision solid
is rebuilt as a coarser height field (--collision_cell_frac) instead of quadric decimation (which opens the solid).

Checks (detail and collision solids): watertight (every edge used by exactly 2 faces, open3d is_watertight), volume > 0,
centre of mass inside the footprint, bottom-layer vertices vs the table along gravity (-n): no penetration > --tol.

  python tools_mesh/solidify_and_merge.py --object_mesh <CoMe napkin mesh> --step14 <step14 dir> \
         --table_mesh <mesh_come_v2_table.ply> --out_dir <step15 dir>
"""
import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mesh_io as mio  # noqa: E402

C0 = 0.28209479177387814


def raycast(V, F, origins, dirs):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
    rays = np.concatenate([origins, dirs], 1).astype(np.float32)
    return scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()


def no_diagonal_contacts(mask):
    """Fill one cell of every 2x2 block whose solid cells touch only by a corner (keeps the surface 2-manifold)."""
    m = mask.copy()
    for _ in range(10):
        a, b, c, d = m[:-1, :-1], m[:-1, 1:], m[1:, :-1], m[1:, 1:]
        diag = (a & d & ~b & ~c) | (b & c & ~a & ~d)
        if not diag.any():
            break
        ys, xs = np.nonzero(diag)
        m[ys, xs + 1] = True
        m[ys + 1, xs] = True
    return m


def heightfield_solid(plane, solid, top, bottom, umin, cell):
    """Watertight mesh of the prism between bottom and top over the solid cells (corner-based grid)."""
    H, W = solid.shape
    sp = np.pad(solid, 1)
    # corner (j, i) touches cells (j-1..j, i-1..i): top = mean of the touching solid cells, bottom = their MAX (the
    # bottom never dips below the support of any touching cell -> no penetration into the table)
    tp = np.pad(np.where(solid, top, 0.0), 1)
    bp = np.pad(np.where(solid, bottom, -np.inf), 1, constant_values=-np.inf)
    cnt = sp[:-1, :-1].astype(int) + sp[:-1, 1:] + sp[1:, :-1] + sp[1:, 1:]
    tsum = tp[:-1, :-1] + tp[:-1, 1:] + tp[1:, :-1] + tp[1:, 1:]
    bmax = np.maximum(np.maximum(bp[:-1, :-1], bp[:-1, 1:]), np.maximum(bp[1:, :-1], bp[1:, 1:]))
    active = cnt > 0
    cb = np.where(active, bmax, 0)
    eps_c = 0.25 * cell                                       # top stays above the (raised) corner bottom
    ct = np.where(active, np.maximum(tsum / np.maximum(cnt, 1), cb + eps_c), 0)
    ids_t = -np.ones((H + 1, W + 1), np.int64)
    ids_t[active] = np.arange(active.sum())
    ids_b = -np.ones((H + 1, W + 1), np.int64)
    ids_b[active] = active.sum() + np.arange(active.sum())
    jj, ii = np.nonzero(active)
    uv = np.stack([umin[0] + ii * cell, umin[1] + jj * cell], 1)
    Vt = mio.from_uvh(plane, np.concatenate([uv, ct[jj, ii][:, None]], 1))
    Vb = mio.from_uvh(plane, np.concatenate([uv, cb[jj, ii][:, None]], 1))
    V = np.concatenate([Vt, Vb])
    F = []
    cy, cx = np.nonzero(solid)
    for j, i in zip(cy, cx):
        a, b, c, d = (j, i), (j, i + 1), (j + 1, i + 1), (j + 1, i)            # (row=v, col=u): CCW in (u, v)
        At, Bt, Ct, Dt = ids_t[a], ids_t[b], ids_t[c], ids_t[d]
        Ab, Bb, Cb, Db = ids_b[a], ids_b[b], ids_b[c], ids_b[d]
        F += [[At, Bt, Ct], [At, Ct, Dt], [Ab, Cb, Bb], [Ab, Db, Cb]]
        if i + 1 >= W or not solid[j, i + 1]:      # +u side: corners b, c
            F += [[Bb, Cb, Ct], [Bb, Ct, Bt]]
        if i - 1 < 0 or not solid[j, i - 1]:       # -u side: corners d, a
            F += [[Db, Ab, At], [Db, At, Dt]]
        if j - 1 < 0 or not solid[j - 1, i]:       # -v side: corners a, b
            F += [[Ab, Bb, Bt], [Ab, Bt, At]]
        if j + 1 >= H or not solid[j + 1, i]:      # +v side: corners c, d
            F += [[Cb, Db, Dt], [Cb, Dt, Ct]]
    return V, np.array(F, np.int64)


def build_object(plane, V, F, g, sup_fn, cell, eps, sup_cell_fn=None):
    """Height-field solid at a given cell size: top from rays onto the CoMe mesh, bottom = support + eps."""
    n = plane["n"]
    ext = float(g["ext"])
    umin = np.asarray(g["umin"], float)
    Hs, Ws = g["footprint"].shape
    W, H = int(np.ceil(Ws * g["cell"] / cell)), int(np.ceil(Hs * g["cell"] / cell))
    jj, ii = np.mgrid[0:H, 0:W]
    cu, cv = umin[0] + (ii + 0.5) * cell, umin[1] + (jj + 0.5) * cell
    sj = np.clip(((cv - umin[1]) / g["cell"]).astype(int), 0, Hs - 1)
    si = np.clip(((cu - umin[0]) / g["cell"]).astype(int), 0, Ws - 1)
    foot = g["footprint"][sj, si]
    support = sup_cell_fn(cu, cv, cell) if sup_cell_fn else sup_fn(cu, cv)   # highest table point under the cell
    start = 0.6 * ext
    org = mio.from_uvh(plane, np.stack([cu, cv, np.full_like(cu, start)], -1)).reshape(-1, 3)
    top = start - raycast(V, F, org, np.tile(-n, (len(org), 1))).reshape(H, W)
    hit = np.isfinite(top) & (top > support + 0.5 * eps)
    solid = ndimage.binary_fill_holes(ndimage.binary_closing(foot & hit, np.ones((3, 3))))
    lab, nlab = ndimage.label(solid)
    if nlab > 1:
        solid = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    solid = no_diagonal_contacts(solid)
    bottom = support + eps
    known = solid & np.isfinite(top)
    _, (iy, ix) = ndimage.distance_transform_edt(~known, return_indices=True)
    top = np.maximum(np.where(known, top, top[iy, ix]), bottom + 2 * eps)
    Vo, Fo = heightfield_solid(plane, solid, top, bottom, umin, cell)
    mesh = mio.to_o3d(Vo, Fo)
    th = np.where(solid, top - bottom, 0)
    com_uvh = np.array([(cu * th).sum(), (cv * th).sum(), ((top + bottom) / 2 * th).sum()]) / th.sum()
    ci, cj = int((com_uvh[0] - umin[0]) / cell), int((com_uvh[1] - umin[1]) / cell)
    wt = bool(mesh.is_watertight())
    info = {"cell": round(cell, 5), "cells": int(solid.sum()), "vertices": int(len(Vo)), "faces": int(len(Fo)),
            "edges": mio.edge_stats(Fo), "watertight": wt, "volume": round(float(mesh.get_volume()), 6) if wt else None,
            "height_max_above_support": round(float((top - support)[solid].max()), 4),
            "center_of_mass": mio.from_uvh(plane, com_uvh).round(5).tolist(), "com_inside_footprint": bool(solid[cj, ci])}
    return mesh, info


def contact(plane, mesh, TVd, TFd, sup_fn, eps, tol):
    """Bottom-layer vertices (within 3 eps of the support) against the table along the gravity direction."""
    n = plane["n"]
    P = np.asarray(mesh.vertices)
    pu = mio.to_uvh(plane, P)
    low = pu[:, 2] - sup_fn(pu[:, 0], pu[:, 1]) < 3 * eps
    Pl = P[low]
    down = raycast(TVd, TFd, Pl, np.tile(-n, (len(Pl), 1)))
    up = raycast(TVd, TFd, Pl, np.tile(n, (len(Pl), 1)))
    pen = np.isfinite(up) & (~np.isfinite(down) | (up < down))      # nearest table surface is above the vertex
    depth = np.where(pen, up, 0.0)
    gap = np.where(np.isfinite(down), down, np.nan)
    return {"bottom_layer_vertices": int(low.sum()), "with_table_below": round(float(np.isfinite(down).mean()), 4),
            "gap_p05_p50_p95": [round(float(np.nanpercentile(gap, q)), 5) for q in (5, 50, 95)],
            "penetrating_vertices": int((depth > tol).sum()), "max_penetration": round(float(depth.max()), 5)}


def main():
    ap = ArgumentParser()
    ap.add_argument("--object_mesh", required=True)
    ap.add_argument("--step14", required=True)
    ap.add_argument("--table_mesh", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--object_id", type=int, default=44)
    ap.add_argument("--cell_frac", type=float, default=0.005, help="height-field cell of the detailed solid (x extent)")
    ap.add_argument("--collision_cell_frac", type=float, default=0.01, help="cell of the collision solid (x extent)")
    ap.add_argument("--table_faces", type=int, default=300000)
    ap.add_argument("--min_table_faces", type=int, default=1000)
    ap.add_argument("--tol", type=float, default=0.001, help="allowed penetration into the table (scene units)")
    args = ap.parse_args()
    out = Path(args.out_dir)
    (out / "collision").mkdir(parents=True, exist_ok=True)
    plane = mio.load_plane()
    n = plane["n"]
    g = np.load(Path(args.step14) / "support_grid.npz")
    ext, eps = float(g["ext"]), float(g["eps"])
    res = {"object_extent": round(ext, 4), "gap_eps": round(eps, 5)}

    # ---------------------------------------------------------------- table first: drop fragments, decimate
    TV, TF, _ = mio.read_mesh(args.table_mesh)
    table = mio.to_o3d(TV, TF)
    labels, counts, _ = table.cluster_connected_triangles()
    labels, counts = np.asarray(labels), np.asarray(counts)
    small = counts < args.min_table_faces
    table.remove_triangles_by_mask(small[labels])
    table.remove_unreferenced_vertices()
    tdec = table.simplify_quadric_decimation(target_number_of_triangles=args.table_faces)
    tdec.remove_degenerate_triangles()
    tdec.remove_unreferenced_vertices()
    o3d.io.write_triangle_mesh(str(out / "collision" / "table_static.obj"), tdec)
    TVd, TFd = np.asarray(tdec.vertices), np.asarray(tdec.triangles)
    res["table"] = {"input_faces": int(len(TF)), "components": int(len(counts)), "removed_components": int(small.sum()),
                    "removed_faces": int(counts[small].sum()), "faces": int(len(TFd))}

    # support = the collision table itself: rays along -n on a fine grid, 3x3 maximum (conservative), table plane
    # (step-14 support) where a ray misses or hits something far from the plane
    sc = 0.0025 * ext
    umin = np.asarray(g["umin"], float)
    Hs, Ws = g["footprint"].shape
    SW, SH = int(np.ceil(Ws * g["cell"] / sc)) + 2, int(np.ceil(Hs * g["cell"] / sc)) + 2
    jj, ii = np.mgrid[0:SH, 0:SW]
    su, sv = umin[0] + (ii + 0.5) * sc, umin[1] + (jj + 0.5) * sc
    start = 0.5 * ext
    org = mio.from_uvh(plane, np.stack([su, sv, np.full_like(su, start)], -1)).reshape(-1, 3)
    hit = start - raycast(TVd, TFd, org, np.tile(-n, (len(org), 1))).reshape(SH, SW)
    old = g["support"][np.clip(((sv - umin[1]) / g["cell"]).astype(int), 0, Hs - 1),
                       np.clip(((su - umin[0]) / g["cell"]).astype(int), 0, Ws - 1)]
    valid = np.isfinite(hit) & (np.abs(hit) < 0.04 * ext)
    sup_grid = ndimage.maximum_filter(np.where(valid, hit, old), size=3)

    def sup_fn(u, v):
        return sup_grid[np.clip(((v - umin[1]) / sc).astype(int), 0, SH - 1), np.clip(((u - umin[0]) / sc).astype(int), 0, SW - 1)]
    def sup_cell_fn(u, v, cell):
        k = int(np.ceil(cell / sc)) + 2                       # cover the whole cell (+ neighbours' half cells)
        return ndimage.maximum_filter(sup_grid, size=k)[np.clip(((v - umin[1]) / sc).astype(int), 0, SH - 1),
                                                        np.clip(((u - umin[0]) / sc).astype(int), 0, SW - 1)]
    res["support"] = {"source": "collision table (decimated), rays along -n, 3x3 max; per cell: max over the cell",
                      "cell": round(sc, 5),
                      "rays_on_table": round(float(valid.mean()), 4)}

    # ---------------------------------------------------------------- object: CoMe mesh -> height-field solids
    V, F, _ = mio.read_mesh(args.object_mesh)
    V, F, info = mio.remove_small_components(V, F, min_frac=0.5)          # largest component only
    uvh = mio.to_uvh(plane, V)
    rel = uvh[:, 2] - sup_fn(uvh[:, 0], uvh[:, 1])
    res["raw_mesh"] = {"vertices": int(len(V)), "faces": int(len(F)), **info,
                       "vertices_below_support": round(float((rel < 0).mean()), 4),
                       "lowest_below_support": round(float(rel.min()), 4)}
    detail, res["object"] = build_object(plane, V, F, g, sup_fn, args.cell_frac * ext, eps, sup_cell_fn)
    coll, res["collision_object"] = build_object(plane, V, F, g, sup_fn, args.collision_cell_frac * ext, eps, sup_cell_fn)
    v = PlyData.read(str(Path(args.step14) / f"point_cloud_{args.object_id}_clean.ply")).elements[0]
    w = 1 / (1 + np.exp(-np.asarray(v["opacity"])))
    rgb = np.clip(0.5 + C0 * np.stack([np.asarray(v[f"f_dc_{k}"]) for k in range(3)], 1), 0, 1)
    col = (rgb * w[:, None]).sum(0) / w.sum()
    res["object"]["colour_rgb"] = col.round(3).tolist()
    mio.write_mesh(out / "mesh_napkin_closed.ply", np.asarray(detail.vertices), np.asarray(detail.triangles),
                   np.tile(col, (len(detail.vertices), 1)))
    o3d.io.write_triangle_mesh(str(out / "collision" / "napkin_dynamic.obj"), coll)
    hull, _ = coll.compute_convex_hull()
    o3d.io.write_triangle_mesh(str(out / "collision" / "napkin_convex_hull.obj"), hull)
    res["contact_detail"] = contact(plane, detail, TVd, TFd, sup_fn, eps, args.tol)
    res["contact_collision"] = contact(plane, coll, TVd, TFd, sup_fn, eps, args.tol)

    # ---------------------------------------------------------------- combined scene + metadata
    tcol = np.tile([0.72, 0.62, 0.48], (len(TVd), 1))
    scene_V = np.concatenate([TVd, np.asarray(detail.vertices)])
    scene_F = np.concatenate([TFd, np.asarray(detail.triangles) + len(TVd)])
    scene_C = np.concatenate([tcol, np.tile(col, (len(detail.vertices), 1))])
    mio.write_mesh(out / "scene_table_napkin.ply", scene_V, scene_F, scene_C)
    co = res["collision_object"]
    meta = {"units": "COLMAP units of the CoMe reconstruction (metric scale unknown)",
            "gravity_direction": (-n).round(6).tolist(), "table_plane_n_d": [*n.round(6).tolist(), plane["d"]],
            "contact_gap": round(eps, 5),
            "bodies": {"table": {"file": "collision/table_static.obj", "type": "static", "note": "open surface"},
                       "napkin": {"file": "collision/napkin_dynamic.obj", "type": "dynamic", "watertight": co["watertight"],
                                  "convex_hull": "collision/napkin_convex_hull.obj",
                                  "center_of_mass": co["center_of_mass"], "volume": co["volume"]}},
            "note": "napkin = rigid solid: top from the CoMe mesh of the step-14 Gaussians, bottom = inpainted table + gap "
                    "(folds filled); convex decomposition (CoACD/V-HACD) not run: convex hull provided"}
    (out / "physics_meta.json").write_text(json.dumps(meta, indent=2))
    ok = all(res[k]["watertight"] and (res[k]["volume"] or 0) > 0 and res[k]["com_inside_footprint"]
             for k in ("object", "collision_object")) and all(
        res[k]["penetrating_vertices"] == 0 and res[k]["with_table_below"] > 0.9 for k in ("contact_detail", "contact_collision"))
    res["check_ok"] = bool(ok)
    (out / "summary.json").write_text(json.dumps(res, indent=2))
    print("STEP15 " + json.dumps(res))


if __name__ == "__main__":
    main()
