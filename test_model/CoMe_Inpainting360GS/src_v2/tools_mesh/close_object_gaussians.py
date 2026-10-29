"""Step 14: predict the unseen underside of an object's 3DGS so it can be meshed as a closed solid (plan.md, 4b).

The object Gaussians (grouped by ID as in Gaussian Grouping: classifier + removal_setup -> point_cloud_<id>.ply) only
cover what the cameras saw: top and rims. The bottom resting on the support was never seen. Steps:
  1. clean: prob(object id) > 0.7 from the distilled classifier (removal_setup's own test; the convex-hull points that
     removal_setup also adds can be table Gaussians under the object), opacity > 0.1, height within
     [-0.5 %, 30 %] of the object extent above the support, largest DBSCAN cluster
  2. support S(u,v) under the object: the inpainted table mesh (step 13) hit by rays along -n (median filtered); the table
     plane where a ray misses or hits a fragment far from the plane
  3. grid on the support (cell = extent / 100): footprint F (Gaussians splatted with their in-plane size, closed, holes
     filled) and top height H_top (highest Gaussian centre per cell, nearest fill inside F)
  4. bottom: one flat Gaussian per footprint cell at S + eps (eps = 0.3 % extent), opacity 0.99, colour / ID feature =
     opacity-weighted mean of the object
  5. walls: along the footprint border, columns of flat vertical Gaussians from S + eps to H_top, facing outwards
The interior between top and bottom is not filled: the CoMe/GOF extractor treats space hidden from a view as inside.
Output keeps the CoMe ply layout of the input (f_dc/f_rest/opacity/scale/rot/obj_dc/confidence/filter_3D).

  python tools_mesh/close_object_gaussians.py --object_ply <point_cloud_44.ply> --classifier <classifier.pth> \
         --object_id 44 --support_mesh <mesh_come_v2_table.ply> --out_dir <step14 dir>
"""
import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
from scipy import ndimage
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mesh_io as mio  # noqa: E402


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logit(p):
    return np.log(p / (1 - p))


def quat_from_matrix(R):
    """Rotation matrices [N,3,3] -> quaternions [N,4] (w, x, y, z), the 3DGS rot_0..rot_3 convention."""
    q = np.zeros((len(R), 4))
    tr = np.trace(R, axis1=1, axis2=2)
    for i, M in enumerate(R):
        if tr[i] > 0:
            s = np.sqrt(tr[i] + 1.0) * 2
            q[i] = [0.25 * s, (M[2, 1] - M[1, 2]) / s, (M[0, 2] - M[2, 0]) / s, (M[1, 0] - M[0, 1]) / s]
        elif M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
            s = np.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2
            q[i] = [(M[2, 1] - M[1, 2]) / s, 0.25 * s, (M[0, 1] + M[1, 0]) / s, (M[0, 2] + M[2, 0]) / s]
        elif M[1, 1] > M[2, 2]:
            s = np.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2
            q[i] = [(M[0, 2] - M[2, 0]) / s, (M[0, 1] + M[1, 0]) / s, 0.25 * s, (M[1, 2] + M[2, 1]) / s]
        else:
            s = np.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2
            q[i] = [(M[1, 0] - M[0, 1]) / s, (M[0, 2] + M[2, 0]) / s, (M[1, 2] + M[2, 1]) / s, 0.25 * s]
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def main():
    ap = ArgumentParser()
    ap.add_argument("--object_ply", required=True)
    ap.add_argument("--classifier", required=True)
    ap.add_argument("--object_id", type=int, required=True)
    ap.add_argument("--support_mesh", default="", help="inpainted background mesh; empty = table plane only")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prob", type=float, default=0.7)
    ap.add_argument("--cells", type=int, default=100, help="grid cells across the object extent")
    ap.add_argument("--eps", type=float, default=0.003, help="gap above the support, fraction of the extent")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plane = mio.load_plane()

    ply = PlyData.read(args.object_ply)
    v = ply.elements[0]
    names = [p.name for p in v.properties]
    A = np.stack([np.asarray(v[nm], np.float64) for nm in names], 1)
    col = {nm: i for i, nm in enumerate(names)}
    xyz = A[:, [col["x"], col["y"], col["z"]]]
    op = sigmoid(A[:, col["opacity"]])
    obj = A[:, [col[f"obj_dc_{i}"] for i in range(16)]]
    uvh = mio.to_uvh(plane, xyz)

    # 1. clean
    import torch
    sd = torch.load(args.classifier, map_location="cpu")
    W, b = sd["weight"][:, :, 0, 0].numpy().astype(np.float64), sd["bias"].numpy().astype(np.float64)
    logits = obj @ W.T + b
    prob = np.exp(logits - logits.max(1, keepdims=True))
    prob = (prob / prob.sum(1, keepdims=True))[:, args.object_id]
    keep = (prob > args.prob) & (op > 0.1)
    ext0 = float(np.ptp(uvh[keep & (np.abs(uvh[:, 2]) < 0.3), :2], 0).max())
    keep &= (uvh[:, 2] > -0.005 * ext0) & (uvh[:, 2] < 0.3 * ext0)
    lab = DBSCAN(eps=ext0 / 20, min_samples=10).fit_predict(xyz[keep])
    big = np.bincount(lab[lab >= 0]).argmax()
    idx = np.nonzero(keep)[0][lab == big]
    clean = A[idx]
    cuvh, cop = uvh[idx], op[idx]
    ext = float(np.ptp(cuvh[:, :2], 0).max())
    stats = {"input_gaussians": int(len(A)), "prob_gt": int((prob > args.prob).sum()),
             "after_opacity_height": int(keep.sum()), "kept_largest_cluster": int(len(idx)), "extent": round(ext, 4)}

    # 3. grid + footprint + top height
    cell = ext / args.cells
    pad = 4
    umin = cuvh[:, :2].min(0) - pad * cell
    G = np.ceil((cuvh[:, :2].max(0) + pad * cell - umin) / cell).astype(int) + 1
    foot = np.zeros((G[1], G[0]), np.uint8)
    smax = np.exp(clean[:, [col["scale_0"], col["scale_1"], col["scale_2"]]]).max(1)
    ij = ((cuvh[:, :2] - umin) / cell).astype(int)
    for (i, j), r, o in zip(ij, smax, cop):
        cv2.circle(foot, (int(i), int(j)), int(np.clip(round(1.5 * r / cell), 0, 3)), 1, -1)
    foot = cv2.morphologyEx(foot, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    foot = ndimage.binary_fill_holes(foot > 0)
    top = np.full(foot.shape, -np.inf)
    sel = cop > 0.3
    np.maximum.at(top, (ij[sel, 1], ij[sel, 0]), cuvh[sel, 2])
    known = np.isfinite(top)
    _, (iy, ix) = ndimage.distance_transform_edt(~known, return_indices=True)
    top = np.where(known, top, top[iy, ix])
    gy, gx = np.mgrid[0:G[1], 0:G[0]]
    cu, cvv = umin[0] + (gx + 0.5) * cell, umin[1] + (gy + 0.5) * cell

    # 2. support under the footprint (rays along -n onto the inpainted table mesh)
    support = np.zeros(foot.shape)
    sup_info = {"source": "table plane"}
    if args.support_mesh:
        V, F, _ = mio.read_mesh(args.support_mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
        start_h = 0.5 * ext
        org = mio.from_uvh(plane, np.stack([cu, cvv, np.full_like(cu, start_h)], -1)).reshape(-1, 3)
        rays = np.concatenate([org, np.tile(-plane["n"], (len(org), 1))], 1).astype(np.float32)
        t = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().reshape(foot.shape)
        hit = start_h - t
        valid = np.isfinite(hit) & (np.abs(hit) < 0.04 * ext)        # fragments above the table are not support
        hit = np.where(valid, hit, 0.0)
        support = ndimage.median_filter(hit, size=5)
        sup_info = {"source": f"rays onto {Path(args.support_mesh).name} (median 5x5), table plane where missed",
                    "cells_on_mesh_in_footprint": round(float(valid[foot].mean()), 4),
                    "support_height_p05_p50_p95_in_footprint": [round(float(np.percentile(support[foot], q)), 4) for q in (5, 50, 95)]}
    eps = args.eps * ext

    # 4. bottom Gaussians
    n, e1, e2 = plane["n"], plane["e1"], plane["e2"]
    w = cop / cop.sum()
    mean_row = (clean * w[:, None]).sum(0)
    Rb = np.stack([e1, e2, n], 1)                                   # columns = local x, y, z axes
    fy, fx = np.nonzero(foot)
    bottom_uvh = np.stack([cu[fy, fx], cvv[fy, fx], support[fy, fx] + eps], 1)
    bottom = np.tile(mean_row, (len(fy), 1))
    bottom[:, [col["x"], col["y"], col["z"]]] = mio.from_uvh(plane, bottom_uvh)
    bottom[:, [col["scale_0"], col["scale_1"], col["scale_2"]]] = np.log([0.75 * cell, 0.75 * cell, eps / 2])
    bottom[:, [col[f"rot_{k}"] for k in range(4)]] = quat_from_matrix(Rb[None])[0]

    # 5. walls along the footprint border, facing outwards
    dist = ndimage.distance_transform_edt(foot)
    gyd, gxd = np.gradient(ndimage.gaussian_filter(dist, 1.0))
    border = foot & ~ndimage.binary_erosion(foot)
    by, bx = np.nonzero(border)
    walls, wall_rot = [], []
    for y, x in zip(by, bx):
        o2 = -np.array([gxd[y, x], gyd[y, x]])
        if np.linalg.norm(o2) < 1e-9:
            continue
        o2 /= np.linalg.norm(o2)
        o = o2[0] * e1 + o2[1] * e2
        tvec = np.cross(n, o)
        R = np.stack([tvec, n, o], 1)
        h0, h1 = support[y, x] + eps, max(top[y, x], support[y, x] + 2 * eps)
        for h in np.arange(h0, h1 + 1e-9, cell):
            walls.append([cu[y, x], cvv[y, x], h])
            wall_rot.append(R)
    wall = np.tile(mean_row, (len(walls), 1))
    if walls:
        wall[:, [col["x"], col["y"], col["z"]]] = mio.from_uvh(plane, np.array(walls))
        wall[:, [col["scale_0"], col["scale_1"], col["scale_2"]]] = np.log([0.75 * cell, 0.75 * cell, eps / 2])
        wall[:, [col[f"rot_{k}"] for k in range(4)]] = quat_from_matrix(np.array(wall_rot))
    for arr in (bottom, wall):
        arr[:, col["opacity"]] = logit(0.99)
        for k in [c for c in names if c.startswith("f_rest_")]:
            arr[:, col[k]] = 0.0
        arr[:, col["confidence"]] = 0.0
        arr[:, col["filter_3D"]] = 0.0                          # recomputed by CoMe from the training cameras

    def save(rows, path):
        el = np.empty(len(rows), dtype=[(nm, "f4") for nm in names])
        for i, nm in enumerate(names):
            el[nm] = rows[:, i]
        PlyData([PlyElement.describe(el, "vertex")]).write(str(path))

    closed = np.concatenate([clean, bottom, wall])
    save(clean, out / f"point_cloud_{args.object_id}_clean.ply")
    save(closed, out / f"point_cloud_{args.object_id}_closed.ply")
    np.savez_compressed(out / "support_grid.npz", umin=umin, cell=cell, footprint=foot, support=support, top=top,
                        eps=eps, ext=ext, plane_n=n, plane_d=plane["d"], e1=e1, e2=e2)

    # checks + pictures
    orig_cells = np.zeros(foot.shape, bool)
    orig_cells[ij[:, 1], ij[:, 0]] = True
    hm = np.where(foot, np.clip(top / (0.2 * ext), 0, 1), np.nan)
    img = cv2.applyColorMap((np.nan_to_num(hm, nan=0) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    img[~foot] = 0
    img[border] = (255, 255, 255)
    cv2.imwrite(str(out / "height_map_top.png"), cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
    sm = np.where(foot, np.clip((support / (0.02 * ext) + 1) / 2, 0, 1), np.nan)
    simg = cv2.applyColorMap((np.nan_to_num(sm, nan=0) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    simg[~foot] = 0
    cv2.imwrite(str(out / "support_height.png"), cv2.resize(simg, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
    bottom_h = bottom_uvh[:, 2] - support[fy, fx]
    stats.update({
        "cell": round(cell, 5), "eps": round(eps, 5), "grid": [int(G[0]), int(G[1])],
        "footprint_cells": int(foot.sum()), "border_cells": int(border.sum()),
        "footprint_covers_gaussian_cells": round(float(foot[orig_cells].mean()), 4),
        "top_height_p50_p95_max": [round(float(np.percentile(top[foot], q)), 4) for q in (50, 95)] + [round(float(top[foot].max()), 4)],
        "bottom_gaussians": int(len(bottom)), "wall_gaussians": int(len(wall)), "closed_gaussians": int(len(closed)),
        "bottom_offset_from_support_max": round(float(np.abs(bottom_h - eps).max()), 6),
        "support": sup_info,
    })
    ok = stats["footprint_covers_gaussian_cells"] >= 0.98 and stats["bottom_offset_from_support_max"] <= 0.005 * ext
    stats["check_ok"] = bool(ok)
    (out / "summary.json").write_text(json.dumps(stats, indent=2))
    print("STEP14 " + json.dumps(stats))


if __name__ == "__main__":
    main()
