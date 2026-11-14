"""
Helpers used by the Modal runner (no change to the authors' code):

  id_iou        Step 3: IoU between render_obj.py's colour-coded predictions (objects_pred/*.jpg, id2rgb colours)
                and the ground-truth ID masks.
  ply_stats     Step 4/5/6: Gaussian count, robust bbox, centre and variance of a 2DGS point_cloud.ply.
  seed_points   Step 5: points3D.ply (x,y,z,nx,ny,nz,rgb) for the refined set, from the object Gaussians, so the
                2DGS refinement does not start from the loader's random cube in [-1.3, 1.3]^3.
  mesh_params   Step 6: TSDF voxel / sdf_trunc / depth_trunc from the object size and the camera distances.
  to_obj        Step 6: fuse_post.ply -> .obj (vertex colours kept in the .ply; .obj via open3d).
  manifest      sha256 of every file in a folder (checked after `modal volume get`).
"""
import argparse
import colorsys
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


def id2rgb(i):  # same formula as render.py::id2rgb, which render_obj.py uses for objects_pred/*.jpg
    if i == 0:     # (extract_object.py has its own id2rgb with golden_ratio 1.14514 - not used for objects_pred)
        return np.zeros(3, np.uint8)
    h = (i * 1.6180339887) % 1
    s = 0.5 + (i % 2) * 0.5
    r, g, b = colorsys.hls_to_rgb(h, 0.5, s)
    return np.array([int(r * 255), int(g * 255), int(b * 255)], np.uint8)


PALETTE = np.stack([id2rgb(i) for i in range(256)]).astype(np.int16)


def id_iou(pred_dir, gt_dir, obj_id, out_json, tol=40):
    from PIL import Image
    color = id2rgb(obj_id).astype(np.int16)
    rows, ious = {}, []
    for gp in sorted(Path(gt_dir).glob("*.png")):
        pp = Path(pred_dir) / f"{gp.stem}.jpg"
        if not pp.exists():
            continue
        gt = np.asarray(Image.open(gp))
        gt = (gt[..., 0] if gt.ndim == 3 else gt) == obj_id
        pred_rgb = np.asarray(Image.open(pp).convert("RGB")).astype(np.int16)
        if pred_rgb.shape[:2] != gt.shape:
            pred_rgb = np.asarray(Image.open(pp).convert("RGB").resize(gt.shape[::-1], Image.NEAREST)).astype(np.int16)
        # a pixel is the object if its colour is close to id2rgb(obj_id) AND that is its nearest palette colour
        pred = np.abs(pred_rgb - color).max(-1) <= tol
        if pred.any():
            cand = pred_rgb[pred]                                               # (N, 3)
            d = np.abs(cand[:, None, :] - PALETTE[None, :, :]).max(-1)        # (N, 256)
            pred[pred] = d.argmin(1) == obj_id
        if not gt.any() and not pred.any():
            continue
        inter, union = int((gt & pred).sum()), int((gt | pred).sum())
        iou = inter / union if union else 0.0
        rows[gp.stem] = {"iou": round(iou, 4), "gt_px": int(gt.sum()), "pred_px": int(pred.sum())}
        if gt.any():
            ious.append(iou)
    summary = {"object_id": obj_id, "color": id2rgb(obj_id).tolist(), "frames_with_gt": len(ious),
               "mean_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
               "median_iou": round(float(np.median(ious)), 4) if ious else 0.0,
               "false_positive_frames": sum(1 for r in rows.values() if r["gt_px"] == 0)}
    Path(out_json).write_text(json.dumps({"summary": summary, "frames": rows}, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _read_xyz(ply_path):
    from plyfile import PlyData
    v = PlyData.read(str(ply_path)).elements[0]
    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], 1).astype(np.float64)
    return v, xyz


def ply_stats(ply_path):
    v, xyz = _read_xyz(ply_path)
    lo, hi = np.percentile(xyz, 1, 0), np.percentile(xyz, 99, 0)
    return {"n_gaussians": int(len(xyz)), "center_mean": xyz.mean(0).tolist(),
            "var_all": float(np.var(xyz, ddof=1)),  # = torch.var(_xyz, unbiased=True) used by dream_gaussian_2D
            "bbox_p01": lo.tolist(), "bbox_p99": hi.tolist(), "extent_p01_p99": (hi - lo).tolist()}


def seed_points(gaussian_ply, out_ply, max_points=100000):
    from plyfile import PlyData, PlyElement
    v, xyz = _read_xyz(gaussian_ply)
    c0 = 0.28209479177387814
    rgb = np.stack([np.asarray(v[f"f_dc_{i}"]) for i in range(3)], 1) * c0 + 0.5
    rgb = (rgb.clip(0, 1) * 255).astype(np.uint8)
    if len(xyz) > max_points:
        idx = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    el = np.empty(len(xyz), dtype=dtype)
    el["x"], el["y"], el["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    el["nx"] = el["ny"] = el["nz"] = 0
    el["red"], el["green"], el["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    PlyData([PlyElement.describe(el, "vertex")]).write(str(out_ply))
    return len(xyz)


def camera_centers(cameras_json):
    cams = json.loads(Path(cameras_json).read_text())
    return np.array([c["position"] for c in cams], dtype=np.float64), cams


def mesh_params(gaussian_ply, cameras_json, voxels_across=320):
    st = ply_stats(gaussian_ply)
    lo, hi = np.array(st["bbox_p01"]), np.array(st["bbox_p99"])
    center, ext = (lo + hi) / 2, float(np.max(hi - lo))
    centers, _ = camera_centers(cameras_json)
    dist = np.linalg.norm(centers - center, axis=1)
    voxel = float(np.clip(ext / voxels_across, 2e-4, 0.02))
    return {"voxel_size": round(voxel, 6), "sdf_trunc": round(4 * voxel, 6),
            "depth_trunc": round(float(dist.max() + ext), 4), "object_extent": round(ext, 5),
            "camera_distance_median": round(float(np.median(dist)), 4)}


def dream_overrides(gaussian_ply, cameras_json):
    """W, H, fovx, fovy (radians) of the dataset cameras and the orbit radius for dream_gaussian_2D.py.
    The script orbits at radius = sqrt(var(xyz)) + opt.radius around mean(xyz); opt.radius is chosen so the
    orbit distance equals the median distance of the real cameras to the object."""
    st = ply_stats(gaussian_ply)
    centers, cams = camera_centers(cameras_json)
    c0 = cams[0]
    W, H, fx, fy = int(c0["width"]), int(c0["height"]), float(c0["fx"]), float(c0["fy"])
    target = np.array(st["center_mean"])
    d = float(np.median(np.linalg.norm(centers - target, axis=1)))
    radius_opt = max(d - float(np.sqrt(st["var_all"])), 0.05)
    return {"W": W, "H": H, "_W": W, "_H": H,
            "fovx": float(2 * np.arctan(W / (2 * fx))), "fovy": float(2 * np.arctan(H / (2 * fy))),
            "radius": round(radius_opt, 5), "orbit_distance": round(d, 5)}


def prune_gaussians(model_ply, ref_gaussian_ply, margin=0.15):
    """Drop Gaussians whose centre is outside the object bbox (ref Gaussians p1-p99 + margin), in place.
    Black floaters of the object-only model (black-background training) otherwise turn into huge TSDF surfaces."""
    from plyfile import PlyData
    st = ply_stats(ref_gaussian_ply)
    lo, hi = np.array(st["bbox_p01"]), np.array(st["bbox_p99"])
    pad = margin * float(np.max(hi - lo))
    lo, hi = lo - pad, hi + pad
    pd = PlyData.read(str(model_ply))
    v = pd.elements[0]
    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], 1)
    keep = np.all((xyz >= lo) & (xyz <= hi), axis=1)
    v.data = v.data[keep]
    pd.write(str(model_ply))
    return {"before": int(len(keep)), "kept": int(keep.sum()), "bbox": [lo.tolist(), hi.tolist()]}


def clean_mesh(ply_in, ref_gaussian_ply, out_ply, out_obj, margin=0.15, min_frac=0.05):
    """The object-only 2DGS model is trained on black-background images, so black floaters cost nothing and still
    produce depth -> TSDF of the raw model is full of floater fragments (43M vertices / 718k clusters at 15k iters).
    Keep the part of the raw TSDF mesh inside the object's bbox (Step 4 Gaussians, p1-p99, +margin) and drop small
    connected pieces (< min_frac of the largest)."""
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(str(ply_in))
    raw_v, raw_t = len(m.vertices), len(m.triangles)
    st = ply_stats(ref_gaussian_ply)
    lo, hi = np.array(st["bbox_p01"]), np.array(st["bbox_p99"])
    pad = margin * float(np.max(hi - lo))
    m = m.crop(o3d.geometry.AxisAlignedBoundingBox(lo - pad, hi + pad))
    crop_v, crop_t = len(m.vertices), len(m.triangles)
    ids, counts, _ = m.cluster_connected_triangles()
    ids, counts = np.asarray(ids), np.asarray(counts)
    kept = 0
    if len(counts):
        keep = counts >= min_frac * counts.max()
        kept = int(keep.sum())
        m.remove_triangles_by_mask(~keep[ids])
        m.remove_unreferenced_vertices()
    o3d.io.write_triangle_mesh(str(out_ply), m)
    o3d.io.write_triangle_mesh(str(out_obj), m, write_vertex_colors=True)
    return {"raw": {"vertices": raw_v, "triangles": raw_t}, "cropped": {"vertices": crop_v, "triangles": crop_t},
            "clusters_in_bbox": int(len(counts)), "clusters_kept": kept,
            "clean": {"vertices": len(m.vertices), "triangles": len(m.triangles)},
            "bbox": [(lo - pad).tolist(), (hi + pad).tolist()], "min_frac": min_frac}


def to_obj(ply_in, obj_out):
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(str(ply_in))
    o3d.io.write_triangle_mesh(str(obj_out), m, write_vertex_colors=True)
    return {"vertices": len(m.vertices), "triangles": len(m.triangles)}


def manifest(folder, out_name="manifest.json", exclude_dirs=()):
    folder = Path(folder)
    files = {}
    for p in sorted(folder.rglob("*")):
        rel_parts = p.relative_to(folder).parts
        if any(part in exclude_dirs for part in rel_parts[:-1]):
            continue
        if p.is_file() and p.name != out_name:
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            files[p.relative_to(folder).as_posix()] = {"bytes": p.stat().st_size, "sha256": h.hexdigest()}
    (folder / out_name).write_text(json.dumps(files, indent=1))
    return len(files)


def verify(folder, name="manifest.json"):
    """Return list of problems (missing / size / hash mismatch) for a downloaded folder."""
    folder = Path(folder)
    problems = []
    for mf in folder.rglob(name):
        ref = json.loads(mf.read_text())
        for rel, info in ref.items():
            p = mf.parent / rel
            if not p.exists():
                problems.append(f"missing {p}")
                continue
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            if p.stat().st_size != info["bytes"] or h.hexdigest() != info["sha256"]:
                problems.append(f"corrupt {p}")
    return problems


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("id_iou"); a.add_argument("pred"); a.add_argument("gt"); a.add_argument("id", type=int); a.add_argument("out")
    a = sub.add_parser("verify"); a.add_argument("folder")
    args = ap.parse_args()
    if args.cmd == "id_iou":
        id_iou(args.pred, args.gt, args.id, args.out)
    elif args.cmd == "verify":
        probs = verify(args.folder)
        print("\n".join(probs) if probs else "OK: every file matches its manifest.json")
        sys.exit(1 if probs else 0)
