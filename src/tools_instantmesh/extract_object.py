"""Step 14.1 (local CPU): napkin Gaussians out of the trained 3DGS + what the real cameras saw of them.

1. Object Gaussians from the ID-distilled scene (step 7: point_cloud_id_iter2000.ply + classifier.pth), same rule as
   basic step 14 (tools_mesh/close_object_gaussians.py): softmax(classifier(obj_dc))[id] > 0.7, opacity > 0.1,
   height above the table plane in [-0.5 %, --max_height_frac (30 %)] of the object extent, largest DBSCAN cluster. The bottom of the
   napkin was never seen, so these Gaussians only describe the top / rims.
2. Observations: Gaussian i is "observed" by real camera j when its centre projects inside image j onto a pixel whose
   synced mask (step 6, ID map) == object id -> also excludes cameras where something else hides it.
3. InstantMesh frame (LRM world): X_im = s * R (X - c), R rows = [e1, e2, n] (table normal -> +z, right-handed),
   c = centre of the object box (uv) and of [support, top] (height), s so that the object reaches |coord| = FILL.
4. Evaluation masks of every real view with the object (1/2 res): 255 object, 128 other object (possible occluder),
   0 background / table.
5. Renderer check: the object Gaussians rendered (gs_render.py) at a few real cameras next to the photo.

  python tools_instantmesh/extract_object.py --run <outputs/LERF/teatime_napkin> [--plane table_plane.json]
"""
from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy import ndimage
from sklearn.cluster import DBSCAN

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools_mesh"))
sys.path.insert(0, str(HERE))
import mesh_io as mio  # noqa: E402
import gs_render  # noqa: E402

FILL = 0.92
EVAL_SCALE = 0.5


def step_dir(run: Path, n: int) -> Path:
    """Folder of step n in a run, whatever its suffix (step1_reuse_come_3dgs of the reference run, step1_come_3dgs of
    the new runners, ...): first existing match of step<n>_*."""
    c = sorted(d for d in Path(run).glob(f"step{n}_*") if d.is_dir())
    if not c:
        raise SystemExit(f"no step{n}_* folder in {run}")
    return c[0]


def cams_from_json(path):
    out = {}
    for c in json.loads(Path(path).read_text()):
        c2w = np.eye(4)
        c2w[:3, :3] = np.array(c["rotation"]); c2w[:3, 3] = np.array(c["position"])
        out[c["img_name"]] = {"c2w_cv": c2w, "K": [c["fx"], c["fy"], c["width"] / 2, c["height"] / 2],
                              "W": c["width"], "H": c["height"]}
    return out


def main():
    ap = ArgumentParser()
    ap.add_argument("--run", required=True, help="outputs/<dataset>/<run>, e.g. outputs/LERF/teatime_napkin")
    ap.add_argument("--object_id", "--object-id", type=int, default=44)
    ap.add_argument("--prob", type=float, default=0.7)
    ap.add_argument("--max_height_frac", type=float, default=3.0,
                    help="keep Gaussians up to this height above the table, as a fraction of the object extent in the "
                         "table plane. 3.0 works for flat AND tall objects (teatime napkin: same Gaussians as 0.3, "
                         "+5 of 3.2k; Replica bin needs it); lower it only if another object sits on top of the target")
    ap.add_argument("--force", action="store_true", help="delete an existing 1_object_gaussians and redo")
    ap.add_argument("--fill_prob", type=float, default=0.5,
                    help="after DBSCAN, also take Gaussians with prob > fill_prob inside the cluster footprint/height "
                         "(recovers sparse thin walls); 0 = off (old behaviour)")
    ap.add_argument("--fill_margin", type=float, default=0.03, help="footprint growth for the fill, x object extent")
    ap.add_argument("--plane", default=None,
                    help="table_plane.json (default: <run>/table_plane.json, <run>/step13_come_mesh/, tools_mesh/)")
    args = ap.parse_args()
    run = Path(args.run)
    out = run / "step14_instantmesh" / "1_object_gaussians"
    if out.exists() and any(out.iterdir()):
        if not args.force:
            raise SystemExit(f"{out} exists - not overwriting (--force to redo)")
        import shutil
        shutil.rmtree(out)
    plane_path = Path(args.plane) if args.plane else mio.find_plane(run)
    print(f"table plane: {plane_path}")
    (out / "masks_eval").mkdir(parents=True, exist_ok=True); (out / "renderer_check").mkdir(exist_ok=True)
    plane = mio.load_plane(plane_path)
    oid = args.object_id

    # ---------------------------------------------------------------- 1. object Gaussians (rule of basic step 14)
    ply = PlyData.read(str(sorted(step_dir(run, 7).glob("point_cloud_id_iter*.ply"))[-1]))
    v = ply.elements[0]
    names = [p.name for p in v.properties]
    A = np.stack([np.asarray(v[nm], np.float64) for nm in names], 1)
    col = {nm: i for i, nm in enumerate(names)}
    xyz = A[:, [col["x"], col["y"], col["z"]]]
    op = 1 / (1 + np.exp(-A[:, col["opacity"]]))
    sd = torch.load(step_dir(run, 7) / "classifier.pth", map_location="cpu")
    Wc, bc = sd["weight"][:, :, 0, 0].numpy().astype(np.float64), sd["bias"].numpy().astype(np.float64)
    logits = A[:, [col[f"obj_dc_{i}"] for i in range(Wc.shape[1])]] @ Wc.T + bc
    prob = np.exp(logits - logits.max(1, keepdims=True)); prob = (prob / prob.sum(1, keepdims=True))[:, oid]
    uvh = mio.to_uvh(plane, xyz)
    keep = (prob > args.prob) & (op > 0.1)
    ext0 = float(np.ptp(uvh[keep & (np.abs(uvh[:, 2]) < 0.3), :2], 0).max())
    keep &= (uvh[:, 2] > -0.005 * ext0) & (uvh[:, 2] < args.max_height_frac * ext0)
    lab = DBSCAN(eps=ext0 / 20, min_samples=10).fit_predict(xyz[keep])
    big = np.bincount(lab[lab >= 0]).argmax()
    idx = np.nonzero(keep)[0][lab == big]
    n_dbscan = int(len(idx))
    if args.fill_prob > 0:
        # Recover the sparse parts of the object: thin walls (Replica bin body: ~1.2k large Gaussians for 4 walls vs
        # ~5k on the rim) are DBSCAN noise at eps = ext/20 and partly below --prob, so the object rendered alone had
        # see-through walls -> InstantMesh got views of a "bin without body". Add every Gaussian with
        # prob > --fill_prob inside the cluster's footprint (convex hull in the plane, grown by --fill_margin x ext)
        # and height range; a neighbour object (other footprint) is not taken.
        from scipy.spatial import ConvexHull
        cu = uvh[idx, :2]
        hull = cu[ConvexHull(cu).vertices]
        ctr = hull.mean(0)
        r = np.linalg.norm(hull - ctr, axis=1, keepdims=True)
        hull = ctr + (hull - ctr) * (r + args.fill_margin * ext0) / np.maximum(r, 1e-9)
        a, b2 = hull, np.roll(hull, -1, 0)
        P = uvh[:, :2]
        cross = (b2[:, 0] - a[:, 0])[None] * (P[:, 1:2] - a[None, :, 1]) - (b2[:, 1] - a[:, 1])[None] * (P[:, 0:1] - a[None, :, 0])
        inside = (cross >= 0).all(1) | (cross <= 0).all(1)
        hmin, hmax = float(uvh[idx, 2].min()), float(uvh[idx, 2].max())
        fill = (prob > args.fill_prob) & (op > 0.1) & inside & (uvh[:, 2] >= hmin) & (uvh[:, 2] <= hmax)
        idx = np.union1d(idx, np.nonzero(fill)[0])
    PlyData([PlyElement.describe(v.data[idx], "vertex")], text=False).write(str(out / f"point_cloud_{oid}_object.ply"))
    X, ouvh, oop = xyz[idx], uvh[idx], op[idx]
    stats = {"scene_gaussians": int(len(A)), "prob_gt": int((prob > args.prob).sum()),
             "after_opacity_height": int(keep.sum()), "after_dbscan": n_dbscan, "object_gaussians": int(len(idx)),
             "fill": {"fill_prob": args.fill_prob, "fill_margin": args.fill_margin, "added": int(len(idx)) - n_dbscan},
             "extent_uv": round(float(np.ptp(ouvh[:, :2], 0).max()), 4),
             "height_p50_p95_max": [round(float(np.percentile(ouvh[:, 2], q)), 4) for q in (50, 95)] + [round(float(ouvh[:, 2].max()), 4)]}

    # ---------------------------------------------------------------- 3. InstantMesh frame
    R = np.stack([plane["e1"], plane["e2"], plane["n"]])
    if np.linalg.det(R) < 0:
        R[1] *= -1                                                # keep it right-handed (no mirrored world)
    sel = oop > 0.3
    lo, hi = np.percentile(ouvh[sel], 0.5, 0), np.percentile(ouvh[sel], 99.5, 0)
    c_uvh = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (min(lo[2], 0.0) + hi[2]) / 2])
    c = mio.from_uvh(plane, c_uvh[None])[0]
    Xi = (X[sel] - c) @ R.T
    s = FILL / np.percentile(np.abs(Xi), 99.5, 0).max()
    frame = {"c": c.tolist(), "R": R.tolist(), "s": float(s), "fill": FILL, "object_id": oid,
             "note": "X_im = s * R @ (X_scene - c); X_scene = c + R.T @ X_im / s. z_im = table normal (up)."}
    (out / "frame.json").write_text(json.dumps(frame, indent=1))

    # ---------------------------------------------------------------- 2 + 4. observations + eval masks
    cams = cams_from_json(step_dir(run, 1) / "cameras.json")
    mdir = step_dir(run, 6) / "1_masks_synced"
    names_c = sorted(n for n in cams if (mdir / f"{n}.png").exists())
    # table label = most frequent label in a ring around the object (never an occluder)
    cnt = {}
    for n in names_c[::6]:
        lab_m = np.array(Image.open(mdir / f"{n}.png"))
        m = lab_m == oid
        if m.any():
            ring = ndimage.binary_dilation(m, iterations=6) & ~m
            for k_, n_ in zip(*np.unique(lab_m[ring], return_counts=True)):
                cnt[int(k_)] = cnt.get(int(k_), 0) + int(n_)
    cnt.pop(0, None); cnt.pop(oid, None)
    table_label = max(cnt, key=cnt.get)
    obs = np.zeros((len(X), len(names_c)), bool)
    real = {}
    for j, n in enumerate(names_c):
        cm = cams[n]
        lab_m = np.array(Image.open(mdir / f"{n}.png"))
        H, W = lab_m.shape
        if (W, H) != (cm["W"], cm["H"]):
            # masks at a lower --resolution than cameras.json (e.g. Inpaint360GS res 2): rescale intrinsics
            sx, sy = W / cm["W"], H / cm["H"]
            assert abs(sx - sy) < 1e-2, (n, W, H, cm["W"], cm["H"])
            cm = cams[n] = {**cm, "W": W, "H": H, "K": [cm["K"][0] * sx, cm["K"][1] * sy, W / 2, H / 2]}
        w2c = np.linalg.inv(cm["c2w_cv"])
        p = X @ w2c[:3, :3].T + w2c[:3, 3]
        fx, fy, cx, cy = cm["K"]
        u = fx * p[:, 0] / p[:, 2] + cx; vv = fy * p[:, 1] / p[:, 2] + cy
        ins = (p[:, 2] > 0.01) & (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
        ui, vi = np.clip(u.astype(int), 0, W - 1), np.clip(vv.astype(int), 0, H - 1)
        obs[:, j] = ins & (lab_m[vi, ui] == oid)
        m = lab_m == oid
        entry = {"c2w_cv": cm["c2w_cv"].tolist(), "K": cm["K"], "W": W, "H": H, "object_pixels": int(m.sum())}
        if m.any():
            ys, xs = np.nonzero(m)
            entry["border"] = bool(ys.min() == 0 or xs.min() == 0 or ys.max() == H - 1 or xs.max() == W - 1)
            entry["inside_fraction"] = round(float(ins.mean()), 4)
            em = np.zeros_like(lab_m)
            em[(lab_m != 0) & (lab_m != table_label) & ~m] = 128
            em[m] = 255
            em = cv2.resize(em, (int(round(W * EVAL_SCALE)), int(round(H * EVAL_SCALE))), interpolation=cv2.INTER_NEAREST)
            Image.fromarray(em).save(out / "masks_eval" / f"{n}.png")
        real[n] = entry
    cam_centers = np.array([cams[n]["c2w_cv"][:3, 3] for n in names_c])
    np.savez_compressed(out / "observations.npz", obs=obs, cam_centers=cam_centers, cam_names=np.array(names_c))
    (out / "real_cameras.json").write_text(json.dumps({"eval_scale": EVAL_SCALE, "table_label": table_label,
                                                        "cameras": real}, indent=1))
    nobs = obs.sum(1)
    stats["observations"] = {"cameras": len(names_c), "cameras_seeing_object": int((obs.sum(0) > 0).sum()),
                             "per_gaussian_p05_p50_p95": [int(np.percentile(nobs, q)) for q in (5, 50, 95)],
                             "never_observed": int((nobs == 0).sum()), "table_label": table_label}

    # ---------------------------------------------------------------- 5. renderer check vs photos (1/2 res)
    g = gs_render.load_ply(out / f"point_cloud_{oid}_object.ply")
    cand = [n for n in names_c if real[n]["object_pixels"] > 0 and not real[n].get("border", True)]
    cand = sorted(cand, key=lambda n: -real[n]["object_pixels"])[:: max(1, len(cand) // 6)][:6]
    img_dir = step_dir(run, 0) / "images"
    checks, tiles = {}, []
    for n in cand:
        cm = cams[n]
        Wh, Hh = cm["W"] // 2, cm["H"] // 2
        Kh = [k / 2 for k in cm["K"]]
        r = gs_render.render(g, cm["c2w_cv"], Kh, Wh, Hh, bg=(0, 0, 0))
        rgb, a = r["rgb"].numpy(), r["alpha"].numpy()
        photo = np.asarray(Image.open(next(img_dir.glob(f"{n}.*"))).convert("RGB").resize((Wh, Hh), Image.BILINEAR), np.float32) / 255
        m = cv2.resize((np.array(Image.open(mdir / f"{n}.png")) == oid).astype(np.uint8), (Wh, Hh), interpolation=cv2.INTER_NEAREST) > 0
        pa = a > 0.5
        iou = float((pa & m).sum() / max((pa | m).sum(), 1))
        both = pa & m
        psnr = float(-10 * np.log10(((rgb[both] - photo[both]) ** 2).mean() + 1e-12)) if both.any() else None
        checks[n] = {"alpha_mask_iou": round(iou, 4), "psnr_inside_both": round(psnr, 2) if psnr else None}
        ys, xs = np.nonzero(m | pa)
        cy_, cx_ = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
        h = max(np.ptp(ys), np.ptp(xs)) // 2 + 12
        sl = (slice(max(cy_ - h, 0), cy_ + h), slice(max(cx_ - h, 0), cx_ + h))
        t = np.concatenate([photo[sl], rgb[sl]], 1)
        tiles.append(Image.fromarray((t * 255).astype(np.uint8)).resize((400, 200)))
    sheet = Image.new("RGB", (800, 200 * ((len(tiles) + 1) // 2)))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % 2) * 400, (i // 2) * 200))
    sheet.save(out / "renderer_check" / "photo_vs_render.jpg", quality=92)
    stats["renderer_check"] = checks
    stats["renderer_check_mean"] = {"alpha_mask_iou": round(float(np.mean([c_["alpha_mask_iou"] for c_ in checks.values()])), 4),
                                    "psnr": round(float(np.mean([c_["psnr_inside_both"] for c_ in checks.values() if c_["psnr_inside_both"]])), 2)}
    (out / "summary.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
