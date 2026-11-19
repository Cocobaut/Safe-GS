"""
Depth Anything 3 (arXiv 2511.10647, ByteDance-Seed/Depth-Anything-3 @ 3d835ec1, checkpoint depth-anything/DA3-BASE, Apache-2.0)
multi-view depth for every COLMAP view, then per-view scale + shift alignment to the COLMAP sparse observations.

1. raw:   DA3 inference in chunks of --chunk views (sorted by name = trajectory order), pose-conditioned on the COLMAP
          w2c extrinsics + intrinsics (DA3 rescales its depth to the input pose scale). One npz per view
          (depth, conf at DA3 process resolution), written atomically; existing files are skipped (resume).
2. align: for each view, every 2D observation with a 3D point gives (pixel, z_cam). Robust fit z = s * d + t
          (RANSAC on 2-point models + least squares on inliers, |res| <= max(0.5 %, 2 MAD)). Saved: aligned depth in COLMAP
          units (float16), conf, and per-view stats (n obs, s, t, inlier ratio, median |res| in scene units).
          Same rule for every view; nothing per object.

    python tools/da3_depth.py --colmap Data/teatime/colmap --out Data/teatime/da3_depth            (local GPU)
Modal wrapper with Volume commit after each chunk: modal_da3.py
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def colmap_views(sparse):
    import pycolmap
    r = pycolmap.Reconstruction(str(sparse))
    views = []
    for iid, im in r.images.items():
        cam = r.cameras[im.camera_id]
        assert str(cam.model).endswith("PINHOLE"), cam.model
        fx, fy, cx, cy = cam.params
        T = np.eye(4); T[:3] = im.cam_from_world().matrix()
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
        uv, z = [], []
        for p2 in im.points2D:
            if p2.has_point3D():
                X = r.points3D[p2.point3D_id].xyz
                Xc = T[:3, :3] @ X + T[:3, 3]
                if Xc[2] > 1e-6:
                    uv.append(p2.xy); z.append(Xc[2])
        views.append({"name": im.name, "K": K, "T": T, "W": cam.width, "H": cam.height,
                      "uv": np.array(uv, np.float64).reshape(-1, 2), "z": np.array(z, np.float64)})
    return sorted(views, key=lambda v: v["name"])


def save_atomic(path, **arrs):
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, path)


def run_raw(views, img_dir, out, model_name, chunk, process_res, on_chunk=None):
    raw = out / "raw"; raw.mkdir(parents=True, exist_ok=True)
    todo = [v for v in views if not (raw / (Path(v["name"]).stem + ".npz")).exists()]
    print(f"[raw] {len(views) - len(todo)}/{len(views)} views already on disk, {len(todo)} to do", flush=True)
    if not todo:
        return
    import torch
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained(model_name).to("cuda").eval()
    # chunks over the FULL ordered list so a resumed run builds the same chunks; a chunk with any missing view is redone
    for i in range(0, len(views), chunk):
        ch = views[i:i + chunk]
        if all((raw / (Path(v["name"]).stem + ".npz")).exists() for v in ch):
            continue
        t0 = time.time()
        with torch.no_grad():
            pr = model.inference([str(img_dir / v["name"]) for v in ch],
                                 extrinsics=np.stack([v["T"] for v in ch]).astype(np.float32),
                                 intrinsics=np.stack([v["K"] for v in ch]).astype(np.float32),
                                 align_to_input_ext_scale=True, process_res=process_res)
        for j, v in enumerate(ch):
            save_atomic(raw / (Path(v["name"]).stem + ".npz"), depth=pr.depth[j].astype(np.float32),
                        conf=pr.conf[j].astype(np.float16), W=v["W"], H=v["H"])
        print(f"[raw] views {i}-{i + len(ch) - 1}: depth {pr.depth.shape} {time.time() - t0:.1f}s", flush=True)
        if on_chunk:
            on_chunk()
        del pr
        torch.cuda.empty_cache()


def robust_fit(d, z, rng, iters=500):
    """z = s d + t. RANSAC (relative residual), then least squares on inliers."""
    best, best_in = (1.0, 0.0), None
    thr = 0.005 * np.median(z)
    for _ in range(iters):
        a, b = rng.choice(len(d), 2, replace=False)
        if abs(d[a] - d[b]) < 1e-6:
            continue
        s = (z[a] - z[b]) / (d[a] - d[b]); t = z[a] - s * d[a]
        if s <= 0:
            continue
        inl = np.abs(s * d + t - z) <= thr
        if best_in is None or inl.sum() > best_in.sum():
            best, best_in = (s, t), inl
    s, t = best
    for _ in range(3):
        res = s * d + t - z
        mad = 1.4826 * np.median(np.abs(res - np.median(res)))
        inl = np.abs(res) <= max(thr, 2 * mad)
        A = np.stack([d[inl], np.ones(inl.sum())], 1)
        s, t = np.linalg.lstsq(A, z[inl], rcond=None)[0]
    res = s * d + t - z
    return float(s), float(t), inl, res


def run_align(views, out):
    raw, al = out / "raw", out / "aligned"
    al.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    stats = {}
    for v in views:
        st = Path(v["name"]).stem
        z_ = np.load(raw / f"{st}.npz")
        D, C = z_["depth"].astype(np.float64), z_["conf"].astype(np.float32)
        h, w = D.shape
        px = np.clip((v["uv"][:, 0] * w / v["W"]).astype(int), 0, w - 1)
        py = np.clip((v["uv"][:, 1] * h / v["H"]).astype(int), 0, h - 1)
        d, z = D[py, px], v["z"]
        ok = np.isfinite(d) & (d > 0)
        d, z = d[ok], z[ok]
        if len(d) < 20:
            stats[st] = {"n_obs": int(len(d)), "ok": False}
            continue
        # raw DA3 (already in input pose scale) vs COLMAP before the per-view fit
        raw_rel = float(np.median(np.abs(d - z) / z))
        s, t, inl, res = robust_fit(d, z, rng)
        A = (s * D + t).astype(np.float16)
        save_atomic(al / f"{st}.npz", depth=A, conf=C.astype(np.float16), W=v["W"], H=v["H"], s=s, t=t)
        stats[st] = {"n_obs": int(len(d)), "ok": True, "s": s, "t": t, "inlier_pct": 100 * float(inl.mean()),
                     "raw_med_rel_err": raw_rel, "med_abs_res": float(np.median(np.abs(res[inl]))),
                     "med_abs_res_all": float(np.median(np.abs(res))), "med_rel_res": float(np.median(np.abs(res) / z)),
                     "res_p90_abs": float(np.percentile(np.abs(res), 90))}
    ok = [s for s in stats.values() if s.get("ok")]
    summ = {"views": len(views), "aligned": len(ok),
            "median_over_views": {k: float(np.median([s[k] for s in ok])) for k in
                                  ("n_obs", "s", "inlier_pct", "raw_med_rel_err", "med_abs_res", "med_rel_res", "res_p90_abs")}}
    (out / "align_stats.json").write_text(json.dumps({"summary": summ, "views": stats}, indent=1))
    print("[align]", json.dumps(summ), flush=True)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--colmap", required=True, help="dir with images/ and sparse/0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="depth-anything/DA3-BASE")
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--process_res", type=int, default=504)
    ap.add_argument("--limit", type=int, default=0, help="only the first N views (incremental check)")
    ap.add_argument("--skip_raw", action="store_true")
    a = ap.parse_args()
    col, out = Path(a.colmap), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    views = colmap_views(col / "sparse" / "0")
    if a.limit:
        views = views[:a.limit]
    print(f"{len(views)} views, median {int(np.median([len(v['z']) for v in views]))} COLMAP obs/view", flush=True)
    if not a.skip_raw:
        run_raw(views, col / "images", out, a.model, a.chunk, a.process_res)
    run_align(views, out)
    (out / "meta.json").write_text(json.dumps({"model": a.model, "repo_commit": "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4",
                                               "chunk": a.chunk, "process_res": a.process_res, "pose_conditioned": True}, indent=1))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Based_Model" / "Depth-Anything-3" / "src"))
    main()
