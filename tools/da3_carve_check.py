"""
Local zero-GPU pre-check for DA3 free-space carving: count DA3 'see-through' votes for every voxel of an existing S3m frame
(COLMAP cameras, aligned DA3 depth from tools/da3_depth.py) and report what the carve would remove, per object, under the
same rule for all objects. Used only to decide whether the Modal s3m run (pipeline/s3m_mold.py --da3_dir) is worth it.

    python tools/da3_carve_check.py --frames Data/teatime/pipeline/v3_sam3/S3m/frame --da3 Data/teatime/da3_depth \
        --colmap Data/teatime/colmap --support Data/teatime/pipeline/v3_sam3/S3_label/support.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from da3_depth import colmap_views  # noqa: E402


def edge_mask(D, rel, r=2):
    """True where DA3 depth is unreliable: local (2r+1)^2 depth range > rel * depth (occlusion boundaries)."""
    import cv2
    k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    return (cv2.dilate(D, k) - cv2.erode(D, k)) > rel * D


def gauss_keep(X, vox, ply, op_min=0.5, dil=1):
    """Voxels within dil voxels of an opaque object Gaussian centre (observed matter): never carved."""
    from plyfile import PlyData
    e = PlyData.read(str(ply))["vertex"]
    G = np.stack([e["x"], e["y"], e["z"]], 1).astype(np.float64)
    G = G[1 / (1 + np.exp(-np.asarray(e["opacity"], np.float64))) > op_min]
    o = X.min(0)
    key = lambda q: (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]
    s = 1 << 20
    gi = np.round((G - o) / vox).astype(np.int64) + s
    ks = [key(gi + np.array(d)) for d in np.ndindex(2 * dil + 1, 2 * dil + 1, 2 * dil + 1) for d in [np.array(d) - dil]]
    ks = np.unique(np.concatenate(ks))
    xi = key(np.round((X - o) / vox).astype(np.int64) + s)
    return np.isin(xi, ks), G


def da3_votes(X, views, da3, tol_units, tol_k, conf_q, edge_rel=0.0):
    stats = json.loads((da3 / "align_stats.json").read_text())["views"]
    thru = np.zeros(len(X), np.int32); seen = np.zeros(len(X), np.int32)
    for v in views:
        st = Path(v["name"]).stem
        if not stats.get(st, {}).get("ok"):
            continue
        z_ = np.load(da3 / "aligned" / f"{st}.npz")
        D, C = z_["depth"].astype(np.float32), z_["conf"].astype(np.float32)
        h, w = D.shape
        cthr = np.percentile(C, conf_q) if conf_q > 0 else -np.inf
        tv = max(tol_units, tol_k * stats[st]["med_abs_res"])
        Xc = X @ v["T"][:3, :3].T + v["T"][:3, 3]
        z = Xc[:, 2]
        u = v["K"][0, 0] * Xc[:, 0] / np.maximum(z, 1e-6) + v["K"][0, 2]
        q = v["K"][1, 1] * Xc[:, 1] / np.maximum(z, 1e-6) + v["K"][1, 2]
        pu, pv = (u * w / v["W"]).astype(int), (q * h / v["H"]).astype(int)
        ok = (z > 1e-3) & (pu >= 0) & (pu < w) & (pv >= 0) & (pv < h)
        d = np.full(len(X), np.nan, np.float32); c = np.zeros(len(X), np.float32)
        d[ok], c[ok] = D[pv[ok], pu[ok]], C[pv[ok], pu[ok]]
        ok &= np.isfinite(d) & (c >= cthr)
        if edge_rel > 0:
            E = edge_mask(D, edge_rel)
            ok[ok] &= ~E[pv[ok], pu[ok]]
        thru += (ok & (z < d - tv)).astype(np.int32)
        seen += (ok & (np.abs(z - d) <= tv)).astype(np.int32)
    return thru, seen


def gauss_lost(G, Xc, vox):
    """% of opaque object Gaussian centres whose nearest voxel was carved (over-carving of observed matter)."""
    if len(Xc) == 0 or len(G) == 0:
        return 0.0
    from scipy.spatial import cKDTree
    return 100 * float((cKDTree(Xc).query(G, distance_upper_bound=0.5 * vox)[0] < np.inf).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--da3", required=True)
    ap.add_argument("--colmap", required=True)
    ap.add_argument("--support", required=True)
    ap.add_argument("--gauss", default="Data/teatime/pipeline/v3_sam3/S3_label/gaussians")
    ap.add_argument("--min_vis", type=int, default=3)
    ap.add_argument("--objects", default="")
    a = ap.parse_args()
    sup = json.loads(Path(a.support).read_text())
    n_t, d_t, cm = np.array(sup["table"]["normal"]), float(sup["table"]["d"]), 100 * float(sup["meters_per_unit"])
    views = colmap_views(Path(a.colmap) / "sparse" / "0")
    da3 = Path(a.da3)
    names = sorted(p.stem for p in Path(a.frames).glob("*.npz"))
    if a.objects:
        names = [n for n in names if n in a.objects.split(",")]
    res = {}
    for n in names:
        z = np.load(Path(a.frames) / f"{n}.npz")
        X = z["xyz"].astype(np.float64); vox = float(z["voxel"])
        h = (X @ n_t + d_t) * cm
        r = {"voxels": int(len(X)), "cm3": float(len(X) * (vox * cm) ** 3), "h_p98": float(np.percentile(h, 98))}
        gk, G = gauss_keep(X, vox, Path(a.gauss) / f"{n}.ply")
        r["gauss_voxels_pct"] = 100 * float(gk.mean())
        for tag, tol_cm, k, cq, er, guard in (("tol1", 1.0, 0, 0, 0, False), ("tol1_edge", 1.0, 0, 0, 0.03, False),
                                              ("tol1_edge_guard", 1.0, 0, 0, 0.03, True), ("tol1_k2_edge_guard", 1.0, 2.0, 0, 0.03, True)):
            th, se = da3_votes(X, views, da3, tol_cm / cm, k, cq, er)
            keep = (th < a.min_vis) | (gk if guard else False)
            r[tag] = {"carved_pct": 100 * float((~keep).mean()), "cm3": float(keep.sum() * (vox * cm) ** 3),
                      "h_p98": float(np.percentile(h[keep], 98)) if keep.any() else 0.0,
                      "thru_med": float(np.median(th)), "seen_med": float(np.median(se)),
                      "gauss_lost_pct": gauss_lost(G, X[~keep], vox)}
        res[n] = r
        print(n, json.dumps(r), flush=True)
    (da3 / "carve_check.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
