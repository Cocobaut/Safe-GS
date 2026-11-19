"""
Free-space carving from Depth Anything 3 depth (arXiv 2511.10647; depth from tools/da3_depth.py, aligned per view to the
COLMAP sparse observations). Pure numpy so it runs both inside pipeline/s3m_mold.py (Modal) and locally.

Vote per voxel and view (all views, no grazing gating): DA3 pixel valid (finite, not on a depth edge) and
    z_voxel < d_DA3 - tol_v,   tol_v = max(tol, k * median |alignment residual| of that view)
-> the camera sees through the voxel. Free = votes >= min_vis, except voxels that contain an opaque object Gaussian
centre (observed matter is never carved). Same rule for every object.
"""
import json
from pathlib import Path

import numpy as np


def edge_mask(D, rel, r=2):
    """True where DA3 depth is unreliable: local (2r+1)^2 depth range > rel * depth (occlusion boundaries)."""
    import cv2
    k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    D = D.astype(np.float32)
    return (cv2.dilate(D, k) - cv2.erode(D, k)) > rel * D


def load_da3(da3_dir):
    da3_dir = Path(da3_dir)
    stats = json.loads((da3_dir / "align_stats.json").read_text())["views"]
    out = {}
    for st, s in stats.items():
        if s.get("ok"):
            z = np.load(da3_dir / "aligned" / f"{st}.npz")
            out[st] = {"D": z["depth"].astype(np.float32), "res": float(s["med_abs_res"])}
    return out


def da3_thru_votes(X, cams, da3, tol, k_res=2.0, edge_rel=0.03):
    """X [N,3] world points. cams: list of dict(name, K 3x3 at (W,H), T 4x4 w2c, W, H). Returns (thru, valid) vote counts."""
    thru = np.zeros(len(X), np.int32); valid = np.zeros(len(X), np.int32)
    for c in cams:
        e = da3.get(c["name"])
        if e is None:
            continue
        D = e["D"]; h, w = D.shape
        E = edge_mask(D, edge_rel) if edge_rel > 0 else np.zeros_like(D, bool)
        tv = max(tol, k_res * e["res"])
        Xc = X @ c["T"][:3, :3].T + c["T"][:3, 3]
        z = Xc[:, 2]
        u = c["K"][0, 0] * Xc[:, 0] / np.maximum(z, 1e-6) + c["K"][0, 2]
        q = c["K"][1, 1] * Xc[:, 1] / np.maximum(z, 1e-6) + c["K"][1, 2]
        pu, pv = np.floor(u * w / c["W"]).astype(int), np.floor(q * h / c["H"]).astype(int)
        ok = (z > 1e-3) & (pu >= 0) & (pu < w) & (pv >= 0) & (pv < h)
        d = np.full(len(X), np.nan, np.float32)
        d[ok] = D[pv[ok], pu[ok]]
        ok &= np.isfinite(d) & (d > 0)
        ok[ok] &= ~E[pv[ok], pu[ok]]
        valid += ok.astype(np.int32)
        thru += (ok & (z < d - tv)).astype(np.int32)
    return thru, valid


def gauss_voxels(X, vox, G):
    """Mask of voxels of X (grid spacing vox, any origin aligned with X) that contain a centre of G."""
    o = X.min(0); s = 1 << 20
    key = lambda q: ((q[:, 0] + s) << 42) | ((q[:, 1] + s) << 21) | (q[:, 2] + s)
    return np.isin(key(np.round((X - o) / vox).astype(np.int64)), np.unique(key(np.round((G - o) / vox).astype(np.int64))))
