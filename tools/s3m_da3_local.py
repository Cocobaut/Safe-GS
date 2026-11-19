"""
Zero-GPU v5_da3 frames: apply the DA3 free-space carve of pipeline/da3_carve.py (same function s3m_mold.py --da3_dir uses on
Modal) to an existing S3m frame, then redo the shell / seen / mold classification exactly as s3m_mold.py does.
Seen votes come from the base frame (CoMe render depth), unchanged; DA3 only removes voxels.

    python tools/s3m_da3_local.py --base Data/teatime/pipeline/v3_sam3/S3m --out Data/teatime/pipeline/v3_sam3/S3m/v5_da3 \
        --da3 Data/teatime/da3_depth --colmap Data/teatime/colmap --s3 Data/teatime/pipeline/v3_sam3/S3_label
Per object output is written then skipped on rerun (resume).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline")); sys.path.insert(0, str(ROOT / "tools"))
from da3_carve import load_da3, da3_thru_votes, gauss_voxels  # noqa: E402
from da3_depth import colmap_views  # noqa: E402


def shell_of(V, vox):
    ijk = np.round((V - V.min(0)) / vox).astype(np.int64)
    s = 1 << 20
    key = lambda q: ((q[:, 0] + s) << 42) | ((q[:, 1] + s) << 21) | (q[:, 2] + s)
    keys = np.sort(key(ijk)); inner = np.ones(len(V), bool)
    for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        k = key(ijk + np.array(d)); i = np.minimum(np.searchsorted(keys, k), len(keys) - 1)
        inner &= keys[i] == k
    return ~inner


def read_gauss(ply, op_min):
    from plyfile import PlyData
    e = PlyData.read(str(ply))["vertex"]
    G = np.stack([e["x"], e["y"], e["z"]], 1).astype(np.float64)
    return G[1 / (1 + np.exp(-np.asarray(e["opacity"], np.float64))) > op_min], G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--out", required=True); ap.add_argument("--da3", required=True)
    ap.add_argument("--colmap", required=True); ap.add_argument("--s3", required=True)
    ap.add_argument("--tol_cm", type=float, default=1.0); ap.add_argument("--min_vis", type=int, default=3)
    ap.add_argument("--k_res", type=float, default=2.0); ap.add_argument("--edge", type=float, default=0.03)
    ap.add_argument("--op", type=float, default=0.5); ap.add_argument("--objects", default="")
    a = ap.parse_args()
    base, out = Path(a.base), Path(a.out); (out / "frame").mkdir(parents=True, exist_ok=True)
    sup = json.loads((Path(a.s3) / "support.json").read_text())
    n_t, d_t, cm = np.array(sup["table"]["normal"]), float(sup["table"]["d"]), 100 * float(sup["meters_per_unit"])
    cams = [{"name": Path(v["name"]).stem, "K": v["K"], "T": v["T"], "W": v["W"], "H": v["H"]}
            for v in colmap_views(Path(a.colmap) / "sparse" / "0")]
    da3 = load_da3(a.da3)
    base_m = json.loads((base / "metrics.json").read_text())["B_carve"]["objects"]
    mf = out / "metrics.json"
    res = json.loads(mf.read_text()) if mf.exists() else {"B_carve": {"objects": {}}}
    res["B_carve"].update({"base": str(base), "tol_cm": a.tol_cm, "min_vis": a.min_vis, "da3_k_res": a.k_res,
                           "da3_edge": a.edge, "da3_op": a.op, "views": len(cams), "da3_views": len(da3)})
    names = sorted(p.stem for p in (base / "frame").glob("*.npz"))
    if a.objects:
        names = [n for n in names if n in a.objects.split(",")]
    for n in names:
        if n in res["B_carve"]["objects"] and (out / "frame" / f"{n}.npz").exists():
            print(n, "skip (done)"); continue
        z = np.load(base / "frame" / f"{n}.npz")
        X = z["xyz"].astype(np.float64); vox = float(z["voxel"]); cls0 = z["cls"]
        th, va = da3_thru_votes(X, cams, da3, a.tol_cm / cm, a.k_res, a.edge)
        Gop, Gall = read_gauss(Path(a.s3) / "gaussians" / f"{n}.ply", a.op)
        matter = gauss_voxels(X, vox, Gop) if len(Gop) else np.zeros(len(X), bool)
        free = (th >= a.min_vis) & ~matter
        keep = ~free
        F = X[keep]; seen = cls0[keep] == 1
        shl = shell_of(F, vox)
        h = (F @ n_t + d_t) * cm
        hp98, hlow = float(np.percentile(h, 98)), float(np.percentile(h, 2))
        bottom = h <= hlow + max(1.0, 0.15 * (hp98 - hlow))
        mold = shl & ~seen
        cls = np.where(mold & bottom, 3, np.where(mold, 4, np.where(seen, 1, 2))).astype(np.uint8)
        np.savez_compressed(out / "frame" / f"{n}.npz", xyz=F.astype(np.float32), cls=cls, voxel=vox,
                            surf=z["surf"][keep], thru=z["thru"][keep], da3_thru=th[keep].astype(np.int16))
        hg = (Gall @ n_t + d_t) * cm
        r = {"frame_cm3": float(len(F) * (vox * cm) ** 3), "base_frame_cm3": float(len(X) * (vox * cm) ** 3),
             "da3_carved_pct": 100 * float(free.mean()), "da3_blocked_by_gaussians": int(((th >= a.min_vis) & matter).sum()),
             "da3_valid_votes_median": float(np.median(va)), "voxel_cm": vox * cm,
             "height_p98_cm": {"hull": base_m.get(n, {}).get("height_p98_cm", {}).get("hull"), "frame": hp98,
                               "base_frame": float(np.percentile((X @ n_t + d_t) * cm, 98)), "gaussians": float(np.percentile(hg, 98))},
             "shell": int(shl.sum()), "shell_seen_pct": 100 * float((shl & seen).sum() / max(shl.sum(), 1)),
             "mold_bottom_pct": 100 * float((mold & bottom).sum() / max(shl.sum(), 1)),
             "mold_side_back_pct": 100 * float((mold & ~bottom).sum() / max(shl.sum(), 1))}
        res["B_carve"]["objects"][n] = r
        mf.write_text(json.dumps(res, indent=2, ensure_ascii=False))     # checkpoint after every object
        print(n, json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k != "height_p98_cm"}),
              {k: round(v, 2) for k, v in r["height_p98_cm"].items() if v is not None}, flush=True)


if __name__ == "__main__":
    main()
