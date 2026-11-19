"""
Place generated 3D assets in the teatime scene and measure them against the observations (local, CPU).

  amodal3r: mesh.npz from modal_amodal3r.py (canonical [-0.5, 0.5]^3, z up, arbitrary yaw and scale).
            Registration, same for every object: up = table normal, bottom on the table plane, then a search over yaw
            (10 deg steps, both up-signs) and scale, refined by Nelder-Mead on the 80 %-trimmed mean distance from the
            observed matter (opaque object Gaussians, S3_label/gaussians/<obj>.ply) to the asset surface. The search only
            sees observed points, so the hidden part is free to be whatever the generator produced.
  trellis:  mesh.npz from modal_trellis.py (already in the frame grid: v * size + center in the table frame of
            trellis_in/<obj>/grid.npz). No registration needed.

Metrics (cm, cm3; table frame, z = height above table):
  volume_cm3     enclosed volume = voxel fill (trimesh voxelized(pitch).fill()) at the S3m voxel size, the same measure as
                 frame_cm3. Not the signed mesh volume: Amodal3R / TRELLIS meshes are often closed thin shells (outer + inner
                 wall), whose signed volume is only the wall material (Amodal3R tao: 8 cm3 vs 226 cm3 enclosed).
  mesh_signed_cm3  |signed mesh volume| (material of the shell), kept for reference
  watertight     trimesh.is_watertight
  height_p98_cm  p98 of surface samples above the table
  obs_cov_1cm    % of observed points (opaque Gaussians) within 1 cm of the asset surface
  obs_to_asset_med_cm / asset_to_obs_med_cm   median distances in both directions
  frame_iou      voxel IoU between the filled asset and the S3m frame of --frame (v3_sam3 by default)
Writes <out>/<obj>.npz (xyz = world surface samples, for tools/build_mold_inspector.py --asset) and <out>/metrics.json.

    python tools/align_generated.py --kind amodal3r --src Data/teatime/amodal3r_out/a1 --out Data/teatime/amodal3r_out/a1_aligned
    python tools/align_generated.py --kind trellis --src Data/teatime/trellis_out/t06_rp --out Data/teatime/trellis_out/t06_rp_aligned
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.optimize import minimize
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]


def table_frame(s3):
    sup = json.loads((s3 / "support.json").read_text())
    n = np.array(sup["table"]["normal"], np.float64); n /= np.linalg.norm(n)
    d = float(sup["table"]["d"]); cm = 100.0 * float(sup["meters_per_unit"])
    b0 = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(n, b0); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
    return np.stack([e1, e2, n], 0), -d * n, cm


def read_gauss(ply, op_min=0.5):
    from plyfile import PlyData
    e = PlyData.read(str(ply))["vertex"]
    G = np.stack([e["x"], e["y"], e["z"]], 1).astype(np.float64)
    return G[1 / (1 + np.exp(-np.asarray(e["opacity"], np.float64))) > op_min]


def transform(S, p):
    yaw, ls, tx, ty, tz = p
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return np.exp(ls) * S @ Rz.T + np.array([tx, ty, tz])


def trimmed(T, tree, q=80):
    d = tree.query(T)[0]
    return float(np.mean(np.sort(d)[: max(1, int(len(d) * q / 100))]))


def register(S, T):
    """S canonical surface samples (z up), T observed points in table frame (cm). Returns params, up sign, score."""
    best = None
    ext_T = np.percentile(T[:, :2], 98, 0) - np.percentile(T[:, :2], 2, 0)
    for up in (1, -1):
        Su = S * np.array([1, up, up])                       # flip about x: z -> -z keeps a proper rotation
        ext_S = np.percentile(Su[:, :2], 98, 0) - np.percentile(Su[:, :2], 2, 0)
        s0 = np.linalg.norm(ext_T) / max(np.linalg.norm(ext_S), 1e-9)
        for sf in (0.8, 0.9, 1.0, 1.1, 1.25):
            for yaw in np.deg2rad(np.arange(0, 360, 10)):
                ls = np.log(s0 * sf)
                X = transform(Su, (yaw, ls, 0, 0, 0))
                t = np.r_[np.median(T[:, :2], 0) - np.median(X[:, :2], 0), -np.percentile(X[:, 2], 0.5)]
                p = (yaw, ls, *t)
                sc = trimmed(T, cKDTree(transform(Su, p)))
                if best is None or sc < best[0]:
                    best = (sc, p, up)
    sc, p, up = best
    Su = S * np.array([1, up, up])
    sub = Su[np.random.default_rng(0).choice(len(Su), min(len(Su), 6000), replace=False)]
    f = lambda q: trimmed(T, cKDTree(transform(sub, q)))
    r = minimize(f, np.array(p), method="Nelder-Mead", options={"maxiter": 600, "xatol": 1e-3, "fatol": 1e-4})
    return r.x, up, float(r.fun), float(sc)


def fill_voxels(mesh, pitch):
    vg = mesh.voxelized(pitch).fill()
    return vg.points, vg


def metrics(mesh_t, T, frame_t, vox_cm):
    S = mesh_t.sample(30000)
    tS, tT = cKDTree(S), cKDTree(T)
    d_ts = tS.query(T)[0]; d_st = tT.query(S)[0]
    wt = bool(mesh_t.is_watertight)
    pts, vg = fill_voxels(mesh_t, vox_cm)
    vol = float(len(pts) * vox_cm ** 3)
    # IoU with the S3m frame on a common grid
    key = lambda P: set(map(tuple, np.round(P / vox_cm).astype(np.int64)))
    A, B = key(pts), key(frame_t)
    iou = len(A & B) / max(len(A | B), 1)
    return {"volume_cm3": vol, "mesh_signed_cm3": float(abs(mesh_t.volume)), "watertight": wt, "height_p98_cm": float(np.percentile(S[:, 2], 98)),
            "obs_cov_1cm": 100 * float((d_ts <= 1.0).mean()), "obs_to_asset_med_cm": float(np.median(d_ts)),
            "asset_to_obs_med_cm": float(np.median(d_st)), "frame_iou": float(iou),
            "frame_cm3": float(len(frame_t) * vox_cm ** 3)}, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["amodal3r", "trellis"], required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--s3", default=str(ROOT / "Data/teatime/pipeline/v3_sam3/S3_label"))
    ap.add_argument("--frame", default=str(ROOT / "Data/teatime/pipeline/v3_sam3/S3m/frame"))
    ap.add_argument("--trellis_in", default=str(ROOT / "Data/teatime/trellis_in"))
    ap.add_argument("--objects", default="dia,banh_quy,coc_su,ly_tra,tao")
    ap.add_argument("--force", action="store_true", help="recompute objects already in metrics.json")
    a = ap.parse_args()
    s3, src, out = Path(a.s3), Path(a.src), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    R, p0, cm = table_frame(s3)
    to_t = lambda X: (X - p0) @ R.T * cm
    to_w = lambda P: (P / cm) @ R + p0
    mf = out / "metrics.json"
    res = json.loads(mf.read_text()) if mf.exists() else {}
    for n in a.objects.split(","):
        if not (src / n / "mesh.npz").exists():
            print(n, "no mesh"); continue
        if n in res and (out / f"{n}.npz").exists() and not a.force:
            print(n, "skip (done)"); continue
        z = np.load(src / n / "mesh.npz")
        V, F = z["v"].astype(np.float64), z["f"].astype(np.int64)
        T = to_t(read_gauss(s3 / "gaussians" / f"{n}.ply"))
        fz = np.load(Path(a.frame) / f"{n}.npz")
        frame_t, vox_cm = to_t(fz["xyz"].astype(np.float64)), float(fz["voxel"]) * cm
        info = {}
        if a.kind == "trellis":
            g = np.load(Path(a.trellis_in) / n / "grid.npz")
            Vw = (V * float(g["size"]) + g["center"]) @ g["R"] + g["p0"]             # grid table frame (units) -> world
            Vt = to_t(Vw)
        else:
            S = trimesh.Trimesh(V, F, process=False).sample(20000)
            p, up, sc, sc0 = register(S, T)
            Vt = transform(V * np.array([1, up, up]), p)
            info = {"reg_yaw_deg": float(np.rad2deg(p[0]) % 360), "reg_scale_cm_per_unit": float(np.exp(p[1])), "reg_up_sign": int(up),
                    "reg_trimmed_cm": sc, "reg_grid_trimmed_cm": sc0}
        mesh_t = trimesh.Trimesh(Vt, F, process=True)
        m, Ssurf = metrics(mesh_t, T, frame_t, vox_cm)
        m.update(info)
        np.savez_compressed(out / f"{n}.npz", xyz=to_w(Ssurf).astype(np.float32))
        mesh_t.export(out / f"{n}_table_cm.ply")
        res[n] = m
        mf.write_text(json.dumps(res, indent=1))                             # checkpoint per object
        print(n, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()}), flush=True)


if __name__ == "__main__":
    main()
