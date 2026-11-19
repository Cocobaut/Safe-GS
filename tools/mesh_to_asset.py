"""
Đổi mesh của tools/frame_mesh.py (<vật>/scene_part_<k>.obj, hệ bàn, mét) thành "asset" cho tools/build_mold_inspector.py:
<out>/<vật>.npz (xyz = điểm lấy mẫu trên bề mặt, hệ thế giới / cảnh COLMAP) + <out>/metrics.json (thể tích, genus, kín, chốt độ dày).
Biến đổi ngược đúng của frame_mesh.py: P_bàn = mpu * R (P - p0)  =>  P = R^T P_bàn / mpu + p0.

    .venv_da3/Scripts/python.exe tools/mesh_to_asset.py --meshes Data/teatime/pipeline/v3_sam3/mesh_v5 \
        --support Data/teatime/pipeline/v3_sam3/S3_label/support.json --out Data/teatime/pipeline/v3_sam3/mesh_v5/asset
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meshes", required=True)
    ap.add_argument("--support", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20000)
    a = ap.parse_args()
    sup = json.loads(Path(a.support).read_text())
    n_t, d_t, mpu = np.array(sup["table"]["normal"]), float(sup["table"]["d"]), float(sup["meters_per_unit"])
    b0 = np.array([1.0, 0, 0]) if abs(n_t[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(n_t, b0); e1 /= np.linalg.norm(e1); e2 = np.cross(n_t, e1)
    R = np.stack([e1, e2, n_t], 0); p0 = -d_t * n_t
    md, out = Path(a.meshes), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = json.loads((md / "report.json").read_text())
    metrics = {}
    for name, r in rep.items():
        parts = sorted((md / name).glob("scene_part_*.obj")) if "parts" in r else []
        if not parts:
            continue
        ms = [trimesh.load(p, force="mesh") for p in parts]
        m = trimesh.util.concatenate(ms)
        pts = m.sample(a.n, seed=0) if hasattr(m, "sample") else m.vertices
        X = (R.T @ (np.asarray(pts) / mpu).T).T + p0
        np.savez_compressed(out / f"{name}.npz", xyz=X.astype(np.float32))
        metrics[name] = {"volume_cm3": float(sum(q["volume_cm3"] for q in r["parts"])),
                         "parts": len(r["parts"]), "genus": [q["genus"] for q in r["parts"]],
                         "watertight": all(q["watertight_edgehash"] for q in r["parts"]),
                         "thick_guard_dilate_vox": r.get("thick_guard_dilate_vox", 0)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    print(f"{len(metrics)} vat -> {out}")


if __name__ == "__main__":
    main()
