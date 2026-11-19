"""
Tự tìm mặt đỡ (bàn) trong một tập Gaussian theo từng vật (mỗi vật một file .ply 3DGS), không cần nhãn "bàn".

Với mỗi vật: lọc opacity, RANSAC mặt phẳng (ngưỡng = 2 x trung vị bán kính Gaussian của vật đó). Mặt đỡ = vật có nhiều
điểm nằm trên mặt phẳng nhất và tỉ lệ điểm trên mặt phẳng >= --min_inlier_ratio. Pháp tuyến hướng về phía đa số các vật khác.
Tỉ lệ mét không biết -> meters_per_unit = 0.01 (đơn vị "cm" trong các tool sau = đơn vị cảnh); dùng kèm support_graph --scale_free.

    python tools/auto_support_plane.py --objects Data/figurines/viewable_ply/objects --out Data/figurines/support_graph
Xuất: support.json (table.normal, table.d, table_id, meters_per_unit, scale_known=false), candidates.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from support_graph import read_ply_gauss  # noqa: E402


def ransac_plane(p, thresh, iters=1500, seed=0):
    rng = np.random.default_rng(seed)
    best, best_n, best_d = None, None, None
    for _ in range(iters):
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n /= nn
        d = -n @ a
        inl = np.abs(p @ n + d) < thresh
        if best is None or inl.sum() > best.sum():
            best, best_n, best_d = inl, n, d
    q = p[best]                                                   # tinh chỉnh bằng SVD trên inlier
    c = q.mean(0)
    n = np.linalg.svd(q - c)[2][-1]
    return n, -n @ c, np.abs(p @ n - n @ c) < thresh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_inlier_ratio", type=float, default=0.5)
    ap.add_argument("--max_points", type=int, default=60000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pts, cand = {}, {}
    for f in sorted(Path(a.objects).glob("*.ply")):
        p = read_ply_gauss(f)
        r = read_ply_gauss.last_scale
        if len(p) < 50:
            continue
        if len(p) > a.max_points:
            idx = np.random.default_rng(0).choice(len(p), a.max_points, replace=False)
            p, r = p[idx], r[idx]
        pts[f.stem] = p
        n, d, inl = ransac_plane(p, 2 * float(np.median(r)))
        cand[f.stem] = {"points": int(len(p)), "inliers": int(inl.sum()), "ratio": float(inl.mean()), "normal": n.tolist(), "d": float(d)}
    ok = {k: c for k, c in cand.items() if c["ratio"] >= a.min_inlier_ratio}
    tid = max(ok, key=lambda k: ok[k]["inliers"])
    n, d = np.array(cand[tid]["normal"]), cand[tid]["d"]
    others = np.concatenate([p for k, p in pts.items() if k != tid])
    if np.median(others @ n + d) < 0:
        n, d = -n, -d
    sup = {"table": {"normal": n.tolist(), "d": float(d), "inliers": cand[tid]["inliers"]}, "table_id": tid,
           "meters_per_unit": 0.01, "scale_known": False,
           "note": "mat do tu tim (RANSAC tung vat, chon vat phang nhieu diem nhat); don vi 'cm' = don vi canh"}
    (out / "support.json").write_text(json.dumps(sup, indent=2))
    (out / "candidates.json").write_text(json.dumps(dict(sorted(cand.items(), key=lambda x: -x[1]["inliers"])), indent=2))
    top = sorted(cand.items(), key=lambda x: -x[1]["inliers"])[:6]
    print("mat do:", tid, cand[tid]["inliers"], f"{cand[tid]['ratio']:.2f}")
    for k, c in top:
        print(f"  {k}: inlier {c['inliers']} / {c['points']} ({c['ratio']:.2f})")


if __name__ == "__main__":
    main()
