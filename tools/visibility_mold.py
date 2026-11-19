"""
Khuôn inpaint cho từng vật: phần khung (visual hull) mà camera THẤY được và phần KHÔNG camera nào thấy (đáy, mặt sau, bị vật khác che).

Bước (cùng một cách cho mọi vật, không tham số riêng):
  1. Lưới chiếm chỗ (hash các ô nguyên) từ mọi Gaussian của cảnh (S3: objects_all + background_with_support), opacity >= 0.5;
     Gaussian lớn hơn 1 ô đánh dấu thêm 6 ô kề. Mặt bàn: mọi điểm thấp hơn mặt bàn coi như bị chặn.
  2. Vỏ hull của vật: voxel hull có ít nhất 1 trong 6 ô kề không thuộc hull.
  3. Với mỗi voxel vỏ và mỗi camera COLMAP: voxel nằm trong ảnh và phía trước camera; đi dọc tia từ voxel tới camera theo bước nửa ô
     (bỏ qua --skip_cm đầu tiên vì chính bề mặt vật nằm ngay đó); bị chặn nếu chạm ô chiếm chỗ hoặc xuống dưới mặt bàn.
     n_vis = số camera thấy voxel.
  4. Phân loại voxel vỏ: thấy (n_vis >= --min_vis), thấy ít (1 .. min_vis-1), KHÔNG thấy (0) = khuôn cần inpaint;
     khuôn chia "đáy" (cao <= max(1 cm, 15 % chiều cao vật)) và "bên / sau".
     Voxel thấy được nhưng không có Gaussian của vật trong --surf_cm = hull dư (chỗ khung còn lỏng).
  5. Cũng tính n_vis cho từng Gaussian của vật: Gaussian KHÔNG camera nào thấy = phần thêm thắt (theo nguyên tắc "không thấy thì để trống").
Xuất: mold_<vật>.ply (vỏ tô màu: xanh thấy, vàng thấy ít, đỏ đáy không thấy, tím bên / sau không thấy, xám thấy nhưng rỗng),
gauss_vis_<vật>.ply (Gaussian: xanh thấy, đỏ không thấy), summary.json, report.md, views_<vật>.png
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Based_Model" / "CoMe" / "scene"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat  # noqa: E402
from support_graph import read_ply_gauss, read_ply_xyz  # noqa: E402


def write_ply(path, xyz, rgb):
    a = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    a["x"], a["y"], a["z"] = xyz.T
    a["red"], a["green"], a["blue"] = rgb.T
    head = ["ply", "format binary_little_endian 1.0", f"element vertex {len(a)}", "property float x", "property float y",
            "property float z", "property uchar red", "property uchar green", "property uchar blue", "end_header", ""]
    with open(path, "wb") as f:
        f.write("\n".join(head).encode())
        f.write(a.tobytes())


class Occ:
    """Tập ô chiếm chỗ: khoá int64 đã sắp xếp, tra bằng searchsorted."""

    def __init__(self, pts, rad, cell, origin):
        self.cell, self.o = cell, origin
        ijk = np.floor((pts - origin) / cell).astype(np.int64)
        big = rad > cell
        nb = [ijk]
        for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            nb.append(ijk[big] + np.array(d))
        ijk = np.concatenate(nb)
        self.keys = np.unique(self.key(ijk))

    @staticmethod
    def key(ijk):
        s = (1 << 20)
        return ((ijk[..., 0] + s) << 42) | ((ijk[..., 1] + s) << 21) | (ijk[..., 2] + s)

    def hit(self, P):
        k = self.key(np.floor((P - self.o) / self.cell).astype(np.int64))
        i = np.searchsorted(self.keys, k)
        i = np.minimum(i, len(self.keys) - 1)
        return self.keys[i] == k


class VisCtx:
    """Camera COLMAP + lưới chiếm chỗ của cả cảnh + mặt bàn; visibility(P) = số camera thấy từng điểm (tia không bị chặn)."""

    def __init__(self, s3, colmap, cell_cm=0.5, skip_cm=0.6):
        s3 = Path(s3)
        sup = json.loads((s3 / "support.json").read_text())
        self.n_t, self.d_t = np.array(sup["table"]["normal"]), float(sup["table"]["d"])
        self.cm = 100 * float(sup["meters_per_unit"])
        sp = Path(colmap)
        intr = read_intrinsics_binary(str(sp / "cameras.bin"))
        ext = read_extrinsics_binary(str(sp / "images.bin"))
        self.cams = []
        for e in ext.values():
            c = intr[e.camera_id]
            p = c.params
            fx, fy, cx, cy = (p[0], p[0], p[1], p[2]) if c.model == "SIMPLE_PINHOLE" else (p[0], p[1], p[2], p[3])
            R = qvec2rotmat(e.qvec)
            self.cams.append((R, e.tvec, fx, fy, cx, cy, c.width, c.height, -R.T @ e.tvec))
        g = s3 / "gaussians"
        P1 = read_ply_gauss(g / "background_with_support.ply", 0.5); r1 = read_ply_gauss.last_scale
        P2 = read_ply_gauss(g / "objects_all.ply", 0.5); r2 = read_ply_gauss.last_scale
        allp, allr = np.concatenate([P1, P2]), np.concatenate([r1, r2])
        self.cell = cell_cm / self.cm
        self.skip = skip_cm / self.cm
        self.occ = Occ(allp, allr, self.cell, allp.min(0))
        self.lo, self.hi = np.percentile(allp, 0.5, 0), np.percentile(allp, 99.5, 0)
        print(f"{len(self.cams)} camera; o chiem cho {len(self.occ.keys)} (o {cell_cm} cm, {len(allp)} Gaussian)")

    def height_cm(self, P):
        return (P @ self.n_t + self.d_t) * self.cm

    def visibility(self, P):
        """n_vis [N] và ma trận thấy [N, cams] (bool)."""
        N = len(P)
        vis = np.zeros((N, len(self.cams)), bool)
        step = self.cell / 2
        for ci, (R, t, fx, fy, cx, cy, W, H, C) in enumerate(self.cams):
            X = P @ R.T + t
            z = X[:, 2]
            u, v = fx * X[:, 0] / np.maximum(z, 1e-9) + cx, fy * X[:, 1] / np.maximum(z, 1e-9) + cy
            inimg = (z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            idx = np.where(inimg)[0]
            if len(idx) == 0:
                continue
            D = C[None] - P[idx]
            L = np.linalg.norm(D, axis=1)
            D /= L[:, None]
            free = np.ones(len(idx), bool)
            tt = self.skip
            tmax = L.max()
            while tt < tmax and free.any():
                act = free & (tt < L)
                if not act.any():
                    break
                Q = P[idx[act]] + tt * D[act]
                inside = np.all((Q >= self.lo) & (Q <= self.hi), 1)
                blocked = np.zeros(len(Q), bool)
                if inside.any():
                    blocked[inside] = self.occ.hit(Q[inside]) | (self.height_cm(Q[inside]) < -0.3)
                ai = np.where(act)[0]
                free[ai[blocked]] = False
                L[ai[~inside]] = 0                                   # ra khỏi hộp cảnh: coi như thoát
                tt += step
            vis[idx, ci] = free
        return vis.sum(1), vis


def shell_of(V, vox):
    o = V.min(0)
    ijk = np.round((V - o) / vox).astype(np.int64)
    keys = np.sort(Occ.key(ijk))
    inner = np.ones(len(V), bool)
    for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        k = Occ.key(ijk + np.array(d))
        i = np.minimum(np.searchsorted(keys, k), len(keys) - 1)
        inner &= keys[i] == k
    return ~inner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s3", required=True, help="thư mục S3_label (gaussians/, support.json)")
    ap.add_argument("--hulls", required=True)
    ap.add_argument("--colmap", required=True, help="sparse/0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--objects", default="", help="danh sách vật, mặc định mọi vật có cả hull và Gaussian")
    ap.add_argument("--cell_cm", type=float, default=0.5)
    ap.add_argument("--skip_cm", type=float, default=0.6)
    ap.add_argument("--min_vis", type=int, default=3)
    ap.add_argument("--surf_cm", type=float, default=1.0)
    ap.add_argument("--max_shell", type=int, default=20000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    s3 = Path(a.s3)
    ctx = VisCtx(s3, a.colmap, a.cell_cm, a.skip_cm)
    n_t, d_t, cm, cell, cams, g = ctx.n_t, ctx.d_t, ctx.cm, ctx.cell, ctx.cams, s3 / "gaussians"
    height_cm, visibility = ctx.height_cm, ctx.visibility

    names = sorted(f.stem[5:] for f in Path(a.hulls).glob("hull_*.ply") if f.stem[5:] not in ("objects", "consistency"))
    names = [n for n in names if (g / f"{n}.ply").exists()]
    if a.objects:
        names = [n for n in names if n in a.objects.split(",")]
    summary = {}
    for n in names:
        V = read_ply_xyz(Path(a.hulls) / f"hull_{n}.ply")
        if len(V) == 0:
            continue
        ux = np.unique(np.round(V[:, 0], 5))
        vox = float(np.min(np.diff(ux))) if len(ux) > 1 else cell
        V = V[height_cm(V) >= -0.5]                                   # không có vật dưới mặt bàn
        sh = shell_of(V, vox)
        S = V[sh]
        sub = 1.0
        if len(S) > a.max_shell:
            rng = np.random.default_rng(0)
            S = S[rng.choice(len(S), a.max_shell, replace=False)]
            sub = sh.sum() / a.max_shell
        G = read_ply_gauss(g / f"{n}.ply", 0.3)
        nv, _ = visibility(S)
        ng, _ = visibility(G) if len(G) else (np.zeros(0, int), None)
        hS = height_cm(S)
        hG = height_cm(G)
        hmax = np.percentile(hS, 98)
        bottom = hS <= max(1.0, 0.15 * hmax)
        dS = cKDTree(G).query(S, k=1)[0] * cm if len(G) else np.full(len(S), 1e9)
        seen, weak, miss = nv >= a.min_vis, (nv > 0) & (nv < a.min_vis), nv == 0
        empty = seen & (dS > a.surf_cm)
        cls = np.where(miss & bottom, 3, np.where(miss, 4, np.where(weak, 2, np.where(empty, 5, 1))))
        pal = np.array([[0, 0, 0], [40, 200, 60], [240, 210, 40], [230, 40, 40], [150, 60, 220], [150, 150, 150]], np.uint8)
        write_ply(out / f"mold_{n}.ply", S, pal[cls])
        gcol = np.where((ng >= a.min_vis)[:, None], [[40, 200, 60]], np.where((ng > 0)[:, None], [[240, 210, 40]], [[230, 40, 40]]))
        write_ply(out / f"gauss_vis_{n}.ply", G, gcol.astype(np.uint8))
        np.save(out / f"gauss_xyz_{n}.npy", G.astype(np.float32))          # để kiểm chứng bằng render (khớp theo toạ độ)
        np.save(out / f"gauss_nvis_{n}.npy", ng.astype(np.int16))
        r = {"hull_voxels": int(len(V)), "shell_voxels": int(sh.sum()), "shell_sampled": int(len(S)),
             "height_cm_p98": float(hmax),
             "shell_seen_pct": 100 * float(seen.mean()), "shell_weak_pct": 100 * float(weak.mean()),
             "shell_missing_pct": 100 * float(miss.mean()),
             "missing_bottom_pct": 100 * float((miss & bottom).mean()), "missing_side_back_pct": 100 * float((miss & ~bottom).mean()),
             "seen_but_empty_pct": 100 * float(empty.mean()),
             "bottom_shell_missing_pct": 100 * float(miss[bottom].mean()) if bottom.any() else None,
             "gaussians": int(len(G)), "gauss_unseen_pct": 100 * float((ng == 0).mean()) if len(G) else None,
             "gauss_weak_pct": 100 * float(((ng > 0) & (ng < a.min_vis)).mean()) if len(G) else None,
             "gauss_unseen_median_height_cm": float(np.median(hG[ng == 0])) if len(G) and (ng == 0).any() else None,
             "median_views_seen_shell": float(np.median(nv[seen])) if seen.any() else 0}
        summary[n] = r
        print(n, json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}))

        # hình: nhìn trên xuống và nhìn ngang theo 2 trục mặt bàn
        a0 = np.array([1.0, 0, 0]) if abs(n_t[0]) < 0.9 else np.array([0, 1.0, 0])
        u_ = np.cross(n_t, a0); u_ /= np.linalg.norm(u_); v_ = np.cross(n_t, u_)
        fig, axs = plt.subplots(1, 3, figsize=(16, 5))
        for ax, (x, y, ttl) in zip(axs, ((S @ u_ * cm, S @ v_ * cm, "tren xuong"), (S @ u_ * cm, hS, "ngang (u)"),
                                         (S @ v_ * cm, hS, "ngang (v)"))):
            for c, lb in ((1, "thay"), (2, "thay it"), (5, "thay nhung rong"), (4, "khong thay: ben/sau"), (3, "khong thay: day")):
                m = cls == c
                ax.scatter(x[m], y[m], s=3, color=pal[c] / 255, label=f"{lb} ({100 * m.mean():.0f}%)")
            ax.set_title(ttl); ax.set_aspect("equal")
        axs[0].legend(fontsize=8, markerscale=4)
        fig.suptitle(f"{n}: khuon (vo hull) theo kha nang camera thay")
        fig.tight_layout(); fig.savefig(out / f"views_{n}.png", dpi=110); plt.close(fig)

    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    L = ["# Khuon inpaint: vo hull theo kha nang camera thay", "",
         f"{len(cams)} camera, o chiem cho {a.cell_cm} cm, bo qua {a.skip_cm} cm dau tia, thay = >= {a.min_vis} camera.", "",
         "| vat | vo (voxel) | thay % | thay it % | KHONG thay % | - day % | - ben/sau % | thay nhung rong % | day khong thay (trong vung day) % | Gaussian | Gaussian khong camera thay % | cao trung vi Gaussian khong thay (cm) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda x: "-" if x is None else f"{x:.1f}"
    for n, r in summary.items():
        L.append(f"| {n} | {r['shell_voxels']} | {f(r['shell_seen_pct'])} | {f(r['shell_weak_pct'])} | {f(r['shell_missing_pct'])} | "
                 f"{f(r['missing_bottom_pct'])} | {f(r['missing_side_back_pct'])} | {f(r['seen_but_empty_pct'])} | "
                 f"{f(r['bottom_shell_missing_pct'])} | {r['gaussians']} | {f(r['gauss_unseen_pct'])} | {f(r['gauss_unseen_median_height_cm'])} |")
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
