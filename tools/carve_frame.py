"""
Khung chặt cho từng vật = visual hull khoét bỏ không gian trống mà camera nhìn xuyên qua (space carving theo tia).

Cùng một quy tắc cho mọi vật:
  - voxel hull (bỏ phần dưới mặt bàn), lấy thưa theo lưới để <= --max_vox voxel / vật;
  - n_vis = số camera thấy voxel (tools/visibility_mold.VisCtx: tia từ voxel tới camera không bị Gaussian nào chặn);
  - voxel TRỐNG (khoét) nếu n_vis >= --min_vis và không có Gaussian của vật trong --surf_cm (camera nhìn thấy chỗ đó mà không có gì);
  - khung = hull - trống. Trong khung: "bề mặt thấy" (n_vis >= min_vis, có Gaussian gần) và "không thấy" (n_vis < min_vis:
    ruột vật hoặc chỗ chưa ai quan sát).
  - vỏ khung (voxel khung có ô kề không thuộc khung): "thấy" -> giữ theo ảnh; "không thấy" -> KHUÔN cần inpaint, chia đáy / bên-sau.
Xuất: frame_<vật>.ply (xanh bề mặt thấy, cam ruột không thấy, đỏ khuôn đáy, tím khuôn bên/sau), frame_<vật>.npz (toạ độ + lớp),
carve_<vật>.png (hull vs khung nhìn ngang), summary.json, report.md
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from visibility_mold import VisCtx, shell_of, write_ply  # noqa: E402
from support_graph import read_ply_gauss, read_ply_xyz  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s3", required=True)
    ap.add_argument("--hulls", required=True)
    ap.add_argument("--colmap", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--objects", default="")
    ap.add_argument("--min_vis", type=int, default=3)
    ap.add_argument("--surf_cm", type=float, default=1.0)
    ap.add_argument("--max_vox", type=int, default=30000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    s3 = Path(a.s3)
    ctx = VisCtx(s3, a.colmap)
    cm = ctx.cm
    names = sorted(f.stem[5:] for f in Path(a.hulls).glob("hull_*.ply") if f.stem[5:] not in ("objects", "consistency"))
    names = [n for n in names if (s3 / "gaussians" / f"{n}.ply").exists()]
    if a.objects:
        names = [n for n in names if n in a.objects.split(",")]
    a0 = np.array([1.0, 0, 0]) if abs(ctx.n_t[0]) < 0.9 else np.array([0, 1.0, 0])
    u_ = np.cross(ctx.n_t, a0); u_ /= np.linalg.norm(u_); v_ = np.cross(ctx.n_t, u_)
    summary = {}
    for n in names:
        V = read_ply_xyz(Path(a.hulls) / f"hull_{n}.ply")
        if len(V) == 0:
            continue
        ux = np.unique(np.round(V[:, 0], 5))
        vox = float(np.min(np.diff(ux)))
        V = V[ctx.height_cm(V) >= -0.5]
        stride = max(1, int(np.ceil((len(V) / a.max_vox) ** (1 / 3))))
        ijk = np.round((V - V.min(0)) / vox).astype(np.int64)
        Vs = V[np.all(ijk % stride == 0, 1)]
        vs = vox * stride
        G = read_ply_gauss(s3 / "gaussians" / f"{n}.ply", 0.3)
        nv, _ = ctx.visibility(Vs)
        dG = cKDTree(G).query(Vs, k=1)[0] * cm
        free = (nv >= a.min_vis) & (dG > a.surf_cm)
        F = Vs[~free]
        nvF, dF = nv[~free], dG[~free]
        seen_surf = nvF >= a.min_vis
        sh = shell_of(F, vs) if len(F) else np.zeros(0, bool)
        h = ctx.height_cm(F)
        hp98 = float(np.percentile(h, 98)) if len(h) else 0.0
        bottom = h <= max(1.0, 0.15 * hp98)
        mold = sh & ~seen_surf
        cls = np.where(mold & bottom, 3, np.where(mold, 4, np.where(seen_surf, 1, 2)))
        pal = np.array([[0, 0, 0], [40, 200, 60], [240, 150, 40], [230, 40, 40], [150, 60, 220]], np.uint8)
        write_ply(out / f"frame_{n}.ply", F, pal[cls])
        np.savez_compressed(out / f"frame_{n}.npz", xyz=F.astype(np.float32), cls=cls.astype(np.uint8), voxel=vs, n_vis=nvF.astype(np.int16))
        hG = ctx.height_cm(G)
        hV = ctx.height_cm(Vs)
        r = {"hull_voxels": int(len(V)), "stride": stride, "voxel_cm": vs * cm,
             "hull_volume_cm3": float(len(Vs) * (vs * cm) ** 3), "frame_volume_cm3": float(len(F) * (vs * cm) ** 3),
             "carved_pct": 100 * float(free.mean()),
             "height_p98_cm": {"hull": float(np.percentile(hV, 98)), "frame": hp98, "gaussians": float(np.percentile(hG, 98))},
             "frame_shell_voxels": int(sh.sum()),
             "shell_seen_pct": 100 * float((sh & seen_surf).sum() / max(sh.sum(), 1)),
             "shell_mold_bottom_pct": 100 * float((mold & bottom).sum() / max(sh.sum(), 1)),
             "shell_mold_side_back_pct": 100 * float((mold & ~bottom).sum() / max(sh.sum(), 1)),
             "interior_unseen_voxels": int((~sh & ~seen_surf).sum())}
        summary[n] = r
        print(n, json.dumps(r))
        fig, axs = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
        for ax, (x_h, x_f, ttl) in zip(axs, ((Vs @ u_ * cm, F @ u_ * cm, "ngang (u)"), (Vs @ v_ * cm, F @ v_ * cm, "ngang (v)"))):
            ax.scatter(x_h, hV, s=2, color="#cccccc", label="hull (truoc khi khoet)")
            for c, lb in ((1, "be mat thay"), (2, "ruot / chua thay"), (4, "khuon ben/sau"), (3, "khuon day")):
                m = cls == c
                ax.scatter(x_f[m], h[m], s=3, color=pal[c] / 255, label=f"{lb} ({100 * m.mean():.0f}%)")
            ax.axhline(np.percentile(hG, 98), color="k", ls="--", lw=0.8)
            ax.set_title(ttl); ax.set_aspect("equal")
        axs[0].legend(fontsize=7, markerscale=4)
        fig.suptitle(f"{n}: hull {r['hull_volume_cm3']:.0f} cm3 -> khung {r['frame_volume_cm3']:.0f} cm3 (khoet {r['carved_pct']:.0f}%), "
                     f"net dut = cao p98 cua Gaussian")
        fig.tight_layout(); fig.savefig(out / f"carve_{n}.png", dpi=110); plt.close(fig)

    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    L = ["# Khung chat (hull khoet theo tia camera)", "",
         f"Trong = thay boi >= {a.min_vis} camera va khong co Gaussian cua vat trong {a.surf_cm} cm.", "",
         "| vat | hull cm3 | khung cm3 | khoet % | cao p98 hull / khung / Gaussian (cm) | vo khung thay % | khuon day % | khuon ben/sau % |",
         "|---|---|---|---|---|---|---|---|"]
    for n, r in summary.items():
        hp = r["height_p98_cm"]
        L.append(f"| {n} | {r['hull_volume_cm3']:.0f} | {r['frame_volume_cm3']:.0f} | {r['carved_pct']:.1f} | "
                 f"{hp['hull']:.1f} / {hp['frame']:.1f} / {hp['gaussians']:.1f} | {r['shell_seen_pct']:.1f} | "
                 f"{r['shell_mold_bottom_pct']:.1f} | {r['shell_mold_side_back_pct']:.1f} |")
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
