"""
So kết quả hoàn thiện TRELLIS (modal_trellis.py) với khung quan sát: hình chiếu + lát cắt, và mesh TRELLIS đổi về hệ bàn (mét).

    python tools/viz_trellis_occ.py --inp Data/teatime/trellis_in --runs Data/teatime/trellis_out --tags t06_rp,seen_t08_carved,frame_t08_carved
Xuất: <runs>/compare_<vật>.png (mỗi hàng một run: 3 hình chiếu + lát giữa; xám = cả hai, đỏ = khung có mà kết quả bỏ,
xanh = kết quả THÊM ngoài khung, xanh dương = bề mặt camera thấy), <runs>/<tag>/<vật>/mesh_table.obj, summary.md
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frame_mesh import edge_stats  # noqa: E402


def colorize(frame, occ, seen, axis):
    a = frame.any(axis)
    b = occ.any(axis)
    s = seen.any(axis)
    img = np.ones(a.shape + (3,))
    img[a & b] = (0.6, 0.6, 0.6)
    img[a & ~b] = (0.9, 0.2, 0.2)
    img[~a & b] = (0.2, 0.75, 0.3)
    img[s & b] = np.minimum(img[s & b], (0.45, 0.55, 0.95))
    return np.transpose(img, (1, 0, 2))[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--tags", required=True)
    a = ap.parse_args()
    tags = a.tags.split(",")
    runs = Path(a.runs)
    objs = sorted({p.parent.name for t in tags for p in (runs / t).glob("*/occ.npz")})
    L = ["| vat | run | o khung | o ket qua | them ngoai khung | bo trong khung | IoU voi khung | mesh kin (bien/non-manifold) | genus |", "|---|---|---|---|---|---|---|---|---|"]
    for n in objs:
        g = np.load(Path(a.inp) / n / "grid.npz")
        frame, seen = g["frame"], g["seen"]
        center, size = g["center"], float(g["size"])
        rows = [t for t in tags if (runs / t / n / "occ.npz").exists()]
        fig, axs = plt.subplots(len(rows), 4, figsize=(16, 4 * len(rows)), squeeze=False)
        for r, t in enumerate(rows):
            occ = np.load(runs / t / n / "occ.npz")["occ_final"]
            for c, (ax_, ttl) in enumerate(((2, "tren xuong"), (1, "ngang (x)"), (0, "ngang (y)"))):
                im = colorize(frame, occ, seen, ax_) if ax_ != 2 else colorize(frame, occ, seen, 2)
                axs[r, c].imshow(im, interpolation="nearest"); axs[r, c].set_title(f"{t}: {ttl}", fontsize=9); axs[r, c].axis("off")
            k = frame.shape[1] // 2
            sl = np.ones(frame[:, k, :].shape + (3,))
            sl[frame[:, k, :] & occ[:, k, :]] = (0.6, 0.6, 0.6)
            sl[frame[:, k, :] & ~occ[:, k, :]] = (0.9, 0.2, 0.2)
            sl[~frame[:, k, :] & occ[:, k, :]] = (0.2, 0.75, 0.3)
            axs[r, 3].imshow(np.transpose(sl, (1, 0, 2))[::-1], interpolation="nearest"); axs[r, 3].set_title("lat cat giua (y)", fontsize=9)
            axs[r, 3].axis("off")
            mz = np.load(runs / t / n / "mesh.npz")
            V = mz["v"] * size + center
            m = trimesh.Trimesh(V, mz["f"], process=True)
            E, nb, nnm = edge_stats(m.faces)
            chi = len(m.vertices) - E + len(m.faces)
            m.export(runs / t / n / "mesh_table.obj")
            add = int((occ & ~frame).sum()); rem = int((frame & ~occ).sum())
            iou = float((occ & frame).sum() / max((occ | frame).sum(), 1))
            L.append(f"| {n} | {t} | {int(frame.sum())} | {int(occ.sum())} | {add} | {rem} | {iou:.3f} | "
                     f"{'co' if nb == 0 and nnm == 0 else 'KHONG'} ({nb}/{nnm}) | {(1 - chi / 2) if nb == 0 and nnm == 0 else '-'} |")
        fig.suptitle(f"{n}: xam = ca hai, do = khung co / ket qua bo, xanh la = ket qua them, xanh duong = be mat camera thay")
        fig.tight_layout(); fig.savefig(runs / f"compare_{n}.png", dpi=90); plt.close(fig)
    (runs / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
