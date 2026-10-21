"""Debug S6: thử opacity_field trên lưới quanh 1 vật với các kích thước khối điểm khác nhau."""
import sys

import numpy as np
import torch

from common import log, make_parser, load_scene, read_ply, gaussians_from_ply, opacity_field
from arguments import get_combined_args


def main():
    p, model, pipe, ss, _, _ = make_parser("debug field")
    p.add_argument("--ply", type=str, required=True)
    p.add_argument("--chunk", type=int, default=100000)
    args = get_combined_args(p)
    S = load_scene(args, model, pipe, ss)
    el = read_ply(args.ply)
    xyz = np.stack([el["x"], el["y"], el["z"]], 1)
    g_o = gaussians_from_ply(args.ply, S["dataset"].sh_degree, S["train"])
    lo, hi = np.percentile(xyz, 1, 0), np.percentile(xyz, 99, 0)
    step = float((hi - lo).max()) / 128
    dims = np.ceil((hi - lo) / step).astype(int) + 1
    axes = [lo[i] + step * np.arange(dims[i]) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)
    log(f"grid {dims} = {len(grid)} điểm, chunk {args.chunk}")
    for s in range(0, len(grid), args.chunk):
        pts = torch.from_numpy(grid[s:s + args.chunk]).cuda().contiguous()
        for k, v in enumerate(S["train"]):
            try:
                f = opacity_field(pts, [v], g_o, S)
                torch.cuda.synchronize()
            except Exception as e:
                ph = torch.cat([pts, torch.ones(len(pts), 1, device="cuda")], 1)
                pp = ph @ v.full_proj_transform
                ndc = pp[:, :2] / pp[:, 3:4]
                z = (ph @ v.world_view_transform)[:, 2]
                inside = (z > 0.2) & (ndc.abs() <= 1).all(1)
                log(f"chunk bắt đầu {s}, view {k} ({v.image_name}) LỖI: {e} | điểm trong khung {int(inside.sum())}, "
                    f"z min/max {float(z.min()):.3f}/{float(z.max()):.3f}, Gaussian {g_o.get_xyz.shape[0]}")
                sys.exit(1)
    log(f"chunk {args.chunk}: mọi view OK")


if __name__ == "__main__":
    main()
