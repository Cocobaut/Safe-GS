"""Make an object-only 3DGS meshable by the original CoMe extractor (step 15).

extract_mesh_tets.py renders every training camera; with the Gaussians of one small object only, many cameras see no
Gaussian at all and the CoMe kernels are launched with an empty configuration (CUDA "invalid configuration argument").
The empty launch is in the rasterizer of `integrate` when a view renders 0 Gaussians (the error surfaces at the next
torch op, `torch.min` in evaluate_alpha). The CoMe code is kept unchanged; instead one small, almost transparent
Gaussian is placed `--dist` in front of every camera:
  - opacity 0.01 > 1/255 (not culled) and > opacity_cutoff_tetra 0.0039 (a few tetrahedra points in its own view)
  - scale 0.01 >> the CoMe 3D filter at that distance (~6e-4): the filter scales opacity by (s / sqrt(s^2+f^2))^3, a
    1e-4 Gaussian would drop to ~0 opacity and be culled again
  - its own opacity field stays far below the 0.5 level -> no surface around it
  - it lowers the transmittance of what is behind it by 1 % -> negligible for the 0.5 level set
  - an anchor point hidden behind the object from another camera counts as "inside" and can leave a small isolated
    piece far from the object: removed in solidify_and_merge.py (components outside the footprint column)

  python tools_mesh/add_view_anchors.py --ply <closed.ply> --cameras <cameras.json> --out <anchored.ply>
"""
import json
from argparse import ArgumentParser

import numpy as np
from plyfile import PlyData, PlyElement


def main():
    ap = ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--cameras", required=True, help="cameras.json of the CoMe model (camera->world rotation, position)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dist", type=float, default=1.0)
    ap.add_argument("--opacity", type=float, default=0.01)
    ap.add_argument("--scale", type=float, default=0.01)
    args = ap.parse_args()

    v = PlyData.read(args.ply).elements[0]
    names = [p.name for p in v.properties]
    rows = np.stack([np.asarray(v[nm], np.float64) for nm in names], 1)
    cams = json.loads(open(args.cameras).read())
    anchors = np.zeros((len(cams), len(names)))
    col = {nm: i for i, nm in enumerate(names)}
    for i, c in enumerate(cams):
        R, p = np.array(c["rotation"]), np.array(c["position"])
        anchors[i, [col["x"], col["y"], col["z"]]] = p + args.dist * R[:, 2]      # +z = viewing direction
    anchors[:, col["opacity"]] = np.log(args.opacity / (1 - args.opacity))
    anchors[:, [col["scale_0"], col["scale_1"], col["scale_2"]]] = np.log(args.scale)
    anchors[:, col["rot_0"]] = 1.0
    out = np.concatenate([rows, anchors])
    el = np.empty(len(out), dtype=[(nm, "f4") for nm in names])
    for i, nm in enumerate(names):
        el[nm] = out[:, i]
    PlyData([PlyElement.describe(el, "vertex")]).write(args.out)
    print(f"[anchors] {len(cams)} view anchors (opacity {args.opacity}) + {len(rows)} Gaussians -> {args.out}")


if __name__ == "__main__":
    main()
