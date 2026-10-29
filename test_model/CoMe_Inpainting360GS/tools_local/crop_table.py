"""Cut the table (+ the objects on it) out of a full-scene CoMe mesh, with the same prism as
CoMe_Infusion/Step_6_Mesh_GOF (mesh_gof_table_crop.json: table plane, 2D table outline on the plane, height range),
so the v1 / v2 / GOF table meshes cover exactly the same region. No decimation, no cleaning: every triangle whose
3 vertices lie in the prism is kept.

  python tools_local/crop_table.py --mesh <full mesh .ply> --out <table mesh .ply>
"""
import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay

PRISM = Path(r"F:/Safe-GS/test_model/CoMe_Infusion/Step_6_Mesh_GOF/mesh_gof_table_crop.json")


def read_mesh(path):
    """Vertices, triangles (+ colors in [0,1] or None). Binary triangle-only PLYs (CoMe / trimesh export) are read
    with numpy: open3d segfaults on some of these large files; anything else goes through open3d."""
    with open(path, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            header += f.readline()
        lines = header.decode().splitlines()
        if "format binary_little_endian 1.0" in lines and "property list uchar int vertex_indices" in lines:
            nv = nf = 0
            vprops, cur = [], None
            for ln in lines:
                t = ln.split()
                if t[:2] == ["element", "vertex"]:
                    nv, cur = int(t[2]), "vertex"
                elif t[:2] == ["element", "face"]:
                    nf, cur = int(t[2]), "face"
                elif t[0] == "property" and cur == "vertex":
                    vprops.append((t[2], {"float": "<f4", "double": "<f8", "uchar": "u1"}[t[1]]))
            vert = np.fromfile(f, dtype=np.dtype(vprops), count=nv)
            face = np.fromfile(f, dtype=np.dtype([("n", "u1"), ("i", "<i4", (3,))]), count=nf)
            if not (face["n"] == 3).all():
                raise ValueError("non-triangle faces")
            P = np.stack([vert["x"], vert["y"], vert["z"]], 1).astype(np.float64)
            F = face["i"].astype(np.int64)
            if F.min() < 0 or F.max() >= nv:
                raise ValueError(f"face index out of range ({F.min()}..{F.max()}, {nv} vertices)")
            names = vert.dtype.names
            colors = (np.stack([vert["red"], vert["green"], vert["blue"]], 1) / 255.0
                      if {"red", "green", "blue"} <= set(names) else None)
            return P, F, colors
    mesh = o3d.io.read_triangle_mesh(path)
    colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
    return np.asarray(mesh.vertices), np.asarray(mesh.triangles), colors


def main():
    ap = ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists (never overwritten)")

    c = json.loads(PRISM.read_text(encoding="utf-8"))
    n, d = np.array(c["plane_normal"]), float(c["plane_d"])
    e1, e2 = np.array(c["basis_e1"]), np.array(c["basis_e2"])
    lo, hi = c["height_range"]
    tri = Delaunay(np.array(c["hull_2d"]))

    P, F, colors = read_mesh(args.mesh)
    print(f"[i] full mesh: {len(P)} vertices, {len(F)} faces")
    h = P @ n + d
    keep_v = (h >= lo) & (h <= hi)
    cand = np.nonzero(keep_v)[0]
    keep_v[cand] = tri.find_simplex(np.stack([P[cand] @ e1, P[cand] @ e2], 1)) >= 0
    keep_f = keep_v[F].all(1)
    used = np.unique(F[keep_f])
    remap = -np.ones(len(P), np.int64)
    remap[used] = np.arange(len(used))
    table = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(P[used]),
                                      o3d.utility.Vector3iVector(remap[F[keep_f]]))
    if colors is not None:
        table.vertex_colors = o3d.utility.Vector3dVector(colors[used])
    table.compute_vertex_normals()
    out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out), table, write_ascii=False, compressed=False)
    meta = {"source": str(Path(args.mesh).resolve()), "prism": str(PRISM), "plane_normal": n.tolist(), "plane_d": d,
            "height_range": [lo, hi], "vertices": int(len(used)), "faces": int(keep_f.sum())}
    out.with_name(out.stem + "_crop.json").write_text(json.dumps(meta, indent=2))
    print(f"[ok] {out}: {len(used)} vertices, {int(keep_f.sum())} faces, {out.stat().st_size / 1024**2:.0f} MB")


if __name__ == "__main__":
    main()
