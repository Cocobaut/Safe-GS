"""Mesh / table-plane helpers shared by step 14-15 (numpy + open3d, CPU)."""
import json
from pathlib import Path

import numpy as np
import open3d as o3d

PLANE_JSON = Path(__file__).with_name("table_plane.json")   # = CoMe_Infusion/Step_6_Mesh_GOF/mesh_gof_table_crop.json


def load_plane(path=PLANE_JSON):
    """Table plane n.X + d = 0 (n towards the cameras), in-plane basis e1, e2 (n, e1, e2 orthonormal)."""
    c = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"n": np.array(c["plane_normal"], float), "d": float(c["plane_d"]),
            "e1": np.array(c["basis_e1"], float), "e2": np.array(c["basis_e2"], float)}


def to_uvh(plane, X):
    return np.stack([X @ plane["e1"], X @ plane["e2"], X @ plane["n"] + plane["d"]], -1)


def from_uvh(plane, uvh):
    return (uvh[..., :1] * plane["e1"] + uvh[..., 1:2] * plane["e2"]
            + (uvh[..., 2:3] - plane["d"]) * plane["n"])


def read_mesh(path):
    """(vertices float64 [N,3], triangles int64 [M,3], colors [N,3] in 0..1 or None). Binary triangle-only PLYs are read
    with numpy and validated (every face has 3 indices in range): a corrupted download fails here instead of being read
    silently (open3d does)."""
    with open(path, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            line = f.readline()
            if not line:
                break
            header += line
        lines = header.decode(errors="replace").splitlines()
        if "format binary_little_endian 1.0" in lines and "property list uchar int vertex_indices" in lines:
            nv = nf = 0
            vprops, cur = [], None
            for ln in lines:
                t = ln.split()
                if t[:2] == ["element", "vertex"]:
                    nv, cur = int(t[2]), "vertex"
                elif t[:2] == ["element", "face"]:
                    nf, cur = int(t[2]), "face"
                elif t and t[0] == "property" and cur == "vertex":
                    vprops.append((t[2], {"float": "<f4", "double": "<f8", "uchar": "u1"}[t[1]]))
            vert = np.fromfile(f, dtype=np.dtype(vprops), count=nv)
            face = np.fromfile(f, dtype=np.dtype([("n", "u1"), ("i", "<i4", (3,))]), count=nf)
            if len(face) != nf or not (face["n"] == 3).all():
                raise ValueError(f"{path}: corrupted or non-triangle face block")
            F = face["i"].astype(np.int64)
            if F.min() < 0 or F.max() >= nv:
                raise ValueError(f"{path}: face index out of range")
            P = np.stack([vert["x"], vert["y"], vert["z"]], 1).astype(np.float64)
            cols = (np.stack([vert["red"], vert["green"], vert["blue"]], 1) / 255.0
                    if {"red", "green", "blue"} <= set(vert.dtype.names) else None)
            return P, F, cols
    m = o3d.io.read_triangle_mesh(str(path))
    cols = np.asarray(m.vertex_colors) if m.has_vertex_colors() else None
    return np.asarray(m.vertices), np.asarray(m.triangles).astype(np.int64), cols


def to_o3d(V, F, colors=None):
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F.astype(np.int32)))
    if colors is not None:
        m.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
    return m


def write_mesh(path, V, F, colors=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    m = to_o3d(V, F, colors)
    m.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(path), m, write_ascii=False, compressed=False)


def edge_stats(F):
    """Counts of edges used once (boundary), twice (manifold) and more (non-manifold)."""
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    return {"boundary": int((cnt == 1).sum()), "manifold": int((cnt == 2).sum()), "non_manifold": int((cnt > 2).sum())}


def remove_small_components(V, F, min_frac=0.01, min_faces=0):
    m = to_o3d(V, F)
    labels, counts, _ = m.cluster_connected_triangles()
    labels, counts = np.asarray(labels), np.asarray(counts)
    keep_lab = np.nonzero((counts >= min_frac * counts.sum()) & (counts >= min_faces))[0]
    keep = np.isin(labels, keep_lab)
    return compact(V, F[keep]) + ({"components": int(len(counts)), "kept": int(len(keep_lab)),
                                   "faces_removed": int((~keep).sum())},)


def compact(V, F, colors=None):
    used = np.unique(F)
    remap = -np.ones(len(V), np.int64)
    remap[used] = np.arange(len(used))
    out = (V[used], remap[F])
    return out + ((colors[used],) if colors is not None else ())
