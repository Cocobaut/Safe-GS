"""Shared helpers of step 16 (numpy only at import time; open3d / scipy / cv2 imported where used).

- run / input resolution (no hardcoded paths, file names or Gaussian counts)
- SimFrame: COLMAP world <-> MuJoCo Z-up frame (table top under the object at z = 0, robot side = -Y)
- OBJ writing and 3DGS colour baking into a texture atlas
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
for p in (SRC, SRC / "tools_mesh", SRC / "tools_instantmesh"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common.paths import OUTPUTS_DIR, get_run_dir, step_dir  # noqa: E402

OUT_NAME = "step16_physics_engine"
SH_C0 = 0.28209479177387814

# configs/datasets/*.yaml -> `physics:` block (defaults < scene < object) overrides these
PHYS_DEFAULTS = {
    "mode": "articulated",        # articulated (3 hinged rigid segments, cloth-like) | flex (<flexcomp>) | rigid
    "sim_scale": 1.0,             # sim units per scene unit (scene units are NOT metres; teatime: 1 unit ~ 0.38 m)
    "object_mass": 0.065,
    "n_segments": 8,              # articulated mode (hinged rigid strips + one continuous <skin> for the visual)
    "flex_grid": [6, 6, 2],       # flex mode
    "flex_young": 600.0,
    "table_tris": 150000,
    "object_tris": 8000,
    "object_color": "mesh",       # mesh = InstantMesh vertex colours (step 14/15) | gs = bake step-8 Gaussians
    "floor_z": None,              # sim z of the floor (robot stands on it); None = pedestal of fixed height
    "gap": 0.001,                 # step-15 clearance (to find the table top under the object)
    "fps": 30,
    "phases": {"SETTLE": 0.8, "APPROACH": 0.8, "DESCEND": 0.9, "GRASP": 0.6, "LIFT": 1.6, "CARRY": 1.4, "HOLD": 0.8},
}


# ---------------------------------------------------------------------------------------------------- run / config
def resolve_run_dir(run: str, dataset: str = "LERF") -> Path:
    p = Path(run)
    if p.is_dir():
        return Path(os.path.abspath(p))
    return get_run_dir(dataset, run)


def load_physics_cfg(run_dir: Path) -> dict:
    """PHYS_DEFAULTS < yaml physics block of the run (outputs/<D>/ours/<run>) + the run's target_id."""
    cfg = json.loads(json.dumps(PHYS_DEFAULTS))
    cfg["target_id"] = None
    try:
        from common.datasets import resolve_run
        rd = Path(run_dir).absolute()
        y = resolve_run(rd.parents[1].name, rd.name)
        for k, v in (y.get("physics") or {}).items():
            cfg[k] = {**cfg[k], **v} if isinstance(cfg.get(k), dict) and isinstance(v, dict) else v
        cfg["target_id"] = y.get("target_id")
        if y.get("gap") is not None:
            cfg["gap"] = float(y["gap"])
    except (Exception, SystemExit) as e:  # noqa: BLE001  (undeclared run / no pyyaml: defaults)
        print(f"[physics] config: defaults only ({type(e).__name__}: {e})"[:200])
    return cfg


def out_dir(run_dir: Path) -> Path:
    return Path(run_dir) / OUT_NAME


def find_inputs(run_dir: Path, object_id=None) -> dict:
    """Every input file of step 16, found from the run folder."""
    run = Path(run_dir)
    # a run can hold several step-8 folders (teatime_napkin: old step8_removal_virtual_cam_v2 + newer
    # step8_removal_virtual_cam): take the one whose removed object is the most recent
    s8_all = [d / "2_object_removal" for d in run.glob("step8_*") if list((d / "2_object_removal").glob("removed_object_id*.ply"))]
    s8 = (max(s8_all, key=lambda d: max(q.stat().st_mtime for q in d.glob("removed_object_id*.ply")))
          if s8_all else step_dir(run, 8) / "2_object_removal")
    if object_id is None:
        cands = sorted(s8.glob("removed_object_id*.ply"))
        if len(cands) != 1:
            raise SystemExit(f"{s8}: {len(cands)} removed_object_id*.ply -> give --object-id")
        object_id = int(re.findall(r"id(\d+)", cands[0].stem)[-1])
    obj_gs = s8 / f"removed_object_id{object_id}.ply"
    s11 = step_dir(run, 11)
    plys = sorted(s11.glob("point_cloud_inpainted_iter*.ply"), key=lambda q: int(re.findall(r"\d+", q.stem)[-1]))
    s15 = step_dir(run, 15)
    table = None
    summ = s15 / "summary.json"
    if summ.is_file():                                         # the table mesh step 15 really used
        rel = json.loads(summ.read_text(encoding="utf-8")).get("table", {}).get("file")
        if rel and (run / rel).is_file():
            table = run / rel
    if table is None:                                          # same rule as place_on_table.py
        s13 = step_dir(run, 13)
        table = s13 / "mesh_come_v2_table.ply"
        if not table.is_file():
            c = sorted(s13.glob("*_table.ply"), key=lambda q: q.stat().st_mtime)
            table = c[-1] if c else None
    cams = step_dir(run, 1) / "cameras.json"
    res = {"run": run, "object_id": int(object_id), "object_gs": obj_gs,
           "background_gs": plys[-1] if plys else None, "table_mesh": table,
           "object_mesh": s15 / "mesh_object_placed.ply", "cameras": cams if cams.is_file() else None}
    miss = [k for k in ("object_gs", "background_gs", "table_mesh", "object_mesh") if not res[k] or not Path(res[k]).is_file()]
    if miss:
        raise SystemExit("missing inputs of step 16: " + ", ".join(f"{k}={res[k]}" for k in miss))
    return res


# ---------------------------------------------------------------------------------------------------- sim frame
class SimFrame:
    """x_sim = s * R (x_world - o). Rows of R = sim X, Y, Z axes in world coordinates (Z = table normal, up)."""

    def __init__(self, R, o, s=1.0):
        self.R, self.o, self.s = np.asarray(R, float), np.asarray(o, float), float(s)

    def to_sim(self, X):
        return self.s * (np.asarray(X, float) - self.o) @ self.R.T

    def to_world(self, S):
        return np.asarray(S, float) @ self.R / self.s + self.o

    def dir_to_sim(self, D):
        return np.asarray(D, float) @ self.R.T

    def dir_to_world(self, D):
        return np.asarray(D, float) @ self.R

    def to_json(self):
        return {"R_sim_from_world": self.R.tolist(), "origin_world": self.o.tolist(), "sim_scale": self.s,
                "formula": "x_sim = sim_scale * R_sim_from_world @ (x_world - origin_world)"}

    @classmethod
    def from_json(cls, d):
        return cls(d["R_sim_from_world"], d["origin_world"], d["sim_scale"])


def nearest_edge_dir(hull_uv, p_uv):
    """Unit 2D vector from p to the closest point of the polygon boundary, and that distance."""
    H = np.asarray(hull_uv, float)
    A, B = H, np.roll(H, -1, 0)
    AB = B - A
    t = np.clip(((p_uv - A) * AB).sum(1) / np.maximum((AB * AB).sum(1), 1e-12), 0, 1)
    Q = A + t[:, None] * AB
    d = np.linalg.norm(Q - p_uv, axis=1)
    i = int(np.argmin(d))
    return (Q[i] - p_uv) / max(d[i], 1e-9), float(d[i])


# ---------------------------------------------------------------------------------------------------- gaussians
def read_gs_points(path, min_opacity=0.0):
    """xyz (world), rgb (SH DC), opacity of a 3DGS ply (any number of properties, read by name)."""
    from plyfile import PlyData
    v = PlyData.read(str(path)).elements[0]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    rgb = np.clip(0.5 + SH_C0 * np.stack([v[f"f_dc_{i}"] for i in range(3)], 1), 0, 1)
    op = 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float64)))
    keep = op > min_opacity
    return xyz[keep], rgb[keep], op[keep]


def knn_colors(query_world, gs_xyz, gs_rgb, k=6, eps=None):
    from scipy.spatial import cKDTree
    tree = cKDTree(gs_xyz)
    d, i = tree.query(query_world, k=k, workers=-1)
    eps = eps if eps is not None else max(float(np.median(d[:, 0])), 1e-4)
    w = 1.0 / (d + eps)
    w /= w.sum(1, keepdims=True)
    return (gs_rgb[i] * w[..., None]).sum(1)


# ---------------------------------------------------------------------------------------------------- meshes
def bake_atlas(F, vcol, cell=4):
    """Texture atlas with one cell x cell block per triangle, filled with the barycentric blend of its 3 vertex
    colours (smooth Gouraud-like colour, no UV bleeding). -> (uv per corner [3M,2], image RGB uint8)."""
    M = len(F)
    g = int(np.ceil(np.sqrt(M)))
    size = g * cell
    img = np.zeros((size, size, 3), np.float32)
    ti = np.arange(M)
    gx, gy = ti % g, ti // g
    # texel centres inside a cell, barycentric w.r.t. the corner texels (0,0) (cell-1,0) (0,cell-1)
    a = np.arange(cell) + 0.5
    tx, ty = np.meshgrid(a, a)                                  # [cell, cell]
    u = np.clip((tx - 0.5) / (cell - 1), 0, 1)
    v = np.clip((ty - 0.5) / (cell - 1), 0, 1)
    s = u + v
    s = np.maximum(s, 1.0)
    u, v = u / s, v / s
    w0 = 1 - u - v
    C = vcol[F]                                                 # [M,3,3]
    blk = w0[None, ..., None] * C[:, None, None, 0] + u[None, ..., None] * C[:, None, None, 1] \
        + v[None, ..., None] * C[:, None, None, 2]              # [M,cell,cell,3]
    img_r = img.reshape(g, cell, g, cell, 3)
    img_r[gy, :, gx, :] = blk
    img = img_r.reshape(size, size, 3)
    px0, py0 = gx * cell, gy * cell
    corners = np.array([[0.5, 0.5], [cell - 0.5, 0.5], [0.5, cell - 0.5]])
    uv = np.zeros((M, 3, 2))
    uv[..., 0] = (px0[:, None] + corners[None, :, 0]) / size
    uv[..., 1] = 1.0 - (py0[:, None] + corners[None, :, 1]) / size  # OBJ v=0 is the bottom row
    return uv.reshape(-1, 2), (np.clip(img, 0, 1) * 255).astype(np.uint8)


def write_obj(path, V, F, uv=None, N=None):
    """Wavefront OBJ; uv = one per triangle corner (as bake_atlas), N = one normal per vertex."""
    lines = ["# Safe-GS step 16"]
    lines += [f"v {a:.6f} {b:.6f} {c:.6f}" for a, b, c in V]
    if N is not None:
        lines += [f"vn {a:.5f} {b:.5f} {c:.5f}" for a, b, c in N]
    if uv is not None:
        lines += [f"vt {a:.6f} {b:.6f}" for a, b in uv]
    for t, (i, j, k) in enumerate(np.asarray(F) + 1):
        if uv is None and N is None:
            lines.append(f"f {i} {j} {k}")
        else:
            c = [3 * t + 1, 3 * t + 2, 3 * t + 3] if uv is not None else ["", "", ""]
            n = [i, j, k] if N is not None else ["", "", ""]
            lines.append("f " + " ".join(f"{q}/{c_}/{n_}" if N is not None else f"{q}/{c_}"
                                         for q, c_, n_ in zip((i, j, k), c, n)))
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")


def decimate(V, F, target, colors=None):
    import open3d as o3d
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F.astype(np.int32)))
    if colors is not None:
        m.vertex_colors = o3d.utility.Vector3dVector(colors)
    if len(F) > target:
        m = m.simplify_quadric_decimation(int(target))
    m.remove_degenerate_triangles()
    m.remove_duplicated_triangles()
    m.remove_duplicated_vertices()
    m.remove_unreferenced_vertices()
    m.compute_vertex_normals()
    V_out = np.asarray(m.vertices)
    if colors is not None:
        cols = np.asarray(m.vertex_colors)
        if len(cols) != len(V_out):
            cols = knn_colors(V_out, V, np.asarray(colors, float), k=3)
    else:
        cols = None
    return V_out, np.asarray(m.triangles).astype(np.int64), np.asarray(m.vertex_normals), cols


def segment_weights(y, y_bounds):
    """Smooth skinning weights of points along the strip axis: hat functions centred on the segment centres
    (partition of unity, 1 beyond the end centres) -> [N, n_segments]."""
    yb = np.asarray(y_bounds, float)
    yc = 0.5 * (yb[:-1] + yb[1:])
    W = np.zeros((len(y), len(yc)))
    for i in range(len(yc)):
        e = np.zeros(len(yc))
        e[i] = 1.0
        W[:, i] = np.interp(y, yc, e)
    return W


def look_at_xyaxes(pos, target, up=(0, 0, 1)):
    """MuJoCo camera xyaxes (camera looks along -z, y up)."""
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    x = np.cross(f, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return np.concatenate([x, y])


def fmt(v, p=5):
    return " ".join(f"{float(x):.{p}f}" for x in np.ravel(v))
