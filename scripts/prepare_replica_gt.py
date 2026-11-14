"""plan.md 1.1.1 / Pha 0 (part 2): Replica GT for the object runs of configs/datasets/replica.yaml.

Per scene (<DATA_DIR>/<data_subdir>/<scene>, made by prepare_replica_colmap.py):
  gt_mesh/<obj>_gt.ply            GT object mesh, cut from replica_v1 habitat/mesh_semantic.ply by instance id
  gt_mesh/<obj>_gt_capped.ply     same + bottom cap on the support plane (R9)
  gt_mesh/<obj>_footprint.json    2D footprint polygon on the support plane + plane fit + bbox + best views
  gt_inst_masks/<frame>.png       16-bit instance-id map per sampled frame (ray cast of mesh_semantic.ply with the
                                  COLMAP poses) -> step 6id: --ref-masks <dir> --ref-id <gt_instance_id>
  gt_check.json                   coordinate check replica_v1 vs NICE-SLAM mesh (vertex max |diff|)

The object instance ids come from `gt_instance_id` of each object in replica.yaml.

  python scripts/prepare_replica_gt.py --scenes office0 room0            # all objects with gt_instance_id
  python scripts/prepare_replica_gt.py --scenes office0 --no-masks
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.datasets import load_dataset_yaml  # noqa: E402
from common.paths import DATA_DIR  # noqa: E402


def read_semantic_ply(path: Path):
    """replica-instance-mesh-format v0: vertex xyz/normal/rgb, quad faces + uint16 object_id."""
    raw = path.read_bytes()
    h = raw.index(b"end_header\n") + 11
    hdr = raw[:h].decode()
    nv = int(hdr.split("element vertex ")[1].split()[0])
    nf = int(hdr.split("element face ")[1].split()[0])
    vdt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                    ("r", "u1"), ("g", "u1"), ("b", "u1")])
    v = np.frombuffer(raw, vdt, nv, h)
    off = h + nv * vdt.itemsize
    fdt = np.dtype([("n", "u1"), ("i", "<u4", 4), ("o", "<u2")])
    f = np.frombuffer(raw, fdt, nf, off)
    if not (f["n"] == 4).all() or off + nf * fdt.itemsize != len(raw):
        raise ValueError(f"{path}: expected only quad faces")
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    rgb = np.stack([v["r"], v["g"], v["b"]], 1)
    return xyz, rgb, f["i"].astype(np.int64), f["o"].astype(np.int64)


def quads_to_tris(q):
    return np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])


def read_colmap_images(path: Path):
    cams = []
    for line in path.read_text().splitlines():
        t = line.split()
        if len(t) == 10 and not line.startswith("#"):
            qw, qx, qy, qz = map(float, t[1:5])
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            cams.append((Path(t[9]).stem, R, np.array(list(map(float, t[5:8])))))
    return cams


def read_colmap_camera(path: Path):
    for line in path.read_text().splitlines():
        t = line.split()
        if t and not line.startswith("#"):
            w, h = int(t[2]), int(t[3])
            p = list(map(float, t[4:]))
            fx, fy, cx, cy = (p[0], p[0], p[1], p[2]) if t[1] == "SIMPLE_PINHOLE" else p[:4]
            return w, h, fx, fy, cx, cy
    raise ValueError(path)


def plane_frame(n):
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(n, a); e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1), n


def process_object(name, iid, xyz, rgb, quads, oid, up, out: Path, cls_name):
    sel = oid == iid
    if not sel.any():
        raise SystemExit(f"instance id {iid} ({name}) not found in mesh_semantic.ply")
    tris = quads_to_tris(quads[sel])
    used, inv = np.unique(tris, return_inverse=True)
    v, f = xyz[used], inv.reshape(-1, 3)
    obj = trimesh.Trimesh(v, f, vertex_colors=rgb[used], process=False)
    obj.export(out / f"{name}_gt.ply")

    # support plane: scene vertices (other instances) in a 12 cm ring around the object, near its lowest point
    h_obj = v @ up
    e1, e2, _ = plane_frame(up)
    uv_obj = np.stack([v @ e1, v @ e2], 1)
    others = np.unique(quads[~sel])
    p = xyz[others]
    uv = np.stack([p @ e1, p @ e2], 1)
    lo, hi = uv_obj.min(0) - 0.12, uv_obj.max(0) + 0.12
    hz = p @ up
    m = (uv >= lo).all(1) & (uv <= hi).all(1) & (np.abs(hz - h_obj.min()) < 0.04)
    plane_pts = p[m]
    if len(plane_pts) >= 20:
        c = plane_pts.mean(0)
        nrm = np.linalg.svd(plane_pts - c)[2][-1]
        nrm = nrm if nrm @ up > 0 else -nrm
        rms = float(np.sqrt((((plane_pts - c) @ nrm) ** 2).mean()))
    else:                                           # fallback: horizontal plane at the object's lowest point
        c, nrm, rms = v[np.argmin(h_obj)], up.copy(), None
    d = -float(nrm @ c)

    # footprint: convex hull of the object projected on the support plane; cap = that polygon on the plane
    a1, a2, _ = plane_frame(nrm)
    q = np.stack([v @ a1, v @ a2], 1)
    hull = ConvexHull(q)
    poly2 = q[hull.vertices]
    poly3 = poly2[:, :1] * a1 + poly2[:, 1:] * a2 - d * nrm          # points on the plane
    centre3 = poly3.mean(0)
    base = len(v)
    cap_v = np.vstack([poly3, centre3])
    k = len(poly3)
    cap_f = np.array([[base + k, base + (i + 1) % k, base + i] for i in range(k)])   # normal = -nrm (downwards)
    capped = trimesh.Trimesh(np.vstack([v, cap_v]), np.vstack([f, cap_f]),
                             vertex_colors=np.vstack([rgb[used], np.full((k + 1, 3), 128, np.uint8)]), process=False)
    capped.export(out / f"{name}_gt_capped.ply")

    height = float((v @ nrm + d).max())
    gap = float((v @ nrm + d).min())
    info = {"object": name, "gt_instance_id": int(iid), "class_name": cls_name,
            "n_faces": int(len(f)), "bbox_min": v.min(0).tolist(), "bbox_max": v.max(0).tolist(),
            "extent_m": (v.max(0) - v.min(0)).tolist(), "height_above_plane_m": height,
            "lowest_point_to_plane_m": gap,
            "support_plane": {"normal": nrm.tolist(), "d": d, "point": c.tolist(),
                              "n_points": int(len(plane_pts)), "rms_m": rms},
            "footprint_polygon_3d": poly3.tolist(), "footprint_area_m2": float(hull.volume),
            "capped_watertight": bool(capped.is_watertight)}
    (out / f"{name}_footprint.json").write_text(json.dumps(info, indent=2))
    return info


def render_masks(xyz, quads, oid, cams, cam, out: Path):
    w, h, fx, fy, cx, cy = cam
    tris = quads_to_tris(quads)
    tri_oid = np.concatenate([oid, oid]).astype(np.uint16)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(xyz.astype(np.float32)), o3d.core.Tensor(tris.astype(np.uint32)))
    K = o3d.core.Tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], o3d.core.Dtype.Float64)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, R, t in cams:
        E = np.eye(4); E[:3, :3] = R; E[:3, 3] = t
        rays = scene.create_rays_pinhole(K, o3d.core.Tensor(E), w, h)
        prim = scene.cast_rays(rays)["primitive_ids"].numpy()
        ids = np.zeros(prim.shape, np.uint16)
        hit = prim != scene.INVALID_ID
        ids[hit] = tri_oid[prim[hit]]
        Image.fromarray(ids).save(out / f"{name}.png")
        for i in np.unique(ids):
            counts[int(i)] = counts.get(int(i), 0) + 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["office0", "room0"])
    ap.add_argument("--no-masks", action="store_true")
    args = ap.parse_args()
    cfg = load_dataset_yaml("Replica")
    dflt = cfg["defaults"]
    for sc in args.scenes:
        scfg = cfg["scenes"][sc]
        root = DATA_DIR / dflt["data_subdir"] / sc
        sem_dir = DATA_DIR / dflt["gt_semantic_subdir"] / scfg["gt_scene_v1"] / "habitat"
        xyz, rgb, quads, oid = read_semantic_ply(sem_dir / "mesh_semantic.ply")
        info_sem = json.loads((sem_dir / "info_semantic.json").read_text())
        up = -np.asarray(info_sem["gravity_dir"], float); up /= np.linalg.norm(up)
        cls = {o["id"]: o["class_name"] for o in info_sem["objects"]}

        check = {"semantic_mesh": str(sem_dir / "mesh_semantic.ply")}
        nice = DATA_DIR / dflt["raw_subdir"] / f"{sc}_mesh.ply"
        if nice.exists():
            from plyfile import PlyData
            vn = PlyData.read(str(nice))["vertex"]
            vn = np.stack([vn["x"], vn["y"], vn["z"]], 1)
            check["nice_slam_mesh"] = str(nice)
            check["max_abs_vertex_diff_m"] = float(np.abs(vn - xyz).max()) if vn.shape == xyz.shape else None
            check["same_frame"] = check["max_abs_vertex_diff_m"] is not None and check["max_abs_vertex_diff_m"] < 1e-4
        (root / "gt_check.json").write_text(json.dumps(check, indent=2))
        print(f"[{sc}] coordinate check: {check.get('max_abs_vertex_diff_m')}  same_frame={check.get('same_frame')}")

        out = root / "gt_mesh"
        out.mkdir(parents=True, exist_ok=True)
        for oname, o in (scfg.get("objects") or {}).items():
            iid = o.get("gt_instance_id")
            if iid is None or oname.endswith("_nocome"):
                continue
            inf = process_object(oname, int(iid), xyz, rgb, quads, oid, up, out, cls.get(iid, "?"))
            print(f"[{sc}] {oname}: id {iid} {inf['class_name']}, extent {np.round(inf['extent_m'], 3).tolist()} m, "
                  f"plane rms {inf['support_plane']['rms_m']}, gap {inf['lowest_point_to_plane_m']:.4f} m, "
                  f"capped watertight {inf['capped_watertight']}")

        if not args.no_masks:
            cams = read_colmap_images(root / "sparse/0/images.txt")
            cam = read_colmap_camera(root / "sparse/0/cameras.txt")
            counts = render_masks(xyz, quads, oid, cams, cam, root / "gt_inst_masks")
            for oname, o in (scfg.get("objects") or {}).items():
                if o.get("gt_instance_id") is not None and not oname.endswith("_nocome"):
                    print(f"[{sc}] {oname}: instance {o['gt_instance_id']} in {counts.get(int(o['gt_instance_id']), 0)}"
                          f"/{len(cams)} frames")
            print(f"[{sc}] masks -> {root / 'gt_inst_masks'}")


if __name__ == "__main__":
    main()
