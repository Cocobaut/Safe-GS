"""Step 16.1 (local CPU): physics assets + MJCF of a run (ported from teatime_napkin prepare_physics_assets.py,
build_real_teatime_env.py, rebuild_smooth_env.py; nothing hardcoded: names / counts / sizes come from the run).

Inputs (all found from the run folder):
  table_plane.json (13p)  step13 *_table.ply (13b, name taken from step15 summary)  step15 mesh_object_placed.ply
  step8 removed_object_id<id>.ply (object Gaussians)  step11 point_cloud_inpainted_iter*.ply (background Gaussians)
  step1 cameras.json (a real capture view for the side-by-side 3DGS video)

Sim frame (sim_transform.json): Z = table normal, z = 0 = table top under the object (lowest object vertex - gap),
origin under the object centre, -Y = direction to the nearest table edge (the robot stands there).

Outputs in <run>/step16_physics_engine/:
  sim_transform.json  robot.json  object_gs_binding.json  mjcf/scene_simulation.xml
  mjcf/assets/{table.obj, table_tex.png, object.obj, object_tex.png, object_seg<i>.obj | object_hull<i>.obj}

  python tools_physics/prepare_assets.py --run <outputs/LERF/ours/teatime_napkin> [--mode articulated|flex|rigid]
"""
from __future__ import annotations

import json
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

import physics_io as pio

L1, L2, L_WRIST_GRIP, L_GRIP_PAD = 0.58, 0.54, 0.12, 0.09     # robot of run_real_teatime_sim.py at k = 1


def main():
    ap = ArgumentParser()
    ap.add_argument("--run", required=True, help="run folder or run name (with --dataset)")
    ap.add_argument("--dataset", default="LERF")
    ap.add_argument("--object-id", type=int, default=None)
    ap.add_argument("--mode", default=None, choices=["articulated", "flex", "rigid"])
    ap.add_argument("--sim-scale", type=float, default=None)
    ap.add_argument("--gs-camera", default=None, help="img_name of the capture view used for the 3DGS video")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    run = pio.resolve_run_dir(args.run, args.dataset)
    cfg = pio.load_physics_cfg(run)
    for k, v in (("mode", args.mode), ("sim_scale", args.sim_scale)):
        if v is not None:
            cfg[k] = v
    oid = args.object_id if args.object_id is not None else cfg.get("target_id")
    inp = pio.find_inputs(run, oid)
    out = pio.out_dir(run)
    if (out / "prepare_done.json").exists() and not args.force:
        raise SystemExit(f"{out} already prepared (--force to redo)")
    A = out / "mjcf" / "assets"
    A.mkdir(parents=True, exist_ok=True)
    import mesh_io as mio
    print(f"[16.1] run={run}\n       inputs: " + ", ".join(f"{k}={Path(v).name}" for k, v in inp.items()
                                                       if k not in ("run", "object_id") and v))

    # ------------------------------------------------------------------ 1. sim frame
    plane = mio.load_plane(run_dir=run)
    plane_file = mio.find_plane(run)
    n, e1, e2 = plane["n"], plane["e1"], plane["e2"]
    if np.cross(e1, e2) @ n < 0:
        e2 = -e2
    OV, OF, OC = mio.read_mesh(inp["object_mesh"])
    uvh = mio.to_uvh({**plane, "e2": e2}, OV)
    c_uv = 0.5 * (uvh[:, :2].min(0) + uvh[:, :2].max(0))
    h0 = float(uvh[:, 2].min()) - cfg["gap"]                       # table top under the object
    hull = json.loads(Path(plane_file).read_text(encoding="utf-8")).get("hull_2d")
    if hull and np.cross(plane["e1"], plane["e2"]) @ n < 0:
        hull = [[a, -b] for a, b in hull]                           # hull_2d is in (e1, e2) coordinates
    if hull:
        a_uv, edge_dist = pio.nearest_edge_dir(hull, c_uv)
    else:
        a_uv, edge_dist = np.array([0.0, -1.0]), 0.0
    y_uv = -a_uv                                                    # sim +Y points away from the robot
    x_uv = np.array([y_uv[1], -y_uv[0]])
    Xw, Yw = x_uv[0] * e1 + x_uv[1] * e2, y_uv[0] * e1 + y_uv[1] * e2
    origin = mio.from_uvh({**plane, "e2": e2}, np.array([c_uv[0], c_uv[1], h0]))
    s = float(cfg["sim_scale"])
    fr = pio.SimFrame(np.stack([Xw, Yw, n]), origin, s)

    # ------------------------------------------------------------------ 2. object mesh (step 15) + colours
    from plyfile import PlyData  # noqa: F401  (fail early if missing)
    OVs = fr.to_sim(OV)
    if cfg["object_color"] == "mesh" and OC is not None:
        ocol = OC
    else:
        gxyz, grgb, _ = pio.read_gs_points(inp["object_gs"], min_opacity=0.3)
        ocol = pio.knn_colors(OV, gxyz, grgb)
    V, F, N, C = pio.decimate(OVs, OF, cfg["object_tris"], colors=ocol)
    lo, hi = V.min(0), V.max(0)
    W, D, H = hi - lo
    cx, cy = 0.5 * (lo[:2] + hi[:2])
    print(f"[16.1] object {len(OV)} -> {len(V)} verts, size W={W:.3f} D={D:.3f} H={H:.3f} (sim units), "
          f"z_min={lo[2]:.4f}; nearest table edge {edge_dist * s:.3f}")
    uv, tex = pio.bake_atlas(F, C)
    import cv2
    cv2.imwrite(str(A / "object_tex.png"), tex[..., ::-1])
    pio.write_obj(A / "object.obj", V, F, uv, N)

    mode = cfg["mode"]
    zh = lo[2] + 0.5 * H
    obj = {"mode": mode, "bbox_min": lo.tolist(), "bbox_max": hi.tolist(), "mass": cfg["object_mass"]}
    if mode == "articulated":
        nseg = int(cfg["n_segments"])
        yb = lo[1] + D * np.arange(nseg + 1) / nseg                 # segment 0 = near edge (robot side, grasped)
        tseg = np.clip(np.searchsorted(yb, V[F].mean(1)[:, 1]) - 1, 0, nseg - 1)
        origins = [np.array([cx, yb[max(i, 1)], zh]) for i in range(nseg)]
        segs = []
        for i in range(nseg):
            fi = np.where(tseg == i)[0]
            used = np.unique(F[fi])
            remap = -np.ones(len(V), np.int64)
            remap[used] = np.arange(len(used))
            pio.write_obj(A / f"object_seg{i}.obj", V[used] - origins[i], remap[F[fi]],
                          uv.reshape(-1, 3, 2)[fi].reshape(-1, 2), N[used])
            segs.append({"body": f"object_seg{i}", "mesh": f"object_seg{i}.obj", "origin": origins[i].tolist(),
                         "faces": int(len(fi)), "z_top": float(V[used, 2].max()), "y_range": [float(yb[i]), float(yb[i + 1])]})
        obj.update(segments=segs, y_bounds=yb.tolist())
        grasp = np.array([cx, 0.5 * (yb[0] + yb[1]), segs[0]["z_top"]])
        hang = D * (nseg - 1) / nseg
    elif mode == "flex":
        nx, ny, nz = cfg["flex_grid"]
        hz = min(H, 0.5 * max(W, D))
        sp = np.array([W / (nx - 1), D / (ny - 1), hz / max(nz - 1, 1)])
        ctr = np.array([cx, cy, lo[2] + 0.5 * hz + 0.002 * s])
        nodes = np.zeros((nx * ny * nz, 3))
        for ix in range(nx):                                        # MuJoCo flexcomp grid order (checked in 16.2)
            for iy in range(ny):
                for iz in range(nz):
                    nodes[(ix * ny + iy) * nz + iz] = ctr + (np.array([ix, iy, iz]) - (np.array([nx, ny, nz]) - 1) / 2) * sp
        edge = [(ix * ny + 0) * nz + (nz - 1) for ix in range(nx) if abs(ix - (nx - 1) / 2) <= 1]
        obj.update(grid=[nx, ny, nz], spacing=sp.tolist(), center=ctr.tolist(), rest_nodes=nodes.tolist(),
                   grasp_nodes=edge, rgba=list(np.r_[C.mean(0), 1.0]))
        grasp = np.array([cx, lo[1] + 0.5 * sp[1], ctr[2] + 0.5 * hz])
        hang = D
    else:                                                           # rigid: visual + convex decomposition
        hulls = convex_parts(V, F)
        for i, (hv, hf) in enumerate(hulls):
            pio.write_obj(A / f"object_hull{i}.obj", hv - [cx, cy, zh], hf)
        obj.update(origin=[cx, cy, zh], hulls=[f"object_hull{i}.obj" for i in range(len(hulls))])
        grasp = np.array([cx, cy, hi[2]])
        hang = 0.0

    # ------------------------------------------------------------------ 3. table mesh (step 13b) + 3DGS texture
    TV, TF, _ = mio.read_mesh(inp["table_mesh"])
    TVs = fr.to_sim(TV)
    under = (np.abs(TVs[:, 0] - cx) <= 0.55 * W) & (np.abs(TVs[:, 1] - cy) <= 0.55 * D) & (TVs[:, 2] < 0.5 * H)
    TVs[under, 2] = np.clip(TVs[under, 2], -0.004 * s, 0.0)        # flat support under the object (z = 0 collider)
    tV, tF, tN, _ = pio.decimate(TVs, TF, cfg["table_tris"])
    bxyz, brgb, _ = pio.read_gs_points(inp["background_gs"], min_opacity=0.3)
    tcol = pio.knn_colors(fr.to_world(tV), bxyz, brgb)
    tuv, ttex = pio.bake_atlas(tF, tcol)
    cv2.imwrite(str(A / "table_tex.png"), ttex[..., ::-1])
    pio.write_obj(A / "table.obj", tV, tF, tuv, tN)
    print(f"[16.1] table {len(TV)} -> {len(tV)} verts ({len(tF)} tris), texture {ttex.shape[0]}px, "
          f"{under.sum()} vertices levelled under the object")
    del TV, TF, TVs

    # ------------------------------------------------------------------ 4. robot placement + waypoints
    edge_y = edge_dist * s - (0 - lo[1])                            # table edge beyond the near side of the object
    ee_clear = 0.02
    lift_h = hang + 0.3 * H + 0.15
    rob = place_robot(grasp, lo, hi, lift_h, edge_y, cfg.get("floor_z"), ee_clear)
    print(f"[16.1] robot k={rob['k']:.2f} shoulder={np.round(rob['shoulder'], 3).tolist()} "
          f"reach ok={rob['reachable']}")

    # ------------------------------------------------------------------ 5. Gaussian binding (object Gaussians -> sim)
    from plyfile import PlyData as _P
    g = _P.read(str(inp["object_gs"])).elements[0]
    G = fr.to_sim(np.stack([g["x"], g["y"], g["z"]], 1).astype(np.float64))
    bind = {"object_gs": Path(inp["object_gs"]).relative_to(run).as_posix(), "n_gaussians": int(len(G)), "mode": mode}
    if mode == "articulated":
        yb = np.array(obj["y_bounds"])
        band = 0.04 * D
        Wt = np.zeros((len(G), len(yb) - 1))
        for i in range(len(yb) - 1):
            a_ = np.clip((G[:, 1] - (yb[i] - band)) / (2 * band), 0, 1) if i > 0 else np.ones(len(G))
            b_ = np.clip(((yb[i + 1] + band) - G[:, 1]) / (2 * band), 0, 1) if i < len(yb) - 2 else np.ones(len(G))
            Wt[:, i] = np.minimum(a_, b_)
        Wt /= np.maximum(Wt.sum(1, keepdims=True), 1e-9)
        bind.update(bodies=[sg["body"] for sg in obj["segments"]], weights=np.round(Wt, 4).tolist())
    elif mode == "flex":
        from scipy.spatial import cKDTree
        d, i = cKDTree(np.array(obj["rest_nodes"])).query(G, k=4)
        w = 1 / (d + 1e-4)
        w /= w.sum(1, keepdims=True)
        bind.update(nodes=i.tolist(), weights=np.round(w, 5).tolist(), rest_nodes=obj["rest_nodes"])
    else:
        bind.update(bodies=["object_body"], weights=[[1.0]] * len(G))
    (out / "object_gs_binding.json").write_text(json.dumps(bind), encoding="utf-8")

    # ------------------------------------------------------------------ 6. cameras + MJCF
    cams = make_cameras(fr, inp["cameras"], lo, hi, lift_h, rob, args.gs_camera)
    xml = build_mjcf(obj, rob, cams, cfg, s)
    (out / "mjcf" / "scene_simulation.xml").write_text(xml, encoding="utf-8")
    try:
        import mujoco
        m = mujoco.MjModel.from_xml_path(str(out / "mjcf" / "scene_simulation.xml"))
        print(f"[16.1] MJCF compiles: nbody={m.nbody} ngeom={m.ngeom} nu={m.nu} neq={m.neq}")
    except ImportError:
        print("[16.1] mujoco not installed: MJCF written but not checked")

    st = {"description": "COLMAP/3DGS world -> MuJoCo Z-up frame of step 16", **fr.to_json(),
          "plane_file": str(plane_file), "plane_normal_world": n.tolist(), "plane_d_world": plane["d"],
          "table_top_height_on_plane": h0, "robot_side_uv": a_uv.tolist(), "nearest_table_edge_scene": edge_dist,
          "object_id": inp["object_id"], "inputs": {k: (Path(v).relative_to(run).as_posix() if v and Path(v).is_relative_to(run)
                                                        else (str(v) if v else None))
                                                    for k, v in inp.items() if k not in ("run", "object_id")},
          "object": {k: v for k, v in obj.items() if k not in ("rest_nodes",)}, "config": cfg}
    (out / "sim_transform.json").write_text(json.dumps(st, indent=1), encoding="utf-8")
    (out / "robot.json").write_text(json.dumps({**rob, "cameras": cams}, indent=1), encoding="utf-8")
    (out / "prepare_done.json").write_text(json.dumps({"ok": True, "seconds": round(time.time() - t0, 1),
                                                       "mode": mode}, indent=1))
    print(f"[16.1] done in {time.time() - t0:.0f}s -> {out}")


# ---------------------------------------------------------------------------------------------------- helpers
def convex_parts(V, F, max_parts=12):
    """CoACD convex decomposition (if installed), else 4 quadrant hulls."""
    try:
        import coacd
        parts = coacd.run_coacd(coacd.Mesh(V, F), threshold=0.08, max_convex_hull=max_parts)
        return [(np.asarray(v), np.asarray(f)) for v, f in parts]
    except Exception as e:  # noqa: BLE001
        print(f"[16.1] CoACD unavailable ({type(e).__name__}) -> quadrant hulls")
    from scipy.spatial import ConvexHull
    c = V.mean(0)
    out = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            P = V[((V[:, 0] - c[0]) * sx >= 0) & ((V[:, 1] - c[1]) * sy >= 0)]
            if len(P) >= 4:
                h = ConvexHull(P)
                out.append((P, h.simplices))
    return out


def place_robot(grasp, lo, hi, lift_h, edge_y, floor_z, ee_clear):
    """Scale k of the run_real_teatime_sim.py arm and shoulder position so that every waypoint is reachable,
    with the base beyond the nearest table edge (-Y side)."""
    W, D, H = hi - lo
    best = None
    for k in np.arange(0.8, 4.01, 0.05):
        R = (L1 + L2) * k
        lee = (L_WRIST_GRIP + L_GRIP_PAD) * k
        ee = grasp + [0, 0, ee_clear * k]
        wps = {"HOVER": ee + [0, 0, max(0.3 * k, 0.6 * H)], "GRASP": ee, "LIFT": ee + [0, 0, lift_h],
               "CARRY": ee + [-0.6 * max(W, 0.3), -0.35 * D, lift_h]}
        off = max(edge_y + 0.25 * k, 0.35 * k)
        sh = np.array([grasp[0] + 0.15 * k, lo[1] - off, 0.45 * k])
        dist = [np.linalg.norm(p + [0, 0, lee] - sh) for p in wps.values()]
        ok = all(0.3 * R <= d_ <= 0.92 * R for d_ in dist)
        best = (k, R, lee, wps, sh, ok)
        if ok:
            break
    k, R, lee, wps, sh, ok = best
    base_z = floor_z if (floor_z is not None and floor_z < sh[2] - 0.8 * k) else sh[2] - 2.38 * k
    return {"k": float(k), "L1": L1 * k, "L2": L2 * k, "L_ee": float(lee), "shoulder": sh.tolist(),
            "base": [float(sh[0]), float(sh[1]), float(base_z)], "waypoints": {a: b.tolist() for a, b in wps.items()},
            "jaw_closed": 0.065 * k, "jaw_open": 0.0, "lift_height": float(lift_h), "reachable": bool(ok)}


def make_cameras(fr, cams_json, lo, hi, lift_h, rob, want=None):
    ext = float(max((hi - lo)[:2].max(), lift_h, rob["L1"]))
    T = np.array([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * (lift_h + hi[2])])
    out = {}
    for name, pos in {"cam_side": T + 2.4 * ext * np.array([0.85, -0.45, 0.35]),
                      "cam_robot": T + 2.4 * ext * np.array([-0.75, -0.55, 0.4]),
                      "cam_top": T + [0.001, -0.6 * ext, 2.6 * ext]}.items():
        out[name] = {"pos": pos.tolist(), "xyaxes": pio.look_at_xyaxes(pos, T).tolist(), "fovy": 50.0}
    if cams_json is None:
        return out
    # a real capture view that sees the whole action volume (object + lift) -> 3DGS renders well there
    box = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (0.0, hi[2] + lift_h)])
    Bw = fr.to_world(box)
    best = None
    for c in json.loads(Path(cams_json).read_text(encoding="utf-8")):
        Rc, p = np.array(c["rotation"]), np.array(c["position"])
        q = (Bw - p) @ Rc
        if (q[:, 2] <= 0).any():
            continue
        u = c["fx"] * q[:, 0] / q[:, 2] + c["width"] / 2
        v = c["fy"] * q[:, 1] / q[:, 2] + c["height"] / 2
        inside = ((u > 0.02 * c["width"]) & (u < 0.98 * c["width"]) & (v > 0.02 * c["height"]) & (v < 0.98 * c["height"])).mean()
        off = np.hypot(u.mean() / c["width"] - 0.5, v.mean() / c["height"] - 0.5)
        area = np.ptp(np.clip(u, 0, c["width"])) * np.ptp(np.clip(v, 0, c["height"])) / (c["width"] * c["height"])
        # whole action volume in frame first, then as large (close) and centred as possible
        score = (1.0 + area - 0.3 * off) if inside >= 0.95 else inside - 0.3 * off
        if want and c["img_name"] == want:
            score = 1e9
        if best is None or score > best[0]:
            best = (score, c, inside)
    if best is None:
        return out
    _, c, inside = best
    Rc, p = np.array(c["rotation"]), np.array(c["position"])
    xs, ys = fr.dir_to_sim(Rc[:, 0]), -fr.dir_to_sim(Rc[:, 1])      # OpenCV -> MuJoCo camera (y up, looks along -z)
    out["cam_gs"] = {"pos": fr.to_sim(p).tolist(), "xyaxes": np.r_[xs, ys].tolist(),
                     "fovy": float(np.degrees(2 * np.arctan(c["height"] / (2 * c["fy"])))),
                     "img_name": c["img_name"], "width": c["width"], "height": c["height"], "fx": c["fx"], "fy": c["fy"],
                     "c2w_cv_world": np.r_[np.c_[Rc, p], [[0, 0, 0, 1]]].tolist(), "box_inside_frac": float(inside)}
    print(f"[16.1] 3DGS camera = capture view {c['img_name']} (action volume {inside * 100:.0f}% in frame)")
    return out


def build_mjcf(obj, rob, cams, cfg, s):
    k = rob["k"]
    S = lambda x: x * k                                              # noqa: E731  robot dimensions
    sh, base = np.array(rob["shoulder"]), np.array(rob["base"])
    turret_z = sh[2] - S(0.08) - base[2]
    col_lo, col_hi = S(0.33), turret_z - S(0.15)
    mass = float(cfg["object_mass"])
    fric = 'condim="4" friction="1.2 0.02 0.005" solref="0.004 1" solimp="0.9 0.95 0.001 0.5 2"'
    dt = 0.001 if obj["mode"] == "flex" else 0.002
    integ = 'integrator="implicitfast"' if obj["mode"] != "flex" else 'integrator="implicitfast" solver="CG" iterations="100"'
    W = max(1280, int(cams.get("cam_gs", {}).get("width", 0)))
    Hh = max(720, int(cams.get("cam_gs", {}).get("height", 0)))
    ext = float(max(np.ptp([obj["bbox_min"], obj["bbox_max"]], 0)[:2].max(), 1.0))

    assets = ['<texture type="2d" name="table_tex" file="table_tex.png"/>',
              '<material name="table_mat" texture="table_tex" specular="0.05" shininess="0.05"/>',
              '<mesh name="table_mesh" file="table.obj" inertia="shell"/>',
              '<texture type="2d" name="object_tex" file="object_tex.png"/>',
              '<material name="object_mat" texture="object_tex" specular="0.05" shininess="0.05"/>',
              '<texture type="skybox" builtin="gradient" rgb1="0.88 0.90 0.92" rgb2="0.72 0.75 0.78" width="512" height="512"/>',
              '<material name="robot_white" rgba="0.92 0.93 0.95 1" specular="0.6" shininess="0.7"/>',
              '<material name="robot_dark" rgba="0.18 0.20 0.24 1" specular="0.5" shininess="0.6"/>',
              '<material name="robot_joint" rgba="0.15 0.55 0.88 1" specular="0.5" shininess="0.6"/>',
              '<material name="pad_mat" rgba="0.92 0.36 0.12 1" specular="0.2"/>']
    welds = []
    if obj["mode"] == "articulated":
        segs = obj["segments"]
        nseg = len(segs)
        for sg in segs:
            assets.append(f'<mesh name="{sg["body"]}" file="{sg["mesh"]}" inertia="shell"/>')
        seg_len = (obj["bbox_max"][1] - obj["bbox_min"][1]) / nseg
        stiff = 0.4 * (mass / nseg) * 9.81 * seg_len
        body = ""
        for i in reversed(range(nseg)):
            sg = segs[i]
            geom = (f'<geom name="{sg["body"]}_geom" type="mesh" mesh="{sg["body"]}" material="object_mat" '
                    f'mass="{mass / nseg:.5f}" group="2" {fric}/>')
            if i == 0:
                body = (f'<body name="{sg["body"]}" pos="{pio.fmt(sg["origin"])}">\n  <freejoint name="object_free"/>\n'
                        f'  {geom}\n{body}</body>\n')
            else:
                rel = np.array(sg["origin"]) - np.array(segs[i - 1]["origin"]) if i > 1 else np.zeros(3)
                body = (f'<body name="{sg["body"]}" pos="{pio.fmt(rel)}">\n'
                        f'  <joint name="fold{i}" type="hinge" axis="1 0 0" range="-1.75 1.75" stiffness="{stiff:.5f}" '
                        f'damping="{0.25 * stiff:.5f}"/>\n  {geom}\n{body}</body>\n')
        obj_xml = body
        welds.append(f'<weld name="grasp_weld" body1="robot_gripper" body2="{segs[0]["body"]}" active="false" '
                     f'solref="0.005 1" solimp="0.95 0.99 0.001"/>')
    elif obj["mode"] == "flex":
        nx, ny, nz = obj["grid"]
        obj_xml = (f'<flexcomp name="object_flex" type="grid" count="{nx} {ny} {nz}" spacing="{pio.fmt(obj["spacing"])}" '
                   f'pos="{pio.fmt(obj["center"])}" dim="3" mass="{mass}" radius="{0.006 * s:.4f}" rgba="{pio.fmt(obj["rgba"], 3)}">\n'
                   f'  <contact selfcollide="none" {fric}/>\n'
                   f'  <elasticity young="{cfg["flex_young"]}" poisson="0.12" damping="3.0"/>\n</flexcomp>\n')
        for j in obj["grasp_nodes"]:
            welds.append(f'<weld name="grasp_weld_{j}" body1="robot_gripper" body2="object_flex_{j}" active="false" '
                         f'solref="0.005 1" solimp="0.95 0.99 0.001"/>')
    else:
        assets.append('<mesh name="object_vis" file="object.obj" inertia="shell"/>')
        hulls = "".join(f'<mesh name="object_hull{i}" file="{h}"/>' for i, h in enumerate(obj["hulls"]))
        assets.append(hulls)
        c = np.array(obj["origin"])
        geoms = "".join(f'  <geom name="object_col{i}" type="mesh" mesh="object_hull{i}" mass="{mass / len(obj["hulls"]):.5f}" '
                        f'group="3" rgba="1 1 1 0" {fric}/>\n' for i in range(len(obj["hulls"])))
        obj_xml = (f'<body name="object_body" pos="{pio.fmt(c)}">\n  <freejoint name="object_free"/>\n'
                   f'  <geom name="object_visual" type="mesh" mesh="object_vis" pos="{pio.fmt(-c)}" material="object_mat" '
                   f'contype="0" conaffinity="0" mass="0" group="2"/>\n{geoms}</body>\n')
        welds.append('<weld name="grasp_weld" body1="robot_gripper" body2="object_body" active="false" '
                     'solref="0.005 1" solimp="0.95 0.99 0.001"/>')

    cam_xml = "".join(f'<camera name="{n_}" pos="{pio.fmt(c["pos"])}" xyaxes="{pio.fmt(c["xyaxes"])}" fovy="{c["fovy"]:.3f}"/>\n'
                      for n_, c in cams.items())
    nc = 'contype="0" conaffinity="0" group="1"'
    floor = (f'<geom name="floor_collider" type="plane" pos="0 0 {base[2]:.4f}" size="{3 * ext:.2f} {3 * ext:.2f} 0.05" '
             f'rgba="0.82 0.82 0.84 1"/>\n' if cfg.get("floor_z") is not None else "")
    robot = f"""<body name="robot_base" pos="{pio.fmt(base)}">
  <geom name="base_chassis" type="box" pos="0 0 {S(0.18):.4f}" size="{S(0.38):.4f} {S(0.32):.4f} {S(0.15):.4f}" material="robot_dark" {nc}/>
  <geom name="spine_column" type="cylinder" fromto="0 0 {col_lo:.4f} 0 0 {col_hi:.4f}" size="{S(0.10):.4f}" material="robot_white" {nc}/>
  <geom name="shoulder_mount" type="box" pos="0 0 {turret_z - S(0.15):.4f}" size="{S(0.14):.4f} {S(0.16):.4f} {S(0.15):.4f}" material="robot_dark" {nc}/>
  <body name="shoulder_turret" pos="0 0 {turret_z:.4f}">
    <joint name="joint_base_yaw" type="hinge" axis="0 0 1" range="-3.14 3.14" damping="{40 * k ** 5:.2f}"/>
    <geom name="turret" type="cylinder" pos="0 0 {S(0.04):.4f}" size="{S(0.11):.4f} {S(0.04):.4f}" material="robot_joint" {nc}/>
    <body name="upper_arm" pos="0 0 {S(0.08):.4f}">
      <joint name="joint_shoulder_pitch" type="hinge" axis="1 0 0" range="-2.5 2.5" damping="{40 * k ** 5:.2f}"/>
      <geom name="shoulder_ball" type="sphere" size="{S(0.09):.4f}" material="robot_joint" {nc}/>
      <geom name="upper_arm_capsule" type="capsule" fromto="0 0 0 0 {S(L1):.4f} 0" size="{S(0.052):.4f}" material="robot_white" {nc}/>
      <body name="forearm" pos="0 {S(L1):.4f} 0">
        <joint name="joint_elbow_pitch" type="hinge" axis="1 0 0" range="-2.8 2.8" damping="{30 * k ** 5:.2f}"/>
        <geom name="elbow_ring" type="cylinder" euler="0 1.5708 0" size="{S(0.072):.4f} {S(0.055):.4f}" material="robot_joint" {nc}/>
        <geom name="forearm_capsule" type="capsule" fromto="0 0 0 0 {S(L2):.4f} 0" size="{S(0.046):.4f}" material="robot_white" {nc}/>
        <body name="wrist" pos="0 {S(L2):.4f} 0">
          <joint name="joint_wrist_pitch" type="hinge" axis="1 0 0" range="-2.8 2.8" damping="{20 * k ** 5:.2f}"/>
          <geom name="wrist_ring" type="cylinder" euler="0 1.5708 0" size="{S(0.055):.4f} {S(0.045):.4f}" material="robot_joint" {nc}/>
          <body name="gripper_base" pos="0 {S(0.06):.4f} 0">
            <joint name="joint_wrist_yaw" type="hinge" axis="0 1 0" range="-3.14 3.14" damping="{15 * k ** 5:.2f}"/>
            <geom name="wrist_shaft" type="cylinder" pos="0 {S(0.03):.4f} 0" euler="1.5708 0 0" size="{S(0.042):.4f} {S(0.03):.4f}" material="robot_dark" {nc}/>
            <body name="robot_gripper" pos="0 {S(0.06):.4f} 0">
              <geom name="palm_bar" type="box" size="{S(0.18):.4f} {S(0.024):.4f} {S(0.035):.4f}" material="robot_dark" {nc}/>
              <body name="left_finger" pos="{-S(0.14):.4f} {S(0.04):.4f} 0">
                <joint name="finger_left" type="slide" axis="1 0 0" range="0 {S(0.16):.4f}" damping="{10 * k ** 3:.2f}"/>
                <geom name="left_finger_arm" type="box" pos="0 {S(0.03):.4f} 0" size="{S(0.016):.4f} {S(0.04):.4f} {S(0.03):.4f}" material="robot_dark" {nc}/>
                <geom name="left_finger_pad" type="box" pos="{S(0.014):.4f} {S(0.05):.4f} 0" size="{S(0.014):.4f} {S(0.035):.4f} {S(0.028):.4f}" material="pad_mat" group="1" condim="4" friction="4.0 0.05 0.005"/>
              </body>
              <body name="right_finger" pos="{S(0.14):.4f} {S(0.04):.4f} 0">
                <joint name="finger_right" type="slide" axis="-1 0 0" range="0 {S(0.16):.4f}" damping="{10 * k ** 3:.2f}"/>
                <geom name="right_finger_arm" type="box" pos="0 {S(0.03):.4f} 0" size="{S(0.016):.4f} {S(0.04):.4f} {S(0.03):.4f}" material="robot_dark" {nc}/>
                <geom name="right_finger_pad" type="box" pos="{-S(0.014):.4f} {S(0.05):.4f} 0" size="{S(0.014):.4f} {S(0.035):.4f} {S(0.028):.4f}" material="pad_mat" group="1" condim="4" friction="4.0 0.05 0.005"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </body>
</body>
"""
    g4 = k ** 5                                                      # link inertia ~ k^5
    act = f"""<position name="act_base_yaw" joint="joint_base_yaw" kp="{2000 * g4:.1f}" kv="{80 * g4:.1f}" ctrlrange="-3.14 3.14"/>
<position name="act_shoulder_pitch" joint="joint_shoulder_pitch" kp="{2500 * g4:.1f}" kv="{100 * g4:.1f}" ctrlrange="-2.5 2.5"/>
<position name="act_elbow_pitch" joint="joint_elbow_pitch" kp="{2000 * g4:.1f}" kv="{80 * g4:.1f}" ctrlrange="-2.8 2.8"/>
<position name="act_wrist_pitch" joint="joint_wrist_pitch" kp="{1000 * g4:.1f}" kv="{40 * g4:.1f}" ctrlrange="-2.8 2.8"/>
<position name="act_wrist_yaw" joint="joint_wrist_yaw" kp="{800 * g4:.1f}" kv="{30 * g4:.1f}" ctrlrange="-3.14 3.14"/>
<position name="act_finger_left" joint="finger_left" kp="{3000 * k ** 3:.1f}" kv="{50 * k ** 3:.1f}" ctrlrange="0 {S(0.16):.4f}"/>
<position name="act_finger_right" joint="finger_right" kp="{3000 * k ** 3:.1f}" kv="{50 * k ** 3:.1f}" ctrlrange="0 {S(0.16):.4f}"/>"""
    ind = lambda t, n_: "\n".join(" " * n_ + ln if ln else ln for ln in t.splitlines())  # noqa: E731
    return f"""<mujoco model="safe_gs_step16">
  <compiler angle="radian" meshdir="assets" texturedir="assets" autolimits="true"/>
  <option timestep="{dt}" gravity="0 0 -9.81" {integ} cone="elliptic" impratio="20"/>
  <visual>
    <headlight diffuse="0.8 0.78 0.76" ambient="0.45 0.45 0.45" specular="0.15 0.15 0.15"/>
    <rgba haze="0.86 0.88 0.90 1"/>
    <global offwidth="{W}" offheight="{Hh}"/>
    <map znear="0.005" zfar="200"/>
  </visual>
  <asset>
{ind(chr(10).join(assets), 4)}
  </asset>
  <worldbody>
    <light pos="0 {-2 * ext:.2f} {3 * ext:.2f}" dir="0 0.3 -1" directional="true" diffuse="0.6 0.58 0.55" castshadow="false"/>
{ind(cam_xml, 4)}
    <body name="environment">
      <geom name="table_visual" type="mesh" mesh="table_mesh" material="table_mat" contype="0" conaffinity="0" group="0"/>
      <geom name="table_collider" type="plane" size="{3 * ext:.2f} {3 * ext:.2f} 0.05" group="3" rgba="0 0 0 0" {fric}/>
{ind(floor, 6)}
    </body>
{ind(obj_xml, 4)}
{ind(robot, 4)}
  </worldbody>
  <equality>
{ind(chr(10).join(welds), 4)}
  </equality>
  <actuator>
{ind(act, 4)}
  </actuator>
</mujoco>
"""


if __name__ == "__main__":
    main()
