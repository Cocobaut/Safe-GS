"""AIRBOT Play (DISCOVERSE, MIT) for step 16: MJCF parts, base placement, grasp planning, numerical IK.

The arm is metric (reach ~0.6 m, parallel gripper opening 0-0.084 m), so the sim frame must be metric too
(physics.sim_scale = metres per scene unit). Gripper frame = body link6: +z = approach direction,
fingers close along +-y, pads centred ~0.005 m along +z, site "endpoint" at +0.015 m along +z.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

ROBOT_DIR = Path(__file__).resolve().parent / "robots" / "airbot_play"
HOME_Q = np.array([0.0, -1.0, 1.2, 1.5708, -1.2, -1.5708])
GRIP_OPEN, GRIP_CLOSED = 0.04, 0.0              # tendon ctrl: 0.04 = 0.084 m opening, 0 = closed
MAX_PINCH = 0.07                                 # widest part that fits between the pads (with margin)
PAD_DEPTH = 0.02                                 # how far the pads reach below the grasp top
BASE_CLEAR = 0.10                                # base centre distance behind the object's near side
MAX_WALL_DEG = 35                                # pinchable wall: within the pad friction cone (atan 1 = 45 deg) - margin


def copy_assets(mjcf_dir: Path) -> str:
    """Copies the robot XML (visual geoms -> group 1) and meshes next to the scene. Returns the include folder name."""
    dst = mjcf_dir / "airbot_play"
    (mjcf_dir / "assets" / "airbot_play").mkdir(parents=True, exist_ok=True)
    for f in (ROBOT_DIR / "meshes").glob("*.obj"):
        shutil.copy2(f, mjcf_dir / "assets" / "airbot_play" / f.name)
    dst.mkdir(exist_ok=True)
    for name in ("airbot_play.xml", "airbot_play_appendix.xml", "airbot_play_control.xml", "airbot_play_dependencies.xml"):
        txt = (ROBOT_DIR / name).read_text(encoding="utf-8")
        if name == "airbot_play_dependencies.xml":
            txt = txt.replace('file="', 'file="airbot_play/').replace('file="airbot_play/airbot_play/', 'file="airbot_play/')
            txt = txt.replace('<geom group="2" type="mesh" contype="0" conaffinity="0"/>',
                              '<geom group="1" type="mesh" contype="0" conaffinity="0"/>')
        (dst / name).write_text(txt, encoding="utf-8")
    return "airbot_play"


def plan(obj, grasp_part_vertices, lo, hi, lift_h, edge_y):
    """Base pose + grasp (point, closing axis, friction or weld) + waypoints, all in the metric sim frame."""
    W, D, H = hi - lo
    Vg = grasp_part_vertices
    z_top = float(Vg[:, 2].max())
    # MuJoCo collides with the convex hull of the part: the pads first touch it at their lower edge,
    # so width and centre are measured on the hull cross-section at z_top - PAD_DEPTH
    # deepest pinch (most pad contact) that still fits the gripper; else weld at the default depth
    for depth in (PAD_DEPTH, 0.017, 0.015, 0.012, 0.010, 0.008):
        sec = hull_section(Vg, z_top - depth)
        widths = {"x": float(np.ptp(sec[:, 0])), "y": float(np.ptp(sec[:, 1]))}
        axis = min(widths, key=widths.get)
        if widths[axis] <= MAX_PINCH:
            break
    mode = "friction" if widths[axis] <= MAX_PINCH else "weld"
    reason = "fits the gripper" if mode == "friction" else f"part wider than the gripper ({widths[axis]:.3f} > {MAX_PINCH})"
    # parallel pads only squeeze walls inside the friction cone: wall angle from vertical < atan(mu) (mu_pad = 1)
    upper = hull_section(Vg, z_top - 0.5 * depth)
    wall = float(np.degrees(np.arctan2(0.5 * (np.ptp(sec[:, "xy".index(axis)]) - np.ptp(upper[:, "xy".index(axis)])),
                                       0.5 * depth)))
    if mode == "friction" and wall > MAX_WALL_DEG:
        mode, reason = "weld", f"walls slope {wall:.0f} deg from vertical (> {MAX_WALL_DEG} deg): pads would only press on top"
    if mode == "weld":
        depth = PAD_DEPTH
        sec = hull_section(Vg, z_top - depth)
        widths = {"x": float(np.ptp(sec[:, 0])), "y": float(np.ptp(sec[:, 1]))}
    c = 0.5 * (sec.min(0) + sec.max(0))
    # pads are 0.04 m tall, centred 0.01 m behind (above) the endpoint site: put their lower edge PAD_DEPTH below the
    # top, i.e. pad centre = z_top - PAD_DEPTH + 0.02 and endpoint = pad centre - 0.01
    pad_c = np.array([c[0], c[1], z_top - depth + 0.02])
    endpoint = pad_c - [0, 0, 0.010]
    # base on the table between the object and the nearest edge (or at the edge if closer)
    by = lo[1] - min(BASE_CLEAR + 0.12, max(edge_y, 0.12))
    base = np.array([c[0], by, 0.0])
    lift = max(lift_h, 0.15)
    hover = endpoint + [0, 0, max(0.08, 0.6 * H)]
    wps = {"HOME": hover + [0, -0.08, 0.06], "HOVER": endpoint + [0, 0, max(0.08, 0.6 * H)], "GRASP": endpoint,
           "LIFT": endpoint + [0, 0, lift],
           "CARRY": endpoint + [-0.6 * max(W, 0.1), -0.25 * D, lift]}
    return {"type": "airbot_play", "base": base.tolist(), "base_euler": [0, 0, np.pi / 2],
            "grasp_mode": mode, "grasp_axis": axis, "pinch_widths": widths, "pinch_depth": depth, "max_pinch": MAX_PINCH,
            "wall_angle_deg": round(wall, 1), "grasp_reason": reason,
            "waypoints": {k: np.asarray(v).tolist() for k, v in wps.items()},
            "lift_height": float(lift), "jaw_closed": GRIP_CLOSED, "jaw_open": GRIP_OPEN,
            # compatibility keys used by the cameras / viewer of the serial arm
            "k": 1.0, "L1": 0.27, "L2": 0.29, "L_ee": 0.0, "shoulder": (base + [0, 0, 0.12]).tolist(), "reachable": True}


def hull_section(V, h):
    """Points of the convex hull of V cut by the plane z = h (edges crossing it, interpolated)."""
    from scipy.spatial import ConvexHull
    hull = ConvexHull(V)
    E = {tuple(sorted((s[i], s[(i + 1) % 3]))) for s in hull.simplices for i in range(3)}
    E = np.array(list(E))
    a, b = V[E[:, 0]], V[E[:, 1]]
    m = (a[:, 2] - h) * (b[:, 2] - h) <= 0
    a, b = a[m], b[m]
    t = np.clip((h - a[:, 2]) / np.where(np.abs(b[:, 2] - a[:, 2]) < 1e-12, 1e-12, b[:, 2] - a[:, 2]), 0, 1)
    P = a + t[:, None] * (b - a)
    return P if len(P) else V[V[:, 2] >= h]


def target_rot(axis):
    """Gripper pointing down (link6 +z = -Z), fingers closing along sim X or Y."""
    z = np.array([0.0, 0.0, -1.0])
    y = np.array([1.0, 0.0, 0.0]) if axis == "x" else np.array([0.0, 1.0, 0.0])
    x = np.cross(y, z)
    return np.stack([x, y, z], 1)


def solve_ik(model, target_pos, target_R, q0=None, w_rot=0.3):
    """Best of several seeds (joint 6 / joint 1 offsets avoid the wrist-roll local minimum)."""
    seeds = [HOME_Q] if q0 is None else [np.asarray(q0, float), HOME_Q]
    seeds = seeds + [HOME_Q + [a, 0, 0, 0, 0, b] for a in (0.0, 0.8, -0.8) for b in (np.pi / 2, -np.pi / 2)]
    best = None
    for s in seeds:
        r = _dls(model, np.asarray(target_pos, float), target_R, s, w_rot=w_rot)
        score = r[1] + 0.05 * r[2]
        if best is None or score < best[0]:
            best = (score, r)
        if r[1] < 1e-3 and r[2] < 1e-2:
            break
    return best[1]


def _dls(model, target_pos, target_R, q0, iters=300, damping=1e-3, w_rot=0.3):
    """Damped least squares on site 'endpoint' (position) + body link6 (orientation). -> (q[6], pos err, rot err)."""
    import mujoco
    d = mujoco.MjData(model)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "endpoint")
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link6")
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 7)]
    qadr = np.array([model.jnt_qposadr[j] for j in jids])
    dadr = np.array([model.jnt_dofadr[j] for j in jids])
    lo, hi = model.jnt_range[jids, 0], model.jnt_range[jids, 1]
    q = np.clip(np.array(q0, float), lo, hi)
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for _ in range(iters):
        d.qpos[qadr] = q
        mujoco.mj_kinematics(model, d)
        mujoco.mj_comPos(model, d)
        R = d.xmat[body].reshape(3, 3)
        ep = target_pos - d.site_xpos[site]
        er = 0.5 * sum(np.cross(R[:, i], target_R[:, i]) for i in range(3))
        if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(er) < 1e-3:
            break
        mujoco.mj_jacSite(model, d, jp, jr, site)
        J = np.vstack([jp[:, dadr], w_rot * jr[:, dadr]])
        e = np.concatenate([ep, w_rot * er])
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), e)
        q = np.clip(q + np.clip(dq, -0.2, 0.2), lo, hi)
    d.qpos[qadr] = q
    mujoco.mj_kinematics(model, d)
    R = d.xmat[body].reshape(3, 3)
    return q, float(np.linalg.norm(target_pos - d.site_xpos[site])), \
        float(np.linalg.norm(0.5 * sum(np.cross(R[:, i], target_R[:, i]) for i in range(3))))
