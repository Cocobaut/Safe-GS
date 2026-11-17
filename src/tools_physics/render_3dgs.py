"""Step 16.3 (local, torch; GPU if available): the 16.2 trajectory rendered with the Gaussians
(ported from teatime_napkin render_cloth_draping_3dgs.py / render_discoverse_3dgs_trajectory.py).

- camera = the real capture view chosen in 16.1 (cam_gs), so the Gaussians are seen from a trained viewpoint
- background = step 11 inpainted Gaussians (object removed + table inpainted), rendered once with expected depth
- object = step 8 Gaussians, moved every frame by the simulated bodies (articulated / rigid: blended rigid
  transforms of the segments; flex: linear blend of the flex node displacements), kept in the COLMAP world frame
  so their SH colours stay valid (the camera is moved instead of the Gaussians)
- robot = MuJoCo render of the robot geoms from the same camera, depth-composited over the splats
Rendering: gs_render.py model (SH, opacity, scale, quaternion, Mip-Splatting filter_3D) with a tiled rasterizer
(gs_render.render is dense pixels x Gaussians and cannot hold the ~2M background Gaussians).

Outputs in <run>/step16_physics_engine/videos/: render_3dgs_synced.mp4, comparison_mujoco_vs_3dgs.mp4,
  frame_3dgs_first.png / frame_3dgs_lift.png

  python tools_physics/render_3dgs.py --run <run dir | run name> [--scale 0.5] [--max-frames N]
"""
from __future__ import annotations

import json
import time
from argparse import ArgumentParser

import numpy as np
import torch

import physics_io as pio
from gs_render import eval_sh, quat_to_rot


# torch <-> numpy without the numpy C bridge (works even when torch was built against another numpy major)
def n2t(a, device="cpu"):
    a = np.ascontiguousarray(a, dtype=np.float32)
    return torch.frombuffer(bytearray(a.tobytes()), dtype=torch.float32).reshape(a.shape).to(device)


def t2n(t):
    import ctypes
    t = t.detach().to("cpu", torch.float32).contiguous()
    buf = (ctypes.c_float * t.numel()).from_address(t.data_ptr())
    return np.ctypeslib.as_array(buf).reshape(tuple(t.shape)).copy()


def load_ply(path, device="cpu"):
    """gs_render.load_ply without torch.from_numpy."""
    from plyfile import PlyData
    v = PlyData.read(str(path)).elements[0]
    names = [p.name for p in v.properties]
    get = lambda k: np.asarray(v[k], np.float32)  # noqa: E731
    n_rest = len([k for k in names if k.startswith("f_rest_")])
    deg = int(round(np.sqrt(n_rest / 3 + 1))) - 1
    dc = np.stack([get(f"f_dc_{i}") for i in range(3)], 1)[:, :, None]
    rest = (np.stack([get(f"f_rest_{i}") for i in range(n_rest)], 1).reshape(len(dc), 3, -1) if n_rest
            else np.zeros((len(dc), 3, 0), np.float32))
    g = {"xyz": np.stack([get("x"), get("y"), get("z")], 1), "sh": np.concatenate([dc, rest], 2),
         "opacity": 1 / (1 + np.exp(-get("opacity"))), "scale": np.exp(np.stack([get(f"scale_{i}") for i in range(3)], 1)),
         "rot": np.stack([get(f"rot_{i}") for i in range(4)], 1),
         "filter3d": get("filter_3D") if "filter_3D" in names else np.zeros(len(dc), np.float32)}
    out = {k: n2t(x, device) for k, x in g.items()}
    out["deg"] = deg
    return out


# ---------------------------------------------------------------------------------------------------- rasterizer
@torch.no_grad()
def project(g, c2w_cv, K, W, H, near=0.02, chunk=400_000):
    """Screen-space Gaussians (front of the camera, on screen), sorted by depth."""
    dev = g["xyz"].device
    c2w = n2t(np.asarray(c2w_cv), dev)
    w2c = torch.linalg.inv(c2w)
    fx, fy, cx, cy = [float(k) for k in K]
    outs = []
    for s in range(0, len(g["xyz"]), chunk):
        sl = slice(s, s + chunk)
        xyz = g["xyz"][sl]
        p = xyz @ w2c[:3, :3].T + w2c[:3, 3]
        z = p[:, 2]
        s2 = g["scale"][sl] ** 2
        s2f = s2 + g["filter3d"][sl][:, None] ** 2
        op = g["opacity"][sl] * torch.sqrt(s2.prod(1) / s2f.prod(1))
        R = quat_to_rot(g["rot"][sl])
        cov = R @ torch.diag_embed(s2f) @ R.transpose(1, 2)
        zc = z.clamp(min=near)
        J = torch.zeros(len(p), 2, 3, device=dev)
        J[:, 0, 0] = fx / zc; J[:, 0, 2] = -fx * p[:, 0] / zc ** 2
        J[:, 1, 1] = fy / zc; J[:, 1, 2] = -fy * p[:, 1] / zc ** 2
        T = J @ w2c[:3, :3]
        cov2 = T @ cov @ T.transpose(1, 2) + 0.3 * torch.eye(2, device=dev)
        det = cov2[:, 0, 0] * cov2[:, 1, 1] - cov2[:, 0, 1] ** 2
        conic = torch.stack([cov2[:, 1, 1], -cov2[:, 0, 1], cov2[:, 0, 0]], 1) / det.clamp(min=1e-12)[:, None]
        mu = torch.stack([fx * p[:, 0] / zc + cx, fy * p[:, 1] / zc + cy], 1)
        lam = 0.5 * (cov2[:, 0, 0] + cov2[:, 1, 1]) + torch.sqrt((0.25 * (cov2[:, 0, 0] - cov2[:, 1, 1]) ** 2
                                                                  + cov2[:, 0, 1] ** 2).clamp(min=0))
        rad = 3 * torch.sqrt(lam)
        vis = ((z > near) & (det > 0) & (op > 1 / 255) & (mu[:, 0] + rad > 0) & (mu[:, 0] - rad < W)
               & (mu[:, 1] + rad > 0) & (mu[:, 1] - rad < H) & (rad < 2 * max(W, H)))
        idx = torch.nonzero(vis)[:, 0]
        col = eval_sh(g["deg"], g["sh"][sl][idx], torch.nn.functional.normalize(xyz[idx] - c2w[:3, 3], dim=1))
        outs.append((mu[idx], conic[idx], op[idx], col, z[idx], rad[idx]))
    mu, con, op, col, z, rad = (torch.cat(t) for t in zip(*outs))
    o = torch.argsort(z)
    return {"mu": mu[o], "conic": con[o], "op": op[o], "col": col[o], "z": z[o], "rad": rad[o]}


@torch.no_grad()
def rasterize(P, W, H, tile=32, chunk=2048):
    """Front-to-back alpha compositing per tile. -> premultiplied rgb [H,W,3], alpha [H,W], expected depth [H,W]."""
    dev = P["mu"].device
    rgb = torch.zeros(H, W, 3, device=dev)
    alpha = torch.zeros(H, W, device=dev)
    dep = torch.zeros(H, W, device=dev)
    if len(P["mu"]) == 0:
        return rgb, alpha, dep
    lo, hi = P["mu"] - P["rad"][:, None], P["mu"] + P["rad"][:, None]
    feat = torch.cat([P["col"], P["z"][:, None]], 1)
    for y0 in range(0, H, tile):
        for x0 in range(0, W, tile):
            x1, y1 = min(x0 + tile, W), min(y0 + tile, H)
            sel = torch.nonzero((hi[:, 0] > x0) & (lo[:, 0] < x1) & (hi[:, 1] > y0) & (lo[:, 1] < y1))[:, 0]
            if len(sel) == 0:
                continue
            ys, xs = torch.meshgrid(torch.arange(y0, y1, device=dev) + 0.5, torch.arange(x0, x1, device=dev) + 0.5,
                                    indexing="ij")
            pix = torch.stack([xs.reshape(-1), ys.reshape(-1)], 1)
            Tr = torch.ones(len(pix), device=dev)
            acc = torch.zeros(len(pix), 4, device=dev)
            for c in range(0, len(sel), chunk):
                ii = sel[c:c + chunk]
                d = pix[:, None, :] - P["mu"][ii][None]
                cn = P["conic"][ii]
                pw = -0.5 * (cn[:, 0] * d[..., 0] ** 2 + cn[:, 2] * d[..., 1] ** 2) - cn[:, 1] * d[..., 0] * d[..., 1]
                a = torch.clamp(P["op"][ii] * torch.exp(pw), max=0.99)
                a = torch.where((pw > 0) | (a < 1 / 255), torch.zeros_like(a), a)
                Tc = Tr[:, None] * torch.cumprod(torch.cat([torch.ones_like(a[:, :1]), 1 - a[:, :-1]], 1), 1)
                acc += (a * Tc) @ feat[ii]
                Tr = Tr * torch.prod(1 - a, 1)
                if float(Tr.max()) < 1e-4:
                    break
            h, w = y1 - y0, x1 - x0
            rgb[y0:y1, x0:x1] = acc[:, :3].view(h, w, 3)
            alpha[y0:y1, x0:x1] = (1 - Tr).view(h, w)
            dep[y0:y1, x0:x1] = acc[:, 3].view(h, w)
    return rgb, alpha, dep / alpha.clamp(min=1e-6)


# ---------------------------------------------------------------------------------------------------- deformation
def quat_mat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def mat_quat(R):
    import scipy.spatial.transform as tr
    x, y, z, w = tr.Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def quat_mul(a, b):
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


class ObjectDeformer:
    def __init__(self, obj_g, bind, traj, fr: pio.SimFrame):
        self.g0, self.fr, self.bind, self.tr = obj_g, fr, bind, traj
        self.dev = obj_g["xyz"].device
        self.P0 = fr.to_sim(t2n(obj_g["xyz"]).astype(np.float64))
        self.Wt = np.asarray(bind["weights"], np.float64)
        if bind["mode"] == "flex":
            self.nodes = np.asarray(bind["nodes"])
            self.n0 = traj["flex"][0]

    def at(self, f):
        g = dict(self.g0)
        if self.bind["mode"] == "flex":
            disp = (self.tr["flex"][f] - self.n0)[self.nodes]                  # [N,4,3]
            P = self.P0 + (self.Wt[..., None] * disp).sum(1)
            g["xyz"] = n2t(self.fr.to_world(P), self.dev)
            return g
        pos, quat = self.tr["body_pos"], self.tr["body_quat"]
        P = np.zeros_like(self.P0)
        Rs = []
        for b in range(pos.shape[1]):
            Rb = quat_mat(quat[f, b]) @ quat_mat(quat[0, b]).T
            P += self.Wt[:, b:b + 1] * ((self.P0 - pos[0, b]) @ Rb.T + pos[f, b])
            Rs.append(Rb)
        g["xyz"] = n2t(self.fr.to_world(P), self.dev)
        Rf = self.fr.R                                                         # sim rotation -> world rotation
        qb = n2t(np.stack([mat_quat(Rf.T @ Rb @ Rf) for Rb in Rs]), self.dev)
        dom = torch.tensor(self.Wt.argmax(1).tolist(), device=self.dev)
        g["rot"] = quat_mul(qb[dom], torch.nn.functional.normalize(self.g0["rot"], dim=1))
        return g


# ---------------------------------------------------------------------------------------------------- main
def main():
    ap = ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dataset", default="LERF")
    ap.add_argument("--scale", type=float, default=1.0, help="resolution scale of the capture camera")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--min-opacity", type=float, default=0.02, help="background Gaussians below are skipped")
    args = ap.parse_args()
    import cv2
    import mujoco
    run = pio.resolve_run_dir(args.run, args.dataset)
    out = pio.out_dir(run)
    st = json.loads((out / "sim_transform.json").read_text(encoding="utf-8"))
    rob = json.loads((out / "robot.json").read_text(encoding="utf-8"))
    bind = json.loads((out / "object_gs_binding.json").read_text(encoding="utf-8"))
    tz = np.load(out / "trajectory.npz", allow_pickle=False)
    traj = {k: tz[k] for k in tz.files}
    cam = rob["cameras"].get("cam_gs")
    if cam is None:
        raise SystemExit("no capture camera in robot.json (step1 cameras.json missing at 16.1)")
    fr = pio.SimFrame.from_json(st)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    W, H = int(round(cam["width"] * args.scale)), int(round(cam["height"] * args.scale))
    K = [cam["fx"] * args.scale, cam["fy"] * args.scale, W / 2, H / 2]
    c2w = np.array(cam["c2w_cv_world"])
    t0 = time.time()

    bg = load_ply(run / st["inputs"]["background_gs"], device="cpu")
    keep = torch.nonzero(bg["opacity"] > args.min_opacity)[:, 0]
    bg = {k: (v[keep].to(dev) if torch.is_tensor(v) else v) for k, v in bg.items()}
    Pb = project(bg, c2w, K, W, H)
    bg_rgb, bg_a, bg_d = rasterize(Pb, W, H)
    del bg, Pb
    print(f"[16.3] background {len(keep)} Gaussians rendered once ({W}x{H}, {dev}) in {time.time() - t0:.0f}s")

    obj = load_ply(run / bind["object_gs"], device=dev)
    deform = ObjectDeformer(obj, bind, traj, fr)

    m = mujoco.MjModel.from_xml_path(str(out / "mjcf" / "scene_simulation.xml"))
    d = mujoco.MjData(m)
    rend = mujoco.Renderer(m, height=H, width=W)
    opt_robot = mujoco.MjvOption()
    opt_robot.geomgroup[:] = 0
    opt_robot.geomgroup[1] = 1
    s = float(st["sim_scale"])

    vids = out / "videos"
    vids.mkdir(exist_ok=True)
    fps = int(st["config"].get("fps", 30))
    four = cv2.VideoWriter_fourcc(*"mp4v")
    w_gs = cv2.VideoWriter(str(vids / "render_3dgs_synced.mp4"), four, fps, (W, H))
    w_cmp = cv2.VideoWriter(str(vids / "comparison_mujoco_vs_3dgs.mp4"), four, fps, (2 * W, H))
    nF = len(traj["t"]) if args.max_frames is None else min(len(traj["t"]), args.max_frames)
    lift_frame = int(np.argmax(traj["qpos"][:, 2])) if len(traj["qpos"]) else 0
    white = torch.ones(3, device=dev)
    for f in range(nF):
        g = deform.at(f)
        o_rgb, o_a, o_d = rasterize(project(g, c2w, K, W, H), W, H)
        front = (o_d < bg_d) | (bg_a < 0.05)
        over = torch.where(front[..., None], o_rgb + (1 - o_a[..., None]) * bg_rgb, bg_rgb + (1 - bg_a[..., None]) * o_rgb)
        a = 1 - (1 - o_a) * (1 - bg_a)
        img = (over + (1 - a[..., None]) * white).clamp(0, 1)
        depth = t2n(torch.where(front, torch.where(o_a > 0.3, o_d, bg_d), bg_d))
        frame = (t2n(img)[..., ::-1] * 255).astype(np.uint8).copy()
        # robot layer from MuJoCo (same camera), depth test against the splats
        d.qpos[:] = traj["qpos"][f]
        mujoco.mj_forward(m, d)
        rend.update_scene(d, camera="cam_gs", scene_option=opt_robot)
        rrgb = rend.render()
        rend.enable_segmentation_rendering()
        seg = rend.render()[..., 0]
        rend.disable_segmentation_rendering()
        rend.enable_depth_rendering()
        rdep = rend.render() / s
        rend.disable_depth_rendering()
        msk = (seg >= 0) & (rdep < depth + 1e-3)
        frame[msk] = rrgb[msk][:, ::-1]
        cv2.putText(frame, f"3DGS  t={traj['t'][f]:.2f}s  {traj['phase'][f]}", (16, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (40, 40, 40), 2, cv2.LINE_AA)
        w_gs.write(frame)
        rend.update_scene(d, camera="cam_gs")
        mj = cv2.cvtColor(rend.render(), cv2.COLOR_RGB2BGR)
        cv2.putText(mj, "MuJoCo (same camera)", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2, cv2.LINE_AA)
        w_cmp.write(np.hstack([mj, frame]))
        if f == 0:
            cv2.imwrite(str(vids / "frame_3dgs_first.png"), frame)
        if f == min(lift_frame, nF - 1) or f == nF - 1:
            cv2.imwrite(str(vids / "frame_3dgs_lift.png"), frame)
        if f % 20 == 0:
            print(f"  frame {f + 1}/{nF} ({(time.time() - t0):.0f}s)")
    w_gs.release(); w_cmp.release(); rend.close()
    print(f"[16.3] {nF} frames in {time.time() - t0:.0f}s -> {vids}")


if __name__ == "__main__":
    main()
