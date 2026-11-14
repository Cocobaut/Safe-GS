"""Small pure-PyTorch Gaussian-splatting renderer for ONE object (a few thousand Gaussians), used by step 14.

Same model as the CoMe ply (3DGS + Mip-Splatting 3D filter): SH degree 3 colour, quaternion (w, x, y, z) rotation,
exp scales, sigmoid opacity, `filter_3D` applied as in Mip-Splatting (scale^2 + f^2, opacity * sqrt(det/det_f)),
EWA projection with the 3DGS 0.3 px low-pass, front-to-back alpha compositing (alpha <= 0.99, alpha < 1/255 skipped).
Not the CoMe CUDA rasterizer (ray-based evaluation, StopThePop sorting): for an isolated object the difference is small
and this runs anywhere torch runs (validated against the real photos in step 14.1).

Cameras: OpenCV convention (x right, y down, z forward), c2w 4x4, pixel intrinsics.
"""
from __future__ import annotations

import numpy as np
import torch
from plyfile import PlyData

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658,
      1.445305721320277, -0.5900435899266435)


def load_ply(path, device="cpu"):
    v = PlyData.read(str(path)).elements[0]
    names = [p.name for p in v.properties]
    get = lambda k: np.asarray(v[k], np.float32)
    n_rest = len([k for k in names if k.startswith("f_rest_")])
    deg = int(round(np.sqrt(n_rest / 3 + 1))) - 1
    dc = np.stack([get(f"f_dc_{i}") for i in range(3)], 1)[:, :, None]
    rest = np.stack([get(f"f_rest_{i}") for i in range(n_rest)], 1).reshape(len(dc), 3, -1) if n_rest else np.zeros((len(dc), 3, 0), np.float32)
    g = {"xyz": np.stack([get("x"), get("y"), get("z")], 1),
         "sh": np.concatenate([dc, rest], 2), "deg": deg,
         "opacity": 1 / (1 + np.exp(-get("opacity"))),
         "scale": np.exp(np.stack([get(f"scale_{i}") for i in range(3)], 1)),
         "rot": np.stack([get(f"rot_{i}") for i in range(4)], 1),
         "filter3d": get("filter_3D") if "filter_3D" in names else np.zeros(len(dc), np.float32)}
    t = {k: (torch.from_numpy(x).to(device) if isinstance(x, np.ndarray) else x) for k, x in g.items()}
    return t


def quat_to_rot(q):
    q = torch.nn.functional.normalize(q, dim=1)
    w, x, y, z = q.unbind(1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).view(-1, 3, 3)


def eval_sh(deg, sh, d):
    x, y, z = d.unbind(1)
    r = C0 * sh[..., 0]
    if deg > 0:
        r = r - C1 * y[:, None] * sh[..., 1] + C1 * z[:, None] * sh[..., 2] - C1 * x[:, None] * sh[..., 3]
    if deg > 1:
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        r = (r + C2[0] * xy[:, None] * sh[..., 4] + C2[1] * yz[:, None] * sh[..., 5]
             + C2[2] * (2 * zz - xx - yy)[:, None] * sh[..., 6] + C2[3] * xz[:, None] * sh[..., 7]
             + C2[4] * (xx - yy)[:, None] * sh[..., 8])
        if deg > 2:
            r = (r + C3[0] * (y * (3 * xx - yy))[:, None] * sh[..., 9] + C3[1] * (xy * z)[:, None] * sh[..., 10]
                 + C3[2] * (y * (4 * zz - xx - yy))[:, None] * sh[..., 11]
                 + C3[3] * (z * (2 * zz - 3 * xx - 3 * yy))[:, None] * sh[..., 12]
                 + C3[4] * (x * (4 * zz - xx - yy))[:, None] * sh[..., 13] + C3[5] * (z * (xx - yy))[:, None] * sh[..., 14]
                 + C3[6] * (x * (xx - 3 * yy))[:, None] * sh[..., 15])
    return torch.clamp(r + 0.5, min=0.0)


@torch.no_grad()
def render(g, c2w_cv, K, W, H, bg=(1.0, 1.0, 1.0), features=None, chunk=8192, near=0.01):
    """-> dict(rgb [H,W,3], alpha [H,W], feat [H,W,C] (composited per-Gaussian features), contrib [N] (sum of
    compositing weights per Gaussian = how many pixels it effectively covers in this view))."""
    dev = g["xyz"].device
    c2w = torch.as_tensor(c2w_cv, dtype=torch.float32, device=dev)
    w2c = torch.linalg.inv(c2w)
    fx, fy, cx, cy = [float(k) for k in K]
    p = g["xyz"] @ w2c[:3, :3].T + w2c[:3, 3]
    z = p[:, 2]
    ok = z > near
    # 3D filter (Mip-Splatting)
    s2 = g["scale"] ** 2
    s2f = s2 + g["filter3d"][:, None] ** 2
    op = g["opacity"] * torch.sqrt(s2.prod(1) / s2f.prod(1))
    R = quat_to_rot(g["rot"])
    cov = R @ torch.diag_embed(s2f) @ R.transpose(1, 2)
    zc = z.clamp(min=near)
    J = torch.zeros(len(p), 2, 3, device=dev)
    J[:, 0, 0] = fx / zc; J[:, 0, 2] = -fx * p[:, 0] / zc ** 2
    J[:, 1, 1] = fy / zc; J[:, 1, 2] = -fy * p[:, 1] / zc ** 2
    T = J @ w2c[:3, :3]
    cov2 = T @ cov @ T.transpose(1, 2) + 0.3 * torch.eye(2, device=dev)
    det = cov2[:, 0, 0] * cov2[:, 1, 1] - cov2[:, 0, 1] ** 2
    conic = torch.stack([cov2[:, 1, 1] / det, -cov2[:, 0, 1] / det, cov2[:, 0, 0] / det], 1)
    mu = torch.stack([fx * p[:, 0] / zc + cx, fy * p[:, 1] / zc + cy], 1)
    campos = c2w[:3, 3]
    col = eval_sh(g["deg"], g["sh"], torch.nn.functional.normalize(g["xyz"] - campos, dim=1))
    feats = col if features is None else torch.cat([col, features], 1)
    # radius cull (3 sigma) + depth sort
    lam = 0.5 * (cov2[:, 0, 0] + cov2[:, 1, 1]) + torch.sqrt((0.25 * (cov2[:, 0, 0] - cov2[:, 1, 1]) ** 2 + cov2[:, 0, 1] ** 2).clamp(min=0))
    rad = 3 * torch.sqrt(lam)
    vis = ok & (mu[:, 0] + rad > 0) & (mu[:, 0] - rad < W) & (mu[:, 1] + rad > 0) & (mu[:, 1] - rad < H) & (det > 0)
    idx = torch.nonzero(vis)[:, 0]
    idx = idx[torch.argsort(z[idx])]
    mu_s, con_s, op_s, f_s = mu[idx], conic[idx], op[idx], feats[idx]
    ys, xs = torch.meshgrid(torch.arange(H, device=dev) + 0.5, torch.arange(W, device=dev) + 0.5, indexing="ij")
    pix = torch.stack([xs.reshape(-1), ys.reshape(-1)], 1)
    out_f = torch.zeros(len(pix), feats.shape[1], device=dev)
    out_a = torch.zeros(len(pix), device=dev)
    contrib = torch.zeros(len(p), device=dev)
    for i in range(0, len(pix), chunk):
        d = pix[i:i + chunk, None, :] - mu_s[None]                         # [P, n, 2]
        power = -0.5 * (con_s[:, 0] * d[..., 0] ** 2 + con_s[:, 2] * d[..., 1] ** 2) - con_s[:, 1] * d[..., 0] * d[..., 1]
        a = torch.clamp(op_s * torch.exp(power), max=0.99)
        a = torch.where((power > 0) | (a < 1 / 255), torch.zeros_like(a), a)
        Tc = torch.cumprod(torch.cat([torch.ones_like(a[:, :1]), 1 - a[:, :-1]], 1), 1)
        w = a * Tc                                                          # compositing weights
        out_f[i:i + chunk] = w @ f_s
        out_a[i:i + chunk] = w.sum(1)
        contrib.index_add_(0, idx, w.sum(0))
    bgt = torch.tensor(bg, device=dev)
    rgb = out_f[:, :3] + (1 - out_a[:, None]) * bgt
    res = {"rgb": rgb.view(H, W, 3).clamp(0, 1), "alpha": out_a.view(H, W), "contrib": contrib}
    if features is not None:
        res["feat"] = out_f[:, 3:].view(H, W, -1)
    return res
