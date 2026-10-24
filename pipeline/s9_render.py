"""
S9: render video kết quả mô phỏng bằng Gaussian (chạy trong CoMe, cần CUDA).

Cảnh mỗi frame = nền đã lấp (S4b background_finetuned.ply) + Gaussian từng vật (S5) đã biến đổi theo pose MuJoCo (S8).
Pose S8 (p, q) ở hệ sim: s' = R(q) s + p. Hệ sim = W2S (world_to_sim.json, có scale): s = W2S x.
=> trong hệ thế giới: x' = M x, M = W2S^-1 D W2S (phần quay của M là phép quay thuần vì scale đều triệt tiêu).
Gaussian: tâm x -> M x; quaternion quay q_g -> q_M * q_g; scale, opacity, màu giữ nguyên (SH bậc cao không xoay, v1).
Camera: các frame trong config (đúng góc camera COLMAP). Video mp4 có chữ: kịch bản, thời gian, frame.
Checkpoint: video nào đã có thì bỏ qua.
Xuất: <kịch bản>/<camera>.mp4, <kịch bản>/<camera>_keyframes.jpg
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from common import log, make_parser, load_scene, view_name, gaussians_from_ply, render_rgbd, to_u8, load_cfg
from arguments import get_combined_args
from gaussian_renderer import GaussianModel


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def R_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if i == 1:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])


def quat_mul(a, b):
    """a, b: (..., 4) w,x,y,z (torch)."""
    w1, x1, y1, z1 = a.unbind(-1); w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def concat(parts, sh_degree):
    g = GaussianModel(sh_degree)
    g._xyz = torch.cat([p["xyz"] for p in parts])
    g._features_dc = torch.cat([p["dc"] for p in parts])
    g._features_rest = torch.cat([p["rest"] for p in parts])
    g._opacity = torch.cat([p["opa"] for p in parts])
    g._scaling = torch.cat([p["scl"] for p in parts])
    g._rotation = torch.cat([p["rot"] for p in parts])
    g._confidence = torch.cat([p["conf"] for p in parts])
    g.filter_3D = torch.cat([p["f3d"] for p in parts])
    g.active_sh_degree = sh_degree
    return g


def pack(g):
    return {"xyz": g._xyz.detach(), "dc": g._features_dc.detach(), "rest": g._features_rest.detach(),
            "opa": g._opacity.detach(), "scl": g._scaling.detach(), "rot": torch.nn.functional.normalize(g._rotation.detach(), dim=-1),
            "conf": g._confidence.detach(), "f3d": g.filter_3D.detach()}


def main():
    p, model, pipe, ss, _, _ = make_parser("S9 render")
    for k in ("s4b", "s5", "s6", "s8"):
        p.add_argument(f"--{k}", type=str, required=True)
    args = get_combined_args(p)
    cfg_all = load_cfg(args.cfg); cfg = cfg_all["S9_render"]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    S = load_scene(args, model, pipe, ss)
    sh = S["dataset"].sh_degree
    W2S = np.array(json.loads((Path(args.s6) / "world_to_sim.json").read_text())["T"])
    S2W = np.linalg.inv(W2S)
    bg = pack(gaussians_from_ply(Path(args.s4b) / "background_finetuned.ply", sh, S["train"]))
    objs = {}
    for od in sorted((Path(args.s5) / "objects").iterdir()):
        f = od / "gaussians.ply"
        if f.exists() and f.stat().st_size > 1000:
            objs[od.name] = pack(gaussians_from_ply(f, sh, S["train"]))
    cams = [v for v in S["views"] if view_name(v) in cfg["cameras"]]
    log(f"nền {bg['xyz'].shape[0]:,} Gaussian, vật: {list(objs)}, camera: {[view_name(v) for v in cams]}")

    for sc_dir in sorted(p_ for p_ in Path(args.s8).iterdir() if (p_ / "trajectory.npz").exists()):
        tr = np.load(sc_dir / "trajectory.npz")
        names = [str(n) for n in tr["names"]]
        t, P, Q = tr["t"], tr["pos"], tr["quat"]
        od = out / sc_dir.name; od.mkdir(exist_ok=True)
        for cam in cams:
            mp4 = od / f"{view_name(cam)}.mp4"
            if mp4.exists():
                continue
            H, W = int(cam.image_height), int(cam.image_width)
            vw = cv2.VideoWriter(str(mp4.with_suffix(".tmp.mp4")), cv2.VideoWriter_fourcc(*"mp4v"), cfg["fps"], (W, H))
            keys = []
            kidx = set(np.linspace(0, len(t) - 1, 5).astype(int).tolist())
            for fi in range(len(t)):
                parts = [bg]
                for i, n in enumerate(names):
                    if n not in objs:
                        continue
                    D = np.eye(4); D[:3, :3] = quat_to_R(Q[fi, i]); D[:3, 3] = P[fi, i]
                    M = S2W @ D @ W2S
                    Rm = M[:3, :3]
                    o = objs[n]
                    Rt = torch.tensor(Rm, dtype=torch.float32, device="cuda")
                    tt = torch.tensor(M[:3, 3], dtype=torch.float32, device="cuda")
                    qm = torch.tensor(R_to_quat(Rm / np.cbrt(np.linalg.det(Rm))), dtype=torch.float32, device="cuda")
                    parts.append(dict(o, xyz=o["xyz"] @ Rt.T + tt, rot=quat_mul(qm.expand_as(o["rot"]), o["rot"])))
                g = concat(parts, sh)
                img = to_u8(render_rgbd(cam, g, S)[0])[..., ::-1].copy()
                txt = f"{sc_dir.name}  t={t[fi]:.2f}s  {fi + 1}/{len(t)}  ({view_name(cam)})"
                cv2.rectangle(img, (0, 0), (14 * len(txt), 36), (0, 0, 0), -1)
                cv2.putText(img, txt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                vw.write(img)
                if fi in kidx:
                    keys.append(cv2.resize(img, (W // 2, H // 2)))
            vw.release()
            mp4.with_suffix(".tmp.mp4").replace(mp4)
            cv2.imwrite(str(od / f"{view_name(cam)}_keyframes.jpg"), np.hstack(keys), [cv2.IMWRITE_JPEG_QUALITY, 85])
            log(f"{sc_dir.name} / {view_name(cam)}: {len(t)} frame -> {mp4}")
    log(f"S9 xong -> {out}")


if __name__ == "__main__":
    main()
