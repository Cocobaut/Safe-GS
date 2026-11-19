"""
Render Gaussian CẢ cảnh theo quỹ đạo MuJoCo (tools/scene_lift.py --record), run v3 (nhãn SAM 3, khung chặt, lấp khuôn).

  nền   = S4d/e8 background_finetuned.ply (nhóm nền theo SAM 3, đã cắt vết chân + lấp bàn + tinh chỉnh);
  vật động (có trong quỹ đạo) = Gaussian của vật từ S3i filled.ply (đã ép khung + lấp khuôn đáy / mặt sau), biến đổi theo pose MuJoCo:
      M = S2W @ D(t) @ W2S, D = [R(quat), pos] của thân vật trong hệ bàn (mesh đặt đúng vị trí cảnh ở t = 0 nên D(0) ~ I);
  vật tĩnh không có trong quỹ đạo, không phải gấu / cừu (vd vũng trà) = giữ nguyên chỗ; gấu / cừu đã nằm trong nền.
Xuất cho mỗi quỹ đạo: <cam>.mp4, <cam>_keyframes.jpg (5 khung), below_<vật>.jpg: camera ảo NHÌN TỪ DƯỚI LÊN vật được nhấc ở đỉnh nhấc
(đáy / mặt khuất, nền thô trước khi lấp vs sau khi lấp).
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from common import log, make_parser, load_scene, view_name, read_ply, write_ply, gaussians_from_ply, render_rgbd, to_u8
from arguments import get_combined_args
from s9_render import concat, pack, quat_to_R, R_to_quat, quat_mul
from s3i_mold_fill import look_at_cam

CAMS = ["frame_00017", "frame_00065", "frame_00113"]
STATIC_SKIP = ("gau", "cuu")


def eval_test(args, S, sh, bg, objs, out):
    """Ảnh test cả cảnh cuối: nền (S4d) + vật S3i ở tư thế gốc (gấu / cừu đã nằm trong nền) so với ảnh thật; mốc đối chiếu = CoMe 30k."""
    from utils.loss_utils import ssim
    from utils.image_utils import psnr
    from lpipsPyTorch import LPIPSEval
    lp = LPIPSEval(device="cuda", net_type="vgg")
    od = out / "test_eval" / args.eval_tag
    od.mkdir(parents=True, exist_ok=True)
    final = concat([bg] + [o for n, o in objs.items() if n not in STATIC_SKIP], sh)
    refs = {"final": final, "come_30k": S["g"]}
    res = {"views": {}, "objects_in_scene": [n for n in objs if n not in STATIC_SKIP], "bg_gaussians": int(bg["xyz"].shape[0]),
           "total_gaussians": int(final._xyz.shape[0])}
    acc = {k: {"psnr": [], "ssim": [], "lpips": []} for k in refs}
    with torch.no_grad():
        for v in S["scene"].getTestCameras():
            nm = view_name(v)
            gt = v.original_image.cuda()[:3]
            res["views"][nm] = {}
            for k, g in refs.items():
                img = render_rgbd(v, g, S)[0]
                m = {"psnr": float(psnr(img[None], gt[None]).mean()), "ssim": float(ssim(img[None], gt[None])),
                     "lpips": float(lp.criterion(img[None], gt[None]))}
                for kk, vv in m.items():
                    acc[k][kk].append(vv)
                res["views"][nm][k] = m
                if k == "final":
                    cv2.imwrite(str(od / f"{nm}.png"), to_u8(img)[..., ::-1])
                    cv2.imwrite(str(od / f"{nm}_gt_vs_render.jpg"), np.hstack([to_u8(gt), to_u8(img)])[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
    res["n_test_views"] = len(res["views"])
    for k in refs:
        res[k] = {kk: float(np.mean(vv)) for kk, vv in acc[k].items()}
    (od / "metrics.json").write_text(json.dumps(res, indent=1))
    log(f"test {res['n_test_views']} view: final {res['final']} | CoMe {res['come_30k']}")


def main():
    p, model, pipe, ss, _, _ = make_parser("S9 v3 render")
    p.add_argument("--bg", required=True, help="S4d/e8/S4like/background_finetuned.ply")
    p.add_argument("--filled", required=True, help="thư mục S3i/<tag> (filled.ply, oid_filled.npy)")
    p.add_argument("--frames", required=True, help="S3m/<tag>/frame (chỉ để lấy danh sách vật theo thứ tự)")
    p.add_argument("--s3", required=True)
    p.add_argument("--lift", default="", help="thư mục lift (traj_lift_*/trajectory.npz, world_to_sim.json); rỗng = chỉ đo ảnh test")
    p.add_argument("--eval_tag", default="", help="đo PSNR / SSIM / LPIPS cả cảnh (nền + vật, tư thế t = 0) trên các view test, lưu vào <out>/test_eval/<tag>")
    args = get_combined_args(p)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    S = load_scene(args, model, pipe, ss)
    sh = S["dataset"].sh_degree
    s3 = Path(args.s3)
    groups = json.loads((s3 / "groups.json").read_text())["objects"]
    names_all = [n for n in groups if (Path(args.frames) / f"{n}.npz").exists()]
    el = read_ply(Path(args.filled) / "filled.ply")
    oid = np.load(Path(args.filled) / "oid_filled.npy")
    assert len(el) == len(oid), (len(el), len(oid))
    if args.lift:
        W2S = np.array(json.loads((Path(args.lift) / "world_to_sim.json").read_text())["T"])
        S2W = np.linalg.inv(W2S)
    objs = {}
    for i, n in enumerate(names_all):
        m = oid == i
        if m.sum() < 5:
            continue
        pth = out / f"_{n}.ply"
        write_ply(pth, el[m])
        objs[n] = pack(gaussians_from_ply(pth, sh, S["train"]))
        pth.unlink(missing_ok=True)
    bg = pack(gaussians_from_ply(Path(args.bg), sh, S["train"]))
    cams = [v for v in S["views"] if view_name(v) in CAMS]
    log(f"nền {bg['xyz'].shape[0]:,} Gaussian; vật: { {n: int(o['xyz'].shape[0]) for n, o in objs.items()} }")

    def place(o, M):
        Rm = M[:3, :3]
        Rt = torch.tensor(Rm, dtype=torch.float32, device="cuda")
        tt = torch.tensor(M[:3, 3], dtype=torch.float32, device="cuda")
        qm = torch.tensor(R_to_quat(Rm / np.cbrt(np.linalg.det(Rm))), dtype=torch.float32, device="cuda")
        return dict(o, xyz=o["xyz"] @ Rt.T + tt, rot=quat_mul(qm.expand_as(o["rot"]), o["rot"]))

    if args.eval_tag:
        eval_test(args, S, sh, bg, objs, out)
    for tdir in sorted(Path(args.lift).glob("traj_lift_*")) if args.lift else []:
        tr = np.load(tdir / "trajectory.npz")
        dyn = [str(n) for n in tr["names"]]
        t, P, Q = tr["t"], tr["pos"], tr["quat"]
        lifted = tdir.name[len("traj_lift_"):]
        od = out / tdir.name; od.mkdir(exist_ok=True)
        static = [n for n in objs if n not in dyn and n not in STATIC_SKIP]
        log(f"{tdir.name}: {len(t)} khung, vật động {dyn}, vật tĩnh thêm {static}")

        def scene_at(fi, with_fill=True):
            parts = [bg] + [objs[n] for n in static]
            for i, n in enumerate(dyn):
                if n not in objs:
                    continue
                D = np.eye(4); D[:3, :3] = quat_to_R(Q[fi, i]); D[:3, 3] = P[fi, i]
                parts.append(place(objs[n], S2W @ D @ W2S))
            return concat(parts, sh)

        for cam in cams:
            mp4 = od / f"{view_name(cam)}.mp4"
            H, W = int(cam.image_height), int(cam.image_width)
            vw = cv2.VideoWriter(str(mp4.with_suffix(".tmp.mp4")), cv2.VideoWriter_fourcc(*"mp4v"), 25, (W, H))
            keys, kidx = [], set(np.linspace(0, len(t) - 1, 5).astype(int).tolist())
            for fi in range(len(t)):
                img = to_u8(render_rgbd(cam, scene_at(fi), S)[0])[..., ::-1].copy()
                txt = f"lift {lifted}  t={t[fi]:.2f}s  ({view_name(cam)})"
                cv2.rectangle(img, (0, 0), (14 * len(txt), 36), (0, 0, 0), -1)
                cv2.putText(img, txt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                vw.write(img)
                if fi in kidx:
                    keys.append(cv2.resize(img, (W // 2, H // 2)))
            vw.release()
            mp4.with_suffix(".tmp.mp4").replace(mp4)
            cv2.imwrite(str(od / f"{view_name(cam)}_keyframes.jpg"), np.hstack(keys), [cv2.IMWRITE_JPEG_QUALITY, 85])
            log(f"{tdir.name} / {view_name(cam)}: {len(t)} khung")

        # nhìn từ dưới lên vật được nhấc, ở đỉnh nhấc: không lấp khuôn vs có lấp khuôn
        if lifted in dyn and lifted in objs:
            i = dyn.index(lifted)
            fi = int(np.argmax(P[:, i, 2]))
            up = S2W[:3, :3] @ np.array([0, 0, 1.0]); up /= np.linalg.norm(up)
            D = np.eye(4); D[:3, :3] = quat_to_R(Q[fi, i]); D[:3, 3] = P[fi, i]
            M = S2W @ D @ W2S
            cen = objs[lifted]["xyz"].mean(0).cpu().numpy()
            cen = M[:3, :3] @ cen + M[:3, 3]
            sub = scene_at(fi)
            rows = []
            for el_ in (-60, -30, 0):
                for az in (0, 120, 240):
                    a0 = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
                    e1 = np.cross(up, a0); e1 /= np.linalg.norm(e1); e2 = np.cross(up, e1)
                    dv = math.cos(math.radians(el_)) * (math.cos(math.radians(az)) * e1 + math.sin(math.radians(az)) * e2) + math.sin(math.radians(el_)) * up
                    cam_, _, _ = look_at_cam(cen + 0.5 * dv, cen, up, 384, 384, 40.0)
                    rows.append(to_u8(render_rgbd(cam_, sub, S)[0])[..., ::-1])
            img = np.vstack([np.hstack(rows[0:3]), np.hstack(rows[3:6]), np.hstack(rows[6:9])])
            cv2.imwrite(str(od / f"below_{lifted}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            log(f"{tdir.name}: ảnh nhìn từ dưới lên -> below_{lifted}.jpg (khung {fi}, cao {P[fi, i, 2] * 100:.1f} cm)")
    log(f"S9 v3 xong -> {out}")


if __name__ == "__main__":
    main()
