"""
S10 (phần GPU, chạy trong CoMe): đo lường S2, S3, S4, S9.

A. Căn COLMAP của LERF-Mask (Gaussian Grouping) về hệ của mình: Umeyama (đồng dạng) trên tâm camera của các frame
   chung tên -> dựng camera cho 3 ảnh test mới (test_0..2) và kiểm tra bằng PSNR render tại các frame chung.
B. S3 trên benchmark LERF-Mask (teatime, 10 câu prompt, mask GT ở test_0 và test_2):
   gán Gaussian theo nhóm ID (config S10_eval.lerf_prompts) bằng vote đóng góp render như S3 -> render riêng nhóm
   (màu 1, nền đen = opacity tích luỹ) -> mask dự đoán = alpha > 0.5 -> IoU và Boundary IoU đúng code
   script/eval_lerf_mask.py của Gaussian Grouping (dilation_ratio 0.02, mean theo prompt).
C. S2 NVS: PSNR / SSIM / LPIPS(vgg) trên 3 ảnh test LERF (góc nhìn mới) và 23 ảnh test của mình (--eval, 1/8).
D. S4 nền sau khi bỏ vật (không có GT nền thật nên đo gián tiếp), trên 23 view test:
   - giữ nguyên phần còn lại: PSNR / LPIPS ngoài vùng vật (mask vật nới 15 px) của cảnh nền vs ảnh GT,
     so với cảnh gốc CoMe (độ suy giảm);
   - "nền sạch": điểm 3D của depth nền trong vùng vật -> khoảng cách tới mặt bàn (cm); % pixel trong 1 cm / 2 cm;
   - độ nét: phương sai Laplacian trong vùng vật / ở vành bàn quanh vật (ảnh render nền), so cùng tỉ số trên ảnh GT.
E. S9 ghép lại: nền S4b + mọi vật S5 ở vị trí nghỉ, so ảnh GT (23 view test) -> mất mát do tách / ghép.
Xuất: eval_gpu.json, preview/lerf_<view>_<prompt>.jpg, preview/align_check.jpg
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from common import (log, make_parser, load_scene, view_name, freeze, read_ply, gaussians_from_ply, render_rgbd,
                    to_u8, load_cfg)
from arguments import get_combined_args
from gaussian_renderer import render
from scene.cameras import MiniCam
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, qvec2rotmat
from utils.image_utils import psnr
from utils.loss_utils import ssim
from utils.depth_utils import depths_to_points
from lpipsPyTorch import LPIPSEval
from s3_label import accumulate


# --------------------------------------------------------------------------- Gaussian Grouping eval_lerf_mask.py
def mask_to_boundary(mask, dilation_ratio=0.02):
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1: h + 1, 1: w + 1]
    return mask - mask_erode


def boundary_iou(gt, dt, dilation_ratio=0.02):
    dt = (dt > 128).astype("uint8"); gt = (gt > 128).astype("uint8")
    gb, db = mask_to_boundary(gt, dilation_ratio), mask_to_boundary(dt, dilation_ratio)
    inter = ((gb * db) > 0).sum(); union = ((gb + db) > 0).sum()
    return inter / union


def iou(gt, dt):
    g, d = gt > 128, dt > 128
    return np.logical_and(g, d).sum() / np.logical_or(g, d).sum()


# --------------------------------------------------------------------------- căn hai COLMAP
def centers(model_dir):
    ext = read_extrinsics_binary(str(Path(model_dir) / "images.bin"))
    intr = read_intrinsics_binary(str(Path(model_dir) / "cameras.bin"))
    out = {}
    for e in ext.values():
        R = qvec2rotmat(e.qvec); t = np.array(e.tvec)
        out[Path(e.name).stem] = (R, t, -R.T @ t, intr[e.camera_id])
    return out


def umeyama(src, dst):
    """dst ~ s R src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / (xs ** 2).sum(1).mean()
    return s, R, mu_d - s * R @ mu_s


def make_cam(Rc, tc, cam, s, R, t, znear=0.01, zfar=100.0):
    """Camera LERF (w2c Rc, tc; PINHOLE fx fy cx cy) đưa về hệ của mình; ma trận chiếu có điểm chính lệch tâm."""
    W, H = int(cam.width), int(cam.height)
    p = cam.params
    fx, fy, cx, cy = (p[0], p[0], p[1], p[2]) if cam.model == "SIMPLE_PINHOLE" else (p[0], p[1], p[2], p[3])
    R2 = Rc @ R.T
    t2 = -R2 @ t + s * tc                         # nhân s cho toạ độ camera: không đổi phép chiếu
    W2V = np.eye(4); W2V[:3, :3] = R2; W2V[:3, 3] = t2
    P = np.zeros((4, 4))
    P[0, 0] = 2 * fx / W; P[0, 2] = 2 * cx / W - 1
    P[1, 1] = 2 * fy / H; P[1, 2] = 2 * cy / H - 1
    P[2, 2] = zfar / (zfar - znear); P[2, 3] = -(zfar * znear) / (zfar - znear); P[3, 2] = 1.0
    wvt = torch.tensor(W2V.T, dtype=torch.float32, device="cuda")
    proj = torch.tensor(P.T, dtype=torch.float32, device="cuda")
    fovx, fovy = 2 * np.arctan(W / (2 * fx)), 2 * np.arctan(H / (2 * fy))
    return MiniCam(W, H, fovy, fovx, znear, zfar, wvt, wvt @ proj)


def load_img(path, size=None):
    im = cv2.imread(str(path))[..., ::-1].astype(np.float32) / 255.0
    if size is not None and (im.shape[1], im.shape[0]) != size:
        im = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
    return torch.from_numpy(im).permute(2, 0, 1).contiguous().cuda()


def nvs(pred, gt, lp):
    return {"psnr": float(psnr(pred[None], gt[None]).mean()), "ssim": float(ssim(pred[None], gt[None])),
            "lpips": float(lp.criterion(pred[None], gt[None]))}


def main():
    p, model, pipe, ss, _, _ = make_parser("S10 eval gpu")
    for k in ("lerf", "s3", "s4b", "s5", "s4a", "labels"):
        p.add_argument(f"--{k}", type=str, required=True)
    p.add_argument("--n_input", type=int, required=True)
    args = get_combined_args(p)
    cfg_all = load_cfg(args.cfg); cfg = cfg_all["S10_eval"]
    out = Path(args.out); (out / "preview").mkdir(parents=True, exist_ok=True)
    S = load_scene(args, model, pipe, ss)
    g = S["g"]; freeze(g)
    lp = LPIPSEval(device="cuda", net_type="vgg")
    res = {}
    lerf = Path(args.lerf)
    black = torch.zeros(3, device="cuda")
    S_black = dict(S, bg=black)

    # ------------------------------------------------------------------ S0. thống kê COLMAP (như colmap model_analyzer)
    sp = Path(args.source_path) / "sparse" / "0"
    ext = read_extrinsics_binary(str(sp / "images.bin"))
    _, _, perr = read_points3D_binary(str(sp / "points3D.bin"))
    nobs = int(sum((e.point3D_ids != -1).sum() for e in ext.values()))
    res["colmap"] = {"registered": len(ext), "input_images": int(args.n_input),
                     "points3D": int(len(perr)), "observations": nobs,
                     "mean_track_length": nobs / len(perr), "mean_observations_per_image": nobs / len(ext),
                     "mean_reprojection_error_px": float(np.mean(perr))}
    log(f"COLMAP: {res['colmap']}")

    # ------------------------------------------------------------------ A. căn COLMAP
    ours = centers(Path(args.source_path) / "sparse" / "0")
    theirs = centers(lerf / "sparse" / "0")
    common = sorted(set(ours) & set(theirs))
    A = np.array([theirs[n][2] for n in common]); B = np.array([ours[n][2] for n in common])
    s, R, t = umeyama(A, B)
    resid = np.linalg.norm((s * (R @ A.T)).T + t - B, axis=1)
    mpu = json.loads((Path(args.s3) / "support.json").read_text())["meters_per_unit"]
    res["align"] = {"common_frames": len(common), "scale": float(s),
                    "center_residual_rms_cm": float(np.sqrt((resid ** 2).mean()) * mpu * 100),
                    "center_residual_max_cm": float(resid.max() * mpu * 100)}
    chk = []
    for n in common[::30]:
        Rc, tc, _, cam = theirs[n]
        c = make_cam(Rc, tc, cam, s, R, t)
        img = render_rgbd(c, g, S)[0]
        gt = load_img(lerf / "images" / f"{n}.jpg", (c.image_width, c.image_height))
        chk.append(float(psnr(img[None], gt[None]).mean()))
        if len(chk) == 1:
            cv2.imwrite(str(out / "preview" / "align_check.jpg"),
                        to_u8(torch.cat([img, torch.ones(3, img.shape[1], 8, device="cuda"), gt], 2))[..., ::-1])
    res["align"]["psnr_render_vs_lerf_frames"] = float(np.mean(chk))
    log(f"căn COLMAP: {res['align']}")
    test_cams = {}
    for n, (Rc, tc, _, cam) in theirs.items():
        if n.startswith("test_"):
            test_cams[n] = make_cam(Rc, tc, cam, s, R, t)

    # ------------------------------------------------------------------ C. NVS
    res["nvs_lerf_test"] = {n: nvs(render_rgbd(c, g, S)[0], load_img(lerf / "images" / f"{n}.jpg",
                                                                     (c.image_width, c.image_height)), lp)
                            for n, c in sorted(test_cams.items())}
    test_views = [v for v in S["scene"].getTestCameras()]
    gts = {view_name(v): v.original_image.cuda()[:3] for v in test_views}
    full = {view_name(v): render_rgbd(v, g, S)[0] for v in test_views}
    m = [nvs(full[k], gts[k], lp) for k in gts]
    res["nvs_ours_test"] = {k: float(np.mean([x[k] for x in m])) for k in ("psnr", "ssim", "lpips")}
    log(f"NVS: {res['nvs_lerf_test']} | ours {res['nvs_ours_test']}")

    # ------------------------------------------------------------------ B. LERF-Mask
    prompts = cfg["lerf_prompts"]
    names = [k for k, v in prompts.items() if v]
    labels_dir = Path(args.labels)
    gid = [torch.tensor(prompts[n], device="cuda") for n in names]

    def gm(view):
        f = labels_dir / f"{view_name(view)}.png"
        if not f.exists():
            return None
        lab = torch.from_numpy(cv2.imread(str(f), cv2.IMREAD_UNCHANGED).astype(np.int64)).cuda()
        return torch.stack([torch.isin(lab, t_).float() for t_ in gid] + [torch.ones_like(lab, dtype=torch.float32)], 0)

    W_ = accumulate(S, g, S["views"], gm, len(names) + 1, out / "partial_lerf.pt")
    ratio = (W_[:, :-1] / W_[:, -1:].clamp(min=1e-12)).cpu().numpy()
    tot = W_[:, -1].cpu().numpy()
    lab = np.where((ratio.max(1) >= cfg_all["S3_label"]["assign_ratio"]) & (tot > 1e-4), ratio.argmax(1), -1)
    ious, bious, per = {}, {}, {}
    for vdir in sorted((lerf / "test_mask").iterdir()):
        cam = test_cams.get(f"test_{vdir.name}")
        if cam is None:
            continue
        base = to_u8(render_rgbd(cam, g, S)[0])[..., ::-1]
        for f in sorted(vdir.glob("*.png")):
            pr = f.stem
            gt = np.array(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE))
            if pr in names:
                sel = torch.from_numpy(lab == names.index(pr)).cuda()
                if sel.any():
                    a = render(cam, g, S["pipe"], black, splat_args=S["splat_args"],
                               override_color=sel[:, None].float().expand(-1, 3).contiguous())["render"][0]
                    pred = ((a > 0.5).cpu().numpy() * 255).astype(np.uint8)
                else:
                    pred = np.zeros_like(gt)
            else:
                pred = np.zeros_like(gt)                 # prompt không có nhóm ID tương ứng -> dự đoán rỗng
            if pred.shape != gt.shape:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            i, b = float(iou(gt, pred)), float(boundary_iou(gt, pred))
            ious.setdefault(pr, []).append(i); bious.setdefault(pr, []).append(b)
            per[f"test_{vdir.name}/{pr}"] = {"iou": i, "biou": b}
            vis = base.copy()
            vis[gt > 128] = (0.5 * vis[gt > 128] + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
            vis[pred > 128] = (0.5 * vis[pred > 128] + np.array([0, 0, 255]) * 0.5).astype(np.uint8)
            cv2.putText(vis, f"{pr}: IoU {i:.3f} (xanh=GT, do=du doan)", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
            cv2.imwrite(str(out / "preview" / f"lerf_test{vdir.name}_{pr.replace(' ', '_')}.jpg"), vis,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
    miou = {k: float(np.mean(v)) for k, v in ious.items()}
    mbiou = {k: float(np.mean(v)) for k, v in bious.items()}
    sup = [k for k in miou if k in names]
    res["lerf_mask"] = {"mIoU_all": float(np.mean(list(miou.values()))), "mBIoU_all": float(np.mean(list(mbiou.values()))),
                        "mIoU_supported": float(np.mean([miou[k] for k in sup])),
                        "mBIoU_supported": float(np.mean([mbiou[k] for k in sup])),
                        "unsupported_prompts": [k for k in miou if k not in names],
                        "per_prompt_iou": miou, "per_prompt_biou": mbiou, "per_view": per}
    log(f"LERF-Mask: mIoU {res['lerf_mask']['mIoU_all']:.3f} mBIoU {res['lerf_mask']['mBIoU_all']:.3f}")

    # ------------------------------------------------------------------ D. S4 nền + E. S9 ghép lại
    sh = S["dataset"].sh_degree
    g_bg = gaussians_from_ply(Path(args.s4b) / "background_finetuned.ply", sh, S["train"])
    sup_ = json.loads((Path(args.s3) / "support.json").read_text())["table"]
    n_t, d_t = np.array(sup_["normal"]), sup_["d"]
    groups = json.loads((Path(args.s3) / "groups.json").read_text())
    obj_ids = sorted({i for o in groups["objects"].values() for i in o["ids"]})
    # vết chân vật trên mặt bàn: chiếu Gaussian vật xuống mặt bàn, lưới 1 cm, nới 1 ô
    ob = read_ply(Path(args.s3) / "gaussians" / "objects_all.ply")
    oxyz = np.stack([ob["x"], ob["y"], ob["z"]], 1)
    a0 = np.array([1.0, 0, 0]) if abs(n_t[0]) < 0.9 else np.array([0, 1.0, 0])
    u_ = np.cross(n_t, a0); u_ /= np.linalg.norm(u_); v_ = np.cross(n_t, u_)
    p0 = -d_t * n_t
    cell = 0.01 / mpu
    oa, obb = np.floor((oxyz - p0) @ u_ / cell).astype(int), np.floor((oxyz - p0) @ v_ / cell).astype(int)
    amin, bmin = oa.min() - 2, obb.min() - 2
    foot = np.zeros((obb.max() - bmin + 3, oa.max() - amin + 3), np.uint8)
    foot[obb - bmin, oa - amin] = 1
    foot = cv2.dilate(foot, np.ones((3, 3), np.uint8)) > 0

    def footprint_pixels(v):
        """Pixel mà tia nhìn cắt mặt bàn tại vết chân vật (chỗ vật vừa đứng)."""
        H, W = int(v.image_height), int(v.image_width)
        P1 = depths_to_points(v, torch.ones(1, H, W, device="cuda")).cpu().numpy()
        o = v.camera_center.cpu().numpy()
        dirs = P1 - o
        tt = -(o @ n_t + d_t) / (dirs @ n_t)
        X = o + tt[:, None] * dirs
        ia = np.floor((X - p0) @ u_ / cell).astype(int) - amin
        ib = np.floor((X - p0) @ v_ / cell).astype(int) - bmin
        okk = (tt > 0) & (ia >= 0) & (ia < foot.shape[1]) & (ib >= 0) & (ib < foot.shape[0])
        fp = np.zeros(H * W, bool)
        fp[okk] = foot[ib[okk], ia[okk]]
        return fp.reshape(H, W)

    out_psnr_bg, out_psnr_full, out_lp_bg, out_lp_full, dist_in, sharp_r, sharp_r_gt = [], [], [], [], [], [], []
    k15 = np.ones((31, 31), np.uint8)
    for v in test_views:
        nm = view_name(v)
        f = labels_dir / f"{nm}.png"
        if not f.exists():
            continue
        objm = np.isin(cv2.imread(str(f), cv2.IMREAD_UNCHANGED), obj_ids).astype(np.uint8)
        if objm.sum() < 50:
            continue
        dil = cv2.dilate(objm, k15) > 0
        fpm = footprint_pixels(v)
        if fpm.sum() < 50:
            continue
        ring = (cv2.dilate(fpm.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0) & ~fpm & ~dil
        rgb_bg, dep_bg = render_rgbd(v, g_bg, S)
        gt = gts[nm]
        keep = torch.from_numpy(~dil).cuda()
        for img, lst, lpl in ((rgb_bg, out_psnr_bg, out_lp_bg), (full[nm], out_psnr_full, out_lp_full)):
            mse = ((img - gt) ** 2)[:, keep].mean()
            lst.append(float(10 * torch.log10(1 / mse)))
            lpl.append(float(lp.criterion((img * keep)[None], (gt * keep)[None])))
        P3 = depths_to_points(v, dep_bg[None]).reshape(dep_bg.shape[0], dep_bg.shape[1], 3).cpu().numpy()
        inside = fpm
        dist_in.append(np.abs(P3[inside] @ n_t + d_t) * mpu * 100)
        lap = lambda im: cv2.Laplacian(cv2.cvtColor((im * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY), cv2.CV_64F)
        Lb = lap(rgb_bg.permute(1, 2, 0).cpu().numpy()); Lg = lap(gt.permute(1, 2, 0).cpu().numpy())
        if ring.sum() > 100:
            sharp_r.append(Lb[inside].var() / max(Lb[ring].var(), 1e-9))
            sharp_r_gt.append(Lg[inside].var() / max(Lg[ring].var(), 1e-9))
    dist_in = np.concatenate(dist_in) if dist_in else np.zeros(1)
    res["s4_background"] = {
        "views": len(out_psnr_bg),
        "psnr_outside_objects_background": float(np.mean(out_psnr_bg)),
        "psnr_outside_objects_full_scene": float(np.mean(out_psnr_full)),
        "lpips_outside_objects_background": float(np.mean(out_lp_bg)),
        "lpips_outside_objects_full_scene": float(np.mean(out_lp_full)),
        "hole_dist_to_table_cm_median": float(np.median(dist_in)),
        "hole_pct_within_1cm": float((dist_in < 1).mean() * 100),
        "hole_pct_within_2cm": float((dist_in < 2).mean() * 100),
        "hole_sharpness_ratio_render": float(np.mean(sharp_r)) if sharp_r else None,
        "hole_sharpness_ratio_gt_reference": float(np.mean(sharp_r_gt)) if sharp_r_gt else None}
    log(f"S4: {res['s4_background']}")

    parts = [g_bg] + [gaussians_from_ply(f, sh, S["train"]) for f in sorted((Path(args.s5) / "objects").glob("*/gaussians.ply"))
                      if f.stat().st_size > 1000]
    from s9_render import pack, concat
    g_all = concat([pack(x) for x in parts], sh)
    m = [nvs(render_rgbd(v, g_all, S)[0], gts[view_name(v)], lp) for v in test_views]
    res["s9_recomposed_test"] = {k: float(np.mean([x[k] for x in m])) for k in ("psnr", "ssim", "lpips")}
    res["s9_recomposed_test"]["gaussians"] = int(g_all.get_xyz.shape[0])
    log(f"S9 ghép lại: {res['s9_recomposed_test']}")

    (out / "eval_gpu.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    for f in out.glob("partial_*.pt"):
        f.unlink()
    log(f"S10 GPU xong -> {out}")


if __name__ == "__main__":
    main()
