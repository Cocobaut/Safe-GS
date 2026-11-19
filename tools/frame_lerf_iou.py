"""
Do chinh xac 3D cua khung (S3m/<tag>/frame/<vat>.npz) bang LERF-Mask: chieu voxel khung cua MOI vat vao camera test LERF
(test_0, test_2; camera LERF dua ve he COLMAP cua minh bang Umeyama tren tam camera chung, nhu pipeline/s10_eval_gpu.py),
z-buffer giua cac vat (vat gan che vat xa), roi IoU silhouette cua tung vat voi mask GT cua prompt tuong ung.
Cung mot quy tac cho moi vat; khung khong co mat nen / ban nen phan bi ban che van tinh (gioi han, ghi trong RESULTS.md).

    .venv_da3/Scripts/python.exe tools/frame_lerf_iou.py --frames Data/teatime/pipeline/v6_adap/S3m/v6_adap/frame \
        --out Data/teatime/gaussian_inspect_v6/frame_lerf_iou_v6.json
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Based_Model" / "CoMe" / "scene"))
from colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat  # noqa: E402

PROMPT_OBJ = {"apple": "tao", "bag of cookies": "tui_banh", "coffee mug": "coc_su", "cookies on a plate": "banh_quy",
              "paper napkin": "khan_giay", "plate": "dia", "sheep": "cuu", "stuffed bear": "gau", "tea in a glass": "ly_tra"}


def centers(d):
    ext = read_extrinsics_binary(str(Path(d) / "images.bin"))
    intr = read_intrinsics_binary(str(Path(d) / "cameras.bin"))
    out = {}
    for e in ext.values():
        R = qvec2rotmat(e.qvec); t = np.array(e.tvec)
        out[Path(e.name).stem] = (R, t, -R.T @ t, intr[e.camera_id])
    return out


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / (xs ** 2).sum(1).mean()
    return s, R, mu_d - s * R @ mu_s


def render_ids(objs, R2, t2, fx, fy, cx, cy, W, H, dev):
    """objs: list of (xyz [N,3], voxel). Tra anh nhan (0 = trong, k+1 = vat k) bang splat vuong + z-buffer."""
    best = torch.full((H * W,), np.iinfo(np.int64).max, dtype=torch.int64, device=dev)
    for k, (X, vox) in enumerate(objs):
        P = torch.tensor(X, dtype=torch.float64, device=dev) @ torch.tensor(R2.T, device=dev) + torch.tensor(t2, device=dev)
        z = P[:, 2]
        ok = z > 1e-3
        P, z = P[ok], z[ok]
        u = fx * P[:, 0] / z + cx; v = fy * P[:, 1] / z + cy
        r = torch.ceil(0.5 * fx * vox / z).long().clamp(min=0, max=12)
        key_z = (z * 1e6).long() * 64 + (k + 1)
        rmax = int(r.max()) if len(r) else 0
        for dx in range(-rmax, rmax + 1):
            for dy in range(-rmax, rmax + 1):
                sel = (r >= abs(dx)) & (r >= abs(dy))
                uu = (u[sel] + dx).long(); vv = (v[sel] + dy).long()
                inb = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
                best.scatter_reduce_(0, (vv[inb] * W + uu[inb]), key_z[sel][inb], reduce="amin")
    lab = torch.where(best == np.iinfo(np.int64).max, torch.zeros_like(best), best % 64)
    return lab.view(H, W).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--colmap", default=str(ROOT / "Data/teatime/colmap/sparse/0"))
    ap.add_argument("--lerf", default=str(ROOT / "Data/benchmarks/lerf_mask_dl/x/teatime"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--preview", default="")
    ap.add_argument("--prompt_map", default="", help="canh khac teatime: JSON {prompt LERF: ten vat}")
    a = ap.parse_args()
    global PROMPT_OBJ
    if a.prompt_map:
        PROMPT_OBJ = json.loads(Path(a.prompt_map).read_text(encoding="utf-8"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    names = sorted(p.stem for p in Path(a.frames).glob("*.npz"))
    objs = []
    for n in names:
        z = np.load(Path(a.frames) / f"{n}.npz")
        objs.append((z["xyz"].astype(np.float64), float(z["voxel"])))
    ours, theirs = centers(a.colmap), centers(Path(a.lerf) / "sparse" / "0")
    common = sorted(set(ours) & set(theirs))
    s, R, t = umeyama(np.array([theirs[n][2] for n in common]), np.array([ours[n][2] for n in common]))
    res = {"frames_dir": a.frames, "objects": names, "per_view": {}, "per_prompt": {}}
    for vdir in sorted((Path(a.lerf) / "test_mask").iterdir()):
        vn = f"test_{vdir.name}"
        Rc, tc, _, cam = theirs[vn]
        p = cam.params
        fx, fy, cx, cy = (p[0], p[0], p[1], p[2]) if cam.model == "SIMPLE_PINHOLE" else (p[0], p[1], p[2], p[3])
        R2 = Rc @ R.T; t2 = -R2 @ t + s * tc
        W, H = int(cam.width), int(cam.height)
        lab = render_ids(objs, R2, t2, fx, fy, cx, cy, W, H, dev)   # toa do camera theo don vi cua minh (= s x LERF), phep chieu khong doi
        vis = np.zeros((H, W, 3), np.uint8)
        for f in sorted(vdir.glob("*.png")):
            pr = f.stem
            if pr not in PROMPT_OBJ or PROMPT_OBJ[pr] not in names:
                continue
            gt = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if gt.shape != (H, W):
                gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
            g = gt > 128
            pm = lab == names.index(PROMPT_OBJ[pr]) + 1
            iou = float((g & pm).sum() / max((g | pm).sum(), 1))
            res["per_view"].setdefault(vn, {})[pr] = {"obj": PROMPT_OBJ[pr], "iou": iou,
                                                       "pred_px": int(pm.sum()), "gt_px": int(g.sum())}
            res["per_prompt"].setdefault(pr, []).append(iou)
            vis[g & pm] = (60, 200, 60); vis[g & ~pm] = (60, 60, 230); vis[~g & pm] = (230, 120, 40)
        if a.preview:
            Path(a.preview).mkdir(parents=True, exist_ok=True)
            img = cv2.imread(str(Path(a.lerf) / "images" / f"{vn}.jpg"))
            if img is not None:
                img = cv2.resize(img, (W, H))
                cv2.imwrite(str(Path(a.preview) / f"frame_iou_{vn}.jpg"), np.hstack([img, cv2.addWeighted(img, 0.4, vis, 0.6, 0)]))
    res["per_prompt"] = {k: float(np.mean(v)) for k, v in res["per_prompt"].items()}
    res["per_object"] = {PROMPT_OBJ[k]: v for k, v in res["per_prompt"].items()}
    res["mIoU"] = float(np.mean(list(res["per_prompt"].values())))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"mIoU": res["mIoU"], **{k: round(v, 3) for k, v in res["per_object"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
