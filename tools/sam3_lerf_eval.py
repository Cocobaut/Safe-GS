"""
Chấm SAM 3 (chế độ ảnh, prompt chữ) trên LERF-Mask teatime bằng ground truth, để chọn ngưỡng / độ nở mask.

    python tools/modal_spawn.py ...  (modal_sam3.py::_probe)  ->  python -m modal run modal_sam3.py::pull_probe
    python tools/sam3_lerf_eval.py --probs Data/teatime/sam3_masks/probe/lerf_probs.npz \
        --gt Data/benchmarks/lerf_mask_dl/x/teatime/test_mask --out Data/teatime/sam3_masks/probe/lerf_eval.json

IoU / Boundary IoU = đúng code script/eval_lerf_mask.py của Gaussian Grouping (dilation_ratio 0.02), trung bình theo view rồi
theo prompt. Quét: ngưỡng xác suất mask p, ngưỡng điểm instance s, độ nở d (px, âm = co), cách gộp (union / instance điểm cao nhất).
"""
import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np


def mask_to_boundary(mask, dilation_ratio=0.02):
    h, w = mask.shape
    dilation = max(1, int(round(dilation_ratio * np.sqrt(h ** 2 + w ** 2))))
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    new_mask_erode = cv2.erode(new_mask, np.ones((3, 3), dtype=np.uint8), iterations=dilation)
    return mask - new_mask_erode[1: h + 1, 1: w + 1]


def boundary_iou(gt, dt):
    dt = (dt > 128).astype("uint8"); gt = (gt > 128).astype("uint8")
    gb, db = mask_to_boundary(gt), mask_to_boundary(dt)
    u = ((gb + db) > 0).sum()
    return ((gb * db) > 0).sum() / u if u else 0.0


def iou(gt, dt):
    g, d = gt > 128, dt > 128
    u = np.logical_or(g, d).sum()
    return np.logical_and(g, d).sum() / u if u else 0.0


def make_pred(P, S, p_thr, s_thr, d, mode):
    if len(S) == 0 or P.shape[-1] < 8:
        return None
    keep = np.flatnonzero(S >= s_thr)
    if len(keep) == 0:
        return None
    if mode == "top1":
        keep = [keep[np.argmax(S[keep])]]
    m = np.zeros(P.shape[1:], bool)
    for j in keep:
        m |= P[j] >= int(p_thr * 255)
    m = m.astype(np.uint8)
    if d > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1)))
    elif d < 0:
        m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (-2 * d + 1, -2 * d + 1)))
    return m * 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    z = np.load(a.probs)
    gts = {}
    for vdir in sorted(Path(a.gt).iterdir()):
        for f in vdir.glob("*.png"):
            gts[(f"test_{vdir.name}", f.stem)] = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
    grid = list(itertools.product((0.3, 0.5, 0.7), (0.3, 0.5, 0.7), (-2, -1, 0, 1, 2, 3), ("union", "top1")))
    rows = []
    for p_thr, s_thr, d, mode in grid:
        per = {}
        for (v, pr), gt in gts.items():
            P = z[f"{v}|{pr}|p"] if f"{v}|{pr}|p" in z.files else np.zeros((0, 1, 1), np.uint8)
            S = z[f"{v}|{pr}|s"] if f"{v}|{pr}|s" in z.files else np.zeros(0)
            pred = make_pred(P, S, p_thr, s_thr, d, mode)
            if pred is None:
                pred = np.zeros_like(gt)
            elif pred.shape != gt.shape:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            per.setdefault(pr, []).append((iou(gt, pred), boundary_iou(gt, pred)))
        mi = {k: float(np.mean([x[0] for x in v])) for k, v in per.items()}
        mb = {k: float(np.mean([x[1] for x in v])) for k, v in per.items()}
        rows.append({"p": p_thr, "s": s_thr, "d": d, "mode": mode, "mIoU": float(np.mean(list(mi.values()))),
                     "mBIoU": float(np.mean(list(mb.values()))), "per_prompt_iou": mi, "per_prompt_biou": mb})
    rows.sort(key=lambda r: -(r["mIoU"] + r["mBIoU"]))
    Path(a.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    base = next(r for r in rows if (r["p"], r["s"], r["d"], r["mode"]) == (0.5, 0.5, 0, "union"))
    print(f"mặc định (p 0.5, s 0.5, d 0, union): mIoU {base['mIoU'] * 100:.1f} mBIoU {base['mBIoU'] * 100:.1f}")
    print("top 8 cấu hình:")
    for r in rows[:8]:
        print(f"  p {r['p']} s {r['s']} d {r['d']:+d} {r['mode']:5s}: mIoU {r['mIoU'] * 100:.1f} mBIoU {r['mBIoU'] * 100:.1f}")
    best = rows[0]
    print("per prompt (cấu hình tốt nhất): IoU / BIoU")
    for k in best["per_prompt_iou"]:
        print(f"  {k:20s} {best['per_prompt_iou'][k] * 100:5.1f} {best['per_prompt_biou'][k] * 100:5.1f}   "
              f"(mặc định {base['per_prompt_iou'][k] * 100:5.1f} {base['per_prompt_biou'][k] * 100:5.1f})")


if __name__ == "__main__":
    main()
