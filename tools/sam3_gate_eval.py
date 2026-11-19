"""
Cổng chọn instance SAM 3 không cần huấn luyện (Relative Top-1 + Neighbor Continuity), chấm trên LERF-Mask teatime bằng GT,
cùng file xác suất và cùng hàm IoU / Boundary IoU với tools/sam3_lerf_eval.py. Cùng một quy tắc cho mọi prompt:

  1. top-1: instance điểm cao nhất của (view, prompt); giữ nếu điểm >= --floor (không thì dự đoán rỗng);
  2. Relative Top-1: instance khác chỉ được xét nếu điểm >= rho x điểm top-1 (ngưỡng tương đối, thích nghi theo prompt / view);
  3. Neighbor Continuity: instance được xét chỉ được gộp nếu mask của nó chạm mask top-1 đã nới r px
     (r = 2 % đường chéo ảnh, cùng hằng số dilation_ratio của Boundary IoU) -> loại instance rời xa (vật khác cùng loại / nhầm).
So với cổng tuyệt đối của sam3_lerf_eval.py (s = 0.5 / 0.3, union). Quét floor x rho x có / không continuity, in toàn bộ bảng
(không chọn theo vật).

    .venv_da3/Scripts/python.exe tools/sam3_gate_eval.py --probs Data/teatime/sam3_masks/probe/lerf_probs.npz \
        --gt Data/benchmarks/lerf_mask_dl/x/teatime/test_mask --out Data/teatime/sam3_masks/probe/gate_eval.json
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sam3_lerf_eval import iou, boundary_iou, make_pred  # noqa: E402


def gate_pred(P, S, floor, rho, cont, p_thr=0.5):
    if len(S) == 0 or P.shape[-1] < 8:
        return None
    j0 = int(np.argmax(S))
    if S[j0] < floor:
        return None
    M = [P[j] >= int(p_thr * 255) for j in range(len(S))]
    m = M[j0].copy()
    if cont:
        h, w = m.shape
        r = max(1, int(round(0.02 * np.sqrt(h * h + w * w))))
        near = cv2.dilate(M[j0].astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    for j in range(len(S)):
        if j == j0 or S[j] < rho * S[j0]:
            continue
        if cont and not (M[j] & near).any():
            continue
        m |= M[j]
    return m.astype(np.uint8) * 255


def score(z, gts, fn):
    per = {}
    for (v, pr), gt in gts.items():
        P = z[f"{v}|{pr}|p"] if f"{v}|{pr}|p" in z.files else np.zeros((0, 1, 1), np.uint8)
        S = z[f"{v}|{pr}|s"] if f"{v}|{pr}|s" in z.files else np.zeros(0)
        pred = fn(P, S)
        if pred is None:
            pred = np.zeros_like(gt)
        elif pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        per.setdefault(pr, []).append((iou(gt, pred), boundary_iou(gt, pred)))
    mi = {k: float(np.mean([x[0] for x in v])) for k, v in per.items()}
    mb = {k: float(np.mean([x[1] for x in v])) for k, v in per.items()}
    return {"mIoU": float(np.mean(list(mi.values()))), "mBIoU": float(np.mean(list(mb.values()))),
            "per_prompt_iou": mi, "per_prompt_biou": mb, "pairs": sum(len(v) for v in per.values())}


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
    rows = []
    for s in (0.5, 0.3):
        r = score(z, gts, lambda P, S: make_pred(P, S, 0.5, s, 0, "union"))
        rows.append({"gate": f"absolute s={s} union", **r})
    for floor, rho, cont in itertools.product((0.1, 0.2, 0.3), (0.3, 0.5, 0.7, 0.9), (False, True)):
        r = score(z, gts, lambda P, S: gate_pred(P, S, floor, rho, cont))
        rows.append({"gate": f"rel-top1 floor={floor} rho={rho}" + (" +continuity" if cont else ""),
                     "floor": floor, "rho": rho, "continuity": cont, **r})
    Path(a.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    keys = ["tea in a glass", "stuffed bear", "cookies on a plate", "spoon handle", "sheep"]
    print(f"{'gate':45s} mIoU  mBIoU | " + " | ".join(k[:12] for k in keys))
    for r in rows:
        print(f"{r['gate']:45s} {r['mIoU'] * 100:5.1f} {r['mBIoU'] * 100:5.1f} | "
              + " | ".join(f"{r['per_prompt_iou'].get(k, float('nan')) * 100:12.1f}" for k in keys))


if __name__ == "__main__":
    main()
