"""
Quan sát + đo chất lượng bộ mask có ID (SAM 3 / SAM2): video overlay, bảng ổn định theo thời gian, so sánh 2 bộ.

    python tools/mask_quality.py --images Data/teatime/colmap/images --labels Data/teatime/sam3_masks/labels \
        --legend Data/teatime/sam3_masks/legend.json --out Data/teatime/sam3_masks/quality [--ref Data/teatime/sam2_masks/labels]

Đo theo từng ID (chỉ so frame liền kề, ảnh đã sắp theo tên):
  frames      số frame ID xuất hiện
  area_jumps  số lần diện tích đổi > 2.5 lần giữa 2 frame liền kề (ID đổi chủ / mất / vỡ mask; cùng cách đo với SAM2 trước đây)
  pieces      số mảnh rời trung bình trong 1 frame (>1 = mask vỡ vụn)
  gaps        số lần ID biến mất rồi hiện lại (tracking đứt)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

SEED_PAL = np.random.default_rng(0).integers(40, 255, (256, 3)).astype(np.uint8)
SEED_PAL[0] = 0


def stats(labels_dir, ids=None):
    files = sorted(Path(labels_dir).glob("*.png"))
    A = {}                                                        # id -> [diện tích theo frame]
    pieces = {}
    for k, f in enumerate(files):
        lab = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        for i in np.unique(lab):
            if i == 0 or (ids is not None and i not in ids):
                continue
            m = (lab == i).astype(np.uint8)
            A.setdefault(int(i), np.zeros(len(files)))[k] = m.sum()
            n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            pieces.setdefault(int(i), []).append(int((st[1:, cv2.CC_STAT_AREA] >= 30).sum()))
    out = {}
    for i, a in A.items():
        pres = a > 0
        both = pres[1:] & pres[:-1]
        r = a[1:][both] / np.maximum(a[:-1][both], 1)
        trans = np.flatnonzero(np.diff(pres.astype(int)) == 1)
        out[i] = {"frames": int(pres.sum()), "area_jumps": int(((r > 2.5) | (r < 0.4)).sum()),
                  "pieces": round(float(np.mean(pieces[i])), 2),
                  "gaps": int(len(trans) + (1 if pres[0] else 0) - 1) if pres.any() else 0,   # số đợt xuất hiện - 1
                  "mean_area_px": int(a[pres].mean())}
    return out, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--legend", default=None)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    legend = json.loads(Path(a.legend).read_text(encoding="utf-8")) if a.legend else {}
    st, files = stats(a.labels)
    (out / "stability.json").write_text(json.dumps(st, indent=1))
    rows = ["| ID | vật (prompt) | frame | nhảy diện tích | mảnh TB | đứt |", "|---|---|---|---|---|---|"]
    for i, v in sorted(st.items()):
        rows.append(f"| {i} | {legend.get(str(i), {}).get('prompt', '')} | {v['frames']} | {v['area_jumps']} | {v['pieces']} | {v['gaps']} |")
    (out / "stability.md").write_text("\n".join(rows), encoding="utf-8")
    print("\n".join(rows))
    if a.ref:
        sr, _ = stats(a.ref)
        tot = lambda s: (sum(v["area_jumps"] for v in s.values()), len(s))
        print(f"\nSAM 3: tổng nhảy diện tích {tot(st)[0]} trên {tot(st)[1]} ID | ref: {tot(sr)[0]} trên {tot(sr)[1]} ID")

    # video: ảnh gốc | overlay có nhãn
    imgs = sorted(Path(a.images).glob("*.jpg"))
    names = {f.stem for f in files}
    imgs = [p for p in imgs if p.stem in names]
    H, W = cv2.imread(str(imgs[0])).shape[:2]
    vw = cv2.VideoWriter(str(out / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 12, (W * 2, H))
    for p, f in zip(imgs, files):
        im = cv2.imread(str(p)); lab = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        ov = (0.5 * im + 0.5 * SEED_PAL[lab][..., ::-1]).astype(np.uint8)
        edge = cv2.morphologyEx((lab > 0).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        for i in np.unique(lab):
            if i == 0:
                continue
            ys, xs = np.nonzero(lab == i)
            if len(xs) < 400:
                continue
            txt = f"{i}" + (f" {legend[str(i)]['prompt']}" if str(i) in legend else "")
            cv2.putText(ov, txt, (int(xs.mean()) - 25, int(ys.mean())), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(ov, txt, (int(xs.mean()) - 25, int(ys.mean())), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(ov, p.stem, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        vw.write(np.hstack([im, ov]))
    vw.release()
    print("video:", out / "overlay.mp4")


if __name__ == "__main__":
    main()
