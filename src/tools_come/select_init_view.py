"""v2: choose the virtual view whose LaMa fill is cleanest, to initialize the new Gaussians (edit_object_inpaint.py,
INPAINT360_INIT_VIEW) instead of the fixed view 00004.

Score per view (lower is better), inside the inpainting mask vs a ring of table around it:
  Lab distance of the mean colors + 20 * excess fraction of fill pixels clearly darker than the table (dark smudges)
Views whose mask touches the image border or is smaller than half the median mask area are not candidates
(the initialization should see the whole hole).

  python tools_come/select_init_view.py -s <data> --out <json>
"""
import json
import os
from argparse import ArgumentParser

import cv2
import numpy as np


def lab(img):
    x = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)
    x[..., 0] *= 100 / 255
    x[..., 1:] -= 128
    return x


def main():
    parser = ArgumentParser()
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("--images", default="images_inpaint_unseen_virtual")
    parser.add_argument("--masks", default="inpaint_2d_unseen_mask_virtual")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    k_in, k_out = np.ones((11, 11), np.uint8), np.ones((51, 51), np.uint8)
    rows = []
    for f in sorted(os.listdir(os.path.join(args.source_path, args.masks))):
        if not f.endswith(".png"):
            continue
        name = f[:-4]
        m = (cv2.imread(os.path.join(args.source_path, args.masks, f), cv2.IMREAD_GRAYSCALE) > 128).astype(np.uint8)
        img = cv2.imread(os.path.join(args.source_path, args.images, name + ".JPG"))
        if img is None or m.sum() == 0:
            continue
        ring = (cv2.dilate(m, k_out) > 0) & ~(cv2.dilate(m, k_in) > 0)
        hole = m > 0
        L = lab(img)
        dist = float(np.linalg.norm(L[hole].mean(0) - L[ring].mean(0)))
        # dark smudges: share of fill pixels clearly darker than the table, minus the same share in the ring itself
        # (wood grain / nearby dark objects), so only the excess darkness of the fill counts
        thr = np.median(L[ring][:, 0]) - 12
        dark = max(0.0, float((L[hole][:, 0] < thr).mean()) - float((L[ring][:, 0] < thr).mean()))
        border = bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())
        rows.append({"view": name, "lab_dist": round(dist, 3), "dark_frac": round(dark, 4),
                     "area_pct": round(100 * float(hole.mean()), 3), "touches_border": border,
                     "score": round(dist + 20 * dark, 3)})

    med = float(np.median([r["area_pct"] for r in rows]))
    cand = [r for r in rows if not r["touches_border"] and r["area_pct"] >= 0.5 * med] or rows
    best = min(cand, key=lambda r: r["score"])
    old = next((r for r in rows if r["view"] == "00004"), None)
    res = {"selected": best["view"], "selected_row": best, "previous_fixed_view_00004": old,
           "criterion": "min(lab_dist + 20*dark_frac) over views not touching the border with area >= 0.5*median",
           "per_view": sorted(rows, key=lambda r: r["score"])}
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"SELECTED {best['view']} score {best['score']} (00004: {old and old['score']})")


if __name__ == "__main__":
    main()
