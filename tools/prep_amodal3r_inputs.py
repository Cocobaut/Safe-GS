"""
Inputs for Amodal3R (arXiv 2503.13439, Sm0kyWu/Amodal3R @ d00e083f): per object, the best view + a 3-level mask in the
repo's convention (README / preprocess_image_w_mask): background 255 (> 230), visible object 188, occluded area 0 (< 127).

Same rule for every object:
  visible  = SAM 3 label of the object (labels/<frame>.png, ids from the pipeline config).
  occluder = pixels of OTHER objects (any non-table SAM 3 id) inside a dilation of the visible mask whose DA3 depth
             (tools/da3_depth.py, aligned to COLMAP) is > --margin_cm in front of the object's depth at the nearest visible
             pixel; only connected components touching the visible mask are kept. Other-object pixels inside the filled
             visible silhouette (holes) are occluders without a depth test.
  best view: not truncated by the image border, maximal visible area; ties broken by more occluder pixels.
  crop:    square around visible + occluder, object box filling --fill of the side, resized to 518 x 518 (Amodal3R resizes
           to 518 without keeping the aspect ratio, so the crop must be square). RGB = object pixels, black elsewhere.
Also written: meta.json (frame, crop box in original pixels, areas) so the result can be checked against the same view.

    python tools/prep_amodal3r_inputs.py --objects dia,banh_quy,coc_su,ly_tra,tao
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/pipeline/teatime_v3_sam3.json"))
    ap.add_argument("--images", default=str(ROOT / "Data/teatime/colmap/images"))
    ap.add_argument("--labels", default=str(ROOT / "Data/teatime/sam3_masks/labels"))
    ap.add_argument("--da3", default=str(ROOT / "Data/teatime/da3_depth"))
    ap.add_argument("--out", default=str(ROOT / "Data/teatime/amodal3r_in"))
    ap.add_argument("--objects", default="dia,banh_quy,coc_su,ly_tra,tao")
    ap.add_argument("--table_ids", default="13")
    ap.add_argument("--ring", type=float, default=0.25, help="occluder search ring = this * sqrt(visible area) pixels")
    ap.add_argument("--margin_cm", type=float, default=2.0, help="occluder must be this much in front of the object locally")
    ap.add_argument("--fill", type=float, default=0.75)
    ap.add_argument("--size", type=int, default=518)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))["S3_label"]["objects"]
    table = {int(t) for t in a.table_ids.split(",")}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    frames = sorted(p.stem for p in Path(a.labels).glob("*.png"))
    s3 = json.loads((ROOT / "Data/teatime/pipeline/v3_sam3/S3_label/support.json").read_text())
    margin = a.margin_cm / (100 * float(s3["meters_per_unit"]))           # cm -> scene units
    summary = {}
    for name in a.objects.split(","):
        ids = cfg[name]
        best = None
        for fr in frames:
            L = cv2.imread(str(Path(a.labels) / f"{fr}.png"), cv2.IMREAD_UNCHANGED)
            vis = np.isin(L, ids)
            n = int(vis.sum())
            if n < 400:
                continue
            H, W = L.shape
            ys, xs = np.nonzero(vis)
            if xs.min() <= 1 or ys.min() <= 1 or xs.max() >= W - 2 or ys.max() >= H - 2:
                continue                                        # truncated by the image border
            r = max(3, int(a.ring * np.sqrt(n)))
            ring = cv2.dilate(vis.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
            other = (L > 0) & ~np.isin(L, list(table)) & ~vis & ring
            occ = np.zeros_like(vis)
            if other.any():
                z = np.load(Path(a.da3) / "aligned" / f"{fr}.npz")
                D = cv2.resize(z["depth"].astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
                # depth of the object at the nearest visible pixel (local comparison, not a global median)
                _, lab = cv2.distanceTransformWithLabels((~vis).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
                vy, vx = np.nonzero(vis)
                near = np.zeros(lab.max() + 1, np.float32)
                near[lab[vy, vx]] = D[vy, vx]
                occ = other & (D < near[lab] - margin)
                # keep only occluder components that touch the visible mask
                n_cc, cc = cv2.connectedComponents(occ.astype(np.uint8))
                touch = cv2.dilate(vis.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
                good = np.unique(cc[touch & occ])
                occ = np.isin(cc, good[good > 0])
                # other-object pixels in a hole of the visible silhouette can only be in front of it (no depth test)
                cs, _ = cv2.findContours(vis.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                filled = cv2.drawContours(np.zeros((H, W), np.uint8), cs, -1, 1, -1) > 0
                occ |= other & filled
            key = (n, int(occ.sum()))
            if best is None or key > best[0]:
                best = (key, fr, vis, occ)
        if best is None:
            print(name, "no valid view"); continue
        (n, n_occ), fr, vis, occ = best
        img = cv2.imread(str(Path(a.images) / f"{fr}.jpg"))
        H, W = vis.shape
        ys, xs = np.nonzero(vis | occ)
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        side = int(np.ceil(max(xs.max() - xs.min(), ys.max() - ys.min()) / a.fill))
        x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
        pad = side                                                # pad so the crop never leaves the canvas
        canvas = lambda arr, val: cv2.copyMakeBorder(arr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=val)
        crop = lambda arr: arr[y0 + pad:y0 + pad + side, x0 + pad:x0 + pad + side]
        rgb = crop(canvas(np.where(vis[..., None], img, 0).astype(np.uint8), (0, 0, 0)))
        m = np.full((H, W), 255, np.uint8); m[vis] = 188; m[occ] = 0
        m = crop(canvas(m, 255))
        rgb = cv2.resize(rgb, (a.size, a.size), interpolation=cv2.INTER_AREA)
        m = cv2.resize(m, (a.size, a.size), interpolation=cv2.INTER_NEAREST)
        od = out / name; od.mkdir(exist_ok=True)
        cv2.imwrite(str(od / "000000.png"), rgb)
        cv2.imwrite(str(od / "000000_mask.png"), m)
        meta = {"frame": fr, "ids": ids, "visible_px": n, "occluder_px": n_occ, "occluded_ratio": n_occ / max(n + n_occ, 1),
                "crop_xywh": [x0, y0, side, side], "image_wh": [W, H], "size": a.size}
        (od / "meta.json").write_text(json.dumps(meta, indent=1))
        summary[name] = meta
        print(name, json.dumps(meta), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
