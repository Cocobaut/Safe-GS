"""
Step 1: validate the SAM2 ID masks of one object (replaces the non-existent script/eval_mask.py of the old plan).

Reads <data>/object_mask/*.png (cropped ID masks from Step 0) and <data>/images/*.png, writes into --out:
  mask_validation_report.json  per-frame area / #components / border contact + summary + pass/fail
  preview_<frame>.jpg          overlay of the object on chosen frames
  preview_grid.jpg             12 evenly spaced frames that contain the object

Pass criteria: object present in >= --min_frames frames, and in >= 90% of those frames the largest connected
component holds >= 80% of the mask area (no badly fragmented masks).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


def ids_of(path):
    a = np.asarray(Image.open(path))
    return a[..., 0] if a.ndim == 3 else a


def overlay(img, mask, color=(255, 60, 60)):
    a = np.asarray(img).astype(np.float32)
    dim = a * 0.35
    out = np.where(mask[..., None], a * 0.5 + np.array(color) * 0.5, dim)
    edge = mask ^ ndimage.binary_erosion(mask, iterations=2)
    out[edge] = (255, 255, 0)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_frames", type=int, default=80)
    ap.add_argument("--preview", nargs="*", default=["frame_00001", "frame_00108", "frame_00140"])
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    per_frame, present = {}, []
    for mp in sorted((data / "object_mask").glob("*.png")):
        m = ids_of(mp) == args.id
        area = int(m.sum())
        if area == 0:
            per_frame[mp.stem] = {"area": 0}
            continue
        lab, n = ndimage.label(m)
        sizes = np.bincount(lab.ravel())[1:]
        h, w = m.shape
        per_frame[mp.stem] = {
            "area": area, "area_pct": round(100.0 * area / m.size, 3), "components": int(n),
            "largest_component_frac": round(float(sizes.max()) / area, 3),
            "touches_border": bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()),
        }
        present.append(mp.stem)

    frac = np.array([per_frame[s]["largest_component_frac"] for s in present]) if present else np.array([])
    coherent = float((frac >= 0.8).mean()) if present else 0.0
    summary = {
        "object_id": args.id, "frames_total": len(per_frame), "frames_present": len(present),
        "first_last_present": [present[0], present[-1]] if present else None,
        "coherent_frames_frac": round(coherent, 3),
        "frames_touching_border": int(sum(per_frame[s]["touches_border"] for s in present)),
        "mean_area_pct": round(float(np.mean([per_frame[s]["area_pct"] for s in present])), 3) if present else 0,
    }
    summary["pass"] = bool(len(present) >= args.min_frames and coherent >= 0.9)

    for stem in args.preview:
        mp, ip = data / "object_mask" / f"{stem}.png", data / "images" / f"{stem}.png"
        if mp.exists() and ip.exists():
            overlay(Image.open(ip).convert("RGB"), ids_of(mp) == args.id).save(out / f"preview_{stem}.jpg", quality=92)
    if present:
        pick = [present[i] for i in np.linspace(0, len(present) - 1, min(12, len(present))).round().astype(int)]
        tiles = []
        for stem in pick:
            t = overlay(Image.open(data / "images" / f"{stem}.png").convert("RGB"),
                        ids_of(data / "object_mask" / f"{stem}.png") == args.id)
            t.thumbnail((330, 330))
            ImageDraw.Draw(t).text((6, 6), stem, fill=(255, 255, 255))
            tiles.append(t)
        tw, th = tiles[0].size
        grid = Image.new("RGB", (tw * 4, th * ((len(tiles) + 3) // 4)))
        for i, t in enumerate(tiles):
            grid.paste(t, ((i % 4) * tw, (i // 4) * th))
        grid.save(out / "preview_grid.jpg", quality=90)

    (out / "mask_validation_report.json").write_text(json.dumps({"summary": summary, "frames": per_frame}, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["pass"]:
        raise SystemExit("[step1] mask check FAILED, see mask_validation_report.json")


if __name__ == "__main__":
    main()
