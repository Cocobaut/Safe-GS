"""
Step 1 (local, CPU): LERF teatime + SAM2 label masks -> RGBA input images of the napkin (object ID 11) for InstantMesh.

Re-uses F:\\LERF Datasets\\teatime\\teatime\\mask\\labels (uint8 ID maps, 0 = background) - no rembg needed.
For each candidate frame:
  * alpha = (label == 11), eroded by ERODE_PX to drop the wooden-table fringe, light 1 px feather
  * crop to the mask bbox, pad to a square, foreground ratio 0.85 (= InstantMesh resize_foreground(…, 0.85),
    which run.py skips when --no_rembg is given), resize to OUT_SIZE
Frames are chosen automatically: napkin mask in one piece, not touching the image border (=not cut off),
area >= MIN_AREA_PCT; the TOP_K largest are kept.

    python modal_instantmesh/prepare_teatime_napkin.py            (from F:\\Safe-GS\\Based_Model\\InstantMesh)
Writes outputs/teatime_napkin/step1_inputs/{rgba,preview}/frame_XXXXX.png + frames.json + contact_sheet.jpg
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

RAW = Path(r"F:\LERF Datasets\teatime\teatime")
OBJ_ID = 11
TOP_K = 8
MIN_AREA_PCT = 1.5
ERODE_PX = 2
RATIO = 0.85
OUT_SIZE = 640
OUT = Path(__file__).resolve().parents[1] / "outputs" / "teatime_napkin" / "step1_inputs"


def candidates() -> list[dict]:
    rows = []
    for p in sorted((RAW / "mask" / "labels").glob("frame_*.png")):
        m = np.array(Image.open(p)) == OBJ_ID
        if not m.any():
            continue
        H, W = m.shape
        ys, xs = np.nonzero(m)
        n_parts = ndimage.label(m)[1]
        border = ys.min() == 0 or xs.min() == 0 or ys.max() == H - 1 or xs.max() == W - 1
        area = 100.0 * m.mean()
        rows.append({"frame": p.stem, "area_pct": round(area, 3), "parts": int(n_parts), "touches_border": bool(border),
                     "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]})
    ok = [r for r in rows if r["parts"] == 1 and not r["touches_border"] and r["area_pct"] >= MIN_AREA_PCT]
    ok.sort(key=lambda r: -r["area_pct"])
    return ok[:TOP_K]


def make_rgba(frame: str) -> Image.Image:
    rgb = np.array(Image.open(RAW / "images" / f"{frame}.jpg").convert("RGB"))
    m = np.array(Image.open(RAW / "mask" / "labels" / f"{frame}.png")) == OBJ_ID
    assert rgb.shape[:2] == m.shape, (rgb.shape, m.shape)
    m = ndimage.binary_erosion(m, iterations=ERODE_PX)
    m = ndimage.binary_fill_holes(m)
    alpha = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.0))
    rgba = np.dstack([rgb, np.array(alpha)])
    rgba[..., :3][~m] = 255                                    # white outside (Zero123++ composites on white anyway)
    ys, xs = np.nonzero(rgba[..., 3] > 0)
    fg = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    side = int(max(fg.shape[:2]) / RATIO)
    canvas = np.zeros((side, side, 4), np.uint8)
    canvas[..., :3] = 255
    y0, x0 = (side - fg.shape[0]) // 2, (side - fg.shape[1]) // 2
    canvas[y0:y0 + fg.shape[0], x0:x0 + fg.shape[1]] = fg
    return Image.fromarray(canvas, "RGBA").resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS)


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit(f"{OUT} already exists - not overwriting (delete it or rename to _vK first)")
    (OUT / "rgba").mkdir(parents=True, exist_ok=True)
    (OUT / "preview").mkdir(parents=True, exist_ok=True)
    sel = candidates()
    tiles = []
    for r in sel:
        im = make_rgba(r["frame"])
        im.save(OUT / "rgba" / f"{r['frame']}.png")
        prev = Image.new("RGB", im.size, (128, 128, 128))      # grey preview: white napkin visible
        prev.paste(im, mask=im.getchannel("A"))
        prev.save(OUT / "preview" / f"{r['frame']}.png")
        tiles.append(prev.resize((256, 256)))
    sheet = Image.new("RGB", (256 * 4, 256 * ((len(tiles) + 3) // 4)), (60, 60, 60))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % 4) * 256, (i // 4) * 256))
    sheet.save(OUT / "contact_sheet.jpg", quality=92)
    cfg = {"object_id": OBJ_ID, "top_k": TOP_K, "min_area_pct": MIN_AREA_PCT, "erode_px": ERODE_PX,
           "ratio": RATIO, "out_size": OUT_SIZE, "frames": sel}
    (OUT / "frames.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps([(r["frame"], r["area_pct"]) for r in sel]))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
