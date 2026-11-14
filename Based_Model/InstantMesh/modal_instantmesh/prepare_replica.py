"""
Step 1 (local CPU, no Modal credit): Replica object -> RGBA input images for the original InstantMesh.

InstantMesh only has rembg (whole-foreground removal), which cannot isolate one object in a Replica room frame.
So the object is cut with the SAM masks the Safe-GS pipeline already produced (no new SAM run, zero GPU cost):
    outputs/Replica/ours/<run>/step6_sync_mask/1_masks_synced/<frame>.png   HQ-SAM (step 5) masks, IDs synced across
                                                                            views (step 6), uint8, 0 = background
    object = ID target_id of configs/datasets/replica.yaml (chosen by max IoU with the GT instance, object_id.json)
    RGB    = outputs/Replica/ours/<run>/step0_colmap/images/<frame>.jpg
Same frame selection / cropping as prepare_teatime_napkin.py: mask in one piece (small specks ignored), not touching the
image border, area >= MIN_AREA_PCT, TOP_K largest; alpha = eroded + hole-filled mask, crop, pad to a square with
foreground ratio 0.85 (= resize_foreground of run.py, skipped with --no_rembg), resize to OUT_SIZE.

    python modal_instantmesh/prepare_replica.py                    (from F:\\Safe-GS\\Based_Model\\InstantMesh)
    python modal_instantmesh/prepare_replica.py --runs room0_obj1 --force
Writes outputs/Replica/Object_mesh_SAM_InstantMesh/<scene>/<obj>/step8_instantmesh_input_sam/
       {rgba,preview}/<frame>.png, frames.json, contact_sheet.jpg
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

SAFE_GS = Path(__file__).resolve().parents[3]                 # F:\Safe-GS
OURS = SAFE_GS / "outputs" / "Replica" / "ours"
OUT_ROOT = SAFE_GS / "outputs" / "Replica" / "Object_mesh_SAM_InstantMesh"
RUNS = {"room0_obj1": 192, "room0_obj2": 18, "office0_obj1": 19, "office0_obj2": 85}   # run -> target_id (replica.yaml)
STEP1 = "step8_instantmesh_input_sam"
TOP_K = 3                      # 3 frames per object: InstantMesh is single-view, keep a few views, cheap on GPU
MIN_AREA_PCT = 0.2             # the room0 candle covers at most 0.4 % of a 1200x680 frame
MIN_PART_PCT = 0.02            # connected parts smaller than this (% of image) are SAM specks, ignored
ERODE_PX = 2
RATIO = 0.85
OUT_SIZE = 640


def out_dir(run: str) -> Path:
    scene, obj = run.split("_")
    return OUT_ROOT / scene / obj / STEP1


def clean_mask(lab: np.ndarray, oid: int) -> tuple[np.ndarray, int]:
    m = lab == oid
    cc, n = ndimage.label(m)
    if n == 0:
        return m, 0
    sizes = ndimage.sum(m, cc, range(1, n + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if 100.0 * s / m.size >= MIN_PART_PCT]
    return np.isin(cc, keep), len(keep)


def candidates(run: str, oid: int) -> list[dict]:
    rows = []
    for p in sorted((OURS / run / "step6_sync_mask" / "1_masks_synced").glob("frame*.png")):
        m, parts = clean_mask(np.array(Image.open(p)), oid)
        if not m.any():
            continue
        H, W = m.shape
        ys, xs = np.nonzero(m)
        border = ys.min() == 0 or xs.min() == 0 or ys.max() == H - 1 or xs.max() == W - 1
        rows.append({"frame": p.stem, "area_pct": round(100.0 * m.mean(), 3), "parts": parts,
                     "touches_border": bool(border),
                     "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]})
    ok = [r for r in rows if r["parts"] == 1 and not r["touches_border"] and r["area_pct"] >= MIN_AREA_PCT]
    ok.sort(key=lambda r: -r["area_pct"])
    return ok[:TOP_K], len(rows)


def make_rgba(run: str, oid: int, frame: str) -> Image.Image:
    rgb = np.array(Image.open(OURS / run / "step0_colmap" / "images" / f"{frame}.jpg").convert("RGB"))
    m, _ = clean_mask(np.array(Image.open(OURS / run / "step6_sync_mask" / "1_masks_synced" / f"{frame}.png")), oid)
    assert rgb.shape[:2] == m.shape, (rgb.shape, m.shape)
    m = ndimage.binary_fill_holes(ndimage.binary_erosion(m, iterations=ERODE_PX))
    alpha = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.0))
    rgba = np.dstack([rgb, np.array(alpha)])
    rgba[..., :3][~m] = 255
    ys, xs = np.nonzero(rgba[..., 3] > 0)
    fg = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    side = int(max(fg.shape[:2]) / RATIO)
    canvas = np.zeros((side, side, 4), np.uint8)
    canvas[..., :3] = 255
    y0, x0 = (side - fg.shape[0]) // 2, (side - fg.shape[1]) // 2
    canvas[y0:y0 + fg.shape[0], x0:x0 + fg.shape[1]] = fg
    return Image.fromarray(canvas, "RGBA").resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS)


def prepare(run: str, oid: int, force: bool) -> None:
    out = out_dir(run)
    if out.exists() and any(out.iterdir()):
        if not force:
            print(f"[skip] {out} exists (--force to redo)")
            return
        shutil.rmtree(out)
    (out / "rgba").mkdir(parents=True, exist_ok=True)
    (out / "preview").mkdir(parents=True, exist_ok=True)
    sel, n_seen = candidates(run, oid)
    if not sel:
        raise SystemExit(f"{run}: no usable frame for ID {oid}")
    tiles = []
    for r in sel:
        im = make_rgba(run, oid, r["frame"])
        im.save(out / "rgba" / f"{r['frame']}.png")
        prev = Image.new("RGB", im.size, (128, 128, 128))
        prev.paste(im, mask=im.getchannel("A"))
        prev.save(out / "preview" / f"{r['frame']}.png")
        tiles.append(prev.resize((256, 256)))
    sheet = Image.new("RGB", (256 * len(tiles), 256), (60, 60, 60))
    for i, t in enumerate(tiles):
        sheet.paste(t, (i * 256, 0))
    sheet.save(out / "contact_sheet.jpg", quality=92)
    cfg = {"run": run, "object_id": oid, "mask_source": f"outputs/Replica/ours/{run}/step6_sync_mask/1_masks_synced "
           "(HQ-SAM step 5, IDs synced step 6; reused, no new SAM run)", "frames_with_object": n_seen,
           "top_k": TOP_K, "min_area_pct": MIN_AREA_PCT, "min_part_pct": MIN_PART_PCT, "erode_px": ERODE_PX,
           "ratio": RATIO, "out_size": OUT_SIZE, "frames": sel}
    (out / "frames.json").write_text(json.dumps(cfg, indent=2))
    print(f"{run} (ID {oid}): {n_seen} frames with object -> {[(r['frame'], r['area_pct']) for r in sel]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=",".join(RUNS))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for run in a.runs.split(","):
        prepare(run, RUNS[run], a.force)


if __name__ == "__main__":
    main()
