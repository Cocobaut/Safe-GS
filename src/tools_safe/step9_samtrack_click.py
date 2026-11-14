"""Step 9 (GPU): inpainting masks on the virtual views with Segment-and-Track-Anything, headless.

Replaces the manual Gradio app of Inpaint360GS (end of run_remove.sh): one positive click on the first virtual view
(removed scene) -> SAM (vit_b, as model_args.py) -> DeAOT propagation over the other views.
Gate: a tracked mask whose area leaves [gate_lo, gate_hi] x first-frame area, or is empty, is replaced by the
fallback mask (the model's own prediction of the target ids on the full-scene virtual render, objects_pred), when given.

Writes the masks where the app writes them (prepare_lama_data.py reads them):
    <samt_dir>/tracking_results/images/images_masks/<name>.png   (uint8 0/255)
and a copy for the run:
    <export>/1_masks/<name>.png, <export>/2_masks_overlay/<name>.jpg, <export>/summary.json
    (summary keys of the reference run: n_masks, n_virtual_views, click, refine, gated_frames, empty_masks,
     area_pct_min, area_pct_max, area_pct_per_view)

  cd src && python tools_safe/step9_samtrack_click.py \
      --frames <M>/virtual/ours_object_removal/iteration_<N>/renders --click 482 371 \
      --export <run>/step9_inpaint_masks_samt [--fallback_pred <M>/virtual/ours<N>/objects_pred --target_ids 44]
Checkpoints (download_ckpt.sh): <samt_dir>/ckpt/sam_vit_b_01ec64.pth, <samt_dir>/ckpt/R50_DeAOTL_PRE_YTB_DAV.pth
"""
from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np

SAMT = Path(__file__).resolve().parents[1] / "Segment-and-Track-Anything"


def overlay(rgb, mask, click=None):
    out = rgb.copy()
    m = mask > 0
    out[m] = (0.5 * out[m] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cnts, -1, (255, 255, 0), 2)
    if click is not None:
        cv2.circle(out, tuple(int(c) for c in click), 6, (0, 255, 0), -1)
    return out


def main():
    ap = ArgumentParser()
    ap.add_argument("--frames", required=True, help="virtual renders of the removed scene (*.png, sorted = video)")
    ap.add_argument("--click", type=float, nargs=2, default=None, metavar=("X", "Y"),
                    help="pixel on the first view; omitted = no SAM-Track, every mask from --fallback_pred")
    ap.add_argument("--neg", type=float, nargs="*", default=[], help="optional negative clicks x1 y1 x2 y2 ...")
    ap.add_argument("--export", required=True)
    ap.add_argument("--samt_dir", default=str(SAMT))
    ap.add_argument("--fallback_pred", default=None, help="objects_pred of the full-scene virtual render")
    ap.add_argument("--target_ids", default="", help="ids of the fallback mask, e.g. 44 or 26,24")
    ap.add_argument("--gate_lo", type=float, default=0.4)
    ap.add_argument("--gate_hi", type=float, default=2.5)
    ap.add_argument("--dilate", type=int, default=0, help="extra dilation (px) of every mask")
    args = ap.parse_args()

    frames = sorted(Path(args.frames).resolve().glob("*.png"))
    if args.fallback_pred:
        args.fallback_pred = str(Path(args.fallback_pred).resolve())
    args.export = str(Path(args.export).resolve())
    if args.click is None and not (args.fallback_pred and args.target_ids):
        raise SystemExit("--click or (--fallback_pred and --target_ids) is required")
    samt = Path(args.samt_dir).resolve()
    os.chdir(samt)                                     # model_args.py / configs use paths relative to the repo
    for p in (samt, samt / "aot", samt / "sam"):
        sys.path.insert(0, str(p))
    import torch  # noqa: E402
    if args.click is not None:
        from aot_tracker import get_aot  # noqa: E402
        from model_args import aot_args, sam_args  # noqa: E402
        from tool.segmentor import Segmentor  # noqa: E402

    assert frames, f"no frames in {args.frames}"
    exp = Path(args.export)
    (exp / "1_masks").mkdir(parents=True, exist_ok=True)
    (exp / "2_masks_overlay").mkdir(parents=True, exist_ok=True)
    app_dir = samt / "tracking_results" / "images" / "images_masks"
    app_dir.mkdir(parents=True, exist_ok=True)
    for f in app_dir.glob("*.png"):
        f.unlink()

    fb_ids = [int(x) for x in args.target_ids.split(",") if x.strip()]

    def fallback(name, shape):
        if not args.fallback_pred or not fb_ids:
            return None
        p = Path(args.fallback_pred) / name
        if not p.exists():
            return None
        m = np.isin(cv2.imread(str(p), cv2.IMREAD_UNCHANGED), fb_ids).astype(np.uint8)
        return cv2.resize(m, shape[::-1], interpolation=cv2.INTER_NEAREST) if m.shape != shape else m

    if args.click is not None:
        segmentor = Segmentor(sam_args)
        tracker = get_aot(aot_args)
        coords = [list(args.click)] + [args.neg[i:i + 2] for i in range(0, len(args.neg), 2)]
        modes = [1] + [0] * (len(coords) - 1)

    area, gated, empty, refine = {}, [], [], []
    a0 = None
    kernel = np.ones((2 * args.dilate + 1,) * 2, np.uint8) if args.dilate > 0 else None
    with torch.no_grad():
        for i, f in enumerate(frames):
            rgb = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
            if args.click is None:
                mask = fallback(f.name, rgb.shape[:2])
                mask = np.zeros(rgb.shape[:2], np.uint8) if mask is None else mask
                refine.append(f.stem)
            elif i == 0:
                mask = segmentor.segment_with_click(rgb, np.array(coords), np.array(modes), multimask=True)
                tracker.add_reference_frame(rgb, mask.astype(np.uint8), 1, 0)
            else:
                pred = tracker.track(rgb)
                tracker.update_memory(pred)
                mask = (pred.squeeze(0).squeeze(0).detach().cpu().numpy() > 0).astype(np.uint8)
            a = float(mask.mean())
            if i == 0:
                a0 = max(a, 1e-9)
            if args.click is None:
                if a == 0:
                    empty.append(f.stem)
            elif a == 0 or not (args.gate_lo * a0 <= a <= args.gate_hi * a0):
                fb = fallback(f.name, mask.shape)
                (empty if a == 0 else gated).append(f.stem)
                if fb is not None and fb.any():
                    mask = fb
                    refine.append(f.stem)
            if kernel is not None:
                mask = cv2.dilate(mask, kernel)
            m255 = (mask > 0).astype(np.uint8) * 255
            cv2.imwrite(str(app_dir / f.name), m255)
            cv2.imwrite(str(exp / "1_masks" / f.name), m255)
            cv2.imwrite(str(exp / "2_masks_overlay" / (f.stem + ".jpg")),
                        cv2.cvtColor(overlay(rgb, m255, args.click if i == 0 and args.click else None),
                                     cv2.COLOR_RGB2BGR))
            area[f.stem] = round(100.0 * float((m255 > 0).mean()), 3)

    vals = list(area.values())
    summary = {"n_masks": len(area), "n_virtual_views": len(frames),
               "click": [int(c) for c in args.click] if args.click else None,
               "mode": "samtrack_click" if args.click else "fallback_objects_pred",
               "refine": refine, "gated_frames": gated, "empty_masks": empty,
               "area_pct_min": min(vals), "area_pct_max": max(vals), "area_pct_per_view": area,
               "frames": str(Path(args.frames)), "app_masks": str(app_dir)}
    (exp / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "area_pct_per_view"}))


if __name__ == "__main__":
    main()
