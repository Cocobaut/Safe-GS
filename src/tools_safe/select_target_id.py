"""Step 3 + 6-id (local CPU): choose the target_id of a run in the 3D-consistent ID space of step 6.

The reference run chose napkin = 44 this way (step3_reuse_napkin_reference + step6_sync_mask/napkin_id.json):
the Inpaint360GS ID whose synced masks overlap best a reference mask of the object over all frames.

Two ways to give the object:
  --ref_masks <dir>   per-frame masks named like the images (binary 0/255, or an ID map with --ref_id <id>), e.g.
                      SAM2 masks of modal_lerf_prep.py, Replica instance masks ... -> copied to step3_object_reference/
  --click <frame> <x> <y>   one pixel of the object on one photo (full resolution): the synced ID under it, checked
                      on all frames for its area
Writes <run>/step6*/object_id.json (keys of the reference napkin_id.json: <object>_id = chosen id, iou_total,
precision_total, recall_total, candidates_top5) and prints the line to put in configs/datasets/<dataset>.yaml.

  python tools_safe/select_target_id.py --run <outputs/LERF/teatime_napkin> --ref_masks <labels> --ref_id 11
  python tools_safe/select_target_id.py --run <outputs/Replica/office0_vase> --click frame_000120 512 300
"""
from __future__ import annotations

import json
import shutil
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image


def step_dir(run: Path, n: int) -> Path:
    c = sorted(d for d in run.glob(f"step{n}_*") if d.is_dir())
    if not c:
        raise SystemExit(f"no step{n}_* folder in {run}")
    return c[0]


def load_ids(p: Path, shape=None) -> np.ndarray:
    im = Image.open(p)
    if shape is not None and im.size != (shape[1], shape[0]):
        im = im.resize((shape[1], shape[0]), Image.NEAREST)
    a = np.asarray(im)
    return a[..., 0] if a.ndim == 3 else a


def main():
    ap = ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ref_masks", default=None)
    ap.add_argument("--ref_id", type=int, default=None, help="id of the object in --ref_masks (default: mask > 0)")
    ap.add_argument("--click", nargs=3, default=None, metavar=("FRAME", "X", "Y"))
    ap.add_argument("--object", default="object", help="name used in the json key (<object>_id)")
    args = ap.parse_args()
    run = Path(args.run)
    s6 = step_dir(run, 6)
    syn = s6 / "1_masks_synced"
    frames = sorted(p.stem for p in syn.glob("*.png"))
    if not frames:
        raise SystemExit(f"no synced masks in {syn}")

    if args.click:
        f, x, y = args.click[0], float(args.click[1]), float(args.click[2])
        photos = sorted(step_dir(run, 0).glob(f"images/{f}.*"))
        W, H = Image.open(photos[0]).size if photos else (None, None)
        ids = load_ids(syn / f"{f}.png")
        if W:                                              # click in photo pixels -> synced mask pixels
            x, y = x * ids.shape[1] / W, y * ids.shape[0] / H
        cid = int(ids[int(round(y)), int(round(x))])
        if cid == 0:
            raise SystemExit(f"the click falls on id 0 (background) in {f}: click inside the object")
        area = {f2: float((load_ids(syn / f"{f2}.png") == cid).mean()) for f2 in frames}
        res = {f"{args.object}_id": cid, "method": f"click {args.click}", "frames_with_id": sum(a > 0 for a in area.values()),
               "mean_area_pct_visible": round(100 * float(np.mean([a for a in area.values() if a > 0] or [0])), 4)}
    elif args.ref_masks:
        ref_dir = Path(args.ref_masks)
        out3 = run / "step3_object_reference" / "masks"
        out3.mkdir(parents=True, exist_ok=True)
        inter, ref_px, id_px = {}, 0, {}
        used = 0
        for f in frames:
            rp = sorted(ref_dir.glob(f"{f}.*"))
            if not rp:
                continue
            ids = load_ids(syn / f"{f}.png")
            ref = load_ids(rp[0], ids.shape)
            m = (ref == args.ref_id) if args.ref_id is not None else (ref > 0)
            Image.fromarray((m * 255).astype(np.uint8)).save(out3 / f"{f}.png")
            used += 1
            ref_px += int(m.sum())
            u, c = np.unique(ids, return_counts=True)
            for i, n in zip(u.tolist(), c.tolist()):
                id_px[i] = id_px.get(i, 0) + n
            if m.any():
                u, c = np.unique(ids[m], return_counts=True)
                for i, n in zip(u.tolist(), c.tolist()):
                    inter[i] = inter.get(i, 0) + n
        if not used or not ref_px:
            raise SystemExit("no reference mask matched the synced frames (names must match)")
        cand = []
        for i, n in inter.items():
            if i == 0:
                continue
            cand.append({"id": i, "iou_total": round(n / (ref_px + id_px[i] - n), 4), "inter_px": n,
                         "precision_total": round(n / id_px[i], 4), "recall_total": round(n / ref_px, 4)})
        cand.sort(key=lambda d: -d["iou_total"])
        best = cand[0]
        res = {f"{args.object}_id": best["id"], "iou_total": best["iou_total"], "precision_total": best["precision_total"],
               "recall_total": best["recall_total"], "candidates_top5": cand[:5], "frames_used": used,
               "method": f"max IoU with {ref_dir}" + (f" (id {args.ref_id})" if args.ref_id is not None else "")}
        (run / "step3_object_reference" / "summary.json").write_text(json.dumps(
            {"source": str(ref_dir), "ref_id": args.ref_id, "frames": used}, indent=2))
    else:
        raise SystemExit("give --ref_masks or --click")
    (s6 / "object_id.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))
    print(f"\n-> {s6 / 'object_id.json'}\nconfigs/datasets/<dataset>.yaml: target_id: {res[f'{args.object}_id']}")


if __name__ == "__main__":
    main()
