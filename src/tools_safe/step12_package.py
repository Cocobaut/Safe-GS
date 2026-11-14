"""Step 12 (CPU): collect the train / test renders of the removed (step 8) and inpainted (step 11) scenes of one run.

Sources inside the model folder <M> (Inpaint360GS layout):
    <M>/{train,test}/ours_object_removal/iteration_<N>/{gt,renders}        edit_object_removal.py --render_video
    <M>/{train,test}/ours_object_inpaint_virtual/iteration_<K>/renders     edit_object_inpaint.py --render_video
    <M>/video/ours_*object_removal*/final_video.mp4, ours_*object_inpaint*/final_video.mp4
Output (layout of the reference run outputs/LERF/teatime_napkin/step12_render):
    <out>/{train,test}/{0_gt,1_removed,2_inpainted,3_gt_removed_inpainted}/<name>.png
    <out>/video_1_removed.mp4, video_2_inpainted.mp4, summary.json (counts, videos, from_step11, note)

  python tools_safe/step12_package.py -m <M> --out <run>/step12_render [--has_gt_object_free]
"""
from __future__ import annotations

import json
import shutil
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np


def newest(globbed):
    c = [p for p in globbed if p.is_dir() and "removal_target" not in str(p)]
    return max(c, key=lambda p: p.stat().st_mtime) if c else None


def main():
    ap = ArgumentParser()
    ap.add_argument("-m", "--model_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--has_gt_object_free", action="store_true", help="dataset has object-free GT (Inpaint360GS)")
    ap.add_argument("--mask_dir", default=None, help="Inpaint360GS <data>/unseen_mask (masks of the inpainted region)")
    args = ap.parse_args()
    M, out = Path(args.model_path), Path(args.out)
    counts, src = {}, {}
    for split in ("train", "test"):
        rem = newest((M / split).glob("ours*object_removal*/iteration_*"))
        inp = newest((M / split).glob("ours*object_inpaint*/iteration_*"))
        if rem is None or inp is None:
            print(f"[step12] {split}: missing removal ({rem}) or inpaint ({inp}) renders")
            continue
        src[split] = {"removed": str(rem), "inpainted": str(inp)}
        d = {k: out / split / k for k in ("0_gt", "1_removed", "2_inpainted", "3_gt_removed_inpainted")}
        for p in d.values():
            p.mkdir(parents=True, exist_ok=True)
        n = {"gt": 0, "removed": 0, "inpainted": 0}
        for r in sorted((rem / "renders").glob("*.png")):
            g, i = rem / "gt" / r.name, inp / "renders" / r.name
            shutil.copy2(r, d["1_removed"] / r.name); n["removed"] += 1
            tiles = []
            if g.exists():
                shutil.copy2(g, d["0_gt"] / r.name); n["gt"] += 1
                tiles.append(cv2.imread(str(g)))
            tiles.append(cv2.imread(str(r)))
            if i.exists():
                shutil.copy2(i, d["2_inpainted"] / r.name); n["inpainted"] += 1
                tiles.append(cv2.imread(str(i)))
            h = min(t.shape[0] for t in tiles)
            sep = np.full((h, 8, 3), 255, np.uint8)
            row = [x for t in tiles for x in (t[:h], sep)][:-1]
            cv2.imwrite(str(d["3_gt_removed_inpainted"] / r.name), np.concatenate(row, 1))
        counts[split] = n
    # Inpaint360GS: the "inpaint" cameras (object-free photos, *test* images) rendered by edit_object_inpaint.py
    inp = newest((M / "inpaint").glob("ours*object_inpaint*/iteration_*")) if (M / "inpaint").exists() else None
    if inp is not None:
        src["inpaint"] = {"inpainted": str(inp)}
        d0, d2 = out / "inpaint" / "0_gt", out / "inpaint" / "2_inpainted"
        d0.mkdir(parents=True, exist_ok=True); d2.mkdir(parents=True, exist_ok=True)
        n = {"gt": 0, "inpainted": 0}
        for r in sorted((inp / "renders").glob("*.png")):
            shutil.copy2(r, d2 / r.name); n["inpainted"] += 1
            if (inp / "gt" / r.name).exists():
                shutil.copy2(inp / "gt" / r.name, d0 / r.name); n["gt"] += 1
        if args.mask_dir and Path(args.mask_dir).exists():          # unseen-region masks shipped with the dataset
            shutil.copytree(args.mask_dir, out / "inpaint" / "mask", dirs_exist_ok=True)
            n["mask"] = len(list((out / "inpaint" / "mask").glob("*")))
        counts["inpaint"] = n
    elif args.has_gt_object_free:
        print("[step12] no <M>/inpaint renders although the dataset has object-free GT")
    videos = {}
    for name, pat in (("video_1_removed.mp4", "*object_removal*"), ("video_2_inpainted.mp4", "*object_inpaint*")):
        cands = sorted((M / "video").glob(f"{pat}/**/final_video.mp4"), key=lambda p: p.stat().st_mtime) + \
            sorted((M / "video").glob(f"{pat}/final_video.mp4"), key=lambda p: p.stat().st_mtime)
        if cands:
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cands[-1], out / name)
        videos[name] = bool(cands)
    summary = {"counts": counts, "videos": videos, "from_step11": "step11_inpaint_3dgs_come_loss", "sources": src,
               "note": ("object-free GT available: run the metrics (src/tools/metrics_fid_masked.py)"
                        if args.has_gt_object_free else
                        "no ground truth without the object: no metrics, visual check only")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
