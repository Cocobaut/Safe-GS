"""
Check whether an existing folder of 2D SAM masks can be used as `raw_hqsam` for Inpaint360GS.

The conditions follow the input contract of seg/mask_associate.py and the paper (Supp. B: HQ-SAM raw masks):
  C1  masks were produced by HQ-SAM / CropFormer (not SAM2 / video tracking)
  C2  exactly one <image_stem>.png per image in images_dir
  C3  single channel uint8, same H x W as the image
  C4  per-frame ids are contiguous {0} U {1..N}  (remap_mask asserts len(labels) == mask.max())

Exit code 0 = usable (copy), 1 = not usable (rerun seg/raw_mask_sam.py --method hqsam).

  python tools_safe/check_sam_masks.py --mask_dir "F:/LERF Datasets/teatime/teatime/mask" \
      --images_dir "F:/LERF Datasets/teatime/teatime/images_2" --report report.json
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def find_label_dir(mask_dir):
    labels = os.path.join(mask_dir, "labels")
    return labels if os.path.isdir(labels) else mask_dir


def check_generator(mask_dir):
    """C1: look for metadata written next to the masks."""
    sam2_cfg = os.path.join(mask_dir, "sam2_config.json")
    if os.path.exists(sam2_cfg):
        with open(sam2_cfg, "r", encoding="utf-8") as f:
            model = json.load(f).get("model", "sam2")
        return False, f"sam2_config.json found (model={model}); paper uses HQ-SAM/CropFormer"
    for name in ("generator.json", "scene.json"):
        meta = os.path.join(mask_dir, name)
        if os.path.exists(meta):
            with open(meta, "r", encoding="utf-8") as f:
                text = f.read().lower()
            if "hqsam" in text or "cropformer" in text:
                return True, f"{name} says hqsam"
    return False, "no metadata proving the masks come from HQ-SAM"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    label_dir = find_label_dir(args.mask_dir)
    images = sorted(f for f in os.listdir(args.images_dir) if f.lower().endswith(IMAGE_EXTS))
    masks = sorted(f for f in os.listdir(label_dir) if f.lower().endswith(".png"))
    image_stems = {os.path.splitext(f)[0]: f for f in images}
    mask_stems = {os.path.splitext(f)[0] for f in masks}

    report = {"mask_dir": args.mask_dir, "label_dir": label_dir, "images_dir": args.images_dir,
              "num_images": len(images), "num_masks": len(masks), "conditions": {}}

    ok, msg = check_generator(args.mask_dir)
    report["conditions"]["C1_generator_hqsam"] = {"pass": ok, "detail": msg}

    missing = sorted(set(image_stems) - mask_stems)
    extra = sorted(mask_stems - set(image_stems))
    report["conditions"]["C2_one_mask_per_image"] = {
        "pass": not missing and not extra,
        "detail": f"missing={len(missing)} extra={len(extra)}",
        "missing_examples": missing[:5], "extra_examples": extra[:5]}

    bad_shape, bad_ids = [], []
    for stem in sorted(set(image_stems) & mask_stems):
        mask = cv2.imread(os.path.join(label_dir, stem + ".png"), cv2.IMREAD_UNCHANGED)
        img = cv2.imread(os.path.join(args.images_dir, image_stems[stem]))
        if mask is None or mask.ndim != 2 or mask.dtype != np.uint8 or mask.shape != img.shape[:2]:
            got = None if mask is None else f"{mask.shape} {mask.dtype}"
            bad_shape.append({"frame": stem, "mask": got, "image": list(img.shape[:2])})
        if mask is not None:
            ids = np.unique(mask)
            ids = ids[ids != 0]
            if len(ids) and (ids.min() != 1 or ids.max() != len(ids)):
                bad_ids.append({"frame": stem, "num_ids": int(len(ids)), "max_id": int(ids.max())})

    report["conditions"]["C3_uint8_same_size"] = {
        "pass": not bad_shape, "detail": f"{len(bad_shape)} frames mismatch", "examples": bad_shape[:3]}
    report["conditions"]["C4_contiguous_ids_per_frame"] = {
        "pass": not bad_ids, "detail": f"{len(bad_ids)} frames non-contiguous", "examples": bad_ids[:3]}

    report["usable"] = all(c["pass"] for c in report["conditions"].values())
    report["decision"] = "copy existing masks to raw_hqsam" if report["usable"] else "rerun seg/raw_mask_sam.py --method hqsam"

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(text)
    sys.exit(0 if report["usable"] else 1)


if __name__ == "__main__":
    main()
