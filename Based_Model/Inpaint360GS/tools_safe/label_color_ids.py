"""
Write object ids onto the associated color masks (visualization only; same placement as tools/add_label_num_hqsam.py:
one label at the centroid of every external contour), with a smaller font and an outline so ids stay readable.

  python tools_safe/label_color_ids.py --step_dir output/teatime_napkin/step3_sync_mask --font_scale 0.3
Output: <step_dir>/associated_hqsam_color_num/ (a new _vK folder if it already exists).
"""

import argparse
import os

import cv2
import numpy as np


def next_free(path):
    if not os.path.exists(path):
        return path
    k = 2
    while os.path.exists(f"{path}_v{k}"):
        k += 1
    return f"{path}_v{k}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step_dir", required=True)
    parser.add_argument("--mask_dir", default="associated_hqsam")
    parser.add_argument("--color_dir", default="associated_hqsam_color")
    parser.add_argument("--font_scale", type=float, default=0.3)
    parser.add_argument("--min_area", type=int, default=20, help="skip contours smaller than this (pixels)")
    args = parser.parse_args()

    mask_dir = os.path.join(args.step_dir, args.mask_dir)
    color_dir = os.path.join(args.step_dir, args.color_dir)
    out_dir = next_free(os.path.join(args.step_dir, args.color_dir + "_num"))
    os.makedirs(out_dir)

    for name in sorted(os.listdir(color_dir)):
        stem = os.path.splitext(name)[0]
        mask_path = os.path.join(mask_dir, stem + ".png")
        image = cv2.imread(os.path.join(color_dir, name))
        if image is None or not os.path.exists(mask_path):
            continue
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        for label in np.unique(mask):
            if label == 0:
                continue
            contours, _ = cv2.findContours((mask == label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                m = cv2.moments(contour)
                if m["m00"] < args.min_area:
                    continue
                org = (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))
                text = str(label)
                cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, args.font_scale, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, args.font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, name), image)
    print("written:", out_dir, len(os.listdir(out_dir)), "images")


if __name__ == "__main__":
    main()
