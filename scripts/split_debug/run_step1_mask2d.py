"""
BUOC 1 cua Split (tao mask 2D ban dau) - chay DUNG logic cua sam2/auto_seg.py (getMask) nhung ghi them ket qua
trung gian vao <out>/01_mask_2d/. KHONG sua code goc: import thang ham MaskMerging / useless_mask cua tac gia,
4 cau hinh SAM2 chep nguyen van tu khoi __main__ cua auto_seg.py.

Khac biet (deu khong doi ket qua):
  - auto_seg.py dung 2 ban SAM2 nhung ban apply_postprocessing=True khong duoc cau hinh nao dung -> chi nap 1 ban
    (do ton VRAM).
  - mask cuoi luu dang nhi phan 0/255 thay vi anh to mau ngau nhien; buoc 3 cua tac gia doc mask bang
    convert("L") > 0 nen tuong duong.
  - khong doi ten file anh tai cho (auto_seg.py tu doi ten anh goc thanh .JPEG; du lieu cua minh da dat ten san).
"""
import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REPO, scene_dirs, id_color, cap_vram

sys.path.insert(0, os.path.join(REPO, "sam2"))  # giong khi chay "python ./sam2/auto_seg.py"

import numpy as np
import torch
from PIL import Image

cap_gb = cap_vram()
import auto_seg  # side effect giong ban goc: autocast bfloat16, tf32, np.random.seed(3)
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def save_bin(mask, path):
    Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255).save(path)


def overlay(img, masks):
    """Anh RGB + mask to mau (mask lon ve truoc, nho de len). ID chi co nghia trong frame nay."""
    out = img.astype(np.float32) * 0.45
    order = sorted(range(len(masks)), key=lambda k: -int(np.count_nonzero(masks[k])))
    color_layer = np.zeros_like(out)
    covered = np.zeros(img.shape[:2], bool)
    for k in order:
        m = np.asarray(masks[k]) > 0
        color_layer[m] = id_color(k)
        covered |= m
    out[covered] = out[covered] + color_layer[covered] * 0.55
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="office0")
    ap.add_argument("--max_frames", type=int, default=0, help="0 = tat ca; >0 de chay thu")
    ap.add_argument("--out", default=None, help="mac dinh <SPLIT_ROOT>/<scene>")
    args = ap.parse_args()

    out_root, work = scene_dirs(args.scene)
    out_root = args.out or out_root
    d1 = os.path.join(out_root, "01_mask_2d")
    step3_input = os.path.join(work, "output", f"{args.scene}_autoseg_mask")  # dau vao buoc 3 (duong dan goc)
    image_dir = os.path.join(work, "data", args.scene, "images")

    ckpt = os.path.join(REPO, "checkpoints", "sam2.1_hiera_large.pt")
    sam2 = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", ckpt, device="cuda", apply_postprocessing=False)

    # ---- chep nguyen van tu auto_seg.py ----
    configs = [
        {"name": "uber_huge", "model": sam2, "points_per_side": 1, "points_per_batch": 1, "pred_iou_thresh": 0.8,
         "stability_score_thresh": 0.85, "stability_score_offset": 0.85, "mask_threshold": 0.5, "crop_n_layers": 1,
         "box_nms_thresh": 0.3, "use_m2m": False},
        {"name": "very_coarse", "model": sam2, "points_per_side": 4, "points_per_batch": 4, "pred_iou_thresh": 0.8,
         "stability_score_thresh": 0.9, "stability_score_offset": 0.85, "mask_threshold": 0.5, "crop_n_layers": 1,
         "box_nms_thresh": 0.3, "use_m2m": False},
        {"name": "coarse", "model": sam2, "points_per_side": 8, "points_per_batch": 8, "pred_iou_thresh": 0.8,
         "stability_score_thresh": 0.85, "stability_score_offset": 0.85, "mask_threshold": 0.3, "crop_n_layers": 1,
         "box_nms_thresh": 0.5, "use_m2m": False},
        {"name": "fine", "model": sam2, "points_per_side": 16, "points_per_batch": 64, "pred_iou_thresh": 0.8,
         "stability_score_thresh": 0.9, "stability_score_offset": 1, "mask_threshold": 0.3, "crop_n_layers": 1,
         "box_nms_thresh": 0.3, "use_m2m": False},
    ]
    cfg_names = [c["name"] for c in configs]

    frames = sorted(f for f in os.listdir(image_dir) if f.endswith(".JPEG"))
    if args.max_frames > 0:
        frames = frames[: args.max_frames]

    for sub in ["per_config", "merged", "filtered", "overlay"]:
        os.makedirs(os.path.join(d1, sub), exist_ok=True)
    stats_path = os.path.join(d1, "stats.csv")
    stats_f = open(stats_path, "w", newline="")
    stats = csv.writer(stats_f)
    stats.writerow(["frame", *[f"so_mask_{n}" for n in cfg_names], "so_mask_sau_gop", "so_mask_sau_loc",
                    "dien_tich_min_pct", "dien_tich_max_pct", "dien_tich_tb_pct", "giay"])

    t_all = time.time()
    torch.cuda.reset_peak_memory_stats()
    for fi, fname in enumerate(frames):
        t0 = time.time()
        frame = fname.replace(".JPEG", "")
        image = np.array(Image.open(os.path.join(image_dir, fname)).convert("RGB"))
        H, W = image.shape[:2]

        # ---- getMask(preload=False) cua tac gia ----
        generated_mask, per_cfg = [], {}
        for cfg in configs:
            mask_generator = SAM2AutomaticMaskGenerator(**{k: v for k, v in cfg.items() if k != "name"})
            try:
                masks1 = mask_generator.generate(image)
                masks = [m["segmentation"].astype(np.uint8) for m in masks1]  # = save_mask(...)
                generated_mask.append(masks)
                per_cfg[cfg["name"]] = masks
            except Exception as e:
                print("NO valid mask!", e)
                per_cfg[cfg["name"]] = []
                continue

        final_masks, merged = [], []
        if len(generated_mask) > 0:
            for i in range(len(generated_mask)):
                if i == 0:
                    sorted_masks_0 = sorted(list(generated_mask[i]), key=lambda x: np.sum(x), reverse=True)
                    final_masks = auto_seg.useless_mask(sorted_masks_0)
                else:
                    final_masks = auto_seg.MaskMerging(final_masks, generated_mask[i])
            merged = list(final_masks)
            final_masks = auto_seg.useless_mask(final_masks)

        # ---- ghi ket qua trung gian ----
        for name in cfg_names:
            dd = os.path.join(d1, "per_config", frame, name)
            os.makedirs(dd, exist_ok=True)
            for k, m in enumerate(per_cfg[name]):
                save_bin(m, os.path.join(dd, f"{k}.png"))
        for sub, ms in [("merged", merged), ("filtered", final_masks)]:
            dd = os.path.join(d1, sub, frame)
            os.makedirs(dd, exist_ok=True)
            for k, m in enumerate(ms):
                save_bin(m, os.path.join(dd, f"{k}.png"))
        # dau vao buoc 3: dung cho goc ./output/<scene>_autoseg_mask/<frame>/<id>.png (luon tao thu muc frame)
        dd = os.path.join(step3_input, frame)
        os.makedirs(dd, exist_ok=True)
        for k, m in enumerate(final_masks):
            save_bin(m, os.path.join(dd, f"{k}.png"))
        Image.fromarray(overlay(image, final_masks)).save(os.path.join(d1, "overlay", f"{frame}.png"))

        areas = [100.0 * np.count_nonzero(m) / (H * W) for m in final_masks] or [0.0]
        dt = time.time() - t0
        stats.writerow([frame, *[len(per_cfg[n]) for n in cfg_names], len(merged), len(final_masks),
                        f"{min(areas):.3f}", f"{max(areas):.3f}", f"{np.mean(areas):.3f}", f"{dt:.1f}"])
        stats_f.flush()
        peak = torch.cuda.max_memory_reserved() / 1024**3
        print(f"[{fi + 1}/{len(frames)}] frame {frame}: {len(final_masks)} mask, {dt:.1f}s, "
              f"VRAM PyTorch dinh {peak:.2f} GiB", flush=True)

        del image, generated_mask, per_cfg, merged, final_masks
        torch.cuda.empty_cache()

    stats_f.close()
    info = {"step1": {"frames": len(frames), "seconds": round(time.time() - t_all, 1),
                      "vram_pytorch_peak_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
                      "vram_cap_gib": cap_gb, "sam2_checkpoint": "sam2.1_hiera_large.pt",
                      "configs": [{k: v for k, v in c.items() if k != "model"} for c in configs]}}
    with open(os.path.join(out_root, "run_info_step1.json"), "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print("[DONE] buoc 1", json.dumps(info["step1"], ensure_ascii=False)[:300])


if __name__ == "__main__":
    main()
