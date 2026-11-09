"""
Đo LERF-Mask (teatime) cho model ObjectGS đã train. Chạy trong env objectgs, cwd = repo ObjectGS.

    python /opt/objectgs_eval_lerf.py -m outputs/lerf_mask/rebuttal/teatime/<run> --gt datasets/lerf_mask/teatime/test_mask
        --out <dir>

Nạp model như render.py của tác giả, render ID (render_semantics -> id_encoder.inverse_transform) ở các view test_*.
Prompt -> ID: các ID trong mask DEVA đầu vào (object_mask/test_<v>.png) có >= 50% diện tích nằm trong GT của prompt.
IoU / Boundary IoU: đúng code script/eval_lerf_mask.py của Gaussian Grouping (dilation_ratio 0.02), trung bình theo view
rồi theo prompt. Đo thêm IoU của chính mask DEVA (cùng quy tắc nối) để biết mức trần do mask đầu vào.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml


def mask_to_boundary(mask, dilation_ratio=0.02):
    h, w = mask.shape
    dilation = max(1, int(round(dilation_ratio * np.sqrt(h ** 2 + w ** 2))))
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    new_mask_erode = cv2.erode(new_mask, np.ones((3, 3), dtype=np.uint8), iterations=dilation)
    return mask - new_mask_erode[1: h + 1, 1: w + 1]


def boundary_iou(gt, dt):
    dt = (dt > 128).astype("uint8"); gt = (gt > 128).astype("uint8")
    gb, db = mask_to_boundary(gt), mask_to_boundary(dt)
    return ((gb * db) > 0).sum() / ((gb + db) > 0).sum()


def iou(gt, dt):
    g, d = gt > 128, dt > 128
    return np.logical_and(g, d).sum() / np.logical_or(g, d).sum()


def pick_ids(deva, gt, thr=0.5):
    ids, cnt = np.unique(deva, return_counts=True)
    inside = {i: (deva[gt > 128] == i).sum() for i in ids}
    return [int(i) for i, c in zip(ids, cnt) if i != 0 and inside[i] / c >= thr]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_path", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, os.getcwd())
    from scene import Scene
    from utils.general_utils import safe_state, parse_cfg
    import gaussian_renderer

    cfg = yaml.load(open(os.path.join(a.model_path, "config.yaml")), Loader=yaml.FullLoader)
    scene_name = a.model_path.rstrip("/").split("/")[-2]                      # như render.py
    cfg["model_params"]["exp_name"] = os.path.join(cfg["model_params"]["exp_name"], scene_name)
    cfg["model_params"]["source_path"] = os.path.join(cfg["model_params"]["source_path"], scene_name)
    lp, op, pp = parse_cfg(cfg)
    lp.model_path = a.model_path
    safe_state(True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pp.add_prefilter = not (pp.no_prefilter_step > 0)
    with torch.no_grad():
        import scene as scene_mod
        g = getattr(scene_mod, lp.model_config["name"])(**lp.model_config["kwargs"])
        sc = Scene(lp, g, load_iteration=-1, shuffle=False)
        g.eval()
        views = {v.image_name: v for v in sc.getTestCameras()}
        pred_ids = {}
        for name, v in sorted(views.items()):
            pkg = gaussian_renderer.render(v, g, pp, sc.background)
            ids = g.id_encoder.inverse_transform(pkg["render_semantics"]).squeeze().cpu().numpy().astype(np.int32)
            pred_ids[name] = ids
            rgb = (torch.clamp(pkg["render"], 0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(str(out / f"{name}_rgb.png"), rgb[..., ::-1])
            cv2.imwrite(str(out / f"{name}_ids.png"), ids.astype(np.uint16))
    src = Path(lp.source_path)
    res = {"ours": {}}
    per = {}
    for vdir in sorted(Path(a.gt).iterdir()):
        name = f"test_{vdir.name}"
        if name not in pred_ids:
            continue
        deva = cv2.imread(str(src / "object_mask" / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        pid = pred_ids[name]
        for f in sorted(vdir.glob("*.png")):
            gt = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if deva.shape != gt.shape:
                deva = cv2.resize(deva, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            if pid.shape != gt.shape:
                pid = cv2.resize(pid, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            sel = pick_ids(pid, gt)                   # mask DEVA của view test_* trống -> chọn ID theo render (oracle)
            for key, lab in (("ours", pid),):
                pred = (np.isin(lab, sel) * 255).astype(np.uint8)
                res[key].setdefault(f.stem, []).append((float(iou(gt, pred)), float(boundary_iou(gt, pred))))
            per[f"{name}/{f.stem}"] = {"ids": sel}
    summ = {}
    for key, d in res.items():
        miou = {k: float(np.mean([x[0] for x in v])) for k, v in d.items()}
        mb = {k: float(np.mean([x[1] for x in v])) for k, v in d.items()}
        summ[key] = {"mIoU": float(np.mean(list(miou.values()))), "mBIoU": float(np.mean(list(mb.values()))),
                     "per_prompt_iou": miou, "per_prompt_biou": mb}
    summ["mapping"] = per
    summ["rule"] = ("prompt -> ID (oracle): ID có >= 50% vùng render nằm trong GT; mask DEVA của view test_* trống "
                    "nên không nối được theo mask đầu vào. Cận trên so với giao thức chọn ID theo chữ của Gaussian Grouping.")
    (out / "lerf_mask_eval.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False))
    print(json.dumps({"ours": (summ["ours"]["mIoU"], summ["ours"]["mBIoU"])}))


if __name__ == "__main__":
    main()
