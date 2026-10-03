"""
BUOC 2 + 3 cua Split: chay ban sao co chen log cua sam2/mask_propagation_scanet.py (tao boi
make_instrumented_propagation.py) dung nhu tac gia chay: cwd = thu muc co data/<scene>/ va output/,
argv = --scene <scene> (KHONG --verbose: co do mo cua so 3D va treo tren server).
Sau do tong hop 04_final/ va run_info.json.
"""
import argparse
import json
import os
import runpy
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REPO, scene_dirs, cap_vram

sys.path.insert(0, os.path.join(REPO, "sam2"))  # giong khi chay "python ./sam2/mask_propagation_scanet.py"

import matplotlib
matplotlib.use("Agg")
matplotlib.use = lambda *a, **k: None  # vo hieu matplotlib.use("TkAgg") cua code goc (server khong co cua so)

import torch

import split_hooks as H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="office0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--work", default=None)
    args = ap.parse_args()
    out_root, work = scene_dirs(args.scene)
    out_root, work = args.out or out_root, args.work or work

    cap_gb = cap_vram()
    script = os.path.join(work, "mask_propagation_scanet_debug.py")
    H.configure(out_root, work, args.scene)
    os.chdir(work)
    sys.argv = [script, "--scene", args.scene]

    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    runpy.run_path(script, init_globals={"_dbg": H}, run_name="__main__")
    t_prop = time.time() - t0
    summary = H.finalize()
    t_all = time.time() - t0

    info = {}
    p1 = os.path.join(out_root, "run_info_step1.json")
    if os.path.exists(p1):
        info.update(json.load(open(p1)))
        os.remove(p1)
    info["step2_3"] = {
        "seconds_propagation": round(t_prop, 1), "seconds_total": round(t_all, 1),
        "vram_pytorch_peak_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 2), "vram_cap_gib": cap_gb,
        "params_goc": {"EPSILON_depth_m": 0.02, "ACCURACY_LABELS": 0.7, "DOWNSAMPLE_points": 100000,
                       "dbscan_moi_mask": {"eps": 0.10, "min_samples": 10},
                       "dbscan_loc_nhan": {"eps": 0.05, "min_samples": 10}, "verbose": False},
        "ket_qua": summary,
    }
    info["du_lieu"] = {
        "scene": args.scene,
        "anh_depth": "Replica Test (250 frame, stride 8 tu 2000)",
        "pose_intrinsic_pointcloud": f"outputs/Replica/sfm/{args.scene}_gof/sparse/0 (SfM gia co san)",
        "data_dir": os.path.join(work, "data", args.scene),
    }
    info["code"] = {
        "goc": "Based_Model/Split_and_Splat (khong sua)",
        "buoc_1": "scripts/split_debug/run_step1_mask2d.py (goi ham cua sam2/auto_seg.py)",
        "buoc_2_3": f"{script} (ban sao, xem mask_propagation_scanet_debug.diff)",
    }
    with open(os.path.join(out_root, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print("[DONE] buoc 2+3", json.dumps(info["step2_3"], ensure_ascii=False))


if __name__ == "__main__":
    main()
