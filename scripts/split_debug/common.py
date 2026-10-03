"""Phan dung chung cho bo debug Split: duong dan, bang mau co dinh theo ID, gioi han VRAM."""
import os

import numpy as np

REPO = "/media/ml4u/Extreme SSD/Safe-GS/Based_Model/Split_and_Splat"
SAFE_GS = "/media/ml4u/Extreme SSD/Safe-GS"
SPLIT_ROOT = os.path.join(SAFE_GS, "outputs/Replica/split/split_and_splat")


def scene_dirs(scene):
    out = os.path.join(SPLIT_ROOT, scene)
    work = os.path.join(SPLIT_ROOT, "_work", scene)
    return out, work


def id_color(i):
    """Mau co dinh cho 1 ID (khong dung RNG toan cuc cua numpy -> khong anh huong ket qua code goc)."""
    rng = np.random.default_rng(int(i) * 7919 + 12345)
    return tuple(int(x) for x in rng.integers(40, 256, size=3))


def cap_vram():
    """Chan bo cap phat PyTorch o SPLIT_VRAM_CAP_GB (mac dinh 6.0 GiB). Cong phan nen CUDA (~0.5-0.7GB) van < 7GB.
    Vuot nguong thi chinh tien trinh nay bao OOM, khong lan sang nguoi khac."""
    import torch
    cap_gb = float(os.environ.get("SPLIT_VRAM_CAP_GB", "6.0"))
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap_gb * 1024**3 / total), 0)
    print(f"[VRAM] gioi han bo cap phat PyTorch = {cap_gb:.1f} GiB / {total / 1024**3:.1f} GiB", flush=True)
    return cap_gb
