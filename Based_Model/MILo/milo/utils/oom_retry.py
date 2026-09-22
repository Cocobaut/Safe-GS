"""
Chiu duoc CUDA OOM TAM THOI khi dung chung GPU voi nguoi khac (khong doi thuat toan/ket qua cua MILo).

Y tuong: khi 1 buoc bi OOM, giai phong bo nho tam, cho vai chuc giay cho nguoi khac nha VRAM, khoi phuc
DUNG trang thai dau buoc (RNG python/numpy/torch, hai danh sach camera, occupancy, mesh_state) roi lam lai
DUNG buoc do -> ket qua nhu chay lien mot mach.

  - oom_retry(fn): boc cac ham render/integrate THUAN (khong sua trang thai) -> lam lai truc tiep.
  - take_snapshot/restore_snapshot: cho ca doan "chon camera -> render -> loss -> mesh -> backward" trong train.py.
  - oom_inject: kiem thu (MILO_OOM_INJECT="8001:1,8500:1"): ep OOM gia o iteration chi dinh va in ra so sanh
    loss luc truoc/sau khi khoi phuc de chung minh lam lai cho ket qua giong.
"""
import functools
import gc
import os
import random
import time

import numpy as np
import torch

MAX_RETRIES = int(os.environ.get("MILO_OOM_MAX_RETRIES", "240"))     # x WAIT_SECONDS = ~2 gio
WAIT_SECONDS = int(os.environ.get("MILO_OOM_WAIT_SECONDS", "30"))


class OOMRetryExhausted(RuntimeError):
    pass


def is_oom(e):
    if isinstance(e, OOMRetryExhausted):
        return False
    return isinstance(e, torch.cuda.OutOfMemoryError) or (
        isinstance(e, RuntimeError) and "out of memory" in str(e).lower()
    )


def wait_for_gpu(attempt, what):
    gc.collect()
    torch.cuda.empty_cache()
    if attempt > MAX_RETRIES:
        raise OOMRetryExhausted(f"[OOM-RETRY] {what}: van thieu VRAM sau {MAX_RETRIES} lan cho, dung han.")
    free, total = torch.cuda.mem_get_info()
    print(f"[OOM-RETRY] {what}: CUDA OOM (lan {attempt}/{MAX_RETRIES}); VRAM trong {free / 2**20:.0f}/{total / 2**20:.0f} MiB; "
          f"cho {WAIT_SECONDS}s roi lam lai buoc nay (khong doi ket qua)", flush=True)
    time.sleep(WAIT_SECONDS)


def oom_retry(fn, name=None):
    """Boc ham THUAN (chi doc tham so, tra ket qua) de tu lam lai khi OOM."""
    what = name or getattr(fn, "__name__", "fn")

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        attempt = 0
        while True:
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if not is_oom(e):
                    raise
            # (ngoai khoi except de traceback duoc giai phong truoc khi cho)
            attempt += 1
            wait_for_gpu(attempt, what)

    return wrapped


# ---------------------------------------------------------------- snapshot cho doan forward+backward
def take_snapshot(gaussians, mesh_state, viewpoint_stack, viewpoint_idx_stack, with_occupancy):
    snap = {
        "rng": (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state()),
        "vs": None if viewpoint_stack is None else list(viewpoint_stack),
        "vis": None if viewpoint_idx_stack is None else list(viewpoint_idx_stack),
        "mesh_state": None if mesh_state is None else dict(mesh_state),
        "occ": None,
    }
    if with_occupancy and getattr(gaussians, "learn_occupancy", False):
        snap["occ"] = (gaussians._base_occupancy.detach().clone(), gaussians._occupancy_shift.detach().clone())
    return snap


def restore_snapshot(snap, gaussians, mesh_state):
    """Tra ve (viewpoint_stack, viewpoint_idx_stack); khoi phuc tai cho RNG, occupancy, mesh_state, grad."""
    random.setstate(snap["rng"][0])
    np.random.set_state(snap["rng"][1])
    torch.set_rng_state(snap["rng"][2])
    torch.cuda.set_rng_state(snap["rng"][3])
    if snap["occ"] is not None:
        gaussians._base_occupancy.data.copy_(snap["occ"][0])
        gaussians._occupancy_shift.data.copy_(snap["occ"][1])
    if mesh_state is not None and snap["mesh_state"] is not None:
        mesh_state.clear()
        mesh_state.update(snap["mesh_state"])
    # backward do dang co the da cong don 1 phan grad
    if getattr(gaussians, "optimizer", None) is not None:
        gaussians.optimizer.zero_grad(set_to_none=True)
    return snap["vs"], snap["vis"]


# ---------------------------------------------------------------- kiem thu bang OOM gia
_INJECT = None
_FIRST_LOSS = {}


def oom_inject(iteration, loss):
    """Neu MILO_OOM_INJECT='8001:1,8500:1': o cac iteration do, lan dau ep OOM gia (sau khi da tinh loss);
    lan lam lai in so sanh loss. Khong dat bien moi truong -> khong lam gi."""
    global _INJECT
    if _INJECT is None:
        _INJECT = {}
        for item in os.environ.get("MILO_OOM_INJECT", "").split(","):
            if item.strip():
                it, n = item.split(":")
                _INJECT[int(it)] = int(n)
    if iteration not in _INJECT:
        return False
    value = float(loss.detach().item())
    if _INJECT[iteration] > 0:
        _INJECT[iteration] -= 1
        _FIRST_LOSS.setdefault(iteration, value)
        print(f"[OOM-SELFTEST] iter {iteration}: loss lan dau = {value:.9f} -> ep OOM gia", flush=True)
        return True
    print(f"[OOM-SELFTEST] iter {iteration}: loss sau khi khoi phuc+lam lai = {value:.9f} | lan dau = "
          f"{_FIRST_LOSS[iteration]:.9f} | lech = {abs(value - _FIRST_LOSS[iteration]):.3e}", flush=True)
    return False
