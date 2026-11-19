"""
Hoàn thiện vật bằng TRELLIS (microsoft/TRELLIS, commit 193e0502, trọng số microsoft/TRELLIS-image-large, MIT) theo ý Points-to-3D
(arXiv 2603.18782): khung đã quan sát làm khởi đầu trong không gian ẩn cấu trúc thưa, chỉ để mạng sinh phần CHƯA thấy.

  1. Lưới 64^3 (tools/prep_trellis_inputs.py): frame / seen / unknown / free.
  2. z0 = SS-encoder(frame) (latent 8 x 16^3).
  3. Lấy mẫu flow từ t0 (SDEdit: x = (1 - t0) z0 + sigma(t0) eps) về 0, có CFG theo nhiều ảnh (trung bình dự đoán theo từng ảnh);
     RePaint trong latent: ô latent mà cả khối 4^3 ô lưới đều ĐÃ BIẾT (seen hoặc free) bị đặt lại = z0 nhiễu đúng mức t ở mỗi bước,
     nên mạng chỉ sinh ở chỗ có ô chưa biết.
  4. occ = decoder(z) > 0; ép quan sát: occ = (occ AND NOT free) OR seen.
  5. Tầng 2 (SLAT) có điều kiện theo các ảnh trên cấu trúc đó -> Gaussian + mesh (hệ [-0.5, 0.5]^3, z lên).
Xuất mỗi vật: occ.npz (occ_pred, occ_final, đếm), gaussian.ply (hệ chuẩn hoá), mesh.npz, report.json

    python -m modal deploy modal_trellis.py
    python tools/modal_spawn.py safe-gs-trellis complete '"run1"' '"coc_su,tao"' 0.6
"""
import json
from pathlib import Path

import modal

app = modal.App("safe-gs-trellis")
VOL = "/vol"
vol = modal.Volume.from_name("safe-gs-come")
TRELLIS_COMMIT = "193e0502b5cb3eb5997ec457d5340e95d09b158e"

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0", "build-essential", "ninja-build", "libegl1", "libgles2", "libx11-6", "libxrender1", "libxext6", "libgomp1", "libusb-1.0-0", "libsm6", "libxi6", "libxxf86vm1", "libxfixes3", "libxkbcommon0", "ca-certificates")
    .pip_install("torch==2.4.0", "torchvision==0.19.0", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("xformers==0.0.27.post2", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("pillow", "imageio", "imageio-ffmpeg", "tqdm", "easydict", "opencv-python-headless", "scipy", "ninja",
                 "trimesh", "xatlas", "pyvista", "pymeshfix", "igraph", "transformers==4.49.0", "safetensors", "huggingface_hub",
                 "plyfile", "spconv-cu120", "rembg", "onnxruntime", "wheel", "setuptools<70", "open3d")
    .pip_install("git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8")
    .pip_install("kaolin", find_links="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html")
    .run_commands(
        f"git clone https://github.com/microsoft/TRELLIS.git /opt/TRELLIS && cd /opt/TRELLIS && git checkout {TRELLIS_COMMIT} && git submodule update --init --recursive",
        "git clone https://github.com/autonomousvision/mip-splatting.git /tmp/mip && "
        "CC=gcc CXX=g++ TORCH_CUDA_ARCH_LIST='8.0;8.6;8.9' pip install --no-build-isolation /tmp/mip/submodules/diff-gaussian-rasterization/",
    )
    .pip_install("certifi")
    .env({"ATTN_BACKEND": "xformers", "SPCONV_ALGO": "native", "PYTHONPATH": "/opt/TRELLIS", "HF_HOME": "/root/hf_cache"})
    .run_commands("cd /opt/TRELLIS && python -c 'import open3d, utils3d, spconv, xformers, kaolin; "
                  "from trellis.pipelines import TrellisImageTo3DPipeline; from trellis import models; "
                  "import torch; from transformers import Dinov2WithRegistersModel, Dinov2WithRegistersConfig; "
                  "m = Dinov2WithRegistersModel(Dinov2WithRegistersConfig(num_hidden_layers=1)); print(1)'")
)


@app.function(image=image, gpu=["A100-40GB", "L40S"], volumes={VOL: vol}, timeout=3 * 3600, cpu=8, memory=32768)
def complete(tag: str, objects: str = "", t0: float = 0.6, steps: int = 25, cfg: float = 5.0, seed: int = 1,
             repaint: bool = True, in_rel: str = "teatime/trellis_in", init: str = "frame", free_mode: str = "all"):
    """init: 'frame' = khởi đầu từ cả khung; 'seen' = chỉ bề mặt camera thấy (kiểu Points-to-3D).
    free_mode: 'all' = mọi ô ngoài khung là trống; 'carved' = chỉ ô camera nhìn xuyên + dưới mặt bàn là trống, còn lại để mạng quyết."""
    import numpy as np
    import torch
    from PIL import Image
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis import models
    from trellis.pipelines.samplers import FlowEulerSampler

    vol.reload()
    # DINOv2 của TRELLIS tải qua torch.hub từ dl.fbaipublicfiles.com (chứng chỉ SSL hết hạn 2026-11-11). Dùng cùng mô hình
    # DINOv2-L/14 có 4 register từ Hugging Face (facebook/dinov2-with-registers-large); TRELLIS cần 'x_prenorm' = đầu ra khối
    # cuối TRƯỚC layernorm cuối = hidden_states[-1] của transformers; thứ tự token giống nhau (cls, 4 register, patch).
    import torch.nn as nn
    import torchvision.transforms as T
    from transformers import Dinov2WithRegistersModel
    import trellis.pipelines.trellis_image_to_3d as TI

    class _Dino(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = Dinov2WithRegistersModel.from_pretrained("facebook/dinov2-with-registers-large").eval()

        def forward(self, x, is_training=True):
            return {"x_prenorm": self.m(pixel_values=x, output_hidden_states=True).hidden_states[-1]}

    def _init_cond(self, name):
        self.models["image_cond_model"] = _Dino().eval()
        self.image_cond_model_transform = T.Compose([T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    TI.TrellisImageTo3DPipeline._init_image_cond_model = _init_cond
    root = Path(VOL) / in_rel
    out_root = Path(VOL) / "teatime" / "trellis_out" / tag
    out_root.mkdir(parents=True, exist_ok=True)
    pipe = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipe.cuda()
    enc = models.from_pretrained("microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16").cuda().eval()
    flow = pipe.models["sparse_structure_flow_model"]
    dec = pipe.models["sparse_structure_decoder"]
    sampler = pipe.sparse_structure_sampler
    smin = sampler.sigma_min
    names = [p.name for p in sorted(root.iterdir()) if (p / "grid.npz").exists()]
    if objects:
        names = [n for n in names if n in objects.split(",")]
    rep_all = {}
    for n in names:
        od = out_root / n
        if (od / "report.json").exists():
            rep_all[n] = json.loads((od / "report.json").read_text()); continue
        od.mkdir(exist_ok=True)
        z = np.load(root / n / "grid.npz")
        frame, seen, free, unknown = (torch.tensor(z[k], device="cuda") for k in ("frame", "seen", "free", "unknown"))
        if free_mode == "carved":
            free = torch.tensor(z["carved"] | z["below"], device="cuda")
            unknown = ~(seen | free)                                  # mọi ô chưa chắc chắn đều để mạng quyết
        x_init = frame if init == "frame" else seen
        imgs = [Image.open(p) for p in sorted((root / n).glob("view_*.png"))]
        imgs = [pipe.preprocess_image(im) for im in imgs]
        with torch.no_grad():
            cond = pipe.get_cond(imgs)
        conds, neg = cond["cond"], cond["neg_cond"][:1]
        torch.manual_seed(seed)
        with torch.no_grad():
            z0 = enc(x_init.float()[None, None], sample_posterior=False)          # [1, 8, 16, 16, 16]
            known = (~unknown).float()
            known_lat = torch.nn.functional.avg_pool3d(known[None, None], 4)[0, 0] > 0.999   # cả khối 4^3 đã biết
            eps = torch.randn_like(z0)
            sig = lambda t: smin + (1 - smin) * t
            x = (1 - t0) * z0 + sig(t0) * eps
            t_seq = np.linspace(1, 0, steps + 1)
            rs = 3.0
            t_seq = rs * t_seq / (1 + (rs - 1) * t_seq)
            t_seq = [t for t in t_seq if t <= t0 + 1e-9]
            if t_seq[0] < t0:
                t_seq = [t0] + t_seq
            for t, tp in zip(t_seq[:-1], t_seq[1:]):
                vs = [FlowEulerSampler._inference_model(sampler, flow, x, t, conds[i:i + 1]) for i in range(len(conds))]
                v = sum(vs) / len(vs)
                if 0.5 <= t <= 1.0:
                    vn = FlowEulerSampler._inference_model(sampler, flow, x, t, neg)
                    v = (1 + cfg) * v - cfg * vn
                x = x - (t - tp) * v
                if repaint and tp > 0:
                    xk = (1 - tp) * z0 + sig(tp) * torch.randn_like(z0)
                    x = torch.where(known_lat[None, None], xk, x)
            if repaint:
                x = torch.where(known_lat[None, None], z0, x)
            occ_pred = dec(x)[0, 0] > 0
            occ = (occ_pred & ~free) | seen
            coords = torch.argwhere(occ)
            coords = torch.cat([torch.zeros(len(coords), 1, dtype=coords.dtype, device="cuda"), coords], 1).int()
        stats = {"t0": t0, "repaint": repaint, "init": init, "free_mode": free_mode, "frame": int(frame.sum()), "seen": int(seen.sum()), "unknown": int(unknown.sum()),
                 "pred": int(occ_pred.sum()), "final": int(occ.sum()),
                 "pred_in_free_removed": int((occ_pred & free).sum()),
                 "unknown_filled": int((occ & unknown).sum()), "unknown_dropped": int((~occ & unknown).sum()),
                 "added_outside_frame": int((occ & ~frame).sum()),
                 "iou_with_frame": float((occ & frame).sum() / max(int((occ | frame).sum()), 1)),
                 "known_latent_cells": int(known_lat.sum())}
        np.savez_compressed(od / "occ.npz", occ_pred=occ_pred.cpu().numpy(), occ_final=occ.cpu().numpy())
        # tầng 2: SLAT theo nhiều ảnh trên cấu trúc đã ép quan sát
        with torch.no_grad(), pipe.inject_sampler_multi_image("slat_sampler", len(imgs), 25, mode="multidiffusion"):
            slat = pipe.sample_slat({"cond": conds, "neg_cond": neg}, coords, {})
            outs = pipe.decode_slat(slat, ["gaussian", "mesh"])
        gs = outs["gaussian"][0]
        gs.save_ply(str(od / "gaussian.ply"), transform=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        mesh = outs["mesh"][0]
        np.savez_compressed(od / "mesh.npz", v=mesh.vertices.detach().cpu().numpy(), f=mesh.faces.detach().cpu().numpy())
        stats["gaussians"] = int(gs.get_xyz.shape[0])
        stats["mesh_v"], stats["mesh_f"] = int(mesh.vertices.shape[0]), int(mesh.faces.shape[0])
        (od / "report.json").write_text(json.dumps(stats, indent=2))
        rep_all[n] = stats
        print(n, json.dumps(stats), flush=True)
        vol.commit()
    (out_root / "report.json").write_text(json.dumps(rep_all, indent=2))
    vol.commit()
    return rep_all
