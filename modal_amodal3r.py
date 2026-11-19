"""
Amodal3R (arXiv 2503.13439; code github.com/Sm0kyWu/Amodal3R @ d00e083f, S-Lab License 1.0 non-commercial; weights
huggingface.co/Sm0kyWu/Amodal3R, CC-BY-4.0): occlusion-aware 3D generation from one view + visible / occluder mask.

Image = self-contained copy of the TRELLIS image definition in modal_trellis.py (same torch 2.4 / cu121 / xformers / spconv /
kaolin / mip-splatting stack that Amodal3R's setup.sh installs) + one layer cloning Amodal3R. Nothing is imported from
modal_trellis: Modal mounts only this entrypoint file into the container, so `from modal_trellis import ...` crashed there.
DINOv2: like modal_trellis.py, the torch.hub download (dl.fbaipublicfiles.com) is replaced by the same DINOv2-L/14 with
4 registers from Hugging Face (facebook/dinov2-with-registers-large), 'x_prenorm' = hidden_states[-1].

Inputs (tools/prep_amodal3r_inputs.py, uploaded to the Volume): teatime/amodal3r_in/<obj>/000000.png + 000000_mask.png
Outputs: teatime/amodal3r_out/<tag>/<obj>/ gaussian.ply (canonical [-0.5, 0.5]^3, z up), mesh.npz (v, f), report.json.
Checkpointing: vol.commit() after every object; an object with report.json is skipped (resume).

    python -m modal run modal_amodal3r.py --tag a1 --objects dia
"""
import json
from pathlib import Path

import modal

VOL = "/vol"
vol = modal.Volume.from_name("safe-gs-come")
TRELLIS_COMMIT = "193e0502b5cb3eb5997ec457d5340e95d09b158e"
AMODAL3R_COMMIT = "d00e083f224d0a4c32c1d3a92740ae57d53b58ca"
app = modal.App("safe-gs-amodal3r")

# copied verbatim from modal_trellis.py (keep in sync by hand)
trellis_image = (
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

image = (
    trellis_image
    .run_commands(f"git clone https://github.com/Sm0kyWu/Amodal3R.git /opt/Amodal3R && cd /opt/Amodal3R && git checkout {AMODAL3R_COMMIT}")
    .env({"PYTHONPATH": "/opt/Amodal3R:/opt/TRELLIS", "ATTN_BACKEND": "xformers", "SPARSE_ATTN_BACKEND": "xformers",
          "HF_HOME": f"{VOL}/hf_cache"})                 # weights cached on the Volume: downloaded once, by the CPU check
)


@app.function(image=image, volumes={VOL: vol}, timeout=30 * 60, cpu=2, memory=8192)
def check():
    """CPU only (no GPU billed): import the stack, download Amodal3R + DINOv2 weights into the Volume cache, list inputs."""
    import importlib
    from huggingface_hub import snapshot_download
    vol.reload()
    mods = {}
    for m in ("torch", "xformers", "spconv.pytorch", "utils3d", "rembg", "amodal3r.pipelines", "transformers"):
        try:
            importlib.import_module(m); mods[m] = "ok"
        except Exception as e:                                           # report, do not hide
            mods[m] = f"FAIL {type(e).__name__}: {str(e)[:200]}"
    p1 = snapshot_download("Sm0kyWu/Amodal3R")
    p2 = snapshot_download("facebook/dinov2-with-registers-large", allow_patterns=["*.json", "*.safetensors"])
    vol.commit()
    ins = sorted(str(p.relative_to(Path(VOL))) for p in (Path(VOL) / "teatime" / "amodal3r_in").rglob("*.png"))
    args = json.loads((Path(p1) / "pipeline.json").read_text())["args"]
    print(json.dumps({"imports": mods}), flush=True)
    return {"imports": mods, "amodal3r": p1, "dinov2": p2, "image_cond_model": args.get("image_cond_model"),
            "models": args.get("models"), "files": sorted(str(q.relative_to(p1)) for q in Path(p1).rglob("*") if q.is_file()),
            "inputs": ins}


@app.function(image=image, gpu=["L40S", "A100-40GB"], volumes={VOL: vol}, timeout=45 * 60, cpu=4, memory=32768)
def complete(tag: str, objects: str, seed: int = 1, ss_steps: int = 12, ss_cfg: float = 7.5, slat_steps: int = 12,
             slat_cfg: float = 3.0, in_rel: str = "teatime/amodal3r_in"):
    import time
    import numpy as np
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from PIL import Image
    from transformers import Dinov2WithRegistersModel
    import amodal3r.pipelines.image_to_3d as A3

    vol.reload()
    root = Path(VOL) / in_rel
    out_root = Path(VOL) / "teatime" / "amodal3r_out" / tag
    out_root.mkdir(parents=True, exist_ok=True)
    names = [n for n in objects.split(",") if (root / n / "000000_mask.png").exists()]
    todo = [n for n in names if not (out_root / n / "report.json").exists()]
    print(f"objects {names}; already done {sorted(set(names) - set(todo))}; to do {todo}", flush=True)
    rep_all = {n: json.loads((out_root / n / "report.json").read_text()) for n in names if n not in todo}
    if not todo:
        return rep_all

    class _Dino(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = Dinov2WithRegistersModel.from_pretrained("facebook/dinov2-with-registers-large").eval()

        def forward(self, x, is_training=True):
            return {"x_prenorm": self.m(pixel_values=x, output_hidden_states=True).hidden_states[-1]}

    def _init_cond(self, name):
        assert "vitl14_reg" in str(name), f"unexpected DINOv2 variant {name}; the HF replacement is ViT-L/14 with registers"
        self.models["image_cond_model"] = _Dino().eval()
        self.image_cond_model_transform = T.Compose([T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    A3.Amodal3RImageTo3DPipeline._init_image_cond_model = _init_cond

    from amodal3r.pipelines import Amodal3RImageTo3DPipeline
    pipe = Amodal3RImageTo3DPipeline.from_pretrained("Sm0kyWu/Amodal3R")
    pipe.cuda()
    for n in todo:
        t0 = time.time()
        od = out_root / n
        od.mkdir(exist_ok=True)
        img = Image.open(root / n / "000000.png").convert("RGB")
        msk = Image.open(root / n / "000000_mask.png").convert("L")
        with torch.no_grad():
            outs = pipe.run_multi_image([img], [msk], seed=seed,
                                        sparse_structure_sampler_params={"steps": ss_steps, "cfg_strength": ss_cfg},
                                        slat_sampler_params={"steps": slat_steps, "cfg_strength": slat_cfg})
        gs, mesh = outs["gaussian"][0], outs["mesh"][0]
        gs.save_ply(str(od / "gaussian.ply"), transform=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        v, f = mesh.vertices.detach().cpu().numpy(), mesh.faces.detach().cpu().numpy()
        np.savez_compressed(od / "mesh.npz", v=v, f=f)
        m = np.array(msk)
        rep = {"seed": seed, "ss_steps": ss_steps, "ss_cfg": ss_cfg, "slat_steps": slat_steps, "slat_cfg": slat_cfg,
               "commit": AMODAL3R_COMMIT, "gaussians": int(gs.get_xyz.shape[0]), "mesh_v": int(len(v)), "mesh_f": int(len(f)),
               "mask_visible_px": int(((m >= 127) & (m <= 230)).sum()), "mask_occluder_px": int((m < 127).sum()),
               "bbox_min": v.min(0).tolist(), "bbox_max": v.max(0).tolist(), "seconds": round(time.time() - t0, 1)}
        (od / "report.json").write_text(json.dumps(rep, indent=1))
        vol.commit()                                                     # checkpoint after every object
        rep_all[n] = rep
        print(n, json.dumps(rep), flush=True)
        torch.cuda.empty_cache()
    (out_root / "report.json").write_text(json.dumps(rep_all, indent=1))
    vol.commit()
    return rep_all


@app.local_entrypoint()
def main(tag: str = "a1", objects: str = "dia", seed: int = 1, only_check: bool = False):
    if only_check:
        print(json.dumps(check.remote(), indent=1)); return
    print(json.dumps(complete.remote(tag, objects, seed), indent=1))
