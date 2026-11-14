"""
OMEGAS Target Replenishment - Stable Diffusion personalization (paper Eq. 8: eps_theta* = P(I~, eps_theta), P = DreamBooth).
The published OMEGAS code only LOADS the personalized weights (sd_utils.py, hf_key == 'refined_sd':
unet/ and text_encoder/ from ./dreamgaussian/dreambooth/{hf_key}_finetuned/); the training script is not included.
This is a minimal DreamBooth (UNet + text encoder, prior preservation) on SD 2.1-base, as in the paper
("we adopt Stable Diffusion 2.1 with the resolution 512x512. We take DreamBooth as our personalization model").

Instance images I~ = the target segmented from the original views: object_<id>/images/*.jpg written by
extract_object.py (original image x predicted 3D-consistent target mask, black background), cropped around the target.

Usage: python tr_dreambooth.py --object_images <dir> --out <dir> [--steps 600 --lr 2e-6 ...]
Writes <out>/unet, <out>/text_encoder (diffusers format), <out>/instance/*.png, <out>/samples/*.png, <out>/info.json
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

BASE = "sd2-community/stable-diffusion-2-1-base"      # mirror of stabilityai/stable-diffusion-2-1-base (now 401)


def build_instance_images(src_dir: Path, out_dir: Path, k: int, min_px: int, margin: float, size: int = 512):
    out_dir.mkdir(parents=True, exist_ok=True)
    cands = []
    for p in sorted(src_dir.glob("*.jpg")):
        a = np.asarray(Image.open(p).convert("RGB"))
        m = a.max(-1) > 12
        n = int(m.sum())
        if n >= min_px:
            cands.append((n, p, m))
    cands.sort(key=lambda x: -x[0])
    # spread the picks over the sequence instead of taking only neighbouring frames
    chosen = sorted(cands[: max(k * 3, k)], key=lambda x: x[1].name)
    if len(chosen) > k:
        idx = np.linspace(0, len(chosen) - 1, k).round().astype(int)
        chosen = [chosen[i] for i in idx]
    names = []
    for n, p, m in chosen:
        ys, xs = np.nonzero(m)
        cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
        side = max(ys.max() - ys.min(), xs.max() - xs.min()) * (1 + 2 * margin)
        side = max(side, 64)
        box = (int(round(cx - side / 2)), int(round(cy - side / 2)), int(round(cx + side / 2)), int(round(cy + side / 2)))
        crop = Image.open(p).convert("RGB").crop(box)            # PIL pads outside the image with black
        crop.resize((size, size), Image.BICUBIC).save(out_dir / f"{p.stem}.png")
        names.append({"frame": p.stem, "pixels": n, "box": box})
    return names


def to_tensor(img: Image.Image, size=512):
    img = img.convert("RGB").resize((size, size), Image.BICUBIC)
    t = torch.from_numpy(np.asarray(img).astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object_images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--instance_prompt", default="a photo of sks napkin")
    ap.add_argument("--class_prompt", default="a photo of napkin")
    ap.add_argument("--num_instance", type=int, default=20)
    ap.add_argument("--min_px", type=int, default=1500)
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--num_class_images", type=int, default=100)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--prior_loss_weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    inst = build_instance_images(Path(args.object_images), out / "instance", args.num_instance, args.min_px, args.margin)
    print(f"[dreambooth] {len(inst)} instance images", flush=True)
    if len(inst) < 3:
        raise SystemExit("[dreambooth] fewer than 3 usable target views")

    # ---- prior-preservation class images (generated once with the base model, cached)
    cls_dir = out / "class_images"
    cls_dir.mkdir(exist_ok=True)
    have = len(list(cls_dir.glob("*.png")))
    if have < args.num_class_images:
        pipe = StableDiffusionPipeline.from_pretrained(BASE, torch_dtype=torch.float16, safety_checker=None).to("cuda")
        pipe.set_progress_bar_config(disable=True)
        g = torch.Generator("cuda").manual_seed(args.seed)
        for i in range(have, args.num_class_images, 4):
            imgs = pipe([args.class_prompt] * min(4, args.num_class_images - i), num_inference_steps=30, generator=g).images
            for j, im in enumerate(imgs):
                im.save(cls_dir / f"class_{i + j:04d}.png")
        del pipe
        torch.cuda.empty_cache()
    print(f"[dreambooth] class images: {len(list(cls_dir.glob('*.png')))}", flush=True)

    tok = CLIPTokenizer.from_pretrained(BASE, subfolder="tokenizer")
    te = CLIPTextModel.from_pretrained(BASE, subfolder="text_encoder").cuda()
    vae = AutoencoderKL.from_pretrained(BASE, subfolder="vae", torch_dtype=torch.float16).cuda().eval()
    unet = UNet2DConditionModel.from_pretrained(BASE, subfolder="unet").cuda()
    sched = DDPMScheduler.from_pretrained(BASE, subfolder="scheduler")
    vae.requires_grad_(False)
    unet.enable_gradient_checkpointing()
    te.gradient_checkpointing_enable()
    unet.train()
    te.train()

    def ids(prompt):
        return tok(prompt, padding="max_length", max_length=tok.model_max_length, truncation=True,
                   return_tensors="pt").input_ids.cuda()

    inst_t = [to_tensor(Image.open(p)) for p in sorted((out / "instance").glob("*.png"))]
    cls_t = [to_tensor(Image.open(p)) for p in sorted(cls_dir.glob("*.png"))]
    ids_inst, ids_cls = ids(args.instance_prompt), ids(args.class_prompt)
    params = list(unet.parameters()) + list(te.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler()
    losses = []
    for step in range(1, args.steps + 1):
        x = torch.stack([random.choice(inst_t), random.choice(cls_t)]).cuda()
        if random.random() < 0.5:
            x = torch.flip(x, dims=[3])
        with torch.no_grad():
            lat = vae.encode(x.half()).latent_dist.sample() * vae.config.scaling_factor
        lat = lat.float()
        noise = torch.randn_like(lat)
        t = torch.randint(0, sched.config.num_train_timesteps, (lat.shape[0],), device=lat.device).long()
        noisy = sched.add_noise(lat, noise, t)
        with torch.autocast("cuda", dtype=torch.float16):
            ehs = te(torch.cat([ids_inst, ids_cls]))[0]
            pred = unet(noisy, t, ehs).sample
        target = noise if sched.config.prediction_type == "epsilon" else sched.get_velocity(lat, noise, t)
        pred = pred.float()
        loss_inst = F.mse_loss(pred[:1], target[:1])
        loss_prior = F.mse_loss(pred[1:], target[1:])
        loss = loss_inst + args.prior_loss_weight * loss_prior
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        losses.append(float(loss))
        if step % 50 == 0 or step == 1:
            print(f"[dreambooth] step {step}/{args.steps} loss {np.mean(losses[-50:]):.4f}", flush=True)

    unet.eval()
    te.eval()
    unet.save_pretrained(out / "unet")
    te.save_pretrained(out / "text_encoder")
    # sanity samples of the personalized model
    pipe = StableDiffusionPipeline.from_pretrained(BASE, unet=unet.half(), text_encoder=te.half(),
                                                   torch_dtype=torch.float16, safety_checker=None).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    (out / "samples").mkdir(exist_ok=True)
    g = torch.Generator("cuda").manual_seed(1)
    for i, im in enumerate(pipe([args.instance_prompt] * 4, num_inference_steps=30, generator=g).images):
        im.save(out / "samples" / f"sks_{i}.png")
    info = {"base": BASE, "instance_prompt": args.instance_prompt, "class_prompt": args.class_prompt,
            "instance_images": inst, "class_images": len(cls_t), "steps": args.steps, "lr": args.lr,
            "prior_loss_weight": args.prior_loss_weight, "final_loss_mean50": float(np.mean(losses[-50:])),
            "seconds": round(time.time() - t0, 1)}
    (out / "info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({k: info[k] for k in ("final_loss_mean50", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
