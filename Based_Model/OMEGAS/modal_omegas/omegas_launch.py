"""
Run an original OMEGAS script unchanged, after applying compatibility fixes in memory only:

  1. CUDA is initialised before the script runs, so the hard-coded `os.environ["CUDA_VISIBLE_DEVICES"] = ...`
     of dream_gaussian_2D.py, render_obj.py, extract_object.py ("3") and mesh_2D.py ("2") no longer hides the
     single GPU of the Modal container. (Items 2-5 apply to dream_gaussian_2D.py only.)
  2. `dreamgaussian.gs_renderer_2D` (missing from the published code) -> modal_omegas/gs_renderer_2D.py.
  3. `dreamgaussian.cam_utils.OrbitCamera` of the release has no `fovx` argument and converts fovy from degrees,
     while dream_gaussian_2D.py passes OrbitCamera(W, H, r=, fovy=<rad>, fovx=<rad>) -> compatible subclass.
  4. `dreamgaussian.cam_utils.save_camera` returns the camera position WITHOUT the orbit target, while the image
     was rendered with orbit_camera(..., target) -> the saved position gets the target added, so the extra
     cameras written to refined/cameras.json match their rendered images.
  5. HuggingFace repos removed since the paper -> mirrors (diffusers/transformers from_pretrained remap).

Usage:  python modal_omegas/omegas_launch.py <script.py> [script args...]
        python modal_omegas/omegas_launch.py --preflight      (download/verify the SD weights only)
"""
import os
import runpy
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))

HF_REMAP = {
    "stabilityai/stable-diffusion-2-1-base": "sd2-community/stable-diffusion-2-1-base",
    # the paper inpaints with stabilityai/stable-diffusion-2-inpainting (footnote 1); the code loads the SD-1.5
    # inpainting model -> use the paper's model (mirror; the stabilityai repo now returns 401)
    "runwayml/stable-diffusion-inpainting": "sd2-community/stable-diffusion-2-inpainting",
    "stabilityai/stable-diffusion-2-inpainting": "sd2-community/stable-diffusion-2-inpainting",
    "runwayml/stable-diffusion-v1-5": "stable-diffusion-v1-5/stable-diffusion-v1-5",
}
# sd_utils.py loads the DreamBooth weights from this literal path (missing f-string in the authors' code)
DREAMBOOTH_KEYS = ("./dreamgaussian/dreambooth/{hf_key}_finetuned/unet",
                   "./dreamgaussian/dreambooth/{hf_key}_finetuned/text_encoder")
if os.environ.get("OMEGAS_DREAMBOOTH_DIR"):
    _db = os.environ["OMEGAS_DREAMBOOTH_DIR"]
    HF_REMAP[DREAMBOOTH_KEYS[0]] = os.path.join(_db, "unet")
    HF_REMAP[DREAMBOOTH_KEYS[1]] = os.path.join(_db, "text_encoder")


def _init_cuda():
    import torch
    torch.cuda.init()  # CUDA_VISIBLE_DEVICES is read once, here; later changes by the script have no effect
    n = torch.cuda.device_count()
    print(f"[launch] CUDA devices visible: {n} ({torch.cuda.get_device_name(0) if n else 'none'})", flush=True)
    if n == 0:
        raise SystemExit("[launch] no CUDA device")


def _patch_hf():
    import diffusers
    import transformers

    def wrap(klass):
        if "from_pretrained" not in vars(klass):
            return
        orig = vars(klass)["from_pretrained"].__func__

        def patched(cls, name, *a, **k):
            new = HF_REMAP.get(name, name) if isinstance(name, str) else name
            if new != name:
                print(f"[launch] HF remap {name} -> {new}", flush=True)
            return orig(cls, new, *a, **k)

        klass.from_pretrained = classmethod(patched)

    for klass in (diffusers.DiffusionPipeline, diffusers.ModelMixin, diffusers.SchedulerMixin,
                  transformers.PreTrainedModel, transformers.PreTrainedTokenizerBase):
        wrap(klass)
    # ConfigMixin.from_config / load_config are used by SchedulerMixin.from_pretrained with the remapped name already


def _patch_dreamgaussian():
    import importlib
    import numpy as np

    sys.path.insert(0, HERE)
    gs2d = importlib.import_module("gs_renderer_2D")
    sys.modules["dreamgaussian.gs_renderer_2D"] = gs2d

    cam_utils = importlib.import_module("dreamgaussian.cam_utils")
    base = cam_utils.OrbitCamera

    class OrbitCamera(base):
        def __init__(self, W, H, r=2, fovy=0.8, fovx=None, near=0.01, far=100):
            super().__init__(W, H, r=r, fovy=60, near=near, far=far)
            self.fovy = float(fovy)                     # radians, as in the OMEGAS dream_*.yaml configs
            self._fovx = None if fovx is None else float(fovx)

        @property
        def fovx(self):
            if self._fovx is not None:
                return self._fovx
            return 2 * np.arctan(np.tan(self.fovy / 2) * self.W / self.H)

    orig_save = cam_utils.save_camera

    def save_camera(elevation, azimuth, radius=1, is_degree=True, target=None, opengl=True):
        position, rotation = orig_save(elevation, azimuth, radius=radius, is_degree=is_degree, target=target, opengl=opengl)
        if target is not None:
            position = position + np.asarray(target, dtype=position.dtype)
        return position, rotation

    def convert_camera_from_orbit_to_pytorch3d(*_a, **_k):
        # imported by dream_gaussian_2D.py but not defined in the released cam_utils.py; only used when
        # render_mesh=true (mesh rendering with pytorch3d), which our config keeps off
        raise NotImplementedError("convert_camera_from_orbit_to_pytorch3d is missing from the released OMEGAS code; "
                                  "keep render_mesh: false")

    cam_utils.OrbitCamera = OrbitCamera
    cam_utils.save_camera = save_camera
    if not hasattr(cam_utils, "convert_camera_from_orbit_to_pytorch3d"):
        cam_utils.convert_camera_from_orbit_to_pytorch3d = convert_camera_from_orbit_to_pytorch3d


def _patch_sd_mask():
    """Paper (Implementation Details): "we fix the timestep t to 991 and sample 10 random seeds to obtain an averaged
    mask result". The released sd_mask() draws 5 seeds; this is the same function with the number of seeds from
    OMEGAS_MASK_SEEDS (default 10). Everything else is unchanged."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from dreamgaussian.guidance import sd_utils

    n_seeds = int(os.environ.get("OMEGAS_MASK_SEEDS", "10"))

    @torch.no_grad()
    def sd_mask(self, rgb, guidance_scale=7.5, steps=100, prompt=['truck'], render_H=546, render_W=979, th=17,
                height=512, width=512):
        seeds = torch.randint(0, 1024, (n_seeds,))
        masks = []
        for seed in seeds:
            torch.manual_seed(seed)
            batch_size = len(prompt)
            rgb_512 = F.interpolate(rgb, (512, 512), mode='bilinear', align_corners=False)
            latent = self.encode_imgs(rgb_512.to(self.dtype))
            text_input = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length,
                                        truncation=True, return_tensors="pt")
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
            uncond_embeddings = torch.cat([uncond_embeddings, uncond_embeddings])      # y = empty prompt
            noise = torch.randn((batch_size, self.unet.in_channels, height // 8, width // 8)).to(self.device).half()
            self.scheduler.set_timesteps(steps)
            t = torch.tensor(991).to("cuda")
            latent_input = self.scheduler.add_noise(latent, noise, t)
            latent_input = torch.cat([latent_input] * 2) if len(prompt) > 0 else latent_input
            noise_pred = self.unet(latent_input, t, encoder_hidden_states=uncond_embeddings).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            a = (noise - noise_pred)                                                    # residual Δε
            image = self.vae.decode(1 / self.vae.config.scaling_factor * a).sample      # D(Δε)
            image = (image / 2 + 0.5).clamp(0, 1)
            image = (image.detach().cpu().permute(0, 2, 3, 1).numpy()[0] * 255).round().astype("uint8")
            image = 1 / 3 * image[:, :, 0] + 1 / 3 * image[:, :, 1] + 1 / 3 * image[:, :, 2]
            mask = (image > image.max() - th).astype(float)                             # binarisation B
            mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0)
            mask = F.interpolate(mask, size=(render_H, render_W), mode='bilinear').squeeze().numpy()
            masks.append(mask)
        return np.mean(np.stack(masks, axis=0), axis=0).round().astype("uint8")

    sd_utils.StableDiffusion.sd_mask = sd_mask
    print(f"[launch] sd_mask: {n_seeds} seeds, t=991", flush=True)


def preflight():
    import torch
    from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline, StableDiffusionPipeline
    _patch_hf()
    print(f"[preflight] HF_HOME={os.environ.get('HF_HOME')}", flush=True)
    StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16)
    DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="scheduler")
    StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16,
                                                   safety_checker=None)
    for k in DREAMBOOTH_KEYS:
        if k in HF_REMAP and not os.path.isdir(HF_REMAP[k]):
            raise SystemExit(f"[preflight] DreamBooth weights missing: {HF_REMAP[k]}")
    print("[preflight] OK: SD 2.1-base + SD-2 inpainting cached"
          + (", DreamBooth weights found" if DREAMBOOTH_KEYS[0] in HF_REMAP else ""), flush=True)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    os.chdir(REPO)
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    if sys.argv[1] == "--preflight":
        preflight()
        return
    script = os.path.join(REPO, sys.argv[1])
    _init_cuda()     # every script: render_obj.py / extract_object.py / mesh_2D.py / dream_gaussian_2D.py
    #                  hard-code CUDA_VISIBLE_DEVICES ("3", "3", "2", "3") on their first lines
    if os.path.basename(script) == "dream_gaussian_2D.py":
        _patch_hf()
        _patch_dreamgaussian()
        _patch_sd_mask()
    sys.argv = [script] + sys.argv[2:]
    print(f"[launch] running {script} {' '.join(sys.argv[1:])}", flush=True)
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
