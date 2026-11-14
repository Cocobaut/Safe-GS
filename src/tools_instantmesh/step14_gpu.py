"""Step 14.2-14.5 (GPU): virtual 360 video of the object Gaussians -> most informative views -> InstantMesh -> mesh.

14.2 orbit   : spiral of N_FRAMES virtual cameras around the object in the InstantMesh frame (distance 4, fov 30, looking
               at the origin, up = table normal = the exact camera model InstantMesh's LRM uses), azimuth 3 turns,
               elevation 80 deg -> -40 deg. The object Gaussians are rendered (gs_render.py, RGBA 512^2 on white) together
               with a per-pixel RELIABILITY map. Reliability of Gaussian i for a virtual camera v:
                 r_i(v) = clamp((max_j cos(d_v, d_j) - cos 60) / (1 - cos 60), 0, 1),  j = real cameras that observed i
               (d = unit direction from the Gaussian to the camera). The napkin bottom was never observed: from below
               the table the Gaussians are seen from the side no real camera saw -> r ~ 0 (shown red in the video).
14.3 select  : greedy "next best view" on the orbit frames. Information of view v about Gaussian i:
                 info_i(v) = min(1, contrib_i(v)) * r_i(v)      (contrib = compositing weight summed over the pixels)
               gain(v | chosen) = sum_i a_i * max(0, info_i(v) - best_i)   (a_i = opacity x area: importance)
               views whose mean reliability over the silhouette is < MIN_REL are excluded; stop when the gain drops
               below GAIN_STOP of the total importance or K_MAX views are chosen (InstantMesh's LRM is trained with 6
               views; 12 was its best and >= 24 degraded in the InstantMesh/src_v2 experiment).
14.4 recon   : InstantMesh LRM (instant-mesh-large) on the chosen views with their exact cameras -> mesh (IM frame);
               baseline "uniform6": 6 views at azimuth 30..330, elevation 20/40 alternating.
14.5 eval    : every mesh back to the scene (X = c + R^T X_im / s) and projected with the REAL cameras (nvdiffrast):
               IoU with the synced object masks (pixels of other objects ignored); the basic step-15 napkin as reference.
               Best variant -> largest component -> mesh_object_instantmesh.ply (scene frame, vertex colours).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gs_render  # noqa: E402

N_FRAMES = 240
TURNS = 3
ELEV_START, ELEV_END = 80.0, -40.0
SIZE = 512
LRM_RADIUS, LRM_FOV = 4.0, 30.0
REL_COS = math.cos(math.radians(60))
MIN_REL = 0.5
GAIN_STOP = 0.02
K_MIN, K_MAX = 4, 12
FLIP = np.diag([1.0, -1.0, -1.0, 1.0])            # OpenGL <-> OpenCV camera axes


# ------------------------------------------------------------------------------------------- cameras
def look_at_gl(pos, target=np.zeros(3), up=np.array([0, 0, 1.0])):
    z = pos - target; z = z / np.linalg.norm(z)
    x = np.cross(up, z); x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4); m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, pos
    return m


def im_camera(az, el):
    a, e = math.radians(az), math.radians(el)
    pos = LRM_RADIUS * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    return look_at_gl(pos)


def im_to_scene_c2w_cv(c2w_gl_im, frame):
    c, R, s = np.array(frame["c"]), np.array(frame["R"]), frame["s"]
    m = np.eye(4)
    m[:3, :3] = R.T @ c2w_gl_im[:3, :3]
    m[:3, 3] = c + R.T @ c2w_gl_im[:3, 3] / s
    return m @ FLIP


def intrinsics(size, fov=LRM_FOV):
    f = 0.5 * size / math.tan(math.radians(fov) / 2)
    return [f, f, size / 2, size / 2]


def orbit_poses():
    t = np.linspace(0, 1, N_FRAMES, endpoint=False)
    return [(float((360 * TURNS * x) % 360), float(ELEV_START + (ELEV_END - ELEV_START) * x)) for x in t]


def reliability(xyz, obs, cam_centers, cam_pos):
    """[N] in [0, 1]: best agreement between the virtual view direction and a real view that observed the Gaussian."""
    dv = torch.nn.functional.normalize(cam_pos[None] - xyz, dim=1)                       # [N,3]
    dj = torch.nn.functional.normalize(cam_centers[None] - xyz[:, None], dim=2)          # [N,J,3]
    cos = (dj * dv[:, None]).sum(2)
    cos = torch.where(obs, cos, torch.full_like(cos, -1.0)).max(1).values
    return ((cos - REL_COS) / (1 - REL_COS)).clamp(0, 1)


def load_object(in_dir, device):
    frame = json.loads(Path(in_dir, "frame.json").read_text())
    g = gs_render.load_ply(Path(in_dir, f"point_cloud_{frame['object_id']}_object.ply"), device)
    o = np.load(Path(in_dir, "observations.npz"))
    obs = torch.from_numpy(o["obs"]).to(device)
    cc = torch.from_numpy(o["cam_centers"]).float().to(device)
    return frame, g, obs, cc


def to_png(rgb, alpha):
    a = np.clip(alpha, 0, 1)
    return Image.fromarray(np.dstack([(np.clip(rgb, 0, 1) * 255).astype(np.uint8), (a * 255).astype(np.uint8)]), "RGBA")


def rel_vis(rgb, alpha, rel):
    """reliability overlay: green = observed by real cameras from a similar direction, red = never seen like this."""
    r = np.clip(rel, 0, 1)
    heat = np.stack([1 - r, r, np.zeros_like(r)], -1)
    out = 0.5 * rgb + 0.5 * heat
    a = alpha[..., None]
    return (np.clip(out * a + (1 - a), 0, 1) * 255).astype(np.uint8)


# ------------------------------------------------------------------------------------------- 14.2 orbit
def step14_2_orbit(in_dir, out_dir, commit=lambda r: None):
    import imageio.v2 as imageio
    dev = torch.device("cuda")
    out = Path(out_dir); (out / "frames").mkdir(parents=True, exist_ok=True)
    frame, g, obs, cc = load_object(in_dir, dev)
    poses = orbit_poses()
    K = intrinsics(SIZE)
    N = len(g["xyz"])
    contrib = np.zeros((N_FRAMES, N), np.float16); rel = np.zeros((N_FRAMES, N), np.float16)
    meta = []
    part = out / "orbit_arrays_partial.npz"
    done_upto = 0
    if part.exists():                                            # resume after preemption
        z = np.load(part); contrib, rel, done_upto = z["contrib"], z["rel"], int(z["done"])
        meta = json.loads((out / "orbit_meta_partial.json").read_text())
        print(f"[14.2] resume at frame {done_upto}", flush=True)
    t0 = time.time()
    for k in range(done_upto, N_FRAMES):
        az, el = poses[k]
        c2w_im = im_camera(az, el)
        c2w_cv = im_to_scene_c2w_cv(c2w_im, frame)
        r_i = reliability(g["xyz"], obs, cc, torch.tensor(c2w_cv[:3, 3], dtype=torch.float32, device=dev))
        res = gs_render.render(g, c2w_cv, K, SIZE, SIZE, bg=(1, 1, 1), features=r_i[:, None])
        rgb, a = res["rgb"].cpu().numpy(), res["alpha"].cpu().numpy()
        relmap = (res["feat"][..., 0] / res["alpha"].clamp(min=1e-6)).cpu().numpy()
        to_png(rgb, a).save(out / "frames" / f"{k:03d}.png")
        Image.fromarray(rel_vis(rgb, a, relmap)).save(out / "frames" / f"{k:03d}_rel.jpg", quality=90)
        contrib[k] = res["contrib"].cpu().numpy(); rel[k] = r_i.cpu().numpy()
        sil = a > 0.5
        meta.append({"frame": k, "azimuth": round(az, 2), "elevation": round(el, 2), "c2w_gl_im": c2w_im.round(6).tolist(),
                     "alpha_area": round(float(sil.mean()), 4),
                     "mean_reliability": round(float(relmap[sil].mean()) if sil.any() else 0.0, 4)})
        if (k + 1) % 40 == 0:
            np.savez_compressed(part, contrib=contrib, rel=rel, done=k + 1)
            (out / "orbit_meta_partial.json").write_text(json.dumps(meta))
            commit(f"14.2 frame {k + 1}")
            print(f"[14.2] {k + 1}/{N_FRAMES} frames, {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(out / "orbit_arrays.npz", contrib=contrib, rel=rel)
    (out / "orbit_cameras.json").write_text(json.dumps({"size": SIZE, "fov": LRM_FOV, "radius_im": LRM_RADIUS,
                                                         "turns": TURNS, "elevation": [ELEV_START, ELEV_END],
                                                         "frames": meta}, indent=1))
    for name, suf in (("orbit_rgb.mp4", ".png"), ("orbit_reliability.mp4", "_rel.jpg")):
        w = imageio.get_writer(out / name, fps=24, quality=8)
        for k in range(N_FRAMES):
            im = np.asarray(Image.open(out / "frames" / f"{k:03d}{suf}").convert("RGBA" if suf == ".png" else "RGB"), np.float32) / 255
            if im.shape[2] == 4:
                im = im[..., :3] * im[..., 3:] + (1 - im[..., 3:])
            w.append_data((im * 255).astype(np.uint8))
        w.close()
    part.unlink(missing_ok=True); (out / "orbit_meta_partial.json").unlink(missing_ok=True)
    (out / "done.json").write_text(json.dumps({"frames": N_FRAMES, "seconds": round(time.time() - t0, 1)}))
    commit("14.2 done")


# ------------------------------------------------------------------------------------------- 14.3 select
def step14_3_select(in_dir, orbit_dir, out_dir):
    """Both selections: k_min = 4 (pure coverage rule) and k_min = 6 (LRM is trained with 6 input views; with 4 views the
    coverage saturates but InstantMesh reconstructs worse - measured in the first run)."""
    r4 = _select(in_dir, orbit_dir, out_dir, K_MIN, "")
    r6 = _select(in_dir, orbit_dir, out_dir, 6, "_k6")
    return r4, r6


def _select(in_dir, orbit_dir, out_dir, k_min, tag):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    g = gs_render.load_ply(Path(in_dir, f"point_cloud_{json.loads(Path(in_dir, 'frame.json').read_text())['object_id']}_object.ply"))
    sc = g["scale"].numpy(); srt = np.sort(sc, 1)
    imp = g["opacity"].numpy() * srt[:, 1] * srt[:, 2]
    imp = imp / imp.sum()
    z = np.load(Path(orbit_dir, "orbit_arrays.npz"))
    contrib, rel = z["contrib"].astype(np.float32), z["rel"].astype(np.float32)
    meta = json.loads(Path(orbit_dir, "orbit_cameras.json").read_text())["frames"]
    info = np.minimum(1.0, contrib) * rel                                       # [F, N]
    valid = np.array([m["mean_reliability"] >= MIN_REL for m in meta])
    best = np.zeros(info.shape[1]); chosen, gains, curve = [], [], []
    while len(chosen) < K_MAX:
        gain = (np.maximum(0, info - best[None]) * imp[None]).sum(1)
        gain[~valid] = -1; gain[chosen] = -1
        k = int(gain.argmax())
        if gain[k] <= 0 or (len(chosen) >= k_min and gain[k] < GAIN_STOP):
            break
        chosen.append(k); gains.append(float(gain[k])); best = np.maximum(best, info[k])
        curve.append(float((best * imp).sum()))
    sel = [{"frame": k, "azimuth": meta[k]["azimuth"], "elevation": meta[k]["elevation"], "gain": round(gn, 4),
            "covered_after": round(cv, 4), "mean_reliability": meta[k]["mean_reliability"]} for k, gn, cv in zip(chosen, gains, curve)]
    res = {"rule": {"info": "min(1, contrib) * reliability", "importance": "opacity x area (2 largest scales)",
                    "min_mean_reliability": MIN_REL, "gain_stop": GAIN_STOP, "k_min": k_min, "k_max": K_MAX},
           "valid_frames": int(valid.sum()), "chosen": sel, "n_views": len(sel),
           "covered_importance": round(curve[-1], 4) if curve else 0.0,
           "max_coverable_importance": round(float((info[valid].max(0) * imp).sum()), 4)}
    (out / f"selected_views{tag}.json").write_text(json.dumps(res, indent=2))
    # contact sheet (render | reliability)
    tiles = []
    for s_ in sel:
        a = Image.open(Path(orbit_dir, "frames", f"{s_['frame']:03d}.png")).convert("RGBA")
        w = Image.new("RGB", a.size, (255, 255, 255)); w.paste(a, mask=a.getchannel("A"))
        r = Image.open(Path(orbit_dir, "frames", f"{s_['frame']:03d}_rel.jpg"))
        t = Image.new("RGB", (512, 256), (255, 255, 255)); t.paste(w.resize((256, 256)), (0, 0)); t.paste(r.resize((256, 256)), (256, 0))
        tiles.append(t)
    cols = 3
    sheet = Image.new("RGB", (cols * 512, max(1, (len(tiles) + cols - 1) // cols) * 256), (255, 255, 255))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * 512, (i // cols) * 256))
    sheet.save(out / f"selected_views{tag}.jpg", quality=92)
    # plot: reliability + gain along the video
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot([m["mean_reliability"] for m in meta], label="mean reliability"); ax[0].axhline(MIN_REL, ls="--", c="gray")
    ax[0].plot([m["elevation"] / 90 for m in meta], label="elevation / 90", alpha=0.6); ax[0].legend()
    ax[1].bar([s_["frame"] for s_ in sel], [s_["gain"] for s_ in sel], width=3, color="tab:red", label="chosen (gain)")
    ax[1].set_xlabel("orbit frame"); ax[1].legend()
    fig.savefig(out / f"selection_plot{tag}.png", dpi=100, bbox_inches="tight"); plt.close(fig)
    print(f"[14.3{tag}] {json.dumps({k: v for k, v in res.items() if k != 'chosen'})} frames={[s_['frame'] for s_ in sel]}", flush=True)
    return res


# ------------------------------------------------------------------------------------------- 14.4 + 14.5
def load_lrm(im_root, device):
    sys.path.insert(0, str(im_root))
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf
    from src.models.lrm_mesh import InstantMesh
    cfg = OmegaConf.load(Path(im_root, "configs", "instant-mesh-large.yaml"))
    model = InstantMesh(**cfg.model_config.params)
    ck = hf_hub_download("TencentARC/InstantMesh", "instant_mesh_large.ckpt", repo_type="model")
    sd = torch.load(ck, map_location="cpu")["state_dict"]
    model.load_state_dict({k[14:]: v for k, v in sd.items() if k.startswith("lrm_generator.")}, strict=True)
    model = model.to(device); model.init_flexicubes_geometry(device, fovy=LRM_FOV)
    return model.eval(), cfg.infer_config


def projection(fx, fy, cx, cy, W, H, near=0.01, far=100.0):
    return torch.tensor([[2 * fx / W, 0, 1 - 2 * cx / W, 0], [0, 2 * fy / H, 2 * cy / H - 1, 0],
                         [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)], [0, 0, -1, 0]], dtype=torch.float32)


class RealViews:
    def __init__(self, in_dir, device):
        rc = json.loads(Path(in_dir, "real_cameras.json").read_text())
        sc = rc["eval_scale"]
        self.names, mvp, masks, ign, clean = [], [], [], [], []
        for n, e in sorted(rc["cameras"].items()):
            p = Path(in_dir, "masks_eval", f"{n}.png")
            if not p.exists():
                continue
            m = np.array(Image.open(p))
            H, W = m.shape
            fx, fy, cx, cy = [k * sc for k in e["K"]]
            w2c = torch.linalg.inv(torch.tensor(np.array(e["c2w_cv"]) @ FLIP, dtype=torch.float32))
            mvp.append(projection(fx, fy, cx, cy, W, H) @ w2c)
            masks.append(torch.from_numpy(m == 255)); ign.append(torch.from_numpy(m == 128))
            clean.append(not e.get("border", True) and e.get("inside_fraction", 0) >= 0.98)
            self.names.append(n)
        self.size = (H, W)
        self.mvp = torch.stack(mvp).to(device); self.masks = torch.stack(masks).to(device)
        self.ignore = torch.stack(ign).to(device); self.clean = np.array(clean)


@torch.no_grad()
def iou_real(ctx, V, F, rv, batch=16):
    import nvdiffrast.torch as dr
    Vt = torch.as_tensor(V, dtype=torch.float32, device=rv.mvp.device); Ft = torch.as_tensor(F, dtype=torch.int32, device=rv.mvp.device)
    vh = torch.cat([Vt, torch.ones_like(Vt[:, :1])], 1)
    out = {}
    for i in range(0, len(rv.names), batch):
        clip = torch.einsum("bij,vj->bvi", rv.mvp[i:i + batch], vh).contiguous()
        rast, _ = dr.rasterize(ctx, clip, Ft, resolution=list(rv.size))
        pred = (rast[..., 3] > 0).flip(1)
        gt, valid = rv.masks[i:i + batch], ~rv.ignore[i:i + batch]
        inter = (pred & gt & valid).flatten(1).sum(1).float(); union = ((pred | gt) & valid).flatten(1).sum(1).float()
        for n, x in zip(rv.names[i:i + batch], (inter / union.clamp(min=1)).tolist()):
            out[n] = round(x, 4)
    return out


def summarise(iou, rv):
    v = np.array([iou[n] for n in rv.names])
    return {"mean_all": round(float(v.mean()), 4), "mean_clean": round(float(v[rv.clean].mean()), 4),
            "n_all": int(len(v)), "n_clean": int(rv.clean.sum())}


def mesh_stats(V, F):
    import trimesh
    m = trimesh.Trimesh(V, F, process=False)
    parts = m.split(only_watertight=False)
    big = max(parts, key=lambda p: len(p.faces)) if len(parts) else m
    return {"vertices": int(len(V)), "faces": int(len(F)), "components": int(len(parts)),
            "largest_component_face_share": round(len(big.faces) / max(len(F), 1), 4), "watertight": bool(m.is_watertight)}


def step14_4_5_recon_eval(in_dir, orbit_dir, select_dir, out_dir, im_root, basic_mesh=None, commit=lambda r: None):
    import nvdiffrast.torch as dr
    import trimesh
    dev = torch.device("cuda")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    frame, g, _, _ = load_object(in_dir, dev)
    c, R, s = np.array(frame["c"]), np.array(frame["R"]), frame["s"]
    to_scene = lambda X: c + (X / s) @ R
    model, infer = load_lrm(im_root, dev)
    ctx = dr.RasterizeCudaContext(device=dev)
    rv = RealViews(in_dir, dev)
    meta = json.loads(Path(orbit_dir, "orbit_cameras.json").read_text())["frames"]
    sel = json.loads(Path(select_dir, "selected_views.json").read_text())["chosen"]
    sel6 = json.loads(Path(select_dir, "selected_views_k6.json").read_text())["chosen"]
    f_norm = 0.5 / math.tan(math.radians(LRM_FOV) / 2)
    K = intrinsics(SIZE)
    variants = {"heuristic": [(meta[x["frame"]]["azimuth"], meta[x["frame"]]["elevation"], x["frame"]) for x in sel],
                "heuristic_k6": [(meta[x["frame"]]["azimuth"], meta[x["frame"]]["elevation"], x["frame"]) for x in sel6],
                "uniform6": [(az, 20.0 if i % 2 == 0 else 40.0, None) for i, az in enumerate(range(30, 360, 60))]}
    results = {}
    for name, views in variants.items():
        vd = out / name
        if (vd / "done.json").exists():
            results[name] = json.loads((vd / "done.json").read_text()); continue
        (vd / "inputs").mkdir(parents=True, exist_ok=True)
        imgs, cams = [], []
        for i, (az, el, fr) in enumerate(views):
            c2w_im = im_camera(az, el)
            if fr is not None:
                im = Image.open(Path(orbit_dir, "frames", f"{fr:03d}.png")).convert("RGBA")
            else:
                r = gs_render.render(g, im_to_scene_c2w_cv(c2w_im, frame), K, SIZE, SIZE, bg=(1, 1, 1))
                im = to_png(r["rgb"].cpu().numpy(), r["alpha"].cpu().numpy())
            im.save(vd / "inputs" / f"{i:02d}_az{az:.0f}_el{el:.0f}.png")
            a = np.asarray(im, np.float32) / 255
            rgb = a[..., :3] * a[..., 3:] + (1 - a[..., 3:])                     # white background
            t = torch.from_numpy(rgb).permute(2, 0, 1)
            imgs.append(torch.nn.functional.interpolate(t[None], size=(320, 320), mode="bicubic", antialias=True)[0].clamp(0, 1))
            cams.append(np.concatenate([c2w_im.reshape(-1)[:12], [f_norm, f_norm, 0.5, 0.5]]))
        t0 = time.time()
        with torch.no_grad():
            planes = model.forward_planes(torch.stack(imgs)[None].to(dev), torch.tensor(np.stack(cams), dtype=torch.float32)[None].to(dev))
            v, f, col = model.extract_mesh(planes, use_texture_map=False, texture_resolution=infer.texture_resolution)
        sec = time.time() - t0
        Vs = to_scene(v.astype(np.float64))
        trimesh.Trimesh(v, f, vertex_colors=col, process=False).export(vd / "mesh_im.obj")
        trimesh.Trimesh(Vs, f, vertex_colors=col, process=False).export(vd / "mesh_scene.ply")
        iou = iou_real(ctx, Vs, f, rv)
        (vd / "iou_real_views.json").write_text(json.dumps(iou, indent=1))
        info = {"variant": name, "views": [{"azimuth": round(a_, 1), "elevation": round(e_, 1), "orbit_frame": fr} for a_, e_, fr in views],
                "n_views": len(views), "lrm_seconds": round(sec, 2), "stats": mesh_stats(v, f), "iou_real": summarise(iou, rv)}
        (vd / "done.json").write_text(json.dumps(info, indent=2))
        results[name] = info
        print(f"[14.4] {json.dumps(info)}", flush=True)
        commit(f"14.4 {name}")
    ref = None
    if basic_mesh and Path(basic_mesh).exists():
        bm = trimesh.load(basic_mesh, process=False, force="mesh")
        ref = {"mesh": "output_napkin_basic step15 mesh_napkin_closed.ply (height field)",
               "iou_real": summarise(iou_real(ctx, np.asarray(bm.vertices), np.asarray(bm.faces), rv), rv)}
    best = max(results, key=lambda k: results[k]["iou_real"]["mean_clean"])
    m = trimesh.load(out / best / "mesh_scene.ply", process=False, force="mesh")
    parts = m.split(only_watertight=False)
    big = max(parts, key=lambda p: len(p.faces)) if len(parts) > 1 else m
    fin = Path(out_dir).parent / "5_final_mesh"; fin.mkdir(parents=True, exist_ok=True)
    big.export(fin / "mesh_object_instantmesh.ply")
    big.export(fin / "mesh_object_instantmesh.obj")
    final = {"chosen_variant": best, "rule": "highest mean IoU on the clean real views", "variants": {k: v["iou_real"] for k, v in results.items()},
             "basic_reference": ref, "final_mesh": {**mesh_stats(np.asarray(big.vertices), np.asarray(big.faces)),
                                                    "iou_real": summarise(iou_real(ctx, np.asarray(big.vertices), np.asarray(big.faces), rv), rv)},
             "frame": "scene (COLMAP units of the CoMe reconstruction), same as step 13"}
    (fin / "summary.json").write_text(json.dumps(final, indent=2))
    print(f"[14.5] {json.dumps(final)}", flush=True)
    commit("14.5 done")
    return final
