"""
Safe-GS pipeline end-to-end trên Modal (S0 -> S6), có checkpoint ở mọi stage.

    python -m modal run --detach modal_pipeline.py::run --config configs/pipeline/teatime_v1.json            # chạy tới S6
    python -m modal run --detach modal_pipeline.py::run --config configs/pipeline/teatime_v1.json --until S3
    python -m modal run modal_pipeline.py::run --config ... --force S4a       # chạy lại 1 stage (và các stage sau)
    python -m modal run modal_pipeline.py::status --config configs/pipeline/teatime_v1.json
    python -m modal run modal_pipeline.py::pull --config configs/pipeline/teatime_v1.json   # tải preview + json về
                                                                               # Data/<scene>/pipeline/<run>/ (không tải .ply)

Output trên Volume safe-gs-come: <scene>/pipeline/<run_name>/<stage>/ gồm:
    _done.json        trạng thái + hash config (đổi config của stage hoặc stage trước -> stage chạy lại)
    config_used.json  bản config đầy đủ lúc chạy (tái sử dụng / so sánh)
    preview/ ...      ảnh để xem nhanh stage đó ra gì
    (dữ liệu của stage)
Checkpoint trong stage: S0/S2 dùng checkpoint của modal_come.py; S3 lưu vote cộng dồn mỗi 20 view; S4a lưu từng đoạn
60 frame; S4b lưu Gaussian mỗi 500 iter; S5/S6 lưu từng vật. Hàm có retries; Volume được commit mỗi 30 giây.
Mỗi stage có log riêng: <scene>/logs/pipeline_<stage>.log
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import modal

import modal_come as mc

app = modal.App("safe-gs-pipeline")
vol = mc.vol
VOL = mc.VOL
REPO_DIR = Path(__file__).resolve().parent
STAGES = ["S0", "S1", "S2", "S3", "S4a", "S4b", "S5", "S6", "S7", "S8", "S9", "S10"]
DEPS = {"S0": [], "S1": [], "S2": ["S0"], "S3": ["S1", "S2"], "S4a": ["S3"], "S4b": ["S3", "S4a"],
        "S5": ["S3"], "S6": ["S3", "S5"], "S7": ["S6"], "S8": ["S7"], "S9": ["S4b", "S5", "S6", "S8"],
        "S10": ["S4b", "S5", "S6", "S8"]}
CFG_KEY = {"S0": "S0_colmap", "S1": "S1_masks", "S2": "S2_come", "S3": "S3_label", "S4a": "S4_background",
           "S4b": "S4_background", "S5": "S5_objects", "S6": "S6_mesh", "S7": "S7_physics", "S8": "S8_sim",
           "S9": "S9_render", "S10": "S10_eval"}
DIRNAME = {"S0": "S0_colmap", "S1": "S1_masks", "S2": "S2_come", "S3": "S3_label", "S4a": "S4a_inpaint",
           "S4b": "S4b_background", "S5": "S5_objects", "S6": "S6_mesh", "S7": "S7_physics", "S8": "S8_sim",
           "S9": "S9_render", "S10": "S10_eval"}

def _src(img):
    """Mọi container cần module modal_come (dùng chung Volume, image, hàm log/checkpoint)."""
    return img.add_local_python_source("modal_come")


pipe_image = _src(
    mc.come_image
    .pip_install("coacd", "trimesh", "rtree", "shapely", "mapbox-earcut")   # shapely + earcut: slice_plane(cap=True)
    .add_local_dir(REPO_DIR / "pipeline", "/opt/pipeline")
)
sim_img = _src(
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("mujoco", "trimesh", "numpy", "scipy", "matplotlib", "rtree")
    .add_local_dir(REPO_DIR / "pipeline", "/opt/pipeline")
)
colmap_img = _src(mc.colmap_image)
come_img = _src(mc.come_image)
light_img = _src(modal.Image.debian_slim(python_version="3.12").pip_install("numpy", "pillow"))

PP_URL = "https://github.com/sczhou/ProPainter.git"
PP_REL = "https://github.com/sczhou/ProPainter/releases/download/v0.1.0"
propainter_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0", "wget")
    .pip_install("torch==2.1.2", "torchvision==0.16.2", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("numpy<2", "av", "addict", "einops", "future", "scipy", "opencv-python", "matplotlib",
                 "scikit-image", "imageio-ffmpeg", "pyyaml", "requests", "timm", "yapf", "pillow")
    .run_commands(
        f"git clone {PP_URL} /opt/ProPainter && git -C /opt/ProPainter rev-parse HEAD > /opt/ProPainter/COMMIT",
        "mkdir -p /opt/ProPainter/weights && cd /opt/ProPainter/weights && "
        f"wget -q {PP_REL}/ProPainter.pth && wget -q {PP_REL}/recurrent_flow_completion.pth && "
        f"wget -q {PP_REL}/raft-things.pth",
    )
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("modal_come")
)


# ============================================================================================ tiện ích
def _hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


def _stage_hash(cfg: dict, stage: str) -> str:
    """Hash = config của stage + hash các stage nó phụ thuộc (đổi upstream -> downstream chạy lại)."""
    return _hash({"cfg": cfg[CFG_KEY[stage]], "deps": [_stage_hash(cfg, s) for s in DEPS[stage]], "stage": stage})


def _root(cfg) -> Path:
    return Path(VOL) / cfg["scene"] / "pipeline" / cfg["run_name"]


def _sdir(cfg, stage) -> Path:
    return _root(cfg) / DIRNAME[stage]


def _done(cfg, stage) -> bool:
    f = _sdir(cfg, stage) / "_done.json"
    return f.exists() and json.loads(f.read_text()).get("hash") == _stage_hash(cfg, stage)


def _begin(cfg, stage, force=False) -> Path:
    d = _sdir(cfg, stage)
    if force and d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    (d / "config_used.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    old = d / "_done.json"
    if old.exists() and json.loads(old.read_text()).get("hash") != _stage_hash(cfg, stage):
        old.unlink()                                             # config đổi -> làm lại, giữ checkpoint con nếu còn hợp lệ
    vol.commit()
    return d


def _finish(cfg, stage, t0, extra=None):
    d = _sdir(cfg, stage)
    info = {"stage": stage, "hash": _stage_hash(cfg, stage), "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
            "minutes_this_run": round((time.time() - t0) / 60, 2), **(extra or {})}
    (d / "_done.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    vol.commit()
    mc._tlog(f"[{stage}] xong: {json.dumps(info, ensure_ascii=False)}")


class _Committer:
    def __enter__(self):
        self.stop = threading.Event()
        def loop():
            while not self.stop.wait(30):
                try:
                    vol.commit()
                except Exception as e:
                    print(f"[commit] {e}", flush=True)
        self.th = threading.Thread(target=loop, daemon=True); self.th.start()
        return self

    def __exit__(self, *a):
        self.stop.set(); self.th.join(timeout=10); vol.commit()


def _prep_come(cfg):
    """Chép COLMAP + model CoMe từ Volume ra local (cùng đường dẫn lúc train để cfg_args khớp)."""
    V = Path(VOL) / cfg["scene"]
    data = Path("/tmp/teatime_colmap")
    for sub in ("images", "sparse"):
        shutil.copytree(V / "colmap" / sub, data / sub, dirs_exist_ok=True)
    local = Path("/tmp/come_out/teatime"); local.mkdir(parents=True, exist_ok=True)
    for name in mc.SMALL_FILES:
        shutil.copy2(V / "come" / name, local / name)
    shutil.copytree(V / "come" / "point_cloud", local / "point_cloud", dirs_exist_ok=True)
    return data, local


def _run_script(cfg, stage, script, extra):
    data, local = _prep_come(cfg)
    cfg_path = _sdir(cfg, stage) / "config_used.json"
    cmd = (f"PYTHONPATH=/opt/CoMe:/opt/pipeline python /opt/pipeline/{script} -m {local} -s {data} "
           f"--iteration {cfg['S2_come']['iterations']} --cfg {cfg_path} --out {_sdir(cfg, stage)} {extra}")
    with _Committer():
        mc._run_train(cmd, "/opt/CoMe", Path(VOL) / cfg["scene"] / "logs" / f"pipeline_{stage}.log")


GPU_KW = dict(gpu="L4", volumes={VOL: vol}, cpu=8, memory=32768,
              retries=modal.Retries(max_retries=5, initial_delay=10.0, backoff_coefficient=1.0))


# ============================================================================================ stages
@app.function(image=colmap_img, timeout=2 * 3600, **GPU_KW)
def s0(cfg: dict, force: bool = False):
    if _done(cfg, "S0") and not force:
        return "skip"
    t0 = time.time(); d = _begin(cfg, "S0", force)
    V = Path(VOL) / cfg["scene"]
    if not (V / "colmap" / "sparse" / "0" / "points3D.bin").exists():
        mc.colmap_teatime.local()                                  # checkpoint của chính colmap_teatime
    vol.reload()
    chk = json.loads((V / "colmap_check.json").read_text())
    (d / "summary.json").write_text(json.dumps({"colmap_dir": f"{cfg['scene']}/colmap", **chk}, indent=2))
    _finish(cfg, "S0", t0, {"registered": chk["registered"], "input_images": chk["input_images"]})
    return "done"


@app.function(image=light_img, volumes={VOL: vol}, timeout=1800)
def s1(cfg: dict, force: bool = False):
    if _done(cfg, "S1") and not force:
        return "skip"
    import numpy as np
    from PIL import Image
    t0 = time.time(); d = _begin(cfg, "S1", force)
    c = cfg["S1_masks"]
    assert c["source"] == "team_sam2", "v1 chỉ hỗ trợ team_sam2 (SAM 3 cần quyền HF)"
    lab = Path(VOL) / c["labels_dir"]
    files = sorted(lab.glob("frame_*.png"))
    ids = set()
    for f in files[::10]:
        ids |= set(np.unique(np.array(Image.open(f))).tolist())
    (d / "summary.json").write_text(json.dumps({"labels_dir": c["labels_dir"], "frames": len(files),
                                                "ids_seen_every_10th_frame": len(ids - {0})}, indent=2))
    _finish(cfg, "S1", t0, {"frames": len(files)})
    return "done"


@app.function(image=come_img, timeout=4 * 3600, **GPU_KW)
def s2(cfg: dict, force: bool = False):
    if _done(cfg, "S2") and not force:
        return "skip"
    t0 = time.time(); d = _begin(cfg, "S2", force)
    V = Path(VOL) / cfg["scene"]
    final = V / "come" / "point_cloud" / f"iteration_{cfg['S2_come']['iterations']}" / "point_cloud.ply"
    if not final.exists():
        mc.train_teatime.local()                                   # checkpoint mỗi 1000 iter của train_teatime
    vol.reload()
    summ = {"model_dir": f"{cfg['scene']}/come", "ply_MB": round(final.stat().st_size / 1e6, 1)}
    if (V / "come" / "metrics.json").exists():
        summ["metrics_test"] = json.loads((V / "come" / "metrics.json").read_text())
    (d / "summary.json").write_text(json.dumps(summ, indent=2))
    _finish(cfg, "S2", t0, summ)
    return "done"


@app.function(image=pipe_image, timeout=3 * 3600, **GPU_KW)
def s3(cfg: dict, force: bool = False):
    if _done(cfg, "S3") and not force:
        return "skip"
    t0 = time.time(); _begin(cfg, "S3", force)
    _run_script(cfg, "S3", "s3_label.py", f"--labels {Path(VOL) / cfg['S1_masks']['labels_dir']}")
    d = _sdir(cfg, "S3")
    _finish(cfg, "S3", t0, {"groups": json.loads((d / "groups.json").read_text())["objects"],
                            "support": json.loads((d / "support.json").read_text())})
    return "done"


@app.function(image=propainter_image, timeout=3 * 3600, **GPU_KW)
def s4a(cfg: dict, force: bool = False):
    """Inpaint video vùng các vật (ProPainter), từng đoạn 60 frame, đoạn nào xong thì lưu luôn."""
    if _done(cfg, "S4a") and not force:
        return "skip"
    import cv2
    import numpy as np
    t0 = time.time(); d = _begin(cfg, "S4a", force)
    c = cfg["S4_background"]["propainter"]
    V = Path(VOL) / cfg["scene"]
    groups = json.loads((_sdir(cfg, "S3") / "groups.json").read_text())
    ids = sorted({i for o in groups["objects"].values() for i in o["ids"]})
    names = sorted(p.stem for p in (V / "colmap" / "input").glob("frame_*.jpg"))
    (d / "inpainted").mkdir(exist_ok=True); (d / "masks").mkdir(exist_ok=True); (d / "preview").mkdir(exist_ok=True)
    chunk = 60
    log = V / "logs" / "pipeline_S4a.log"
    with _Committer():
        for k0 in range(0, len(names), chunk):
            part = names[k0:k0 + chunk]
            if all((d / "inpainted" / f"{n}.png").exists() for n in part):
                continue
            w = Path(f"/tmp/pp/chunk{k0}"); fr, ms = w / "frames", w / "masks"
            shutil.rmtree(w, ignore_errors=True); fr.mkdir(parents=True); ms.mkdir()
            for n in part:
                src = V / "colmap" / "images" / f"{n}.jpg"
                if not src.exists():
                    src = V / "colmap" / "input" / f"{n}.jpg"
                shutil.copy2(src, fr / f"{n}.jpg")
                lp = Path(VOL) / cfg["S1_masks"]["labels_dir"] / f"{n}.png"
                lab = cv2.imread(str(lp), cv2.IMREAD_UNCHANGED)
                assert lab is not None, f"không đọc được label {lp}"
                m = (np.isin(lab, ids).astype(np.uint8) * 255)
                cv2.imwrite(str(ms / f"{n}.png"), m)
                cv2.imwrite(str(d / "masks" / f"{n}.png"), m)
            cmd = (f"python inference_propainter.py --video {fr} --mask {ms} --output {w / 'out'} --save_frames "
                   f"--mask_dilation {c['mask_dilation']} --subvideo_length {c['subvideo_length']} "
                   f"--neighbor_length {c['neighbor_length']} --ref_stride {c['ref_stride']}"
                   + (" --fp16" if c["fp16"] else ""))
            mc._run_train(cmd, "/opt/ProPainter", log)
            outs = sorted((w / "out" / "frames" / "frames").glob("*.png"))
            assert len(outs) == len(part), f"ProPainter trả {len(outs)} frame cho đoạn {len(part)} frame"
            for o, n in zip(outs, part):
                shutil.copy2(o, d / "inpainted" / f"{n}.png")
            vol.commit()
            mc._tlog(f"[S4a] xong đoạn {k0}-{k0 + len(part) - 1}")
    for n in cfg["S3_label"]["preview_frames"]:
        a = cv2.imread(str(V / "colmap" / "input" / f"{n}.jpg"))
        m = cv2.imread(str(d / "masks" / f"{n}.png"), 0)
        b = cv2.imread(str(d / "inpainted" / f"{n}.png"))
        ov = a.copy(); ov[m > 0] = (0.5 * ov[m > 0] + np.array([0, 0, 255]) * 0.5).astype(np.uint8)
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
        cv2.imwrite(str(d / "preview" / f"{n}.jpg"), np.hstack([a, ov, b]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    _finish(cfg, "S4a", t0, {"frames": len(names), "mask_ids": ids,
                             "propainter_commit": Path("/opt/ProPainter/COMMIT").read_text().strip()})
    return "done"


@app.function(image=pipe_image, timeout=3 * 3600, **GPU_KW)
def s4b(cfg: dict, force: bool = False):
    if _done(cfg, "S4b") and not force:
        return "skip"
    t0 = time.time(); _begin(cfg, "S4b", force)
    _run_script(cfg, "S4b", "s4_background.py",
                f"--s3 {_sdir(cfg, 'S3')} --inpainted {_sdir(cfg, 'S4a') / 'inpainted'}")
    d = _sdir(cfg, "S4b")
    _finish(cfg, "S4b", t0, {"fill": json.loads((d / "fill.json").read_text()) if (d / "fill.json").exists() else {}})
    return "done"


@app.function(image=pipe_image, timeout=2 * 3600, **GPU_KW)
def s5(cfg: dict, force: bool = False):
    if _done(cfg, "S5") and not force:
        return "skip"
    t0 = time.time(); _begin(cfg, "S5", force)
    _run_script(cfg, "S5", "s5_objects.py", f"--s3 {_sdir(cfg, 'S3')}")
    _finish(cfg, "S5", t0)
    return "done"


@app.function(image=pipe_image, timeout=3 * 3600, **GPU_KW)
def s6(cfg: dict, force: bool = False):
    if _done(cfg, "S6") and not force:
        return "skip"
    t0 = time.time(); _begin(cfg, "S6", force)
    _run_script(cfg, "S6", "s6_mesh.py", f"--s3 {_sdir(cfg, 'S3')} --s5 {_sdir(cfg, 'S5')}")
    _finish(cfg, "S6", t0)
    return "done"


def _run_cpu(cfg, stage, script, extra):
    cmd = f"python /opt/pipeline/{script} --cfg {_sdir(cfg, stage) / 'config_used.json'} --out {_sdir(cfg, stage)} {extra}"
    with _Committer():
        mc._run_train(cmd, "/opt/pipeline", Path(VOL) / cfg["scene"] / "logs" / f"pipeline_{stage}.log")


CPU_KW = dict(volumes={VOL: vol}, cpu=4, memory=8192,
              retries=modal.Retries(max_retries=5, initial_delay=10.0, backoff_coefficient=1.0))


@app.function(image=sim_img, timeout=1800, **CPU_KW)
def s7(cfg: dict, force: bool = False):
    if _done(cfg, "S7") and not force:
        return "skip"
    t0 = time.time(); _begin(cfg, "S7", force)
    _run_cpu(cfg, "S7", "s7_physics.py", f"--s6 {_sdir(cfg, 'S6')}")
    import mujoco
    _finish(cfg, "S7", t0, {"mujoco": mujoco.__version__,
                            "check": json.loads((_sdir(cfg, "S7") / "physics.json").read_text())["check"]})
    return "done"


@app.function(image=sim_img, timeout=3600, **CPU_KW)
def s8(cfg: dict, force: bool = False):
    if _done(cfg, "S8") and not force:
        return "skip"
    t0 = time.time(); _begin(cfg, "S8", force)
    _run_cpu(cfg, "S8", "s8_sim.py", f"--s7 {_sdir(cfg, 'S7')}")
    _finish(cfg, "S8", t0, {"summary": json.loads((_sdir(cfg, "S8") / "summary.json").read_text())})
    return "done"


@app.function(image=pipe_image, timeout=3 * 3600, **GPU_KW)
def s9(cfg: dict, force: bool = False):
    if _done(cfg, "S9") and not force:
        return "skip"
    t0 = time.time(); _begin(cfg, "S9", force)
    _run_script(cfg, "S9", "s9_render.py", f"--s4b {_sdir(cfg, 'S4b')} --s5 {_sdir(cfg, 'S5')} "
                                           f"--s6 {_sdir(cfg, 'S6')} --s8 {_sdir(cfg, 'S8')}")
    _finish(cfg, "S9", t0)
    return "done"


@app.function(image=sim_img, timeout=1800, **CPU_KW)
def _s10_cpu(cfg: dict):
    """Phần CPU của S10 (mesh S6 + mô phỏng S8) rồi gộp bảng; mỗi phần có json riêng làm checkpoint."""
    vol.reload()
    d = _sdir(cfg, "S10")
    if not (d / "eval_cpu.json").exists():
        _run_cpu(cfg, "S10", "s10_eval_cpu.py", f"--s5 {_sdir(cfg, 'S5')} --s6 {_sdir(cfg, 'S6')} "
                                                f"--s7 {_sdir(cfg, 'S7')} --s8 {_sdir(cfg, 'S8')}")
    mc._run_train(f"python /opt/pipeline/s10_report.py --gpu {d / 'eval_gpu.json'} --cpu {d / 'eval_cpu.json'} --out {d}",
                  "/opt/pipeline", Path(VOL) / cfg["scene"] / "logs" / "pipeline_S10.log")
    vol.commit()


@app.function(image=pipe_image, timeout=3 * 3600, **GPU_KW)
def s10(cfg: dict, force: bool = False):
    """Đo lường mọi stage (pipeline/s10_*.py). Cần LERF-Mask teatime trên Volume tại S10_eval.lerf_dir."""
    if _done(cfg, "S10") and not force:
        return "skip"
    t0 = time.time(); d = _begin(cfg, "S10", force)
    V = Path(VOL) / cfg["scene"]
    if not (d / "eval_gpu.json").exists():
        _run_script(cfg, "S10", "s10_eval_gpu.py",
                    f"--lerf {Path(VOL) / cfg['S10_eval']['lerf_dir']} --s3 {_sdir(cfg, 'S3')} "
                    f"--s4b {_sdir(cfg, 'S4b')} --s5 {_sdir(cfg, 'S5')} --s4a {_sdir(cfg, 'S4a')} "
                    f"--labels {Path(VOL) / cfg['S1_masks']['labels_dir']} "
                    f"--n_input {len(list((V / 'colmap' / 'input').glob('*.jpg')))}")
    vol.commit()
    _s10_cpu.remote(cfg)
    vol.reload()
    _finish(cfg, "S10", t0)
    return "done"


@app.function(image=pipe_image, timeout=1800, gpu="L4", volumes={VOL: vol}, cpu=8, memory=32768)
def debug_tsdf(cfg: dict, obj: str = "coc_su", chunks: str = "50000,200000,1000000"):
    data, local = _prep_come(cfg)
    objs = list(cfg["S3_label"]["objects"]) if obj == "all" else [obj]
    for ob, ch in [(o, c) for o in objs for c in chunks.split(",")]:
        obj = ob
        cmd = (f"CUDA_LAUNCH_BLOCKING=1 PYTHONPATH=/opt/CoMe:/opt/pipeline python /opt/pipeline/debug_tsdf.py "
               f"-m {local} -s {data} --iteration {cfg['S2_come']['iterations']} "
               f"--cfg {_sdir(cfg, 'S6') / 'config_used.json'} --out /tmp/dbg "
               f"--ply {_sdir(cfg, 'S5') / 'objects' / obj / 'gaussians.ply'} --chunk {ch}")
        try:
            mc._run_train(cmd, "/opt/CoMe", Path(VOL) / cfg["scene"] / "logs" / "pipeline_debug.log")
        except Exception as e:
            print(f"chunk {ch}: lỗi {str(e)[-300:]}", flush=True)


@app.local_entrypoint()
def dbg(config: str = "configs/pipeline/teatime_v1.json", obj: str = "coc_su", chunks: str = "50000,200000,1000000"):
    debug_tsdf.remote(_load_cfg(config), obj, chunks)


FN = {"S0": s0, "S1": s1, "S2": s2, "S3": s3, "S4a": s4a, "S4b": s4b, "S5": s5, "S6": s6, "S7": s7, "S8": s8,
      "S9": s9, "S10": s10}


# ============================================================================================ điều phối (máy local)
def _load_cfg(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@app.local_entrypoint()
def run(config: str = "configs/pipeline/teatime_v1.json", until: str = "S6", only: str = "", force: str = ""):
    cfg = _load_cfg(config)
    stages = STAGES[:STAGES.index(until) + 1] if not only else [only]
    forced = set()
    if force:
        forced = set(STAGES[STAGES.index(force):])                # ép chạy lại stage này và mọi stage sau
    print(f"Pipeline {cfg['scene']}/{cfg['run_name']}: {stages}  (ép chạy lại: {sorted(forced) or 'không'})")
    done = _done_map.remote(cfg)                                   # kiểm tra bằng container CPU nhẹ, khỏi bật GPU
    for st in stages:
        if done.get(st) and st not in forced:
            print(f"<-- {st}: skip (đã xong, config không đổi)", flush=True)
            continue
        t = time.time()
        print(f"--> {st} ...", flush=True)
        r = FN[st].remote(cfg, st in forced)
        print(f"<-- {st}: {r} ({(time.time() - t) / 60:.1f} phút)", flush=True)
    print("XONG. Xem kết quả: python -m modal run modal_pipeline.py::pull --config", config)


@app.function(image=light_img, volumes={VOL: vol}, timeout=300)
def _done_map(cfg: dict) -> dict:
    vol.reload()
    return {st: _done(cfg, st) for st in STAGES}


@app.function(image=light_img, volumes={VOL: vol}, timeout=300)
def _status(cfg: dict):
    vol.reload()
    rows = []
    for st in STAGES:
        f = _sdir(cfg, st) / "_done.json"
        if f.exists():
            info = json.loads(f.read_text())
            ok = info.get("hash") == _stage_hash(cfg, st)
            rows.append(f"{st:4s} {'XONG' if ok else 'CŨ (config đã đổi)'}  {info.get('finished')}  "
                        f"{info.get('minutes_this_run')} phút")
        else:
            rows.append(f"{st:4s} chưa xong")
    return "\n".join(rows)


@app.local_entrypoint()
def status(config: str = "configs/pipeline/teatime_v1.json"):
    print(_status.remote(_load_cfg(config)))


@app.local_entrypoint()
def pull(config: str = "configs/pipeline/teatime_v1.json", with_ply: bool = False):
    """Tải mọi file trừ .ply/.pt (trừ khi --with-ply) về Data/<scene>/pipeline/<run>/."""
    cfg = _load_cfg(config)
    remote_root = f"/{cfg['scene']}/pipeline/{cfg['run_name']}"
    local_root = REPO_DIR / "Data" / cfg["scene"] / "pipeline" / cfg["run_name"]
    n = 0
    for e in vol.listdir(remote_root, recursive=True):
        if e.type != modal.volume.FileEntryType.FILE:
            continue
        if not with_ply and (e.path.endswith(".ply") or e.path.endswith(".pt")):
            continue
        rel = e.path[len(remote_root.lstrip("/")):].lstrip("/") if e.path.startswith(remote_root.lstrip("/")) else e.path
        dst = local_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "wb") as f:
            for chunk in vol.read_file(e.path):
                f.write(chunk)
        n += 1
    print(f"Đã tải {n} file -> {local_root}")
