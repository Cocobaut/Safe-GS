"""
S1 v2: mask có ID cho teatime bằng SAM 3 (facebookresearch/sam3 @ f1647e8, checkpoint HF facebook/sam3) theo prompt chữ.

    python -m modal deploy modal_sam3.py && python -c "import modal; modal.Function.from_name('safe-gs-sam3','_run').spawn()"
    python -m modal run modal_sam3.py::pull                 # tải labels + preview về Data/teatime/sam3_masks/

Dùng đúng API video của README / examples/sam3_video_predictor_example.ipynb: start_session (thư mục JPEG đặt tên
<số>.jpg), mỗi prompt: reset_session -> add_prompt(text, frame 0) -> propagate_in_video (mặc định cả 2 chiều).
Ảnh đầu vào = 177 ảnh đã undistort của COLMAP (cùng ảnh CoMe train) -> mask khớp camera.
Gộp: mỗi instance của prompt p nhận nhãn p*10 + k (k theo thứ tự xuất hiện); prompt sau đè prompt trước (vật nhỏ sau).
Checkpoint: mỗi prompt lưu masks_<p>.npz lên Volume ngay khi xong; chạy lại thì bỏ qua prompt đã có.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import modal

import modal_come as mc

app = modal.App("safe-gs-sam3")
vol, VOL = mc.vol, mc.VOL
SAM3_COMMIT = "f1647e8d0c1d280991f4d1b7d974c2704537fe66"
ROOT = Path(VOL) / "teatime" / "sam3_masks"

# thứ tự = độ ưu tiên tăng dần (prompt sau đè lên)
PROMPTS = ["table", "teddy bear", "stuffed sheep", "paper napkin", "paper bag", "plate", "glass of tea",
           "coffee mug", "apple", "cookie", "spoon", "creamer cup",
           # lần 2: 4 và 12 ra 0 instance; cà phê trong cốc không thuộc mask cốc
           "napkin", "coffee", "creamer",
           # lần 3 (thử bằng chế độ ảnh, xem probe/cand_*.jpg): vật nhỏ còn thiếu
           "small white cup", "puddle of liquid", "blue snack package"]
# instance bắt nhầm (xem preview lần 1): 21 = "teddy bear" trên con cừu, 31 = "stuffed sheep" trên con gấu,
# 61 / 91 = plate / apple chỉ 2 / 7 frame; 142 = "coffee" trên túi giấy
# + prompt "table": chỉ giữ 13 (bàn chính). 10-12, 14-19 bắt nhầm đầu gấu, cổ cừu, vũng trà lốm đốm, bàn xa
DROP = [21, 31, 61, 91, 142, 10, 11, 12, 14, 15, 16, 17, 18, 19]
# gộp: cà phê vào cốc, trà vào ly (mask cốc / ly bị thủng đúng chỗ đó), các miếng bánh quy thành 1 nhóm
MERGE = {140: 80, 141: 70, 101: 100, 102: 100, 103: 100, 104: 100}
# prompt lần 3 hay bắt cả vật đã có nhãn (vd "small white cup" bắt cả cốc lớn): bỏ instance trong frame đó nếu
# > 30% diện tích đã có nhãn vật khác (trừ bàn 13)
CHECK_OVERLAP = ["small white cup", "puddle of liquid", "blue snack package"]
MIN_PIECE = 40                      # xoá mảnh rời nhỏ hơn 40 px của mỗi nhãn trong mỗi frame

img = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.10.0", "torchvision", index_url="https://download.pytorch.org/whl/cu128")   # README
    .run_commands(f"git clone -q https://github.com/facebookresearch/sam3.git /opt/sam3 && "
                  f"git -C /opt/sam3 checkout -q {SAM3_COMMIT} && cd /opt/sam3 && pip install -e .")
    .pip_install("opencv-python-headless", "pillow", "einops", "pycocotools",
                 "psutil")                                    # sam3_video_predictor.py import, không có trong pyproject
    .env({"PYTHONUNBUFFERED": "1", "HF_HOME": f"{VOL}/hf_cache"})
    .add_local_python_source("modal_come")
)


@app.function(image=img, gpu="L4", volumes={VOL: vol}, timeout=6 * 3600, cpu=8, memory=49152,
              secrets=[modal.Secret.from_name("huggingface")],
              retries=modal.Retries(max_retries=3, initial_delay=10.0, backoff_coefficient=1.0))
def _run(prompts: list = None):
    import numpy as np
    import cv2
    from sam3.model_builder import build_sam3_video_predictor

    vol.reload()
    prompts = prompts or PROMPTS
    src = sorted((Path(VOL) / "teatime" / "colmap" / "images").glob("frame_*.jpg"))
    frames = Path("/tmp/frames"); shutil.rmtree(frames, ignore_errors=True); frames.mkdir()
    for i, f in enumerate(src):
        shutil.copy2(f, frames / f"{i:05d}.jpg")                # loader cần tên <số>.jpg
    names = [f.stem for f in src]
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "frames.json").write_text(json.dumps(names))
    H, W = cv2.imread(str(src[0])).shape[:2]

    pred = build_sam3_video_predictor()
    sid = pred.handle_request(dict(type="start_session", resource_path=str(frames)))["session_id"]
    for p, text in enumerate(prompts, start=1):
        out_f = ROOT / f"masks_{p:02d}.npz"
        if out_f.exists():
            mc._tlog(f"[sam3] skip '{text}'"); continue
        t0 = time.time()
        pred.handle_request(dict(type="reset_session", session_id=sid))
        pred.handle_request(dict(type="add_prompt", session_id=sid, frame_index=0, text=text))
        per = {}
        for r in pred.handle_stream_request(dict(type="propagate_in_video", session_id=sid)):
            o = r["outputs"]
            per[r["frame_index"]] = {int(i): np.packbits(m) for i, m in zip(o["out_obj_ids"].tolist(), o["out_binary_masks"])
                                     if m.any()}
        ids = sorted({i for d in per.values() for i in d}, key=lambda i: min(f for f, d in per.items() if i in d))
        arr = {}
        for f, d in per.items():
            for i, m in d.items():
                arr[f"{f}_{i}"] = m
        np.savez_compressed(out_f, **arr)
        info = {"prompt": text, "instances": [{"obj_id": i, "label": p * 10 + k,
                                                "frames": sum(i in d for d in per.values())}
                                               for k, i in enumerate(ids[:10])],
                "minutes": round((time.time() - t0) / 60, 2)}
        (ROOT / f"masks_{p:02d}.json").write_text(json.dumps(info, ensure_ascii=False))
        vol.commit()
        mc._tlog(f"[sam3] '{text}': {len(ids)} instance, {info['minutes']} phút")

    # gộp thành ảnh nhãn (uint8) cho từng frame, prompt sau đè prompt trước
    lab_dir = ROOT / "labels"; lab_dir.mkdir(exist_ok=True)
    prev = ROOT / "preview"; prev.mkdir(exist_ok=True)
    labels = np.zeros((len(names), H, W), np.uint8)
    prompt_of = {}
    for p in range(1, len(prompts) + 1):
        info = json.loads((ROOT / f"masks_{p:02d}.json").read_text())
        z = np.load(ROOT / f"masks_{p:02d}.npz")
        lab_of = {d["obj_id"]: d["label"] for d in info["instances"] if d["label"] not in DROP}
        for d in info["instances"]:
            if d["label"] not in DROP:
                prompt_of[MERGE.get(d["label"], d["label"])] = info["prompt"] if d["label"] not in MERGE else \
                    prompt_of.get(MERGE[d["label"]], info["prompt"])
        for key in z.files:
            f, i = map(int, key.split("_"))
            if i in lab_of:
                m = np.unpackbits(z[key])[: H * W].reshape(H, W).astype(bool)
                if info["prompt"] in CHECK_OVERLAP:
                    occ = (labels[f] != 0) & (labels[f] != 13)
                    if (m & occ).sum() > 0.3 * m.sum():
                        continue
                labels[f][m] = MERGE.get(lab_of[i], lab_of[i])
    # xoá mảnh rời nhỏ
    for f in range(len(names)):
        for lb in np.unique(labels[f]):
            if lb == 0:
                continue
            n, cc, st, _ = cv2.connectedComponentsWithStats((labels[f] == lb).astype(np.uint8), connectivity=8)
            for c in range(1, n):
                if st[c, cv2.CC_STAT_AREA] < MIN_PIECE:
                    labels[f][cc == c] = 0
    legend = {}
    for lb in np.unique(labels):
        if lb:
            legend[int(lb)] = {"prompt": prompt_of.get(int(lb), "?"), "frames": int((labels == lb).any((1, 2)).sum())}
    merged_note = {str(k): v for k, v in MERGE.items()}
    (ROOT / "merge_rules.json").write_text(json.dumps({"merge": merged_note, "drop": DROP, "min_piece_px": MIN_PIECE,
                                                       "check_overlap": CHECK_OVERLAP}, indent=1))
    rng = np.random.default_rng(0); pal = rng.integers(40, 255, (256, 3)).astype(np.uint8); pal[0] = 0
    for k, n in enumerate(names):
        cv2.imwrite(str(lab_dir / f"{n}.png"), labels[k])
        if k % 20 == 0:
            im = cv2.imread(str(src[k]))
            ov = (0.45 * im + 0.55 * pal[labels[k]][..., ::-1]).astype(np.uint8)
            for lb in np.unique(labels[k]):
                if lb == 0:
                    continue
                ys, xs = np.nonzero(labels[k] == lb)
                cv2.putText(ov, f"{lb}:{legend[int(lb)]['prompt']}", (int(xs.mean()) - 30, int(ys.mean())),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.imwrite(str(prev / f"{n}.jpg"), np.hstack([im, ov]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    (ROOT / "legend.json").write_text(json.dumps(legend, indent=1, ensure_ascii=False))
    vol.commit()
    return legend


LERF_PROMPTS = ["apple", "bag of cookies", "coffee mug", "cookies on a plate", "paper napkin", "plate", "sheep",
                "spoon handle", "stuffed bear", "tea in a glass"]
CANDIDATES = ["sugar packet", "tea bag", "plastic creamer cup", "small white cup", "spilled tea", "puddle of liquid",
              "blue snack package", "candy wrapper", "coffee creamer", "cookie crumbs"]


@app.function(image=img, gpu="L4", volumes={VOL: vol}, timeout=3600, cpu=8, memory=32768,
              secrets=[modal.Secret.from_name("huggingface")])
def _probe():
    """Chế độ ảnh của SAM 3 (README: Sam3Processor): (1) 10 prompt LERF-Mask trên ảnh test, lưu xác suất mask để chọn
    ngưỡng / độ nở bằng ground truth; (2) prompt ứng viên cho vật nhỏ còn thiếu trên vài frame của video."""
    import cv2
    import numpy as np
    from PIL import Image
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    import torch
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()     # như examples/sam3_image_predictor_example.ipynb
    vol.reload()
    proc = Sam3Processor(build_sam3_image_model(), confidence_threshold=0.05)
    out = ROOT / "probe"; out.mkdir(parents=True, exist_ok=True)

    def run(path, prompt):
        st = proc.set_image(Image.open(path).convert("RGB"))
        st = proc.set_text_prompt(state=st, prompt=prompt)
        if "masks_logits" not in st or len(st["scores"]) == 0:
            return np.zeros((0, 1, 1), np.uint8), np.zeros(0, np.float32)
        p = (st["masks_logits"][:, 0].float().cpu().numpy() * 255).astype(np.uint8)
        return p, st["scores"].float().cpu().numpy()

    lerf = Path(VOL) / "teatime" / "benchmarks" / "lerf_mask" / "teatime" / "images"
    arr = {}
    for v in ("test_0", "test_2"):
        st0 = proc.set_image(Image.open(lerf / f"{v}.jpg").convert("RGB"))
        for pr in LERF_PROMPTS:
            proc.reset_all_prompts(st0)
            st = proc.set_text_prompt(state=st0, prompt=pr)
            if len(st["scores"]):
                arr[f"{v}|{pr}|p"] = (st["masks_logits"][:, 0].float().cpu().numpy() * 255).astype(np.uint8)
                arr[f"{v}|{pr}|s"] = st["scores"].float().cpu().numpy()
    np.savez_compressed(out / "lerf_probs.npz", **arr)
    vol.commit()
    mc._tlog(f"[sam3] probe LERF: {len(arr) // 2} (view, prompt) có mask")

    src = sorted((Path(VOL) / "teatime" / "colmap" / "images").glob("frame_*.jpg"))
    ks = [0, 20, 60, 90, 120, 150]
    rows = []
    rng = np.random.default_rng(1); pal = rng.integers(60, 255, (64, 3)).astype(np.uint8)
    for pr in CANDIDATES:
        tiles = []
        for k in ks:
            im = cv2.imread(str(src[k]))
            p, s = run(src[k], pr)
            ov = im.copy()
            n = 0
            for j in range(len(s)):
                if s[j] < 0.3:
                    continue
                m = cv2.resize(p[j], (im.shape[1], im.shape[0])) > 127
                ov[m] = (0.4 * ov[m] + 0.6 * pal[j % 64][::-1]).astype(np.uint8); n += 1
            cv2.putText(ov, f"{pr} #{k}: {n} inst, max {s.max() if len(s) else 0:.2f}", (6, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            tiles.append(cv2.resize(ov, (480, int(ov.shape[0] * 480 / ov.shape[1]))))
        rows.append(np.hstack(tiles[:3])); rows.append(np.hstack(tiles[3:]))
    for g in range(0, len(rows), 6):
        cv2.imwrite(str(out / f"cand_{g // 6}.jpg"), np.vstack(rows[g:g + 6]), [cv2.IMWRITE_JPEG_QUALITY, 80])
    vol.commit()
    return "ok"


@app.local_entrypoint()
def pull_probe():
    out = Path(__file__).resolve().parent / "Data" / "teatime" / "sam3_masks" / "probe"
    out.mkdir(parents=True, exist_ok=True)
    for e in vol.listdir("teatime/sam3_masks/probe"):
        with open(out / Path(e.path).name, "wb") as f:
            for c in vol.read_file(e.path):
                f.write(c)
    print(f"đã tải về {out}")


@app.local_entrypoint()
def pull():
    out = Path(__file__).resolve().parent / "Data" / "teatime" / "sam3_masks"
    for sub in ("labels", "preview"):
        (out / sub).mkdir(parents=True, exist_ok=True)
        for e in vol.listdir(f"teatime/sam3_masks/{sub}"):
            with open(out / sub / Path(e.path).name, "wb") as f:
                for c in vol.read_file(e.path):
                    f.write(c)
    for name in ("legend.json", "frames.json"):
        with open(out / name, "wb") as f:
            for c in vol.read_file(f"teatime/sam3_masks/{name}"):
                f.write(c)
    print(f"đã tải về {out}")
