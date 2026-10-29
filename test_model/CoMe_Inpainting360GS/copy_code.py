"""Copy Inpaint360GS + CoMe rasterizer into src/ (once; code is then edited there) and build the reuse steps 0-3.

Run locally once (plan.md, sections 2 and 4):
    python copy_code.py                 # src + step0-3 + checks
    python copy_code.py --only src      # code only (refuses if src/ already exists)
    python copy_code.py --only reuse    # step0-3 only

Originals (Based_Model/Inpaint360GS, Based_Model/CoMe, test_model/CoMe_Infusion) are only read.
"""
import argparse
import filecmp
import fnmatch
import json
import os
import shutil
import struct
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
SAFE = HERE.parents[1]                                   # F:\Safe-GS
INP = SAFE / "Based_Model" / "Inpaint360GS"
COME = SAFE / "Based_Model" / "CoMe"
INF = SAFE / "test_model" / "CoMe_Infusion"
INF_SUB = INF / "src" / "gaussian_splatting" / "submodules"
STEP2 = INF / "Step_2_Output_train_3DGS" / "teatime_come_step2"

SRC = HERE / "src"
OUT_ROOT = HERE / "outputs"
SCENE = "teatime_napkin"
NAPKIN_ID = 11                                           # verified in CoMe_Infusion/Step_1_Sync_ID

LAMA_COMMIT = "786f5936b27fb3dacd2b1ad799e4de968ea697e7"
LAMA_FILES = ["__init__.py", "aug.py", "datasets.py", "masks.py"]

# plan section 2: everything of Inpaint360GS except these
INP_IGNORE = [
    "outputs", "data", "output", "Paper", "media", "SIBR_viewers",
    "*/third_party/glm/doc", "*/third_party/glm/test",
    "seg/Entity", "seg/detectron2/docs", "seg/detectron2/tests",
    "*__pycache__", "*/build", "build", "*.pyc", "*.egg-info", ".git",
]
GLM_IGNORE = ["third_party/glm/doc", "third_party/glm/test", "*__pycache__", "build", "*.egg-info", ".git"]


def log(msg, fh=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n")
        fh.flush()


def ignored(rel, patterns):
    rel = rel.replace("\\", "/")
    parts = rel.split("/")
    for p in patterns:
        if fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(rel, "*/" + p):
            return True
        if "/" not in p and any(fnmatch.fnmatch(x, p) for x in parts):
            return True
    return False


def copy_tree(src, dst, patterns):
    """Copy src -> dst skipping ignored paths; returns list of (src_file, dst_file)."""
    pairs = []
    for root, dirs, files in os.walk(src):
        rel_root = os.path.relpath(root, src)
        rel_root = "" if rel_root == "." else rel_root
        dirs[:] = [d for d in dirs if not ignored(os.path.join(rel_root, d), patterns)]
        for f in files:
            rel = os.path.join(rel_root, f)
            if ignored(rel, patterns):
                continue
            s, d = Path(src) / rel, Path(dst) / rel
            d.parent.mkdir(parents=True, exist_ok=True)
            if not (d.exists() and d.stat().st_size == s.stat().st_size and filecmp.cmp(s, d, shallow=False)):
                shutil.copy2(s, d)
            pairs.append((s, d))
    return pairs


def next_free(path):
    """No overwrite of results: <path>, <path>_v2, <path>_v3 ..."""
    if not path.exists() or not any(path.iterdir()):
        return path
    k = 2
    while (path.parent / f"{path.name}_v{k}").exists():
        k += 1
    return path.parent / f"{path.name}_v{k}"


# --------------------------------------------------------------------------------------------------
# src/
# --------------------------------------------------------------------------------------------------
def build_src(fh):
    if SRC.exists() and any(SRC.iterdir()):
        # code is edited directly in src/ -> never copy over it again
        raise SystemExit(f"{SRC} already exists (code is edited there); refusing to overwrite")
    t0 = time.time()
    copied = []
    log(f"copy {INP} -> {SRC}", fh)
    copied += copy_tree(INP, SRC, INP_IGNORE)
    log(f"  {len(copied)} files", fh)

    for name, dst in [("diff-gaussian-rasterization-come", "diff-gaussian-rasterization-come"),
                      ("simple-knn", "simple-knn-come")]:
        pairs = copy_tree(INF_SUB / name, SRC / "submodules" / dst, GLM_IGNORE)
        log(f"copy {INF_SUB / name} -> src/submodules/{dst}: {len(pairs)} files", fh)
        copied += pairs

    d = SRC / "utils" / "come_depth_utils.py"
    shutil.copy2(COME / "utils" / "depth_utils.py", d)
    copied.append((COME / "utils" / "depth_utils.py", d))

    # original GaussianModel kept as parent class for the CoMe-aware gaussian_model.py edited in src/
    d = SRC / "scene" / "gaussian_model_inpaint360.py"
    shutil.copy2(INP / "scene" / "gaussian_model.py", d)
    copied.append((INP / "scene" / "gaussian_model.py", d))

    # LaMa training/data (missing from the released repo)
    lama_dir = SRC / "LaMa" / "saicinpainting" / "training" / "data"
    lama_dir.mkdir(parents=True, exist_ok=True)
    for f in LAMA_FILES:
        p = lama_dir / f
        if not p.exists() or p.stat().st_size == 0:
            url = f"https://raw.githubusercontent.com/advimman/lama/{LAMA_COMMIT}/saicinpainting/training/data/{f}"
            urllib.request.urlretrieve(url, p)
        log(f"LaMa data/{f}: {p.stat().st_size} B", fh)

    # check: byte-compare every copied file
    bad = [str(d) for s, d in copied if not filecmp.cmp(s, d, shallow=False)]
    summary = {"files_copied": len(copied), "byte_mismatch": bad, "seconds": round(time.time() - t0, 1)}
    (HERE / "copy_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"src check: {len(copied)} copied, {len(bad)} mismatched", fh)
    if bad:
        raise SystemExit(f"byte mismatch: {bad[:10]}")


# --------------------------------------------------------------------------------------------------
# COLMAP binary readers (enough for checks)
# --------------------------------------------------------------------------------------------------
def read_cameras_bin(path):
    model_params = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
                    4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8)}
    cams = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cid, mid, w, h = struct.unpack("<iiQQ", f.read(24))
            name, k = model_params[mid]
            cams[cid] = {"model": name, "width": w, "height": h,
                         "params": list(struct.unpack("<" + "d" * k, f.read(8 * k)))}
    return cams


def read_images_bin(path):
    imgs = []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            iid = struct.unpack("<i", f.read(4))[0]
            f.read(8 * 7)
            cid = struct.unpack("<i", f.read(4))[0]
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            npts = struct.unpack("<Q", f.read(8))[0]
            f.read(24 * npts)
            imgs.append({"id": iid, "camera_id": cid, "name": name.decode()})
    return imgs


def ply_header(path):
    props, n = [], None
    with open(path, "rb") as f:
        while (line := f.readline().decode("ascii", "ignore").strip()) != "end_header":
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                props.append(line.split()[-1])
    return n, props


# --------------------------------------------------------------------------------------------------
# outputs/<scene>/step0-3
# --------------------------------------------------------------------------------------------------
def step0(out, fh):
    d = out / "step0_reuse_colmap"
    src = STEP2 / "colmap"
    shutil.copytree(src / "images", d / "images")
    shutil.copytree(src / "sparse" / "0", d / "sparse" / "0")
    cams = read_cameras_bin(d / "sparse" / "0" / "cameras.bin")
    imgs = sorted(read_images_bin(d / "sparse" / "0" / "images.bin"), key=lambda x: x["name"])
    names = [x["name"] for x in imgs]
    (d / "registered_images.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    files = sorted(os.listdir(d / "images"))
    sizes = {Image.open(d / "images" / f).size for f in files}
    ok = (len(files) == 177 and files == names and all(c["model"] == "PINHOLE" for c in cams.values())
          and sizes == {(994, 738)})
    s = {"source": str(src), "images": len(files), "registered": len(names), "image_sizes": sorted(sizes),
         "cameras": cams, "missing_vs_raw_180": sorted({f"frame_{i:05d}.jpg" for i in range(1, 181)} - set(names)),
         "check_ok": ok}
    (d / "summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    log(f"step0: {len(files)} images, {len(names)} registered, sizes {sorted(sizes)}, "
        f"cameras {[c['model'] for c in cams.values()]} -> {'OK' if ok else 'FAIL'}", fh)
    return ok, names


def step1(out, fh):
    d = out / "step1_reuse_come_3dgs"
    src = STEP2 / "come"
    shutil.copytree(src, d, ignore=shutil.ignore_patterns("chkpnt*.pth"))
    ply = d / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    n, props = ply_header(ply)
    same = filecmp.cmp(src / "point_cloud" / "iteration_30000" / "point_cloud.ply", ply, shallow=False)
    cfg = json.loads((d / "config.json").read_text())
    ok = same and "confidence" in props and "filter_3D" in props and n > 0
    s = {"source": str(src), "excluded": ["chkpnt29000.pth"], "num_gaussians": n, "ply_properties": props,
         "splatting_config": cfg, "ply_byte_identical": same,
         "files": sorted(str(p.relative_to(d)).replace("\\", "/") for p in d.rglob("*") if p.is_file()),
         "check_ok": ok}
    (d / "summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    log(f"step1: {n} Gaussians, confidence={'confidence' in props}, filter_3D={'filter_3D' in props}, "
        f"byte-identical={same} -> {'OK' if ok else 'FAIL'}", fh)
    return ok


def step2(out, fh, names):
    d = out / "step2_reuse_come_renders" / "renders"
    src = INF / "Step_3_Output_render_3DGS"
    shutil.copytree(src, d)
    files = sorted(os.listdir(d))
    sizes = {Image.open(d / f).size for f in files}
    stems = [Path(f).stem for f in files]
    ok = len(files) == 177 and stems == [Path(x).stem for x in names] and sizes == {(994, 738)}
    s = {"source": str(src), "renders": len(files), "sizes": sorted(sizes),
         "names_match_registered": stems == [Path(x).stem for x in names], "check_ok": ok}
    (d.parent / "summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    log(f"step2: {len(files)} renders, sizes {sorted(sizes)}, names match COLMAP={s['names_match_registered']} "
        f"-> {'OK' if ok else 'FAIL'}", fh)
    return ok


def step3(out, fh, names):
    d = out / "step3_reuse_napkin_reference" / f"napkin_mask_id{NAPKIN_ID}"
    d.mkdir(parents=True)
    src = INF / "Step_1_Sync_ID" / "labels"
    per_frame = {}
    for x in names:
        stem = Path(x).stem
        lab = np.array(Image.open(src / f"{stem}.png"))
        m = (lab == NAPKIN_ID).astype(np.uint8) * 255
        Image.fromarray(m).save(d / f"{stem}.png")
        per_frame[stem] = round(100.0 * (m > 0).mean(), 4)
    vis = [k for k, v in per_frame.items() if v > 0]
    ok = len(per_frame) == 177 and len(vis) > 0
    s = {"source": str(src), "napkin_id": NAPKIN_ID, "values": [0, 255], "frames": len(per_frame),
         "frames_with_napkin": len(vis), "mean_area_pct_visible": round(float(np.mean([per_frame[k] for k in vis])), 4),
         "area_pct_per_frame": per_frame,
         "note": "reference only: used to pick the napkin ID after HQ-SAM (step6), never used for training",
         "check_ok": ok}
    (d.parent / "summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    log(f"step3: {len(per_frame)} masks, napkin visible in {len(vis)} frames, "
        f"mean area {s['mean_area_pct_visible']}% -> {'OK' if ok else 'FAIL'}", fh)
    return ok


def build_reuse():
    out = next_free(OUT_ROOT / SCENE)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    log(f"outputs -> {out}")
    results = {}
    with open(out / "logs" / "step0_reuse_colmap.log", "w", encoding="utf-8") as fh:
        results["step0_reuse_colmap"], names = step0(out, fh)
    with open(out / "logs" / "step1_reuse_come_3dgs.log", "w", encoding="utf-8") as fh:
        results["step1_reuse_come_3dgs"] = step1(out, fh)
    with open(out / "logs" / "step2_reuse_come_renders.log", "w", encoding="utf-8") as fh:
        results["step2_reuse_come_renders"] = step2(out, fh, names)
    with open(out / "logs" / "step3_reuse_napkin_reference.log", "w", encoding="utf-8") as fh:
        results["step3_reuse_napkin_reference"] = step3(out, fh, names)
    log(f"reuse steps: {results}")
    if not all(results.values()):
        raise SystemExit("a reuse step failed its check")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["src", "reuse"])
    a = ap.parse_args()
    if a.only in (None, "src"):
        (HERE / "logs_local").mkdir(exist_ok=True)
        with open(HERE / "logs_local" / "copy_code_src.log", "w", encoding="utf-8") as fh:
            build_src(fh)
    if a.only in (None, "reuse"):
        build_reuse()
