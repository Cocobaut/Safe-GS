"""Chuan hoa mot canh COLMAP (da undistort, PINHOLE) ve dang pipeline dung: <out>/images/frame_XXXXX.jpg + <out>/sparse/0/*.bin.

Ten anh trong images.bin duoc doi thanh frame_<so>.jpg (giu so thu tu goc) va file anh duoc chep / doi duoi cho khop.
Khong chep points3D.ply (3DGS se tu tao tu points3D.bin). Dung chung cho moi canh, khong viet cung ten canh.

    python tools/prep_scene_colmap.py --src Data/figurines/prep/x/figurines --out Data/figurines/colmap
"""
import argparse
import re
import shutil
import struct
from pathlib import Path


def rename(stem: str) -> str:
    m = re.search(r"(\d+)$", stem)
    return f"frame_{int(m.group(1)):05d}" if m else stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="thu muc co images/ va sparse/0/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", default="", help="regex ten anh (stem) bi bo khoi images.bin, vd ^test_ (view test LERF giu lai de cham)")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    sp = src / "sparse" / "0"
    for f in ("cameras.bin", "points3D.bin"):
        shutil.copy2(sp / f, out / "sparse" / "0" / f)
    cam = (sp / "cameras.bin").read_bytes()
    assert struct.unpack("<i", cam[12:16])[0] in (0, 1), "can camera PINHOLE / SIMPLE_PINHOLE (anh da undistort)"
    files = {p.stem: p for p in (src / "images").iterdir() if p.is_file()}
    b = (sp / "images.bin").read_bytes()
    n = struct.unpack("<Q", b[:8])[0]
    o, res, mapping, kept = 8, [], {}, 0
    for _ in range(n):
        head = b[o:o + 64]; o += 64
        e = b.index(b"\0", o); name = b[o:e].decode(); o = e + 1
        npts = struct.unpack("<Q", b[o:o + 8])[0]
        pts = b[o:o + 8 + 24 * npts]; o += 8 + 24 * npts
        if a.exclude and re.search(a.exclude, Path(name).stem):
            continue
        kept += 1
        new = rename(Path(name).stem) + ".jpg"
        mapping[name] = new
        res += [head, new.encode() + b"\0", pts]
        f = files[Path(name).stem]
        dst = out / "images" / new
        if f.suffix.lower() in (".jpg", ".jpeg"):
            shutil.copy2(f, dst)
        else:
            from PIL import Image
            Image.open(f).convert("RGB").save(dst, quality=95)
    assert o == len(b)
    assert len(set(mapping.values())) == kept, "trung ten anh sau khi doi ten"
    (out / "sparse" / "0" / "images.bin").write_bytes(struct.pack("<Q", kept) + b"".join(res))
    print(f"{kept}/{n} anh -> {out}; vd {list(mapping.items())[:2]}")


if __name__ == "__main__":
    main()
