"""
Lọc Gaussian của CoMe theo opacity, ghi ra file .ply mới (giữ nguyên mọi thuộc tính, CoMe load_ply đọc được).

Opacity trong .ply lưu dạng logit (GaussianModel.save_ply). Mặc định lọc theo opacity HIỆU DỤNG lúc render, giống
GaussianModel.get_opacity_with_3D_filter:
    opacity = sigmoid(o) * sqrt( prod(s^2) / prod(s^2 + filter_3D^2) ),   s = exp(scale_i)
--raw: lọc theo sigmoid(o), không tính 3D filter.

Ví dụ:
    python tools/filter_gaussians_opacity.py                      # giữ opacity >= 0.5
    python tools/filter_gaussians_opacity.py --min 0.1 --raw
"""
import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_IN = REPO / "Data" / "teatime" / "come" / "point_cloud" / "iteration_30000" / "point_cloud.ply"


def read_ply(path: Path):
    with open(path, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: thiếu end_header")
            header += line
        lines = header.decode("ascii").splitlines()
        assert "format binary_little_endian 1.0" in lines, "chỉ hỗ trợ binary_little_endian"
        n = int(next(l for l in lines if l.startswith("element vertex")).split()[-1])
        props = [l.split()[-1] for l in lines if l.startswith("property")]
        assert all(l.split()[1] == "float" for l in lines if l.startswith("property")), "thuộc tính không phải float"
        data = np.fromfile(f, dtype=np.dtype([(p, "<f4") for p in props]), count=n)
    assert len(data) == n, f"đọc được {len(data)}/{n} điểm"
    return data


def write_ply(path: Path, data: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(data)}"]
    header += [f"property float {p}" for p in data.dtype.names] + ["end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        data.tofile(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inp", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--min", type=float, default=0.5, help="giữ Gaussian có opacity >= giá trị này")
    ap.add_argument("--raw", action="store_true", help="dùng sigmoid(opacity), bỏ qua 3D filter")
    args = ap.parse_args()

    d = read_ply(args.inp)
    op = 1.0 / (1.0 + np.exp(-d["opacity"].astype(np.float64)))
    if not args.raw:
        s2 = np.stack([np.exp(d[f"scale_{i}"].astype(np.float64)) ** 2 for i in range(3)], 1)
        f2 = d["filter_3D"].astype(np.float64)[:, None] ** 2
        op = op * np.sqrt(s2.prod(1) / (s2 + f2).prod(1))
    keep = op >= args.min
    kind = "sigmoid" if args.raw else "hieu_dung"
    out = args.out or args.inp.parent.parent.parent / "filtered" / f"point_cloud_opacity_{kind}_ge_{args.min}.ply"
    write_ply(out, d[keep])

    print(f"vào : {args.inp}  ({len(d):,} Gaussian)")
    print(f"opacity ({'sigmoid' if args.raw else 'hiệu dụng, có 3D filter'}): "
          f"p10 {np.percentile(op, 10):.3f} | p50 {np.median(op):.3f} | p90 {np.percentile(op, 90):.3f}")
    for t in (0.1, 0.3, 0.5, 0.7, 0.9):
        print(f"  >= {t}: {(op >= t).sum():>9,} ({(op >= t).mean() * 100:5.1f}%)")
    print(f"ra  : {out}  ({keep.sum():,} Gaussian giữ lại, bỏ {(~keep).sum():,}, {out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
