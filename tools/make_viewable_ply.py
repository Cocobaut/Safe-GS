"""
Chuyển Gaussian của Split&Splat (Stage 2) sang file .ply 3DGS chuẩn để xem bằng trình xem web (SuperSplat, ...).

- Bỏ 384 thuộc tính `desc_*` (vector ngữ nghĩa) và nx/ny/nz -> chỉ giữ các thuộc tính chuẩn của 3DGS.
- Bỏ Gaussian bị nhuộm đen (f_dc < -1.75, ngưỡng filter_points của tác giả) và Gaussian gần trong suốt
  (sigmoid(opacity) < --min-opacity): phần sót lại từ lúc khởi tạo bằng point cloud cả cảnh.
- Ghi từng vật ra <out>/objects/<id>.ply và gộp mọi vật thành <out>/all_objects.ply.

Ví dụ:
    python tools/make_viewable_ply.py Data/figurines/split_splat_stage2/raw Data/figurines/viewable_ply
"""
import argparse
from pathlib import Path

import numpy as np

STD_PREFIX = ("x", "y", "z", "f_dc_", "f_rest_", "opacity", "scale_", "rot_")


def read_ply(path: Path):
    with open(path, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: thiếu end_header")
            header += line
        lines = header.decode("ascii").splitlines()
        assert "format binary_little_endian 1.0" in lines, f"{path}: chỉ hỗ trợ binary_little_endian"
        n = int(next(l for l in lines if l.startswith("element vertex")).split()[-1])
        props = [l.split()[-1] for l in lines if l.startswith("property")]
        assert all(l.split()[1] == "float" for l in lines if l.startswith("property")), f"{path}: thuộc tính không phải float"
        data = np.fromfile(f, dtype=np.dtype([(p, "<f4") for p in props]), count=n)
    return data


def write_ply(path: Path, data: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(data)}"]
    header += [f"property float {name}" for name in data.dtype.names] + ["end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        data.tofile(f)


def clean(data: np.ndarray, min_opacity: float) -> np.ndarray:
    keep = [p for p in data.dtype.names if p.startswith(STD_PREFIX)]
    out = np.empty(len(data), dtype=np.dtype([(p, "<f4") for p in keep]))
    for p in keep:
        out[p] = data[p]
    f_dc = np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], axis=1)
    not_black = ~np.all(f_dc < -1.75, axis=1)
    visible = 1.0 / (1.0 + np.exp(-data["opacity"])) >= min_opacity
    return out[not_black & visible]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir", type=Path, help="thư mục raw/ chứa <id>/point_cloud/iteration_<it>/point_cloud.ply")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--iteration", type=int, default=1000)
    ap.add_argument("--min-opacity", type=float, default=0.1)
    args = ap.parse_args()

    plys = sorted(args.raw_dir.glob(f"*/point_cloud/iteration_{args.iteration}/point_cloud.ply"),
                  key=lambda p: int(p.parts[-4]) if p.parts[-4].isdigit() else 10**9)
    assert plys, f"không thấy point_cloud.ply nào trong {args.raw_dir}"
    parts = []
    for p in plys:
        obj_id = p.parts[-4]
        src = read_ply(p)
        dst = clean(src, args.min_opacity)
        write_ply(args.out_dir / "objects" / f"{obj_id}.ply", dst)
        parts.append(dst)
        print(f"vật {obj_id:>4}: {len(src):>8} -> {len(dst):>7} Gaussian ({dst.nbytes / 1e6:.1f} MB)")
    merged = np.concatenate(parts)
    write_ply(args.out_dir / "all_objects.ply", merged)
    print(f"\n{len(plys)} vật -> {args.out_dir / 'all_objects.ply'}: {len(merged)} Gaussian ({merged.nbytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
