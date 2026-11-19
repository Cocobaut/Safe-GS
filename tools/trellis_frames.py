"""
Đổi kết quả TRELLIS (occ_final: vỏ 64^3 trong hệ bàn chuẩn hoá) thành khung ĐẶC theo định dạng frame/<vật>.npz của tools/frame_mesh.py,
để dựng mesh kín + thử MuJoCo bằng đúng công cụ cũ. Cùng một quy tắc cho mọi vật.

  đặc = lấp lỗ (binary_fill_holes) sau khi đóng hình thái 1 ô của vỏ TRELLIS; hợp với khung quan sát (phần đã thấy luôn được giữ).
Xuất: <out>/<vật>.npz (xyz = tâm ô theo hệ cảnh COLMAP, cls = 1, voxel = cạnh ô theo đơn vị cảnh), summary.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="trellis_in (grid.npz)")
    ap.add_argument("--run", required=True, help="trellis_out/<tag> (<vật>/occ.npz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--union_frame", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    summ = {}
    for od in sorted(Path(a.run).iterdir()):
        f = od / "occ.npz"
        gf = Path(a.inp) / od.name / "grid.npz"
        if not f.exists() or not gf.exists():
            continue
        occ = np.load(f)["occ_final"]
        g = np.load(gf)
        res = occ.shape[0]
        sh = np.pad(occ, 2)
        solid = binary_fill_holes(binary_closing(sh, iterations=1))[2:-2, 2:-2, 2:-2]
        filled_by_holes = int((solid & ~occ).sum())
        if a.union_frame:
            solid = solid | g["frame"]
        idx = np.argwhere(solid)
        center, size, R, p0 = g["center"], float(g["size"]), g["R"], g["p0"]
        Pt = ((idx + 0.5) / res - 0.5) * size + center
        Pw = Pt @ R + p0
        np.savez_compressed(out / f"{od.name}.npz", xyz=Pw.astype(np.float32), cls=np.ones(len(Pw), np.uint8), voxel=size / res)
        summ[od.name] = {"shell": int(occ.sum()), "solid": int(solid.sum()), "interior_filled": filled_by_holes,
                         "frame": int(g["frame"].sum()), "added_vs_frame": int((solid & ~g["frame"]).sum()),
                         "cell_cm": size / res * float(g["mpu"]) * 100}
        print(od.name, summ[od.name])
    (out / "summary.json").write_text(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
