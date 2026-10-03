"""
Dung lai SfM gia da dung san (outputs/Replica/sfm/<scene>_gof/sparse/0) de tao dau vao cho buoc Split cua
Split&Splat, sap xep dung khuon thu muc ScanNet ma code goc doc cung (KHONG sua code goc):

  intrinsic/intrinsic_depth.txt <- cameras.bin cua SfM (PINHOLE fx, fy, cx, cy) dang 4x4
  pose/0000.txt                 <- images.bin cua SfM (world->camera, dao lai thanh camera->world nhu ScanNet)
  <scene>_vh_clean_2.ply        <- points3D.ply cua SfM (diem back-project tu depth, cung he toa do voi pose)
  images/0000.JPEG              <- Replica Test (ban rut gon 250 anh)
  depth/0000.png                <- Replica Test, doi don vi /6553.5 (m) * 1000 -> mm uint16
                                   (code goc doc depth = png / 1000.0; SfM khong co depth nen phai lay o day)

Frame j cua Replica Test = frame goc j*stride (stride 8) = anh "frame{j*stride:06d}.jpg" trong SfM.
Chay bang env safe-gs (co pycolmap).
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pycolmap
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="office0")
    ap.add_argument("--sfm", default=None, help="mac dinh outputs/Replica/sfm/<scene>_gof/sparse/0")
    ap.add_argument("--replica_test", default="/media/ml4u/Extreme SSD/Replica 8 Scene/Replica Test")
    ap.add_argument("--stride", type=int, default=8, help="Replica Test lay 1/stride frame goc")
    ap.add_argument("--out_data", required=True, help="thu muc data/ (se tao data/<scene>/...)")
    args = ap.parse_args()

    sfm = Path(args.sfm or f"/media/ml4u/Extreme SSD/Safe-GS/outputs/Replica/sfm/{args.scene}_gof/sparse/0")
    src = Path(args.replica_test) / args.scene
    dst = Path(args.out_data) / args.scene
    for sub in ["images", "depth", "pose", "intrinsic"]:
        (dst / sub).mkdir(parents=True, exist_ok=True)

    rec = pycolmap.Reconstruction(str(sfm))
    cam = next(iter(rec.cameras.values()))
    assert cam.model.name == "PINHOLE", cam.model.name
    fx, fy, cx, cy = cam.params
    K = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    for name in ["intrinsic_depth.txt", "intrinsic_color.txt"]:
        np.savetxt(dst / "intrinsic" / name, K, fmt="%.6f")

    by_name = {img.name: img for img in rec.images.values()}
    depth_scale = json.loads((Path(args.replica_test) / "cam_params.json").read_text())["camera"]["scale"]
    traj_test = np.loadtxt(src / "traj.txt").reshape(-1, 4, 4)
    n = traj_test.shape[0]
    max_err, max_depth_mm = 0.0, 0

    for j in range(n):
        img = by_name[f"frame{j * args.stride:06d}.jpg"]
        w2c = np.eye(4)
        w2c[:3, :4] = img.cam_from_world().matrix()
        c2w = np.linalg.inv(w2c)
        max_err = max(max_err, float(np.abs(c2w - traj_test[j]).max()))
        np.savetxt(dst / "pose" / f"{j:04d}.txt", c2w, fmt="%.8f")

        shutil.copy2(src / "results" / "image" / f"frame{j:06d}.jpg", dst / "images" / f"{j:04d}.JPEG")

        d = np.array(Image.open(src / "results" / "depth_image" / f"depth{j:06d}.png")).astype(np.float64)
        mm = np.clip(np.round(d / depth_scale * 1000.0), 0, 65535).astype(np.uint16)
        max_depth_mm = max(max_depth_mm, int(mm.max()))
        Image.fromarray(mm).save(dst / "depth" / f"{j:04d}.png")

    assert max_err < 1e-4, f"pose SfM lech traj.txt cua Replica Test: {max_err}"
    shutil.copy2(sfm / "points3D.ply", dst / f"{args.scene}_vh_clean_2.ply")
    print(f"[{args.scene}] {n} frame -> {dst}")
    print(f"  pose SfM vs traj.txt: lech lon nhat {max_err:.2e} | depth lon nhat {max_depth_mm} mm | "
          f"point cloud: {len(rec.points3D)} diem tu SfM")


if __name__ == "__main__":
    main()
