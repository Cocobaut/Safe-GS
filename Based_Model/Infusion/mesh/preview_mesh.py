"""Ảnh xem trước nhanh của mesh (splat màu đỉnh, nhìn từ trên xuống theo hệ trục mặt bàn) – không cần OpenGL."""
import json
import os
import sys

import numpy as np
import trimesh
from PIL import Image

mesh_path, crop_json, out_png = sys.argv[1:4]
fr = json.loads(open(crop_json).read())
n, d, e1, e2 = (np.array(fr[k]) for k in ["plane_normal", "plane_d", "basis_e1", "basis_e2"])
m = trimesh.load(mesh_path, process=False)
P, C = np.asarray(m.vertices), np.asarray(m.visual.vertex_colors)[:, :3]
u, v, h = P @ e1, P @ e2, P @ n + d
W = 1000
s = (W - 1) / max(np.ptp(u), np.ptp(v))
x, y = ((u - u.min()) * s).astype(int), ((v - v.min()) * s).astype(int)
H = y.max() + 1
img = np.full((H, W, 3), 255, np.uint8)
order = np.argsort(h)  # điểm cao vẽ sau (nhìn từ trên xuống)
img[y[order], x[order]] = C[order]
os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
Image.fromarray(img[::-1]).save(out_png)
print(f"[✓] {out_png} {img.shape}")
