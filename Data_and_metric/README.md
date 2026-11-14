# Data

Dữ liệu cục bộ của dự án, sắp xếp theo scene. Danh sách dataset tham khảo nằm ở `DATASETS.md`.

```
Data/
├── teatime/                  INPUT CHÍNH (team đã chốt, 2026-10-15)
│   ├── images/               180 ảnh RGB frame_00001.jpg .. frame_00180.jpg, 994x738
│   └── sam2_masks/
│       ├── labels/           180 PNG grayscale uint8, mỗi pixel = ID vật (0 = không thuộc vật nào),
│       │                     ID nhất quán giữa các frame (SAM2 video propagation)
│       ├── vis/              180 ảnh overlay để xem mask bằng mắt
│       ├── objects.json      106 vật: frame seed, số frame thấy vật, diện tích, frame rõ nhất
│       │                     (bản gốc: Modal tài khoản quangkhaisaigonhcm, Volume safe-gs-outputs:
│       │                      lerf_masks/teatime/, trùng hash; code sinh ra chưa có trong repo)
│       ├── sam2_config.json  cấu hình SAM2 (sam2.1_hiera_large, 6 keyframe)
│       └── overview.jpg
│   └── colmap/               COLMAP (modal_come.py::colmap_teatime), dùng làm `-s` cho CoMe
│       ├── images/           177 ảnh đã undistort (PINHOLE, vẫn 994x738); thiếu frame_00148, 00155, 00156
│       ├── sparse/0/         cameras.bin, images.bin, points3D.bin (24685 điểm)
│       ├── input/            180 ảnh gốc
│       ├── database.db       database COLMAP
│       └── colmap_check.json kết quả kiểm tra
│   └── come/                 CoMe đã train 30k iter (modal_come.py::train_teatime)
│       ├── point_cloud/iteration_30000/point_cloud.ply        1.80M Gaussian (440 MB), có confidence + filter_3D
│       ├── point_cloud/iteration_30000/appearance_embedding.pth
│       ├── cfg_args, config.json, mesh_args, rem_args, cameras.json   cấu hình lúc train (cần cho render/mesh)
│       ├── tensorboard/
│       ├── renders/{train,test}/ ảnh render + ảnh ghép so với ảnh gốc
│   ├── pipeline/v1/          kết quả pipeline end-to-end S0..S6 (modal_come pipeline, xem pipeline/README.md)
│       ├── edit/delete_all_object/ xoá mọi ID mask (label > 0), cùng cấu trúc với remove_id22
│       ├── edit/delete_all_object_v2_sdf/ + dọn bên trong SDF và floater < 0.5 quanh vật (có diag.npz trên Volume)
│       ├── edit/delete_all_object_v3_freevote/ + vote bỏ qua che trong bán kính 0.2 quanh vật (bản tốt nhất hiện tại)
│       ├── edit/remove_id22/ cảnh đã xoá cốc (scene_without.ply), phần cốc (object.ply), preview trước|sau, stats.json
│       ├── filtered/         Gaussian đã lọc theo opacity (tools/filter_gaussians_opacity.py), cùng định dạng CoMe
│       ├── tracking/         video tracking ID mask trên ảnh render (tools/make_tracking_video.py)
│       ├── sdf_vis/          mặt SDF = 0 (opacity 0.5) nhìn từ 177 góc camera (tools/come_sdf_vis.py):
│       │                     contour/ (viền vàng tô đậm), normal/, sdf_contour.mp4, sdf_normal.mp4
│       └── renders_all/      177 ảnh render frame_XXXXX.png, đúng góc camera và kích thước 994x738 của ảnh gốc
│                             (thiếu frame_00148, 00155, 00156 vì COLMAP không đăng ký được)
│                             (checkpoint chkpnt29000.pth 1.3 GB chỉ để trên Volume safe-gs-come:/teatime/come)
│
├── figurines/                Kết quả thử nghiệm cũ trên LERF figurines (Split&Splat)
│   ├── sam2_masks/           mask SAM2 auto_seg từng frame (Split&Splat stage 1)
│   ├── propagation/          lan truyền mask (stage 1 bước 2-4)
│   ├── prep/                 figurines_prep.zip: COLMAP + Murre depth + 3DGS dày (modal_prep.py)
│   ├── depth_ckpt/           checkpoint depth Murre + colmap.zip của lần chạy trước
│   ├── split_splat_stage2/   Gaussian từng instance (raw/<id>/point_cloud/...), logs, progress_2a.csv  (~44 GB)
│   └── viewable_ply/         Gaussian từng vật đã lọc để xem (tools/make_viewable_ply.py)
│
├── weights/
│   └── Murre-ckpt.zip        checkpoint Murre (depth)
│
└── _archive/                 Không dùng nữa, giữ lại để tham khảo / khôi phục
    ├── teatime_sam2_masks_old/   bộ mask teatime cũ (auto_seg từng frame, ID không nhất quán giữa frame),
    │                             đã thay bằng teatime/sam2_masks
    └── zips/                 các file zip gốc đã giải nén (có thể xoá nếu cần chỗ trống)
        ├── teatime_images.zip           = teatime/images
        ├── teatime_sam2_masks_v2.zip    = teatime/sam2_masks
        ├── teatime_sam2_masks.zip       = _archive/teatime_sam2_masks_old
        ├── figurines_sam2_masks.zip     = figurines/sam2_masks
        ├── figurines_sam2_masks__modal_inputs_copy.zip   bản copy tải từ Volume Modal
        └── ckpt-20261008T085313Z-1-001.zip              = figurines/depth_ckpt
```

Quy ước khi thêm dữ liệu mới:
- Mỗi scene một thư mục `Data/<scene>/`, bên trong chia theo bước: `images/`, `sam2_masks/`, `colmap/`, `come/`, ...
- Kết quả tải từ Modal về thì đặt vào đúng `Data/<scene>/<bước>/`, không để thư mục `modal_<tài khoản>/` ở gốc.
- Zip gốc sau khi giải nén thì chuyển vào `_archive/zips/`.
