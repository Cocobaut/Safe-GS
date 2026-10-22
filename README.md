## 📂 Cấu trúc dự án (Project Structure)

```text
Project_Safe_GS/
│
├── configs/                            # Quản lý cấu hình huấn luyện và mô phỏng
│   ├── base_scene.toml                 # Config huấn luyện Scene-GS (20k iters, camera settings)
│   ├── object_roi.toml                 # Config huấn luyện Object-GS (AABB, densification threshold)
│   ├── gof_meshing.toml                # Config GOF (Tetrahedral grid, level set, binary search steps)
│   └── simulation.toml                 # Config Physics (MuJoCo/PhysX/PyBullet parameters, timestep)
│
├── data/                               # Dữ liệu & kết quả trung gian (chuẩn hóa chữ thường)
│   ├── raw/                            # Dữ liệu ảnh thô (RGB images, camera captures)
│   ├── sfm/                            # Kết quả Colmap/SfM (cameras.bin, images.bin, points3D.bin)
│   ├── segmentation/                   # 2D Masks trích xuất từ SAM / GroundingDINO
│   └── workspace/                      # Cache trung gian qua từng module
│       ├── roi_boxes/                  # Tọa độ 3D AABB / Voxel grid metadata (JSON/YAML)
│       ├── checkpoints/                # Checkpoints: scene_gs.ply, object_gs.ply, composed_3dgs.ply
│       └── meshes/                     # Mesh xuất ra: raw_gof.obj, watertight.obj, decimated.obj
│
├── submodules/                         # Custom CUDA Kernels hoặc C++ extensions
│   ├── diff_gaussian_rasterization/    # Rasterizer gốc hoặc tùy biến
│   └── gof_cuda/                       # Custom Ray-Gaussian intersection & Opacity evaluation CUDA
│
├── src/                                # Mã nguồn chính chuẩn hóa khớp 100% với 6 Modules
│   ├── __init__.py
│   │
│   ├── module1_perception/             # Module 1: SfM & 2D Segmentation
│   │   ├── sfm_pipeline.py             # Tự động hóa COLMAP / trích xuất camera poses, intrinsics
│   │   └── segmenter.py                # SAM2 / LangSAM trích xuất 2D Mask
│   │
│   ├── module2_scene_and_roi/          # Module 2: Nhánh song song Scene-GS & 3D ROI
│   │   ├── scene_branch/               # Nhánh 1: Scene-GS
│   │   │   ├── scene_trainer.py        # Huấn luyện Scene-GS (20K iters theo chiến lược ROI-GS)
│   │   │   └── camera_scheduler.py     # Phân loại góc nhìn toàn cảnh vs góc nhìn cận cảnh
│   │   └── roi_branch/                 # Nhánh 2: 3D ROI & Space Partitioning
│   │       ├── mask_to_3d.py           # Chiếu ngược Mask 2D + Mây điểm SfM lên không gian 3D
│   │       ├── aabb_generator.py       # Tính toán 3D Bounding Box (AABB/OBB) và Voxel Grid
│   │       └── view_selection.py       # Lọc danh sách Camera tối ưu cho vùng ROI
│   │
│   ├── module3_object_composition/     # Module 3: Object-GS & 3DGS Hoàn chỉnh
│   │   ├── object_trainer.py           # Huấn luyện Object-GS khởi tạo từ Scene-GS, densify trong ROI
│   │   └── gs_composer.py              # Hoán đổi hạt trong AABB -> sinh Composed-3DGS
│   │
│   ├── module4_gof_meshing/            # Module 4: Trích xuất Mesh bằng GOF (Tách riêng biệt)
│   │   ├── opacity_field.py            # Ray-Gaussian intersection & Min Opacity field
│   │   ├── regularization.py           # Depth Distortion Loss & Normal Consistency Loss
│   │   ├── marching_tetrahedra.py      # Marching Tetrahedra từ Delaunay tetrahedral grid
│   │   └── level_set_solver.py         # Binary Search 8 bước tìm đẳng diện 0.5
│   │
|   +-- Module4-Physic_Engine/          # Module 4: Physics engine integration
|       +-- preprocessing/             # Mesh cleaning, topology, adaptive QEM, collision assets
|       |   +-- Input/<dataset>/
|       |   +-- Output/<dataset>/
|       +-- simulation/                # MuJoCo scenes, tasks, topology guard, benchmarks
|       |   +-- Output/<dataset>/
|       +-- docs/
│
├── scripts/                            # Entry points trực quan theo tiến trình
│   ├── 01_run_perception.py            # Chạy Module 1 (SfM + Segment)
│   ├── 02_run_scene_gs.py              # Chạy Module 2 - Nhánh Scene-GS
│   ├── 02_run_roi_3d.py                # Chạy Module 2 - Nhánh ROI 3D
│   ├── 03_run_object_and_compose.py    # Chạy Module 3 (Object-GS -> Composed 3DGS)
│   ├── 04_run_gof_meshing.py           # Chạy Module 4 (Trích xuất Mesh thô từ Composed-3DGS)
│   ├── 05_run_postprocessing.py        # Chạy Module 4 preprocessing (Decimation + Watertight Mesh)
│   └── 06_run_simulation.py            # Chạy Module 4 simulation (Nạp Mesh vào Simulator kiểm thử)
│
├── tests/                              # Unit tests kiểm tra từng module
│   ├── test_roi_projection.py
│   ├── test_gof_extraction.py
│   ├── test_watertight.py
│   └── test_topology_guard.py
│
├── requirements.txt
└── README.md
```


Physics preprocessing and MuJoCo simulation now live inside `src/Module4-Physic_Engine/`. Keep using scripts 05/06; dataset paths are documented there.
