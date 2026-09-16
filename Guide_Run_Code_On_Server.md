# Hướng dẫn chạy — theo từng giai đoạn

## Hướng dẫn setup trước khi chạy

```bash
Bước 1: cd "/media/ml4u/Extreme SSD/Safe-GS"
Bước 2: "conda activate safe-gs" hoặc lệnh "conda activate /home/conda_envs/safe-gs"  (để kích hoạt môi trường ảo conda)
Bước 3: Muốn tải một thư viện nào đó thì "conda install <ten thu vien>"
```
## Giai đoạn 1: Khảo sát SfM (COLMAP) trên DTU scan24

```bash
cd "/media/ml4u/Extreme SSD/Safe-GS"
export DISPLAY=:0
/home/ml4u/conda_envs/safe-gs/bin/python scripts/01_run_perception.py --images_dir "../DTU Preprocess/DTU/scan24/images" --workspace_dir "outputs/DTU/sfm/dtu_scan24" --skip_seg
/home/ml4u/conda_envs/safe-gs/bin/python scripts/inspect_sfm.py --no_viz --export_ply "outputs/DTU/sfm/dtu_scan24/sparse_cloud.ply" --dtu_cameras_npz "../DTU Preprocess/DTU/scan24/cameras.npz"
/home/ml4u/conda_envs/safe-gs/bin/python scripts/inspect_sfm.py --color_by_density
/home/ml4u/conda_envs/safe-gs/bin/python scripts/inspect_sfm.py --view_ply "../DTU Dataset/Points/Points/stl/stl024_total.ply"
xdg-open "../DTU Preprocess/DTU/scan24/images/0000.png"
```

Giải thích từng dòng (theo đúng thứ tự trên):
1. Vào thư mục project.
2. Trỏ `DISPLAY` vào màn hình thật (bắt buộc, không có thì COLMAP/Open3D crash hoặc chạy chậm bằng CPU giả lập).
3. Chạy COLMAP (feature extraction → matching → mapping) trên 49 ảnh scan24, ra `outputs/DTU/sfm/dtu_scan24/`.
4. In số liệu (camera intrinsics, pose ảnh, thống kê point, visibility graph), xuất `sparse_cloud.ply`, và so pose với ground-truth DTU — tất cả trong 1 lệnh, không mở cửa sổ 3D.
5. Mở cửa sổ Open3D xem sparse cloud, tô màu theo mật độ (đỏ = dày, xanh dương = thưa/rỗng) để tìm vùng thiếu điểm.
6. Mở file ground-truth laser scan gốc của DTU (5.17 triệu điểm) để so bằng mắt với sparse cloud COLMAP.
7. Mở 1 ảnh gốc bất kỳ (đổi số `0000`) để đối chiếu vùng thưa/rỗng thấy trên point cloud với texture ảnh thật.

**Kết quả đã chạy trên scan24:** 49/49 ảnh đăng ký, 11,917 điểm 3D, reprojection error trung bình 0.51px, mỗi điểm trung bình 5.25 ảnh thấy, sai số vị trí camera so với DTU ~0.81 (đơn vị DTU, dưới 1mm), sai số góc xoay 0.09°.

**Muốn chạy trên Replica office0 (2000 ảnh) thay vì DTU scan24** — đổi các đường dẫn tương ứng:
```bash
/home/ml4u/conda_envs/safe-gs/bin/python scripts/01_run_perception.py --images_dir "../Replica 8 Scene/Replica/office0/results/image" --workspace_dir "outputs/Replica/sfm/office0" --skip_seg
/home/ml4u/conda_envs/safe-gs/bin/python scripts/inspect_sfm.py --no_viz --sparse_dir "outputs/Replica/sfm/office0/sparse/0" --visibility_json "outputs/Replica/sfm/office0/visibility_graph.json"
```

---

## Giai đoạn 2: Train Scene-GS (3D Gaussian Splatting toàn cảnh) trên scan24

```bash
cd "/media/ml4u/Extreme SSD/Safe-GS"
export PYTHONNOUSERSITE=1
export DISPLAY=:0
/home/ml4u/conda_envs/safe-gs/bin/python scripts/02_run_scene_gs.py
```

Giải thích:
1-3. Vào project, cô lập env khỏi `~/.local` dùng chung, trỏ màn hình thật (bắt buộc cho CUDA rasterizer).
4. Chạy train Scene-GS 30,000 iterations trên sparse cloud + ảnh scan24 đã có sẵn (config đã trỏ sẵn trong `configs/base_scene.toml`). Tự động lưu checkpoint tại:
   - `outputs/DTU/3dgs/dtu_scan24/scene_gs_7000.ply`
   - `outputs/DTU/3dgs/dtu_scan24/scene_gs_30000.ply`
   - `outputs/DTU/3dgs/dtu_scan24/scene_gs.ply` (bản cuối cùng, dùng cho Module 3)

Lưu ý: đã cài `diff-gaussian-rasterization` + `simple-knn` (CUDA rasterizer thật) vào env `safe-gs` — nếu env bị tạo lại từ đầu thì cần cài lại 2 gói này:
```bash
export PYTHONNOUSERSITE=1
export CUDA_HOME=/usr/local/cuda-12
export TORCH_CUDA_ARCH_LIST="8.9"
/home/ml4u/conda_envs/safe-gs/bin/pip install "git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git" --no-build-isolation
/home/ml4u/conda_envs/safe-gs/bin/pip install "git+https://gitlab.inria.fr/bkerbl/simple-knn.git" --no-build-isolation
```

Muốn theo dõi tiến trình (loss, số Gaussian) khi train đang chạy: thanh progress bar tự in trực tiếp ra terminal, không cần lệnh gì thêm.

**Train thêm bản SH degree = 0** (tắt hẳn Spherical Harmonics bậc cao, chỉ giữ màu cơ bản — không có hiệu ứng đổi màu theo góc nhìn), lưu checkpoint 7k/30k với hậu tố `_sh0`:
```bash
export PYTHONNOUSERSITE=1
export DISPLAY=:0
/home/ml4u/conda_envs/safe-gs/bin/python scripts/02_run_scene_gs.py --sh_degree 0 --checkpoint_tag "_sh0"
```
→ Ra 3 file: `scene_gs_7000_sh0.ply`, `scene_gs_30000_sh0.ply`, `scene_gs_sh0.ply` (cùng thư mục 3dgs, không đè lên bản SH3 đã train trước đó).

Muốn train biến thể khác (vd sh_degree=1, hay tên tag khác) chỉ cần đổi 2 tham số này, không cần sửa code/config gì thêm.

**Xem thử checkpoint Gaussian vừa train bằng Open3D** (giống hệt cách xem sparse cloud/ground-truth trước đó, chỉ đổi tên file):
```bash
export DISPLAY=:0
/home/ml4u/conda_envs/safe-gs/bin/python scripts/inspect_sfm.py --view_ply "outputs/DTU/3dgs/dtu_scan24/scene_gs_7000.ply"
/home/ml4u/conda_envs/safe-gs/bin/python scripts/inspect_sfm.py --view_ply "outputs/DTU/3dgs/dtu_scan24/scene_gs_30000.ply"
```
(script tự nhận diện đây là file checkpoint Gaussian và tự chuyển màu đúng — chỉ cần đổi tên file trong `--view_ply` để xem checkpoint khác)

**Quan sát chất lượng tốt hơn — render ảnh/video thật thay vì chấm điểm** (Open3D chỉ hiện tâm điểm, không thấy được hình dạng/màu/độ mờ thật của Gaussian — cách này render bằng đúng CUDA rasterizer nên chất lượng như ảnh thật):

Render vài góc camera đã train, so sánh với ảnh gốc:
```bash
export PYTHONNOUSERSITE=1
export DISPLAY=:0
/home/ml4u/conda_envs/safe-gs/bin/python scripts/render_gs.py --checkpoint "outputs/DTU/3dgs/dtu_scan24/scene_gs_30000.ply" --output_dir "outputs/DTU/3dgs/dtu_scan24/renders_30000" --save_compare
```
(mở ảnh trong `outputs/DTU/3dgs/dtu_scan24/renders_30000/` bằng `xdg-open <file>.png` — mỗi ảnh là [render | ảnh gốc] ghép cạnh nhau)

Render video xoay 360° quanh vật thể (trực quan nhất, dễ quan sát nhất):
```bash
/home/ml4u/conda_envs/safe-gs/bin/python scripts/render_gs.py --checkpoint "outputs/DTU/3dgs/dtu_scan24/scene_gs_30000.ply" --output_dir "outputs/DTU/3dgs/dtu_scan24/orbit_30000" --orbit --num_frames 90
xdg-open "/media/ml4u/Extreme SSD/Safe-GS/outputs/DTU/3dgs/dtu_scan24/orbit_30000/orbit.mp4"
```
Muốn xem checkpoint khác (7000 thay vì 30000) chỉ cần đổi `--checkpoint` và `--output_dir` trong 2 lệnh trên, mọi thứ khác giữ nguyên.

Lưu ý: `xdg-open` với file video **phải dùng đường dẫn tuyệt đối** (bắt đầu bằng `/media/...`), dùng đường dẫn tương đối VLC sẽ mở lên nhưng không load được file (chỉ hiện logo hình nón mặc định, tưởng video bị đen/lỗi nhưng thực ra video vẫn bình thường).

**Soi kỹ 1 hạt Gaussian duy nhất** (phóng to, xoay quanh nó, xem hình dạng + màu đổi thế nào theo góc nhìn):
```bash
/home/ml4u/conda_envs/safe-gs/bin/python scripts/render_gs.py --checkpoint "outputs/DTU/3dgs/dtu_scan24/scene_gs_30000.ply" --output_dir "outputs/DTU/3dgs/dtu_scan24/isolate_gaussian" --isolate_gaussian --num_frames 60
xdg-open "/media/ml4u/Extreme SSD/Safe-GS/outputs/DTU/3dgs/dtu_scan24/isolate_gaussian/isolate_gaussian.mp4"
```
- Không cần chỉ định hạt nào — script tự chọn 1 hạt to + rõ (opacity cao) làm ví dụ, phóng to gấp 15 lần để dễ nhìn.
- Muốn soi đúng 1 hạt cụ thể: thêm `--gaussian_idx <số>` (số này là chỉ số hạt trong file .ply, 0 đến tổng số hạt trừ 1).
- Muốn phóng to/nhỏ hơn: đổi `--scale_boost` (mặc định 15).
- **Về màu đổi theo góc nhìn**: 3DGS mã màu bằng Spherical Harmonics (SH) — nếu hạt đó nằm ở vùng bề mặt bóng/phản chiếu ánh sáng, xoay quanh sẽ thấy màu đổi rõ; nếu nằm ở vùng mờ/matte thì màu gần như không đổi (đúng bản chất vật lý, không phải lỗi). Checkpoint càng train nhiều iteration (30000 hơn 7000) thì hiệu ứng này càng rõ vì SH degree tăng dần theo quá trình train.

---

## Giai đoạn 3: Chạy full toàn bộ DTU dataset (15 scan) + Replica 8 Scene (8 scene)

```bash
cd "/media/ml4u/Extreme SSD/Safe-GS"
bash scripts/run_all_scenes.sh
```

Script tự động, tuần tự, cho từng scene: chạy COLMAP (nếu chưa có) → train Scene-GS 30,000 iterations
(checkpoint 7k/30k) → lưu vào `outputs/<DTU|Replica>/3dgs/<tên_scene>/`. Danh sách scene:
- DTU: `dtu_scan24, dtu_scan37, dtu_scan40, dtu_scan55, dtu_scan63, dtu_scan65, dtu_scan69, dtu_scan83, dtu_scan97, dtu_scan105, dtu_scan106, dtu_scan110, dtu_scan114, dtu_scan118, dtu_scan122`
- Replica: `replica_office0 ... replica_office4, replica_room0 ... replica_room2`

**Thời gian**: DTU (49-64 ảnh/scan) nhanh, Replica (2000 ảnh/scene) chậm hơn nhiều — tổng 23 scene có
thể mất vài giờ. Dừng giữa chừng bằng `Ctrl+C`, chạy lại đúng lệnh trên để tiếp tục — scene nào đã có
`scene_gs_30000.ply` sẽ tự bỏ qua (không train lại từ đầu, không chạy lại COLMAP nếu đã có sẵn `sfm/sparse/0`).

Xem/render checkpoint của bất kỳ scene nào sau khi xong, dùng lại đúng các lệnh ở Giai đoạn 2, chỉ đổi
đường dẫn, ví dụ với `office1`:
```bash
/home/ml4u/conda_envs/safe-gs/bin/python scripts/render_gs.py --checkpoint "outputs/Replica/3dgs/office1/scene_gs_30000.ply" --sfm_dir "outputs/Replica/sfm/office1/sparse/0" --images_dir "../Replica 8 Scene/Replica/office1/results/image" --output_dir "outputs/Replica/3dgs/office1/orbit_30000" --orbit --num_frames 90
```


*(Các giai đoạn tiếp theo — Module 3 Object-GS, v.v. — sẽ bổ sung vào đây khi làm tới.)*
