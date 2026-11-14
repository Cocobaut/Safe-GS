# CLAUDE.md — Hướng dẫn & Quy tắc Tối ưu Token cho Dự án `Safe-GS`

Dự án `F:\Safe-GS` nghiên cứu và phát triển pipeline **3D Gaussian Splatting (3DGS)**, **Object Removal / Inpainting 3D (InFusion, CoMe, Inpaint360GS, SplatFill, OMEGAS)**, **Trích xuất Mesh bề mặt (GOF, MILo, InstantMesh)** và **Mô phỏng Vật lý (MuJoCo)**.

---

## ⚡ 1. QUY TẮC TIẾT KIỆM TOKEN (BẮT BUỘC TUÂN THỦ)

1. **Ưu tiên dùng công cụ MCP `serena` cho code Python:**
   - Khi cần tìm hiểu cấu trúc file `.py` lớn (như `gaussian_model.py`, `train.py`, `compose.py`, `modal_*.py`), hãy dùng `get_symbols_overview` hoặc `find_symbol` của **Serena** để đọc đúng hàm/class cần thiết thay vì dùng `Read` đọc toàn bộ file.
   - Khi sửa hàm/class Python, ưu tiên dùng `replace_symbol_body` hoặc `Edit` phạm vi hẹp.
   - Đối với file kế hoạch `.md`, cấu hình `.toml`/`.yaml`/`.json`, script `.sh` và kernel CUDA/C++ (`.cu`, `.cpp`, `.h`), sử dụng công cụ `Read` / `Edit` mặc định của Claude Code.

2. **CẤM quét (`Grep` / `Glob` / `find`) không giới hạn ở thư mục gốc `F:\Safe-GS`:**
   - Dự án chứa **>11,500 files** do có 8 repo gốc trong `Based_Model/` và nhiều bản sao thử nghiệm trong `test_model/`.
   - **Luôn chỉ định rõ đường dẫn thư mục con (`path`)** khi gọi `Grep` hoặc `Glob` (ví dụ: chỉ tìm trong `src/`, hoặc `Based_Model/CoMe`, hoặc `test_model/CoMe_Infusion/src`).

3. **CẤM đọc các file dữ liệu nặng / lockfile:**
   - Không bao giờ đọc: `uv.lock` (1.15 MB), các file `.ply`, `.pth`, `.pt`, `.ckpt`, `.npy`, `.npz`, `.so`, `.pyd`, `.db`, `.mp4`, hoặc các file `.log` nguyên bản dài hàng nghìn dòng (nếu cần kiểm tra log, chỉ đọc 50–100 dòng cuối bằng `tail` hoặc `Read` với `offset`/`limit`).

---

## 🗺️ 2. BẢN ĐỒ KIẾN TRÚC THƯ MỤC (`F:\Safe-GS`)

### A. Base Framework (CoMe + Inpaint360GS + InstantMesh) — tại thư mục gốc
> Kế hoạch & trạng thái: `PLAN_REBUILD_BASE_FRAMEWORK.md`. Hướng dẫn chạy: `README.md`. Pipeline cũ (GOF/MILo, 6 module) nằm ở `legacy/` (`src_v0/`, `scripts_v0/`, `README_v0.md`) — chỉ đọc khi người dùng yêu cầu.
- `src/` (chạy với `cwd=src`, import kiểu `from scene import ...`): Inpaint360GS + rasterizer CoMe cho Step 4–12.
  - `src/common/`: `paths.py` (PROJECT_ROOT, DATA_DIR, OUTPUTS_DIR, `step_dir`, `get_run_dir`), `datasets.py` (`resolve_run` đọc `configs/datasets/*.yaml`).
  - `src/come/`: CoMe (`r4dl/CoMe @ caf62f6`, đã vá resume checkpoint trong `train.py`) — Step 1, 2, 13, chạy với `cwd=src/come` (có `scene/`, `arguments/` TRÙNG TÊN với `src/` → không đưa cả hai vào `sys.path`). `src/come/tools/`: `convert_colmap.py`, `come_apply_appearance.py`, `come_sdf_vis.py`, `come_remove_object.py`.
  - `src/tools_come/` (S4 `render_check.py`, S11 `select_init_view.py`), `src/tools_safe/` (S9 `step9_samtrack_click.py`, S12 `step12_package.py`), `src/tools_mesh/` (`mesh_io.py`, S13b `crop_table.py`, `table_plane.json`), `src/tools_instantmesh/` (S14.1 `extract_object.py`, S14.2–14.5 `step14_gpu.py`, S15 `place_on_table.py`).
  - Code bên thứ ba, KHÔNG đọc/quét trừ khi cần: `src/seg/detectron2/`, `src/LaMa/`, `src/Segment-and-Track-Anything/`, `src/instantmesh/`, `src/gaussian_splatting/`, `src/**/submodules/`.
- `scripts/`: `run_pipeline.py` (entry-point, `--check-env`, `--list`, `--dry-run`, `--modal`), `modal_come_train.py` (S0–2, Volume `safe-gs-come`), `modal_come_pipeline.py` + `pipeline_steps.py` (S4–12, bảng lệnh, Volume `inpaint360gs`), `modal_come_mesh.py` (S13), `modal_instantmesh.py` (S14, Volume `instantmesh`), `modal_common.py` (helper + image CoMe), `evaluate_all_metrics.py` (Phase 5 → `outputs/council_report_summary.{md,csv}`), `lerf_select_object.py`, `lerf_masks_3d_consistent.py`.
- `configs/`: `paths.yaml`, `datasets/{lerf,replica,inpaint360gs}.yaml` (target_id, resolution, samtrack_click ...); các `.toml` cũ của pipeline GOF.
- `outputs/` (gitignored): `outputs/<LERF|Replica|Inpaint360GS>/ours/<scene>_<object>/step0..15/`, `evaluation_summary.json`; bản mẫu `outputs/LERF/ours/teatime_napkin` (7.9 GB, KHÔNG đọc file nặng). Cũ: `outputs/_legacy/`.
  - Baseline đặt cạnh `ours/`: `outputs/<D>/{CoMe,MILo,Inpaint360GS,NeRFiller}/`. CoMe gốc (mesh cả scene, mỗi scene 1 lần): `outputs/Replica/CoMe/<scene>/{step0_colmap,step1_3DGS,step2_Mesh}`, chạy bằng `Based_Model/CoMe/modal_come_scene_mesh.py` (`::run` / `::download --dataset Replica --scene <scene>`).
- Gốc (pipeline cũ, vẫn dùng được): `modal_infusion.py`, `modal_lerf_prep.py`, `modal_sync.py` (Volumes `safe-gs-dataset`, `safe-gs-outputs`, `safe-gs-hf-cache`), `README_MODAL.md`; `modal_train.py` còn trỏ `src/Module*` (nay ở `legacy/src_v0/`).

### B. Kho mô hình gốc làm tham chiếu (`Based_Model/`) & Nhánh GitHub `come-setup`
> **Quy tắc:**
> - Các thư mục trong `Based_Model/` là **bản chuẩn dùng để đối chiếu/tham khảo**, KHÔNG tự ý sửa code trực tiếp trong `Based_Model/` trừ khi người dùng yêu cầu rõ.
> - **Code & Script huấn luyện `CoMe 3DGS` (Step 0 COLMAP, Step 1 Train CoMe 3DGS, Step 2 Render):** Nằm trên nhánh **`come-setup` trên GitHub (`origin/come-setup`)**, cụ thể là file `modal_come.py` (844 dòng), `tools/come_apply_appearance.py`, `modal_inpaint360.py`, và bộ đo lường `pipeline/s10_eval_{gpu,cpu}.py`. Có thể đọc trực tiếp bằng `git show origin/come-setup:<path>`.
- `Based_Model/GOF`: Gaussian Opacity Fields (trích xuất mesh bằng Marching Tetrahedra + Binary Search).
- `Based_Model/MILo`: Mesh-In-the-Loop Gaussian Splatting.
- `Based_Model/Split_and_Splat`: Phân tách đối tượng 3DGS + SAM2 (`sam2/`, `point_projection/`).
- `Based_Model/Infusion`: Xóa vật và hoàn thiện 3DGS bằng Depth Inpainting.
- `Based_Model/CoMe`: Confidence-based Mesh / 3DGS (`filter_3D`, `confidence`, `appearance_embedding`).
- `Based_Model/Inpaint360GS`: Inpainting 360° 3DGS.
- `Based_Model/OMEGAS`: Hoàn thiện vùng khuất 3DGS.
- `Based_Model/InstantMesh`: Tái tạo Mesh 3D từ ảnh đa góc nhìn.

### C. Không gian ghép Model & Thử nghiệm (`test_model/`)
> **Có thể đã bị xóa** (người dùng dự định xóa `test_model/` và `legacy/`). Mọi thứ cần cho pipeline đã chuyển sang `src/`, `scripts/`, `configs/`, `outputs/LERF/{teatime_napkin,teatime_bear}` và `docs/reference/`. Nếu thư mục không tồn tại, bỏ qua mục này.
> **Lưu ý quan trọng:** Một số thư mục trong `test_model/` nằm trong `.gitignore` (như `CoMe_Infusion/`, `CoMe_Infusion_Inpainting/`). Khi làm việc với chúng, hãy truy cập trực tiếp bằng đường dẫn cụ thể thay vì tìm kiếm toàn cục.
- `test_model/CoMe_Infusion/`:
  - Kế hoạch chuẩn: `PLAN_InFusion_CoMe_rasterizer.md`.
  - Code chính: `test_model/CoMe_Infusion/src/` (ghép InFusion với rasterizer và định dạng 3DGS của CoMe).
  - Dữ liệu từng bước: `Step_0_Mask`, `Step_1_Sync_ID`, `Step_2_Output_train_3DGS`, `Step_3_Output_render_3DGS`, `Step_4_Output_Inpainting`, `Step_5_Combine_3DGS`, `Step_6_Mesh_GOF`.
- `test_model/CoMe_Inpainting360GS/`:
  - Kế hoạch chuẩn: `plan_mesh_completion_instantMesh.md`.
  - Các nhánh thử nghiệm: `src_napkin_basic/`, `src_napkin_omegas/`, `src_napkin_omegas_improve/`, `src_napkin_instantMesh/`. **Chỉ đọc nhánh `src_*` mà người dùng đang chỉ định**, không đọc cùng lúc cả 4 nhánh.
- `test_model/CoMe_Infusion_Inpainting/`:
  - Code chính: `test_model/CoMe_Infusion_Inpainting/src/` và `modal_sdxl_plane.py`.
- `test_model/2D_inpainting/` & `test_model/ID_synchornization/`: Notebooks thử nghiệm LaMa, SDXL, SD2 và đồng bộ ID SAM-Track.

---

## 🔧 3. QUY TRÌNH THỰC HIỆN TÁC VỤ CHUẨN

### Khi người dùng yêu cầu: *"Ghép 2 model theo file `.md`"*
1. Đọc kỹ file `.md` kế hoạch được chỉ định bằng công cụ `Read`.
2. Xác định rõ **thư mục nguồn tham chiếu** (VD: `Based_Model/CoMe`) và **thư mục đích chỉnh sửa** (VD: `test_model/CoMe_Infusion/src`).
3. Dùng **Serena** (`find_symbol` / `get_symbols_overview`) để đối chiếu chính xác các hàm/class giữa 2 model (như `GaussianModel`, `render`, `load_ply`, `save_ply`, `RasterizationSettings`).
4. Thực hiện từng Phase theo đúng file `.md`, kiểm tra đầy đủ điều kiện nghiệm thu (PSNR, số lượng Gaussian, định dạng tensor/ply) trước khi chuyển sang Phase tiếp theo.

### Khi người dùng yêu cầu: *"Chạy model trên một tập Dataset khác (Replica / LERF / SPIn-NeRF / DTU)"*
1. Kiểm tra định dạng camera & dữ liệu đầu vào:
   - COLMAP (`sparse/0/cameras.bin`, `images.bin`, `points3D.bin`) hay NeRF (`transforms.json`).
   - Mô hình camera (`PINHOLE` vs `SIMPLE_PINHOLE`, độ phân giải ảnh, `cx/cy`, hệ số scale/radius).
2. Dùng **Serena** kiểm tra nhanh `scene/dataset_readers.py`, `scene/colmap_loader.py`, hoặc các hàm trong `modal_*.py`.
3. Trên Windows PowerShell, khi chạy lệnh `modal run`, luôn đặt `$env:PYTHONIOENCODING="utf-8"` nếu script có in ký tự Unicode.
4. Với Base Framework: khai báo scene/vật trong `configs/datasets/<dataset>.yaml` (run = `<scene>_<object>`), kiểm tra bằng `python scripts/run_pipeline.py --list` và `--dataset <D> --run <run> --steps 4-15 --dry-run`, rồi chạy theo bảng trong `README.md` (mục 3). Dữ liệu Inpaint360GS phải có chữ `inpaint360` trong đường dẫn, LERF/Replica thì không (`src/scene/dataset_readers.py` rẽ nhánh theo chữ này).
5. Đánh giá: `python scripts/evaluate_all_metrics.py --dataset <D> --run <run>` (hoặc `--all`).
6. Không hardcode đường dẫn tuyệt đối hay tên profile Modal: dùng `src/common/paths.py` / `scripts/modal_common.py`.

### 💾 Lưu kết quả vào `F:\Safe-GS\outputs` (BẮT BUỘC sau mỗi lần chạy)
1. **Mọi bước chạy xong phải có kết quả nằm trong `outputs/`** theo cấu trúc `outputs/<LERF|Replica|Inpaint360GS>/ours/<scene>_<object>/step<N>/` (lấy đường dẫn qua `src/common/paths.py`: `OUTPUTS_DIR`, `get_run_dir`, `step_dir` — không ghi kết quả ra chỗ khác như `src/`, `scripts/`, thư mục tạm hay `test_model/`).
2. **Chạy trên Modal:** kết quả nằm trên Volume, chưa có ở máy local. Khi job (kể cả job `--detach`) chạy xong, **luôn chạy entrypoint `::download`** tương ứng để kéo về `outputs/`:
   - S0–2: `python -m modal run scripts/modal_come_train.py::download --dataset <D> --scene <scene> --run <run>`
   - S4–12: `python -m modal run scripts/modal_come_pipeline.py::download --dataset <D> --run <run>`
   - S13: `python -m modal run scripts/modal_come_mesh.py::download --dataset <D> --run <run>`
   - S14: `python -m modal run scripts/modal_instantmesh.py::download --dataset <D> --run-name <run>`
   - Hoặc dùng `python scripts/run_pipeline.py ... --modal` (tự chèn bước download sau mỗi bước Modal).
3. **Kiểm tra sau khi chạy:** xác nhận thư mục `outputs/<D>/ours/<run>/step<N>/` tồn tại và có file kết quả (liệt kê bằng `ls`, KHÔNG đọc file nặng `.ply/.pth/.npy`). Nếu thiếu → báo cho người dùng, không coi bước đó là hoàn thành.
4. Kết quả đánh giá (`evaluate_all_metrics.py`) cũng lưu trong `outputs/` (`evaluation_summary.json`, `outputs/council_report_summary.{md,csv}`).

