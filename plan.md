# KẾ HOẠCH THỰC NGHIỆM TOÀN DIỆN (EXPERIMENT PLAN) — `Safe-GS` (`src/` + `scripts/`)

Tài liệu này hướng dẫn **từng bước cụ thể (step-by-step)** để thực thi toàn bộ hệ thống thực nghiệm đối đầu (**Main Experiment — Table 1 & Table 2**) và nghiên cứu bóc tách (**Ablation Study — Table 3**) trực tiếp trên kiến trúc mã nguồn hiện tại (`src/` + `scripts/` + `configs/`).

> **Lưu ý:** Các bảng số liệu trong tài liệu này là **số liệu minh họa mục tiêu** để định hình cấu trúc báo cáo, KHÔNG phải kết quả. Số minh họa đã được chỉnh để nhất quán logic giữa các bảng (xem mục R).

---

## R. Nhật ký rà soát (rev. 2026-11-10) — Những chỗ đã sửa so với bản trước

| # | Vấn đề ở bản trước | Đã sửa thành |
| :-- | :-- | :-- |
| R1 | "16 bước (0→15)" nhưng sơ đồ có Step 16 (Physics); thiếu `6id`, `13p`. | Pipeline = Step `0–15` (thứ tự thật trong `run_pipeline.py`: `0,1,2,4,5,6,6id,7,8…12,13,13p,13b,14.1,14,15`) + **Step 16 Physics** (script mới). Step 3 không còn chạy. |
| R2 | `--steps 0-15` một lệnh cho run mới. | Không chạy được: run mới chưa có `target_id` → Step 8–12 cần `target_id` chọn ở `6id` (`run_pipeline.py` tự bỏ `6id` nếu thiếu `--click`). Tách thành 3 lượt `0-2` → `4-7` → `6id` → `8-15`. |
| R3 | Replica: Step `4-7` chỉ chạy cho `obj1`, rồi `8-12` cho `obj2`. | Output Step 4–7 nằm theo **run** (`outputs/Replica/<run>/step4..7`), nên `obj2` sẽ thiếu. Thêm cờ `--reuse-from <run>` (Pha 1) để copy Step 4–7 của `obj1` sang `obj2` (cùng không gian ID → `target_id` của 2 vật nhất quán). |
| R4 | `--come-flags` cho `modal_come_train.py::train`. | Không tồn tại. `train_scene(..., extra_flags)` chỉ **nối thêm** vào `COME_FLAGS` cố định ⇒ không tắt được gì. Cần cờ `--flags-preset {come,vanilla,gof}` **thay thế** `COME_FLAGS` (Pha 1). |
| R5 | "Vanilla 3DGS = chạy `src/come/train.py` không có các cờ CoMe". | Sai: mặc định trong `src/come/arguments` có `lambda_distortion=1000`, `lambda_depth_normal=0.05`, `lambda_opacity_field=0.004`, `splatting_config=configs/hierarchical.json` ⇒ vẫn là GOF/CoMe có ràng buộc. Preset `vanilla` phải **đặt về 0 tường minh** và thay `splatting_config` (xem 1.1.2-(2)). Cùng lỗi ở Ablation (a). |
| R6 | GOF baseline dùng `--lambda_normal_variance 0.05` gọi là "depth-normal". | Tên cờ sai (`lambda_normal_variance` là của CoMe; depth-normal là `lambda_depth_normal`). Ưu tiên chạy **GOF gốc** (`Based_Model/GOF`) cho baseline có thể trích dẫn; preset `gof` trong code CoMe chỉ là phương án dự phòng, phải ghi rõ trong paper. |
| R7 | Ablation (a) chỉ tắt `distortion/variance/normal_variance`, Step 11 không có cách tắt. | Dùng preset `vanilla` ở Step 1 và biến môi trường `COME_LOSS=0` ở Step 11 (`src/utils/come_loss.py::come_loss_enabled`). Biến này phải được truyền **vào container Modal** qua `spec` (Pha 1), đặt ở máy local không có tác dụng. |
| R8 | Ablation (a) chạy `--run teatime_napkin_ablation_no_come` không khai báo. | `resolve_run` sẽ suy scene = `teatime_napkin_ablation_no` (sai) và không có `target_id`. Phải khai báo run trong YAML (`scene: teatime`), dùng `--volume-key` riêng để không ghi đè model CoMe, và chọn lại `target_id` (ID chỉ hợp lệ trong run tạo ra nó). |
| R9 | Đo **Object CD** so với GT scan Replica "nguyên vẹn". | GT scan của Replica **không có mặt đáy** vật (bị bàn che khi quét) ⇒ mesh kín nước của Ours bị phạt oan ở đáy. Phải **đóng đáy GT** bằng mặt phẳng đỡ (cap) trước khi đo; nêu rõ trong paper. Replica bản NICE-SLAM **không có mesh từng vật** → tách từ `mesh_semantic.ply` + `info_semantic.json` của `replica_v1`. |
| R10 | Background CD trên `scene_gt.ply` (đã cắt bbox vật). | Vùng dưới vật **không có GT** (bị che khi quét) ⇒ điểm inpaint ở đó bị phạt oan. Background CD chỉ đo **ngoài footprint vật (giãn 5 cm)**; vùng dưới vật đo riêng bằng **Under-object completeness / RMSE so với mặt phẳng GT**. |
| R11 | Căn chỉnh COLMAP↔GT bằng ICP mesh riêng cho từng method; `registration.py` của TNT. | ICP theo từng method là không công bằng (method tệ → căn chỉnh tệ/tốt bất thường). Tất cả method dùng **chung COLMAP** ⇒ tính **một** Sim(3) mỗi scene bằng Umeyama trên tâm camera (COLMAP ↔ `traj.txt` GT), tinh chỉnh ICP trên vùng nền tĩnh, rồi áp dụng chung cho mọi method. Sim(3) cũng cho **thang đo mét** (COLMAP không có scale). |
| R12 | Đơn vị cm/mm trên LERF. | LERF không có thang đo mét ⇒ CD/Penetration trên LERF không có đơn vị. Table 3: `Object CD` **chỉ trên Replica**; Penetration trên LERF quy đổi theo kích thước thật đo tay của vật (ví dụ chiều dài khăn) hoặc báo theo % đường chéo bbox vật. |
| R13 | Baseline 4 CoMe (Direct) và Ablation (b) là **cùng một cấu hình** nhưng số khác nhau (4.05 vs 4.30). | Đồng nhất: (b) = CoMe (Direct) ⇒ cùng Object CD. Đổi tên (b) thành `w/o InstantMesh (CoMe direct mesh)` (không phải "GOF only"). |
| R14 | Table 1: CoMe (Direct) Background CD (1.42) ≠ Ours (1.21) dù dùng **cùng** mesh nền Step 13b. | Hai dòng phải bằng nhau. Table 2: CoMe + Direct Mesh và Ours dùng **cùng** nền Step 11 ⇒ cột Background Inpainting phải bằng nhau. |
| R15 | Table 2 LERF Composite PSNR mục tiêu 27.15. | Không khả thi: render CoMe **chưa chỉnh sửa** trên tập test teatime chỉ đạt **PSNR 23.34** (`step4_renderer_check/summary.json`) — đó là cận trên. Thêm dòng **Upper bound (CoMe 3DGS không chỉnh)** và đặt lại số minh họa. Ours (vật là mesh sinh bởi InstantMesh) nhiều khả năng **thấp hơn nhẹ** Inpaint360GS (ghép lại chính Gaussian gốc) ở PSNR toàn ảnh ⇒ câu chuyện đúng là "ngang ngửa về ảnh, vượt trội về hình học/vật lý". |
| R16 | Inpaint360GS: Composite so với "ảnh GT gốc có vật" ở split test. | Với đường dẫn chứa `inpaint360`, split **test là ảnh KHÔNG có vật** ⇒ không có GT "có vật" ở test. Phải giữ lại 1/8 ảnh **train (có vật)** làm hold-out composite, loại khỏi huấn luyện cho **mọi** method (Pha 1). Ngoài ra GT sạch của Inpaint360GS xóa **tất cả** vật mục tiêu (doppelherz: `26` + surrounding `24,10`) ⇒ phải xóa đúng tập vật đó. |
| R17 | Composite: so render mesh (màu đỉnh) với render Gaussian. | Không công bằng. Quy ước chung: **nền = Gaussian (Step 11)**, **vật = biểu diễn của method**; vật dạng mesh được raster bằng `nvdiffrast` và ghép theo z-buffer với ảnh/độ sâu Gaussian. Báo thêm PSNR/LPIPS **trong mask vật** (masked) vì đó là chỗ các method khác nhau. |
| R18 | Baseline "SPIn-NeRF" chạy bằng `modal_infusion.py`. | `modal_infusion.py` là **InFusion**, không phải SPIn-NeRF. Đổi baseline thành **InFusion** (đã có code). SPIn-NeRF (NeRF, thiết kế cho cảnh forward-facing, chạy rất lâu) chỉ đưa vào nếu bắt buộc, khi đó trích số từ bài Inpaint360GS **đúng giao thức** và ghi chú "reported". |
| R19 | Ablation (d) "raw triangle mesh vào MuJoCo gây va chạm lõm–lõm, FPS 22". | Sai cơ chế: MuJoCo **luôn thay mesh geom bằng bao lồi (convex hull)** khi tính va chạm. "Không CoACD" trong MuJoCo = **1 bao lồi duy nhất** ⇒ hỏng ở vật lõm (lấp chỗ lõm, sai điểm tựa) chứ không phải nổ lực/FPS thấp. (d) hiện **đã bỏ khỏi Table 3** cùng toàn bộ đánh giá vật lý; ghi chú này giữ lại cho khi làm Step 16. Bàn nên mô hình hóa bằng `box`/`plane` theo `table_plane.json` hoặc CoACD; không cần CoACD cho bàn phẳng. |
| R20 | Table 3 đo trên LERF + Replica nhưng lệnh chỉ cho `teatime_napkin`. | Liệt kê đủ 4 run và vòng lặp. FPS đo trên **cùng một máy, 1 luồng CPU**, ghi cấu hình máy. |
| R21 | Hai nơi tạo script bảng (`scripts/eval_table1_replica.py` vs mở rộng `evaluate_all_metrics.py`). | Thống nhất: `evaluate_all_metrics.py` giữ chỉ số theo run; thêm `scripts/eval_tables.py` (Table 1/2/3) đọc JSON theo run. |
| R22 | Tên file cứng `mesh_napkin_placed.ply`, `mesh_come_table.ply`. | Thực tế (sau khi đổi tên, mọi run): `mesh_object_placed.ply`, `scene_scene_demo_object.ply`, Step 14 `mesh_object_instantmesh.ply` (tên cũ `mesh_napkin_*` vẫn được đọc dự phòng); mesh bàn có thể mang hậu tố biến thể (ví dụ `mesh_come_v2_table.ply` ở teatime_napkin) ⇒ script đọc tên từ `step13_come_mesh/done.json`, không hardcode. |
| R23 | Pha 1 "1–2 giờ code". | Ước lượng lại ~2 tuần (xem mục 4). |
| R24 | Chạy COLMAP SfM trên Replica dễ fail do bề mặt render phẳng, ít texture. | Dùng script `scripts/prepare_replica_colmap.py` (Pha 0) chuyển đổi trực tiếp `cam_params.json` + `traj.txt` (pose GT) sang format COLMAP (`sparse/0/cameras.*`, `images.*`, `points3D.*` lấy mẫu 150k điểm từ mesh). Đảm bảo 100% frame đăng ký thành công, scale metric chuẩn ($Sim(3) = I, \text{scale} = 1.0$). |
| R25 | Quản lý hạn mức $30 free của Modal. | Theo dõi credits; khi hết $30 free tier trên workspace/profile hiện tại, dừng ngay và thông báo người dùng đổi workspace khác (`modal profile use <profile>` hoặc biến `MODAL_PROFILE`). |
| R26 | Quy trình Replica: SfM nhận từ Pha 0 và target_id chọn sau Step 7. | Không chạy COLMAP SfM trên Modal (upload thẳng sparse/0 có sẵn qua `upload_colmap`); `target_id` để null ban đầu, sau khi HQ-SAM/Tracking xong ở Step 4–7 mới mở ảnh click tọa độ tại Step 6id để xác định ID rồi mới chạy Step 8–15. |
| R27 | "Step 6id tự động ghi `target_id` vào YAML". | Sai: `src/tools_safe/select_target_id.py` chỉ ghi `<run>/step6*/object_id.json` và **in ra** dòng `target_id: N` ⇒ phải tự chép vào YAML (hoặc thêm cờ `--write-yaml`, Pha 1 mục 10). Với Replica nên dùng `--ref-masks` (mask instance GT render từ `mesh_semantic.ply` với pose GT — COLMAP = GT nên khớp pixel) thay vì `--click`: có IoU/precision/recall khách quan và phát hiện vật bị tách thành nhiều ID. |
| R28 | Ablation (a) dùng `--volume-key lerf/teatime_nocome` cho Step 4–7. | **Lỗi nghiêm trọng:** `--volume-key` chỉ đổi nguồn model ở `_prep` (`--come-key`), còn dữ liệu/model/trạng thái Step 4–7 trên Volume `inpaint360gs` khóa theo `dk = <dataset>/<scene>` (`pipeline_steps.make_spec`) ⇒ (1) model vanilla **ghi đè** `output/lerf/teatime` của Ours; (2) `state/lerf/teatime` báo 4–7 đã xong ⇒ Step 6–7 **không chạy lại** trên model vanilla. Sửa Pha 1 mục 11: `dk = come_key` khi có `--volume-key` (hoặc khai báo scene riêng `teatime_nocome` cùng `data_subdir`). |
| R29 | R3: "Output Step 4–7 nằm theo run". | Chính xác hơn: Step `prep,4–7` **tính theo scene** trên Volume (`SCENE_STEPS`, `state/<dk>`), chỉ phần **export/download** theo run. ⇒ `--reuse-from` chỉ cần copy local `step0..7` của `obj1` sang `obj2` (không cần GPU, không cần copy trên Volume); Step 8 của `obj2` tự tạo `runs/<dk>/<obj2>` từ `data/<dk>`, `output/<dk>`. `obj2` cũng cần `step0–2` local (Step 1–2 chỉ download vào run `obj1`). |
| R30 | Preset `vanilla` "thay thế `COME_FLAGS`". | Phải **giữ** các cờ không thuộc hình học của `COME_FLAGS`: `--eval --far_plane 100. --detach_alpha False` (cùng split train/test 1/8, cùng far plane) — chỉ bỏ/0 các thành phần hình học & appearance. Lưu ý `--eval` ⇒ Replica chỉ train trên ~219/250 frame; ghi rõ trong paper. |
| R31 | Baseline 1 / Ablation (a): `::train --volume-key replica/<scene>_vanilla`. | Key mới **chưa có COLMAP** ⇒ `train` fail (`_stage_colmap`). Phải `upload_colmap --volume-key ...` trước (Replica) hoặc copy COLMAP từ key gốc (LERF, Pha 1 mục 12). Entrypoint `train` hiện cũng **không nhận** cờ nào ngoài `iterations` ⇒ `--flags-preset` phải thêm cả vào `train` lẫn `train_scene`. TSDF chưa có lệnh ⇒ thêm entrypoint `modal_baselines.py::tsdf`. |
| R32 | Ablation (a) LERF chạy `--steps 0-2` với key mới. | Step 0 chạy lại COLMAP ⇒ **khác COLMAP của Ours** (phá giả định "chung COLMAP", thêm 20–40 phút). Chỉ chạy `1-2`, COLMAP copy từ `lerf/teatime` (`::copy_colmap`, Pha 1 mục 12). |
| R33 | Inpaint360GS chạy Step 0 (COLMAP lại). | Dữ liệu đã có `sparse/` của tác giả ⇒ chạy COLMAP lại cho hệ tọa độ khác baseline Inpaint360GS gốc và khác `unseen_mask`/camera test. Dùng `upload_colmap` (tổng quát hóa cho Inpaint360GS: kiểm tra camera model là PINHOLE, nếu không phải thì undistort trước) rồi chạy từ Step 1. |
| R34 | Inpaint360GS: "`target_id` phải bao gồm MỌI vật". | `target_id` là **một số nguyên**; các vật khác vào `target_surrounding_id: "a,b"` (Step 8 truyền `--target_surronding_id`). Step 14.1/14 chỉ dựng mesh cho `target_id` ⇒ Composite I360 của Ours chỉ có 1 vật mesh. Quy ước chung cho mọi method: vật phụ được ghép lại bằng **Gaussian gốc**, masked metric chỉ tính trên mask `target_id`. Kiểm tra lại ngữ nghĩa `surrounding` trong `Based_Model/Inpaint360GS` trước khi chốt. |
| R35 | Table 3 vs Table 1 không nhất quán: Ours F@1cm 68.0 vs 84.3, (b) 41.0 vs 72.8 nhưng Object CD giống hệt (1.87, 4.05). | Table 3 chỉ dùng `obj1` ⇒ hoặc CD cũng khác, hoặc lấy cùng tập run. Đã chỉnh: Object CD/F@1cm của Table 3 lấy **cùng tập 4 run Replica** như Table 1 (Ours, (b) trùng số Table 1); (a)/(c) đặt lại cho đơn điệu CD↔F. |
| R36 | Chi phí: Table 1 Ours "S0–7", Table 3 (a) "S7–14", Pha 4 "train lại 3 scene". | Replica không chạy S0 ⇒ "S1–7"; (a) chạy `S4–15` (4–7 phải chạy lại trên model vanilla); chỉ **1** scene train mới (`teatime`), `room0/office0` dùng lại Baseline 1. |

---

## 0. Kiến trúc Ánh xạ Pipeline `Safe-GS` (`src/` + `scripts/`)

Luồng End-to-End của **Ours (Full Pipeline)** gồm Step `0–15` (điều phối bởi [`scripts/run_pipeline.py`](scripts/run_pipeline.py)) + Step `16` Physics (script mới):

```
RGB Images
  ──[S0-2 : scripts/modal_come_train.py (src/come/train.py)  | theo SCENE]────► CoMe 3DGS (phẳng, confidence + filter_3D)
  ──[S4-7 : scripts/modal_come_pipeline.py (HQ-SAM, sync ID, distill)  | theo SCENE*]
  ──[S6id : src/tools_safe/select_target_id.py (local, --click)]─────────────► target_id → configs/datasets/*.yaml
  ──[S8-12: modal_come_pipeline.py (removal, SAM-Track, LaMa, inpaint+CoMe loss) | theo RUN]► nền 3DGS đã inpaint
  ──[S13 / 13p / 13b: modal_come_mesh.py + estimate_table_plane.py + crop_table.py]──► mesh nền + table_plane.json
  ──[S14.1 local + S14.2-14.4 modal_instantmesh.py]──────────────────────────► mesh vật kín nước (InstantMesh)
  ──[S15 : src/tools_instantmesh/place_on_table.py]──────────────────────────► căn chỉnh & ghép (scene_table_<obj>.ply)
  ──[S16 : src/tools_mesh/run_physics_eval.py (CoACD + MuJoCo, local CPU)]───► physics_summary.json   (HOÃN — chưa dùng cho Table 3)
```
`*` S4–7 được **tính theo scene** trên Volume (`state/<dk>`), chỉ export/download theo run ⇒ vật thứ 2 cùng scene dùng `--reuse-from` (Pha 1: copy local `step0..7`, R29).

### Kiểm tra môi trường trước khi chạy
```powershell
$env:PYTHONIOENCODING = "utf-8"
python scripts/run_pipeline.py --check-env
python scripts/run_pipeline.py --list
python scripts/run_pipeline.py --dataset LERF --run teatime_napkin --steps 4-15 --dry-run
```

### Quy trình chuẩn cho MỘT run mới (áp dụng cho mọi dataset)
```powershell
# (0) Khai báo run trong configs/datasets/<d>.yaml với target_id: null
python scripts/run_pipeline.py --dataset <D> --run <run> --steps 0-2 --modal          # theo scene (1 lần / scene)
python scripts/run_pipeline.py --dataset <D> --run <run> --steps 4-7 --modal          # tự download sau khi xong
python scripts/run_pipeline.py --dataset <D> --run <run> --step 6id --click <frame> <x> <y>   # <frame> = tên ảnh không đuôi, (x,y) theo pixel ảnh gốc
#     → script CHỈ in "target_id: N" + ghi step6*/object_id.json: TỰ chép target_id (+ samtrack_click nếu cần) vào YAML (R27)
python scripts/run_pipeline.py --dataset <D> --run <run> --steps 8-15 --dry-run      # kiểm tra target_id đã được đọc đúng
python scripts/run_pipeline.py --dataset <D> --run <run> --steps 8-15 --modal
python scripts/evaluate_all_metrics.py --dataset <D> --run <run>
```
> Dùng `--detach` cho job dài: `run_pipeline.py` dừng sau job detach đầu tiên, phải chạy lại lệnh `download` khi job xong.

---

## BƯỚC 1: MAIN EXPERIMENT

### Bước 1.1: Geometric & Surface Reconstruction Quality (Replica — Table 1)

#### 1.1.1. Chuẩn bị dữ liệu & cấu hình (`configs/datasets/replica.yaml`)
- **2 scene (tối ưu chi phí):** `room0`, `office0`; **2 vật/scene** đặt trên bàn/sàn ⇒ **4 object runs** (`room0_obj1`, `room0_obj2`, `office0_obj1`, `office0_obj2`).
- **Tiêu chí chọn vật:** nhìn thấy trong ≥ 30% số frame, kích thước 10–40 cm, có mặt đỡ phẳng, không bị cắt bởi biên ảnh ở đa số frame (InstantMesh cần một góc nhìn sạch ở Step 11–14).
- **Nguồn dữ liệu & Tiền xử lý (Pha 0):**
  - Ảnh RGB + `traj.txt` (pose GT) + `cam_params.json` (intrinsics): bản render Replica (NICE-SLAM). Trajectory là chuỗi liên tục ~2000 frame ⇒ **lấy mẫu 250 frame** đều (đủ tái tạo chi tiết, giảm tải huấn luyện).
  - **Tạo pseudo-COLMAP trực tiếp (R24):** Chạy `python scripts/prepare_replica_colmap.py --scenes office0 room0`. Script tự động tính $T_{w2c} = T_{c2w}^{-1}$, xuất quaternion và translation chuẩn COLMAP vào `sparse/0/cameras.*`, `images.*` (cả text và binary). Đồng thời lấy mẫu 150.000 điểm 3D từ `<scene>_mesh.ply` lưu thành `points3D.*` và `points3D.ply` có đầy đủ màu RGB. Đảm bảo 100% camera được đăng ký và giữ nguyên hệ tọa độ metric thực.
  - GT mesh từng vật: tách từ `replica_v1/<scene>/habitat/mesh_semantic.ply` theo `instance id` (`info_semantic.json`). **Phải kiểm tra** mesh `replica_v1` và mesh NICE-SLAM cùng hệ tọa độ (so khớp trên vài điểm; nếu lệch, ICP 1 lần và lưu ma trận).
  - **Đóng đáy GT vật** (R9): chiếu viền dưới lên mặt phẳng đỡ và vá bằng tam giác ⇒ `<obj>_gt_capped.ply`.
- **Cấu trúc thư mục:**
  ```text
  # DATA_DIR = F:/ (.env: SAFE_GS_DATA_DIR). Raw (chỉ đọc): F:/Replica 8 Scene/Replica/<scene>/{results/image, traj.txt}
  #                                         GT semantic: F:/Replica/Replica-Dataset/<room_0|office_0>/habitat/
  F:/Replica 8 Scene/Replica_safegs/<scene>/     # data_subdir trong replica.yaml, do Pha 0 tạo ra
    ├── images/                       # 250 frame lấy mẫu (frame000000.jpg, ...)
    ├── traj_gt.txt                   # pose GT của đúng 250 frame trên
    ├── sparse/0/                     # Format COLMAP chuẩn (cameras.txt/.bin, images.txt/.bin, points3D.txt/.bin/.ply)
    ├── gt_mesh/
    │   ├── scene_gt.ply              # mesh GT toàn cảnh (copy từ <scene>_mesh.ply)
    │   ├── <obj>_gt.ply              # mesh GT vật (tách theo instance)
    │   ├── <obj>_gt_capped.ply       # mesh GT vật đã đóng đáy
    │   └── <obj>_footprint.json      # đa giác footprint + mặt phẳng đỡ GT
    └── T_colmap_to_gt.json           # Ma trận Sim(3) Identity (tỷ lệ 1.0 m, do COLMAP sinh trực tiếp từ GT)
  ```
- **Khai báo YAML trước khi chạy** (target_id điền sau `6id`):
  ```yaml
  scenes:
    room0:
      objects:
        obj1: {run: room0_obj1, target_id: null, samtrack_click: null}
        obj2: {run: room0_obj2, target_id: null, samtrack_click: null}
    office0:
      objects:
        obj1: {run: office0_obj1, target_id: null, samtrack_click: null}
        obj2: {run: office0_obj2, target_id: null, samtrack_click: null}
  ```

#### 1.1.2. Cách chạy 5 phương pháp

Công bằng: mọi method dùng **chung ảnh + COLMAP Step 0** của scene và **chung một prism 3D của vật** (footprint từ `table_plane.json` + chiều cao từ Step 14.1 của Ours, giãn 2 cm) để tách mesh nền / mesh vật. Nhờ vậy baseline **không cần** chạy lại phân đoạn ID (Step 4–8).

##### (1) Ours (Full Pipeline)

> [!IMPORTANT]
> **Lưu ý thực thi cho Replica (R24, R26):**
> 1. **Đầu vào SfM (Step 0):** Replica đã được tạo sẵn COLMAP offline ở Pha 0 (`sparse/0/` chuẩn từ GT). **KHÔNG chạy Step 0 (COLMAP SfM) trên Modal** để tránh tốn GPU và tránh fail. Thay vào đó, tải thẳng COLMAP có sẵn lên Volume bằng `upload_colmap`, rồi chạy thẳng từ Step 1 (Train CoMe).
> 2. **Cơ chế chọn `target_id` (Step 6id):** Trước khi chạy Step 4–7, mô hình HQ-SAM và module Tracking chưa chạy nên **chưa ai biết ID của các vật là bao nhiêu** $\Rightarrow$ Bắt buộc để `target_id: null` trong `replica.yaml`. Sau khi Step 4–7 chạy xong, chạy Step 6id ở local (khuyến nghị `--ref-masks` từ mask instance GT; hoặc `--click` trên 1 ảnh RGB), đọc `target_id` script in ra và **chép tay** vào YAML trước khi sang Step 8–15 (R27).

```powershell
$scenes = @("room0","office0")

# 1. Tải thẳng COLMAP có sẵn từ Pha 0 lên Modal Volume (Bỏ qua chạy SfM trên Modal)
foreach ($s in $scenes) {
    python -m modal run scripts/modal_come_train.py::upload_colmap --dataset Replica --scene $s
}

# 2. Huấn luyện 3DGS ban đầu (Step 1-2) và Phân đoạn ngữ nghĩa toàn cảnh (Step 4-7)
foreach ($s in $scenes) {
    # Train CoMe 3DGS toàn cảnh (Step 1-2)
    python scripts/run_pipeline.py --dataset Replica --run "${s}_obj1" --steps 1-2 --modal
    
    # Phân đoạn HQ-SAM + Tracking (Step 4-7) cho vật 1
    python scripts/run_pipeline.py --dataset Replica --run "${s}_obj1" --steps 4-7 --modal
    
    # Vật 2 dùng chung không gian phân đoạn với vật 1 (4-7 tính theo scene, R29): copy local step0..7, không tốn GPU
    python scripts/run_pipeline.py --dataset Replica --run "${s}_obj2" --steps 0-7 --reuse-from "${s}_obj1"
}
# Kiểm tra mắt trước khi đi tiếp: step4_renderer_check/summary.json (PSNR), step6_sync_mask/2_masks_synced_color/
# (vật mục tiêu có MỘT màu ổn định qua các frame, không bị tách nhiều ID).

# 3. TẠI ĐÂY MỚI CHỌN TARGET_ID (Step 6id - Chạy Local CPU, 0đ):
# Khuyến nghị (Replica): mask instance GT render từ mesh_semantic.ply + pose GT (script Pha 0) -> --ref-masks:
#   python scripts/run_pipeline.py --dataset Replica --run room0_obj1 --step 6id --ref-masks <gt_inst_masks/room0> --ref-id <instance_id>
# Hoặc click (frame = tên ảnh không đuôi, chọn frame vật nằm giữa ảnh, (X,Y) theo pixel ảnh gốc 1200x680):
#   python scripts/run_pipeline.py --dataset Replica --run room0_obj1 --step 6id --click frame000000 <X> <Y>
#   ... tương tự room0_obj2, office0_obj1, office0_obj2
# -> Script in "target_id: N" và ghi step6*/object_id.json. TỰ CHÉP N vào configs/datasets/replica.yaml (R27);
#    xem thêm candidates_top5 / iou_total: nếu IoU thấp, vật bị tách nhiều ID -> cân nhắc target_surrounding_id.
#    obj1 và obj2 cùng scene phải ra 2 ID KHÁC nhau.

# 4. Khi YAML đã có target_id: Chạy Xóa vật, Inpaint nền & Mesh InstantMesh (Step 8-15)
foreach ($s in $scenes) { foreach ($o in @("obj1","obj2")) {
    python scripts/run_pipeline.py --dataset Replica --run "${s}_$o" --steps 8-15 --modal
}}
```
- Output: mesh nền `step13_come_mesh/<tên trong done.json>_table.ply`; mesh vật `step15_object_mesh_merge/mesh_<obj>_placed.ply`.

##### (2) Baseline 1: 3DGS (không ràng buộc hình học) + TSDF
- Train bằng code CoMe với preset `vanilla` (Pha 1) — **đặt tường minh**:
  `--lambda_distortion 0 --lambda_depth_normal 0 --lambda_opacity_field 0 --lambda_extent 0 --lambda_smoothness 0 --lambda_variance 0 --lambda_normal_variance 0 --splatting_config <config không hierarchical>` và **không** `--color_confidence`, `--use_ssimdecoupled_appearance`.
  **Giữ** `--eval --far_plane 100. --detach_alpha False` như Ours (R30).
  Gọi tên trong paper là *"3DGS (w/o geometric regularization) + TSDF"* (vẫn có 3D filter của rasterizer GOF), hoặc dùng `src/gaussian_splatting` nếu muốn đúng "Vanilla 3DGS" (nhưng rasterizer gốc không xuất depth ⇒ phải thêm).
  ```powershell
  # key mới chưa có COLMAP -> upload trước (R31); model này dùng lại cho Ablation (a) Replica
  python -m modal run scripts/modal_come_train.py::upload_colmap --dataset Replica --scene <scene> --volume-key replica/<scene>_vanilla
  python -m modal run --detach scripts/modal_come_train.py::train --dataset Replica --scene <scene> --volume-key replica/<scene>_vanilla --flags-preset vanilla
  python -m modal run scripts/modal_come_train.py::render --dataset Replica --scene <scene> --volume-key replica/<scene>_vanilla
  python -m modal run scripts/modal_baselines.py::tsdf --dataset Replica --scene <scene> --volume-key replica/<scene>_vanilla   # Pha 1 mục 4
  ```
- Mesh: `src/come/extract_mesh_tsdf.py` trên **model toàn cảnh** (cần thư mục model + camera, không chạy trên `.ply` rời) ⇒ cắt bằng prism chung thành `mesh_bg.ply` / `mesh_obj.ply`.

##### (3) Baseline 2: GOF
- **Khuyến nghị:** `Based_Model/GOF` gốc (train + `extract_mesh.py` Marching Tetrahedra) qua wrapper `scripts/modal_baselines.py::gof` (Pha 1).
- Dự phòng: preset `gof` trong code CoMe (`--lambda_distortion 100 --lambda_depth_normal 0.05`, tắt `color_confidence`, `splatting_config` không hierarchical) — ghi rõ "GOF (re-impl. in CoMe codebase)".
- Mesh toàn cảnh → cắt bằng prism chung. Mesh vật sẽ hở đáy (camera không thấy mặt dưới).

##### (4) Baseline 3: MILo
- Wrapper `scripts/modal_baselines.py::milo` gọi `Based_Model/MILo` (train + mesh extraction) trên cùng ảnh + COLMAP. Cần image Modal riêng (submodule CUDA của MILo), dự kiến GPU L40S.
- Cắt bằng prism chung → `outputs/Replica/<run>/baselines/milo/{mesh_bg,mesh_obj}.ply`.

##### (5) Baseline 4: CoMe (Direct)
- Nền: **chính mesh nền Step 13b của Ours** ⇒ Background CD = Ours (R14).
- Vật: `extract_mesh_tets.py` trên Gaussian vật (`point_cloud_<tid>_object.ply`, Step 14.1):
  ```powershell
  python -m modal run scripts/modal_come_mesh.py::run_object_direct --dataset Replica --run <run>
  ```
  → `step13_come_mesh/mesh_come_direct_obj.ply`. (`extract_mesh_tets.py` cần thư mục model + camera ⇒ entrypoint mới phải dựng model tạm chỉ chứa Gaussian vật.)

> **Vì sao baseline 1–3 không inpaint nền?** Vùng dưới vật không có GT nên không ảnh hưởng Background CD (R10). Khác biệt về inpainting được đo bởi cột **Under-object completeness**, nơi baseline (để lại lỗ) sẽ ≈ 0%. Nếu reviewer yêu cầu, chạy thêm Step 8–12 cho baseline (chi phí +≈ $20–36, xem mục 4).

#### 1.1.3. Đo lường (`scripts/eval_tables.py --table 1`)
1. **Căn chỉnh (một lần mỗi scene, R11):** Umeyama Sim(3) giữa tâm camera COLMAP và `traj_gt.txt` → tinh chỉnh ICP point-to-plane trên mesh nền **ngoài** footprint vật → `T_colmap_to_gt.json`. Kiểm tra: sai số tâm camera sau căn chỉnh < 1 cm.
2. **Chỉ số** (mẫu 100k điểm đều trên bề mặt, trung bình trên 4 run):
   - **Background CD ↓ (cm):** CD L2 hai chiều, chỉ ngoài footprint giãn 5 cm, trong bbox cảnh chung.
   - **Under-object completeness ↑ (%):** % diện tích footprint có mesh nền nằm trong ±1 cm so với mặt phẳng đỡ GT.
   - **Object CD ↓ (cm):** so với `<obj>_gt_capped.ply`.
   - **Object F-Score@1cm / @2cm ↑ (%)** (tự cài precision/recall; `EvaluateHisto` của TNT có voxel-downsample + crop riêng, không dùng trực tiếp).
   - **Normal Consistency ↑:** trung bình $|\mathbf{n}_{pred}\cdot\mathbf{n}_{gt}|$ theo cặp láng giềng gần nhất, hai chiều.
   - **Watertight?:** `mesh_io.edge_stats` (`boundary == 0` và `non_manifold == 0`, như `eval_step15` trong `evaluate_all_metrics.py`).
```powershell
python scripts/eval_tables.py --table 1 --dataset Replica --out outputs/table1_replica_geometry
```

#### Table 1: Quantitative geometry evaluation on Replica (số minh họa mục tiêu)

| Method | Background CD ↓ (cm) | Under-object compl. ↑ (%) | Object CD ↓ (cm) | Object F@1cm ↑ (%) | Object F@2cm ↑ (%) | Normal Cons. ↑ | Watertight? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3DGS (w/o geo. reg.) + TSDF** | 3.12 | 0 | 5.48 | 56.4 | 72.1 | 0.785 | No |
| **GOF** | 2.15 | 0 | 4.82 | 63.5 | 78.9 | 0.824 | No |
| **MILo** | 1.85 | 0 | 4.30 | 69.2 | 82.5 | 0.861 | No |
| **CoMe (Direct)** | **1.21** | **91.5** | 4.05 | 72.8 | 84.6 | 0.879 | No |
| **Ours** | **1.21** | **91.5** | **1.87** | **84.3** | **93.1** | **0.924** | **Yes (100%)** |

> CoMe (Direct) và Ours chung mesh nền ⇒ hai cột nền bằng nhau; khác biệt nằm ở vật (InstantMesh đóng kín đáy).

---

### Bước 1.2: Novel View Synthesis & Inpainting Quality (Inpaint360GS & LERF — Table 2)

#### 1.2.1. Dữ liệu & cấu hình
- **Inpaint360GS** (`inpaint360gs.yaml`, có ảnh **Clean GT** ở split test): `doppelherz_main` (đã có `target_id: 26`, surrounding `24,10` — kiểm tra lại sau Step 6), `toys`, `fruits` (**chưa có ID** → khai báo run + `6id`).
  - Dữ liệu đã có sẵn tại `F:/inpaint360/inpaint360/<scene>/{images*,sparse,unseen_mask}` (không cần tải lại).
  - Đường dẫn phải chứa `inpaint360` (`data_subdir: inpaint360/inpaint360` trong `inpaint360gs.yaml`, DATA_DIR = `F:/`).
- **LERF:** `F:/LERF Datasets/<scene>/<scene>/` (mỗi scene khai báo `data_subdir: LERF Datasets/<scene>`); `ramen` chỉ có `transforms.json`, chưa có `sparse/0`.
  - Clean GT xóa **mọi** vật mục tiêu của scene ⇒ cấu hình removal phải xóa đúng tập ID đó (R16).
- **LERF** (`lerf.yaml`, `has_object_free_gt: false`): `teatime_napkin` (**đã xong 0–15** tại `outputs/LERF/teatime_napkin`), `teatime_cafe_box` (**đã xong 0–15** tại `outputs/LERF/teatime_cafe_box` — hộp cafe cao, lệch tâm, đối lập hình thái với khăn dẹt `napkin`), `ramen_main` (khai báo scene `ramen` mới).

#### 1.2.2. Giao thức đo
1. **Background Inpainting (chỉ Inpaint360GS):** render nền Gaussian sau xóa + inpaint tại camera test (`inpaint/`), so với Clean GT: `PSNR / SSIM / LPIPS` (+ masked FID) — đã có ở `eval_step12` của `evaluate_all_metrics.py`.
2. **Composite Scene (Inpaint360GS & LERF):**
   - Ảnh GT **có vật**: LERF dùng split test 1/8 (`--eval`); Inpaint360GS phải **giữ lại 1/8 ảnh train có vật** làm hold-out (loại khỏi training của mọi method, Pha 1) vì split test là ảnh không có vật (R16).
   - Render: nền Gaussian (Step 11) + vật của method, ghép z-buffer (`nvdiffrast` cho vật dạng mesh) (R17).
   - Báo `PSNR / SSIM / LPIPS` toàn ảnh **và** trong mask vật (masked).
   - Thêm dòng **Upper bound** = render CoMe 3DGS chưa chỉnh (Step 2/4) tại cùng view.

#### 1.2.3. Các phương pháp & Tận dụng kết quả đã có sẵn trên máy
- **(1) InFusion** (thay SPIn-NeRF, R18): `modal_infusion.py` / `Based_Model/Infusion`, 3DGS cơ bản; composite = Gaussian vật gốc ghép lại. Output `outputs/<D>/<run>/baselines/infusion/{inpaint,composite}/`.
  - **Tận dụng sẵn cho `teatime_napkin`:** lấy trực tiếp từ `test_model/CoMe_Infusion/{Step_4_Output_Inpainting,Step_5_Combine_3DGS,Step_6_Mesh_GOF}` (không cần chạy lại).
- **(2) Inpaint360GS (gốc):** `Based_Model/Inpaint360GS` với rasterizer 3DGS chuẩn; composite bằng `src/tools/combine_gaussian_scene.py -s <src> -m <model> --iteration 5000 --object_list <target_id>`.
- **(3) CoMe + Direct Mesh:** nền Step 11 của Ours (⇒ cột Background = Ours) + `mesh_come_direct_obj.ply` (màu đỉnh).
  - **Tận dụng sẵn cho `teatime_napkin`:** đã có mesh khăn CoMe trực tiếp tại `test_model/CoMe_Inpainting360GS/output_napkin_basic/teatime_napkin/step15_object_mesh_merge/mesh_come_napkin_v2_iter1.ply`.
- **(4) Ours:** nền Step 11 + `mesh_<obj>_placed.ply` (texture/màu đỉnh InstantMesh).
  - **Đã xong 100% cho `teatime_napkin` và `teatime_cafe_box`** (`outputs/LERF/{teatime_napkin,teatime_cafe_box}/step15_object_mesh_merge/scene_table_*.ply`).
> ⚠️ **Điều kiện tiên quyết (Pha 1 mục 7):** hold-out 1/8 ảnh train có vật của Inpaint360GS phải được cài vào dataset reader **TRƯỚC** khi chạy Step 0–2 cho bất kỳ method nào (Ours, InFusion, Inpaint360GS gốc). Train trước khi có hold-out ⇒ ảnh hold-out đã lọt vào training ⇒ phải train lại toàn bộ.

```powershell
# --- Inpaint360GS: 3 run mới, chạy đủ quy trình (cả 3 đều chưa có output) ---
$runs = @("doppelherz_main","toys_main","fruits_main")
$scenes = @{doppelherz_main="doppelherz"; toys_main="toys"; fruits_main="fruits"}
foreach ($run in $runs) {
    # KHÔNG chạy Step 0 (COLMAP lại): dùng sparse/ của tác giả để cùng hệ tọa độ với Inpaint360GS gốc (R33)
    python -m modal run scripts/modal_come_train.py::upload_colmap --dataset Inpaint360GS --scene $scenes[$run]
    python scripts/run_pipeline.py --dataset Inpaint360GS --run $run --steps 1-2 --modal   # SAU khi có hold-out
    python scripts/run_pipeline.py --dataset Inpaint360GS --run $run --steps 4-7 --modal
}
# 6id thủ công từng run (mở ảnh chọn điểm), rồi điền target_id vào configs/datasets/inpaint360gs.yaml.
# doppelherz_main đã có target_id 26 + surrounding "24,10" (ID của tác giả) -> vẫn chạy 6id để kiểm tra lại sau Step 6.
# toys_main / fruits_main: Clean GT xóa MỌI vật mục tiêu (R16) -> target_id = vật chính (1 số),
#   các vật còn lại vào target_surrounding_id: "a,b" (R34). Chạy 6id một lần cho mỗi vật để lấy đủ ID.
python scripts/run_pipeline.py --dataset Inpaint360GS --run doppelherz_main --step 6id --click <frame> <x> <y>
python scripts/run_pipeline.py --dataset Inpaint360GS --run toys_main       --step 6id --click <frame> <x> <y>
python scripts/run_pipeline.py --dataset Inpaint360GS --run fruits_main     --step 6id --click <frame> <x> <y>
foreach ($run in $runs) {
    python scripts/run_pipeline.py --dataset Inpaint360GS --run $run --steps 8-15 --modal
    python scripts/evaluate_all_metrics.py --dataset Inpaint360GS --run $run
}

# --- LERF ---
# teatime_napkin & teatime_cafe_box: ĐÃ XONG Step 0-15, không cần chạy lại GPU
# ramen_main: run mới đầy đủ (scene ramen chưa có sparse/0 -> Step 0 chạy COLMAP)
python scripts/run_pipeline.py --dataset LERF --run ramen_main --steps 0-2 --modal
python scripts/run_pipeline.py --dataset LERF --run ramen_main --steps 4-7 --modal
python scripts/run_pipeline.py --dataset LERF --run ramen_main --step 6id --click <frame> <x> <y>   # -> điền lerf.yaml
python scripts/run_pipeline.py --dataset LERF --run ramen_main --steps 8-15 --modal
python scripts/evaluate_all_metrics.py --dataset LERF --run ramen_main

python scripts/eval_tables.py --table 2 --out outputs/table2_nvs_inpainting
```

#### Table 2: Novel view synthesis and inpainting quality (số minh họa mục tiêu)

| Method | BG Inpaint (I360) PSNR ↑ | SSIM ↑ | LPIPS ↓ | Composite (I360) PSNR ↑ | SSIM ↑ | LPIPS ↓ | Composite (LERF) PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| *Upper bound (CoMe 3DGS, không chỉnh)* | — | — | — | *28.9* | *0.905* | *0.110* | *23.3* | *0.820* | *0.190* |
| **InFusion** | 25.10 | 0.860 | 0.205 | 27.6 | 0.885 | 0.135 | 22.6 | 0.800 | 0.210 |
| **Inpaint360GS** | 28.15 | 0.912 | 0.138 | **28.4** | **0.898** | **0.118** | **22.9** | **0.812** | **0.198** |
| **CoMe + Direct Mesh** | **28.60** | **0.918** | **0.125** | 26.9 | 0.880 | 0.145 | 21.6 | 0.790 | 0.225 |
| **Ours** | **28.60** | **0.918** | **0.125** | 28.0 | 0.893 | 0.124 | 22.6 | 0.806 | 0.204 |

> Cận trên LERF lấy từ số đo thật: PSNR test 23.34 (`outputs/LERF/teatime_napkin/step4_renderer_check/summary.json`). Mục tiêu hợp lý của Ours ở Composite là **sát Inpaint360GS (≤ 0.5 dB)** và **vượt CoMe + Direct Mesh**; lợi thế chính của Ours được chứng minh ở Table 1 & 3.

---

## BƯỚC 2: ABLATION STUDY (Table 3)

**Phạm vi:** chỉ chạy **3 biến thể (a), (b), (c)** so với Ours (Full). **Chưa** đánh giá trong physics engine (không MuJoCo/CoACD, không Step 16, bỏ biến thể (d) w/o CoACD) — Table 3 chỉ dùng chỉ số hình học/đặt vật đo được từ output Step 13–15.

**Run:** LERF `teatime_napkin`, `teatime_cafe_box` (cả 2 đã có sẵn trọn bộ Step 0–15 của Ours); Replica `room0_obj1`, `office0_obj1`. `Object CD`/`F@1cm` chỉ tính trên 2 run Replica (R12); các cột còn lại trung bình trên 4 run.

**Chỉ số (không cần mô phỏng):**
- **Watertight?:** `boundary_edges == 0` và `non_manifold_edges == 0` trên mesh vật cuối.
- **Object CD ↓ (cm)** và **Object F@1cm ↑ (%):** Replica, so với `<obj>_gt_capped.ply` (cùng hàm metric Table 1).
- **Under-object height RMSE ↓ (mm):** `under_object.height_rmse` — độ gợn mặt bàn dưới vật so với mặt phẳng đỡ.
- **Static penetration ↓:** từ JSON của `place_on_table.py` (`after.vertices_inside_table` và p05 của `clearance_min_p05_p50`, đổi sang mm) — vật xuyên vào bàn bao nhiêu sau khi đặt.

### Bước 2.1 — (a) w/o CoMe
- Step 1 preset `vanilla` (R5) + Step 11 `COME_LOSS=0` (R7); giữ Step 13–15.
- **Tận dụng sẵn:**
  - Trên `teatime_napkin`, đã có sẵn so sánh mặt bàn khi tắt/bật CoMe loss ở Step 11 (`mesh_come_v1_table.ply` vs `mesh_come_v2_table.ply`) tại `outputs/LERF/teatime_napkin/step13_come_mesh/eval_table_v1_vs_v2/mesh_table_check.json` và `eval_under_napkin_4meshes/under_napkin.json` (`mesh_no_napkin_GOF_table.ply`).
  - Trên Replica (`room0`, `office0`), **tái sử dụng luôn model Vanilla 3DGS đã train ở Table 1 (Baseline 1)** và dùng lại mask HQ-SAM (Step 5–6) của Ours.
- Khai báo run mới (R8), ví dụ trong `lerf.yaml`: `teatime: objects: napkin_nocome: {run: teatime_napkin_nocome, target_id: null}`; model ở `--volume-key lerf/teatime_nocome`.
> ⚠️ **Chặn bởi Pha 1 mục 11 (R28):** chưa sửa `dk` thì Step 4–7 với `--volume-key` sẽ ghi đè model Ours trên Volume `inpaint360gs` và bị bỏ qua do `state/lerf/teatime` đã xong.
```powershell
# LERF: COLMAP copy từ key của Ours (KHÔNG chạy Step 0 lại, R32), train vanilla từ Step 1
python -m modal run scripts/modal_come_train.py::copy_colmap --src-key lerf/teatime --volume-key lerf/teatime_nocome   # Pha 1 mục 12
python scripts/run_pipeline.py --dataset LERF --run teatime_napkin_nocome --steps 1-2 --volume-key lerf/teatime_nocome --flags-preset vanilla --modal
python scripts/run_pipeline.py --dataset LERF --run teatime_napkin_nocome --steps 4-7 --volume-key lerf/teatime_nocome --modal
python scripts/run_pipeline.py --dataset LERF --run teatime_cafe_box_nocome --steps 0-7 --reuse-from teatime_napkin_nocome
python scripts/run_pipeline.py --dataset LERF --run teatime_napkin_nocome   --step 6id --click <frame> <x> <y>   # -> chép target_id vào lerf.yaml
python scripts/run_pipeline.py --dataset LERF --run teatime_cafe_box_nocome --step 6id --click <frame> <x> <y>
foreach ($r in @("teatime_napkin_nocome","teatime_cafe_box_nocome")) {
    python scripts/run_pipeline.py --dataset LERF --run $r --steps 8-15 --volume-key lerf/teatime_nocome --no-come-loss --modal
}

# Replica: model vanilla đã train ở Table 1 Baseline 1 (key replica/<s>_vanilla) -> bắt đầu từ Step 2 download
foreach ($s in @("room0","office0")) {
    $r = "${s}_obj1_nocome"
    python -m modal run scripts/modal_come_train.py::download --dataset Replica --scene $s --run $r --volume-key "replica/${s}_vanilla"   # chỉ download step0-2 (render đã chạy ở Baseline 1)
    python scripts/run_pipeline.py --dataset Replica --run $r --steps 4-7 --volume-key "replica/${s}_vanilla" --modal
    python scripts/run_pipeline.py --dataset Replica --run $r --step 6id --ref-masks <gt_inst_masks/$s> --ref-id <instance_id>   # -> chép vào replica.yaml
    python scripts/run_pipeline.py --dataset Replica --run $r --steps 8-15 --volume-key "replica/${s}_vanilla" --no-come-loss --modal
}
```
> Step 5 (HQ-SAM 2D trên ảnh gốc) không phụ thuộc model ⇒ có thể copy `raw_hqsam` của Ours để tiết kiệm; Step 6 (`mask_associate.py` dùng model) và 7 **phải** chạy lại ⇒ ID mới, bắt buộc 6id lại.
- **Giả thuyết:** mặt bàn gợn (`under_object.height_rmse` tăng) ⇒ đặt vật kém ổn định, Static penetration tăng; Object CD xấu đi nhẹ.

### Bước 2.2 — (b) w/o InstantMesh (CoMe direct mesh)
- = Baseline 4 của Table 1 (R13): dùng `mesh_come_direct_obj.ply` đưa vào Step 15.
- **Tận dụng sẵn:**
  - `teatime_napkin`: lấy trực tiếp mesh CoMe của khăn đã trích sẵn tại `test_model/CoMe_Inpainting360GS/output_napkin_basic/teatime_napkin/step15_object_mesh_merge/mesh_come_napkin_v2_iter1.ply` (không cần chạy lại GPU).
  - `room0_obj1`, `office0_obj1`: lấy trực tiếp từ dòng `CoMe (Direct)` đã chạy ở Table 1.
  - Chỉ cần chạy thêm 1 lệnh `::run_object_direct` trên Modal cho `teatime_cafe_box` (~2 phút).
```powershell
python -m modal run scripts/modal_come_mesh.py::run_object_direct --dataset LERF --run teatime_cafe_box
python src/tools_instantmesh/place_on_table.py --run outputs/LERF/teatime_napkin --object_mesh test_model/CoMe_Inpainting360GS/output_napkin_basic/teatime_napkin/step15_object_mesh_merge/mesh_come_napkin_v2_iter1.ply --out_subdir step15_ablation_no_instantmesh
python src/tools_instantmesh/place_on_table.py --run outputs/LERF/teatime_cafe_box --object_mesh outputs/LERF/teatime_cafe_box/step13_come_mesh/mesh_come_direct_obj.ply --out_subdir step15_ablation_no_instantmesh
foreach ($r in @("room0_obj1","office0_obj1")) {
    python src/tools_instantmesh/place_on_table.py --run outputs/Replica/$r --object_mesh outputs/Replica/$r/step13_come_mesh/mesh_come_direct_obj.ply --out_subdir step15_ablation_no_instantmesh
}
```
> `test_model/` có thể bị xóa: **copy ngay** `mesh_come_napkin_v2_iter1.ply` (và output InFusion của `test_model/CoMe_Infusion` dùng cho Table 2) sang `outputs/LERF/teatime_napkin/baselines/` rồi trỏ lệnh vào đó.
> Lưu ý: `mesh_come_napkin_v2_iter1.ply` được trích theo quy trình cũ (`output_napkin_basic`), không chắc trùng `run_object_direct` ⇒ để nhất quán giữa 4 run, nên chạy luôn `run_object_direct` cho `teatime_napkin` (~2 phút).
- **Giả thuyết:** mesh hở (Watertight = No), mặt khuất/đáy thiếu ⇒ Object CD và F@1cm kém.

### Bước 2.3 — (c) w/o Sim(3)/ICP alignment
- Cờ `--no-sim3-icp` cho `place_on_table.py`: đặt mesh canonical InstantMesh vào tâm AABB của cụm Gaussian vật, scale = cạnh lớn nhất AABB, **không** xoay theo pháp tuyến bàn, **không** chọn view theo Silhouette IoU, **không** dịch theo tia BVH.
- **Chạy 100% trên CPU local** từ kết quả Step 13–14 đã có sẵn của `teatime_napkin`, `teatime_cafe_box`, `room0_obj1`, `office0_obj1`:
```powershell
python src/tools_instantmesh/place_on_table.py --run outputs/LERF/teatime_napkin --no-sim3-icp --out_subdir step15_ablation_no_align
python src/tools_instantmesh/place_on_table.py --run outputs/LERF/teatime_cafe_box --no-sim3-icp --out_subdir step15_ablation_no_align
python src/tools_instantmesh/place_on_table.py --run outputs/Replica/room0_obj1   --no-sim3-icp --out_subdir step15_ablation_no_align
python src/tools_instantmesh/place_on_table.py --run outputs/Replica/office0_obj1 --no-sim3-icp --out_subdir step15_ablation_no_align
```
> `--out_subdir` phải giữ nguyên `step15_object_mesh_merge/` của Ours (không ghi đè); kiểm tra `place_on_table.py` không bỏ qua do "đã có kết quả" (cờ `--force` hiện dựa trên thư mục mặc định).
- **Giả thuyết:** mesh vẫn kín nhưng lệch tư thế/tỉ lệ ⇒ Object CD tăng mạnh, vật xuyên/lơ lửng so với bàn.

> Tổng hợp: `scripts/eval_tables.py --table 3` đọc mesh cuối của từng biến thể (`step15_*`) + JSON của `place_on_table.py`, không cần `physics_summary.json`.

#### Table 3: Ablation (số minh họa mục tiêu; Object CD/F@1cm trên Replica — Ours & (b) trùng Table 1 (R35), còn lại trung bình 4 run)

| Configuration | Watertight? | Object CD ↓ (cm) | Object F@1cm ↑ (%) | Under-object RMSE ↓ (mm) | Static penetration ↓ (mm) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **(a) w/o CoMe** | Yes | 2.85 | 76.0 | 6.5 | 2.4 |
| **(b) w/o InstantMesh (CoMe direct mesh)** | No | 4.05 | 72.8 | 3.1 | 5.0 |
| **(c) w/o Sim(3)/ICP alignment** | Yes | 6.70 | 25.0 | 3.1 | 8.2 |
| **Ours (Full)** | **Yes** | **1.87** | **84.3** | **3.1** | **0.4** |

> Nếu muốn (a) chỉ chạy trên `obj1` (tiết kiệm), thì Object CD/F của **mọi** dòng Table 3 phải tính trên cùng tập `{room0_obj1, office0_obj1}` (khi đó Ours/(b) sẽ khác Table 1 — ghi chú rõ).

> Mô phỏng vật lý (Step 16: CoACD + MuJoCo, Penetration động / Sim Success / FPS, biến thể w/o CoACD) **hoãn lại** — làm sau nếu cần.

---

## 3. Lộ trình triển khai (Checklist)

### Pha 0: Dữ liệu GT Replica & Chuyển đổi COLMAP (bắt buộc trước Pha 3)
- [x] Tạo script `scripts/prepare_replica_colmap.py` tự động chuyển đổi từ render Replica (NICE-SLAM: `cam_params.json` + `traj.txt` + `mesh.ply`) sang định dạng COLMAP (`sparse/0/cameras.*`, `images.*`, `points3D.*` có màu RGB) và lấy mẫu 250 frame ảnh.
- [x] Chạy chuẩn bị dữ liệu cho 2 scene chính:
  ```powershell
  python scripts/prepare_replica_colmap.py --scenes office0 room0 --num-frames 250 --num-points 150000
  ```
  -> Đã tạo xong `F:/Replica 8 Scene/Replica_safegs/office0` và `room0` với 100% camera được đăng ký chuẩn xác, hệ tọa độ metric $1.0\text{ m}$ và khởi tạo SfM point cloud dày 150k điểm.
- [x] (2026-11-10, `scripts/prepare_replica_gt.py`: office0 obj1=bin #34, obj2=desk-organizer #44; room0 obj1=pot #43, obj2=candle #79; replica_v1 ≡ NICE-SLAM mesh, max|Δ|=0 ⇒ không cần ICP; thêm `gt_inst_masks/` 16-bit cho `6id --ref-masks`; lưu ý: `*_gt_capped.ply` chưa watertight vì mesh GT vật hở) Tách mesh GT 4 vật (2 vật × `room0`, `office0`) theo instance id từ `F:/Replica/Replica-Dataset/<scene>/habitat/mesh_semantic.ply` (`info_semantic.json`), kiểm tra trùng hệ tọa độ, đóng đáy (`*_gt_capped.ply`), lưu footprint.

### Pha 1: Code bổ sung
- [ ] **1.** `modal_come_train.py`: `--flags-preset {come,vanilla,gof}` **thay thế** `COME_FLAGS` (R4, R5); đưa qua `run_pipeline.py`.
- [ ] **2.** `run_pipeline.py`: `--reuse-from <run>` (copy Step 4–7, local + Volume), `--no-come-loss` (truyền `COME_LOSS=0` vào spec → env của container Step 11).
- [ ] **3.** `modal_come_mesh.py::run_object_direct` (dựng model tạm chỉ chứa Gaussian vật → `extract_mesh_tets.py`).
- [ ] **4.** `scripts/modal_baselines.py::{gof,milo}` (image Modal riêng cho GOF gốc và MILo) + `crop_by_prism.py` (tách nền/vật bằng prism chung).
- [ ] **5.** `place_on_table.py --no-sim3-icp`, `--object_mesh`, `--out_subdir`.
- [ ] ~~**6.** `src/tools_mesh/run_physics_eval.py` (CoACD + MuJoCo)~~ — **hoãn**, Table 3 hiện không cần mô phỏng vật lý.
- [ ] **7.** Hold-out 1/8 ảnh train có vật cho Inpaint360GS (dataset reader) + render composite z-buffer (Gaussian + mesh, `nvdiffrast`).
- [ ] **8.** `scripts/eval_tables.py` (Table 1/2/3 → `.md` + `.csv`), căn chỉnh Sim(3) Umeyama + ICP, CD/F-score/NC/Under-object completeness.
- [ ] **9.** Unit test nhanh trên `teatime_napkin` (đã có đủ 0–15): tính Table 3 dòng Ours trước khi tốn GPU.
- [ ] **10.** `select_target_id.py --write-yaml` (hoặc chép tay, R27) + script render mask instance GT Replica (`mesh_semantic.ply` + pose GT) cho `--ref-masks`.
- [ ] **11.** `pipeline_steps.make_spec`: `dk = come_key` khi có `--volume-key` (R28) — **chặn** Ablation (a).
- [ ] **12.** `modal_come_train.py`: `train` entrypoint nhận `--flags-preset`; thêm `::copy_colmap --src-key` (R31, R32); `upload_colmap` dùng được cho Inpaint360GS (R33).
- [ ] **13.** Copy các file cần từ `test_model/` (mesh khăn CoMe, output InFusion teatime) sang `outputs/.../baselines/` trước khi xóa `test_model/`.
- [x] **14.** Sửa comment đầu `configs/datasets/replica.yaml` (đang ghi "COLMAP by ::colmap", trái với R24/R26).

### Pha 2: Table 2 (Inpaint360GS & LERF)
- [ ] **Chặn:** Pha 1 mục 7 (hold-out 1/8 ảnh train có vật cho Inpaint360GS) xong **trước** mọi lệnh Step 0–2 / train baseline trên Inpaint360GS.
- [ ] Thử script Table 2 trên `teatime_napkin` và `teatime_cafe_box` (cả 2 đã có đủ kết quả Step 0–15).
- [ ] `ramen_main` run mới (0-2 → 4-7 → `6id` → 8-15); 3 run Inpaint360GS (0-2 → 4-7 → `6id` ×3 → 8-15; doppelherz kiểm tra lại ID 26/24,10).
- [ ] Baseline InFusion (tái sử dụng `test_model/CoMe_Infusion` cho `teatime_napkin`), Inpaint360GS gốc, CoMe + Direct Mesh → Table 2.

### Pha 3: Table 1 (Replica 2 scene × 2 vật)
- [ ] Upload COLMAP đã chuẩn bị sẵn từ Pha 0 bằng `scripts/modal_come_train.py::upload_colmap` cho `room0` và `office0` (bỏ qua chạy SfM trên Modal).
- [ ] Ours: Chạy S1–2 (Train CoMe 3DGS), S4–7 (+ `--reuse-from`) cho `room0`, `office0`.
- [ ] Chạy `6id` (`--ref-masks` mask GT, hoặc `--click`) cho 4 vật, **chép** `target_id` in ra vào `replica.yaml`, kiểm tra bằng `--steps 8-15 --dry-run`.
- [ ] Ours: Chạy tiếp S8–15 trên Modal cho 4 vật (`room0_obj1`, `room0_obj2`, `office0_obj1`, `office0_obj2`).
- [ ] Baseline: 3DGS+TSDF, GOF, MILo (train theo scene), CoMe Direct (theo run).
- [ ] Đánh giá các chỉ số hình học qua `scripts/eval_tables.py --table 1` → Table 1.

### Pha 4: Table 3 (Ablation)
- [ ] (a) train Vanilla `teatime` (còn `room0`, `office0` tái sử dụng từ Table 1) + pipeline `nocome` cho 4 run (`teatime_napkin_nocome`, `teatime_cafe_box_nocome`, `room0_obj1_nocome`, `office0_obj1_nocome`).
- [ ] (b), (c) — tái sử dụng kết quả: (b) dùng sẵn `mesh_come_napkin_v2_iter1.ply` + 2 mesh `CoMe Direct` của Replica, chỉ trích thêm 1 mesh vật trực tiếp cho `teatime_cafe_box` (Modal); còn lại chạy local.
- [ ] Metric hình học (Watertight, Object CD/F@1cm, Under-object RMSE, Static penetration) cho Ours + 3 biến thể → Table 3.

---

## 4. Ước lượng thời gian & chi phí

> [!IMPORTANT]
> **Quy định Giới hạn $30 Free Credits trên Modal (R25):**
> Mỗi tài khoản Modal có \$30 credits miễn phí. Trong quá trình chạy thực nghiệm:
> - Cần theo dõi lượng credits đã dùng trên workspace Modal (`modal profile current` hoặc trang quản lý Modal).
> - **Ngay khi hết $30 free credits trên workspace hiện tại: Agent PHẢI DỪNG LẠI và thông báo cho người dùng** để người dùng chuyển đổi sang workspace/profile khác (dùng lệnh `modal profile use <tên_profile>` hoặc cập nhật biến `MODAL_PROFILE` trong file `.env`).
> - Sau khi người dùng chuyển workspace xong mới tiếp tục thực hiện các lệnh tiếp theo.

### 4.1. Đơn giá Modal dùng để tính
Theo bảng giá Modal (giá theo giây, **kiểm tra lại trên modal.com/pricing trước khi chạy** vì có thể đã thay đổi): T4 ≈ $0.59/h, L4 ≈ $0.80/h, A10G ≈ $1.10/h, L40S ≈ $1.95/h; CPU ≈ $0.047/core·h; RAM ≈ $0.008/GiB·h. Modal tính cả CPU/RAM **đã xin**, nên chi phí thật theo cấu hình trong code:

| Hàm Modal (cấu hình trong code) | GPU + CPU + RAM | ≈ $/giờ |
| :--- | :--- | :---: |
| `colmap_scene`, `train_scene`, `render_scene` (L4, 8 core, 32 GiB) | 0.80 + 0.38 + 0.26 | **1.44** |
| `main_parts` S4–12 (L4, 8 core, 64 GiB) | 0.80 + 0.38 + 0.51 | **1.69** |
| `lama_part` S10 (T4, 4 core, 32 GiB) | 0.59 + 0.19 + 0.26 | **1.04** |
| `mesh_run` S13 (L40S, 16 core, 128 GiB) | 1.95 + 0.75 + 1.02 | **3.72** |
| `run_step14` InstantMesh (A10G/L4, 4 core, 32 GiB) | 1.10 + 0.19 + 0.26 | **1.55** |

> Gợi ý tiết kiệm: `mesh_run` xin 128 GiB RAM chiếm ~27% giá; nếu mesh Replica nhỏ hơn teatime (16M đỉnh), giảm còn 64 GiB.

### 4.2. Thời gian GPU cho một đơn vị công việc
(Từ README: COLMAP 20–40 phút, train 30k 1–2 giờ, render 5–10 phút, S13 5–10 phút compute; còn lại ước lượng, ±50%.)

| Đơn vị | Thời gian GPU | Chi phí |
| :--- | :---: | :---: |
| **Scene** (S0 + S1 + S2) | 1.5–2.8 h | $2.2–4.0 |
| **Scene** S4–7 (HQ-SAM, sync, distill) | 0.8–1.5 h | $1.4–2.5 |
| **Run** S8–12 (gồm LaMa) | 1.0–1.8 h | $1.7–3.0 |
| **Run** S13 (kể cả nạp model, ghi 16M đỉnh) | 0.25–0.5 h | $0.9–1.9 |
| **Run** S14 InstantMesh | 0.25–0.5 h | $0.4–0.8 |
| **Run** S13p/13b/14.1/15 (local CPU) | 10–30 phút | $0 |
| Build image lần đầu (mỗi image, CPU) | 20–40 phút | ~$0.2 |

⇒ **Ours, 1 scene mới + 1 vật ≈ $7–12, ~4–7 giờ GPU**; vật thứ 2 cùng scene (dùng `--reuse-from`) ≈ $3–6.

### 4.3. Chi phí theo bảng (Đã tận dụng `teatime_napkin` + `teatime_cafe_box` & khử trùng lặp Table 1 ↔ Table 3)

| Hạng mục | Khối lượng | Chi phí ước lượng |
| :--- | :--- | :---: |
| **Table 1 — Ours** | 2 scene × (S1–7, không S0) + 4 run × (S8–14) | $20–37 |
| Table 1 — 3DGS + TSDF | 2 train (~1 h) + 2 TSDF | $4–6 |
| Table 1 — GOF gốc | 2 train (1–2 h) + 2 mesh (L40S) | $6–12 |
| Table 1 — MILo | 2 train (1–1.5 h, L40S) + mesh + debug image | $10–20 |
| Table 1 — CoMe Direct | 4 × mesh vật (L40S, ~10 phút) | $2–4 |
| **Table 2 — Ours** | 3 run Inpaint360GS + `ramen_main` mới (`teatime_napkin` & `teatime_cafe_box` đã xong 0–15) | $28–45 |
| Table 2 — Inpaint360GS gốc | 6 run (train + grouping + inpaint, 2–3 h L4) | $20–35 |
| Table 2 — InFusion | 5 run mới (~1 h; `teatime_napkin` đã có sẵn ở `test_model/CoMe_Infusion`) | $8–13 |
| Table 2 — composite render / eval | 6 run × ~15 phút | $3–5 |
| **Table 3 — (a) w/o CoMe** | 1 scene train mới (`teatime`, S1–2), 3 scene × S4–7 (model vanilla) + 4 run S8–14 | $18–30 |
| Table 3 — (b), (c) | tái sử dụng + 1 mesh vật trực tiếp (`teatime_cafe_box`); đặt vật chạy local | $1–2 |
| **Tổng tính toán (chạy đúng 1 lần)** | | **≈ $122–212** |
| Dự phòng chạy lại / debug (×1.5) | | **≈ $180–320** |
| (Tùy chọn) Step 8–12 cho baseline Table 1 | 12 run (3 baseline × 4) | +$20–36 |
| (Tùy chọn) SPIn-NeRF chạy thật | 6 run × 4–8 h A100 | +$60–130 |

> Volume Modal: dữ liệu + output dự kiến 100–180 GB (teatime_napkin local đã 7.9 GB/run). Kiểm tra chính sách giá lưu trữ Volume hiện hành; tải về local chỉ những file cần cho eval (mesh, ảnh render, JSON), không tải checkpoint.

### 4.4. Thời gian lịch (1 người, GPU chạy song song bằng `--detach`)

| Pha | Nội dung | Thời gian người làm | Ghi chú |
| :--- | :--- | :---: | :--- |
| 0 | GT Replica (2 scene, 4 vật: tách vật, căn hệ tọa độ, đóng đáy) | 2–3 ngày | Rủi ro: lệch hệ tọa độ `replica_v1` ↔ NICE-SLAM |
| 1 | Code bổ sung (9 mục) | 8–12 ngày | MILo/GOF image Modal và render composite là phần lâu nhất |
| 2 | Table 2 | 4–6 ngày | 3 lần `6id` thủ công; kiểm tra Clean GT đúng tập vật |
| 3 | Table 1 | 3–5 ngày | 4 lần `6id` + kiểm tra mắt Step 9/11/14 từng run |
| 4 | Table 3 | 3–5 ngày | (a) train mới 1 scene (`teatime`), `room0/office0` dùng lại Baseline 1; S4–7 chạy lại cả 3 scene |
| — | Tổng hợp bảng, hình, viết | 3–4 ngày | |
| **Tổng** | | **≈ 23–35 ngày làm việc (5–7 tuần)** | Pha 2 có thể chạy song song với cuối Pha 1 |

**Đường găng (critical path):** Pha 0 → Pha 1 mục 8 (Sim(3) + metric) → Pha 3. Nên làm Pha 0 song song với Pha 1 mục 1–3 và chạy GPU Ours của Replica (S0–7) ngay khi preset & `--reuse-from` xong.

### 4.5. Rủi ro chính
1. **COLMAP trên Replica** (texture trơn, ánh sáng render đồng đều): Đã giải quyết triệt để ở Pha 0 bằng `scripts/prepare_replica_colmap.py` (chuyển đổi pose GT `traj.txt` + `cam_params.json` sang `sparse/0/` chuẩn COLMAP và lấy mẫu 150k điểm từ mesh). Kết quả 100% frame đăng ký thành công, Sim(3) là Identity ($scale = 1.0\text{ m}$), loại bỏ hoàn toàn rủi ro SfM bị fail.
2. **Vật nhỏ trên Replica** (độ phân giải 1200×680) → InstantMesh từ crop nhỏ chất lượng kém; ưu tiên vật ≥ 15 cm.
3. **Composite PSNR của Ours thấp hơn Inpaint360GS** là kỳ vọng hợp lý (R15) — cần lập luận trong paper bằng masked metrics + Table 1/3.
4. **MILo** có yêu cầu bộ nhớ/CUDA riêng; nếu build quá 2 ngày, cân nhắc dùng 2DGS hoặc PGSR làm baseline thay thế.
