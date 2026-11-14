# PLAN: Chạy Inpaint360GS trên LERF `teatime`, xóa tờ giấy/khăn napkin (ID 44) trên Modal GPU

Lập: 2026-10-24 · Viết lại: 2026-10-26 (folder local bị xóa, volume Modal vẫn còn)
Workspace Modal: **`nguyenkhacd10`** · Volume: **`inpaint360gs`** · GPU: **L4**
Paper: Inpaint360GS, WACV 2026 (arXiv 2511.06457)

---

## 0. Nguyên tắc

1. **Không đổi quy trình của paper/code gốc.** Chỉ gọi lại đúng các lệnh trong `run_seg.sh`, `run_remove.sh`, `run_inpaint.sh` (và `scripts/run_data_prepare.sh` cho dữ liệu tự có), đúng thứ tự, đúng tham số mặc định. Không sửa file gốc nào; mọi thứ thêm vào là file mới: `modal_inpaint360gs.py` và `tools_safe/`.
2. **Mỗi step lưu thành một folder riêng** `outputs/teatime_napkin/stepN_<tên>/`. Chạy lại một step thì tạo `_v2`, `_v3`, không ghi đè bản cũ.
3. **Chạy trên Modal `nguyenkhacd10`.** Script tự kiểm tra `modal profile current` và dừng nếu sai workspace.
4. **Dùng `modal run --detach`** cho các step GPU, để tắt máy local thì job trên Modal vẫn chạy tiếp.

## 1. Dữ liệu và các sai khác đã biết

| Mục | Ghi chú |
|---|---|
| Dữ liệu | `F:\LERF Datasets\teatime\teatime\images`: 180 ảnh 994×738, chỉ có `transforms.json` (nerfstudio), không có COLMAP |
| Nhánh code | `dataset_name="others"`, giống `kitchen`, vì không có ảnh `test_`. Không chạy metrics (script gốc cũng chỉ chạy cho `inpaint360`) |
| `resolution` | 2 (`images_2`, 494×365 sau undistort). Không dùng được 1, vì `raw_mask_sam.py` và `add_label_num_hqsam.py` đọc `images_{res}` |
| Mask SAM có sẵn `F:\LERF Datasets\teatime\teatime\mask` | **Không dùng được.** C1: do SAM2.1 tạo, không phải HQ-SAM; C3: 994×738, sai kích thước; C4: ID không liên tục trong từng frame. Đã chạy lại HQ-SAM đúng như paper |

**Paper và code phát hành lệch nhau.** Plan dùng giá trị mặc định của code (`run_*.sh` là quy trình chính thức):

| Tham số | Paper (Supp. B) | Code / config |
|---|---|---|
| Ngưỡng GS-IoU σ | 0.2 | `mask_associate.py`: 0.1 |
| λ distill | 0.0005 | `train_distill.json`: `reg3d_lambda_val: 2` (cùng 2000 vòng) |
| 3D inpaint λ1 / λ2 / số vòng | 0.2 / 0.005 / 2000 | `lambda_dssim 0.8`, `lambda_lpips 0.0005`, `finetune_iteration 5000` |
| Số bước tối ưu latent LaMa | 10 | `default.yaml`: `refiner.n_iters 15` |

**Môi trường khác `install.sh`** (thuật toán giữ nguyên):
- COLMAP chạy `--no_gpu`, vì bản cài từ apt không có CUDA.
- Build bằng gcc/g++, vì Python của Modal mặc định gọi clang++.
- `pip install . --no-build-isolation`, và `MMCV_WITH_OPS=0` (code không dùng mmcv).
- LaMa dùng Python 3.10 + torch 1.12.1+cu113, vì CUDA 10.2 không chạy được trên L4 và Modal không hỗ trợ Python dưới 3.10. Kèm sklearn 1.0.2, opencv 4.5.5.64.
- Weight CropFormer (HQ-SAM) nằm trong dataset HuggingFace có kiểm soát truy cập, nên không đóng vào image; được upload lên `/vol/weights/`.
- `PYTHONPATH`:
  - Step 2–4: repo + `seg/detectron2` (như `run_seg.sh`) + thư mục `CropFormer`, vì `criterion_view.py` import thẳng `mask2former`.
  - Step 5–7: thư mục gốc repo, vì `tools/virtual_pose.py` import `render`.

## 2. Lưu checkpoint định kỳ (Modal có thể tắt GPU bất cứ lúc nào)

1. **Commit volume định kỳ:** mỗi step chạy trong `PeriodicCommit`, tự commit mỗi 300 giây, khi nhận SIGTERM và khi kết thúc step.
2. **Checkpoint 3DGS (Step 1):** `--checkpoint_iterations 5000 … 25000` (tham số có sẵn, không đổi quá trình train), giữ 2 bản mới nhất. Khi chạy lại, script dùng `--start_checkpoint` với checkpoint mới nhất còn đọc được.
3. **Trạng thái từng step** nằm trong `/vol/state/stepN.json`. Modal retry (`retries=2`) sẽ chạy tiếp đúng run đó và đúng folder export. Step đã `done` thì bỏ qua; `--force` tạo `_vK` mới.
4. **Step ngắn** (HQ-SAM, distill, LaMa, 3D inpaint): chạy lại cả step. Riêng Step 6 chia thành `prepare` / `color` / `depth` / `collect`, phần nào xong rồi thì không chạy lại.

## 3. Các bước và trạng thái (cập nhật 2026-10-26)

`D=data/others/teatime`, `M=output/others/teatime` (trên volume).

| Step | Lệnh gốc | Trạng thái trên Modal | Kết quả |
|---|---|---|---|
| 0 Dữ liệu | `convert.py -s D --resize` (COLMAP) | ✅ done | 177/180 ảnh. Thiếu `frame_00148/155/156`; người dùng đồng ý chạy tiếp (2026-10-24) |
| 1 3DGS | `gaussian_splatting/train.py --eval -r 2` (30k vòng) | ✅ done | PSNR test 24,36 dB / train 32,94 dB; 154 view train, 23 view test |
| 2 Mask2D | `seg/raw_mask_sam.py --method hqsam` | ✅ done | `check_report.json`: không dùng được mask LERF, nên chạy lại HQ-SAM. Tối đa 101 vùng mỗi ảnh |
| 3 Sync mask | `seg/mask_associate.py` + `tools/add_label_num_hqsam.py` | ✅ done | 256 object (đúng trần của code). Code lưu ID dạng uint8 nên object có ID > 255 bị tràn; ảnh hưởng không đáng kể |
| 4 Gắn ID 3DGS | `seg/distillation.py` (2000 vòng) + `render.py --render_video` | ✅ done | `point_cloud/iteration_2000` + `classifier.pth`, render `objects_pred_color` |
| 5 Xóa vật + virtual camera | `init_configs.py --target_id 44` → `edit_object_removal.py` → `tools/virtual_pose.py` | ✅ done (lần 3, sau khi sửa `PYTHONPATH`); 30 view ảo | ID **44** = tờ giấy/khăn trắng trước con gấu (frame_00001), `surrounding = None`. Có mặt ở 105/177 view |
| 5c SAM-Track | `Segment-and-Track-Anything/app.py` (paper: tương tác) → **tự động hóa** theo yêu cầu người dùng (`::samt_auto_run --target-id 44`) | ✅ done (`samt_masks_v2`) | Xem mục 3b |
| 6 Inpaint 2D | `prepare_lama_data.py --inpaint2lama` → LaMa `predict_color/depth.py` → `prepare_lama_data.py` | ✅ done (`step6_inpaint2d_v3`) | LaMa màu + depth trên 30 view ảo của `step5_virtual_cam_v4` |
| 7 3DGS inpaint | `edit_object_removal_plyfusion.py` + `edit_object_inpaint.py --render_video` (qua wrapper bỏ view rỗng) | ✅ done (`step7_inpaint_3dgs`) | Train trên 24/30 view; bỏ 00018–00023; kết quả `point_cloud_object_inpaint_virtual/iteration_5000` |
| 8 Render | gom render/video của Step 7 | ✅ done (`step8_render`) | train/test renders + 3 video `final_video.mp4` |

### 3b. Mask SAM-Track tự động (2026-10-26, theo yêu cầu người dùng)

Mask vẫn do **chính SAM-Track tạo ra** (paper Eq. 6: M_i = SAMT(C_i)), dùng đúng các hàm `app.py` gọi (`SegTracker.seg_acc_click` → `img_seq_type_input_tracking`) với tham số mặc định của giao diện (r50_deaotl, sam_gap 1000, max_obj 255, points_per_side 16, fps 8). Chỉ khác ở chỗ **thao tác click tay được thay bằng click tự động**:
1. Click Positive tại tâm của ID 44 trên view ảo 0 (lấy từ `virtual/ours_2000/objects_pred`), tức pixel (112, 196).
2. **Tự động refine** (giống nút "Start tracking to refine"): ở frame đầu tiên mà vùng ID 44 nhìn thấy được nhưng mask phủ dưới 30%, click lại vào tâm ID 44 và track tiếp từ frame đó. Đã refine tại frame 24, pixel (97, 352).
3. **Lọc phần bám nhầm**: bỏ các mảnh mask không chạm vùng ID 44 (đã nới 15 px). Ở frame 18–23, tờ giấy nằm ngoài khung hình và tracker đã bám nhầm sang tai con cừu.

Cả bộ mask thô (`samt_masks_v2_raw_track`) lẫn ảnh overlay đều được lưu trong `step5_virtual_cam/`. Lần thử đầu, chỉ click mà không refine/lọc, nằm ở `samt_masks/` (để đối chiếu, không dùng).

### 3c. Sự cố và cách xử lý (2026-10-26)

| Sự cố | Nguyên nhân | Xử lý |
|---|---|---|
| Giao diện SAM-Track không mở | `app.py` cần video ví dụ `assets/blackswan.mp4`; ngoài ra `uvicorn`/`starlette`/`pydantic` bản mới không tương thích gradio 3.39 | Tạo sẵn `blackswan.mp4`; ghim `uvicorn==0.23.2 fastapi==0.104.1 pydantic==2.5.3` (layer cuối của image) |
| Mask SAM-Track làm tay bị mất | Bấm Start Tracking lần 2: `img_seq_type_input_tracking` xóa `images_masks/` trước, rồi lỗi | `samt_app` tự sao lưu mỗi khi đủ 30 mask; chuyển sang `samt_auto` |
| Step 6: `No module named saicinpainting.training.data` | Repo phát hành thiếu `LaMa/saicinpainting/training/data/` | Bổ sung 4 file từ `advimman/lama@786f593` (image LaMa). Phần đọc depth `.npy` của tác giả nằm ở `evaluation/data.py` (có sẵn, không đổi) |
| Step 7: `mask_to_bbox` IndexError | Một view ảo có mask rỗng. Camera ảo nhìn vào **tâm hội tụ của camera train** (`generate_ellipse_path`, `object_centered=False`), không nhìn vào vật; với `circle_radius` tự tính = 0,2356, tờ giấy ra khỏi khung hình ở frame 18–22 | Mô phỏng trên máy bằng chính `generate_ellipse_path`: chọn **`--circle_radius 0.6`** (tham số chính thức, `run_remove.sh` gợi ý 0.5). Kết quả: 0 view mất vật, lỗ chiếm 0,76–3,9% ảnh (paper: 1–50%). Chạy lại Step 5 → `step5_virtual_cam_v2`, Step 6 → `_v2` |

Bản đầu (`step5_virtual_cam`, `step6_inpaint2d`, với r = 0,2356) được giữ nguyên để đối chiếu.

**Cập nhật:** bán kính 0,6 (`step5_virtual_cam_v2`) và 0,8 (`_v3`) cũng thất bại. Với 0,6, camera đi xuyên qua con gấu ở frame 24–29; với 0,8, tờ giấy bị con gấu che ở frame 10 và 25–27. Ngoài ra, mô phỏng che khuất bằng point cloud không bắt được lông gấu (Gaussian lớn, mờ). Không bán kính nào cho cả 30 view đều thấy vật, vì camera ảo của code nhìn vào tâm cảnh chứ không nhìn vào vật.

**Quyết định của người dùng (2026-10-26):** giữ quỹ đạo của code (bán kính tự tính), và **bỏ qua các view ảo có mask rỗng khi train Step 7**. Cách làm:
- Wrapper `edit_object_inpaint_skip_empty.py` nhúng trong `modal_inpaint360gs.py` (`SKIP_EMPTY_WRAPPER`), lưu kèm trong folder export của Step 7.
- Wrapper import `edit_object_inpaint.py`, thay `finetune_inpaint` bằng bản chỉ nhận các view có `objects > 128`, rồi chạy nguyên văn khối `__main__` của file gốc (lấy ra bằng `ast`).
- Danh sách view được dùng và bị bỏ được ghi vào `skipped_virtual_views.json`.

Chuỗi `chain_5_to_8` chạy hoàn toàn trên Modal: Step 5 (`_v4`, bán kính tự tính) → mask tự động → Step 6 (`_v3`) → Step 7 → Step 8.

### Điểm dừng cần người dùng
1. ✅ Chọn ID vật cần xóa: **44** (đã chọn).
2. ✅ Mask Segment-and-Track: đã tự động hóa (mục 3b), không cần thao tác tay nữa.

## 4. Output local

Đã dọn lại (2026-10-26): chỉ giữ kết quả cuối của từng bước, 4,4 GB. Mô tả từng folder nằm trong `outputs/teatime_napkin/README.md`.
```
F:\Safe-GS\Based_Model\Inpaint360GS\outputs\teatime_napkin\
  README.md  logs\
  step0_colmap\  step1_3dgs\  step2_mask2d_hqsam\  step3_sync_mask\  step4_id_3dgs\
  step5_removal_virtual_cam\  step6_inpaint2d_lama\  step7_inpaint_3dgs\  step8_render\{train,test}\{0_gt,1_removed,2_inpainted}
```
Dữ liệu trung gian chỉ còn trên Modal volume (`export/teatime_napkin/stepN_*`, tên folder kiểu cũ). Lưu ý: `::download` tải về theo tên cũ, nên đừng chạy nó vào folder đã dọn này.

Công cụ xem ID (chỉ chạy local, không ảnh hưởng pipeline): `tools_safe/label_color_ids.py` ghi số ID lên ảnh màu (`associated_hqsam_color`, `objects_pred_color`), cỡ chữ 0.3.

## 5. File

- `modal_inpaint360gs.py`: Modal app (2 image, các hàm step, entrypoint `upload`, `upload_weight`, `run_step`, `accept_step0`, `samt`, `samt_stop`, `status`, `download`).
- `tools_safe/check_sam_masks.py`: kiểm tra mask SAM có sẵn (C1–C4). File này nằm trong image, nên giữ nguyên nội dung để không phải build lại.
- `tools_safe/label_color_ids.py`: ghi số ID lên ảnh màu (chỉ chạy local).

## 6. Việc tiếp theo

1. Chạy tiếp Step 5 (resume): `modal run --detach modal_inpaint360gs.py::run_step --step 5 --target-id 44 --surrounding-id None`.
2. Tải lại Step 0–5 về `outputs/teatime_napkin/`.
3. Mở SAM-Track (`::samt`) → người dùng vẽ mask → `::samt_stop`.
4. Step 6 → 7 → 8 → tải về.
