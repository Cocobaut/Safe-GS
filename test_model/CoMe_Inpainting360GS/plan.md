# PLAN: Inpaint360GS + rasterizer/loss CoMe trên LERF `teatime`, xóa khăn giấy

Lập: 2026-10-27 · Workspace Modal: **`nguyenkhacd10`** · GPU: **L4** · luôn chạy `modal run --detach`

Mục tiêu:
1. Copy code **Inpaint360GS** và **rasterizer CoMe** sang `F:\Safe-GS\test_model\CoMe_Inpainting360GS`. Bản gốc (`Based_Model/Inpaint360GS`, `Based_Model/CoMe`, `test_model/CoMe_Infusion`) giữ nguyên.
2. Sửa các sai sót của lần chạy Inpaint360GS trước (`Based_Model/Inpaint360GS/outputs/teatime_napkin`) bằng cách **chọn lại bước tốt nhất**, không train lại phần đã có.
3. **Thay rasterizer** của Inpaint360GS bằng rasterizer CoMe, và **thêm loss CoMe** vào bước train lại 3DGS (fine-tune inpaint).
4. Dùng lại các kết quả có sẵn: COLMAP + 3DGS CoMe, ảnh render CoMe. Mask 2D **chạy lại bằng HQ-SAM** như paper (mask SAM 2.1 có sẵn chỉ dùng để tự chọn ID khăn).
5. Mỗi step một thư mục trong `outputs/`; các step **dùng lại** được đánh số trước.

---

## 1. Rà lại các sai sót của lần chạy Inpaint360GS trước và chọn bước tốt nhất

| # | Sai sót / điểm yếu ở lần trước | Nguyên nhân | Chọn lại (không train lại) |
|---|---|---|---|
| 1 | COLMAP chỉ đăng ký 177/180 ảnh; ảnh đã undistort và hạ xuống `images_2` (494×365) | `convert.py` của repo; `raw_mask_sam.py` không chạy được với resolution 1 | Dùng **COLMAP + ảnh của CoMe** (`teatime_come_step2/colmap`): cũng 177 ảnh nhưng PINHOLE 994×738, principal point đã đặt đúng tâm, **độ phân giải đầy đủ** (`--resolution 1`) |
| 2 | 3DGS vanilla tự train (PSNR test 24,36 dB), depth nhiễu nên vùng inpaint bị mờ và lơ lửng | 3DGS gốc không có ràng buộc hình học | Dùng **3DGS đã train của CoMe** (`come/point_cloud/iteration_30000`, 1,8 triệu Gaussian, có loss depth/normal/variance/distortion). Không train lại Step 1 |
| 3 | Mask HQ-SAM + Key Object Management sinh 256 object (chạm trần), ID uint8 bị tràn; mask chạy trên `images_2` đã undistort nên không khớp ảnh CoMe | Cảnh có nhiều vật nhỏ; COLMAP khác | **Chạy lại HQ-SAM (CropFormer) + `mask_associate.py` đúng như paper**, nhưng trên **ảnh CoMe 994×738** và **3DGS CoMe** (association chỉ chiếu tâm Gaussian, không cần rasterizer). Tham số mặc định của code. Kiểm tra số object; nếu vẫn chạm 256, ghi nhận (giới hạn của code) và kiểm tra ID khăn không bị trùng. **ID khăn chọn tự động**: ID có độ trùng (IoU) lớn nhất với mask khăn đã kiểm chứng của `CoMe_Infusion/Step_1_Sync_ID` (ID 11, đã đối chiếu bằng điểm 3D COLMAP); kết quả được lưu ra để người dùng duyệt trên ảnh `ids_on_images` trước Step 7 |
| 4 | Camera ảo nhìn vào **tâm cảnh** (`focus_point_fn`), không nhìn vào vật; khăn ra khỏi khung hình ở view 18–23, nên Step 7 crash | `tools/virtual_pose.py` gọi `generate_ellipse_path` với `object_centered=False` | Theo đúng paper (Sec. 3.4: *"trajectory centered on the removed object"*): **đặt tâm quỹ đạo vào tâm vật bị xóa** (trung bình các Gaussian của vật, lấy từ `point_cloud_<id>.ply`), dùng thống nhất cho `virtual_pose`, `plyfusion`, `edit_object_inpaint` |
| 5 | Step 7 phải bỏ 6/30 view có mask rỗng | Hệ quả của #4 | Với #4 sẽ không còn view rỗng. Vẫn giữ **bộ lọc view rỗng** làm lưới an toàn |
| 6 | Mask SAM-Track: click tay dễ mất (bấm Start Tracking lần 2 xóa mask); tracker bám nhầm sang tai cừu | Thiết kế của `app.py`; vật ra khỏi khung hình | Dùng **mask SAM-Track tự động** (click tại tâm ID khăn trên view ảo 0, tự refine khi mất dấu, lọc mảnh không chạm vùng vật). Với #4, vật luôn trong khung hình nên ít phải refine |
| 7 | LaMa lấp mảng nâu sẫm ở view sát con gấu | Mask hẹp (nới 10 px) nên LaMa có ít ngữ cảnh | Giữ `expand_pixels = 10` (mặc định) cho lần chạy chính. Nếu vẫn sẫm: chạy bản `_v2` với 20 px (README FAQ cho phép) |
| 8 | Repo phát hành thiếu `LaMa/saicinpainting/training/data/` | Bị `.gitignore` | Bổ sung 4 file từ `advimman/lama@786f593` (đã kiểm chứng ở lần trước) |
| 9 | Lỗi môi trường: `PYTHONPATH` (`render`, `mask2former`), gradio/uvicorn/fastapi/pydantic, Python 3.9 của LaMa | Repo không ghi đủ môi trường | Dùng lại đúng các bản sửa trong `Based_Model/Inpaint360GS/modal_inpaint360gs.py` (image SAM-Track và LaMa đã build, có cache) |

Không đổi: tham số mặc định của code (distill 2000 vòng, finetune 5000 vòng, `removal_thresh 0.7`, LaMa refine 15 bước, 30 view ảo).

---

## 2. Copy code (script `copy_code.py`, chạy local một lần)

```
CoMe_Inpainting360GS/
  plan.md, copy_code.py, modal_come_inpaint360gs.py
  src/                          # = bản copy Inpaint360GS + rasterizer CoMe; sửa code TRỰC TIẾP ở đây (code gốc đã nằm ở Based_Model)
  outputs/teatime_napkin/       # kết quả từng step
```

| Nguồn | Đích trong `src/` | Ghi chú |
|---|---|---|
| `Based_Model/Inpaint360GS/` (trừ `outputs`, `data`, `output`, `Paper`, `media`, `SIBR_viewers`, docs/test của glm, `seg/Entity`, `seg/detectron2/{docs,tests}`, `__pycache__`, `build`) | `src/` | Toàn bộ code Inpaint360GS |
| `test_model/CoMe_Infusion/src/gaussian_splatting/submodules/diff-gaussian-rasterization-come` | `src/submodules/diff-gaussian-rasterization-come` | Rasterizer CoMe (StopThePop, 13 kênh đầu ra), đã kiểm chứng render trùng từng pixel với CoMe |
| `test_model/CoMe_Infusion/src/gaussian_splatting/submodules/simple-knn` | `src/submodules/simple-knn-come` | Build cùng torch 2.4.1 |
| `Based_Model/CoMe/utils/depth_utils.py` | `src/utils/come_depth_utils.py` | `depth_to_normal`, `central_diff` cho loss CoMe |
| `src/scene/gaussian_model.py` (gốc) | chép thêm thành `src/scene/gaussian_model_inpaint360.py` | Giữ nguyên làm lớp cha |
| LaMa `training/data` (4 file) | `src/LaMa/saicinpainting/training/data/` | Tải từ GitHub, pin commit |

Script kiểm tra sau khi copy: đếm file, so khớp từng byte phần đã copy. Copy chỉ chạy **một lần**: nếu `src/` đã có thì script dừng, không ghi đè code đã sửa. Muốn biết đã sửa gì: so `src/` với `Based_Model/Inpaint360GS`.

---

## 3. Thay rasterizer + thêm loss CoMe (sửa trực tiếp trong `src/`)

| File | Thay đổi |
|---|---|
| `src/gaussian_renderer/__init__.py` | `render()` dùng **rasterizer CoMe** (`ExtendedSettings` từ `come/config.json` qua biến môi trường `COME_SPLATTING_CONFIG`, `inv_viewprojmatrix` tính lại mỗi lần vì camera ảo bị sửa trực tiếp, `filter_3d`, `confidences`). Giữ đúng chữ ký và các key trả về của Inpaint360GS (`render`, `radii`, `visibility_filter`, `viewspace_points`, `depth_3dgs`), thêm `alpha` và `rendered_all` (13 kênh). **Feature ID 16 chiều** (`render_object`) được render bằng cùng rasterizer qua `colors_precomp`, 6 lượt × 3 kênh, cùng trọng số blend với RGB, **chỉ tính khi được đọc** (distill/ảnh ID). Không phải sửa CUDA. Sau mỗi `Scene(...)`, gắn camera train vào model để tính lại `filter_3D` giống CoMe |
| `src/scene/gaussian_model.py` | Lớp con của `GaussianModel` gốc: thêm `confidence`, `filter_3D`; `load_ply`/`save_ply` đọc/ghi thêm 2 thuộc tính này (ply ra vẫn mở được bằng CoMe); `prune_points` cắt đồng bộ. Khi số điểm đổi (`inpaint_setup`, densify), renderer tự tính lại `filter_3D` |
| `src/utils/come_loss.py` (mới) | `come_regularization()`: depth-normal 0.05, distortion 100, smoothness 0.01, opacity-field 0.004, extent 0.1, variance 0.5, normal-variance 0.005, đúng `mesh_args` của lần train CoMe Step 2. Không dùng appearance embedding (view ảo không có) |
| `src/edit_object_inpaint.py` | `loss = <loss gốc> + come_regularization(...)` (tắt được bằng `COME_LOSS=0`); lọc view có mask rỗng ở đầu `finetune_inpaint` |
| `src/utils/pose_utils.py` | `center = focus_point_fn(poses)` → tâm vật nếu có biến môi trường `INPAINT360_OBJECT_CENTER` (tọa độ world, đổi sang hệ PCA bằng `transform`); không có thì giữ hành vi gốc |

Loss CoMe chỉ thêm vào **bước train lại 3DGS** (Step 7, fine-tune vùng inpaint). Distill ID (Step 4) chỉ tối ưu `obj_dc`, hình học CoMe đứng yên, nên không cần loss hình học.

---

## 4. Các bước và thư mục `outputs/teatime_napkin/`

Các step dùng lại (đánh số trước, copy từ `test_model/CoMe_Infusion`):

| Thư mục | Nguồn | Nội dung |
|---|---|---|
| `step0_reuse_colmap/` | `Step_2_Output_train_3DGS/teatime_come_step2/colmap` | `images/` (177 jpg 994×738), `sparse/0` |
| `step1_reuse_come_3dgs/` | `.../teatime_come_step2/come` (bỏ `chkpnt29000.pth`) | `point_cloud/iteration_30000/point_cloud.ply`, `config.json`, `cfg_args`, `cameras.json` |
| `step2_reuse_come_renders/` | `Step_3_Output_render_3DGS` | 177 ảnh render CoMe (ảnh đối chiếu cho kiểm tra renderer) |
| `step3_reuse_napkin_reference/` | `Step_1_Sync_ID/labels` (chỉ pixel ID 11) | Mask khăn đã kiểm chứng (0/255), **chỉ dùng để tự chọn ID khăn** sau HQ-SAM, không dùng để train |

Các step chạy mới trên Modal:

| Thư mục | Bước | Lệnh (code trong `src/`) | Kiểm tra đạt |
|---|---|---|---|
| `step4_renderer_check/` | Render 3DGS CoMe bằng renderer mới trên 177 view | script kiểm tra | PSNR so với `step2_reuse_come_renders` ≥ 35 dB (kỳ vọng trùng gần như từng pixel) |
| `step5_mask2d_hqsam/` | Mask 2D HQ-SAM trên ảnh CoMe | `seg/raw_mask_sam.py --dataset_path <data> --scene_name teatime --image_folder images --method hqsam` (image cũ có detectron2 + weight CropFormer trên volume) | 177 mask, ID liên tục trong từng frame |
| `step6_sync_mask/` | Đồng bộ ID đa view (Key Object Management) trên 3DGS CoMe | `seg/mask_associate.py --source_path <data> --model_path <3dgs CoMe> --resolution 1 --mask_generator hqsam --eval` → `tools/add_label_num_hqsam.py --resolution 1` (`images_1` = link tới `images`, chỉ là môi trường) | `scene.json` (số object); `napkin_id.json` = ID trùng nhiều nhất với mask khăn ở step3, kèm IoU từng frame |
| `step7_id_distill/` | Gắn ID vào Gaussian CoMe | `seg/distillation.py --resolution 1 --object_path associated_hqsam --eval` (+ `render.py`, ảnh ID có ghi số) | ID khăn phủ đúng khăn trên `objects_pred_color` |
| `step8_removal_virtual_cam/` | Xóa ID khăn, dựng 30 view ảo quanh tâm vật | `init_configs.py --target_id <napkin_id>` → `edit_object_removal.py` → tính tâm vật → `virtual_pose.py` | Vật nằm trong khung hình ở cả 30 view |
| `step9_inpaint_masks_samt/` | Mask vùng cần inpaint trên view ảo | SAM-Track tự động (image SAM-Track cũ) | 30/30 mask khác rỗng, 1–50% ảnh |
| `step10_inpaint2d_lama/` | LaMa inpaint màu + depth | `prepare_lama_data` → `predict_color/depth` (image LaMa cũ) → `prepare_lama_data` | Không có mảng sẫm lớn |
| `step11_inpaint_3dgs_come_loss/` | Train lại 3DGS vùng inpaint với loss Inpaint360GS + CoMe | `edit_object_removal_plyfusion.py` → `edit_object_inpaint.py --resolution 1 --render_video` | Loss hội tụ; ply có `confidence`, `filter_3D` |
| `step12_render/` | Render train/test + video | do `--render_video` tạo | So với lần trước: vùng khăn sạch hơn, không mờ |
| `step13_come_mesh/` | Trích mesh cảnh đã inpaint (không còn khăn) bằng CoMe, cắt riêng mặt bàn | `modal_come_mesh.py` (`extract_mesh_tets.py` gốc) → `tools_local/crop_table.py` (lăng trụ của `CoMe_Infusion/Step_6_Mesh_GOF`) | Vùng dưới khăn phủ mặt bàn ≥ 90 %, không lỗ (`tools_local/eval_under_napkin.py`) |
| `step14_object_closed_3dgs/` | Từ 3DGS của vật (ID khăn), **dự đoán 3DGS phía dưới vật** (đáy + vách) để vật thành khối kín | `tools_mesh/close_object_gaussians.py` (mục 4b) | Đáy nằm trên mặt đỡ (lệch ≤ 0,5 % kích thước vật), chân vật phủ ≥ 98 % |
| `step15_object_mesh_merge/` | Trích mesh vật bằng CoMe, cắt theo mặt đỡ, đóng kín, **gộp với mesh bàn** cho engine vật lý | `modal_come_mesh.py --variants napkin` → `tools_mesh/solidify_and_merge.py` (mục 4b) | Mesh vật kín (watertight), không xuyên vào bàn, mặt bàn dưới vật liền |

Mỗi thư mục chỉ giữ kết quả cuối (giống cách đã dọn `Inpaint360GS/outputs`), kèm `logs/<thư mục>.log` và `README.md` tổng hợp.

### 4b. Step 14–15: mesh vật kín + mesh bàn đã lấp, gộp lại cho engine vật lý

**Mục tiêu.** Engine vật lý cần **hai vật thể riêng** nằm đúng chỗ: (1) mặt bàn **đã lấp phía dưới vật** (Step 13, v2: phủ 94 % vùng dưới khăn, không lõm, không gồ) và (2) **bản thân vật** dưới dạng khối kín (watertight) đặt lên mặt bàn, không xuyên, không hở.

**Vì sao không làm như `CoMe_Infusion`.** Ở đó 3DGS khăn được ghép lại vào cảnh đã lấp (`Step_5_Combine_3DGS`) rồi trích mesh GOF một lần. Bộ trích GOF/CoMe coi một điểm là "bên trong" nếu nó bị che ở **bất kỳ** view nào (độ mờ = 1 − min theo view của transmittance), nên khoảng giữa mặt giấy và mặt bàn bị lấp đặc: khăn và bàn **dính thành một khối**. Đo được: 51 % ô dưới khăn không có mặt bàn, chỉ còn mặt trên của khăn, khăn không tách ra được (`eval_under_napkin_gof_v1_v2`). Ở đây 3DGS vật đã tách sẵn theo ID (gom kiểu Gaussian Grouping: `classifier` + `removal_setup`, ra `point_cloud_object_removal/iteration_2000/point_cloud_44.ply`, 3 734 Gaussian), nên trích **riêng** mesh vật rồi gộp ở mức mesh.

**Vấn đề của vật tách riêng.** Gaussian của khăn chỉ phủ **mặt nhìn thấy** (mặt trên, mép). Mặt dưới (tiếp xúc bàn) chưa từng được nhìn thấy nên không có Gaussian. Nếu trích mesh thẳng từ 3DGS vật, theo quy tắc "bị che ở một view là bên trong" ở trên, vùng dưới mặt giấy cũng thành "bên trong" và **kéo dài xuống dưới mặt bàn** tới hết hộp bao các Gaussian. Vì vậy phải (a) dự đoán phần dưới của vật và (b) chặn bằng **mặt đỡ** (support surface).

#### Step 14: dự đoán 3DGS phía dưới vật (`tools_mesh/close_object_gaussians.py`)

Đầu vào: `point_cloud_44.ply` (định dạng CoMe), mặt đỡ, camera train. Mọi độ cao đo theo pháp tuyến mặt đỡ.

1. **Làm sạch Gaussian vật.** Giữ Gaussian có `prob(ID 44) > 0,7` (đúng như `removal_setup`), opacity > 0,1, độ cao trong [−0,5 %, 30 %] kích thước vật so với mặt đỡ; DBSCAN trên tâm, giữ cụm lớn nhất (bỏ điểm lạc của bao lồi).
2. **Mặt đỡ** `S(u,v)`: với vật trên mặt bàn phẳng là **mặt phẳng bàn** (`mesh_gof_table_crop.json`, đã kiểm chứng: vân gỗ khớp tốt nhất đúng ở mặt phẳng, lệch ≤ 1 cm). **Tổng quát cho nền không phẳng:** `S(u,v)` = độ cao bề mặt nền **sau inpaint** ngay dưới vật, lấy bằng bắn tia theo trọng lực xuống mesh Step 13 (hoặc độ sâu render cảnh đã xóa vật). Không dùng giả định phẳng ở các bước sau.
3. **Bản đồ độ cao mặt trên** trên lưới (ô = kích thước vật / 100) trong hệ toạ độ mặt đỡ: `H_top(u,v)` = độ cao lớn nhất có trọng số opacity của các Gaussian chiếu vào ô; **chân vật** `F` = các ô có Gaussian, đóng hình thái (close 3×3) và lấp lỗ.
4. **Gaussian đáy:** mỗi ô trong `F` một Gaussian dẹt nằm tại `S(u,v) + ε` (ε = 0,3 % kích thước vật, để không chạm đúng mặt bàn), scale trong mặt = kích thước ô, scale theo pháp tuyến = ε/2, opacity 0,99, màu = màu trung bình mặt trên của vật (mặt dưới không có quan sát), `obj_dc` = đặc trưng ID của vật, `confidence` = 0.
5. **Gaussian vách:** dọc biên của `F`, các cột Gaussian dẹt đứng (pháp tuyến hướng ra ngoài biên) từ `S + ε` lên tới `H_top` của ô biên, bước = kích thước ô. Bên trong không cần lấp: bộ trích CoMe tự coi khoảng bị che giữa mặt trên và đáy là "bên trong".
6. Lưu `point_cloud_44_closed.ply` (định dạng CoMe: thêm `confidence`, `filter_3D` tính lại từ camera train). Kiểm tra: đáy lệch mặt đỡ ≤ 0,5 % kích thước vật; chân vật phủ ≥ 98 % ô có Gaussian gốc; render 177 view của vật đóng kín so với vật gốc chỉ khác ở vùng không nhìn thấy (PSNR mặt trên ≥ 35 dB).

Giới hạn: đáy là **dự đoán hình học** (vật đặt trên mặt đỡ, đặc từ mặt trên xuống đáy). Đúng cho vật nằm/đặt trên bề mặt (khăn, hộp, cốc úp…); với vật có phần rỗng bên dưới (ghế, quai cốc) sẽ bị lấp đặc. Đủ cho va chạm vật rắn, không đủ cho vật biến dạng (khăn thật là tấm mỏng).

#### Step 15: trích mesh vật bằng CoMe + gộp mesh (`modal_come_mesh.py --variants napkin`, `tools_mesh/solidify_and_merge.py`)

1. **Trích mesh vật** bằng `extract_mesh_tets.py` gốc của CoMe trên `point_cloud_44_closed.ply` (thư mục model kiểu CoMe như Step 13, cùng 154 camera train) → `mesh_napkin_raw.ply`.
2. **Cắt theo mặt đỡ:** bỏ phần dưới `S + ε`, cắt tam giác cắt ngang mặt đỡ (không chỉ bỏ cả tam giác) → biên đáy khép kín.
3. **Đóng kín:** nắp đáy = đa giác biên đáy chiếu lên mặt đỡ, tam giác hoá có ràng buộc; bỏ thành phần rời < 1 % số mặt; vá lỗ nhỏ còn lại. Kiểm tra `is_watertight`, thể tích dương, không tự cắt. **Dự phòng** nếu mesh vật có lỗ lớn: dựng thẳng từ `H_top` (mặt trên = mesh CoMe cắt theo `F`, vách = đùn biên xuống `S + ε`, đáy = nắp phẳng) — không cần Step 14.
4. **Gộp với mesh bàn:** mesh bàn = `mesh_come_v2_table.ply` (Step 13 v2, hoặc v3 nếu tốt hơn) đã dọn mảnh rác và giảm lưới (100–200 nghìn mặt); mesh vật giảm lưới (≤ 20 nghìn mặt). Hai vật thể **giữ riêng**:
   - `scene_table_napkin.ply`: cả hai trong một file để xem / render;
   - `collision/table_static.obj` (tĩnh, mặt hở được), `collision/napkin_dynamic.obj` + tách khối lồi (CoACD / V-HACD) nếu engine yêu cầu khối lồi;
   - `physics_meta.json`: mặt đỡ, hướng trọng lực (= −pháp tuyến bàn), hệ số đổi đơn vị COLMAP → mét (ước lượng từ kích thước vật thật), khe hở tiếp xúc ε.
5. **Kiểm tra gộp:** không đỉnh nào của vật nằm dưới mặt bàn quá ε (khoảng cách có dấu tới mesh bàn); mặt bàn dưới chân vật phủ ≥ 90 % (đúng kết quả Step 13); vật kín; tâm khối nằm trong chân vật (đứng vững khi thả trọng lực).

Thời gian: Step 14 vài phút CPU; Step 15 ≈ 3 phút GPU (vật nhỏ) + vài phút CPU.

#### Hiện thực cho v2 (2026-10-28, `src_v2/tools_mesh/`, kết quả `output_v2/teatime_napkin/step14_*`, `step15_*`)

Kiểm tra plan với v2: dùng được nguyên (v2 dùng lại step 0–9 nên có `removed_object_id44.ply` + `classifier.pth`; mesh bàn v2 = `step13_come_mesh/mesh_come_v2_table.ply` làm mặt đỡ). Các điểm **lệch plan** khi chạy:

1. **Kiểm tra Step 14:** bỏ "render 177 view ≥ 35 dB" (cần GPU) → kiểm tra hình học (chân vật phủ ≥ 98 % ô có Gaussian gốc, đáy = mặt đỡ + ε).
2. **Trích mesh vật bằng CoMe chỉ với camera thấy trọn vật** (87/177 ảnh, `napkin_views.json`, dữ liệu con `/vol/come360/data/teatime_napkin_views`). Lý do: code CUDA của CoMe gọi `identifyPixelRanges<<<(num_integrated+255)/256>>>` không chặn `num_integrated = 0`; với 3DGS chỉ có vật, view không thấy vật có 0 điểm → `invalid configuration argument`. Đã thử "Gaussian neo" trước mỗi camera (`add_view_anchors.py`): qua được lượt đánh giá đầu nhưng vẫn lỗi ở bước binary search (điểm giữa chỉ nằm quanh vật) → bỏ, dùng tập camera con. Code CoMe giữ nguyên.
3. **Đóng kín bằng lưới độ cao** (phương án dự phòng của plan thành phương án chính): mesh CoMe thô kín nhưng **9 % đỉnh lọt dưới mặt đỡ, sâu tới 14,6 cm** (vùng bị lớp đáy che coi là "bên trong"); vá đáy theo vòng cắt không bền → mặt trên = tia xuống mesh CoMe, đáy = mặt đỡ + ε, vách ở biên. Mất phần nếp gấp chìa ra (lấp đặc) — đủ cho vật rắn.
4. **Mặt đỡ đo lại trên mesh bàn va chạm** (đã giảm lưới), lấy **max** trong ô và ở góc ô → không xuyên; lần 1 (trung vị trên mesh bàn gốc) có 298 đỉnh xuyên tới 1,1 cm (`_attempt1_penetration/`), lần 2 góc ô bị lộn (đáy > đỉnh) → không kín (`_attempt2_not_watertight/`).
5. **Mesh va chạm** = lưới độ cao ô thô (0,85 cm) thay vì quadric decimation (làm hở khối). Không có CoACD/V-HACD → xuất bao lồi.

Kết quả: Step 14: 3 734 → 2 975 Gaussian khăn (loại 409 có P(ID 44) ≤ 0,7, phần lớn là Gaussian mặt bàn dưới khăn bị bao lồi gom vào), chân khăn 6 295 ô, tia trúng mesh bàn 99,8 %, + 6 295 Gaussian đáy + 2 483 Gaussian vách. Step 15: mesh CoMe khăn 252 096 mặt (0,2 phút); khối kín chi tiết 102 248 mặt (thể tích 0,0393), khối va chạm 25 972 mặt (0,0389), cả hai watertight, trọng tâm trong chân khăn, **0 đỉnh xuyên bàn**, khe đáy–bàn trung vị 0,42 / 0,49 cm; bàn bỏ 415 mảnh rời, giảm còn 300 000 mặt. File: `mesh_napkin_closed.ply`, `scene_table_napkin.ply`, `collision/{table_static,napkin_dynamic,napkin_convex_hull}.obj`, `physics_meta.json`, `cross_sections.png`.

### Bước cuối: tải kết quả về `F:\Safe-GS\test_model\CoMe_Inpainting360GS\outputs` và đặt tên lại

Chạy local sau khi Step 12 (hoặc 13) xong: `python -m modal run modal_come_inpaint360gs.py::download` (entrypoint chỉ đọc volume, không tốn GPU).

1. **Tải:** `modal volume get inpaint360gs come360/export/<step>` cho từng step vào thư mục tạm `outputs/_download/` (tải song song). So kích thước từng file với volume; file thiếu hoặc lỗi thì tải lại. Không ghi đè file đã có ([[no-overwrite-outputs]]): nếu `outputs/teatime_napkin` đã tồn tại thì ghi vào `teatime_napkin_v2`, `_v3`…
2. **Giữ kết quả cuối, bỏ phần trung gian:** depth `.npy` và `fused_*_ply` theo từng view, frame PNG của video (giữ `.mp4`), `gt_objects_color`, `objects_pred` dạng số thô, checkpoint, `state.json`, ảnh đầu vào đã chép sang LaMa, `images.zip`, mask SAM-Track thô. Phần này vẫn còn trên volume.
3. **Đặt tên lại**: thư mục con đánh số theo thứ tự chạy, tên file nói rõ nội dung:

```
outputs/teatime_napkin/
  README.md                         # bảng: step | thư mục | script gốc | lệnh | kết quả kiểm tra | điểm lệch
  logs/stepN_<tên>.log
  step0_reuse_colmap/               sparse/0, registered_images.txt
  step1_reuse_come_3dgs/            point_cloud_come_iter30000.ply, config.json, cameras.json
  step2_reuse_come_renders/         renders/ (177)
  step3_reuse_napkin_reference/     napkin_mask_id11/ (177)
  step4_renderer_check/             summary.json, per_view_psnr.json, 1_come_render/, 2_new_renderer/ (vài view mẫu)
  step5_mask2d_hqsam/               1_masks/, 2_masks_color/
  step6_sync_mask/                  1_masks_synced/, 2_masks_synced_color/, 3_ids_on_images/, 4_ids_on_masks_color/, scene.json, napkin_id.json
  step7_id_distill/                 point_cloud_id_iter2000.ply, classifier.pth, 1_renders_rgb/{train,test}, 2_renders_id_color_num/{train,test}, video_rgb.mp4
  step8_removal_virtual_cam/        1_config/, 2_object_removal/ (point_cloud_removed.ply, removed_object_id<k>.ply, object_center.json, video_removed_object.mp4),
                                    3_virtual_views_original/, 4_virtual_views_removed/
  step9_inpaint_masks_samt/         1_masks/, 2_masks_overlay/
  step10_inpaint2d_lama/            1_masks_dilated/, 2_inpainted_rgb/, 3_inpainted_depth_npy/, 4_inpainted_depth_vis/
  step11_inpaint_3dgs_come_loss/    point_cloud_inpainted_iter5000.ply (định dạng CoMe), loss_log.json, used_virtual_views.json
  step12_render/                    train/{0_gt,1_removed,2_inpainted}, test/{0_gt,1_removed,2_inpainted}, video_1_removed.mp4, video_2_inpainted.mp4
  step13_come_mesh/                 mesh_come_v1_iter5000.ply, mesh_come_v1_table.ply (+ _crop.json), logs/
  step14_object_closed_3dgs/        point_cloud_44_clean.ply, point_cloud_44_closed.ply, height_map_top.png, footprint.png, summary.json
  step15_object_mesh_merge/         mesh_napkin_raw.ply, mesh_napkin_closed.ply, scene_table_napkin.ply,
                                    collision/{table_static.obj, napkin_dynamic.obj, napkin_convex_*.obj}, physics_meta.json, summary.json
```

4. **Kiểm tra sau khi đặt tên:** mỗi thư mục có đủ số file như bảng mục 4 (vd. 177 mask, 30 view ảo, 154 + 23 render), mở thử ply cuối bằng `plyfile` (có `confidence`, `filter_3D`), rồi xóa `outputs/_download/`.
5. Cập nhật `README.md` của outputs và mục 8 (Trạng thái) của plan.

---

## 5. Chạy trên Modal (`modal_come_inpaint360gs.py`)

| Image | Dùng cho | Nội dung |
|---|---|---|
| `come_image` (mới) | Step 4, 7, 8, 10 (phần chuẩn bị), 11–12 | CUDA 12.1, Python 3.10, torch 2.4.1 cu121 (giống lần train CoMe), rasterizer CoMe + simple-knn build với `TORCH_CUDA_ARCH_LIST=8.9`, `CC=gcc CXX=g++`; thư viện của Inpaint360GS (plyfile, open3d, lpips, scikit-learn, wandb, …) |
| `main_image` cũ (import từ `Based_Model/Inpaint360GS/modal_inpaint360gs.py`) | Step 5–6 (HQ-SAM + association, code Inpaint360GS gốc), Step 9 (SAM-Track tự động) | Đã build và có cache: detectron2 + CropFormer ops, SAM-Track. Weight `CropFormer_hornet_3x_03823a.pth` đã có trên volume `inpaint360gs` (`/vol/weights`). `PYTHONPATH` như lần trước (repo + `seg/detectron2` + thư mục CropFormer). Association chỉ chiếu tâm Gaussian nên đọc được ply CoMe bằng `GaussianModel` gốc |
| `lama_image` cũ | Step 10 (LaMa) | Đã build, có 4 file LaMa bổ sung |

- Volume: dùng `inpaint360gs`, thư mục gốc riêng `/vol/come360/` (`data/teatime`, `output/teatime`, `lama/`, `state/`, `export/`).
- Biến môi trường cho mọi lệnh của `come_image`: `COME_SPLATTING_CONFIG=<model>/3dgs_output/config.json`, `PYTHONPATH=<src>`; Step 8–11 thêm `INPAINT360_OBJECT_CENTER`; Step 11 thêm `COME_LOSS=1`.
- Lưu định kỳ: commit volume mỗi 5 phút, trạng thái từng step trong `/vol/come360/state/stepN.json` (chạy lại thì tiếp tục, step đã xong thì bỏ qua), `retries=2`.
- Cả chuỗi Step 4→12 chạy trong **một hàm trên app đã deploy** (`modal deploy` + `spawn`), nên tắt máy local vẫn chạy tiếp.

Lệnh dự kiến:
```
python copy_code.py                                              # local: copy + đưa step0-3 vào outputs
python -m modal run modal_come_inpaint360gs.py::upload           # đưa step0-3 lên volume
python -m modal run --detach modal_come_inpaint360gs.py::check   # step4 (cổng kiểm tra renderer)
python -m modal deploy modal_come_inpaint360gs.py && python -m modal run modal_come_inpaint360gs.py::chain
python -m modal run modal_come_inpaint360gs.py::download
```

---

## 6. Ước tính

| Hạng mục | Công sức | GPU (L4) |
|---|---|---|
| Copy + sửa code + kiểm tra local | 1–2 giờ | — |
| Build `come_image` + step4 renderer check | 0,5 giờ | ~20 phút (build) + ~5 phút |
| Step 5–6 HQ-SAM (177 ảnh 994×738) + association + chọn ID khăn | 2–3 giờ | ~25–35 phút |
| Step 7 distill (6 lượt render feature/vòng, 2000 vòng, ảnh 994×738) | gỡ lỗi 0,5 ngày | ~20–30 phút |
| Step 8–10 | gỡ lỗi 0,5 ngày | ~20 phút |
| Step 11 fine-tune 5000 vòng với loss CoMe | gỡ lỗi 0,5 ngày | ~20–30 phút |
| Step 12–13 | 0,5 ngày | ~20–40 phút |
| Bước cuối: tải về + đặt tên lại | ~30 phút (tải ~4–6 GB) | — |
| **Tổng** | **~2,5–3 ngày** | **~2,5–3 giờ GPU (~4–6 USD)** |

## 7. Rủi ro và điểm cần xác nhận

1. **HQ-SAM + association có thể lại chạm trần 256 object** (cảnh nhiều vật nhỏ, ID uint8). Kiểm tra ở Step 6: nếu ID khăn bị trùng với vật khác (tràn số), báo lại trước khi chạy tiếp. Phương án dự phòng (lệch paper): dùng mask SAM 2.1 đã đồng bộ ID của `CoMe_Infusion/Step_1_Sync_ID`.
2. **Camera ảo nhìn vào vật:** đúng mô tả của paper, nhưng khác code phát hành; giữ bộ lọc view rỗng làm lưới an toàn.
3. Feature ID render qua `colors_precomp` không hoàn toàn giống rasterizer feature của Inpaint360GS (khác cách sort/blend) → kiểm tra ở Step 5 rằng ID 11 vẫn tách đúng khăn.
4. `filter_3D` tính lại từ 154 camera train: phải khớp cách CoMe tính (đã kiểm chứng ở `CoMe_Infusion` Phase 0). Step 4 là cổng kiểm tra.
5. Rasterizer CoMe chậm hơn khoảng 1,5–2 lần so với rasterizer gốc.
6. Kết quả là biến thể **"Inpaint360GS + CoMe"**, không phải Inpaint360GS nguyên bản; cần ghi rõ khi báo cáo.

## 8. Trạng thái

- [x] plan.md
- [x] Sửa code trong `src/`: renderer (`gaussian_renderer/__init__.py`, thêm `gt_color` cho kênh phương sai màu), `scene/gaussian_model.py` (lớp con), `scene/__init__.py` (camera tính `filter_3D`), `utils/come_loss.py` (7 loss CoMe), `edit_object_inpaint.py` (loss CoMe + lọc view rỗng + `loss_log.json`), `utils/pose_utils.py` (tâm quỹ đạo = tâm vật qua `INPAINT360_OBJECT_CENTER`; **chuẩn hóa ma trận xoay** của camera ảo), `tools_come/render_check.py`, `tools_come/virtual_check.py`
- [x] `copy_code.py` (copy + kiểm tra + đưa step0–3 vào outputs) — 2026-10-27: src 2706 file khớp từng byte; step0–3 đạt kiểm tra (`summary.json` từng thư mục)
- [x] `modal_come_inpaint360gs.py`: upload, step 4–12, chain, status, download, diag (`lama_image` chép nguyên văn từ script cũ)
- [x] Step 4 (2026-10-27): renderer mới trùng **từng pixel** với render CoMe ở 177/177 view (max diff 0); `filter_3D` tính lại từ 154 camera, lệch tối đa 0,0037 so với ply; PSNR test 23,34 dB (khớp CoMe)
- [x] Step 5: 177 mask HQ-SAM 994×738, 4–120 ID/frame (TB 36), không hổng ID
- [x] Step 6: association **chạm trần 256 object** (rủi ro #1). **ID khăn tự chọn = 44**: IoU tổng 0,73, recall 0,97, precision 0,74. Phần ngoài tham chiếu nằm ở frame 94–98 (và 164), đã xem ảnh: **đó là khăn thật** mà mask SAM 2.1 tham chiếu bỏ sót → không thấy ID 44 bị trùng vật khác. Chờ người dùng duyệt trên `step6_sync_mask/3_ids_on_images` trước Step 7
- [x] Step 7 (2026-10-27): distill 2000 vòng (~23 phút); ID 44 trên render khớp mask đồng bộ IoU 0,92; 177 view. Video `video_rgb_id.mp4` render trước khi sửa pose nên bị vỡ (ảnh view không ảnh hưởng)
- [x] Step 8 (`_v2`): lần 1 thất bại: view ảo vỡ nát vì `generate_ellipse_path` để lại hệ số scale PCA (0,168) trong ma trận xoay; rasterizer 3DGS gốc bất biến với scale đều này, rasterizer CoMe (tia trong không gian world) thì không. Đã chuẩn hóa ma trận xoay (`tools_come/virtual_check.py` xác nhận). Lần 2: tâm khăn (trung vị Gaussian) `1.117,2.230,0.970`, khăn hiện ở **30/30** view ảo
- [x] Step 9: SAM-Track tự động, 30/30 mask, không phải click lại, không phải cắt mảnh lệch; diện tích 3,7–4,2 % ảnh. (`unzip` của main_image báo lỗi với zip hợp lệ → giải nén bằng `zipfile`)
- [x] Step 10: LaMa 30 màu + 30 depth (bị preempt 1 lần, tự chạy lại)
- [x] Step 11: plyfusion + fine-tune 5000 vòng với loss Inpaint360GS + CoMe trên 30/30 view ảo (~13,5 phút; preempt 1 lần); 1 873 793 Gaussian, ply có `confidence`, `filter_3D`
- [x] Step 12: render train 154 / test 23 (gt, removed, inpainted) + 2 video
- [x] Step 13 (2026-10-28, `modal_come_mesh.py`): mesh bằng code CoMe gốc (`extract_mesh_tets.py`, mặc định) cho v1 và v2 (~6 phút/mesh, L40S); cắt mặt bàn bằng đúng lăng trụ của `CoMe_Infusion/Step_6_Mesh_GOF` (`tools_local/crop_table.py`) -> `*_table.ply`; đánh giá vùng khăn trên mesh (`tools_local/eval_mesh_table.py`)
- [x] **v2 (phương án C, `src_v2/`, `modal_come_inpaint360gs_v2.py`, kết quả `output_v2/`)**: dùng lại step 0–9, chạy lại 10–12 với mask LaMa nới 20 px + view khởi tạo tự chọn (00020 thay 00004) + sửa render `frame_00002`. Vệt tối: 0 % (v1 TB 6,3 %); mesh vùng khăn phẳng hơn (đỉnh nhô >2 % kích thước khăn: 1,2 % vs 11,7 % ở v1; phủ bàn 98 % vs 94 %)
- [x] Step 14 (v2, 2026-10-28): dự đoán 3DGS phía dưới khăn (đáy + vách trên mesh bàn v2) → `output_v2/.../step14_object_closed_3dgs` (mục 4b, phần "Hiện thực cho v2")
- [x] Step 15 (v2, 2026-10-28): mesh khăn bằng CoMe (87 camera thấy khăn) → khối kín lưới độ cao, 0 đỉnh xuyên bàn, gộp với mesh bàn v2 → `output_v2/.../step15_object_mesh_merge` (mục 4b)
- [ ] Bước cuối: tải về `outputs/`, đặt tên lại, kiểm tra, viết README
