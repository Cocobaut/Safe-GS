# Kết quả Safe-GS trên teatime (chốt đến 2026-11-04)

Số đo thật, có nguồn file; chỗ nào chưa kết luận được thì ghi rõ. Nhật ký chi tiết theo ngày nằm trong `plan.md`
(mục 8). Khảo sát paper nằm trong `paper/` (đặc biệt `14_Inpainting_SOTA_2026.md`, `15_ObjectAware_Training_2026.md`).

## 0. Tóm tắt

- **Pipeline v1 (CoMe + tự làm) là pipeline duy nhất chạy trọn từ ảnh tới video vật lý** (S0 đến S10): dựng cảnh, tách vật,
  lấp nền, mesh, MuJoCo, render lại. Nhấc cốc 20 cm rồi thả, cốc về đúng chỗ cũ (lệch 0.00 cm), các vật khác dời 0.14 cm.
- **Điểm yếu lớn nhất của v1:** (1) chỗ mặt bàn vừa nhấc vật bị nhoè (độ nét 0.31 so với 1.82 của ảnh thật), (2) tách vật yếu
  (LERF-Mask 68.0 / 65.0), (3) mesh chưa vật nào kín (0/7) nên va chạm đang dùng bao lồi.
- **Tách vật tốt nhất hiện có:** mask SAM 3 (2D) và ObjectGS (3D). Trên LERF-Mask: SAM 3 ảnh 77.6 / 75.9 (79.2 / 77.6 khi chỉnh
  ngưỡng), ObjectGS 79.3 / 76.8 (88.2 nếu bỏ prompt "spoon handle", paper 88.9), v1 68.0 / 65.0.
- **Xoá sạch vật khỏi cảnh Gaussian chưa giải được.** Cả ObjectGS lẫn các cách cắt theo ID, hull, vị trí Gaussian đều còn sót lớp
  "vỏ" của vật dẹt (gói kẹo, vỏ đĩa), vì vật dẹt bị vẽ thẳng lên các Gaussian của mặt bàn. Xoá và lấp mặt bàn là một việc.
- **Khuyến nghị:** giữ v1 làm xương sống (Gaussian tường minh, nối được mesh và MuJoCo); đưa vào v1 (a) mask SAM 3 cho bước gán nhãn
  S3, (b) cắt theo bán kính ảnh hưởng của Gaussian, (c) lấp mặt bàn bằng plane fill / texture. ObjectGS dùng anchor thần kinh nên
  khó đưa sang mesh, chỉ giữ làm tham chiếu.

## 1. Những gì đã chạy

| # | Việc | Trạng thái | Kết quả ở đâu | Code |
|---|---|---|---|---|
| 1 | COLMAP teatime | xong | `Data/teatime/colmap/` | `modal_come.py::colmap_teatime` |
| 2 | CoMe 30k iter | xong | `Data/teatime/come/` | `modal_come.py::train_teatime` |
| 3 | Pipeline v1 S0 đến S9 + đo S10 | xong | `Data/teatime/pipeline/v1/` (báo cáo `S10_eval/eval_report.md`) | `modal_pipeline.py`, `pipeline/` |
| 4 | Inpaint360GS (WACV 2026) nguyên pipeline tác giả | xong, mới đánh giá bằng mắt | `Data/teatime/i360gs/result/` | `modal_inpaint360.py` |
| 5 | ObjectGS (ICCV 2025) trên LERF-Mask | xong, tái lập đúng paper | `Data/teatime/objectgs/eval_lerf/` | `modal_objectgs.py::_train`, `tools/objectgs_eval_lerf.py` |
| 6 | Mask SAM 3 teatime | xong (bản 3, 17 ID) | `Data/teatime/sam3_masks/` | `modal_sam3.py`, `tools/mask_quality.py` |
| 7 | Visual hull từ mask + camera | xong | `Data/teatime/mask_hull/` (SAM2), `mask_hull_sam3/` | `tools/mask_hull.py` |
| 8 | ObjectGS trên COLMAP của mình + mask SAM 3, 3 mức lambda | xong | `Data/teatime/objectgs/eval_removal_*` | `modal_objectgs.py::_pipeline_sam3`, `tools/objectgs_eval_removal.py` |
| 9 | Chẩn đoán phần còn sót sau khi xoá | xong | `Data/teatime/objectgs/eval_removal_sam3_lam1.0_probe/` | `--probe` của `objectgs_eval_removal.py` |

## 2. Pipeline v1: bảng đo S10

Nguồn: `Data/teatime/pipeline/v1/S10_eval/eval_report.md` (+ `s10_ghost.py` cho số "ma").

| Stage | Chỉ số | Giá trị |
|---|---|---|
| S0 COLMAP | ảnh đăng ký / sai số chiếu lại / độ dài track / điểm 3D | 177 trên 180 / 0.66 px / 4.74 / 24,685 |
| S0 | độ lệch tâm camera so với COLMAP của nhóm LERF (177 frame chung) | RMS 0.49 cm, lớn nhất 2.64 cm |
| S2 CoMe | PSNR / SSIM / LPIPS(vgg), 23 view test | 23.34 / 0.805 / 0.263 |
| S2 | 3 góc mới của LERF (test_0, 1, 2) PSNR | 25.40 / 24.45 / 24.63 |
| S3 | LERF-Mask mIoU / mBIoU, 10 prompt | 68.0 / 65.0 (bỏ "spoon handle": 75.6 / 72.2) |
| S4 nền | PSNR ngoài vùng vật: nền đã lấp / cảnh gốc | 23.74 / 23.73 dB (đã sửa rò rỉ view test) |
| S4 | chỗ vật vừa đứng: % pixel trong 1 cm / 2 cm quanh mặt bàn; độ cao lệch trung vị | 48.2% / 65.4%; 1.10 cm |
| S4 | độ nét chỗ lấp so với vành bàn (render / ảnh thật) | 0.31 / 1.82 |
| S4 | "ma" cấp Gaussian (cao hơn mặt bàn > 1.5 cm trong cột vật): số Gaussian, alpha TB, % pixel bị che | 6,962; 0.157; 28.7% |
| S9 ghép lại | PSNR / SSIM / LPIPS, 23 view test | 22.62 / 0.794 / 0.271 |
| S6 mesh | vật kín / Chamfer mesh với Gaussian | 0 trên 7 / 2.5 đến 3.9 mm |
| S6 | Hausdorff mesh tới mảnh lồi | 14 đến 40 mm (ly trà bị CoACD chia 210 mảnh; hộp kem sữa bị bỏ) |
| S8 mô phỏng | để yên 2 s: dời lớn nhất / xoay lớn nhất / lún / tốc độ | 1.4 mm / 0.01 độ / 0 mm / 6.8 lần thời gian thực |
| S8 | nhấc cốc 20 cm rồi thả: về chỗ cũ / vật khác bị dời | 0.00 cm / 0.14 cm |

LERF-Mask v1 theo prompt (IoU): táo 93.4, túi bánh 96.0, cốc 83.7, bánh trên đĩa 56.2, khăn giấy 84.3, đĩa 89.1, cừu 61.6,
thìa 0.0, gấu 90.9, ly trà 25.2.

## 3. So sánh phương pháp theo khâu

### 3.1 Dựng cảnh (23 view test của mình, cùng cách chia 1 trên 8)

| Phương pháp | PSNR | SSIM | LPIPS |
|---|---|---|---|
| CoMe (v1) | 23.34 | 0.805 | 0.263 |
| ObjectGS + mask SAM 3, lambda 0.01 | 23.19 | 0.792 | 0.258 |
| ObjectGS, lambda 0.1 | 22.24 | 0.790 | 0.262 |
| ObjectGS, lambda 1.0 | 22.46 | 0.789 | 0.273 |
| Inpaint360GS (3DGS gốc) | 23.30 | không đo | không đo |

### 3.2 Tách vật, LERF-Mask teatime (mIoU / mBIoU)

Giá trị tham chiếu lấy từ Table 1 của Gaga (arXiv 2404.07977), đã kiểm trên bản html.

| Phương pháp | mIoU | mBIoU | Ghi chú |
|---|---|---|---|
| SA3D | 42.5 | 39.2 | tham chiếu |
| LERF | 49.7 | 42.6 | tham chiếu |
| LangSplat | 59.8 | 52.7 | tham chiếu |
| **Safe-GS v1 (SAM2 + vote)** | **68.0** | **65.0** | 10 prompt |
| OmniSeg3D | 69.8 | 63.8 | tham chiếu |
| Gaga | 71.2 | 68.4 | tham chiếu |
| Gaussian Grouping | 71.7 | 66.1 | tham chiếu |
| **SAM 3 chế độ ảnh, mặc định** | **77.6** | **75.9** | prompt chữ, ngưỡng 0.5 |
| SAM 3 chế độ ảnh, ngưỡng 0.3 | 79.2 | 77.6 | chọn trên chính 2 ảnh test, hơi lạc quan |
| **ObjectGS (dữ liệu LERF)** | **79.3** | **76.8** | chọn ID kiểu oracle; bỏ "spoon handle": 88.2 (paper 88.9) |
| ILGS | 84.3 | 75.5 | tham chiếu |

Lưu ý: ObjectGS và v1 nối prompt với ID bằng cách tra tay hoặc oracle (mask DEVA của view test trống), nên là cận trên so với
giao thức chọn ID theo chữ.

SAM 3 ảnh, theo prompt (IoU mặc định): táo 95.0, túi bánh 96.3, cốc 75.7, bánh trên đĩa 93.0, **khăn giấy 0.0 (prompt "paper
napkin" hỏng, "napkin" thì chạy)**, đĩa 90.2, cừu 97.0, thìa 81.5, gấu 98.9, ly trà 48.3 (ngưỡng 0.3 đưa lên 96.2 nhưng làm gấu
tụt từ 99 xuống 65). Nở mask +1 px còn 75.8 / 74.0, +2 px còn 71.2 / 68.4, co -1 px còn 75.2 / 73.3: **biên mask đã đúng,
không nên lấp khe 2 px bằng cách nở mask**.

### 3.3 Xoá vật và lấp nền

| Phương pháp | Xoá | Lấp mặt bàn | Ghi chú |
|---|---|---|---|
| v1 | vote theo gradient, cắt theo ID | Gaussian phẳng + tinh chỉnh theo ảnh ProPainter | nhoè chỗ vật vừa đứng |
| Inpaint360GS | xoá theo ID tự phát hiện | LaMa trên 30 view ảo (màu + độ sâu) rồi tối ưu 3D | chỗ cốc, đĩa, táo, ly thành mặt bàn sạch; còn khối nhoè lớn chỗ túi bánh + ly (chỉ nhìn ảnh, chưa đo S10) |
| ObjectGS | bỏ anchor theo ID | không có bước lấp | còn "vỏ" vật dẹt, xem mục 6 |

## 4. Mask SAM 3 (bản 3)

Prompt chữ, video 177 ảnh COLMAP, `facebookresearch/sam3 @ f1647e8`. Kết quả `Data/teatime/sam3_masks/labels/`,
`legend.json`, video kiểm tra `quality_v3/overlay.mp4`.

| | SAM2 của team | SAM 3 bản 1 | SAM 3 bản 3 (dùng) |
|---|---|---|---|
| số ID | 106 | 26 | 17 |
| số lần diện tích nhảy bất thường giữa 2 frame liền kề | 1010 | 262 | (chỉ cải thiện hậu xử lý, tracker không đổi) |
| gấu | đổi ID 2 / 9 / 6 | một ID 20 | một ID 20 |
| thìa | gộp vào cốc | ID riêng 110 | ID riêng 110 |

Sửa từ bản 1 sang bản 3: bàn chỉ giữ ID 13 (các ID "table" khác bắt nhầm đầu gấu, cổ cừu); gộp cà phê vào cốc (140 vào 80),
trà vào ly (141 vào 70), các miếng bánh quy thành nhóm 100; thêm prompt "small white cup", "puddle of liquid", "blue snack package"
(chọn bằng cách thử ảnh, `probe/cand_*.jpg`; bỏ instance trùng vật đã có nhãn); xoá mảnh rời nhỏ hơn 40 px.

Tác động ở 3D (visual hull, tỉ lệ vùng mập mờ trên vùng chắc chắn, càng nhỏ càng tốt):

| Vật | SAM2 | SAM 3 |
|---|---|---|
| ly trà | 7.64 | 0.84 |
| cừu | 15.9 | 1.46 |
| bánh quy | 1.35 | 0.05 |
| đĩa | 2.23 | 1.62 |
| cốc | 1.28 | 1.18 |
| khăn giấy | 1.38 | 1.50 |

Hull vẫn phình ở vật phẳng (đĩa 2766 cm3 so với mesh v1 1220; bánh 691 so với 167) vì camera nhìn từ trên xuống. Cốc kem sữa
chỉ có 58 frame nên hull = 0. Chưa kiểm các đoạn vật biến mất ở frame 140 đến 176.

## 5. ObjectGS trên COLMAP của mình với mask SAM 3

Config `lerf_mask` của tác giả giữ nguyên, chỉ đổi tên dataset và đường dẫn; chạy A100-80GB (L4 và L40S hết bộ nhớ ở bước đánh giá).
Đã vá lỗi checkpoint của tác giả (`tools/patch_objectgs_ckpt.py`: `capture` và `restore` không khớp nên không resume được).

| lambda_object_loss | PSNR / SSIM / LPIPS | IoU ID (`metrics.py`) | anchor bị xoá theo ID | anchor ma > 1.5 cm | alpha ma (cấp anchor) | % phủ |
|---|---|---|---|---|---|---|
| 0.01 (gốc) | 23.19 / 0.792 / 0.258 | 0.905 | 98,515 | 11,433 | 0.205 | 66.5 |
| 0.1 | 22.24 / 0.790 / 0.262 | 0.945 | 115,098 | 3,391 | 0.227 | 70.9 |
| 1.0 | 22.46 / 0.789 / 0.273 | 0.944 | 151,392 | 2,693 | 0.151 | 41.5 |

Số theo độ sâu (so được giữa các bản ObjectGS, không so được với v1): lambda 1.0 làm mặt bàn dưới vật vừa xoá **phẳng hơn hẳn**:
phẳng trong 1 cm 51.5% (lambda 0.01: 16.5%), độ cao lệch trung vị 1.02 cm (4.16 cm). Đổi lại mất 0.7 đến 1 dB PSNR.

ID render so với mask SAM 3 trên view test (lambda 0.01): pixel vật đúng 98.9%; IoU theo vật 0.78 đến 0.95 (bàn 0.92, gấu 0.94,
cừu 0.94, cốc 0.90, đĩa 0.92, táo 0.93, khăn giấy 0.95); yếu: gói kẹo 0.46 (nhãn 180) và 0.75 (181), vũng trà 0.61 (nhãn 171).
Hơi lạc quan vì point cloud gán nhãn dùng cả mask của view test.

## 6. Xoá vật: các cách cắt và nguyên nhân phần còn sót

Thước đo "ma": alpha của RIÊNG các Gaussian còn lại nằm trong cột trên vết chân của vật đã xoá, cao hơn mặt bàn > 1.5 cm
(mặt bàn lấy từ S3, cùng hệ toạ độ COLMAP). 0 là sạch.

Model lambda 1.0, tập vật đầy đủ:

| Cách xoá | alpha ma (cấp Gaussian) | % pixel bị che | mặt bàn phẳng trong 1 cm | mặt bàn thủng |
|---|---|---|---|---|
| theo ID | 0.068 | 19.3 | 51.5 | 19.9 |
| + lọc Gaussian có tâm trong hull, cao hơn 0.3 cm | 0.036 | 10.4 | 56.5 | 21.0 |
| + lọc Gaussian trong hull, kể cả sát và dưới mặt bàn | 0.036 | 10.4 | 48.5 | 29.8 |
| + cắt cả cột trên vết chân hull (mọi độ cao) | 0.023 | 6.2 | 33.4 | 52.8 |
| v1 (CoMe, nền S4b), cùng định nghĩa | 0.157 | 28.7 | không áp dụng | không áp dụng |

Cắt theo vị trí anchor thay vì vị trí Gaussian cho kết quả kém hơn: lambda 0.01: alpha 0.205 xuống 0.119 (phủ 66.5% xuống 43.2%);
lambda 1.0: 0.151 xuống 0.061 (phủ 41.5% xuống 19.4%). Số đo cấp anchor phóng đại so với cấp Gaussian (0.151 so với 0.068
trước khi cắt), nên **không dùng số cấp anchor để so với v1**.

**Chẩn đoán** (`--probe`, 3 view test):
- Hull phủ 97 đến 100% mask của mọi vật (gói kẹo 97.4%), nên không phải hull lệch.
- Phần che pixel vật sau khi xoá ID đến chủ yếu từ Gaussian **sát mặt bàn (dưới 0.3 cm) mang nhãn bàn 13** (alpha 0.92 đến 1.0
  ở gói kẹo, đĩa, ly) và nhãn nền 0; ở gói kẹo các dải cao hơn gần 0 (0.005).
- Kết luận: vật dẹt (dày vài mm) bị vẽ lên chính lớp Gaussian của mặt bàn, không có Gaussian riêng. Ngay cả cắt cả cột trên vết
  chân vẫn còn mảng xanh gói kẹo nhỏ hơn, vì còn Gaussian mặt bàn lớn có tâm ngoài vết chân nhưng vệt phủ vào vật.
- Ảnh minh hoạ cùng view 00065: `eval_removal_sam3_lam1.0*/B_*_10.jpg` (trái trước, phải sau khi xoá).

## 7. Giới hạn đo lường và điều chưa kết luận

- Số "ma" của v1 và ObjectGS dùng vết chân khác nhau (v1 1,260 ô 1 cm, ObjectGS 1,500 đến 4,198 ô vì tập vật xoá khác nhau); khi
  ép cùng tập 6 vật, ObjectGS cho alpha 0.183 / 0.199 / 0.150 (lambda 0.01 / 0.1 / 1.0) ở cấp anchor. So với v1 chỉ mang tính tương đối.
- Chưa đo sàn nhiễu của mặt bàn ở vùng không có vật, nên chưa biết alpha bao nhiêu là "sạch thật".
- Thước đo "ma" mù với thứ nằm sát mặt bàn (dưới 1.5 cm). Phần còn sót nhìn thấy trong ảnh thuộc vùng này.
- Chưa có ground truth 3D cho mask 177 frame; chất lượng mask video đo gián tiếp (độ ổn định, hull) và nhìn bằng mắt.
- Inpaint360GS chưa đo bằng S10.
- Độ sâu của ObjectGS là độ sâu kỳ vọng (pha trộn lớp trong suốt), của v1 là độ sâu tại T = 0.5: các % phẳng / thủng không so chéo được.

## 8. Bài học kỹ thuật

- Modal `deploy` bỏ qua file gắn vào bằng `add_local_file` nếu định nghĩa hàm không đổi ("no changes detected"). Đổi hậu tố đường
  dẫn đích mỗi lần sửa script (xem `EVAL_RM` trong `modal_objectgs.py`).
- Chạy lâu dùng `deploy` rồi `spawn` (xem `tools/modal_spawn.py`) để job không chết theo phiên làm việc; checkpoint commit lên
  Volume mỗi 60 giây.
- GPU L4 đôi khi hết chỗ: cho phép `["L4", "L40S"]` (cùng sm_89). ObjectGS cần A100-80GB.
- Repo clone trên Windows bị CRLF làm hỏng `make.sh`; repo Inpaint360GS thiếu `LaMa/saicinpainting/training/data` (bị `.gitignore`
  loại), lấy từ `advimman/lama`.
- Token HuggingFace đã bị dán vào chat: nên xoá và tạo lại, rồi cập nhật secret `huggingface` của Modal.

## 9. Việc tiếp theo (đề xuất)

1. Đưa mask SAM 3 vào S3 của v1 (đổi `S3_label.objects` theo ID SAM 3: 80, 70, 60, 100, 90, 50, 130, 110, 170, 180, 160) và chạy lại S3 đến S10.
2. Cắt theo bán kính ảnh hưởng của Gaussian (tâm cộng độ lớn), không chỉ theo tâm, rồi lấp mặt bàn bằng plane fill + texture sạch
   (tái dùng `pipeline/s4_background.py`; hướng 3DTexMOR hoặc baseline Inpaint360GS để so).
3. Đo S4 mới bằng S10 kèm thước đo "ma" (`pipeline/s10_ghost.py`) và đo thêm "ma sát mặt bàn" (0.1 đến 1.5 cm).
4. Mesh kín cho vật (remesh) rồi chạy lại CoACD; đo Chamfer, Hausdorff.
5. Đo Inpaint360GS bằng S10 để có bảng so sánh công bằng.

## 10. Tái lập

```
python -m modal run --detach modal_pipeline.py::run --config configs/pipeline/teatime_v1.json --until S10
python -m modal run modal_pipeline.py::pull --config configs/pipeline/teatime_v1.json
python -m modal deploy modal_sam3.py            # rồi gọi hàm _run (mask) hoặc _probe (thử prompt); ::pull, ::pull_probe tải về
python -m modal deploy modal_objectgs.py        # _pipeline_sam3 (dữ liệu + train), _train_sam3(lam, tag), _eval_removal(...)
python -m modal run modal_inpaint360.py::seg    # rồi ::remove --target ..., ::inpaint
python tools/mask_hull.py --scheme sam3 --colmap Data/teatime/colmap/sparse/0 --labels Data/teatime/sam3_masks/labels --out Data/teatime/mask_hull_sam3
python tools/mask_quality.py --images Data/teatime/colmap/images --labels Data/teatime/sam3_masks/labels --legend Data/teatime/sam3_masks/legend.json --out Data/teatime/sam3_masks/quality_v3
python tools/sam3_lerf_eval.py --probs Data/teatime/sam3_masks/probe/lerf_probs.npz --gt Data/benchmarks/lerf_mask_dl/x/teatime/test_mask --out Data/teatime/sam3_masks/probe/lerf_eval.json
```

Cần Modal secret `huggingface` (token đọc) cho SAM 3 và weights CropFormer của Inpaint360GS. `Data/` không đưa lên git.

## 11. Chưa commit (nhánh `minhnt`, commit cuối `397bce3`)

`modal_sam3.py`, `pipeline/s10_ghost.py`, `tools/` (`mask_hull.py`, `mask_quality.py`, `modal_spawn.py`, `objectgs_eval_lerf.py`,
`objectgs_eval_removal.py`, `patch_objectgs_ckpt.py`, `sam3_lerf_eval.py`), thay đổi trong `modal_pipeline.py`,
`modal_inpaint360.py`, `modal_objectgs.py`, `plan.md`, thư mục `Based_Model/sam3/` (nên thêm dạng submodule như hai repo kia),
`RESULTS.md`. Không đưa `Data/` và `.vscode/settings.json`.

## S4c: tinh chỉnh CoMe với hai nhóm Gaussian (vật / nền), mask SAM 3 (2026-11-04)

Script `pipeline/s4c_decouple.py`, chạy `modal_pipeline.py::_s4c`. Gán nhóm bằng bỏ phiếu mask, 8000 bước, loss alpha theo nhóm
trong ảnh hợp thành (sil + leak). Kết quả trên 23 view test (kết quả ở Volume `teatime/pipeline/v1/S4c/<tag>`):

| run | PSNR | SSIM | LPIPS | Gaussian vật | còn giống vật % | thủng % | alpha ma > 1.5 cm | phủ % |
|---|---|---|---|---|---|---|---|---|
| ctrl (0 bước) | 23.34 | 0.805 | 0.263 | 47060 | 10.8 | 0.0 | 0.035 | 7.5 |
| w1 (sil 0.1, leak 0.5) | 23.33 | 0.800 | 0.262 | 47060 | 11.3 | 0.0 | 0.034 | 9.0 |
| w2 (sil 0.5, leak 2.5) | 23.30 | 0.798 | 0.265 | 47060 | 11.3 | 0.0 | 0.038 | 9.2 |
| w2_r25 (ngưỡng gán 0.25) | 23.29 | 0.798 | 0.264 | 50024 | 10.7 | 0.0 | 0.042 | 9.9 |

Kết luận: không đạt. Gói kẹo và đĩa vẫn còn nguyên trong ảnh chỉ-nền (frame_00065) ở cả ctrl lẫn w2; tinh chỉnh không giảm phần còn sót
(còn giống vật ~11 %). Chỉ số ma 0.035 của ctrl đã thấp hơn v1 (0.157) nhờ gán nhóm bằng SAM 3 + bỏ phiếu, không do huấn luyện thêm.
Phần còn sót là Gaussian nền bám vào mặt vật phẳng, loss alpha không đẩy được chúng sang nhóm vật vì RGB loss giữ chúng ở đó.

## S4d: cắt Gaussian nền trong vết chân + plane_fill, nền S4c/ctrl (2026-11-04)

Script `pipeline/s4d_cutfill.py`, chạy `modal_pipeline.py::_s4d`, kết quả Volume `teatime/pipeline/v1/S4d/<tag>`, ảnh ở `Data/teatime/pipeline/v1/S4d/`.
Cắt (a) tâm trong vết chân (nới cut_dilate ô, trên mặt bàn -1 cm) và (b) Gaussian lớn gần vết chân (khoảng cách < reach x scale, cao < low_cm).

| run | cut_dilate / reach / low_cm / fill_dilate | cắt | thêm | còn giống vật % | thủng % | alpha ma |
|---|---|---|---|---|---|---|
| (S4c ctrl, không cắt) | | 0 | 0 | 10.8 | 0.0 | 0.035 |
| d1 | 2 / 2.0 / 3 / 4 | 70910 | 3659 | 6.3 | 0.0 | 0.000 |
| d2 | 4 / 3.0 / 5 / 6 | 96930 | 4553 | 5.6 | 0.0 | 0.000 |
| d3 | 2 / 0 / 3 / 3 | 70038 | 3202 | 6.3 | 0.0 | 0.000 |

- Ảnh chỉ-nền frame_00065 (d1, d2): gói kẹo, đĩa, cốc biến mất, mặt bàn liền, không thủng. Đây là lần đầu hết "ma" nhìn thấy được.
- alpha ma = 0.000 là hiển nhiên vì cột bị cắt sạch, chỉ số này không còn phân biệt được; cần thước đo mới (chất lượng mặt bàn lấp).
- 5 đến 6 % pixel còn "giống vật" chủ yếu do màu bàn vùng lấp gần màu vật (đĩa trắng không trùng), chưa kiểm từng ô.
- Hạn chế: vùng lấp là màu phẳng, mất vân gỗ, có vệt tam giác ở biên Gaussian lớn; cắt (a)+(b) cũng ăn vào một phần con cừu bên trái (view 65, d1) vì vết chân gồm cả vật sát cừu. Chưa tinh chỉnh theo ảnh inpaint nên chưa có NVS.
- d2 sạch hơn d1 nhưng cắt nhiều hơn 37 %. d3 (không cắt theo kích thước) kết quả gần như d1 nên điều kiện (b) gần vô tác dụng, điều kiện (a) là chính.
- Việc tiếp: chạy tinh chỉnh kiểu S4b (3000 bước, ảnh inpaint) trên nền đã cắt+lấp để phục hồi vân gỗ; giới hạn cắt không ăn Gaussian của cừu/gấu; đo PSNR trên vùng không-vật.

### S4d, vòng 2: bảo vệ cừu/gấu + tinh chỉnh (2026-11-04)

`pipeline/s4d_cutfill.py` thêm: `--protect` (Gaussian cừu/gấu theo bỏ phiếu SAM 3 ID 20, 30 không bị cắt), `--ft_iters`, `--ft_mode inpaint|masked`,
`--freeze_region`. Đo trên 22 view test có mask. PSNR vùng không-vật = ngoài mask vật nới 15 px; đối chứng nền S4c chưa cắt: 23.47 (cừu/gấu 24.74).
Bảng dùng cắt cut_dilate 2 (e3, e6 dùng 4 / reach 3 / low 5 / fill 6).

| run | tinh chỉnh | PSNR không-vật | PSNR cừu/gấu | còn giống vật % | nhận xét ảnh frame_00065 |
|---|---|---|---|---|---|
| e1 (cắt + lấp) | không | 22.83 | 24.72 | 6.4 | sạch, cừu nguyên; màu bàn phẳng, vệt tam giác |
| e2 | ảnh inpaint, 3000 | 23.49 | 24.71 | 22.8 | gói kẹo hiện lại (đích ProPainter vẫn còn vật) |
| e3 | ảnh inpaint, cắt rộng | 23.46 | 24.71 | 22.9 | như e2 |
| e4 | ảnh inpaint, lr x1.0 | 23.45 | 24.66 | 23.4 | như e2 |
| e5 | ảnh thật, bỏ giám sát trong mask vật, 3000 | 23.56 | 24.69 | 12.1 | không còn gói kẹo; vết chân lốm đốm, bóng viền đĩa |
| e6 | như e5, cắt rộng | 23.52 | 24.69 | 12.1 | như e5 |
| e7 | như e5, lr x1.0, 6000 | 23.60 | 24.37 | 14.9 | như e5, cừu/gấu giảm |
| **e8** | như e5 + đóng băng Gaussian lấp | 23.33 | 24.68 | **6.7** | **sạch, vân gỗ ngoài vết chân khôi phục, còn vệt tam giác** |
| e9 | như e8 + đóng băng vùng quanh vết chân 6 ô | 23.02 | 24.76 | 6.6 | sạch, kém e8 về PSNR |

Kết luận:
- Bảo vệ cừu/gấu hoạt động: PSNR cừu/gấu giữ 24.7 (đối chứng 24.74), không còn cắt lẹm.
- Ảnh inpaint ProPainter (S4a, mask SAM 2) KHÔNG dùng làm đích được: vẫn còn gói kẹo nên tinh chỉnh vẽ lại vật (22.8 %).
- Tinh chỉnh ảnh thật có bỏ giám sát trong mask vật (masked) tránh được việc đó; để Gaussian trong vết chân tự do thì trôi và lốm đốm (e5 đến e7);
  đóng băng Gaussian lấp (e8) cho kết quả tốt nhất: PSNR không-vật 23.33 (đối chứng 23.47, mất 0.14 dB), còn giống vật 6.7 % (đối chứng 10.8 %), không thủng.
- alpha ma ~ 0.0005 ở mọi run (cột đã cắt sạch), không còn phân biệt; 6 % "còn giống vật" chưa kiểm từng ô (màu bàn gần màu vật ở một số ô).
- Còn lại: vân gỗ vùng lấp là màu phẳng + vệt tam giác ở biên Gaussian lớn (e8 giảm nhưng chưa hết); chưa nối vào S5 đến S9 (mesh, MuJoCo) và chưa đo trên các view train.
- Cấu hình đề xuất cho v2: SAM 3 nhãn + cut_dilate 2 / protect + plane_fill + tinh chỉnh masked đóng băng Gaussian lấp 3000 bước (e8, Volume `teatime/pipeline/v1/S4d/e8`).

### Run v2: nền S4b thay bằng S4d/e8, chạy lại S9 và S10 (2026-11-04)

`configs/pipeline/teatime_v2.json` (run_name v2). `modal_pipeline.py::_mk_v2` sao chép v1 sang v2, thay `S4b_background/background_finetuned.ply`
bằng `S4d/e8/S4like/background_finetuned.ply` (có NOTE.txt), xoá S9, S10; chỉ S9 (0.8 phút) và S10 (2.0 phút) chạy lại. S5 đến S8 (vật, mesh, MuJoCo) không đổi.
Báo cáo đầy đủ: `Data/teatime/pipeline/v2/S10_eval/eval_report.md` (v1: `.../v1/S10_eval/eval_report.md`).

| Chỉ số S10 | v1 | v2 (nền e8) |
|---|---|---|
| Chỗ vật vừa đứng: % pixel nền trong 1 cm / 2 cm quanh mặt bàn ↑ | 48.2 / 65.4 | **93.0 / 97.9** |
| Chỗ vật vừa đứng: khoảng cách trung vị tới mặt bàn ↓ | 1.10 cm | **0.09 cm** |
| Độ nét chỗ lấp / vành bàn (ảnh thật 1.82) | 0.31 | 0.84 |
| PSNR ngoài vùng vật: nền đã lấp / cảnh gốc | 23.74 / 23.73 | **21.97** / 23.73 |
| LPIPS ngoài vùng vật: nền đã lấp / cảnh gốc | 0.230 / 0.230 | 0.263 / 0.230 |
| S9 ghép lại PSNR / SSIM / LPIPS | 22.62 / 0.794 / 0.271 | 21.62 / 0.775 / 0.304 |

Đọc kết quả:
- Hình học chỗ vật vừa đứng sạch hơn hẳn (nền bám mặt bàn, ma biến mất), đây là mục tiêu cho MuJoCo (nhấc cốc, bàn liền).
- Đổi lại điểm ảnh tệ hơn: PSNR ngoài vùng vật giảm 1.77 dB và S9 giảm 1.0 dB. Chỉ số S10 "ngoài vùng vật" dùng mask SAM 2 không nới,
  nên gồm cả vành quanh vết chân (bóng đổ, vân gỗ) nơi nền đã cắt + lấp phẳng; số riêng của S4d (ngoài mask nới 15 px) chỉ mất 0.14 dB. Phần mất nằm ở vành sát vật,
  chưa kiểm sâu hơn.
- Một nửa cải thiện "độ nét" (0.31 -> 0.84) vẫn dưới ảnh thật (1.82): vùng lấp còn mờ.
- e10 (ô lấp 0.5 cm, 14144 Gaussian, đóng băng): PSNR không-vật 23.35, còn giống vật 6.5 %, ảnh gần như e8; vệt tam giác không đổi. Vệt đến từ Gaussian lớn
  của nền gốc quanh vết chân, không phải từ lớp lấp, nên thu nhỏ ô lấp không giúp.
- Chưa làm: kiểm lại cắt lẹm vành bàn (thu cut_dilate / thay lấp phẳng bằng lấy mẫu texture ảnh thật), kiểm 6 % còn giống vật, đo mesh/đáy vật v2 (S5, S6 giữ nguyên v1).
