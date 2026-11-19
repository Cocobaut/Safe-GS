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

## Do thi chong do (support graph) tu hull va tu Gaussian (2026-11-12)

`tools/support_graph.py` (union-find cum dinh nhau, topo sort Kahn cho thu tu go, hau due = vat mang theo khi nhac, kiem tra do nhay theo tol).
Ket qua: `Data/teatime/support_graph/{hull,gauss}/` (report.md, graph.json, sensitivity.json, top_view.png, side_view.png, graph.png, matrices.png).
- Nguon hull (SAM 3, 12 vat): do dung 8/8 vat ky vong nam tren ban va cap coc_su do thia (khe -0.62 cm, on dinh tu tol 1.0), nhung SAI cap dia do banh_quy
  (chi ra "touch", khe -2.73 cm) vi hull dia cao toi 10.2 cm (vat that ~3.7 cm) va hull lan xuong duoi ban; 14 canh cham ben chua dang tin.
- Nguon Gaussian S3 (SAM 2 v1, 8 vat): cap dia do banh_quy_1 / banh_quy_2 dung va on dinh o moi tol (khe +0.12 / +0.06 cm), dia, coc_su,
  ly_tra, tui_banh tren ban; nhung thieu goi_keo, khan_giay, thia, vung_tra (khong co nhan S3 v1).
- Ket luan: hull do duoc "tren ban" nhung khong du chinh xac de xac dinh quan he doc giua hai vat; Gaussian cho ket qua doc dung nhung can nhan day du.

### Do thi chong do, ban tong quat (2026-11-12)

Muc tieu: khong chinh tham so theo vat / theo canh. Thay doi trong `tools/support_graph.py`:
- bo bang ten va ky vong gan cung (ky vong chuyen ra `configs/support_graph/teatime_expect.json`, chi de doi chieu, `--expect`);
- suy luan chong do theo trong luc: vat khong co vat khac do thi do ban do, neu day (tru ban kinh Gaussian) du gan mat ban, tinh tuong doi theo chieu cao vat;
- `--scale_free`: moi nguong tinh theo do nhieu mat ban sigma (MAD / IQR cua Gaussian ban) va ban kinh Gaussian cua tung vat:
  nguong cap = he so x (sigma + (r_A + r_B)/2), quet he so 1, 1.5, 2, 3, 4; canh giu lai khi xuat hien o >= 60 % muc quet (dong thuan);
- cum phu thuoc = union-find chi tren canh do (canh cham ben chi liet ke, truoc day noi het 13 vat thanh 1 cum);
- `tools/auto_support_plane.py`: tu tim mat do bang RANSAC tung vat khi khong co nhan ban.

Nhan SAM 3 cho Gaussian: run moi `configs/pipeline/teatime_v3_sam3.json` (S0, S2 chep tu v1 bang `_clone_stages`, S1 + S3 chay lai, S3 4 phut),
13 vat + ban. Ket qua `Data/teatime/support_graph/gauss_sam3_scalefree/`:

| Nguon | Vat | Tren ban dung | Cap chong dung | Ghi chu |
|---|---|---|---|---|
| hull SAM 3 | 12 | 8/8 | 1/2 | sai dia do banh_quy (hull dia cao 10 cm, that ~3.7 cm) |
| Gaussian S3 v1 (SAM 2) | 8 | 5/5 | 2/2 | thieu 5 vat chua co nhan |
| Gaussian S3 SAM 3, nguong cm | 13 | 9/9 | 2/2 | coc_su>thia chi 0.6 phieu |
| **Gaussian S3 SAM 3, scale_free** | 13 | **9/9** | **2/2** | dia>banh_quy 1.0, coc_su>thia 0.8 phieu, khong vat lo lung |

- gau, cuu bi tinh "ban do" (tay / chan dat len ban; ghe khong co nhan nen khong co nut ghe).
- Thu canh thu hai (LERF figurines, Gaussian Split&Splat stage 2 cu, 39 vat): du lieu suy bien (nhieu vat chi 1 o vet chan, dong cuc o
  1.5 den 2 don vi tren "mat do" tu tim; ty le met khong biet), KHONG dung lam bang chung tong quat. Can mot canh thu hai chay du pipeline
  (COLMAP + CoMe + SAM 3 + S3) de kiem tra that.

## Khung chat + khuon inpaint (pipeline: prompt -> khung -> chi giu cho thay duoc -> khuon cho phan khong thay) (2026-11-12)

### Lan 1 (SAI, da bo): vat can gia dinh
`tools/visibility_mold.py`, `tools/carve_frame.py`: coi moi tam Gaussian (opacity >= 0.5) la vat can cung, ban tia tu voxel / Gaussian toi camera.
Kiem chung bang render (`pipeline/s3v_unseen_check.py`, Modal `_unseen_check`): bo 42,774 Gaussian "khong thay" -> PSNR test 22.20,
bo cung so Gaussian "thay" (doi chung) -> 23.16, day du 23.34. Tuc la nhom "khong thay" thuc ra van gop mau (3DGS cong don nhieu lop ban
trong suot). So "27 % Gaussian coc su khong ai thay" la SAI. Hai tool nay khong dung nua.

### Lan 2 (dung): do bang chinh bo render CoMe, `pipeline/s3m_mold.py`, Modal `_s3m`, run v3_sam3, ket qua `Data/teatime/pipeline/v3_sam3/S3m/`
A. Gaussian duoc quan sat = tong trong so dong gop render qua 154 view train (s3_label.accumulate, mask = 1) >= 1e-4.
   Kiem chung tren 23 view test (khong dung de tinh):

| Bo | PSNR test | sai khac trung binh trong mask vat | % pixel vat doi > 2/255 |
|---|---|---|---|
| day du | 23.3447 | - | - |
| bo 3,264 Gaussian vat khong dong gop | 23.3447 | 0.00001 | 0.01 % |
| bo 3,264 Gaussian vat dong gop (doi chung) | 23.3382 | 0.0010 | 2.93 % |

   -> tieu chi nay dung voi render, ke ca o goc nhin moi. Ty le Gaussian vat khong dong gop: 0.8 den 4.3 % (goi_keo 13.3 %, thia 8.9 %).
   Ca canh: 408,346 / 1,804,024 Gaussian (22.6 %) khong dong gop o view train nao (chua kiem bo het ca canh).
B. Khoet hull theo anh do sau render (177 view, tol 1 cm, >= 3 view nhin xuyen -> trong):

| vat | hull cm3 | khung cm3 | khoet % | cao p98 hull / khung / Gaussian (cm) | vo thay % | khuon day % | khuon ben/sau % |
|---|---|---|---|---|---|---|---|
| coc_su | 824 | 542 | 34.3 | 13.3 / 11.0 / 11.9 | 95.2 | 4.2 | 0.5 |
| ly_tra | 828 | 402 | 51.5 | 13.3 / 9.8 / 11.9 | 94.0 | 5.3 | 0.7 |
| dia | 2275 | 915 | 59.8 | 8.3 / 2.7 / 2.9 | 64.7 | 34.7 | 0.6 |
| banh_quy | 558 | 321 | 42.4 | 5.7 / 2.9 / 2.9 | 58.5 | 41.5 | 0.0 |
| tao | 240 | 224 | 6.4 | 7.8 / 7.3 / 7.8 | 95.4 | 3.4 | 1.2 |
| tui_banh | 1680 | 1378 | 18.0 | 19.0 / 18.6 / 20.5 | 89.3 | 6.2 | 4.5 |
| khan_giay | 1100 | 670 | 39.1 | 6.5 / 5.3 / 6.2 | 71.6 | 26.9 | 1.5 |
| goi_keo | 128 | 79 | 38.1 | 3.0 / 1.3 / 1.2 | 78.6 | 21.4 | 0.0 |
| thia | 9 | 4 | 61.2 | 15.6 / 15.5 / 15.8 | 100 | 0 | 0 |
| gau | 45812 | 30879 | 32.6 | 44.5 / 45.4 / 56.3 | 56.1 | 20.3 | 23.7 |
| cuu | 9298 | 7522 | 19.1 | 46.1 / 46.0 / 47.9 | 67.7 | 3.6 | 28.7 |

- Hull cua vat det duoc ep dung do day: dia 8.3 -> 2.7 cm (Gaussian 2.9), banh_quy 5.7 -> 2.9, goi_keo 3.0 -> 1.3 (Gaussian 1.2).
- Khuon day lon o vat det nam tren vat khac / ban (dia 35 %, banh_quy 42 %, khan_giay 27 %); mat sau cuu / gau (24 den 29 %) la khuon ben/sau
  (camera khong quay ra sau). Hinh: `S3m/frames_side.png`.
- Han che: tol 1 cm lam vung sat mat ban (< 1 cm) co the bi tinh "be mat thay" do trung do sau mat ban (khuon day coc_su chi 4.2 %);
  ly_tra / coc_su khung thap hon Gaussian 1 den 2 cm o mieng (vanh mong bi khoet). vung_tra hull gan rong (1 cm3).

## Buoc 1: tinh chinh Gaussian bi gioi han trong khung (`pipeline/s3c_constrain.py`, Modal `_s3c`, run v3_sam3/S3c) (2026-11-12)

Quy tac chung: bo Gaussian vat khong dong gop (3,243); loss = L1/SSIM anh that + lam_bound * mean(opacity * khoang cach ra ngoai khung, cm);
5000 buoc, khong densify; sau do tia Gaussian ngoai khung > 1 cm co opacity < 0.05. 23 view test.

Lan 1 (ep moi vat, c0..c3): Gaussian vat ngoai khung 35.1 % -> 1.9 den 2.9 %, nhung PSNR 23.345 -> 23.07 / 22.65 / 22.46 va do phu trong mask
0.986 -> 0.93 / 0.88 / 0.87: vat bi thung. Nguyen nhan: hull gau / cuu bi cat cut boi vung voxel (mask_hull.py lay vung tu vat tren ban,
bo gau / cuu, nen hull cham bien vung; 44 den 45 % Gaussian cua chung ngoai khung) va hull vung_tra gan rong (97 % ngoai).

Lan 2 (them kiem khung: vat co > 20 % Gaussian ngoai khung > 0.5 cm thi khung khong dang tin, khong ep; tu dong loai gau, cuu, vung_tra):

| run | lam_bound | PSNR | SSIM | LPIPS | ngoai khung (vat duoc ep) | do phu trung binh |
|---|---|---|---|---|---|---|
| truoc | - | 23.345 | 0.8053 | 0.2629 | xem bang duoi | 0.986 |
| r0 (doi chung, chi tinh chinh) | 0 | 23.385 | 0.8004 | 0.2621 | khong doi | 0.984 |
| r1 | 0.1 | 23.375 | 0.8003 | 0.2630 | tui_banh 3.9 -> 0.1 %, ly_tra 8.1 -> 5.7 %, goi_keo 3.2 -> 1.7 % | 0.981 |
| r2 | 1.0 | 23.344 | 0.7991 | 0.2650 | tui_banh 3.9 -> 0.1 %, ly_tra 8.1 -> 5.0 %, goi_keo 3.2 -> 1.6 % | 0.980 |

- Ep khung khong lam hai anh khi khung dang tin (PSNR r1 / r2 ngang doi chung), nhung loi rat it: Gaussian vat cua CoMe + S3 (nhan SAM 3)
  da nam trong khung san (0 den 8 %), do ro ra ngoai mask cua tung vat ~ 0.000 o moi run. Do phu giam nhe (goi_keo 0.960 -> 0.909, thia 0.969 -> 0.920 o r2).
- Ket luan: voi teatime, buoc "ep khung" khong phai cho gay loi chinh; loi chinh la phan KHONG thay (day, mat sau) -> buoc 2 (inpaint khuon).
- Can sua: mask_hull.py dang bo gau / cuu khoi vung dung hull (gan cung ten) -> hull cua vat lon bi cut. Sua thanh vung tinh tu moi vat.

## Buoc 2 (ban co so, khong mo hinh sinh): lap khuon bang Gaussian det (`pipeline/s3i_mold_fill.py`, Modal `_s3i`) (2026-11-12)

Chuoi: hull SAM 3 v2 (`tools/mask_hull.py` sua: vung tu moi vat, bo gan cung gau/cuu) -> S3m v3 (khung chat; cat duoi ban chi khi vat dat tren ban,
xet bang Gaussian < -5 cm < 1 %) -> S3c r1v3 (ep khung lam 0.1; kiem khung tu dong: chi vung_tra bi loai) -> S3i.
Quy tac lap chung cho moi vat:
- khuon = vo khung khong camera nao thay; dia Gaussian det tai tam voxel, phap tuyen ra ngoai khung (day: -phap tuyen ban), mau = trung binh
  1/d cua 8 Gaussian gan nhat cua chinh vat;
- day chi trong vet chan cua chinh Gaussian vat tren mat phang (dong lo, giu manh roi);
- khuon tiep xuc theo do thi chong do: mat vat do duoi vet chan vat ben tren, o do cao trung vi mat vat do quanh vien;
- LOC: bo moi Gaussian lap ma o mot view train no roi vao pixel trong (alpha < 0.5) hoac nam truoc be mat render (> 0.3 cm): phan lap
  khong duoc doi pixel da quan sat.

Ket qua cuoi (S3i/i_r1v3k). Anh test ca canh: PSNR 23.3405 -> 23.2862, sai khac trung binh 0.00075.
"Thung" = % pixel trong bong cua khung ma vat dung RIENG co alpha < 0.5, camera ao 5 goc cao (-75 tu duoi len .. 60) x 8 phuong vi:

| vat | khuon (day / ben-sau) | lap giu / bo vi lo | thung truoc -> sau % |
|---|---|---|---|
| banh_quy | 1218 / 47 | 937 / 13 (315 ngoai vet chan) | 9.91 -> 1.65 |
| dia | 2716 / 151 (+339 tiep xuc duoi banh) | 2973 / 233 | 2.26 -> 0.29 |
| gau | 45537 / 101870 | 59016 / 72691 | 13.94 -> 9.40 |
| cuu | 8850 / 4557 | 5736 / 5138 | 3.84 -> 1.73 |
| goi_keo | 234 / 0 | 231 / 3 | 0.38 -> 0.32 |
| khan_giay | 1424 / 97 | 1496 / 25 | 0.01 -> 0.00 |
| coc_su, ly_tra, tao, tui_banh | nho | | ~0 -> ~0 |

- Banh quy nhin tu duoi: truoc trong suot / rong, sau co day kin mau banh, hai banh tach roi (preview `S3i/i_r1v3k/preview/banh_quy.jpg`).
- Dia nhin tu tren: cho banh tung nam con vien lo, lap tiep xuc mot phan (233 Gaussian bi loc vi lo o view that).
- Thung o coc / tao / tui_banh ~0 ngay tu dau: Gaussian ben trong vat da che kin bong (do kin theo alpha, chua noi gi ve mau mat day).
- Gau: 49 % phan lap bi loc vi lo o view that -> khung gau rong hon gau that o nhieu cho; "thung" 9.4 % con lai la do khung qua beo, khong han la lo.
- Han che thuoc do: chi do alpha, khong do mau / hinh mat day co hop ly khong.

Su co ky thuat: Modal giu ban thu muc `pipeline/` gan vao cu (add_local_dir) du da deploy lai, ke ca khi doi env theo hash; 3 lan chay
lien tiep thuc te dung code cu (phat hien qua dong "phien ban" trong log). Cach chac chan: `tools/push_code.py` day pipeline/*.py len Volume
theo hash noi dung, ham Modal nhan `code_rel` va chay script tu do (hien moi ap dung cho `_s3i`).

## Buoc 3: mesh kin tu khung + MuJoCo (`tools/frame_mesh.py`, chay may local) (2026-11-12)

Khung S3m v3 (ca ruot) -> lam min Gauss 0.8 o -> marching cubes -> he ban (met, z = phap tuyen ban); vat dat tren ban (mesh khong xuong duoi
ban qua 2 cm) thi KHOA PHANG DAY: cat o z = 0 va dong nap; kiem kin bang bang bam canh; MuJoCo: ban = plane, vat = freejoint + mesh,
500 kg/m3, solref 0.002, dt 1 ms, dat day cach ban 1 mm, 2 s; chi thu rieng vat ma do thi chong do noi la ban do. Ket qua `mesh_v3/report.md`.

| vat | kin (canh bien / non-manifold) | Euler / genus | the tich cm3 (bao loi) | MuJoCo truot mm / xoay do / lun mm |
|---|---|---|---|---|
| coc_su | co (0/0) | 0 / 1 (quai coc) | 521 (687) | 0.00 / 0.00 / 0.03 |
| ly_tra | co | 2 / 0 | 383 (548) | 0.00 / 0.00 / 0.03 |
| dia | co | -4 / 3 (lo gia) | 776 (1431) | 0.00 / 0.00 / 0.03 |
| tao | co | 2 / 0 | 220 (224) | 0.00 / 0.01 / 0.03 |
| tui_banh | co | 0 / 1 (lo gia) | 1348 (1543) | 0.00 / 0.00 / 0.03 |
| khan_giay | co | 0 / 1 (lo gia) | 580 (955) | 0.00 / 0.00 / 0.03 |
| goi_keo | co | 2 / 0 | 56 (68) | 0.00 / 0.00 / 0.03 |
| banh_quy | co | 2 / 0 | 268 | khong thu rieng (dia do) |
| thia | co | 2 / 0 | 1 | khong thu rieng (coc_su do) |
| gau / cuu | co | -228 / 115, -6 / 4 | 156528 / 35724 | khong thu rieng (khong dat tren ban) |
| vung_tra | khung qua mong | | | |

- 11/11 mesh dung duoc deu kin (0 canh bien, 0 non-manifold). v1 S6: khong mesh nao kin, phai dung bao loi lam va cham.
- Truoc khi khoa phang day: tao lan 138 mm / xoay 105 do (day tron do lam min). Sau khi khoa: moi vat do ban dung yen (truot 0, xoay <= 0.01 do).
- Lun 0.03 mm la do tiep xuc mem cua MuJoCo (solref 0.002); voi solref mac dinh la 0.88 mm.
- Han che: genus gia (dia 3, khan_giay / tui_banh 1, gau 115) do khung co duong ham nho; 3 mieng banh_quy dinh thanh 1 manh (khung lien);
  coc_su dac (khong rong ben trong) vi phan ruot khong thay duoc lap day; MuJoCo va cham bang bao loi cua tung mesh (chua chia loi CoACD).

### Buoc 3b: lam sach topo, tach manh, CoACD, thu nhac trong ca canh (2026-11-12)

`tools/frame_mesh.py` them: dong hinh thai ban kinh 2 o (lap duong ham nho), co 2 o + danh nhan de tach manh noi bang cau mong,
chia loi CoACD (threshold 0.05) cho va cham. `tools/scene_lift.py`: ca canh MuJoCo (ban = plane; vat dat tren ban / duoc vat khac do =
freejoint; gau, cuu khong dat tren ban -> co dinh lam vat can; boundmass 0.01 kg), go chong lan ban dau bang cach day vat ben tren len,
de yen 1 s, roi han tung vat vao mocap nhac 10 cm, so vat "mang theo" voi hau due trong do thi chong do. Ket qua `mesh_v4/`, `lift_v4/`.

| | truoc (mesh_v3) | sau (mesh_v4) |
|---|---|---|
| banh_quy | 1 khoi | 3 manh (118 / 134 / 25 cm3), genus 0 |
| genus gau / cuu | 115 / 4 | 7 / 2 |
| genus dia / khan_giay / tui_banh | 3 / 1 / 1 | 3 / 1 / 1 (duong ham rong hon 0.65 cm, chua lap) |
| coc_su | genus 1 (quai) | genus 1 (quai giu nguyen) |
| so manh loi CoACD | - | coc_su 42, ly_tra 40, tui_banh 33, dia 21, ... |

Thu nhac (lift_v4/lift.md):
- Lan dau that bai: thia (0.7 g, mong) lam mo phong NaN -> MuJoCo reset du lieu (tat luon khop han); banh_quy lun vao dia 6.6 mm.
  Sua chung: boundmass / boundinertia, go chong lan truoc khi chay.
- Sau khi sua: 9/9 vat nhac len du 10 cm. Nhac dia -> mang theo banh_quy (khop do thi). Nhac moi vat khac khong xo vat nao ngoai thia.
- 8/9 khop do thi; lech: nhac coc_su khong mang theo thia vi thia (mesh chi 1 cm3, phan can thia; phan mui thia trong coc khong thay) roi
  khoi coc ngay khi de yen (dich 448 mm).
- banh_quy phai day len 24 mm moi het chong lan voi dia: khung dia con chiem phan the tich duoi banh (khuon tiep xuc), roi 25 mm xuong dia khi de yen.

## Mesh ca canh + render ca canh theo MuJoCo, run v3 (2026-11-12)

Mesh be mat ca canh: TSDF cua CoMe (`Based_Model/CoMe/extract_mesh_tsdf.py`, Modal `_full_mesh`) tren model 30k:
`Data/teatime/full_scene_mesh/tsdf_vox010.ply` (73 MB, o 0.01 don vi canh ~ 2.5 mm) va `tsdf_vox020.ply` (18 MB, ~5 mm). Don vi canh COLMAP
(1 don vi ~ 24.9 cm, xem `support.json` meters_per_unit); khong cat, khong dong kin (mesh be mat ca canh, khong phai mesh vat ly).

Render Gaussian ca canh theo quy dao MuJoCo (`pipeline/s9v3_render.py`, Modal `_s9v3`; `tools/scene_lift.py --record`):
nen = S4d/e8 (nhom nen SAM 3, cat vet chan + lap ban + tinh chinh), vat dong = Gaussian tu S3i filled.ply (ep khung + lap khuon)
bien doi theo pose MuJoCo, quy dao: de yen 0.5 s, han + nhac 15 cm trong 1.5 s, giu 0.5 s, tha 1 s (25 khung/s, 87 khung).
Ket qua `Data/teatime/pipeline/v3_sam3/S9v3/r1/traj_lift_{dia,coc_su,tao}/`: `frame_00017|00065|00113.mp4`, `*_keyframes.jpg`, `below_<vat>.jpg`
(camera ao nhin tu duoi len vat dang nhac, 3 goc cao -60/-30/0 x 3 phuong vi).
- Nhac dia (frame_00113): dia nhac len cung hai banh quy, mat ban phia duoi con nguyen (con vet mo nhat).
- Nhac coc su (frame_00017): coc len roi roi xuong; khong vat nao khac di theo (thia khong di theo coc, da bao truoc).
- below_coc_su.jpg: day coc kin, nhung la mang mau theo dang cac tam (mau lay tu Gaussian gan nhat), chua phai texture that.
- Han che: nen e8 co nhuoc diem da neu (vet tam giac / bong mo quanh vet chan); coc_kem_sua khong co trong canh (da bi bo khoi nen, khung rong);
  gau / cuu chi o trong nen, khong co phan lap mat sau; chua co video khoi tao tu goc ngoai cac goc train.

## Hoan thien vat bang mang sinh: TRELLIS + khung quan sat (kieu Points-to-3D) (2026-11-12)

Tham khao (tra 2026-11-12): Points-to-3D https://arxiv.org/abs/2603.18782 (2026-03-19, dung TRELLIS, thay nhieu khoi dau cua latent cau truc
bang latent cua dam may diem thay duoc, xem nhu inpainting latent; khong tim thay code); Amodal3R https://arxiv.org/abs/2503.13439 (2025-03-17,
code github.com/Sm0kyWu/Amodal3R); SAM 3D Objects https://arxiv.org/abs/2511.16624 (2025-11-20, trong so HF bi khoa thu cong);
MV-SAM3D https://arxiv.org/abs/2603.11633 (nhieu goc, dung SAM 3D); GENA3D https://arxiv.org/abs/2511.21945 (2025-11-26);
DinoComplete https://arxiv.org/abs/2605.26949 (2026-05-26, hoan thien hinh tu quet mot phan). TRELLIS: github.com/microsoft/TRELLIS (MIT),
trong so microsoft/TRELLIS-image-large (khong khoa).

Cach lam (`tools/prep_trellis_inputs.py`, `modal_trellis.py`, TRELLIS commit 193e0502, A100): luoi 64^3 he ban (z len) quanh vat:
frame / seen (be mat thay) / unknown / free; z0 = SS-encoder(khoi dau); lay mau flow tu t0 voi CFG trung binh theo 6 anh (mask SAM 3);
RePaint trong latent (o latent da biet het giu = z0 nhieu dung muc); occ = (decoder > 0 AND NOT free) OR seen; tang SLAT theo nhieu anh -> Gaussian + mesh.
DINOv2 cua TRELLIS (torch.hub, dl.fbaipublicfiles.com) loi chung chi SSL het han 2026-11-11 -> dung facebook/dinov2-with-registers-large
qua transformers 4.49 (hidden_states[-1] = x_prenorm); CHUA kiem so voi trong so goc.

| run | khoi dau | o "chac chan trong" | t0 | ghi chu |
|---|---|---|---|---|
| t06_rp | ca khung | moi o ngoai khung | 0.6 | mang gan nhu chep lai khung (IoU 0.97 den 0.997, them 0 o) -> thi nghiem sai: ngoai hull chua chac trong |
| seen_t08_carved | chi be mat thay | o camera nhin xuyen + duoi ban | 0.8 | dung tinh than Points-to-3D |
| frame_t08_carved | ca khung | nhu tren | 0.8 | |

seen_t08_carved (`Data/teatime/trellis_out/compare_<vat>.png`, `mesh_views_seen_t08_carved.png`, `summary.md`):
- Ket qua la VO (TRELLIS bieu dien be mat, ruot rong), giu dung be mat thay va o trong da biet; THEM ngoai khung: tao 2697, dia 1339,
  banh_quy 557, coc_su 167 o (chu yeu o mat ben / day).
- Day duoc dong kin o ca 4 vat (nhin thang tu duoi: tao, coc_su, banh_quy kin; dia con 1 lo nho). Quai coc giu nguyen.
- Mesh TRELLIS kin theo canh nhung nhieu manh (14 den 72 manh) -> genus tinh ra am, khong dung duoc truc tiep cho MuJoCo; can loc manh.
- Chua lam: ghep Gaussian TRELLIS vao canh (chi vung khuon) + loc "khong doi pixel da quan sat" + so voi lap phang (S3i); kiem huong anh dieu kien
  co khop huong khung; khong co ground truth de do sai so hinh.

### TRELLIS cho moi vat: ghep Gaussian vao canh + mesh + MuJoCo, so voi ban quy tac (2026-11-12)

TRELLIS (seen_t08_carved) chay cho ca 13 vat; gau / cuu chay lai voi luoi sua (khong ep "duoi ban la trong" cho vat khong dat tren ban:
`tools/prep_trellis_inputs.py` xet khung co xuong duoi ban > 2 cm khong).

A. Ghep Gaussian TRELLIS vao canh (`pipeline/s3i_mold_fill.py --trellis_dir/--trellis_grid`): Gaussian TRELLIS doi ve he canh (vi tri khop:
lech trung vi 0.2 cm voi Gaussian vat, do cao khop), chi giu Gaussian o o chua quan sat, roi loc "khong doi pixel da quan sat".

| ban | PSNR test (goc 23.3405) | thung banh_quy | thung dia | thung cuu | thung gau |
|---|---|---|---|---|---|
| lap phang (S3i/i_r1v3k) | 23.2862 | 1.65 | 0.29 | 1.73 | 9.40 |
| TRELLIS, bo o be mat thay noi 1 o (i_trellis) | 23.1271 | 9.86 | 1.57 | 3.41 | 11.22 |
| TRELLIS, khong noi (i_trellis_d0) | 23.0540 | 9.78 | 1.57 | 3.37 | 11.20 |

- Ban TRELLIS KEM hon lap phang o ca PSNR test va do thung. Day banh quy nhin tu duoi van trong mo du them 13k Gaussian: Gaussian TRELLIS rat nho
  (trung vi ~0.17 mm, 32 Gaussian / o) nen phu kem khi render bang bo render CoMe; nghi do khac bo loc / kernel giua hai bo render, CHUA kiem.
B. Khung dac tu TRELLIS (`tools/trellis_frames.py`: vo TRELLIS lap lo + hop khung quan sat) -> `tools/frame_mesh.py` -> `mesh_trellis/`:

| vat | genus quy tac (mesh_v4) | genus TRELLIS | the tich quy tac / TRELLIS cm3 |
|---|---|---|---|
| dia | 3 | 1 | 783 / 1099 |
| khan_giay | 1 | 0 | 586 / 715 |
| tui_banh | 1 | 0 | 1375 / 1519 |
| coc_su | 1 (quai) | 1 (quai) | 526 / 597 |
| thia | 0 | 0 | 1.0 / 6.0 (them phan mui thia) |
| gau / cuu | 7 / 2 | 19 / 3 | 158234 / 198431, 35927 / 43573 |

- 13/13 mesh kin; vat do ban dung yen (truot 0, xoay <= 0.01 do). TRELLIS lam sach topo o dia / khan_giay / tui_banh nhung phong the tich 10 den 40 %.
C. Thu nhac ca canh voi mesh TRELLIS (`lift_trellis/lift.md`): 10/10 vat nhac duoc, dia mang theo banh_quy (khop), nhung chong lan ban dau tang
  (banh_quy|dia 29 mm, coc_su|gau 10 mm do gau phong to), thia van roi khoi coc va bi xo xa.

Ket luan: tich hop TRELLIS chay tron pipeline, nhung tren teatime va cac thuoc do nay KHONG hon ban quy tac (lap phang + khung khoet);
loi duy nhat ro rang la lam sach duong ham gia o vai vat. Ban de xuat giu: khung khoet + lap phang + mesh_v4. Muon TRELLIS co ich can:
khop kich thuoc / bo loc Gaussian giua TRELLIS va CoMe, kiem DINOv2 thay the, chi dung TRELLIS cho vat co phan khong thay lon, thu Amodal3R / SAM 3D.

## Thi nghiem tu dong 2026-11-17 (chieu): v5_da3 tron chuoi, lap texture anh that, cong SAM 3 khong huan luyen

Modal: profile `quangkhaisaigonhcm`, Volume `safe-gs-come` (chi co du lieu Amodal3R luc dau). Da day len: colmap (images, sparse, input), come
(file nho + point_cloud/iteration_30000), sam3_masks/labels + legend, S3_label (support, groups, stats, labels.npy, config_used), S3m/v5_da3
(frame + contrib.npy lay tu S3m/v4_p0: contrib chi phu thuoc Gaussian CoMe, khong phu thuoc khung), support_graph/graph.json, nen e8, S4c/ctrl.
Log: `logs/v5chain.log`, `logs/s4dtex_l40s_v2.log`, `logs/s9v5.log`, `logs/lift_*.log`, `logs/frame_mesh_*.log`, `logs/sam3_gate_eval.log`.

Su co du lieu (anh huong ket qua, da sua):
- `Data/teatime/support_graph/graph.json` o may la ban CU (hull, khong co canh dia -> banh_quy). Ban dung = `support_graph/gauss_sam3_scalefree/graph.json`
  (trung voi ban tren Volume cu). Da thay tren Volume truoc khi S3i chay; moi lan chay MuJoCo hom nay dung ban dung.
- `Data/teatime/pipeline/v1/S4c/ctrl/S4like/background_finetuned.ply` o may bi hong: 695,991 / 1,756,964 dong toan so 0 (kich thuoc file van
  trung ban tren Volume). Render file nay doi 51.9 GiB (OOM tren L40S o moi view). Tai lai tu Volume cu: 0 dong hong. Lan tai e8 dau tien cung
  hong (11,007 dong 0), lan 2 sach => `modal volume get` tren may nay doi khi ghi khoi 0; phai kiem noi dung, khong chi kich thuoc.
  Cac file khac da kiem: 177 nhan PNG giai ma duoc, 357 anh khong co khoi 0, objects_all.ply 0 quaternion 0, frame npz huu han.

### TN1. v5_da3 tron chuoi S3c -> S3i -> frame_mesh -> MuJoCo -> S9v3

S3c `r1v5` (lam_bound 0.1, 5000 iter, nhu r1v3; loc khoi dong 2 cm / opacity 0.2) va S3i `i_r1v5`, 23 view test
(`Data/teatime/pipeline/v3_sam3/S3c/r1v5/metrics.json`, `S3i/i_r1v5/metrics.json`; doi chung `S3c/r1v3/metrics.json`, `S3i/i_r1v3k/metrics.json`):

| | S3c sau: PSNR / SSIM / LPIPS | Gaussian vat ngoai khung sau | S3i PSNR test truoc -> sau |
|---|---|---|---|
| r1v3 / i_r1v3k (khung S3m_v3) | 23.3405 / 0.7991 / 0.2645 | 2.35 % | 23.3405 -> 23.2862 |
| r1v5 / i_r1v5 (khung v5_da3) | 23.3739 / 0.8001 / 0.2634 | 35.12 % | 23.3739 -> 23.3336 |

- Loc khoi dong chi bo 29 Gaussian; bo sau tinh chinh 441 (r1v3: 5834).
- Kiem khung tu dong: voi v5_da3, 44.82 % Gaussian gau va 45.68 % Gaussian cuu nam ngoai khung -> khung "khong dang tin", KHONG ep va S3i KHONG
  lap gau / cuu (r1v3: 6.46 / 2.91 %, co ep). Ti le 35.12 % ngoai khung la do hai vat nay. PSNR cao hon mot phan vi gau / cuu khong bi ep / lap.
  Chua tach duoc phan nay do DA3 hay do khung goc S3m (khong chay S3c tren khung goc).
- Thung sau lap (% pixel alpha < 0.5, camera ao): dia 0.29 -> 0.28, banh_quy 1.65 -> 2.60 (kem hon), goi_keo 0.32 -> 0.19, coc_su / ly_tra / tao /
  tui_banh ~0 o ca hai.
- S10: `s10_eval_cpu` / `s10_eval_gpu` KHONG chay cho v5: hai script doc cac stage bo cuc v1 (S4b, S5, S6, S8) ma chuoi v3 / v5 khong tao ra;
  so PSNR / SSIM / LPIPS o tren la cua s3c / s3i tren cung 23 view test. s10_ghost chay trong TN2.

Mesh (`tools/frame_mesh.py`) va MuJoCo (`tools/scene_lift.py`). De tach rieng tac dung DA3, mesh them khung goc ma v5_da3 duoc khoet tu do
(`S3m/frame` -> `mesh_v5base/`); `mesh_v4` lam tu `S3m_v3/frame` (khung khac).

| vat | mesh_v4 (S3m_v3): cm3 / genus | mesh_v5base (S3m) | mesh_v5 (v5_da3) |
|---|---|---|---|
| dia | 782.8 / 3 | 781.0 / 3 | 738.5 / 3 |
| coc_su | 525.9 / 1 | 527.1 / 1 | 490.5 / 4 |
| ly_tra | 386.4 / 0 | 387.5 / 0 | 307.3 / 1 |
| khan_giay | 586.1 / 1 | 585.9 / 0 | 511.9 / 2 |
| banh_quy | 276.4 (3 manh) | 276.5 (2 manh) | 273.9 (2 manh) |
| thia | 1.0 | 1.0 | 0.0 (khung con 57 o) |
| gau | 158234 / 7 | 30762 / 4 | 27996 / 53 |
| cuu | 35927 / 2 | 7526 / 1 | 6981 / 16 |

- DA3 (v5base -> v5) giam the tich vua phai nhung TANG genus gia (coc_su 1 -> 4, gau 4 -> 53, cuu 1 -> 16, ly_tra 0 -> 1, khan_giay 0 -> 2) va xoa thia.
- gau / cuu nho di ~5 lan la do doi khung S3m_v3 -> S3m (truoc DA3). Voi khung S3m, cuu "dat tren ban" theo mesh -> thanh vat dong va DO
  (thu rieng tren ban: truot 591.9 mm, xoay 98.7 do o v5base; 573.9 mm / 96.9 do o v5). Hoi quy nay co tu khung goc, khong phai do DA3.

Sua do luong trong `tools/scene_lift.py` (so cu cua lift_v4 bi anh huong):
- vi tri do bang `xpos` = goc than = goc canh (cach vat den ~1 m) -> moi phep xoay phong dai do dich; nay do bang tam khoi `xipos`.
  Vd thia "dich 448 mm" (lift_v4) = 135.2 mm theo tam khoi (cung code cu, `lift_v4_rerun`).
- moi han nhac neo o goc canh -> vat chung xuong (tam khoi chi len 4.8 den 9.8 cm khi nhac 10 cm, `lift_v4_rerun`). `--com_frame 1`: khung than dat
  tai tam khoi, mocap cung tu the dau. Kiem (`.tmp_scripts/check_com_frame.py`): vi tri moi geom giong het (sai khac lon nhat 7e-18 m), tam khoi 1.7e-16 m.
- dem canh bao phan ky MuJoCo (BADQACC / QPOS / QVEL; MuJoCo tu reset khi gap): 0 o moi lan chay duoi day.
- them thu nhac coc 20 cm roi tha (`--drop_test coc_su --drop_cm 20`), ghi `carried_rise_cm`.

| (--com_frame 1, `lift_*_com/`) | v4 | v5base | v5 |
|---|---|---|---|
| chong lan banh_quy / dia luc dau | 6.6 mm | 8.7 mm | 10.7 mm |
| day banh_quy len de go chong lan ("tre 24 mm") | 24.2 mm | 24.7 mm | 24.1 mm |
| nhac dia 10 cm: tam khoi dia / banh_quy len (cm) | 9.986 / 9.985 | 9.986 / 9.984 | 9.986 / 9.985 |
| khop do thi chong do | 8/9 (lech: coc_su, thia khong theo) | 10/11 (nhu tren) | 10/11 (nhu tren) |
| de yen 1 s: vat dich > 1 mm (tam khoi, mm) | banh_quy 5.65, thia 130.6 | banh_quy 3.5, cuu 322.8, dia 1.4, gau 4.3, thia 159.4 | banh_quy 2.85, cuu 303.1, dia 1.41, gau 4.26, thia 150.5 |
| coc 20 cm roi tha 2 s: len / lech ngang / lech doc / nghieng | 19.99 cm / 0.000 mm / 0.0 mm / 0.001 do | 19.99 / 0.003 / 0.0 / 0.001 | 19.99 / 0.003 / 0.0 / 0.006 |

- Ket luan "tre banh_quy khi nhac dia": KHONG cai thien co y nghia (24.2 -> 24.1 mm; tren cung khung goc, DA3 chi 24.7 -> 24.1). Khi nhac, banh_quy
  theo dia khong tre (lech 0.001 cm theo tam khoi).
- Neu giu khung goc canh cu (`--com_frame 0`, cung code): coc 20 cm tha -> lech 117.9 mm, nghieng 91.4 do (v4); 189.4 mm / 179.1 do (v5): do mo hinh
  moi han, khong phai do mesh.
- Thia: roi khoi coc o moi ban. Chot do day (`frame_mesh.py --min_thick_vox 2`, cung quy tac cho moi vat; chi thia (loi 1.41 / 1.0 o) va vung_tra
  (1.0) bi no 1 o, moi vat khac loi >= 3.16 o nen khong doi): thia 1.0 -> 8.2 cm3 (`mesh_v4g`), 0.0 -> 5.2 cm3 (`mesh_v5g`) nhung VAN ROI
  (de yen: 143.2 mm v4g, 133.0 mm v5g; khong chot: 130.6 / 150.5). Nguyen nhan la thieu cho do (phan mui thia trong coc khong quan sat duoc), khong
  phai do mong -> KHONG bat mac dinh.
- S9v3 `r_v5` (nen e8 + S3i/i_r1v5 theo quy dao `lift_v5_com`, --com_frame 1 da doi ve tu the khung goc canh): 
  `Data/teatime/pipeline/v3_sam3/S9v3/r_v5/traj_lift_{dia,coc_su,tao}/`. Nhac dia mang banh quy; cuu nghieng dan trong video (dung voi mo phong v5).
- De xuat: KHONG dung v5_da3 lam mac dinh. Neu giu khoet DA3 can chan khoet o cau truc mong va kiem genus truoc / sau; can xem lai khung S3m
  (cuu do) truoc khi dung cho mo phong.
- `Data/teatime/mold_inspector_360.html` dung lai: them lop mesh_v4 / mesh_v5 / mesh_v5g (diem tren be mat, `tools/mesh_to_asset.py`); khung / khuon
  khong doi (truoc S3m, sau v5_da3).

### TN2. Lap mat ban bang texture anh that (`pipeline/s4d_cutfill.py --texture_fill 1`)

Cach lam (cung quy tac moi vat): moi Gaussian lap cua plane_fill chieu vao 154 view train; chi lay pixel co nhan SAM 3 = ban (ID 13, co 3 px);
>= 2 view thay -> mau = trung vi anh that. Vung lien thong khong view nao thay (duoi day vat): chep texture ban tu mot phep tinh tien ca vung
(16 huong x khoang cach 1 / 1.5 / 2 / 3 lan duong kinh vung, chon ngan nhat co >= 90 % diem dich duoc thay). O lap 4 mm, scale trong mat phang
0.5 o. Con lai giu tham so e8 (cat vet chan + bao ve cuu / gau + tinh chinh anh that masked 3000 iter, dong bang Gaussian lap). L40S.
Them do vet chan cung dinh nghia S10 (`hole_*`) + PSNR trong vet chan o pixel anh that thay ban. Ket qua `Data/teatime/pipeline/v1/S4d/{e8r,e8g4,e8tex}/`
(metrics.json, ghost.json, preview/).
Kiem tai lap: e8r (cung code moi) cho cut 64,354, bao ve 6,556, PSNR khong-vat sau lap 22.8288 = e8 goc.
e8tex: 14,191 diem lap, 7,962 thay truc tiep (trung vi 7 view), 6,229 chep tu 20 vung tinh tien (20/20 vung tim duoc), 0 diem lay mau gan nhat.

| ban (sau tinh chinh) | PSNR khong-vat | PSNR vet chan (pixel thay ban) | % trong 1 cm / 2 cm | trung vi cach ban (cm) | do net / anh that | con giong vat % | alpha ma TB (s10_ghost) |
|---|---|---|---|---|---|---|---|
| nen S4c/ctrl chua cat (doi chung) | 23.4692 | 26.176 | 76.19 / 88.61 | 0.453 | 2.050 / 5.975 | 10.84 | - |
| e8r: lap phang mau vanh, o 1 cm | 23.3338 | 21.829 | 89.09 / 94.60 | 0.094 | 0.627 / 5.975 | 6.57 | 0.000427 |
| e8g4: lap phang mau vanh, o 4 mm | 23.3203 | 21.556 | 83.37 / 92.98 | 0.060 | 0.741 / 5.975 | 6.60 | 0.000427 |
| e8tex: texture anh that, o 4 mm | 23.3911 | 23.835 | 91.78 / 95.36 | 0.042 | 0.986 / 5.975 | 6.62 | 0.000427 |

- Texture anh that tot hon lap phang o moi chi so vet chan: PSNR vet chan +2.0 dB, do net 0.63 -> 0.99, trong 1 cm 89.1 -> 91.8 %; PSNR khong-vat
  +0.06 dB. Doi chung e8g4 cho thay loi do texture, khong do o nho. Alpha ma va % con giong vat khong doi.
- Do net van thap hon ty so cua anh that (5.975; ty so nay tinh tren anh that co vat trong vet chan nen chi la moc tham khao). 5.975 khac 1.82 cua
  bao cao S10 v2 vi vet chan o day lay tu Gaussian vat cua S4c/ctrl (nhan SAM 3), S10 v2 lay tu S3 v1.
- Chua chay lai S9 / S10 day du voi nen e8tex (can bo cuc run v1 / v2 day du tren Volume moi).

### TN3. Cong SAM 3 khong huan luyen: Relative Top-1 + Neighbor Continuity (`tools/sam3_gate_eval.py`, may local)

Du lieu: `Data/teatime/sam3_masks/probe/lerf_probs.npz` (mask + diem moi instance; 15 cap view-prompt co GT LERF-Mask, view test 0 va 2),
GT `Data/benchmarks/lerf_mask_dl/x/teatime/test_mask`, cung ham IoU / Boundary IoU voi `tools/sam3_lerf_eval.py`.
Cong: top-1 giu neu diem >= floor; instance khac giu neu diem >= rho x diem top-1 va (neu bat continuity) mask cham mask top-1 no 2 % duong cheo.
Bang day du (26 cau hinh): `Data/teatime/sam3_masks/probe/gate_eval.json`, `logs/sam3_gate_eval.log`.

| cong | mIoU | mBIoU | ly_tra (tea in a glass) | gau (stuffed bear) | khan_giay (paper napkin) |
|---|---|---|---|---|---|
| tuyet doi s 0.5 (mac dinh) | 77.6 | 75.9 | 48.3 | 98.9 | 0.0 |
| tuyet doi s 0.3 | 79.0 | 76.8 | 95.5 | 65.7 | 0.0 |
| rel-top1 floor 0.2, rho 0.5 (co / khong continuity nhu nhau) | 92.0 | 90.2 | 95.5 | 98.9 | 96.5 |
| rel-top1 floor 0.2, rho 0.3, khong continuity | 84.5 | 82.4 | 74.4 | 65.7 | 96.5 |
| rel-top1 floor 0.2, rho 0.3 + continuity | 91.9 | 90.2 | 95.5 | 98.9 | 96.5 |
| rel-top1 floor 0.3, rho 0.5 | 82.3 | 80.6 | 95.5 | 98.9 | 0.0 |

- ly_tra 48.3 -> 95.5 ma gau giu 98.9. Voi rho >= 0.5, rieng Relative Top-1 da du; Neighbor Continuity chi can khi rho thap (0.3).
- floor 0.1 va 0.2 cho cung diem; floor 0.3 lam mat khan giay (top-1 = 0.293).
- Gioi han: chi 15 cap, 2 view; LERF-Mask chi cham prompt CO trong view nen khong do duoc bao dong gia khi vat vang mat. Trong 2 cap vang mat co
  ung vien (view 2: bag of cookies top-1 0.100, stuffed bear 0.198), floor 0.2 khong bat cap nao, floor 0.1 bat ca 2. Chua ap dung vao mask video
  (`modal_sam3.py`) nen chua co ket qua 3D.

## Chot ha Winning Stack 2026-11-17

Chuoi chot: S3 SAM 3 (v3_sam3) -> khung S3m v3 -> S3i `i_r1v3k` -> mesh kin `mesh_final` (12/13 vat; vung_tra bi bo vi khung qua mong) -> MuJoCo; nen S4d `e8tex` (`--texture_fill`).

MuJoCo (`tools/scene_lift.py --com_frame 1 --drop_test coc_su --drop_cm 20`, do thi `support_graph/gauss_sam3_scalefree/graph.json`):
- Log `logs/lift_final_com.log`; ket qua `Data/teatime/pipeline/v3_sam3/lift_final_com/` (lift.json, lift.md, scene.xml, traj_lift_{dia,coc_su,tao}).
- Mesh: `Data/teatime/pipeline/v3_sam3/mesh_final/` (report.md, report.json). Vat de rieng tren ban: truot 0.00 mm, xoay <= 0.01 do, lun 0.03 mm (tui_banh truot 8.8e-5 mm).
- Nhac 10 cm: 9 vat dong, 9/9 len 9.986 - 9.989 cm, 0 canh bao phan ky, khop do thi 8/9. Dia mang banh_quy (khop). Coc_su khong mang thia (khong khop).
- De yen 1 s: banh_quy 5.65 mm, thia 130.62 mm, vat khac 0.00 mm. Thia van roi khoi coc (thieu cho do, da biet; so trung `lift_v4_com`).
- Coc_su nhac 20 cm roi tha 2 s: len 19.99 cm, lech ngang 0.0005 mm, lech doc ~0, nghieng 0.0007 do, NaN khong, canh bao phan ky 0, vat khac doi: thia 2.38 mm, banh_quy 0.16 mm.

Ca canh S9 (`s9final`, Modal profile quangkhaisaigonhcm, nen `e8tex` + S3i `i_r1v3k` + khung S3m v3, 23 view test, tu the t = 0):
- Log `logs/s9_final_v3_tex.log`; ket qua `Data/teatime/pipeline/v3_sam3/S9v3/final_v3_tex/` (`test_eval/final_v3_tex/metrics.json`, quy dao `traj_lift_{dia,coc_su,tao}/*.mp4`).
- Final: PSNR 22.937, SSIM 0.7849, LPIPS 0.2886. CoMe 30k goc: 23.345 / 0.8053 / 0.2629. Chenh -0.41 dB / -0.020 / +0.026.
- 1 757 093 Gaussian (nen 1 706 801 + 10 vat); 10 vat trong canh: coc_su, ly_tra, dia, banh_quy, tao, tui_banh, khan_giay, thia, vung_tra, goi_keo; cuu, gau khong nam trong danh sach.
- Luu y: quy dao render dung `lift_final_com` (mesh_final), nhung vat render la S3i `i_r1v3k` tren khung v3, khong phai mesh. Dung de so sanh voi `r_v5` (nen e8, chuoi v5_da3).

## v6_adap: mask SAM 3 cong Relative Top-1 -> S3 -> S3m -> S3c, so voi v3_sam3 (2026-11-18)

Chay tren Modal profile quangkhaisaigonhcm. Cung code, cung tham so cho moi vat; chi doi nguon mask.
- Mask: `modal_sam3.py::adaptive` (floor 0.2, rho 0.5), log `logs/v6_adap_sam3.log`, EXIT 0, 177/177 `labels/*.png` tren Volume
  `teatime/sam3_masks_adaptive/` (100 frame da co tu lan truoc, resume 77 frame). So frame co nhan: table 177, stuffed bear 144, plate 138,
  bag of cookies 130, cookies on a plate 126, coffee mug 117, apple 105, tea in a glass 98, puddle of liquid 98, spoon 93, blue snack package 75,
  sheep 71, paper napkin 31, small white cup 15.
- Hull: Volume quangkhaisaigonhcm KHONG co thu muc hull nao, nen dung lai tai may local: `tools/mask_hull.py --scheme sam3 --voxel_mm 3.24
  --min_views 15 --ratio 0.85` tren mask moi -> `Data/teatime/mask_hull_sam3_adap/` (put len `teatime/mask_hull_sam3_adap/`, 11 hull).
  Cung tham so voi `mask_hull_sam3` / `mask_hull_sam3_v2`; khong kiem duoc khung S3m v3 da dung thu muc hull nao (gioi han so sanh).
  Hull bi mat: khan_giay (paper napkin chi co nhan 31/177 frame, khong dat 15 phieu + 85 %), coc_kem_sua (15 frame). vung_tra con 12 voxel
  (0.4 cm3, v2: 4.7) va khong ra khung.
- Chuoi: `modal_pipeline.py::v6chain` (S3 -> S3m `v6_adap` -> S3c `r1v6`, lam_bound 0.1, 5000 iter), log `logs/v6_adap_chain.log`, EXIT 0.
  Doi chung `r1v3` da co san tren Volume tai `v3_sam3/S3c/r1v3` (khong co `r1v3_rerun`), v6chain tu bo qua.

### Khung S3m: the tich va dz (p98 chieu cao tren mat ban), cung cong thuc (`tools/build_mold_inspector.py`, summary json)

| vat | the tich v3 (cm3) | v6 (cm3) | dz p98 v3 (cm) | v6 (cm) | khuon v3 (cm3) | v6 (cm3) |
|---|---|---|---|---|---|---|
| coc_su | 537.6 | 454.6 | 11.0 | 11.6 | 5.7 | 17.7 |
| ly_tra | 398.4 | 522.9 | 9.9 | 11.6 | 6.2 | 4.9 |
| dia | 908.4 | 1078.5 | 2.7 | 3.4 | 99.1 | 83.1 |
| banh_quy | 319.5 | 374.1 | 2.9 | 3.7 | 43.7 | 32.7 |
| tao | 222.6 | 245.8 | 7.3 | 8.0 | 2.0 | 2.0 |
| tui_banh | 1365.4 | 1553.7 | 18.5 | 18.8 | 29.1 | 25.6 |
| khan_giay | 665.4 | - | 5.3 | - | 52.6 | - |
| thia | 3.5 | 5.5 | 15.4 | 15.3 | 0.0 | 0.0 |
| vung_tra | 1.3 | - | 0.2 | - | 0.0 | - |
| goi_keo | 78.4 | 73.2 | 1.3 | 1.8 | 8.1 | 0.1 |
| gau | 155330.8 | 155680.5 | 50.9 | 51.4 | 5097.5 | 6398.7 |
| cuu | 35523.4 | 37231.3 | 43.7 | 44.0 | 463.6 | 574.5 |

### Do chinh xac 3D khung bang LERF-Mask (`tools/frame_lerf_iou.py`, test_0 + test_2)

Ket qua `Data/teatime/gaussian_inspect_v6/frame_lerf_iou_v6.json` (v3: `frame_lerf_iou_v3.json`).

| prompt | v3 | v6 |
|---|---|---|
| apple | 87.6 | 84.4 |
| bag of cookies | 94.3 | 94.6 |
| coffee mug | 87.0 | 76.8 |
| cookies on a plate | 84.0 | 60.3 |
| paper napkin | 92.2 | khong co khung |
| plate | 87.8 | 81.8 |
| sheep | 88.9 | 87.1 |
| stuffed bear | 82.7 | 82.0 |
| tea in a glass | 81.2 | 93.6 |
| **mIoU** | **87.3** (9 prompt) | **82.6** (8 prompt) |

- Cung 8 prompt (bo paper napkin): v3 86.7, v6 82.6. Tinh paper napkin = 0 tren 9 prompt: v6 73.4.
- Chi tea in a glass tang ro (+12.4); cookies on a plate (-23.7), coffee mug (-10.2), plate (-6.0) giam. Cong thich nghi tot o muc mask 2D
  (TN3: 92.0 mIoU) nhung khong chuyen thanh khung 3D tot hon o chuoi nay.

### S3c `r1v3` (khung v3) vs `r1v6` (khung v6_adap), 23 view test, lam_bound 0.1, 5000 iter

| | r1v3 truoc | r1v3 sau | r1v6 truoc | r1v6 sau |
|---|---|---|---|---|
| PSNR | 23.342 | 23.343 | 23.345 | 23.352 |
| SSIM | 0.8053 | 0.7991 | 0.8053 | 0.7988 |
| LPIPS | 0.2629 | 0.2648 | 0.2629 | 0.2652 |
| Gaussian vat | 164 577 | 159 127 | 144 745 | 134 082 |
| Gaussian ngoai khung | 9 318 (5.66 %) | 3 734 (2.35 %) | 11 694 (8.08 %) | 931 (0.69 %) |
| tong opacity ngoai khung | 8075.1 | 2509.1 | 10409.1 | 89.0 |
| alpha vat trong / ngoai mask | 0.986 / 0.0033 | 0.968 / 0.0033 | 0.928 / 0.0247 | 0.890 / 0.0204 |

- Loc khoi dong / sau: r1v3 bo 3243 khong quan sat + 392 khoi dong, cat 5450 sau; r1v6: 2740 + 608, cat 10663.
- Alpha trong / ngoai mask tinh theo mask cua chinh run do (v3 vs adaptive) nen khong so truc tiep duoc.
- frame_check (% ngoai khung theo vat, sau): r1v6 cao hon r1v3 o goi_keo 19.6 vs 1.9, gau 14.8 vs 6.5, coc_su 4.6 vs 0.4; thap hon o ly_tra
  0.9 vs 8.0, tui_banh 0.2 vs 4.0, thia 0.4 vs 4.5. r1v6 khong co khan_giay, vung_tra (khong co khung); r1v3 vung_tra 96.3 % (khong rang buoc).

### Orbit 72 frame + Gaussian tung vat (`modal_pipeline.py::inspect`, tag v6_vs_v3, nguong ngoai khung 0.5 cm)

Log `logs/v6_adap_inspect.log`, EXIT 0. Gaussian / ngoai khung / % / ngoai khung co opacity > 0.1:

| vat | v3_sam3 (r1v3) | v6_adap (r1v6) |
|---|---|---|
| ly_tra | 2003 / 101 / 5.04 / 65 | 1747 / 8 / 0.46 / 3 |
| coc_su | 5105 / 11 / 0.22 / 0 | 4084 / 30 / 0.73 / 0 |
| dia | 8249 / 93 / 1.13 / 6 | 8140 / 88 / 1.08 / 0 |
| banh_quy | 9019 / 3 / 0.03 / 0 | 9054 / 0 / 0.00 / 0 |
| tao | 2056 / 6 / 0.29 / 3 | 2069 / 0 / 0.00 / 0 |
| tui_banh | 8957 / 12 / 0.13 / 5 | 8887 / 13 / 0.15 / 1 |
| goi_keo | 1712 / 16 / 0.93 / 2 | 1418 / 144 / 10.16 / 32 |
| thia | 244 / 10 / 4.10 / 1 | 234 / 0 / 0.00 / 0 |
| khan_giay | 3379 / 13 / 0.38 / 2 | - |
| tat ca (tru gau, cuu) | 43692 / 3118 / 7.14 / 2873 | 35633 / 283 / 0.79 / 36 |

File (deu kiem tra khong rong):
- Frame orbit: `Data/teatime/gaussian_inspect_v6/v6_vs_v3/<vat>/f_000..071.jpg` (10 thu muc x 72), `<vat>_sheet.jpg`, `stats.json`.
- Video (72 frame, 24 fps, 3.0 s, tren: render, duoi: ngoai khung to do): `Data/teatime/gaussian_inspect_v6/turntable_{all,banh_quy,coc_su,dia,goi_keo,khan_giay,ly_tra,tao,thia,tui_banh}.mp4`.
- Gaussian tung vat: `.../v6_vs_v3/gauss/{v3_sam3,v6_adap}/<vat>.npz` (9 + 8) va PLY tach tu final.ply theo oid_final:
  `Data/teatime/gaussian_inspect_v6/ply/r1v3/` (12 vat) va `ply/r1v6/` (10 vat) (`.tmp_scripts/split_obj_ply.py`).
- `Data/teatime/mold_inspector_360.html` dung lai: truoc = khung S3m v3 (v3_sam3), sau = khung S3m v6_adap, lop Gaussian r1v3 va r1v6
  (gau / cuu khong co lop Gaussian vi inspect khong render 2 vat nay). Ban cu (co lop Amodal3R / TRELLIS / mesh_v4/v5/v5g):
  `Data/teatime/mold_inspector_360.pre_v6.html`.

Ket luan: v6_adap giam manh Gaussian ngoai khung sau S3c (931 vs 3734) voi PSNR / SSIM / LPIPS gan nhu nhau, nhung khung 3D kem hon tren
LERF-Mask (82.6 vs 86.7 cung 8 prompt) va mat han khan_giay, vung_tra. Chua nen thay v3 bang v6 lam mac dinh.

## S4e: lap be mat cong co vat do (kieu 3DGIC), dia va mat ban, so voi e8tex (2026-11-18)

Modal profile quangkhaisaigonhcm. Code: `pipeline/s4e_surface_fill.py` (viet lai), `pipeline/s4d_cutfill.py --surface_fill 1`, `modal_s4e.py`.
Cung quy tac moi vat do: khung cuc bo (PCA vat do, hoac mat phang ban trong support.json) -> vet chan = o 4 mm chua tam Gaussian vat bi nhac ->
be mat tham chieu tu Gaussian vat do o VANH vet chan (moi o: diem cao nhat, hoac `--ring_layer_cm 0.25`: trung vi lop tren day 2.5 mm) ->
h(u, v) = da thuc bac 2 (2 vong bo diem > 2.5 sigma) + IDW phan du -> Gaussian lap det tai h(u, v), XOAY THEO PHAP TUYEN cuc bo (ban truoc ghi
rot = don vi nen dia det nam theo truc the gioi) -> mau 3DGIC (nhan SAM 3 vat do + |z - D_render| <= 2.5 cm, >= 2 view, trung vi; vung khong
thay: chep tinh tien tren mat cong, ty le thay >= 0.9) -> (dia) cat Gaussian vat do trong vet chan nong 2 o cao hon mat > -0.5 cm (giu mat duoi)
-> tinh chinh masked 1000 iter (loss chi o pixel nhan dia co 2 px, ngoai mask banh_quy nong 15 px; khong dong bang).
Kiem tra tong hop (`.tmp_scripts/test_s4e_geo.py`, dia cong nghieng 0.3 rad, day 0.3): sai so cao do 0.000 / 0.024 (lop 0.1), phap tuyen
0.0 / 0.5 do (p95), mat duoi khong bi cat.
Do tren 23 view test (20 view co vet chan >= 50 px): hole_pct = % pixel alpha < 0.5 trong (mask banh_quy co 3 px) giao (hinh chieu Gaussian lap
render opacity 1); psnr_support_visible = PSNR pixel nhan dia ngoai mask banh_quy nong 15 px (render rieng dia, nen den); psnr_footprint_visible
= PSNR trong hinh chieu vet chan o pixel anh that la dia.

### Dia bo banh_quy (`S4e/dia_banhquy_ft`, `dia_banhquy_ft_l`), log `logs/s4e_dia_banhquy_ft{,_l}.log`, EXIT 0

Lap: 688 o vet chan, 1322 Gaussian lap, 702 thay truc tiep (trung vi 7.5 view), 620 chep tu 3/3 vung, 0 gan nhat; cat 888 / 1058 Gaussian dia trong
vung cat (mat duoi giu); vanh 860 diem, RMS khop 0.107 cm; mat lap thap hon mat phang PCA 0.81 cm (trung vi), nghieng phap tuyen p50/p95 7.4 / 12.3 do.

| dia_banhquy_ft (lop _l) | truoc | chi cat | cat + lap | + tinh chinh 1000 |
|---|---|---|---|---|
| hole_pct (vet chan) | 0.593 (0.592) | 0.657 (0.656) | 0.000 (0.000) | 0.000 (0.000) |
| psnr_support_visible | 18.928 | 18.926 | 18.925 | 21.222 (21.277) |
| psnr_footprint_visible | 25.211 (25.198) | 22.535 (22.498) | 22.617 (22.646) | 21.944 (21.954) |

- Dia cua CoMe da co Gaussian duoi banh_quy nen % thung truoc chi 0.59 %; lap dua ve 0.00 %. Van de chinh khong phai thung ma la texture banh
  in len dia (anh `preview/frame_00009.jpg`, cot gt | truoc | cat | cf | ft): cot "truoc" va "cat" con vet banh nau / xanh; cf / ft gan trang,
  con vien banh mo. Mot vet ma sang duoi dia co san o moi cot (khong do S4e).
- Tinh chinh masked tang PSNR dia thay duoc +2.3 dB (18.93 -> 21.22) nhung PSNR trong vet chan o pixel thay dia giam 25.21 -> 22.62 (lap) -> 21.94 (ft):
  o 4 mm + chep tinh tien muot hon Gaussian dia goc o sat mep banh. Ban cu (v8, rot don vi, khong cat, khong ft): PSNR dia 20.79 -> 20.73,
  dinh nghia mask khac (toan bo nhan dia) nen khong so truc tiep.
- `_l` (lop 2.5 mm) gan nhu trung ban max (khac <= 0.06 dB).

### Mat ban bo 10 vat (`S4e/table_all`, `table_all_l`) vs `S4d/e8tex`

Chay trong `s4d_cutfill.py` voi MOI tham so cua e8tex (cut_dilate 2, reach 2, low_cm 3, fill_dilate 4, o 4 mm, bao ve cuu / gau, ft masked 3000,
dong bang vung lap, tex_scale 0.5, L40S), chi doi buoc lap -> so sanh truc tiep. Vet chan = `S4c/ctrl/S3like/gaussians/objects_all.ply` nhu e8tex
(cac vat tren ban tru gau / cuu). Cat 64 354, bao ve 6 556, 14 191 Gaussian lap: giong het e8tex.
Mau: e8tex 7 962 thay / 6 229 chep (20 vung); S4e 7 644 / 6 547 (13 vung), _l 7 695 / 6 496; 0 gan nhat ca ba.

| sau tinh chinh (23 view test, 22 dung) | e8tex | S4e table_all | S4e table_all_l |
|---|---|---|---|
| PSNR khong-vat | 23.3911 | 23.3688 | 23.3660 |
| PSNR vet chan (pixel thay ban) | 23.835 | 24.076 | 24.059 |
| % trong 1 cm / 2 cm | 91.78 / 95.36 | 91.39 / 95.38 | 92.16 / 95.40 |
| trung vi cach mat phang RANSAC (cm) | 0.042 | 0.470 | 0.426 |
| do net / anh that | 0.986 / 5.975 | 1.051 / 5.975 | 1.085 / 5.975 |
| con giong vat % | 6.62 | 6.41 | 6.42 |
| PSNR cuu / gau | 24.679 | 24.678 | 24.672 |
| ghost Gaussian > 0.5 / 1.5 / 3 cm (s10_ghost) | 1801 / 1777 / 1636 | 2484 / 1777 / 1636 | 1811 / 1778 / 1637 |
| alpha ma TB | 0.000427 | 0.000427 | 0.000427 |

Truoc tinh chinh (cf): PSNR vet chan 22.719 / 22.933 / 22.961, khong-vat 22.957 / 22.966 / 22.968.
- S4e tot hon e8tex o texture vet chan (+0.24 dB sau ft, +0.21..0.24 truoc ft), do net (1.05..1.09 vs 0.99), con giong vat (6.41 vs 6.62);
  kem 0.02 dB o PSNR khong-vat.
- Mat lap S4e nam cao hon mat phang RANSAC 0.38..0.42 cm (p5..p95 0.25..0.50). Nen goc chua cat (src) cung cach mat phang 0.453 cm (trung vi)
  -> ban tai cho vat dung that su cao hon mat phang toan cuc ~4 mm; chi so "cach mat phang" uu tien e8tex theo dinh nghia. Ban max (table_all)
  lam 683 Gaussian lap vuot nguong ghost 0.5 cm; ban lop (_l) tra ve 1811 (e8tex 1801).
- Chua chay S9 / S10 day du voi nen S4e.

File (deu kiem tra khong rong): `Data/teatime/pipeline/v1/S4e/{dia_banhquy_ft,dia_banhquy_ft_l}/{metrics.json,fill.json,preview/*.jpg (4)}`,
`Data/teatime/pipeline/v1/S4e/{table_all,table_all_l}/{metrics.json,cutfill.json,ghost/ghost.json,preview/*.jpg (3)}`.

## v7_adap: cong SAM 3 co tre thoi gian + bao ve khung theo Gaussian that (2026-11-18)

Modal profile quangkhaisaigonhcm (SAM 3 chay duoc: `hf_cache/hub/models--facebook--sam3` co tren Volume).

### Chan doan v6 (gate.json / legend.json keo lai tu Volume, md5 trung ban local; `.tmp_scripts/cmp_adap_v3.py`)

So frame top1 >= nguong (177 frame) theo prompt:

| prompt | top1 = 0 | >= 0.20 | >= 0.12 | >= 0.10 | >= 0.05 |
|---|---|---|---|---|---|
| paper napkin | 89 | 31 | 49 | 55 | 88 |
| coffee mug | 39 | 117 | 126 | 128 | 138 |
| spoon | 63 | 93 | 101 | 107 | 114 |
| cookies on a plate | 48 | 126 | 129 | 129 | 129 |
| plate | 34 | 138 | 140 | 141 | 143 |

- Ha floor chi giup paper napkin (31 -> 55 o 0.10, toi da 88 vi 89 frame khong co phat hien nao o nguong 0.05 cua Sam3Processor).
- banh_quy KHONG mat vi cong: mask v6 vs v3 (video) trung 126 frame, IoU trung vi 0.976, dien tich 1.01x; hull adaptive 734.6 vs 691.5 cm3.
  khan_giay: 82.5 % pixel v3 khong co nhan trong v6 (74 frame chi v3 co). coc_su: hull 468 vs 814 cm3 (19 % pixel v3 khong nhan; v3 gop
  "coffee" vao coc bang MERGE tay, ban adaptive khong co prompt do).
- Nguyen nhan banh_quy / dia / coc_su giam la S3m, khong phai mask: khung v6 to hon v3 (voxel dia +19 %, banh_quy +17 %, ly_tra +31 %).
  Cung mask v3, code S3m hien tai: `v3_sam3/S3m/v4` (protect_vis 3) cho banh_quy 59.1, `v4_p0` (protect_vis 0) 84.6
  (`tools/frame_lerf_iou.py`). v3 goc (87.3) lam truoc khi co bao ve mat mong. Bao ve (>= 3 view thay be mat) giu ca voxel trong trong dai
  dung sai do sau phia tren vat det -> khung phong len.

### Thay doi (opt-in, mac dinh giu hanh vi cu)

- `modal_sam3.py::adaptive --low_floor 0.05 --hyst_overlap 0.5 --hyst_radius 1`: raw luu mask moi prompt co top1 >= 0.05; phat hien yeu
  (0.05 <= top1 < 0.2) chi duoc nhan neu frame ke (+-1) co phat hien manh cung prompt va >= 50 % mask yeu nam trong mask manh nong 2 % duong
  cheo (= quy tac continuity cua TN3). Khong noi suy mask khi frame khong co phat hien.
- `pipeline/s3m_mold.py --protect_matter 1 --protect_op 0.5`: voxel chi duoc bao ve neu cach <= 1 voxel mot tam Gaussian CUA CHINH VAT
  (opacity > 0.5). Ghi `protected_voxels_before_matter_gate`.
- `modal_pipeline.py::v6chain`: them `--protect-vis/--protect-matter/--ablate/--stop-after-s3m`; `configs/pipeline/teatime_v7_adap.json`.

### Ket qua

Mask `teatime/sam3_masks_v7` (log `logs/v7_adap_sam3.log`, EXIT 0, 177/177 labels): tre nhan / tu choi: paper napkin 8 / 49, stuffed bear 8 / 9,
sheep 6 / 1, bag of cookies 3 / 6, coffee mug 2 / 19, spoon 1 / 20, cookies on a plate 2 / 1. So frame co nhan v6 -> v7: paper napkin 31 -> 39,
stuffed bear 144 -> 152, sheep 71 -> 77, coffee mug 117 -> 119, cookies on a plate 126 -> 128. Hull `mask_hull_sam3_v7` (local, cung tham so v6,
put len Volume): gan nhu v6; khan_giay van 0 voxel (39 frame < dieu kien 15 phieu + 85 %), vung_tra 2.1 cm3 (v6 0.4).
Chuoi: S3 `v7_adap` (2.6 phut) -> S3m 3 bien the cung hull (log `logs/v7_adap_chain_s3m.log`; tien trinh local bi dung o gioi han 10 phut
nhung ca 3 S3m da xong va commit) -> S3c `r1v7` (lam_bound 0.1, 5000 iter, log `logs/v7_adap_chain.log`, EXIT 0).

LERF frame IoU (`tools/frame_lerf_iou.py`, test_0 + test_2, mIoU tren 8 prompt co khung; paper napkin khong co khung o v6 / v7):

| khung S3m | mIoU 8 prompt | apple | bag of cookies | coffee mug | cookies on a plate | plate | sheep | stuffed bear | tea in a glass |
|---|---|---|---|---|---|---|---|---|---|
| v3 (mask v3, S3m cu) | 86.7 (9 prompt: 87.3) | 87.6 | 94.3 | 87.0 | 84.0 | 87.8 | 88.9 | 82.7 | 81.2 |
| v6_adap (protect 3) | 82.6 | 84.4 | 94.6 | 76.8 | 60.3 | 81.8 | 87.1 | 82.0 | 93.6 |
| v7_p3 (mask v7, protect 3) | 82.7 | 84.2 | 94.6 | 77.0 | 60.9 | 82.2 | 86.8 | 82.5 | 93.6 |
| v7_p0 (mask v7, khong bao ve) | 85.4 | 87.1 | 95.1 | 76.9 | 83.5 | 88.1 | 88.1 | 82.8 | 81.4 |
| **v7_adap (mask v7, protect 3 + matter)** | **85.2** | 84.3 | 94.7 | 76.9 | 77.6 | 87.3 | 87.2 | 82.6 | 90.9 |

Khung (cm3) p0 / p3 / matter: dia 749 / 1056 / 833, banh_quy 272 / 366 / 336, ly_tra 371 / 517 / 428, coc_su 381 / 446 / 433; voxel duoc bao ve
p3 -> matter: dia 9076 -> 2475, ly_tra 4320 -> 1675, banh_quy 2785 -> 1882. p98 chieu cao khung / Gaussian dia 2.97 / 2.84 cm (p3 3.41).

S3c, 23 view test:

| | r1v3 sau | r1v6 sau | r1v7 sau |
|---|---|---|---|
| PSNR / SSIM / LPIPS | 23.343 / 0.7991 / 0.2648 | 23.352 / 0.7988 / 0.2652 | 23.352 / 0.7989 / 0.2645 |
| Gaussian vat | 159 127 | 134 082 | 145 170 |
| ngoai khung (metrics.json) | 3 734 (2.35 %) | 931 (0.69 %) | 3 425 (2.36 %) |
| ngoai khung, bo nhom vung_tra (khong rang buoc) | 872 / 156 159 (0.56 %) | 931 / 134 082 (0.69 %) | 629 / 142 350 (0.44 %) |
| alpha vat trong / ngoai mask | 0.968 / 0.0033 | 0.890 / 0.0204 | 0.932 / 0.0113 |

r1v7: bo 2865 khong quan sat + 580 khoi dong, cat 9614 sau; tong opacity ngoai khung 2444.8. Theo vat (% ngoai khung): goi_keo 8.98, coc_su 0.93,
ly_tra 0.85, dia 1.11, gau 0.39, cuu 0.24, vung_tra 99.15 (khong rang buoc).

- So sanh "2.35 % -> 0.69 %" cua v6 chu yeu do v6 KHONG co nhom vung_tra (khung v3 / v7 cua vung_tra qua nho nen S3c khong rang buoc, 96-99 %
  Gaussian cua no ngoai khung). Bo nhom do: r1v3 0.56 %, r1v6 0.69 %, r1v7 0.44 %.
- v7 KHONG dat dieu kien (>= 87 mIoU va < 1.5 % ngoai khung): mIoU 85.2; ngoai khung 2.36 % theo dinh nghia cu (0.44 % neu bo nhom khong rang buoc).
  Khong chay inspect / turntable `gaussian_inspect_v7` (dieu kien buoc 4).
- Con thieu so voi v3: coffee mug 76.9 vs 87.0 (hull coc_su nho: thieu phan ca phe ma v3 gop tay), paper napkin khong co khung (39 frame).
  Tre thoi gian +-1 frame chi them 8 frame khan_giay vi 49 phat hien yeu khong co frame manh lien ke.

File (deu kiem tra khong rong): `Data/teatime/sam3_masks_v7/{labels (177), preview, gate.json, legend.json, frames.json}`,
`Data/teatime/mask_hull_sam3_v7/`, `Data/teatime/pipeline/v7_adap/S3m/{v7_adap,v7_p3,v7_p0}/{frame (22), metrics.json}`,
`Data/teatime/pipeline/v7_adap/S3c/r1v7/{metrics.json, metrics_before.json, preview/*.jpg (3)}`,
`Data/teatime/gaussian_inspect_v7/frame_lerf_iou_{v7_adap,v7_p3,v7_p0}.json` + `frame_iou_*/`.

## v8_hullv4: visual hull v4 (mask SAM 3 thich nghi) -> S3m -> S3c (2026-11-18)

Modal profile quangkhaisaigonhcm. Hull `Data/teatime/mask_hull_sam3_adap_v4/` (`tools/mask_hull.py --adaptive_min_views 1
--soft_empty_weight 0.0 --clean_3d 1 --sfm_anchor 1`, nhan `sam3_masks_adaptive/labels`) put len Volume `teatime/mask_hull_sam3_adap_v4/`
(12 `hull_<vat>.ply` + `stats.json`, kich thuoc trung local; bo `hull_objects.ply` / `hull_consistency.ply` vi S3m khong doc).
Config `configs/pipeline/teatime_v8_hullv4.json` (= v6_adap, run_name `v8_hullv4`). Lenh:
`modal run modal_pipeline.py::v6chain --config configs/pipeline/teatime_v8_hullv4.json --mold-tag v8_hullv4 --hull-rel mask_hull_sam3_adap_v4
--s3c-tag r1v8 --lam-bound 0.1 --iters 5000 --protect-vis 3 --protect-matter 1` (log `logs/v8_hullv4_chain.log`, EXIT 0).

### Loi voxel va sua (`pipeline/s3m_mold.py`)

- Lan chay dau (log `logs/v8_hullv4_chain_badvox.log`): S3c chet o `FrameDist` (`_ArrayMemoryError`, can 3.77 PiB). Nguyen nhan: toa do hull v4
  khong nam dung tren luoi (nhieu float32 ~1e-5 giua cac cot x; hull v7 cach deu 0.01311), con S3m uoc luong voxel = khoang cach nho nhat giua
  cac x khac nhau -> voxel 0.00025 cm thay vi 0.326 cm cho dia, banh_quy, tui_banh, gau, cuu. Voxel sai di vao `frame/*.npz` va lam hong ca
  khoet khung lan IoU (mIoU 87.3 cua lan do KHONG hop le, luu `frame_lerf_iou_v8_badvox.json` de doi chieu).
- Sua: bo cac khoang cach < 0.5 x trung vi truoc khi lay min. Tren 7 hull v7 da thu cho dung gia tri cu (0.01311), tren hull v4 cho 0.01311
  cho moi vat; lan chay lai log voxel_cm 0.326 cho ca 12 vat. Da xoa `S3m/v8_hullv4` + `S3c/r1v8` loi tren Volume truoc khi chay lai.

### LERF frame IoU (`tools/frame_lerf_iou.py`, test_0 + test_2)

| khung S3m | mIoU 9 prompt | mIoU 8 prompt (bo napkin) | apple | bag of cookies | coffee mug | cookies on a plate | paper napkin | plate | sheep | stuffed bear | tea in a glass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v3 | 87.3 | 86.7 | 87.6 | 94.3 | 87.0 | 84.0 | 92.2 | 87.8 | 88.9 | 82.7 | 81.2 |
| v6_adap | - (khong co khung napkin) | 82.6 | 84.4 | 94.6 | 76.8 | 60.3 | - | 81.8 | 87.1 | 82.0 | 93.6 |
| v7_adap | - (khong co khung napkin) | 85.2 | 84.3 | 94.7 | 76.9 | 77.6 | - | 87.3 | 87.2 | 82.6 | 90.9 |
| **v8_hullv4** | **85.9** | **85.1** | 84.8 | 93.8 | 77.4 | 76.8 | 92.2 | 86.9 | 87.9 | 82.5 | 90.9 |

Khung (hull cm3 / frame cm3) v7 -> v8: dia 2167/833 -> 2248/854, banh_quy 568/336 -> 587/349, ly_tra 710/428 -> 759/455, coc_su 460/433 ->
497/466, khan_giay 0 -> 1307/924, vung_tra 0 -> 131/61, goi_keo 101/64 -> 140/82, cuu 40980/35498 -> 43170/36530, gau 211420/159649 ->
223001/168046.

### S3c r1v8 (23 view test, lam_bound 0.1, 5000 iter)

| | r1v3 sau | r1v7 sau | **r1v8 sau** |
|---|---|---|---|
| PSNR / SSIM / LPIPS | 23.343 / 0.7991 / 0.2648 | 23.352 / 0.7989 / 0.2645 | **23.351 / 0.7990 / 0.2649** |
| Gaussian vat | 159 127 | 145 170 | 137 570 |
| ngoai khung (metrics.json, tong) | 3 734 (2.35 %) | 3 425 (2.36 %) | **386 (0.28 %)** |
| ngoai khung, bo nhom vung_tra | 872 / 156 159 (0.56 %) | 629 / 142 350 (0.44 %) | **316 / 135 193 (0.23 %)** |
| alpha vat trong / ngoai mask | 0.968 / 0.0033 | 0.932 / 0.0113 | 0.897 / 0.0229 |

r1v8 truoc S3c: 10 601 / 147 729 ngoai khung (7.18 %). Bo 2779 khong quan sat + 609 khoi dong, cat 10 159 sau; tong opacity ngoai khung 20.4
(r1v7 2444.8). Theo vat (% ngoai khung sau): dia 1.13, ly_tra 0.74, goi_keo 0.70, coc_su 0.59, cuu 0.19, gau 0.16, tui_banh 0.06, banh_quy 0.01,
tao / thia / khan_giay 0, vung_tra 2.94 (nay CO rang buoc: frame_check 18.3 %, constrained=true; r1v7 99.1 %, constrained=false).

- v8 dat dieu kien ngoai khung (< 1.5 %: 0.28 % tong, 0.23 % bo vung_tra) va la lan dau vung_tra / khan_giay co khung. mIoU 85.9 (9 prompt) /
  85.1 (8 prompt): ngang v7, duoi nguong 87 va duoi v3 (87.3 / 86.7). Khoang cach van nam o coffee mug (77.4 vs 87.0) va cookies on a plate
  (76.8 vs 84.0).
- alpha vat trong mask giam (0.897 so voi 0.932 o r1v7): S3c ep chat hon thi vat mong di trong anh. khan_giay chi co 166 Gaussian (nhan S3 tu
  mask adaptive), du khung 924 cm3.
- Hull gau / cuu van qua lon (bbox gau ~96 x 91 x 100 cm, giong v7); hull v4 cuu 43 170 cm3 vs v7 40 980 nen loc cum ma bang SfM khong lam
  giam the tich tong.

File (deu kiem tra khong rong): `Data/teatime/pipeline/v8_hullv4/S3m/v8_hullv4/{frame (24), metrics.json, preview}`,
`Data/teatime/pipeline/v8_hullv4/S3c/r1v8/{metrics.json, metrics_before.json, preview/*.jpg (3)}`,
`Data/teatime/gaussian_inspect_v6/frame_lerf_iou_v8.json` + `frame_iou_v8/`; doi chieu loi: `*_badvox`.

## Track A: S3c + L_leak / L_cover (2026-11-18, Modal quangkhaisaigonhcm)

Code: `pipeline/s3c_constrain.py` them co opt-in `--lam_leak`, `--lam_cover`, `--aux_every` (mac dinh 0 = nhu cu). Moi `aux_every` = 2 buoc:
chon ngau nhien 1 vat co khung dang tin va co mask SAM 3 >= 200 px tren view train dang dung, render alpha RIENG vat do (Gaussian khac van che,
mau 0), `L_leak = mean(a_i ngoai mask_i no 15 px)`, `L_cover = mean((1 - a_i) trong mask_i co 3 px)` (trung binh tren vung mask, khop cach do
leak / cover; ban tren ca anh lam L_cover qua nho). Gradient cua 2 hang nay chi giu cho Gaussian cua vat i (vat khac khong duoc che bot de ha loss).
ID mask lay tu `groups.json` cua S3 (bo bang SAM3_IDS viet cung). Chay: `modal run modal_pipeline.py::leakrun` (log `logs/r1v8_leak*.log`).
r1v8_leak / r1v8_leak_s dung lai start.ply / start_f.ply / metrics_before cua r1v8 (cung khung v8_hullv4, cung loc khoi dong) nen 'before' trung r1v8
va startup_pruned ghi 0 trong metrics.json (that ra = 609 nhu r1v8). Thu tu view ngau nhien khac r1v8 (rng dung chung) nen +-0.05 dB co the la nhieu.

| 23 view test, sau S3c (lam_bound 0.1, 5000 iter) | PSNR / SSIM / LPIPS | outside_pct | leak (alpha ngoai mask) | cover (alpha trong mask) |
|---|---|---|---|---|
| r1v3 (khung v3) | 23.343 / 0.7991 / 0.2648 | 2.35 % | 0.0033 | 0.968 |
| r1v8 (khong aux) | 23.351 / 0.7990 / 0.2649 | 0.28 % | 0.0229 | 0.897 |
| **r1v8_leak** (leak 0.05, cover 0.02) | 23.406 / 0.7997 / 0.2639 | 0.28 % | 0.0249 | 0.903 |
| r1v8_leak_s (leak 0.5, cover 0.2) | 23.352 / 0.7989 / 0.2649 | 0.31 % | 0.0304 | 0.922 |

Cover theo vat (r1v8 -> r1v8_leak -> r1v8_leak_s): coc_su 0.89/0.90/0.93, ly_tra 0.83/0.88/0.97, dia 0.97/0.97/0.97, banh_quy 0.98/0.97/0.97, tao 0.97/0.96/0.93, tui_banh 0.93/0.92/0.93, khan_giay 0.07/0.15/0.27, thia 0.69/0.71/0.70, vung_tra 0.87/0.88/0.94, goi_keo 0.92/0.94/0.92, gau 0.71/0.71/0.73, cuu 0.90/0.91/0.92

- Leak theo tung vat (alpha rieng vat i ngoai mask i no 15 px) o r1v8 da ~1e-6 .. 1e-3 (gau lon nhat 1.3e-3), nen L_leak 0.05 gan nhu khong co tac dung;
  leak tong (alpha moi Gaussian vat ngoai hop mask moi vat) tang nhe 0.0229 -> 0.0249 -> 0.0304: ep cover lam vat day len ca o mep.
- L_cover la phan co tac dung: cover 0.897 -> 0.903 (0.02) -> 0.922 (0.2); ly_tra 0.83 -> 0.97, khan_giay 0.07 -> 0.27. PSNR khong giam (23.35 - 23.41),
  outside_pct giu 0.28 % (0.02) / 0.31 % (0.2). Chua ve muc r1v3 (0.968 / 0.0033) vi khung v8 chat hon nhieu (r1v3 2.35 % Gaussian ngoai khung).
- Danh doi cover / leak la that; lam_cover 0.2 la diem tot hon cho cover nhung leak tong x1.3. Chua thu lam_leak lon voi cover nho.

File: `Data/teatime/pipeline/v8_hullv4/S3c/{r1v8_leak,r1v8_leak_s}/metrics.json`, `r1v8_leak/preview/*.jpg`.

## Track B: S3i --image_inpaint (2026-11-18)

Code: `pipeline/s3i_mold_fill.py --image_inpaint 1` (`modal run modal_pipeline.py::imgfill`). Sau khi lap khuon nhu cu + bo loc nhin thay:
camera ao quanh tung vat (elev -60/-15/15/45, az 22.5 + 45k: TACH khoi 40 camera do hole %), render vat truoc lap, lo = pixel trong bong khung co
alpha < 0.5; anh dich = OpenCV Telea (LaMa khong co trong container) chi tu pixel vat thay duoc (ngoai bong cung la chua biet de nen khong loang).
Toi uu f_dc / opacity / scale / rotation cua RIENG Gaussian lap 600 buoc (L1 mau trong bong + 0.5 (1 - alpha) trong bong), Gaussian goc dong bang,
vi tri giu o voxel khuon (d_out = 0), scale <= 1 voxel. Khong co chieu mau da view va khong co loss tren 154 view that: anh that duoc giu bang bo
loc nhin thay (vi tri khong doi) va kiem lai bang PSNR test. Doi chung i_r1v8_base = cung base r1v8_leak, khong inpaint.

| | base S3c | PSNR test truoc -> sau lap | doi trung binh | hole % camera ao (10 vat) truoc -> sau |
|---|---|---|---|---|
| i_r1v3k (khung v3) | r1v3 | 23.341 -> 23.286 | 7.5e-04 | 3.05 -> 1.35 |
| i_r1v8_base | r1v8_leak | 23.406 -> 23.354 | 7.2e-04 | 6.66 -> 4.43 |
| **i_r1v8_img** | r1v8_leak | 23.406 -> **23.358** | 5.4e-04 | 6.66 -> **4.06** |

Hole % sau theo vat (base / img): coc_su 0.73/0.74, ly_tra 1.07/1.07, dia 0.69/0.79, banh_quy 4.88/4.64, tao 0.14/0.14, tui_banh 0.23/0.23, khan_giay 14.04/11.62, goi_keo 3.29/3.29, gau 16.76/16.09, cuu 2.46/2.04

- i_r1v8_img: PSNR test 23.358 (> i_r1v3k 23.286, +0.07 dB), hole % 4.43 -> 4.06 so voi doi chung cung base (-8 % tuong doi); lon nhat o khan_giay
  14.0 -> 11.6, cuu 2.46 -> 2.04, gau 16.8 -> 16.1.
- Hole % KHONG so truc tiep duoc voi i_r1v3k (1.35): bong khung = chieu khung cua chinh run; khung v3 to hon nhieu (r1v3 2.35 % Gaussian ngoai khung)
  va lap khuon v3 co nhieu voxel hon. Phan lo con lai cua gau (16 %) / khan_giay (12 %) nam ngoai voxel khuon chua quan sat (alpha cuoi van
  thieu 0.22 / 0.06): inpaint chi to mau / lam day Gaussian lap, khong them Gaussian o cho da quan sat nhung mong.

File: `Data/teatime/pipeline/v8_hullv4/S3i/{i_r1v8_img,i_r1v8_base}/{metrics.json, preview/*.jpg}`.

## Track C: canh thu hai LERF figurines (2026-11-18, Modal quangkhaisaigonhcm)

Du lieu: `Data/figurines/prep/figurines_prep.zip` = COLMAP da undistort (PINHOLE 994 x 738, 1 camera) 299 anh `images/000NN.JPEG` + sparse/0
(images.bin ghi ten `000NN.jpg` -> khong khop duoi file) + depth Murre (khong dung) + points3D.ply 3.1 GB (khong dung). `tools/prep_scene_colmap.py`
chuan hoa thanh `Data/figurines/colmap/{images/frame_000NN.jpg, sparse/0}` (doi ten trong images.bin) -> Volume `figurines/colmap`.
GT LERF-Mask figurines: `figurines.zip` (412.5 MB) tu https://huggingface.co/mqye/Gaussian-Grouping/tree/main/data/lerf_mask (Gaussian Grouping,
dataset phat hanh 2024-01-16 theo README github.com/lkeab/gaussian-grouping) -> `Data/benchmarks/lerf_mask_dl/fig_dl/x/figurines`: 4 view test
(test_0..3) x 7 prompt: green apple, green toy chair, old camera, porcelain hand, red apple, red toy chair, rubber duck with red hat.

Chuoi (khong sua nguong theo canh; chi doi: danh sach prompt + ten vat):
1. CoMe 30k (`modal_come.py::train --scene figurines`, cung co run_tnt/run_scn nhu teatime, L4, 67 phut; 261 view train / 38 test, --eval).
2. SAM 3 thich nghi (`modal_sam3.py::adaptive_scene --prompts-json configs/sam3/figurines_prompts.json`, floor 0.2, rho 0.5): 20 prompt chon tu
   anh frame_00001 / frame_00150. Lan gop 1 (luat chong lap cu area30) bo nham vat nho nam trong mask vat lon (waldo: 233 frame top1 >= 0.2
   nhung chi 44 frame co nhan); them `overlap_mode swallow` (chi bo instance nuot > 50 % mot vat da co nhan) va gop lai. So frame co nhan: table 287, plastic box of candy 226, camera 191, candy bag 119, green apple 219, yellow rubber duck 263, porcelain hand 112, yellow dog toy 221, blue elephant toy 236, green toy chair 223, red toy chair 198, rubik's cube 219, cat figurine 79, pumpkin 163, rubber duck with pirate hat 32, silver lamp 148, red apple 124, rabbit figurine 32, waldo figurine 65, play-doh can 15.
   Han che con lai: raw luu HOP mask moi prompt nen frame nao prompt pirate-duck bat ca vit lon thi ca frame bi bo (32 frame).
3. Hull `tools/mask_hull.py --objects_json configs/pipeline/figurines_v1.json --adaptive_min_views 1 --soft_empty_weight 0.0 --clean_3d 1
   --sfm_anchor 1 --vox_rel 40` (local RTX 4060, 3.4 phut). `--res 320` mac dinh cho voxel 0.283 don vi (~2 voxel / vat) vi vung bi vat ngoai lai
   (hop keo, vit) lam phinh 8.8 x 4.7 x 9.1 don vi; them `--vox_rel N`: voxel = trung vi kich thuoc vat / N -> 0.0154 don vi (teatime 0.0131).
   17/19 vat co hull; rubber_duck_with_pirate_hat, rabbit_figurine = 0 voxel (it frame co nhan).
4. S3 -> S3m m1_fig (protect_vis 3, protect_matter 1) -> S3c r1_fig (lam_bound 0.1, lam_leak 0.05, lam_cover 0.02, 5000 iter):
   `modal run modal_pipeline.py::figchain` (log `logs/figurines_chain.log`).

**Ty le chua tin duoc:** S3 tim mat san song song ban cach 0.963 don vi -> meters_per_unit 0.779 (gia dinh ban cao 0.75 m). Chieu cao p98
Gaussian: rubik_s_cube 32 cm, green_apple 39 cm (that ~6 / ~8 cm) => ty le lon ~5x; moi nguong tinh bang cm (tol_cm, margin_cm, prune_cm,
startup_cm, d_out > 0.5 cm khi dem outside) tren figurines chat hon ~5x theo kich thuoc that so voi teatime. So cm3 / cm la theo ty le do.

### LERF-Mask frame IoU (khung S3m chieu vao 4 view test, `tools/frame_lerf_iou.py --prompt_map configs/sam3/figurines_lerf_map.json`)

| prompt | green apple | green toy chair | old camera | porcelain hand | red apple | red toy chair | rubber duck with red hat |
|---|---|---|---|---|---|---|---|
| IoU | 0.861 | 0.840 | 0.391 | 0.798 | 0.863 | 0.856 | khong co khung |

mIoU 76.8 (6 prompt co khung) / 65.8 (7 prompt, vit mu do = 0). old camera 0.39: mask GT gom ca den flash
+ day deo; pipeline tach den flash thanh vat silver_lamp (prompt cua minh) nen pred chi ~50 % pixel GT (xem `frame_iou_test_0.jpg`): loi dinh nghia
prompt chu khong phai hinh hoc. 5 vat con lai 0.80 - 0.86.

### S3c r1_fig (38 view test)

| | PSNR / SSIM / LPIPS | Gaussian vat | ngoai khung (tong) | ngoai khung (vat co rang buoc) | leak | cover |
|---|---|---|---|---|---|---|
| truoc | 22.448 / 0.8318 / 0.2088 | 105231 | 9425 (8.96 %) | 7705 / 98142 (7.85 %) | 0.0179 | 0.916 |
| **sau** | **22.657** / 0.8289 / 0.2061 | 101993 | 4512 (4.42 %) | 2795 / 94904 (2.95 %) | 0.0177 | 0.898 |

Bo 6985 Gaussian vat khong quan sat, 366 khoi dong, cat 3238 sau. Khung khong dang tin (> 20 % Gaussian ngoai khung truoc S3c, khong ep):
green_apple 22.6 %, waldo_figurine 60.6 %. play_doh_can: S3 gan 0 Gaussian (15 frame co nhan). Theo vat sau (ngoai khung % / cover): plastic_box_of_candy 2.8/0.89, camera 3.6/0.65, silver_lamp 5.1/0.65, candy_bag 1.5/0.75, green_apple 22.6/0.97, yellow_rubber_duck 2.1/0.93, porcelain_hand 1.2/0.83, yellow_dog_toy 3.1/0.93, blue_elephant_toy 2.5/0.97, green_toy_chair 4.3/0.97, red_toy_chair 5.7/0.98, play_doh_can 0.0/0.00, rubik_s_cube 6.4/0.97, cat_figurine 2.4/0.40, pumpkin 1.1/0.85, red_apple 2.4/0.90, waldo_figurine 59.4/0.26.

- Chuoi chung chay het tren canh thu hai khong sua nguong: PSNR test tang 22.45 -> 22.66 sau S3c, Gaussian ngoai khung 8.96 -> 4.42 % (vat co
  rang buoc 7.85 -> 2.95 %), leak 0.0179 -> 0.0177, cover 0.916 -> 0.898 (cung chieu giam nhu teatime r1v8 0.932 -> 0.897).
- Chua dat muc teatime (0.28 %): mot phan do nguong d_out 0.5 cm dang chat ~5x theo kich thuoc that (ty le sai), mot phan do hull 2 vat tren
  bi bo / sai. cover thap o camera 0.65, silver_lamp 0.65, cat_figurine 0.40, waldo 0.26 (vat mong / nho / bi chia doi bang prompt).

File: `Data/figurines/colmap`, `Data/figurines/sam3_masks_adaptive/{labels, preview, legend.json, gate.json}`, `Data/figurines/mask_hull_sam3_adap/`,
`Data/figurines/pipeline/v1_fig/S3m/m1_fig/{frame, metrics.json}`, `Data/figurines/pipeline/v1_fig/S3c/r1_fig/{metrics.json, preview}`,
`Data/figurines/gaussian_inspect/frame_lerf_iou_m1_fig.json` + `frame_iou_m1_fig/`. Config: `configs/pipeline/figurines_v1.json`, `configs/sam3/figurines_*.json`.


## Track A tiep: r1v8_leak_bal + L_leak chuan hoa theo mask (2026-11-18, Modal quangkhaisaigonhcm)

Chay: `modal run modal_pipeline.py::leakrun --tag r1v8_leak_bal --lam-bound 0.1 --lam-leak 2.0 --lam-cover 0.2 --aux-every 2` (log `logs/r1v8_leak_bal.log`).
Them co opt-in `--leak_norm mask` (`pipeline/s3c_constrain.py`, mac dinh `image` = nhu cu): `L_leak = tong alpha_i ngoai mask_i no 15 px / dien tich mask_i`
thay vi trung binh tren ca anh. Ly do: ban `image` chia cho ~toan bo pixel ngoai mask nen gia tri ~1e-4 (log ghi `leak 0.0000` o moi checkpoint ca voi
lam_leak 2.0) -> gradient gan bang 0, tang lam_leak khong co tac dung. r1v8_leak_mn: `--lam-leak 0.5 --lam-cover 0.2 --leak-norm mask` (log `logs/r1v8_leak_mn.log`).
Ca hai run tinh lai 'before' (start.ply moi, cung khung v8_hullv4, cung loc khoi dong 609): PSNR 23.345, outside 7.18 %, leak 0.0261, cover 0.931.

| 23 view test, sau S3c (lam_bound 0.1, 5000 iter) | PSNR / SSIM / LPIPS | outside_pct | leak (alpha ngoai mask) | cover (alpha trong mask) |
|---|---|---|---|---|
| r1v8 (khong aux) | 23.351 / 0.7990 / 0.2649 | 0.28 % | 0.0229 | 0.897 |
| r1v8_leak (leak 0.05, cover 0.02) | 23.406 / 0.7997 / 0.2639 | 0.28 % | 0.0249 | 0.903 |
| r1v8_leak_s (leak 0.5, cover 0.2) | 23.352 / 0.7989 / 0.2649 | 0.31 % | 0.0304 | 0.922 |
| r1v8_leak_bal (leak 2.0, cover 0.2) | 23.358 / 0.7991 / 0.2649 | 0.31 % | 0.0292 | 0.922 |
| r1v8_leak_mn (leak 0.5 mask-norm, cover 0.2) | 23.368 / 0.7991 / 0.2646 | 0.29 % | 0.0273 | 0.918 |

Cover theo vat (bal / mn): coc_su 0.93/0.92, ly_tra 0.97/0.88, dia 0.97/0.97, banh_quy 0.97/0.97, tao 0.93/0.97, tui_banh 0.93/0.93, khan_giay 0.26/0.25, thia 0.71/0.71, vung_tra 0.96/0.89, goi_keo 0.93/0.92, gau 0.73/0.73, cuu 0.92/0.93

- r1v8_leak_bal ~ r1v8_leak_s (cover 0.922 vs 0.922, leak 0.029 vs 0.030): tang lam_leak 0.5 -> 2.0 khong doi gi, khop chan doan gradient L_leak (ban `image`) ~0.
- r1v8_leak_mn: leak 0.0273 (< 0.029 cua bal), cover 0.918, outside 0.29 %, PSNR 23.368. Van khong dat muc tieu (cover >= 0.92 VA leak <= 0.023).
- Leak RIENG tung vat co rang buoc (alpha vat i ngoai mask i no 15 px) da rat nho: max 7.7e-04 (gau), con lai 1e-7 .. 7e-5. Leak tong 0.027 la alpha
  moi Gaussian vat ngoai HOP mask moi vat; GIA THUYET (chua do tach theo nguon): phan lon den tu vat khong nam trong tap rang buoc (khung khong dang tin, khong co L_leak) va view test ma
  mask SAM 3 thieu vat (am tinh gia), khong phai Gaussian tran khoi vat cua chinh no. Vi vay L_leak theo vat khong ha duoc leak tong; can sua o
  nhan / khung (vat khong rang buoc) chu khong phai them trong so.
- Theo dieu kien de ra (chi chay S3i neu tot hon r1v8_leak), KHONG chay S3i tren r1v8_leak_bal / r1v8_leak_mn: ca hai danh doi cover +0.015..0.019 lay
  leak +0.0024..0.0043, khong troi hon. i_r1v8_img (base r1v8_leak) van la ban S3i tot nhat teatime.

File: `Data/teatime/pipeline/v8_hullv4/S3c/{r1v8_leak_bal,r1v8_leak_mn}/metrics.json`, `r1v8_leak_bal/preview/*.jpg`.


## Track C2: canh thu ba LERF ramen (2026-11-18, Modal quangkhaisaigonhcm)

Du lieu: `ramen.zip` (196.8 MB) tu https://huggingface.co/mqye/Gaussian-Grouping/tree/main/data/lerf_mask (cung nguon voi figurines / teatime;
dataset Gaussian Grouping phat hanh 2024-01-16 theo README github.com/lkeab/gaussian-grouping) -> `Data/ramen/dl/x/ramen`: images/ (131 frame + test_0..3),
sparse/0 (PINHOLE 989 x 731, 1 camera), test_mask/0..2 x 6 prompt (chopsticks, egg, glass of water, pork belly, wavy noodles in bowl, yellow bowl).
`tools/prep_scene_colmap.py` them `--exclude "^test_"` (bo view test LERF khoi images.bin; neu khong test_0 -> frame_00000 trung ten) ->
`Data/ramen/colmap` (131 anh) -> Volume `ramen/colmap`.

Chuoi (cung code / nguong voi figurines; chi khac prompt + ten vat + vat chuan ty le):
1. CoMe 30k `modal run modal_come.py::train --scene ramen` (22.5 phut, log `logs/ramen_come.log`).
2. SAM 3 thich nghi: lan 1 dung nguyen cau LERF -> "pork belly", "sake cup", "sake bottle" = 0 frame (top1 SAM 3 = 0 moi frame),
   "wavy noodles in bowl" 55 frame. Them `modal_sam3.py::probe_prompts` (cham top1 tren 12 frame deu) va 1 quy tac chung: giu cau LERF neu trung vi
   top1 >= 0.2 (= floor cong), neu khong lay ung vien co trung vi cao nhat: pork belly 0.00 -> "slice of meat" 0.91, wavy noodles in bowl 0.17 -> "noodles"
   0.81, sake cup 0.00 -> "small metal cup" 0.86, sake bottle 0.00 -> "bottle" 0.35, paper napkin 0.03 -> "napkin" 0.84 (`configs/sam3/ramen_prompts_v2.json`,
   per_instance 1, overlap swallow). So frame co nhan (131): metal plate 123, yellow bowl 131, noodles 126, slice of meat 127, egg 119, seaweed 121,
   bottle 57, small metal cup 94, glass of water 109, napkin 113, chopsticks 114.
3. Hull `tools/mask_hull.py` (co v4 nhu figurines + `--vox_rel 40 --assign_mode ratio`), 11/11 vat co hull.
4. S3 -> S3m m2_ramen -> S3c r2_ramen -> S3i i_r2_ramen (`modal run modal_pipeline.py::figchain --config configs/pipeline/ramen_v2.json --s3i-tag i_r2_ramen`,
   log `logs/ramen_v2_chain.log`).

**Ty le:** khong co san / vat chuan kich thuoc chieu cao. Mat "san" S3 tim duoc cach ban 1.79 don vi -> 0.418 m/don vi (gia dinh ban 0.75 m) cho qua trung
24 - 41 cm => sai (mat do la ghe / ke, khong phai san). `scale_reference` dua dung mot lan dai 0.21 m (stat length, truc chinh trong mat ban):
lan 1 (v1_ramen, da dung giua chung) p98 - p2 = 8.48 don vi (Gaussian lac gan nhan dua) -> 0.0248 m/don vi, sai ~2.8x. Them buoc bo Gaussian lac
(giu diem cach trung vi < 3 x trung vi khoang cach) -> 3.03 don vi -> **0.0694 m/don vi**; kiem cheo: bat vang 2.66 don vi = 18.5 cm, ca 2 nua trung
1.02 don vi = 7.1 cm (hop ly). Ban san lech 6.0x.

### LERF-Mask frame IoU (khung S3m m2_ramen chieu vao 3 view test, `--prompt_map configs/sam3/ramen_lerf_map.json`)

| prompt | chopsticks | egg | glass of water | pork belly | wavy noodles in bowl | yellow bowl | mIoU |
|---|---|---|---|---|---|---|---|
| IoU | 0.520 | 0.671 | 0.858 | 0.740 | 0.606 | 0.746 | **69.0** |

Loi chinh (`Data/ramen/gaussian_inspect/frame_iou_m2_ramen/`): khung noodles lan sang nuoc dung; dua mong (vai voxel) nen lech 1 voxel mat nhieu IoU;
egg = 2 nua trung trong 1 vat.

### S3c r2_ramen (lam_bound 0.1, lam_leak 0.05, lam_cover 0.02, 5000 iter; 11 vat; 17 view test CoMe --eval, 114 train)

| | PSNR / SSIM / LPIPS | Gaussian vat | ngoai khung (tong) | leak | cover |
|---|---|---|---|---|---|
| truoc | 24.540 / 0.8700 / 0.1789 | 140916 | 28394 (20.15 %) | 0.0062 | 0.949 |
| **sau** | **24.739** / 0.8679 / 0.1792 | 137520 | 24986 (18.17 %) | 0.0078 | 0.937 |

Bo 6541 Gaussian vat khong quan sat, 341 khoi dong. Theo vat (ngoai khung % truoc -> sau / cover; * = khung khong dang tin
> 20 %, khong ep): metal_plate 42.3->42.3/0.94*, yellow_bowl 11.6->2.6/0.96, wavy_noodles_in_bowl 22.5->22.5/0.83*, pork_belly 0.2->0.1/0.87, egg 0.9->0.0/0.88, seaweed 0.1->0.0/0.99, sake_bottle 1.6->0.4/0.96, sake_cup 18.1->0.6/0.95, glass_of_water 56.3->56.3/0.87*, paper_napkin 23.6->23.6/0.91*, chopsticks 12.2->0.3/0.97.

- Vat co rang buoc ve 0.0 - 2.6 % ngoai khung (yellow_bowl 11.6 -> 2.6, sake_cup 18.1 -> 0.6, chopsticks 12.2 -> 0.3). Tong 18.2 % do 4 vat khong ep (nguyen nhan duoi day la GIA THUYET, chua kiem):
  glass_of_water 56 % (trong suot: hull SAM 3 khong khop Gaussian khuc xa), metal_plate 42 % va noodles 22.5 % (vat long nhau: dia duoi bat, mi trong bat,
  Gaussian vote sang vat ngoai), napkin 24 %.
- PSNR test tang +0.199 dB; leak 0.0062 -> 0.0078, cover 0.949 -> 0.937 (cung chieu nhu teatime / figurines).

### S3i i_r2_ramen (--image_inpaint 1)

PSNR test 24.739 -> 24.726 (doi trung binh 1.4e-04). Hole % camera ao (7 vat co khung dang tin)
10.16 -> **7.11**: yellow_bowl 2.78->2.76; pork_belly 7.76->2.48; egg 2.49->0.94; seaweed 33.41->33.41; sake_bottle 0.76->0.76; sake_cup 19.15->5.10; chopsticks 4.77->4.34. seaweed 33 % khong doi (tam rong mong, mat sau khong co voxel khuon).

File: `Data/ramen/{colmap, sam3_masks_adaptive_v2, mask_hull_sam3_adap}`, `Data/ramen/pipeline/v2_ramen/{S3_label/support.json, S3m/m2_ramen, S3c/r2_ramen, S3i/i_r2_ramen}`,
`Data/ramen/gaussian_inspect/frame_lerf_iou_m2_ramen.json` + `frame_iou_m2_ramen/`. Config `configs/pipeline/ramen_v2.json`, `configs/sam3/ramen_*.json`.
v1_ramen (ty le 0.0248 sai) bi dung o S3c, khong dung.


## Track C tiep: figurines v2 (ty le + mask theo instance + hull ratio) va v3 / v4 (vote S3 theo view co nhan) (2026-11-18, Modal quangkhaisaigonhcm)

Thay doi (tat ca opt-in, mac dinh = hanh vi cu, teatime khong doi):
1. **Ty le** (`pipeline/s3_label.py`, config `S6_mesh.scale_reference` / `S6_mesh.meters_per_unit`): `meters_per_unit` = kich thuoc vat chuan / kich thuoc
   Gaussian vat do (opacity >= 0.5, bo Gaussian lac cach trung vi > 3 x trung vi khoang cach; stat `height` = p98 - p2 theo phap tuyen ban, `length` =
   p98 - p2 theo truc chinh trong mat ban). Gia tri theo san van luu `meters_per_unit_floor`. figurines: rubik 3x3 chuan 5.7 cm -> 0.3631 don vi ->
   **0.157 m/don vi** (theo san 0.779, lech 4.96x; v1 rubik 32 cm). Moi nguong cm (tol, margin, prune, startup, d_out > 0.5 cm) nay theo kich thuoc that.
   (figurines v2 chay truoc khi them buoc bo Gaussian lac; v3 / v4 sau, cung 0.157 vi rubik gon.)
2. **SAM 3** (`modal_sam3.py`, `per_instance 1`): raw luu tung instance qua cong (thay vi hop); loc swallow xet tung instance. Prompt v2
   (`configs/sam3/figurines_prompts_v2.json`): cau LERF "old camera", "rubber duck with red hat"; bo "silver lamp" (den flash thuoc may anh trong GT).
   So frame co nhan v1 -> v2: vit mu do 32 -> 161, tho 32 -> 127, waldo 65 -> 158, camera 191 -> 207 (khong con silver lamp 148).
3. **Hull** (`tools/mask_hull.py --assign_mode ratio`): voxel thuoc vat co ti le phieu trong / so view CO nhan vat do va thay voxel cao nhat (thay vi so
   phieu). Lan 1 (count) vit mu do = 0 voxel du 161 frame: prompt chung "yellow rubber duck" cung phu vit mu do o 263 frame va thang phieu. ratio: 18/18 vat
   co hull (vit mu do 5780 voxel, tho 1247, waldo 1279).
4. **S3 vote** (`S3_label.vote_norm`): `present` = cung y tuong cho Gaussian (ti le = dong gop trong mask k / dong gop o view co nhan k);
   `present_gated` = them dieu kien dong gop trong mask k >= 20 % tong. v3 = present, v4 = present_gated (cung mask / hull v2).

Chay: `modal run modal_pipeline.py::figchain --config configs/pipeline/figurines_v{2,3,4}.json --s3i-tag i_r1_fig_v{2,3} --image-inpaint 1`
(log `logs/figurines_v{2,3,4}_chain.log`, `logs/figurines_v3_s3i.log`). figchain them buoc S3i (tuy chon) chay tren L40S (L4 22 GB OOM o v3).
v4 dung sau S3c (ket qua ~ v3, khong chay S3i).

### LERF-Mask frame IoU (khung S3m, 4 view test, 7 prompt, `--prompt_map configs/sam3/figurines_lerf_map_v2.json`)

| | green apple | green toy chair | old camera | porcelain hand | red apple | red toy chair | rubber duck with red hat | mIoU (7) |
|---|---|---|---|---|---|---|---|---|
| v1 m1_fig | 0.861 | 0.840 | 0.391 | 0.798 | 0.863 | 0.856 | khong khung (0) | 65.8 (76.8 tren 6) |
| v2 m2_fig | 0.882 | 0.824 | 0.600 | 0.798 | 0.867 | 0.843 | 0.869 | **81.2** |
| v3 / v4 m3_fig / m4_fig | 0.882 | 0.824 | 0.600 | 0.798 | 0.866 | 0.843 | 0.849 | 80.9 |

old camera 0.39 -> 0.60: bo prompt silver lamp, den flash gio la mot phan khung may anh; phan con thieu: chao phan xa den flash + day deo
(`Data/figurines/gaussian_inspect/frame_iou_m2_fig/frame_iou_test_0.jpg`). Vit mu do: 0 -> 0.869.

### S3c (lam_bound 0.1, lam_leak 0.05, lam_cover 0.02, 5000 iter; 38 view test)

| | LERF mIoU | PSNR truoc -> sau / SSIM / LPIPS | ngoai khung (tong) | ngoai khung (vat co rang buoc) | leak | cover |
|---|---|---|---|---|---|---|
| v1 r1_fig (ty le sai 5x) | 65.8 | 22.448 -> 22.657 / 0.8289 / 0.2061 | 8.96 -> 4.42 % | 2.95 % | 0.0179 -> 0.0177 | 0.916 -> 0.898 |
| v2 (count vote) | 81.2 | 22.449 -> **22.670** / 0.8297 / 0.2050 | 9.01 -> 8.26 % | 1.05 % (14 vat) | 0.0226 -> 0.0228 | 0.917 -> 0.906 |
| v3 (vote present) | 80.9 | 22.449 -> **22.684** / 0.8299 / 0.2047 | 11.53 -> 10.77 % | 1.11 % (13 vat) | 0.0306 -> 0.0301 | 0.972 -> 0.957 |
| v4 (vote present_gated) | 80.9 | 22.449 -> **22.685** / 0.8298 / 0.2047 | 11.59 -> 10.84 % | 1.11 % (13 vat) | 0.0305 -> 0.0300 | 0.971 -> 0.956 |

Theo vat sau S3c (ngoai khung % / cover; * = khung khong dang tin > 20 %, khong ep):
- v2: plastic_box_of_candy 0.3/0.89, camera 24.3/0.96*, candy_bag 0.0/0.76, green_apple 2.2/0.97, yellow_rubber_duck 44.2/0.90*, porcelain_hand 0.0/0.86, yellow_dog_toy 1.7/0.95, blue_elephant_toy 2.8/0.98, green_toy_chair 1.4/0.98, red_toy_chair 1.1/0.98, rubik_s_cube 1.2/0.97, cat_figurine 0.0/0.38, pumpkin 0.0/0.91, red_apple 5.0/0.89, rubber_duck_with_red_hat 83.9/0.13*, rabbit_figurine 57.3/0.50*, waldo_figurine 0.0/0.22
- v3: plastic_box_of_candy 0.3/0.89, camera 27.3/0.99*, candy_bag 0.0/0.98, green_apple 3.2/0.97, yellow_rubber_duck 37.2/0.77*, porcelain_hand 0.0/0.93, yellow_dog_toy 1.8/0.95, blue_elephant_toy 2.9/0.98, green_toy_chair 1.5/0.98, red_toy_chair 1.1/0.99, play_doh_can 0.0/0.05, rubik_s_cube 1.2/0.97, cat_figurine 50.3/0.86*, pumpkin 0.0/0.92, red_apple 5.1/0.90, rubber_duck_with_red_hat 0.0/0.35, rabbit_figurine 64.3/0.60*, waldo_figurine 35.8/0.57*

- v2 so v1: LERF mIoU 65.8 -> 81.2 (7 prompt), PSNR sau S3c 22.657 -> 22.670. Ngoai khung tong XAU hon (4.42 -> 8.26 %): 4 vat khong ep vi > 20 % Gaussian
  S3 nam ngoai khung: vit mu do 84 %, tho 57 %, vit vang 44 %, camera 24 %. Khung cua chung dung (vit mu do IoU 0.87) -> loi nam o nhan Gaussian S3:
  vote dem phieu cho prompt chung (vit vang) thang Gaussian cua vit mu do (cung loi da sua o hull). So % ngoai khung v1 / v2 khong cung thuoc do: d_out
  > 0.5 cm o v2 rong hon ~5x theo don vi canh.
- v3 (vote present): vit mu do duoc ep (84 -> 0.0 %), cover tong 0.906 -> 0.957 (candy_bag 0.76 -> 0.98, cat 0.38 -> 0.86, waldo 0.22 -> 0.57), PSNR 22.684
  (cao nhat). Nhung vat nho co mask nhieu (cat, tho, waldo) nhan them Gaussian ngoai khung (cat 0.0 -> 50 %, waldo 0.0 -> 36 %) -> tong 10.8 %.
  v4 (present_gated 0.2) gan nhu trung v3 (cat 52 %, tho 64 %, waldo 42 %): Gaussian thua co ty phan dong gop >= 20 %, nen nguyen nhan khong phai
  Gaussian nen dong gop nho; GIA THUYET (chua kiem): prompt cat / rabbit / waldo bat nham vat khac o mot so frame, mau so nho phong dai.
- Khong ban nao troi han: v2 tot hon ve ngoai khung (8.3 vs 10.8 %), v3 tot hon ve cover (0.957 vs 0.906) va vit mu do. Huong tiep: tieu chi chung de chon
  vote theo vat, chua lam.

### S3i --image_inpaint 1

| | base | PSNR test truoc -> sau | doi trung binh | hole % camera ao truoc -> sau (vat co khung dang tin) |
|---|---|---|---|---|
| i_r1_fig_v2 | r1_fig_v2 | 22.670 -> 22.669 | 5.0e-06 | 3.51 -> **2.75** (11 vat) |
| i_r1_fig_v3 | r1_fig_v3 | 22.684 -> 22.683 | 8.0e-06 | 9.50 -> **8.61** (12 vat) |

Giam lo lon nhat: v2 cat_figurine 27.2 -> 19.6; v3 rubber_duck_with_red_hat 18.8 -> 11.7, play_doh_can 86 -> 83 (21 Gaussian, gan nhu khong co vat).
PSNR test khong doi (< 0.002 dB): lap chi them Gaussian o phan khuat, bo loc nhin thay giu anh that. Trung binh hole % v2 / v3 khac tap vat.

File: `Data/figurines/sam3_masks_adaptive_v2/`, `Data/figurines/mask_hull_sam3_adap_v2/`, `Data/figurines/pipeline/v2_fig/{S3_label/support.json, S3m/m2_fig, S3c/r1_fig_v2, S3i/i_r1_fig_v2}`,
`Data/figurines/pipeline/v3_fig/{S3m/m3_fig, S3c/r1_fig_v3, S3i/i_r1_fig_v3}`, `Data/figurines/pipeline/v4_fig/{S3m/m4_fig, S3c/r1_fig_v4}`,
`Data/figurines/gaussian_inspect/frame_lerf_iou_m{2,3,4}_fig.json` + `frame_iou_m{2,4}_fig/`. Config `configs/pipeline/figurines_v{2,3,4}.json`.


## Uu tien 1 & 2: Vote nhat quan Hull 3D (`fg_present_hull`) va Joint CoMe + Safe-GS (2026-11-19, Modal quangkhaisaigonhcm)

### 1. S3_label `fg_present` va `fg_present_hull` tren figurines (v5, v6, v7) va ramen (v3, v4)

Thay doi trong `pipeline/s3_label.py` (opt-in qua `S3_label.vote_norm`):
- `fg_present`: chi Gaussian co tong dong gop tien canh (`0..n_obj-1`) >= `assign_ratio` va `obj_share >= 0.05` moi duoc dung ty le `present` (`W_k / Wp_k`).
- `all_then_fg_present`: giu `ratio_all` neu `max(ratio_all) >= assign_ratio`, chi dung `fg_present` khi tranh chap.
- `fg_present_hull`: kiem tra nhat quan khong gian 3D voi Visual Hull (`cKDTree` tren `hull_<name>.ply`, buoc luoi `vs` tinh tu `np.diff(np.unique(pts[:, 0]))`):
  chi cho phep `fg_present` kich diem khi Gaussian nam trong/sat `hull_k` (`d_vox <= hull_gate_vox`, mac dinh `2.5`); neu Gaussian nam ngoai `hull_k` nhung nam trong `hull_j` cua vat ung vien khac (`ratio >= 0.20`), giam diem cua `k` (`x 0.25`) de tra ve vat `j`.

#### Bảng S3c & S3i trên `figurines` (38 view test)

| Phien ban (`vote_norm`) | PSNR sau S3c / SSIM / LPIPS | ngoai khung (tong) | ngoai khung (vat rang buoc) | leak | cover | S3i PSNR | S3i hole % TB |
|---|---|---|---|---|---|---|---|
| v2 (`all`) | 22.670 / 0.8297 / 0.2050 | 8.26 % | 1.05 % (14 vat) | 0.0228 | 0.906 | 22.669 | 3.51 -> 2.75 (11 vat) |
| v3 (`present`) | 22.684 / 0.8299 / 0.2047 | 10.77 % | 1.11 % (13 vat) | 0.0301 | 0.957 | 22.683 | 9.50 -> 8.61 (12 vat) |
| v5 (`fg_present`) | 22.685 / 0.8298 / 0.2047 | 10.36 % | 1.10 % (13 vat) | 0.0256 | 0.939 | 22.684 | - |
| v6 (`all_then_fg_present`) | 22.676 / 0.8299 / 0.2049 | 11.33 % | 1.12 % (12 vat) | 0.0256 | 0.940 | 22.676 | - |
| **v7 (`fg_present_hull`)** | **22.6914** / **0.8302** / **0.2046** | **5.32 %** | **0.95 % (16 vat)** | 0.0229 | 0.9092 | **22.6906** | 1.50 -> **1.22** (13 vat) |

Chi tiet theo vat o **v7 (`r1_fig_v7`)** (ngoai khung % sau S3c / cover):
- `yellow_rubber_duck`: **0.0 %** / 0.57 (tu 44.2 % o v2 va 37.2 % o v5)
- `rubber_duck_with_red_hat`: **0.0 %** / 0.45 (1486 Gaussian, tu 83.9 % o v2 va 941 Gaussian o v5)
- `cat_figurine`: **0.0 %** / 0.52 (tu 47.8 % o v5)
- `waldo_figurine`: **0.0 %** / 0.46 (tu 35.8 % o v5)
- `porcelain_hand` 0.0/0.89, `candy_bag` 0.0/0.76, `pumpkin` 0.0/0.91, `plastic_box_of_candy` 0.33/0.89, `red_toy_chair` 1.06/0.98, `rubik_s_cube` 1.17/0.97, `green_toy_chair` 1.42/0.98, `yellow_dog_toy` 1.83/0.95, `green_apple` 2.26/0.97, `blue_elephant_toy` 2.87/0.98, `red_apple` 4.69/0.89.
- Chi con 2 vat ngoai khung > 20 % (khong ep): `camera` 24.31 % (day deo + den flash), `rabbit_figurine` 49.42 % (tu 61.3 % o v5).

#### Bảng S3c & S3i trên `ramen` (17 view test)

| Phien ban (`vote_norm`) | PSNR sau S3c / SSIM / LPIPS | ngoai khung (tong) | ngoai khung (vat rang buoc) | leak | cover | S3i PSNR | S3i hole % TB (7 vat) |
|---|---|---|---|---|---|---|---|
| v2 (`all`) | 24.739 / 0.8679 / 0.1792 | 18.17 % | 0.68 % (7 vat) | 0.0078 | 0.937 | 24.726 | 10.16 -> 7.11 |
| **v3 (`fg_present`)** | **24.7465** / **0.8681** / 0.1792 | 18.14 % | 0.68 % (7 vat) | 0.0081 | **0.9433** | **24.7371** | - |
| **v4 (`fg_present_hull`)** | 24.7403 / **0.8681** / 0.1792 | **17.80 %** | **0.67 % (7 vat)** | 0.0078 | 0.9394 | 24.7300 | 9.96 -> **7.12** |

File: `logs/v{5,6,7}_fig_s3{c,i}_metrics.json`, `logs/v{3,4}_ramen_s3{c,i}_metrics.json`, `configs/pipeline/figurines_v{5,6,7}.json`, `configs/pipeline/ramen_v{3,4}.json`.

### 2. Joint CoMe + Safe-GS tren `teatime` (`tools/come_joint_train.py`, `modal_come.py::joint_train`)

Tich hop `fg_present` + `L_bound` (3D EDT tren `mask_hull_sam3_adap_v4`) + `L_leak` (mask-norm) + `L_cover` truc tiep vao CoMe tu iter `10,000 -> 30,000`:

| Run | PSNR test (23 view) | Tong Gaussian | Gaussian tien canh | Ngoai khung tong (ngay tai 30k) | Vat dang tin |
|---|---|---|---|---|---|
| CoMe 30k goc (baseline) | 23.345 | 1,732,750 | - | - | - |
| S3c `r1v8_leak` (hau ky 5k) | 23.406 | - | 139,144 | 0.28 % (sau cat S3c) | 12/13 |
| **`come_joint` (`lam_ghost=0`)** | **23.5520** | 1,651,693 | 140,117 | **0.88 %** (chua qua S3c) | **12/12** |
| `come_joint_clean` (`lam_ghost=0.05`) | 23.0956 | 1,601,165 | 119,372 | 0.87 % | 12/12 |

Theo vat tren **`come_joint`** (`logs/teatime_joint_metrics.json`, ngoai khung % ngay tai iter 30k): `tao` 0.00 %, `thia` 0.00 %, `banh_quy` 0.01 %, `tui_banh` 0.18 %, `coc_su` 0.31 %, `dia` 0.35 %, `cuu` 0.37 %, `goi_keo` 0.52 %, `ly_tra` 0.58 %, `khan_giay` 0.70 %, `gau` 1.47 %, `vung_tra` 3.85 %.

### 3. Kiem tra nhac `gau` & `cuu` va xu ly vo ma (`tools/inspect_joint_lift.py`)

- Khi chi an `oid == k` (`Data/teatime/come_joint_lift/`), lop long sau / day deo co xanh cua `cuu` va chan duoi ban cua `gau` con sot lai do `T_i ~ 0` o lop trong (bi gan `oid = -1`, `13 (ban)`, hoac `9 (goi_keo)`).
- Ket hop `free_vote` (chieu tam 3D bo qua che khuat) + `faint` (`opa < 0.5`) + `peel` (`Data/teatime/come_joint_lift_free/`, `come_joint_lift_peel/`, `come_joint_lift_exact/`):
  - `come_joint_lift_free`: don sach 99 % tay long `gau` sat coc su (`view_2_frame_00089.jpg`, gom them `16,407` Gaussian cho `gau`, `6,701` cho `cuu`).
  - `come_joint_lift_peel`: xoa sach 100 % vo ma va day deo co xanh cua `cuu` o cot 3 (`cuu/view_0_frame_00001.jpg`, `cuu/view_2_frame_00109.jpg`); chan doan `peel-diag` xac nhan quanh `cuu` co `12,605` hat `oid=-1`, `5,949` hat `oid=13(ban)`, `22` hat `oid=9(goi_keo)`.
  - `come_joint_lift_exact`: gioi han vo `peel` 2.0 cm tren mat ban + khoa bao ve toan bo vat tren ban (`{0..8, 10}`), giu nguyen tao, ly tra, dia banh khi nhac `cuu` o cot 2.
