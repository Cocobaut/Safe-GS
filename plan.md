# Plan nghiên cứu Safe-GS

> File này để nhớ context. Mỗi lần đổi hướng, xong một bước, hay phát hiện điều gì quan trọng thì cập nhật vào đây
> (đánh dấu checkbox, ghi vào mục "Nhật ký" ở cuối).
>
> **Kết quả đã chốt (bảng số liệu, so sánh phương pháp, chẩn đoán, việc tiếp theo) nằm trong [RESULTS.md](RESULTS.md).**
> Mục dưới đây là kế hoạch gốc và nhật ký theo ngày.

## 1. Mục tiêu

Đưa **cả một cảnh thật** (ví dụ LERF `teatime`: bàn, ly, bánh, ...) dựng từ ảnh bằng Gaussian Splatting vào
physics engine (MuJoCo), sao cho:

- Nhấc một vật lên (ví dụ cái ly) thì **vật có đáy kín**, va chạm đúng.
- **Mặt bàn chỗ vật vừa đứng vẫn liền**, cả về hình học (va chạm) lẫn hình ảnh (render không bị thủng).
- Kết quả vừa **mô phỏng** được (mesh cho MuJoCo) vừa **render** được (Gaussian), ra video cảnh thật có vật
  chuyển động theo vật lý.

## 2. Ý tưởng tổng thể

Chia cảnh thành 2 phần, mỗi phần giữ 2 dạng biểu diễn:

| Phần | Gaussian (để render) | Mesh (để MuJoCo va chạm) |
|---|---|---|
| Nền tĩnh (bàn, tường, sàn) | Gaussian nền, đã lấp lỗ dưới các vật | Mesh nền chia mảnh lồi, khối tĩnh |
| Mỗi vật động (ly, bánh, ...) | Gaussian của vật, có thêm đáy | Mesh kín chia mảnh lồi, khối tự do |

Vòng lặp: MuJoCo cho pose (vị trí + xoay) của từng vật theo thời gian -> áp phép biến đổi đó lên Gaussian của vật
-> render cùng Gaussian nền -> video.

Có **2 loại lỗ** phải lấp:

1. **Đáy vật**: mặt chạm bàn, không ảnh nào thấy.
2. **Mặt bàn dưới vật**: khi nhấc vật ra thì chỗ đó trống.

## 3. Pipeline

### Giai đoạn A: Dựng cảnh
- [x] **A1** Ảnh + camera (COLMAP) cho teatime: `modal_come.py::colmap_teatime` -> `Data/teatime/colmap/`
      (177/180 ảnh đăng ký, 24685 điểm 3D, PINHOLE, ảnh giữ nguyên 994x738).
- [x] **A2** (xong 2026-10-15, `Data/teatime/come/`, 1.80M Gaussian, 73.6 phút L4) Train **CoMe** cả cảnh (`Based_Model/CoMe/train.py`) -> Gaussian có hình học tốt.
      Cờ chuẩn của tác giả: `--use_ssimdecoupled_appearance --color_confidence --color_confidence_max 0.075
      --color_confidence_from_iter 500 --lambda_variance 0.5 --lambda_normal_variance 0.005 --lambda_distortion 100`
      (xem `Based_Model/CoMe/scripts/run_tnt.py`).
- [x] **A3** (v1 xong: vote gradient `pipeline/s3_label.py`, LERF-Mask 68.0; mask SAM 3 và ObjectGS tốt hơn, xem RESULTS.md) Gán nhãn từng Gaussian theo mask SAM2: chiếu tâm Gaussian lên mọi ảnh, vote theo mask
      -> tách thành Gaussian nền | Gaussian từng vật.

### Giai đoạn B: Lấp lỗ mặt bàn (nền)
- [x] **B1** (v1 `pipeline/s4_background.py`) Nới rộng mask vật trên mọi ảnh (bao cả bóng đổ).
- [x] **B2** (v1 dùng ProPainter, màu; chưa có độ sâu; còn nhoè: độ nét 0.31 so với 1.82. Đã thử Inpaint360GS dùng LaMa, sạch hơn ở
      vài vật) Inpaint 2D vùng mask (cả màu lẫn depth).
- [x] **B3** (v1 xong, chất lượng chưa đạt) Train tiếp Gaussian nền bằng ảnh đã inpaint -> mặt bàn liền.
- Lưu ý: inpaint từng ảnh riêng sẽ không nhất quán giữa các view -> nhòe. Hướng ổn định: inpaint **1 ảnh tham chiếu**
  (màu + depth), chiếu thành Gaussian đặt vào lỗ, rồi tinh chỉnh nhẹ đa view. **Đây là chỗ có thể có đóng góp nghiên cứu.**

### Giai đoạn C: Lấp đáy vật
- [x] **C1** (S3: RANSAC mặt bàn; scale 1 đơn vị = 0.247 m) Tìm mặt phẳng tiếp xúc (mặt bàn) bằng RANSAC trên Gaussian nền quanh vật.
- [x] **C2** (S6: marching cubes trên trường opacity GOF; chưa vật nào kín 0/7, va chạm tạm dùng bao lồi) Trích mesh vật (`extract_mesh_tets.py` của CoMe trên Gaussian của vật), cắt tại mặt bàn,
      đóng nắp phẳng -> mesh kín (watertight).
- [x] **C3** (S5, vết tiếp xúc trên mặt bàn) Thêm Gaussian phẳng trên nắp đáy (màu lấy từ viền đáy) -> nhìn từ dưới không rỗng.
- Lý do dùng nắp phẳng: vật đặt trên bàn thì đáy chính là phần chạm bàn = mặt phẳng. Đúng vật lý hơn để AI đoán.

### Giai đoạn D: Physics
- [x] **D1** (S6, `world_to_sim.json`) Chỉnh scale về mét + xoay cảnh để mặt bàn ở z = 0, trọng lực hướng -z.
- [x] **D2** (S6: bàn = hộp, sàn = mặt phẳng, vật chia CoACD) Trích mesh nền đã lấp; chia mảnh lồi cho nền và từng vật (CoACD).
- [x] **D3** (S7, khối lượng ước lượng tay trong config) Sinh MJCF: nền tĩnh, mỗi vật là khối tự do, khối lượng = thể tích x khối lượng riêng.
- [x] **D4** (S8: để yên, thả, nhấc cốc 20 cm) Chạy MuJoCo, xuất pose từng vật theo thời gian.
- [x] **D5** (S9, video `Data/teatime/pipeline/v1/S9_render/`) Áp pose lên Gaussian vật, render cùng nền -> video.

## 4. Baseline trước, nghiên cứu sau

Chạy hết A -> D với baseline để có video end-to-end sớm, rồi mới nâng cấp từng phần và so với baseline.

| Phần | Baseline | Phiên bản nghiên cứu |
|---|---|---|
| Lỗ mặt bàn | Lấp bằng Gaussian phẳng, màu lấy từ vùng bàn xung quanh | Inpaint ảnh tham chiếu (màu + depth) -> chiếu 3D -> tinh chỉnh đa view |
| Đáy vật | Nắp phẳng | Nắp phẳng (+ có thể thử completion bằng model sinh) |
| Va chạm nền | Mặt bàn = hộp/mặt phẳng primitive, phần còn lại chia mảnh lồi | Toàn bộ nền chia mảnh lồi |

**Thứ tự làm đề xuất:** A1 + A2 (teatime trên Modal) -> A3 -> D1..D5 với baseline -> rồi mới B, C bản nghiên cứu.

## 5. Những điều cần nhớ (đã kiểm chứng / đã chốt)

- **CoMe** = 3DGS + confidence cho mỗi Gaussian + variance loss màu/normal + appearance embedding SSIM-decoupled.
  Chỉ giúp **phần nhìn thấy** có hình học tốt hơn; không tự lấp được vùng không ai thấy.
- **Appearance embedding của CoMe không dùng riêng để "chỉnh ảnh" được**: trọng số học cùng Gaussian của từng cảnh,
  và nó học `hệ số x render ≈ ảnh gốc` (thêm lỗi phơi sáng vào render), không phải sửa ảnh gốc.
  Script xem thử (cần model đã train): `tools/come_apply_appearance.py`.
- "Luminance" trong SSIM của CoMe = trung bình cục bộ cửa sổ Gaussian 11x11, **từng kênh R/G/B riêng**,
  không đổi sang ảnh xám. Luminance so với ảnh đã qua appearance net; contrast-structure so với ảnh render gốc.
- **Chỉnh ảnh / inpaint 2D không tạo ra được đáy vật** vì không có pixel nào chứa đáy.
- **MuJoCo coi mesh là bao lồi khi va chạm, kể cả vật tĩnh** -> mesh nền và vật lõm phải chia mảnh lồi.
- **COLMAP không có đơn vị thật** -> phải đưa về mét trước khi mô phỏng.
- Split&Splat cho Gaussian từng vật nhưng là 3DGS thường -> mesh xốp, không dùng làm hình học cuối.

## 6. Hiện trạng dữ liệu & code

Cấu trúc thư mục `Data/` mô tả trong `Data/README.md`.

| Thứ | Ở đâu | Trạng thái |
|---|---|---|
| **Ảnh teatime (input chính, team chốt)** | `Data/teatime/images/` (180 ảnh, 994x738) | Có |
| **Mask SAM2 teatime (ID nhất quán giữa frame, 106 vật)** | `Data/teatime/sam2_masks/` | Có |
| Prep figurines (COLMAP + Murre depth + 3DGS) | `Data/figurines/prep/`, `modal_prep.py` | Có |
| Mask SAM2 + lan truyền figurines | `Data/figurines/sam2_masks/`, `Data/figurines/propagation/` | Có |
| Gaussian từng vật figurines (Split&Splat) | `Data/figurines/split_splat_stage2/`, `Data/figurines/viewable_ply/` | Có (3DGS thường) |
| Code CoMe | `Based_Model/CoMe/` (submodule) | Đã chạy: `modal_come.py` |
| **COLMAP teatime** (input cho CoMe `-s`) | `Data/teatime/colmap/` (+ Volume `safe-gs-come:/teatime/colmap`) | Có |
| **CoMe đã train teatime (30k iter)** | `Data/teatime/come/point_cloud/iteration_30000/` (+ Volume `teatime/come`, có chkpnt29000.pth) | Có |
| **Pipeline v1 S0 đến S10** (A3, B, C, D + đo lường) | `modal_pipeline.py`, `pipeline/`, `configs/pipeline/teatime_v1.json`; kết quả `Data/teatime/pipeline/v1/` | Xong, có video và bảng đo S10 |
| Mask SAM 3 (17 ID) | `Data/teatime/sam3_masks/`, `modal_sam3.py` | Xong, chưa dùng trong v1 |
| Visual hull từ mask | `Data/teatime/mask_hull*/`, `tools/mask_hull.py` | Xong |
| Inpaint360GS (nguyên pipeline tác giả) | `Based_Model/Inpaint360GS` (submodule), `modal_inpaint360.py`, `Data/teatime/i360gs/` | Chạy xong, mới nhìn bằng mắt |
| ObjectGS (LERF-Mask và trên dữ liệu của mình) | `Based_Model/ObjectGS` (submodule), `modal_objectgs.py`, `Data/teatime/objectgs/` | Xong, xoá vật chưa sạch |
| Tổng hợp kết quả | `RESULTS.md` | Có |

## 7. Câu hỏi mở

- **Gom ID mask thành vật (để sau, tự động thay vì tra tay):** SAM2 tách 1 vật thành nhiều ID (bàn = 1 + 65 chân,
  cừu = 4 + 27 chân, gấu = 2/6/9, đĩa 17 + bánh 99/61/75) và có ID không phải vật (63 = vũng trà đổ).
  Tra tay sơ bộ trên frame_00108 (chưa chốt): 1 mặt bàn, 65 chân bàn, 29 ly trà, 22 cốc sứ + thìa, 17 đĩa,
  99/61 bánh quy, 75 vụn bánh, 44 táo, 15 túi bánh, 67 hộp kem sữa, 53 gói đường, 11 khăn giấy, 63 vũng trà,
  4/27 cừu bông, 2/6/9 gấu bông (chưa chắc), 69 gói "NOTES" (chưa chắc), 13/16/24... ghế.

- ~~Đo scale thật bằng cách nào cho teatime?~~ Đã chốt v1: scale theo chiều cao bàn 0.75 m (1 đơn vị COLMAP = 0.247 m). Chưa kiểm
  bằng kích thước vật thật ngoài việc kích thước ra hợp lý (táo 7.4 cm, cốc 15.7 cm kể cả thìa, đĩa 25 cm).
- ~~Đánh giá vùng inpaint mặt bàn thế nào khi không có ảnh thật?~~ Đo gián tiếp bằng S10: PSNR ngoài vùng vật, mặt bàn phẳng trong
  1 cm, độ nét so với vành bàn, và "ma" (alpha các Gaussian còn lại trong cột vật đã xoá). Còn thiếu sàn nhiễu của mặt bàn ở vùng
  không có vật.
- **Mở, quan trọng:** vật dẹt (gói kẹo, vỏ đĩa) bị vẽ lên chính Gaussian mặt bàn nên không xoá sạch được bằng ID, hull hay vị trí
  Gaussian; hướng thử tiếp là cắt theo bán kính ảnh hưởng rồi lấp mặt bàn (RESULTS.md mục 6, 9).
- Có cần vật biến dạng (bánh) hay chỉ vật rắn?

## 8. Nhật ký

- 2026-10-15: Đọc hiểu CoMe; chốt mục tiêu "cả cảnh vào physics, nhấc vật lên bàn vẫn liền"; lập plan này.
- 2026-10-15: Team chốt input teatime = 180 ảnh + mask SAM2 v2 (ID nhất quán giữa frame). Sắp xếp lại `Data/` theo scene
  (xem `Data/README.md`); mask teatime cũ chuyển vào `Data/_archive/`. Chuyển Modal sang tài khoản `ntm12112005`
  (chưa có Volume nào, chi phí tháng này $0).
- 2026-10-15: A1 xong. COLMAP 3.11.1 CUDA trên L4, đúng lệnh convert.py của 3DGS, camera PINHOLE, giữ tên frame_XXXXX.
  Mapper tạo 2 model (model '0' rác 2 ảnh, model '1' 177 ảnh); convert.py luôn lấy sparse/0 nên đã thêm bước đưa model
  lớn nhất về '0' trước undistort. 3 ảnh không đăng ký: frame_00148, 00155, 00156 (mask của 3 frame này không dùng được).
  Volume Modal: `safe-gs-come` (tài khoản ntm12112005).
- 2026-10-15: A2 bắt đầu: `modal_come.py::train_teatime` (L4, 30k iter, cờ đúng run_tnt/run_scn: có `--eval` nên 23/177
  ảnh làm test, không train). Môi trường: CUDA 12.1, py3.10, torch 2.4.1; cần thêm matplotlib/imageio/scipy/skimage/sklearn
  và libegl1/libusb (không có trong environment.yml). Sửa lỗi resume trong train.py của tác giả (chỉ trong container).
  Checkpoint mỗi 1000 iter lên Volume `teatime/come/`. Smoke test 700 iter + resume: OK (~10 it/s lúc đầu).
- 2026-10-15: A2 xong: 30k iter trong 73.6 phút trên L4, không bị ngắt lần nào, 1.80M Gaussian, ply 440 MB.
  Tổng chi phí Modal tài khoản ntm12112005 tới giờ: $2.18 (metered).
- 2026-10-15: Render 177 ảnh (`modal_come.py::render_teatime`, render.py + metrics.py của tác giả) ->
  `Data/teatime/come/renders/{train,test}/` (frame_XXXXX.png + __render_vs_gt.jpg). Test 23 ảnh: PSNR 23.34,
  SSIM 0.805, LPIPS 0.262. Render là màu "chuẩn" (không qua appearance embedding) nên màu/độ sáng lệch nhẹ so với ảnh gốc.
- 2026-10-16: Video tracking mask trên render (`tools/make_tracking_video.py`, `Data/teatime/come/tracking/`).
  Cốc ID 22: ổn. Gấu (phức tạp): ID KHÔNG nhất quán. Cả con gấu đổi ID 2 -> 9 giữa 2 frame liền nhau (40 -> 41,
  89 -> 90), có frame gấu bị chia đôi 2 + 9 (85, 90), ID 6 lúc ở gấu lúc ở cừu; 25 lần diện tích nhảy >2.5x giữa frame
  liền nhau. Frame không có gấu (57-67, 133-139) là đúng (gấu ra khỏi khung). => Bước A3 phải gom ID / sửa ID-switch
  trước khi gán nhãn Gaussian, nhất là vật to, mềm, chiếm cả khung hình.
- 2026-10-16: Quan sát mặt SDF = 0 (`modal_come.py::sdf_vis_teatime`, `tools/come_sdf_vis.py`, `Data/teatime/come/sdf_vis/`).
  Đã kiểm trong kernel: kênh 6 (depth) = chỗ T qua 0.5 (giải chính xác); kênh 7 KHÔNG phải opacity tích luỹ mà là giá trị
  trường opacity tại điểm bề mặt (~0.5, trung vị 0.52). Kết quả: bàn phẳng, cốc/ly/táo/gấu/cừu có viền khuất rõ;
  vật mỏng sát bàn (đĩa, bánh, khăn, gói đường, vũng trà) hầu như không có viền (cao < 4% depth); nền xa nhiều mảnh vụn.
- 2026-10-20: Thử xoá vật (A3 bản đầu): `modal_come.py::remove_object --ids 22`, `tools/come_remove_object.py`.
  Chiếu tâm Gaussian lên 177 view, chỉ tính view không bị che (z <= depth SDF * 1.03 + 0.01), xoá nếu >= 3 view thấy
  và >= 50% phiếu thuộc ID. Cốc: xoá 5,632 / 1,804,024 Gaussian -> `Data/teatime/come/edit/remove_id22/`.
  Cốc biến mất sạch, mặt bàn phía sau hiện ra đúng. Còn sót: (1) vệt mờ trong suốt đúng chỗ đáy cốc chạm bàn,
  (2) vệt mảnh của cái thìa. 374,660 Gaussian (21%) không view nào thấy (nằm trong/khuất) nên không được bỏ phiếu ->
  có thể là phần sót ở đáy. Đây đúng là "lỗ dưới vật" của giai đoạn B.
- 2026-10-20: `remove_object --ids all` (mọi label > 0) -> `edit/delete_all_object/`: xoá 586,200, giữ 1,217,824.
  Vật biến mất nhưng để lại "vỏ ma" trong suốt (bàn, gấu, cừu) + nhiều Gaussian dài như gai: chủ yếu là Gaussian không
  view nào thấy (bên trong vật) và Gaussian ở khe/mép giữa các mask (label 0). => Cần: xoá thêm Gaussian không ai thấy
  nằm trong vùng 3D của vật đã xoá, nới mask. Ghi chú: trong cảnh có 331 Gaussian dài > 5 đơn vị (lớn nhất 120) ở xa
  65-320 đơn vị (nền ngoài cửa kính); viewer bay sát/xuyên qua sẽ thấy "đĩa tròn mờ có vòng đồng tâm".
- 2026-10-20: Khảo sát paper -> `paper/` (README.md tổng hợp + 13 note). Có code: FlashSplat, Gradient-Driven Seg,
  Trace3D, COB-GS, Unified-Lift, InstaScene, DecoupledGaussian, Gaussian Grouping, SAGA. Paper 2026 (Seed2GS,
  Multi-Cue, Occlusion-Robust Decoupling, GASE, Progressive Removal) chưa có code. Chưa paper nào xử lý Gaussian bên
  trong vật (vỏ ma) bằng cách vote theo render; chỉ InstaScene / DecoupledGaussian hoàn thiện vật trong 3D.
  Đề xuất thử trước: Gradient-Driven + FlashSplat trên cảnh CoMe (không train lại), rồi ý gộp mask của Trace3D.
- 2026-10-20: `--cleanup` (xoá thêm quanh vật: tâm trong SDF >= 0.5 hoặc opacity < 0.5) -> `edit/delete_all_object_v2_sdf/`:
  thêm 52k (21k trong SDF, 31k floater), nhìn gần như không đổi. Chẩn đoán (`--diag_ids 2,6,9`, diag.npz) "ma" ở gấu:
  (1) phần lớn pixel gấu (frame 6: 82%) giờ hiện thứ ở SAU gấu (độ sâu >= 1.3x mặt gấu): ghế/sàn bị gấu che ở hầu hết
  view nên dựng kém -> đây là lỗ nền (giai đoạn B), không phải xoá sót; (2) ~1.3k Gaussian gấu sót ngay sau mặt trước
  (lông sâu / mặt sau): to gấp 4 (trục 0.06 vs 0.015), đặc (opacity 0.96), 83% chỉ được 1-2 view "thấy" theo test che
  tâm-vs-depth 3%, bị coi là che ở 98% view, nhưng bỏ qua che thì >= 50% view rơi vào mask gấu => lỗi do CÁCH VOTE
  (tâm + ngưỡng che + tối thiểu 3 view), không do mask. Luật SDF không bắt được vì chúng nằm trên mặt có view nhìn thấy
  (trường GOF = min theo view < 0.5). 374k Gaussian "không view nào thấy" gần như không nằm gần vật (chỉ 37).
- 2026-10-20: v3 `--free-radius 0.2` (trong ~0.2 quanh vật đã xoá: vote bỏ qua che, >= 50% view trong khung rơi vào mask)
  -> `edit/delete_all_object_v3_freevote/`: xoá thêm 45,520 (tổng 683,742). Cừu gần như mất hẳn, gấu mỏng đi rõ.
  Còn: bóng tối hình gấu, mặt bàn lem (vùng bàn / sau gấu bị che ở hầu hết view nên dựng kém -> cần inpaint, giai đoạn B).
- 2026-10-20: Khảo sát SOTA 2025-2026 theo khâu -> `paper/PIPELINE_SOTA.md` (sơ đồ S0-S9 + 5 gap). Mới đáng chú ý:
  SAM 3 (mask theo prompt chữ, giữ ID video), SAM 3D Objects (hoàn thiện vật từ 1 ảnh, xuất Gaussian), VGGT-Omega,
  ObjectGS (train có ID), MiniMax-Remover (video removal), Inpaint360GS / 3D-GIMP / 3DTexMOR / CoGeo-GS (3D inpainting,
  đa số chưa có code), PhysGS / PhysVGGT (thuộc tính vật lý). Bài toán CÓ 3D inpainting: nền (dưới/sau vật) + hoàn thiện vật.
- 2026-10-20: **Pipeline v1 chạy end-to-end S0 -> S6** (`modal_pipeline.py`, `pipeline/`, config `configs/pipeline/teatime_v1.json`,
  kết quả `Data/teatime/pipeline/v1/`, hướng dẫn `pipeline/README.md`). Mỗi stage có _done.json (hash config),
  config_used.json, preview/, checkpoint trong stage. Thời gian: S3 7.5', S4a 4.8', S4b 10.7', S5 0.4', S6 3.1'.
  Kết quả: gộp ID 3D tìm ra gấu 2+9 (overlap 0.77); scale theo bàn 0.75 m -> 1 đơn vị = 0.247 m, kích thước vật hợp lý
  (táo 7.4 cm, bánh 7 cm, ly 11 cm, túi 21 cm, đĩa 25 cm). Mesh vật = marching cubes trên trường opacity GOF của riêng
  vật (TSDF Open3D cho rỗng mọi biến thể; sửa lỗi CUDA integrate khi 0 điểm cần tính).
  Vấn đề còn: hộp kem sữa (ID 67) chỉ 11 Gaussian -> bỏ; ly thuỷ tinh (trong suốt) xoá không sạch; S4 inpaint
  ProPainter ra vệt mờ vì vùng dưới vật KHÔNG frame nào thấy (video inpainting chỉ mượn được nội dung đã thấy) ->
  nền sau tinh chỉnh bị nhoè đúng chỗ vật; mesh sau khi cắt + đóng đáy đa số chưa kín (va chạm tạm dùng bao lồi);
  ly trà CoACD ra 210 mảnh (mesh nhiễu bên trong); hộp bàn 5.1 x 3.7 m (cả dãy bàn).
- 2026-10-21: **S7-S9 xong -> pipeline chạy hết S0 -> S9, có video.** MuJoCo 3.16 (8 body, 218 geom, mesh đúng chỗ).
  Khối lượng thật ước lượng tay trong config. Lỗi đầu: bánh quy nằm lọt trong bao lồi của đĩa -> văng 0.96 m khi để yên;
  sửa bằng `attach` (bánh gắn vào đĩa). Sau sửa: settle dời tối đa 1.4 mm; lift_coc_su cốc nhấc 20 cm, rơi về đúng chỗ;
  drop_all: cốc đổ, ly lăn, táo lăn 0.84 m (hợp lý). Video: `Data/teatime/pipeline/v1/S9_render/<kịch bản>/<camera>.mp4`.
  Còn: chỗ cốc vừa nhấc lên mặt bàn nhoè (S4); SH bậc cao không xoay theo vật; chất lỏng trong ly không mô phỏng.
- 2026-10-21: **S10 đo lường** (`modal_pipeline.py::run --only S10`, `pipeline/s10_eval_{gpu,cpu}.py`, `s10_report.py`;
  bảng: `Data/teatime/pipeline/v1/S10_eval/eval_report.md`). Benchmark LERF-Mask teatime (Gaussian Grouping, đặt trên
  Volume `teatime/benchmarks/lerf_mask/teatime`), căn COLMAP của họ về của mình bằng Umeyama (RMS tâm camera 0.49 cm).
  Kết quả: LERF-Mask mIoU/mBIoU 68.0/65.0 (GG 71.7/66.1, Gaga 71.2/68.4, ILGS 84.3/75.5 theo Gaga Table 1);
  yếu: spoon handle 0 (không có ID riêng), tea in a glass 25, cookies on a plate 56, sheep 62.
  Sửa lỗi rò rỉ: S4b trước tinh chỉnh cả 23 view test -> giờ chỉ view train (S4b chạy lại; video S9 vẫn từ nền cũ).
  Sau sửa: nền ngoài vùng vật 23.74 dB = cảnh gốc 23.73; chỗ vật vừa đứng chỉ 48% pixel trong 1 cm mặt bàn, độ nét
  0.31 vs 1.82 ảnh thật (nhoè); ghép lại 22.62 dB (mất 0.7 dB so với S2). Mesh: 0/7 kín, Chamfer 2.5-3.9 mm.
  `pull` lỗi FileExistsError ở S7_physics (chưa xem), file S10 đã tải đủ.
- 2026-10-23..24: **Inpaint360GS** (WACV 2026, `modal_inpaint360.py`) chạy hết pipeline tác giả trên teatime (COLMAP của mình):
  3DGS test PSNR 23.30; ID của họ khớp vật mình (túi 30, cốc 28 + thìa 34, đĩa 43, bánh 40, táo 56, ly 49; kem sữa không
  khớp). Sửa để chạy: CRLF make.sh, PYTHONPATH (CropFormer, gốc repo), repo thiếu LaMa/saicinpainting/training/data (lấy
  từ advimman/lama @786f593), bước chọn mask bằng tay -> mask tự động từ objects_pred. Kết quả (định tính,
  `Data/teatime/i360gs/result/`): chỗ cốc / đĩa / táo thành mặt bàn sạch; còn khối nhoè lớn chỗ túi bánh + ly. Chưa đo S10.
- 2026-10-23: **ObjectGS** (ICCV 2025, `modal_objectgs.py`) tái lập trên dữ liệu LERF-Mask: LERF-Mask mIoU 79.3 / mBIoU 76.8
  (bỏ spoon handle 88.2; paper 88.9), chọn ID kiểu oracle vì mask DEVA của view test trống. PSNR 3 view test LERF 22.3
  (CoMe 24.4-25.4 trên cùng view). Lỗi tác giả: capture/restore checkpoint không khớp -> vá `tools/patch_objectgs_ckpt.py`
  (test resume qua). Bộ nhớ: L4 hết ở iter 10k, L40S hết ở 30k, xong trên A100-80GB. Chưa thử xoá vật, chưa train trên
  COLMAP của mình.
- 2026-11-03: **Mask SAM 3** (`modal_sam3.py`, facebookresearch/sam3 @ f1647e8, prompt chữ, video 177 ảnh COLMAP) ->
  `Data/teatime/sam3_masks/` (labels, legend.json, quality_v3/overlay.mp4). Bản 1: 26 ID (SAM2 cũ: 106 ID), nhảy diện tích
  262 lần (SAM2: 1010). Lỗi thấy khi xem: ID "table" khác 13 bắt nhầm đầu gấu / cổ cừu / vũng trà; cốc và ly bị thủng lỗ ở
  chỗ cà phê / trà (ID riêng 140 / 141); thiếu cốc kem sữa, vũng trà, gói kẹo; khe ~2 px không nhãn giữa vật và bàn.
  Đo bằng GT LERF-Mask (SAM 3 chế độ ảnh, `tools/sam3_lerf_eval.py`): mIoU/mBIoU 77.6/75.9 mặc định, 79.2/77.6 ngưỡng 0.3;
  nở mask +1 px còn 75.8/74.0, co -1 px còn 75.2/73.3 => biên đã đúng, KHÔNG lấp khe bằng cách nở mask (coi là "chưa biết").
  "paper napkin" = 0.0 (đổi prompt "napkin" thì chạy); ngưỡng điểm 0.3 cứu ly trà (48 -> 96) nhưng làm gấu tụt (99 -> 65).
  Bản 3 (17 ID): bàn chỉ giữ 13; gộp cà phê -> cốc (80), trà -> ly (70), bánh quy thành 1 nhóm (100); thêm prompt
  "small white cup" / "puddle of liquid" / "blue snack package" (chọn bằng thử ảnh `probe/cand_*.jpg`, bỏ instance trùng
  vật đã có nhãn); xoá mảnh rời < 40 px. Visual hull (`tools/mask_hull.py --scheme sam3`): tỉ lệ vùng mập mờ / chắc chắn
  giảm ở 6/9 vật (ly 7.64 -> 0.84, cừu 15.9 -> 1.46, bánh 1.35 -> 0.05, đĩa 2.23 -> 1.62); hull vẫn phình vật phẳng
  (đĩa 2766 cm3 so với mesh v1 1220; bánh 691 so với 167) vì camera nhìn từ trên xuống. Chưa cải thiện: ID đứt theo
  thời gian (đợt xuất hiện 7-15 / vật, chưa kiểm đoạn frame 140-176), cốc kem sữa chỉ 58 frame nên hull = 0.
- 2026-11-03: **ObjectGS trên COLMAP của mình + mask SAM 3** (`modal_objectgs.py::_pipeline_sam3`, 1 h 13 phút trên A100-80GB,
  config lerf_mask giữ nguyên, 23 view test chia như CoMe: mỗi 8 ảnh theo tên). NVS 23 view test: PSNR 23.19 / SSIM 0.792 /
  LPIPS 0.258 (CoMe 23.34 / 0.805 / 0.263). ID render so với mask SAM 3 trên view test: pixel vật đúng 98.9%, IoU theo
  vật 0.78-0.95 (gói kẹo 180 chỉ 0.46, vũng trà 171 0.61); số này hơi lạc quan vì point cloud gán nhãn dùng cả mask view test.
  Xoá vật (`tools/objectgs_eval_removal.py`, `Data/teatime/objectgs/eval_removal_sam3/`): bỏ 98,515 anchor của vật trên bàn
  thì vật biến mất nhưng để lại "vỏ kính" trong suốt đúng hình vật (đĩa, ly, gói kẹo, túi) và còn 27,737 anchor ID 0 / 13
  lơ lửng trong khoảng trống (~28% số anchor đã xoá; có thể gồm nhiễu mặt bàn vì ngưỡng cao 0.5 cm). Các % ghost / flat / sunk
  theo độ sâu (31.9 / 16.5 / 51.6) KHÔNG so được với số S10 của v1 vì ObjectGS chỉ có độ sâu kỳ vọng (pha trộn lớp trong
  suốt), v1 dùng độ sâu T = 0.5. Nghi nguyên nhân: config tác giả để lambda_object_loss = 0.01 nên ràng buộc "anchor
  phải nằm trong mask của ID mình" rất yếu. Việc kế tiếp đề xuất: đo bằng alpha của anchor còn lại (độc lập độ sâu), rồi thử
  tăng ràng buộc mask lúc train.
- 2026-11-04: **Thử tăng ràng buộc mask của ObjectGS** (chỉ đổi lambda_object_loss, chạy song song 2 bản trên A100-80GB; loss
  của tác giả = cross-entropy trên bản đồ ID đã render, ignore_index = 0). Thước đo mới = alpha của RIÊNG các anchor còn lại
  cao hơn mặt bàn > 1.5 cm trong cột khoảng trống vật (`tools/objectgs_eval_removal.py`; v1: `pipeline/s10_ghost.py`).
  | lambda | PSNR / SSIM / LPIPS (23 view test) | IoU ID | anchor ma > 1.5 cm | alpha ma TB | % pixel vết chân alpha > 0.1 |
  | 0.01 (gốc) | 23.19 / 0.792 / 0.258 | 0.905 | 11,433 | 0.205 | 66.5 |
  | 0.1 | 22.24 / 0.790 / 0.262 | 0.945 | 3,391 | 0.227 | 70.9 |
  | 1.0 | 22.46 / 0.789 / 0.273 | 0.944 | 2,693 | 0.151 | 41.5 |
  Cùng tập vật như v1 (6 vật): ObjectGS 0.183 / 0.199 / 0.150 (% phủ 60.0 / 66.5 / 42.2); v1 nền S4b: alpha ma 0.157, phủ 28.7%
  (6,962 Gaussian ma; vết chân v1 1260 ô, ObjectGS 1500-2800 ô, đơn vị anchor khác Gaussian => so chỉ mang tính tương đối).
  Kết luận: ép mask mạnh hơn giảm mạnh số anchor ma (-76%) nhưng alpha nhìn thấy chỉ giảm tới 0.15 (ngang v1), ảnh test mất
  0.7-1 dB, ảnh xoá ở lambda = 1.0 vẫn còn vỏ trong suốt (đĩa, ly) và gói kẹo xanh còn nguyên hình. Chưa đo sàn nhiễu của
  mặt bàn (vùng không có vật) nên chưa biết alpha tối thiểu "sạch" là bao nhiêu.
- 2026-11-04: **Cắt theo visual hull trên ObjectGS** (`--hull_dir` của `tools/objectgs_eval_removal.py`, hull SAM 3 trong
  `Data/teatime/mask_hull_sam3/`): ngoài xoá theo ID, bỏ thêm MỌI anchor có vị trí cách hull vật đã xoá <= 1.5 cm và cao hơn
  mặt bàn > 1.5 cm. Alpha ma / % phủ / anchor ma: lambda 0.01: 0.205 / 66.5 / 11,433 -> 0.119 / 43.2 / 4,915; lambda 1.0:
  0.151 / 41.5 / 2,693 -> 0.061 / 19.4 / 1,894 (v1 nền S4b: 0.157 / 28.7, vết chân khác nhau nên chỉ so tương đối).
  Số đo theo độ sâu (so được giữa các bản ObjectGS): lambda 1.0 làm mặt bàn dưới vật phẳng hơn hẳn: phẳng trong 1 cm 51.5%
  so với 16.5%, độ cao lệch trung vị 1.02 cm so với 4.16 cm. NHƯNG ảnh xoá vẫn còn nguyên gói kẹo xanh và vỏ đĩa dù hạ ngưỡng
  cắt xuống 0.3 cm, thậm chí cắt tất cả 152,559 anchor trong hull kể cả sát / dưới mặt bàn: phần còn thấy không nằm ở anchor
  có vị trí trong hull + 1.5 cm. Nguyên nhân CHƯA xác định (giả thuyết: Gaussian sinh ra có offset lớn so với vị trí anchor).
  Việc kế tiếp đề xuất: lọc theo vị trí của chính Gaussian sinh ra (sau generate_neural_gaussians) thay vì vị trí anchor.
- 2026-11-04: **Chẩn đoán phần còn sót** (`--probe`, `Data/teatime/objectgs/eval_removal_sam3_lam1.0_probe/`): hull phủ 97-100%
  mask của mọi vật trên 3 view test (kể cả gói kẹo 97.4%) => KHÔNG phải hull lệch. Alpha che pixel vật sau khi xoá ID chủ yếu
  đến từ Gaussian sát mặt bàn (< 0.3 cm) nhãn 13 (alpha 0.92-1.0 ở gói kẹo / đĩa / ly) và nhãn 0; dải cao hơn ~0. Tức vật dẹt
  (gói kẹo dày vài mm) bị "vẽ" lên chính lớp Gaussian mặt bàn, không có Gaussian riêng. Lọc theo vị trí Gaussian (alpha ma cấp
  Gaussian 0.068 -> 0.036; v1 cùng định nghĩa: 0.157) và cắt cả cột trên vết chân hull kể cả lớp mặt bàn (0.023, phủ 6.2%,
  nhưng mặt bàn thủng: sunk 19.9 -> 52.8%, phẳng 51.5 -> 33.4%) đều vẫn để lại mảng xanh gói kẹo: Gaussian mặt bàn LỚN có
  tâm ngoài vết chân nhưng vệt phủ vào vật. Kết luận: với vật dẹt, xoá và lấp là một việc; cần cắt theo bán kính ảnh hưởng
  của Gaussian (không chỉ tâm) rồi lấp mặt bàn bằng texture sạch. Bài học mã: Modal bỏ qua file gắn vào (add_local_file) nếu
  định nghĩa hàm không đổi => đổi hậu tố đường dẫn đích mỗi lần sửa script (xem EVAL_RM trong modal_objectgs.py).


### 2026-11-04 S4c (tách hai nhóm khi tinh chỉnh)
Chạy 4 cấu hình (ctrl, w1, w2, w2_r25). Không đạt tiêu chí: gói kẹo/đĩa còn trong ảnh chỉ-nền; NVS 23.29 đến 23.34 dB (không mất đáng kể), ma 0.034 đến 0.042 (ctrl đã 0.035, vậy cải thiện so với v1 0.157 đến từ gán nhóm SAM 3, không từ huấn luyện). Chi tiết ở RESULTS.md. Việc tiếp: huấn luyện hai nhóm từ đầu (không chỉ tinh chỉnh), hoặc xoá + lấp mặt bàn.

### 2026-11-04 S4d (cắt vết chân + plane_fill)
d1/d2/d3: ảnh chỉ-nền hết gói kẹo/đĩa/cốc, bàn liền, không thủng; còn giống vật 6.3/5.6/6.3 % (từ 10.8). Nhược điểm: màu phẳng mất vân gỗ, cắt lẹm cừu. Tiếp: tinh chỉnh với ảnh inpaint (S4b) trên nền đã lấp, bảo vệ cừu/gấu. Chi tiết RESULTS.md.

### 2026-11-04 S4d vòng 2 (kết luận)
e8 (cắt vết chân + bảo vệ cừu/gấu + plane_fill + tinh chỉnh ảnh thật, bỏ giám sát trong mask vật, đóng băng Gaussian lấp) là bản tốt nhất: bàn sạch, không thủng, PSNR không-vật 23.33 (đối chứng 23.47), còn giống vật 6.7 % (10.8 %). Ảnh inpaint S4a làm hỏng tinh chỉnh (còn vật). Tiếp: thay nền S4b của v1 bằng e8 rồi chạy S5 đến S9 + S10, vá vệt tam giác (fill nhiều Gaussian nhỏ hơn / lấy texture từ ảnh thật), kiểm 6 % còn lại. Chi tiết RESULTS.md.

### 2026-11-04 run v2 (nền e8) + e10
v2 = v1 với nền S4b thay bằng S4d/e8, chạy lại S9, S10. Hình học chỗ vật đứng: trong 1 cm 93.0 % (v1 48.2 %), trung vị 0.09 cm (1.10 cm). Đổi lại PSNR ngoài vùng vật 21.97 (v1 23.74), S9 21.62 (22.62). e10 (ô lấp 0.5 cm) không tốt hơn e8; vệt tam giác do Gaussian lớn của nền gốc. Tiếp: giảm mất mát vành sát vật (lấy texture ảnh thật vào vùng lấp), kiểm đáy vật / mesh v2, nhận định cuối cho MuJoCo.
