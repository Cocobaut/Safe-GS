# Plan nghiên cứu Safe-GS

> File này để nhớ context. Mỗi lần đổi hướng, xong một bước, hay phát hiện điều gì quan trọng thì cập nhật vào đây
> (đánh dấu checkbox, ghi vào mục "Nhật ký" ở cuối).

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
- [ ] **A3** Gán nhãn từng Gaussian theo mask SAM2: chiếu tâm Gaussian lên mọi ảnh, vote theo mask
      -> tách thành Gaussian nền | Gaussian từng vật.

### Giai đoạn B: Lấp lỗ mặt bàn (nền)
- [ ] **B1** Nới rộng mask vật trên mọi ảnh (bao cả bóng đổ).
- [ ] **B2** Inpaint 2D vùng mask (cả màu lẫn depth).
- [ ] **B3** Train tiếp Gaussian nền bằng ảnh đã inpaint -> mặt bàn liền.
- Lưu ý: inpaint từng ảnh riêng sẽ không nhất quán giữa các view -> nhòe. Hướng ổn định: inpaint **1 ảnh tham chiếu**
  (màu + depth), chiếu thành Gaussian đặt vào lỗ, rồi tinh chỉnh nhẹ đa view. **Đây là chỗ có thể có đóng góp nghiên cứu.**

### Giai đoạn C: Lấp đáy vật
- [ ] **C1** Tìm mặt phẳng tiếp xúc (mặt bàn) bằng RANSAC trên Gaussian nền quanh vật.
- [ ] **C2** Trích mesh vật (`extract_mesh_tets.py` của CoMe trên Gaussian của vật), cắt tại mặt bàn,
      đóng nắp phẳng -> mesh kín (watertight).
- [ ] **C3** Thêm Gaussian phẳng trên nắp đáy (màu lấy từ viền đáy) -> nhìn từ dưới không rỗng.
- Lý do dùng nắp phẳng: vật đặt trên bàn thì đáy chính là phần chạm bàn = mặt phẳng. Đúng vật lý hơn để AI đoán.

### Giai đoạn D: Physics
- [ ] **D1** Chỉnh scale về mét + xoay cảnh để mặt bàn ở z = 0, trọng lực hướng -z.
- [ ] **D2** Trích mesh nền đã lấp; chia mảnh lồi cho nền và từng vật (CoACD).
- [ ] **D3** Sinh MJCF: nền tĩnh, mỗi vật là khối tự do, khối lượng = thể tích x khối lượng riêng.
- [ ] **D4** Chạy MuJoCo, xuất pose từng vật theo thời gian.
- [ ] **D5** Áp pose lên Gaussian vật, render cùng nền -> video.

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
| Code CoMe | `Based_Model/CoMe/` | Có, chưa chạy |
| **COLMAP teatime** (input cho CoMe `-s`) | `Data/teatime/colmap/` (+ Volume `safe-gs-come:/teatime/colmap`) | Có |
| **CoMe đã train teatime (30k iter)** | `Data/teatime/come/point_cloud/iteration_30000/` (+ Volume `teatime/come`, có chkpnt29000.pth) | Có |
| A3, B, C, D (script mới) | (dự kiến `tools/`) | Chưa |

## 7. Câu hỏi mở

- **Gom ID mask thành vật (để sau, tự động thay vì tra tay):** SAM2 tách 1 vật thành nhiều ID (bàn = 1 + 65 chân,
  cừu = 4 + 27 chân, gấu = 2/6/9, đĩa 17 + bánh 99/61/75) và có ID không phải vật (63 = vũng trà đổ).
  Tra tay sơ bộ trên frame_00108 (chưa chốt): 1 mặt bàn, 65 chân bàn, 29 ly trà, 22 cốc sứ + thìa, 17 đĩa,
  99/61 bánh quy, 75 vụn bánh, 44 táo, 15 túi bánh, 67 hộp kem sữa, 53 gói đường, 11 khăn giấy, 63 vũng trà,
  4/27 cừu bông, 2/6/9 gấu bông (chưa chắc), 69 gói "NOTES" (chưa chắc), 13/16/24... ghế.

- Đo scale thật bằng cách nào cho teatime? (kích thước thật của ly / bàn, hay depth metric?)
- Đánh giá vùng inpaint mặt bàn thế nào khi không có ảnh thật "bàn không có ly"?
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
