# Pipeline SOTA: ảnh RGB -> cảnh Gaussian tách vật -> physics

Cập nhật 2026-10-20. Ghép phương pháp mạnh nhất hiện có (2025-2026) cho từng khâu, đánh dấu khâu nào đã có trong repo,
khâu nào có code, và chỗ nào còn trống (gap) để đào sâu. "Code" = đã thấy repo công khai (chưa chạy thử).
Note chi tiết từng paper cũ: các file `01_...` đến `13_...` trong thư mục này.

## Sơ đồ

```
                 ẢNH RGB (video teatime, 180 frame)
                              |
     +------------------------+------------------------+
     |                                                 |
 [S0] Camera + depth                            [S1] Mask 2D + ID theo video
 COLMAP (có)  | VGGT-Omega (CVPR26, code)         SAM2 auto (có) -> SAM 3 (11/2025, code)
     |        |  làm prior depth / kiểm pose       prompt chữ: "cup", "teddy bear"... ID giữ qua video
     +------------------------+------------------------+
                              |
 [S2] Gaussian bám bề mặt:  CoMe (có, ECCV26)  | thay thế: MILo, GSPrior (CVPR26)
                              |
 [S3] Gán Gaussian cho vật + đồng bộ ID trong 3D
      vote theo đóng góp render: FlashSplat / Gradient-Driven (code, không train lại)
      sửa ID lệch giữa view: Trace3D (code) | train có ID: ObjectGS (ICCV25, code), Unified-Lift (code)
      mép sạch: COB-GS (code)                 <-- GAP 1: phân vùng đầy đủ (lông, mặt sau, vật bị che)
                              |
            +-----------------+------------------+
            |                                    |
 [S4] NỀN: xoá vật + 3D inpainting         [S5] VẬT: hoàn thiện phần không thấy
  2D video removal: MiniMax-Remover (code),    SAM 3D Objects (11/2025, code):
   ProPainter (code)                            1 ảnh + mask -> vật 3D đầy đủ (kể cả phần bị che)
  3D: DecoupledGaussian (Poisson, code),       Amodal3R (ICCV25) | InstaScene (ICCV25)
   Inpaint360GS (WACV26), 3D-GIMP (07/26),      baseline: cắt mặt bàn + đóng đáy phẳng
   3DTexMOR (10/26), CoGeo-GS (ICME26)          <-- GAP 3: ghép vật sinh ra vào đúng vị trí/scale
  <-- GAP 2: chỗ tiếp xúc nhiều vật /             của cảnh, giữ phần đã quan sát
      hình học mặt bàn sau khi lấp
            |                                    |
            +-----------------+------------------+
                              |
 [S6] Mesh va chạm: vật = mesh kín (CoMe tets / TSDF + đóng đáy) -> CoACD
      nền = mesh nền đã lấp hoặc primitive (mặt bàn = hộp, sàn = mặt phẳng); scale về mét
                              |
 [S7] Thuộc tính vật lý: VLM chọn vật liệu -> khối lượng riêng, ma sát (kiểu SceneSmith);
      PhysGS (Bayes trên Gaussian), PhysVGGT (09/26, 1 ảnh)   <-- GAP 4: chưa có chuẩn đánh giá cho cảnh thật
                              |
 [S8] Mô phỏng: MuJoCo (rigid, MJCF) | Gaussian gắn lên mesh để render theo pose (kiểu SplatMesh)
                              |
 [S9] Render ngược + đánh giá: video nhấc vật; IoU vật render riêng vs mask;
      độ sâu nền ở chỗ vật vừa đứng; vật rơi có nằm yên trên bàn  <-- GAP 5: thiếu benchmark "nhấc vật lên"
```

## Có phải làm 3D inpainting không?

**Có, và là khâu trung tâm.** Có 2 loại:

1. **Inpainting nền (S4):** chỗ dưới cái cốc, sau con gấu chưa từng được nhìn thấy. Đã đo trên bản xoá v3: ở frame 6,
   82% vùng gấu sau khi xoá hiện ra thứ nằm sau gấu và được dựng kém. Muốn "nhấc vật lên, bàn vẫn liền" khi render thì
   phải lấp vùng này nhất quán giữa các view = 3D inpainting. Riêng physics thì mặt bàn có thể thay bằng hộp phẳng,
   nhưng hình ảnh thì không.
2. **Hoàn thiện vật (S5):** đáy cốc, mặt sau con gấu không có trong ảnh nào. Vật trong physics phải là khối kín, nên
   phải "inpaint" hình học của vật trong 3D (amodal completion).

## Từng khâu: lựa chọn

| Khâu | Đang có | SOTA nên thử (code?) | Chạy L4 | Ghi chú |
|---|---|---|---|---|
| S0 camera, depth | COLMAP 177/180, depth từ CoMe | VGGT-Omega (CVPR26 oral, code, checkpoint xin quyền trên HF) | 1B tham số, chắc được ở 512 px | Repo ghi chú 18/08/2026: số benchmark của bản 1B có thể bị ảnh hưởng do dữ liệu nhiễm |
| S1 mask + ID | SAM2 auto, 106 ID, ID vụn | **SAM 3** (11/2025, code, ~840M tham số, xin quyền trên HF): prompt chữ, detector + tracker giữ ID qua che khuất | Được | Prompt theo tên vật thay vì tự sinh hàng trăm mask -> ít ID vụn hơn |
| S2 Gaussian bề mặt | CoMe 30k (74 phút L4) | Giữ CoMe. MILo nếu cần mesh nhẹ | Có | CoMe tự báo TnT F1 0.521 |
| S3 gán Gaussian | Vote theo tâm (v3) | FlashSplat, Gradient-Driven (không train lại); Trace3D; ObjectGS (mỗi Gaussian có ID ngay từ lúc train) | Được | Xem note 01-05 |
| S4 nền | Chưa | Video: MiniMax-Remover (code, DiT, 6 bước), ProPainter (code). 3D: DecoupledGaussian (code), Inpaint360GS, 3D-GIMP, 3DTexMOR, CoGeo-GS (chưa thấy code) | Video diffusion có thể nặng, cần kiểm | Teatime là video nên dùng video inpainting sẽ nhất quán hơn inpaint từng ảnh |
| S5 vật | Chưa | **SAM 3D Objects** (code, xuất Gaussian .ply từ 1 ảnh + mask, chịu được che khuất); Amodal3R | Cần kiểm VRAM | Kết quả ở hệ toạ độ riêng của vật -> phải căn về cảnh |
| S6 mesh | Chưa (CoMe tets cần build thêm) | CoMe tets / TSDF + đóng đáy theo mặt bàn; CoACD | Có | |
| S7 vật lý | Chưa | VLM chọn vật liệu (đơn giản); PhysGS (nộp CVPR26, chưa thấy code); PhysVGGT (09/2026) | Có | v0 dùng bảng khối lượng riêng theo vật liệu |
| S8 mô phỏng | Chưa | MuJoCo (đã chọn) | CPU | |
| S9 đánh giá | Chưa | Tự định nghĩa (xem GAP 5); YUBI-Real2Sim (10/2026) bàn về kiểm định cảnh 3DGS cho robot | | |

## Gap để đào sâu (đã có bằng chứng từ thí nghiệm của mình)

1. **Phân vùng đầy đủ (S3):** vote theo render / theo tâm đều bỏ sót lớp lông sâu, mặt sau, Gaussian to vắt qua
   2 vật (đã đo: ~1.3k Gaussian gấu sót, 98% view bị coi là che). Trace3D tự nhận giới hạn tương tự. Chưa có phương pháp
   đảm bảo "mỗi Gaussian thuộc đúng 1 chủ" cho cảnh nhiều vật.
2. **Chỗ tiếp xúc nhiều vật (S4):** DecoupledGaussian chỉ thử 1 vật / 1 mặt tiếp xúc; paper nhiều vật (2606.29303)
   chưa có code. Teatime có hàng chục vật nằm sát nhau trên bàn.
3. **Ghép vật hoàn thiện vào cảnh (S5):** SAM 3D Objects / Amodal3R sinh vật đầy đủ nhưng ở hệ toạ độ riêng; cần căn
   pose + scale theo phần Gaussian đã quan sát và giữ nguyên phần thật, chỉ thay phần bị thiếu.
4. **Thuộc tính vật lý cho cảnh thật (S7):** các phương pháp mới dựa VLM, chưa có chuẩn đánh giá trên cảnh kiểu teatime.
5. **Đánh giá "nhấc vật lên" (S9):** chưa có benchmark đo đồng thời: vật đủ, nền sạch, mô phỏng ổn. Đây có thể là
   đóng góp riêng (metric + video).

## Đề xuất v1 (SOTA có code, chạy được)

S0 COLMAP -> S1 **SAM 3** (prompt tên vật) -> S2 CoMe -> S3 **FlashSplat / Gradient vote** + gộp ID kiểu Trace3D ->
S4 **MiniMax-Remover hoặc ProPainter** cho ảnh nền + **tinh chỉnh riêng Gaussian nền** (ý DecoupledGaussian) ->
S5 **SAM 3D Objects** cho từng vật (baseline: đóng đáy phẳng) -> S6 mesh + CoACD -> S7 VLM chọn vật liệu ->
S8 MuJoCo -> S9 video + metric.

## Nguồn (2025-2026, ngoài các note 01-13)

- SAM 3: https://arxiv.org/abs/2511.16719 , https://ai.meta.com/research/sam3/ , repo facebookresearch/sam3
- SAM 3D Objects: https://arxiv.org/abs/2511.16624 , repo facebookresearch/sam-3d-objects
- VGGT-Omega: https://arxiv.org/abs/2605.15195 , https://github.com/facebookresearch/vggt-omega
- ObjectGS: https://arxiv.org/abs/2507.15454 , https://github.com/RuijieZhu94/ObjectGS
- Inpaint360GS (WACV 2026): https://arxiv.org/abs/2511.06457 , project https://dfki-av.github.io/inpaint360gs/
- 3D-GIMP: https://arxiv.org/abs/2607.20789 (chưa thấy code)
- 3DTexMOR: https://arxiv.org/html/2610.11198 (chưa thấy code)
- CoGeo-GS (ICME 2026): https://arxiv.org/abs/2608.26656 (chưa thấy code)
- MiniMax-Remover: https://arxiv.org/abs/2505.24873 , https://minimax-remover.github.io
- VOID (video object + interaction deletion, 04/2026): https://arxiv.org/html/2604.02296v1
- Amodal3R (ICCV 2025): https://arxiv.org/abs/2503.13439
- PhysGS: https://arxiv.org/abs/2511.18570 ; PhysVGGT: https://arxiv.org/abs/2609.18920 ; SceneSmith: https://arxiv.org/pdf/2602.09153
- SplatMesh (real-to-sim, MuJoCo MJX): https://arxiv.org/abs/2506.04120 ; YUBI-Real2Sim: https://www.alphaxiv.org/abs/2610.17521
- MILo: https://arxiv.org/abs/2506.24096 ; GSPrior (CVPR 2026): https://github.com/takeshie/GSPrior
