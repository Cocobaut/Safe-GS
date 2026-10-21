# Pipeline v1 (đang chạy): mỗi bước dùng paper / model nào, đọc gì để cải thiện

Code: `modal_pipeline.py` + `pipeline/` ; config: `configs/pipeline/teatime_v1.json` ; kết quả: `Data/teatime/pipeline/v1/`.
Cột "Tự viết" = phần không có trong paper nào, mình tự ghép / tự viết cho bài toán này.

```
Ảnh RGB (video 180 frame)
  |
  S0  COLMAP ............................. camera + điểm thưa
  S1  SAM 2 (team chạy) .................. mask, ID theo video
  S2  CoMe ............................... Gaussian bám bề mặt (1.8M)
  S3  vote gradient + gộp ID 3D + RANSAC . Gaussian nào thuộc vật nào, mặt bàn, scale mét
  S4a ProPainter ......................... ảnh đã xoá vật (inpaint video)
  S4b lấp phẳng + tinh chỉnh ............. nền liền sau khi bỏ vật
  S5  tách vật ........................... Gaussian từng vật
  S6  trường opacity GOF + marching cubes + CoACD .... mesh kín + mảnh lồi (mét)
  S7  MuJoCo MJCF ........................ khối lượng, ma sát
  S8  MuJoCo ............................. mô phỏng (settle / drop / lift)
  S9  render Gaussian theo pose .......... video
```

| Bước | Đang dùng (v1) | Paper / tool gốc | Tự viết | Đọc gì để cải thiện |
|---|---|---|---|---|
| **S0** camera | COLMAP 3.11.1, đúng lệnh `convert.py` của 3DGS, camera PINHOLE | COLMAP: *Structure-from-Motion Revisited* (Schönberger & Frahm, CVPR 2016); `convert.py` của *3D Gaussian Splatting* (Kerbl et al., SIGGRAPH 2023, arXiv 2308.04079) | Chọn model lớn nhất làm `sparse/0` | VGGT-Omega (CVPR 2026, arXiv 2605.15195) |
| **S1** mask + ID | Mask team đưa: SAM 2 hiera-large, lan truyền theo video từ 6 keyframe | *SAM 2: Segment Anything in Images and Videos* (Ravi et al., arXiv 2408.00714) | | **SAM 3** (arXiv 2511.16719, prompt chữ, giữ ID); Multi-Cue Mask Refinement (ICPR 2026, arXiv 2607.01708) |
| **S2** Gaussian | **CoMe** 30k iter, cờ của tác giả | *Confidence-Based Mesh Extraction from 3D Gaussians* (CoMe, ECCV 2026, arXiv 2603.24725). CoMe xây trên: StopThePop (SIGGRAPH 2024, arXiv 2402.00525), SOF, *Gaussian Opacity Fields* (GOF, arXiv 2404.10772), Mip-Splatting (CVPR 2024, arXiv 2311.16493); appearance theo VastGaussian (arXiv 2402.17427) | Sửa lỗi resume trong `train.py` | MILo (arXiv 2506.24096) nếu cần mesh nhẹ; GSPrior (CVPR 2026) |
| **S3** gán Gaussian cho vật | Vote theo **đóng góp render** (gradient theo màu giả trên mask); đồ thị **gộp ID** theo Gaussian dùng chung; kNN cho Gaussian không đóng góp; RANSAC mặt bàn + sàn; scale theo bàn cao 0.75 m | Ý vote gradient: *Gradient-Driven 3D Segmentation* (arXiv 2409.11681); cùng đại lượng với *FlashSplat* (ECCV 2024, arXiv 2409.08270). RANSAC (Fischler & Bolles 1981) | Đồ thị gộp ID 3D, kNN bên trong, scale theo bàn | **Trace3D** (ICCV 2025, arXiv 2508.03227) cho gộp ID; Unified-Lift (CVPR 2025); **ObjectGS** (ICCV 2025, train có ID); COB-GS (CVPR 2025) cho mép; InstaScene (ICCV 2025) |
| **S4a** inpaint ảnh | **ProPainter**, từng đoạn 60 frame, fp16 | *ProPainter: Improving Propagation and Transformer for Video Inpainting* (Zhou et al., ICCV 2023, arXiv 2309.03897) | Ghép mask các vật | **Điểm yếu nhất.** 3D-GIMP (arXiv 2607.20789), Inpaint360GS (WACV 2026), 3DTexMOR (arXiv 2610.11198), CoGeo-GS (ICME 2026), MiniMax-Remover (arXiv 2505.24873) |
| **S4b** lấp nền | Gaussian phẳng lấp vết vật trên bàn + tinh chỉnh Gaussian nền theo ảnh inpaint (3000 iter) | Ý "xoá rồi tinh chỉnh nền theo ảnh inpaint": Gaussian Grouping (ECCV 2024), DecoupledGaussian (CVPR 2025, arXiv 2503.05484) | Lấp phẳng theo mặt bàn | **DecoupledGaussian** (Poisson field cho mặt tiếp xúc) |
| **S5** vật | Gaussian từng vật, đáy = mặt phẳng cắt ở S6 | | Toàn bộ | **SAM 3D Objects** (arXiv 2511.16624); Amodal3R (ICCV 2025, arXiv 2503.13439); InstaScene |
| **S6** mesh | Trường opacity **GOF** của riêng vật (hàm `integrate` của CoMe) -> marching cubes mức 0.5 -> hệ mét -> cắt mặt bàn + đóng nắp (trimesh) -> **CoACD** | GOF (arXiv 2404.10772); Marching Cubes (Lorensen & Cline 1987); *CoACD: Approximate Convex Decomposition ... Collision-Aware Concavity and Tree Search* (Wei et al., SIGGRAPH 2022) | Ghép các bước; sửa lỗi CUDA `integrate` khi 0 điểm | Marching tetrahedra của CoMe/GOF (`extract_mesh_tets.py`); remesh cho kín |
| **S7** vật lý | Khối lượng / ma sát nhập tay; gắn bánh vào đĩa | MuJoCo (Todorov et al., IROS 2012), bản 3.16 | Toàn bộ | PhysGS (arXiv 2511.18570); PhysVGGT (arXiv 2609.18920); SceneSmith (arXiv 2602.09153, VLM chọn vật liệu) |
| **S8** mô phỏng | MuJoCo: settle, drop_all, lift | MuJoCo | Kịch bản | SplatMesh (arXiv 2506.04120, MuJoCo MJX khả vi) |
| **S9** video | Rasterizer CoMe, Gaussian vật biến đổi cứng theo pose | Rasterizer CoMe / StopThePop | Biến đổi pose sim -> thế giới | Gắn Gaussian lên mesh như SplatMesh; xoay SH theo vật |

## Đọc theo thứ tự ưu tiên (theo điểm yếu thấy được trong video)

1. **S4 nền mờ chỗ vật vừa nhấc** -> DecoupledGaussian, 3D-GIMP, Inpaint360GS. Ý cần nắm: vùng dưới vật không frame nào
   thấy, nên phải *sinh* nội dung (diffusion) ở 1 view rồi *lan* sang view khác bằng hình học, không mượn từ frame khác được.
2. **S3 tách vật chưa đủ (vật phức tạp, mép, vật trong suốt)** -> Trace3D, ObjectGS, FlashSplat.
3. **S5/S6 vật chưa kín, đáy phẳng giả định** -> SAM 3D Objects, Amodal3R, CoMe `extract_mesh_tets.py`.
4. **S1 ID vụn** -> SAM 3.
5. **S7 khối lượng nhập tay** -> PhysGS, PhysVGGT.

Note chi tiết + code / chạy L4: các file `01_...` đến `13_...` và `PIPELINE_SOTA.md` trong thư mục này.
