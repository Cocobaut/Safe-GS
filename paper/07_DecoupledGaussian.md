# DecoupledGaussian: Object-Scene Decoupling for Physics-Based Interaction

- **Tác giả:** Miaowei Wang, Yibo Zhang, Rui Ma, Weiwei Xu, Changqing Zou, Daniel Morris
- **Venue:** CVPR 2025
- **Paper:** https://arxiv.org/abs/2503.05484
- **Project:** https://wangmiaowei.github.io/DecoupledGaussian.github.io/
- **Code:** Có. https://github.com/wangmiaowei/DecoupledGaussian (`gs_simulation.py`,
  `gs_simulation_multiobject.py`, `mpm_solver_warp/`, `particle_filling/`; xây trên PhysGaussian; README ghi đã
  phát hành toàn bộ code). Không thấy file license.

## Ý tưởng chính

- Bài toán **giống của mình nhất**: tách vật tĩnh khỏi mặt nó đang tựa (từ video thật) để mô phỏng vật **rời hẳn**
  khỏi mặt tiếp xúc (các phương pháp trước chỉ rung / biến dạng đàn hồi tại chỗ).
- Dựng cảnh bằng planar Gaussian (PGSR) để có depth tốt; depth làm điểm proxy cho vật.
- **Joint Poisson fields:** sửa và mở rộng Gaussian của **cả vật lẫn mặt tiếp xúc** sau khi tách (lấp đáy vật và
  mặt bàn). Lập luận: inpainting 2D không khôi phục được vị trí 3D.
- **Multi-carve:** gọt hình học vật cho đúng.
- Mô phỏng MPM (Warp): tách rời, va chạm, vỡ theo lực người dùng.

## Số liệu (theo paper)

- Poisson field 128^3 < 20 s; TSDF < 100 s; tinh chỉnh Gaussian (1,000 iter cảnh + 3,000 iter vật) < 4 phút;
  phục hồi cảnh ~1 phút (VR-GS ~7 phút, GScream ~70 phút).
- Mô phỏng: A40 + Xeon, ~10 FPS cho video 50 frame.
- Thành phần: COLMAP, SAM2, LaMa, RAFT, PoissonRecon. Dữ liệu có cả **LERF figurines**.

## Liên quan bài toán của mình

- Bao phủ **giai đoạn B, C, D** trong plan: lấp mặt bàn + đáy vật + đưa vào physics. Rất đáng đọc kỹ phần Poisson field.

## Chạy trên L4

- Được: các bước đều vài phút; mô phỏng MPM trên A40 (48 GB). Mình chưa có số bộ nhớ, nên L4 24 GB có thể phải giảm
  số hạt MPM.
- Phải train lại cảnh bằng PGSR, không dùng thẳng cảnh CoMe.

## Ưu điểm

- Đúng mục tiêu "nhấc vật lên, mặt bàn vẫn liền, vật có đáy". Nhanh, có code, có physics.

## Nhược điểm (paper tự nêu)

- **Chưa thử cảnh nhiều vật với nhiều kiểu tiếp xúc** (teatime có rất nhiều vật sát nhau).
- Khó hoàn thiện texture tần số cao cho vật; FID của vật cao vì texture nội suy từ ảnh train.
- Khó tách các bộ phận nhỏ của một vật.
- Dùng MPM (PhysGaussian) chứ không phải mesh rigid body như MuJoCo của mình.
