# Khảo sát SOTA 2026: Ép khuôn (Mold Carving) & Ép Gaussian vào khung vật thể

Ngày khảo sát: 2026-11-17. Toàn bộ thông tin dưới đây trích trực tiếp từ trang arXiv gốc của từng paper.

## 1. Danh sách Paper SOTA 2026 đã kiểm chứng

| Paper | arXiv ID & Ngày | Ý tưởng cốt lõi (Từ Abstract gốc) | Code |
|---|---|---|---|
| **CleanGS** | [arXiv:2601.00913](https://arxiv.org/abs/2601.00913)<br>(01/2026, v2 10/2026) | Lọc hậu kỳ không cần retrain: chọn tập con góc nhìn tối ưu theo **Pareto front** (độ tin cậy mask, camera baseline, đa dạng góc nhìn, hỗ trợ 3D thưa) + rasterization-aware voting. Chạy mất 20 giây, 1.9 GB VRAM, vượt FlashSplat (19.03 vs 16.44 dB). | [smlab-niser/clean-gs](https://github.com/smlab-niser/clean-gs/) |
| **Probabilistic Object-Level GS** | [arXiv:2603.14316](https://arxiv.org/abs/2603.14316)<br>(03/2026, v2 07/2026) | Thay mask nhị phân cứng bằng **continuous probability masks** từ SAM để triệt tiêu nhập nhằng đường biên. Gắn xác suất foreground-background lên từng Gaussian, tỉa động các hạt xác suất thấp, lọc 2 giai đoạn lúc khởi tạo, và tự sửa lỗi biên qua rendered probability mask. Giảm số Gaussian còn ~1/10. | Chưa công bố |
| **OP2GS** | [arXiv:2605.20044](https://arxiv.org/abs/2605.20044)<br>(05/2026) | **Dual-Opacity Primitives**: Tách rời độ mờ hiển thị $\sigma$ và độ mờ vật thể $\sigma^*$. Giải quyết triệt để việc các hạt Gaussian ở ranh giới bị gán nhãn sai (label contamination) giữa vật thể và mặt bàn. | Chưa công bố |
| **ObjectSplat** | [arXiv:2608.30423](https://arxiv.org/abs/2608.30423)<br>(08/2026) | **Decompose-before-reconstruct**: Tách instance trên từng frame $\rightarrow$ inpaint nền trước $\rightarrow$ dựng độc lập từng vật thể và nền bằng **mesh splatting** rồi ghép lại. Tăng >5% mesh F-score. | Sẽ công bố |
| **GaussianObject** | [arXiv:2402.10259](https://arxiv.org/abs/2402.10259)<br>(SIGGRAPH Asia 2024) | Dùng **Visual Hull + Floater Elimination** ngay bước khởi tạo từ ít góc nhìn để ép Gaussian nằm trong khung hình học hợp lệ trước khi tinh chỉnh. | [GaussianObject](https://github.com/GaussianObject/GaussianObject) |

## 2. Kế thừa ý tưởng trực tiếp vào Safe-GS (`s3m_mold.py` & `s3c_constrain.py`)

1. **Từ CleanGS (2026)**: Thay vì cho tất cả 177 camera vote bình đẳng vào khuôn `S3m` (khiến các góc nhìn xiên sát mặt bàn hoặc mask bị lem làm phồng khuôn `dia` lên $2.7\text{ cm}$), ta lọc góc nhìn bằng tiêu chí **Mask Reliability + Viewing-Direction Diversity** (ưu tiên các góc nhìn có độ tin cậy SAM 3 cao và góc nhìn nghiêng tốt để gọt mỏng vật dẹt).
2. **Từ Probabilistic Object-Level GS (2026)**: Thay mask nhị phân cứng ở `s3m_mold` bằng **Continuous Probability / Confidence Map** kết hợp **Dual-Stage Startup Filtering** khi train `s3c_constrain`, giúp đường biên cắt sắc gọn mà không làm mất vật mỏng như `spoon` hay làm phồng `dia`.
3. **Từ OP2GS (2026)**: Áp dụng tư duy **Dual-Opacity ($\sigma$ vs $\sigma^*$)** tại vùng chân đế tiếp xúc mặt bàn để các hạt dùng để render bóng đổ không làm phồng khung vật lý của vật thể.
