# Occlusion-Robust Multi-Object Decoupling for Physics-Based Robotic Interaction

- **Tác giả:** Xin Dong, Lihan Zhang, Tianru Dai, Wenfeng Deng, Yansong Tang
- **Venue:** arXiv 2606.29303 (28/06/2026, v2 30/06/2026), chưa có hội nghị
- **Code:** **Không**

## Ý tưởng chính

- Dựng **nhiều vật** từ ít ảnh thật, bị che lẫn nhau, không cần mask cho trước, ra vật dùng được cho mô phỏng MPM.
- Segmentation field học từ SAM2 cho phân vùng instance thô.
- **SDS kết hợp** (giám sát view tham chiếu + tổng hợp view mới) với prior diffusion 2D và 3D để **hoàn thiện phần bị che**.
- Prior hình học: tương đồng trong cùng vật và giữa các vật.

## Số liệu

- Abstract không ghi GPU, thời gian, tên dataset (chỉ nói "synthetic, robotic and real-world").

## Liên quan bài toán của mình

- Nhắm **nhiều vật + bị che + physics**: đúng teatime (vật sát nhau, che nhau). Là bản "nhiều vật" của DecoupledGaussian.

## Chạy trên L4

- Không có code. SDS với prior diffusion thường nặng và chậm (mình ước tính, không có số trong paper).

## Ưu điểm

- Bài toán gần nhất với mục tiêu cuối (cả cảnh nhiều vật vào physics). Rất mới.

## Nhược điểm

- Không code, chưa bình duyệt. Hoàn thiện bằng SDS là "đoán", không đảm bảo hình học thật.
