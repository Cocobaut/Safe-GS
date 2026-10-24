# GASE: Gaussian Splatting-Based Automated System for Reconstructing Embodied-Simulation Environments

- **Tác giả:** Jiawei Zhang, Yiming Yan, Chao Liang và cộng sự (11 tác giả)
- **Venue:** arXiv 2606.17520 (16/06/2026), mục cs.RO
- **Code:** **Chưa có**. Abstract ghi "Code will be released."

## Ý tưởng chính

- Hệ thống tự động: quét môi trường bằng video đa view từ dàn camera panorama.
- Tách vật qua các frame ở 2D bằng chiến lược dựa trên pose camera, rồi **inpaint cảnh** chỗ vật để lại.
- Dựng riêng vật và nền tĩnh, đưa vào physics simulator để train policy robot.

## Số liệu (theo abstract)

- Phân đoạn tốt hơn các phương pháp 3D Gaussian khác hơn 10%; inpainting SOTA; policy train trong mô phỏng kém
  dưới 10% so với train bằng dữ liệu thật (thử trên robot thật).

## Liên quan bài toán của mình

- Là **cả pipeline của mình** dưới dạng hệ thống: tách vật + lấp nền + vào simulator. Đáng theo dõi khi có code.

## Chạy trên L4

- Không rõ. Dùng dàn camera panorama riêng, khác dữ liệu teatime.

## Ưu điểm

- Có kiểm chứng end-to-end bằng robot thật (sim-to-real), đúng mục đích physics.

## Nhược điểm

- Chưa có code; dữ liệu thu bằng phần cứng riêng; abstract không nói rõ vật có được hoàn thiện (đáy, mặt sau) hay không.
