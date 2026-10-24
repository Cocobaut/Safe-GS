"""
S10 report: gộp eval_gpu.json (có thống kê COLMAP) + eval_cpu.json -> eval_report.md (bảng thống kê) + eval_report.json.
Số tham khảo từ paper ghi kèm nguồn; số của mình là đo thật trên kết quả pipeline.
"""
import argparse
import json
from pathlib import Path

# Số tham khảo LERF-Mask teatime (mIoU, mBIoU), giao thức script/eval_lerf_mask.py của Gaussian Grouping
# Nguồn: Table 1 của Gaga (arXiv 2404.07977 v4, bản html); dấu * = Gaga chép từ Gaussian Grouping
REF_LERF = [
    ("SA3D*", 42.5, 39.2, "Gaga Table 1"),
    ("LEGaussians", 42.8, 35.3, "Gaga Table 1"),
    ("LERF*", 49.7, 42.6, "Gaga Table 1"),
    ("GOI", 52.6, None, "Gaga Table 1 (mBIoU không báo)"),
    ("LangSplat", 59.8, 52.7, "Gaga Table 1"),
    ("OmniSeg3D", 69.8, 63.8, "Gaga Table 1"),
    ("Gaga", 71.2, 68.4, "Gaga Table 1"),
    ("Gaussian Grouping*", 71.7, 66.1, "Gaga Table 1 / Gaussian Grouping (ECCV 2024)"),
    ("ILGS", 84.3, 75.5, "Gaga Table 1"),
]


def f(x, nd=2):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--cpu", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    G = json.loads(Path(a.gpu).read_text())
    C = G["colmap"]
    P = json.loads(Path(a.cpu).read_text())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    L = ["# Bảng đo lường pipeline Safe-GS v1 (teatime)", "",
         "Số của mình = đo thật trên kết quả pipeline. Mũi tên: ↑ càng cao càng tốt, ↓ càng thấp càng tốt.", ""]

    L += ["## 1. Tổng hợp theo stage", "",
          "| Stage | Chỉ số | Giá trị | Ghi chú |", "|---|---|---|---|"]
    L += [f"| S0 COLMAP | Ảnh đăng ký ↑ | {C['registered']}/{C['input_images']} | |",
          f"| S0 COLMAP | Sai số chiếu lại trung bình ↓ | {f(C['mean_reprojection_error_px'])} px | |",
          f"| S0 COLMAP | Độ dài track trung bình ↑ | {f(C['mean_track_length'])} | số ảnh thấy 1 điểm 3D |",
          f"| S0 COLMAP | Quan sát / ảnh ↑ | {f(C['mean_observations_per_image'], 0)} | |",
          f"| S0 COLMAP | Số điểm 3D | {C['points3D']} | |"]
    n = G["nvs_ours_test"]
    L += [f"| S2 CoMe | PSNR ↑ / SSIM ↑ / LPIPS(vgg) ↓, 23 ảnh test của mình | {f(n['psnr'])} / {f(n['ssim'], 3)} / {f(n['lpips'], 3)} | --eval, 1/8 ảnh |"]
    for k, v in G["nvs_lerf_test"].items():
        L += [f"| S2 CoMe | PSNR / SSIM / LPIPS, ảnh {k} của LERF-Mask (góc nhìn mới) | {f(v['psnr'])} / {f(v['ssim'], 3)} / {f(v['lpips'], 3)} | |"]
    al = G["align"]
    L += [f"| (căn hệ) | Căn COLMAP LERF -> của mình: sai số tâm camera RMS / max | {f(al['center_residual_rms_cm'])} / {f(al['center_residual_max_cm'])} cm | {al['common_frames']} frame chung; PSNR render ở frame LERF {f(al['psnr_render_vs_lerf_frames'])} dB |"]
    lm = G["lerf_mask"]
    L += [f"| S3 tách vật | **LERF-Mask mIoU ↑ / mBIoU ↑ (10 prompt)** | **{f(lm['mIoU_all'] * 100, 1)} / {f(lm['mBIoU_all'] * 100, 1)}** | benchmark chuẩn, so được với paper (mục 2) |",
          f"| S3 tách vật | LERF-Mask mIoU / mBIoU (bỏ prompt không có nhóm: {', '.join(lm['unsupported_prompts']) or 'không'}) | {f(lm['mIoU_supported'] * 100, 1)} / {f(lm['mBIoU_supported'] * 100, 1)} | |"]
    s4 = G["s4_background"]
    L += [f"| S4 nền | PSNR ngoài vùng vật: nền đã lấp / cảnh gốc ↑ | {f(s4['psnr_outside_objects_background'])} / {f(s4['psnr_outside_objects_full_scene'])} dB | chênh = phần nền bị ảnh hưởng do bỏ vật + tinh chỉnh |",
          f"| S4 nền | LPIPS ngoài vùng vật: nền đã lấp / cảnh gốc ↓ | {f(s4['lpips_outside_objects_background'], 3)} / {f(s4['lpips_outside_objects_full_scene'], 3)} | |",
          f"| S4 nền | Chỗ vật vừa đứng: % pixel nền nằm trong 1 cm / 2 cm quanh mặt bàn ↑ | {f(s4['hole_pct_within_1cm'], 1)}% / {f(s4['hole_pct_within_2cm'], 1)}% | nền sạch về hình học |",
          f"| S4 nền | Chỗ vật vừa đứng: khoảng cách trung vị tới mặt bàn ↓ | {f(s4['hole_dist_to_table_cm_median'])} cm | |",
          f"| S4 nền | Độ nét chỗ lấp / vành bàn quanh đó (render) vs cùng tỉ số trên ảnh thật | {f(s4['hole_sharpness_ratio_render'])} vs {f(s4['hole_sharpness_ratio_gt_reference'])} | < tỉ số ảnh thật = bị mờ hơn bàn thật |"]
    r9 = G["s9_recomposed_test"]
    L += [f"| S9 ghép lại | PSNR / SSIM / LPIPS: nền S4b + vật S5 ở vị trí nghỉ, 23 ảnh test | {f(r9['psnr'])} / {f(r9['ssim'], 3)} / {f(r9['lpips'], 3)} | so với S2 = mất mát do tách + ghép |"]
    st = P["s8_sim"]["settle"]
    L += [f"| S8 mô phỏng | Để yên 2 s: dời lớn nhất ↓ / xoay lớn nhất ↓ | {f(st['max_displacement_mm'])} mm / {f(st['max_rotation_deg'])}° | |",
          f"| S8 mô phỏng | Lún sâu nhất lúc nghỉ ↓ | {f(st['max_penetration_mm'])} mm | |",
          f"| S8 mô phỏng | Vận tốc lớn nhất cuối ↓ / NaN | {f(st['max_speed_end_m_s'], 4)} m/s / {'không' if st['finite'] else 'CÓ'} | |",
          f"| S8 mô phỏng | Tốc độ so thời gian thực ↑ | {f(st['realtime_factor'], 1)}x | CPU |"]
    for k, v in P["s8_sim"].items():
        if k.startswith("lift_"):
            L += [f"| S8 mô phỏng | {k}: vật rơi về lệch chỗ cũ ↓ / vật khác bị dời ↓ | {f(v['return_error_cm'])} cm / {f(v['others_max_displacement_cm'])} cm | |"]

    L += ["", "## 2. LERF-Mask (teatime) so với paper", "",
          "| Phương pháp | mIoU ↑ | mBIoU ↑ | Nguồn |", "|---|---|---|---|"]
    for name, mi, mb, src in REF_LERF:
        L += [f"| {name} | {mi} | {f(mb)} | {src} |"]
    L += [f"| **Safe-GS v1 (của mình)** | **{f(lm['mIoU_all'] * 100, 1)}** | **{f(lm['mBIoU_all'] * 100, 1)}** | đo thật, cùng code tính IoU / Boundary IoU |", "",
          "Lưu ý: mình nối prompt -> ID mask bằng tay (config S10_eval.lerf_prompts), còn các paper dùng Grounded-SAM theo chữ;",
          "mask của mình lấy từ SAM 2 lan truyền video (team), không phải từ prompt chữ.", "",
          "| Prompt | IoU | Boundary IoU |", "|---|---|---|"]
    for k in lm["per_prompt_iou"]:
        L += [f"| {k} | {f(lm['per_prompt_iou'][k] * 100, 1)} | {f(lm['per_prompt_biou'][k] * 100, 1)} |"]

    L += ["", "## 3. Mesh từng vật (S6)", "",
          "| Vật | Kín | Số mặt | Thể tích (cm3) | Kích thước (cm) | Chamfer TB / p95 (mm) ↓ | Mảnh CoACD | Thể tích mảnh / mesh | Hausdorff mesh->mảnh (mm) ↓ |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, v in P["s6_mesh"].items():
        if v.get("skipped"):
            L += [f"| {k} | bỏ qua: {v['skipped']} | | | | | | | |"]
            continue
        L += [f"| {k} | {'có' if v['watertight'] else 'không'} | {v['faces']} | {f(v['volume_cm3'], 1)} | {' x '.join(str(x) for x in v['bbox_cm'])} | "
              f"{f(v['chamfer_mm'], 1)} / {f(v['chamfer_p95_mm'], 1)} | {v['coacd_parts']} | {f(v['hull_volume_ratio'])} | {f(v['hausdorff_mesh_to_hulls_mm'], 1)} |"]

    L += ["", "## 4. Định nghĩa chỉ số", "",
          "- **mIoU / mBIoU (LERF-Mask)**: IoU giữa mask dự đoán (Gaussian của nhóm render riêng, opacity tích luỹ > 0.5) và mask GT; Boundary IoU chỉ tính dải biên dày round(0.02 x đường chéo ảnh) px; trung bình theo view rồi theo prompt (đúng script/eval_lerf_mask.py của Gaussian Grouping).",
          "- **PSNR / SSIM / LPIPS(vgg)**: như metrics.py của 3DGS / CoMe.",
          "- **Ngoài vùng vật**: mask vật (SAM2) nới 15 px, chỉ tính phần còn lại -> đo mức nền bị hỏng khi bỏ vật.",
          "- **Chỗ vật vừa đứng**: pixel có tia nhìn cắt mặt bàn trong vết chân vật (chiếu Gaussian vật xuống mặt bàn, lưới 1 cm); đo khoảng cách điểm depth của nền tới mặt bàn.",
          "- **Độ nét**: phương sai Laplacian (ảnh xám) trong vùng lấp / ở vành bàn quanh đó.",
          "- **Chamfer mesh <-> Gaussian**: 20k điểm trên mesh vs tâm Gaussian (opacity >= 0.5) của vật, trung bình 2 chiều.",
          "- **Hausdorff mesh -> mảnh lồi**: khoảng cách xa nhất từ điểm trên mesh tới hợp các mảnh CoACD (độ hở / độ phình của va chạm).",
          "- **Lún**: khoảng cách tiếp xúc âm lớn nhất của MuJoCo khi để yên 2 s."]
    (out / "eval_report.md").write_text("\n".join(L), encoding="utf-8")
    (out / "eval_report.json").write_text(json.dumps({"gpu": G, "cpu": P}, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
