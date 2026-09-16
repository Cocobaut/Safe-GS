"""Module Simulation: Visualize 3D Point Cloud or Mesh (.ply) with Open3D.

Hỗ trợ xem trực quan tương tác các file .ply (Point Cloud hoặc Triangle Mesh)
được xuất ra từ Module 1 hoặc Module Simulation (DTU / Replica).

Tự động nhận diện Display Server (:0 trên Linux) và hỗ trợ cả Point Cloud lẫn Triangle Mesh.

Cách chạy:
    1. Xem file mặc định (scan24_with_bbox.ply):
        python simulation/visualize_point_cloud.py
    
    2. Chỉ định file PLY cụ thể (đường dẫn tương đối hoặc tên file trong result_simulation):
        python simulation/visualize_point_cloud.py --ply simulation/result_simulation/visualization_roi_combined_office0.ply
        hoặc:
        python simulation/visualize_point_cloud.py --ply office0
        python simulation/visualize_point_cloud.py --ply scan24
    
    3. Tùy chỉnh kích thước điểm và màu nền:
        python simulation/visualize_point_cloud.py --point-size 3.0 --bg-color 0.1 0.1 0.1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import open3d as o3d

# Tự động phát hiện và gán biến môi trường DISPLAY trên Linux nếu chưa có
if sys.platform.startswith("linux") and "DISPLAY" not in os.environ:
    x11_dir = Path("/tmp/.X11-unix")
    if x11_dir.exists():
        sockets = sorted(x11_dir.glob("X*"))
        if sockets:
            disp_num = sockets[0].name.replace("X", "")
            os.environ["DISPLAY"] = f":{disp_num}"
    if "DISPLAY" not in os.environ:
        os.environ["DISPLAY"] = ":0"

# Thêm repo root vào sys.path để import an toàn
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def resolve_ply_path(ply_input: Optional[Union[str, Path]]) -> Path:
    """Tự động resolve file PLY từ đường dẫn tương đối hoặc tên gợi nhớ."""
    sim_dir = REPO_ROOT / "simulation" / "result_simulation"

    if not ply_input:
        # Mặc định ưu tiên file scan24 hoặc file đầu tiên trong result_simulation
        candidates = [
            sim_dir / "scan24_with_bbox.ply",
            sim_dir / "visualization_roi_combined_office0.ply",
        ]
        for c in candidates:
            if c.is_file():
                return c

        existing_plys = list(sim_dir.glob("*.ply"))
        if existing_plys:
            return existing_plys[0]
        return sim_dir / "scan24_with_bbox.ply"

    p = Path(ply_input)
    if p.is_file():
        return p

    # Thử resolve relative to REPO_ROOT
    if (REPO_ROOT / p).is_file():
        return REPO_ROOT / p

    # Thử tìm trong simulation/result_simulation/
    if (sim_dir / p).is_file():
        return sim_dir / p
    if (sim_dir / f"{ply_input}.ply").is_file():
        return sim_dir / f"{ply_input}.ply"
    if (sim_dir / f"{ply_input}_with_bbox.ply").is_file():
        return sim_dir / f"{ply_input}_with_bbox.ply"
    if (sim_dir / f"visualization_roi_combined_{ply_input}.ply").is_file():
        return sim_dir / f"visualization_roi_combined_{ply_input}.ply"

    return p if p.is_absolute() else REPO_ROOT / p


def view_3d_geometry(
    ply_path: Union[str, Path],
    point_size: float = 4.0,
    gamma_correction: bool = True,
    gamma_value: float = 0.55,
    bg_color: tuple[float, float, float] = (0.85, 0.85, 0.85),
    show_coord_frame: bool = True,
    save_image: Optional[str] = None,
) -> None:
    """Đọc và hiển thị tương tác Point Cloud hoặc Triangle Mesh."""
    ply_path = resolve_ply_path(ply_path)

    if not ply_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy file PLY: {ply_path}\n"
            f"Hãy kiểm tra lại thư mục simulation/result_simulation/."
        )

    print(f"[*] Đang tải dữ liệu 3D: {ply_path}")

    # 1. Thử đọc dạng Triangle Mesh (nếu là mesh có faces)
    geometry_to_show = None
    is_mesh = False
    try:
        mesh = o3d.io.read_triangle_mesh(str(ply_path))
        if len(mesh.triangles) > 0:
            is_mesh = True
            mesh.compute_vertex_normals()
            geometry_to_show = mesh
            print(f"[*] Dạng dữ liệu: Triangle Mesh ({len(mesh.vertices):,} đỉnh, {len(mesh.triangles):,} tam giác)")
    except Exception:
        is_mesh = False

    # 2. Nếu không phải mesh, đọc dạng Point Cloud
    if not is_mesh:
        pcd = o3d.io.read_point_cloud(str(ply_path))
        print(f"[*] Dạng dữ liệu: Point Cloud ({len(pcd.points):,} điểm)")
        print(f"[*] Dữ liệu màu sắc: {'Có' if pcd.has_colors() else 'Không'}")

        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            if colors.max() > 1.0:
                colors = colors / 255.0

            if gamma_correction:
                colors = np.power(colors, gamma_value)

            pcd.colors = o3d.utility.Vector3dVector(colors)

        # Xóa vector pháp tuyến để tránh Open3D diffuse shading làm tối điểm
        pcd.normals = o3d.utility.Vector3dVector([])
        geometry_to_show = pcd

    if geometry_to_show is None:
        raise RuntimeError(f"Không thể đọc geometry từ file: {ply_path}")

    # 3. Khởi tạo cửa sổ Visualizer
    scene_title = ply_path.stem
    vis = o3d.visualization.Visualizer()

    window_created = vis.create_window(
        window_name=f"Safe-GS 3D Viewer - {scene_title}",
        width=1280,
        height=720,
        visible=True,
    )

    if not window_created:
        print("[!] Cảnh báo: Không thể tạo cửa sổ GLFW trực tiếp. Thử hiển thị qua draw_geometries...")
        try:
            geoms = [geometry_to_show]
            if show_coord_frame:
                geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0]))
            o3d.visualization.draw_geometries(
                geoms,
                window_name=f"Safe-GS 3D Viewer - {scene_title}",
                width=1280,
                height=720,
            )
            return
        except Exception as e:
            print(f"[!] Lỗi khi mở cửa sổ GUI: {e}")
            print("Gợi ý: Đảm bảo biến môi trường DISPLAY được cấu hình đúng (ví dụ: export DISPLAY=:0).")
            return

    vis.add_geometry(geometry_to_show)

    # Thêm trục tọa độ gốc (RGB tương ứng X, Y, Z) với độ dài 0.5m
    if show_coord_frame:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        vis.add_geometry(coord_frame)

    # Tùy chỉnh chế độ hiển thị
    render_option = vis.get_render_option()
    if render_option is not None:
        render_option.point_size = float(point_size)
        render_option.background_color = np.asarray(bg_color)
        render_option.light_on = False if not is_mesh else True

    print("\n--- HƯỚNG DẪN ĐIỀU KHIỂN ---")
    print("- Chuột trái + kéo : Xoay camera")
    print("- Chuột phải + kéo: Di chuyển tịnh tiến (Pan)")
    print("- Con lăn chuột   : Zoom phóng to / thu nhỏ")
    print("- Phím [ / ]      : Tăng / giảm kích thước hạt điểm trực tiếp")
    print("- Phím Q hoặc ESC : Thoát cửa sổ")

    if save_image:
        vis.poll_events()
        vis.update_renderer()
        img_out = Path(save_image)
        img_out.parent.mkdir(parents=True, exist_ok=True)
        vis.capture_screen_image(str(img_out), do_render=True)
        print(f"[✓] Đã lưu ảnh chụp màn hình: {img_out}")

    vis.run()
    vis.destroy_window()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hiển thị 3D Point Cloud hoặc Mesh (.ply) trực quan với Open3D."
    )
    parser.add_argument(
        "--ply",
        dest="ply_path",
        default=None,
        help="Đường dẫn file .ply hoặc tên scan/scene (vd: scan24, office0, simulation/result_simulation/scan24_with_bbox.ply).",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=4.0,
        help="Kích thước hạt điểm Point Cloud (mặc định: 4.0).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.55,
        help="Hệ số gamma correction (mặc định: 0.55).",
    )
    parser.add_argument(
        "--no-gamma",
        action="store_true",
        help="Tắt chế độ gamma correction.",
    )
    parser.add_argument(
        "--bg-color",
        nargs=3,
        type=float,
        default=[0.85, 0.85, 0.85],
        help="Màu nền RGB từ 0.0 đến 1.0 (mặc định: 0.85 0.85 0.85).",
    )
    parser.add_argument(
        "--no-coord",
        action="store_true",
        help="Không hiển thị trục tọa độ gốc (origin frame).",
    )
    parser.add_argument(
        "--save-image",
        default=None,
        help="Đường dẫn lưu ảnh chụp màn hình nếu cần.",
    )

    args = parser.parse_args()

    view_3d_geometry(
        ply_path=args.ply_path,
        point_size=args.point_size,
        gamma_correction=(not args.no_gamma),
        gamma_value=args.gamma,
        bg_color=tuple(args.bg_color),
        show_coord_frame=(not args.no_coord),
        save_image=args.save_image,
    )


if __name__ == "__main__":
    main()