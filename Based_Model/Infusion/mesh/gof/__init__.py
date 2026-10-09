"""
Logic trích xuất mesh của Gaussian Opacity Fields (GOF, Yu et al. 2024), copy từ Based_Model/GOF
và rút gọn cho Gaussians của InFusion (3DGS gốc, không có 3D filter của GOF):
  - graphics.py       : getWorld2View2 / getProjectionMatrix / fov <-> focal   (GOF utils/graphics_utils.py)
  - cameras.py        : camera tối thiểu dựng từ cameras.json của InFusion       (GOF scene/cameras.py)
  - gaussian_model.py : load ply, 3D filter, get_tetra_points, frustum mask     (GOF scene/gaussian_model.py)
  - renderer.py       : integrate() - tích phân opacity tại điểm 3D             (GOF gaussian_renderer/__init__.py)
  - tetmesh.py        : marching tetrahedra (copy nguyên từ GOF utils/tetmesh.py)
"""
