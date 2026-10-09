import os


def unique_path(path: str) -> str:
    """Trả về path chưa tồn tại: mesh.ply -> mesh_v2.ply -> mesh_v3.ply ... (không bao giờ ghi đè mesh cũ)."""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    i = 2
    while os.path.exists(f"{stem}_v{i}{ext}"):
        i += 1
    return f"{stem}_v{i}{ext}"
