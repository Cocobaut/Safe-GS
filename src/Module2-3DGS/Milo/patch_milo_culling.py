"""
Va MILo scene/gaussian_model.py: doi tensor `_culling` (N_gaussian x N_camera, bool, tren GPU) thanh
danh sach cot, chi de GIAM DINH BO NHO GPU khi dataset co 2000 anh (Replica). Ket qua toan hoc GIONG HET
ban goc: moi thao tac (lay cot, gan cot, loc hang, noi hang, repeat) duoc lam tung cot mot.
Ban goc luu o scene/gaussian_model.py.orig.
"""
import re
from pathlib import Path

P = Path("/media/ml4u/Extreme SSD/Safe-GS/Based_Model/MILo/milo/scene/gaussian_model.py")
src = P.read_text()
assert "_ColumnCulling" not in src, "da va roi"

WRAPPER = '''

class _ColumnCulling:
    """Thay cho tensor bool (N, V): luu V cot rieng biet de noi/loc hang chi ton them 1 cot bo nho tam
    (tensor goc phai giu ban cu + ban moi cung luc -> tran VRAM khi V=2000). Ket qua giong het."""

    def __init__(self, cols):
        self.cols = cols

    @classmethod
    def zeros(cls, n, num_views, device='cuda'):
        return cls([torch.zeros(n, dtype=torch.bool, device=device) for _ in range(num_views)])

    def __getitem__(self, key):
        # cach dung duy nhat trong code: culling[:, uid]
        assert isinstance(key, tuple) and key[0] == slice(None)
        return self.cols[int(key[1])]

    def __setitem__(self, key, value):
        assert isinstance(key, tuple) and key[0] == slice(None)
        self.cols[int(key[1])] = value.to(torch.bool).clone()

    def select_rows_(self, mask):
        for i, c in enumerate(self.cols):
            self.cols[i] = c[mask]
        torch.cuda.empty_cache()

    def append_rows_(self, mask, repeat=1):
        for i, c in enumerate(self.cols):
            new = c[mask]
            if repeat != 1:
                new = new.repeat(repeat)
            self.cols[i] = torch.cat((c, new))
        torch.cuda.empty_cache()

'''

# 1) chen class wrapper ngay truoc `class GaussianModel`
assert src.count("\nclass GaussianModel") == 1
src = src.replace("\nclass GaussianModel", WRAPPER + "class GaussianModel", 1)

# 2) helper _new_culling trong GaussianModel: giai phong khoi cu TRUOC khi cap phat khoi moi
helper = '''    def _new_culling(self, n, num_views):
        self._culling = None
        torch.cuda.empty_cache()
        return _ColumnCulling.zeros(n, num_views)

    def init_culling(self, num_views):'''
assert src.count("    def init_culling(self, num_views):") == 1
src = src.replace("    def init_culling(self, num_views):", helper, 1)

def sub(pattern, repl, expected):
    global src
    src, n = re.subn(pattern, repl, src)
    assert n == expected, (pattern, n, expected)

# 3) khoi tao (7 cho: 1 trong init_culling + 6 trong cac ham culling_with_*)
sub(r"self\._culling=torch\.zeros\(\(self\._xyz\.shape\[0\], num_views\), dtype=torch\.bool, device='cuda'\)",
    "self._culling = self._new_culling(self._xyz.shape[0], num_views)", 1)
sub(r"self\._culling=torch\.zeros\(\(self\._xyz\.shape\[0\], len\(views\)\), dtype=torch\.bool, device='cuda'\)",
    "self._culling = self._new_culling(self._xyz.shape[0], len(views))", 6)

# 4) noi hang (repeat=N truoc, roi khong repeat)
sub(r"( +)new_culling = self\._culling\[selected_pts_mask\]\.repeat\(N,1\)\n +self\._culling = torch\.cat\(\(self\._culling, new_culling\)\)",
    r"\1self._culling.append_rows_(selected_pts_mask, repeat=N)", 3)
sub(r"( +)new_culling = self\._culling\[selected_pts_mask\]\n +self\._culling = torch\.cat\(\(self\._culling, new_culling\)\)",
    r"\1self._culling.append_rows_(selected_pts_mask)", 3)

# 5) loc hang khi prune
sub(r"self\._culling = self\._culling\[valid_points_mask\]", "self._culling.select_rows_(valid_points_mask)", 1)

# an toan: khong con thao tac tensor nguyen khoi tren _culling
assert "torch.cat((self._culling" not in src
assert not re.search(r"self\._culling\s*=\s*torch\.zeros", src)
P.write_text(src)
print("patched OK")
