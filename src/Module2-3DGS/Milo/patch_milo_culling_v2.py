"""
Nang cap _ColumnCulling (da va trong scene/gaussian_model.py) len NEN BIT: moi cot bool (N phan tu) duoc luu
thanh N/8 byte uint8 (1 bit / gia tri). Khong mat mat gi - lay cot ra la giai nen dung nguyen mang bool goc,
nen ket qua toan hoc GIONG HET ban goc; chi giam bo nho 8 lan (17GB -> ~2GB khi N ~ 8.5M x 2000 camera).
"""
import re
from pathlib import Path

P = Path("/media/ml4u/Extreme SSD/Safe-GS/Based_Model/MILo/milo/scene/gaussian_model.py")
src = P.read_text()
assert "class _ColumnCulling" in src and "_pack" not in src, "chua va v1 hoac da va v2"

NEW = '''class _ColumnCulling:
    """Thay cho tensor bool (N, V) tren GPU. Luu V cot rieng, moi cot NEN BIT (N/8 byte).
    Lay cot (culling[:, uid]) giai nen ra mang bool dung nhu ban goc -> ket qua giong het,
    chi giam ~8 lan bo nho va noi/loc hang chi ton them bo nho tam cho 1 cot."""

    _W = {}
    _S = {}

    @classmethod
    def _consts(cls, device):
        key = str(device)
        if key not in cls._W:
            cls._W[key] = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=device)
            cls._S[key] = torch.arange(8, dtype=torch.uint8, device=device)
        return cls._W[key], cls._S[key]

    @classmethod
    def _pack(cls, col):
        w, _ = cls._consts(col.device)
        b = col.to(torch.uint8)
        pad = (-b.numel()) % 8
        if pad:
            b = torch.nn.functional.pad(b, (0, pad))
        return (b.view(-1, 8) * w).sum(dim=1).to(torch.uint8)

    @classmethod
    def _unpack(cls, packed, n):
        _, s = cls._consts(packed.device)
        return ((packed.unsqueeze(1) >> s) & 1).view(-1)[:n].to(torch.bool)

    def __init__(self, cols, n):
        self.cols = cols
        self.n = n

    @classmethod
    def zeros(cls, n, num_views, device='cuda'):
        return cls([torch.zeros((n + 7) // 8, dtype=torch.uint8, device=device) for _ in range(num_views)], n)

    def __getitem__(self, key):
        # cach dung duy nhat trong code: culling[:, uid]
        assert isinstance(key, tuple) and key[0] == slice(None)
        return self._unpack(self.cols[int(key[1])], self.n)

    def __setitem__(self, key, value):
        assert isinstance(key, tuple) and key[0] == slice(None)
        assert value.numel() == self.n
        self.cols[int(key[1])] = self._pack(value.to(torch.bool))

    def select_rows_(self, mask):
        new_n = None
        for i, c in enumerate(self.cols):
            b = self._unpack(c, self.n)[mask]
            new_n = b.numel()
            self.cols[i] = self._pack(b)
        if new_n is not None:
            self.n = new_n
        torch.cuda.empty_cache()

    def append_rows_(self, mask, repeat=1):
        new_n = None
        for i, c in enumerate(self.cols):
            b = self._unpack(c, self.n)
            new = b[mask]
            if repeat != 1:
                new = new.repeat(repeat)
            b = torch.cat((b, new))
            new_n = b.numel()
            self.cols[i] = self._pack(b)
        if new_n is not None:
            self.n = new_n
        torch.cuda.empty_cache()


'''

# thay class cu (tu 'class _ColumnCulling' den truoc 'class GaussianModel')
pat = re.compile(r"class _ColumnCulling:.*?(?=class GaussianModel)", re.S)
assert len(pat.findall(src)) == 1
src = pat.sub(lambda m: NEW, src, count=1)

# _new_culling: zeros(n, V) khong doi chu ky -> khong can sua them
assert "_ColumnCulling.zeros(n, num_views)" in src
P.write_text(src)
print("patched v2 (nen bit) OK")
