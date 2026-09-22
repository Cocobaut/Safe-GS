"""
Va MILo scene/cameras.py: luu anh training dang uint8 thay vi float32 de giam RAM he thong ~4 lan
(2000 anh x 2 do phan giai: ~24GB float32 -> ~6GB uint8). KET QUA GIONG HET: anh goc vao Camera la
uint8/255.0, nen `uint8 / 255.0` khoi phuc dung tung bit; moi anh deu duoc kiem tra torch.equal luc nap,
anh nao khong khop se giu float32 nhu ban goc. Ban goc: scene/cameras.py.orig
"""
from pathlib import Path

P = Path("/media/ml4u/Extreme SSD/Safe-GS/Based_Model/MILo/milo/scene/cameras.py")
src = P.read_text()
assert "_image_u8" not in src, "da va roi"

old_init = '''        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
'''
new_init = '''        _img = image.clamp(0.0, 1.0)
        _u8 = (_img * 255.0).round().to(torch.uint8)
        if torch.equal(_u8 / 255.0, _img):
            # luu uint8 (giam RAM ~4 lan), khoi phuc dung tung bit trong property original_image
            self._image_u8 = _u8.to(self.data_device)
            self._image_f32 = None
        else:
            self._image_u8 = None
            self._image_f32 = _img.to(self.data_device)
        self.image_width = _img.shape[2]
        self.image_height = _img.shape[1]
'''
assert src.count(old_init) == 1
src = src.replace(old_init, new_init, 1)

old_cls_end = "class MiniCam:"
prop = '''    @property
    def original_image(self):
        if self._image_u8 is not None:
            return self._image_u8 / 255.0
        return self._image_f32


'''
assert src.count(old_cls_end) == 1
src = src.replace(old_cls_end, prop + old_cls_end, 1)
P.write_text(src)
print("patched OK")
