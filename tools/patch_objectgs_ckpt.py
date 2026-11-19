"""
Vá checkpoint của ObjectGS (scene/base_model.py @ 0f6842b) để --start_checkpoint chạy tiếp được.

Bản gốc: capture() trả 11 phần tử, restore() đòi 13 -> resume crash; và không lưu _anchor_feat, label_ids, bộ mã ID,
offset_opacity_accum. Bản vá chỉ thay 2 hàm này (lưu / nạp đủ trạng thái), không đổi cách train.
    python tools/patch_objectgs_ckpt.py /opt/ObjectGS/scene/base_model.py
"""
import sys
from pathlib import Path

NEW = '''    def capture(self):
        # [Safe-GS] full state for resume (original lacks anchor_feat / label_ids; restore did not match capture)
        param_dict = {}
        param_dict['optimizer'] = self.optimizer.state_dict()
        param_dict['opacity_mlp'] = self.mlp_opacity.state_dict()
        param_dict['cov_mlp'] = self.mlp_cov.state_dict()
        param_dict['color_mlp'] = self.mlp_color.state_dict()
        if self.appearance_dim > 0:
            param_dict['appearance'] = self.embedding_appearance.state_dict()
        return {
            'anchor': self._anchor.detach(), 'offset': self._offset.detach(), 'anchor_feat': self._anchor_feat.detach(),
            'scaling': self._scaling.detach(), 'rotation': self._rotation.detach(), 'label_ids': self.label_ids,
            'classes': self.id_encoder.class_tensor,
            'anchor_opacity_accum': self.anchor_opacity_accum, 'anchor_demon': self.anchor_demon,
            'offset_opacity_accum': self.offset_opacity_accum, 'offset_gradient_accum': self.offset_gradient_accum,
            'offset_denom': self.offset_denom, 'max_radii2D': self.max_radii2D,
            'param_dict': param_dict, 'spatial_lr_scale': self.spatial_lr_scale, 'voxel_size': self.voxel_size,
            'active_sh_degree': self.active_sh_degree, 'training_stage': getattr(self, 'training_stage', None),
        }

    def restore(self, model_args, training_args):
        d = model_args
        self._anchor = nn.Parameter(d['anchor'].cuda().requires_grad_(True))
        self._offset = nn.Parameter(d['offset'].cuda().requires_grad_(True))
        self._anchor_feat = nn.Parameter(d['anchor_feat'].cuda().requires_grad_(True))
        self._scaling = nn.Parameter(d['scaling'].cuda().requires_grad_(True))
        self._rotation = nn.Parameter(d['rotation'].cuda().requires_grad_(False))
        self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device="cuda")
        self.label_ids = d['label_ids'].cuda()
        self.id_encoder = OneHotEncoder(d['classes'].cuda())
        self.spatial_lr_scale = d['spatial_lr_scale']
        self.voxel_size = d['voxel_size']
        self.active_sh_degree = d['active_sh_degree']
        if d['training_stage'] is not None:
            self.training_stage = d['training_stage']
        param_dict = d['param_dict']
        self.mlp_opacity.load_state_dict(param_dict['opacity_mlp'])
        self.mlp_cov.load_state_dict(param_dict['cov_mlp'])
        self.mlp_color.load_state_dict(param_dict['color_mlp'])
        if self.appearance_dim > 0:
            self.embedding_appearance.load_state_dict(param_dict['appearance'])
        self.training_setup(training_args)
        self.optimizer.load_state_dict(param_dict['optimizer'])
        for k in ('anchor_opacity_accum', 'anchor_demon', 'offset_opacity_accum', 'offset_gradient_accum',
                  'offset_denom', 'max_radii2D'):
            setattr(self, k, d[k].cuda())

'''


def main(path):
    p = Path(path)
    s = p.read_text(encoding="utf-8")
    if "[Safe-GS]" in s:
        print("already patched"); return
    a = s.index("    def capture(self):")
    b = s.index("    @property\n    def get_anchor(self):")
    assert a < b and "def restore(self, model_args, training_args):" in s[a:b], "cấu trúc base_model.py khác dự kiến"
    p.write_text(s[:a] + NEW + s[b:], encoding="utf-8")
    print(f"patched {p}")


if __name__ == "__main__":
    main(sys.argv[1])
