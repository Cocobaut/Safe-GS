"""
Va MILo train.py de chiu duoc CUDA OOM tam thoi (GPU dung chung) - KHONG doi thuat toan/ket qua:
  1) boc cac ham render/integrate THUAN bang oom_retry (lam lai truc tiep khi OOM);
  2) boc doan "chon camera -> render -> loss -> mesh-in-the-loop -> backward" trong vong lap
     `while True: snapshot; try: ...; break; except OOM: khoi phuc dung trang thai; cho; lam lai`.
Ghi ra train.py.new (kiem thu roi moi mv len train.py). Ban goc: train.py.orig
"""
from pathlib import Path

D = Path("/media/ml4u/Extreme SSD/Safe-GS/Based_Model/MILo/milo")
src = (D / "train.py").read_text()
assert "oom_retry" not in src, "da va roi"

# ---- 1) import + boc ham render thuan (muc module) ----
anchor = "\ndef training(\n"
assert src.count(anchor) == 1
wrap_block = '''
from utils.oom_retry import (
    oom_retry, is_oom, take_snapshot, restore_snapshot, wait_for_gpu, oom_inject,
)
render_imp = oom_retry(render_imp, "render_imp")
render_simp = oom_retry(render_simp, "render_simp")
render_depth = oom_retry(render_depth, "render_depth")
render_full = oom_retry(render_full, "render_full")

'''
src = src.replace(anchor, wrap_block + anchor, 1)

anchor2 = "        from gaussian_renderer.gof import integrate_gof as integrate\n"
assert src.count(anchor2) == 1
src = src.replace(anchor2, anchor2 + '    render = oom_retry(render, "render")\n    integrate = oom_retry(integrate, "integrate")\n', 1)

# ---- 2) boc doan forward+backward ----
START = "        # ---Select random viewpoint---\n"
END = "        # ---Backward pass---\n        loss.backward()\n"
assert src.count(START) == 1 and src.count(END) == 1
i0 = src.index(START)
i1 = src.index(END) + len(END)
region = src[i0:i1]

# hook kiem thu OOM gia ngay truoc backward (vo hieu neu khong dat MILO_OOM_INJECT)
region = region.replace(
    "        loss.backward()\n",
    '        if oom_inject(iteration, loss):\n            raise torch.cuda.OutOfMemoryError("synthetic OOM (self-test)")\n        loss.backward()\n', 1)

indented = "".join(("        " + ln if ln.strip() else ln) for ln in region.splitlines(keepends=True))

head = '''        _oom_attempt = 0
        while True:
            _oom_snap = take_snapshot(
                gaussians, locals().get("mesh_state"), viewpoint_stack, locals().get("viewpoint_idx_stack"),
                with_occupancy=bool(args.mesh_regularization and iteration >= mesh_config["start_iter"]),
            )
            try:
'''
tail = '''                break
            except Exception as _oom_e:
                if not is_oom(_oom_e):
                    raise
            # OOM tam thoi (GPU dung chung): tra VRAM, khoi phuc dung trang thai dau buoc, cho, lam lai
            _oom_attempt += 1
            render_pkg = image = viewspace_point_tensor = visibility_filter = radii = gt_image = loss = None
            rendered_depth_to_normals = rendered_normals = normal_error_map = depth_normal_loss = None
            detached_render_pkg = mesh_regularization_pkg = mesh_loss = mesh_render_pkg = None
            viewpoint_stack, viewpoint_idx_stack = restore_snapshot(_oom_snap, gaussians, locals().get("mesh_state"))
            wait_for_gpu(_oom_attempt, f"iteration {iteration}")
        _oom_snap = None
'''
src = src[:i0] + head + indented + tail + src[i1:]
(D / "train.py.new").write_text(src)
print("wrote train.py.new")
