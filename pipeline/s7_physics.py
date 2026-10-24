"""
S7: thuộc tính vật lý + mô hình MuJoCo (MJCF) cho cảnh.

v1: khối lượng / ma sát lấy từ config (S7_physics.objects); vật không có trong config thì khối lượng = khối lượng riêng
mặc định x thể tích mesh va chạm (lưu ý: vật rỗng như cốc sẽ bị nặng quá, nên khai khối lượng thật trong config).
Nâng cấp sau: VLM chọn vật liệu, PhysGS / PhysVGGT.

Hệ toạ độ: hệ sim của S6 (mét, mặt bàn z = 0, z hướng lên). Mỗi vật = 1 body có freejoint, gốc body = gốc hệ sim,
geom = các mảnh lồi collision_XX.obj của S6 (đã ở toạ độ sim) -> qpos ban đầu = đơn vị, pose sau này = phép biến đổi
cứng áp lên toạ độ sim của vật. Bàn = hộp, sàn = mặt phẳng.
Kiểm tra: compile, so tâm khối geom MuJoCo với tâm mesh gốc (đảm bảo MuJoCo không dời mesh).
Xuất: scene.xml, meshes/<vật>_XX.obj, physics.json
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import trimesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--s6", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg_all = json.loads(Path(a.cfg).read_text(encoding="utf-8"))
    cfg = cfg_all["S7_physics"]
    s6, out = Path(a.s6), Path(a.out)
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    box = json.loads((s6 / "support" / "table_box.json").read_text())
    floor = json.loads((s6 / "support" / "floor.json").read_text())

    bodies, phys = [], {}
    for od in sorted((s6 / "objects").iterdir()):
        name = od.name
        info = json.loads((od / "info.json").read_text())
        parts = sorted(od.glob("collision_*.obj"))
        if info.get("skipped") or not parts:
            phys[name] = {"skipped": info.get("skipped", "không có mesh va chạm")}
            continue
        meshes = [trimesh.load(p, force="mesh") for p in parts]
        vols = np.array([max(abs(m.volume), 1e-9) for m in meshes])
        oc = cfg["objects"].get(name, {})
        mass = oc.get("mass_kg", cfg["default_density_kg_m3"] * float(vols.sum()))
        fric = oc.get("friction", cfg["default_friction"])
        geoms = []
        for k, (p, m, v) in enumerate(zip(parts, meshes, vols)):
            mn = f"{name}_{k:02d}"
            shutil.copy2(p, out / "meshes" / f"{mn}.obj")
            geoms.append((mn, mass * v / vols.sum(), m.center_mass if m.is_volume else m.centroid))
        bodies.append((name, geoms, fric))
        if name in cfg.get("attach", {}):
            phys[name] = {"attached_to": cfg["attach"][name]}
        phys[name] = {**phys.get(name, {}),"mass_kg": mass, "mass_source": "config" if "mass_kg" in oc else "density x volume",
                      "friction": fric, "collision_parts": len(parts), "collision_volume_m3": float(vols.sum()),
                      "watertight_visual": info.get("watertight")}

    # gắn vật con vào vật cha (cùng 1 body cứng), vd bánh quy nằm trên đĩa có va chạm là bao lồi (khối đặc phủ cả bánh)
    attach = {c: p for c, p in cfg.get("attach", {}).items() if c in {b[0] for b in bodies} and p in {b[0] for b in bodies}}
    merged = []
    for name, geoms, fric in bodies:
        if name in attach:
            continue
        extra = [g for c, gs, _ in bodies if attach.get(c) == name for g in gs]
        merged.append((name, geoms + extra, fric))
    bodies = merged

    L = ['<mujoco model="safe-gs">',
         '  <compiler meshdir="meshes" angle="radian"/>',
         f'  <option timestep="{cfg_all["S8_sim"]["timestep"]}" gravity="0 0 -9.81" integrator="implicitfast"/>',
         '  <default><geom condim="4" solref="0.005 1" solimp="0.95 0.99 0.001"/></default>',
         '  <asset>']
    for name, geoms, _ in bodies:
        for mn, _, _ in geoms:
            L.append(f'    <mesh name="{mn}" file="{mn}.obj"/>')
    L.append('  </asset>')
    L.append('  <worldbody>')
    L.append(f'    <light pos="0 0 3" dir="0 0 -1"/>')
    L.append(f'    <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 {floor["z"]}" friction="{cfg["table_friction"]} 0.005 0.0001"/>')
    c, h = box["center"], box["half_size"]
    L.append(f'    <geom name="table" type="box" pos="{c[0]} {c[1]} {c[2]}" size="{h[0]} {h[1]} {h[2]}" '
             f'friction="{cfg["table_friction"]} 0.005 0.0001" rgba="0.8 0.7 0.5 1"/>')
    for name, geoms, fric in bodies:
        L.append(f'    <body name="{name}" pos="0 0 0">')
        L.append(f'      <freejoint name="{name}_free"/>')
        for mn, gm, _ in geoms:
            L.append(f'      <geom name="{mn}" type="mesh" mesh="{mn}" mass="{gm:.6f}" friction="{fric} 0.005 0.0001"/>')
        L.append('    </body>')
    L.append('  </worldbody>')
    L.append('</mujoco>')
    (out / "scene.xml").write_text("\n".join(L))

    import mujoco
    model = mujoco.MjModel.from_xml_path(str(out / "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # kiểm tra MuJoCo đặt mesh đúng chỗ: tâm geom (thế giới) vs tâm khối mesh gốc
    err = []
    for name, geoms, _ in bodies:
        for mn, _, cm in geoms:
            gid = model.geom(mn).id
            err.append(float(np.linalg.norm(data.geom_xpos[gid] - cm)))
    check = {"compiled": True, "nbody": int(model.nbody), "ngeom": int(model.ngeom),
             "geom_center_offset_max_m": max(err) if err else None,
             "total_mass_kg": float(sum(model.body_mass))}
    (out / "physics.json").write_text(json.dumps({"objects": phys, "attached": attach, "check": check},
                                                 indent=2, ensure_ascii=False))
    print(json.dumps(check), flush=True)
    assert check["geom_center_offset_max_m"] is None or check["geom_center_offset_max_m"] < 1e-3, \
        "MuJoCo dời mesh khỏi vị trí gốc -> pose của vật sẽ sai"


if __name__ == "__main__":
    main()
