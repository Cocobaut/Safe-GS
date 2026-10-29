"""End-to-end pipeline figure of Inpaint360GS + CoMe (v2) on LERF teatime (napkin, ID 44), same idea as
CoMe_Infusion/pipeline/Framework-Safe-GS.json: RGB images -> CoMe 3DGS -> IDs -> removal -> inpainting -> mesh -> object
mesh placed on the filled table.

  1. copies one representative image per step into images/, renamed step<N>_frame<NNNNN>[_tag] (real view 00127, the napkin is fully inside the image)
     or step<N>_vframe<NNNNN>[_tag] (virtual view 00020 = the v2 init view)
  2. renders the meshes (open3d hidden window): from the real camera of frame 00127 and from a side view below the table
  3. writes Framework-Inpaint360GS-CoMe.drawio (thumbnails embedded) and Framework-Inpaint360GS-CoMe.json (same
     node/edge listing as the InFusion file)

  python pipeline/build_pipeline.py            # from CoMe_Inpainting360GS/
"""
import base64
import io
import json
import shutil
import sys
from html import escape
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_v2" / "tools_mesh"))
import mesh_io as mio  # noqa: E402

OUT = ROOT / "pipeline"
IMG = OUT / "images"
V1 = ROOT / "outputs" / "teatime_napkin"
V2 = ROOT / "output_v2" / "teatime_napkin"
FR, VF = "00127", "00020"
W, H = 994, 738

# ------------------------------------------------------------------------------------------------ 1. copy images
COPIES = {
    f"step0_frame{FR}.jpg": V1 / f"step0_reuse_colmap/images/frame_{FR}.jpg",
    f"step2_frame{FR}_come_render.png": V1 / f"step2_reuse_come_renders/renders/frame_{FR}.png",
    f"step4_frame{FR}_new_renderer.png": V1 / f"step4_renderer_check/2_new_renderer/frame_{FR}.png",
    f"step5_frame{FR}_hqsam.png": V1 / f"step5_mask2d_hqsam/2_masks_color/frame_{FR}.png",
    f"step6_frame{FR}_synced.png": V1 / f"step6_sync_mask/2_masks_synced_color/frame_{FR}.png",
    f"step6_frame{FR}_ids.jpg": V1 / f"step6_sync_mask/3_ids_on_images/frame_{FR}.jpg",
    f"step7_frame{FR}_id_render.png": V1 / f"step7_id_distill/2_renders_id_color_num/train/frame_{FR}.png",
    f"step8_vframe{VF}_original.png": V1 / f"step8_removal_virtual_cam_v2/3_virtual_views_original/{VF}.png",
    f"step8_vframe{VF}_removed.png": V1 / f"step8_removal_virtual_cam_v2/4_virtual_views_removed/{VF}.png",
    f"step9_vframe{VF}_mask.png": V1 / f"step9_inpaint_masks_samt/2_masks_overlay/{VF}.png",
    f"step10_vframe{VF}_mask_dilated.png": V2 / f"step10_inpaint2d_lama/1_masks_dilated/{VF}.png",
    f"step10_vframe{VF}_lama_rgb.jpg": V2 / f"step10_inpaint2d_lama/2_inpainted_rgb/{VF}.JPG",
    f"step10_vframe{VF}_lama_depth.png": V2 / f"step10_inpaint2d_lama/4_inpainted_depth_vis/{VF}.png",
    f"step12_frame{FR}_removed.png": V2 / f"step12_render/train/1_removed/frame_{FR}.png",
    f"step12_frame{FR}_inpainted.png": V2 / f"step12_render/train/2_inpainted/frame_{FR}.png",
    "step14_height_map_top.png": V2 / "step14_object_closed_3dgs/height_map_top.png",
    "step14_support_height.png": V2 / "step14_object_closed_3dgs/support_height.png",
    "step15_cross_sections.png": V2 / "step15_object_mesh_merge/cross_sections.png",
}


# ------------------------------------------------------------------------------------------------ 2. renders
def cam_frame():
    c = next(x for x in json.loads((V1 / "step1_reuse_come_3dgs/cameras.json").read_text()) if x["img_name"] == f"frame_{FR}")
    R, p = np.array(c["rotation"]), np.array(c["position"])           # camera-to-world rotation, centre
    E = np.eye(4)
    E[:3, :3], E[:3, 3] = R.T, -R.T @ p
    return E, c["fx"], c["fy"]


def look_at(eye, target, up):
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, -up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    E = np.eye(4)
    E[:3, :3] = np.stack([x, y, z])
    E[:3, 3] = -E[:3, :3] @ eye
    return E


def render(geoms, E, fx, fy, path, point_size=2.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=W, height=H, visible=False)
    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.background_color = np.ones(3)
    opt.point_size = point_size
    opt.mesh_show_back_face = True
    prm = o3d.camera.PinholeCameraParameters()
    prm.intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, W / 2 - 0.5, H / 2 - 0.5)
    prm.extrinsic = E
    vis.get_view_control().convert_from_pinhole_camera_parameters(prm, allow_arbitrary=True)
    vis.poll_events()
    vis.update_renderer()
    img = np.asarray(vis.capture_screen_float_buffer(True))
    vis.destroy_window()
    Image.fromarray((img * 255).astype(np.uint8)).save(path)
    print("rendered", path.name, flush=True)


def o3d_mesh(path, colour=None):
    V, F, C = mio.read_mesh(path)
    m = mio.to_o3d(V, F)
    m.compute_vertex_normals()
    if colour is not None:
        m.paint_uniform_color(colour)
    elif C is not None:
        m.vertex_colors = o3d.utility.Vector3dVector(C)
    return m


def renders():
    plane = mio.load_plane()
    n, e1, e2 = plane["n"], plane["e1"], plane["e2"]
    E, fx, fy = cam_frame()
    g = np.load(V2 / "step14_object_closed_3dgs/support_grid.npz")
    ext = float(g["ext"])
    meta = json.loads((V2 / "step15_object_mesh_merge/summary.json").read_text())
    com = np.array(meta["object"]["center_of_mass"])
    # side view slightly below the table plane, looking at the napkin: shows its underside / what is under the table
    side = look_at(com + 1.1 * ext * (0.8 * e1 + 0.6 * e2) - 0.15 * ext * n, com - 0.05 * ext * n, n)
    above = look_at(com + 1.4 * ext * (0.8 * e1 + 0.6 * e2) + 1.8 * ext * n, com, n)
    sfx = 1.1 * W

    s13, s15 = V2 / "step13_come_mesh", V2 / "step15_object_mesh_merge"
    # the full mesh also holds far background / floaters: keep the part within 3 units of the napkin
    full = o3d_mesh(s13 / "mesh_come_v2_iter5000.ply")
    full.remove_vertices_by_mask(np.linalg.norm(np.asarray(full.vertices) - com, axis=1) > 3.0)
    render([full], E, fx, fy, IMG / f"step13_frame{FR}_mesh_scene.png")
    table = o3d_mesh(s13 / "mesh_come_v2_table.ply")
    render([table], E, fx, fy, IMG / f"step13_frame{FR}_mesh_table_filled.png")
    render([table], above, sfx, sfx, IMG / "step13_view_above_table_filled.png")

    # step 14: Gaussian centres, object (own colour) / added bottom (red) / added walls (blue)
    clean = PlyData.read(str(V2 / "step14_object_closed_3dgs/point_cloud_44_clean.ply")).elements[0]
    closed = PlyData.read(str(V2 / "step14_object_closed_3dgs/point_cloud_44_closed.ply")).elements[0]
    P = np.stack([np.asarray(closed[k], float) for k in "xyz"], 1)
    nc, nb = len(clean.data), int(np.asarray(g["footprint"]).sum())
    C0 = 0.28209479177387814
    col = np.clip(0.5 + C0 * np.stack([np.asarray(closed[f"f_dc_{k}"], float) for k in range(3)], 1), 0, 1)
    col[nc:nc + nb] = [0.85, 0.15, 0.15]
    col[nc + nb:] = [0.15, 0.35, 0.9]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc.colors = o3d.utility.Vector3dVector(col)
    render([pc], side, sfx, sfx, IMG / "step14_view_side_closed_gaussians.png", point_size=4.0)
    pc_clean = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[:nc]))
    pc_clean.colors = o3d.utility.Vector3dVector(col[:nc])
    render([pc_clean], side, sfx, sfx, IMG / "step14_view_side_object_gaussians.png", point_size=6.0)

    # step 15: raw CoMe napkin mesh vs closed solid, then placed on the filled table
    tstat = o3d.geometry.TriangleMesh(table)
    tstat.paint_uniform_color([0.75, 0.65, 0.5])
    raw = o3d_mesh(s15 / "mesh_come_napkin_v2_iter1.ply")
    closed_m = o3d_mesh(s15 / "mesh_napkin_closed.ply", colour=[0.85, 0.85, 0.9])
    render([raw], side, sfx, sfx, IMG / "step15_view_side_raw_come_mesh.png")
    render([closed_m], side, sfx, sfx, IMG / "step15_view_side_closed_solid.png")
    render([tstat, closed_m], above, sfx, sfx, IMG / "step15_view_above_scene.png")
    scene = o3d_mesh(s15 / "scene_table_napkin.ply")
    render([scene], E, fx, fy, IMG / f"step15_frame{FR}_scene_table_napkin.png")


# ------------------------------------------------------------------------------------------------ 3. diagram
# (id, x, y, w, h, kind, text) ; kind: stage | proc | data | new (v2 change) | img:<file>
B = "<b>{}</b><br><font style='font-size:10px'>{}</font>"
NODES = []
EDGES = []


def node(i, x, y, w, h, kind, text=""):
    NODES.append((i, x, y, w, h, kind, text))


def edge(a, b, label=""):
    EDGES.append((a, b, label))


def layout():
    TW, TH = 150, 112          # thumbnail size
    # ---------------- stage 1
    node("s1", 20, 20, 2110, 300, "stage", "Stage 1 – Data & CoMe 3DGS (reused from CoMe_Infusion, step 0–4)")
    node("rgb", 40, 70, 150, 50, "data", f"Multi-view RGB<br>(177 views, 994×738)")
    node("i_rgb", 40, 140, TW, TH, f"img:step0_frame{FR}.jpg")
    node("colmap", 240, 70, 150, 50, "proc", B.format("SfM", "COLMAP poses (step 0)"))
    node("train", 440, 70, 190, 50, "proc", B.format("3DGS Training", "CoMe: 3DGS + CoMe losses, 30k it (step 1)"))
    node("g3d", 680, 70, 150, 50, "data", "Scene 3DGS<br>(render, step 2)")
    node("i_g3d", 680, 140, TW, TH, f"img:step2_frame{FR}_come_render.png")
    node("rchk", 880, 70, 220, 50, "proc", B.format("Renderer check", "Inpaint360GS code + CoMe rasterizer, PSNR 44 dB (step 4)"))
    node("i_rchk", 915, 140, TW, TH, f"img:step4_frame{FR}_new_renderer.png")
    edge("rgb", "colmap"); edge("colmap", "train"); edge("train", "g3d"); edge("g3d", "rchk")

    # ---------------- stage 2
    node("s2", 20, 340, 2110, 300, "stage", "Stage 2 – Segmentation & 3D ID (Inpaint360GS, step 5–7)")
    node("hqsam", 40, 390, 170, 50, "proc", B.format("2D Segmentation", "HQ-SAM / CropFormer (step 5)"))
    node("i_hqsam", 50, 460, TW, TH, f"img:step5_frame{FR}_hqsam.png")
    node("m2d", 260, 390, 150, 50, "data", "2D masks<br>(per-frame IDs)")
    node("sync", 460, 390, 200, 50, "proc", B.format("Mask ID Sync", "Key Object Management on 3DGS (step 6)"))
    node("i_sync", 400, 460, TW, TH, f"img:step6_frame{FR}_synced.png")
    node("i_ids", 560, 460, TW, TH, f"img:step6_frame{FR}_ids.jpg")
    node("nid", 710, 390, 170, 50, "data", "Napkin ID = 44<br>(auto: IoU 0.73 with ref. mask)")
    node("distill", 930, 390, 210, 50, "proc", B.format("ID Distillation", "ID feature per Gaussian + classifier (step 7)"))
    node("i_distill", 960, 460, TW, TH, f"img:step7_frame{FR}_id_render.png")
    node("g3did", 1190, 390, 170, 50, "data", "3DGS with IDs<br>+ classifier.pth")
    edge("hqsam", "m2d"); edge("m2d", "sync"); edge("sync", "nid"); edge("nid", "distill"); edge("distill", "g3did")
    edge("rgb", "hqsam", "RGB"); edge("g3d", "sync", "3DGS")

    # ---------------- stage 3
    node("s3", 20, 660, 2110, 330, "stage", "Stage 3 – Removal & inpainting (Inpaint360GS v2 + CoMe loss, step 8–12)")
    node("remove", 40, 710, 190, 50, "proc", B.format("Object Removal", "ID 44 Gaussians (classifier + hull) (step 8)"))
    node("vcam", 280, 710, 190, 50, "proc", B.format("Virtual cameras", "30 views around object, rotation fixed (step 8)"))
    node("i_vorig", 250, 780, TW, TH, f"img:step8_vframe{VF}_original.png")
    node("i_vrem", 410, 780, TW, TH, f"img:step8_vframe{VF}_removed.png")
    node("samt", 520, 710, 170, 50, "proc", B.format("Inpaint masks", "SAM-Track on removed views (step 9)"))
    node("i_samt", 570, 780, TW, TH, f"img:step9_vframe{VF}_mask.png")
    node("lama", 740, 710, 220, 50, "new", B.format("2D Inpainting (v2)", "LaMa colour + depth, mask dilated 20 px (step 10)"))
    node("i_mdil", 730, 780, TW, TH, f"img:step10_vframe{VF}_mask_dilated.png")
    node("i_lrgb", 890, 780, TW, TH, f"img:step10_vframe{VF}_lama_rgb.jpg")
    node("i_ldep", 1050, 780, TW, TH, f"img:step10_vframe{VF}_lama_depth.png")
    node("initv", 1240, 710, 190, 50, "new", B.format("Init view selection (v2)", f"cleanest LaMa fill: view {VF}"))
    node("inp3d", 1480, 710, 240, 50, "proc", B.format("3DGS Inpainting", "Inpaint360GS loss + CoMe loss, 5000 it (step 11)"))
    node("final", 1770, 710, 170, 50, "data", "Final inpainted 3DGS<br>(render, step 12)")
    node("i_frem", 1700, 780, TW, TH, f"img:step12_frame{FR}_removed.png")
    node("i_finp", 1860, 780, TW, TH, f"img:step12_frame{FR}_inpainted.png")
    edge("remove", "vcam"); edge("vcam", "samt"); edge("samt", "lama"); edge("lama", "initv"); edge("initv", "inp3d")
    edge("inp3d", "final"); edge("g3did", "remove", "IDs")

    # ---------------- stage 4
    node("s4", 20, 1010, 2110, 300, "stage", "Stage 4 – Background mesh (CoMe, step 13)")
    node("mesh", 40, 1060, 200, 50, "proc", B.format("Mesh Extraction", "CoMe tetrahedra (extract_mesh_tets)"))
    node("i_mesh", 65, 1130, TW, TH, f"img:step13_frame{FR}_mesh_scene.png")
    node("crop", 300, 1060, 200, 50, "proc", B.format("Crop table", "prism over the RANSAC table plane"))
    node("table", 560, 1060, 200, 50, "data", "Filled table mesh<br>mesh_come_v2_table.ply")
    node("i_table", 490, 1130, TW, TH, f"img:step13_frame{FR}_mesh_table_filled.png")
    node("i_table2", 650, 1130, TW, TH, "img:step13_view_above_table_filled.png")
    edge("final", "mesh", "inpainted 3DGS"); edge("mesh", "crop"); edge("crop", "table")

    # ---------------- stage 5
    node("s5", 20, 1330, 2110, 330, "stage", "Stage 5 – Object mesh placed on the filled table (step 14–15)")
    node("objg", 40, 1380, 190, 50, "data", "Object Gaussians ID 44<br>(prob > 0.7, DBSCAN)")
    node("i_objg", 60, 1450, TW, TH, "img:step14_view_side_object_gaussians.png")
    node("close", 280, 1380, 250, 50, "new", B.format("Predict underside (step 14)", "bottom = support + ε (rays onto filled table), walls"))
    node("i_close", 240, 1450, TW, TH, "img:step14_view_side_closed_gaussians.png")
    node("i_hmap", 400, 1450, TW, TH, "img:step14_support_height.png")
    node("omesh", 580, 1380, 210, 50, "proc", B.format("Object mesh", "CoMe tetrahedra, 87 views seeing the napkin"))
    node("i_raw", 600, 1450, TW, TH, "img:step15_view_side_raw_come_mesh.png")
    node("solid", 840, 1380, 250, 50, "new", B.format("Height-field solid (step 15)", "top = CoMe mesh, bottom = table + ε, watertight"))
    node("i_solid", 800, 1450, TW, TH, "img:step15_view_side_closed_solid.png")
    node("i_xs", 960, 1450, TW, TH, "img:step15_cross_sections.png")
    node("place", 1140, 1380, 230, 50, "proc", B.format("Place on table", "same COLMAP frame, no vertex stitching"))
    node("i_place", 1160, 1450, TW, TH, "img:step15_view_above_scene.png")
    node("out", 1420, 1380, 250, 60, "data", "table_static.obj (static)<br>napkin_dynamic.obj (dynamic)<br>scene_table_napkin.ply")
    node("i_out", 1470, 1460, TW, TH, f"img:step15_frame{FR}_scene_table_napkin.png")
    edge("objg", "close"); edge("close", "omesh"); edge("omesh", "solid"); edge("solid", "place"); edge("place", "out")
    edge("remove", "objg", "removed ID 44"); edge("table", "close", "support"); edge("table", "place", "table")


STYLE = {
    "stage": "swimlane;horizontal=1;startSize=26;fillColor=#f5f5f5;strokeColor=#666666;fontStyle=1;align=left;"
             "spacingLeft=10;html=1;container=0;",
    "proc": "rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;fontSize=11;",
    "new": "rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;fontSize=11;",
    "data": "rounded=0;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=11;",
}


def thumb_uri(name, w=300):
    im = Image.open(IMG / name).convert("RGB")
    im.thumbnail((w, w))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82)
    return "data:image/jpeg," + base64.b64encode(buf.getvalue()).decode()


def write_drawio():
    cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
    for i, x, y, w, h, kind, text in NODES:
        if kind.startswith("img:"):
            name = kind[4:]
            st = (f"shape=image;verticalLabelPosition=bottom;verticalAlign=top;imageAspect=1;aspect=fixed;"
                  f"fontSize=8;image={thumb_uri(name)};")
            val = escape(name)
        else:
            st, val = STYLE[kind], escape(text)
        cells.append(f'<mxCell id="{i}" value="{val}" style="{st}" vertex="1" parent="1">'
                     f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    for k, (a, b, lab) in enumerate(EDGES):
        st = "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;fontSize=9;strokeColor=#333333;"
        if lab:
            st += "dashed=1;"
        cells.append(f'<mxCell id="e{k}" value="{escape(lab)}" style="{st}" edge="1" parent="1" source="{a}" target="{b}">'
                     f'<mxGeometry relative="1" as="geometry"/></mxCell>')
    legend = [("lg1", "proc", "process"), ("lg2", "new", "changed / added in v2"), ("lg3", "data", "data")]
    for k, (i, kind, text) in enumerate(legend):
        cells.append(f'<mxCell id="{i}" value="{text}" style="{STYLE[kind]}" vertex="1" parent="1">'
                     f'<mxGeometry x="{1700 + 130 * k}" y="70" width="120" height="30" as="geometry"/></mxCell>')
    xml = ('<mxfile host="build_pipeline.py"><diagram id="pipeline" name="Pipeline">'
           '<mxGraphModel dx="1900" dy="1700" grid="1" gridSize="10" page="0"><root>'
           + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>")
    (OUT / "Framework-Inpaint360GS-CoMe.drawio").write_text(xml, encoding="utf-8")

    # listing in the format of CoMe_Infusion/pipeline/Framework-Safe-GS.json
    import re
    plain = lambda s: re.sub(r"<[^>]+>", "", s.replace("<br>", "\n")).replace("&amp;", "&").replace("&gt;", ">")
    js = [{"id": "1", "type": "layer"}]
    for i, *_, kind, text in NODES:
        c = {"id": i, "type": "node", "parent": "1"}
        if kind.startswith("img:"):
            c["image"] = f"images/{kind[4:]}"
        else:
            c["label"] = plain(text)
            if "<" in text:
                c["html"] = text
        js.append(c)
    for k, (a, b, lab) in enumerate(EDGES):
        js.append({"id": f"e{k}", "type": "edge", "parent": "1", **({"label": lab} if lab else {}), "source": a, "target": b})
    (OUT / "Framework-Inpaint360GS-CoMe.json").write_text(
        json.dumps({"version": "32.6.1", "pages": [{"id": "pipeline", "name": "Pipeline", "cells": js}]},
                   indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    for name, src in COPIES.items():
        shutil.copy2(src, IMG / name)
    if "--no-render" not in sys.argv:
        renders()
    layout()
    missing = [k[4:] for *_, k, _ in NODES if k.startswith("img:") and not (IMG / k[4:]).exists()]
    if missing:
        raise SystemExit(f"missing images: {missing}")
    write_drawio()
    print("done:", OUT / "Framework-Inpaint360GS-CoMe.drawio")


if __name__ == "__main__":
    main()
