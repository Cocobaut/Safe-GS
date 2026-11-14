"""End-to-end pipeline figure of the Safe-GS base framework (CoMe + Inpaint360GS + InstantMesh), built from the
reference run outputs/LERF/ours/teatime_napkin (LERF teatime, object = napkin, ID 44):
RGB images -> CoMe 3DGS -> IDs -> removal -> inpainting -> CoMe mesh of the filled table -> InstantMesh object mesh
placed on the table.

  1. copies one representative image per step into images/, renamed step<N>_frame<NNNNN>[_tag] (real view 00127,
     the object is fully inside the image) or step<N>_vframe<NNNNN>[_tag] (virtual view 00020 = the init view)
  2. renders the meshes (open3d hidden window): from the real camera of frame 00127, from above and from the side
  3. writes Framework-Inpaint360GS-CoMe.drawio (thumbnails embedded) and Framework-Inpaint360GS-CoMe.json
     (node / edge listing); every number in the boxes is read from the summary.json files of the run

  python framework_diagram/build_pipeline.py [--run outputs/LERF/ours/teatime_napkin] [--no-render]
"""
import base64
import io
import json
import re
import sys
from html import escape
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "tools_mesh"))
import mesh_io as mio  # noqa: E402

OUT = Path(__file__).resolve().parent
IMG = OUT / "images"
RUN = ROOT / "outputs" / "LERF" / "ours" / "teatime_napkin"
if "--run" in sys.argv:
    RUN = Path(sys.argv[sys.argv.index("--run") + 1]).resolve()
FR, VF = "00127", "00020"
W, H = 994, 738

S = {  # step folders of the run
    0: "step0_reuse_colmap", 1: "step1_reuse_come_3dgs", 2: "step2_reuse_come_renders", 4: "step4_renderer_check",
    5: "step5_mask2d_hqsam", 6: "step6_sync_mask", 7: "step7_id_distill", 8: "step8_removal_virtual_cam",
    9: "step9_inpaint_masks_samt", 10: "step10_inpaint2d_lama", 11: "step11_inpaint_3dgs_come_loss",
    12: "step12_render", 13: "step13_come_mesh", 14: "step14_instantmesh", 15: "step15_object_mesh_merge",
}


def d(step):
    return RUN / S[step]


def js(path):
    p = RUN / path
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# ------------------------------------------------------------------------------------------------ 1. copy images
def copies():
    s14 = d(14)
    return {
        f"step0_frame{FR}.jpg": d(0) / f"images/frame_{FR}.jpg",
        f"step2_frame{FR}_come_render.png": d(2) / f"renders/frame_{FR}.png",
        f"step4_frame{FR}_new_renderer.png": d(4) / f"2_new_renderer/frame_{FR}.png",
        f"step5_frame{FR}_hqsam.png": d(5) / f"2_masks_color/frame_{FR}.png",
        f"step6_frame{FR}_synced.png": d(6) / f"2_masks_synced_color/frame_{FR}.png",
        f"step6_frame{FR}_ids.jpg": d(6) / f"3_ids_on_images/frame_{FR}.jpg",
        f"step7_frame{FR}_id_render.png": d(7) / f"2_renders_id_color_num/train/frame_{FR}.png",
        f"step8_vframe{VF}_original.png": d(8) / f"3_virtual_views_original/{VF}.png",
        f"step8_vframe{VF}_removed.png": d(8) / f"4_virtual_views_removed/{VF}.png",
        f"step9_vframe{VF}_mask.png": d(9) / f"2_masks_overlay/{VF}.png",
        f"step10_vframe{VF}_mask_dilated.png": d(10) / f"1_masks_dilated/{VF}.png",
        f"step10_vframe{VF}_lama_rgb.jpg": d(10) / f"2_inpainted_rgb/{VF}.JPG",
        f"step10_vframe{VF}_lama_depth.png": d(10) / f"4_inpainted_depth_vis/{VF}.png",
        f"step12_frame{FR}_removed.png": d(12) / f"train/1_removed/frame_{FR}.png",
        f"step12_frame{FR}_inpainted.png": d(12) / f"train/2_inpainted/frame_{FR}.png",
        f"step14_1_frame{FR}_object_mask_eval.png": s14 / f"1_object_gaussians/masks_eval/frame_{FR}.png",
        "step14_2_orbit_rgb.png": s14 / "2_orbit_video/frames/000.png",
        "step14_2_orbit_reliability.jpg": s14 / "2_orbit_video/frames/000_rel.jpg",
        "step14_3_selected_views.jpg": s14 / "3_view_selection/selected_views_k6.jpg",
    }


def grid_inputs(variant, name):
    """the 6 input views of the chosen InstantMesh variant as one 3x2 image"""
    ims = [Image.open(p).convert("RGB") for p in sorted((d(14) / "4_instantmesh" / variant / "inputs").glob("*.png"))]
    if not ims:
        return
    w, h = ims[0].size
    g = Image.new("RGB", (3 * w, 2 * h), "white")
    for k, im in enumerate(ims[:6]):
        g.paste(im.resize((w, h)), ((k % 3) * w, (k // 3) * h))
    g.save(IMG / name)


# ------------------------------------------------------------------------------------------------ 2. renders
def cam_frame():
    c = next(x for x in json.loads((d(1) / "cameras.json").read_text()) if x["img_name"] == f"frame_{FR}")
    R, p = np.array(c["rotation"]), np.array(c["position"])           # camera-to-world rotation, centre
    E = np.eye(4)
    E[:3, :3], E[:3, 3] = R.T, -R.T @ p
    return E, c["fx"], c["fy"]


def look_at(eye, target, up):
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)                       # open3d camera: x right, y down, z forward
    x /= np.linalg.norm(x)
    y = np.cross(z, x)                        # = -up projected: image up = world up
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


def table_mesh_path():
    s13 = d(13)
    return s13 / "mesh_come_v2_table.ply" if (s13 / "mesh_come_v2_table.ply").exists() else \
        max(s13.glob("*_table.ply"), key=lambda q: q.stat().st_mtime)


def renders():
    plane = mio.load_plane(run_dir=RUN)
    n, e1, e2 = plane["n"], plane["e1"], plane["e2"]
    E, fx, fy = cam_frame()
    s13, s14, s15 = d(13), d(14), d(15)
    obj = o3d_mesh(s15 / "mesh_object_placed.ply")
    OV = np.asarray(obj.vertices)
    com = OV.mean(0)
    ext = float(np.ptp(mio.to_uvh(plane, OV)[:, :2], 0).max())
    side = look_at(com + 2.0 * ext * (0.8 * e1 + 0.6 * e2) + 0.45 * ext * n, com, n)
    above = look_at(com + 1.4 * ext * (0.8 * e1 + 0.6 * e2) + 1.8 * ext * n, com, n)
    sfx = 1.1 * W

    # step 13: full CoMe mesh (keep the part within 3 units of the object: far background / floaters) + cropped table
    full_ply = next(iter(sorted(s13.glob("mesh_come_*iter*.ply"))), None)
    if full_ply is not None:
        full = o3d_mesh(full_ply)
        full.remove_vertices_by_mask(np.linalg.norm(np.asarray(full.vertices) - com, axis=1) > 3.0)
        render([full], E, fx, fy, IMG / f"step13_frame{FR}_mesh_scene.png")
        del full
    table = o3d_mesh(table_mesh_path())
    render([table], E, fx, fy, IMG / f"step13_frame{FR}_mesh_table_filled.png")
    render([table], above, sfx, sfx, IMG / "step13_view_above_table_filled.png")

    # step 14.1: object Gaussians (centres, own colour)
    pc_ply = next(iter(sorted((s14 / "1_object_gaussians").glob("point_cloud_*_object.ply"))), None)
    if pc_ply is not None:
        el = PlyData.read(str(pc_ply)).elements[0]
        P = np.stack([np.asarray(el[k], float) for k in "xyz"], 1)
        C0 = 0.28209479177387814
        col = np.clip(0.5 + C0 * np.stack([np.asarray(el[f"f_dc_{k}"], float) for k in range(3)], 1), 0, 1)
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
        pc.colors = o3d.utility.Vector3dVector(col)
        render([pc], side, sfx, sfx, IMG / "step14_1_view_side_object_gaussians.png", point_size=5.0)
    # step 14.5: InstantMesh mesh in the scene frame (before the step-15 translation)
    im = o3d_mesh(s14 / "5_final_mesh" / "mesh_object_instantmesh.ply")
    render([im], side, sfx, sfx, IMG / "step14_5_view_side_instantmesh.png")

    # step 15: object placed on the filled table
    tstat = o3d.geometry.TriangleMesh(table)
    tstat.paint_uniform_color([0.75, 0.65, 0.5])
    render([tstat, obj], side, sfx, sfx, IMG / "step15_view_side_placed.png")
    render([tstat, obj], above, sfx, sfx, IMG / "step15_view_above_scene.png")
    render([o3d_mesh(s15 / "scene_scene_demo_object.ply")], E, fx, fy, IMG / f"step15_frame{FR}_scene_demo_object.png")


# ------------------------------------------------------------------------------------------------ 3. diagram
# (id, x, y, w, h, kind, text) ; kind: stage | proc | data | new (added vs. Inpaint360GS / CoMe) | img:<file>
B = "<b>{}</b><br><font style='font-size:10px'>{}</font>"
NODES = []
EDGES = []


def node(i, x, y, w, h, kind, text=""):
    NODES.append((i, x, y, w, h, kind, text))


def edge(a, b, label=""):
    EDGES.append((a, b, label))


def f2(x, nd=2):
    return "?" if x is None else f"{x:.{nd}f}"


def mega(x):
    return "?" if x is None else f"{x / 1e6:.2f}M"


def layout():
    TW, TH = 150, 112          # thumbnail size
    s0, s1, s4 = js(f"{S[0]}/summary.json"), js(f"{S[1]}/summary.json"), js(f"{S[4]}/summary.json")
    s5, s6, s7 = js(f"{S[5]}/summary.json"), js(f"{S[6]}/napkin_id.json"), js(f"{S[7]}/summary.json")
    s8, s9, s11 = js(f"{S[8]}/summary.json"), js(f"{S[9]}/summary.json"), js(f"{S[11]}/summary.json")
    s13, crop = js(f"{S[13]}/done.json"), js(f"{S[13]}/{table_mesh_path().stem}_crop.json")
    s141, s145 = js(f"{S[14]}/1_object_gaussians/summary.json"), js(f"{S[14]}/5_final_mesh/summary.json")
    s142, s143 = js(f"{S[14]}/2_orbit_video/done.json"), js(f"{S[14]}/3_view_selection/selected_views_k6.json")
    s15, ev = js(f"{S[15]}/summary.json"), js("evaluation_summary.json")
    tid = s6.get("napkin_id", s7.get("napkin_id", "?"))
    size = (s0.get("image_sizes") or [["?", "?"]])[0]
    ev12 = ev.get("step12_inpainting_quality", {})
    best = s145.get("chosen_variant", "?")
    fm = s145.get("final_mesh", {})
    k = len(s143.get("chosen", [])) or "?"

    # ---------------- stage 1
    node("s1", 20, 20, 2110, 300, "stage", "Stage 1 – Data & CoMe 3DGS (step 0–4)")
    node("rgb", 40, 70, 150, 50, "data", f"Multi-view RGB<br>({s0.get('images', '?')} views, {size[0]}×{size[1]})")
    node("i_rgb", 40, 140, TW, TH, f"img:step0_frame{FR}.jpg")
    node("colmap", 240, 70, 150, 50, "proc", B.format("SfM", "COLMAP poses, PINHOLE (step 0)"))
    node("train", 440, 70, 190, 50, "proc", B.format("3DGS Training", f"CoMe: 3DGS + CoMe losses, 30k it, "
                                                     f"{mega(s1.get('num_gaussians'))} Gaussians (step 1)"))
    node("g3d", 680, 70, 150, 50, "data", "Scene 3DGS<br>(CoMe render, step 2)")
    node("i_g3d", 680, 140, TW, TH, f"img:step2_frame{FR}_come_render.png")
    tr, te = s4.get("psnr_vs_gt_image_train", {}), s4.get("psnr_vs_gt_image_test", {})
    same = "identical to CoMe render" if s4.get("max_abs_diff_vs_come_render") == 0 else "≈ CoMe render"
    node("rchk", 880, 70, 240, 60, "proc", B.format("Renderer check", f"Inpaint360GS code + CoMe rasterizer: {same};"
                                                    f" PSNR vs GT {f2(tr.get('mean'), 1)} / {f2(te.get('mean'), 1)} dB"
                                                    f" train / test (step 4)"))
    node("i_rchk", 925, 140, TW, TH, f"img:step4_frame{FR}_new_renderer.png")
    edge("rgb", "colmap"); edge("colmap", "train"); edge("train", "g3d"); edge("g3d", "rchk")

    # ---------------- stage 2
    node("s2", 20, 340, 2110, 300, "stage", "Stage 2 – Segmentation & 3D ID (Inpaint360GS, step 5–7)")
    ipf = s5.get("ids_per_frame", {})
    node("hqsam", 40, 390, 170, 50, "proc", B.format("2D Segmentation", f"HQ-SAM, {f2(ipf.get('mean'), 0)} IDs/frame (step 5)"))
    node("i_hqsam", 50, 460, TW, TH, f"img:step5_frame{FR}_hqsam.png")
    node("m2d", 260, 390, 150, 50, "data", "2D masks<br>(per-frame IDs)")
    node("sync", 460, 390, 200, 50, "proc", B.format("Mask ID Sync", "Key Object Management on 3DGS (step 6)"))
    node("i_sync", 400, 460, TW, TH, f"img:step6_frame{FR}_synced.png")
    node("i_ids", 560, 460, TW, TH, f"img:step6_frame{FR}_ids.jpg")
    node("nid", 710, 390, 170, 50, "data", f"Object ID = {tid}<br>(auto: IoU {f2(s6.get('iou_total'))} with ref. mask)")
    node("distill", 930, 390, 210, 50, "proc", B.format("ID Distillation", f"ID feature per Gaussian + classifier, "
                                                       f"render IoU {f2(s7.get('napkin_iou_render_vs_synced_mask'))} (step 7)"))
    node("i_distill", 960, 460, TW, TH, f"img:step7_frame{FR}_id_render.png")
    node("g3did", 1190, 390, 170, 50, "data", "3DGS with IDs<br>+ classifier.pth")
    edge("hqsam", "m2d"); edge("m2d", "sync"); edge("sync", "nid"); edge("nid", "distill"); edge("distill", "g3did")
    edge("rgb", "hqsam", "RGB"); edge("g3d", "sync", "3DGS")

    # ---------------- stage 3
    node("s3", 20, 660, 2110, 330, "stage", "Stage 3 – Removal & inpainting (Inpaint360GS + CoMe loss, step 8–12)")
    node("remove", 40, 710, 190, 50, "proc", B.format("Object Removal", f"ID {tid} Gaussians (classifier + hull) (step 8)"))
    node("vcam", 280, 710, 190, 50, "proc", B.format("Virtual cameras", f"{s8.get('n_virtual_views', '?')} views around "
                                                    f"the object (step 8)"))
    node("i_vorig", 250, 780, TW, TH, f"img:step8_vframe{VF}_original.png")
    node("i_vrem", 410, 780, TW, TH, f"img:step8_vframe{VF}_removed.png")
    node("samt", 520, 710, 190, 50, "proc", B.format("Inpaint masks", f"SAM-Track on removed views, "
                                                    f"{f2(s9.get('area_pct_min'), 1)}–{f2(s9.get('area_pct_max'), 1)} % (step 9)"))
    node("i_samt", 570, 780, TW, TH, f"img:step9_vframe{VF}_mask.png")
    node("lama", 760, 710, 220, 50, "new", B.format("2D Inpainting", "LaMa colour + depth, dilated mask (step 10)"))
    node("i_mdil", 730, 780, TW, TH, f"img:step10_vframe{VF}_mask_dilated.png")
    node("i_lrgb", 890, 780, TW, TH, f"img:step10_vframe{VF}_lama_rgb.jpg")
    node("i_ldep", 1050, 780, TW, TH, f"img:step10_vframe{VF}_lama_depth.png")
    node("initv", 1240, 710, 190, 50, "new", B.format("Init view selection", f"cleanest LaMa fill: view "
                                                     f"{s11.get('init_view', VF)} (step 11)"))
    node("inp3d", 1480, 710, 240, 50, "proc", B.format("3DGS Inpainting", f"Inpaint360GS loss + CoMe loss, 5000 it, "
                                                      f"{mega(s11.get('n_gaussians'))} Gaussians (step 11)"))
    node("final", 1770, 710, 190, 60, "data", f"Final inpainted 3DGS<br>(render, step 12)<br>test PSNR outside obj. "
                                              f"{f2(ev12.get('psnr_inpainted_outside'), 1)} dB")
    node("i_frem", 1700, 790, TW, TH, f"img:step12_frame{FR}_removed.png")
    node("i_finp", 1860, 790, TW, TH, f"img:step12_frame{FR}_inpainted.png")
    edge("remove", "vcam"); edge("vcam", "samt"); edge("samt", "lama"); edge("lama", "initv"); edge("initv", "inp3d")
    edge("inp3d", "final"); edge("g3did", "remove", "IDs")

    # ---------------- stage 4
    node("s4", 20, 1010, 2110, 300, "stage", "Stage 4 – Background mesh (CoMe, step 13)")
    node("mesh", 40, 1060, 220, 60, "proc", B.format("Mesh Extraction", f"CoMe tetrahedra (extract_mesh_tets), "
                                                    f"{mega(s13.get('n_vertices'))} V / {mega(s13.get('n_faces'))} F"))
    node("i_mesh", 65, 1140, TW, TH, f"img:step13_frame{FR}_mesh_scene.png")
    node("crop", 320, 1060, 200, 50, "proc", B.format("Crop table", "prism over the RANSAC table plane"))
    node("table", 580, 1060, 220, 60, "data", f"Filled table mesh<br>{table_mesh_path().name}<br>"
                                              f"{mega(crop.get('vertices'))} V / {mega(crop.get('faces'))} F")
    node("i_table", 510, 1140, TW, TH, f"img:step13_frame{FR}_mesh_table_filled.png")
    node("i_table2", 670, 1140, TW, TH, "img:step13_view_above_table_filled.png")
    edge("final", "mesh", "inpainted 3DGS"); edge("mesh", "crop"); edge("crop", "table")

    # ---------------- stage 5
    node("s5", 20, 1330, 2110, 340, "stage", "Stage 5 – Object mesh (InstantMesh, step 14) placed on the filled table (step 15)")
    obs = s141.get("observations", {})
    node("objg", 40, 1380, 210, 60, "new", B.format("14.1 Object Gaussians", f"ID {tid}: {s141.get('object_gaussians', '?')} "
                                                   f"Gaussians, seen by {obs.get('cameras_seeing_object', '?')} cameras"))
    node("i_objg", 30, 1460, TW, TH, "img:step14_1_view_side_object_gaussians.png")
    node("i_objm", 190, 1460, TW, TH, f"img:step14_1_frame{FR}_object_mask_eval.png")
    node("orbit", 300, 1380, 200, 60, "new", B.format("14.2 Orbit video", f"object-only 3DGS, {s142.get('frames', '?')} "
                                                     f"frames + reliability"))
    node("i_orb", 350, 1460, TW, TH, "img:step14_2_orbit_rgb.png")
    node("i_orbr", 510, 1460, TW, TH, "img:step14_2_orbit_reliability.jpg")
    node("vsel", 550, 1380, 200, 60, "new", B.format("14.3 View selection", f"coverage × reliability, k = {k} "
                                                    f"(+ uniform 6 views)"))
    node("i_vsel", 670, 1475, 225, 75, "img:step14_3_selected_views.jpg")
    node("lrm", 800, 1380, 210, 60, "proc", B.format("14.4 InstantMesh", "multi-view diffusion + LRM, 3 view sets"))
    node("i_lrm", 905, 1460, TW, 100, f"img:step14_4_inputs_{best}.png")
    vb = (s145.get("variants") or {}).get(best, {})
    node("pick", 1060, 1380, 230, 60, "new", B.format("14.5 Pick + largest part", f"best = {best}: IoU "
                                                     f"{f2(vb.get('mean_clean'), 3)} on real views, "
                                                     f"{'watertight' if fm.get('watertight') else 'open'}"))
    node("i_pick", 1065, 1460, TW, TH, "img:step14_5_view_side_instantmesh.png")
    node("place", 1340, 1380, 240, 60, "proc", B.format("Place on table (step 15)", f"one rigid shift along n "
                                                       f"{f2(100 * s15.get('translation_along_n', 0), 1)} cm, "
                                                       f"contact sunk {f2(100 * s15.get('after', {}).get('contact_sunk_frac', 0), 1)} %"))
    node("i_place", 1260, 1460, TW, TH, "img:step15_view_side_placed.png")
    node("i_place2", 1420, 1460, TW, TH, "img:step15_view_above_scene.png")
    node("out", 1630, 1380, 260, 60, "data", f"mesh_object_placed.ply ({fm.get('vertices', '?')} V)<br>"
                                             f"scene_scene_demo_object.ply<br>(table + object, two pieces)")
    node("i_out", 1680, 1460, TW, TH, f"img:step15_frame{FR}_scene_demo_object.png")
    edge("objg", "orbit"); edge("orbit", "vsel"); edge("vsel", "lrm"); edge("lrm", "pick"); edge("pick", "place")
    edge("place", "out")
    edge("g3did", "objg", f"ID {tid} Gaussians"); edge("table", "place", "table")


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
                  f"fontSize=8;image={thumb_uri(name, 450 if w > TWIDE else 300)};")
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
    legend = [("lg1", "proc", "process"), ("lg2", "new", "added / changed by Safe-GS"), ("lg3", "data", "data")]
    for k, (i, kind, text) in enumerate(legend):
        cells.append(f'<mxCell id="{i}" value="{text}" style="{STYLE[kind]}" vertex="1" parent="1">'
                     f'<mxGeometry x="{1640 + 160 * k}" y="70" width="150" height="30" as="geometry"/></mxCell>')
    xml = ('<mxfile host="build_pipeline.py"><diagram id="pipeline" name="Pipeline">'
           '<mxGraphModel dx="1900" dy="1700" grid="1" gridSize="10" page="0"><root>'
           + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>")
    (OUT / "Framework-Inpaint360GS-CoMe.drawio").write_text(xml, encoding="utf-8")

    plain = lambda s: re.sub(r"<[^>]+>", "", s.replace("<br>", "\n")).replace("&amp;", "&").replace("&gt;", ">")
    cells_js = [{"id": "1", "type": "layer"}]
    for i, *_, kind, text in NODES:
        c = {"id": i, "type": "node", "parent": "1"}
        if kind.startswith("img:"):
            c["image"] = f"images/{kind[4:]}"
        else:
            c["label"] = plain(text)
            if "<" in text:
                c["html"] = text
        cells_js.append(c)
    for k, (a, b, lab) in enumerate(EDGES):
        cells_js.append({"id": f"e{k}", "type": "edge", "parent": "1", **({"label": lab} if lab else {}),
                         "source": a, "target": b})
    (OUT / "Framework-Inpaint360GS-CoMe.json").write_text(
        json.dumps({"version": "32.6.1", "source_run": RUN.relative_to(ROOT).as_posix() if RUN.is_relative_to(ROOT) else str(RUN),
                    "pages": [{"id": "pipeline", "name": "Pipeline", "cells": cells_js}]},
                   indent=2, ensure_ascii=False), encoding="utf-8")


TWIDE = 160


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    for old in IMG.iterdir():                       # images/ only holds files generated by this script
        if old.is_file():
            old.unlink()
    for name, src in copies().items():
        Image.open(src).save(IMG / name) if src.suffix.lower() != Path(name).suffix.lower() else \
            (IMG / name).write_bytes(src.read_bytes())
    layout()
    best = next(k[len("img:step14_4_inputs_"):-4] for *_, k, _ in NODES if k.startswith("img:step14_4_inputs_"))
    grid_inputs(best, f"step14_4_inputs_{best}.png")
    if "--no-render" not in sys.argv:
        renders()
    missing = [k[4:] for *_, k, _ in NODES if k.startswith("img:") and not (IMG / k[4:]).exists()]
    if missing:
        raise SystemExit(f"missing images: {missing}")
    write_drawio()
    print("done:", OUT / "Framework-Inpaint360GS-CoMe.drawio")


if __name__ == "__main__":
    main()
