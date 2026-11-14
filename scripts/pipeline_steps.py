"""
Step 4-12 of Safe-GS (CoMe + Inpaint360GS): commands and exports, plain Python (no modal import).

Used by scripts/modal_come_pipeline.py (containers) and scripts/run_pipeline.py --dry-run (local).
The commands are the ones recorded in outputs/LERF/teatime_napkin/logs/*.log (reference run), with the paths and
ids taken from configs/datasets/*.yaml. Step 9 and Step 12 had no recorded command: they use the new
src/tools_safe/step9_samtrack_click.py and src/tools_safe/step12_package.py.

Volume layout (Volume `inpaint360gs`, mounted at /vol; dk = <dataset lower>/<scene>, e.g. lerf/teatime):
    come360/data/<dk>                       scene data S: images, images_<r>, sparse/0 (+ raw_hqsam, associated_hqsam)
    come360/output/<dk>                     scene model M: 3dgs_output (= CoMe step 1), distilled ID model (step 7)
    come360/ref/<dk>/come_renders           CoMe renders of step 2 (flat <image>.png) for step 4
    come360/runs/<dk>/<run>/data/<scene>    per-run copy RS of S  (steps 8-12 write into the data folder)
    come360/runs/<dk>/<run>/output/<scene>  per-run copy RM of M
    come360/runs/<dk>/<run>/{config,LaMa_data,LaMa_output,tracking_results,samt_assets}  linked into the code tree
    come360/export/<Dataset>/<run>/stepN_*  exported results (= outputs/<Dataset>/<run>/stepN_*), logs/
    come360/state/<dk>/state.json, come360/runs/<dk>/<run>/state.json
src/scene/dataset_readers.py switches on "inpaint360" in the source path: only the Inpaint360GS dataset has it (dk).
"""
from __future__ import annotations

VOL = "/vol"
ROOT = f"{VOL}/come360"
ROOT_REL = "come360"
CODE = "/opt/safegs"                       # src/ in the Step 4-12 image
LAMA_CODE = "/opt/safegs_lama"             # src/LaMa in the LaMa image

SCENE_STEPS = ["prep", "4", "5", "6", "7"]
RUN_STEPS = ["8", "9", "10a", "10l", "10b", "11", "12"]
ALL_STEPS = SCENE_STEPS + RUN_STEPS
STEP_DIRS = {"4": "step4_renderer_check", "5": "step5_mask2d_hqsam", "6": "step6_sync_mask", "7": "step7_id_distill",
             "8": "step8_removal_virtual_cam", "9": "step9_inpaint_masks_samt", "10": "step10_inpaint2d_lama",
             "11": "step11_inpaint_3dgs_come_loss", "12": "step12_render"}
LOG_NAMES = {"prep": "step3_prep_volume", "4": "step4_renderer_check", "5": "step5_mask2d_hqsam",
             "6": "step6_sync_mask", "7": "step7_id_distill", "8": "step8_removal_virtual_cam",
             "9": "step9_inpaint_masks_samt", "10a": "step10_inpaint2d_lama", "10l": "step10_inpaint2d_lama",
             "10b": "step10_inpaint2d_lama", "11": "step11_inpaint_3dgs_come_loss", "12": "step12_render"}


def expand_steps(spec: str) -> list[str]:
    """'4-12' / '8,9' / 'all' / '10' -> ordered list of parts (10 = 10a, 10l, 10b)."""
    spec = spec.strip().lower()
    if spec in ("all", ""):
        return list(ALL_STEPS)
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-")
            nums = [str(i) for i in range(int(a), int(b) + 1)]
        else:
            nums = [tok]
        for n in nums:
            if n == "10":
                out += ["10a", "10l", "10b"]
            elif n in ALL_STEPS:
                out.append(n)
            elif n in ("0", "1", "2", "3"):
                if "prep" not in out:
                    out.append("prep")
            else:
                raise ValueError(f"unknown step '{n}' (Step 4-12 runner: {ALL_STEPS})")
    return [s for s in ALL_STEPS if s in out]


def make_spec(cfg: dict) -> dict:
    """Small json-able dict sent to the containers (from common.datasets.resolve_run)."""
    ds, scene = cfg["dataset"], cfg["scene"]
    dk = f"{ds.lower()}/{scene}"
    click = cfg.get("samtrack_click")
    return {
        "dataset": ds, "scene": scene, "run": cfg["run"], "object": cfg.get("object"), "dk": dk,
        "resolution": int(cfg.get("resolution", 1)), "dataset_name": cfg.get("dataset_name", "others"),
        "target_id": cfg.get("target_id"), "surrounding": str(cfg.get("target_surrounding_id") or "None"),
        "click": list(click) if click else None, "has_gt": bool(cfg.get("has_object_free_gt", False)),
        "come_iteration": int((cfg.get("come") or {}).get("iterations", 30000)),
        "inpaint_iteration": int(cfg.get("inpaint_iteration", 5000)),
        "distill_iteration": int(cfg.get("distill_iteration", 2000)),
        "object_centered": bool(cfg.get("virtual_object_centered", False)),
        "removal_thresh": cfg.get("removal_thresh"),
    }


def paths(spec: dict) -> dict:
    dk, scene, run, ds = spec["dk"], spec["scene"], spec["run"], spec["dataset"]
    rr = f"{ROOT}/runs/{dk}/{run}"
    return {"S": f"{ROOT}/data/{dk}", "M": f"{ROOT}/output/{dk}", "REF": f"{ROOT}/ref/{dk}/come_renders",
            "RR": rr, "RS": f"{rr}/data/{scene}", "RM": f"{rr}/output/{scene}",
            "EXPORT": f"{ROOT}/export/{ds}/{run}", "EXPORT_REL": f"{ROOT_REL}/export/{ds}/{run}",
            "SCENE_STATE": f"{ROOT}/state/{dk}", "RUN_STATE": rr}


def image_folder(spec: dict) -> str:
    return "images" if spec["resolution"] == 1 else f"images_{spec['resolution']}"


PP = (f"export PYTHONPATH={CODE}:{CODE}/seg/detectron2:{CODE}/seg/detectron2/detectron2/projects/CropFormer:"
      "$PYTHONPATH; ")


def commands(spec: dict) -> dict[str, list[str]]:
    """part -> shell commands (cwd = CODE, except 10l which runs in the LaMa image, cwd = LAMA_CODE)."""
    p = paths(spec)
    S, M, RS, RM, E = p["S"], p["M"], p["RS"], p["RM"], p["EXPORT"]
    scene, r, ds_name = spec["scene"], spec["resolution"], spec["dataset_name"]
    itd = spec["distill_iteration"]
    cfg_rm = f"config/object_removal/{ds_name}/{scene}.json"
    cfg_in = f"config/object_inpaint/{ds_name}/{scene}.json"
    s_parent = S.rsplit("/", 1)[0]
    tid = spec["target_id"]
    s9 = (f"python tools_safe/step9_samtrack_click.py --frames {RM}/virtual/ours_object_removal/iteration_{itd}/renders "
          f"--export {E}/{STEP_DIRS['9']} --fallback_pred {RM}/virtual/ours_{itd}/objects_pred --target_ids {tid}")
    if spec["click"]:
        s9 += f" --click {spec['click'][0]} {spec['click'][1]}"
    sel = f"{E}/{STEP_DIRS['11']}/init_view_selection.json"
    lama = f"360_{scene}_virtual"
    return {
        "prep": [],                                   # python (images_<r>, flat CoMe renders), see modal_come_pipeline
        "4": [f"python tools_come/render_check.py -s {S} -m {M}/3dgs_output --resolution {r} --eval "
              f"--iteration {spec['come_iteration']} --come_renders {p['REF']} --out_dir {E}/{STEP_DIRS['4']}"],
        "5": [PP + f"python seg/raw_mask_sam.py --dataset_path {s_parent}/ --scene_name {scene} "
                   f"--image_folder {image_folder(spec)} --method hqsam"],
        "6": [PP + f"python seg/mask_associate.py --source_path {S} --model_path {M}/3dgs_output --resolution {r} "
                   f"--mask_generator hqsam --eval",
              PP + f"python tools/add_label_num_hqsam.py --source_path {S} --resolution {r} --mask_generator hqsam"],
        "7": [PP + f"python seg/distillation.py --source_path {S} --model_path {M} --vanilla_3dgs_path {M}/3dgs_output "
                   f"--resolution {r} --object_path associated_hqsam --eval",
              f"python render.py -m {M} --render_video"],
        "8": [f"python tools/init_configs.py --dataset_name {ds_name} --scene {scene} --target_id {tid} "
              f"--target_surronding_id {spec['surrounding']}"]
             + ([f"python -c \"import json; ps=['{cfg_rm}', '{cfg_in}']; "
                 f"[json.dump({{**json.load(open(p)), 'removal_thresh': {float(spec['removal_thresh'])}}}, "
                 f"open(p + '.tmp', 'w'), indent=4) for p in ps]\" && mv {cfg_rm}.tmp {cfg_rm} && mv {cfg_in}.tmp {cfg_in}"]
                if spec.get("removal_thresh") is not None else [])
             + [
f"python edit_object_removal.py --source_path {RS} -m {RM} --config_file {cfg_rm} --render_video",
              f"python tools/virtual_pose.py -s {RS} -m {RM} --config_file {cfg_rm}"
              + (" --object_centered" if spec.get("object_centered") else "")],
        "9": [s9],
        "10a": [f"python tools/prepare_lama_data.py -s {RS} -m {RM} -r {r} --inpaint2lama"],
        "10l": [f"python bin/predict_color.py --data_name {lama}", f"python bin/predict_depth.py --data_name {lama}"],
        "10b": [f"python tools/prepare_lama_data.py -s {RS} -m {RM} -r {r}"],
        "11": [f"mkdir -p {E}/{STEP_DIRS['11']} && python tools_come/select_init_view.py -s {RS} --out {sel}",
               f"export INPAINT360_INIT_VIEW=$(python -c \"import json;print(json.load(open('{sel}'))['selected'])\"); "
               f"python edit_object_removal_plyfusion.py -s {RS} -m {RM} --config_file {cfg_rm} && "
               f"python edit_object_inpaint.py -s {RS} -m {RM} --config_file {cfg_in} --resolution {r} --render_video"],
        "12": [f"python tools_safe/step12_package.py -m {RM} --out {E}/{STEP_DIRS['12']}"
               + (f" --has_gt_object_free --mask_dir {RS}/unseen_mask" if spec["has_gt"] else "")],
    }


def exports(spec: dict) -> dict[str, list[tuple[str, str]]]:
    """part -> [(source path or glob on the Volume, destination inside the step folder)] copied after the part.
    A source ending in '/' is a folder (copied whole); '*' globs are resolved at export time (newest match)."""
    p = paths(spec)
    S, M, RS, RM = p["S"], p["M"], p["RS"], p["RM"]
    itd, iti, r = spec["distill_iteration"], spec["inpaint_iteration"], spec["resolution"]
    scene, ds_name, tid = spec["scene"], spec["dataset_name"], spec["target_id"]
    lama = f"360_{scene}_virtual"
    return {
        "5": [(f"{S}/raw_hqsam/", "1_masks")],
        "6": [(f"{S}/associated_hqsam/", "1_masks_synced"), (f"{S}/associated_hqsam_color/", "2_masks_synced_color"),
              (f"{S}/{image_folder(spec) if r != 1 else 'images_1'}_num/", "3_ids_on_images"),
              (f"{S}/associated_hqsam/scene.json", "scene.json")],
        "7": [(f"{M}/point_cloud/iteration_{itd}/point_cloud.ply", f"point_cloud_id_iter{itd}.ply"),
              (f"{M}/point_cloud/iteration_{itd}/classifier.pth", "classifier.pth"),
              (f"{M}/train/ours_{itd}/renders/", "1_renders_rgb"),
              (f"{M}/train/ours_{itd}/objects_pred_color/", "2_renders_id_color_num"),
              (f"{M}/video/ours_{itd}/final_video.mp4", "video_rgb_id.mp4")],
        "8": [(f"{CODE}/config/object_removal/{ds_name}/{scene}.json", "1_config/object_removal.json"),
              (f"{CODE}/config/object_inpaint/{ds_name}/{scene}.json", "1_config/object_inpaint.json"),
              (f"{RM}/point_cloud_object_removal/iteration_{itd}/point_cloud.ply", "2_object_removal/point_cloud_removed.ply"),
              (f"{RM}/point_cloud_object_removal/iteration_{itd}/point_cloud_{tid}.ply",
               f"2_object_removal/removed_object_id{tid}.ply"),
              (f"{RM}/virtual/ours_{itd}/renders/", "3_virtual_views_original"),
              (f"{RM}/virtual/ours_object_removal/iteration_{itd}/renders/", "4_virtual_views_removed"),
              (f"{RM}/virtual/ours_{itd}/objects_pred_color/", "5_virtual_views_ids"),
              (f"{RM}/video/ours_ours_object_removal_*/final_video.mp4", "2_object_removal/video_removed_scene.mp4"),
              (f"{RM}/video/ours_ours_removed_obj_{tid}/final_video.mp4", "2_object_removal/video_removed_object.mp4")],
        "10b": [(f"{RS}/inpaint_2d_unseen_mask_virtual/", "1_masks_dilated"),
                (f"{RR_LAMA_OUT(spec)}/color/{lama}/", "2_inpainted_rgb"),
                (f"{RR_LAMA_OUT(spec)}/depth/{lama}/", "3_inpainted_depth_npy")],
        "11": [(f"{RM}/point_cloud_object_inpaint_virtual/iteration_{iti}/point_cloud.ply",
                f"point_cloud_inpainted_iter{iti}.ply"),
               (f"{RM}/loss_log.json", "loss_log.json"), (f"{RM}/used_virtual_views.json", "used_virtual_views.json")],
    }


def RR_LAMA_OUT(spec: dict) -> str:  # noqa: N802
    return f"{paths(spec)['RR']}/LaMa_output"


def run_links(spec: dict) -> dict[str, str]:
    """code-tree path -> per-run Volume folder (symlinks made before every run step)."""
    rr, ds_name = paths(spec)["RR"], spec["dataset_name"]
    return {f"{CODE}/config/object_removal/{ds_name}": f"{rr}/config/object_removal/{ds_name}",
            f"{CODE}/config/object_inpaint/{ds_name}": f"{rr}/config/object_inpaint/{ds_name}",
            f"{CODE}/LaMa/data": f"{rr}/LaMa_data", f"{CODE}/LaMa/output": f"{rr}/LaMa_output",
            f"{CODE}/Segment-and-Track-Anything/tracking_results": f"{rr}/tracking_results",
            f"{CODE}/Segment-and-Track-Anything/assets": f"{rr}/samt_assets",
            f"{LAMA_CODE}/data": f"{rr}/LaMa_data", f"{LAMA_CODE}/output": f"{rr}/LaMa_output"}


def describe(spec: dict, parts: list[str]) -> str:
    """Human-readable plan (dry run)."""
    p, cmds = paths(spec), commands(spec)
    lines = [f"# {spec['dataset']}/{spec['run']}  scene={spec['scene']} target_id={spec['target_id']} "
             f"surrounding={spec['surrounding']} resolution={spec['resolution']} click={spec['click']}",
             f"#   S={p['S']}  M={p['M']}", f"#   RS={p['RS']}  RM={p['RM']}", f"#   export={p['EXPORT']}"]
    for part in parts:
        where = "LaMa image (T4), cwd src/LaMa" if part == "10l" else "Step 4-12 image (L4/L40S), cwd src"
        lines.append(f"\n[{part}] {LOG_NAMES[part]}  ({where})")
        if part == "prep":
            lines.append("  images_<r> from images, CoMe step-2 renders flattened to REF")
        if part == "8":
            lines.append(f"  copy S -> RS and M -> RM (without chkpnt*.pth), link {list(run_links(spec))[:2]} ...")
        for c in cmds[part]:
            lines.append("  $ " + c.replace(PP, "<PYTHONPATH=src:detectron2:CropFormer> "))
    return "\n".join(lines)
