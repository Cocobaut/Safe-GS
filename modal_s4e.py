"""Run S4e Support-Aware Non-Planar Surface Fill (3DGIC-style) on Modal.

    modal run modal_s4e.py::main --support-name dia --remove-names banh_quy --tag dia_banhquy_ft --ft-iters 1000
    modal run modal_s4e.py::table --tag table_all          # S4e fill inside S4d (same cut / protect / ft / eval as e8tex)
    modal run modal_s4e.py::pull --tag dia_banhquy_ft      # metrics.json + preview/*.jpg -> Data/teatime/pipeline/v1/S4e/<tag>/

Volume commits every 30 s while a job runs (modal_pipeline._Committer); S4e / S4d checkpoint their own fill PLY and ckpt/latest.ply,
so a re-run resumes.
"""
import json
from pathlib import Path

import modal

import modal_come as mc
import modal_pipeline as mp

app = modal.App("safe-gs-s4e")
V = Path(mc.VOL) / "teatime"
ENV = "PYTHONPATH=/opt/CoMe:/opt/pipeline python /opt/pipeline"
img = mp.pipe_image.add_local_file("modal_pipeline.py", "/root/modal_pipeline.py")


def _cfg():
    return {"scene": "teatime", "run_name": "v1", "S2_come": {"iterations": 30000}}


@app.function(image=img, timeout=3 * 3600, **{**mp.GPU_KW, "gpu": ["L40S", "L4"]})
def run_s4e(support_name: str = "dia", remove_names: str = "banh_quy", tag: str = "dia_banhquy", ft_iters: int = 0,
            extra: str = ""):
    mc.vol.reload()
    data, local = mp._prep_come(_cfg())
    src = V / "pipeline" / "v1" / "S4c" / "ctrl"
    out = V / "pipeline" / "v1" / "S4e" / tag
    out.mkdir(parents=True, exist_ok=True)
    cfgp = V / "pipeline" / "v3_sam3" / "S3_label" / "config_used.json"
    log = V / "logs" / f"s4e_{tag}.log"
    cmd = (f"{ENV}/s4e_surface_fill.py -m {local} -s {data} --iteration 30000 --cfg {cfgp} --out {out} --src {src} "
           f"--masks {V / 'sam3_masks' / 'labels'} --support_name {support_name} --remove_names {remove_names} "
           f"--ft_iters {ft_iters} {extra}")
    with mp._Committer():
        if not (out / "metrics.json").exists():
            mc._run_train(cmd, "/opt/CoMe", log)
    mc.vol.commit()
    return json.loads((out / "metrics.json").read_text())


@app.function(image=img, timeout=3 * 3600, **{**mp.GPU_KW, "gpu": ["L40S"]})
def run_table(tag: str = "table_all", ft_iters: int = 3000, extra: str = ""):
    """S4d_cutfill with --surface_fill 1 and every other argument copied from the e8tex run (logs/s4dtex_l40s*.log)."""
    mc.vol.reload()
    cfg = _cfg()
    data, local = mp._prep_come(cfg)
    src = V / "pipeline" / "v1" / "S4c" / "ctrl"
    out = V / "pipeline" / "v1" / "S4e" / tag
    out.mkdir(parents=True, exist_ok=True)
    cfgp = V / "pipeline" / "v1" / "S10_eval" / "config_used.json"
    labels = V / "sam3_masks" / "labels"
    log = V / "logs" / f"s4e_{tag}.log"
    common = f"-m {local} -s {data} --iteration 30000 --cfg {cfgp}"
    with mp._Committer():
        if not (out / "metrics.json").exists():
            mc._run_train(f"{ENV}/s4d_cutfill.py --surface_fill 1 --texture_fill 0 --tex_scale 0.5 {common} --out {out} --src {src} "
                          f"--labels {labels} --cut_dilate 2 --reach 2.0 --low_cm 3.0 --fill_dilate 4 --protect 1 "
                          f"--ft_iters {ft_iters} --lr_scale 0.5 --ft_mode masked --freeze_region 1 --freeze_dilate 0 --fill_grid 0.004 "
                          f"--inpainted {V / 'pipeline' / 'v1' / 'S4a_inpaint' / 'inpainted'} {extra}", "/opt/CoMe", log)
        if not (out / "ghost" / "ghost.json").exists():
            mc._run_train(f"{ENV}/s10_ghost.py {common} --out {out / 'ghost'} --s3 {out / 'S3like'} "
                          f"--s4b {out / 'S4like'} --labels {labels} --occluder_ids 20,30", "/opt/CoMe", log)
    mc.vol.commit()
    return {"metrics": json.loads((out / "metrics.json").read_text()),
            "ghost": json.loads((out / "ghost" / "ghost.json").read_text())}


@app.local_entrypoint()
def main(support_name: str = "dia", remove_names: str = "banh_quy", tag: str = "dia_banhquy", ft_iters: int = 0, extra: str = ""):
    res = run_s4e.remote(support_name=support_name, remove_names=remove_names, tag=tag, ft_iters=ft_iters, extra=extra)
    print("S4E_RESULT:", json.dumps(res, ensure_ascii=False, indent=2))


@app.local_entrypoint()
def table(tag: str = "table_all", ft_iters: int = 3000, extra: str = ""):
    res = run_table.remote(tag=tag, ft_iters=ft_iters, extra=extra)
    print("S4E_TABLE_RESULT:", json.dumps(res, ensure_ascii=False, indent=2))


@app.local_entrypoint()
def pull(tag: str = "dia_banhquy_ft"):
    out = Path(__file__).resolve().parent / "Data" / "teatime" / "pipeline" / "v1" / "S4e" / tag
    (out / "preview").mkdir(parents=True, exist_ok=True)
    rel = f"teatime/pipeline/v1/S4e/{tag}"
    files = [f"{rel}/{n}" for n in ("metrics.json", "fill.json", "cutfill.json", "ghost/ghost.json")]
    files += [e.path for e in mc.vol.listdir(f"{rel}/preview")]
    got = []
    for f in files:
        try:
            dst = out / Path(f).relative_to(rel)
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(dst, "wb") as fh:
                for c in mc.vol.read_file(f):
                    fh.write(c)
            got.append((str(dst.relative_to(out)), dst.stat().st_size))
        except FileNotFoundError:
            pass
    print(json.dumps(got, indent=1))
