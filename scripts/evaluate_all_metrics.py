"""
Safe-GS evaluation for the council report (PLAN_REBUILD_BASE_FRAMEWORK.md, Phase 5). CPU only, relative paths.

Reads outputs/<Dataset>/<run>/ and writes
    <run_dir>/evaluation_summary.json
    outputs/council_report_summary.md + outputs/council_report_summary.csv   (one row per evaluated run)

Metric groups (JSON keys checked against the reference run outputs/LERF/teatime_napkin):
  Appearance
    step 4   step4_renderer_check/summary.json: psnr_vs_come_render.{mean,min}, max_abs_diff_vs_come_render,
             psnr_vs_gt_image_{train,test}.mean, n_gaussians
    step 12  step12_render/test/{0_gt,1_removed,2_inpainted}:
             - outside the object mask (all datasets): PSNR / SSIM of removed and inpainted vs the photo
               (mask = step6 synced ID == target_id, dilated) -> the edit must not damage the rest of the scene
             - dataset with object-free GT (Inpaint360GS): PSNR / SSIM inside + full image, LPIPS and FID
               if the `lpips` / `pytorch_fid` packages are installed (else reported as null)
             - LERF / Replica: no GT under the object -> no inside-mask metric (stated in the report)
    step 11  init view of LaMa (init_view_selection.json), n Gaussians after inpainting
  Geometry
    step 13  done.json (full mesh) + *_table_crop.json (table mesh); flatness of the table under the object:
             heights of the table vertices inside the object footprint w.r.t. the table plane (RMSE, p05/p50/p95,
             coverage of the footprint grid within tolerance)
    step 14  5_final_mesh/summary.json: chosen variant, mean IoU (clean / all real views) per variant
    step 15  step15_object_mesh_merge/summary.json: vertices inside the table (after, before), min clearance (= gap),
             share of vertices with table below, boundary / non-manifold edges of the object mesh

  python scripts/evaluate_all_metrics.py --dataset LERF --run teatime_napkin
  python scripts/evaluate_all_metrics.py --all              # every outputs/<Dataset>/<run> with step folders
  python scripts/evaluate_all_metrics.py --dataset LERF --run teatime_napkin --skip-geometry --skip-images
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path[:0] = [str(SRC), str(SRC / "tools_mesh")]

from common.paths import DATASETS, OUTPUTS_DIR, get_method_dir, get_run_dir, step_dir  # noqa: E402


def _json(p: Path):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _r(x, n=4):
    return None if x is None else round(float(x), n)


# ==================================================================================================== appearance
def eval_step4(run: Path) -> dict | None:
    s = _json(step_dir(run, 4) / "summary.json")
    if not s:
        return None
    g = lambda k, f="mean": (s.get(k) or {}).get(f)  # noqa: E731
    return {"n_views": s.get("n_views"), "n_gaussians": s.get("n_gaussians"),
            "psnr_vs_come_render_mean": g("psnr_vs_come_render"), "psnr_vs_come_render_min": g("psnr_vs_come_render", "min"),
            "max_abs_diff_vs_come_render": s.get("max_abs_diff_vs_come_render"),
            "psnr_vs_gt_train_mean": g("psnr_vs_gt_image_train"), "psnr_vs_gt_test_mean": g("psnr_vs_gt_image_test"),
            "renderer_identical": s.get("max_abs_diff_vs_come_render") == 0}


def _load_rgb(p: Path) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(p).convert("RGB"), np.float64) / 255.0


def _psnr(a, b, m=None):
    d = (a - b) ** 2
    mse = d.mean() if m is None else (d[m].mean() if m.any() else np.nan)
    return float("inf") if mse == 0 else float(-10 * np.log10(mse))


def _ssim_map(a, b):
    """SSIM map (gaussian window sigma 1.5, as the 3DGS / Inpaint360GS ssim), averaged over RGB."""
    from scipy.ndimage import gaussian_filter
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    out = []
    for ch in range(3):
        x, y = a[..., ch], b[..., ch]
        f = lambda z: gaussian_filter(z, 1.5, truncate=3.5)  # noqa: E731
        mx, my = f(x), f(y)
        sxx, syy, sxy = f(x * x) - mx * mx, f(y * y) - my * my, f(x * y) - mx * my
        out.append(((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2)))
    return np.mean(out, 0)


def _object_mask(run: Path, name: str, target_id, shape, dilate: int) -> np.ndarray | None:
    from PIL import Image
    if target_id is None:
        return None
    p = step_dir(run, 6) / "1_masks_synced" / f"{name}.png"
    if not p.exists():
        return None
    ids = Image.open(p)
    if ids.size != (shape[1], shape[0]):
        ids = ids.resize((shape[1], shape[0]), Image.NEAREST)
    m = np.asarray(ids) == int(target_id)
    if dilate > 0 and m.any():
        from scipy.ndimage import binary_dilation
        m = binary_dilation(m, iterations=dilate)
    return m


def _dataset_mask(split_dir: Path, name: str, shape) -> np.ndarray | None:
    """unseen_mask of the Inpaint360GS dataset (copied by step12_package.py to <split>/mask)."""
    from PIL import Image
    c = sorted((split_dir / "mask").glob(f"{name}.*")) if (split_dir / "mask").exists() else []
    if not c:
        return None
    im = Image.open(c[0]).convert("L")
    if im.size != (shape[1], shape[0]):
        im = im.resize((shape[1], shape[0]), Image.NEAREST)
    return np.asarray(im) > 127


def eval_step12(run: Path, cfg: dict, dilate: int = 10, max_views: int = 0) -> dict | None:
    s12 = step_dir(run, 12)
    test = s12 / "test"
    if not (test / "0_gt").exists():
        return None
    has_gt = bool(cfg.get("has_object_free_gt"))
    split = "test"
    if has_gt and (s12 / "inpaint" / "0_gt").exists():   # Inpaint360GS: object-free photos of the "inpaint" cameras
        test, split = s12 / "inpaint", "inpaint"
    names = sorted(p.stem for p in (test / "0_gt").glob("*.png"))
    if max_views:
        names = names[:max_views]
    tid = cfg.get("target_id")
    rows = []
    for n in names:
        gt = _load_rgb(test / "0_gt" / f"{n}.png")
        rem_p, inp_p = test / "1_removed" / f"{n}.png", test / "2_inpainted" / f"{n}.png"
        if not inp_p.exists():
            continue
        inp = _load_rgb(inp_p)
        rem = _load_rgb(rem_p) if rem_p.exists() else None
        m = _dataset_mask(test, n, gt.shape[:2]) if split == "inpaint" else None
        m = m if m is not None else _object_mask(run, n, tid, gt.shape[:2], dilate)
        out = ~m if m is not None else np.ones(gt.shape[:2], bool)
        ss_i = _ssim_map(inp, gt)
        row = {"view": n, "object_px_pct": _r(100 * (m.mean() if m is not None else 0), 3),
               "psnr_inpainted_outside": _psnr(inp, gt, out), "ssim_inpainted_outside": float(ss_i[out].mean())}
        if rem is not None:
            row["psnr_removed_outside"] = _psnr(rem, gt, out)
            row["ssim_removed_outside"] = float(_ssim_map(rem, gt)[out].mean())
        if has_gt:                                   # GT is object-free: the whole image is comparable
            row["psnr_inpainted_full"] = _psnr(inp, gt)
            row["ssim_inpainted_full"] = float(ss_i.mean())
            if m is not None and m.any():
                row["psnr_inpainted_inside"] = _psnr(inp, gt, m)
                row["ssim_inpainted_inside"] = float(ss_i[m].mean())
        rows.append(row)
    if not rows:
        return None

    def mean(k):
        v = [r[k] for r in rows if k in r and r[k] is not None and np.isfinite(r[k])]
        return _r(np.mean(v)) if v else None
    res = {"split": split, "n_test_views": len(rows), "views_with_object": sum(1 for r in rows if r["object_px_pct"]),
           "mask": ("dataset unseen_mask (step12_render/inpaint/mask)" if split == "inpaint" else
                    f"step6 synced id == {tid}, dilated {dilate}px" if tid is not None else "none (no target_id)"),
           "has_object_free_gt": has_gt,
           **{k: mean(k) for k in ("psnr_inpainted_outside", "ssim_inpainted_outside", "psnr_removed_outside",
                                   "ssim_removed_outside", "psnr_inpainted_full", "ssim_inpainted_full",
                                   "psnr_inpainted_inside", "ssim_inpainted_inside")}}
    if has_gt:
        res.update(_lpips_fid(test))
    else:
        res["note"] = "no ground truth without the object (LERF / Replica): only metrics outside the object mask"
    res["per_view"] = [{k: (_r(v) if isinstance(v, float) else v) for k, v in r.items()} for r in rows]
    return res


def _lpips_fid(test: Path) -> dict:
    out = {"lpips_inpainted_full": None, "fid_inpainted": None}
    try:
        import lpips
        import torch
        net = lpips.LPIPS(net="vgg", verbose=False)
        vals = []
        for p in sorted((test / "2_inpainted").glob("*.png")):
            g = test / "0_gt" / p.name
            t = [torch.from_numpy(_load_rgb(x)).permute(2, 0, 1)[None].float() * 2 - 1 for x in (p, g)]
            with torch.no_grad():
                vals.append(float(net(t[0], t[1])))
        out["lpips_inpainted_full"] = _r(np.mean(vals))
    except Exception as e:  # noqa: BLE001
        out["lpips_note"] = f"skipped ({type(e).__name__}: {e})"[:200]
    try:
        from pytorch_fid import fid_score
        out["fid_inpainted"] = _r(fid_score.calculate_fid_given_paths(
            [str(test / "2_inpainted"), str(test / "0_gt")], batch_size=8, device="cpu", dims=2048), 3)
    except Exception as e:  # noqa: BLE001
        out["fid_note"] = f"skipped ({type(e).__name__}: {e})"[:200]
    return out


def eval_step11(run: Path) -> dict | None:
    d = step_dir(run, 11)
    s, sel = _json(d / "summary.json"), _json(d / "init_view_selection.json")
    if not s and not sel:
        return None
    s = s or {}
    if s.get("n_gaussians") is None:                     # new runner: count from the ply header
        plys = sorted(d.glob("point_cloud_inpainted_iter*.ply"))
        if plys:
            with open(plys[-1], "rb") as f:
                props = []
                for line in f:
                    t = line.decode(errors="replace").split()
                    if t[:2] == ["element", "vertex"]:
                        s["n_gaussians"] = int(t[2])
                    elif t[:1] == ["property"]:
                        props.append(t[-1])
                    if t[:1] == ["end_header"]:
                        break
            s.setdefault("has_confidence", "confidence" in props)
            s.setdefault("has_filter_3D", "filter_3D" in props)
    return {"n_gaussians": s.get("n_gaussians"), "has_confidence": s.get("has_confidence"),
            "has_filter_3D": s.get("has_filter_3D"), "virtual_views_used": s.get("virtual_views_used"),
            "init_view": (sel or {}).get("selected", s.get("init_view")),
            "init_view_score": ((sel or {}).get("selected_row") or {}).get("score")}


# ==================================================================================================== geometry
def _table_mesh(run: Path) -> Path | None:
    d = step_dir(run, 13)
    if (d / "mesh_come_v2_table.ply").exists():
        return d / "mesh_come_v2_table.ply"
    c = sorted(d.glob("*_table.ply"), key=lambda p: p.stat().st_mtime)
    return c[-1] if c else None


def eval_step13(run: Path, plane_path: str | None, skip_geometry: bool) -> dict | None:
    d = step_dir(run, 13)
    done = _json(d / "done.json")
    if not done and not d.exists():
        return None
    res = {"full_mesh_vertices": (done or {}).get("n_vertices"), "full_mesh_faces": (done or {}).get("n_faces"),
           "source_ply": Path(str((done or {}).get("source_ply", ""))).name or None}
    table = _table_mesh(run)
    crop = _json(table.with_name(table.stem + "_crop.json")) if table else None
    if crop:
        res.update(table_mesh=table.name, table_vertices=crop.get("vertices"), table_faces=crop.get("faces"))
    if skip_geometry or table is None:
        return res
    # object-agnostic names first, then the pre-rename napkin names (results re-downloaded from old Modal Volumes)
    cands = [step_dir(run, 15) / "mesh_object_placed.ply", step_dir(run, 15) / "mesh_napkin_placed.ply",
             step_dir(run, 14) / "5_final_mesh" / "mesh_object_instantmesh.ply",
             step_dir(run, 14) / "5_final_mesh" / "mesh_napkin_instantmesh.ply"]
    obj = next((p for p in cands if p.exists()), cands[0])
    if not obj.exists():
        res["under_object"] = "skipped: no object mesh (step 14/15) to define the footprint"
        return res
    res["under_object"] = _under_object(run, table, obj, plane_path)
    return res


def _under_object(run: Path, table: Path, obj: Path, plane_path: str | None) -> dict:
    """Table flatness in the object footprint: grid of cells (extent/40) over the footprint (convex hull of the
    object in the table plane); a cell is 'covered' when it holds a table vertex within tol = 2 % of the object
    extent of the plane; RMSE / percentiles of the heights of those near-plane vertices."""
    import mesh_io as mio
    from scipy.spatial import Delaunay
    plane = mio.load_plane(plane_path, run_dir=run)
    OV, _, _ = mio.read_mesh(obj)
    ouvh = mio.to_uvh(plane, OV)
    extent = float(np.ptp(ouvh[:, :2], 0).max())
    tol, cell = 0.02 * extent, extent / 40
    hull = Delaunay(ouvh[:, :2])
    TV, _, _ = mio.read_mesh(table)
    tuvh = mio.to_uvh(plane, TV)
    lo, hi = ouvh[:, :2].min(0), ouvh[:, :2].max(0)
    box = np.all((tuvh[:, :2] >= lo) & (tuvh[:, :2] <= hi), 1)
    cand = tuvh[box]
    inside = cand[hull.find_simplex(cand[:, :2]) >= 0]
    near = inside[np.abs(inside[:, 2]) <= 10 * tol]                 # the table layer (not the floor / far geometry)
    gx = np.arange(lo[0], hi[0] + cell, cell)
    gy = np.arange(lo[1], hi[1] + cell, cell)
    cx, cy = np.meshgrid(gx[:-1] + cell / 2, gy[:-1] + cell / 2)
    cells = np.stack([cx.ravel(), cy.ravel()], 1)
    cells = cells[hull.find_simplex(cells) >= 0]
    ij = lambda P: (np.floor((P[:, 0] - lo[0]) / cell).astype(int), np.floor((P[:, 1] - lo[1]) / cell).astype(int))  # noqa: E731
    covered = set(zip(*ij(near[np.abs(near[:, 2]) <= tol]))) if len(near) else set()
    ci = set(zip(*ij(cells)))
    h = near[:, 2] if len(near) else np.zeros(1)
    return {"object_mesh": obj.name, "object_extent": _r(extent), "cell": _r(cell), "tol": _r(tol),
            "footprint_cells": len(ci), "table_vertices_in_footprint": int(len(near)),
            "coverage_within_tol": _r(len(ci & covered) / max(len(ci), 1)),
            "height_rmse": _r(np.sqrt(np.mean(h ** 2)), 5), "height_p05_p50_p95": [_r(v, 5) for v in np.percentile(h, [5, 50, 95])]}


def eval_step14(run: Path) -> dict | None:
    s = _json(step_dir(run, 14) / "5_final_mesh" / "summary.json")
    if not s:
        return None
    var = s.get("variants") or {}
    ch = s.get("chosen_variant")
    return {"chosen_variant": ch, "rule": s.get("rule"),
            "mean_iou_clean": (var.get(ch) or {}).get("mean_clean"), "mean_iou_all": (var.get(ch) or {}).get("mean_all"),
            "n_views_clean": (var.get(ch) or {}).get("n_clean"), "n_views_all": (var.get(ch) or {}).get("n_all"),
            "variants": {k: {kk: v.get(kk) for kk in ("mean_clean", "mean_all")} for k, v in var.items()}}


def eval_step15(run: Path) -> dict | None:
    s = _json(step_dir(run, 15) / "summary.json")
    if not s:
        return None
    af, bf = s.get("after") or {}, s.get("before") or {}
    ob = next((v for k, v in s.items() if isinstance(v, dict) and "edges" in v), {})
    e = ob.get("edges") or {}
    cl = af.get("clearance_min_p05_p50") or [None]
    return {"vertices_inside_table_after": af.get("vertices_inside_table"),
            "vertices_inside_table_before": bf.get("vertices_inside_table"),
            "clearance_min": cl[0], "clearance_p05_p50": cl[1:], "gap": s.get("gap"),
            "vertices_with_table_below": af.get("vertices_with_table_below"),
            "translation_along_n": s.get("translation_along_n"),
            "object_vertices": ob.get("vertices"), "object_faces": ob.get("faces"),
            "boundary_edges": e.get("boundary"), "non_manifold_edges": e.get("non_manifold"),
            "watertight": e.get("boundary") == 0 and e.get("non_manifold") == 0,
            "physically_plausible": af.get("vertices_inside_table") == 0 and (cl[0] or -1) >= 0}


# ==================================================================================================== report
def evaluate_run(dataset: str, run: str, args) -> dict:
    from common.datasets import resolve_run
    try:
        cfg = resolve_run(dataset, run)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] {dataset}/{run}: no config ({e}); target_id unknown")
        cfg = {"dataset": dataset, "run": run, "run_dir": str(get_run_dir(dataset, run))}
    if args.target_id is not None:
        cfg["target_id"] = args.target_id
    rd = Path(cfg["run_dir"])
    if not rd.exists():
        raise SystemExit(f"{rd} does not exist")
    t0 = time.time()
    res = {"dataset": dataset, "run": run, "scene": cfg.get("scene"), "object": cfg.get("object"),
           "target_id": cfg.get("target_id"), "has_object_free_gt": bool(cfg.get("has_object_free_gt")),
           "steps_present": sorted(p.name for p in rd.iterdir() if p.is_dir() and p.name.startswith("step")),
           "step4_renderer_check": eval_step4(rd),
           "step11_inpaint_3dgs": eval_step11(rd),
           "step12_inpainting_quality": None if args.skip_images else eval_step12(rd, cfg, args.dilate, args.max_views),
           "step13_table_mesh": eval_step13(rd, args.plane, args.skip_geometry),
           "step14_object_mesh": eval_step14(rd),
           "step15_merge_physics": eval_step15(rd)}
    res["seconds"] = round(time.time() - t0, 1)
    (rd / "evaluation_summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"[ok] {dataset}/{run} -> {rd / 'evaluation_summary.json'} ({res['seconds']} s)")
    return res


COLUMNS = [  # (header, path in evaluation_summary.json)
    ("Dataset", "dataset"), ("Run", "run"), ("Target ID", "target_id"),
    ("S4 renderer max diff", "step4_renderer_check.max_abs_diff_vs_come_render"),
    ("S4 test PSNR (CoMe)", "step4_renderer_check.psnr_vs_gt_test_mean"),
    ("S12 PSNR outside obj", "step12_inpainting_quality.psnr_inpainted_outside"),
    ("S12 SSIM outside obj", "step12_inpainting_quality.ssim_inpainted_outside"),
    ("S12 PSNR inside (GT)", "step12_inpainting_quality.psnr_inpainted_inside"),
    ("S12 SSIM inside (GT)", "step12_inpainting_quality.ssim_inpainted_inside"),
    ("S12 LPIPS (GT)", "step12_inpainting_quality.lpips_inpainted_full"),
    ("S12 FID (GT)", "step12_inpainting_quality.fid_inpainted"),
    ("S11 Gaussians", "step11_inpaint_3dgs.n_gaussians"),
    ("S13 table V", "step13_table_mesh.table_vertices"), ("S13 table F", "step13_table_mesh.table_faces"),
    ("S13 under-obj RMSE", "step13_table_mesh.under_object.height_rmse"),
    ("S13 under-obj coverage", "step13_table_mesh.under_object.coverage_within_tol"),
    ("S14 variant", "step14_object_mesh.chosen_variant"),
    ("S14 IoU clean", "step14_object_mesh.mean_iou_clean"), ("S14 IoU all", "step14_object_mesh.mean_iou_all"),
    ("S15 V inside table", "step15_merge_physics.vertices_inside_table_after"),
    ("S15 clearance min", "step15_merge_physics.clearance_min"),
    ("S15 V with table below", "step15_merge_physics.vertices_with_table_below"),
    ("S15 boundary edges", "step15_merge_physics.boundary_edges"),
    ("S15 non-manifold", "step15_merge_physics.non_manifold_edges"),
]


def _get(d, path):
    for k in path.split("."):
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def write_report(results: list[dict]):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [[_get(r, p) for _, p in COLUMNS] for r in results]
    with open(OUTPUTS_DIR / "council_report_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([h for h, _ in COLUMNS])
        w.writerows(rows)
    fmt = lambda v: "–" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))  # noqa: E731
    md = ["# Safe-GS – Council report summary", "",
          f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/evaluate_all_metrics.py` from `outputs/<Dataset>/<run>/`.",
          "", "| " + " | ".join(h for h, _ in COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    md += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]
    md += ["", "## Notes", "",
           "- **S4**: Inpaint360GS renderer vs CoMe renders of the same 3DGS (max abs diff 0 = identical); "
           "test PSNR = CoMe 3DGS vs photos (held-out 1/8 views).",
           "- **S12 outside obj**: inpainted render vs photo outside the (dilated) object mask on the test views - "
           "measures that the rest of the scene is preserved. **Inside (GT)**, LPIPS and FID only for datasets with an "
           "object-free ground truth (Inpaint360GS); LERF / Replica have none.",
           "- **S13 under-obj**: heights of the table mesh vertices inside the object footprint w.r.t. the table plane "
           "(RMSE) and share of footprint cells with a table surface within 2 % of the object size.",
           "- **S14 IoU**: silhouette IoU of the InstantMesh mesh vs the object masks of the real cameras.",
           "- **S15**: object placed on the table: vertices inside the table (0 = no penetration), minimum clearance "
           "(= gap), share of object vertices with table support below, boundary / non-manifold edges (0 = watertight)."]
    for r in results:
        s12 = r.get("step12_inpainting_quality") or {}
        md += ["", f"### {r['dataset']} / {r['run']}", "",
               f"- steps present: {', '.join(r['steps_present'])}",
               f"- step 12: {s12.get('n_test_views', '–')} test views, mask: {s12.get('mask', '–')}"
               + (f"; {s12['note']}" if s12.get("note") else ""),
               f"- step 14 variants: {json.dumps((r.get('step14_object_mesh') or {}).get('variants'))}"]
    (OUTPUTS_DIR / "council_report_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[ok] {OUTPUTS_DIR / 'council_report_summary.md'}\n[ok] {OUTPUTS_DIR / 'council_report_summary.csv'}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="LERF")
    ap.add_argument("--run", default=None, help="run name (default: every run of --dataset)")
    ap.add_argument("--all", action="store_true", help="every outputs/<Dataset>/<run>")
    ap.add_argument("--target-id", type=int, default=None, help="override target_id (object mask of step 12)")
    ap.add_argument("--plane", default=None, help="table_plane.json (default: run folder, then src/tools_mesh)")
    ap.add_argument("--dilate", type=int, default=10, help="object mask dilation (px) for the outside-mask metrics")
    ap.add_argument("--max-views", type=int, default=0, help="limit step-12 test views (0 = all)")
    ap.add_argument("--skip-images", action="store_true", help="skip step 12 image metrics")
    ap.add_argument("--skip-geometry", action="store_true", help="skip the under-object table analysis (big meshes)")
    args = ap.parse_args()
    targets = []
    if args.all or not args.run:
        for ds in (DATASETS if args.all else (args.dataset,)):
            for d in sorted(get_method_dir(ds).glob("*")) if get_method_dir(ds).exists() else []:
                if d.is_dir() and any(p.name.startswith("step") for p in d.iterdir()):
                    targets.append((ds, d.name))
    else:
        targets = [(args.dataset, args.run)]
    if not targets:
        raise SystemExit("no run with step folders found under outputs/")
    write_report([evaluate_run(ds, rn, args) for ds, rn in targets])


if __name__ == "__main__":
    main()
