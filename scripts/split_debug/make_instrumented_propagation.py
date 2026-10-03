"""
Tao BAN SAO cua sam2/mask_propagation_scanet.py (file goc giu nguyen) co chen cac dong goi _dbg de ghi ket qua
trung gian. Chi CHEN dong moi (doc bien), khong doi phep tinh nao. Thay doi duy nhat tren dong co san:
  BASE_DIR = Path(__file__)...  ->  tro ve thu muc sam2 goc (vi ban sao nam cho khac, de tim dung checkpoint).
Moi diem chen deu kiem tra la DUY NHAT trong file; file goc doi khac di thi dung han, khong chen bua.
Ghi ban sao + file diff vao _work/<scene>/.
"""
import argparse
import difflib
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REPO, scene_dirs

SRC = os.path.join(REPO, "sam2", "mask_propagation_scanet.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="office0")
    args = ap.parse_args()
    _, work = scene_dirs(args.scene)
    os.makedirs(work, exist_ok=True)

    lines = open(SRC).read().split("\n")
    main_start = next(i for i, l in enumerate(lines) if l.startswith('if __name__ == "__main__":'))
    out = list(lines)
    inserts = []  # (index dong neo, "truoc"/"sau", noi dung chen)

    def find_one(pattern, start=main_start, end=None):
        end = len(lines) if end is None else end
        hits = [i for i in range(start, end) if lines[i].rstrip() == pattern.rstrip()]
        assert len(hits) == 1, f"neo khong duy nhat/khong thay ({len(hits)}): {pattern!r}"
        return hits[0]

    def indent_of(i):
        return lines[i][: len(lines[i]) - len(lines[i].lstrip())]

    # BUOC 2: sau khi biet diem 3D nao nhin thay o frame nay
    i = find_one("        projected_view[image_id] = projected_point")
    inserts.append((i, "sau", '_dbg.at("frame_visible", locals())'))
    # BUOC 3 - frame dau: moi mask tao 1 nhan goc
    i = find_one("                gt_labels_idx +=1 #index update")
    inserts.append((i, "sau", '_dbg.at("first_frame_mask", locals())'))
    # BUOC 3 - frame sau: "mask ao" da dung xong, truoc khi xet tung mask SAM
    i = find_one("            # Step 3: iterate over the masks of the current view to update the instances")
    inserts.append((i, "truoc", '_dbg.at("virtual_mask", locals())'))
    # BUOC 3 - truong hop cua tung mask SAM
    i = find_one("                label_in_mask = label_in_mask[label_in_mask >= 0] # remove -1 labels(void)")
    inserts.append((i, "sau", '_dbg.at("mask_case", locals())'))
    # BUOC 3 - so diem 3D thuc su them vao (4 lan goi DBSCAN_points trong main, theo thu tu)
    db = [k for k in range(main_start, len(lines)) if lines[k].strip() == "_, p2D, ids = DBSCAN_points(t_pcd_points, t_points_2D)"]
    assert len(db) == 4, f"can 4 lan goi DBSCAN_points, thay {len(db)}"
    for k, tag in zip(db[1:], ["dbscan_new", "dbscan_multi", "dbscan_step4"]):  # lan 1 (frame dau) da co first_frame_mask
        inserts.append((k, "sau", f'_dbg.at("{tag}", locals())'))
    # BUOC 3 - het 1 frame
    i = find_one("        iter+=1")
    inserts.append((i, "truoc", '_dbg.at("frame_end", locals())'))
    # BUOC 3 - sau bau chon
    i = find_one("    pointsLabels_def = majority_per_key(pointsLabels)")
    inserts.append((i, "sau", '_dbg.at("votes_final", locals())'))
    # BUOC 3 - DBSCAN loc tung nhan
    i = find_one("        unique, counts = np.unique(valid_labels, return_counts=True)")
    inserts.append((i, "sau", '_dbg.at("cluster_label", locals())'))
    i = find_one("    if(VERBOSE): print(labeled_clusters.keys())")
    inserts.append((i, "sau", '_dbg.at("clusters", locals())'))
    # BUOC 3 - nguon goc tung mask cuoi (moi lan goi save_mask trong phan gan mask)
    reprompt = find_one("        for label in (visiblelabels - created_mask):")
    def_line = find_one("    def save_mask(label, image_name, mask):")
    calls = [k for k in range(def_line + 1, len(lines)) if re.match(r"\s*save_mask\(", lines[k])]
    assert len(calls) == 6, f"can 6 lan goi save_mask, thay {len(calls)}"
    for k in calls:
        arg = re.match(r"\s*save_mask\((\w+),", lines[k]).group(1)
        src = "sam_sinh_lai" if k > reprompt else ("sam_goc_gop" if "merged_mask" in lines[k] else "sam_goc")
        inserts.append((k, "sau", f'_dbg.mask_saved("{src}", {arg}, save_name)'))

    # thay duong dan checkpoint (ban sao nam o thu muc khac)
    bi = [k for k, l in enumerate(lines) if l.strip() == "BASE_DIR = Path(__file__).resolve().parent"]
    assert len(bi) == 1
    out[bi[0]] = f'BASE_DIR = Path({os.path.join(REPO, "sam2")!r})  # [debug] = thu muc sam2 goc'

    for idx, where, code in sorted(inserts, key=lambda t: (t[0], t[1] == "sau"), reverse=True):
        line = indent_of(idx) + code + "  # [debug]"
        out.insert(idx + 1 if where == "sau" else idx, line)

    dst = os.path.join(work, "mask_propagation_scanet_debug.py")
    open(dst, "w").write("\n".join(out))
    diff = list(difflib.unified_diff(lines, out, "goc/mask_propagation_scanet.py", "ban_sao_debug.py", lineterm="", n=1))
    open(os.path.join(work, "mask_propagation_scanet_debug.diff"), "w").write("\n".join(diff))
    added = sum(1 for d in diff if d.startswith("+") and not d.startswith("+++"))
    removed = sum(1 for d in diff if d.startswith("-") and not d.startswith("---"))
    print(f"ban sao: {dst} | them {added} dong, doi {removed} dong co san (BASE_DIR)")
    compile(open(dst).read(), dst, "exec")


if __name__ == "__main__":
    main()
