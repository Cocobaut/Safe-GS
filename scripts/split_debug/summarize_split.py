"""Doc cac bang da xuat cua bo debug Split va ghi <out>/SUMMARY.txt (tong quan 1 trang)."""
import csv
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import scene_dirs


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else "office0"
    out, _ = scene_dirs(scene)
    L = []
    info = json.load(open(os.path.join(out, "run_info.json"))) if os.path.exists(os.path.join(out, "run_info.json")) else {}
    L.append(f"=== TOM TAT SPLIT - {scene} ===")
    s1 = info.get("step1", {})
    s23 = info.get("step2_3", {})
    if s1:
        L.append(f"Buoc 1: {s1.get('frames')} frame, {s1.get('seconds')} s, VRAM PyTorch dinh {s1.get('vram_pytorch_peak_gib')} GiB")
    if s23:
        L.append(f"Buoc 2+3: {s23.get('seconds_total')} s, VRAM PyTorch dinh {s23.get('vram_pytorch_peak_gib')} GiB")
        L.append(f"  ket qua: {s23.get('ket_qua')}")

    p = os.path.join(out, "01_mask_2d", "stats.csv")
    if os.path.exists(p):
        r = rows(p)
        after = [int(x["so_mask_sau_loc"]) for x in r]
        L.append(f"\n[Buoc 1] so mask sau loc / frame: min {min(after)}, tb {sum(after) / len(after):.1f}, max {max(after)}")
        for k in ["uber_huge", "very_coarse", "coarse", "fine"]:
            v = [int(x[f"so_mask_{k}"]) for x in r]
            L.append(f"   cau hinh {k:<12}: tb {sum(v) / len(v):.1f} mask/frame")

    p = os.path.join(out, "02_skeleton_3d", "depth_check.csv")
    if os.path.exists(p):
        r = rows(p)
        pct = [float(x["ti_le_qua_pct"]) for x in r]
        bad = [x["frame"] for x in r if float(x["ti_le_qua_pct"]) < 80]
        L.append(f"\n[Buoc 2] diem 3D qua kiem tra depth: min {min(pct):.1f}%, tb {sum(pct) / len(pct):.1f}%, max {max(pct):.1f}%"
                 f" | frame < 80%: {bad[:10] if bad else 'khong co'}")

    p = os.path.join(out, "03_propagation", "label_growth.csv")
    if os.path.exists(p):
        r = rows(p)
        tot = [int(x["tong_so_nhan"]) for x in r]
        L.append(f"\n[Buoc 3] so nhan: frame dau {tot[0]}, dinh {max(tot)}, cuoi {tot[-1]}; tong nhan moi {sum(int(x['nhan_moi']) for x in r)}, "
                 f"tong nhan bi xoa {sum(int(x['nhan_bi_xoa']) for x in r)}")
    p = os.path.join(out, "03_propagation", "decisions.csv")
    if os.path.exists(p):
        c = Counter((x["truong_hop"], x["ket_qua"].split(" ")[0]) for x in rows(p))
        L.append("   quyet dinh cho moi mask SAM (truong hop / ket qua):")
        for (a, b), n in sorted(c.items(), key=lambda t: -t[1]):
            L.append(f"      {n:>6}  {a} / {b}")
    p = os.path.join(out, "03_propagation", "mask_source.csv")
    if os.path.exists(p):
        c = Counter(x["nguon"] for x in rows(p))
        tot = sum(c.values())
        L.append("   nguon mask cuoi: " + ", ".join(f"{k} {v} ({100 * v / tot:.1f}%)" for k, v in c.items()))
    p = os.path.join(out, "03_propagation", "clusters", "clusters.csv")
    if os.path.exists(p):
        r = rows(p)
        L.append(f"   nhan sau DBSCAN: {len(r)} nhan, giu {sum(x['duoc_giu'] == 'co' for x in r)}, "
                 f"phan manh (>3 cum) {sum(x['phan_manh_gt3_cum'] == 'co' for x in r)}")

    p = os.path.join(out, "04_final", "instances.csv")
    if os.path.exists(p):
        r = rows(p)
        nf = [int(x["so_frame_xuat_hien"]) for x in r]
        L.append(f"\n[Ket qua] {len(r)} ID cuoi cung; so frame xuat hien: min {min(nf)}, tb {sum(nf) / len(nf):.1f}, max {max(nf)}")
        L.append("   top 10 ID theo so frame:")
        for x in sorted(r, key=lambda x: -int(x["so_frame_xuat_hien"]))[:10]:
            L.append(f"      ID {x['id']:>4}: {x['so_frame_xuat_hien']:>3} frame, {x['so_diem_3d']:>6} diem 3D, dien tich tb {x['dien_tich_tb_pct']}%")
        few = [x["id"] for x in r if int(x["so_frame_xuat_hien"]) <= 5]
        L.append(f"   ID xuat hien <= 5 frame (nghi nhieu/phan manh): {len(few)}")
    open(os.path.join(out, "SUMMARY.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
