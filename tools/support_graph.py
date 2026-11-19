"""
Đồ thị chống đỡ giữa các vật từ visual hull (mask SAM 3 + camera), không dùng Gaussian.

Đầu vào: thư mục hull_<vật>.ply (tâm voxel, cùng hệ toạ độ COLMAP, tools/mask_hull.py), support.json (mặt bàn, mét / đơn vị).
Với mỗi vật: độ cao so với mặt bàn, vết chân 1 cm, thể tích. Với mỗi cặp vật: khoảng cách ngắn nhất giữa hai hull,
số voxel trùng nhau (hull chồng lấn), diện tích vết chân chồng nhau, và trong các ô chồng nhau: khe hở giữa đáy vật trên và
đỉnh vật dưới. Cạnh "A đỡ B" khi hai vết chân chồng nhau đủ lớn và |khe hở| <= tol (hull dư vài mm nên cho phép âm / dương).
Cạnh "chạm bên" khi hull gần nhau (<= touch_cm) nhưng không thoả điều kiện đỡ.
Vật nằm trên bàn khi đáy hull (phân vị 2 %) cách mặt bàn <= tol.
Giải thuật: union-find gom cụm dính nhau; sắp xếp topo (Kahn) cho đồ thị đỡ ra thứ tự gỡ từ trên xuống; hậu duệ trong
đồ thị đỡ = những gì bị mang theo khi nhấc vật.
Kiểm tra độ nhạy: lặp với nhiều tol, báo cạnh nào ổn định.

    python tools/support_graph.py --hulls Data/teatime/mask_hull_sam3 --support Data/teatime/pipeline/v1/S3_label/support.json \
        --out Data/teatime/support_graph
Xuất: graph.json, report.md, sensitivity.json, top_view.png, side_view.png, graph.png, matrices.png
"""
import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

NAMES = {}          # tên hiển thị; mặc định dùng chính tên vật (không gắn cứng theo cảnh)


def read_ply_gauss(path, min_opacity=0.3):
    """Tâm Gaussian (x, y, z) của file 3DGS PLY, lọc theo opacity (sigmoid) >= min_opacity."""
    tm = {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4", "uint": "<u4", "short": "<i2", "ushort": "<u2"}
    with open(path, "rb") as f:
        n, props = 0, []
        while True:
            line = f.readline()
            if line.startswith(b"element vertex"):
                n = int(line.split()[2])
            if line.startswith(b"property"):
                _, t, nm = line.decode().split()
                props.append((nm, tm[t]))
            if line.startswith(b"end_header"):
                break
        a = np.frombuffer(f.read(n * np.dtype(props).itemsize), np.dtype(props))
    keep = 1 / (1 + np.exp(-a["opacity"].astype(np.float64))) >= min_opacity
    read_ply_gauss.last_scale = np.exp(np.stack([a["scale_0"], a["scale_1"], a["scale_2"]], 1).astype(np.float64)).max(1)[keep]
    return np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)[keep]


def read_ply_xyz(path):
    with open(path, "rb") as f:
        n, off = 0, 0
        while True:
            line = f.readline()
            off += len(line)
            if line.startswith(b"element vertex"):
                n = int(line.split()[2])
            if line.startswith(b"end_header"):
                break
        dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
        a = np.frombuffer(f.read(n * dt.itemsize), dt)
    return np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)


class UF:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def analyse(objs, hts, ab, tol_cm, touch_cm, min_overlap_cm2, cell_cm, vox_cm, tree, rad_cm, sigma_rel=None):
    """sigma_rel != None: tol_cm là HỆ SỐ; ngưỡng cặp = hệ số x (sigma + (r_A + r_B) / 2), ngưỡng trên bàn = hệ số x (sigma + r)."""
    """Trả về (đối tượng edges, on_table) cho một giá trị tol."""
    names = list(objs)
    edges = {}
    for i, A in enumerate(names):
        for B in names[i + 1:]:
            ca, cb = ab[A][1], ab[B][1]               # cell index tuples
            ov = np.intersect1d(ca, cb)
            ov_cm2 = len(ov) * cell_cm ** 2
            info = {"a": A, "b": B, "min_dist_cm": None, "overlap_cm2": ov_cm2, "gap_cm": None, "type": "none"}
            dmin = pair_dist(objs, tree, A, B)
            info["min_dist_cm"] = dmin
            if ov_cm2 >= min_overlap_cm2:
                ia, ib = np.searchsorted(ab[A][1], ov), np.searchsorted(ab[B][1], ov)
                a_top, a_bot = ab[A][2][ia], ab[A][3][ia]
                b_top, b_bot = ab[B][2][ib], ab[B][3][ib]
                a_mid, b_mid = (a_top + a_bot) / 2, (b_top + b_bot) / 2
                lower_is_A = np.median(a_mid) <= np.median(b_mid)
                if lower_is_A:
                    gap = np.median(b_bot - a_top)
                    low, up = A, B
                else:
                    gap = np.median(a_bot - b_top)
                    low, up = B, A
                info.update({"lower": low, "upper": up, "gap_cm": float(gap)})
                t_pair = tol_cm if sigma_rel is None else tol_cm * (sigma_rel + (rad_cm[A] + rad_cm[B]) / 2)
                info["tol_used_cm"] = float(t_pair)
                if abs(gap) <= t_pair:
                    info["type"] = "support"
            if info["type"] == "none" and dmin is not None and dmin <= touch_cm:
                info["type"] = "touch"
            if info["type"] != "none":
                edges[(A, B)] = info
    # suy luận chống đỡ theo trọng lực: vật nào không có vật khác đỡ thì do bàn đỡ, nếu đáy đủ gần mặt bàn.
    # Đáy vật thường không quan sát được (bị che), nên "đủ gần" tính tương đối theo chiều cao vật: <= max(tol, 0.25 x cao).
    uppers = {e["upper"] for e in edges.values() if e["type"] == "support"}
    on_table = {}
    for n in names:
        h = hts[n]
        p2, p98 = np.percentile(h, 2), np.percentile(h, 98)
        t_n = tol_cm if sigma_rel is None else tol_cm * (sigma_rel + rad_cm[n])
        on_table[n] = bool(n not in uppers and p2 - rad_cm[n] <= max(t_n, 0.25 * (p98 - p2)))
    return edges, on_table


def pair_dist(objs, tree, A, B):
    pa = objs[A]
    if len(pa) > 150000:
        pa = pa[np.random.RandomState(0).choice(len(pa), 150000, replace=False)]
    d, _ = tree[B].query(pa, k=1)
    return float(d.min())


def toposort(nodes, children):
    indeg = {n: 0 for n in nodes}
    for p, cs in children.items():
        for c in cs:
            indeg[c] += 1
    q = deque(sorted(n for n in nodes if indeg[n] == 0))
    out = []
    while q:
        n = q.popleft()
        out.append(n)
        for c in sorted(children.get(n, [])):
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    return out, len(out) != len(nodes)


def descendants(n, children):
    seen, st = set(), [n]
    while st:
        x = st.pop()
        for c in children.get(x, []):
            if c not in seen:
                seen.add(c)
                st.append(c)
    return sorted(seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hulls", required=True)
    ap.add_argument("--support", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", choices=["hull", "gauss"], default="hull",
                    help="hull: hull_<vật>.ply; gauss: <vật>.ply (Gaussian S3, lọc opacity)")
    ap.add_argument("--clip_cm", type=float, default=-0.5, help="bỏ điểm thấp hơn mặt bàn quá mức này (vật không nằm dưới bàn)")
    ap.add_argument("--tol_cm", type=float, default=None,
                    help="ngưỡng chính; mặc định = 4 x độ nhiễu mặt bàn (--table_ply) hoặc 1.5 cm nếu không có")
    ap.add_argument("--table_ply", type=str, default="", help="Gaussian mặt bàn (S3 ban.ply) để ước lượng độ nhiễu sigma")
    ap.add_argument("--exclude", type=str, default="", help="tên file (không đuôi) bỏ qua, cách nhau dấu phẩy (vd mặt đỡ)")
    ap.add_argument("--scale_free", action="store_true",
                    help="mọi ngưỡng tính theo độ nhiễu mặt bàn sigma (ô 4s, chạm 2.5s, cắt -4s, chồng >= 9 ô), dùng khi không biết tỉ lệ mét")
    ap.add_argument("--consensus", type=float, default=0.6,
                    help="giữ cạnh đỡ / quan hệ trên bàn nếu xuất hiện ở >= tỉ lệ này trong các ngưỡng quét")
    ap.add_argument("--expect", type=str, default="", help="JSON kỳ vọng tuỳ chọn {on_table: [...], pairs: [[dưới, trên], ...]}")
    ap.add_argument("--touch_cm", type=float, default=0.3)
    ap.add_argument("--min_overlap_cm2", type=float, default=2.0)
    ap.add_argument("--cell_cm", type=float, default=1.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sup = json.loads(Path(a.support).read_text())
    n_t, d_t, mpu = np.array(sup["table"]["normal"], np.float64), float(sup["table"]["d"]), float(sup["meters_per_unit"])
    cm = 100.0 * mpu                                           # cm / đơn vị COLMAP

    objs = {}
    raw_n = {}
    if a.source == "hull":
        files = [(f.stem[5:], f) for f in sorted(Path(a.hulls).glob("hull_*.ply")) if f.stem[5:] not in ("objects", "consistency")]
    else:
        files = [(f.stem, f) for f in sorted(Path(a.hulls).glob("*.ply"))
                 if f.stem not in ("background", "background_with_support", "ban", "objects_all")]
    excl = {x for x in a.exclude.split(",") if x}
    files = [(k, f) for k, f in files if k not in excl]
    radius = {}                                  # bán kính đại diện của 1 phần tử (Gaussian / voxel), đơn vị COLMAP
    for k, f in files:
        p = read_ply_xyz(f) if a.source == "hull" else read_ply_gauss(f)
        if len(p):
            raw_n[k] = len(p)
            objs[k] = p
            if a.source == "gauss":
                radius[k] = float(np.median(read_ply_gauss.last_scale))
    s_w = 1.0 if np.median([np.median(p @ n_t + d_t) for p in objs.values()]) >= 0 else -1.0
    sigma = None
    if a.table_ply:
        tp = read_ply_gauss(a.table_ply)
        th = s_w * (tp @ n_t + d_t) * cm
        mad0 = np.median(np.abs(th - np.median(th)))
        th = th[np.abs(th - np.median(th)) < 5 * mad0]                # bỏ Gaussian lệch xa mặt bàn (cạnh, chân bàn)
        q25, q75 = np.percentile(th, [25, 75])
        # MAD có thể = 0 nếu nhiều Gaussian trùng đúng một mặt (điểm khởi tạo suy biến) -> lấy thêm IQR
        sigma = float(max(1.4826 * np.median(np.abs(th - np.median(th))), (q75 - q25) / 1.349))
        if sigma < 1e-6:                                           # vẫn suy biến: bỏ các giá trị trùng rồi đo lại
            tu = np.unique(np.round(th, 6))
            sigma = float(1.4826 * np.median(np.abs(tu - np.median(tu))))
        print(f"sigma: {len(th)} Gaussian mat ban, {len(np.unique(np.round(th, 6)))} gia tri khac nhau")
    if a.scale_free:
        assert sigma, "--scale_free cần --table_ply"
        a.cell_cm, a.touch_cm, a.clip_cm = 4 * sigma, 2.5 * sigma, -4 * sigma
        a.min_overlap_cm2 = 9 * a.cell_cm ** 2
        print(f"scale_free: cell {a.cell_cm:.3f}, touch {a.touch_cm:.3f}, clip {a.clip_cm:.3f}, min_overlap {a.min_overlap_cm2:.3f} (don vi cm theo mpu)")
    clipped = {}
    for k in list(objs):
        h0 = s_w * (objs[k] @ n_t + d_t) * cm
        keep = h0 >= a.clip_cm
        clipped[k] = int((~keep).sum())
        objs[k] = objs[k][keep]
        if len(objs[k]) == 0:
            del objs[k]
    allp = np.concatenate(list(objs.values()))
    if a.source == "hull":
        ux = np.unique(np.round(max(objs.values(), key=len)[:, 0], 5))
        vox_units = float(np.min(np.diff(ux)))
        vox_cm = vox_units * cm
    else:
        vox_units, vox_cm = 0.0, 0.0
    print(f"nguon {a.source}, vat: {len(objs)}, voxel {vox_cm:.3f} cm, dau mat ban s_w={s_w:+.0f}")

    hts = {k: s_w * (p @ n_t + d_t) * cm for k, p in objs.items()}            # cm so với mặt bàn
    rad_cm = {k: (radius[k] * cm if k in radius else vox_cm / 2) for k in objs}   # tâm phần tử nằm cách bề mặt ~ bán kính
    a0 = np.array([1.0, 0, 0]) if abs(n_t[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n_t, a0); u /= np.linalg.norm(u); v = np.cross(n_t, u)
    p0 = -d_t * n_t
    cell_u = a.cell_cm / cm
    uv = {k: np.stack([(p - p0) @ u, (p - p0) @ v], 1) / cell_u for k, p in objs.items()}
    amin = min(np.floor(x[:, 0]).min() for x in uv.values())
    bmin = min(np.floor(x[:, 1]).min() for x in uv.values())
    W = int(max(np.floor(x[:, 0]).max() for x in uv.values()) - amin + 1)
    ab = {}
    for k in objs:
        ia = (np.floor(uv[k][:, 0]) - amin).astype(np.int64)
        ib = (np.floor(uv[k][:, 1]) - bmin).astype(np.int64)
        key = ib * W + ia
        uk, inv = np.unique(key, return_inverse=True)
        top = np.full(len(uk), -1e9); bot = np.full(len(uk), 1e9)
        np.maximum.at(top, inv, hts[k]); np.minimum.at(bot, inv, hts[k])
        ab[k] = (None, uk, top, bot, (ia, ib))
    tree = {k: cKDTree(p) for k, p in objs.items()}

    # --- bảng vật
    rows = {}
    for k, p in objs.items():
        h = hts[k]
        rows[k] = {"voxels": int(len(p)), "clipped_below_plane": clipped[k], "volume_cm3": float(len(p) * vox_cm ** 3),
                   "footprint_cm2": float(len(ab[k][1]) * a.cell_cm ** 2),
                   "h_min_cm": float(h.min()), "h_p2_cm": float(np.percentile(h, 2)), "h_median_cm": float(np.median(h)),
                   "h_p98_cm": float(np.percentile(h, 98)), "h_max_cm": float(h.max()),
                   "element_radius_cm": rad_cm[k], "centroid_cm_uv": [float(uv[k][:, 0].mean() * a.cell_cm), float(uv[k][:, 1].mean() * a.cell_cm)]}

    # --- phân tích chính + độ nhạy
    if a.scale_free:
        tols = [1.0, 1.5, 2.0, 3.0, 4.0]                       # hệ số x (sigma + bán kính phần tử)
        main_tol = a.tol_cm if a.tol_cm is not None else 2.0
    elif sigma:
        tols = sorted({round(max(0.3, f * sigma), 2) for f in (2, 3, 4, 6, 8)})
        main_tol = a.tol_cm if a.tol_cm is not None else round(max(0.3, 4 * sigma), 2)
    else:
        tols = [0.5, 1.0, 1.5, 2.5, 4.0]
        main_tol = a.tol_cm if a.tol_cm is not None else 1.5
    a.tol_cm = main_tol
    print(f"do nhieu mat ban sigma = {sigma} cm; quet tol = {tols}; tol chinh = {main_tol}")
    results = {}
    for t in sorted(set(tols + [main_tol])):
        edges_t, on_t = analyse(objs, hts, ab, t, a.touch_cm, a.min_overlap_cm2, a.cell_cm, vox_cm, tree, rad_cm,
                                sigma if a.scale_free else None)
        results[t] = (edges_t, on_t)
    # đồng thuận qua các ngưỡng: cạnh đỡ / trên bàn giữ nếu xuất hiện ở >= consensus số ngưỡng quét
    edges, on_table_main = results[main_tol]
    edges = dict(edges)
    n_t_ = len(tols)
    sup_votes = defaultdict(int)
    for t in tols:
        for key, e in results[t][0].items():
            if e["type"] == "support":
                sup_votes[key] += 1
    for t in tols:                       # cạnh đỡ có ở ngưỡng khác nhưng thiếu ở ngưỡng chính
        for key, e in results[t][0].items():
            if key not in edges:
                edges[key] = dict(e)
    for key, e in list(edges.items()):
        frac = sup_votes[key] / n_t_
        e["support_vote"] = round(frac, 2)
        if frac >= a.consensus:
            e["type"] = "support"
            if "lower" not in e:
                src = next(results[t][0][key] for t in tols if results[t][0].get(key, {}).get("type") == "support")
                e.update({k: src[k] for k in ("lower", "upper", "gap_cm")})
        elif e["type"] == "support":
            e["type"] = "touch" if (e["min_dist_cm"] is not None and e["min_dist_cm"] <= a.touch_cm) else "weak_support"
        if e["type"] == "none":
            del edges[key]
    on_table = {n: (sum(results[t][1][n] for t in tols) / n_t_) >= a.consensus for n in objs}
    uppers_c = {e["upper"] for e in edges.values() if e["type"] == "support"}
    on_table = {n: (x and n not in uppers_c) for n, x in on_table.items()}
    floating = sorted(n for n in objs if not on_table[n] and n not in uppers_c)      # không tìm được vật đỡ

    # trùng voxel 3D (hull chồng lấn) trên cùng lưới
    lat = {k: set(map(tuple, np.round((p - allp.min(0)) / vox_units).astype(np.int64))) for k, p in objs.items() if len(p) < 400000}         if a.source == "hull" else {}
    overlap_vox = {}
    for (A, B), e in edges.items():
        if A in lat and B in lat:
            overlap_vox[(A, B)] = len(lat[A] & lat[B])

    # --- đồ thị đỡ
    children = defaultdict(list)                  # dưới -> trên
    for (A, B), e in edges.items():
        if e["type"] == "support":
            children[e["lower"]].append(e["upper"])
    for n in objs:
        if on_table[n]:
            children["TABLE"].append(n)
    nodes = ["TABLE"] + list(objs)
    order, cyc = toposort(nodes, children)
    remove_order = [n for n in reversed(order) if n != "TABLE"]
    uf = UF(list(objs))                          # cụm phụ thuộc: chỉ nối theo cạnh đỡ (không qua bàn); cạnh chạm bên chỉ liệt kê
    for (A, B), e in edges.items():
        if e["type"] == "support":
            uf.union(A, B)
    clusters = defaultdict(list)
    for n in objs:
        clusters[uf.find(n)].append(n)
    clusters = sorted(clusters.values(), key=lambda c: (-len(c), c))
    support_parent = defaultdict(list)
    for p, cs in children.items():
        for c in cs:
            support_parent[c].append(p)

    # --- JSON
    graph = {"params": {"tol_cm": a.tol_cm, "tol_sweep_cm": tols, "table_noise_sigma_cm": sigma, "consensus": a.consensus, "touch_cm": a.touch_cm, "min_overlap_cm2": a.min_overlap_cm2, "cell_cm": a.cell_cm,
                        "voxel_cm": vox_cm}, "objects": rows, "on_table": on_table,
             "edges": [{**e, "overlap_voxels_3d": overlap_vox.get((A, B))} for (A, B), e in edges.items()],
             "support_children": {k: sorted(v) for k, v in children.items()}, "topo_order_bottom_up": order,
             "removal_order_top_down": remove_order, "cycle": cyc, "no_support_found": floating, "contact_clusters": clusters,
             "descendants": {n: descendants(n, children) for n in objs}}
    (out / "graph.json").write_text(json.dumps(graph, indent=2, ensure_ascii=False))
    sens = {}
    for t, (ed, ot) in results.items():
        sens[str(t)] = {"support": sorted(f"{e['lower']}>{e['upper']}" for e in ed.values() if e["type"] == "support"),
                        "touch": sorted(f"{A}~{B}" for (A, B), e in ed.items() if e["type"] == "touch"),
                        "on_table": sorted(k for k, x in ot.items() if x)}
    (out / "sensitivity.json").write_text(json.dumps(sens, indent=2, ensure_ascii=False))

    # --- hình
    cols = {k: plt.cm.tab20(i % 20) for i, k in enumerate(objs)}
    fp = {k: len(ab[k][1]) for k in objs}
    big = {k for k in objs if fp[k] > 4 * np.median(list(fp.values()))}       # vật rất lớn: chỉ làm nhạt trên hình
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))
    for k in sorted(objs, key=lambda k: -len(ab[k][1])):
        ia, ib = ab[k][4]
        m = hts[k] < 1000
        ax.scatter(np.unique(ia) * a.cell_cm, np.unique(ib) * a.cell_cm, s=0) if False else None
        cx = (ab[k][1] % W) * a.cell_cm; cy = (ab[k][1] // W) * a.cell_cm
        ax.scatter(cx, cy, s=14, color=cols[k], alpha=0.35 if k in big else 0.8, marker="s", label=NAMES.get(k, k))
        ax.annotate(NAMES.get(k, k).split(" (")[0], (cx.mean(), cy.mean()), fontsize=9, weight="bold")
    ax.set_aspect("equal"); ax.set_title("Nhin tu tren xuong (vet chan hull, o 1 cm)"); ax.legend(fontsize=7, ncol=2)
    ax.set_xlabel("cm (truc u mat ban)"); ax.set_ylabel("cm (truc v mat ban)")
    fig.tight_layout(); fig.savefig(out / "top_view.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for k in sorted(objs, key=lambda k: -len(ab[k][1])):
        if k in big:
            continue
        x = uv[k][:, 0] * a.cell_cm
        step = max(1, len(x) // 6000)
        ax.scatter(x[::step], hts[k][::step], s=3, color=cols[k], label=NAMES.get(k, k))
        ax.annotate(NAMES.get(k, k).split(" (")[0], (x.mean(), np.percentile(hts[k], 98) + 0.3), fontsize=8, ha="center")
    ax.axhline(0, color="k", lw=1); ax.axhline(a.tol_cm, color="gray", lw=0.6, ls="--")
    ax.set_xlabel("cm doc truc u"); ax.set_ylabel("do cao tren mat ban (cm)"); ax.set_title(f"Nhin ngang (bo vat rat lon: {sorted(big)})")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout(); fig.savefig(out / "side_view.png", dpi=130); plt.close(fig)

    # đồ thị theo tầng
    level = {"TABLE": 0}
    for n in order:
        if n == "TABLE":
            continue
        level[n] = 1 + max([level.get(p, 0) for p in support_parent.get(n, [])] or [0])
    byl = defaultdict(list)
    for n, l in level.items():
        byl[l].append(n)
    pos = {}
    for l, ns in byl.items():
        for i, n in enumerate(sorted(ns)):
            pos[n] = ((i - (len(ns) - 1) / 2) * 2.0, -l)
    fig, ax = plt.subplots(1, 1, figsize=(13, 6))
    for (A, B), e in edges.items():
        if e["type"] == "touch":
            ax.plot([pos[A][0], pos[B][0]], [pos[A][1], pos[B][1]], color="orange", ls=":", lw=1.5)
    for p, cs in children.items():
        for c in cs:
            ax.annotate("", xy=pos[c], xytext=pos[p], arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.5, shrinkA=14, shrinkB=14))
    for n, (x, y) in pos.items():
        ax.text(x, y, NAMES.get(n, n).replace(" (", "\n("), ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round", fc="white" if n != "TABLE" else "#ddd", ec="k"))
    ax.set_xlim(min(p[0] for p in pos.values()) - 2, max(p[0] for p in pos.values()) + 2)
    ax.set_ylim(min(p[1] for p in pos.values()) - 1, 1); ax.axis("off")
    ax.set_title(f"Do thi chong do (mui ten xanh: dua len, cham cam: cham ben), tol {a.tol_cm} cm")
    fig.tight_layout(); fig.savefig(out / "graph.png", dpi=130); plt.close(fig)

    ks = list(objs)
    M_d = np.full((len(ks), len(ks)), np.nan); M_o = np.zeros((len(ks), len(ks))); M_g = np.full((len(ks), len(ks)), np.nan)
    for i, A in enumerate(ks):
        for j, B in enumerate(ks):
            if i == j:
                continue
            e = edges.get((A, B)) or edges.get((B, A))
            ov = len(np.intersect1d(ab[A][1], ab[B][1])) * a.cell_cm ** 2
            M_o[i, j] = ov
            if e is not None:
                M_d[i, j] = e["min_dist_cm"]
                M_g[i, j] = e["gap_cm"] if e["gap_cm"] is not None else np.nan
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, M, ttl in ((axs[0], M_o, "Dien tich vet chan chong nhau (cm2)"), (axs[1], M_g, "Khe ho dai/day (cm), cap co canh")):
        im = ax.imshow(M, cmap="viridis"); ax.set_xticks(range(len(ks))); ax.set_yticks(range(len(ks)))
        ax.set_xticklabels([k for k in ks], rotation=60, ha="right", fontsize=8); ax.set_yticklabels(ks, fontsize=8)
        ax.set_title(ttl); fig.colorbar(im, ax=ax, shrink=0.8)
        for i in range(len(ks)):
            for j in range(len(ks)):
                if not np.isnan(M[i, j]) and M[i, j] != 0:
                    ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=6, color="w")
    fig.tight_layout(); fig.savefig(out / "matrices.png", dpi=130); plt.close(fig)

    # --- báo cáo
    L = []
    L += ["# Do thi chong do tu visual hull (teatime, mask SAM 3)", "",
          f"Tham so: tol {a.tol_cm} cm, touch {a.touch_cm} cm, chong vet chan toi thieu {a.min_overlap_cm2} cm2, o {a.cell_cm} cm, "
          f"nguon {a.source}, cat duoi ban < {a.clip_cm} cm, voxel {vox_cm:.3f} cm, {len(objs)} vat.", "",
          "## 1. Tung vat", "",
          "| vat | diem | bi cat duoi ban | the tich cm3 | vet chan cm2 | cao min | cao p2 | cao median | cao p98 | cao max | tren ban (<= tol) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in rows.items():
        L.append(f"| {k} | {r['voxels']} | {r['clipped_below_plane']} | {r['volume_cm3']:.0f} | {r['footprint_cm2']:.0f} | {r['h_min_cm']:.1f} | {r['h_p2_cm']:.1f} | "
                 f"{r['h_median_cm']:.1f} | {r['h_p98_cm']:.1f} | {r['h_max_cm']:.1f} | {'co' if on_table[k] else 'khong'} |")
    L += ["", "Luu y: voi nguon hull, the tich lon hon vat that (hull la can ngoai). Cao so voi mat ban RANSAC cua S3.", "",
          "## 2. Cac canh (cap co chong vet chan hoac hull gan nhau)", "",
          "| cap | loai | duoi -> tren | khe cm | chong vet chan cm2 | khoang cach hull cm | voxel hull trung nhau |",
          "|---|---|---|---|---|---|---|"]
    for (A, B), e in sorted(edges.items(), key=lambda x: (x[1]["type"], x[0])):
        lo = f"{e.get('lower','-')} -> {e.get('upper','-')}" if e.get("lower") else "-"
        g = f"{e['gap_cm']:+.2f}" if e["gap_cm"] is not None else "-"
        L.append(f"| {A} / {B} | {e['type']} | {lo} | {g} | {e['overlap_cm2']:.0f} | {e['min_dist_cm']:.2f} | {overlap_vox.get((A, B), '-')} |")
    L += ["", "## 3. Do thi chong do", "", "Cha -> con (con nam tren cha):", ""]
    for p in ["TABLE"] + list(objs):
        if p in children:
            L.append(f"- {p}: {', '.join(sorted(children[p]))}")
    L += ["", f"Vat KHONG tim duoc vat do (lo lung / do do vat chua co nhan): {', '.join(floating) or '-'}", "",
          f"Co chu trinh: {'CO' if cyc else 'khong'}", "",
          f"Thu tu topo (duoi len): {' > '.join(order)}", "",
          f"Thu tu go bo an toan (tren xuong): {' > '.join(remove_order)}", "",
          "## 4. Cum phu thuoc (union-find tren canh do, khong tinh ban; canh cham ben xem muc 2)", ""]
    for i, c in enumerate(clusters):
        L.append(f"- cum {i + 1} ({len(c)} vat): {', '.join(sorted(c))}")
    L += ["", "## 5. Nhac mot vat thi mang theo gi (hau due trong do thi do)", "", "| nhac | mang theo | dang cham ben |", "|---|---|---|"]
    for n in objs:
        touch = sorted([B if A == n else A for (A, B), e in edges.items() if e["type"] == "touch" and n in (A, B)])
        L.append(f"| {n} | {', '.join(descendants(n, children)) or '-'} | {', '.join(touch) or '-'} |")
    L += ["", "## 6. Do nhay theo tol", "", "| tol cm | canh doc | canh cham ben | tren ban |", "|---|---|---|---|"]
    for t, s in sens.items():
        L.append(f"| {t} | {len(s['support'])}: {'; '.join(s['support']) or '-'} | {len(s['touch'])} | {len(s['on_table'])}: {', '.join(s['on_table'])} |")
    stable = set.intersection(*[set(s["support"]) for s in sens.values()])
    L += ["", f"Canh doc on dinh o moi tol: {', '.join(sorted(stable)) or 'khong co'}",
          f"Canh doc theo dong thuan >= {a.consensus}: " + ", ".join(sorted(f"{e['lower']}>{e['upper']} ({e['support_vote']})"
                                                                     for e in edges.values() if e["type"] == "support")), ""]
    if a.expect:
        ex = json.loads(Path(a.expect).read_text(encoding="utf-8"))
        eot = set(ex.get("on_table", [])) & set(objs)
        got = {k for k, x in on_table.items() if x}
        L += ["## 7. So voi ky vong (file --expect, do nguoi ghi, khong phai do dac)", "",
              f"- tren ban: dung {len(eot & got)}/{len(eot)}; thieu: {', '.join(sorted(eot - got)) or '-'}; "
              f"thua (do la tren ban nhung khong co trong ky vong): {', '.join(sorted(got - set(ex.get('on_table', [])))) or '-'}"]
        hit = 0
        pairs = [(lo, up) for lo, up in ex.get("pairs", []) if lo in objs and up in objs]
        for lo, up in pairs:
            f_ = [e for e in edges.values() if {e["a"], e["b"]} == {lo, up}]
            ok = bool(f_) and f_[0]["type"] == "support" and f_[0].get("lower") == lo
            hit += ok
            L.append(f"- cap {lo} do {up}: " + (f"{f_[0]['type']}, khe {f_[0]['gap_cm']}, phieu {f_[0].get('support_vote')}" if f_ else "khong co canh")
                     + (" -> DUNG" if ok else " -> SAI"))
        L.append(f"- cap do dung: {hit}/{len(pairs)}")
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
