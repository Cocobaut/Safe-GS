"""
Tạo Data/teatime/mold_inspector_360.html: trình xem WebGL độc lập (không cần mạng, không thư viện ngoài).

Chọn một trong các vật (13 vật), bật/tắt:
  - Gaussian của vật (S3_label/gaussians/<vật>.ply, màu SH bậc 0)
  - bề mặt thấy được của khung (frame cls 1)
  - khuôn dự đoán dạng khung dây (voxel cls 3/4), trước và sau (nếu có)
  - mặt phẳng tiếp xúc bàn (z = 0, mặt phẳng RANSAC trong support.json)
Hiển thị: dz (cao p98 của khung trên mặt bàn, cm), dz bề mặt thấy được, thể tích khung (cm3), thể tích khuôn (cm3).

Hệ tọa độ: hệ bàn, z = chiều cao trên bàn (cm), x/y nằm trên bàn. Chuyển đổi giống tools/frame_mesh.py.

    python tools/build_mold_inspector.py \
        --s3 Data/teatime/pipeline/v3_sam3/S3_label \
        --before Data/teatime/pipeline/v3_sam3/S3m \
        --after  Data/teatime/pipeline/v3_sam3/S3m/v4 \
        --out    Data/teatime/mold_inspector_360.html

--after bỏ trống hoặc thư mục không có frame/ thì trình xem chỉ hiện bản trước.
--before_label / --after_label: tên hiển thị (vd v3_sam3, v5_da3). Voxel có ở bản trước mà mất ở bản sau = "bị khoét"
(lớp riêng, đỏ), vd khoét theo độ sâu Depth Anything 3.
--asset NHAN=THU_MUC (lặp được): tài sản 3D đã đặt vào cảnh (tools/align_generated.py): THU_MUC/<vật>.npz (xyz hệ thế giới)
+ THU_MUC/metrics.json {vật: {...}}; hiện dạng điểm + bảng chỉ số (vd Amodal3R, TRELLIS).
"""
import argparse
import base64
import json
from pathlib import Path

import numpy as np

MAX_GAUSS = 12000       # Gaussian hiển thị mỗi vật (lấy mẫu đều, cố định seed)
MAX_CUBES = 4000        # voxel khuôn vẽ mỗi vật mỗi bản


def read_ply(path):
    """PLY nhị phân (float/uchar) -> structured array. Đủ cho file 3DGS và file ghi bởi pipeline/common.py."""
    with open(path, "rb") as f:
        props, n = [], 0
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            t = line.split()
            if t[:2] == ["element", "vertex"]:
                n = int(t[2])
            elif t[:1] == ["property"]:
                props.append((t[2], t[1]))
            if line == "end_header":
                break
        tmap = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1", "int": "<i4"}
        dt = np.dtype([(name, tmap[ty]) for name, ty in props])
        return np.frombuffer(f.read(n * dt.itemsize), dt, count=n)


def table_frame(s3):
    sup = json.loads((s3 / "support.json").read_text(encoding="utf-8"))
    n = np.array(sup["table"]["normal"], np.float64)
    n /= np.linalg.norm(n)
    d = float(sup["table"]["d"])
    cm = 100.0 * float(sup["meters_per_unit"])
    b0 = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(n, b0); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    p0 = -d * n
    R = np.stack([e1, e2, n], 0)
    return n, d, cm, p0, R


def to_table(X, p0, R, cm):
    return ((X - p0) @ R.T) * cm            # columns: x, y on the table, z = height above table (cm)


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else 0.0


def frame_stats(P, cls, cell):
    h = P[:, 2]
    seen = cls == 1
    mold = (cls == 3) | (cls == 4)
    bottom = cls == 3
    return {
        "voxels": int(len(P)),
        "dz_p98_cm": pct(h, 98), "dz_max_cm": float(h.max()) if len(h) else 0.0,
        "dz_seen_p98_cm": pct(h[seen], 98), "dz_seen_p50_cm": pct(h[seen], 50),
        "volume_cm3": float(len(P) * cell ** 3), "mold_cm3": float(mold.sum() * cell ** 3),
        "mold_bottom_cm3": float(bottom.sum() * cell ** 3), "mold_pct": 100.0 * float(mold.sum()) / max(len(P), 1),
    }


def load_variant(d, name, p0, R, cm):
    z = np.load(d / "frame" / f"{name}.npz")
    X = z["xyz"].astype(np.float64)
    cls = z["cls"]
    cell = float(z["voxel"]) * cm
    P = to_table(X, p0, R, cm)
    return P, cls, cell


def sample(idx_n, k, seed=0):
    if idx_n <= k:
        return np.arange(idx_n)
    return np.sort(np.random.default_rng(seed).choice(idx_n, k, replace=False))


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


ASSET_COLORS = [[0.95, 0.35, 0.85], [1.0, 0.9, 0.3], [0.6, 0.5, 1.0]]


def build(args):
    s3, before, after = Path(args.s3), Path(args.before), (Path(args.after) if args.after else None)
    assets = []
    for i, a in enumerate(args.asset):
        lab, _, d = a.partition("=")
        d = Path(d)
        m = json.loads((d / "metrics.json").read_text(encoding="utf-8")) if (d / "metrics.json").exists() else {}
        assets.append((lab, d, m, ASSET_COLORS[i % len(ASSET_COLORS)]))
    if after is not None and not (after / "frame").is_dir():
        print(f"khong co {after}/frame: bo qua ban sau")
        after = None
    n_t, d_t, cm, p0, R = table_frame(s3)
    groups = json.loads((s3 / "groups.json").read_text(encoding="utf-8"))["objects"]
    objs, summary = {}, {}
    for name in groups:
        if not (before / "frame" / f"{name}.npz").exists():
            continue
        entry = {"variants": {}}
        # Gaussian của vật (trước tinh chỉnh S3c): vị trí + màu SH bậc 0
        gp = s3 / "gaussians" / f"{name}.ply"
        gstats = None
        if gp.exists():
            g = read_ply(gp)
            X = np.stack([g["x"], g["y"], g["z"]], 1).astype(np.float64)
            P = to_table(X, p0, R, cm)
            sel = sample(len(P), MAX_GAUSS)
            col = np.clip(0.5 + 0.28209479 * np.stack([g["f_dc_0"], g["f_dc_1"], g["f_dc_2"]], 1), 0, 1)
            entry["gauss"] = {"p": b64(P[sel].astype(np.float32)), "c": b64(col[sel].astype(np.float32)), "n": int(len(sel))}
            gstats = {"gaussians": int(len(P)), "dz_p98_cm": pct(P[:, 2], 98)}
        summary[name] = {"gauss": gstats}
        ext = 0.0
        for tag, d in (("before", before), ("after", after)):
            if d is None or not (d / "frame" / f"{name}.npz").exists():
                entry["variants"][tag] = None
                continue
            P, cls, cell = load_variant(d, name, p0, R, cm)
            st = frame_stats(P, cls, cell)
            seen = P[cls == 1]
            mold_idx = np.where((cls == 3) | (cls == 4))[0]
            mold_idx = mold_idx[sample(len(mold_idx), MAX_CUBES, seed=1)]
            entry["variants"][tag] = {
                "stats": st,
                "seen": {"p": b64(seen.astype(np.float32)), "n": int(len(seen))},
                "mold": {"p": b64(P[mold_idx].astype(np.float32)), "n": int(len(mold_idx)), "cell": cell,
                         "bottom": b64(((cls[mold_idx]) == 3).astype(np.uint8)), "cls": "u8"},
            }
            ext = max(ext, float(np.abs(P[:, :2]).max()) if len(P) else 0.0)
            summary[name][tag] = st
            if tag == "after" and entry["variants"].get("before"):
                Pb, _, _ = load_variant(before, name, p0, R, cm)
                zb = np.load(before / "frame" / f"{name}.npz")["xyz"]; za = np.load(d / "frame" / f"{name}.npz")["xyz"]
                vb = np.ascontiguousarray(zb.astype(np.float32)).view([("", np.float32)] * 3).ravel()
                va = np.ascontiguousarray(za.astype(np.float32)).view([("", np.float32)] * 3).ravel()
                gone = ~np.isin(vb, va)
                cidx = np.where(gone)[0]
                cidx = cidx[sample(len(cidx), 3 * MAX_CUBES, seed=2)]
                entry["variants"]["after"]["carved"] = {"p": b64(Pb[cidx].astype(np.float32)), "n": int(len(cidx)),
                                                        "cm3": float(gone.sum() * cell ** 3)}
                summary[name]["carved_cm3"] = float(gone.sum() * cell ** 3)
        entry["assets"] = []
        for lab, d, m, col in assets:
            f = d / f"{name}.npz"
            if not f.exists():
                continue
            Xa = np.load(f)["xyz"].astype(np.float64)
            Pa = to_table(Xa, p0, R, cm)
            sel = sample(len(Pa), MAX_GAUSS, seed=3)
            entry["assets"].append({"label": lab, "p": b64(Pa[sel].astype(np.float32)), "n": int(len(sel)), "col": col,
                                    "metrics": m.get(name, {})})
            summary[name][f"asset_{lab}"] = m.get(name, {})
        entry["plane_half_cm"] = max(ext, 1.0) + 15.0
        entry["gauss_n_total"] = gstats["gaussians"] if gstats else 0
        objs[name] = entry
    return objs, summary, cm


HTML = r"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8">
<title>Mold inspector 360 - teatime</title>
<style>
 html,body{margin:0;height:100%;background:#15171c;color:#ddd;font:13px/1.4 system-ui,Segoe UI,sans-serif}
 #wrap{display:flex;height:100%}
 #side{width:330px;padding:12px;box-sizing:border-box;overflow:auto;border-right:1px solid #2a2d35;background:#1b1e25}
 #view{flex:1;position:relative}
 canvas{display:block;width:100%;height:100%;cursor:grab}
 h1{font-size:15px;margin:0 0 8px} h2{font-size:13px;margin:14px 0 6px;color:#9ab}
 select{width:100%;padding:4px;background:#101216;color:#ddd;border:1px solid #333}
 label{display:block;margin:3px 0;cursor:pointer} input[type=checkbox]{margin-right:6px}
 table{border-collapse:collapse;width:100%} td,th{padding:3px 4px;border-bottom:1px solid #2a2d35;text-align:right}
 th:first-child,td:first-child{text-align:left} th{color:#9ab;font-weight:600}
 .note{color:#888;font-size:12px;margin-top:8px}
 #legend span{display:inline-block;width:12px;height:12px;margin-right:4px;vertical-align:-2px}
 #hint{position:absolute;left:10px;bottom:8px;color:#777;font-size:12px}
</style></head>
<body><div id="wrap">
<div id="side">
 <h1>Mold inspector 360 - teatime</h1>
 <label>Vat <select id="obj"></select></label>
 <h2>Hien thi</h2>
 <label><input type="checkbox" id="t_gauss" checked>Gaussian cua vat (truoc S3c)</label>
 <label><input type="checkbox" id="t_seen" checked>Be mat thay duoc (khung)</label>
 <label><input type="checkbox" id="t_mold_b">Khuon du doan - TRUOC = __LB0__ (khung day)</label>
 <label><input type="checkbox" id="t_mold_a" checked>Khuon du doan - SAU = __LB1__ (khung day)</label>
 <label><input type="checkbox" id="t_carved" checked>Voxel bi khoet (TRUOC -> SAU, do)</label>
 <label><input type="checkbox" id="t_plane" checked>Mat tiep xuc ban (z = 0)</label>
 <div id="assetToggles"></div>
 <h2>Chi so (cm, cm3)</h2>
 <table id="stats"></table>
 <h2>Tai san 3D sinh (dat vao canh)</h2>
 <table id="astats"></table>
 <div class="note" id="legend"><span style="background:#ff8a2b"></span>khuon TRUOC &nbsp;<span style="background:#4fd1ff"></span>khuon SAU &nbsp;<span style="background:#6c6"></span>be mat thay &nbsp;<span style="background:#e33"></span>bi khoet &nbsp;<span style="background:#ccc"></span>Gaussian</div>
 <div class="note">dz = cao p98 cua khung tren mat ban (cm). Khoi tich = so voxel khung x (cell)^3. Khuon = voxel khong camera nao thay (cls 3 day + 4 ben/sau). Keo chuot: xoay; cuon: zoom; vuot/Shift: dich.</div>
</div>
<div id="view"><canvas id="c"></canvas><div id="hint">z = chieu cao tren ban (cm)</div></div>
</div>
<script>
const OBJS = __DATA__;
const LAB = __LAB__;
const CM = __CM__;
const names = Object.keys(OBJS);

function b64f32(s){const bin=atob(s);const u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return new Float32Array(u.buffer);}
function b64u8(s){const bin=atob(s);const u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u;}

const cv=document.getElementById('c');
const gl=cv.getContext('webgl',{antialias:true});
const VS=`attribute vec3 aP; attribute vec3 aC; uniform mat4 uM; uniform float uPt; varying vec3 vC;
void main(){gl_Position=uM*vec4(aP,1.0); gl_PointSize=uPt; vC=aC;}`;
const FS=`precision mediump float; varying vec3 vC; uniform float uA; uniform float uRound;
void main(){ if(uRound>0.5){vec2 d=gl_PointCoord-vec2(0.5); if(dot(d,d)>0.25) discard;}
 gl_FragColor=vec4(vC,uA);}`;
function sh(t,s){const o=gl.createShader(t);gl.shaderSource(o,s);gl.compileShader(o);if(!gl.getShaderParameter(o,gl.COMPILE_STATUS))throw gl.getShaderInfoLog(o);return o;}
const pr=gl.createProgram();gl.attachShader(pr,sh(gl.VERTEX_SHADER,VS));gl.attachShader(pr,sh(gl.FRAGMENT_SHADER,FS));gl.linkProgram(pr);gl.useProgram(pr);
const L={aP:gl.getAttribLocation(pr,'aP'),aC:gl.getAttribLocation(pr,'aC'),uM:gl.getUniformLocation(pr,'uM'),uPt:gl.getUniformLocation(pr,'uPt'),uA:gl.getUniformLocation(pr,'uA'),uRound:gl.getUniformLocation(pr,'uRound')};

// ---- mat4 (column-major)
function mul(a,b){const o=new Float32Array(16);for(let c=0;c<4;c++)for(let r=0;r<4;r++){let s=0;for(let k=0;k<4;k++)s+=a[k*4+r]*b[c*4+k];o[c*4+r]=s;}return o;}
function persp(f,asp,n,fa){const o=new Float32Array(16);const t=1/Math.tan(f/2);o[0]=t/asp;o[5]=t;o[10]=(fa+n)/(n-fa);o[11]=-1;o[14]=2*fa*n/(n-fa);return o;}
function look(e,c,u){let z=norm(sub(e,c)),x=norm(cross(u,z)),y=cross(z,x);const o=new Float32Array(16);
 o[0]=x[0];o[4]=x[1];o[8]=x[2];o[1]=y[0];o[5]=y[1];o[9]=y[2];o[2]=z[0];o[6]=z[1];o[10]=z[2];
 o[12]=-dot(x,e);o[13]=-dot(y,e);o[14]=-dot(z,e);o[15]=1;return o;}
function sub(a,b){return[a[0]-b[0],a[1]-b[1],a[2]-b[2]];}
function dot(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
function cross(a,b){return[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];}
function norm(a){const l=Math.hypot(a[0],a[1],a[2])||1;return[a[0]/l,a[1]/l,a[2]/l];}

// ---- camera (orbit about target, z up)
let cam={yaw:0.8,pitch:0.6,dist:90,tgt:[0,0,0]};
let cur=null, geo={};

function buf(data){const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,data,gl.STATIC_DRAW);return b;}
function drawLayer(l){
 gl.bindBuffer(gl.ARRAY_BUFFER,l.p);gl.enableVertexAttribArray(L.aP);gl.vertexAttribPointer(L.aP,3,gl.FLOAT,false,0,0);
 gl.bindBuffer(gl.ARRAY_BUFFER,l.c);gl.enableVertexAttribArray(L.aC);gl.vertexAttribPointer(L.aC,3,gl.FLOAT,false,0,0);
 gl.uniform1f(L.uPt,l.pt||1); gl.uniform1f(L.uA,l.a); gl.uniform1f(L.uRound,l.round?1:0);
 gl.drawArrays(l.mode,0,l.n);
}
function layerFrom(pts,cols,mode,opts){return Object.assign({p:buf(pts),c:buf(cols),n:pts.length/3,mode:mode},opts);}
function grey(n,v,a){const c=new Float32Array(n*3);for(let i=0;i<n;i++){c[3*i]=v[0];c[3*i+1]=v[1];c[3*i+2]=v[2];}return c;}

// cube wireframe around each voxel center (half size = 0.45 cell)
function cubes(centers,cell,col){
 const n=centers.length/3,h=0.45*cell,E=[[0,1],[1,3],[3,2],[2,0],[4,5],[5,7],[7,6],[6,4],[0,4],[1,5],[2,6],[3,7]];
 const P=new Float32Array(n*24*3),C=new Float32Array(n*24*3);let k=0;
 for(let i=0;i<n;i++){const cx=centers[3*i],cy=centers[3*i+1],cz=centers[3*i+2];
  const V=[];for(let j=0;j<8;j++)V.push([cx+((j&1)?h:-h),cy+((j&2)?h:-h),cz+((j&4)?h:-h)]);
  for(const [a,b] of E){for(const v of [V[a],V[b]]){P[k*3]=v[0];P[k*3+1]=v[1];P[k*3+2]=v[2];C[k*3]=col[0];C[k*3+1]=col[1];C[k*3+2]=col[2];k++;}}
 }
 return [P,C];
}

function selectObj(name){
 cur=name; const o=OBJS[name]; geo={};
 const v=o.variants;
 if(o.gauss){geo.gauss=layerFrom(b64f32(o.gauss.p),b64f32(o.gauss.c),gl.POINTS,{pt:2.0,a:0.9,round:true});}
 // seen surface: green points
 if(v.before&&v.before.seen&&v.before.seen.n){const s=b64f32(v.before.seen.p);geo.seen=layerFrom(s,grey(s.length/3,[0.35,0.85,0.35]),gl.POINTS,{pt:3.0,a:1.0,round:true});}
 for(const tag of ['before','after']){
  const m=v[tag]&&v[tag].mold; if(!m) continue;
  const cen=b64f32(m.p); const bottomArr=b64u8(m.bottom);
  const col=tag==='before'?[1.0,0.55,0.17]:[0.31,0.82,1.0];
  const [P,C]=cubes(cen,m.cell,col);
  geo['mold_'+tag]=layerFrom(P,C,gl.LINES,{a:0.9});
 }
 if(v.after&&v.after.carved&&v.after.carved.n){const s=b64f32(v.after.carved.p);geo.carved=layerFrom(s,grey(s.length/3,[0.9,0.2,0.2]),gl.POINTS,{pt:2.5,a:0.9,round:true});}
 const at=document.getElementById('assetToggles'); at.innerHTML='';
 (o.assets||[]).forEach((a,i)=>{const s=b64f32(a.p);geo['asset'+i]=layerFrom(s,grey(s.length/3,a.col),gl.POINTS,{pt:2.0,a:0.85,round:true});
   const c='#'+a.col.map(x=>Math.round(x*255).toString(16).padStart(2,'0')).join('');
   at.insertAdjacentHTML('beforeend',`<label><input type="checkbox" id="t_asset${i}"><span style="color:${c}">&#9632;</span> ${a.label} (dat vao canh)</label>`);});
 // table plane z=0: two triangles, translucent
 const R=o.plane_half_cm;
 const PP=new Float32Array([-R,-R,0, R,-R,0, R,R,0, -R,-R,0, R,R,0, -R,R,0]);
 geo.plane=layerFrom(PP,grey(6,[0.5,0.5,0.55]),gl.TRIANGLES,{a:0.22});
 // view centre: bbox of seen + gauss
 let mn=[1e9,1e9,0],mx=[-1e9,-1e9,-1e9];
 const src=(v.before&&v.before.seen)?b64f32(v.before.seen.p):(o.gauss?b64f32(o.gauss.p):new Float32Array(3));
 for(let i=0;i<src.length;i+=3){for(let k=0;k<3;k++){mn[k]=Math.min(mn[k],src[i+k]);mx[k]=Math.max(mx[k],src[i+k]);}}
 const c=[(mn[0]+mx[0])/2,(mn[1]+mx[1])/2,(mn[2]+mx[2])/2];
 cam.tgt=[c[0],c[1],Math.max(0,c[2])];
 cam.dist=Math.max(30,1.8*Math.max(mx[0]-mn[0],mx[1]-mn[1],mx[2]-mn[2],20));
 stats(o);
}

function fmt(x,d){return x==null||isNaN(x)?'-':Number(x).toFixed(d===undefined?2:d);}
function stats(o){
 const AS=o.assets||[]; const mk=new Set(); AS.forEach(a=>Object.keys(a.metrics||{}).forEach(k=>{if(typeof a.metrics[k]==='number'||typeof a.metrics[k]==='boolean')mk.add(k);}));
 document.getElementById('astats').innerHTML=AS.length?(`<tr><th>chi so</th>${AS.map(a=>`<th>${a.label}</th>`).join('')}</tr>`+
   [...mk].map(k=>`<tr><td>${k}</td>${AS.map(a=>{const x=a.metrics[k];return `<td>${typeof x==='boolean'?x:fmt(x,2)}</td>`;}).join('')}</tr>`).join('')):'<tr><td class=note>chua co</td></tr>';
 const b=o.variants.before&&o.variants.before.stats, a=o.variants.after&&o.variants.after.stats;
 const keys=[['dz_p98_cm','dz p98 khung (cm)',2],['dz_max_cm','dz max khung (cm)',2],['dz_seen_p98_cm','dz bề mặt thấy p98 (cm)',2],
   ['volume_cm3','the tich khung (cm3)',1],['mold_cm3','the tich khuon (cm3)',1],['mold_pct','khuon % khung',1]];
 const cells=[]; for(const [k,lab,d] of keys){
   const bv=b?b[k]:null, av=a?a[k]:null;
   cells.push(`<tr><td>${lab}</td><td>${fmt(bv,d)}</td><td>${a?fmt(av,d):'-'}</td><td>${a&&b?fmt(av-bv,d):'-'}</td></tr>`);
 }
 const g=o.gauss_n_total?`<tr><td>Gaussian vat (so)</td><td>${o.gauss_n_total}</td><td>-</td><td>-</td></tr>`:'';
 document.getElementById('stats').innerHTML=`<tr><th>${o.name||''}</th><th>${LAB[0]}</th><th>${LAB[1]}</th><th>delta</th></tr>`+cells.join('')+g+
   (a&&o.variants.after.carved?`<tr><td>bi khoet (cm3)</td><td>-</td><td>${fmt(o.variants.after.carved.cm3,1)}</td><td>-</td></tr>`:'')+(a?'':'<tr><td colspan=4 class=note>Chua co ban sau.</td></tr>');
}

function resize(){const w=cv.clientWidth,h=cv.clientHeight;cv.width=w*devicePixelRatio;cv.height=h*devicePixelRatio;gl.viewport(0,0,cv.width,cv.height);}
function render(){
 resize();
 gl.clearColor(0.08,0.09,0.11,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
 gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
 const cp=Math.cos(cam.pitch), eye=[cam.tgt[0]+cam.dist*cp*Math.cos(cam.yaw), cam.tgt[1]+cam.dist*cp*Math.sin(cam.yaw), cam.tgt[2]+cam.dist*Math.sin(cam.pitch)];
 const M=mul(persp(0.9,cv.width/cv.height,1,5000),look(eye,cam.tgt,[0,0,1]));
 gl.uniformMatrix4fv(L.uM,false,M);
 const ui={gauss:'t_gauss',seen:'t_seen',mold_before:'t_mold_b',mold_after:'t_mold_a',plane:'t_plane'};
 gl.depthMask(false); if(geo.plane&&document.getElementById('t_plane').checked) drawLayer(geo.plane); gl.depthMask(true);
 for(const k of ['gauss','seen','mold_before','mold_after','carved']){ const id=k==='carved'?'t_carved':ui[k]; if(geo[k]&&document.getElementById(id).checked) drawLayer(geo[k]); }
 for(let i=0;geo['asset'+i];i++){const e=document.getElementById('t_asset'+i); if(e&&e.checked) drawLayer(geo['asset'+i]);}
 requestAnimationFrame(render);
}

let drag=null;
cv.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,shift:e.shiftKey};cv.setPointerCapture(e.pointerId);});
cv.addEventListener('pointerup',()=>drag=null);
cv.addEventListener('pointermove',e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.x=e.clientX;drag.y=e.clientY;
 if(e.shiftKey){cam.tgt[0]-=dx*cam.dist*0.0015;cam.tgt[1]+=dy*cam.dist*0.0015;}else{cam.yaw-=dx*0.006;cam.pitch=Math.max(-1.5,Math.min(1.5,cam.pitch+dy*0.006));}});
cv.addEventListener('wheel',e=>{e.preventDefault();cam.dist*=Math.exp(e.deltaY*0.001);},{passive:false});

const sel=document.getElementById('obj');
for(const n of names){const o=document.createElement('option');o.value=n;o.textContent=n;sel.appendChild(o);}
sel.addEventListener('change',()=>{OBJS[sel.value].name=sel.value;selectObj(sel.value);});
OBJS[names[0]].name=names[0];selectObj(names[0]);
requestAnimationFrame(render);
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s3", required=True, help="thư mục S3_label (support.json, groups.json, gaussians/)")
    ap.add_argument("--before", required=True, help="thư mục S3m trước (frame/)")
    ap.add_argument("--after", default="", help="thư mục S3m sau (frame/); bỏ trống = không có")
    ap.add_argument("--out", required=True)
    ap.add_argument("--before_label", default="truoc")
    ap.add_argument("--after_label", default="sau")
    ap.add_argument("--asset", action="append", default=[], help="NHAN=THU_MUC (lặp được)")
    args = ap.parse_args()
    objs, summary, cm = build(args)
    data = json.dumps(objs, ensure_ascii=False, separators=(",", ":"))
    html = (HTML.replace("__DATA__", data).replace("__CM__", repr(cm)).replace("__LB0__", args.before_label)
            .replace("__LB1__", args.after_label).replace("__LAB__", json.dumps([args.before_label, args.after_label])))
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"viet {args.out} ({len(html) / 1e6:.1f} MB), {len(objs)} vat")
    print(f"{'vat':<12}{'dz truoc':>9}{'dz sau':>8}{'V truoc':>9}{'V sau':>8}{'khuon truoc':>13}{'khuon sau':>11}")
    for name, s in summary.items():
        b, a = s.get("before"), s.get("after")
        if not b:
            continue
        ds = f"{a['dz_p98_cm']:.2f}" if a else "-"
        Vs = f"{a['volume_cm3']:.1f}" if a else "-"
        km = f"{a['mold_cm3']:.1f}" if a else "-"
        print(f"{name:<12}{b['dz_p98_cm']:>9.2f}{ds:>8}{b['volume_cm3']:>9.1f}{Vs:>8}{b['mold_cm3']:>13.1f}{km:>11}")
    (Path(args.out).with_suffix(".summary.json")).write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
