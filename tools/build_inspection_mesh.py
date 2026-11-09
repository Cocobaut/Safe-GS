"""
Build Interactive 3D Mesh Inspection Files (OBJ and Standalone 3D HTML Viewer).
Xuat vao Data/teatime/inspection_mesh/

Giup xoay 360 do de quan sat ro rang:
- Phan nhin thay duoc: Mesh vo ngoai tron min
- Phan thieu goc nhin (day vat): Lo thung rong thuc te hoac nap day phang
"""
import json
import struct
from pathlib import Path
import numpy as np


def parse_ply(path):
    with open(path, "rb") as f:
        n_verts = 0
        n_faces = 0
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            elif line.startswith("element face"):
                n_faces = int(line.split()[-1])
            elif line == "end_header":
                break
        verts = np.fromfile(f, dtype="<f4", count=n_verts * 3).reshape(n_verts, 3)
        faces = []
        for _ in range(n_faces):
            k = struct.unpack("<B", f.read(1))[0]
            faces.append(struct.unpack(f"<{k}i", f.read(k * 4)))
        return verts, np.array(faces)


def export_obj(path, verts, faces):
    with open(path, "w") as f:
        f.write("# Safe-GS 3D Object Mesh\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def simplify_mesh(verts, faces, target_faces=5000):
    """Giam so mat nhe nhang de nhung vao HTML viewer nhe va muot."""
    if len(faces) <= target_faces:
        return verts, faces
    # Step sampling of faces and used vertices
    step = len(faces) // target_faces
    sub_faces = faces[::step]
    used_v = np.unique(sub_faces)
    new_idx = np.full(len(verts), -1, dtype=int)
    new_idx[used_v] = np.arange(len(used_v))
    new_verts = verts[used_v]
    new_faces = new_idx[sub_faces]
    return new_verts, new_faces


def main():
    base_dir = Path("Data/teatime/pipeline/v1/S6_mesh/objects")
    out_dir = Path("Data/teatime/inspection_mesh")
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        ("coc_su", "Cốc sứ"),
        ("ly_tra", "Ly trà"),
        ("tao", "Quả táo"),
        ("dia", "Đĩa"),
    ]

    models_data = {}

    for name, title in targets:
        ply_path = base_dir / name / "visual.ply"
        if not ply_path.exists():
            continue

        v, f = parse_ply(ply_path)

        # Can tam vat ve (0, 0, 0) de xoay 360 do muot ma quanh truc tam
        center = (v.min(0) + v.max(0)) / 2
        v_centered = v - center

        # Phan biet mat vo ngoai va mat cat day (z sat mat ban)
        f_z = v[f, 2]
        is_bottom = (f_z < 0.001).all(axis=1)

        f_hollow = f[~is_bottom]
        f_capped = f

        # Xuat file OBJ do phan giai cao
        export_obj(out_dir / f"{name}_hollow.obj", v_centered, f_hollow)
        export_obj(out_dir / f"{name}_capped.obj", v_centered, f_capped)

        print(f"[+] Da xuat: {name}_hollow.obj ({len(f_hollow):,} faces) va {name}_capped.obj ({len(f_capped):,} faces)")

        # Chuan bi du lieu don gian cho WebGL HTML viewer
        # Scale ve kich thuoc chuan [-1, 1] cho WebGL
        scale = np.max(np.abs(v_centered))
        v_norm = v_centered / scale

        # Simplify nhe de file HTML nhe (~1MB)
        v_s_h, f_s_h = simplify_mesh(v_norm, f_hollow, target_faces=4000)
        v_s_c, f_s_c = simplify_mesh(v_norm, f_capped, target_faces=4000)

        # Tinh normals co ban
        models_data[name] = {
            "title": title,
            "verts_hollow": np.round(v_s_h, 4).flatten().tolist(),
            "faces_hollow": f_s_h.flatten().tolist(),
            "verts_capped": np.round(v_s_c, 4).flatten().tolist(),
            "faces_capped": f_s_c.flatten().tolist(),
            "hole_ratio": f"{float(is_bottom.sum() / len(f) * 100):.1f}% diện tích đáy khuyết",
        }

    # Tao file HTML tu chua WebGL 3D Viewer hoan toan doc lap (khong can mang, khong can Three.js CDN)
    html_template = """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <title>Safe-GS 360° 3D Mesh Inspection</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #121316; color: #e1e3e6; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; overflow: hidden; height: 100vh; display: flex; }
    #sidebar { width: 320px; background: #1a1c22; border-right: 1px solid #2a2e39; padding: 20px; display: flex; flex-direction: column; gap: 18px; z-index: 10; box-shadow: 4px 0 15px rgba(0,0,0,0.3); }
    h1 { font-size: 18px; font-weight: 600; color: #fff; border-bottom: 1px solid #2e3340; padding-bottom: 10px; }
    .section-title { font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: #8e95a5; margin-bottom: 6px; }
    .btn-group { display: flex; flex-direction: column; gap: 8px; }
    button { background: #262a36; color: #d0d4de; border: 1px solid #363c4d; border-radius: 8px; padding: 10px 14px; font-size: 14px; font-weight: 500; cursor: pointer; text-align: left; transition: all 0.2s; }
    button:hover { background: #323746; color: #fff; border-color: #4a5368; }
    button.active { background: #2563eb; color: #fff; border-color: #3b82f6; font-weight: 600; }
    .toggle-box { background: #20242f; border: 1px solid #2f3545; border-radius: 8px; padding: 12px; display: flex; flex-direction: column; gap: 10px; }
    .toggle-row { display: flex; align-items: center; justify-content: space-between; font-size: 13px; }
    input[type="checkbox"] { width: 18px; height: 18px; cursor: pointer; accent-color: #2563eb; }
    .guide-box { background: #161820; border-left: 3px solid #3b82f6; padding: 10px 12px; border-radius: 4px; font-size: 12px; line-height: 1.5; color: #9da5b4; }
    #canvas-container { flex: 1; position: relative; width: 100%; height: 100%; }
    canvas { width: 100%; height: 100%; display: block; }
    #info-tag { position: absolute; top: 20px; right: 20px; background: rgba(26,28,34,0.85); backdrop-filter: blur(8px); border: 1px solid #2a2e39; padding: 12px 18px; border-radius: 8px; font-size: 13px; color: #e1e3e6; }
    #info-tag b { color: #60a5fa; }
  </style>
</head>
<body>
  <div id="sidebar">
    <h1>Safe-GS 360° Inspection</h1>
    
    <div>
      <div class="section-title">Chọn vật thể quan sát</div>
      <div class="btn-group" id="obj-selector">
        <button class="active" onclick="switchObj('coc_su')">Cốc sứ (Coffee Mug)</button>
        <button onclick="switchObj('ly_tra')">Ly trà (Tea Glass)</button>
        <button onclick="switchObj('tao')">Quả táo (Apple)</button>
        <button onclick="switchObj('dia')">Đĩa bánh (Plate)</button>
      </div>
    </div>

    <div>
      <div class="section-title">Chế độ kiểm tra đáy (Hole Inspection)</div>
      <div class="toggle-box">
        <div class="toggle-row">
          <span>Xem đáy rỗng (Phần khuyết)</span>
          <input type="checkbox" id="chk-hollow" checked onchange="toggleHollow()">
        </div>
        <div class="toggle-row">
          <span>Khung lưới tam giác (Wireframe)</span>
          <input type="checkbox" id="chk-wire" onchange="toggleWire()">
        </div>
      </div>
    </div>

    <div class="guide-box">
      <b>Thao tác chuột:</b><br>
      - <b>Chuột trái:</b> Giữ và rê để xoay 360° (lật ngửa đáy lên xem).<br>
      - <b>Cuộn chuột:</b> Phóng to / thu nhỏ.<br>
      - <b>Chuột phải:</b> Di chuyển vật thể (Pan).
    </div>
  </div>

  <div id="canvas-container">
    <canvas id="glcanvas"></canvas>
    <div id="info-tag">Đang xem: <b id="lbl-title">Cốc sứ</b> | <span id="lbl-status" style="color: #f87171;">Đáy thủng (chỗ thiếu góc nhìn)</span></div>
  </div>

  <script>
    const MODELS = """ + json.dumps(models_data) + """;
    let curObj = 'coc_su';
    let isHollow = true;
    let isWire = false;

    // Mini WebGL 3D Orbit Engine
    const canvas = document.getElementById('glcanvas');
    const gl = canvas.getContext('webgl', { antialias: true });

    function resize() {
      canvas.width = canvas.parentElement.clientWidth;
      canvas.height = canvas.parentElement.clientHeight;
      gl.viewport(0, 0, canvas.width, canvas.height);
    }
    window.addEventListener('resize', resize);
    resize();

    // Shaders
    const vsSource = `
      attribute vec3 aPos;
      attribute vec3 aNormal;
      uniform mat4 uMVP;
      uniform mat4 uModel;
      varying vec3 vNormal;
      varying vec3 vWorldPos;
      void main() {
        vNormal = (uModel * vec4(aNormal, 0.0)).xyz;
        vWorldPos = (uModel * vec4(aPos, 1.0)).xyz;
        gl_Position = uMVP * vec4(aPos, 1.0);
      }
    `;
    const fsSource = `
      precision mediump float;
      varying vec3 vNormal;
      varying vec3 vWorldPos;
      uniform vec3 uColor;
      uniform bool uIsWire;
      void main() {
        if (uIsWire) {
          gl_FragColor = vec4(0.2, 0.7, 1.0, 1.0);
          return;
        }
        vec3 N = normalize(vNormal);
        vec3 L1 = normalize(vec3(1.0, 1.5, 1.0));
        vec3 L2 = normalize(vec3(-1.0, -1.0, -1.0));
        float diff1 = max(dot(N, L1), 0.0) * 0.75;
        float diff2 = max(dot(N, L2), 0.0) * 0.35;
        float amb = 0.25;
        vec3 col = uColor * (diff1 + diff2 + amb);
        gl_FragColor = vec4(col, 1.0);
      }
    `;

    function createShader(gl, type, source) {
      const s = gl.createShader(type);
      gl.shaderSource(s, source);
      gl.compileShader(s);
      return s;
    }
    const program = gl.createProgram();
    gl.attachShader(program, createShader(gl, gl.VERTEX_SHADER, vsSource));
    gl.attachShader(program, createShader(gl, gl.FRAGMENT_SHADER, fsSource));
    gl.linkProgram(program);
    gl.useProgram(program);

    const aPos = gl.getAttribLocation(program, 'aPos');
    const aNormal = gl.getAttribLocation(program, 'aNormal');
    const uMVP = gl.getUniformLocation(program, 'uMVP');
    const uModel = gl.getUniformLocation(program, 'uModel');
    const uColor = gl.getUniformLocation(program, 'uColor');
    const uIsWire = gl.getUniformLocation(program, 'uIsWire');

    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);

    let vBuffer = gl.createBuffer();
    let nBuffer = gl.createBuffer();
    let iBuffer = gl.createBuffer();
    let indexCount = 0;

    function buildMesh(verts, faces) {
      const V = new Float32Array(verts);
      const F = new Uint16Array(faces);
      const N = new Float32Array(V.length);

      // Compute vertex normals
      for (let i = 0; i < F.length; i += 3) {
        let i1 = F[i] * 3, i2 = F[i+1] * 3, i3 = F[i+2] * 3;
        let ax = V[i2]-V[i1], ay = V[i2+1]-V[i1+1], az = V[i2+2]-V[i1+2];
        let bx = V[i3]-V[i1], by = V[i3+1]-V[i1+1], bz = V[i3+2]-V[i1+2];
        let nx = ay*bz - az*by, ny = az*bx - ax*bz, nz = ax*by - ay*bx;
        N[i1] += nx; N[i1+1] += ny; N[i1+2] += nz;
        N[i2] += nx; N[i2+1] += ny; N[i2+2] += nz;
        N[i3] += nx; N[i3+1] += ny; N[i3+2] += nz;
      }
      for (let i = 0; i < N.length; i += 3) {
        let l = Math.hypot(N[i], N[i+1], N[i+2]) || 1;
        N[i] /= l; N[i+1] /= l; N[i+2] /= l;
      }

      gl.bindBuffer(gl.ARRAY_BUFFER, vBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, V, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, nBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, N, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, iBuffer);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, F, gl.STATIC_DRAW);
      indexCount = F.length;
    }

    function loadActiveMesh() {
      const data = MODELS[curObj];
      if (isHollow) {
        gl.disable(gl.CULL_FACE); // Double sided rendering so we can look inside the hollow bottom
        buildMesh(data.verts_hollow, data.faces_hollow);
        document.getElementById('lbl-status').innerHTML = '<span style="color:#f87171;">Đáy thủng (chỗ thiếu góc nhìn)</span>';
      } else {
        gl.enable(gl.CULL_FACE);
        buildMesh(data.verts_capped, data.faces_capped);
        document.getElementById('lbl-status').innerHTML = '<span style="color:#4ade80;">Đáy đã lấp kín (Capped)</span>';
      }
      document.getElementById('lbl-title').innerText = data.title;
    }

    // Camera & Orbit Controls
    let rotX = 0.3, rotY = -0.5, dist = 2.4;
    let panX = 0, panY = 0;
    let isDragging = false, dragButton = 0, lastX = 0, lastY = 0;

    canvas.addEventListener('mousedown', (e) => {
      isDragging = true;
      dragButton = e.button;
      lastX = e.clientX;
      lastY = e.clientY;
    });
    window.addEventListener('mouseup', () => isDragging = false);
    canvas.addEventListener('contextmenu', e => e.preventDefault());
    window.addEventListener('mousemove', (e) => {
      if (!isDragging) return;
      let dx = e.clientX - lastX, dy = e.clientY - lastY;
      lastX = e.clientX; lastY = e.clientY;
      if (dragButton === 0) {
        rotY += dx * 0.008;
        rotX += dy * 0.008;
      } else if (dragButton === 2) {
        panX += dx * 0.003;
        panY -= dy * 0.003;
      }
    });
    canvas.addEventListener('wheel', (e) => {
      dist *= (e.deltaY > 0) ? 1.08 : 0.92;
      dist = Math.max(0.5, Math.min(8.0, dist));
      e.preventDefault();
    });

    // Matrix helpers
    function mat4Perspective(fovy, aspect, near, far) {
      let f = 1.0 / Math.tan(fovy / 2), nf = 1 / (near - far);
      return [f/aspect, 0, 0, 0,  0, f, 0, 0,  0, 0, (far+near)*nf, -1,  0, 0, 2*far*near*nf, 0];
    }
    function mat4Multiply(a, b) {
      let out = new Float32Array(16);
      for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 4; j++) {
          out[i*4+j] = a[i*4+0]*b[0*4+j] + a[i*4+1]*b[1*4+j] + a[i*4+2]*b[2*4+j] + a[i*4+3]*b[3*4+j];
        }
      }
      return out;
    }

    function render() {
      gl.clearColor(0.08, 0.09, 0.11, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

      // Model matrix: rotations + centering
      let cx = Math.cos(rotX), sx = Math.sin(rotX);
      let cy = Math.cos(rotY), sy = Math.sin(rotY);
      let model = [
        cy, sx*sy, -cx*sy, 0,
        0,  cx,    sx,     0,
        sy, -sx*cy, cx*cy, 0,
        panX, panY, 0,     1
      ];

      // View matrix (lookAt origin from dist)
      let view = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, -dist, 1
      ];

      let proj = mat4Perspective(45 * Math.PI / 180, canvas.width / canvas.height, 0.1, 100.0);
      let mv = mat4Multiply(view, model);
      let mvp = mat4Multiply(proj, mv);

      gl.uniformMatrix4fv(uMVP, false, mvp);
      gl.uniformMatrix4fv(uModel, false, new Float32Array(model));

      // Color: warm cream for objects
      gl.uniform3f(uColor, 0.88, 0.86, 0.82);
      gl.uniform1i(uIsWire, isWire ? 1 : 0);

      gl.bindBuffer(gl.ARRAY_BUFFER, vBuffer);
      gl.enableVertexAttribArray(aPos);
      gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);

      gl.bindBuffer(gl.ARRAY_BUFFER, nBuffer);
      gl.enableVertexAttribArray(aNormal);
      gl.vertexAttribPointer(aNormal, 3, gl.FLOAT, false, 0, 0);

      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, iBuffer);
      gl.drawElements(isWire ? gl.LINES : gl.TRIANGLES, indexCount, gl.UNSIGNED_SHORT, 0);

      requestAnimationFrame(render);
    }

    function switchObj(name) {
      curObj = name;
      document.querySelectorAll('#obj-selector button').forEach(b => b.classList.remove('active'));
      event.target.classList.add('active');
      loadActiveMesh();
    }
    function toggleHollow() {
      isHollow = document.getElementById('chk-hollow').checked;
      loadActiveMesh();
    }
    function toggleWire() {
      isWire = document.getElementById('chk-wire').checked;
    }

    loadActiveMesh();
    requestAnimationFrame(render);
  </script>
</body>
</html>
"""
    viewer_path = out_dir / "viewer_360.html"
    viewer_path.write_text(html_template, encoding="utf-8")
    print(f"[+] Da tao viewer 360 do tuong tac doc lap: {viewer_path}")


if __name__ == "__main__":
    main()
