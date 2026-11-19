"""
Đẩy pipeline/*.py lên Volume ở thư mục theo hash nội dung (code/pipeline_<hash>), in ra đường dẫn tương đối để truyền vào
các hàm Modal có tham số code_rel. Dùng khi Modal giữ bản thư mục gắn vào cũ (add_local_dir) dù đã deploy lại.

    python tools/push_code.py            -> in "code/pipeline_<hash>"
"""
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
files = sorted((REPO / "pipeline").glob("*.py"))
h = hashlib.sha1()
for f in files:
    h.update(f.name.encode()); h.update(f.read_bytes())
rel = f"code/pipeline_{h.hexdigest()[:12]}"
with tempfile.TemporaryDirectory() as d:
    for f in files:
        shutil.copy2(f, Path(d) / f.name)
    r = subprocess.run([sys.executable, "-m", "modal", "volume", "put", "--force", "safe-gs-come", d, rel], capture_output=True, text=True, encoding="utf-8", errors="replace", env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    if r.returncode != 0:
        sys.exit(r.stderr)
print(rel)
