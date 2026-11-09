"""
Gọi 1 hàm của app Modal đã deploy, chạy độc lập với máy local (tắt máy / hết phiên vẫn chạy).

    python -m modal deploy modal_objectgs.py
    python tools/modal_spawn.py safe-gs-objectgs _train
    python tools/modal_spawn.py safe-gs-inpaint360 _remove '"1,2,3"' '"None"'   # tham số là JSON
"""
import json
import sys

import modal

app, fn, *args = sys.argv[1:]
call = modal.Function.from_name(app, fn).spawn(*[json.loads(a) for a in args])
print(f"spawned {app}.{fn} call_id={call.object_id}")
