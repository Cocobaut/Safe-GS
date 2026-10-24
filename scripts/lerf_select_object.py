"""
Chọn vật cần xóa cho InFusion (chạy LOCAL, có cửa sổ GUI)
========================================================
Mở ảnh của scene LERF, bạn click lên vật cần xóa -> lưu prompt điểm cho SAM2 (outputs/lerf_prompts/<scene>.json).
SAM2 trên Modal sẽ lan mask từ các điểm này ra toàn bộ video.

    python scripts/lerf_select_object.py --scene teatime
    python scripts/lerf_select_object.py --scene figurines --frame frame_00120.jpg

Phím tắt trong cửa sổ:
    Chuột trái  : điểm DƯƠNG (thuộc vật cần xóa)
    Chuột phải  : điểm ÂM (không thuộc vật, dùng để sửa khi mask lan sang vùng khác)
    A / D       : ảnh trước / sau (W / S: nhảy 10 ảnh)
    N           : bắt đầu vật mới (xóa nhiều vật cùng lúc)
    U           : hoàn tác điểm cuối
    V           : đặt ảnh hiện tại làm view inpaint cho InFusion (mặc định: ảnh đầu tiên có click)
    Enter / P   : lưu prompt và thoát
    Esc / Q     : thoát không lưu
Mẹo: click vật trên 2-3 ảnh nhìn từ các góc khác nhau giúp SAM2 lan mask ổn định hơn.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

COLORS = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 200, 255), (255, 0, 255)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, help="figurines hoặc teatime")
    parser.add_argument("--lerf-root", default="F:/LERF Datasets")
    parser.add_argument("--frame", default=None, help="ảnh bắt đầu, ví dụ frame_00042.jpg")
    parser.add_argument("--out", default=None, help="mặc định: outputs/lerf_prompts/<scene>.json")
    args = parser.parse_args()

    img_dir = Path(args.lerf_root) / args.scene / args.scene / "images"
    frames = sorted(p.name for p in img_dir.glob("*.jpg"))
    if not frames:
        raise SystemExit(f"Không thấy ảnh trong {img_dir}")
    out = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "outputs" / "lerf_prompts" / f"{args.scene}.json"

    # objects[i] = {frame_name: {"points": [[x, y]], "labels": [1/0]}}
    objects: list[dict] = [{}]
    history: list[tuple[int, str]] = []
    state = {"idx": frames.index(args.frame) if args.frame else 0, "view": None}
    if out.exists():
        data = json.loads(out.read_text(encoding="utf-8"))
        objects = [{p["frame"]: {"points": p["points"], "labels": p["labels"]} for p in o["prompts"]} for o in data["objects"]]
        state["view"] = data.get("view")
        print(f"[i] Đã nạp prompt cũ từ {out}")

    def on_mouse(event, x, y, flags, param):
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        name = frames[state["idx"]]
        entry = objects[-1].setdefault(name, {"points": [], "labels": []})
        entry["points"].append([int(x), int(y)])
        entry["labels"].append(1 if event == cv2.EVENT_LBUTTONDOWN else 0)
        history.append((len(objects) - 1, name))

    win = f"Chon vat can xoa - {args.scene}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        name = frames[state["idx"]]
        img = cv2.imread(str(img_dir / name))
        for oi, obj in enumerate(objects):
            if name not in obj:
                continue
            color = COLORS[oi % len(COLORS)]
            for (x, y), lab in zip(obj[name]["points"], obj[name]["labels"]):
                cv2.circle(img, (x, y), 7, color, -1 if lab == 1 else 2)
                if lab == 0:
                    cv2.line(img, (x - 6, y - 6), (x + 6, y + 6), color, 2)
        view_tag = "  [VIEW INPAINT]" if state["view"] == Path(name).stem else ""
        info = f"{name} ({state['idx'] + 1}/{len(frames)})  vat #{len(objects)}{view_tag}"
        cv2.putText(img, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(img, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.imshow(win, img)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("a"), ord("A")):
            state["idx"] = max(0, state["idx"] - 1)
        elif key in (ord("d"), ord("D")):
            state["idx"] = min(len(frames) - 1, state["idx"] + 1)
        elif key in (ord("w"), ord("W")):
            state["idx"] = max(0, state["idx"] - 10)
        elif key in (ord("s"), ord("S")):
            state["idx"] = min(len(frames) - 1, state["idx"] + 10)
        elif key in (ord("n"), ord("N")):
            if objects[-1]:
                objects.append({})
        elif key in (ord("u"), ord("U")) and history:
            oi, fname = history.pop()
            e = objects[oi][fname]
            e["points"].pop()
            e["labels"].pop()
            if not e["points"]:
                del objects[oi][fname]
        elif key in (ord("v"), ord("V")):
            state["view"] = Path(name).stem
        elif key in (13, ord("p"), ord("P")):
            break
        elif key in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            print("[i] Thoát, không lưu.")
            return
    cv2.destroyAllWindows()

    objects = [o for o in objects if o]
    if not objects:
        print("[X] Chưa click điểm nào, không lưu.")
        return
    prompted = sorted({f for o in objects for f in o})
    data = {
        "scene": args.scene,
        "view": state["view"] or Path(prompted[0]).stem,
        "objects": [
            {"id": i + 1, "prompts": [{"frame": f, **o[f]} for f in sorted(o)]} for i, o in enumerate(objects)
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[✓] Đã lưu {len(objects)} vật, {len(prompted)} ảnh có click -> {out}")
    print(f"[✓] View inpaint cho InFusion: {data['view']}")


if __name__ == "__main__":
    main()
