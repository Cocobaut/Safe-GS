"""
Video tracking 1 hoặc nhiều ID mask SAM2 trên ảnh render CoMe (cùng góc camera và kích thước với ảnh gốc).

- Mỗi frame: ảnh render; ID nào có trong label PNG của frame đó -> tô màu bán trong suốt + viền + nhãn ID
  (mỗi ID một màu).
- Frame không có ID nào (hoặc không có mask) -> giữ nguyên ảnh render, ghi "(no ID)".
- Góc trên trái: số thứ tự frame trong video (1, 2, 3, ...) và tên frame gốc.

Ví dụ:
    python tools/make_tracking_video.py --id 22
    python tools/make_tracking_video.py --id 29 --fps 6 --name ly_tra
    python tools/make_tracking_video.py --id 2 9 6 --name gau_bong      # nhiều ID, mỗi ID một màu
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
TEATIME = REPO / "Data" / "teatime"
# BGR: cam, xanh lơ, hồng, xanh lá, vàng, tím
COLORS = [(0, 140, 255), (255, 200, 0), (255, 0, 200), (0, 200, 0), (0, 230, 230), (200, 80, 120)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", type=int, nargs="+", required=True, help="1 hoặc nhiều ID vật trong sam2_masks/labels")
    ap.add_argument("--renders", type=Path, default=TEATIME / "come" / "renders_all")
    ap.add_argument("--labels", type=Path, default=TEATIME / "sam2_masks" / "labels")
    ap.add_argument("--out", type=Path, default=TEATIME / "come" / "tracking")
    ap.add_argument("--name", type=str, default="", help="tên thêm vào file video, vd 'coc_su'")
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--alpha", type=float, default=0.45)
    args = ap.parse_args()

    frames = sorted(args.renders.glob("frame_*.png"))
    assert frames, f"không có ảnh trong {args.renders}"
    h, w = cv2.imread(str(frames[0])).shape[:2]
    colors = {k: np.array(COLORS[j % len(COLORS)], np.uint8) for j, k in enumerate(args.id)}

    args.out.mkdir(parents=True, exist_ok=True)
    stem = "id" + "_".join(map(str, args.id)) + (f"_{args.name}" if args.name else "")
    mp4 = args.out / f"tracking_{stem}.mp4"
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    assert writer.isOpened(), "không mở được VideoWriter (mp4v)"

    hits = {k: 0 for k in args.id}
    for i, f in enumerate(frames, 1):
        img = cv2.imread(str(f))
        lab_path = args.labels / f"{f.stem}.png"
        lab = cv2.imread(str(lab_path), cv2.IMREAD_UNCHANGED) if lab_path.exists() else None
        if lab is not None:
            assert lab.shape[:2] == (h, w), f"{lab_path} khác kích thước ảnh render"
        found = []
        for k in args.id:
            if lab is None:
                continue
            mask = lab == k
            if not mask.any():
                continue
            found.append(k); hits[k] += 1
            c = colors[k]
            img[mask] = (img[mask] * (1 - args.alpha) + c * args.alpha).astype(np.uint8)
            cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, c.tolist(), 2)
            ys, xs = np.nonzero(mask)
            pos = (max(int(xs.mean()) - 30, 0), max(int(ys.min()) - 8, 60))
            cv2.putText(img, f"ID {k}", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.putText(img, f"ID {k}", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, c.tolist(), 2, cv2.LINE_AA)
        # số frame ở góc
        txt = f"{i}/{len(frames)}  {f.stem}" + ("" if found else "  (no ID)")
        cv2.rectangle(img, (0, 0), (12 + 13 * len(txt), 36), (0, 0, 0), -1)
        cv2.putText(img, txt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(img)
    writer.release()
    print(f"{mp4}  |  {len(frames)} frame, {args.fps} fps ({len(frames) / args.fps:.0f} giây) | số frame có mỗi ID: {hits}")


if __name__ == "__main__":
    main()
