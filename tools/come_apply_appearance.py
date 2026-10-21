"""
Cho một ảnh RGB đi qua appearance network của CoMe (đã train) để xem ảnh sau khi được chỉnh độ sáng/màu.

CoMe học cho mỗi ảnh train một embedding 64 chiều + một CNN nhỏ. CNN nhìn ảnh (downsample x32), embedding
của view và toạ độ (x, y, r) -> sinh ra một map hệ số nhân (3 kênh, mượt, tần số thấp). Ảnh chỉnh = map * ảnh.
Vì embedding gắn với từng ảnh train, phải chọn "chỉnh theo kiểu của view nào" bằng --view.

Cần thư mục model đã train với --use_vastgaussian_appearance hoặc --use_ssimdecoupled_appearance, có:
    <model>/cameras.json
    <model>/point_cloud/iteration_<N>/appearance_embedding.pth

Đầu ra trong --out:
    mapped.png      ảnh sau khi chỉnh
    multiplier.png  map hệ số nhân, hiển thị [0.5, 1.5] -> [0, 1] (xám 50% = không đổi)
    multiplier.npy  map hệ số nhân thô (3xHxW)
    compare.png     [ảnh gốc | ảnh chỉnh | map hệ số]

Ví dụ (chạy trong môi trường conda của CoMe, cần CUDA):
    python tools/come_apply_appearance.py -m output/figurines --image my.png --view frame_00012
    python tools/come_apply_appearance.py -m output/figurines --list
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

COME_DIR = Path(__file__).resolve().parents[1] / "Based_Model" / "CoMe"
sys.path.insert(0, str(COME_DIR))

from scene.appearance_network import AppearanceEmbedding, VastGaussianAppearanceEmbedding  # noqa: E402


def find_embedding(model: Path, iteration: int | None) -> Path:
    pc_dir = model / "point_cloud"
    if iteration is None:
        iters = [int(p.name.split("_")[-1]) for p in pc_dir.glob("iteration_*")]
        if not iters:
            raise FileNotFoundError(f"không có {pc_dir}/iteration_*")
        iteration = max(iters)
    path = pc_dir / f"iteration_{iteration}" / "appearance_embedding.pth"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def train_cameras(model: Path) -> list[dict]:
    # train.py đánh camera.idx theo thứ tự train camera sau khi shuffle, cũng là thứ tự ghi cameras.json
    with open(model / "cameras.json") as f:
        cams = json.load(f)
    return [c for c in cams if c.get("is_train_camera", True)]


def resolve_view(view: str, cams: list[dict]) -> int:
    if view.isdigit():
        return int(view)
    for c in cams:
        if c["img_name"] == view or Path(c["img_name"]).stem == Path(view).stem:
            return c["id"]
    raise ValueError(f"không tìm thấy view '{view}' trong cameras.json (dùng --list để xem)")


def to_png(t: torch.Tensor, path: Path):
    arr = (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--model", type=Path, required=True)
    ap.add_argument("--image", type=Path)
    ap.add_argument("--view", type=str, help="tên ảnh train (img_name) hoặc chỉ số embedding")
    ap.add_argument("--iteration", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--list", action="store_true", help="liệt kê các view train và chỉ số embedding")
    args = ap.parse_args()

    cams = train_cameras(args.model)
    if args.list:
        for c in sorted(cams, key=lambda c: c["id"]):
            print(f"{c['id']:4d}  {c['img_name']}")
        return
    if args.image is None or args.view is None:
        ap.error("cần --image và --view (hoặc --list)")

    capture = torch.load(find_embedding(args.model, args.iteration), map_location="cuda")
    emb = AppearanceEmbedding.load_from_capture(capture).cuda().eval()
    if not isinstance(emb, VastGaussianAppearanceEmbedding):
        raise SystemExit("model này train không có appearance embedding (AppearanceEmbedding mặc định) -> không có gì để áp")

    idx = resolve_view(args.view, cams)
    if not 0 <= idx < emb.num_view:
        raise SystemExit(f"chỉ số {idx} ngoài khoảng [0, {emb.num_view})")

    rgb = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.float32) / 255.0
    image = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().cuda()

    with torch.no_grad():
        mapped = emb.appearance_mapping(image, idx)
    multiplier = mapped / image.clamp_min(1e-4)

    out = args.out or args.model / "appearance_preview" / f"{args.image.stem}__view{idx}"
    out.mkdir(parents=True, exist_ok=True)
    to_png(mapped, out / "mapped.png")
    to_png(multiplier - 0.5, out / "multiplier.png")
    np.save(out / "multiplier.npy", multiplier.cpu().numpy())
    to_png(torch.cat([image, mapped, multiplier - 0.5], dim=2), out / "compare.png")

    m = multiplier.flatten(1)
    print(f"view {idx} ({next((c['img_name'] for c in cams if c['id'] == idx), '?')})")
    print("hệ số nhân trung bình R/G/B:", [round(v, 3) for v in m.mean(1).tolist()],
          " min/max:", round(m.min().item(), 3), round(m.max().item(), 3))
    print("đã ghi:", out)


if __name__ == "__main__":
    main()
