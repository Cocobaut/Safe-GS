# Safe-GS pipeline (v1, S0 -> S6)

Ảnh RGB -> camera -> mask có ID -> Gaussian bám bề mặt (CoMe) -> gán Gaussian cho từng vật -> lấp nền -> chuẩn bị vật
-> mesh va chạm (mét) cho vật + mặt đỡ. Sơ đồ SOTA và gap: `paper/PIPELINE_SOTA.md`.

## Chạy

```bash
python -m modal run --detach modal_pipeline.py::run --config configs/pipeline/teatime_v1.json            # tới S6
python -m modal run --detach modal_pipeline.py::run --config configs/pipeline/teatime_v1.json --until S3
python -m modal run modal_pipeline.py::run --config configs/pipeline/teatime_v1.json --force S4a          # chạy lại S4a và mọi stage sau
python -m modal run modal_pipeline.py::run --config configs/pipeline/teatime_v1.json --only S5            # chỉ 1 stage
python -m modal run modal_pipeline.py::status --config configs/pipeline/teatime_v1.json
python -m modal run modal_pipeline.py::pull --config configs/pipeline/teatime_v1.json                     # preview + json
python -m modal run modal_pipeline.py::pull --config configs/pipeline/teatime_v1.json --with-ply          # cả .ply
```

Kết quả tải về: `Data/<scene>/pipeline/<run_name>/<stage>/`. Trên Volume `safe-gs-come`: `<scene>/pipeline/<run_name>/`.

## Stage

| Stage | Làm gì (v1) | Code | Checkpoint trong stage | Output chính |
|---|---|---|---|---|
| S0 colmap | Dùng lại COLMAP (`modal_come.py::colmap_teatime`) | modal_come.py | database.db, mapper | `summary.json` |
| S1 masks | Mask SAM2 của team (ID theo video) | modal_pipeline.py | - | `summary.json` |
| S2 come | Dùng lại CoMe 30k (`modal_come.py::train_teatime`) | modal_come.py | mỗi 1000 iter | `summary.json` |
| S3 label | Vote theo đóng góp render (gradient theo màu) cho từng ID thô -> đồ thị gộp ID; vote cho nhóm vật / mặt đỡ; Gaussian không đóng góp gần vật -> kNN; RANSAC mặt bàn + sàn -> mét | `s3_label.py` | vote cộng dồn mỗi 20 view | `labels.npy`, `groups.json`, `merge_report.json`, `support.json`, `gaussians/*.ply`, `preview/` [cảnh / nền + bàn / vật tô màu] |
| S4a inpaint | ProPainter inpaint video vùng các vật | modal_pipeline.py | từng đoạn 60 frame | `inpainted/`, `masks/`, `preview/` [gốc / mask / inpaint] |
| S4b background | Gaussian phẳng lấp vết vật trên bàn + tinh chỉnh Gaussian nền theo ảnh inpaint | `s4_background.py` | mỗi 500 iter | `background_finetuned.ply`, `fill.json`, `loss.csv`, `preview/` [đích / trước / sau] |
| S5 objects | Xuất Gaussian từng vật + thông tin đáy / chiều cao | `s5_objects.py` | từng vật | `objects/<tên>/{gaussians.ply, info.json, preview.jpg}` |
| S6 mesh | Marching cubes mức 0.5 trên trường opacity GOF của riêng vật -> hệ sim (mét, bàn z = 0) -> cắt + đóng đáy -> CoACD; bàn = hộp, sàn = mặt phẳng | `s6_mesh.py` | từng vật | `objects/<tên>/{visual.ply, collision_*.obj, info.json, preview.png}`, `support/`, `world_to_sim.json`, `scene.glb` |

| S7 physics | Khối lượng / ma sát theo config, gắn vật con vào vật cha (`attach`, vd bánh trên đĩa), sinh MJCF, kiểm MuJoCo đặt mesh đúng chỗ | `s7_physics.py` | - | `scene.xml`, `meshes/`, `physics.json` |
| S8 sim | MuJoCo: `settle` (để yên), `drop_all` (nâng + nghiêng nhẹ rồi thả), `lift_<vật>` (nhấc lên, giữ, thả) | `s8_sim.py` | - | `<kịch bản>/{trajectory.npz, report.json, height.png}`, `summary.json` |
| S9 render | Gaussian nền (S4b) + Gaussian vật (S5) biến đổi theo pose MuJoCo, render từ camera thật | `s9_render.py` | từng video | `<kịch bản>/<camera>.mp4`, `<camera>_keyframes.jpg` |

`run` kiểm tra trạng thái các stage bằng container CPU nhẹ trước, stage đã xong (hash khớp) thì không bật GPU.
Mỗi stage: `_done.json` (hash config), `config_used.json` (bản config đầy đủ), log `<scene>/logs/pipeline_<stage>.log`.
Hash của stage gồm config của nó + hash các stage nó phụ thuộc, nên đổi config ở đâu thì từ đó trở đi tự chạy lại.

## Config

`configs/pipeline/teatime_v1.json`. Muốn thử biến thể: copy file, đổi `run_name` (vd `v1_ratio04`) để kết quả nằm thư mục
riêng, không đè bản cũ.

## Giới hạn v1 (cố ý, nâng cấp sau)

- S1: chưa dùng SAM 3 (cần quyền HuggingFace `facebook/sam3`).
- S5: chưa hoàn thiện vật bằng SAM 3D Objects (cần quyền `facebook/sam-3d-objects`); đáy = mặt phẳng cắt ở S6.
- S6: scale mét dựa giả định bàn cao 0.75 m; vật không kín thì va chạm bằng bao lồi.
- S4: inpaint từng đoạn 60 frame độc lập (có thể lệch nhẹ ở chỗ nối đoạn).
