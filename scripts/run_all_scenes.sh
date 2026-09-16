#!/bin/bash
# Chạy COLMAP (SfM) + train Scene-GS (30k iters, checkpoint 7k/30k) cho TOÀN BỘ:
#   - 15 scan DTU đã có sẵn preprocessed images (DTU Preprocess/DTU/scan*)
#   - 8 scene Replica (Replica 8 Scene/Replica/{office0-4,room0-2})
#
# Chạy SONG SONG nhiều scene cùng lúc (mặc định 3) để tận dụng GPU 4090 (mỗi scene chỉ
# dùng ~4-8GB/24GB VRAM). Đổi số scene chạy cùng lúc bằng biến PARALLEL, vd:
#   PARALLEL=4 bash scripts/run_all_scenes.sh
#
# Có thể dừng giữa chừng (Ctrl+C) rồi chạy lại lệnh này - scene nào đã có checkpoint
# scene_gs_30000.ply rồi sẽ tự bỏ qua, không train lại từ đầu.
#
# Chạy:
#   cd "/media/ml4u/Extreme SSD/Safe-GS"
#   bash scripts/run_all_scenes.sh
#
# Xem tiến trình từng scene riêng: outputs/DTU/3dgs/log/<tên_scene>.log hoặc outputs/Replica/3dgs/log/<tên_scene>.log

set -e
cd "$(dirname "$0")/.."

PARALLEL="${PARALLEL:-3}"
DTU_ROOT="../DTU Preprocess/DTU"
REPLICA_ROOT="../Replica 8 Scene/Replica"

DTU_SCANS="24 37 40 55 63 65 69 83 97 105 106 110 114 118 122"
REPLICA_SCENES="office0 office1 office2 office3 office4 room0 room1 room2"

scene_list_file=$(mktemp)
for scan in $DTU_SCANS; do
    echo "dtu_scan${scan}|${DTU_ROOT}/scan${scan}/images" >> "$scene_list_file"
done
for scene in $REPLICA_SCENES; do
    echo "replica_${scene}|${REPLICA_ROOT}/${scene}/results/image" >> "$scene_list_file"
done

echo "Chạy ${PARALLEL} scene song song cùng lúc. Tổng $(wc -l < "$scene_list_file") scene."
echo "Theo dõi tiến trình riêng từng scene tại: outputs/DTU/3dgs/log/ hoặc outputs/Replica/3dgs/log/"

echo ""

xargs -a "$scene_list_file" -P "$PARALLEL" -I{} bash scripts/run_one_scene.sh {}

rm -f "$scene_list_file"

echo ""
echo "======================================================================"
echo ">>> HOÀN THÀNH TOÀN BỘ"
echo "======================================================================"
