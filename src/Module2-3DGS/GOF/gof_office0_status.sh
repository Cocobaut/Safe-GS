#!/bin/bash
# Xem nhanh tien do train GOF goc cho office0_gof.
# Chay 1 lan:      bash tmp/gof_office0_status.sh
# Tu refresh 30s:  watch -n 30 'bash "/media/ml4u/Extreme SSD/Safe-GS/tmp/gof_office0_status.sh"'
cd "$(dirname "$0")/.."

TRAIN_LOG="outputs/Replica/3dgs/log/office0_gof.log"
MESH_LOG="outputs/Replica/mesh/log/office0_gof.log"
CKPT_DIR="outputs/Replica/3dgs/office0_gof/point_cloud"
MESH_DIR="outputs/Replica/mesh/office0_gof"

echo "===================== GOF office0_gof — $(date '+%H:%M:%S') ====================="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "---------------------------------------------------------------------------"

if pgrep -f "train\.py -s.*office0_gof" > /dev/null; then
    echo "[TRAIN] dang chay..."
elif [ -d "$CKPT_DIR/iteration_30000" ]; then
    echo "[TRAIN] DA XONG (checkpoint 30000 co san)"
else
    echo "[TRAIN] khong thay process (chua chay hoac da dung)"
fi

if [ -f "$TRAIN_LOG" ]; then
    line=$(grep -oE '[0-9]+%\|[^|]*\|\s*[0-9]+/30000[^]]*\]' "$TRAIN_LOG" | tail -1)
    echo "  Tien do: ${line:-chua co du lieu iter}"
fi

echo "---------------------------------------------------------------------------"
if pgrep -f "extract_mesh\.py.*office0_gof" > /dev/null; then
    echo "[MESH] dang chay..."
    if [ -f "$MESH_LOG" ]; then
        tail -n 3 "$MESH_LOG"
    fi
elif find "$MESH_DIR" -iname "mesh_binary_search_*.ply" 2>/dev/null | grep -q .; then
    echo "[MESH] DA XONG"
else
    echo "[MESH] chua chay"
fi
