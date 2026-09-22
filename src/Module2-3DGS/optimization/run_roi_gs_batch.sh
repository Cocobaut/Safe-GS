#!/bin/bash
# Chay ROI-GS (dung y chang paper) cho tat ca object y nghia cua office0, SONG SONG 2
# object cung luc (moi object ~10GB VRAM, card 24GB con du). Moi object ~30k iterations
# (~10 phut o toc do hien tai) -> 22 object / 2 song song ~ 110 phut.
#
# Chay:
#   cd "/media/ml4u/Extreme SSD/Safe-GS"
#   bash tmp/run_roi_gs_batch.sh

set -e
cd "$(dirname "$0")/.."
export PYTHONNOUSERSITE=1
export DISPLAY=:0
PY=/home/ml4u/conda_envs/safe-gs/bin/python

OUT_DIR="tmp/roi_gs_objects"
LOG_DIR="tmp/roi_gs_logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

# instance_id:category (loc san, bo wall/ceiling/floor/switch/wall-plug/panel/vent/camera/unknown)
OBJECTS="4:chair 5:rug 7:sofa 9:sofa 12:table 15:blinds 16:door 19:bin 20:blinds 21:lamp 22:indoor-plant 23:plant-stand 28:tissue-paper 34:bin 37:lamp 44:desk-organizer 56:pillar 57:clock 58:table 61:chair 64:tablet 66:tv-screen"

run_one() {
    IFS=':' read -r iid cat <<< "$1"
    ckpt="${OUT_DIR}/office0_${iid}_${cat}_roigs.ply"
    if [ -f "$ckpt" ]; then
        echo "[SKIP] object ${iid}:${cat} đã xong."
        return 0
    fi
    "$PY" tmp/refine_object_in_scene.py --instance_id "$iid" --category "$cat" \
        > "${LOG_DIR}/${iid}_${cat}.log" 2>&1
    echo "[DONE] object ${iid}:${cat}"
}
export -f run_one
export OUT_DIR LOG_DIR PY

PARALLEL="${PARALLEL:-2}"
echo "$OBJECTS" | tr ' ' '\n' | xargs -P "$PARALLEL" -I{} bash -c 'run_one "$@"' _ {}

echo ""
echo "======================================================================"
echo ">>> HOAN THANH ROI-GS BATCH (22 object)"
echo "======================================================================"
