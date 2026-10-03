#!/bin/bash
# Chay bo debug giai doan Split cua Split&Splat (khong sua code goc), co canh VRAM <= 7000 MiB.
#   STEPS=1  : buoc 1 (SAM2 cat mask 2D)          -> outputs/Replica/split/split_and_splat/<scene>/01_mask_2d
#   STEPS=23 : buoc 2+3 (khung 3D + dong bo mask) -> 02_skeleton_3d, 03_propagation, 04_final, run_info.json
#   STEPS=all: ca hai
# Chay nen: STEPS=all setsid nohup bash scripts/split_debug/run_split_debug.sh office0 > /dev/null 2>&1 < /dev/null & disown
# Theo doi: tail -f outputs/Replica/split/split_and_splat/_work/<scene>/logs/*.log
SCENE="${1:-office0}"
STEPS="${STEPS:-all}"
LIMIT_MIB="${LIMIT_MIB:-7000}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/home/ml4u/conda_envs/split_and_splat/bin/python
WORK="/media/ml4u/Extreme SSD/Safe-GS/outputs/Replica/split/split_and_splat/_work/$SCENE"
LOGS="$WORK/logs"
mkdir -p "$LOGS"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPLIT_VRAM_CAP_GB="${SPLIT_VRAM_CAP_GB:-6.0}"

wait_gpu() {  # cho den khi GPU con trong >= 7500 MiB (may dung chung)
    while true; do
        free=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1-$2}' | head -1)
        [ "$free" -ge 7500 ] && return
        echo "[$(date '+%H:%M:%S')] GPU con trong $free MiB < 7500, cho 60s..." >> "$LOGS/run.log"; sleep 60
    done
}

run_guarded() {  # run_guarded <ten_log> <lenh...>
    local name="$1"; shift
    wait_gpu
    echo "[$(date '+%H:%M:%S')] BAT DAU $name" >> "$LOGS/run.log"
    "$@" > "$LOGS/$name.log" 2>&1 &
    local pid=$!
    bash "$HERE/vram_guard.sh" "$pid" "$LIMIT_MIB" "$LOGS/${name}_vram.log" &
    wait "$pid"; local rc=$?
    echo "[$(date '+%H:%M:%S')] KET THUC $name (exit $rc), $(tail -1 "$LOGS/${name}_vram.log" 2>/dev/null)" >> "$LOGS/run.log"
    return $rc
}

cd "$HERE"
if [ "$STEPS" = "1" ] || [ "$STEPS" = "all" ]; then
    run_guarded step1_mask2d "$PY" -u run_step1_mask2d.py --scene "$SCENE" || exit 1
fi
if [ "$STEPS" = "23" ] || [ "$STEPS" = "all" ]; then
    "$PY" make_instrumented_propagation.py --scene "$SCENE" >> "$LOGS/run.log" 2>&1 || exit 1
    run_guarded step23_propagation "$PY" -u run_step23_propagation.py --scene "$SCENE" || exit 1
fi
echo "[$(date '+%H:%M:%S')] XONG ($STEPS)" >> "$LOGS/run.log"
