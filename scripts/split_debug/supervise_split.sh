#!/bin/bash
# Giam sat doc lap cho run_split_debug.sh (buoc 2+3): cho lan chay hien tai ket thuc; neu chua co ket qua
# (04_final/instances.csv) thi chay lai (toi da MAX_TRIES lan, giua cac lan cho GPU/RAM trong), khong doi thuat toan.
# Xong thi ghi <out>/SUMMARY.txt. Log: _work/<scene>/logs/supervisor.log
SCENE="${1:-office0}"
MAX_TRIES="${MAX_TRIES:-3}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="/media/ml4u/Extreme SSD/Safe-GS/outputs/Replica/split/split_and_splat/$SCENE"
LOGS="/media/ml4u/Extreme SSD/Safe-GS/outputs/Replica/split/split_and_splat/_work/$SCENE/logs"
mkdir -p "$LOGS"
log() { echo "[$(date '+%d/%m %H:%M:%S')] $*" >> "$LOGS/supervisor.log"; }

done_ok() { [ -s "$OUT/04_final/instances.csv" ] && [ -s "$OUT/run_info.json" ]; }

log "giam sat bat dau (scene $SCENE)"
while pgrep -f "run_split_debug.sh $SCENE" > /dev/null; do sleep 30; done
tries=0
while ! done_ok && [ $tries -lt "$MAX_TRIES" ]; do
    tries=$((tries + 1))
    log "chua co ket qua -> chay lai buoc 2+3 lan $tries/$MAX_TRIES"
    while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 15 ]; do log "RAM trong < 15GB, cho 5 phut"; sleep 300; done
    rm -rf "/media/ml4u/Extreme SSD/Safe-GS/outputs/Replica/split/split_and_splat/_work/$SCENE/output/${SCENE}_masks"
    STEPS=23 bash "$HERE/run_split_debug.sh" "$SCENE"
    done_ok || { log "lan $tries that bai: $(tail -1 "$LOGS/run.log")"; sleep 300; }
done
if done_ok; then
    log "XONG, ghi tom tat"
    PYTHONNOUSERSITE=1 /home/ml4u/conda_envs/split_and_splat/bin/python "$HERE/summarize_split.py" "$SCENE" >> "$LOGS/supervisor.log" 2>&1
else
    log "THAT BAI sau $MAX_TRIES lan, can xem log"
fi
