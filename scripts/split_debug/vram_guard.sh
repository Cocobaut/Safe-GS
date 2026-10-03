#!/bin/bash
# Canh VRAM 1 tien trinh: doc nvidia-smi moi giay, neu tien trinh <PID> vuot <LIMIT_MIB> thi kill ngay
# (tu kill tien trinh cua minh de khong lam nguoi dung chung GPU bi OOM). Ghi dinh VRAM vao <LOG>.
# Cach goi: bash vram_guard.sh <PID> <LIMIT_MIB> <LOG>
PID="$1"; LIMIT="$2"; LOG="$3"
peak=0
echo "[$(date '+%H:%M:%S')] canh PID $PID, nguong $LIMIT MiB" >> "$LOG"
while kill -0 "$PID" 2>/dev/null; do
    used=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
           | awk -F', ' -v p="$PID" '$1==p {print $2}')
    used=${used:-0}
    if [ "$used" -gt "$peak" ]; then peak=$used; fi
    if [ "$used" -gt "$LIMIT" ]; then
        echo "[$(date '+%H:%M:%S')] VUOT NGUONG: $used MiB > $LIMIT MiB -> kill PID $PID" >> "$LOG"
        kill -9 "$PID"
        break
    fi
    sleep 1
done
echo "[$(date '+%H:%M:%S')] ket thuc, dinh VRAM = $peak MiB" >> "$LOG"
echo "peak_mib=$peak" >> "$LOG"
