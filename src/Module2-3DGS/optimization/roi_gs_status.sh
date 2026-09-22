#!/bin/bash
# Xem nhanh tiến độ batch ROI-GS (22 object office0).
# Chạy 1 lần:      bash tmp/roi_gs_status.sh
# Tự refresh 30s:  watch -n 30 bash tmp/roi_gs_status.sh
cd "$(dirname "$0")/.."

OBJECTS="4:chair 5:rug 7:sofa 9:sofa 12:table 15:blinds 16:door 19:bin 20:blinds 21:lamp 22:indoor-plant 23:plant-stand 28:tissue-paper 34:bin 37:lamp 44:desk-organizer 56:pillar 57:clock 58:table 61:chair 64:tablet 66:tv-screen"

echo "===================== ROI-GS office0 — $(date '+%H:%M:%S') ====================="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "---------------------------------------------------------------------------"
printf "%-4s %-16s %-10s %-8s %-14s\n" "ID" "CATEGORY" "STATUS" "PCT" "ITER"

done_n=0; run_n=0; pend_n=0
for pair in $OBJECTS; do
    iid="${pair%%:*}"
    cat="${pair##*:}"
    ckpt="tmp/roi_gs_objects/office0_${iid}_${cat}_roigs.ply"
    log="tmp/roi_gs_logs/${iid}_${cat}.log"
    if [ -f "$ckpt" ]; then
        status="DONE"; pct="100%"; iter="30000/30000"; done_n=$((done_n+1))
    elif [ -f "$log" ]; then
        line=$(grep -oE '[0-9]+%\|[^|]*\| [0-9]+/30000' "$log" | tail -1)
        if [ -n "$line" ]; then
            pct=$(echo "$line" | grep -oE '^[0-9]+%')
            iter=$(echo "$line" | grep -oE '[0-9]+/30000')
            status="running"; run_n=$((run_n+1))
        else
            status="starting"; pct="0%"; iter="0/30000"; run_n=$((run_n+1))
        fi
    else
        status="queued"; pct="-"; iter="-"; pend_n=$((pend_n+1))
    fi
    printf "%-4s %-16s %-10s %-8s %-14s\n" "$iid" "$cat" "$status" "$pct" "$iter"
done

echo "---------------------------------------------------------------------------"
echo "Done: $done_n   Running: $run_n   Queued: $pend_n   Total: 22"
