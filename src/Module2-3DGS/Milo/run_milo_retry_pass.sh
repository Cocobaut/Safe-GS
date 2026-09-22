#!/bin/bash
# Luot chay lai thu 2 cho MILo: cho run_milo_supervisor.sh ket thuc, roi voi moi scene CHUA co mesh:
# cho RAM trong >= 20GB VA VRAM trong >= 19GB (MILo dinh ~18GB), chay lai NGUYEN pipeline tu dau
# (bash scripts/run_milo_one_scene.sh - khong resume, khong doi tham so), toi da 12 lan, nghi 10 phut giua cac lan.
# Moi lan that bai ghi lai iteration dat toi + co phai CUDA OOM khong + VRAM cua nguoi khac de chan doan.
#
# Chay: setsid nohup bash scripts/run_milo_retry_pass.sh > outputs/Replica/milo_retry_pass.log 2>&1 < /dev/null & disown

cd "$(dirname "$0")/.."
SCENES="office1 office2 office3 office4 room0 room1 room2"
MIN_RAM_GB=20
MIN_FREE_VRAM_MB=19000
MAX_TRIES=12

log() { echo "[$(date '+%d/%m %H:%M:%S')] $*"; }
avail_gb() { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }
free_vram_mb() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F, '{print $1-$2}' | head -1; }
has_mesh() { [ -f "outputs/Replica/mesh/$1_milo/mesh_learnable_sdf.ply" ]; }

log "luot chay lai thu 2 khoi dong; cho run_milo_supervisor.sh ket thuc..."
while pgrep -f "run_milo_supervisor.sh" > /dev/null; do sleep 60; done
log "supervisor dau da ket thuc."

for scene in $SCENES; do
    tries=0
    while ! has_mesh "$scene" && [ $tries -lt $MAX_TRIES ]; do
        tries=$((tries + 1))
        while [ "$(avail_gb)" -lt $MIN_RAM_GB ] || [ "$(free_vram_mb)" -lt $MIN_FREE_VRAM_MB ]; do
            log "[$scene] cho tai nguyen: RAM trong $(avail_gb)GB (can ${MIN_RAM_GB}), VRAM trong $(free_vram_mb)MiB (can ${MIN_FREE_VRAM_MB}); nghi 5 phut"; sleep 300
        done
        log "[$scene] chay lai lan $tries/$MAX_TRIES (RAM $(avail_gb)GB, VRAM trong $(free_vram_mb)MiB)"
        bash scripts/run_milo_one_scene.sh "$scene" >> "outputs/Replica/milo_retry_pass_${scene}.log" 2>&1
        if has_mesh "$scene"; then
            log "[$scene] XONG (co mesh)"
        else
            last=$(tr '\r' '\n' < "outputs/Replica/3dgs/log/${scene}_milo.log" | grep "Training progress" | tail -1 | cut -c1-110)
            oom=$(grep -c "OutOfMemoryError" "outputs/Replica/3dgs/log/${scene}_milo.log")
            gpu_last=$(tail -1 outputs/Replica/gpu_watch.log 2>/dev/null)
            log "[$scene] THAT BAI lan $tries: OOM_count=$oom | dat toi: $last | GPU: $gpu_last"
            sleep 600
        fi
    done
    has_mesh "$scene" || log "[$scene] THAT BAI sau $MAX_TRIES lan - can nguoi xem log"
done
log "luot chay lai xong. Scene co mesh: $(for s in $SCENES; do has_mesh $s && echo -n "$s "; done)"
