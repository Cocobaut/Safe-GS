#!/bin/bash
# Giam sat: cho batch run_milo_all_remaining.sh dang chay xong, roi voi moi scene CHUA co mesh MILo thi
# cho RAM he thong con trong >= 20GB, chay lai NGUYEN pipeline tu dau (bash scripts/run_milo_one_scene.sh,
# khong resume, khong doi tham so), toi da 6 lan, giua cac lan nghi 10 phut. Chay doc lap, khong can nguoi canh.
#
# Chay: setsid nohup bash scripts/run_milo_supervisor.sh > outputs/Replica/milo_supervisor.log 2>&1 < /dev/null & disown

cd "$(dirname "$0")/.."
SCENES="office1 office2 office3 office4 room0 room1 room2"
MIN_AVAIL_GB=20
MAX_TRIES=6

log() { echo "[$(date '+%d/%m %H:%M:%S')] $*"; }
avail_gb() { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }
has_mesh() { [ -f "outputs/Replica/mesh/$1_milo/mesh_learnable_sdf.ply" ]; }

log "giam sat khoi dong; cho batch chinh (run_milo_all_remaining.sh) ket thuc..."
while pgrep -f "run_milo_all_remaining.sh" > /dev/null; do sleep 60; done
log "batch chinh da ket thuc."

for scene in $SCENES; do
    tries=0
    while ! has_mesh "$scene" && [ $tries -lt $MAX_TRIES ]; do
        tries=$((tries + 1))
        while [ "$(avail_gb)" -lt $MIN_AVAIL_GB ]; do
            log "[$scene] RAM trong $(avail_gb)GB < ${MIN_AVAIL_GB}GB, cho 5 phut..."; sleep 300
        done
        log "[$scene] chay lai lan $tries/$MAX_TRIES (RAM trong $(avail_gb)GB)"
        bash scripts/run_milo_one_scene.sh "$scene" >> "outputs/Replica/milo_supervisor_${scene}.log" 2>&1
        if has_mesh "$scene"; then log "[$scene] XONG (co mesh)"; else log "[$scene] van chua co mesh, nghi 10 phut roi thu lai"; sleep 600; fi
    done
    has_mesh "$scene" || log "[$scene] THAT BAI sau $MAX_TRIES lan - can xem log outputs/Replica/3dgs/log/${scene}_milo.log"
done

log "giam sat xong. Scene co mesh: $(for s in $SCENES; do has_mesh $s && echo -n "$s "; done)"
