#!/bin/bash
# Chay tuan tu (KHONG song song - moi scene da can ~50GB+ RAM luc trich mesh, chay
# cung luc se OOM) FULL pipeline chuan GOF cho 7 scene Replica con lai (office0 da xong).
#
# Dung lai giua chung (Ctrl+C hoac tat may) roi chay lai duoc - scene nao da xong (co
# checkpoint 30000 / co mesh) se tu bo qua buoc do.
#
# Chay:
#   cd "/media/ml4u/Extreme SSD/Safe-GS"
#   setsid nohup bash scripts/run_gof_full_pipeline_all_remaining.sh > outputs/Replica/gof_all_remaining.log 2>&1 < /dev/null &
#   disown
#
# Theo doi: tail -f outputs/Replica/gof_all_remaining.log
#           hoac tail -f outputs/Replica/3dgs/log/<scene>_gof.log / outputs/Replica/mesh/log/<scene>_gof.log

set -e
cd "$(dirname "$0")/.."

SCENES="office1 office2 office3 office4 room0 room1 room2"

for scene in $SCENES; do
    bash scripts/run_gof_full_pipeline_one_scene.sh "$scene"
done

echo ""
echo "======================================================================"
echo ">>> HOAN THANH TOAN BO 7 SCENE CON LAI (GOF full pipeline)"
echo "======================================================================"
