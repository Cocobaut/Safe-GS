#!/bin/bash
# Chay pipeline GOF (chi Gaussian, khong mesh) cho toan bo 8 scene Replica.
# Moi scene: sinh COLMAP sparse model gia tu ground-truth pose (khong can chay COLMAP
# that, vi vay khong bi cham nhu truoc), roi train.py cua GOF 30000 iterations
# (checkpoint 7k/30k tu dong, --data_device cpu de tranh OOM anh).
#
# Chay song song nhieu scene cung luc (mac dinh 2, vi moi scene giu ~18GB anh tren RAM
# CPU - khong phai VRAM). Doi so scene chay cung luc bang bien PARALLEL, vd:
#   PARALLEL=3 bash scripts/run_gof_replica_all.sh
#
# Dung giua chung (Ctrl+C) roi chay lai duoc - scene nao da co checkpoint 30000 se tu
# bo qua.
#
# Chay:
#   cd "/media/ml4u/Extreme SSD/Safe-GS"
#   bash scripts/run_gof_replica_all.sh
#
# Xem tien trinh tung scene: outputs/Replica/3dgs/log/<scene>.log

set -e
cd "$(dirname "$0")/.."

PARALLEL="${PARALLEL:-2}"
SCENES="office0 office1 office2 office3 office4 room0 room1 room2"

echo "Chay ${PARALLEL} scene song song. Tong $(echo $SCENES | wc -w) scene."
echo "Theo doi tien trinh: outputs/Replica/3dgs/log/<scene>.log"

echo ""

echo "$SCENES" | tr ' ' '\n' | xargs -P "$PARALLEL" -I{} bash scripts/run_gof_replica_one_scene.sh {}

echo ""
echo "======================================================================"
echo ">>> HOAN THANH TOAN BO GOF REPLICA (chi Gaussian)"
echo "======================================================================"
