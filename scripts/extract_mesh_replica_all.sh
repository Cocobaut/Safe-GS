#!/bin/bash
# Trich mesh GOF cho toan bo 8 scene Replica da train Gaussian xong.
# Chay TUAN TU (khong song song) de tranh qua tai may dung chung (da tung gay treo may).
#
# Chay:
#   cd "/media/ml4u/Extreme SSD/Safe-GS"
#   bash scripts/extract_mesh_replica_all.sh

cd "$(dirname "$0")/.."

SCENES="office0 office1 office2 office3 office4 room0 room1 room2"

for scene in $SCENES; do
    bash scripts/extract_mesh_replica_one_scene.sh "$scene" || echo "[WARN] ${scene} that bai, bo qua, tiep tuc scene sau."
done

echo ""
echo "======================================================================"
echo ">>> HOAN THANH TRICH MESH TOAN BO REPLICA"
echo "======================================================================"
