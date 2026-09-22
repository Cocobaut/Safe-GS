#!/bin/bash
# Chay tuan tu (KHONG song song - moi scene dung ~40GB RAM) pipeline MILo goc cho 7 scene Replica con lai
# (office0 da xong). Tu bo qua buoc da co ket qua; 1 scene loi thi ghi lai roi chay tiep scene sau.
#
# Chay:
#   cd "/media/ml4u/Extreme SSD/Safe-GS"
#   setsid nohup bash scripts/run_milo_all_remaining.sh > outputs/Replica/milo_all_remaining.log 2>&1 < /dev/null &
#   disown
# Theo doi: tail -f outputs/Replica/milo_all_remaining.log ; log tung scene: outputs/Replica/3dgs/log/<scene>_milo.log

cd "$(dirname "$0")/.."
SCENES="office1 office2 office3 office4 room0 room1 room2"
FAILED=""

for scene in $SCENES; do
    if ! bash scripts/run_milo_one_scene.sh "$scene"; then
        echo "!!! SCENE LOI: ${scene} ($(date)) - xem outputs/Replica/3dgs/log/${scene}_milo.log va outputs/Replica/mesh/log/${scene}_milo.log"
        FAILED="$FAILED $scene"
    fi
done

echo ""
echo "======================================================================"
echo ">>> HOAN THANH MILO 7 SCENE. Scene loi:${FAILED:- khong co}"
echo "======================================================================"
