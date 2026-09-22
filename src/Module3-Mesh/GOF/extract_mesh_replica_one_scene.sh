#!/bin/bash
# Trich mesh GOF cho 1 scene Replica da train xong. Tu dong giam so camera (50 -> 25 ->
# 13 -> 7) neu bi CUDA OOM, vi cac scene co so luong Gaussian khac nhau nen VRAM can
# cũng khac nhau.
#
# Cach goi: bash scripts/extract_mesh_replica_one_scene.sh <ten_scene>

SCENE="$1"
cd "$(dirname "$0")/.."
export PYTHONNOUSERSITE=1
export DISPLAY=:0

SAFEGS_PY=/home/ml4u/conda_envs/safe-gs/bin/python
GOF_PY=/home/ml4u/conda_envs/gof/bin/python
GOF_DIR="Based_Model/GOF"

MESH_COLMAP_DIR="outputs/Replica/sfm/${SCENE}_mesh_tmp"
MODEL_DIR="outputs/Replica/3dgs/${SCENE}"
MESH_DIR="outputs/Replica/mesh/${SCENE}"
IMAGES_DIR="../Replica 8 Scene/Replica/${SCENE}/results/image"
LOG_DIR="outputs/Replica/mesh/log"
LOG_FILE="${LOG_DIR}/${SCENE}.log"

mkdir -p "$MESH_DIR" "$LOG_DIR"

if find "${MESH_DIR}" -iname "mesh_binary_search_*.ply" 2>/dev/null | grep -q .; then
    echo "[SKIP] ${SCENE} da co mesh trong outputs/Replica/mesh/${SCENE}, bo qua."
    exit 0
fi

: > "$LOG_FILE"
for stride in 40 80 160 320; do
    {
        echo "======================================================================"
        echo ">>> TRICH MESH SCENE: ${SCENE}  (camera_stride=${stride})  ($(date))"
        echo "======================================================================"

        rm -rf "$MESH_COLMAP_DIR"
        echo "--- [${SCENE}] Sinh COLMAP sparse model (camera_stride=${stride}) ---"
        "$SAFEGS_PY" scripts/replica_to_colmap.py --scene "$SCENE" --camera_stride "$stride" --output_dir "$MESH_COLMAP_DIR"

        echo "--- [${SCENE}] Chay extract_mesh.py ---"
        cd "$GOF_DIR"
        "$GOF_PY" -u extract_mesh.py \
            -s "$(pwd)/../../${MESH_COLMAP_DIR}" \
            -i "$(pwd)/../../${IMAGES_DIR}" \
            -m "$(pwd)/../../${MODEL_DIR}" \
            --data_device cpu
        RC=$?
        cd - > /dev/null

        # Don dep temp colmap mesh dir
        rm -rf "$MESH_COLMAP_DIR"

        if [ $RC -eq 0 ]; then
            # Chuyen file mesh sang outputs/Replica/mesh/${SCENE}
            if [ -d "${MODEL_DIR}/test/ours_30000/fusion" ]; then
                cp -r "${MODEL_DIR}/test/ours_30000/fusion"/* "${MESH_DIR}/" 2>/dev/null || true
            fi
        fi

        (exit $RC)
    } >> "$LOG_FILE" 2>&1
    RC=$?

    if [ $RC -eq 0 ]; then
        echo "[DONE] ${SCENE} (camera_stride=${stride}) -> log tai ${LOG_FILE}"
        exit 0
    fi


    if grep -q "CUDA out of memory" "$LOG_FILE"; then
        echo "[RETRY] ${SCENE} bi OOM voi camera_stride=${stride}, giam view thu tiep..."
        continue
    else
        echo "[ERROR] ${SCENE} loi khac (khong phai OOM) -> xem ${LOG_FILE}"
        exit 1
    fi
done

echo "[FAIL] ${SCENE} van OOM du da giam xuong stride=320, can xem lai thu cong."
exit 1
