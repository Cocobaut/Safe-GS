#!/bin/bash
# Chay 1 scene Replica bang pipeline GOF goc:
#   1. Sinh COLMAP sparse model gia tu pose ground-truth (replica_to_colmap.py, env safe-gs)
#   2. Train Gaussian bang train.py cua GOF (env gof, --data_device cpu de tranh OOM anh)
# CHI train Gaussian, KHONG trich mesh (extract_mesh.py) - theo dung yeu cau.
#
# Cach goi: bash scripts/run_gof_replica_one_scene.sh <ten_scene>

set -e
SCENE="$1"
cd "$(dirname "$0")/.."   # Safe-GS/
export PYTHONNOUSERSITE=1
export DISPLAY=:0

SAFEGS_PY=/home/ml4u/conda_envs/safe-gs/bin/python
GOF_PY=/home/ml4u/conda_envs/gof/bin/python
GOF_DIR="Based_Model/GOF"

COLMAP_DIR="outputs/Replica/sfm/${SCENE}"
MODEL_DIR="outputs/Replica/3dgs/${SCENE}"
IMAGES_DIR="../Replica 8 Scene/Replica/${SCENE}/results/image"
LOG_DIR="outputs/Replica/3dgs/log"
LOG_FILE="${LOG_DIR}/${SCENE}.log"
mkdir -p "$COLMAP_DIR" "$MODEL_DIR" "$LOG_DIR"

CKPT="${MODEL_DIR}/point_cloud/iteration_30000/point_cloud.ply"
if [ -f "$CKPT" ]; then
    echo "[SKIP] ${SCENE} da co checkpoint 30000, bo qua."
    exit 0
fi


{
    echo "======================================================================"
    echo ">>> GOF REPLICA SCENE: ${SCENE}  ($(date))"
    echo "======================================================================"

    if [ ! -f "${COLMAP_DIR}/sparse/0/points3D.bin" ]; then
        echo "--- [${SCENE}] Sinh COLMAP sparse model gia tu ground-truth pose ---"
        "$SAFEGS_PY" scripts/replica_to_colmap.py --scene "$SCENE" --output_dir "$COLMAP_DIR"
    else
        echo "[SKIP] Da co sparse model gia, bo qua buoc sinh."
    fi

    echo "--- [${SCENE}] Train Gaussian bang GOF train.py (30000 iters, checkpoint 7k/30k) ---"
    cd "$GOF_DIR"
    "$GOF_PY" -u train.py \
        -s "$(pwd)/../../${COLMAP_DIR}" \
        -i "$(pwd)/../../${IMAGES_DIR}" \
        -m "$(pwd)/../../${MODEL_DIR}" \
        --data_device cpu

    echo ">>> XONG SCENE: ${SCENE}  ($(date))"
} > "$LOG_FILE" 2>&1

echo "[DONE] ${SCENE} -> log tai ${LOG_FILE}"
