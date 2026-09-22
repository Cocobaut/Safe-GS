#!/bin/bash
# Chay FULL pipeline "chuan GOF" cho 1 scene Replica (SfM mat do cao -> train.py GOF goc ->
# extract_mesh.py GOF goc, dung DU camera that, khong stride) - dung y het cach da lam
# thanh cong cho office0_gof.
#
# Yeu cau: scene/gaussian_model.py cua GOF da duoc va (chunk get_frustum_mask + bat lai
# opacity filter trong get_tetra_points) - khong sua gi them o day.
#
# Bo qua tung buoc neu da co san ket qua (an toan de chay lai/resume).
#
# Cach goi: bash scripts/run_gof_full_pipeline_one_scene.sh <ten_scene>

set -e
SCENE="$1"
cd "$(dirname "$0")/.."
export PYTHONNOUSERSITE=1
export DISPLAY=:0

SAFEGS_PY=/home/ml4u/conda_envs/safe-gs/bin/python
GOF_PY=/home/ml4u/conda_envs/gof/bin/python
GOF_DIR="Based_Model/GOF"
BASE="$(pwd)"

SFM_DIR="outputs/Replica/sfm/${SCENE}_gof"
MODEL_DIR="outputs/Replica/3dgs/${SCENE}_gof"
MESH_DIR="outputs/Replica/mesh/${SCENE}_gof"
IMAGES_DIR="../Replica 8 Scene/Replica/${SCENE}/results/image"
TRAIN_LOG="outputs/Replica/3dgs/log/${SCENE}_gof.log"
MESH_LOG="outputs/Replica/mesh/log/${SCENE}_gof.log"
mkdir -p "$SFM_DIR" "$MODEL_DIR" "$MESH_DIR" "outputs/Replica/3dgs/log" "outputs/Replica/mesh/log"

echo "======================================================================"
echo ">>> GOF FULL PIPELINE: ${SCENE}  ($(date))"
echo "======================================================================"

# --- Buoc 1: SfM mat do cao (giong het office0_gof: stride_frames=15, stride_pixels=5) ---
if [ ! -f "${SFM_DIR}/sparse/0/points3D.bin" ]; then
    echo "--- [${SCENE}] Sinh SfM mat do cao ---"
    "$SAFEGS_PY" scripts/replica_to_colmap.py --scene "$SCENE" \
        --output_dir "$SFM_DIR" \
        --point_cloud_stride_frames 15 --point_cloud_stride_pixels 5
else
    echo "[SKIP] ${SCENE} da co SfM, bo qua."
fi

# --- Buoc 2: Train bang train.py goc GOF ---
CKPT="${MODEL_DIR}/point_cloud/iteration_30000/point_cloud.ply"
if [ ! -f "$CKPT" ]; then
    echo "--- [${SCENE}] Train Gaussian bang train.py GOF (30000 iter) ---"
    cd "$GOF_DIR"
    "$GOF_PY" -u train.py \
        -s "${BASE}/${SFM_DIR}" \
        -i "${BASE}/${IMAGES_DIR}" \
        -m "${BASE}/${MODEL_DIR}" \
        --data_device cpu > "${BASE}/${TRAIN_LOG}" 2>&1
    cd "$BASE"
else
    echo "[SKIP] ${SCENE} da co checkpoint 30000, bo qua train."
fi

# --- Buoc 3: Trich mesh full camera (extract_mesh.py goc GOF, da va OOM) ---
if find "$MESH_DIR" -iname "mesh_binary_search_*.ply" 2>/dev/null | grep -q .; then
    echo "[SKIP] ${SCENE} da co mesh, bo qua."
else
    echo "--- [${SCENE}] Trich mesh full camera ---"
    cd "$GOF_DIR"
    "$GOF_PY" -u extract_mesh.py \
        -s "${BASE}/${SFM_DIR}" \
        -i "${BASE}/${IMAGES_DIR}" \
        -m "${BASE}/${MODEL_DIR}" \
        --data_device cpu > "${BASE}/${MESH_LOG}" 2>&1
    cd "$BASE"

    FUSION_DIR="${MODEL_DIR}/test/ours_30000/fusion"
    if [ -d "$FUSION_DIR" ]; then
        cp "${FUSION_DIR}"/mesh_binary_search_*.ply "$MESH_DIR/" 2>/dev/null || true
    fi
fi

echo ">>> XONG SCENE: ${SCENE}  ($(date))"
