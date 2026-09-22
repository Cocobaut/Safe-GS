#!/bin/bash
# Chay pipeline MILo GOC cho 1 scene Replica, dung y lenh Quickstart trong README cua MILo:
#   python train.py -s <sfm> -m <out> --imp_metric indoor --rasterizer radegs
#   python mesh_extract_sdf.py -s <sfm> -m <out> --rasterizer radegs
# (Replica la scene trong nha -> --imp_metric indoor; khong them --dense_gaussians / --decoupled_appearance.)
# Khac README duy nhat: --data_device cpu de 2000 anh khong len GPU (chi doi cho luu anh, khong doi thuat toan).
#
# Dau vao: dung lai SfM da sinh cho GOF (outputs/Replica/sfm/<scene>_gof) de GOF va MILo cung du lieu.
# Bo qua tung buoc neu da co ket qua. Cach goi: bash scripts/run_milo_one_scene.sh <ten_scene>

set -e
SCENE="$1"
cd "$(dirname "$0")/.."
export PYTHONNOUSERSITE=1
export DISPLAY=:0

MILO_PY=/home/ml4u/conda_envs/milo/bin/python
MILO_DIR="Based_Model/MILo/milo"
BASE="$(pwd)"

# Bien moi truong build/chay CUDA cho env milo (nvdiffrast JIT can nvcc)
export PATH="/home/ml4u/conda_envs/milo/bin:/usr/local/cuda-12.2/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.2
export CPATH=/usr/local/cuda-12/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda-12/targets/x86_64-linux/lib:/home/ml4u/conda_envs/milo/lib:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.9"
# Chong phan manh bo nho GPU (chi doi cach cap phat cua PyTorch, khong doi ket qua)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SFM_DIR="${BASE}/outputs/Replica/sfm/${SCENE}_gof"
MODEL_DIR="${BASE}/outputs/Replica/3dgs/${SCENE}_milo"
MESH_DIR="${BASE}/outputs/Replica/mesh/${SCENE}_milo"
IMAGES_DIR="${BASE}/../Replica 8 Scene/Replica/${SCENE}/results/image"
TRAIN_LOG="${BASE}/outputs/Replica/3dgs/log/${SCENE}_milo.log"
MESH_LOG="${BASE}/outputs/Replica/mesh/log/${SCENE}_milo.log"
mkdir -p "$MODEL_DIR" "$MESH_DIR" "${BASE}/outputs/Replica/3dgs/log" "${BASE}/outputs/Replica/mesh/log"

echo "======================================================================"
echo ">>> MILO PIPELINE: ${SCENE}  ($(date))"
echo "======================================================================"

[ -f "${SFM_DIR}/sparse/0/points3D.bin" ] || { echo "THIEU SfM ${SFM_DIR}, can chay GOF pipeline truoc"; exit 1; }

cd "$MILO_DIR"

# --- Buoc 1: Train MILo ---
if [ ! -f "${MODEL_DIR}/point_cloud/iteration_18000/point_cloud.ply" ]; then
    echo "--- [${SCENE}] MILo train.py ---"
    "$MILO_PY" -u train.py \
        -s "$SFM_DIR" -i "$IMAGES_DIR" -m "$MODEL_DIR" \
        --imp_metric indoor --rasterizer radegs \
        --data_device cpu > "$TRAIN_LOG" 2>&1
else
    echo "[SKIP] ${SCENE} da train xong."
fi

# --- Buoc 2: Trich mesh SDF ---
if ! find "$MODEL_DIR" "$MESH_DIR" -iname "*.ply" -path "*mesh*" 2>/dev/null | grep -q .; then
    echo "--- [${SCENE}] MILo mesh_extract_sdf.py ---"
    "$MILO_PY" -u mesh_extract_sdf.py \
        -s "$SFM_DIR" -i "$IMAGES_DIR" -m "$MODEL_DIR" \
        --rasterizer radegs --data_device cpu > "$MESH_LOG" 2>&1
else
    echo "[SKIP] ${SCENE} da co mesh."
fi

# --- Buoc 3: dua mesh vao outputs/Replica/mesh/<scene>_milo (ban goc van con trong thu muc model) ---
if [ -f "${MODEL_DIR}/mesh_learnable_sdf.ply" ]; then
    cp -n "${MODEL_DIR}/mesh_learnable_sdf.ply" "$MESH_DIR/"
else
    echo "CANH BAO: khong thay mesh_learnable_sdf.ply trong ${MODEL_DIR}"; exit 1
fi

echo ">>> XONG SCENE: ${SCENE}  ($(date))"
