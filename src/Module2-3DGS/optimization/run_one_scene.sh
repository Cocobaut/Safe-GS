#!/bin/bash
# Chạy COLMAP + train Scene-GS cho ĐÚNG 1 scene. Dùng nội bộ bởi run_all_scenes.sh
# (chạy song song nhiều scene qua xargs -P), có thể tự gọi riêng lẻ nếu muốn.
#
# Cách gọi: bash scripts/run_one_scene.sh "<tên_scene>|<đường_dẫn_ảnh>"

set -e
cd "$(dirname "$0")/.."

IFS='|' read -r name images_dir <<< "$1"

export PYTHONNOUSERSITE=1
export DISPLAY=:0
PYTHON=/home/ml4u/conda_envs/safe-gs/bin/python

if [[ "$name" == dtu_* ]]; then
    dataset="DTU"
    scene_name="$name"
elif [[ "$name" == replica_* ]]; then
    dataset="Replica"
    scene_name="${name#replica_}"
else
    dataset="DTU"
    scene_name="$name"
fi

sfm_dir="outputs/${dataset}/sfm/${scene_name}"
three_dir="outputs/${dataset}/3dgs/${scene_name}"
log_dir="outputs/${dataset}/3dgs/log"
log_file="${log_dir}/${scene_name}.log"
ckpt="${three_dir}/scene_gs_30000.ply"

mkdir -p "$sfm_dir" "$three_dir" "$log_dir"

if [ -f "$ckpt" ]; then
    echo "[SKIP] ${name} đã có checkpoint 30000, bỏ qua."
    exit 0
fi

{
    echo "======================================================================"
    echo ">>> SCENE: ${name}  ($(date))"
    echo "======================================================================"

    if [ ! -d "${sfm_dir}/sparse/0" ]; then
        echo "--- [${name}] Chạy COLMAP (SfM) ---"
        "$PYTHON" scripts/01_run_perception.py \
            --images_dir "$images_dir" \
            --workspace_dir "$sfm_dir" \
            --skip_seg
    else
        echo "[SKIP] Đã có output SfM, bỏ qua COLMAP."
    fi

    echo "--- [${name}] Train Scene-GS (30000 iterations) ---"
    "$PYTHON" scripts/02_run_scene_gs.py \
        --sfm_dir "${sfm_dir}/sparse/0" \
        --raw_image_dir "$images_dir" \
        --workspace_dir "$three_dir"

    echo ">>> XONG SCENE: ${name}  ($(date))"
} > "$log_file" 2>&1

echo "[DONE] ${name} -> xem log tại ${log_file}"

