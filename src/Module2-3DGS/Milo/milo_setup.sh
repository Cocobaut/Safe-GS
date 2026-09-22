#!/bin/bash
# Dung env milo theo dung cac buoc install.py cua MILo (nhanh CUDA 12.1 - README noi chi test 11.8).
# Torch 2.3.1 da cai bang conda truoc do. Script nay cai requirements + build submodules.
# Chay: setsid nohup bash tmp/milo_setup.sh > tmp/milo_setup.log 2>&1 < /dev/null & disown
set -x
export PYTHONNOUSERSITE=1
ENV=/home/ml4u/conda_envs/milo
export PATH="$ENV/bin:/usr/local/cuda-12.2/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.2
export CPATH=/usr/local/cuda-12/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda-12/targets/x86_64-linux/lib:$ENV/lib:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.9"
export MAX_JOBS=4

cd "/media/ml4u/Extreme SSD/Safe-GS/Based_Model/MILo"
PIP="$ENV/bin/python -m pip"

$PIP install -r requirements.txt || exit 1

for sub in diff-gaussian-rasterization_ms diff-gaussian-rasterization diff-gaussian-rasterization_gof simple-knn fused-ssim; do
    echo "===== building $sub ====="
    $PIP install "submodules/$sub" || { echo "FAILED: $sub"; exit 1; }
done
echo "===== ALL PIP SUBMODULES DONE ====="
