#!/bin/bash
# Build tetra_triangulation cho MILo (cmake . ; make ; pip install -e .) - dung buoc README.
# Bien moi truong: CPATH cho cuda_runtime.h, CMAKE_POLICY_VERSION_MINIMUM cho pybind11 FetchContent cu (giong luc build GOF).
set -x
export PYTHONNOUSERSITE=1
ENV=/home/ml4u/conda_envs/milo
export PATH="$ENV/bin:/usr/local/cuda-12.2/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.2
export CPATH=/usr/local/cuda-12/targets/x86_64-linux/include:$ENV/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda-12/targets/x86_64-linux/lib:$ENV/lib:$LD_LIBRARY_PATH
export CMAKE_POLICY_VERSION_MINIMUM=3.5
export MAX_JOBS=4
cd "/media/ml4u/Extreme SSD/Safe-GS/Based_Model/MILo/submodules/tetra_triangulation"
cmake . && make -j4 && "$ENV/bin/python" -m pip install -e . && echo "===== TETRA DONE ====="
