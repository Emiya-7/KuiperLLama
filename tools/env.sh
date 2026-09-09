# Toolchain for building this project on this machine. Source it, don't run it:
#   source tools/env.sh
#
# Why these are needed here rather than system-wide:
#   * cmake is not installed system-wide, so a user-local 3.28 is used.
#   * The distro's CUDA is 11.5, which cannot target this GPU (sm_89 is not a
#     valid -arch value before CUDA 11.8) and whose nvcc fails to parse GCC 11's
#     <functional>. CUDA 12.8 is installed under ~/.local/opt to fix both.
#   * The same CUDA 12.8 installation provides ncu and ncu-ui (Nsight Compute
#     2025.1.1), so no separate profiler installation or PATH entry is needed.
export CUDA_HOME="${CUDA_HOME:-$HOME/.local/opt/cuda-12.8}"
export PATH="$HOME/.local/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH}"

# Configure with:
#   cmake -B build -DUSE_CPM=ON -DQWEN35_SUPPORT=ON -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
#         -DCMAKE_CUDA_ARCHITECTURES=89 -DCUDAToolkit_ROOT=$CUDA_HOME
#
# CMAKE_CUDA_ARCHITECTURES is passed explicitly: cmake/cuda.cmake autodetects the
# installed GPU only when it is not already set.
