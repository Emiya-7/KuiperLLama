#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 OUTPUT_DIRECTORY" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
OUTPUT_DIR="$1"
mkdir -p "$OUTPUT_DIR"

# Use the project's pinned CUDA 12.8 and CMake paths when they are not already
# overridden by the caller.
source "$REPO_ROOT/tools/env.sh"

{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "repo=$REPO_ROOT"
  echo "git_commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
  echo "git_branch=$(git -C "$REPO_ROOT" branch --show-current)"
  echo "git_status_begin"
  git -C "$REPO_ROOT" status --short
  echo "git_status_end"
  echo "uname=$(uname -a)"
  echo "cmake=$(cmake --version | head -n 1)"
  echo "nvcc_begin"
  nvcc --version
  echo "nvcc_end"
  echo "ncu_begin"
  ncu --version
  echo "ncu_end"
  echo "gpu_begin"
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,compute_cap,temperature.gpu,power.draw,clocks.sm,clocks.mem --format=csv,noheader
  echo "gpu_end"
} >"$OUTPUT_DIR/environment.txt"

echo "Wrote $OUTPUT_DIR/environment.txt"
