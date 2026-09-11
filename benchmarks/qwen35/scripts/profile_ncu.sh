#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || "$3" != "--" ]]; then
  echo "Usage: $0 OUTPUT_PREFIX KERNEL_REGEX -- COMMAND [ARGS...]" >&2
  echo "Example KERNEL_REGEX: .*matmul_kernel_cu_fp32bf16.*" >&2
  exit 2
fi

OUTPUT_PREFIX="$1"
KERNEL_REGEX="$2"
shift 3

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
source "$REPO_ROOT/tools/env.sh"

NCU_SET="${NCU_SET:-detailed}"
NCU_REPLAY_MODE="${NCU_REPLAY_MODE:-kernel}"
NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-1}"
OUTPUT_DIR="$(dirname -- "$OUTPUT_PREFIX")"
mkdir -p "$OUTPUT_DIR"

"$SCRIPT_DIR/capture_environment.sh" "$OUTPUT_DIR"
{
  printf 'NCU_SET=%q NCU_REPLAY_MODE=%q NCU_LAUNCH_COUNT=%q ' \
    "$NCU_SET" "$NCU_REPLAY_MODE" "$NCU_LAUNCH_COUNT"
  printf '%q ' ncu --set "$NCU_SET" --section SchedulerStats --section WarpStateStats \
    --cache-control all --replay-mode "$NCU_REPLAY_MODE" \
    --kernel-name-base function --kernel-name "regex:$KERNEL_REGEX" \
    --launch-count "$NCU_LAUNCH_COUNT" --force-overwrite --export "$OUTPUT_PREFIX" "$@"
  printf '\n'
} >"${OUTPUT_PREFIX}.command.txt"
ncu \
  --set "$NCU_SET" \
  --section SchedulerStats \
  --section WarpStateStats \
  --cache-control all \
  --replay-mode "$NCU_REPLAY_MODE" \
  --kernel-name-base function \
  --kernel-name "regex:$KERNEL_REGEX" \
  --launch-count "$NCU_LAUNCH_COUNT" \
  --force-overwrite \
  --export "$OUTPUT_PREFIX" \
  "$@"

if [[ ! -s "${OUTPUT_PREFIX}.ncu-rep" ]]; then
  echo "NCU did not create ${OUTPUT_PREFIX}.ncu-rep." >&2
  echo "If ERR_NVGPUCTRPERM was printed, enable GPU performance counters on the Windows host." >&2
  exit 3
fi

sha256sum "${OUTPUT_PREFIX}.ncu-rep" >"${OUTPUT_PREFIX}.ncu-rep.sha256"
ncu --import "${OUTPUT_PREFIX}.ncu-rep" --page raw --csv >"${OUTPUT_PREFIX}.raw.csv"
echo "Wrote ${OUTPUT_PREFIX}.ncu-rep"
