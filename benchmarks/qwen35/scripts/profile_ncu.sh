#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || "$3" != "--" ]]; then
  echo "Usage: $0 OUTPUT_PREFIX NVTX_INCLUDE -- COMMAND [ARGS...]" >&2
  echo "Example NVTX_INCLUDE: qwen35@matmul/" >&2
  exit 2
fi

OUTPUT_PREFIX="$1"
NVTX_INCLUDE="$2"
shift 3

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
source "$REPO_ROOT/tools/env.sh"

NCU_SET="${NCU_SET:-basic}"
NCU_REPLAY_MODE="${NCU_REPLAY_MODE:-application}"
OUTPUT_DIR="$(dirname -- "$OUTPUT_PREFIX")"
mkdir -p "$OUTPUT_DIR"

"$SCRIPT_DIR/capture_environment.sh" "$OUTPUT_DIR"
{
  printf 'NCU_SET=%q NCU_REPLAY_MODE=%q ' "$NCU_SET" "$NCU_REPLAY_MODE"
  printf '%q ' ncu --set "$NCU_SET" --replay-mode "$NCU_REPLAY_MODE" --nvtx \
    --nvtx-include "$NVTX_INCLUDE" --force-overwrite --export "$OUTPUT_PREFIX" "$@"
  printf '\n'
} >"${OUTPUT_PREFIX}.command.txt"
ncu \
  --set "$NCU_SET" \
  --replay-mode "$NCU_REPLAY_MODE" \
  --nvtx \
  --nvtx-include "$NVTX_INCLUDE" \
  --force-overwrite \
  --export "$OUTPUT_PREFIX" \
  "$@"

sha256sum "${OUTPUT_PREFIX}.ncu-rep" >"${OUTPUT_PREFIX}.ncu-rep.sha256"
echo "Wrote ${OUTPUT_PREFIX}.ncu-rep"
