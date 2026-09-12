#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 OUTPUT_DIRECTORY [CASE ...]" >&2
  echo "Cases: gdn gdn_qkv gdn_gate gdn_out mlp_up mlp_down lm_head" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
OUTPUT_DIR="$1"
shift
BENCH="$REPO_ROOT/build/demo/qwen35_bench"
MATMUL_KERNEL='.*matmul_kernel_cu_fp32bf16.*'
GDN_KERNEL='.*gated_delta_step_kernel.*'

if [[ $# -eq 0 ]]; then
  set -- gdn gdn_qkv gdn_gate gdn_out mlp_up mlp_down lm_head
fi

mkdir -p "$OUTPUT_DIR"

profile_matmul() {
  local name="$1"
  local input_size="$2"
  local output_size="$3"
  "$SCRIPT_DIR/profile_ncu.sh" "$OUTPUT_DIR/$name" "$MATMUL_KERNEL" -- \
    "$BENCH" --mode matmul --device cuda --dtype bf16 \
    --m "$input_size" --k "$output_size" --cache warm --warmup 0 --repeat 1
}

for case_name in "$@"; do
  case "$case_name" in
    gdn)
      "$SCRIPT_DIR/profile_ncu.sh" "$OUTPUT_DIR/gdn" "$GDN_KERNEL" -- \
        "$BENCH" --mode gdn --device cuda --warmup 0 --repeat 1
      ;;
    gdn_qkv) profile_matmul matmul-gdn-qkv 2560 8192 ;;
    gdn_gate) profile_matmul matmul-gdn-gate 2560 32 ;;
    gdn_out) profile_matmul matmul-gdn-out 4096 2560 ;;
    mlp_up) profile_matmul matmul-mlp-up 2560 9216 ;;
    mlp_down) profile_matmul matmul-mlp-down 9216 2560 ;;
    lm_head) profile_matmul matmul-lm-head 2560 248320 ;;
    *)
      echo "Unknown NCU case: $case_name" >&2
      exit 2
      ;;
  esac
done

echo "Wrote operator NCU baselines to $OUTPUT_DIR"
