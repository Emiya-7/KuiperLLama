#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 OUTPUT_DIRECTORY [WARMUP=5] [REPEAT=30]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
OUTPUT_DIR="$1"
WARMUP="${2:-5}"
REPEAT="${3:-30}"
CACHE_MODES="${CACHE_MODES:-cold}"
BENCH="$REPO_ROOT/build/demo/qwen35_bench"
SHAPES="$REPO_ROOT/benchmarks/qwen35/configs/qwen35_4b_matmul_shapes.csv"

source "$REPO_ROOT/tools/env.sh"
mkdir -p "$OUTPUT_DIR"
"$SCRIPT_DIR/capture_environment.sh" "$OUTPUT_DIR"

while IFS=, read -r name input_size output_size dtype workload; do
  if [[ "$name" == "name" ]]; then
    continue
  fi
  for cache in $CACHE_MODES; do
    echo "benchmarking $name ($cache cache)"
    "$BENCH" \
      --mode matmul --device cuda --dtype "$dtype" \
      --m "$input_size" --k "$output_size" --cache "$cache" \
      --warmup "$WARMUP" --repeat "$REPEAT" \
      --output "$OUTPUT_DIR/matmul-${name}-${cache}.json" >/dev/null
  done
done <"$SHAPES"

echo "Wrote matmul baselines to $OUTPUT_DIR"
