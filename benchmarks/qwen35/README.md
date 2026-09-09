# Qwen3.5 CUDA performance baseline

`qwen35_bench` is the reproducible timing harness used before and after CUDA
kernel optimizations. It deliberately does not use `qwen35_trace`: trace mode
synchronizes and copies every selected hidden state, which changes the workload.

The current model prefill implementation is token-by-token. Results therefore
label it `token-by-token-baseline`; a later optimization will add a batched
prefill implementation without changing the meaning of this baseline.

## Build

```bash
source tools/env.sh
cmake -S . -B build -DUSE_CPM=ON -DQWEN35_SUPPORT=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCUDAToolkit_ROOT="$CUDA_HOME"
cmake --build build --target qwen35_bench -j2
```

## Matmul microbenchmark

The notation is `input[M] * weight[K,M] -> output[K]`. The default shape is the
Qwen3.5-4B `in_proj_z` matrix (`K=4096`, `M=2560`). Initialization and host to
device copies are outside the timed region.

```bash
./build/demo/qwen35_bench \
  --mode matmul --device cuda --dtype bf16 \
  --m 2560 --k 4096 --warmup 5 --repeat 30 \
  --output /tmp/qwen35-matmul-baseline.json
```

## Model benchmark

Model loading and weight upload are reported separately as `load_ms`. Every
sample starts after `reset_state()` and uses deterministic token IDs. Decode
samples first rebuild the same prompt state outside the timed range. The
end-to-end total range contains no synchronization between prefill and decode;
phase timings are gathered in a separate reset run.

```bash
./build/demo/qwen35_bench \
  --mode prefill --device cuda \
  --checkpoint /path/to/qwen35_4b_stage4_bf16.bin \
  --tokenizer /path/to/Qwen3.5-4B/tokenizer.json \
  --prompt-length 128 --warmup 1 --repeat 5 \
  --output /tmp/qwen35-prefill-t128-baseline.json

./build/demo/qwen35_bench \
  --mode decode --device cuda \
  --checkpoint /path/to/qwen35_4b_stage4_bf16.bin \
  --tokenizer /path/to/Qwen3.5-4B/tokenizer.json \
  --prompt-length 128 --decode-steps 32 --warmup 1 --repeat 5 \
  --output /tmp/qwen35-decode-t128-baseline.json
```

Keep generated JSON and Nsight Compute reports outside Git. When publishing a
result, record the Git commit, command, environment manifest, summary tables,
and SHA-256 of the corresponding `.ncu-rep` artifact.
