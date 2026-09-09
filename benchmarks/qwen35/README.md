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
  -DKUIPER_ENABLE_NVTX=ON \
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
  --m 2560 --k 4096 --cache cold --warmup 5 --repeat 30 \
  --output /tmp/qwen35-matmul-baseline.json
```

Matmul defaults to `--cache cold`: before each timed launch it touches a buffer
twice the reported L2 size on the same stream, then records the start event.
The flush is ordered before, but excluded from, the measurement. This models
decode's stream through matrices much larger than L2. `--cache warm` is kept as
a diagnostic mode, but very short warm-cache kernels can be distorted by host
submission and WSL/WDDM scheduling. It is not included in the primary baseline.

Run every registered 4B shape in the primary cold-cache mode with:

```bash
benchmarks/qwen35/scripts/run_matmul_baseline.sh \
  /tmp/qwen35-matmul-baseline 5 30
```

Set `CACHE_MODES="cold warm"` only when collecting an explicitly separate cache
study.

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

## Nsight Compute

CUDA 12.8 in `tools/env.sh` includes Nsight Compute CLI. Verify the selected
installation before profiling:

```bash
source tools/env.sh
ncu --version
ncu --list-sets
ncu --list-sections
```

On the current RTX 4070 SUPER workstation this resolves to Nsight Compute
2025.1.1 (`ncu` and `ncu-ui`). The CLI and section discovery work, but the first
counter collection returned `ERR_NVGPUCTRPERM`; the Windows-host permission
step below must be completed before baseline reports can be collected.

The build enables NVTX annotations by default. The `qwen35` domain contains
top-level `prefill`, `decode`, `end-to-end`, and `matmul/...` ranges. Model
ranges are nested by `layer_NN/gdn|full` and then by operator, for example
`matmul.gdn_qkv`, `delta.gdn`, and `matmul.mlp_down`.

Use application replay for the stateful model benchmark. The helper captures
the environment, exports an NCU report, and creates its SHA-256 file:

```bash
benchmarks/qwen35/scripts/profile_ncu.sh \
  /tmp/qwen35-ncu/matmul-baseline 'qwen35@matmul/' -- \
  ./build/demo/qwen35_bench \
    --mode matmul --device cuda --dtype bf16 \
    --m 2560 --k 4096 --cache cold --warmup 5 --repeat 5

benchmarks/qwen35/scripts/profile_ncu.sh \
  /tmp/qwen35-ncu/decode-baseline 'qwen35@decode/' -- \
  ./build/demo/qwen35_bench \
    --mode decode --device cuda \
    --checkpoint /path/to/qwen35_4b_stage4_bf16.bin \
    --tokenizer /path/to/Qwen3.5-4B/tokenizer.json \
    --prompt-length 128 --decode-steps 4 --warmup 1 --repeat 1
```

Override `NCU_SET` and `NCU_REPLAY_MODE` only for targeted investigations. For
example, an isolated stateless matmul can use `NCU_SET=detailed` and
`NCU_REPLAY_MODE=kernel`; avoid that combination over an entire model run.

If NCU reports `ERR_NVGPUCTRPERM` under WSL2, enable access on the Windows host:

1. Open NVIDIA Control Panel as administrator.
2. Enable **Desktop > Enable Developer Settings**.
3. Open **Developer > Manage GPU Performance Counters**.
4. Select **Allow access to the GPU performance counters to all users**.
5. Restart WSL with `wsl --shutdown`, open it again, and rerun the command.

The JSON from an NCU-instrumented run is not a latency result: replay and
counter collection intentionally perturb execution. Use the standalone CUDA
Event benchmark for latency and the `.ncu-rep` only for bottleneck evidence.
