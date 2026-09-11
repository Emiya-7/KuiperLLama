# Qwen3.5 CUDA performance baseline

`qwen35_bench` is the reproducible timing harness used before and after CUDA
kernel optimizations. It deliberately does not use `qwen35_trace`: trace mode
synchronizes and copies every selected hidden state, which changes the workload.

The optimization case study is deliberately operator-scoped. Nsight Compute is
used on a small set of GDN and BF16 matrix-vector kernels; Nsight Systems and
whole-program timeline tuning are outside this work. Standalone model timing is
retained only as an optional regression check.

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

## GDN microbenchmark

The standalone GDN mode exercises exactly one `gated_delta_step_kernel` launch
with the Qwen3.5-4B dimensions: 16 K heads, 32 V heads, 128-wide K/V heads, and
a 2 MiB recurrent state. It does not load model weights. State is reset before
every sample, and CUDA output/state are checked against the CPU implementation.

```bash
./build/demo/qwen35_bench \
  --mode gdn --device cuda --warmup 5 --repeat 30 \
  --output /tmp/qwen35-gdn-baseline.json
```

Use `--warmup 0 --repeat 1` under NCU so exactly one target kernel is collected.

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

## Operator-focused Nsight Compute

CUDA 12.8 in `tools/env.sh` includes Nsight Compute CLI. Verify the selected
installation before profiling:

```bash
source tools/env.sh
ncu --version
ncu --list-sets
ncu --list-sections
```

On the current RTX 4070 SUPER workstation this resolves to Nsight Compute
2025.1.1 (`ncu` and `ncu-ui`). The initial collection returned
`ERR_NVGPUCTRPERM`; Windows-host permission has since been enabled and the
2026-09-11 operator baseline was collected successfully.

The profiling helper filters by CUDA kernel function, profiles one matching
launch, captures the environment, exports the `.ncu-rep`, writes a SHA-256, and
also exports a reviewable raw CSV. Kernel replay is safe here because each
microbenchmark has deterministic inputs and NCU restores memory modified by a
replayed launch. NCU uses `--cache-control all`; the matmul command uses its
`warm` setting only to disable the benchmark's own 96 MiB memset. NCU therefore
profiles a cold-cache kernel without dirty cache-flush traffic contaminating
the target kernel's DRAM counters.

```bash
benchmarks/qwen35/scripts/profile_ncu.sh \
  /tmp/qwen35-ncu/matmul-gdn-qkv '.*matmul_kernel_cu_fp32bf16.*' -- \
  ./build/demo/qwen35_bench \
    --mode matmul --device cuda --dtype bf16 \
    --m 2560 --k 8192 --cache warm --warmup 0 --repeat 1

benchmarks/qwen35/scripts/profile_ncu.sh \
  /tmp/qwen35-ncu/gdn '.*gated_delta_step_kernel.*' -- \
  ./build/demo/qwen35_bench --mode gdn --device cuda --warmup 0 --repeat 1
```

The curated 4B suite can run all cases or selected cases:

```bash
benchmarks/qwen35/scripts/run_ncu_operator_baseline.sh /tmp/qwen35-ncu-baseline
benchmarks/qwen35/scripts/run_ncu_operator_baseline.sh \
  /tmp/qwen35-ncu-baseline gdn gdn_qkv mlp_up mlp_down

benchmarks/qwen35/scripts/summarize_ncu.py /tmp/qwen35-ncu-baseline
```

The cases intentionally cover different bottleneck regimes:

| Case | Shape or state | Reason |
|---|---:|---|
| `gdn` | 32 × 128 × 128 FP32 state | recurrent update and state traffic |
| `gdn_qkv` | K=8192, M=2560 | large GDN projection |
| `gdn_gate` | K=32, M=2560 | launch/under-utilization boundary |
| `gdn_out` | K=2560, M=4096 | GDN output projection |
| `mlp_up` | K=9216, M=2560 | large expansion projection |
| `mlp_down` | K=2560, M=9216 | wide reduction projection |
| `lm_head` | K=248320, M=2560 | bandwidth-heavy vocabulary head |

The default `detailed` set records Speed of Light, occupancy, compute workload,
memory workload, and source counters. The helper also adds Scheduler Statistics
and Warp State Statistics so issue efficiency and stall reasons are available
in every before/after report. Override `NCU_SET` only for a focused follow-up;
the `full` set is too expensive for routine collection.

If NCU reports `ERR_NVGPUCTRPERM` under WSL2, enable access on the Windows host:

1. Open NVIDIA Control Panel as administrator.
2. Enable **Desktop > Enable Developer Settings**.
3. Open **Developer > Manage GPU Performance Counters**.
4. Select **Allow access to the GPU performance counters to all users**.
5. Restart WSL with `wsl --shutdown`, open it again, and rerun the command.

The JSON from an NCU-instrumented run is not a latency result: replay and
counter collection intentionally perturb execution. Use the standalone CUDA
Event benchmark for latency and the `.ncu-rep` only for bottleneck evidence.
