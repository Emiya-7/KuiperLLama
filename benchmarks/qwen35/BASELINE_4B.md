# Qwen3.5-4B core-operator baseline

This document is the frozen starting point for the CUDA optimization case
study. The scope is individual kernels, not whole-program scheduling:

- Nsight Compute for hardware-counter analysis;
- CUDA Events for uninstrumented latency before/after comparison;
- BF16 matrix-vector projections and the Gated Delta Rule core;
- Qwen3.5-4B only. The 9B and INT8 work is intentionally postponed.

Nsight Systems is not part of this study. NCU itself is a measurement tool and
does not change inference performance; only subsequent kernel changes may do
so.

## Fixed environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 SUPER, 12 GiB |
| Compute capability | 8.9 |
| CUDA toolkit/runtime | 12.8.93 / 12.8 |
| NCU | 2025.1.1 |
| Build | Release, `CMAKE_CUDA_ARCHITECTURES=89` |
| Matmul baseline commit | `f42bda0` |
| GDN harness commit | `96fadda` |

The decode path supplies one activation vector to each projection. Although
the framework calls the operator `MatmulLayer`, these measured launches are
technically GEMV (`input[M] * weight[K,M] -> output[K]`), not multi-row GEMM.
A real GEMM case will be introduced with batched/chunked prefill and measured
separately rather than conflated with this baseline.

## Uninstrumented CUDA Event baseline

Matmul used cold cache, 5 warmups, 30 samples, and reports the median. The cache
flush is ordered before but excluded from each event interval. GDN used 2
warmups and 10 samples; its 2 MiB recurrent state was reset before each sample.

| Case | Dimensions | Median | Effective bandwidth | Correctness |
|---|---:|---:|---:|---|
| GDN step | nk=16, nv=32, kd=vd=128 | 0.021504 ms | not yet counter-derived | CUDA/CPU max abs error 0 |
| GDN QKV projection | K=8192, M=2560 | 0.128000 ms | 328.016 GB/s | checksum recorded |
| GDN gate projection | K=32, M=2560 | 0.006128 ms | 28.428 GB/s | checksum recorded |
| GDN output projection | K=2560, M=4096 | 0.067584 ms | 310.697 GB/s | checksum recorded |
| MLP up/gate projection | K=9216, M=2560 | 0.143360 ms | 329.471 GB/s | checksum recorded |
| MLP down projection | K=2560, M=9216 | 0.148480 ms | 318.110 GB/s | checksum recorded |
| LM head | K=248320, M=2560 | 3.315200 ms | 383.808 GB/s | checksum recorded |

The effective-bandwidth number is a model from tensor byte sizes and event
time, not a hardware-counter reading. It is useful for consistent before/after
screening, but the final bottleneck claims must come from NCU counters.

## NCU baseline status

The CUDA 12.8 NCU CLI is installed, kernel filtering was verified to match
`gated_delta_step_kernel`, but counter collection currently stops with:

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
```

No NCU metric, report, or NCU-instrumented latency is accepted as a baseline at
this point. In particular, profiler-perturbed timing printed by the benchmark
must not be copied into comparison tables.

To unblock collection on the Windows host:

1. Run NVIDIA Control Panel as administrator.
2. Enable **Desktop > Enable Developer Settings**.
3. Open **Developer > Manage GPU Performance Counters**.
4. Select **Allow access to the GPU performance counters to all users**.
5. Run `wsl --shutdown`, reopen WSL, and rerun the commands below.

```bash
cd /home/tuesday/workspace/cuda_project/KuiperLLama
source tools/env.sh

benchmarks/qwen35/scripts/run_ncu_operator_baseline.sh \
  /tmp/qwen35-ncu-before gdn gdn_qkv gdn_gate gdn_out mlp_up mlp_down lm_head
```

Each successful case must contain `.ncu-rep`, `.ncu-rep.sha256`, `.raw.csv`,
`.command.txt`, and `environment.txt`. Reports stay outside Git; the final
optimization report will record their hashes and summarize the following
before/after metrics:

- kernel duration and achieved memory/compute throughput;
- DRAM/L1/L2 traffic and hit rates;
- achieved occupancy and its register/shared-memory/block limits;
- warp issue efficiency and dominant stall reasons;
- instruction mix, including Tensor Core use where applicable.

## Comparison rules

An optimization is accepted only when all of these hold:

1. The same 4B dimensions, input initialization, cache policy, GPU clocks/power
   conditions, build type, and measurement commands are used.
2. Existing tests pass and the operator-specific CPU/CUDA correctness check
   remains within its established tolerance.
3. Standalone CUDA Event latency improves over repeated runs; median and p95 are
   both reported.
4. NCU counters explain the improvement. A latency change without a plausible
   counter-level cause is not presented as an operator optimization result.
