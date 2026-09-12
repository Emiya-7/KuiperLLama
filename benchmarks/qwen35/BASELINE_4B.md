# Qwen3.5-4B core-operator baseline

This document freezes the starting point for the CUDA optimization case study.
The scope is individual kernels, not whole-program scheduling:

- Nsight Compute for hardware-counter analysis;
- CUDA Events for uninstrumented before/after latency;
- BF16 matrix-vector projections and the Gated Delta Rule core;
- Qwen3.5-4B only. The 9B and INT8 work is intentionally postponed.

Nsight Systems is not part of this study. NCU is a measurement tool and does
not change inference performance; only subsequent kernel changes may do so.

## Fixed environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 SUPER, 12 GiB |
| Compute capability | 8.9 |
| CUDA toolkit/runtime | 12.8.93 / 12.8 |
| NCU | 2025.1.1 |
| Build | Release, `CMAKE_CUDA_ARCHITECTURES=89` |
| Profiled code commit | `304eea3` |
| NCU collection | `detailed` + SchedulerStats + WarpStateStats, kernel replay, cache control all |

The decode path supplies one activation vector to each projection. Although
the framework calls the operator `MatmulLayer`, these launches are technically
GEMV (`input[M] * weight[K,M] -> output[K]`), not multi-row GEMM. A real GEMM
case will be introduced with batched/chunked prefill and measured separately.

## Uninstrumented CUDA Event baseline

Matmul used cold cache, 5 warmups and 30 samples. The cache flush is ordered
before but excluded from each event interval. GDN also used 5 warmups and 30
samples; its 2 MiB recurrent state was reset before every sample.

| Case | Dimensions | Median | p95 | Effective bandwidth | Correctness |
|---|---:|---:|---:|---:|---|
| GDN step | nk=16, nv=32, kd=vd=128 | 0.016384 ms | 0.024256 ms | not applicable | CUDA/CPU max abs error 0 |
| GDN QKV projection | K=8192, M=2560 | 0.128000 ms | 0.402432 ms | 328.016 GB/s | checksum recorded |
| GDN gate projection | K=32, M=2560 | 0.006144 ms | 0.006144 ms | 28.354 GB/s | checksum recorded |
| GDN output projection | K=2560, M=4096 | 0.067584 ms | 0.086016 ms | 310.697 GB/s | checksum recorded |
| MLP up projection | K=9216, M=2560 | 0.143360 ms | 0.408576 ms | 329.471 GB/s | checksum recorded |
| MLP down projection | K=2560, M=9216 | 0.149488 ms | 0.701440 ms | 315.965 GB/s | checksum recorded |
| LM head | K=248320, M=2560 | 3.616256 ms | 4.277248 ms | 351.856 GB/s | checksum recorded |

The effective-bandwidth value is calculated from tensor sizes and event time;
it is not a hardware counter. Median is the primary latency comparator. The
large p95 outliers on several short kernels are retained rather than hidden and
must be re-sampled under identical WSL/GPU conditions after optimization.

## NCU baseline results

Counter permission was enabled and all seven reports were collected on
2026-09-11. NCU's own cache control was used. The benchmark's manual 96 MiB
cache-flush memset was disabled during NCU capture so dirty cache evictions
cannot pollute the target-kernel counters.

| Case | Duration (us) | DRAM (%) | DRAM (GB/s) | SM (%) | L2 hit (%) | Occupancy (%) | Waves/SM | Eligible (%) | Long scoreboard (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GDN step | 26.880 | 25.15 | 123.41 | 5.02 | 48.35 | 7.71 | 0.05 | 5.37 | 79.48 |
| GDN gate projection | 3.776 | 10.70 | 51.76 | 3.36 | 67.03 | 8.07 | 0.05 | 4.09 | 56.33 |
| GDN output projection | 59.680 | 90.18 | 443.02 | 24.16 | 14.65 | 97.85 | 3.81 | 12.07 | 77.66 |
| GDN QKV projection | 122.464 | 93.34 | 458.72 | 26.07 | 7.89 | 96.55 | 12.19 | 11.61 | 86.73 |
| LM head | 4009.280 | 91.36 | 343.58 | 31.54 | 15.42 | 98.39 | 369.52 | 14.24 | 88.15 |
| MLP down projection | 140.896 | 92.14 | 452.77 | 20.72 | 35.38 | 97.74 | 3.81 | 9.28 | 70.97 |
| MLP up projection | 136.224 | 93.95 | 461.62 | 26.37 | 7.72 | 96.97 | 13.71 | 15.15 | 73.14 |

Every kernel has zero local-memory load/store instructions. GDN uses 40
registers/thread; every BF16 GEMV uses 37. Tensor-pipe utilization is zero for
all BF16 GEMVs, confirming scalar FP32 accumulation after BF16 conversion
rather than Tensor Core MMA.

### Bottleneck diagnosis

1. **GDN is grid- and latency-limited.** It launches only 32 blocks on 56 SMs,
   reaches 7.71% occupancy and has an eligible warp in only 5.37% of scheduler
   cycles. Long-scoreboard dependencies account for 79.48% of warp cycles per
   issued instruction. Its 25.15% DRAM and 5.02% SM throughput show that neither
   compute nor global bandwidth is saturated.
2. **The 32-output GDN gate projection is also too small to fill the GPU.** Its
   32-block grid, 0.05 waves/SM and 3.776 us duration make launch amortization or
   fusion of the paired A/B projections more promising than generic GEMV tuning.
3. **Large BF16 GEMVs are DRAM-bound.** GDN/MLP/LM-head cases reach
   90.18-93.95% DRAM throughput while SM throughput remains 20.72-31.54%.
   Long-scoreboard stalls dominate and Tensor Core utilization is zero. Decode
   optimization should focus on weight-traffic efficiency, memory-level
   parallelism and shape-specific fusion. Tensor Core GEMM belongs to the later
   multi-token prefill path, not this N=1 path.
4. **LM head is the dominant individual decode projection.** Its NCU duration is
   4.009 ms and it sustains less bandwidth than medium projections, so it needs
   a separate large-output GEMV case rather than being represented by an MLP
   shape.

NCU kernel duration is valid for counter-level before/after comparison. The
approximately one-second JSON timings printed while NCU replays the application
are profiler overhead and are explicitly excluded.

## Artifacts and integrity

Persistent artifacts are stored outside Git at:

```text
/home/tuesday/workspace/icd/profiles/KuiperLLama/qwen35/4b-stage1-before-304eea3/
├── cuda-events/
└── ncu/
```

Each NCU case contains `.ncu-rep`, `.ncu-rep.sha256`, `.raw.csv`, and
`.command.txt`. The report hashes are:

| Report | SHA-256 |
|---|---|
| `gdn.ncu-rep` | `0e1946ecd9ee2f8c1bbadc964e11b47946a0cce02c47a0032110e67bc4c9e4d8` |
| `matmul-gdn-gate.ncu-rep` | `38b19b24eb43d530a4cd58130a99d49cd01faf51448db6ae1cb90ce6832939d1` |
| `matmul-gdn-out.ncu-rep` | `56641061886bcce13ed5ce649b53f2ef6f61e3e6a0cb796e8a3da087d8ffe202` |
| `matmul-gdn-qkv.ncu-rep` | `5da8a7be6f97ba16832feee0dd35c62bdbe32c6606ec75370937cdc03ad4f6bb` |
| `matmul-lm-head.ncu-rep` | `385ab42b1a2ad9f66a0688b305b2dd7dad4f869e060e9aec811df3d799371cd6` |
| `matmul-mlp-down.ncu-rep` | `d088354ffdc50264ad8b5b76fe172a4344cd98c52158eccc53e34c44833203d9` |
| `matmul-mlp-up.ncu-rep` | `26902c7282c7bc656a0ff2499523da62719537c75b95c8b29d4f3c0260036d5a` |

Open a report locally with, for example:

```bash
ncu-ui /home/tuesday/workspace/icd/profiles/KuiperLLama/qwen35/4b-stage1-before-304eea3/ncu/gdn.ncu-rep
```

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
