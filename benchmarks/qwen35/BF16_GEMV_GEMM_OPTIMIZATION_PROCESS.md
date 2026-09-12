# BF16 GEMV/GEMM 优化过程

本文独立记录 Qwen3.5-4B BF16 线性层的瓶颈分析、优化假设、具体实现、失败实验和
前后实测结果，供代码复盘和面试讲解使用。文中严格区分已经由 CUDA Event/NCU
验证的事实与仍待实验的方案，不用理论带宽或单次最小值代替真实加速结果。

当前状态：**GEMV R2 已实现，真正的多 token GEMM 尚未开始。** R1 修复了 CUDA
BF16 路径忽略 `scale` 的接口语义，并用奇数 M 覆盖向量化尾部；R2 为偶数 M 增加
BF16x2/FP32x2 成对读取、双 accumulator、直接 CUB reduction，并通过 128/256/512
threads 实测选择 256。当前 decode 和
token-by-token prefill 都只向 `MatmulLayer` 传入一个 FP32 activation vector，执行的是

```text
x[M] * W[K, M]^T -> y[K]
```

也就是 FP32 input、BF16 weight、FP32 accumulate/output 的 GEMV。只有并行 prefill
把多个 token 组成矩阵后，计算才会成为 GEMM；在此之前不能把 Tensor Core GEMM 的
结论套到当前 N=1 workload 上。

## 1. 面试讲解主线

当前可以先这样描述问题和优化方向；最终数字会在各轮实验完成后更新：

> 我先把框架中的 MatmulLayer 按实际 workload 拆成 decode GEMV 和后续 prefill
> GEMM。优化前 BF16 GEMV 每个输出行启动一个 128-thread block，每个线程以标量方式
> 读取 BF16 权重、转 FP32 并累加，最后用 CUB 做 block reduction。NCU 显示中大型
> 投影的 DRAM throughput 达 90.18%–93.95%，SM throughput 只有 20.72%–31.54%，
> long-scoreboard 占 70.97%–88.15%，说明主要受权重流量和访存等待限制；但 32 输出
> 的 gate 投影只有 32 blocks、0.05 waves/SM，是 launch/并行度受限。于是我不会用同一
> 个 kernel 策略解释所有形状：中大型 GEMV 优先尝试 BF16x2/FP32x2 向量读取、双累加器
> 提升 memory-level parallelism，并比较 128/256 threads；小输出投影单独评估融合或
> 多行映射；LM head 则作为超大 K 的独立带宽案例。真正多 token prefill 完成后，再以
> cuBLASLt 作为可靠基准并评估 Tensor Core tiled GEMM。

## 2. 优化对象与 4B 代表形状

记号统一为 `M=输入维度，K=输出维度`。当前基准覆盖三种不同瓶颈区间：

| 类别 | 代表层 | K × M | 目的 |
|---|---|---:|---|
| 小输出 GEMV | GDN gate | 32 × 2560 | 暴露小 grid、launch 和低 occupancy |
| 中大型 GEMV | GDN out | 2560 × 4096 | 典型收缩投影 |
| 中大型 GEMV | GDN QKV | 8192 × 2560 | 典型宽输出投影 |
| 中大型 GEMV | MLP up | 9216 × 2560 | FFN 扩展投影 |
| 长 reduction GEMV | MLP down | 2560 × 9216 | 更长的单行 reduction |
| 超大输出 GEMV | LM head | 248320 × 2560 | decode 中最大的单个权重流 |

Qwen3.5-4B 的 BF16 权重流远大于 FP32 activation：例如 LM head 权重约 1.18 GiB，
而输入 activation 只有 10 KiB。因而大 GEMV 的主优化目标是更有效地读取一次性权重并
隐藏其延迟，而不是缓存整个矩阵。

## 3. 优化前实现

实现位于 `kuiper/source/op/kernels/cuda/matmul_kernel.cu`。基线 BF16 kernel 为：

```text
grid  = K blocks                 // 每个输出行一个 block
block = 128 threads

thread t:
    sum = Σ input[i] * fp32(weight[row, i]), i = t, t + 128, ...
block:
    CUB BlockReduce(sum)
thread 0:
    output[row] = total
```

它有以下优点：同一 warp 读取连续权重，访存合并；输出行彼此独立；FP32 累加精度明确；
没有 local-memory spill。问题则是：

1. BF16 权重逐元素 load 和逐元素转换，没有使用 `__nv_bfloat162`；
2. 每次循环只有一个 accumulator，load/convert/FMA 依赖链较长，memory-level
   parallelism 有限；
3. 固定 128 threads，没有针对 `M=2560/4096/9216` 比较 block size；
4. 先把每线程结果写入额外的 `sdata[128]`，再交给 CUB reduction，且 reduction 前后有
   多余同步；CUB 本身可以直接规约寄存器中的 `sum`；
5. 优化前 CUDA kernel 忽略接口传入的 `scale`，与 CPU 语义不一致；R1 已修复；
6. CUDA 接口声称接受最多二维 input，但只校验 `input.get_dim(0)==M`、只产出 K 个值，
   并未实现 CPU 路径的多列输入；当前 `MatmulLayer::check()` 也只接受长度 M 的单向量；
7. 每行一个 block 对 K=32 只产生 32 blocks，不能填满 56-SM GPU；但盲目让一个 block
   处理多行会进一步减少 grid，必须另做小 K specialization 或上层融合；
8. decode N=1 无法有效利用 Tensor Core 的矩阵 tile，强行 padding 成 GEMM 可能让布局、
   padding 和同步开销超过 MMA 收益。

其中第 5、6 项是接口/正确性问题，必须和性能优化分开记录。当前模型调用的 scale 恒为
1，因此旧 4B trace 没暴露第 5 项，但公共 kernel API 仍应保持 CPU/CUDA 一致。

## 4. 优化前 NCU 证据

固定环境为 RTX 4070 SUPER（sm_89）、CUDA 12.8.93、Nsight Compute 2025.1.1、
Release build。基线代码 commit 为 `304eea3`，报告目录和 SHA-256 见
`benchmarks/qwen35/BASELINE_4B.md`。

| Case | NCU duration | DRAM | DRAM GB/s | SM | L2 hit | Occupancy | Waves/SM | Eligible | Long scoreboard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gate | 3.776 us | 10.70% | 51.76 | 3.36% | 67.03% | 8.07% | 0.05 | 4.09% | 56.33% |
| GDN out | 59.680 us | 90.18% | 443.02 | 24.16% | 14.65% | 97.85% | 3.81 | 12.07% | 77.66% |
| GDN QKV | 122.464 us | 93.34% | 458.72 | 26.07% | 7.89% | 96.55% | 12.19 | 11.61% | 86.73% |
| LM head | 4009.280 us | 91.36% | 343.58 | 31.54% | 15.42% | 98.39% | 369.52 | 14.24% | 88.15% |
| MLP down | 140.896 us | 92.14% | 452.77 | 20.72% | 35.38% | 97.74% | 3.81 | 9.28% | 70.97% |
| MLP up | 136.224 us | 93.95% | 461.62 | 26.37% | 7.72% | 96.97% | 13.71 | 15.15% | 73.14% |

所有 BF16 GEMV 均为 37 registers/thread、0 local load/store、Tensor Pipe 利用率 0。
这些指标给出三个结论：

- 中大型投影 achieved occupancy 已接近满值，降寄存器不是首要矛盾；DRAM 接近饱和、
  long scoreboard 高，重点应放在权重 load 指令效率和延迟隐藏；
- gate 的 occupancy 低是总 grid 只有 32 blocks 导致的，和寄存器/shared-memory 上限无关；
- LM head 虽然同样 memory-bound，但只有 343.58 GB/s，明显低于中型投影的
  443–462 GB/s，需要单独验证 cache policy、block size 和超大 grid 行为。

CUDA Event 的优化前 median/p95 见基线文档；前后对照必须使用相同 cold-cache 策略、
shape、warmup/repeat 和同一台机器。短 kernel 的 WSL 调度噪声较大，因此至少重复三组，
NCU duration 也采多次确认。

## 5. 分阶段计划

### 阶段 A：正确性护栏与接口修复

- 增加非 1 `scale` 的 CPU/CUDA 对照，修复 CUDA 忽略 scale；
- 增加奇数 M，确保后续 BF16x2 向量路径有安全尾部；
- 保留 4B `[4096,2560]` 大尺寸专项测试；
- 明确二维 input 当前不受 CUDA/MatmulLayer 支持，不在本轮偷偷扩展接口。

这一阶段不声称性能收益，并单独提交。

### 阶段 B：BF16x2 + 双累加器 GEMV

对偶数 M 使用成对读取：

```text
load float2 input
load __nv_bfloat162 weight
convert weight pair -> float2
acc0 += x0 * w0
acc1 += x1 * w1
```

循环结束再相加并做 block reduction。预期收益来自减少权重 load/地址计算/转换指令，
并用两个独立 accumulator 增加 ILP；不是减少必须读取的 BF16 权重字节数。奇数 M 保留
标量安全路径或尾部处理。候选 block size 为 128 和 256，按所有 4B shape 实测分派，
不能只为单一矩阵硬编码。

同时删除额外的 shared `sdata` 和不必要同步，让 CUB 直接规约寄存器 `sum`。该变化会
影响 reduction 次序，正确性使用相对误差而非 bitwise equality。

### 阶段 C：按瓶颈分类的 specialization

- **小 K=32**：通用 GEMV kernel 无法产生足够 blocks。优先评估把成对 gate A/B
  projection 融合为一个 launch，或让一个输出行由多个独立 reduction tile 协作；后者
  需要第二阶段归并/atomic，只有实测覆盖 launch 开销才接受；
- **中大型 K**：比较 128/256 threads、每线程展开次数和只读/cache hint；主要观察
  DRAM GB/s、long scoreboard 和指令数；
- **LM head**：单独比较 block size/cache policy，避免一个对中型投影有利的策略在
  248320-block 超大 grid 上退化；还可评估把 argmax 与 logits 生成结合，但这会改变
  API/trace 可见结果，必须作为独立功能。

### 阶段 D：真正的 prefill GEMM

先让模型和 `MatmulLayer` 明确支持 `[M,N]` 或等价 token-major layout，再建立
`N={8,32,128,256}` 的正确性与基线。实现顺序为：

1. 以 cuBLASLt BF16-weight/FP32-output 路径作为正确性和性能参照；
2. 根据 activation dtype 决策：若维持 FP32 activation，要确认 Tensor Core 支持和转换
   成本；若改为 BF16 activation，要逐层验证数值误差；
3. 实现 shared-memory tiled kernel，采用 vectorized global load、double buffering，
   再评估 `cp.async` 和 MMA；
4. 对不同 N 做 dispatch，短 prompt 不应被大 tile 的 padding/启动成本拖慢；
5. GEMM 和 token-by-token baseline 分开报告，不能把算法级并行收益全部归因于 kernel。

## 6. 接受标准

每个候选版本都按以下顺序验证：

1. operator CPU/CUDA 对照，包括 scale、奇数 M 和 4B 大尺寸；
2. 完整 GTest、CTest；对可能改变模型数值次序的版本补真实 4B trace；
3. 六个 4B GEMV shape 的 cold-cache CUDA Event，至少三组，报告 median/p95；
4. gate、中型投影、MLP down、LM head 的重复 NCU，报告 duration、DRAM、SM、stall、
   instructions、registers 和 local spill；
5. 至少主要中大型 shape 不退化，若采用 shape dispatch，要明确适用区间和 fallback；
6. 每项完整功能一个 commit；无收益实验记录原因后回退，不让无效代码留在主路径。

## 7. 实验总表

| 轮次 | Commit | 变化 | 正确性 | Event | NCU | 结论 |
|---|---|---|---|---|---|---|
| R0 | `304eea3` | 128-thread 标量 BF16 GEMV | 4B 大投影通过 | 见基线文档 | 见第 4 节 | 优化前基线 |
| R1 | `cf66f5e` | scale + 奇数 M 护栏/修复 | focused/完整测试通过 | 不适用 | 不适用 | 已完成 |
| R2 | 本轮提交 | BF16x2、双 accumulator、直接 CUB reduction | 3/3 focused | 全 shape 已测 | 4 类 shape 已测 | 接受 256 threads |
| R3 | R2 已完成 block 搜索 | 128/256/512 threads 对照 | 同 R2 | 256 通用最优 | 256 已测 | 接受 256 threads |
| R4 | 待实验 | 小 K specialization 或 gate 融合 | 待测 | 待测 | 待测 | 待定 |
| R5 | prefill 后 | 真正的多 token GEMM | 待测 | 待测 | 待测 | 尚未开始 |

以后每轮在本文件追加实现映射、命令、原始报告目录、SHA-256 和结论；失败版本同样保留
数据与原因。最终总结必须同时回答“为什么快”“在哪些 shape 快”“有没有数值或适用范围
代价”，而不只给出一个最佳加速比。

### 7.1 R1：补齐 scale 语义和奇数 M 护栏

优化前 CPU BF16 kernel 在写输出时计算 `sum * scale`，CUDA BF16 kernel 却只写
`sum`。Qwen3.5 的线性层固定传 1，所以模型 trace 无法发现该问题；直接使用公共
kernel API（例如 attention 中的缩放 matmul）时会产生静默错误。

R1 把 `scale` 传入 BF16 CUDA kernel 并在 reduction 后由 thread 0 应用，同时增加
`M=259, K=37, scale=-0.375` 的 CPU/CUDA 对照。奇数 M 不是为了当前 4B shape，而是
提前保护 R2 的 BF16x2 快路径：向量循环必须只覆盖完整 pair，最后一个元素仍要正确
参与累加。focused BF16 测试 3/3、完整 GTest 56/56（由 CTest fixture 提供 tiny
模型环境）和 CTest 4/4 均通过。该轮不改变 scale=1 的 4B 性能路径，因而不采性能数据。

### 7.2 R2：BF16x2、双累加器和 block-size 搜索

R2 对偶数 M 走专用 fast path。线程每次用 `float2` 读取两个 FP32 activation，用
`__nv_bfloat162` 读取两个 BF16 weight，再由 `__bfloat1622float2` 转成 FP32；两路
FMA 分别累加到 `sum0/sum1`，循环结束后合并。奇数 M 继续走 R1 验证过的 scalar
fallback。两条路径都不再先把 partial sum 写入 `sdata[thread]`，而是把寄存器值直接交给
`cub::BlockReduce`，因此删除了额外 shared-memory round trip 和多余同步。

这里的关键安全条件是：CUDA allocator 保证 tensor 基地址对齐；只有 M 为偶数时每一行
BF16 weight 的起点才始终满足 `__nv_bfloat162` 的 4-byte alignment，所以 dispatch
不能只在 kernel 内简单处理一个奇数 tail。4B 的所有真实 M 都为偶数，能进入 fast path。

block-size 探索使用 cold cache、20 warmup、200 samples。下表列代表 shape；单位均为
微秒。p95 在 WSL 下仍有离散调度尖峰，因此 block 选择以全 shape median 和 NCU 为主。

| Shape | R1 scalar/128 | BF16x2/128 | BF16x2/256 | BF16x2/512 | 选择 |
|---|---:|---:|---:|---:|---|
| gate `32×2560` | 6.144 | 6.144 | 5.120 | 5.120 | 256；512 无新增收益 |
| full K/V `1024×2560` | 21.504 | 21.504 | 20.480 | 22.528–22.656 | 256 |
| GDN out `2560×4096` | 67.584 | 66.560 | 66.560 | 66.560 | 256；避免其他 shape 退化 |
| GDN QKV `8192×2560` | 129.024 | 129.024 | 128.000 | 135.168 | 256 |
| MLP up `9216×2560` | 143.376 | 143.360 | 143.360 | 148.480 | 256 |
| MLP down `2560×9216` | 148.480 | 143.360 | 144.384 | 143.360 | 128 略快但差距小，先用通用 256 |
| LM head `248320×2560` | 3595.264 | 3638.272 | 3570.688 | 3569.664 | 256；512 收益可忽略且中型退化 |

256 相对同一时段 R1 的 median：gate 降 16.7%，`1024×2560` 降 4.8%，中型投影约降
0.8%–1.5%，MLP down 降 2.8%，LM head 降 0.7%。这不是一个大幅降低权重流量的
优化：BF16 weight 仍必须完整读取一次，因此越接近 DRAM 上限的 shape，收益越小。

首轮 NCU 使用和 R0 相同的 `detailed + SchedulerStats + WarpStateStats`、kernel replay 和
cache control。结果为：

| Shape | R0 duration | R2/256 duration | 改善 | R0 global ld | R2 global ld | R2 registers | local ld/st |
|---|---:|---:|---:|---:|---:|---:|---:|
| gate | 3.776 us | 3.360 us | 11.0% | 5,120 | 2,560 | 37 | 0 / 0 |
| GDN out | 59.680 us | 56.800 us | 4.8% | 655,360 | 327,680 | 37 | 0 / 0 |
| MLP down | 140.896 us | 133.984 us | 4.9% | 1,474,560 | 737,280 | 37 | 0 / 0 |
| LM head | 4009.280 us | 3971.936 us | 0.9% | 39,731,200 | 19,865,600 | 37 | 0 / 0 |

global-load SASS 指令恰好减半，证明 BF16x2/FP32x2 load 确实生成了预期的成对访问；
registers 仍为 37 且无 spill。256 threads 也把 gate 的 achieved occupancy 从 8.07%
提高到 14.58%、eligible 从 4.09% 提高到 8.30%；LM head 的 eligible 从 14.24%
提高到 21.94%。代价是更多线程参与边界判断和 CUB reduction，总指令并未下降：例如
GDN out 从 2,426,880 增到 2,836,480，LM head 从 179,783,680 增到 271,165,440。
最终仍有小幅加速，说明减少 load 指令并增加可调度 warp 的收益覆盖了控制/reduction
开销，但它也解释了为什么大带宽 shape 只有个位数百分比提升。

512 threads 在 LM head 上只比 256 快 0.03%，却让 QKV 慢 5.6%、MLP up 慢 3.6%，
所以拒绝。128 threads 对 MLP down 有约 0.7% 优势，但 LM head 退化 1.2%，其余 shape
收益更弱；当前先保留一个 256-thread 通用 fast path，只有重复 NCU 证明 M-based dispatch
能稳定覆盖成本时才增加分派复杂度。

探索报告目录（源码尚未提交时，JSON 内嵌 commit 仍显示 R1，目录名才是实验身份）：

```text
/home/tuesday/workspace/icd/profiles/KuiperLLama/qwen35/
├── 4b-stage2-gemv-r1-cf66f5e/cuda-events
├── 4b-stage2-gemv-r2-bf16x2-128/cuda-events
├── 4b-stage2-gemv-r2-bf16x2-256/{cuda-events,ncu}
└── 4b-stage2-gemv-r2-bf16x2-512/cuda-events
```

R2 提交后还需重建 benchmark 并采三次正式 NCU，使报告的 Git commit 与源码一致；正式
哈希和重复结果将在下一次文档提交补齐。
