# BF16 GEMV/GEMM 优化过程

本文独立记录 Qwen3.5-4B BF16 线性层的瓶颈分析、优化假设、具体实现、失败实验和
前后实测结果，供代码复盘和面试讲解使用。文中严格区分已经由 CUDA Event/NCU
验证的事实与仍待实验的方案，不用理论带宽或单次最小值代替真实加速结果。

当前状态：**GEMV R2 与 GEMM G2 均已完成独立算子优化和正式验证。**
R1 修复了 CUDA
BF16 路径忽略 `scale` 的接口语义，并用奇数 M 覆盖向量化尾部；R2 为偶数 M 增加
BF16x2/FP32x2 成对读取、双 accumulator、直接 CUB reduction，并通过 128/256/512
threads 实测选择 256。G0 定义了 token-major `input[N,M] × weight[K,M]^T →
output[N,K]`，G2 已加入按 N 分派的 register tiling；但模型 prefill 尚未接入，当前
decode 和 token-by-token prefill 仍只向
`MatmulLayer` 传入一个 FP32 activation vector，执行的是

```text
x[M] * W[K, M]^T -> y[K]
```

也就是 FP32 input、BF16 weight、FP32 accumulate/output 的 GEMV。只有并行 prefill
把多个 token 组成矩阵后，计算才会成为 GEMM；在此之前不能把 Tensor Core GEMM 的
结论套到当前 N=1 workload 上。

## 1. 面试讲解主线

当前 R2 可以这样完整描述：

> 我先把框架中的 MatmulLayer 按实际 workload 拆成 decode GEMV 和后续 prefill
> GEMM。优化前 BF16 GEMV 每个输出行启动一个 128-thread block，每个线程以标量方式
> 读取 BF16 权重、转 FP32 并累加，最后用 CUB 做 block reduction。NCU 显示中大型
> 投影的 DRAM throughput 达 90.18%–93.95%，SM throughput 只有 20.72%–31.54%，
> long-scoreboard 占 70.97%–88.15%，说明主要受权重流量和访存等待限制；但 32 输出
> 的 gate 投影只有 32 blocks、0.05 waves/SM，是 launch/并行度受限。于是我不会用同一
> 个 kernel 策略解释所有形状。我为偶数 M 实现 BF16x2/FP32x2 成对读取、双累加器和
> 直接 CUB reduction，并实测 128/256/512 threads，最终选择 256；奇数 M 保留 scalar
> fallback。global-load 指令减半、37 registers 且没有 spill。三次正式 NCU 的中位
> duration 相对基线在 gate、GDN out、MLP down 上分别下降 16.9%、10.6%、5.7%，真实
> 4B 的逐层误差和 10 个 token 均通过。LM head 的 Event/NCU 波动跨过基线，所以只记录
> 为未证实的微小收益，并留作独立 specialization。真正多 token prefill 完成后，再以
> cuBLASLt 作为可靠基准并评估 Tensor Core tiled GEMM。

GEMM 部分可以接着讲：初版每线程只算一个输出，N=32 的 NCU 虽有 77.72% occupancy，
但只有 28.59% eligible warps，且重复 N tile 产生大量 L2/shared load。于是把 block 从
16×16 输出扩成 32×32/32×64，并让每线程保留 2×2 或 2×4 个独立 accumulator；再按 N
选择 tile，避免小 batch 被过宽 tile 拖慢。N=32 的 global/shared load 均减半、总指令
减少 37.6%、无 spill，NCU duration 下降 30.3%；独立 Event 上 N=128 下降 62.8%。这也
是一个“occupancy 下降但 kernel 更快”的例子，因为优化目标是有效工作量与数据复用，
不是孤立地追求 occupancy 数字。

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
3. 全部 13 个已注册 4B GEMV shape 的 cold-cache CUDA Event，至少三组，报告
   median/p95；
4. gate、中型投影、MLP down、LM head 的重复 NCU，报告 duration、DRAM、SM、stall、
   instructions、registers 和 local spill；
5. 至少主要中大型 shape 不退化，若采用 shape dispatch，要明确适用区间和 fallback；
6. 每项完整功能一个 commit；无收益实验记录原因后回退，不让无效代码留在主路径。

## 7. 实验总表

| 轮次 | Commit | 变化 | 正确性 | Event | NCU | 结论 |
|---|---|---|---|---|---|---|
| R0 | `304eea3` | 128-thread 标量 BF16 GEMV | 4B 大投影通过 | 见基线文档 | 见第 4 节 | 优化前基线 |
| R1 | `cf66f5e` | scale + 奇数 M 护栏/修复 | focused/完整测试通过 | 不适用 | 不适用 | 已完成 |
| R2 | `00b03df` | BF16x2、双 accumulator、直接 CUB reduction | 56/56 + 真实 4B | 全 shape 三组 | 4 类 shape 三组 | 接受 256 threads |
| R3 | R2 已完成 block 搜索 | 128/256/512 threads 对照 | 同 R2 | 256 通用最优 | 256 已测 | 接受 256 threads |
| R4 | 待实验 | 小 K specialization 或 gate 融合 | 待测 | 待测 | 待测 | 待定 |
| G0 | `0dcc8d5` | `[N,M]` 接口 + `16×16×32` tiled GEMM | 58/58 | 待建立 | 待建立 | 算子功能完成 |
| G1 | `c903be9` | N 维 benchmark + GEMV-loop 公平对照 | 58/58，CTest 5/5 | 基准入口完成 | 不适用 | 测量基础设施完成 |
| G2 | `bae70ee` | N-aware tile + 每线程多输出 register tile | 58/58，CTest 5/5 | 全 shape 三组 | N=32/128 正式复测 | 接受 |

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

### 7.3 R2 提交后的正式复测

正式报告在 commit `00b03df` 提交并重新 configure/build 后采集，JSON 中的完整 commit
为 `00b03df74d136767b74839a275598207acde3d9e`。CUDA Event 进行了三组全 shape
复测，每组 cold cache、5 warmup、30 samples：

| Shape | R0 median | R2 三组 median | 稳定结论 |
|---|---:|---:|---|
| gate `32×2560` | 6.144 us | 5.120 / 5.120 / 5.120 us | 下降 16.7% |
| full K/V `1024×2560` | 21.504 us | 20.480 / 20.480 / 20.480 us | 下降 4.8% |
| GDN out `2560×4096` | 67.584 us | 66.560 / 66.560 / 66.560 us | 下降 1.5% |
| GDN QKV `8192×2560` | 129.024 us | 128.000 / 127.072 / 128.000 us | 下降约 0.8%–1.5% |
| MLP up `9216×2560` | 143.376 us | 142.336 / 143.360 / 142.400 us | 下降约 0%–0.7% |
| MLP down `2560×9216` | 148.480 us | 143.360 / 143.360 / 143.360 us | 下降 3.4% |
| LM head `248320×2560` | 3595.264 us | 3614.720 / 3750.912 / 3642.368 us | 跨时段未证实收益 |

LM head 的三组短 Event median 没有复现探索阶段的 3570.688 us，并且整套测试中可见
周期性 WSL 调度离散值。为区分 kernel 选择和跨时段波动，又做了紧邻的 200-sample A/B：
改良 scalar/128 fallback 为 3593.728 us，vector/256 为 3573.760 us，vector 快 0.56%。
因此保留 vector/256，但只把 LM head 结论写为“收益微小且尚不稳健”；下一轮 LM-head
specialization 必须改善绝对 weight bandwidth，不能依赖 0.5% 级噪声。

三组正式 NCU 的 duration 如下。括号中是三组的中位数，相对 R0 只用该中位数计算：

| Shape | R0 | R2 三次 NCU duration | 中位数变化 |
|---|---:|---:|---:|
| gate | 3.776 us | 3.136 / 3.136 / 3.072 us（3.136） | −16.9% |
| GDN out | 59.680 us | 53.376 / 56.352 / 46.208 us（53.376） | −10.6% |
| MLP down | 140.896 us | 134.816 / 125.184 / 132.800 us（132.800） | −5.7% |
| LM head | 4009.280 us | 3984.224 / 3517.760 / 4437.888 us（3984.224） | −0.6%，波动跨过 R0 |

gate、GDN out、MLP down 三次都快于 R0，支持接受 R2；LM head 的范围跨过 R0，不能
用中位数包装成确定收益。硬件层面的稳定事实仍是 global-load 指令减半、37 registers、
0 local load/store；性能波动主要来自 NCU/WSL 下的频率和调度，而不是 spill。

真实 Qwen3.5-4B BF16 checkpoint 也完成了 CUDA 逐层 trace。相对优化前保存的 Kuiper
CPU reference，decoder 最大相对误差为 `2.14677e-05`，final norm 为 `2.88499e-06`，
logits 为 `2.60097e-06`；10 个 greedy token 完全一致。相对 R2 前 CUDA trace，decoder
最大相对误差为 `1.38607e-06`，logits 为 `4.15089e-07`，token 同样完全一致。这说明
双 accumulator 改变 reduction 次序后只有预期的 FP32 舍入差异，没有改变模型行为。

正式产物目录：

```text
/home/tuesday/workspace/icd/profiles/KuiperLLama/qwen35/4b-stage2-gemv-after-00b03df/
├── cuda-events-run1/
├── cuda-events-run2/
├── cuda-events-run3/
├── ncu-run1/
├── ncu-run2/
├── ncu-run3/
└── model-validation/cuda/
```

正式 NCU report SHA-256：

| Run | gate | GDN out | MLP down | LM head |
|---|---|---|---|---|
| 1 | `863c604f4d6a988ca20c24ffe940e998ef0c62064d99a142d1af6cc6401e4b0f` | `f4dfb2c5de218529b742199944774d7c601bbf48cdad272668419af23691d343` | `90ee1fdb83407eba5789293a06e953adc4022efdb3b7d0ebebedb9fa9e91b08f` | `f6e6f84d88377f8b0bf661e333d6d1513d1dba85ee0e5a1fce0b8464419060ed` |
| 2 | `0ccfe5d459b88577d6c42ccea2f117d82451c7db07f9f4aa37429ed6dfb28641` | `3dfdd6d36803b98838de2a8fb24232caa33ec1b9c62c681c707be1558d1fa88c` | `211f7d6d43ed692eb845b77e60a829b2389f779b966c2f7c705c89e4546bf938` | `174ea19a8b6fedbe9fb1b5137681ceb51cf0fa71a70cce9f44169ab8cbd00266` |
| 3 | `ed2faa2c998d19275f6cedd7e6abd8b77ef950f3e99b989bacb47784b714eb87` | `e80a64d5dbadce9a03b41a718b499258bb24a40343246b21203eecb2a9b882ab` | `485c97c2a1871bb5a51a7346d27d2e9c1029c5f0a64fe829af9d91567ad49245` | `0d58f845e63aa2e55fe437624826ca5bcf0437977aed4cdc397ae1da9371a5c6` |

## 8. GEMV R2 完成时确定的后续路线（历史）

R2 已经拿到“指令级向量化 + block 并行度”的第一轮收益，但大投影仍然是 weight
bandwidth-bound。下一轮优先级为：

1. 单独处理 K=32 的成对 A/B gate projection，判断 fusion 能否继续减少 launch；
2. 对 LM head 研究真正减少或更高效组织权重流量的 specialization，并保留完整 logits
   API 的正确性；
3. 不在 N=1 GEMV 上强行使用 Tensor Core；先实现并行 prefill 接口，再开始多 token
   GEMM、cuBLASLt baseline 和自研 tiled kernel 对照。

## 9. GEMM G0：二维接口和初版 tiled kernel

G0 先解决“框架名为 Matmul，但只有单向量路径”的功能缺口。新的公共布局为：

```text
input  [N,M] FP32，token-major，单个 token 的 M 个元素连续
weight [K,M] BF16，输出行优先
output [N,K] FP32，token-major
```

选择 token-major 而不是沿用旧 CPU kernel 未被调用过的 `[M,N]` 解释，有两个原因：

1. `EmbeddingLayer` 本来就产生 `[tokens, hidden]`，未来 prefill 不需要额外转置；
2. 可以把同一输入按 token 切成 N 个连续 `[M]` 向量，建立 N 次 GEMV 与一次 GEMM 的
   公平对照。

`MatmulLayer::check()` 现在分别检查 `[M]→[K]` 和 `[N,M]→[N,K]`，二维路径当前明确
只接受 BF16 weight；INT8、FP32 weight 和 bias broadcast 暂不假装支持，而是返回错误。
CPU BF16/FP32 实现也统一成 token-major 语义。仓库中没有旧的二维调用者，因此这项
布局修正不会改变已有模型路径。

初版 CUDA kernel 使用 `TILE_N=16, TILE_K=16, TILE_M=32`：

```text
grid.x = ceil(K / 16)
grid.y = ceil(N / 16)
block  = (16,16) = 256 threads

每个 M tile：
  协作加载 16×32 FP32 activation 到 shared memory
  协作加载并转换 16×32 BF16 weight 到 FP32 shared memory
  每个线程计算一个 output[n,k] 的 32 次 FMA
```

shared tile 的 reduction 维增加一个 padding column，避免同一 warp 在不同 K row、相同
M coordinate 上读取 weight 时形成 32-way bank conflict。N、K、M 均通过边界判断支持
非 tile 整倍数；测试特意使用 `N=13, K=37, M=259` 覆盖三种 tail。

G0 的 focused BF16 测试为 5/5，带自动 tiny fixture 的完整 GTest 为 58/58，CTest 为
4/4。这一提交只确认接口、布局和数值正确，不在没有 GEMV-loop/cuBLAS 对照时宣称性能
收益。下一步 G1 会给 benchmark 增加 N 维和实现选择，冻结不同 prompt tile 的基线后再
搜索 tile、向量化和库实现。

## 10. GEMM G1：可复现的 GEMM 与 GEMV-loop 对照入口

G1 扩展 `qwen35_bench --mode matmul`，新增 `--n` 与
`--matmul-implementation auto|gemm|gemv-loop`。`auto` 在 N=1 时走既有 GEMV，在 N>1
时走 GEMM；显式 `gemv-loop` 则为每个 token 建立零拷贝的一维 tensor view，并在同一
CUDA stream 内发出 N 次既有 GEMV。tensor view 和所有内存分配均在计时区间之外，因此
对照测到的是“一个二维 kernel”与“N 个一维 kernel”的执行差异，不混入 host allocation。

JSON schema 升级为 2，并记录 `batch_size_n`、实际 implementation、任务 GFLOP/s、
logical bytes 和 estimated memory bytes。logical bytes 对两种实现都只计算一次 weight，
适合表达完成同一数学任务的有效吞吐；estimated memory bytes 对 GEMV-loop 按 N 次完整
weight 读取建模。后者只用于解释流量放大，真实 DRAM bytes 仍以 NCU counter 为准。

固定 shape 表覆盖 `N=8/32/128` 的 `2560→4096` GDN projection，以及 N=32 的 GDN
output、MLP up/down。`run_gemm_baseline.sh` 默认依次生成 GEMV-loop 与 GEMM 的 cold-cache
JSON，避免只展示对自研 kernel 有利的单边结果。CPU CTest smoke 同时覆盖一次二维 GEMM
和三次 GEMV-loop，CUDA 数值正确性继续由 G0 的非整 tile 单元测试负责。

本轮仍然不记录“优化百分比”：G1 是测量基础设施。下一轮 G2 才会在相同命令、相同 shape
和相同 cache policy 下冻结 G0 kernel 的数据，随后进行 tile/register/vectorization 搜索并
用 NCU 解释接受或拒绝每个候选的原因。

本机验证中，CPU `N=3,M=64,K=32` 的两条路径 checksum 均为 `0.647461`；CUDA
`N=13,M=259,K=37`（三维均含 tail）的两条路径 checksum 均为 `0.347168`。完整 CTest
为 5/5，内部 GTest 仍为 58/58。该小 CUDA shape 的单次观测只用于验证 harness 确实执行
了不同路径，不作为正式性能结论。

## 11. GEMM G2：register tiling 与按 N 分派

### 11.1 G0 性能和 NCU 暴露的问题

在 commit `c903be9` 上使用 cold cache、5 warmup、30 samples 冻结 G0。下表把一次
`16×16×32` GEMM 与同一输入的 N 次 R2 GEMV 并列；单位为毫秒：

| Shape | GEMV-loop | G0 GEMM | G0 相对 loop | 结论 |
|---|---:|---:|---:|---|
| GDN z，N=8，M=2560，K=4096 | 0.173056 | 0.248848 | 慢 43.8% | 小 N 无法摊薄 tile/同步成本 |
| GDN z，N=32 | 0.592384 | 0.433152 | 快 26.9% | 已有 weight reuse，但计算核心低效 |
| GDN z，N=128 | 2.730560 | 1.733088 | 快 36.5% | GEMM 方向正确，仍有较大优化空间 |
| GDN out，N=32，M=4096，K=2560 | 0.397936 | 0.381952 | 快 4.0% | 初版只取得边缘收益 |
| MLP up，N=32，M=2560，K=9216 | 1.007104 | 0.777216 | 快 22.8% | K 大时 block 数足够 |
| MLP down，N=32，M=9216，K=2560 | 0.985088 | 0.974848 | 快 1.0% | 长 reduction 的串行依赖明显 |

G0 的 N=32 GDN z NCU baseline 为：duration `530.496 us`、SM throughput `80.30%`、
DRAM throughput 仅 `10.75%` / `52.82 GB/s`、L2 hit `81.80%`、achieved occupancy
`77.72%`、40 registers/thread、eligible warps `28.59%`、long scoreboard `21.87%`。
这组指标说明瓶颈不是“DRAM 已打满”：G0 需要两个 N tile，同一 weight 的第二次加载大量
命中 L2；同时每线程只有一个 accumulator，长 FMA dependency chain 使高 occupancy 没有
转化成高 issue efficiency。因此 G2 的首要目标是增加单线程独立输出、减少 block 与重复
tile load，而不是继续堆线程数。

### 11.2 接受的实现

G2 仍保持 FP32 activation × BF16 weight、FP32 accumulation/output，不为使用 Tensor Core
而把 activation 静默降成 BF16。每个线程改为计算多个 output：

| N 区间 | tile `N×K×M` | 每线程输出 | block threads | 设计原因 |
|---|---:|---:|---:|---|
| N≤8 | `8×32×32` | `1×2` | 128 | 避免 G0 中无效 N row，保留足够 warps |
| 9–16 | `16×32×32` | `1×2` | 256 | 覆盖 prompt tail，K 方向提供两个独立 accumulator |
| 17–32 | `32×32×32` | `2×2` | 256 | 一个 block 覆盖完整 N=32，weight 只加载一次 |
| N>32 | `32×64×32` | `2×4` | 256 | 大 N 可摊薄 8 accumulators，进一步降低 block 数 |

以 N=32 为例，G0 的 block 数为 `(4096/16)×(32/16)=512`；G2 降为
`(4096/32)×(32/32)=128`。每线程的四个独立 accumulator 同时增加 instruction-level
parallelism，缩短单 accumulator dependency chain；N=128 的 K=64 版本则让每线程计算
八个输出。shared-memory padding 和 M/K/N tail guard 保留。

### 11.3 被拒绝的候选

1. 对所有 N 使用 K=64、每线程四个 K 输出：N=128 从 K=32 候选的 `0.803840 ms`
   降到 `0.645632 ms`，但 N=32 从 `0.306176 ms` 退化到 `0.359424 ms`，N=8 从
   `0.215040 ms` 退化到 `0.313344 ms`。因此只在 N>32 使用。
2. N=8 使用 64 threads、每线程四个 K 输出得到 `0.289792 ms`，少量 warps 无法隐藏
   shared/global latency，拒绝。
3. 沿 M 把每个输出拆成四个 accumulator，N=8 得到 `0.247808 ms`，比单 accumulator
   register-tile 的 `0.215040 ms` 慢；N=32 没有改善。额外寄存器与最终 reduction 没有被
   dependency 缩短收益覆盖，已从源码移除。

### 11.4 CUDA Event 三组复测

最终候选使用 cold cache、每组 5 warmup + 30 samples。表中为三组 median，百分比使用
三组中位数与 G0 比较：

| Shape | G0 | G2 run1 / run2 / run3 | 三组中位数改善 |
|---|---:|---:|---:|
| GDN z N=8 | 0.248848 | 0.212992 / 0.212992 / 0.200704 | 14.4% |
| GDN z N=32 | 0.433152 | 0.307136 / 0.306176 / 0.274432 | 29.3% |
| GDN z N=128 | 1.733088 | 0.645632 / 0.644096 / 0.566272 | 62.8% |
| GDN out N=32 | 0.381952 | 0.369664 / 0.369680 / 0.332800 | 3.2% |
| MLP up N=32 | 0.777216 | 0.528896 / 0.528928 / 0.530416 | 31.9% |
| MLP down N=32 | 0.974848 | 1.030656 / 0.848384 / 0.835584 | 13.0% |

MLP down 的 run1 比 G0 慢 5.7%，而后两组快 13%–14%；该 shape 暂记为“有收益但存在
跨 run 波动”，不把最好一次包装成稳定结论。N=8 虽较 G0 改善，仍慢于 N 次 GEMV，说明
小 prompt 的后续策略应是 operator 内部 GEMV fallback 或更适合小 M/N 的专用 kernel，
不能声称当前 GEMM 已覆盖所有 batch 区间。

正确性测试在同一个 GTest 中使用 N=7/13/23/35 覆盖四个实际模板族，并用 M=259、K=37
覆盖 reduction/output tail；每组都与 CPU BF16-weight/FP32-accumulation 参考比较。完整
CTest 5/5、内部 GTest 58/58。正式 NCU 后测将在 G2 commit 上采集，以确保报告内嵌 commit
与最终 kernel 一致。

原始 Event 目录：

```text
/home/tuesday/workspace/icd/profiles/KuiperLLama/qwen35/
├── 4b-stage4-gemm-g0-c903be9
└── 4b-stage4-gemm-g2-regtile-run{1,2,3}
```

### 11.5 提交后的正式 NCU 对照

G2 commit `bae70ee6f19d5821b649821119a8cbf4373e2716` 上重新运行 NCU。N=32 可与
G0 的同 shape、同 `detailed + SchedulerStats + WarpStateStats` 配置直接对比：

| 指标 | G0 `c903be9` | G2 `bae70ee` | 变化 |
|---|---:|---:|---:|
| Duration | 530.496 us | 369.856 us | −30.3% |
| Executed instructions | 61,849,600 | 38,593,536 | −37.6% |
| Global-load SASS instructions | 1,310,720 | 655,360 | −50.0% |
| Shared-load SASS instructions | 20,971,520 | 10,485,760 | −50.0% |
| Shared-store SASS instructions | 1,310,720 | 655,360 | −50.0% |
| Registers/thread | 40 | 40 | 不变 |
| Local load / store | 0 / 0 | 0 / 0 | 无 spill |
| Achieved occupancy | 77.72% | 33.60% | 下降 |
| Waves/SM | 1.52 | 0.38 | block 总数下降 |
| Eligible warps | 28.59% | 29.51% | 基本不变 |
| Long scoreboard | 21.87% | 24.61% | 略升 |
| DRAM read | 21.318 MB | 21.319 MB | 基本不变 |
| L2 hit rate | 81.80% | 65.46% | 重复 tile load 减少后下降 |

这个结果也说明不能把 occupancy 当作单一优化目标。G2 的 occupancy 和 waves 明显下降，
但每个 block 完成的有效输出更多，global/shared 指令减半，总指令减少 37.6%，最终 duration
下降 30.3%。G0 第二个 N tile 对 weight 的重复读取多从 L2 命中，因此 G2 的 DRAM read
没有减半、L2 hit 反而降低；真正被消除的是 L2→SM、shared-memory 与控制指令层面的重复
工作。40 registers 且无 local spill 证明 2×2 register tile 没有以寄存器溢出换速度。

N=128 的 K=64/2×4 模板 NCU duration 为 `790.688 us`，SM throughput `81.54%`、DRAM
throughput `7.44%` / `36.59 GB/s`、occupancy `54.72%`、eligible warps `38.54%`、long
scoreboard `12.32%`、40 registers 且无 local spill。这里 NCU 多 pass replay 下的 duration
高于独立 Event median `0.644096 ms`，性能百分比仍只使用不受 profiler 扰动的 Event 数据；
NCU 用于解释资源和指令行为。

正式报告目录与校验值：

```text
/home/tuesday/workspace/icd/profiles/KuiperLLama/qwen35/4b-stage4-gemm-after-bae70ee/
```

| Report | SHA-256 |
|---|---|
| `ncu-gdn-z-n32.ncu-rep` | `54183206527380059ecf8393bccb16fa345b3876e6e1bf37e07434c19fd1bbb2` |
| `ncu-gdn-z-n128.ncu-rep` | `f31c85282f2627c6cd3f99ac978e6cc32c3e71d4363d039e34a9f7ab51244727` |

## 12. GEMM 当前边界与后续工作

本轮完成的是可独立运行、可用 NCU 分析的 BF16-weight GEMM 算子，不等于模型已经拥有
并行 prefill。当前仍需后续完成：

1. N≤8 的 G2 GEMM 虽比 G0 快 14.4%，仍慢于 N 次成熟 GEMV；需要明确的小 N fallback
   或专用 kernel，并在 benchmark JSON 中暴露实际策略。
2. 将 embedding 后的 `[tokens,hidden]` 沿 projection/MLP 传递到二维 Matmul；GDN 的
   recurrent state update 和 causal attention 仍必须保持时序/因果语义，不能只把线性层
   改成二维就宣称 prefill 完成。
3. 当前二维路径只支持 BF16 weight、无 bias broadcast；FP32、INT8/量化与 bias 要么实现
   完整语义和测试，要么继续显式拒绝。
4. cuBLAS/cuBLASLt 对照必须保持同一语义：当前是 FP32 activation、BF16 weight、FP32
   accumulation/output。若库路径要求先把 activation 降为 BF16，或把 weight 扩为 FP32，
   就分别改变了数值语义或 4B 显存占用，不能与本轮 kernel 混写成公平加速比。后续先验证
   mixed-type 支持组合，再决定采用库 baseline 还是单独标注的 BF16-activation Tensor Core
   实验。
