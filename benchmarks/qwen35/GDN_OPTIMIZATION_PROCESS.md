# GDN 优化过程

本文是 Qwen3.5-4B `gated_delta_step` CUDA kernel 的独立优化档案，也是面试时讲解算子分析、实验设计与结果的主线材料。它不是一份只展示最终加速比的总结：基线、瓶颈证据、优化假设、失败方案、数值风险和每轮实测结果都会保留。

当前状态：**R0/R1 已完成；R2 消冗余实验因退化被回退；R3 的 4B 二维 tiled kernel 已合入并完成重复 NCU 对照。** R4 的 state 单次读取仍待实验。文中标为“待测”的内容不能作为已经取得的结果对外陈述。

## 1. 面试讲解主线

可以先用下面这段话概括项目，再按后续章节展开：

> 我先为 Qwen3.5-4B 的 Gated DeltaNet recurrent step 建立了可复现的 CUDA Event 和 Nsight Compute 基线。原 kernel 数值正确，但每个 V head 只启动一个 block，整个 launch 只有 32 个 block；在 56 SM 的 RTX 4070 SUPER 上 achieved occupancy 只有 7.71%，94.63% 的 scheduler cycle 没有 eligible warp，long scoreboard 占 79.48%。同时 DRAM 和 SM 吞吐分别只有 25.15% 和 5.02%，所以它不是算力或显存带宽饱和，而是并行度不足时被 state load latency 卡住。我先试过缓存 q 和外提循环不变量：global load 减少 33.1%、总指令减少 21.3%，但延迟反而增加 13.0%，因此回退。随后采用 8-column × 16-K-lane 的二维 tile，把 grid 从 32 提升到 512 blocks；三次 NCU duration 为 12.960–13.344 us，相比 26.880 us 基线约为 2.01–2.07×，没有 local-memory spill。这个过程说明优化目标不是让指令数最少，而是让 GPU 有足够并行工作去隐藏 state latency。

CUDA Event 对十几微秒 kernel 存在明显的 WSL/时钟离散值，因此本文同时给出历史冻结基线、同一时段 200 次 A/B 和重复 NCU，不从单次最小值推导加速比。R4 仍将继续测试 state 单次读取。

## 2. 问题背景与算子语义

Qwen3.5 的线性注意力层使用 Gated DeltaNet。当前 decode 路径一次处理一个 token，核心递推可以写成：

```text
decay   = exp(g[h])
memory  = decay * (k^T * S)
delta   = beta[h] * (v - memory)
S'      = decay * S + k * delta^T
out     = q_scale * q^T * S'
```

其中每个 V head 有一个二维 recurrent state `S[k_head_dim, v_head_dim]`。Qwen3.5-4B 的实际尺寸为：

| 参数 | 4B 数值 |
|---|---:|
| K heads | 16 |
| V heads | 32 |
| K head dim | 128 |
| V head dim | 128 |
| V/K head 分组 | 2:1 |
| state dtype | FP32 |
| state 总大小 | `32 * 128 * 128 * 4 B = 2 MiB` |

一次 recurrent step 会更新全部 32 个 state。按算法逻辑计，当前实现对 state 做两次读取和一次写回，流量约为 6 MiB；第二次读取可能命中 cache，所以该数值不能直接等同于 DRAM 实际流量。

## 3. 优化前 kernel 如何工作

实现位于 `kuiper/source/op/kernels/cuda/qwen35_kernel.cu`。基线映射为：

```text
grid  = 32 blocks             // 一个 V head 一个 block
block = 128 threads           // 一个线程负责一个 V-dim column
```

线程 `j` 独占 state 的第 `j` 列：

1. 第一次遍历 128 个 K 维，计算该列的 `k^T * S[:, j]`；
2. 得到 `delta[j]`；
3. 第二次遍历 128 个 K 维，原位更新 `S[:, j]`，并累计 `q^T * S'[:, j]`；
4. 写出 `out[j]`。

该设计的优点是容易验证、没有原子操作，而且同一 warp 在固定 `i` 时访问连续的 `S[i, j]`，state 访问天然合并。`k` 会被所有线程重复使用，因此先缓存到 512 B dynamic shared memory。

它的主要代价是：

- 整个 GPU 只有 32 个 block，少于设备的 56 个 SM；
- 每个线程串行执行两段长度为 128 的 state 依赖链；
- `q[i]` 没有像 `k[i]` 一样进入 shared memory，所有线程都会发出 broadcast load；
- 每个 block 的 128 个线程都重复计算同一个 `exp(g[h])` 并加载同一个 `beta[h]`；
- state 在更新前后被显式读取两次。

这里要特别说明：基线 kernel 的 state 访问已经是 coalesced 的。优化目标不是修复错误访存，而是在保留合并访问的前提下，提高可调度 warp 数量、隐藏访存延迟并减少冗余指令。

## 4. 基线环境与测量方法

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 SUPER 12 GiB |
| Compute capability | 8.9 |
| CUDA toolkit/runtime | 12.8.93 / 12.8 |
| Nsight Compute | 2025.1.1 |
| Build | Release, `CMAKE_CUDA_ARCHITECTURES=89` |
| 基线代码 commit | `304eea3` |
| NCU 采集 | `detailed` + SchedulerStats + WarpStateStats，kernel replay，cache control all |

测量职责严格分开：

- **CUDA Event** 测未插桩 latency，主指标是 median，同时保留 p95；
- **Nsight Compute** 解释硬件瓶颈和优化前后 counter 变化；
- NCU replay 时程序自己打印的耗时包含 profiler 开销，不能当作性能结果；
- 本项目只做核心算子分析，不使用 Nsight Systems，也不把整程序调度优化混入 GDN 案例。

基线报告保存在 Git 仓库之外：

```text
/home/tuesday/workspace/icd/profiles/KuiperLLama/qwen35/4b-stage1-before-304eea3/
├── cuda-events/
└── ncu/
```

详细哈希和其他算子的对照见 `benchmarks/qwen35/BASELINE_4B.md`。

## 5. 优化前核心指标

### 5.1 延迟与正确性

| 指标 | 基线 |
|---|---:|
| CUDA Event median | 0.016384 ms |
| CUDA Event p95 | 0.024256 ms |
| warmup / repeat | 5 / 30 |
| CUDA/CPU max abs error | 0 |
| NCU kernel duration | 26.880 us |

NCU duration 高于未插桩的 CUDA Event median 是正常现象。前后延迟比较以同一套 CUDA Event 命令为主，NCU duration 只和同配置 NCU 报告比较。

### 5.2 并行度与调度

| 指标 | 基线 | 含义 |
|---|---:|---|
| Grid / block | 32 / 128 | 只有 32 个 block 覆盖 56 个 SM |
| Waves per SM | 0.05 | launch 提供的工作量非常小 |
| Theoretical occupancy | 100% | 寄存器和 shared memory 没有限制驻留上限 |
| Achieved occupancy | 7.71% | 实际运行时可驻留/活跃 warp 很少 |
| Active warps per scheduler | 1.02 | scheduler 几乎没有可切换的 warp |
| Eligible warps per scheduler | 0.05 | 真正可以发射指令的 warp 极少 |
| One or more eligible | 5.37% | 只有少量 cycle 存在可发射 warp |
| No eligible | 94.63% | 绝大多数 cycle 无工作可发射 |
| Warp cycles per issued instruction | 18.94 | 发射间隔很长 |

“Theoretical occupancy 100%，但 achieved occupancy 只有 7.71%”并不矛盾。前者说明如果有足够多 block，资源允许较高驻留；后者说明本次 launch 根本没有提供足够多的 block/warp。它直接排除了“先降寄存器以提高 occupancy”这一错误优先级。

### 5.3 stall、吞吐与访存

| 指标 | 基线 | 判断 |
|---|---:|---|
| Long scoreboard | 15.05 cycles / 79.48% | 主要在等待远端 memory dependency |
| DRAM throughput | 25.15% / 123.41 GB/s | 没有打满显存带宽 |
| SM throughput | 5.02% | 不是计算吞吐受限 |
| L1 hit rate | 68.50% | 存在复用，但不足以隐藏长延迟 |
| L2 hit rate | 48.35% | 一部分请求仍需更远层次返回 |
| Registers per thread | 40 | 不构成理论 occupancy 限制 |
| Dynamic shared memory | 512 B | shared memory 资源非常宽松 |
| Local load/store | 0 / 0 | 没有 register spill |
| Branch efficiency | 100% | 不是分支发散问题 |
| Global load/store SASS instructions | 49,664 / 16,512 | load 冗余值得继续拆解 |

粗略按 warp 指令拆解 global load：state 两趟读取约占 32,768 条，重复的 q broadcast load 约占 16,384 条，其余来自 k、v、g 和 beta。这个估算用于提出假设，最终仍要用 Source/SASS counter 验证，不能把估算写成实测事实。

## 6. 根因判断

基线证据组成了一条完整因果链：

```text
32-block 小 grid
    -> 不能覆盖 56 个 SM，也没有足够 warp 隐藏延迟
    -> 每个线程又连续等待两趟 state load
    -> long scoreboard 占 79.48%，94.63% cycle 无 eligible warp
    -> DRAM 只有 25.15%、SM 只有 5.02%，两者都无法被喂满
    -> kernel 是“低并行度下的访存延迟受限”，不是带宽饱和或算力受限
```

因此首要目标不是单纯减少一条算术指令，而是制造更多可独立调度的工作，并缩短单线程串行依赖链。冗余计算和 q load 优化风险低，可以先做，但预期收益应保持克制；真正可能改变瓶颈的是二维 tile 和 state 单次读取。

## 7. 为什么暂不优先做这些方案

### 7.1 不先做 Tensor Core

GDN step 是带 recurrent state 原位更新的 rank-1 递推，不是规则的大矩阵 GEMM。当前数据和累加又是 FP32。强行改写为 MMA 会引入布局变换、同步和精度取舍，不能解决 32-block 小 grid 的根因。Tensor Core 更适合后续多 token 的 chunked prefill GEMM。

### 7.2 不先做 `float4` 向量化

当前一个 warp 在固定 K 行上读取连续 V 列，访问已经合并。直接把一个线程改为处理 4 列会把线程数从 128 降到 32，进一步减少 warp 数；即使指令数下降，也可能恶化延迟隐藏。向量化只能在二维 tile 已经提供足够并行度后作为独立实验。

### 7.3 不先压寄存器

基线只有 40 registers/thread、没有 local load/store，且 theoretical occupancy 已是 100%。`--maxrregcount` 可能制造 spill，却不会凭空增加 grid 中的 block，所以不符合 counter 证据。

### 7.4 不把 DRAM 百分比越高视为越好

若 state 从两读一写降到一读一写，总流量和 kernel 时间可能同时下降，此时 DRAM 利用率也可能下降。评价标准是正确性和绝对延迟，counter 用于解释结果，而不是追求某个百分比最大化。

### 7.5 不用简单的 64-thread 拆分冒充并行化

把每个 128-thread block 拆成两个 64-thread block，如果总 warp 数和每条 warp 的工作量基本不变，只是在 block 之间重新分配工作，并没有解决 K 维串行链。有效方案必须增加可独立推进的 K/V tile 或减少每线程的依赖长度。

## 8. 分阶段优化计划与假设

每个可独立说明的优化形成一个 commit。每轮先做正确性，再采 CUDA Event，最后用 NCU 解释；没有稳定收益的版本会记录后回退，不把失败数据删除。

### 阶段 A：加强 GDN 正确性护栏

基线 microbenchmark 每次把 state 清零，只验证单步输出，无法充分覆盖 recurrent state 在非零条件下的读改写。优化前先增加：

- 随机/确定性的非零初始 state；
- 连续 16 步和 128 步 CPU/CUDA state 与 output 对照；
- `num_v_heads:num_k_heads = 2:1` 的 4B 实际形状；
- 保留 generic shape 和已有 tiny/真实模型回归。

这是测试增强，不预期改变性能。它保护后续的循环重排、shared reduction 和数值重结合。

### 阶段 B：消除低风险冗余

候选修改：

1. 每个 block 只由一个线程计算 `decay=exp(g[h])` 并加载 `beta[h]`，写入 shared 后广播。4B 下 `exp` 调用理论上可由 `32 * 128 = 4096` 次降为 32 次；
2. 在已有 `k` shared cache 后追加 `q` 或 `q_scaled` cache，去掉第二趟循环中每个 warp 的 q broadcast global load；
3. 将第一次循环改成 `kv_raw += k * S`，循环外再乘一次 `decay`；
4. 将固定的 `q_scale` 移出内层，或预先进入 `q_scaled`。

实验结果：global load 从 49,664 降到 33,216（−33.1%），总指令从 310,016 降到 244,096（−21.3%），说明代码改动达到了“消除冗余”的局部目标。但 shared load 增到 24,832，long-scoreboard 等待由 15.05 增至 20.41 cycles/issue，NCU duration 由 26.880 增至 28.384 us。同一时段 200 次 Event median 由原 kernel 的 23.552 增至 26.624 us（慢 13.0%）。因此 R2 被回退：减少指令不等于缩短关键路径，原 q broadcast 经过 cache 后的代价低于新增 shared dependency。

### 阶段 C：二维 K/V tile，提高真实并行度

初始原型为：

```text
block = (32 V columns, 4 K lanes) = 128 threads
grid  = (4 V-column tiles, 32 heads) = 128 blocks
```

实际比较了 `64x2`、`32x4`、`16x8`、`8x16` 和 `4x32`。最终选择 `8x16`：每个 block 仍为 128 threads，每个线程只处理 8 个 K 元素，grid 为 `16 V tiles * 32 heads = 512 blocks`。每组 8 个相邻线程访问连续 state columns，K lanes 之间用 shared memory 做确定性 reduction，不使用 atomic。

结果符合“增加可调度工作”的核心假设。代表性 NCU 报告中 Waves/SM 从 0.05 增至 0.76，achieved occupancy 从 7.71% 增至 57.39%，eligible warps/scheduler 从 0.054 增至 0.171，DRAM 吞吐从 123.41 增至 350.94 GB/s。虽然 reduction 和每个 tile 重复加载使总指令与 global load 增加，绝对 duration 仍下降约一半，证明基线的首要问题确实是并行度而不是指令数量。

### 阶段 D：把 state 两次读取降为一次

这一阶段应建立在二维 tile 之上。两种候选：

- **寄存器暂存**：每个线程持有自己的 state 小片段，完成 `kv_mem` reduction 后直接用暂存值更新；
- **shared state tile**：例如 `128 x 32` FP32 tile 需要 16 KiB/block，先读入、reduce，再更新并写回。

预期：state logical traffic 从“两读一写”降为“一读一写”，global load 指令明显减少。寄存器版的硬门槛是 local load/store 仍为 0；shared 版需要观察 occupancy、bank conflict、barrier stall。任何因为 spill 或同步导致的延迟回退都应拒绝。

### 阶段 E：4B shape specialization

若通用 kernel 的动态维度、除法或边界分支仍有可见成本，为 `nk=16, nv=32, kd=128, vd=128, v_per_k=2` 增加模板 specialization，并保留 generic fallback，不能破坏 Qwen3.5-0.8B/2B 等其他配置。

### 阶段 F：GDN 周边融合

在核心 state kernel 收敛后，再独立评估 decay、beta 或相邻归一化/门控的融合，目标是减少小 kernel launch 和中间向量读写。融合会扩大正确性边界，不与核心 kernel 重排放在同一 commit，避免无法归因。

## 9. 每轮实验记录

### 9.1 总表

| 轮次 | Commit | 主要变化 | Event median | Event p95 | NCU duration | Achieved occupancy | Eligible | Long scoreboard | Registers | Local ld/st | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| R0 | `304eea3` | 原始一列一线程、state 两趟扫描 | 16.384 us | 24.256 us | 26.880 us | 7.71% | 5.37% | 79.48% | 40 | 0 / 0 | 优化前基线 |
| R1 | `test(qwen35): strengthen recurrent GDN coverage` | 非零 state、多步正确性护栏 | 不变 | 不变 | 不适用 | 不适用 | 不适用 | 不适用 | 不适用 | 不适用 | 已完成；55/55 GTest、4/4 CTest |
| R2 | 未提交，已回退 | 标量/q cache 与循环不变量外提 | 26.624 us | 28.672 us | 28.384 us | 8.43% | 4.00% | 78.73% | 40 | 0 / 0 | 退化 13.0%，拒绝 |
| R3 | `89224fe` | 4B `8x16` 二维 K/V tile | 16.848–17.408 us | 23.552–26.624 us | 12.960–13.344 us | 53.77–57.39% | 11.80–12.00% | 73.12–77.53% | 35 | 0 / 0 | 接受；NCU 约 2.01–2.07× |
| R4 | 待提交 | state 单次读取 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待完成 |
| R5 | 待提交 | shape specialization / 周边融合 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待完成 |

### 9.2 单轮记录模板

以后每完成一个版本，在这里追加，而不是覆盖上一版：

```text
轮次与 commit：
改动范围：
优化假设：
代码映射变化：
正确性：CPU/CUDA max abs/rel error；16/128-step state/output；CTest；真实 4B
CUDA Event：warmup/repeat；median；p95；相对 R0；相对上一轮
NCU：duration；grid/block；waves/SM；achieved occupancy；eligible/no eligible；
     long scoreboard；DRAM；L1/L2；global load/store；registers；local ld/st
证据是否支持假设：
副作用与适用范围：
接受、继续调整或回退：
报告目录与 SHA-256：
```

### 9.3 R1：优化前正确性护栏

R1 没有修改 kernel，也没有声称取得性能收益。新增测试直接比较 CPU 与 CUDA 的 output 和完整 recurrent state：

- 通用 grouped shape：`nk=2, nv=4, kd=32, vd=48`，非零初始 state，在第 1、16 步检查；
- Qwen3.5-4B 实际 shape：`nk=16, nv=32, kd=vd=128`，非零初始 state，在第 1、16、128 步检查；
- q/k 按 head 归一化，g、beta、v 和 state 使用确定性非平凡数据；
- 误差门槛为 `2e-4 * max(reference_scale, 1)`，并检查所有 CUDA 结果为有限值；
- focused GDN 测试 3/3 通过，完整 GTest 55/55、CTest 4/4 通过。

这一轮补上了原 benchmark “零 state、单步”无法覆盖的递推风险。后续循环重排即使第一步看似正确，只要误差随 state 累积或 state 原位更新发生错误，16/128 步检查都能暴露问题。

### 9.4 R2：减少了指令，但 kernel 更慢

R2 将 q、k 和 head scalar 放入 shared memory，并把 decay、q_scale 移出内层循环。结果是一个很有价值的反例：

| 指标 | R0 | R2 | 变化 |
|---|---:|---:|---:|
| Global load instructions | 49,664 | 33,216 | −33.1% |
| Total instructions | 310,016 | 244,096 | −21.3% |
| Shared load instructions | 8,192 | 24,832 | +203.1% |
| Long scoreboard cycles/issue | 15.05 | 20.41 | +35.6% |
| NCU duration | 26.880 us | 28.384 us | +5.6% |
| 同时段 Event median（200 samples） | 23.552 us | 26.624 us | +13.0% |

结论是原 q global broadcast 的 cache 行为已经较好；替换成 shared load 后新增的依赖和 shared 指令反而拉长关键路径。该版本源码已回退，报告和同一时段 A/B JSON 保存在 `/home/tuesday/workspace/icd/profiles/KuiperLLama/qwen35/4b-stage2-gdn-rejected-r2-5925f79/`。

### 9.5 R3：二维 tile 解决低并行度

所有候选均使用 128 threads/block。Event 探索采用 20 warmup、200 samples：

| V tile × K lanes | Grid blocks | Median | p95 | 判断 |
|---|---:|---:|---:|---|
| R0 原 kernel | 32 | 23.552 us | 24.576 us | 同一时段对照 |
| `64x2` | 64 | 29.056 us | 30.720 us | block 仍不足，拒绝 |
| `32x4` | 128 | 19.472 us | 20.544 us | 有收益 |
| `16x8` | 256 | 17.408 us | 22.496 us | 继续改善 |
| `8x16` | 512 | 17.408 us | 19.456 us | p95 更好，NCU 最优，接受 |
| `4x32` | 1024 | 20.144 us | 29.344 us | reduction/重复工作超过收益，拒绝 |

`8x16` 相对同时段 R0 的 Event median 为 `23.552 / 17.408 = 1.35x`，延迟下降 26.1%。正式提交后的三组 5/30 median 为 17.408、16.848、17.408 us；历史 R0 单组为 16.384 us，因此不能声称短 Event 的跨时段结果稳定优于历史基线。200-sample 稳定性组 median/p95 为 17.408/19.072 us。

三份提交后 NCU 报告的 duration 为 13.344、13.152、12.960 us；相比 R0 的 26.880 us 为 2.01–2.07×。首份报告的 derived long-scoreboard 百分比为异常的 108.25%，原始报告仍保留，但该派生项不用于结论；另外两份为 73.12% 和 77.53%。选择中间一份 `gdn-run2` 展示其余 counter：

| 指标 | R0 | R3 `gdn-run2` | 变化 |
|---|---:|---:|---:|
| NCU duration | 26.880 us | 13.152 us | −51.1%，2.04× |
| Grid blocks | 32 | 512 | 16× |
| Waves/SM | 0.05 | 0.76 | 15.2× |
| Achieved occupancy | 7.71% | 57.39% | +49.68 pp |
| Eligible warps/scheduler | 0.054 | 0.171 | 3.18× |
| DRAM bandwidth | 123.41 GB/s | 350.94 GB/s | 2.84× |
| Global load instructions | 49,664 | 52,736 | +6.2% |
| Total instructions | 310,016 | 466,944 | +50.6% |
| Registers/thread | 40 | 35 | −5 |
| Local load/store | 0 / 0 | 0 / 0 | 无 spill |

这里最重要的结论是：R3 做了更多 reduction/shared 指令，却用更短时间完成，因为更多 block 和更短的单线程 K 链让 GPU 能并发发出更多 state memory request。优化的是可执行并行度和延迟隐藏，不是静态指令数。

正式报告目录：

```text
/home/tuesday/workspace/icd/profiles/KuiperLLama/qwen35/4b-stage2-gdn-after-89224fe/
├── cuda-events/
└── ncu/
```

报告 SHA-256：

| 报告 | SHA-256 |
|---|---|
| `gdn.ncu-rep` | `d78ef53e8175c0ce23069fa7e6e0a12cee7ed26b2aad25a6db4032fc886b8c07` |
| `gdn-run2.ncu-rep` | `419c533e7518d372aff41df804b3746a4628596df4a7a3a750b05a593e63f8b7` |
| `gdn-run3.ncu-rep` | `ae56e266eee43c310151855daef39ca202d552520b9d8a03abe24376c13c1c19` |

## 10. 前后结果表（优化完成后填写）

| 指标 | R0 Before | Final After | 变化 | 如何解释 |
|---|---:|---:|---:|---|
| CUDA Event median | 16.384 us | 待测 | 待测 | 最终主要性能结论 |
| CUDA Event p95 | 24.256 us | 待测 | 待测 | 稳定性，不隐藏长尾 |
| NCU duration | 26.880 us | 待测 | 待测 | 插桩环境中的同口径对照 |
| Grid blocks | 32 | 待测 | 待测 | 是否增加真实并行度 |
| Achieved occupancy | 7.71% | 待测 | 待测 | 是否有更多 active warp |
| One or more eligible | 5.37% | 待测 | 待测 | scheduler 是否更常有指令可发射 |
| No eligible | 94.63% | 待测 | 待测 | 延迟隐藏是否改善 |
| Long scoreboard | 79.48% | 待测 | 待测 | memory dependency 是否缓解 |
| Global load instructions | 49,664 | 待测 | 待测 | q cache/state 单读是否生效 |
| Registers/thread | 40 | 待测 | 待测 | 资源代价 |
| Local load/store | 0 / 0 | 待测 | 待测 | 必须避免 spill |
| CPU/CUDA error | max abs 0 | 待测 | 待测 | 数值正确性 |

加速比统一写成：

```text
speedup = before_median / after_median
latency_reduction = (before_median - after_median) / before_median * 100%
```

不要把百分比下降和“提升倍数”混写，也不要从单次最小值计算加速比。

## 11. 正确性与接受标准

一个版本只有同时满足以下条件才接受：

1. 非零初始 state 下单步和多步 CPU/CUDA output、最终 state 均在约定误差内；
2. 现有 CTest 全部通过，内部 GTest、tiny fixture 和真实 4B trace/inference 无回归；
3. CUDA Event 使用相同输入、build、warmup/repeat 和设备条件，重复采样仍有稳定收益；
4. NCU counter 能支持优化假设，而不是只看到一次偶然的 latency 波动；
5. local load/store 保持为 0，除非有充分数据证明少量 spill 仍带来稳定净收益；
6. 4B specialization 必须保留 generic fallback；
7. `.ncu-rep`、raw CSV、命令、环境与 SHA-256 全部归档。

若结果不稳定或变慢，记录“失败原因/新认识”后回退代码。失败实验能体现性能工程中的证据闭环，比只保留成功版本更有面试价值。

## 12. 复现实验命令

### 12.1 Build 与测试

```bash
source tools/env.sh
cmake -S . -B build -DUSE_CPM=ON -DQWEN35_SUPPORT=ON \
  -DKUIPER_ENABLE_NVTX=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCUDAToolkit_ROOT="$CUDA_HOME"
cmake --build build --target qwen35_bench test_llm -j2
ctest --test-dir build --output-on-failure --timeout 300
```

### 12.2 未插桩 CUDA Event

```bash
./build/demo/qwen35_bench \
  --mode gdn --device cuda --warmup 5 --repeat 30 \
  --output /tmp/qwen35-gdn.json
```

正式对照应连续采集多组，保留原始 JSON，不只抄录最好的一组。

### 12.3 Nsight Compute

```bash
benchmarks/qwen35/scripts/profile_ncu.sh \
  /tmp/qwen35-ncu/gdn '.*gated_delta_step_kernel.*' -- \
  ./build/demo/qwen35_bench \
    --mode gdn --device cuda --warmup 0 --repeat 1

benchmarks/qwen35/scripts/summarize_ncu.py /tmp/qwen35-ncu
```

用 UI 查看报告：

```bash
ncu-ui /tmp/qwen35-ncu/gdn.ncu-rep
```

每轮最终报告应从 `/tmp` 复制到仓库外的持久化 profile 目录，并按 `before/after + commit` 命名；二进制 `.ncu-rep` 不提交 Git。

## 13. 面试时如何解释结果

建议按“问题—证据—假设—实验—结果—取舍”讲，而不是逐行介绍 CUDA 代码：

1. **问题**：原实现正确，但 Qwen3.5-4B GDN 只有 32 blocks，每个线程串行扫描 state 两次；
2. **证据**：低 achieved occupancy、极低 eligible warp、高 no-eligible 和 long-scoreboard，同时 SM/DRAM 都未饱和；
3. **假设**：主要问题是并行度不足导致访存延迟无法隐藏，其次才是重复 q/scalar load 和 state 二次读取；
4. **实验**：把低风险消冗余、二维 tile、单读 state 分成独立版本，对每版做 Event + NCU + correctness；
5. **结果**：填写最终加速比，并用 eligible warp、long scoreboard、global load 和 spill 情况解释；
6. **取舍**：说明为何没先上 Tensor Core/`float4`/寄存器限制，以及为何保留 generic fallback；
7. **系统视角**：GDN 是自定义递推算子的代表案例，但 4B decode 的大 GEMV/LM head 仍可能占更大绝对时间，不能把单算子收益夸大为端到端同等收益。

面试官可能追问：

- **为什么 theoretical occupancy 是 100%，性能还差？** 因为资源允许驻留不等于 launch 提供了足够 block；本例只有 32 blocks。
- **为什么 DRAM 只有 25.15%，还说在等内存？** 带宽没打满不代表没有 memory latency。warp 数太少时，请求之间缺少并发，单个 warp 会长时间等待 load 返回。
- **为什么 state 单读不是第一步？** 它需要跨 K lane reduction 后仍保留旧 state，可能显著增加寄存器/shared memory 和同步；先建立测试护栏和二维映射更容易隔离风险。
- **为什么不用 Tensor Core？** decode GDN 是 FP32 recurrent rank-1 update，不是适合 MMA 的规则 GEMM；Tensor Core 放在 chunked prefill 更合理。
- **如何证明不是测量噪声？** 相同代码/输入/环境，多组 Event median/p95；NCU 只解释原因；保留命令、原始结果和报告哈希。

## 14. 文档维护规则

后续每完成一次 GDN 优化，都要在同一个功能 commit 中同步更新本文：

- 把 commit、代码映射和优化假设写入“每轮实验记录”；
- 写明所有测试和误差，不只写“正确”；
- 填入 Event median/p95 和相对 R0、相对上一轮的变化；
- 填入能证明或否定假设的 NCU counter；
- 记录报告路径和 SHA-256；
- 明确版本是接受、继续调整还是回退；
- 只有最终方案确定后，才更新开头的面试概括和“前后结果表”；
- 未测数字始终标为“待测”，不得用预期收益代替实测结果。

这套记录方式确保最终展示的不只是“我把 kernel 写快了”，而是一个可复现的算子性能工程闭环：先测量和定位，再提出可证伪的假设，控制变量实验，验证正确性，最后用硬件 counter 解释收益与代价。
