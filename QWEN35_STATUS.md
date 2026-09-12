# Qwen3.5 Dense 支持：设计、现状与待办

目标：用本框架跑通并重点优化 Qwen3.5-4B 纯文本推理。9B 因本机内存限制暂缓。

本文记录已完成的设计与实现、已验证到什么程度、存在哪些缺陷，以及达成目标还差什么。

**当前状态**：Qwen3.5-0.8B 的 FP32 和 BF16-matrix 两种真实权重路径均已在 CPU
上与 Transformers FP32/eager 逐层对齐；Qwen3.5-4B 的真实 BF16 权重也已完成
导出、Kuiper 端到端推理和 Transformers BF16 参考比较，覆盖了 4B 特有的 GDN
v:k=2:1 分组。两种模型的前 10 个 greedy token 均完全一致。4B checkpoint 为
8.41 GB，Kuiper CPU 运行峰值内存约 8.0 GiB。当前共定义 55 个 GTest；RTX 4070
SUPER 上真实 4B CUDA 推理和 tiny CPU/CUDA 对齐均已跑通，完整测试结果为
55/55 passed；真实 4B CUDA 的逐层 hidden、final norm、完整 logits 和生成 token 也已
分别与 Kuiper CPU、Transformers BF16 对齐。详细结果见
[`QWEN35_PROJECT_REPORT.md`](QWEN35_PROJECT_REPORT.md)。

当前开发重点已调整为 4B 核心 CUDA 算子：使用 Nsight Compute 对 GDN 与 BF16
GEMV/MatMul 做优化前后对照。9B 独立 `lm_head`、INT8 暂停，长 prompt 分块 prefill
仍未实现。GDN 已完成 512-block 二维 tile 和 state 单读优化，三次正式 NCU 为
2.47–2.69×；完整过程见
[`benchmarks/qwen35/GDN_OPTIMIZATION_PROCESS.md`](benchmarks/qwen35/GDN_OPTIMIZATION_PROCESS.md)。
BF16 线性层已进入优化阶段；decode GEMV 与后续并行 prefill GEMM 的边界、基线指标和
分轮计划见
[`benchmarks/qwen35/BF16_GEMV_GEMM_OPTIMIZATION_PROCESS.md`](benchmarks/qwen35/BF16_GEMV_GEMM_OPTIMIZATION_PROCESS.md)。

---

## 1. 架构调研结论

一个必须先纠正的判断：**Qwen3.5-Dense 不是 Qwen3 架构**。最初基于搜索标题推测「架构一致、无需新增算子」，是错的。以下事实全部来自实测 `Qwen/Qwen3.5-{0.8B,2B,4B,9B}` 的 `config.json` 与 safetensors 头部张量形状。

### 1.1 它是多模态 checkpoint

```
architectures: ["Qwen3_5ForConditionalGeneration"]
model_type:    qwen3_5
```

含 `vision_config` + `text_config`。权重前缀：

| 前缀 | 内容 | 本实现 |
|---|---|---|
| `model.language_model.*` | 文本塔 | ✅ 使用 |
| `model.visual.*` | 24 层 ViT | ❌ 跳过 |
| `mtp.*` | multi-token prediction | ❌ 跳过 |
| `lm_head.weight` | 输出头（仅 9B 有） | ✅ 使用 |

按你的决策，只做文本塔。

### 1.2 混合注意力（核心差异）

`full_attention_interval: 4`，`layer_types` 为 `[linear, linear, linear, full] × N`：

| 模型 | 层数 | linear | full | hidden | inter | heads/kv | head_dim |
|---|---|---|---|---|---|---|---|
| 0.8B | 24 | 18 | 6 | 1024 | 3584 | 8/2 | 256 |
| 2B | 24 | 18 | 6 | 2048 | 6144 | 8/2 | 256 |
| **4B** | **32** | **24** | **8** | **2560** | **9216** | **16/4** | **256** |
| **9B** | **32** | **24** | **8** | **4096** | **12288** | **16/4** | **256** |

**3/4 的层是 Gated DeltaNet 线性注意力**，框架原先完全没有对应实现。这是工作量的主体。

### 1.3 full_attention 层（实测 4B 形状）

```
q_proj  [8192, 2560]   = 16 heads × 256 × 2   ← 后一半是输出门
k_proj  [1024, 2560]   v_proj [1024, 2560]    = 4 kv heads × 256
q_norm  [256]          k_norm [256]           = head_dim，QK-norm
o_proj  [2560, 4096]
```

与 Qwen3 的差异：
- `attn_output_gate: true` → `attn_out *= sigmoid(gate)`
- `partial_rotary_factor: 0.25` → RoPE **只旋转 head_dim 的前 64 维**，后 192 维原样通过
- `rope_theta: 1e7`（Qwen3 是 1e6）
- `head_dim=256` 独立于 `hidden_size/head_num`（2560/16=160 ≠ 256）
- QK-norm 保留（这一点与 Qwen3 相同）

### 1.4 linear_attention 层（Gated DeltaNet，实测 4B）

```
in_proj_qkv  [8192, 2560]  = k_dim(16×128)×2 + v_dim(32×128) = 2048×2+4096
in_proj_z    [4096, 2560]  = v_dim，gated RMSNorm 的门
in_proj_a    [32, 2560]    in_proj_b [32, 2560]  = num_v_heads
conv1d.weight[8192, 1, 4]  depthwise 因果卷积，groups=conv_dim，k=4
A_log [32] fp32            dt_bias [32]
norm.weight  [128] fp32    = v_head_dim，gated RMSNorm
out_proj     [2560, 4096]
```

`linear_num_key_heads=16`、`linear_num_value_heads=32` → **每个 k head 带 2 个 v head**（0.8B/2B 是 1:1，4B/9B 是 2:1，实现必须支持分组）。

递推式（对齐 transformers `torch_recurrent_gated_delta_rule`）：

```
mixed = silu(causal_depthwise_conv1d(in_proj_qkv(x), w, k=4))
q,k,v = split(mixed, [2048, 2048, 4096])
q = l2norm(q, eps=1e-6) / sqrt(128);   k = l2norm(k, eps=1e-6)
beta  = sigmoid(in_proj_b(x))
g     = -exp(A_log) * softplus(in_proj_a(x) + dt_bias)

每个 v head 维护状态 S[128,128]：
  S      = S * exp(g_t)
  kv_mem = kᵀ·S
  S     += outer(k_t, (v_t - kv_mem) * beta_t)
  out_t  = Sᵀ·q_t

out = rmsnorm(out_t) * norm.w * silu(z);   y = out_proj(out)
```

关键：**状态大小与序列长度无关**。4B 的 24 个 linear 层共 `24 × 32 × 128 × 128 × 4B ≈ 50MB`，替代了这些层本该有的 KV cache。

Qwen3.5 把投影拆成独立的 `in_proj_qkv/z/a/b`（Qwen3-Next 是融合的 `in_proj_qkvz/in_proj_ba`），数学等价但**更好导出** —— 不需要做 head 内交错重排。

### 1.5 其他

- `tie_word_embeddings`: 0.8B/2B/4B 为 true（无 `lm_head`，复用 embedding）；9B 为 false
- `vocab_size: 248320`；tokenizer 有 **248044** 个基础 token 和 26 个特殊 token，
  有效 ID 连续覆盖 `[0, 248070)`，其余 embedding 行是 padding
- `rms_norm_eps: 1e-6`，MLP 仍是 SwiGLU
- `max_position_embeddings: 262144`
- 特殊 token：`<|im_start|>=248045`、`<|im_end|>=248046`、`<|endoftext|>=248044`

---

## 2. 环境升级（CUDA 11.5 → 12.8）

原环境 CUDA 11.5 是两个独立故障的共同根因：

| 故障 | 原因 |
|---|---|
| `nvcc fatal: Value 'sm_89' is not defined` | 11.5 最高支持 `compute_87`，认不了 4070 SUPER 的 sm_89 |
| 所有 `.cu` 编译失败（`std_function.h` 参数包报错） | 11.5 的 nvcc 解析不了 GCC 11 的 `<functional>`。**项目原有 CUDA 文件同样过不去** |

曾临时降到 `sm_86` 靠 PTX JIT 运行，是妥协，已废弃。

**处置**（全部用户目录安装，无 sudo，未动系统驱动）：

| 组件 | 版本 | 位置 |
|---|---|---|
| CUDA toolkit | 12.8.93 | `~/.local/opt/cuda-12.8` |
| cmake | 3.28.6 | `~/.local/opt/cmake` |

驱动 610.62 支持到 CUDA 13.3，无需改动。环境变量固化在 [`tools/env.sh`](tools/env.sh)。

顺带修了 CMake 两处在非标准 CUDA 路径下会失败的硬编码：

- `CMakeLists.txt` 无条件 `set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)`，普通变量优先于缓存变量，会覆盖命令行 `-D`。改为仅在该路径存在且未显式指定时设置。
- `cmake/cuda.cmake` 的 `CUDA_DETECT_INSTALLED_GPUS` 会覆盖显式指定的 `CMAKE_CUDA_ARCHITECTURES`。改为仅在未指定时探测。

### 构建方式

```bash
source tools/env.sh
cmake -B build -DUSE_CPM=ON -DQWEN35_SUPPORT=ON -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
      -DCMAKE_CUDA_ARCHITECTURES=89 -DCUDAToolkit_ROOT=$CUDA_HOME
cmake --build build -j$(nproc)
```

---

## 3. 实现内容

按你的要求：**保留原有算子，新增 Qwen3.5 专用算子**，不改动任何现有 kernel 的语义。

### 3.1 新增算子（10 个 Layer 类型，CPU + CUDA 双实现）

全部在独立文件：`kuiper/source/op/kernels/{cpu/qwen35_kernel.{h,cpp},cuda/qwen35_kernel.{cuh,cu}}`

| 算子 | 用途 | 为何不复用现有 |
|---|---|---|
| `l2norm` | GDN 的 q/k 归一 | 与 rmsnorm 不同：无 weight、不除以维度、按 `sqrt(Σx²+eps)` |
| `zero_centered_rmsnorm` | decoder/final norm 与逐 head QK-norm，`rmsnorm(x)*(1+w)` | Qwen3.5 的普通 RMSNorm 权重以 0 为中心，且必须使用配置中的 epsilon；现有 RMSNorm 是 `*w` 且 epsilon 由编译宏决定 |
| `gated_rmsnorm` | `rmsnorm(x)*w*silu(z)` | 现有 rmsnorm 无门控 |
| `causal_conv1d_decode` | GDN 前的深度可分离因果卷积 + 状态管理 | 框架无卷积算子 |
| `gated_delta_step` | **GDN 递推核心** | 全新，有状态串行递推 |
| `softplus_decay` | `g = -exp(A_log)*softplus(a+dt_bias)` | 全新 |
| `rope_partial` | 只旋转前 `rotary_dim` 维 | 现有 `rope_kernel` 假定整个 head 旋转，且配对方式是 `(2i,2i+1)`，而 HF 用 rotate-half 的 `(i,i+half)`，**两者不可互换** |
| `split_head_interleaved` | 拆分 q_proj 的 `[head][q\|gate]` 交错布局 | 全新（见 4.3 的 bug） |
| `sigmoid` / `mul` | beta、输出门 | 框架无独立实现 |

### 3.2 Layer 封装

`kuiper/include/op/qwen35_ops.h` + `kuiper/source/op/qwen35_ops.cpp`，10 个 Layer 类。

每个类都加了 `using Layer::forward;` —— 派生类声明 `forward()` 会**隐藏基类的多参数重载**（C++ 名字隐藏），这是编译期踩到的坑。项目原有代码靠「统一用 `shared_ptr<op::Layer>` 存放」规避，本实现两种手段都用。

### 3.3 模型层

| 文件 | 内容 |
|---|---|
| `kuiper/include/model/qwen35_config.h` | `Qwen35RawConfig`（磁盘头）+ `Qwen35Config`（运行时，含派生尺寸与 `layer_type()`/`type_local_idx()`） |
| `kuiper/include/model/qwen35.h` | `Qwen35Layers`、`Qwen35Buffer`、`Qwen35Model` |
| `kuiper/source/model/qwen35.cpp` | 权重加载、两类层的 forward、状态管理 |

设计要点：

**层分派**：`forward()` 按 `layer_type(i)` 调 `attention_full` 或 `attention_linear`；norm 与 FFN 两类层共用。

**按类型局部索引**：`type_local_idx()` 把绝对层号映射为「同类层内的序号」。KV cache 只为 8 个 full 层分配（4B 下省 3/4 显存），recurrent state 只为 24 个 linear 层分配。这是混合架构的关键收益。

**状态管理**：`reset_state()` 清零 recurrent + conv state。GDN 状态跨位置累积，**新序列开始前必须清零**，否则会串味 —— 与 KV cache「按位置覆盖」的语义根本不同。demo 与测试都显式调用。

**布局断言**：`create_param_layers()` 结尾有
```cpp
CHECK_EQ(pos + header_size_, raw_model_data_->file_size)
```
消耗的权重字节数必须正好落在文件末尾。导出器与读取器一旦漂移立即失败，而不是读到垃圾数据后静默产出乱码。这是最有价值的一道防线。

**vocab 修正**：`create_encode_layer()` 会用 tokenizer 的 248044 覆盖 `config_->vocab_size_`，而 embedding 是 248320。在 `gen_model_from_file()` 里恢复为 header 值，否则采样会越界读。

### 3.4 导出器

[`tools/export_qwen35/export.py`](tools/export_qwen35/export.py)

- 直读 safetensors，**不依赖 torch / transformers / numpy**（本机无 pip，这是必要约束）
- 只取 `model.language_model.*`，按 `layer_types` 分流两类层
- `--weight_dtype bf16|fp32` 控制 embedding/二维投影矩阵的存储类型，默认 BF16；
  norm、conv、decay 等小参数和全部运行时激活/状态仍保持 FP32
- `--dry_run` 校验全部张量存在且形状正确，不写文件
- 多 shard 懒加载，一次只驻留一个 shard 的 header

磁盘格式 v3（magic `K35D`）：19 个 int32 + 2 个 float 的头，新增矩阵权重类型，
随后按 `create_param_layers` 的顺序排列混合精度权重。读取器仍兼容全 FP32 的 v2
checkpoint。原有 7-int `ModelConfig` 描述不了混合模型，故另立版本化格式。

### 3.5 初始接入时修改的既有文件（8 个，共 +52/-14 行）

| 文件 | 改动 | 性质 |
|---|---|---|
| `CMakeLists.txt` | 加 `QWEN35_SUPPORT` 选项；修 nvcc 路径硬编码 | 新增 + 修复 |
| `cmake/cuda.cmake` | 显式架构不被自动探测覆盖 | 修复 |
| `demo/CMakeLists.txt` | 加 `qwen35_infer` 目标 | 新增 |
| `test/CMakeLists.txt` | 条件加 `QWEN35_SUPPORT` | 新增 |
| `kuiper/include/op/layer.h` | `LayerType` 追加 11–20（**尾部追加，原值不变**） | 新增 |
| `kuiper/include/op/encode.h`<br>`kuiper/source/op/encode.cpp`<br>`kuiper/source/model/model.cpp` | 宏条件加 `QWEN35_SUPPORT`，复用 `QwenEncodeLayer` | 新增 |

初始接入未改动任何现有算子的实现或语义。阶段 3 另行修复了所有 GPT-2 byte-level
BPE 共用的空格预处理错误，详见 4.9。

---

## 4. 验证做到了什么程度

### 4.1 算子级：CPU 对参考实现

`/tmp/k35/verify.cpp`，独立参考实现（float64）对比：

| 算子 | max\|diff\| |
|---|---|
| l2norm | 3.0e-08 |
| softplus_decay | 4.8e-07 |
| gated_rmsnorm | 1.2e-07 |
| causal_conv1d_decode（多步递推 vs 显式卷积） | 1.2e-07 |
| gated_delta_step | 1.2e-07 |
| rope_partial | 3.7e-09 |

### 4.2 算子级：CUDA 对 CPU，用 4B/9B 真实尺寸

`/tmp/k35/verify_cu.cu`，尺寸为 `conv_dim=8192`、`nk=16/nv=32/kd=vd=128`、`head_dim=256/rotary_dim=64/theta=1e7`：

| 算子 | max\|diff\| |
|---|---|
| l2norm | 3.0e-08 |
| gated_rmsnorm | 1.8e-07 |
| softplus_decay | 4.8e-07 |
| causal_conv1d(8192) | 3.6e-07 |
| **gated_delta_step（4 步递推）** | **3.1e-06** |
| rope_partial + cache | ≤1.9e-06 |

### 4.3 最关键一项：GDN 对官方实现

`gated_delta_step` 对 transformers 的 `torch_recurrent_gated_delta_rule`（float64 计算），4B 形状、4 步递推、含 v:k=2:1 分组：

```
max|diff| = 1.68e-08    ref absmax = 4.94e-02    相对误差 = 3.39e-07
```

**正确性锚定到了官方参考实现**，不是自证。这是整个工作最重要的一道验证。

### 4.4 导出链路

- bf16→fp32 转换：已知值（1.0/-2.0/0.5/3.140625/0/-1.0）逐个正确
- 落盘数据与源 safetensors **逐字节一致**
- 头部 18int+2float 读回全部字段正确，magic 校验通过

### 4.5 端到端（合成模型）

`test/test_model/test_qwen35.cpp`，当前 11 个配置/模型测试，其中核心测试如下：

| 测试 | 覆盖 |
|---|---|
| `Qwen35Config.LayerTypeAndLocalIndex` | 4:1 分派、按类型局部索引、4B/9B 派生尺寸 |
| `Qwen35Tiny.LoadsOnCpu` | 权重布局断言（导出器 ↔ 读取器对齐） |
| `Qwen35Tiny.ForwardProducesFiniteLogits` | 两类层都跑通，logits 无 NaN/Inf |
| `Qwen35Tiny.ResetStateMakesRunsReproducible` | GDN 状态清零正确，重放逐位一致 |
| `Qwen35Tiny.CudaMatchesCpu` | CPU/CUDA 端到端相对误差 **<2e-3** |
| `Qwen35Tiny.SamplingSkipsEmbeddingPadding` | 保留有效特殊 token，排除 `[248070, 248320)` padding 行 |
| `Qwen35Config.ReportsItsOwnModelType` | 模型类型不再误报为 Llama2 |
| `Qwen35Config.RejectsZeroIntervalInModelHeader` | 非法 header 在派生尺寸前返回解析错误，不触发除零 |
| `Qwen35Config.RejectsUnknownMatrixWeightType` | v3 header 中未知矩阵 dtype 返回解析错误 |
| `Qwen35Tokenizer.MatchesTransformersChatPrompt` | fixture 保留真实 token ID，同一 chat prompt 的编码/解码与 Transformers 一致 |

合成模型与 4B **同构**：真实 vocab 248320、head_dim 256、rotary_dim 64、interval 4、v:k=2:1、tie_word_embeddings —— 只缩小 hidden/inter/layers。生成器 [`test/test_model/make_tiny_qwen35.py`](test/test_model/make_tiny_qwen35.py) 已入库，可字节级复现。

另在 `test/test_op/test_qwen35_norm.cpp` 新增 3 个 zero-centered RMSNorm 测试：

- weight 全零时退化为无权重 RMSNorm（用于直接区分 `w` 与 `1+w`）
- 非零 weight 的 `(1+w)` 参考值及 CPU in-place 路径
- CUDA 对 CPU（无 CUDA 设备时 skip）

当前共定义 55 个 GTest，已在 RTX 4070 SUPER 上全部通过，包括 Qwen3.5 tiny 的
CPU/CUDA 端到端对齐、Tensor BF16 存储/转换、BF16 matmul 和 BF16 embedding。

### 4.6 真实 Qwen3.5-0.8B 对 Transformers（阶段 3）

使用 `Qwen/Qwen3.5-0.8B` 的真实 safetensors 导出 3.01 GB FP32 checkpoint，
在 CPU 上与 Transformers 5.6.2 的 FP32/eager 路径对齐。验证工具位于
[`tools/verify_qwen35/`](tools/verify_qwen35/)，会保存两侧的 token、每层 hidden
state、最终 norm、完整 logits 和前 10 个 greedy token。

默认 prompt 经两侧 tokenizer 均编码为 12 个 token。数值结果：

| 对比项 | 结果 |
|---|---|
| decoder 层 0–23 | 最大绝对误差 `1.54972e-05`，无层超过 `2e-3` 相对阈值 |
| final norm | 最大绝对误差 `9.99570e-05`，相对误差 `2.54569e-06` |
| 完整 logits（12×248320） | 最大绝对误差 `7.67708e-05`，平均绝对误差 `6.35322e-06` |
| 前 10 个 greedy token | **完全一致** |

前 10 个 token ID：

```text
[248068, 271, 248069, 271, 332, 9010, 16004, 20736, 318, 15015]
```

解码开头为 `<think>\n\n</think>\n\n**Artificial Intelligence (AI`。这证明 0.8B
路径上的真实权重顺序、混合层调度、GDN 状态、full attention、RoPE、norm、MLP、
KV cache 与自回归状态推进均已对齐，而不再只是合成权重自测。

### 4.7 BF16 权重路径（阶段 4）

阶段 4 采用 **BF16 大矩阵存储 + FP32 激活、累加、KV cache 和 GDN state**：

- `Tensor` 增加 BF16 dtype 和 CPU RNE 转换工具。
- CPU/CUDA matmul 支持 FP32 输入、BF16 权重、FP32 累加和输出。
- CPU/CUDA embedding 支持从 BF16 table 读取并输出 FP32。
- v3 checkpoint 仅压缩 embedding 和全部二维投影；小参数保留 FP32。
- tied embedding 在 CUDA 上只上传一次并与 LM head 共享，4B 避免额外约 1.27 GB。

真实 0.8B 导出大小 `3.01 GB → 1.51 GB`。与同一 Transformers reference 比较：

| 对比项 | BF16-matrix 结果 |
|---|---|
| decoder 层 0–23 | 最大绝对误差 `2.37823e-05` |
| final norm | 最大绝对误差 `1.20163e-04`，相对误差 `3.06030e-06` |
| 完整 logits | 最大绝对误差 `1.01328e-04`，平均绝对误差 `8.89898e-06` |
| 前 10 个 greedy token | **完全一致** |

这里的真实模型矩阵原本就是 BF16，因此并非重新量化造成的近似；误差主要来自
Kuiper 标量/Armadillo 与 PyTorch 的 FP32 累加顺序不同。

### 4.8 真实 Qwen3.5-4B 端到端验证（阶段 4）

通过 `hf-mirror.com` 下载两片官方 safetensors 后，检查了每个张量的 dtype、shape、
数据区间与分片边界；738 个源张量及索引声明的 `9,319,737,856` 字节完全一致。
两片文件的 SHA-256 也分别与官方 LFS 元数据一致：
`26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61` 和
`cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188`。
导出器 dry-run 验证了 Kuiper 使用的 426 个文本塔张量，随后生成 8.413 GB 的 v3
BF16-matrix checkpoint。

默认 12-token prompt 加 10-token greedy generation 的实测结果：

| 项目 | 结果 |
|---|---|
| Kuiper CPU | 75.10 秒，峰值 RSS `8,374,084 KB` |
| Transformers BF16/eager 文本塔 | 35.01 秒，峰值 RSS `9,225,216 KB` |
| decoder 层 0–31 | 最大相对误差 `1.86040e-02`，低于 BF16 对比阈值 `2e-2` |
| final norm | 最大绝对误差 `4.12555e-01`，相对误差 `8.04986e-03` |
| 完整 logits | 最大绝对误差 `4.16996e-01`，平均绝对误差 `2.94676e-02`，相对误差 `1.32907e-02` |
| 前 10 个 greedy token | **完全一致** |

前 10 个 token ID：

```text
[248068, 271, 248069, 271, 332, 15015, 332, 12965, 364, 2972]
```

Transformers BF16 会在层间把激活舍入回 BF16，而 Kuiper 只把大矩阵存为 BF16、
激活与累加保持 FP32，因此这组逐层误差不能使用 0.8B FP32 reference 的 `2e-3`
阈值。生成序列完全一致且所有层低于 `2e-2`，确认真实 4B 的权重顺序、v:k=2:1
分组递推、tied embedding 和自回归状态推进均已跑通。

参考工具现在直接加载 `Qwen3_5ForCausalLM` 文本塔，不再把未使用的 vision/MTP
权重放入内存；`--dtype bf16` 用于本机无法容纳 FP32 4B reference 的场景。

RTX 4070 SUPER（12,282 MiB，CUDA 12.8，sm_89）真机上，8.413 GB checkpoint
成功完成 CUDA 加载和默认 prompt 的 21 步运行；forward/generation 阶段耗时
0.513 秒，约 40.95 steps/s，含加载总耗时 7.22 秒。

`qwen35_trace` 已增加 `--device cpu|cuda`。CUDA 模式在模型自有 stream 上逐层复制
并同步，真实 4B 保存了 32 层 decoder hidden、final norm、完整 logits 和 10 个生成
token。GDN 优化后 CUDA vs Kuiper CPU 的 decoder 最大相对误差为 `2.09229e-05`、logits 为
`2.61240e-06`；CUDA vs Transformers BF16 分别为 `1.86247e-02` 和
`1.32916e-02`，两组比较的 10 个 token 均完全一致。CUDA trace 含加载总耗时
7.29 秒，峰值主机 RSS `8,429,096 KB`。

新增 CUDA launch 检查后，Qwen3.5 专用 kernel 以及 matmul/embedding 的各 dtype
分支都会在 launch 点立即检查 `cudaGetLastError()` 并报告 kernel 名称。另增加
Qwen3.5-4B `in_proj_z [4096, 2560]` 真实尺寸的 BF16 CUDA matmul 专项测试，覆盖
10,485,760 个权重，与 CPU 参考按 `2e-5` 相对阈值比较。RTX 4070 SUPER 上专项测试、
全量 55 项测试和真实 4B CUDA trace 均通过。

### 4.9 已发现并修复的实现错误

**q_proj 的门拆分（严重）**。核对官方 `modeling_qwen3_5.py` 发现：

```python
query_states, gate = torch.chunk(
    self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
```

先 view 成 `(num_heads, head_dim*2)` 再按最后一维切分，内存布局是**按 head 交错**：

```
[h0_q(256) | h0_gate(256) | h1_q(256) | h1_gate(256) | ...]
```

我原先当作了「连续两半」`[全部q(4096) | 全部gate(4096)]`，会把不同 head 的 query 和 gate 混在一起 —— 除 head 0 外全错。已新增 `split_head_interleaved` 算子修正，并加了专项测试（含 head1 的值校验）。

同时核实**通过**的两处：
- partial RoPE：官方 `emb=cat((freqs,freqs))` 使 cos 宽 64 且 `cos[i]==cos[i+32]`，与我「缓存 32 个值、配对 (i,i+32)」一致；`apply_rotary_pos_emb` 按 `cos.shape[-1]` 切出 `q_rot/q_pass`，与我「只旋转前 rotary_dim」一致
- mRoPE：纯文本下三个维度的 `position_ids` 相同，`apply_interleaved_mrope` 退化为标准 RoPE

**普通 RMSNorm 的 zero-centered 权重（严重，阶段 1 已修复）**。官方
`Qwen3_5RMSNorm` 的计算是 `rmsnorm(x) * (1 + weight)`，不是现有
Llama/Qwen2 RMSNorm 的 `rmsnorm(x) * weight`。此前 input/post/final norm 和
full-attention Q/K norm 都会读错真实权重；而且只启用 `QWEN35_SUPPORT` 时，旧
RMSNorm 会使用 `1e-5`，不是配置中的 `1e-6`。现已新增专用
`ZeroCenteredRMSNormLayer`，单行和逐 head 共用，CPU/CUDA 都显式接收
`q35_.rms_norm_eps`。GDN 的 gated RMSNorm 仍保留普通 `weight` 乘法语义。

合成模型也已同步修正：普通 norm 权重在 0 附近，GDN gated norm 权重在 1 附近，
避免继续用错误的合成权重掩盖该问题。

**阶段 2 安全性修复（已完成）**。

- CPU GDN 的固定 `float delta[512]` 改为按 `v_head_dim` 分配的一行复用缓冲，
  不再静默忽略第 512 列之后的 state/output；新增 513 维回归测试。
- Qwen3.5 采样上界改为 tokenizer 的有效 ID 数 248070，而 logits/embedding
  仍保持 248320 宽；因此特殊 token 248044–248069 可正常生成，padding 行不会
  被选中。
- 新增 `kModelTypeQwen35`，模型不再复用 `kModelTypeLLama2`。
- 模型文件头的所有除数和关键尺寸在 `derive()` 前校验，并给 `derive()` 的 interval
  增加防御性保护，非法文件返回 `kModelParseError` 而不是触发除零。
- CUDA argmax 的异步 D2H copy 现在会在读取栈上结果前同步 stream，并释放每次
  采样申请的 device index，避免结果竞争和逐 token allocator 泄漏。

**BPE 空格预处理错误（阶段 3 已修复）**。tokenizer 构造阶段已经把 GPT-2
unicode-byte 词表键还原为原始字节，encode 时却又把 ASCII 空格替换成 UTF-8
字符 `Ġ`，导致默认 prompt 从官方的 12 个 token 膨胀为 16 个。现已改为把原始
UTF-8 字节直接交给 tiktoken，decode 同样不再做反向替换，并增加真实 tokenizer
回归测试。

---

## 5. 关键缺口

### 5.1 9B 真实权重尚未端到端验证

0.8B 的真实权重覆盖了 1:1 的 GDN v:k head 布局；4B 的真实权重已进一步覆盖
v:k=2:1 分组、32 层调度和 tied embedding 输出头。现在只剩 9B checkpoint 才能
最终确认的模型分支：

- 9B 使用独立 `lm_head`；导出和加载分支已实现，但尚未跑真实 9B 权重。

### 5.2 4B 内存阻塞已解除，9B 仍超出本机容量

| 项 | 4B | 9B |
|---|---|---|
| 权重（fp32） | ~17 GB | ~36 GB |
| 权重（BF16 matrix） | ~8.5 GB | ~18 GB |
| KV cache（8 full 层，8192 ctx） | ~0.5 GB | ~0.5 GB |
| GDN state（24 linear 层） | ~50 MB | ~50 MB |

本机：**GPU 12 GB、系统内存 15 GB（swap 4 GB）**。

- GPU/CPU 跑 4B FP32：不可能或会重度 swap。
- 4B BF16 matrix 实际为 8.41 GB；本次 CPU trace 峰值 RSS 约 8.0 GiB，已确认
  可在 15 GB 系统内存中运行。
- 9B BF16 matrix 仍超过本机 GPU 和物理内存，需要 int8、更多内存或分层卸载。

CUDA 12.8/sm_89 已在 RTX 4070 SUPER 上完成 tiny CPU/CUDA 对齐、真实 4B 推理和
真实 4B CUDA/CPU/HF 逐层 trace 对齐。
Codex 默认沙箱不暴露 GPU，需要宿主权限运行；普通本机 WSL 终端不受此限制。

### 5.3 prompt 阶段逐 token 串行

GDN 递推本身串行（官方的分块并行版 `torch_chunk_gated_delta_rule` 未实现），full 层也沿用项目既有的单 token MHA。功能正确，但长 prompt 慢。

### 5.4 其他已知问题

| 问题 | 位置 | 严重性 |
|---|---|---|
| conv state 每步 O(k) 左移，未用环形缓冲 | `causal_conv1d_decode` | 低（k=4） |
| `gated_delta_step_cu` 两趟扫描 state，可融合减少一半访存 | `cuda/qwen35_kernel.cu` | 低（性能） |
| 旧 CUDA kernel 尚未全部使用统一 launch helper | add/MHA/RMSNorm/RoPE/SwiGLU | 低 |
| `view()` 返回非拥有张量，无生命周期保护 | `qwen35.cpp` | 低（buffer 在 init 一次性分配） |
| 导出器 F16 分支逐元素 `struct.unpack`，很慢 | `export.py` | 低（真实 checkpoint 是 BF16，不走该分支） |
| 无 int8 量化（`create_param_quant_layers` 直接 FATAL） | — | 中（9B 本机运行仍需要） |
| batch size 固定为 1 | 全框架既有限制 | 低 |
| `max_seq_len` 导出时固定，运行时不可变 | — | 低 |

---

## 6. 跑通 4B 还需要什么

按依赖顺序。

### 步骤 1：0.8B 真实权重对齐（阶段 3，已完成）

0.8B FP32 已完成逐层、logits 和 10-token greedy 对齐，结果见 4.6。

### 步骤 2：解决 4B 内存（阶段 4，已完成）

fp32 的 4B 需要 17 GB，本机 GPU 12 GB、内存 15 GB，两条路都不通。三个选项：

| 方案 | 工作量 | 说明 |
|---|---|---|
| **A. 加 BF16 权重支持（已完成）** | 大 | embedding/投影约减半至 ~8.5 GB；Tensor、CPU/CUDA matmul、embedding、导出器和 v3 loader 已接入 |
| **B. 先用 0.8B 验证正确性（已完成）** | 小 | 0.8B fp32 ≈ 3.4 GB；阶段 3 已验证权重顺序、RoPE、GDN 和生成状态 |
| C. CPU + mmap 惰性加载 | 中 | 靠 mmap 让 OS 按需换页，能跑但极慢；15 GB 内存下仍会 swap |

方案 A 的 BF16 weight-only 路径已在阶段 4 完成，并已用真实 4B 权重端到端验证；
方案 B 已在阶段 3 完成。

注：0.8B 是 `nk=16/nv=16`（1:1），4B/9B 是 2:1；这两条分组路径现在都已有
真实模型端到端结果。

### 步骤 3：复现 0.8B 与 HF 对齐（已完成）

```bash
# 1. 下载真实模型
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir ~/models/Qwen3.5-0.8B

# 2. 导出（先 dry_run 校验形状）
python3 tools/export_qwen35/export.py --model_dir ~/models/Qwen3.5-0.8B \
        --output ~/models/qwen35_0.8b.bin --max_seq_len 4096 --dry_run
python3 tools/export_qwen35/export.py --model_dir ~/models/Qwen3.5-0.8B \
        --output ~/models/qwen35_0.8b.bin --max_seq_len 4096 --weight_dtype bf16

# 3. 生成两侧 trace 并比较
source tools/env.sh
export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
./build/demo/qwen35_trace ~/models/qwen35_0.8b.bin \
        ~/models/Qwen3.5-0.8B/tokenizer.json /tmp/qwen35-kuiper
python3 tools/verify_qwen35/hf_reference.py \
        --model_dir ~/models/Qwen3.5-0.8B --output_dir /tmp/qwen35-hf
python3 tools/verify_qwen35/compare_traces.py \
        --kuiper_dir /tmp/qwen35-kuiper --reference_dir /tmp/qwen35-hf
```

工具逐层比较并报告首个超阈值层，同时要求前 10 个 greedy token 完全一致。

### 步骤 4：修掉 5.4 中危项（阶段 2 已完成）

- `float delta[512]` 已换成按实际 `v_head_dim` 分配的一行复用缓冲
- 采样范围已限制到 tokenizer 的 248070 个连续有效 ID
- 已增加 `kModelTypeQwen35` 枚举
- 额外完成非法模型头的前置校验和 CUDA argmax 同步/allocator 释放

### 步骤 5：跑 4B（已完成）/9B

4B 已完成分片完整性检查、BF16 导出、Kuiper CPU trace 和 Transformers BF16
对比，结果见 4.8。9B 无 tie_word_embeddings、有独立 `lm_head`，且 BF16 仍约
18 GB；导出器已处理该分支，但本机运行还需 int8、更多内存或分层卸载。

### 步骤 6（可选）：性能

- GDN 分块并行（prompt 阶段）
- `gated_delta_step_cu` 两趟融合
- conv state 环形缓冲

---

## 7. 文件清单

### 新增

```
kuiper/include/model/qwen35_config.h            磁盘头 + 运行时配置
kuiper/include/base/bfloat16.h                  CPU BF16 转换工具
kuiper/include/model/qwen35.h                   模型类
kuiper/include/op/qwen35_ops.h                 10 个 Layer 声明
kuiper/source/model/qwen35.cpp                  权重加载 + forward
kuiper/source/op/qwen35_ops.cpp                 Layer 实现
kuiper/source/op/kernels/cpu/qwen35_kernel.h    CPU kernel 声明
kuiper/source/op/kernels/cpu/qwen35_kernel.cpp  CPU kernel 实现
kuiper/source/op/kernels/cuda/qwen35_kernel.cuh CUDA kernel 声明
kuiper/source/op/kernels/cuda/qwen35_kernel.cu  CUDA kernel 实现
demo/main_qwen35.cpp                            推理 demo
test/test_model/test_qwen35.cpp                 11 个配置/模型测试
test/test_model/make_tiny_qwen35.py             合成模型生成器
test/prepare_qwen35_fixture.cmake                CTest fixture 生成与导出驱动
.github/workflows/qwen35-ci.yml                  自托管 GPU CI workflow
tools/export_qwen35/export.py                   导出器
tools/verify_qwen35/kuiper_trace.cpp            Kuiper 逐层 trace 与 10-token 生成
tools/verify_qwen35/hf_reference.py             Transformers FP32/eager 参考 trace
tools/verify_qwen35/compare_traces.py           逐层误差与 token 比较器
kuiper/source/op/kernels/cuda/cuda_launch_check.cuh  CUDA launch 错误检查 helper
tools/verify_qwen35/README.md                    对齐工具使用说明
tools/env.sh                                    工具链环境
```

### 修改（8 个，均为新增分支或修复，未改既有语义）

```
CMakeLists.txt                cmake/cuda.cmake              demo/CMakeLists.txt
test/CMakeLists.txt           kuiper/include/op/layer.h     kuiper/include/op/encode.h
kuiper/source/op/encode.cpp   kuiper/source/model/model.cpp
```

### 运行测试

```bash
source tools/env.sh
cmake -S . -B build -DUSE_CPM=ON -DQWEN35_SUPPORT=ON -DBUILD_TESTING=ON \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
      -DCMAKE_CUDA_ARCHITECTURES=89 -DCUDAToolkit_ROOT="$CUDA_HOME" \
      -DPython3_EXECUTABLE=/home/tuesday/miniconda3/bin/python
cmake --build build --target test_llm -j2
ctest --test-dir build --output-on-failure --timeout 300
```

CTest 的 `qwen35_tiny_fixture` setup 会自动生成 safetensors、确定性 tokenizer 和 BF16
checkpoint，再为 `test_llm` 设置路径；无需下载或复制真实 tokenizer。本机结果为 CTest
`4/4 passed`（含 matmul/GDN benchmark smoke），内部 GTest `55/55 passed`，fixture
相关测试没有 skip。GitHub Actions 使用
`self-hosted, linux, x64, gpu` runner 执行同一套 configure/build/ctest 命令；为避免不受信任
的代码直接运行在自托管机器上，workflow 只响应仓库 push 和手动触发。当前仓库尚未注册
self-hosted runner，远端 job 需要完成 runner 注册并添加 `gpu` 标签后才能实际调度；本次
未自动安装常驻 runner 服务。

---

## 8. 参考

- [Qwen3.5 · HuggingFace 文档](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
- `transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py` — 本实现对齐的权威来源
- `transformers/src/transformers/models/qwen3_next/modular_qwen3_next.py` — GDN 递推的参考实现
- [Gated Delta Networks (arXiv:2412.06464)](https://arxiv.org/abs/2412.06464)
- 实测配置来源：`hf-mirror.com/Qwen/Qwen3.5-{0.8B,2B,4B,9B}` 的 `config.json` 与 safetensors 头部
