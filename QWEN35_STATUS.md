# Qwen3.5 Dense 支持：设计、现状与待办

目标：用本框架跑通 Qwen3.5-4B（及 9B）的纯文本推理。

本文记录已完成的设计与实现、已验证到什么程度、存在哪些缺陷，以及达成目标还差什么。

**当前状态**：推理骨架已打通；已修正 Qwen3.5 zero-centered RMSNorm 语义，
当前共定义 42 个 GTest（新增 3 个专用 norm 测试）。CPU 专项与 tiny-model
端到端测试通过；本次环境无可用 CUDA 设备，新增 CUDA 对比测试会自动 skip。
**但尚未用真实权重验证过** —— 见「第 5 节 关键缺口」。

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
- `vocab_size: 248320`，但 tokenizer.json 只有 **248044** 个 token（embedding 有 padding）
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
CHECK_EQ(pos * sizeof(float) + sizeof(Qwen35RawConfig), raw_model_data_->file_size)
```
消耗的 float 数必须正好落在文件末尾。导出器与读取器一旦漂移立即失败，而不是读到垃圾数据后静默产出乱码。这是最有价值的一道防线。

**vocab 修正**：`create_encode_layer()` 会用 tokenizer 的 248044 覆盖 `config_->vocab_size_`，而 embedding 是 248320。在 `gen_model_from_file()` 里恢复为 header 值，否则采样会越界读。

### 3.4 导出器

[`tools/export_qwen35/export.py`](tools/export_qwen35/export.py)

- 直读 safetensors，**不依赖 torch / transformers / numpy**（本机无 pip，这是必要约束）
- 只取 `model.language_model.*`，按 `layer_types` 分流两类层
- bf16→fp32 用字节切片赋值（走 C 层）：1.79GB→3.58GB 耗时 10.7s，外推 9B 约 3.5 分钟
- `--dry_run` 校验全部张量存在且形状正确，不写文件
- 多 shard 懒加载，一次只驻留一个 shard 的 header

磁盘格式（v2，magic `K35D`）：18 个 int32 + 2 个 float 的头，随后按 `create_param_layers` 的顺序排列 fp32 权重。原有 7-int `ModelConfig` 描述不了混合模型，故另立版本化格式而非扩展。

### 3.5 修改的既有文件（8 个，共 +52/-14 行）

| 文件 | 改动 | 性质 |
|---|---|---|
| `CMakeLists.txt` | 加 `QWEN35_SUPPORT` 选项；修 nvcc 路径硬编码 | 新增 + 修复 |
| `cmake/cuda.cmake` | 显式架构不被自动探测覆盖 | 修复 |
| `demo/CMakeLists.txt` | 加 `qwen35_infer` 目标 | 新增 |
| `test/CMakeLists.txt` | 条件加 `QWEN35_SUPPORT` | 新增 |
| `kuiper/include/op/layer.h` | `LayerType` 追加 11–20（**尾部追加，原值不变**） | 新增 |
| `kuiper/include/op/encode.h`<br>`kuiper/source/op/encode.cpp`<br>`kuiper/source/model/model.cpp` | 宏条件加 `QWEN35_SUPPORT`，复用 `QwenEncodeLayer` | 新增 |

**未改动任何现有算子的实现或语义。**

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

`test/test_model/test_qwen35.cpp`，5 个模型测试：

| 测试 | 覆盖 |
|---|---|
| `Qwen35Config.LayerTypeAndLocalIndex` | 4:1 分派、按类型局部索引、4B/9B 派生尺寸 |
| `Qwen35Tiny.LoadsOnCpu` | 权重布局断言（导出器 ↔ 读取器对齐） |
| `Qwen35Tiny.ForwardProducesFiniteLogits` | 两类层都跑通，logits 无 NaN/Inf |
| `Qwen35Tiny.ResetStateMakesRunsReproducible` | GDN 状态清零正确，重放逐位一致 |
| `Qwen35Tiny.CudaMatchesCpu` | CPU/CUDA 端到端相对误差 **<2e-3** |

合成模型与 4B **同构**：真实 vocab 248320、head_dim 256、rotary_dim 64、interval 4、v:k=2:1、tie_word_embeddings —— 只缩小 hidden/inter/layers。生成器 [`test/test_model/make_tiny_qwen35.py`](test/test_model/make_tiny_qwen35.py) 已入库，可字节级复现。

另在 `test/test_op/test_qwen35_norm.cpp` 新增 3 个 zero-centered RMSNorm 测试：

- weight 全零时退化为无权重 RMSNorm（用于直接区分 `w` 与 `1+w`）
- 非零 weight 的 `(1+w)` 参考值及 CPU in-place 路径
- CUDA 对 CPU（无 CUDA 设备时 skip）

当前共定义 42 个 GTest。本次阶段 1 验证中，3 个 CPU/config 专项测试和 3 个
tiny-model CPU 端到端测试通过，CUDA 专项因运行环境无可用设备而跳过。

### 4.6 已发现并修复的实现错误

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

---

## 5. 关键缺口

### 5.1 未用真实权重验证（最大风险）

**所有验证都基于合成权重（随机数）。与 HF 的 logits 对比这一关没过。**

合成模型能证明的：形状、布局、调度、状态管理、CPU/CUDA 一致性。
**不能证明的**：权重顺序是否真的对得上、RoPE 相位是否正确、GDN 各项语义是否匹配。

`gated_delta_step` 已单独对齐官方实现，但**组装后的整体**没有验证过。上面刚发现的 q_proj bug 就是这类问题的例子 —— 合成测试全绿，但真实推理会输出乱码。

**这是达成目标的必经一步，也是最可能暴露新问题的一步。**

### 5.2 显存/内存不足

| 项 | 4B | 9B |
|---|---|---|
| 权重（fp32） | ~17 GB | ~36 GB |
| KV cache（8 full 层，8192 ctx） | ~0.5 GB | ~0.5 GB |
| GDN state（24 linear 层） | ~50 MB | ~50 MB |

本机：**GPU 12 GB、系统内存 15 GB（swap 4 GB）**。

- GPU 跑 4B fp32：**不可能**（17 GB > 12 GB）
- CPU 跑 4B fp32：**也不够**（17 GB > 15 GB，会重度 swap）

框架目前**只支持 fp32**。这是硬阻塞。

### 5.3 prompt 阶段逐 token 串行

GDN 递推本身串行（官方的分块并行版 `torch_chunk_gated_delta_rule` 未实现），full 层也沿用项目既有的单 token MHA。功能正确，但长 prompt 慢。

### 5.4 其他已知问题

| 问题 | 位置 | 严重性 |
|---|---|---|
| `gated_delta_step_cpu` 用固定栈数组 `float delta[512]`，`v_head_dim>512` 会**静默算错**而非报错 | `cpu/qwen35_kernel.cpp` | 中（当前所有型号 vd=128，但缺防护） |
| `model_type_` 复用了 `kModelTypeLLama2`，未新增枚举 → `model_type()` 报告错误 | `qwen35.cpp` 构造函数 | 低（仅信息性） |
| 采样可能命中 embedding padding 对应的无效 token id；不能从 `248044` 起简单截断，因为其后仍存在有效特殊 token | `post_processing` | 中（需按 tokenizer 的实际有效 ID 集合构造 mask） |
| conv state 每步 O(k) 左移，未用环形缓冲 | `causal_conv1d_decode` | 低（k=4） |
| `gated_delta_step_cu` 两趟扫描 state，可融合减少一半访存 | `cuda/qwen35_kernel.cu` | 低（性能） |
| kernel launch 后普遍缺 `cudaGetLastError()` 检查 | 各处 | 低 |
| `view()` 返回非拥有张量，无生命周期保护 | `qwen35.cpp` | 低（buffer 在 init 一次性分配） |
| 导出器 F16 分支逐元素 `struct.unpack`，很慢 | `export.py` | 低（真实 checkpoint 是 BF16，不走该分支） |
| 无 int8 量化（`create_param_quant_layers` 直接 FATAL） | — | 中（与 5.2 相关） |
| batch size 固定为 1 | 全框架既有限制 | 低 |
| `max_seq_len` 导出时固定，运行时不可变 | — | 低 |

---

## 6. 跑通 4B 还需要什么

按依赖顺序。

### 步骤 1：解决内存（阻塞项，必须先做）

fp32 的 4B 需要 17 GB，本机 GPU 12 GB、内存 15 GB，两条路都不通。三个选项：

| 方案 | 工作量 | 说明 |
|---|---|---|
| **A. 加 bf16/fp16 权重支持** | 大 | 4B 降到 ~8.5 GB，可放进 12 GB GPU。需要改 Tensor 的 dtype 体系、全部 matmul kernel、导出器。**一劳永逸，推荐** |
| **B. 先用 0.8B 验证正确性** | 小 | 0.8B fp32 ≈ 3.4 GB，GPU 完全放得下。**验证权重顺序/RoPE/GDN 语义与 4B 完全等价**（同构，只是维度小）。**推荐作为第一步** |
| C. CPU + mmap 惰性加载 | 中 | 靠 mmap 让 OS 按需换页，能跑但极慢；15 GB 内存下仍会 swap |

**建议：先做 B**（用 0.8B 打通并对齐 HF），**再做 A**（拿到 4B/9B 的实际能力）。B 能以最小代价暴露 5.1 里的全部语义问题。

注：0.8B 是 `nk=16/nv=16`（1:1），4B/9B 是 2:1。两者都要测到才算覆盖分组路径 —— 分组逻辑已在算子级用 4B 尺寸验证过（4.2、4.3），但端到端只测了合成模型。

### 步骤 2：与 HF 对齐 logits（正确性关口）

```bash
# 1. 下载真实模型
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir ~/models/Qwen3.5-0.8B

# 2. 导出（先 dry_run 校验形状）
python3 tools/export_qwen35/export.py --model_dir ~/models/Qwen3.5-0.8B \
        --output ~/models/qwen35_0.8b.bin --max_seq_len 4096 --dry_run
python3 tools/export_qwen35/export.py --model_dir ~/models/Qwen3.5-0.8B \
        --output ~/models/qwen35_0.8b.bin --max_seq_len 4096

# 3. 推理
source tools/env.sh
export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
./build/demo/qwen35_infer ~/models/qwen35_0.8b.bin \
        ~/models/Qwen3.5-0.8B/tokenizer.json 64
```

对齐方法（**必须逐层做，不要只比最终输出**）：

1. 用 HF 跑同一 prompt，`output_hidden_states=True` 存下每层输出
2. 在 `Qwen35Model::forward` 里加临时 dump，导出每层 hidden state
3. 逐层比对，定位**第一个**发散的层
4. 若某 linear 层发散 → 查 GDN 各中间量（conv 输出、l2norm 后的 q/k、beta、g、state）
5. 若某 full 层发散 → 查 q/gate 拆分、QK-norm、RoPE 相位、输出门

判定标准：前 10 个 token 的 greedy 输出与 HF 完全一致。

预期会踩的坑（基于 4.6 的经验）：权重顺序、RoPE 相位、门拆分这三类最容易出错，且**只有真实权重能暴露**。zero-centered RMSNorm 已有专项测试，但仍需在真实逐层对齐中确认。

### 步骤 3：修掉 5.4 中危项

- `float delta[512]` 换成动态分配或加 `CHECK`（静默算错比崩溃更危险）
- logits 超出 tokenizer vocab 的部分 mask 掉
- 加 `kModelTypeQwen35` 枚举

### 步骤 4：跑 4B/9B

依赖步骤 1A（bf16）。9B 无 tie_word_embeddings、有独立 `lm_head`，导出器已处理该分支但未实测。

### 步骤 5（可选）：性能

- GDN 分块并行（prompt 阶段）
- `gated_delta_step_cu` 两趟融合
- conv state 环形缓冲

---

## 7. 文件清单

### 新增

```
kuiper/include/model/qwen35_config.h            磁盘头 + 运行时配置
kuiper/include/model/qwen35.h                   模型类
kuiper/include/op/qwen35_ops.h                 10 个 Layer 声明
kuiper/source/model/qwen35.cpp                  权重加载 + forward
kuiper/source/op/qwen35_ops.cpp                 Layer 实现
kuiper/source/op/kernels/cpu/qwen35_kernel.h    CPU kernel 声明
kuiper/source/op/kernels/cpu/qwen35_kernel.cpp  CPU kernel 实现
kuiper/source/op/kernels/cuda/qwen35_kernel.cuh CUDA kernel 声明
kuiper/source/op/kernels/cuda/qwen35_kernel.cu  CUDA kernel 实现
demo/main_qwen35.cpp                            推理 demo
test/test_model/test_qwen35.cpp                 5 个单元测试
test/test_model/make_tiny_qwen35.py             合成模型生成器
tools/export_qwen35/export.py                   导出器
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
# 生成合成模型（需 numpy，本机在 miniconda 里）
/home/tuesday/miniconda3/bin/python test/test_model/make_tiny_qwen35.py --out_dir /tmp/tiny35
cp <某个真实 Qwen3.5 的 tokenizer.json> /tmp/tiny35/
python3 tools/export_qwen35/export.py --model_dir /tmp/tiny35 \
        --output /tmp/tiny35.bin --max_seq_len 128

export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH GLOG_logtostderr=1
export KUIPER_TINY_QWEN35=/tmp/tiny35.bin
export KUIPER_TINY_QWEN35_TOKENIZER=/tmp/tiny35/tokenizer.json
./build/test/test_llm --gtest_filter='Qwen35*'
```

未设这两个环境变量时，`Qwen35Tiny.*` 会 skip 而非失败，`ctest` 在没有合成模型的机器上仍能通过。

---

## 8. 参考

- [Qwen3.5 · HuggingFace 文档](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
- `transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py` — 本实现对齐的权威来源
- `transformers/src/transformers/models/qwen3_next/modular_qwen3_next.py` — GDN 递推的参考实现
- [Gated Delta Networks (arXiv:2412.06464)](https://arxiv.org/abs/2412.06464)
- 实测配置来源：`hf-mirror.com/Qwen/Qwen3.5-{0.8B,2B,4B,9B}` 的 `config.json` 与 safetensors 头部
