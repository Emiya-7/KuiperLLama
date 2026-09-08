# KuiperLLama Qwen3.5 扩展实施报告

更新时间：2026-09-08

当前分支：`feat/qwen35-stage4-low-precision`

目标模型：Qwen3.5 Dense 纯文本模型，重点为 4B，兼顾 0.8B/2B/9B 的结构兼容性

## 1. 当前结论

项目已经从“原框架只支持 Llama/Qwen2/Qwen3，Qwen3.5 方案尚不完整”的状态，推进到：

- 能识别并执行 Qwen3.5 Dense 的混合架构：每 4 层中 3 层 Gated DeltaNet、1 层
  full attention。
- 已实现 Qwen3.5 所需的 CPU/CUDA 专用算子、模型调度、状态管理和权重导出。
- 已实现 v3 混合精度 checkpoint：大矩阵保存为 BF16，激活、累加、小参数和状态保持
  FP32。
- Qwen3.5-0.8B 已与 Transformers FP32/eager 做逐层、logits 和生成序列严格对齐。
- Qwen3.5-4B 官方权重已完整下载并通过官方 SHA-256 校验，已完成 CPU 推理、
  Transformers BF16 参考比较和 RTX 4070 SUPER CUDA 真机推理。
- 当前 4B CUDA 推理不会因 12 GB 显存不足而失败；实测 21 个位置的 forward/generation
  阶段为 0.513 秒，约 40.95 steps/s。

因此，**Qwen3.5-4B 的 BF16 weight-only 纯文本推理主链路已经跑通**。

当前还不能称为全部完成，主要原因是：

- Qwen3.5-9B 的独立 `lm_head` 虽然已经实现导出/加载分支，但没有真实权重验证；
- 9B BF16 文本权重约 18 GB，超过本机 12 GB 显存和 15 GB 物理内存，需要 INT8、
  分层卸载或更大设备；
- CUDA 的真实 4B 逐层 trace 工具尚未实现，当前真实 4B CUDA 验证是生成结果验证，
  逐层数值对齐由 4B CPU/HF 和 tiny CPU/CUDA 两条测试共同覆盖；
- 全量 52 项测试中有 1 项测试代码重复销毁 CUDA stream，会在 WSL 驱动中段错误。

更细的架构推导、张量形状和公式见 [`QWEN35_STATUS.md`](QWEN35_STATUS.md)。本文重点
记录从初始状态到现在完成了什么、实际验证到哪里、剩余问题和可直接执行的命令。

---

## 2. 与最初状态的对比

### 2.1 最初面对的问题

最初项目的主体是 Llama/Qwen2/Qwen3 推理框架。Qwen3.5 扩展的早期判断中曾把
Qwen3.5 当成接近 Qwen3 的普通 attention 模型，这个前提不成立。真实
`Qwen/Qwen3.5-{0.8B,2B,4B,9B}` checkpoint 表明：

- `architectures` 是 `Qwen3_5ForConditionalGeneration`，checkpoint 同时包含文本塔、
  vision tower 和 MTP 权重；
- 文本塔不是全 attention，而是 full attention 与 Gated DeltaNet 的混合模型；
- Qwen3.5 有独立 `head_dim=256`、partial RoPE、attention output gate、Q/K norm；
- 普通 RMSNorm 使用 zero-centered 权重，即乘以 `(1 + weight)`；
- 4B/9B 的 GDN 是 `num_v_heads:num_k_heads = 2:1`；
- 0.8B/2B/4B 复用 embedding 作为输出头，9B 则有独立 `lm_head`。

这些差异意味着不能只修改配置名称或复用 Qwen3 forward，必须增加模型类、状态缓冲、
混合层调度和一组新算子。

### 2.2 当前相对上游的代码规模

以 `upstream/main` 为基准，当前阶段 4 分支在生成本文档前的差异为：

- 修改/新增 43 个文件；
- 新增约 5,221 行，删除约 62 行；
- 主要新增内容集中在 Qwen3.5 模型、CPU/CUDA kernel、导出器、真实模型对齐工具和测试；
- 原有 Llama/Qwen2/Qwen3 的枚举值和算子语义尽量保持不变。

### 2.3 能力变化总览

| 能力 | 最初状态 | 当前状态 |
|---|---|---|
| Qwen3.5 架构识别 | 不能可靠描述，曾错误类比 Qwen3 | 独立 `Qwen35Config`，支持混合层和派生尺寸 |
| Gated DeltaNet | 框架不存在 | CPU/CUDA decode 路径已实现 |
| partial RoPE | 原 RoPE 语义不匹配 | Qwen3.5 专用 rotate-half partial RoPE |
| attention output gate | 不支持 | 已实现 q/gate 按 head 交错拆分和 sigmoid gate |
| zero-centered RMSNorm | 不支持 | CPU/CUDA 专用算子已实现 |
| v:k=2:1 GDN 分组 | 不支持 | 算子、tiny、真实 4B 均已覆盖 |
| Qwen3.5 checkpoint | 无稳定格式 | v2 FP32 + v3 BF16-matrix 兼容加载 |
| BF16 Tensor/权重 | 无 | Tensor、matmul、embedding、导出器、loader 均支持 |
| tied embedding | CPU 指针可复用，CUDA 可能重复上传 | CUDA 共享同一设备 Tensor，避免重复大表 |
| tokenizer | GPT-2 byte-level 空格处理错误 | 已与 Transformers 的 12-token chat prompt 对齐 |
| 真实模型验证 | 缺失 | 0.8B 严格对齐，4B CPU/HF + CUDA 真机跑通 |
| CUDA 工具链 | 系统默认 11.5，不支持 sm_89 | 用户默认 CUDA 12.8，项目编译到 sm_89 |

---

## 3. CUDA 12.8 默认环境

### 3.1 实测环境

| 项目 | 实测值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 SUPER |
| 显存 | 12,282 MiB |
| Compute Capability | 8.9 |
| Windows/WSL 驱动 | 610.62，CUDA UMD 13.3 |
| 项目 CUDA Toolkit | 12.8.93 |
| 项目 CMake | 3.28.6 |
| PyTorch | 2.10.0+cu128 |
| CMake 架构 | `compute_89,sm_89` |
| 构建类型 | Release |

驱动声明的 CUDA 13.3 是驱动能够支持的最高 CUDA UMD 能力，不要求项目也使用 13.3；
它向下兼容项目使用的 CUDA Toolkit 12.8。

### 3.2 已做的默认环境修改

用户级环境文件 `~/.local/bin/env` 已加入：

```bash
export CUDA_HOME="$HOME/.local/opt/cuda-12.8"
export CUDAToolkit_ROOT="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

`~/.bash_profile` 和交互式 `~/.bashrc` 都会加载这个文件，因此新登录 shell 和新交互
shell 默认都会选择 CUDA 12.8。使用干净环境启动新登录 shell 后的验证结果为：

```text
CUDA_HOME=/home/tuesday/.local/opt/cuda-12.8
CUDAToolkit_ROOT=/home/tuesday/.local/opt/cuda-12.8
nvcc=/home/tuesday/.local/opt/cuda-12.8/bin/nvcc
Build cuda_12.8.r12.8/compiler.35583870_0
```

系统文件 `/usr/bin/nvcc` 仍然是 CUDA 11.5，但它已经排在用户 CUDA 12.8 之后。没有
读取用户 shell 配置的 systemd service、IDE task 或脚本仍应显式指定 CUDA 路径。

项目内保留了可复现配置 [`tools/env.sh`](tools/env.sh)：

```bash
source tools/env.sh
```

即使默认环境已经修改，构建项目时仍建议执行这一行，使 CI、IDE 和手动终端使用相同工具链。

### 3.3 当前项目构建配置

现有 `build/CMakeCache.txt` 已确认：

```text
CMAKE_BUILD_TYPE=Release
CMAKE_CUDA_COMPILER=/home/tuesday/.local/opt/cuda-12.8/bin/nvcc
CMAKE_CUDA_ARCHITECTURES=89
CUDAToolkit_ROOT=/home/tuesday/.local/opt/cuda-12.8
QWEN35_SUPPORT=ON
USE_CPM=ON
```

CUDA 编译命令含：

```text
--generate-code=arch=compute_89,code=[compute_89,sm_89]
```

`qwen35_infer` 的运行时依赖解析到 CUDA 12.8 的 `libcudart.so.12`，未发现缺失动态库。

### 3.4 WSL/Codex 执行限制

- 普通 WSL 终端能够访问 GPU。
- 当前 Codex 默认文件/进程沙箱会屏蔽 GPU，沙箱内 `nvidia-smi` 和
  `torch.cuda.is_available()` 会失败；以宿主权限执行后 GPU 正常可见。这不是本机 CUDA
  配置错误。
- Compute Sanitizer/cuda-gdb 的设备调试接口在当前 WDDM/WSL 环境未启用，常规 CUDA
  编译、kernel、测试和推理不受影响，但暂时不能依赖 Compute Sanitizer 做设备内存诊断。

---

## 4. 已完成的开发工作

### 4.1 阶段 1：Qwen3.5 Dense 推理基线

核心提交：`8dbaba3 feat(qwen35): add dense inference baseline`

完成内容：

1. 新增 `Qwen35RawConfig`、`Qwen35Config` 和 `Qwen35Model`。
2. 支持 `[linear, linear, linear, full] × N` 混合层分派。
3. full attention 只为对应层分配 KV cache，GDN 层只分配 recurrent state 和 conv state。
4. 支持 `head_dim` 独立于 `hidden_size / num_heads`。
5. 支持 Qwen3.5 partial RoPE、Q/K norm、attention output gate。
6. 支持 Gated DeltaNet decode 递推及 v:k 分组。
7. 新增 Qwen3.5 专用 CPU/CUDA kernel 和 Layer 封装。
8. 新增 safetensors 导出器、tiny checkpoint 生成器和模型测试。
9. 修正普通 RMSNorm 为 `(1 + weight)` 的 zero-centered 语义。
10. 修正 q_proj 输出中每个 head 的 q/gate 交错布局。
11. 修正 CUDA 11.5 与 sm_89/GCC 11 不兼容问题，接入 CUDA 12.8。

### 4.2 阶段 2：安全性和边界条件

核心提交：`045019b fix(qwen35): harden inference edge cases`

完成内容：

- CPU GDN 移除固定 `float delta[512]`，支持维度大于 512，避免静默截断。
- 采样范围限制到 tokenizer 的有效 ID 248070，同时保留 248320 宽的 embedding/logits。
- 新增独立 `kModelTypeQwen35`，不再冒充 Llama2。
- 模型头关键尺寸和除数在派生计算前验证，非法文件返回 parse error。
- CUDA argmax 在读取异步 D2H 结果前同步 stream，并释放临时 device index。
- 增加非法 interval、采样 padding、超 512 维 GDN 等回归测试。

### 4.3 阶段 3：tokenizer 和真实 0.8B 对齐

主要提交：

- `b89bd2b fix(tokenizer): preserve raw spaces in BPE input`
- `d9b1482 test(qwen35): skip end-to-end CUDA check without a device`
- `11c59c8 test(qwen35): align real 0.8b inference with transformers`

完成内容：

- 修复 GPT-2 byte-level BPE 空格被二次转换成 `Ġ` 的错误。
- 默认 chat prompt 从错误的 16 tokens 恢复为与 Transformers 一致的 12 tokens。
- 新增 Kuiper trace、Transformers reference 和逐层比较工具。
- 保存并比较每层 hidden state、final norm、完整 logits 和前 10 个 greedy tokens。
- CUDA 不可用时相关测试正确 skip，而不是初始化后 FATAL。

### 4.4 阶段 4：BF16 weight-only 推理

主要提交：

- `b109535 feat(tensor): add BF16 storage type`
- `1dc2133 feat(matmul): support BF16 weights with FP32 accumulation`
- `006a9c0 feat(embedding): support BF16 weight tables`
- `f5a450d feat(qwen35): load mixed FP32 and BF16 checkpoints`
- `5328736 fix(qwen35): share tied embedding weights on CUDA`
- `0424ee2 test(qwen35): reject unknown checkpoint weight types`

完成内容：

- 新增 BF16 Tensor dtype、BF16/FP32 转换和 round-to-nearest-even。
- CPU/CUDA matmul 支持 FP32 input + BF16 weight + FP32 accumulation/output。
- CPU/CUDA embedding 支持 BF16 table，输出 FP32。
- checkpoint 升级为 v3：embedding/二维 projection 为 BF16，小参数保持 FP32。
- loader 同时兼容 v2 全 FP32 checkpoint。
- 4B tied embedding 与 LM head 在 CUDA 上共享一次设备权重，避免重复约 1.18 GiB。
- 导出器增加 `--weight_dtype bf16|fp32`，默认 BF16。

### 4.5 真实 4B 和本机 CUDA 验证

主要提交：

- `ecbf18a test(qwen35): support memory-efficient HF references`
- `292edc6 docs(qwen35): record real 4b validation`
- `2781e10 docs(qwen35): record 4b checkpoint checksums`

完成内容：

- 通过 `hf-mirror.com` 下载 Qwen3.5-4B 两片官方 safetensors。
- 验证全部 738 个源张量的 shape、dtype、data offsets、分片边界和索引总字节数。
- 两片文件 SHA-256 与 Hugging Face 官方 LFS 元数据一致。
- 导出 8.413 GB v3 BF16-matrix checkpoint，共 426 个文本塔张量。
- Transformers reference 改为只加载 `Qwen3_5ForCausalLM` 文本塔，避免加载 vision/MTP。
- reference 增加 `--dtype bf16`，使 4B 可以在本机内存中完成参考计算。
- 完成真实 4B CPU、HF BF16 和 RTX 4070 SUPER CUDA 推理。

---

## 5. 算子层当前状态

### 5.1 已实现的 Qwen3.5 专用算子

| 算子 | CPU | CUDA | 用途 |
|---|---|---|---|
| `l2norm` | 已实现 | 已实现 | GDN q/k 归一化 |
| `zero_centered_rmsnorm` | 已实现 | 已实现 | decoder/final/QK norm，乘 `(1+w)` |
| `gated_rmsnorm` | 已实现 | 已实现 | GDN 输出 norm 与 SiLU gate |
| `causal_conv1d_decode` | 已实现 | 已实现 | GDN 深度可分离因果卷积和状态更新 |
| `gated_delta_step` | 已实现 | 已实现 | GDN recurrent state 递推 |
| `softplus_decay` | 已实现 | 已实现 | 计算 GDN 衰减因子 |
| `rope_partial` | 已实现 | 已实现 | rotate-half partial RoPE |
| `split_head_interleaved` | 已实现 | 已实现 | 正确拆分每个 head 的 q/gate |
| `sigmoid` | 已实现 | 已实现 | beta/output gate |
| `mul` | 已实现 | 已实现 | attention output gate |

此外，通用 matmul 和 embedding 已增加 BF16 weight 分支。

### 5.2 算子正确性已经覆盖的部分

- 独立 float64/Transformers 参考验证了 L2Norm、decay、gated RMSNorm、conv、GDN、RoPE。
- 4B/9B 真实尺寸的 `nk=16,nv=32,kd=vd=128` GDN 分组已经做过 CPU/CUDA 参考测试。
- tiny 模型同时覆盖 linear/full 两类层、v:k=2:1、partial RoPE、tied embedding 和 BF16。
- RTX 4070 SUPER 上 tiny CPU/CUDA 完整 logits 对齐测试通过。
- 真实 4B CUDA 生成成功，说明全部实际尺寸 kernel、设备权重和状态缓冲能协同运行。

### 5.3 算子层仍然不足的部分

#### BF16 matmul 性能

当前 CUDA BF16 matmul 是每个输出行一个 block，逐元素执行
`__bfloat162float` 后做 FP32 reduction。它保证正确性，但没有使用：

- Tensor Core BF16 MMA；
- cuBLAS/cuBLASLt；
- `__nv_bfloat162` 向量化读取和计算；
- 多输出 tile、shared-memory tile 或权重预排布。

因此目前属于“能正确推理”的 kernel，不是充分发挥 RTX 4070 SUPER 性能的 kernel。
CPU BF16 matmul 同样是显式三重循环，没有走 Armadillo/BLAS 的矩阵乘优化。

#### Gated DeltaNet 性能

- `gated_delta_step_cu` 对 state 做两趟扫描，可融合以减少一半左右的 state 访存。
- prompt 阶段仍逐 token 递推，没有实现官方 chunk/parallel GDN prefill。
- `causal_conv1d_decode` 每步移动 k-1 个历史值；k=4 时影响较小，但可换成环形缓冲。

#### kernel 安全性和调试

- 多数新 kernel launch 后没有统一执行 `cudaGetLastError()`。
- 缺少面向真实大尺寸的独立 BF16 matmul CUDA 数值/性能测试；目前由 tiny 端到端覆盖。
- Compute Sanitizer 在当前 WSL/WDDM 环境不能初始化设备调试接口。

#### 未实现算子/精度路径

- Qwen3.5 INT8 权重加载和量化未实现；`create_param_quant_layers()` 当前直接报未实现。
- 没有 FP16 matrix checkpoint 路径；导出器会将来源 F16 转为目标 FP32/BF16。
- 没有 batch inference，模型和大部分 kernel 按 batch size 1 设计。
- 没有 vision tower 和 MTP 推理；这是当前纯文本目标下的有意裁剪，不是回归。

---

## 6. 验证结果

### 6.1 官方权重完整性

Qwen3.5-4B 源模型目录：

```text
/home/tuesday/workspace/icd/models/Qwen__Qwen3.5-4B
```

| 分片 | 大小 | SHA-256 |
|---|---:|---|
| `model.safetensors-00001-of-00002.safetensors` | 5,329,398,688 bytes | `26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61` |
| `model.safetensors-00002-of-00002.safetensors` | 3,990,429,408 bytes | `cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188` |

索引声明的 tensor data 总量为 `9,319,737,856` bytes。文件大小还包括 safetensors
header；全部 offsets 均落在对应分片内，未发现截断、错片或索引不一致。

### 6.2 0.8B 严格对齐

0.8B BF16-matrix Kuiper 与 Transformers FP32/eager：

| 项目 | 结果 |
|---|---|
| decoder 0–23 最大绝对误差 | `2.37823e-05` |
| final norm 最大绝对误差 | `1.20163e-04` |
| final norm 相对误差 | `3.06030e-06` |
| logits 最大绝对误差 | `1.01328e-04` |
| logits 平均绝对误差 | `8.89898e-06` |
| 前 10 个 greedy token | 完全一致 |

### 6.3 4B CPU 与 Transformers BF16

默认 prompt：

```text
<|im_start|>user
What is AI?<|im_end|>
<|im_start|>assistant
```

| 项目 | 结果 |
|---|---|
| prompt tokens | 12 |
| Kuiper CPU trace | 75.10 s，峰值 RSS 8,374,084 KB |
| Transformers BF16/eager 文本塔 | 35.01 s，峰值 RSS 9,225,216 KB |
| decoder 最大相对误差 | `1.86040e-02` |
| final norm 相对误差 | `8.04986e-03` |
| logits 相对误差 | `1.32907e-02` |
| 前 10 个 greedy token | 完全一致 |

Transformers BF16 会在层间舍入激活，Kuiper 只以 BF16 保存大矩阵而保持 FP32 激活，
因此 4B BF16 reference 使用明确的 `2e-2` 相对阈值，不能与 0.8B FP32 reference 的
`2e-3` 阈值混用。

4B 前 10 个生成 token：

```text
[248068, 271, 248069, 271, 332, 15015, 332, 12965, 364, 2972]
```

### 6.4 RTX 4070 SUPER 真机结果

实测确认：

- `nvidia-smi` 正常识别 RTX 4070 SUPER；
- PyTorch 2.10.0+cu128 返回 `torch.cuda.is_available() == True`；
- 最小 CUDA kernel 测试通过；
- 除已知测试生命周期错误外的 51 项测试全部通过；
- `Qwen35Tiny.CudaMatchesCpu` 独立通过；
- 真实 8.413 GB Qwen3.5-4B checkpoint CUDA 加载和生成成功。

4B CUDA 实测：

```text
steps: 21
duration: 0.513 s
steps/s: 40.95
process elapsed including load: 7.22 s
host max RSS: 8,428,820 KB
```

生成文本开头：

```text
<think>

</think>

**AI** stands for **
```

进程结束后显存恢复到基线附近，未发现 8 GB 级设备权重泄漏。

### 6.5 测试状态

测试二进制当前包含 52 项。带 tiny fixture、在宿主 GPU 权限下运行的结果：

- 排除 `Qwen35ZeroCenteredRMSNorm.CudaMatchesCpu` 后：`51/51 passed`；
- `Qwen35Tiny.CudaMatchesCpu`：单独通过；
- 全量 52 项会在 RMSNorm 测试退出时发生 SIGSEGV。

崩溃原因已经由 cuda-gdb 调用栈确定：

1. [`test/test_op/test_qwen35_norm.cpp`](test/test_op/test_qwen35_norm.cpp) 手动调用
   `cudaStreamDestroy(cuda_config->stream)`；
2. [`kuiper/include/base/cuda_config.h`](kuiper/include/base/cuda_config.h) 中
   `CudaConfig::~CudaConfig()` 随后再次销毁同一 stream；
3. WSL `libcuda.so` 在第二次销毁无效句柄时段错误。

kernel 已执行、同步和完成数值断言，崩溃发生在清理阶段。应删除测试中的手动销毁，
或销毁后把 `cuda_config->stream` 置为 `nullptr`。本报告任务只要求分析和记录，尚未修改
该测试。

此外，项目当前没有向 CTest 注册测试，所以 `ctest --test-dir build` 会显示
`No tests were found`；必须直接执行 `build/test/test_llm`。这是测试基础设施缺口。

---

## 7. 已实现、已编码但未验证、未实现

| 项目 | 状态 | 说明 |
|---|---|---|
| Qwen3.5-0.8B CPU FP32/BF16 | 已实现并严格验证 | 与 HF 逐层/logits/token 对齐 |
| Qwen3.5-4B CPU BF16 | 已实现并验证 | 真实权重与 HF BF16 比较通过 |
| Qwen3.5-4B CUDA BF16 | 已实现并真机运行 | 4070 SUPER 生成成功 |
| GDN v:k=1:1 | 已验证 | 0.8B 真实模型 |
| GDN v:k=2:1 | 已验证 | tiny CPU/CUDA + 4B 真实模型 |
| tied embedding | 已验证 | 0.8B/4B，CUDA 设备权重共享 |
| checkpoint v2 FP32 | 已实现并回归验证 | 保持旧格式兼容 |
| checkpoint v3 BF16 matrix | 已实现并验证 | 0.8B/4B |
| 9B 独立 `lm_head` | 已编码，未真实验证 | 需要 9B 权重和更低精度/更大内存 |
| 真实 4B CUDA 逐层 trace | 未实现 | trace runner 当前硬编码 CPU |
| Qwen3.5 INT8 | 未实现 | 9B 本机运行的关键前置项 |
| 并行/分块 prefill | 未实现 | 长 prompt 性能关键项 |
| batch > 1 | 未实现 | 继承框架现有限制 |
| 运行时可变 context | 未实现 | `max_seq_len` 固化在导出文件头 |
| vision tower | 不在当前范围 | 只做文本塔 |
| MTP | 不在当前范围 | 普通自回归生成不使用 |

---

## 8. 当前问题和风险优先级

### P0：应立即修复

1. **CUDA RMSNorm 测试双重销毁 stream**
   - 影响：全量测试无法得到 52/52 green。
   - 范围：测试代码，一行级修复；模型推理本身不受影响。
   - 修复后应重新运行全部 52 项并单独提交 commit。

### P1：影响验证完整性或更大模型

1. **真实 CUDA trace 缺失**
   - `qwen35_trace` 当前固定 CPU，hidden callback 也要求 CPU Tensor。
   - 应增加 `--device cpu|cuda`，CUDA trace 时按层同步并复制到 host。

2. **INT8 未实现**
   - 9B BF16 约 18 GB，无法在 12 GB 4070 SUPER 上常驻。
   - 建议优先做 weight-only INT8：embedding、matmul、checkpoint v4/量化元数据。

3. **测试未接入 CTest**
   - CI 或开发者执行 `ctest` 会误以为没有测试。
   - 应在 `test/CMakeLists.txt` 中启用并注册 `test_llm`，同时提供 tiny fixture。

### P2：性能和工程质量

1. BF16 CUDA matmul 没有使用 Tensor Core/cuBLASLt。
2. CPU BF16 matmul没有 BLAS 优化。
3. GDN prompt 逐 token 串行，长 prompt 性能差。
4. GDN state 两趟扫描可融合。
5. kernel launch 错误检查不统一。
6. conv state 可改环形缓冲。
7. `view()` 是不拥有内存的 Tensor，生命周期依赖模型 buffer 长期存在。
8. demo prompt 硬编码为 `What is AI?`，命令行不能直接输入任意 prompt。

---

## 9. Qwen3.5-4B 推理命令

### 9.1 进入项目并确认 CUDA 12.8

```bash
cd /home/tuesday/workspace/cuda_project/KuiperLLama
source tools/env.sh

command -v nvcc
nvcc --version
nvidia-smi
```

预期 `nvcc`：

```text
/home/tuesday/.local/opt/cuda-12.8/bin/nvcc
```

### 9.2 配置和构建

已有 `build/` 可以直接增量构建：

```bash
cmake --build build -j2
```

需要从头配置时：

```bash
cmake -B build \
  -DUSE_CPM=ON \
  -DQWEN35_SUPPORT=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCUDAToolkit_ROOT="$CUDA_HOME"

cmake --build build -j2
```

### 9.3 使用当前已经导出的 4B checkpoint 在 CUDA 上推理

当前可用文件：

```text
checkpoint: /tmp/qwen35_4b_stage4_bf16.bin
tokenizer : /home/tuesday/workspace/icd/models/Qwen__Qwen3.5-4B/tokenizer.json
```

运行 12-token 默认 prompt，并生成 10 个新 token：

```bash
GLOG_logtostderr=1 ./build/demo/qwen35_infer \
  /tmp/qwen35_4b_stage4_bf16.bin \
  /home/tuesday/workspace/icd/models/Qwen__Qwen3.5-4B/tokenizer.json \
  21
```

`qwen35_infer` 默认使用 CUDA。当前 demo 的第三个数值参数是总 position 上限，不是
`max_new_tokens`；默认 prompt 有 12 个 token，因此 `21 = 12 - 1 + 10`，对应 10 个
生成 token。模型遇到终止 token 时会提前结束。

CPU 对照命令：

```bash
GLOG_logtostderr=1 ./build/demo/qwen35_infer \
  /tmp/qwen35_4b_stage4_bf16.bin \
  /home/tuesday/workspace/icd/models/Qwen__Qwen3.5-4B/tokenizer.json \
  --cpu 21
```

注意：`/tmp` 可能被系统清理。需要长期保留时，建议重新导出到模型目录或其他持久目录。

### 9.4 `/tmp` checkpoint 丢失后的重新导出命令

```bash
MODEL_DIR=/home/tuesday/workspace/icd/models/Qwen__Qwen3.5-4B
CHECKPOINT=/home/tuesday/workspace/icd/models/Qwen__Qwen3.5-4B/kuiper-qwen35-4b-bf16-seq128.bin

/home/tuesday/miniconda3/bin/python tools/export_qwen35/export.py \
  --model_dir "$MODEL_DIR" \
  --output "$CHECKPOINT" \
  --max_seq_len 128 \
  --weight_dtype bf16 \
  --dry_run

/home/tuesday/miniconda3/bin/python tools/export_qwen35/export.py \
  --model_dir "$MODEL_DIR" \
  --output "$CHECKPOINT" \
  --max_seq_len 128 \
  --weight_dtype bf16
```

然后运行：

```bash
GLOG_logtostderr=1 ./build/demo/qwen35_infer \
  "$CHECKPOINT" \
  "$MODEL_DIR/tokenizer.json" \
  21
```

如果需要更长上下文，必须重新导出更大的 `--max_seq_len`。该值会增加 full-attention
KV cache 和 score buffer；在 12 GB 显存上应逐级测试 512、2048、4096、8192，而不是
直接使用原配置的 262144。

### 9.5 当前可通过的 GPU 回归命令

重新生成 tiny fixture：

```bash
/home/tuesday/miniconda3/bin/python test/test_model/make_tiny_qwen35.py \
  --out_dir /tmp/qwen35-stage4-tiny

/home/tuesday/miniconda3/bin/python tools/export_qwen35/export.py \
  --model_dir /tmp/qwen35-stage4-tiny \
  --output /tmp/kuiper-qwen35-stage4-tiny-bf16.bin \
  --max_seq_len 128 \
  --weight_dtype bf16
```

在修复双重 stream destroy 前，运行其余 51 项：

```bash
GLOG_logtostderr=1 \
KUIPER_TINY_QWEN35=/tmp/kuiper-qwen35-stage4-tiny-bf16.bin \
KUIPER_TINY_QWEN35_TOKENIZER=/home/tuesday/workspace/icd/models/Qwen__Qwen3.5-0.8B/tokenizer.json \
./build/test/test_llm \
  --gtest_filter='-Qwen35ZeroCenteredRMSNorm.CudaMatchesCpu'
```

---

## 10. 建议的后续开发顺序

1. 修复 RMSNorm CUDA 测试中的 stream 双重销毁，确保 52/52 tests passed，并单独提交。
2. 给 `qwen35_trace` 增加 CUDA 模式，用真实 4B 保存逐层 hidden/logits，与 CPU/HF 比较。
3. 为新 CUDA kernel 统一增加 launch error 检查，并增加 BF16 大尺寸 matmul 专项测试。
4. 将 `test_llm` 和 tiny fixture 接入 CTest/CI。
5. 优化 BF16 matmul：优先评估 cuBLASLt BF16 weight + FP32 compute，或实现 Tensor Core
   tiled kernel。
6. 实现 GDN chunk prefill，解决长 prompt 逐 token 串行问题。
7. 设计 Qwen3.5 INT8 checkpoint 和 kernel，目标是在 12 GB 4070 SUPER 上运行 9B。
8. 下载并验证 9B 独立 `lm_head` 路径。

建议继续遵守“一项完整功能一个 commit”的规则：实现、测试、文档属于同一功能时放入
同一个 commit；互不依赖的修复和优化分别提交。

---

## 11. 仓库和产物状态

- GitHub 用户仓库：`Emiya-7/KuiperLLama`
- 当前功能分支：`feat/qwen35-stage4-low-precision`
- 当前 PR：`#3 feat(qwen35): add BF16 weight-only inference`
- 上游仓库：`zjhellofss/KuiperLLama`
- 权重、导出 checkpoint、trace、`build/`、`lib/` 和 Python cache 均不应提交 Git。

关键文件：

| 文件 | 作用 |
|---|---|
| [`QWEN35_STATUS.md`](QWEN35_STATUS.md) | 架构公式、详细实现状态和历史验证 |
| [`tools/env.sh`](tools/env.sh) | 项目 CUDA 12.8 可复现环境 |
| [`tools/export_qwen35/export.py`](tools/export_qwen35/export.py) | HF safetensors 到 Kuiper checkpoint |
| [`demo/main_qwen35.cpp`](demo/main_qwen35.cpp) | Qwen3.5 CUDA/CPU 推理 demo |
| [`tools/verify_qwen35/`](tools/verify_qwen35/) | Kuiper/HF trace 和比较工具 |
| [`kuiper/source/model/qwen35.cpp`](kuiper/source/model/qwen35.cpp) | 模型加载、buffer、forward、状态管理 |
| [`kuiper/source/op/kernels/cuda/qwen35_kernel.cu`](kuiper/source/op/kernels/cuda/qwen35_kernel.cu) | Qwen3.5 CUDA kernel |
