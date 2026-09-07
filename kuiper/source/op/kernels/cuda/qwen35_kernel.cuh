#ifndef KUIPER_SOURCE_OP_KERNELS_CUDA_QWEN35_KERNEL_CUH_
#define KUIPER_SOURCE_OP_KERNELS_CUDA_QWEN35_KERNEL_CUH_
#include <cuda_runtime_api.h>
#include <cstdint>

// CUDA counterparts of cpu/qwen35_kernel.h. Same semantics, same argument order;
// see that header for what each one computes. Pointers are device pointers.
namespace kernel {

void l2norm_cu(const float* in, float* out, int32_t n, int32_t dim, float eps,
               cudaStream_t stream);

void split_head_interleaved_cu(const float* in, float* first, float* second, int32_t num_heads,
                               int32_t width, cudaStream_t stream);

void silu_cu(const float* in, float* out, int32_t n, cudaStream_t stream);
void sigmoid_cu(const float* in, float* out, int32_t n, cudaStream_t stream);
void mul_cu(const float* a, const float* b, float* out, int32_t n, cudaStream_t stream);

void softplus_decay_cu(const float* a, const float* A_log, const float* dt_bias, float* g,
                       int32_t n, cudaStream_t stream);

void gated_rmsnorm_cu(const float* in, const float* gate, const float* weight, float* out,
                      int32_t n, int32_t dim, float eps, cudaStream_t stream);

void zero_centered_rmsnorm_cu(const float* in, const float* weight, float* out, int32_t n,
                              int32_t dim, float eps, cudaStream_t stream);

void causal_conv1d_decode_cu(const float* in, float* state, const float* weight, float* out,
                             int32_t dim, int32_t k, cudaStream_t stream);

void gated_delta_step_cu(const float* q, const float* k, const float* v, const float* g,
                         const float* beta, float* state, float* out, int32_t num_k_heads,
                         int32_t num_v_heads, int32_t k_head_dim, int32_t v_head_dim,
                         cudaStream_t stream);

void rope_partial_cu(const float* sin_cache, const float* cos_cache, float* q, float* k,
                     int32_t pos, int32_t num_q_heads, int32_t num_k_heads, int32_t head_dim,
                     int32_t rotary_dim, cudaStream_t stream);

void rope_partial_cache_cu(float* sin_cache, float* cos_cache, int32_t max_seq_len,
                           int32_t rotary_dim, float theta, cudaStream_t stream);

}  // namespace kernel
#endif  // KUIPER_SOURCE_OP_KERNELS_CUDA_QWEN35_KERNEL_CUH_
