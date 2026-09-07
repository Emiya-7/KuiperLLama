#ifndef KUIPER_SOURCE_OP_KERNELS_CPU_QWEN35_KERNEL_H_
#define KUIPER_SOURCE_OP_KERNELS_CPU_QWEN35_KERNEL_H_
#include <cstdint>

// Kernels that only Qwen3.5 needs. Kept separate from the existing rmsnorm /
// rope / mha kernels so the LLama and Qwen2/3 paths are untouched.
//
// These take raw pointers rather than tensor::Tensor: the gated-delta recurrence
// is easier to unit-test against a reference implementation this way, and the
// op::Layer wrappers do the tensor unpacking.
namespace kernel {

// x[n, dim] -> each row scaled to unit L2 norm. Matches flash-linear-attention's
// l2norm (no weight, no mean subtraction), which is what Qwen3.5 applies to the
// GDN query/key. Distinct from rmsnorm: divides by ||x|| not by RMS.
void l2norm_cpu(const float* in, float* out, int32_t n, int32_t dim, float eps);

// Split a per-head interleaved [num_heads, width*2] buffer into two contiguous
// [num_heads, width] buffers.
//
// Qwen3.5's q_proj emits num_heads * head_dim * 2 values that HF reads as
// .view(..., num_heads, head_dim * 2) then chunks in half on the last axis, so on
// disk the layout is [h0_query | h0_gate | h1_query | h1_gate | ...] -- the query
// and the gate of one head are adjacent, NOT two contiguous halves of the whole
// tensor. Treating it as the latter mixes heads together.
void split_head_interleaved_cpu(const float* in, float* first, float* second, int32_t num_heads,
                                int32_t width);

void silu_cpu(const float* in, float* out, int32_t n);
void sigmoid_cpu(const float* in, float* out, int32_t n);

// out = a * b, elementwise. Used for the full-attention output gate.
void mul_cpu(const float* a, const float* b, float* out, int32_t n);

// g = -exp(A_log) * softplus(a + dt_bias), one value per v-head.
void softplus_decay_cpu(const float* a, const float* A_log, const float* dt_bias, float* g,
                        int32_t n);

// out = rmsnorm(x) * weight * silu(gate), per row of `dim`.
void gated_rmsnorm_cpu(const float* in, const float* gate, const float* weight, float* out,
                       int32_t n, int32_t dim, float eps);

// Qwen3.5 zero-centered RMSNorm, independently over each of `n` rows:
// out = rmsnorm(in) * (1 + weight). The weight vector is shared by all rows.
// Used for decoder/final norms and the per-head QK-norm.
void zero_centered_rmsnorm_cpu(const float* in, const float* weight, float* out, int32_t n,
                               int32_t dim, float eps);

// One decode step of the depthwise causal conv1d.
//
// state holds the previous (k-1) inputs per channel, laid out [dim, k-1] with
// the oldest at index 0. It is shifted left in place and the new input appended,
// then out[c] = silu(sum_j state_row[j] * weight[c, j]) where state_row is the
// full k-wide window ending at the new input.
//
// weight is [dim, k] (the [dim,1,k] conv1d.weight with the singleton squeezed).
void causal_conv1d_decode_cpu(const float* in, float* state, const float* weight, float* out,
                              int32_t dim, int32_t k);

// One decode step of the gated delta rule, for every v-head of one layer.
//
//   S       = S * exp(g[h])
//   kv_mem  = k_h^T S                      [v_head_dim]
//   S      += outer(k_h, (v_h - kv_mem) * beta[h])
//   out_h   = S^T q_h                      [v_head_dim]
//
// q/k are [num_k_heads, k_head_dim] and are consumed already L2-normalised;
// this function applies the 1/sqrt(k_head_dim) query scale itself. v is
// [num_v_heads, v_head_dim]. When num_v_heads > num_k_heads (4B and 9B run 2
// v-heads per k-head) v-head h reads k-head h / v_per_k.
//
// state is [num_v_heads, k_head_dim, v_head_dim], updated in place.
void gated_delta_step_cpu(const float* q, const float* k, const float* v, const float* g,
                          const float* beta, float* state, float* out, int32_t num_k_heads,
                          int32_t num_v_heads, int32_t k_head_dim, int32_t v_head_dim);

// RoPE over the leading `rotary_dim` of each head; the rest passes through.
// Qwen3.5 sets partial_rotary_factor=0.25, so only 64 of 256 dims rotate.
// Separate from rope_kernel_cpu, which assumes the whole head rotates.
void rope_partial_cpu(const float* sin_cache, const float* cos_cache, float* q, float* k,
                      int32_t pos, int32_t num_q_heads, int32_t num_k_heads, int32_t head_dim,
                      int32_t rotary_dim);

// sin/cos tables for the partial-RoPE path: [max_seq_len, rotary_dim/2].
void rope_partial_cache_cpu(float* sin_cache, float* cos_cache, int32_t max_seq_len,
                            int32_t rotary_dim, float theta);

}  // namespace kernel
#endif  // KUIPER_SOURCE_OP_KERNELS_CPU_QWEN35_KERNEL_H_
