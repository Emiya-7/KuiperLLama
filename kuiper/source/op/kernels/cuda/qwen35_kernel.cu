#include "qwen35_kernel.cuh"
#include <cub/block/block_reduce.cuh>
#include "cuda_launch_check.cuh"

namespace kernel {

static constexpr int kThreads = 128;

__device__ __forceinline__ float silu_f(float x) { return x / (1.f + __expf(-x)); }

// ------------------------------------------------------------------- l2norm
// One block per row. Rows are k_head_dim (128) wide in the GDN path.
template <int kBlock>
__global__ void l2norm_kernel(const float* in, float* out, int dim, float eps) {
  const int row = blockIdx.x;
  const float* src = in + static_cast<int64_t>(row) * dim;
  float* dst = out + static_cast<int64_t>(row) * dim;

  float partial = 0.f;
  for (int i = threadIdx.x; i < dim; i += kBlock) {
    partial += src[i] * src[i];
  }
  using Reduce = cub::BlockReduce<float, kBlock>;
  __shared__ typename Reduce::TempStorage tmp;
  __shared__ float scale;
  const float total = Reduce(tmp).Sum(partial);
  if (threadIdx.x == 0) {
    scale = rsqrtf(total + eps);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < dim; i += kBlock) {
    dst[i] = src[i] * scale;
  }
}

void l2norm_cu(const float* in, float* out, int32_t n, int32_t dim, float eps,
               cudaStream_t stream) {
  l2norm_kernel<kThreads><<<n, kThreads, 0, stream>>>(in, out, dim, eps);
  check_cuda_kernel_launch("l2norm_kernel");
}

// ------------------------------------------------------ split head interleaved
__global__ void split_head_interleaved_kernel(const float* in, float* first, float* second,
                                              int num_heads, int width) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_heads * width) return;
  const int h = tid / width;
  const int i = tid % width;
  const float* src = in + static_cast<int64_t>(h) * width * 2;
  first[tid] = src[i];
  second[tid] = src[width + i];
}

void split_head_interleaved_cu(const float* in, float* first, float* second, int32_t num_heads,
                               int32_t width, cudaStream_t stream) {
  const int total = num_heads * width;
  split_head_interleaved_kernel<<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      in, first, second, num_heads, width);
  check_cuda_kernel_launch("split_head_interleaved_kernel");
}

// --------------------------------------------------------------- elementwise
__global__ void silu_kernel(const float* in, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = silu_f(in[i]);
}
__global__ void sigmoid_kernel(const float* in, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = 1.f / (1.f + __expf(-in[i]));
}
__global__ void mul_kernel(const float* a, const float* b, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] * b[i];
}

void silu_cu(const float* in, float* out, int32_t n, cudaStream_t stream) {
  silu_kernel<<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(in, out, n);
  check_cuda_kernel_launch("silu_kernel");
}
void sigmoid_cu(const float* in, float* out, int32_t n, cudaStream_t stream) {
  sigmoid_kernel<<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(in, out, n);
  check_cuda_kernel_launch("sigmoid_kernel");
}
void mul_cu(const float* a, const float* b, float* out, int32_t n, cudaStream_t stream) {
  mul_kernel<<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(a, b, out, n);
  check_cuda_kernel_launch("mul_kernel");
}

// ------------------------------------------------------------ softplus decay
__global__ void softplus_decay_kernel(const float* a, const float* A_log, const float* dt_bias,
                                      float* g, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float x = a[i] + dt_bias[i];
  // Guard large x the same way the CPU path does, so the two agree bit-for-bit
  // in the tail where log1p(exp(x)) would overflow.
  const float sp = x > 20.f ? x : __logf(1.f + __expf(x));
  g[i] = -__expf(A_log[i]) * sp;
}

void softplus_decay_cu(const float* a, const float* A_log, const float* dt_bias, float* g,
                       int32_t n, cudaStream_t stream) {
  softplus_decay_kernel<<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(a, A_log, dt_bias,
                                                                                g, n);
  check_cuda_kernel_launch("softplus_decay_kernel");
}

// ----------------------------------------------------------- gated rmsnorm
template <int kBlock>
__global__ void gated_rmsnorm_kernel(const float* in, const float* gate, const float* weight,
                                     float* out, int dim, float eps) {
  const int row = blockIdx.x;
  const float* src = in + static_cast<int64_t>(row) * dim;
  const float* gsrc = gate + static_cast<int64_t>(row) * dim;
  float* dst = out + static_cast<int64_t>(row) * dim;

  float partial = 0.f;
  for (int i = threadIdx.x; i < dim; i += kBlock) {
    partial += src[i] * src[i];
  }
  using Reduce = cub::BlockReduce<float, kBlock>;
  __shared__ typename Reduce::TempStorage tmp;
  __shared__ float scale;
  const float total = Reduce(tmp).Sum(partial);
  if (threadIdx.x == 0) {
    scale = rsqrtf(total / static_cast<float>(dim) + eps);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < dim; i += kBlock) {
    dst[i] = src[i] * scale * weight[i] * silu_f(gsrc[i]);
  }
}

void gated_rmsnorm_cu(const float* in, const float* gate, const float* weight, float* out,
                      int32_t n, int32_t dim, float eps, cudaStream_t stream) {
  gated_rmsnorm_kernel<kThreads><<<n, kThreads, 0, stream>>>(in, gate, weight, out, dim, eps);
  check_cuda_kernel_launch("gated_rmsnorm_kernel");
}

// ----------------------------------------------- zero-centered rmsnorm
// One block per row. Qwen3.5 stores a zero-centered scale, hence (1 + weight).
template <int kBlock>
__global__ void zero_centered_rmsnorm_kernel(const float* in, const float* weight, float* out,
                                             int dim, float eps) {
  const int row = blockIdx.x;
  const float* src = in + static_cast<int64_t>(row) * dim;
  float* dst = out + static_cast<int64_t>(row) * dim;

  float partial = 0.f;
  for (int i = threadIdx.x; i < dim; i += kBlock) {
    partial += src[i] * src[i];
  }
  using Reduce = cub::BlockReduce<float, kBlock>;
  __shared__ typename Reduce::TempStorage tmp;
  __shared__ float scale;
  const float total = Reduce(tmp).Sum(partial);
  if (threadIdx.x == 0) {
    scale = rsqrtf(total / static_cast<float>(dim) + eps);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < dim; i += kBlock) {
    dst[i] = src[i] * scale * (1.f + weight[i]);
  }
}

void zero_centered_rmsnorm_cu(const float* in, const float* weight, float* out, int32_t n,
                              int32_t dim, float eps, cudaStream_t stream) {
  zero_centered_rmsnorm_kernel<kThreads><<<n, kThreads, 0, stream>>>(in, weight, out, dim, eps);
  check_cuda_kernel_launch("zero_centered_rmsnorm_kernel");
}

// ------------------------------------------------------- causal conv1d decode
// Depthwise and channel-independent, so one thread per channel. k is 4, so the
// window fits in registers and the shift is three moves.
__global__ void causal_conv1d_decode_kernel(const float* in, float* state, const float* weight,
                                            float* out, int dim, int k) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= dim) return;
  const int hist = k - 1;
  float* srow = state + static_cast<int64_t>(c) * hist;
  const float* w = weight + static_cast<int64_t>(c) * k;

  float acc = 0.f;
  for (int j = 0; j < hist; ++j) {
    acc += srow[j] * w[j];
  }
  acc += in[c] * w[hist];
  for (int j = 0; j + 1 < hist; ++j) {
    srow[j] = srow[j + 1];
  }
  if (hist > 0) {
    srow[hist - 1] = in[c];
  }
  out[c] = silu_f(acc);
}

void causal_conv1d_decode_cu(const float* in, float* state, const float* weight, float* out,
                             int32_t dim, int32_t k, cudaStream_t stream) {
  causal_conv1d_decode_kernel<<<(dim + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      in, state, weight, out, dim, k);
  check_cuda_kernel_launch("causal_conv1d_decode_kernel");
}

// ---------------------------------------------------------- gated delta step
// One block per v-head, one thread per v-dim column. The state for a head is
// [k_head_dim, v_head_dim]; thread j owns column j, so the outer-product update
// and the query read are both column-local and need no cross-thread traffic.
// The only reduction is kv_mem = k^T S, which is a per-column dot product --
// also column-local. Hence no __syncthreads inside the k loop.
template <int kBlock>
__global__ void gated_delta_step_kernel(const float* q, const float* k, const float* v,
                                        const float* g, const float* beta, float* state,
                                        float* out, int num_k_heads, int num_v_heads,
                                        int k_head_dim, int v_head_dim, float q_scale) {
  const int h = blockIdx.x;
  if (h >= num_v_heads) return;
  const int v_per_k = num_v_heads / num_k_heads;
  const int kh = h / v_per_k;

  const float* q_h = q + static_cast<int64_t>(kh) * k_head_dim;
  const float* k_h = k + static_cast<int64_t>(kh) * k_head_dim;
  const float* v_h = v + static_cast<int64_t>(h) * v_head_dim;
  float* S = state + static_cast<int64_t>(h) * k_head_dim * v_head_dim;
  float* out_h = out + static_cast<int64_t>(h) * v_head_dim;

  const float decay = __expf(g[h]);
  const float beta_h = beta[h];

  // Cache k in shared memory: every thread sweeps the whole k vector twice.
  extern __shared__ float k_sh[];
  for (int i = threadIdx.x; i < k_head_dim; i += kBlock) {
    k_sh[i] = k_h[i];
  }
  __syncthreads();

  for (int j = threadIdx.x; j < v_head_dim; j += kBlock) {
    // Pass 1: kv_mem for this column, with the decay folded in.
    float kv_mem = 0.f;
    for (int i = 0; i < k_head_dim; ++i) {
      kv_mem += k_sh[i] * S[static_cast<int64_t>(i) * v_head_dim + j] * decay;
    }
    const float delta = (v_h[j] - kv_mem) * beta_h;

    // Pass 2: apply decay + rank-1 update, and accumulate the query read.
    float acc = 0.f;
    for (int i = 0; i < k_head_dim; ++i) {
      const int64_t idx = static_cast<int64_t>(i) * v_head_dim + j;
      const float s = S[idx] * decay + k_sh[i] * delta;
      S[idx] = s;
      acc += q_h[i] * q_scale * s;
    }
    out_h[j] = acc;
  }
}

// Qwen3.5-4B uses 32 V heads with a [128, 128] state per head. An 8-column V
// tile and 16-way K split raise the launch from 32 to 512 blocks. Each group
// of eight neighboring threads accesses eight contiguous state columns; the
// four groups in a warp touch independent K rows. Shared memory is used for
// the cross-warp reductions, the resulting delta, and the reused K vector.
constexpr int kGdn4BKHeadDim = 128;
constexpr int kGdn4BVHeadDim = 128;
constexpr int kGdnVTile = 8;
constexpr int kGdnKLanes = 16;

__global__ void gated_delta_step_kernel_4b_tiled(const float* q, const float* k, const float* v,
                                                  const float* g, const float* beta, float* state,
                                                  float* out, float q_scale) {
  const int v_lane = threadIdx.x % kGdnVTile;
  const int k_lane = threadIdx.x / kGdnVTile;
  const int j = blockIdx.x * kGdnVTile + v_lane;
  const int h = blockIdx.y;
  const int kh = h / 2;

  const float* q_h = q + static_cast<int64_t>(kh) * kGdn4BKHeadDim;
  const float* k_h = k + static_cast<int64_t>(kh) * kGdn4BKHeadDim;
  const float* v_h = v + static_cast<int64_t>(h) * kGdn4BVHeadDim;
  float* S = state + static_cast<int64_t>(h) * kGdn4BKHeadDim * kGdn4BVHeadDim;
  float* out_h = out + static_cast<int64_t>(h) * kGdn4BVHeadDim;

  __shared__ float k_sh[kGdn4BKHeadDim];
  __shared__ float partial[kGdnKLanes][kGdnVTile];
  __shared__ float delta_sh[kGdnVTile];
  __shared__ float head_scalars[2];
  if (threadIdx.x < kGdn4BKHeadDim) {
    k_sh[threadIdx.x] = k_h[threadIdx.x];
  }
  if (threadIdx.x == 0) {
    head_scalars[0] = __expf(g[h]);
    head_scalars[1] = beta[h];
  }
  __syncthreads();

  const float decay = head_scalars[0];
  constexpr int kStateValuesPerThread = kGdn4BKHeadDim / kGdnKLanes;
  float state_values[kStateValuesPerThread];
  float kv_partial = 0.f;
#pragma unroll
  for (int item = 0; item < kStateValuesPerThread; ++item) {
    const int i = k_lane + item * kGdnKLanes;
    const float state_value = S[static_cast<int64_t>(i) * kGdn4BVHeadDim + j];
    state_values[item] = state_value;
    kv_partial += k_sh[i] * state_value * decay;
  }
  partial[k_lane][v_lane] = kv_partial;
  __syncthreads();

  if (k_lane == 0) {
    float kv_mem = partial[0][v_lane];
#pragma unroll
    for (int lane = 1; lane < kGdnKLanes; ++lane) {
      kv_mem += partial[lane][v_lane];
    }
    delta_sh[v_lane] = (v_h[j] - kv_mem) * head_scalars[1];
  }
  __syncthreads();

  const float delta = delta_sh[v_lane];
  float out_partial = 0.f;
#pragma unroll
  for (int item = 0; item < kStateValuesPerThread; ++item) {
    const int i = k_lane + item * kGdnKLanes;
    const int64_t idx = static_cast<int64_t>(i) * kGdn4BVHeadDim + j;
    const float s = state_values[item] * decay + k_sh[i] * delta;
    S[idx] = s;
    out_partial += q_h[i] * q_scale * s;
  }
  partial[k_lane][v_lane] = out_partial;
  __syncthreads();

  if (k_lane == 0) {
    float value = partial[0][v_lane];
#pragma unroll
    for (int lane = 1; lane < kGdnKLanes; ++lane) {
      value += partial[lane][v_lane];
    }
    out_h[j] = value;
  }
}

void gated_delta_step_cu(const float* q, const float* k, const float* v, const float* g,
                         const float* beta, float* state, float* out, int32_t num_k_heads,
                         int32_t num_v_heads, int32_t k_head_dim, int32_t v_head_dim,
                         cudaStream_t stream) {
  const float q_scale = rsqrtf(static_cast<float>(k_head_dim));
  if (num_k_heads == 16 && num_v_heads == 32 && k_head_dim == kGdn4BKHeadDim &&
      v_head_dim == kGdn4BVHeadDim) {
    constexpr dim3 block(kGdnVTile * kGdnKLanes);
    constexpr dim3 grid(kGdn4BVHeadDim / kGdnVTile, 32);
    gated_delta_step_kernel_4b_tiled<<<grid, block, 0, stream>>>(q, k, v, g, beta, state, out,
                                                                q_scale);
    check_cuda_kernel_launch("gated_delta_step_kernel_4b_tiled");
    return;
  }

  const size_t shmem = static_cast<size_t>(k_head_dim) * sizeof(float);
  gated_delta_step_kernel<kThreads><<<num_v_heads, kThreads, shmem, stream>>>(
      q, k, v, g, beta, state, out, num_k_heads, num_v_heads, k_head_dim, v_head_dim, q_scale);
  check_cuda_kernel_launch("gated_delta_step_kernel");
}

// -------------------------------------------------------------- partial rope
__global__ void rope_partial_kernel(const float* sin_cache, const float* cos_cache, float* q,
                                    float* k, int pos, int num_q_heads, int num_k_heads,
                                    int head_dim, int rotary_dim) {
  const int half = rotary_dim / 2;
  const int total = (num_q_heads + num_k_heads) * half;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total) return;

  const int q_pairs = num_q_heads * half;
  float* base;
  int i;
  if (tid < q_pairs) {
    base = q + static_cast<int64_t>(tid / half) * head_dim;
    i = tid % half;
  } else {
    const int t = tid - q_pairs;
    base = k + static_cast<int64_t>(t / half) * head_dim;
    i = t % half;
  }
  const float c = cos_cache[static_cast<int64_t>(pos) * half + i];
  const float s = sin_cache[static_cast<int64_t>(pos) * half + i];
  const float x1 = base[i];
  const float x2 = base[i + half];
  base[i] = x1 * c - x2 * s;
  base[i + half] = x2 * c + x1 * s;
  // Dims >= rotary_dim are deliberately left alone.
}

void rope_partial_cu(const float* sin_cache, const float* cos_cache, float* q, float* k,
                     int32_t pos, int32_t num_q_heads, int32_t num_k_heads, int32_t head_dim,
                     int32_t rotary_dim, cudaStream_t stream) {
  const int total = (num_q_heads + num_k_heads) * (rotary_dim / 2);
  rope_partial_kernel<<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      sin_cache, cos_cache, q, k, pos, num_q_heads, num_k_heads, head_dim, rotary_dim);
  check_cuda_kernel_launch("rope_partial_kernel");
}

__global__ void rope_partial_cache_kernel(float* sin_cache, float* cos_cache, int max_seq_len,
                                          int rotary_dim, float theta) {
  const int half = rotary_dim / 2;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= max_seq_len * half) return;
  const int pos = tid / half;
  const int i = tid % half;
  const float freq =
      1.f / powf(theta, static_cast<float>(2 * i) / static_cast<float>(rotary_dim));
  const float val = static_cast<float>(pos) * freq;
  sin_cache[tid] = sinf(val);
  cos_cache[tid] = cosf(val);
}

void rope_partial_cache_cu(float* sin_cache, float* cos_cache, int32_t max_seq_len,
                           int32_t rotary_dim, float theta, cudaStream_t stream) {
  const int total = max_seq_len * (rotary_dim / 2);
  rope_partial_cache_kernel<<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      sin_cache, cos_cache, max_seq_len, rotary_dim, theta);
  check_cuda_kernel_launch("rope_partial_cache_kernel");
}

}  // namespace kernel
