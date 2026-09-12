#include <tensor/tensor.h>
#include <cub/block/block_reduce.cuh>
#include <cuda_bf16.h>
#include "../kernels_interface.h"
#include "cuda_launch_check.cuh"
#include "matmul_kernel.cuh"
namespace kernel {
template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32(const float* input, const float* weight, float* output, int M,
                                      int K) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }

  constexpr int pack_size = 4;
  const int pack_num = M / pack_size;
  const int pack_off = pack_size * pack_num;

#pragma unroll
  for (int p = start_row; p < end_row; ++p) {
    sdata[tid] = 0;
    int row_offset = p * M;
    float4* input_float4_ptr = (float4*)input;
    float4* weight_float4_ptr = (float4*)(weight + row_offset);

#pragma unroll
    for (int i = tid; i < pack_num; i += blockDim.x) {
      float4 input_float4 = *(input_float4_ptr + i);
      float4 weight_float4 = *(weight_float4_ptr + i);
      float part_sum = input_float4.x * weight_float4.x + input_float4.y * weight_float4.y +
                       input_float4.z * weight_float4.z + input_float4.w * weight_float4.w;
      sdata[tid] += part_sum;
    }

    for (int i = pack_off + tid; i < M; i += blockDim.x) {
      sdata[tid] += input[i] * weight[row_offset + i];
    }

    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[p] = part_sum;
    }
    __syncthreads();
  }
}

template <int THREAD_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32bf16_scalar(const float* input,
                                                 const __nv_bfloat16* weight, float* output,
                                                 int M, int K, float scale) {
  const unsigned int tid = threadIdx.x;
  const int row = blockIdx.x;
  if (row >= K) {
    return;
  }

  float sum = 0.f;
  const int row_offset = row * M;
  for (int i = tid; i < M; i += THREAD_PER_BLOCK) {
    sum = fmaf(input[i], __bfloat162float(weight[row_offset + i]), sum);
  }

  using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
  __shared__ typename BlockReduce::TempStorage temp;
  const float total = BlockReduce(temp).Sum(sum);
  if (tid == 0) {
    output[row] = total * scale;
  }
}

template <int THREAD_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32bf16x2(const float* input, const __nv_bfloat16* weight,
                                            float* output, int M, int K, float scale) {
  const unsigned int tid = threadIdx.x;
  const int row = blockIdx.x;
  if (row >= K) {
    return;
  }

  const int pair_count = M / 2;
  const float2* input_pairs = reinterpret_cast<const float2*>(input);
  const __nv_bfloat162* weight_pairs =
      reinterpret_cast<const __nv_bfloat162*>(weight + static_cast<size_t>(row) * M);
  float sum0 = 0.f;
  float sum1 = 0.f;
  for (int pair = tid; pair < pair_count; pair += THREAD_PER_BLOCK) {
    const float2 input_pair = input_pairs[pair];
    const float2 weight_pair = __bfloat1622float2(weight_pairs[pair]);
    sum0 = fmaf(input_pair.x, weight_pair.x, sum0);
    sum1 = fmaf(input_pair.y, weight_pair.y, sum1);
  }

  using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
  __shared__ typename BlockReduce::TempStorage temp;
  const float total = BlockReduce(temp).Sum(sum0 + sum1);
  if (tid == 0) {
    output[row] = total * scale;
  }
}

template <int TILE_N, int TILE_K, int TILE_M>
__global__ void matmul_kernel_cu_fp32bf16_gemm(const float* input,
                                               const __nv_bfloat16* weight, float* output, int N,
                                               int M, int K, float scale) {
  // Padding the reduction dimension avoids a shared-memory bank conflict when
  // adjacent threads consume different weight rows at the same M coordinate.
  __shared__ float input_tile[TILE_N][TILE_M + 1];
  __shared__ float weight_tile[TILE_K][TILE_M + 1];

  const int local_k = threadIdx.x;
  const int local_n = threadIdx.y;
  const int output_k = blockIdx.x * TILE_K + local_k;
  const int output_n = blockIdx.y * TILE_N + local_n;
  const int linear_tid = local_n * TILE_K + local_k;
  constexpr int THREADS = TILE_N * TILE_K;
  float sum = 0.f;

  for (int m_start = 0; m_start < M; m_start += TILE_M) {
    for (int index = linear_tid; index < TILE_N * TILE_M; index += THREADS) {
      const int tile_n = index / TILE_M;
      const int tile_m = index % TILE_M;
      const int global_n = blockIdx.y * TILE_N + tile_n;
      const int global_m = m_start + tile_m;
      input_tile[tile_n][tile_m] =
          global_n < N && global_m < M
              ? input[static_cast<size_t>(global_n) * M + global_m]
              : 0.f;
    }
    for (int index = linear_tid; index < TILE_K * TILE_M; index += THREADS) {
      const int tile_k = index / TILE_M;
      const int tile_m = index % TILE_M;
      const int global_k = blockIdx.x * TILE_K + tile_k;
      const int global_m = m_start + tile_m;
      weight_tile[tile_k][tile_m] =
          global_k < K && global_m < M
              ? __bfloat162float(weight[static_cast<size_t>(global_k) * M + global_m])
              : 0.f;
    }
    __syncthreads();

#pragma unroll
    for (int tile_m = 0; tile_m < TILE_M; ++tile_m) {
      sum = fmaf(input_tile[local_n][tile_m], weight_tile[local_k][tile_m], sum);
    }
    __syncthreads();
  }

  if (output_n < N && output_k < K) {
    output[static_cast<size_t>(output_n) * K + output_k] = sum * scale;
  }
}

template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32int8(const float* input, const int8_t* weight,
                                          const float* scales, const int32_t group_size,
                                          float* output, int M, int K) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }
  for (int p = start_row; p < end_row; ++p) {
    sdata[tid] = 0;
    for (int i = tid; i < M; i += THREAD_PER_BLOCK) {
      const int weight_idx = p * M + i;
      const int group_idx = weight_idx / group_size;
      sdata[tid] += input[i] * scales[group_idx] * static_cast<float>(weight[weight_idx]);
    }
    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[p] = part_sum;
    }
    __syncthreads();
  }
}

void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, const float scale, const CudaConfig* config) {
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  const int32_t K = weight.get_dim(0);  // row
  const int32_t M = weight.get_dim(1);  // col

  CHECK(output.is_empty() == false);
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);
  const bool batched = input.dims_size() == 2;
  const int32_t N = batched ? input.get_dim(0) : 1;
  CHECK_EQ(M, batched ? input.get_dim(1) : input.get_dim(0));
  CHECK_EQ(output.size(), static_cast<size_t>(N) * K);
  cudaStream_t stream = config ? config->stream : nullptr;
  if (weight.data_type() == base::DataType::kDataTypeBf16) {
    if (batched) {
      constexpr int TILE_N = 16;
      constexpr int TILE_K = 16;
      constexpr int TILE_M = 32;
      const dim3 block(TILE_K, TILE_N);
      const dim3 grid((K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N);
      matmul_kernel_cu_fp32bf16_gemm<TILE_N, TILE_K, TILE_M><<<grid, block, 0, stream>>>(
          input.ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(weight.ptr<uint16_t>()),
          const_cast<float*>(output.ptr<float>()), N, M, K, scale);
      check_cuda_kernel_launch("matmul_kernel_cu_fp32bf16_gemm");
    } else if (M % 2 == 0) {
      matmul_kernel_cu_fp32bf16x2<256><<<K, 256, 0, stream>>>(
          input.ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(weight.ptr<uint16_t>()),
          const_cast<float*>(output.ptr<float>()), M, K, scale);
      check_cuda_kernel_launch("matmul_kernel_cu_fp32bf16x2");
    } else {
      matmul_kernel_cu_fp32bf16_scalar<128><<<K, 128, 0, stream>>>(
          input.ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(weight.ptr<uint16_t>()),
          const_cast<float*>(output.ptr<float>()), M, K, scale);
      check_cuda_kernel_launch("matmul_kernel_cu_fp32bf16_scalar");
    }
  } else {
    CHECK(weight.data_type() == base::DataType::kDataTypeFp32);
    CHECK(!batched) << "Batched CUDA Matmul currently requires BF16 weights.";
    matmul_kernel_cu_fp32<128, 1><<<K, 128, 0, stream>>>(
        input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), M, K);
    check_cuda_kernel_launch("matmul_kernel_cu_fp32");
  }
}

void matmul_kernel_cu_qint8(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, int32_t group_size,
                            const tensor::Tensor& scale, const CudaConfig* config) {
  CHECK(config != nullptr);
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  const int32_t K = weight.get_dim(0);  // row
  const int32_t M = weight.get_dim(1);  // col
  int packet_size = 4;
  CHECK_EQ(M % packet_size, 0);
  CHECK_EQ(M, input.get_dim(0));
  if (config->stream) {
    matmul_kernel_cu_fp32int8<128, 1><<<K, 128, 0, config->stream>>>(
        input.ptr<float>(), weight.ptr<int8_t>(), scale.ptr<float>(), group_size,
        const_cast<float*>(output.ptr<float>()), M, K);
    check_cuda_kernel_launch("matmul_kernel_cu_fp32int8");
  } else {
    matmul_kernel_cu_fp32int8<128, 1><<<K, 128>>>(input.ptr<float>(), weight.ptr<int8_t>(),
                                                  scale.ptr<float>(), group_size,
                                                  const_cast<float*>(output.ptr<float>()), M, K);
    check_cuda_kernel_launch("matmul_kernel_cu_fp32int8");
  }
}
}  // namespace kernel
