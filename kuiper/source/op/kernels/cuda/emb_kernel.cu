#include "emb_kernel.cuh"
#include <cuda_bf16.h>
#include "cuda_launch_check.cuh"
namespace kernel {
__global__ void emb_kernel_cu_fp32(int32_t vocab_size, int32_t token_num, int32_t weight_dim,
                                   const int32_t* input_ptr, const float* weight_ptr,
                                   float* output_ptr) {
  int32_t token_idx = blockIdx.x;
  if (token_idx >= token_num) {
    return;
  }
  int32_t token = input_ptr[token_idx];
  if (token >= vocab_size) {
    return;
  }

  float* output_ptr_start = output_ptr + token_idx * weight_dim;
  const float* weight_ptr_start = weight_ptr + token * weight_dim;

  for (int32_t i = threadIdx.x; i < weight_dim; i += blockDim.x) {
    output_ptr_start[i] = weight_ptr_start[i];
  }
}

__global__ void emb_kernel_cu_bf16(int32_t vocab_size, int32_t token_num, int32_t weight_dim,
                                   const int32_t* input_ptr, const __nv_bfloat16* weight_ptr,
                                   float* output_ptr) {
  const int32_t token_idx = blockIdx.x;
  if (token_idx >= token_num) {
    return;
  }
  const int32_t token = input_ptr[token_idx];
  if (token < 0 || token >= vocab_size) {
    return;
  }

  float* output_start = output_ptr + token_idx * weight_dim;
  const __nv_bfloat16* weight_start = weight_ptr + token * weight_dim;
  for (int32_t i = threadIdx.x; i < weight_dim; i += blockDim.x) {
    output_start[i] = __bfloat162float(weight_start[i]);
  }
}

void emb_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                   const tensor::Tensor& output, int32_t vocab_size, void* stream) {
  tensor::Tensor input_cu;
  if (input.device_type() != base::DeviceType::kDeviceCUDA) {
    input_cu = input.clone();
    input_cu.to_cuda();
  }
  const int32_t input_num = static_cast<int32_t>(input.size());
  const int32_t weight_dim = weight.get_dim(1);
  CHECK(weight.device_type() == output.device_type());
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

  constexpr int32_t max_seq_len = 512;
  constexpr int32_t thread_num = 128;
  int32_t* in_ptr = input_cu.ptr<int32_t>();
  float* out_ptr = const_cast<float*>(output.ptr<float>());
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  if (weight.data_type() == base::DataType::kDataTypeBf16) {
    emb_kernel_cu_bf16<<<max_seq_len, thread_num, 0, cuda_stream>>>(
        vocab_size, input_num, weight_dim, in_ptr,
        reinterpret_cast<const __nv_bfloat16*>(weight.ptr<uint16_t>()), out_ptr);
    check_cuda_kernel_launch("emb_kernel_cu_bf16");
  } else {
    CHECK(weight.data_type() == base::DataType::kDataTypeFp32);
    emb_kernel_cu_fp32<<<max_seq_len, thread_num, 0, cuda_stream>>>(
        vocab_size, input_num, weight_dim, in_ptr, weight.ptr<float>(), out_ptr);
    check_cuda_kernel_launch("emb_kernel_cu_fp32");
  }
}
}  // namespace kernel
