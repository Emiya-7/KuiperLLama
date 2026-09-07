#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../source/op/kernels/kernels_interface.h"
#include "base/buffer.h"
#include "base/bfloat16.h"

TEST(test_emb_bf16, converts_selected_rows_to_fp32_on_cpu) {
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor tokens(base::DataType::kDataTypeInt32, 2, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeBf16, 3, 4, true, alloc_cpu);
  tensor::Tensor output(base::DataType::kDataTypeFp32, 2, 4, true, alloc_cpu);
  tokens.index<int32_t>(0) = 2;
  tokens.index<int32_t>(1) = 0;
  for (int32_t i = 0; i < 12; ++i) {
    weight.index<uint16_t>(i) = base::float_to_bfloat16(static_cast<float>(i) / 2.f);
  }

  kernel::get_emb_kernel(base::DeviceType::kDeviceCPU)(tokens, weight, output, 3, nullptr);
  for (int32_t i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(output.index<float>(i), static_cast<float>(8 + i) / 2.f);
    EXPECT_FLOAT_EQ(output.index<float>(4 + i), static_cast<float>(i) / 2.f);
  }
}
TEST(test_emb_cu, emb1_nostream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t token = 4;
  int32_t dim = 512;
  int32_t size = 2048;

  tensor::Tensor input(base::DataType::kDataTypeFp32, 1, true, alloc_cpu);
  input.index<int32_t>(0) = 1;

  tensor::Tensor weight(base::DataType::kDataTypeFp32, token, dim, true, alloc_cpu);
  tensor::Tensor output(base::DataType::kDataTypeFp32, dim, true, alloc_cu);

  for (int i = 0; i < size; ++i) {
    weight.index<float>(i) = static_cast<float>(i);
  }
  weight.to_cuda();
  kernel::get_emb_kernel(base::DeviceType::kDeviceCUDA)(input, weight, output, token,
                                                        nullptr);

  output.to_cpu();
  for (int i = 0; i < dim; ++i) {
    ASSERT_EQ(output.index<float>(i), 512 + i);
  }
}

TEST(test_emb_cu, emb2_nostream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t token = 4;
  int32_t dim = 512;
  int32_t size = 2048;

  tensor::Tensor input(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  input.index<int32_t>(0) = 2;

  tensor::Tensor weight(base::DataType::kDataTypeFp32, token, dim, true, alloc_cpu);
  tensor::Tensor output(base::DataType::kDataTypeFp32, dim, true, alloc_cu);

  for (int i = 0; i < size; ++i) {
    weight.index<float>(i) = static_cast<float>(i);
  }
  weight.to_cuda();
  kernel::get_emb_kernel(base::DeviceType::kDeviceCUDA)(input, weight, output, token,
                                                        nullptr);

  output.to_cpu();
  for (int i = 0; i < dim; ++i) {
    ASSERT_EQ(output.index<float>(i), 1024 + i);
  }
}

TEST(test_emb_cu, emb1_stream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t token = 4;
  int32_t dim = 512;
  int32_t size = 2048;

  tensor::Tensor input(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  input.index<int32_t>(0) = 1;

  tensor::Tensor weight(base::DataType::kDataTypeFp32, token, dim, true, alloc_cpu);
  tensor::Tensor output(base::DataType::kDataTypeFp32, dim, true, alloc_cu);

  for (int i = 0; i < size; ++i) {
    weight.index<float>(i) = static_cast<float>(i);
  }
  weight.to_cuda();
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  kernel::get_emb_kernel(base::DeviceType::kDeviceCUDA)(input, weight, output, token,
                                                        stream);

  output.to_cpu();
  for (int i = 0; i < dim; ++i) {
    ASSERT_EQ(output.index<float>(i), 512 + i);
  }

  cudaStreamDestroy(stream);
}
