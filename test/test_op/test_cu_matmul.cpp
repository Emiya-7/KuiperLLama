#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include "../source/op/kernels/cpu/matmul_kernel.h"
#include "../source/op/kernels/kernels_interface.h"
#include "../utils.cuh"
#include "base/buffer.h"
#include "base/bfloat16.h"
#include "op/matmul.h"
using namespace kernel;

TEST(test_matmul_bf16, fp32_input_bf16_weight_cpu) {
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor input(base::DataType::kDataTypeFp32, 4, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeBf16, 3, 4, true, alloc_cpu);
  tensor::Tensor output(base::DataType::kDataTypeFp32, 3, true, alloc_cpu);

  const float input_values[4] = {1.f, -2.f, 0.5f, 3.f};
  const float weight_values[12] = {1.f,  2.f,  3.f,  4.f,  -1.f, 0.5f,
                                   2.f,  -3.f, 0.25f, 0.5f, 1.f,  -1.f};
  for (int i = 0; i < 4; ++i) {
    input.index<float>(i) = input_values[i];
  }
  for (int i = 0; i < 12; ++i) {
    weight.index<uint16_t>(i) = base::float_to_bfloat16(weight_values[i]);
  }

  matmul_kernel_cpu(input, weight, output);
  EXPECT_FLOAT_EQ(output.index<float>(0), 10.5f);
  EXPECT_FLOAT_EQ(output.index<float>(1), -10.f);
  EXPECT_FLOAT_EQ(output.index<float>(2), -3.25f);
}

TEST(test_matmul_bf16, batched_cpu_uses_token_major_layout) {
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor input(base::DataType::kDataTypeFp32, 2, 3, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeBf16, 2, 3, true, alloc_cpu);
  tensor::Tensor output(base::DataType::kDataTypeFp32, 2, 2, true, alloc_cpu);

  const float input_values[6] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  const float weight_values[6] = {1.f, 0.f, -1.f, 0.5f, 2.f, 0.f};
  for (int i = 0; i < 6; ++i) {
    input.index<float>(i) = input_values[i];
    weight.index<uint16_t>(i) = base::float_to_bfloat16(weight_values[i]);
  }

  matmul_kernel_cpu(input, weight, output);
  EXPECT_FLOAT_EQ(output.index<float>(0), -2.f);
  EXPECT_FLOAT_EQ(output.index<float>(1), 4.5f);
  EXPECT_FLOAT_EQ(output.index<float>(2), -2.f);
  EXPECT_FLOAT_EQ(output.index<float>(3), 12.f);
}

TEST(test_matmul_bf16, batched_cuda_tiled_gemm_matches_cpu) {
  int device_count = 0;
  const cudaError_t device_status = cudaGetDeviceCount(&device_count);
  if (device_status != cudaSuccess || device_count == 0) {
    cudaGetLastError();
    GTEST_SKIP() << "CUDA device unavailable";
  }

  // All three dimensions have tails relative to the 16x16x32 CUDA tile.
  constexpr int32_t kBatchSize = 13;
  constexpr int32_t kInputSize = 259;
  constexpr int32_t kOutputSize = 37;
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
  auto alloc_cuda = base::CUDADeviceAllocatorFactory::get_instance();
  tensor::Tensor input_cpu(base::DataType::kDataTypeFp32, kBatchSize, kInputSize, true,
                           alloc_cpu);
  tensor::Tensor weight_cpu(base::DataType::kDataTypeBf16, kOutputSize, kInputSize, true,
                            alloc_cpu);
  tensor::Tensor output_cpu(base::DataType::kDataTypeFp32, kBatchSize, kOutputSize, true,
                            alloc_cpu);

  for (int64_t i = 0; i < static_cast<int64_t>(kBatchSize) * kInputSize; ++i) {
    input_cpu.index<float>(i) = static_cast<float>(i % 23 - 11) / 16.f;
  }
  for (int64_t i = 0; i < static_cast<int64_t>(kOutputSize) * kInputSize; ++i) {
    weight_cpu.index<uint16_t>(i) =
        base::float_to_bfloat16(static_cast<float>((i * 7) % 29 - 14) / 32.f);
  }
  matmul_kernel_cpu(input_cpu, weight_cpu, output_cpu);

  tensor::Tensor input_cuda = input_cpu.clone();
  tensor::Tensor weight_cuda = weight_cpu.clone();
  input_cuda.to_cuda();
  weight_cuda.to_cuda();
  tensor::Tensor output_cuda(base::DataType::kDataTypeFp32, kBatchSize, kOutputSize, true,
                             alloc_cuda);
  auto config = std::make_shared<CudaConfig>();
  ASSERT_EQ(cudaStreamCreate(&config->stream), cudaSuccess);
  op::MatmulLayer layer(base::DeviceType::kDeviceCUDA, kOutputSize, kInputSize);
  layer.set_cuda_config(config);
  ASSERT_TRUE(layer.set_weight(0, weight_cuda));
  op::Layer& base_layer = layer;
  ASSERT_TRUE(base_layer.forward(input_cuda, output_cuda));
  ASSERT_EQ(cudaStreamSynchronize(config->stream), cudaSuccess);
  output_cuda.to_cpu();

  double max_abs = 0.0;
  double reference_scale = 0.0;
  for (int64_t i = 0; i < static_cast<int64_t>(kBatchSize) * kOutputSize; ++i) {
    const double expected = output_cpu.index<float>(i);
    const double actual = output_cuda.index<float>(i);
    ASSERT_TRUE(std::isfinite(actual)) << "index=" << i;
    max_abs = std::max(max_abs, std::abs(actual - expected));
    reference_scale = std::max(reference_scale, std::abs(expected));
  }
  EXPECT_LT(max_abs / std::max(reference_scale, 1e-6), 2e-5)
      << "max|cpu-cuda|=" << max_abs << ", |reference|max=" << reference_scale;
}

TEST(test_matmul_bf16, qwen35_4b_projection_cuda_matches_cpu) {
  int device_count = 0;
  const cudaError_t device_status = cudaGetDeviceCount(&device_count);
  if (device_status != cudaSuccess || device_count == 0) {
    cudaGetLastError();
    GTEST_SKIP() << "CUDA device unavailable";
  }

  // Qwen3.5-4B in_proj_z is [4096, 2560]. This exercises more than ten
  // million BF16 weights instead of only validating a toy matrix.
  constexpr int32_t kInputSize = 2560;
  constexpr int32_t kOutputSize = 4096;
  constexpr float kScale = -0.375f;
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
  auto alloc_cuda = base::CUDADeviceAllocatorFactory::get_instance();
  tensor::Tensor input_cpu(base::DataType::kDataTypeFp32, kInputSize, true, alloc_cpu);
  tensor::Tensor weight_cpu(base::DataType::kDataTypeBf16, kOutputSize, kInputSize, true,
                            alloc_cpu);
  tensor::Tensor output_cpu(base::DataType::kDataTypeFp32, kOutputSize, true, alloc_cpu);

  for (int32_t column = 0; column < kInputSize; ++column) {
    input_cpu.index<float>(column) = static_cast<float>(column % 29 - 14) / 32.f;
  }
  for (int32_t row = 0; row < kOutputSize; ++row) {
    for (int32_t column = 0; column < kInputSize; ++column) {
      const int32_t pattern = (row * 17 + column * 13) % 31 - 15;
      weight_cpu.index<uint16_t>(static_cast<int64_t>(row) * kInputSize + column) =
          base::float_to_bfloat16(static_cast<float>(pattern) / 64.f);
    }
  }
  matmul_kernel_cpu(input_cpu, weight_cpu, output_cpu, kScale);

  tensor::Tensor input_cuda = input_cpu.clone();
  tensor::Tensor weight_cuda = weight_cpu.clone();
  input_cuda.to_cuda();
  weight_cuda.to_cuda();
  tensor::Tensor output_cuda(base::DataType::kDataTypeFp32, kOutputSize, true, alloc_cuda);

  CudaConfig config;
  ASSERT_EQ(cudaStreamCreate(&config.stream), cudaSuccess);
  get_matmul_kernel(base::DeviceType::kDeviceCUDA)(input_cuda, weight_cuda, output_cuda, kScale,
                                                    &config);
  ASSERT_EQ(cudaStreamSynchronize(config.stream), cudaSuccess);
  output_cuda.to_cpu();

  double max_abs = 0.0;
  double reference_scale = 0.0;
  for (int32_t row = 0; row < kOutputSize; ++row) {
    const double expected = output_cpu.index<float>(row);
    const double actual = output_cuda.index<float>(row);
    ASSERT_TRUE(std::isfinite(actual)) << "row=" << row;
    max_abs = std::max(max_abs, std::abs(actual - expected));
    reference_scale = std::max(reference_scale, std::abs(expected));
  }
  EXPECT_LT(max_abs / std::max(reference_scale, 1e-6), 2e-5)
      << "max|cpu-cuda|=" << max_abs << ", |reference|max=" << reference_scale;
}

TEST(test_matmul_bf16, cuda_applies_scale_and_handles_odd_input_size) {
  int device_count = 0;
  const cudaError_t device_status = cudaGetDeviceCount(&device_count);
  if (device_status != cudaSuccess || device_count == 0) {
    cudaGetLastError();
    GTEST_SKIP() << "CUDA device unavailable";
  }

  // An odd M protects the scalar tail needed by the later BF16x2 fast path.
  constexpr int32_t kInputSize = 259;
  constexpr int32_t kOutputSize = 37;
  constexpr float kScale = -0.375f;
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
  auto alloc_cuda = base::CUDADeviceAllocatorFactory::get_instance();
  tensor::Tensor input_cpu(base::DataType::kDataTypeFp32, kInputSize, true, alloc_cpu);
  tensor::Tensor weight_cpu(base::DataType::kDataTypeBf16, kOutputSize, kInputSize, true,
                            alloc_cpu);
  tensor::Tensor output_cpu(base::DataType::kDataTypeFp32, kOutputSize, true, alloc_cpu);

  for (int32_t column = 0; column < kInputSize; ++column) {
    input_cpu.index<float>(column) = static_cast<float>(column % 23 - 11) / 16.f;
  }
  for (int32_t row = 0; row < kOutputSize; ++row) {
    for (int32_t column = 0; column < kInputSize; ++column) {
      const int32_t pattern = (row * 19 + column * 7) % 29 - 14;
      weight_cpu.index<uint16_t>(static_cast<int64_t>(row) * kInputSize + column) =
          base::float_to_bfloat16(static_cast<float>(pattern) / 32.f);
    }
  }
  matmul_kernel_cpu(input_cpu, weight_cpu, output_cpu, kScale);

  tensor::Tensor input_cuda = input_cpu.clone();
  tensor::Tensor weight_cuda = weight_cpu.clone();
  input_cuda.to_cuda();
  weight_cuda.to_cuda();
  tensor::Tensor output_cuda(base::DataType::kDataTypeFp32, kOutputSize, true, alloc_cuda);

  CudaConfig config;
  ASSERT_EQ(cudaStreamCreate(&config.stream), cudaSuccess);
  get_matmul_kernel(base::DeviceType::kDeviceCUDA)(input_cuda, weight_cuda, output_cuda, kScale,
                                                    &config);
  ASSERT_EQ(cudaStreamSynchronize(config.stream), cudaSuccess);
  output_cuda.to_cpu();
  ASSERT_EQ(cudaStreamDestroy(config.stream), cudaSuccess);
  config.stream = nullptr;

  double max_abs = 0.0;
  double reference_scale = 0.0;
  for (int32_t row = 0; row < kOutputSize; ++row) {
    const double expected = output_cpu.index<float>(row);
    const double actual = output_cuda.index<float>(row);
    ASSERT_TRUE(std::isfinite(actual)) << "row=" << row;
    max_abs = std::max(max_abs, std::abs(actual - expected));
    reference_scale = std::max(reference_scale, std::abs(expected));
  }
  EXPECT_LT(max_abs / std::max(reference_scale, 1e-6), 2e-5)
      << "max|cpu-cuda|=" << max_abs << ", |reference|max=" << reference_scale;
}

TEST(test_matmul_cu, matmul_linear_stream5) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  tensor::Tensor input(base::DataType::kDataTypeFp32, 4, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeFp32, 4, 4, true, alloc_cpu);

  for (int i = 0; i < 4; ++i) {
    input.index<float>(i) = float(i);
  }

  for (int i = 0; i < 16; ++i) {
    weight.index<float>(i) = float(i);
  }
  tensor::Tensor input_cpu = input.clone();
  tensor::Tensor weight_cpu = weight.clone();

  input.to_cuda(nullptr);
  weight.to_cuda(nullptr);

  tensor::Tensor out_cu(base::DataType::kDataTypeFp32, 4, true, alloc_cu);
  tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, 4, true, alloc_cpu);

  CudaConfig* config = new CudaConfig;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  config->stream = stream;
  kernel::get_matmul_kernel(base::DeviceType::kDeviceCUDA)(input, weight, out_cu, 1.f, config);

  kernel::get_matmul_kernel(base::DeviceType::kDeviceCPU)(input_cpu, weight_cpu, out_cpu, 1.f,
                                                          config);

  out_cu.to_cpu();
  for (int i = 0; i < out_cu.size(); ++i) {
    ASSERT_EQ(out_cu.index<float>(i), out_cpu.index<float>(i));
  }
}

TEST(test_matmul_cu, matmul_linear_course) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  tensor::Tensor input(base::DataType::kDataTypeFp32, 3, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeFp32, 3, 3, true, alloc_cpu);

  input.index<float>(0) = float(1);
  input.index<float>(1) = float(1);
  input.index<float>(2) = float(-1);

  for (int i = 1; i <= 9; ++i) {
    weight.index<float>(i - 1) = float(i);
  }
  tensor::Tensor input_cpu = input.clone();
  tensor::Tensor weight_cpu = weight.clone();

  input.to_cuda(nullptr);
  weight.to_cuda(nullptr);

  tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, 3, true, alloc_cpu);

  kernel::get_matmul_kernel(base::DeviceType::kDeviceCPU)(input_cpu, weight_cpu, out_cpu, 1.f,
                                                          nullptr);

  ASSERT_EQ(out_cpu.index<float>(0), 0);
  ASSERT_EQ(out_cpu.index<float>(1), 3);
  ASSERT_EQ(out_cpu.index<float>(2), 6);
}

TEST(test_matmul_cu, matmul_linear_course_cuda) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  tensor::Tensor input(base::DataType::kDataTypeFp32, 3, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeFp32, 3, 3, true, alloc_cpu);

  input.index<float>(0) = float(1);
  input.index<float>(1) = float(1);
  input.index<float>(2) = float(-1);

  for (int i = 1; i <= 9; ++i) {
    weight.index<float>(i - 1) = float(i);
  }

  input.to_cuda();
  weight.to_cuda();

  tensor::Tensor out_cu(base::DataType::kDataTypeFp32, 3, true, alloc_cu);

  kernel::get_matmul_kernel(base::DeviceType::kDeviceCUDA)(input, weight, out_cu, 1.f, nullptr);

  tensor::Tensor out_cpu = out_cu.clone();
  out_cpu.to_cpu();

  ASSERT_EQ(out_cpu.index<float>(0), 0);
  ASSERT_EQ(out_cpu.index<float>(1), 3);
  ASSERT_EQ(out_cpu.index<float>(2), 6);
}
