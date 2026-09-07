#ifdef QWEN35_SUPPORT
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include "base/alloc.h"
#include "base/cuda_config.h"
#include "op/qwen35_ops.h"

namespace {

constexpr int32_t kRows = 2;
constexpr int32_t kDim = 4;
constexpr float kEps = 0.25f;

void fill_input_and_weight(tensor::Tensor& input, tensor::Tensor& weight) {
  const float values[kRows * kDim] = {1.f, 2.f, 3.f, 4.f, -2.f, 1.f, -4.f, 3.f};
  const float scales[kDim] = {0.f, 0.25f, -0.5f, 1.f};
  for (int32_t i = 0; i < kRows * kDim; ++i) {
    input.index<float>(i) = values[i];
  }
  for (int32_t i = 0; i < kDim; ++i) {
    weight.index<float>(i) = scales[i];
  }
}

void expect_zero_centered_reference(const tensor::Tensor& input, const tensor::Tensor& weight,
                                    const tensor::Tensor& output) {
  for (int32_t row = 0; row < kRows; ++row) {
    float sum = 0.f;
    for (int32_t col = 0; col < kDim; ++col) {
      const float x = input.index<float>(row * kDim + col);
      sum += x * x;
    }
    const float inv_rms = 1.f / std::sqrt(sum / static_cast<float>(kDim) + kEps);
    for (int32_t col = 0; col < kDim; ++col) {
      const int32_t idx = row * kDim + col;
      const float expected =
          input.index<float>(idx) * inv_rms * (1.f + weight.index<float>(col));
      EXPECT_NEAR(output.index<float>(idx), expected, 1e-6f) << "row=" << row << " col=" << col;
    }
  }
}

}  // namespace

TEST(Qwen35ZeroCenteredRMSNorm, WeightZeroMeansUnitScaleAndUsesConfiguredEpsilon) {
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor input(base::DataType::kDataTypeFp32, kRows, kDim, true, alloc);
  tensor::Tensor weight(base::DataType::kDataTypeFp32, kDim, true, alloc);
  tensor::Tensor output(base::DataType::kDataTypeFp32, kRows, kDim, true, alloc);

  for (int32_t i = 0; i < kRows * kDim; ++i) {
    input.index<float>(i) = static_cast<float>(i + 1);
  }
  for (int32_t i = 0; i < kDim; ++i) {
    weight.index<float>(i) = 0.f;
  }

  op::ZeroCenteredRMSNormLayer norm(base::DeviceType::kDeviceCPU, kDim, kEps);
  ASSERT_TRUE(norm.set_weight(0, weight));
  ASSERT_TRUE(norm.forward(input, output));

  for (int32_t row = 0; row < kRows; ++row) {
    float sum = 0.f;
    for (int32_t col = 0; col < kDim; ++col) {
      const float x = input.index<float>(row * kDim + col);
      sum += x * x;
    }
    const float inv_rms = 1.f / std::sqrt(sum / static_cast<float>(kDim) + kEps);
    for (int32_t col = 0; col < kDim; ++col) {
      const int32_t idx = row * kDim + col;
      EXPECT_NEAR(output.index<float>(idx), input.index<float>(idx) * inv_rms, 1e-6f);
    }
  }
}

TEST(Qwen35ZeroCenteredRMSNorm, AppliesOnePlusWeightAndSupportsInPlace) {
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor input(base::DataType::kDataTypeFp32, kRows, kDim, true, alloc);
  tensor::Tensor weight(base::DataType::kDataTypeFp32, kDim, true, alloc);
  fill_input_and_weight(input, weight);
  const tensor::Tensor original = input.clone();

  op::ZeroCenteredRMSNormLayer norm(base::DeviceType::kDeviceCPU, kDim, kEps);
  ASSERT_TRUE(norm.set_weight(0, weight));
  ASSERT_TRUE(norm.forward(input, input));
  expect_zero_centered_reference(original, weight, input);
}

TEST(Qwen35ZeroCenteredRMSNorm, CudaMatchesCpu) {
  int device_count = 0;
  const cudaError_t device_status = cudaGetDeviceCount(&device_count);
  if (device_status != cudaSuccess || device_count == 0) {
    cudaGetLastError();
    GTEST_SKIP() << "CUDA device unavailable";
  }

  auto cpu_alloc = base::CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor input_cpu(base::DataType::kDataTypeFp32, kRows, kDim, true, cpu_alloc);
  tensor::Tensor weight_cpu(base::DataType::kDataTypeFp32, kDim, true, cpu_alloc);
  tensor::Tensor output_cpu(base::DataType::kDataTypeFp32, kRows, kDim, true, cpu_alloc);
  fill_input_and_weight(input_cpu, weight_cpu);

  op::ZeroCenteredRMSNormLayer cpu_norm(base::DeviceType::kDeviceCPU, kDim, kEps);
  ASSERT_TRUE(cpu_norm.set_weight(0, weight_cpu));
  ASSERT_TRUE(cpu_norm.forward(input_cpu, output_cpu));

  tensor::Tensor input_cuda = input_cpu.clone();
  tensor::Tensor weight_cuda = weight_cpu.clone();
  tensor::Tensor output_cuda = output_cpu.clone();
  input_cuda.to_cuda();
  weight_cuda.to_cuda();
  output_cuda.to_cuda();

  auto cuda_config = std::make_shared<kernel::CudaConfig>();
  ASSERT_EQ(cudaStreamCreate(&cuda_config->stream), cudaSuccess);
  op::ZeroCenteredRMSNormLayer cuda_norm(base::DeviceType::kDeviceCUDA, kDim, kEps);
  cuda_norm.set_cuda_config(cuda_config);
  ASSERT_TRUE(cuda_norm.set_weight(0, weight_cuda));
  ASSERT_TRUE(cuda_norm.forward(input_cuda, output_cuda));
  ASSERT_EQ(cudaStreamSynchronize(cuda_config->stream), cudaSuccess);
  output_cuda.to_cpu();

  for (int32_t i = 0; i < kRows * kDim; ++i) {
    EXPECT_NEAR(output_cuda.index<float>(i), output_cpu.index<float>(i), 1e-6f) << "index=" << i;
  }
  EXPECT_EQ(cudaStreamDestroy(cuda_config->stream), cudaSuccess);
}

#endif  // QWEN35_SUPPORT
