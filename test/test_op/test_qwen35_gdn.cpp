#ifdef QWEN35_SUPPORT
#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "base/alloc.h"
#include "base/cuda_config.h"
#include "op/qwen35_ops.h"

namespace {

struct GdnShape {
  int32_t num_k_heads;
  int32_t num_v_heads;
  int32_t k_head_dim;
  int32_t v_head_dim;
};

void normalize_heads(tensor::Tensor& tensor, int32_t num_heads, int32_t head_dim) {
  for (int32_t head = 0; head < num_heads; ++head) {
    double square_sum = 0.0;
    for (int32_t column = 0; column < head_dim; ++column) {
      const float value = tensor.index<float>(head * head_dim + column);
      square_sum += static_cast<double>(value) * value;
    }
    const float inverse_norm = 1.f / std::sqrt(static_cast<float>(square_sum));
    for (int32_t column = 0; column < head_dim; ++column) {
      tensor.index<float>(head * head_dim + column) *= inverse_norm;
    }
  }
}

void expect_cuda_matches_cpu(const tensor::Tensor& cpu, const std::vector<float>& cuda,
                             int32_t step, const char* value_name) {
  ASSERT_EQ(cpu.size(), cuda.size());
  double max_abs_error = 0.0;
  double reference_scale = 0.0;
  int64_t max_error_index = 0;
  for (int64_t index = 0; index < static_cast<int64_t>(cuda.size()); ++index) {
    const double expected = cpu.index<float>(index);
    const double actual = cuda[index];
    ASSERT_TRUE(std::isfinite(actual))
        << value_name << " contains a non-finite value at step=" << step << " index=" << index;
    const double error = std::abs(actual - expected);
    if (error > max_abs_error) {
      max_abs_error = error;
      max_error_index = index;
    }
    reference_scale = std::max(reference_scale, std::abs(expected));
  }

  const double tolerance = 2e-4 * std::max(reference_scale, 1.0);
  EXPECT_LE(max_abs_error, tolerance)
      << value_name << " mismatch at step=" << step << " index=" << max_error_index
      << " reference_scale=" << reference_scale;
}

void run_recurrent_cuda_reference(const GdnShape& shape,
                                  const std::vector<int32_t>& checkpoints) {
  ASSERT_FALSE(checkpoints.empty());
  ASSERT_TRUE(std::is_sorted(checkpoints.begin(), checkpoints.end()));
  ASSERT_GT(checkpoints.front(), 0);
  ASSERT_EQ(shape.num_v_heads % shape.num_k_heads, 0);

  const int32_t qk_size = shape.num_k_heads * shape.k_head_dim;
  const int32_t value_size = shape.num_v_heads * shape.v_head_dim;
  const int64_t state_size = static_cast<int64_t>(shape.num_v_heads) * shape.k_head_dim *
                             shape.v_head_dim;
  auto cpu_alloc = base::CPUDeviceAllocatorFactory::get_instance();
  auto cuda_alloc = base::CUDADeviceAllocatorFactory::get_instance();

  tensor::Tensor q_cpu(base::DataType::kDataTypeFp32, qk_size, true, cpu_alloc);
  tensor::Tensor k_cpu(base::DataType::kDataTypeFp32, qk_size, true, cpu_alloc);
  tensor::Tensor v_cpu(base::DataType::kDataTypeFp32, value_size, true, cpu_alloc);
  tensor::Tensor g_cpu(base::DataType::kDataTypeFp32, shape.num_v_heads, true, cpu_alloc);
  tensor::Tensor beta_cpu(base::DataType::kDataTypeFp32, shape.num_v_heads, true, cpu_alloc);
  tensor::Tensor state_cpu(base::DataType::kDataTypeFp32, state_size, true, cpu_alloc);
  tensor::Tensor output_cpu(base::DataType::kDataTypeFp32, value_size, true, cpu_alloc);

  for (int32_t index = 0; index < qk_size; ++index) {
    q_cpu.index<float>(index) = static_cast<float>((index * 17) % 31 - 15) / 32.f;
    k_cpu.index<float>(index) = static_cast<float>((index * 13 + 7) % 29 - 14) / 32.f;
  }
  normalize_heads(q_cpu, shape.num_k_heads, shape.k_head_dim);
  normalize_heads(k_cpu, shape.num_k_heads, shape.k_head_dim);
  for (int32_t index = 0; index < value_size; ++index) {
    v_cpu.index<float>(index) = static_cast<float>((index * 11 + 3) % 37 - 18) / 128.f;
  }
  for (int32_t head = 0; head < shape.num_v_heads; ++head) {
    g_cpu.index<float>(head) = -0.125f - static_cast<float>(head % 5) / 64.f;
    beta_cpu.index<float>(head) = 0.25f + static_cast<float>(head % 7) / 32.f;
  }
  for (int64_t index = 0; index < state_size; ++index) {
    state_cpu.index<float>(index) =
        static_cast<float>((index * 7 + 5) % 41 - 20) / 4096.f;
  }

  tensor::Tensor q_cuda = q_cpu.clone();
  tensor::Tensor k_cuda = k_cpu.clone();
  tensor::Tensor v_cuda = v_cpu.clone();
  tensor::Tensor g_cuda = g_cpu.clone();
  tensor::Tensor beta_cuda = beta_cpu.clone();
  tensor::Tensor state_cuda = state_cpu.clone();
  tensor::Tensor output_cuda(base::DataType::kDataTypeFp32, value_size, true, cuda_alloc);

  auto cuda_config = std::make_shared<kernel::CudaConfig>();
  ASSERT_EQ(cudaStreamCreate(&cuda_config->stream), cudaSuccess);
  q_cuda.to_cuda(cuda_config->stream);
  k_cuda.to_cuda(cuda_config->stream);
  v_cuda.to_cuda(cuda_config->stream);
  g_cuda.to_cuda(cuda_config->stream);
  beta_cuda.to_cuda(cuda_config->stream);
  state_cuda.to_cuda(cuda_config->stream);

  op::GatedDeltaLayer cpu_layer(base::DeviceType::kDeviceCPU, shape.num_k_heads,
                                shape.num_v_heads, shape.k_head_dim, shape.v_head_dim);
  cpu_layer.set_input(0, q_cpu);
  cpu_layer.set_input(1, k_cpu);
  cpu_layer.set_input(2, v_cpu);
  cpu_layer.set_input(3, g_cpu);
  cpu_layer.set_input(4, beta_cpu);
  cpu_layer.set_input(5, state_cpu);
  cpu_layer.set_output(0, output_cpu);

  op::GatedDeltaLayer cuda_layer(base::DeviceType::kDeviceCUDA, shape.num_k_heads,
                                 shape.num_v_heads, shape.k_head_dim, shape.v_head_dim);
  cuda_layer.set_cuda_config(cuda_config);
  cuda_layer.set_input(0, q_cuda);
  cuda_layer.set_input(1, k_cuda);
  cuda_layer.set_input(2, v_cuda);
  cuda_layer.set_input(3, g_cuda);
  cuda_layer.set_input(4, beta_cuda);
  cuda_layer.set_input(5, state_cuda);
  cuda_layer.set_output(0, output_cuda);

  std::vector<float> cuda_output(static_cast<size_t>(value_size));
  std::vector<float> cuda_state(static_cast<size_t>(state_size));
  size_t checkpoint_index = 0;
  for (int32_t step = 1; step <= checkpoints.back(); ++step) {
    ASSERT_TRUE(cpu_layer.forward()) << "CPU step=" << step;
    ASSERT_TRUE(cuda_layer.forward()) << "CUDA step=" << step;
    if (step != checkpoints[checkpoint_index]) {
      continue;
    }

    ASSERT_EQ(cudaMemcpyAsync(cuda_output.data(), output_cuda.ptr<float>(),
                              output_cuda.byte_size(), cudaMemcpyDeviceToHost,
                              cuda_config->stream),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(cuda_state.data(), state_cuda.ptr<float>(), state_cuda.byte_size(),
                              cudaMemcpyDeviceToHost, cuda_config->stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(cuda_config->stream), cudaSuccess);
    expect_cuda_matches_cpu(output_cpu, cuda_output, step, "output");
    expect_cuda_matches_cpu(state_cpu, cuda_state, step, "state");

    ++checkpoint_index;
    if (checkpoint_index == checkpoints.size()) {
      break;
    }
  }
}

bool cuda_device_available() {
  int device_count = 0;
  const cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return device_count > 0;
}

TEST(Qwen35GatedDelta, NonZeroStateGroupedShapeMatchesCpuAcross16Steps) {
  if (!cuda_device_available()) {
    GTEST_SKIP() << "CUDA device unavailable";
  }
  run_recurrent_cuda_reference({2, 4, 32, 48}, {1, 16});
}

TEST(Qwen35GatedDelta, Qwen35FourBShapeMatchesCpuAcross128Steps) {
  if (!cuda_device_available()) {
    GTEST_SKIP() << "CUDA device unavailable";
  }
  run_recurrent_cuda_reference({16, 32, 128, 128}, {1, 16, 128});
}

}  // namespace
#endif  // QWEN35_SUPPORT
