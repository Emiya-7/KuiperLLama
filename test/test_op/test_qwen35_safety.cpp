#ifdef QWEN35_SUPPORT
#include <gtest/gtest.h>
#include "base/alloc.h"
#include "op/qwen35_ops.h"

namespace {

TEST(Qwen35GatedDelta, ComputesColumnsBeyondFormerFixedScratchLimit) {
  constexpr int32_t kNumHeads = 1;
  constexpr int32_t kKeyDim = 1;
  constexpr int32_t kValueDim = 513;
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();

  tensor::Tensor q(base::DataType::kDataTypeFp32, kKeyDim, true, alloc);
  tensor::Tensor k(base::DataType::kDataTypeFp32, kKeyDim, true, alloc);
  tensor::Tensor v(base::DataType::kDataTypeFp32, kValueDim, true, alloc);
  tensor::Tensor g(base::DataType::kDataTypeFp32, kNumHeads, true, alloc);
  tensor::Tensor beta(base::DataType::kDataTypeFp32, kNumHeads, true, alloc);
  tensor::Tensor state(base::DataType::kDataTypeFp32, kKeyDim * kValueDim, true, alloc);
  tensor::Tensor output(base::DataType::kDataTypeFp32, kValueDim, true, alloc);

  q.index<float>(0) = 1.f;
  k.index<float>(0) = 1.f;
  g.index<float>(0) = 0.f;
  beta.index<float>(0) = 1.f;
  for (int32_t i = 0; i < kValueDim; ++i) {
    v.index<float>(i) = static_cast<float>(i + 1) / kValueDim;
    state.index<float>(i) = 0.f;
    output.index<float>(i) = -1.f;
  }

  op::GatedDeltaLayer layer(base::DeviceType::kDeviceCPU, kNumHeads, kNumHeads, kKeyDim,
                            kValueDim);
  layer.set_input(0, q);
  layer.set_input(1, k);
  layer.set_input(2, v);
  layer.set_input(3, g);
  layer.set_input(4, beta);
  layer.set_input(5, state);
  layer.set_output(0, output);
  ASSERT_TRUE(layer.forward());

  // With zero initial state, q=k=beta=1 and g=0, both the updated state and
  // output equal v. Index 512 was silently left untouched by the old kernel.
  EXPECT_NEAR(output.index<float>(kValueDim - 1), v.index<float>(kValueDim - 1), 1e-6f);
  EXPECT_NEAR(state.index<float>(kValueDim - 1), v.index<float>(kValueDim - 1), 1e-6f);
}

}  // namespace
#endif  // QWEN35_SUPPORT
