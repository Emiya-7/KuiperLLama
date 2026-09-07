#ifdef QWEN35_SUPPORT
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <cmath>
#include <cstdlib>
#include <string>
#include "model/qwen35.h"

// These exercise the hybrid plumbing on a small synthetic checkpoint written by
// test/test_model/make_tiny_qwen35.py: same structure as the published 4B/9B
// models (full_attention_interval 4, 2 v-heads per k-head, partial_rotary_factor
// 0.25) at a size that fits in a unit test.
//
// The checkpoint path comes from KUIPER_TINY_QWEN35; the tests skip when it is
// unset so a plain `ctest` still passes on a machine without it.
namespace {

const char* tiny_model_path() { return std::getenv("KUIPER_TINY_QWEN35"); }
const char* tiny_token_path() { return std::getenv("KUIPER_TINY_QWEN35_TOKENIZER"); }

class Qwen35Tiny : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!tiny_model_path() || !tiny_token_path()) {
      GTEST_SKIP() << "set KUIPER_TINY_QWEN35 and KUIPER_TINY_QWEN35_TOKENIZER to run";
    }
  }
};

// Layer typing and per-kind indexing: the 4:1 pattern must put full attention on
// layers 3 and 7, and each kind must number its own layers from zero so the KV
// cache and the recurrent state stay densely packed.
TEST(Qwen35Config, LayerTypeAndLocalIndex) {
  model::Qwen35Config c;
  c.layer_num = 8;
  c.full_attention_interval = 4;
  c.head_num = 16;
  c.kv_head_num = 4;
  c.head_dim = 256;
  c.linear_num_k_heads = 16;
  c.linear_num_v_heads = 32;
  c.linear_k_head_dim = 128;
  c.linear_v_head_dim = 128;
  c.conv_kernel_size = 4;
  c.derive();

  const model::Qwen35LayerType kLin = model::Qwen35LayerType::kLinearAttention;
  const model::Qwen35LayerType kFull = model::Qwen35LayerType::kFullAttention;
  const model::Qwen35LayerType want[8] = {kLin, kLin, kLin, kFull, kLin, kLin, kLin, kFull};
  const int32_t want_local[8] = {0, 1, 2, 0, 3, 4, 5, 1};
  for (int32_t i = 0; i < 8; ++i) {
    EXPECT_EQ(c.layer_type(i), want[i]) << "layer " << i;
    EXPECT_EQ(c.type_local_idx(i), want_local[i]) << "layer " << i;
  }
  EXPECT_EQ(c.full_layer_num, 2);
  EXPECT_EQ(c.linear_layer_num, 6);

  // Derived sizes for the published 4B/9B linear-attention block.
  EXPECT_EQ(c.linear_k_dim, 2048);
  EXPECT_EQ(c.linear_v_dim, 4096);
  EXPECT_EQ(c.conv_dim, 8192);
  EXPECT_EQ(c.v_per_k, 2);
  EXPECT_EQ(c.q_proj_out, 8192);  // 16 heads * 256 * 2, second half is the gate
  EXPECT_EQ(c.kv_dim, 1024);
  EXPECT_EQ(c.kv_mul, 4);
}

// Loading is where an export/reader layout drift would show up: create_param_layers
// asserts that the offsets it walks land exactly on the end of the file.
TEST_F(Qwen35Tiny, LoadsOnCpu) {
  model::Qwen35Model model(base::TokenizerType::kEncodeBpe, tiny_token_path(), tiny_model_path(),
                           false);
  auto status = model.init(base::DeviceType::kDeviceCPU);
  ASSERT_TRUE(status) << status.get_err_msg();

  const auto& c = model.qwen35_config();
  EXPECT_EQ(c.full_attention_interval, 4);
  EXPECT_EQ(c.v_per_k, 2);
  EXPECT_EQ(c.rotary_dim, c.head_dim / 4);  // partial_rotary_factor 0.25
  EXPECT_GT(c.linear_layer_num, 0);
  EXPECT_GT(c.full_layer_num, 0);
}

// A forward pass over both layer kinds must produce finite logits. NaNs here
// would point at the recurrent state or the conv state being read uninitialised.
TEST_F(Qwen35Tiny, ForwardProducesFiniteLogits) {
  model::Qwen35Model model(base::TokenizerType::kEncodeBpe, tiny_token_path(), tiny_model_path(),
                           false);
  ASSERT_TRUE(model.init(base::DeviceType::kDeviceCPU));
  const auto& c = model.qwen35_config();

  model.reset_state();
  std::vector<int32_t> tokens{1, 2, 3};
  const auto emb = model.embedding(tokens);
  auto pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);

  int next = -1;
  for (int32_t pos = 0; pos < 3; ++pos) {
    pos_tensor.index<int32_t>(0) = pos;
    auto input = model.fill_input(pos_tensor, emb, true);
    ASSERT_TRUE(model.predict(input, pos_tensor, /*is_prompt=*/pos < 2, next));
  }

  const auto logits = model.get_buffer(model::ModelBufferType::kForwardOutput);
  ASSERT_EQ(static_cast<int32_t>(logits.size()), c.vocab_size);
  int finite = 0;
  for (int32_t i = 0; i < c.vocab_size; ++i) {
    const float v = logits.index<float>(i);
    ASSERT_FALSE(std::isnan(v)) << "NaN logit at " << i;
    ASSERT_FALSE(std::isinf(v)) << "inf logit at " << i;
    if (v != 0.f) {
      ++finite;
    }
  }
  EXPECT_GT(finite, 0) << "all logits are exactly zero, the forward pass did nothing";
  EXPECT_GE(next, 0);
  EXPECT_LT(next, c.vocab_size);
}

// reset_state has to actually clear the recurrent state: replaying the same
// tokens after a reset must reproduce the same logits bit for bit. Without it,
// the GDN state would carry over and the second run would differ.
TEST_F(Qwen35Tiny, ResetStateMakesRunsReproducible) {
  model::Qwen35Model model(base::TokenizerType::kEncodeBpe, tiny_token_path(), tiny_model_path(),
                           false);
  ASSERT_TRUE(model.init(base::DeviceType::kDeviceCPU));
  const auto& c = model.qwen35_config();

  auto run = [&]() {
    model.reset_state();
    std::vector<int32_t> tokens{5, 6};
    const auto emb = model.embedding(tokens);
    auto pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);
    int next = -1;
    for (int32_t pos = 0; pos < 2; ++pos) {
      pos_tensor.index<int32_t>(0) = pos;
      auto input = model.fill_input(pos_tensor, emb, true);
      CHECK(model.predict(input, pos_tensor, pos < 1, next));
    }
    const auto logits = model.get_buffer(model::ModelBufferType::kForwardOutput);
    std::vector<float> out(c.vocab_size);
    for (int32_t i = 0; i < c.vocab_size; ++i) {
      out[i] = logits.index<float>(i);
    }
    return out;
  };

  const auto first = run();
  const auto second = run();
  for (int32_t i = 0; i < c.vocab_size; ++i) {
    ASSERT_FLOAT_EQ(first[i], second[i]) << "logit " << i << " differs after reset_state";
  }
}

// CPU and CUDA must agree end to end. This is the check that covers the pieces
// the kernel-level tests cannot: buffer wiring, the recurrent-state slicing per
// linear layer, and KV-cache addressing by full-layer ordinal.
TEST_F(Qwen35Tiny, CudaMatchesCpu) {
  auto logits_for = [](base::DeviceType device) {
    model::Qwen35Model model(base::TokenizerType::kEncodeBpe, tiny_token_path(),
                             tiny_model_path(), false);
    CHECK(model.init(device));
    const auto& c = model.qwen35_config();
    model.reset_state();
    std::vector<int32_t> tokens{7, 8, 9};
    const auto emb = model.embedding(tokens);
    auto pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);
    int next = -1;
    for (int32_t pos = 0; pos < 3; ++pos) {
      pos_tensor.index<int32_t>(0) = pos;
      auto input = model.fill_input(pos_tensor, emb, true);
      CHECK(model.predict(input, pos_tensor, pos < 2, next));
    }
    auto logits = model.get_buffer(model::ModelBufferType::kForwardOutput);
    if (device == base::DeviceType::kDeviceCUDA) {
      logits.to_cpu();
    }
    std::vector<float> out(c.vocab_size);
    for (int32_t i = 0; i < c.vocab_size; ++i) {
      out[i] = logits.index<float>(i);
    }
    return out;
  };

  const auto cpu = logits_for(base::DeviceType::kDeviceCPU);
  const auto gpu = logits_for(base::DeviceType::kDeviceCUDA);
  ASSERT_EQ(cpu.size(), gpu.size());

  double max_abs = 0.0;
  double ref_scale = 0.0;
  for (size_t i = 0; i < cpu.size(); ++i) {
    ASSERT_FALSE(std::isnan(gpu[i])) << "NaN CUDA logit at " << i;
    max_abs = std::max(max_abs, std::abs(static_cast<double>(cpu[i]) - gpu[i]));
    ref_scale = std::max(ref_scale, std::abs(static_cast<double>(cpu[i])));
  }
  // Both paths are fp32 but reduce in a different order, so compare relative to
  // the logit magnitude rather than demanding equality.
  EXPECT_LT(max_abs / std::max(ref_scale, 1e-6), 2e-3)
      << "max|cpu-cuda| = " << max_abs << ", |logit|max = " << ref_scale;
}

}  // namespace
#endif  // QWEN35_SUPPORT
