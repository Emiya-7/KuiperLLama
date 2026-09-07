#ifdef QWEN35_SUPPORT
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>
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

class TestableQwen35Model : public model::Qwen35Model {
 public:
  using Qwen35Model::Qwen35Model;

  int32_t tokenizer_vocab_size_for_test() const {
    CHECK(encode_layer_ != nullptr);
    return encode_layer_->vocab_size();
  }

  int32_t sample_for_test() const {
    tensor::Tensor unused_pos(base::DataType::kDataTypeInt32, 1);
    return post_processing(unused_pos, false);
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

TEST(Qwen35Config, ReportsItsOwnModelType) {
  model::Qwen35Model model(base::TokenizerType::kEncodeBpe, "unused-tokenizer", "unused-model",
                           false);
  EXPECT_EQ(model.model_type(), base::ModelType::kModelTypeQwen35);
}

TEST(Qwen35Config, InvalidIntervalDoesNotDivideByZeroDuringDerivation) {
  model::Qwen35Config c;
  c.layer_num = 8;
  c.full_attention_interval = 0;
  c.derive();
  EXPECT_EQ(c.full_layer_num, 0);
  EXPECT_EQ(c.linear_layer_num, 8);
}

TEST(Qwen35Config, RejectsZeroIntervalInModelHeader) {
  char path[] = "/tmp/kuiper_qwen35_invalid_XXXXXX";
  const int fd = mkstemp(path);
  ASSERT_NE(fd, -1);

  model::Qwen35RawConfig raw{};
  raw.magic = model::kQwen35Magic;
  raw.version = model::kQwen35Version;
  raw.hidden_size = 8;
  raw.intermediate_size = 16;
  raw.layer_num = 1;
  raw.vocab_size = 32;
  raw.max_seq_len = 8;
  raw.head_num = 1;
  raw.kv_head_num = 1;
  raw.head_dim = 8;
  raw.rotary_dim = 2;
  raw.full_attention_interval = 0;  // used to divide before being validated
  raw.linear_num_k_heads = 1;
  raw.linear_num_v_heads = 1;
  raw.linear_k_head_dim = 4;
  raw.linear_v_head_dim = 4;
  raw.conv_kernel_size = 4;
  raw.rope_theta = 10000.f;
  raw.rms_norm_eps = 1e-6f;

  const ssize_t written = write(fd, &raw, sizeof(raw));
  close(fd);
  ASSERT_EQ(written, static_cast<ssize_t>(sizeof(raw)));

  model::Qwen35Model qwen35(base::TokenizerType::kEncodeBpe, "unused-tokenizer", path, false);
  const base::Status status = qwen35.init(base::DeviceType::kDeviceCPU);
  EXPECT_FALSE(status);
  EXPECT_EQ(status.get_err_code(), base::StatusCode::kModelParseError);
  EXPECT_NE(status.get_err_msg().find("full_attention_interval"), std::string::npos);
  EXPECT_EQ(std::remove(path), 0);
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

// Qwen3.5 pads the embedding/logit matrix beyond the tokenizer's valid IDs.
// A padding row may have the largest logit but must never be returned as a
// generated token; valid special tokens at the end of the tokenizer stay in
// the sampling range.
TEST_F(Qwen35Tiny, SamplingSkipsEmbeddingPadding) {
  TestableQwen35Model model(base::TokenizerType::kEncodeBpe, tiny_token_path(),
                            tiny_model_path(), false);
  ASSERT_TRUE(model.init(base::DeviceType::kDeviceCPU));

  const int32_t token_limit = model.tokenizer_vocab_size_for_test();
  auto& logits = model.get_buffer(model::ModelBufferType::kForwardOutput);
  ASSERT_GT(token_limit, 0);
  EXPECT_EQ(token_limit, 248070);  // 248044 base tokens + 26 valid special tokens
  ASSERT_LT(static_cast<size_t>(token_limit), logits.size());

  for (size_t i = 0; i < logits.size(); ++i) {
    logits.index<float>(static_cast<int32_t>(i)) = -10.f;
  }
  logits.index<float>(token_limit - 1) = 1.f;
  logits.index<float>(token_limit) = 100.f;  // first padded embedding row

  EXPECT_EQ(model.sample_for_test(), token_limit - 1);
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
