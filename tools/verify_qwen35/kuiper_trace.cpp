#include <base/base.h>
#include <glog/logging.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "model/qwen35.h"

namespace {

const char* kDefaultPrompt =
    "<|im_start|>user\nWhat is AI?<|im_end|>\n<|im_start|>assistant\n";
constexpr int32_t kGenerationLength = 10;

template <typename T>
void write_raw(const std::filesystem::path& path, const T* data, size_t count) {
  std::ofstream out(path, std::ios::binary);
  CHECK(out.is_open()) << "Failed to open trace output " << path;
  out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count * sizeof(T)));
  CHECK(out.good()) << "Failed to write trace output " << path;
}

std::string hidden_name(int32_t layer_idx) {
  char name[32];
  snprintf(name, sizeof(name), "hidden_%03d.f32", layer_idx);
  return name;
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc < 4 || argc > 5) {
    std::cerr << "Usage: qwen35_trace <checkpoint.bin> <tokenizer.json> <output_dir> [prompt]\n";
    return 2;
  }

  const std::string checkpoint = argv[1];
  const std::string tokenizer = argv[2];
  const std::filesystem::path output_dir = argv[3];
  const std::string prompt = argc == 5 ? argv[4] : kDefaultPrompt;
  std::filesystem::create_directories(output_dir);

  model::Qwen35Model qwen35(base::TokenizerType::kEncodeBpe, tokenizer, checkpoint, false);
  const base::Status init_status = qwen35.init(base::DeviceType::kDeviceCPU);
  if (!init_status) {
    std::cerr << "Model init failed: " << init_status.get_err_msg() << "\n";
    return 1;
  }

  const std::vector<int32_t> tokens = qwen35.encode(prompt);
  if (tokens.empty()) {
    std::cerr << "Prompt encoded to no tokens\n";
    return 1;
  }

  const auto& config = qwen35.qwen35_config();
  const int32_t seq_len = static_cast<int32_t>(tokens.size());
  const int32_t trace_layers = config.layer_num + 1;
  std::vector<std::vector<float>> hidden(
      trace_layers, std::vector<float>(static_cast<size_t>(seq_len) * config.hidden_size));
  std::vector<float> logits(static_cast<size_t>(seq_len) * config.vocab_size);

  qwen35.set_hidden_state_callback(
      [&](int32_t position, int32_t layer_idx, const tensor::Tensor& state) {
        CHECK_GE(position, 0);
        CHECK_LT(position, seq_len);
        CHECK_GE(layer_idx, 0);
        CHECK_LT(layer_idx, trace_layers);
        CHECK(state.device_type() == base::DeviceType::kDeviceCPU);
        CHECK_EQ(state.size(), static_cast<size_t>(config.hidden_size));
        float* destination = hidden[layer_idx].data() +
                             static_cast<size_t>(position) * config.hidden_size;
        std::copy_n(state.ptr<float>(), config.hidden_size, destination);
      });

  qwen35.reset_state();
  const op::EmbeddingOutput embeddings =
      qwen35.embedding(std::vector<int>(tokens.begin(), tokens.end()));
  tensor::Tensor pos_tensor = qwen35.get_buffer(model::ModelBufferType::kInputPos);
  int next = -1;
  for (int32_t pos = 0; pos < seq_len; ++pos) {
    pos_tensor.index<int32_t>(0) = pos;
    tensor::Tensor input = qwen35.fill_input(pos_tensor, embeddings, true);
    const base::Status status = qwen35.predict(input, pos_tensor, pos + 1 < seq_len, next);
    if (!status) {
      std::cerr << "Forward failed at position " << pos << ": " << status.get_err_msg() << "\n";
      return 1;
    }
    const auto& row = qwen35.get_buffer(model::ModelBufferType::kForwardOutput);
    std::copy_n(row.ptr<float>(), config.vocab_size,
                logits.data() + static_cast<size_t>(pos) * config.vocab_size);
  }

  std::vector<int32_t> generated;
  generated.reserve(kGenerationLength);
  generated.push_back(next);
  qwen35.set_hidden_state_callback({});
  for (int32_t index = 1; index < kGenerationLength; ++index) {
    const int32_t pos = seq_len + index - 1;
    pos_tensor.index<int32_t>(0) = pos;
    const op::EmbeddingOutput token_embedding = qwen35.embedding({next});
    tensor::Tensor input = qwen35.fill_input(pos_tensor, token_embedding, false);
    const base::Status status = qwen35.predict(input, pos_tensor, false, next);
    if (!status) {
      std::cerr << "Generation failed at position " << pos << ": " << status.get_err_msg()
                << "\n";
      return 1;
    }
    generated.push_back(next);
  }

  write_raw(output_dir / "tokens.i32", tokens.data(), tokens.size());
  write_raw(output_dir / "generated.i32", generated.data(), generated.size());
  write_raw(output_dir / "logits.f32", logits.data(), logits.size());
  for (int32_t layer = 0; layer < trace_layers; ++layer) {
    write_raw(output_dir / hidden_name(layer), hidden[layer].data(), hidden[layer].size());
  }

  std::ofstream metadata(output_dir / "metadata.json");
  metadata << "{\n"
           << "  \"implementation\": \"kuiper\",\n"
           << "  \"sequence_length\": " << seq_len << ",\n"
           << "  \"hidden_size\": " << config.hidden_size << ",\n"
           << "  \"num_hidden_layers\": " << config.layer_num << ",\n"
           << "  \"vocab_size\": " << config.vocab_size << ",\n"
           << "  \"generation_length\": " << generated.size() << ",\n"
           << "  \"next_token\": " << generated.front() << "\n"
           << "}\n";
  CHECK(metadata.good()) << "Failed to write metadata";

  std::cout << "tokens=" << seq_len << " layers=" << config.layer_num
            << " generated=" << generated.size() << " first_token=" << generated.front()
            << " trace_dir=" << output_dir << "\n";
  return 0;
}
