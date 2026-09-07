#include <base/base.h>
#include <base/tick.h>
#include <glog/logging.h>
#include <chrono>
#include <iostream>
#include "model/qwen35.h"

// Qwen3.5 chat special tokens, read from the shipped tokenizer.json rather than
// carried over from Qwen2/Qwen3, whose vocabulary is a different size.
static constexpr int32_t kImStart = 248045;
static constexpr int32_t kImEnd = 248046;

int32_t generate(const model::Qwen35Model& model, const std::string& sentence, int total_steps,
                 bool need_output = false) {
  auto tokens = model.encode(sentence);
  const int32_t prompt_len = static_cast<int32_t>(tokens.size());
  LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";

  // The Gated DeltaNet layers carry a recurrent state that accumulates across
  // positions, so it has to start from zero for each new sequence.
  model.reset_state();

  int32_t pos = 0;
  int32_t next = tokens.at(pos);
  bool is_prompt = true;
  const auto& prompt_embedding = model.embedding(tokens);
  tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);

  std::vector<int32_t> words;
  while (pos < total_steps) {
    pos_tensor.index<int32_t>(0) = pos;
    if (pos < prompt_len - 1) {
      tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
    } else {
      is_prompt = false;
      tokens = std::vector<int32_t>{next};
      const auto& token_embedding = model.embedding(tokens);
      tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
      if (next != kImEnd && next != kImStart) {
        words.push_back(next);
      }
    }
    if (model.is_sentence_ending(next)) {
      break;
    }
    if (is_prompt) {
      next = tokens.at(pos + 1);
    }
    pos += 1;
  }
  if (need_output) {
    printf("%s ", model.decode(words).data());
    fflush(stdout);
  }
  return std::min(pos, total_steps);
}

static std::string fill_template(const std::string& content) {
  return "<|im_start|>user\n" + content + "<|im_end|>\n<|im_start|>assistant\n";
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    LOG(INFO) << "Usage: ./qwen35_infer <checkpoint.bin> <tokenizer path> [--cpu] [steps]";
    return -1;
  }
  const char* checkpoint_path = argv[1];
  const char* tokenizer_path = argv[2];

  auto device = base::DeviceType::kDeviceCUDA;
  int steps_limit = 256;
  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--cpu") {
      device = base::DeviceType::kDeviceCPU;
    } else {
      steps_limit = std::atoi(argv[i]);
    }
  }

  model::Qwen35Model model(base::TokenizerType::kEncodeBpe, tokenizer_path, checkpoint_path, false);
  auto init_status = model.init(device);
  if (!init_status) {
    LOG(FATAL) << "Model init failed, error code: " << init_status.get_err_code() << " "
               << init_status.get_err_msg();
  }

  const std::string prompt = "What is AI?";
  std::cout << prompt << "\n";
  fflush(stdout);

  const auto start = std::chrono::steady_clock::now();
  const int steps = generate(model, fill_template(prompt), steps_limit, true);
  const auto end = std::chrono::steady_clock::now();
  const double duration = std::chrono::duration<double>(end - start).count();

  printf("\nsteps: %d\n", steps);
  printf("duration: %.3f s\n", duration);
  printf("steps/s: %.2f\n", static_cast<double>(steps) / duration);
  fflush(stdout);
  return 0;
}
