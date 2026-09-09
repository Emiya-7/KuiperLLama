#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "base/alloc.h"
#include "base/bfloat16.h"
#include "base/nvtx.h"
#include "model/qwen35.h"
#include "op/matmul.h"

#ifndef KUIPER_BUILD_TYPE
#define KUIPER_BUILD_TYPE "unknown"
#endif

#ifndef KUIPER_CUDA_ARCHITECTURES
#define KUIPER_CUDA_ARCHITECTURES "unknown"
#endif

namespace {

struct Options {
  std::string mode = "matmul";
  std::string device = "cuda";
  std::string dtype = "bf16";
  std::string checkpoint;
  std::string tokenizer;
  std::string output;
  std::string cache = "cold";
  int32_t input_size = 2560;
  int32_t output_size = 4096;
  int32_t prompt_length = 12;
  int32_t decode_steps = 16;
  int32_t warmup = 1;
  int32_t repeat = 5;
};

struct Stats {
  std::vector<double> samples_ms;
  double minimum_ms = 0.0;
  double median_ms = 0.0;
  double p95_ms = 0.0;
  double maximum_ms = 0.0;
  double mean_ms = 0.0;
};

struct RuntimeInfo {
  std::string gpu = "cpu";
  int driver_version = 0;
  int runtime_version = 0;
  size_t total_memory_bytes = 0;
  int compute_major = 0;
  int compute_minor = 0;
};

std::string current_git_commit() {
  namespace fs = std::filesystem;
  fs::path directory = fs::current_path();
  for (;;) {
    const fs::path git = directory / ".git";
    std::ifstream head(git / "HEAD");
    if (head) {
      std::string value;
      std::getline(head, value);
      constexpr const char* prefix = "ref: ";
      if (value.rfind(prefix, 0) == 0) {
        std::ifstream ref(git / value.substr(std::char_traits<char>::length(prefix)));
        if (!ref) return "unknown";
        std::getline(ref, value);
      }
      return value.empty() ? "unknown" : value;
    }
    if (!directory.has_parent_path() || directory.parent_path() == directory) break;
    directory = directory.parent_path();
  }
  return "unknown";
}

[[noreturn]] void usage(const char* program, const std::string& error = {}) {
  if (!error.empty()) {
    std::cerr << "error: " << error << "\n\n";
  }
  std::cerr
      << "Usage:\n"
      << "  " << program
      << " --mode matmul [--m 2560] [--k 4096] [--dtype bf16|fp32]\n"
      << "  " << program
      << " --mode prefill|decode|end-to-end --checkpoint MODEL --tokenizer TOKENIZER\n\n"
      << "Common options:\n"
      << "  --device cuda|cpu       Execution device (default: cuda)\n"
      << "  --warmup N              Untimed workload repetitions (default: 1)\n"
      << "  --repeat N              Timed workload repetitions (default: 5)\n"
      << "  --output FILE           Also write the JSON result to FILE\n"
      << "  --cache cold|warm       Matmul cache state (default: cold)\n"
      << "Model options:\n"
      << "  --prompt-length N       Deterministic synthetic prompt length (default: 12)\n"
      << "  --decode-steps N        Deterministic decode steps (default: 16)\n"
      << "Matmul notation: input[M] * weight[K,M] -> output[K].\n";
  std::exit(error.empty() ? EXIT_SUCCESS : EXIT_FAILURE);
}

int32_t parse_positive(const std::string& value, const std::string& name, bool allow_zero = false) {
  size_t parsed = 0;
  long result = 0;
  try {
    result = std::stol(value, &parsed);
  } catch (const std::exception&) {
    throw std::runtime_error(name + " must be an integer");
  }
  if (parsed != value.size() || result < (allow_zero ? 0 : 1) || result > INT32_MAX) {
    throw std::runtime_error(name + " is outside the supported range");
  }
  return static_cast<int32_t>(result);
}

Options parse_options(int argc, char** argv) {
  Options options;
  auto value_after = [&](int& index, const std::string& name) -> std::string {
    if (++index >= argc) {
      usage(argv[0], "missing value after " + name);
    }
    return argv[index];
  };

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
    } else if (arg == "--mode") {
      options.mode = value_after(i, arg);
    } else if (arg == "--device") {
      options.device = value_after(i, arg);
    } else if (arg == "--dtype") {
      options.dtype = value_after(i, arg);
    } else if (arg == "--checkpoint") {
      options.checkpoint = value_after(i, arg);
    } else if (arg == "--tokenizer") {
      options.tokenizer = value_after(i, arg);
    } else if (arg == "--output") {
      options.output = value_after(i, arg);
    } else if (arg == "--cache") {
      options.cache = value_after(i, arg);
    } else if (arg == "--m") {
      options.input_size = parse_positive(value_after(i, arg), arg);
    } else if (arg == "--k") {
      options.output_size = parse_positive(value_after(i, arg), arg);
    } else if (arg == "--prompt-length") {
      options.prompt_length = parse_positive(value_after(i, arg), arg);
    } else if (arg == "--decode-steps") {
      options.decode_steps = parse_positive(value_after(i, arg), arg, true);
    } else if (arg == "--warmup") {
      options.warmup = parse_positive(value_after(i, arg), arg, true);
    } else if (arg == "--repeat") {
      options.repeat = parse_positive(value_after(i, arg), arg);
    } else {
      usage(argv[0], "unknown option " + arg);
    }
  }

  if (options.mode != "matmul" && options.mode != "prefill" && options.mode != "decode" &&
      options.mode != "end-to-end") {
    throw std::runtime_error("--mode must be matmul, prefill, decode, or end-to-end");
  }
  if (options.device != "cuda" && options.device != "cpu") {
    throw std::runtime_error("--device must be cuda or cpu");
  }
  if (options.dtype != "bf16" && options.dtype != "fp32") {
    throw std::runtime_error("--dtype must be bf16 or fp32");
  }
  if (options.cache != "cold" && options.cache != "warm") {
    throw std::runtime_error("--cache must be cold or warm");
  }
  if (options.mode != "matmul" && (options.checkpoint.empty() || options.tokenizer.empty())) {
    throw std::runtime_error("model modes require --checkpoint and --tokenizer");
  }
  return options;
}

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

RuntimeInfo runtime_info(base::DeviceType device) {
  RuntimeInfo info;
  if (device == base::DeviceType::kDeviceCPU) {
    return info;
  }
  cudaDeviceProp property{};
  check_cuda(cudaGetDeviceProperties(&property, 0), "cudaGetDeviceProperties");
  info.gpu = property.name;
  info.total_memory_bytes = property.totalGlobalMem;
  info.compute_major = property.major;
  info.compute_minor = property.minor;
  check_cuda(cudaDriverGetVersion(&info.driver_version), "cudaDriverGetVersion");
  check_cuda(cudaRuntimeGetVersion(&info.runtime_version), "cudaRuntimeGetVersion");
  return info;
}

Stats summarize(std::vector<double> samples) {
  if (samples.empty()) {
    throw std::runtime_error("cannot summarize an empty sample set");
  }
  Stats stats;
  stats.samples_ms = samples;
  std::sort(samples.begin(), samples.end());
  stats.minimum_ms = samples.front();
  stats.maximum_ms = samples.back();
  const size_t middle = samples.size() / 2;
  stats.median_ms = samples.size() % 2 == 0 ? (samples[middle - 1] + samples[middle]) / 2.0
                                           : samples[middle];
  const size_t p95_index = static_cast<size_t>(std::ceil(samples.size() * 0.95)) - 1;
  stats.p95_ms = samples[std::min(p95_index, samples.size() - 1)];
  stats.mean_ms = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  return stats;
}

double measure_cuda_ms(cudaStream_t stream, const std::function<void()>& workload) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
  try {
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    check_cuda(cudaEventRecord(start, stream), "cudaEventRecord(start)");
    workload();
    check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord(stop)");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
    float elapsed = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime");
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return elapsed;
  } catch (...) {
    if (stop) cudaEventDestroy(stop);
    cudaEventDestroy(start);
    throw;
  }
}

double measure_cpu_ms(const std::function<void()>& workload) {
  const auto start = std::chrono::steady_clock::now();
  workload();
  const auto stop = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(stop - start).count();
}

double measure_ms(base::DeviceType device, cudaStream_t stream,
                  const std::function<void()>& workload) {
  return device == base::DeviceType::kDeviceCUDA ? measure_cuda_ms(stream, workload)
                                                 : measure_cpu_ms(workload);
}

std::vector<int32_t> deterministic_tokens(int32_t count, int32_t offset = 0) {
  std::vector<int32_t> tokens;
  tokens.reserve(count);
  for (int32_t i = 0; i < count; ++i) {
    // Stays well inside every supported Qwen3.5 vocabulary, including the tiny fixture.
    tokens.push_back(100 + ((offset + i) * 131) % 1024);
  }
  return tokens;
}

void run_tokens(const model::Qwen35Model& model, const std::vector<int32_t>& tokens,
                int32_t position_offset, bool prompt_embeddings) {
  if (prompt_embeddings && position_offset != 0) {
    throw std::runtime_error("prompt embedding batches must start at position zero");
  }
  const auto embeddings = model.embedding(tokens);
  tensor::Tensor pos = model.get_buffer(model::ModelBufferType::kInputPos);
  int next = -1;
  for (int32_t i = 0; i < static_cast<int32_t>(tokens.size()); ++i) {
    pos.index<int32_t>(0) = position_offset + i;
    // fill_input uses the absolute position to index a prompt embedding batch,
    // while a decode embedding always contains one token at index zero.
    tensor::Tensor input = model.fill_input(pos, embeddings, prompt_embeddings);
    const base::Status status = model.forward(input, pos, next);
    if (!status) {
      throw std::runtime_error("Qwen3.5 forward failed: " + status.get_err_msg());
    }
  }
}

struct ModelResults {
  double load_ms = 0.0;
  Stats prefill;
  Stats decode;
  Stats total;
  bool has_prefill = false;
  bool has_decode = false;
  bool has_total = false;
};

ModelResults benchmark_model(const Options& options, base::DeviceType device) {
  const auto load_start = std::chrono::steady_clock::now();
  model::Qwen35Model model(base::TokenizerType::kEncodeBpe, options.tokenizer, options.checkpoint,
                           false);
  const base::Status init_status = model.init(device);
  if (!init_status) {
    throw std::runtime_error("model init failed: " + init_status.get_err_msg());
  }
  if (device == base::DeviceType::kDeviceCUDA) {
    check_cuda(cudaStreamSynchronize(model.cuda_stream()), "synchronize after model init");
  }
  ModelResults results;
  results.load_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - load_start)
                        .count();

  const auto prompt = deterministic_tokens(options.prompt_length);
  const auto decode = deterministic_tokens(options.decode_steps, options.prompt_length);

  auto prepare = [&] {
    model.reset_state();
    if (device == base::DeviceType::kDeviceCUDA) {
      check_cuda(cudaStreamSynchronize(model.cuda_stream()), "synchronize after state reset");
    }
  };
  auto prefill_work = [&] {
    base::ScopedNvtxRange range("prefill");
    run_tokens(model, prompt, 0, true);
  };
  auto decode_work = [&] {
    base::ScopedNvtxRange range("decode");
    for (int32_t i = 0; i < options.decode_steps; ++i) {
      run_tokens(model, {decode.at(i)}, options.prompt_length + i, false);
    }
  };

  auto execute_once = [&](bool record, std::vector<double>* prefill_ms,
                          std::vector<double>* decode_ms, std::vector<double>* total_ms) {
    prepare();
    if (options.mode == "prefill") {
      const double value = measure_ms(device, model.cuda_stream(), prefill_work);
      if (record) prefill_ms->push_back(value);
      return;
    }

    if (options.mode == "decode") {
      // Establish exactly the same context and recurrent state before every timed decode range.
      prefill_work();
      if (device == base::DeviceType::kDeviceCUDA) {
        check_cuda(cudaStreamSynchronize(model.cuda_stream()), "synchronize after decode setup");
      }
      const double value = measure_ms(device, model.cuda_stream(), decode_work);
      if (record) decode_ms->push_back(value);
      return;
    }

    double prefill_value = 0.0;
    double decode_value = 0.0;
    const double total_value = measure_ms(device, model.cuda_stream(), [&] {
      base::ScopedNvtxRange range("end-to-end");
      prefill_work();
      decode_work();
    });
    // Collect phase timings in a second, identically reset run. This keeps the total range free
    // of an artificial synchronization point between prefill and decode.
    prepare();
    prefill_value = measure_ms(device, model.cuda_stream(), prefill_work);
    decode_value = measure_ms(device, model.cuda_stream(), decode_work);
    if (record) {
      prefill_ms->push_back(prefill_value);
      decode_ms->push_back(decode_value);
      total_ms->push_back(total_value);
    }
  };

  std::vector<double> prefill_ms;
  std::vector<double> decode_ms;
  std::vector<double> total_ms;
  for (int32_t i = 0; i < options.warmup; ++i) {
    execute_once(false, &prefill_ms, &decode_ms, &total_ms);
  }
  for (int32_t i = 0; i < options.repeat; ++i) {
    execute_once(true, &prefill_ms, &decode_ms, &total_ms);
  }

  if (!prefill_ms.empty()) {
    results.prefill = summarize(prefill_ms);
    results.has_prefill = true;
  }
  if (!decode_ms.empty()) {
    results.decode = summarize(decode_ms);
    results.has_decode = true;
  }
  if (!total_ms.empty()) {
    results.total = summarize(total_ms);
    results.has_total = true;
  }
  return results;
}

struct MatmulResults {
  Stats elapsed;
  double checksum = 0.0;
  double gflops = 0.0;
  double effective_gbps = 0.0;
  size_t cache_flush_bytes = 0;
};

MatmulResults benchmark_matmul(const Options& options, base::DeviceType device) {
  const auto cpu_alloc = base::CPUDeviceAllocatorFactory::get_instance();
  const base::DataType weight_type = options.dtype == "bf16" ? base::DataType::kDataTypeBf16
                                                              : base::DataType::kDataTypeFp32;
  tensor::Tensor input(base::DataType::kDataTypeFp32, options.input_size, true, cpu_alloc);
  tensor::Tensor weight(weight_type, options.output_size, options.input_size, true, cpu_alloc);
  for (int32_t i = 0; i < options.input_size; ++i) {
    input.index<float>(i) = static_cast<float>(i % 29 - 14) / 32.0f;
  }
  const int64_t weight_elements =
      static_cast<int64_t>(options.input_size) * options.output_size;
  for (int64_t i = 0; i < weight_elements; ++i) {
    const float value = static_cast<float>((i * 13) % 31 - 15) / 64.0f;
    if (weight_type == base::DataType::kDataTypeBf16) {
      weight.index<uint16_t>(i) = base::float_to_bfloat16(value);
    } else {
      weight.index<float>(i) = value;
    }
  }

  std::shared_ptr<kernel::CudaConfig> cuda_config;
  std::shared_ptr<base::DeviceAllocator> output_alloc = cpu_alloc;
  void* cache_flush_buffer = nullptr;
  size_t cache_flush_bytes = 0;
  if (device == base::DeviceType::kDeviceCUDA) {
    cuda_config = std::make_shared<kernel::CudaConfig>();
    check_cuda(cudaStreamCreate(&cuda_config->stream), "cudaStreamCreate");
    input.to_cuda(cuda_config->stream);
    weight.to_cuda(cuda_config->stream);
    output_alloc = base::CUDADeviceAllocatorFactory::get_instance();
    if (options.cache == "cold") {
      int32_t l2_bytes = 0;
      check_cuda(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, 0),
                 "query CUDA L2 cache size");
      cache_flush_bytes = std::max<size_t>(static_cast<size_t>(l2_bytes) * 2, 1);
      check_cuda(cudaMalloc(&cache_flush_buffer, cache_flush_bytes), "allocate cache flush buffer");
    }
    check_cuda(cudaStreamSynchronize(cuda_config->stream), "synchronize matmul setup");
  }
  tensor::Tensor output(base::DataType::kDataTypeFp32, options.output_size, true, output_alloc);
  op::MatmulLayer layer(device, options.output_size, options.input_size);
  layer.set_cuda_config(cuda_config);
  const base::Status weight_status = layer.set_weight(0, weight);
  if (!weight_status) {
    throw std::runtime_error("failed to bind matmul weight: " + weight_status.get_err_msg());
  }
  const std::string range_name = "matmul/" + options.dtype + "/m" +
                                 std::to_string(options.input_size) + "_k" +
                                 std::to_string(options.output_size);
  auto workload = [&] {
    base::ScopedNvtxRange range(range_name.c_str());
    // MatmulLayer::forward() hides Layer's convenience overloads, so dispatch
    // through the base type just as the model's shared_ptr<Layer> plumbing does.
    op::Layer& base_layer = layer;
    const base::Status status = base_layer.forward(input, output);
    if (!status) {
      throw std::runtime_error("matmul failed: " + status.get_err_msg());
    }
  };

  for (int32_t i = 0; i < options.warmup; ++i) {
    workload();
  }
  if (device == base::DeviceType::kDeviceCUDA) {
    check_cuda(cudaStreamSynchronize(cuda_config->stream), "synchronize matmul warmup");
  }

  std::vector<double> samples;
  samples.reserve(options.repeat);
  for (int32_t i = 0; i < options.repeat; ++i) {
    if (cache_flush_buffer) {
      // Ordered before the start event on the same stream, so cache eviction is
      // effective but excluded from the measured kernel duration.
      check_cuda(cudaMemsetAsync(cache_flush_buffer, i + 1, cache_flush_bytes,
                                 cuda_config->stream),
                 "flush CUDA L2 cache");
    }
    samples.push_back(measure_ms(device, cuda_config ? cuda_config->stream : nullptr, workload));
  }

  if (device == base::DeviceType::kDeviceCUDA) {
    output.to_cpu();
  }
  MatmulResults results;
  results.elapsed = summarize(samples);
  for (int32_t i = 0; i < options.output_size; ++i) {
    results.checksum += output.index<float>(i);
  }
  if (cache_flush_buffer) {
    check_cuda(cudaFree(cache_flush_buffer), "free cache flush buffer");
  }
  results.cache_flush_bytes = cache_flush_bytes;
  const double seconds = results.elapsed.median_ms / 1000.0;
  results.gflops = 2.0 * options.input_size * options.output_size / seconds / 1e9;
  const double bytes = static_cast<double>(options.input_size) * sizeof(float) +
                       static_cast<double>(weight_elements) * base::DataTypeSize(weight_type) +
                       static_cast<double>(options.output_size) * sizeof(float);
  results.effective_gbps = bytes / seconds / 1e9;
  return results;
}

std::string json_escape(const std::string& value) {
  std::ostringstream out;
  for (const char c : value) {
    if (c == '\\' || c == '"') out << '\\';
    if (c == '\n') {
      out << "\\n";
    } else {
      out << c;
    }
  }
  return out.str();
}

void write_stats(std::ostream& out, const Stats& stats, int indent) {
  const std::string pad(indent, ' ');
  out << "{\n"
      << pad << "  \"min_ms\": " << stats.minimum_ms << ",\n"
      << pad << "  \"median_ms\": " << stats.median_ms << ",\n"
      << pad << "  \"p95_ms\": " << stats.p95_ms << ",\n"
      << pad << "  \"max_ms\": " << stats.maximum_ms << ",\n"
      << pad << "  \"mean_ms\": " << stats.mean_ms << ",\n"
      << pad << "  \"samples_ms\": [";
  for (size_t i = 0; i < stats.samples_ms.size(); ++i) {
    if (i) out << ", ";
    out << stats.samples_ms[i];
  }
  out << "]\n" << pad << "}";
}

std::string result_json(const Options& options, const RuntimeInfo& runtime,
                        const MatmulResults* matmul, const ModelResults* model) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(6);
  out << "{\n"
      << "  \"schema_version\": 1,\n"
      << "  \"git_commit\": \"" << json_escape(current_git_commit()) << "\",\n"
      << "  \"mode\": \"" << json_escape(options.mode) << "\",\n"
      << "  \"device\": \"" << json_escape(options.device) << "\",\n"
      << "  \"build_type\": \"" << json_escape(KUIPER_BUILD_TYPE) << "\",\n"
      << "  \"cuda_architectures\": \"" << json_escape(KUIPER_CUDA_ARCHITECTURES) << "\",\n"
      << "  \"environment\": {\n"
      << "    \"gpu\": \"" << json_escape(runtime.gpu) << "\",\n"
      << "    \"compute_capability\": \"" << runtime.compute_major << "."
      << runtime.compute_minor << "\",\n"
      << "    \"driver_version\": " << runtime.driver_version << ",\n"
      << "    \"cuda_runtime_version\": " << runtime.runtime_version << ",\n"
      << "    \"gpu_memory_bytes\": " << runtime.total_memory_bytes << "\n"
      << "  },\n"
      << "  \"warmup\": " << options.warmup << ",\n"
      << "  \"repeat\": " << options.repeat << ",\n";
  if (matmul) {
    out << "  \"matmul\": {\n"
        << "    \"input_size_m\": " << options.input_size << ",\n"
        << "    \"output_size_k\": " << options.output_size << ",\n"
        << "    \"weight_dtype\": \"" << options.dtype << "\",\n"
        << "    \"cache\": \"" << options.cache << "\",\n"
        << "    \"cache_flush_bytes\": " << matmul->cache_flush_bytes << ",\n"
        << "    \"elapsed\": ";
    write_stats(out, matmul->elapsed, 4);
    out << ",\n"
        << "    \"median_gflops\": " << matmul->gflops << ",\n"
        << "    \"median_effective_gbps\": " << matmul->effective_gbps << ",\n"
        << "    \"output_checksum\": " << matmul->checksum << "\n"
        << "  }\n";
  } else {
    out << "  \"model\": {\n"
        << "    \"checkpoint\": \"" << json_escape(options.checkpoint) << "\",\n"
        << "    \"prompt_length\": " << options.prompt_length << ",\n"
        << "    \"decode_steps\": " << options.decode_steps << ",\n"
        << "    \"prefill_implementation\": \"token-by-token-baseline\",\n"
        << "    \"load_ms\": " << model->load_ms;
    if (model->has_prefill) {
      out << ",\n    \"prefill\": ";
      write_stats(out, model->prefill, 4);
      out << ",\n    \"prefill_tokens_per_second\": "
          << options.prompt_length / (model->prefill.median_ms / 1000.0);
    }
    if (model->has_decode) {
      out << ",\n    \"decode\": ";
      write_stats(out, model->decode, 4);
      out << ",\n    \"decode_ms_per_token\": "
          << model->decode.median_ms / std::max(options.decode_steps, 1)
          << ",\n    \"decode_tokens_per_second\": "
          << options.decode_steps / (model->decode.median_ms / 1000.0);
    }
    if (model->has_total) {
      out << ",\n    \"total\": ";
      write_stats(out, model->total, 4);
    }
    out << "\n  }\n";
  }
  out << "}\n";
  return out.str();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const base::DeviceType device = options.device == "cuda" ? base::DeviceType::kDeviceCUDA
                                                              : base::DeviceType::kDeviceCPU;
    const RuntimeInfo runtime = runtime_info(device);
    MatmulResults matmul;
    ModelResults model;
    const MatmulResults* matmul_ptr = nullptr;
    const ModelResults* model_ptr = nullptr;
    if (options.mode == "matmul") {
      matmul = benchmark_matmul(options, device);
      matmul_ptr = &matmul;
    } else {
      model = benchmark_model(options, device);
      model_ptr = &model;
    }
    const std::string json = result_json(options, runtime, matmul_ptr, model_ptr);
    std::cout << json;
    if (!options.output.empty()) {
      std::ofstream file(options.output);
      if (!file) {
        throw std::runtime_error("failed to open output file " + options.output);
      }
      file << json;
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "qwen35_bench: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
