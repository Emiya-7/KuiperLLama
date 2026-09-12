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
#include "op/qwen35_ops.h"

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
  std::string matmul_implementation = "auto";
  int32_t batch_size = 1;
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
      << " --mode matmul [--n 1] [--m 2560] [--k 4096] [--dtype bf16|fp32]\n"
      << "  " << program
      << " --mode gdn [--device cuda|cpu]\n"
      << "  " << program
      << " --mode prefill|decode|end-to-end --checkpoint MODEL --tokenizer TOKENIZER\n\n"
      << "Common options:\n"
      << "  --device cuda|cpu       Execution device (default: cuda)\n"
      << "  --warmup N              Untimed workload repetitions (default: 1)\n"
      << "  --repeat N              Timed workload repetitions (default: 5)\n"
      << "  --output FILE           Also write the JSON result to FILE\n"
      << "  --cache cold|warm       Matmul cache state (default: cold)\n"
      << "  --matmul-implementation auto|gemm|gemv-loop (default: auto)\n"
      << "Model options:\n"
      << "  --prompt-length N       Deterministic synthetic prompt length (default: 12)\n"
      << "  --decode-steps N        Deterministic decode steps (default: 16)\n"
      << "Matmul notation: input[N,M] * weight[K,M]^T -> output[N,K].\n";
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
    } else if (arg == "--matmul-implementation") {
      options.matmul_implementation = value_after(i, arg);
    } else if (arg == "--n") {
      options.batch_size = parse_positive(value_after(i, arg), arg);
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

  if (options.mode != "matmul" && options.mode != "gdn" && options.mode != "prefill" &&
      options.mode != "decode" && options.mode != "end-to-end") {
    throw std::runtime_error("--mode must be matmul, gdn, prefill, decode, or end-to-end");
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
  if (options.matmul_implementation != "auto" && options.matmul_implementation != "gemm" &&
      options.matmul_implementation != "gemv-loop") {
    throw std::runtime_error("--matmul-implementation must be auto, gemm, or gemv-loop");
  }
  const bool requests_gemm = options.matmul_implementation == "gemm" ||
                             (options.matmul_implementation == "auto" &&
                              options.batch_size > 1);
  if (options.mode == "matmul" && requests_gemm && options.dtype != "bf16") {
    throw std::runtime_error("the batched GEMM kernel currently requires --dtype bf16");
  }
  if (options.matmul_implementation != "auto" && options.matmul_implementation != "gemm" &&
      options.matmul_implementation != "gemv-loop") {
    throw std::runtime_error("--matmul-implementation must be auto, gemm, or gemv-loop");
  }
  if (options.mode == "matmul" && options.batch_size > 1 && options.dtype != "bf16") {
    throw std::runtime_error("batched matmul currently requires --dtype bf16");
  }
  if (options.mode == "matmul" && options.matmul_implementation == "gemm" &&
      options.batch_size == 1) {
    throw std::runtime_error("--matmul-implementation gemm requires --n greater than 1");
  }
  if (options.mode != "matmul" && options.mode != "gdn" &&
      (options.checkpoint.empty() || options.tokenizer.empty())) {
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
  std::string implementation;
  double checksum = 0.0;
  double gflops = 0.0;
  double effective_gbps = 0.0;
  double estimated_memory_gbps = 0.0;
  size_t logical_bytes = 0;
  size_t estimated_memory_bytes = 0;
  size_t cache_flush_bytes = 0;
};

struct GdnResults {
  Stats elapsed;
  double output_checksum = 0.0;
  double state_checksum = 0.0;
  double max_abs_error = 0.0;
  size_t state_bytes = 0;
};

GdnResults benchmark_gdn(const Options& options, base::DeviceType device) {
  // Qwen3.5-4B GDN dimensions from config.json. Keeping this fixture fixed makes
  // before/after NCU reports directly comparable and avoids loading an 8.4 GiB checkpoint.
  constexpr int32_t kNumKHeads = 16;
  constexpr int32_t kNumVHeads = 32;
  constexpr int32_t kHeadDim = 128;
  constexpr int32_t kValueDim = 128;
  constexpr int32_t kQKSize = kNumKHeads * kHeadDim;
  constexpr int32_t kValueSize = kNumVHeads * kValueDim;
  constexpr int32_t kStateSize = kNumVHeads * kHeadDim * kValueDim;

  const auto cpu_alloc = base::CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor q(base::DataType::kDataTypeFp32, kQKSize, true, cpu_alloc);
  tensor::Tensor k(base::DataType::kDataTypeFp32, kQKSize, true, cpu_alloc);
  tensor::Tensor v(base::DataType::kDataTypeFp32, kValueSize, true, cpu_alloc);
  tensor::Tensor g(base::DataType::kDataTypeFp32, kNumVHeads, true, cpu_alloc);
  tensor::Tensor beta(base::DataType::kDataTypeFp32, kNumVHeads, true, cpu_alloc);
  tensor::Tensor state(base::DataType::kDataTypeFp32, kStateSize, true, cpu_alloc);

  const float unit = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
  for (int32_t i = 0; i < kQKSize; ++i) {
    q.index<float>(i) = unit;
    k.index<float>(i) = unit;
  }
  for (int32_t i = 0; i < kValueSize; ++i) {
    v.index<float>(i) = static_cast<float>(i % 37 - 18) / 64.0f;
  }
  for (int32_t i = 0; i < kNumVHeads; ++i) {
    g.index<float>(i) = -0.125f - static_cast<float>(i % 3) / 32.0f;
    beta.index<float>(i) = 0.5f + static_cast<float>(i % 5) / 32.0f;
  }
  std::fill(state.ptr<float>(), state.ptr<float>() + state.size(), 0.0f);

  std::shared_ptr<kernel::CudaConfig> cuda_config;
  std::shared_ptr<base::DeviceAllocator> output_alloc = cpu_alloc;
  if (device == base::DeviceType::kDeviceCUDA) {
    cuda_config = std::make_shared<kernel::CudaConfig>();
    check_cuda(cudaStreamCreate(&cuda_config->stream), "cudaStreamCreate");
    q.to_cuda(cuda_config->stream);
    k.to_cuda(cuda_config->stream);
    v.to_cuda(cuda_config->stream);
    g.to_cuda(cuda_config->stream);
    beta.to_cuda(cuda_config->stream);
    state.to_cuda(cuda_config->stream);
    output_alloc = base::CUDADeviceAllocatorFactory::get_instance();
  }
  tensor::Tensor output(base::DataType::kDataTypeFp32, kValueSize, true, output_alloc);

  op::GatedDeltaLayer layer(device, kNumKHeads, kNumVHeads, kHeadDim, kValueDim);
  layer.set_cuda_config(cuda_config);
  layer.set_input(0, q);
  layer.set_input(1, k);
  layer.set_input(2, v);
  layer.set_input(3, g);
  layer.set_input(4, beta);
  layer.set_input(5, state);
  layer.set_output(0, output);

  auto reset_state = [&] {
    if (device == base::DeviceType::kDeviceCUDA) {
      check_cuda(cudaMemsetAsync(state.ptr<float>(), 0, state.byte_size(), cuda_config->stream),
                 "reset GDN state");
    } else {
      std::fill(state.ptr<float>(), state.ptr<float>() + state.size(), 0.0f);
    }
  };
  auto workload = [&] {
    base::ScopedNvtxRange range("gdn/step/nk16_nv32_k128_v128");
    const base::Status status = layer.forward();
    if (!status) {
      throw std::runtime_error("GDN failed: " + status.get_err_msg());
    }
  };

  for (int32_t i = 0; i < options.warmup; ++i) {
    reset_state();
    workload();
  }
  if (device == base::DeviceType::kDeviceCUDA) {
    check_cuda(cudaStreamSynchronize(cuda_config->stream), "synchronize GDN warmup");
  }

  std::vector<double> samples;
  samples.reserve(options.repeat);
  for (int32_t i = 0; i < options.repeat; ++i) {
    reset_state();
    samples.push_back(measure_ms(device, device == base::DeviceType::kDeviceCUDA
                                             ? cuda_config->stream
                                             : nullptr,
                                 workload));
  }
  if (device == base::DeviceType::kDeviceCUDA) {
    output.to_cpu();
    state.to_cpu();
  }

  GdnResults results;
  results.elapsed = summarize(samples);
  results.state_bytes = static_cast<size_t>(kStateSize) * sizeof(float);
  for (int32_t i = 0; i < kValueSize; ++i) results.output_checksum += output.index<float>(i);
  for (int32_t i = 0; i < kStateSize; ++i) results.state_checksum += state.index<float>(i);

  if (device == base::DeviceType::kDeviceCUDA) {
    tensor::Tensor q_ref(base::DataType::kDataTypeFp32, kQKSize, true, cpu_alloc);
    tensor::Tensor k_ref(base::DataType::kDataTypeFp32, kQKSize, true, cpu_alloc);
    tensor::Tensor v_ref(base::DataType::kDataTypeFp32, kValueSize, true, cpu_alloc);
    tensor::Tensor g_ref(base::DataType::kDataTypeFp32, kNumVHeads, true, cpu_alloc);
    tensor::Tensor beta_ref(base::DataType::kDataTypeFp32, kNumVHeads, true, cpu_alloc);
    tensor::Tensor state_ref(base::DataType::kDataTypeFp32, kStateSize, true, cpu_alloc);
    tensor::Tensor output_ref(base::DataType::kDataTypeFp32, kValueSize, true, cpu_alloc);
    for (int32_t i = 0; i < kQKSize; ++i) {
      q_ref.index<float>(i) = unit;
      k_ref.index<float>(i) = unit;
    }
    for (int32_t i = 0; i < kValueSize; ++i) {
      v_ref.index<float>(i) = static_cast<float>(i % 37 - 18) / 64.0f;
    }
    for (int32_t i = 0; i < kNumVHeads; ++i) {
      g_ref.index<float>(i) = -0.125f - static_cast<float>(i % 3) / 32.0f;
      beta_ref.index<float>(i) = 0.5f + static_cast<float>(i % 5) / 32.0f;
    }
    std::fill(state_ref.ptr<float>(), state_ref.ptr<float>() + state_ref.size(), 0.0f);
    op::GatedDeltaLayer reference(base::DeviceType::kDeviceCPU, kNumKHeads, kNumVHeads, kHeadDim,
                                  kValueDim);
    reference.set_input(0, q_ref);
    reference.set_input(1, k_ref);
    reference.set_input(2, v_ref);
    reference.set_input(3, g_ref);
    reference.set_input(4, beta_ref);
    reference.set_input(5, state_ref);
    reference.set_output(0, output_ref);
    const base::Status status = reference.forward();
    if (!status) throw std::runtime_error("CPU GDN reference failed: " + status.get_err_msg());
    for (int32_t i = 0; i < kValueSize; ++i) {
      results.max_abs_error =
          std::max(results.max_abs_error,
                   std::abs(static_cast<double>(output.index<float>(i) -
                                                output_ref.index<float>(i))));
    }
    for (int32_t i = 0; i < kStateSize; ++i) {
      results.max_abs_error =
          std::max(results.max_abs_error,
                   std::abs(static_cast<double>(state.index<float>(i) -
                                                state_ref.index<float>(i))));
    }
  }
  return results;
}

MatmulResults benchmark_matmul(const Options& options, base::DeviceType device) {
  const auto cpu_alloc = base::CPUDeviceAllocatorFactory::get_instance();
  const base::DataType weight_type = options.dtype == "bf16" ? base::DataType::kDataTypeBf16
                                                              : base::DataType::kDataTypeFp32;
  const std::string implementation =
      options.matmul_implementation == "auto"
          ? (options.batch_size == 1 ? "gemv" : "gemm")
          : options.matmul_implementation;
  const bool use_batched_tensor = implementation == "gemm" || options.batch_size > 1;
  const std::vector<int32_t> input_dims =
      use_batched_tensor ? std::vector<int32_t>{options.batch_size, options.input_size}
                         : std::vector<int32_t>{options.input_size};
  tensor::Tensor input(base::DataType::kDataTypeFp32, input_dims, true, cpu_alloc);
  tensor::Tensor weight(weight_type, options.output_size, options.input_size, true, cpu_alloc);
  const int64_t input_elements = static_cast<int64_t>(options.batch_size) * options.input_size;
  for (int64_t i = 0; i < input_elements; ++i) {
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
  size_t cache_scratch_bytes = 0;
  if (device == base::DeviceType::kDeviceCUDA) {
    cuda_config = std::make_shared<kernel::CudaConfig>();
    check_cuda(cudaStreamCreate(&cuda_config->stream), "cudaStreamCreate");
    input.to_cuda(cuda_config->stream);
    weight.to_cuda(cuda_config->stream);
    output_alloc = base::CUDADeviceAllocatorFactory::get_instance();
    int32_t l2_bytes = 0;
    check_cuda(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, 0),
               "query CUDA L2 cache size");
    cache_scratch_bytes = std::max<size_t>(static_cast<size_t>(l2_bytes) * 2, 1);
    check_cuda(cudaMalloc(&cache_flush_buffer, cache_scratch_bytes),
               "allocate cache flush buffer");
    // Give both cache modes the same memory-heavy clock warmup. Without this,
    // launch-bound K=32 kernels can appear slower in warm mode simply because
    // the cold-cache memset raised the GPU clock first.
    for (int32_t i = 0; i < 8; ++i) {
      check_cuda(cudaMemsetAsync(cache_flush_buffer, i + 1, cache_scratch_bytes,
                                 cuda_config->stream),
                 "warm GPU clocks");
    }
    check_cuda(cudaStreamSynchronize(cuda_config->stream), "synchronize matmul setup");
  }
  const std::vector<int32_t> output_dims =
      use_batched_tensor ? std::vector<int32_t>{options.batch_size, options.output_size}
                         : std::vector<int32_t>{options.output_size};
  tensor::Tensor output(base::DataType::kDataTypeFp32, output_dims, true, output_alloc);
  op::MatmulLayer layer(device, options.output_size, options.input_size);
  layer.set_cuda_config(cuda_config);
  const base::Status weight_status = layer.set_weight(0, weight);
  if (!weight_status) {
    throw std::runtime_error("failed to bind matmul weight: " + weight_status.get_err_msg());
  }
  const std::string range_name = "matmul/" + implementation + "/" + options.dtype + "/n" +
                                 std::to_string(options.batch_size) + "_m" +
                                 std::to_string(options.input_size) + "_k" +
                                 std::to_string(options.output_size);
  std::vector<tensor::Tensor> input_rows;
  std::vector<tensor::Tensor> output_rows;
  if (implementation == "gemv-loop") {
    input_rows.reserve(options.batch_size);
    output_rows.reserve(options.batch_size);
    for (int32_t row = 0; row < options.batch_size; ++row) {
      input_rows.emplace_back(base::DataType::kDataTypeFp32, options.input_size, false, nullptr,
                              input.ptr<float>(static_cast<int64_t>(row) * options.input_size));
      output_rows.emplace_back(base::DataType::kDataTypeFp32, options.output_size, false, nullptr,
                               output.ptr<float>(static_cast<int64_t>(row) * options.output_size));
      input_rows.back().set_device_type(device);
      output_rows.back().set_device_type(device);
    }
  }
  auto launch = [&] {
    // MatmulLayer::forward() hides Layer's convenience overloads, so dispatch
    // through the base type just as the model's shared_ptr<Layer> plumbing does.
    op::Layer& base_layer = layer;
    if (implementation == "gemv-loop") {
      for (int32_t row = 0; row < options.batch_size; ++row) {
        const base::Status status = base_layer.forward(input_rows[row], output_rows[row]);
        if (!status) {
          throw std::runtime_error("matmul GEMV loop failed: " + status.get_err_msg());
        }
      }
    } else {
      const base::Status status = base_layer.forward(input, output);
      if (!status) {
        throw std::runtime_error("matmul failed: " + status.get_err_msg());
      }
    }
  };
  auto workload = [&] {
    base::ScopedNvtxRange range(range_name.c_str());
    launch();
  };

  for (int32_t i = 0; i < options.warmup; ++i) {
    workload();
  }
  if (device == base::DeviceType::kDeviceCUDA) {
    check_cuda(cudaStreamSynchronize(cuda_config->stream), "synchronize matmul warmup");
  }

  std::vector<double> samples;
  samples.reserve(options.repeat);
  if (device == base::DeviceType::kDeviceCUDA) {
    std::vector<cudaEvent_t> starts(options.repeat, nullptr);
    std::vector<cudaEvent_t> stops(options.repeat, nullptr);
    for (int32_t i = 0; i < options.repeat; ++i) {
      check_cuda(cudaEventCreate(&starts[i]), "create batched start event");
      check_cuda(cudaEventCreate(&stops[i]), "create batched stop event");
    }
    for (int32_t i = 0; i < options.repeat; ++i) {
      if (options.cache == "cold") {
        check_cuda(cudaMemsetAsync(cache_flush_buffer, i + 1, cache_scratch_bytes,
                                   cuda_config->stream),
                   "flush CUDA L2 cache");
      }
      check_cuda(cudaEventRecord(starts[i], cuda_config->stream), "record batched start event");
      workload();
      check_cuda(cudaEventRecord(stops[i], cuda_config->stream), "record batched stop event");
    }
    check_cuda(cudaEventSynchronize(stops.back()), "synchronize batched matmul measurements");
    for (int32_t i = 0; i < options.repeat; ++i) {
      float elapsed = 0.0f;
      check_cuda(cudaEventElapsedTime(&elapsed, starts[i], stops[i]),
                 "read batched matmul measurement");
      samples.push_back(elapsed);
      cudaEventDestroy(stops[i]);
      cudaEventDestroy(starts[i]);
    }
  } else {
    for (int32_t i = 0; i < options.repeat; ++i) {
      samples.push_back(measure_cpu_ms(workload));
    }
  }

  if (device == base::DeviceType::kDeviceCUDA) {
    output.to_cpu();
  }
  MatmulResults results;
  results.elapsed = summarize(samples);
  results.implementation = implementation;
  const int64_t output_elements =
      static_cast<int64_t>(options.batch_size) * options.output_size;
  for (int64_t i = 0; i < output_elements; ++i) {
    results.checksum += output.index<float>(i);
  }
  if (cache_flush_buffer) {
    check_cuda(cudaFree(cache_flush_buffer), "free cache flush buffer");
  }
  results.cache_flush_bytes = options.cache == "cold" ? cache_scratch_bytes : 0;
  const double seconds = results.elapsed.median_ms / 1000.0;
  results.gflops = 2.0 * options.batch_size * options.input_size * options.output_size / seconds /
                   1e9;
  results.logical_bytes = static_cast<size_t>(input_elements) * sizeof(float) +
                          static_cast<size_t>(weight_elements) * base::DataTypeSize(weight_type) +
                          static_cast<size_t>(output_elements) * sizeof(float);
  results.estimated_memory_bytes =
      static_cast<size_t>(input_elements) * sizeof(float) +
      static_cast<size_t>(weight_elements) * base::DataTypeSize(weight_type) *
          (implementation == "gemv-loop" ? options.batch_size : 1) +
      static_cast<size_t>(output_elements) * sizeof(float);
  results.effective_gbps = static_cast<double>(results.logical_bytes) / seconds / 1e9;
  results.estimated_memory_gbps =
      static_cast<double>(results.estimated_memory_bytes) / seconds / 1e9;
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
                        const MatmulResults* matmul, const GdnResults* gdn,
                        const ModelResults* model) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(6);
  out << "{\n"
      << "  \"schema_version\": 2,\n"
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
        << "    \"batch_size_n\": " << options.batch_size << ",\n"
        << "    \"input_size_m\": " << options.input_size << ",\n"
        << "    \"output_size_k\": " << options.output_size << ",\n"
        << "    \"implementation\": \"" << matmul->implementation << "\",\n"
        << "    \"weight_dtype\": \"" << options.dtype << "\",\n"
        << "    \"cache\": \"" << options.cache << "\",\n"
        << "    \"cache_flush_bytes\": " << matmul->cache_flush_bytes << ",\n"
        << "    \"elapsed\": ";
    write_stats(out, matmul->elapsed, 4);
    out << ",\n"
        << "    \"median_gflops\": " << matmul->gflops << ",\n"
        << "    \"logical_bytes\": " << matmul->logical_bytes << ",\n"
        << "    \"estimated_memory_bytes\": " << matmul->estimated_memory_bytes << ",\n"
        << "    \"median_effective_gbps\": " << matmul->effective_gbps << ",\n"
        << "    \"median_estimated_memory_gbps\": " << matmul->estimated_memory_gbps << ",\n"
        << "    \"output_checksum\": " << matmul->checksum << "\n"
        << "  }\n";
  } else if (gdn) {
    out << "  \"gdn\": {\n"
        << "    \"num_k_heads\": 16,\n"
        << "    \"num_v_heads\": 32,\n"
        << "    \"k_head_dim\": 128,\n"
        << "    \"v_head_dim\": 128,\n"
        << "    \"state_bytes\": " << gdn->state_bytes << ",\n"
        << "    \"elapsed\": ";
    write_stats(out, gdn->elapsed, 4);
    out << ",\n"
        << "    \"output_checksum\": " << gdn->output_checksum << ",\n"
        << "    \"state_checksum\": " << gdn->state_checksum << ",\n"
        << "    \"cpu_reference_max_abs_error\": " << gdn->max_abs_error << "\n"
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
    GdnResults gdn;
    ModelResults model;
    const MatmulResults* matmul_ptr = nullptr;
    const GdnResults* gdn_ptr = nullptr;
    const ModelResults* model_ptr = nullptr;
    if (options.mode == "matmul") {
      matmul = benchmark_matmul(options, device);
      matmul_ptr = &matmul;
    } else if (options.mode == "gdn") {
      gdn = benchmark_gdn(options, device);
      gdn_ptr = &gdn;
    } else {
      model = benchmark_model(options, device);
      model_ptr = &model;
    }
    const std::string json = result_json(options, runtime, matmul_ptr, gdn_ptr, model_ptr);
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
