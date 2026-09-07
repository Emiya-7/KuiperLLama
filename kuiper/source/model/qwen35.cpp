#ifdef QWEN35_SUPPORT
#include "model/qwen35.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <op/mha.h>
#include <cstdio>
#include <cstring>
#include <utility>
#include "../op/kernels/cpu/qwen35_kernel.h"
#include "../op/kernels/cuda/qwen35_kernel.cuh"

namespace model {

void Qwen35Layers::to_cuda(std::shared_ptr<kernel::CudaConfig> config) {
  auto move = [&config](const std::shared_ptr<op::Layer>& layer) {
    if (layer) {
      layer->set_cuda_config(config);
      layer->to_cuda();
    }
  };

  move(add_layer_);
  move(swiglu_layer_);
  move(embedding_layer_);
  move(cls_layer_);
  move(final_norm_);
  move(mha_layer_);
  move(rope_layer_);
  move(sigmoid_layer_);
  move(mul_layer_);
  move(q_l2norm_);
  move(k_l2norm_);
  move(gated_delta_);
  move(split_qgate_);

  for (auto& v : {&input_norms_, &post_attn_norms_, &w1_layers_, &w2_layers_, &w3_layers_}) {
    for (auto& l : *v) {
      move(l);
    }
  }

  for (auto& f : full_layers_) {
    move(f.wq);
    move(f.wk);
    move(f.wv);
    move(f.wo);
    move(f.q_norm);
    move(f.k_norm);
  }

  for (auto& l : linear_layers_) {
    move(l.in_proj_qkv);
    move(l.in_proj_z);
    move(l.in_proj_a);
    move(l.in_proj_b);
    move(l.conv);
    move(l.decay);
    move(l.norm);
    move(l.out_proj);
  }
}

Qwen35Model::Qwen35Model(base::TokenizerType tokenizer_type, std::string token_path,
                         std::string model_path, bool is_quant_model)
    : Model(tokenizer_type, base::ModelType::kModelTypeLLama2, std::move(token_path),
            std::move(model_path), is_quant_model) {}

tensor::Tensor& Qwen35Model::q35_buffer(Qwen35Buffer idx) const {
  auto it = q35_buffers_.find(idx);
  CHECK(it != q35_buffers_.end()) << "Missing Qwen3.5 buffer " << static_cast<int>(idx);
  return it->second;
}

tensor::Tensor Qwen35Model::view(Qwen35Buffer idx, int32_t offset, int32_t count) const {
  auto& base_tensor = q35_buffer(idx);
  CHECK_LE(static_cast<size_t>(offset) + count, base_tensor.size())
      << "Qwen3.5 buffer view runs past the end of the buffer.";
  float* p = const_cast<float*>(base_tensor.ptr<float>(offset));
  tensor::Tensor v(base::DataType::kDataTypeFp32, count, false, nullptr, p);
  v.set_device_type(device_type_);
  return v;
}

base::Status Qwen35Model::insert_q35_buffer(Qwen35Buffer idx, const tensor::Tensor& tensor) {
  if (q35_buffers_.count(idx) > 0) {
    return base::error::KeyHasExits("Qwen3.5 buffer already exists.");
  }
  if (tensor.is_empty()) {
    return base::error::InvalidArgument("Refusing to insert an empty Qwen3.5 buffer.");
  }
  q35_buffers_.insert({idx, tensor});
  return base::error::Success();
}

// The base class reads a 7-int ModelConfig; Qwen3.5's header is Qwen35RawConfig,
// so the whole read is replaced rather than adapted. The mmap setup mirrors the
// base implementation.
base::Status Qwen35Model::read_model_file() {
  using namespace base;
  if (model_path_.empty()) {
    return error::PathNotValid("The model path is empty.");
  }
  int32_t fd = open(model_path_.data(), O_RDONLY);
  if (fd == -1) {
    return error::PathNotValid("Failed to open the weight file " + model_path_);
  }
  FILE* file = fopen(model_path_.data(), "rb");
  if (!file) {
    close(fd);
    return error::PathNotValid("Failed to open the weight file " + model_path_);
  }

  Qwen35RawConfig raw{};
  if (fread(&raw, sizeof(Qwen35RawConfig), 1, file) != 1) {
    fclose(file);
    close(fd);
    return error::ModelParseError("Failed to read the Qwen3.5 header.");
  }
  fclose(file);

  if (raw.magic != kQwen35Magic) {
    close(fd);
    return error::ModelParseError(
        "Not a Qwen3.5 model file (bad magic). Export it with tools/export_qwen35/export.py.");
  }
  if (raw.version != kQwen35Version) {
    close(fd);
    return error::ModelParseError("Unsupported Qwen3.5 model file version.");
  }

  q35_.hidden_size = raw.hidden_size;
  q35_.intermediate_size = raw.intermediate_size;
  q35_.layer_num = raw.layer_num;
  q35_.vocab_size = raw.vocab_size;
  q35_.max_seq_len = raw.max_seq_len;
  q35_.head_num = raw.head_num;
  q35_.kv_head_num = raw.kv_head_num;
  q35_.head_dim = raw.head_dim;
  q35_.rotary_dim = raw.rotary_dim;
  q35_.full_attention_interval = raw.full_attention_interval;
  q35_.linear_num_k_heads = raw.linear_num_k_heads;
  q35_.linear_num_v_heads = raw.linear_num_v_heads;
  q35_.linear_k_head_dim = raw.linear_k_head_dim;
  q35_.linear_v_head_dim = raw.linear_v_head_dim;
  q35_.conv_kernel_size = raw.conv_kernel_size;
  q35_.tie_word_embeddings = raw.tie_word_embeddings != 0;
  q35_.rope_theta = raw.rope_theta;
  q35_.rms_norm_eps = raw.rms_norm_eps;
  q35_.derive();

  if (q35_.full_attention_interval <= 0) {
    close(fd);
    return error::ModelParseError("full_attention_interval must be positive.");
  }
  if (q35_.linear_num_k_heads <= 0 ||
      q35_.linear_num_v_heads % q35_.linear_num_k_heads != 0) {
    close(fd);
    return error::ModelParseError("linear_num_v_heads must be a multiple of linear_num_k_heads.");
  }

  // The base class's config_ drives shared plumbing (sampler, encode layer), so
  // fill in the fields those paths read. dim_/kv_dim_ deliberately describe the
  // full-attention layers; the GDN layers use q35_ directly.
  config_ = std::make_unique<TransformerConfig>();
  config_->dim_ = q35_.hidden_size;
  config_->hidden_dim_ = q35_.hidden_size;
  config_->layer_num_ = q35_.layer_num;
  config_->head_num_ = q35_.head_num;
  config_->kv_head_num_ = q35_.kv_head_num;
  config_->vocab_size_ = q35_.vocab_size;
  config_->seq_len_ = q35_.max_seq_len;
  config_->head_size_ = q35_.head_dim;
  config_->kv_dim_ = q35_.kv_dim;
  config_->kv_mul_ = q35_.kv_mul;
  config_->is_shared_weight_ = q35_.tie_word_embeddings;
#ifdef QWEN3_SUPPORT
  config_->immediate_dim_ = q35_.intermediate_size;
#endif

  raw_model_data_ = std::make_shared<RawModelDataFp32>();
  struct stat sb {};
  if (fstat(fd, &sb) == -1) {
    close(fd);
    return error::ModelParseError("Failed to stat the weight file.");
  }
  raw_model_data_->file_size = sb.st_size;
  raw_model_data_->fd = fd;
  raw_model_data_->data =
      mmap(nullptr, raw_model_data_->file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (raw_model_data_->data == MAP_FAILED || raw_model_data_->data == nullptr) {
    return error::ModelParseError("Failed to mmap the weight file " + model_path_);
  }
  raw_model_data_->weight_data =
      static_cast<int8_t*>(raw_model_data_->data) + sizeof(Qwen35RawConfig);

  LOG(INFO) << "Qwen3.5: hidden=" << q35_.hidden_size << " layers=" << q35_.layer_num << " ("
            << q35_.full_layer_num << " full / " << q35_.linear_layer_num << " linear)"
            << " head_dim=" << q35_.head_dim << " rotary_dim=" << q35_.rotary_dim
            << " tie_emb=" << q35_.tie_word_embeddings;
  return error::Success();
}

base::Status Qwen35Model::gen_model_from_file() {
  auto status = read_model_file();
  if (!status) {
    return status;
  }
  status = create_encode_layer();
  if (!status) {
    return status;
  }
  // create_encode_layer sets config_->vocab_size_ from the tokenizer, which for
  // Qwen3.5 reports fewer entries (248044) than the padded embedding matrix
  // (248320). The logits width follows the weights, so restore the header value;
  // otherwise sampling would read past the end of the row it was given.
  config_->vocab_size_ = q35_.vocab_size;
  return create_layers();
}

void Qwen35Model::create_param_quant_layers() {
  LOG(FATAL) << "Qwen3.5 int8 quantisation is not implemented.";
}

void Qwen35Model::create_nonparam_layers() {
  CHECK(layers_ != nullptr);
  layers_->add_layer_ = std::make_shared<op::VecAddLayer>(device_type_);
  layers_->swiglu_layer_ =
      std::make_shared<op::SwiGLULayer>(device_type_, q35_.intermediate_size);

  layers_->mha_layer_ = std::make_shared<op::MultiHeadAttention>(
      device_type_, 0, q35_.kv_mul, q35_.kv_dim, q35_.max_seq_len, q35_.head_num, q35_.head_dim);

  layers_->rope_layer_ = std::make_shared<op::RoPEPartialLayer>(
      device_type_, q35_.head_num, q35_.kv_head_num, q35_.head_dim, q35_.rotary_dim);

  layers_->sigmoid_layer_ = std::make_shared<op::SigmoidLayer>(device_type_);
  layers_->mul_layer_ = std::make_shared<op::MulLayer>(device_type_);

  // GDN query/key are L2-normalised per k-head.
  layers_->q_l2norm_ =
      std::make_shared<op::L2NormLayer>(device_type_, q35_.linear_k_head_dim, 1e-6f);
  layers_->k_l2norm_ =
      std::make_shared<op::L2NormLayer>(device_type_, q35_.linear_k_head_dim, 1e-6f);

  layers_->split_qgate_ = std::make_shared<op::SplitHeadInterleavedLayer>(
      device_type_, q35_.head_num, q35_.head_dim);

  layers_->gated_delta_ = std::make_shared<op::GatedDeltaLayer>(
      device_type_, q35_.linear_num_k_heads, q35_.linear_num_v_heads, q35_.linear_k_head_dim,
      q35_.linear_v_head_dim);
}

void Qwen35Model::create_param_layers() {
  CHECK(layers_ != nullptr);
  const auto cpu = base::DeviceType::kDeviceCPU;
  const int32_t hidden = q35_.hidden_size;
  const int32_t inter = q35_.intermediate_size;

  // Offsets are in floats and must track tools/export_qwen35/export.py exactly.
  size_t pos = 0;

  layers_->embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
      device_type_, hidden, q35_.max_seq_len, q35_.vocab_size);
  layers_->embedding_layer_->set_weight(0, {q35_.vocab_size, hidden},
                                        raw_model_data_->weight(pos), cpu);
  const size_t embedding_pos = pos;
  pos += static_cast<size_t>(q35_.vocab_size) * hidden;

  auto final_norm = std::make_shared<op::ZeroCenteredRMSNormLayer>(
      device_type_, hidden, q35_.rms_norm_eps);
  final_norm->set_weight(0, {hidden}, raw_model_data_->weight(pos), cpu);
  layers_->final_norm_ = final_norm;
  pos += hidden;

  layers_->input_norms_.resize(q35_.layer_num);
  layers_->post_attn_norms_.resize(q35_.layer_num);
  layers_->w1_layers_.resize(q35_.layer_num);
  layers_->w2_layers_.resize(q35_.layer_num);
  layers_->w3_layers_.resize(q35_.layer_num);
  layers_->full_layers_.resize(q35_.full_layer_num);
  layers_->linear_layers_.resize(q35_.linear_layer_num);

  for (int32_t i = 0; i < q35_.layer_num; ++i) {
    auto in_norm = std::make_shared<op::ZeroCenteredRMSNormLayer>(
        device_type_, hidden, q35_.rms_norm_eps);
    in_norm->set_weight(0, {hidden}, raw_model_data_->weight(pos), cpu);
    layers_->input_norms_[i] = in_norm;
    pos += hidden;

    const int32_t local = q35_.type_local_idx(i);
    if (q35_.layer_type(i) == Qwen35LayerType::kFullAttention) {
      auto& f = layers_->full_layers_[local];

      f.wq = std::make_shared<op::MatmulLayer>(device_type_, q35_.q_proj_out, hidden, false);
      f.wq->set_weight(0, {q35_.q_proj_out, hidden}, raw_model_data_->weight(pos), cpu);
      pos += static_cast<size_t>(q35_.q_proj_out) * hidden;

      f.wk = std::make_shared<op::MatmulLayer>(device_type_, q35_.kv_dim, hidden, false);
      f.wk->set_weight(0, {q35_.kv_dim, hidden}, raw_model_data_->weight(pos), cpu);
      pos += static_cast<size_t>(q35_.kv_dim) * hidden;

      f.wv = std::make_shared<op::MatmulLayer>(device_type_, q35_.kv_dim, hidden, false);
      f.wv->set_weight(0, {q35_.kv_dim, hidden}, raw_model_data_->weight(pos), cpu);
      pos += static_cast<size_t>(q35_.kv_dim) * hidden;

      // Qwen3.5 uses the same zero-centered (1 + weight) RMSNorm for Q/K,
      // independently over every head.
      f.q_norm = std::make_shared<op::ZeroCenteredRMSNormLayer>(
          device_type_, q35_.head_dim, q35_.rms_norm_eps);
      f.q_norm->set_weight(0, {q35_.head_dim}, raw_model_data_->weight(pos), cpu);
      pos += q35_.head_dim;

      f.k_norm = std::make_shared<op::ZeroCenteredRMSNormLayer>(
          device_type_, q35_.head_dim, q35_.rms_norm_eps);
      f.k_norm->set_weight(0, {q35_.head_dim}, raw_model_data_->weight(pos), cpu);
      pos += q35_.head_dim;

      f.wo = std::make_shared<op::MatmulLayer>(device_type_, hidden, q35_.q_dim, false);
      f.wo->set_weight(0, {hidden, q35_.q_dim}, raw_model_data_->weight(pos), cpu);
      pos += static_cast<size_t>(hidden) * q35_.q_dim;
    } else {
      auto& l = layers_->linear_layers_[local];

      l.in_proj_qkv = std::make_shared<op::MatmulLayer>(device_type_, q35_.conv_dim, hidden, false);
      l.in_proj_qkv->set_weight(0, {q35_.conv_dim, hidden}, raw_model_data_->weight(pos), cpu);
      pos += static_cast<size_t>(q35_.conv_dim) * hidden;

      l.in_proj_z = std::make_shared<op::MatmulLayer>(device_type_, q35_.linear_v_dim, hidden, false);
      l.in_proj_z->set_weight(0, {q35_.linear_v_dim, hidden}, raw_model_data_->weight(pos), cpu);
      pos += static_cast<size_t>(q35_.linear_v_dim) * hidden;

      l.in_proj_a =
          std::make_shared<op::MatmulLayer>(device_type_, q35_.linear_num_v_heads, hidden, false);
      l.in_proj_a->set_weight(0, {q35_.linear_num_v_heads, hidden}, raw_model_data_->weight(pos),
                              cpu);
      pos += static_cast<size_t>(q35_.linear_num_v_heads) * hidden;

      l.in_proj_b =
          std::make_shared<op::MatmulLayer>(device_type_, q35_.linear_num_v_heads, hidden, false);
      l.in_proj_b->set_weight(0, {q35_.linear_num_v_heads, hidden}, raw_model_data_->weight(pos),
                              cpu);
      pos += static_cast<size_t>(q35_.linear_num_v_heads) * hidden;

      l.conv = std::make_shared<op::CausalConv1DLayer>(device_type_, q35_.conv_dim,
                                                       q35_.conv_kernel_size);
      // conv1d.weight ships as [conv_dim, 1, k]; the singleton dim is dropped so
      // the kernel sees a plain [conv_dim, k].
      l.conv->set_weight(0, {q35_.conv_dim, q35_.conv_kernel_size}, raw_model_data_->weight(pos),
                         cpu);
      pos += static_cast<size_t>(q35_.conv_dim) * q35_.conv_kernel_size;

      l.decay = std::make_shared<op::SoftplusDecayLayer>(device_type_, q35_.linear_num_v_heads);
      l.decay->set_weight(0, {q35_.linear_num_v_heads}, raw_model_data_->weight(pos), cpu);
      pos += q35_.linear_num_v_heads;
      l.decay->set_weight(1, {q35_.linear_num_v_heads}, raw_model_data_->weight(pos), cpu);
      pos += q35_.linear_num_v_heads;

      l.norm = std::make_shared<op::GatedRMSNormLayer>(device_type_, q35_.linear_v_head_dim,
                                                       q35_.rms_norm_eps);
      l.norm->set_weight(0, {q35_.linear_v_head_dim}, raw_model_data_->weight(pos), cpu);
      pos += q35_.linear_v_head_dim;

      l.out_proj =
          std::make_shared<op::MatmulLayer>(device_type_, hidden, q35_.linear_v_dim, false);
      l.out_proj->set_weight(0, {hidden, q35_.linear_v_dim}, raw_model_data_->weight(pos), cpu);
      pos += static_cast<size_t>(hidden) * q35_.linear_v_dim;
    }

    auto post_norm = std::make_shared<op::ZeroCenteredRMSNormLayer>(
        device_type_, hidden, q35_.rms_norm_eps);
    post_norm->set_weight(0, {hidden}, raw_model_data_->weight(pos), cpu);
    layers_->post_attn_norms_[i] = post_norm;
    pos += hidden;

    auto w1 = std::make_shared<op::MatmulLayer>(device_type_, inter, hidden, false);
    w1->set_weight(0, {inter, hidden}, raw_model_data_->weight(pos), cpu);
    layers_->w1_layers_[i] = w1;
    pos += static_cast<size_t>(inter) * hidden;

    auto w3 = std::make_shared<op::MatmulLayer>(device_type_, inter, hidden, false);
    w3->set_weight(0, {inter, hidden}, raw_model_data_->weight(pos), cpu);
    layers_->w3_layers_[i] = w3;
    pos += static_cast<size_t>(inter) * hidden;

    auto w2 = std::make_shared<op::MatmulLayer>(device_type_, hidden, inter, false);
    w2->set_weight(0, {hidden, inter}, raw_model_data_->weight(pos), cpu);
    layers_->w2_layers_[i] = w2;
    pos += static_cast<size_t>(hidden) * inter;
  }

  auto lm_head = std::make_shared<op::MatmulLayer>(device_type_, q35_.vocab_size, hidden, false);
  if (q35_.tie_word_embeddings) {
    // 4B and 2B ship no lm_head; the embedding matrix is reused, which is also
    // why the export stops before writing one.
    lm_head->set_weight(0, {q35_.vocab_size, hidden}, raw_model_data_->weight(embedding_pos), cpu);
  } else {
    lm_head->set_weight(0, {q35_.vocab_size, hidden}, raw_model_data_->weight(pos), cpu);
    pos += static_cast<size_t>(q35_.vocab_size) * hidden;
  }
  layers_->cls_layer_ = lm_head;

  // The header is excluded from weight_data, so `pos` floats must land exactly on
  // the end of the file. A mismatch means the export order and this reader have
  // drifted apart -- fail loudly rather than infer garbage.
  const size_t expect_bytes = pos * sizeof(float) + sizeof(Qwen35RawConfig);
  CHECK_EQ(expect_bytes, raw_model_data_->file_size)
      << "Weight layout mismatch: consumed " << pos << " floats (" << expect_bytes
      << " bytes with header) but the file is " << raw_model_data_->file_size << " bytes.";
}

base::Status Qwen35Model::create_layers() {
  using namespace base;
  if (!layers_) {
    layers_ = std::make_unique<Qwen35Layers>();
  }
  if (is_quant_model_) {
    return error::FunctionNotImplement("Qwen3.5 int8 quantisation is not implemented.");
  }
  create_param_layers();
  create_nonparam_layers();

  if (!layers_->embedding_layer_ || !layers_->cls_layer_ || !layers_->final_norm_) {
    return error::InternalError("Failed to create the Qwen3.5 embedding/cls/norm layers.");
  }
  for (int32_t i = 0; i < q35_.layer_num; ++i) {
    if (!layers_->input_norms_[i] || !layers_->post_attn_norms_[i] || !layers_->w1_layers_[i] ||
        !layers_->w2_layers_[i] || !layers_->w3_layers_[i]) {
      return error::InternalError("Failed to create the Qwen3.5 per-layer norm/FFN layers.");
    }
  }
  for (const auto& f : layers_->full_layers_) {
    if (!f.wq || !f.wk || !f.wv || !f.wo || !f.q_norm || !f.k_norm) {
      return error::InternalError("Failed to create a Qwen3.5 full-attention layer.");
    }
  }
  for (const auto& l : layers_->linear_layers_) {
    if (!l.in_proj_qkv || !l.in_proj_z || !l.in_proj_a || !l.in_proj_b || !l.conv || !l.decay ||
        !l.norm || !l.out_proj) {
      return error::InternalError("Failed to create a Qwen3.5 linear-attention layer.");
    }
  }
  if (!layers_->mha_layer_ || !layers_->rope_layer_ || !layers_->gated_delta_ ||
      !layers_->sigmoid_layer_ || !layers_->mul_layer_ || !layers_->q_l2norm_ ||
      !layers_->k_l2norm_ || !layers_->add_layer_ || !layers_->swiglu_layer_ ||
      !layers_->split_qgate_) {
    return error::InternalError("Failed to create the Qwen3.5 stateless layers.");
  }
  return error::Success();
}

void Qwen35Model::init_mem() {
  using namespace base;
  std::shared_ptr<DeviceAllocator> alloc;
  if (device_type_ == DeviceType::kDeviceCPU) {
    alloc = CPUDeviceAllocatorFactory::get_instance();
  } else {
    alloc = CUDADeviceAllocatorFactory::get_instance();
  }
  auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();

  if (device_type_ == DeviceType::kDeviceCUDA) {
    CHECK_NE(cuda_config_, nullptr);
    layers_->to_cuda(cuda_config_);
  }

  const auto f32 = DataType::kDataTypeFp32;
  const int32_t hidden = q35_.hidden_size;

  tensor::Tensor input_tokens(DataType::kDataTypeInt32, 1, true, alloc_cpu);
  tensor::Tensor input_embeddings(f32, 1, hidden, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kInputTokens, input_tokens));
  CHECK(insert_buffer(ModelBufferType::kInputEmbeddings, input_embeddings));

  tensor::Tensor pos_tensor(DataType::kDataTypeInt32, 1, true, alloc_cpu);
  CHECK(insert_buffer(ModelBufferType::kInputPos, pos_tensor));

  // Partial RoPE tables are [max_seq_len, rotary_dim/2]: sized by rotary_dim, not
  // head_dim, which is what keeps them small despite the 262144-token config.
  const int32_t rope_half = q35_.rotary_dim / 2;
  tensor::Tensor sin_cache(f32, q35_.max_seq_len * rope_half, true, alloc);
  tensor::Tensor cos_cache(f32, q35_.max_seq_len * rope_half, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kSinCache, sin_cache));
  CHECK(insert_buffer(ModelBufferType::kCosCache, cos_cache));

  tensor::Tensor rms_output(f32, hidden, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kOutputRMSNorm, rms_output));
  CHECK(insert_buffer(ModelBufferType::kFFNRMSNorm, rms_output));
  CHECK(insert_buffer(ModelBufferType::kW2Output, rms_output));

  tensor::Tensor w1_output(f32, q35_.intermediate_size, true, alloc);
  tensor::Tensor w3_output(f32, q35_.intermediate_size, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kW1Output, w1_output));
  CHECK(insert_buffer(ModelBufferType::kW3Output, w3_output));

  // KV cache covers only the full-attention layers, indexed by type_local_idx.
  // At the 4:1 ratio that is a quarter of what a uniform model would need.
  tensor::Tensor key_cache(f32, q35_.full_layer_num, q35_.max_seq_len, q35_.kv_dim, true, alloc);
  tensor::Tensor value_cache(f32, q35_.full_layer_num, q35_.max_seq_len, q35_.kv_dim, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kKeyCache, key_cache));
  CHECK(insert_buffer(ModelBufferType::kValueCache, value_cache));

  tensor::Tensor score_storage(f32, q35_.head_num, q35_.max_seq_len, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kScoreStorage, score_storage));

  tensor::Tensor attn_output(f32, hidden, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kAttnOutput, attn_output));

  tensor::Tensor forward_output(f32, q35_.vocab_size, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kForwardOutput, forward_output));
  if (device_type_ == DeviceType::kDeviceCUDA) {
    tensor::Tensor forward_output_cpu(f32, q35_.vocab_size, true, alloc_cpu);
    CHECK(insert_buffer(ModelBufferType::kForwardOutputCPU, forward_output_cpu));
  }

  // ---- Qwen3.5-specific buffers ----
  auto add = [&](Qwen35Buffer id, int32_t n) {
    CHECK(insert_q35_buffer(id, tensor::Tensor(f32, n, true, alloc)));
  };
  add(Qwen35Buffer::kQProj, q35_.q_proj_out);
  add(Qwen35Buffer::kQuery, q35_.q_dim);
  add(Qwen35Buffer::kQueryGate, q35_.q_dim);
  add(Qwen35Buffer::kAttnOut, q35_.q_dim);
  add(Qwen35Buffer::kMixedQKV, q35_.conv_dim);
  add(Qwen35Buffer::kConvOut, q35_.conv_dim);
  add(Qwen35Buffer::kGdnZ, q35_.linear_v_dim);
  add(Qwen35Buffer::kGdnA, q35_.linear_num_v_heads);
  add(Qwen35Buffer::kGdnB, q35_.linear_num_v_heads);
  add(Qwen35Buffer::kGdnG, q35_.linear_num_v_heads);
  add(Qwen35Buffer::kGdnBeta, q35_.linear_num_v_heads);
  add(Qwen35Buffer::kGdnCore, q35_.linear_v_dim);
  add(Qwen35Buffer::kGdnNormed, q35_.linear_v_dim);

  // Recurrent + conv state: one slot per linear layer, independent of sequence
  // length. This is what replaces a KV cache for three quarters of the layers.
  CHECK(insert_q35_buffer(
      Qwen35Buffer::kRecurrentState,
      tensor::Tensor(f32, q35_.linear_layer_num, q35_.state_size, true, alloc)));
  CHECK(insert_q35_buffer(
      Qwen35Buffer::kConvState,
      tensor::Tensor(f32, q35_.linear_layer_num, q35_.conv_state_size, true, alloc)));
  reset_state();
}

void Qwen35Model::reset_state() const {
  auto zero = [this](Qwen35Buffer id) {
    auto& t = q35_buffer(id);
    const size_t bytes = t.size() * sizeof(float);
    void* p = const_cast<void*>(static_cast<const void*>(t.ptr<float>()));
    if (device_type_ == base::DeviceType::kDeviceCUDA) {
      CHECK_NE(cuda_config_, nullptr);
      cudaMemsetAsync(p, 0, bytes, cuda_config_->stream);
    } else {
      std::memset(p, 0, bytes);
    }
  };
  zero(Qwen35Buffer::kRecurrentState);
  zero(Qwen35Buffer::kConvState);
  if (device_type_ == base::DeviceType::kDeviceCUDA && cuda_config_) {
    cudaStreamSynchronize(cuda_config_->stream);
  }
}

base::Status Qwen35Model::init(base::DeviceType device_type) {
  using namespace base;
  if (token_path_.empty()) {
    return error::PathNotValid(token_path_);
  }
  if (device_type == DeviceType::kDeviceCPU && is_quant_model_) {
    return error::InternalError("The cpu device does not support int8 quant models.");
  }
  device_type_ = device_type;
  if (device_type == DeviceType::kDeviceCUDA) {
    cudaSetDevice(0);
    cuda_config_ = std::make_shared<kernel::CudaConfig>();
    cudaStreamCreate(&cuda_config_->stream);
    if (cudaGetLastError() != cudaSuccess) {
      return error::InternalError("Failed to create the cuda stream.");
    }
  }

  auto status = gen_model_from_file();
  if (!status) {
    return status;
  }
  init_mem();

  if (device_type_ == DeviceType::kDeviceCPU) {
    kernel::rope_partial_cache_cpu(
        const_cast<float*>(get_buffer(ModelBufferType::kSinCache).ptr<float>()),
        const_cast<float*>(get_buffer(ModelBufferType::kCosCache).ptr<float>()), q35_.max_seq_len,
        q35_.rotary_dim, q35_.rope_theta);
  } else {
    CHECK_NE(cuda_config_, nullptr);
    kernel::rope_partial_cache_cu(
        const_cast<float*>(get_buffer(ModelBufferType::kSinCache).ptr<float>()),
        const_cast<float*>(get_buffer(ModelBufferType::kCosCache).ptr<float>()), q35_.max_seq_len,
        q35_.rotary_dim, q35_.rope_theta, cuda_config_->stream);
    cudaStreamSynchronize(cuda_config_->stream);
  }

  sampler_ = std::make_unique<sampler::ArgmaxSampler>(device_type_);
  return error::Success();
}

op::EmbeddingOutput Qwen35Model::embedding(const std::vector<int>& tokens) const {
  auto input_tokens = get_buffer(ModelBufferType::kInputTokens);
  auto input_embeddings = get_buffer(ModelBufferType::kInputEmbeddings);
  if (input_tokens.size() != tokens.size()) {
    input_tokens.reshape({static_cast<int32_t>(tokens.size())});
    input_embeddings.reshape({static_cast<int32_t>(tokens.size()), q35_.hidden_size});
  }
  for (size_t i = 0; i < tokens.size(); ++i) {
    input_tokens.index<int32_t>(static_cast<int32_t>(i)) = tokens.at(i);
  }
  auto input_token_num =
      tensor::Tensor(base::DataType::kDataTypeInt32, static_cast<int32_t>(tokens.size()));
  LOG_IF(FATAL, !layers_->embedding_layer_) << "The Qwen3.5 embedding layer is null.";
  STATUS_CHECK(
      layers_->embedding_layer_->forward(input_tokens, input_token_num, input_embeddings));
  return op::EmbeddingOutput(input_tokens, input_embeddings, input_token_num);
}

void Qwen35Model::attention_full(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  const int32_t local = q35_.type_local_idx(layer_idx);
  const auto& f = layers_->full_layers_.at(local);
  const int32_t pos = pos_tensor.index<int32_t>(0);

  auto normed = get_buffer(ModelBufferType::kOutputRMSNorm);

  // q_proj emits num_heads * head_dim * 2 values, laid out per head as
  // [query | gate] pairs -- HF views it as (num_heads, head_dim * 2) and chunks
  // the last axis. So the query and gate of one head are adjacent, and the split
  // is a strided deinterleave rather than two contiguous halves.
  auto qproj = q35_buffer(Qwen35Buffer::kQProj);
  STATUS_CHECK(f.wq->forward(normed, qproj));
  auto query = q35_buffer(Qwen35Buffer::kQuery);
  auto gate = q35_buffer(Qwen35Buffer::kQueryGate);
  // Two outputs, so bind explicitly: the 3-argument forward() overload reads its
  // middle argument as a second input, not as an output.
  auto& split = *layers_->split_qgate_;
  split.set_input(0, qproj);
  split.set_output(0, query);
  split.set_output(1, gate);
  STATUS_CHECK(split.forward());

  // Per-head QK-norm over head_dim, then partial RoPE. The KV cache holds only
  // full-attention layers, so it is addressed by the full-layer ordinal; passing
  // the absolute layer index here would run off the end of the buffer.
  auto [key, val] = slice_kv_cache(local, pos);
  STATUS_CHECK(f.wk->forward(normed, key));
  STATUS_CHECK(f.wv->forward(normed, val));

  // ZeroCenteredRMSNormLayer derives the row count from the tensor size, so no
  // reshape is needed around these.
  STATUS_CHECK(f.q_norm->forward(query, query));
  STATUS_CHECK(f.k_norm->forward(key, key));

  STATUS_CHECK(layers_->rope_layer_->forward(query, key, pos_tensor,
                                             get_buffer(ModelBufferType::kSinCache),
                                             get_buffer(ModelBufferType::kCosCache),
                                             tensor::Tensor{}));

  // Cast only to reach the setters; forward() goes through the base pointer so
  // the multi-argument overload is visible.
  auto mha = std::dynamic_pointer_cast<op::MultiHeadAttention>(layers_->mha_layer_);
  CHECK_NE(mha, nullptr);
  mha->set_pos(pos);
  mha->set_layer_idx(local);  // KV cache is indexed by full-layer ordinal
  auto attn_out = q35_buffer(Qwen35Buffer::kAttnOut);
  STATUS_CHECK(layers_->mha_layer_->forward(query, get_buffer(ModelBufferType::kScoreStorage),
                                            get_buffer(ModelBufferType::kKeyCache),
                                            get_buffer(ModelBufferType::kValueCache), attn_out));

  // attn_out *= sigmoid(gate) -- the gate reuses its own buffer for the sigmoid.
  STATUS_CHECK(layers_->sigmoid_layer_->forward(gate, gate));
  STATUS_CHECK(layers_->mul_layer_->forward(attn_out, gate, attn_out));

  STATUS_CHECK(f.wo->forward(attn_out, get_buffer(ModelBufferType::kAttnOutput)));
}

void Qwen35Model::attention_linear(int32_t layer_idx) const {
  const int32_t local = q35_.type_local_idx(layer_idx);
  const auto& l = layers_->linear_layers_.at(local);
  auto normed = get_buffer(ModelBufferType::kOutputRMSNorm);

  auto mixed = q35_buffer(Qwen35Buffer::kMixedQKV);
  auto conv_out = q35_buffer(Qwen35Buffer::kConvOut);
  STATUS_CHECK(l.in_proj_qkv->forward(normed, mixed));

  // Depthwise causal conv over the packed [q | k | v], silu fused in. Each linear
  // layer owns one slot of the conv state.
  auto conv_state =
      view(Qwen35Buffer::kConvState, local * q35_.conv_state_size, q35_.conv_state_size);
  STATUS_CHECK(l.conv->forward(mixed, conv_state, conv_out));

  // q, k, v are views into the conv output rather than copies. l2norm below
  // writes in place, which keeps them consistent with what the kernel reads.
  auto q = view(Qwen35Buffer::kConvOut, 0, q35_.linear_k_dim);
  auto k = view(Qwen35Buffer::kConvOut, q35_.linear_k_dim, q35_.linear_k_dim);
  auto v = view(Qwen35Buffer::kConvOut, 2 * q35_.linear_k_dim, q35_.linear_v_dim);

  // L2-normalise q and k per k-head. The 1/sqrt(k_head_dim) query scale is
  // applied inside the delta-rule kernel, matching the reference order.
  q.reshape({q35_.linear_num_k_heads, q35_.linear_k_head_dim});
  STATUS_CHECK(layers_->q_l2norm_->forward(q, q));
  q.reshape({q35_.linear_k_dim});
  k.reshape({q35_.linear_num_k_heads, q35_.linear_k_head_dim});
  STATUS_CHECK(layers_->k_l2norm_->forward(k, k));
  k.reshape({q35_.linear_k_dim});

  // beta = sigmoid(in_proj_b(x)); g = -exp(A_log) * softplus(in_proj_a(x) + dt_bias)
  auto a = q35_buffer(Qwen35Buffer::kGdnA);
  auto b = q35_buffer(Qwen35Buffer::kGdnB);
  auto g = q35_buffer(Qwen35Buffer::kGdnG);
  auto beta = q35_buffer(Qwen35Buffer::kGdnBeta);
  STATUS_CHECK(l.in_proj_a->forward(normed, a));
  STATUS_CHECK(l.in_proj_b->forward(normed, b));
  STATUS_CHECK(l.decay->forward(a, g));
  STATUS_CHECK(layers_->sigmoid_layer_->forward(b, beta));

  auto state = view(Qwen35Buffer::kRecurrentState, local * q35_.state_size, q35_.state_size);
  auto core = q35_buffer(Qwen35Buffer::kGdnCore);
  // Six inputs exceeds the widest forward() overload, so bind them explicitly.
  auto& gdn = *layers_->gated_delta_;
  gdn.set_input(0, q);
  gdn.set_input(1, k);
  gdn.set_input(2, v);
  gdn.set_input(3, g);
  gdn.set_input(4, beta);
  gdn.set_input(5, state);
  gdn.set_output(0, core);
  STATUS_CHECK(gdn.forward());

  // out = gated_rmsnorm(core, z) over v_head_dim, then out_proj.
  auto z = q35_buffer(Qwen35Buffer::kGdnZ);
  STATUS_CHECK(l.in_proj_z->forward(normed, z));
  auto normed_core = q35_buffer(Qwen35Buffer::kGdnNormed);
  core.reshape({q35_.linear_num_v_heads, q35_.linear_v_head_dim});
  z.reshape({q35_.linear_num_v_heads, q35_.linear_v_head_dim});
  normed_core.reshape({q35_.linear_num_v_heads, q35_.linear_v_head_dim});
  STATUS_CHECK(l.norm->forward(core, z, normed_core));
  core.reshape({q35_.linear_v_dim});
  z.reshape({q35_.linear_v_dim});
  normed_core.reshape({q35_.linear_v_dim});

  STATUS_CHECK(l.out_proj->forward(normed_core, get_buffer(ModelBufferType::kAttnOutput)));
}

void Qwen35Model::feed_forward(int32_t layer_idx, const tensor::Tensor& input) const {
  STATUS_CHECK(layers_->add_layer_->forward(input, get_buffer(ModelBufferType::kAttnOutput),
                                            input));

  auto ffn_norm = get_buffer(ModelBufferType::kFFNRMSNorm);
  STATUS_CHECK(layers_->post_attn_norms_.at(layer_idx)->forward(input, ffn_norm));

  auto w1_out = get_buffer(ModelBufferType::kW1Output);
  auto w3_out = get_buffer(ModelBufferType::kW3Output);
  STATUS_CHECK(layers_->w1_layers_.at(layer_idx)->forward(ffn_norm, w1_out));
  STATUS_CHECK(layers_->w3_layers_.at(layer_idx)->forward(ffn_norm, w3_out));
  STATUS_CHECK(layers_->swiglu_layer_->forward(w1_out, w3_out, w1_out));

  auto w2_out = get_buffer(ModelBufferType::kW2Output);
  STATUS_CHECK(layers_->w2_layers_.at(layer_idx)->forward(w1_out, w2_out));
  STATUS_CHECK(layers_->add_layer_->forward(input, w2_out, input));
}

void Qwen35Model::cls_logits(const tensor::Tensor& input) const {
  STATUS_CHECK(layers_->final_norm_->forward(input, input));
  STATUS_CHECK(
      layers_->cls_layer_->forward(input, get_buffer(ModelBufferType::kForwardOutput)));
}

base::Status Qwen35Model::forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                                 int& next) const {
  if (input.is_empty()) {
    return base::error::InvalidArgument("The input tensor is empty.");
  }
  for (int32_t i = 0; i < q35_.layer_num; ++i) {
    STATUS_CHECK(layers_->input_norms_.at(i)->forward(
        input, get_buffer(ModelBufferType::kOutputRMSNorm)));
    if (q35_.layer_type(i) == Qwen35LayerType::kFullAttention) {
      attention_full(i, pos_tensor);
    } else {
      attention_linear(i);
    }
    feed_forward(i, input);
  }
  cls_logits(input);
  return base::error::Success();
}

base::Status Qwen35Model::predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                                  bool is_prompt, int& next) const {
  auto status = forward(input, pos_tensor, next);
  if (!status) {
    return status;
  }
  next = post_processing(pos_tensor, is_prompt);
  return base::error::Success();
}

int32_t Qwen35Model::post_processing(const tensor::Tensor& pos, bool is_prompt) const {
  auto forward_output = get_buffer(ModelBufferType::kForwardOutput);
  if (is_prompt) {
    return -1;
  }
  return static_cast<int32_t>(sampler_->sample(forward_output.ptr<float>(), forward_output.size(),
                                               cuda_config_ ? cuda_config_->stream : nullptr));
}

}  // namespace model

#endif  // QWEN35_SUPPORT
