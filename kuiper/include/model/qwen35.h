#ifndef KUIPER_INCLUDE_MODEL_QWEN35_H_
#define KUIPER_INCLUDE_MODEL_QWEN35_H_
#include <base/cuda_config.h>
#include <memory>
#include <vector>
#include "model.h"
#include "op/add.h"
#include "op/embedding.h"
#include "op/matmul.h"
#include "op/mha.h"
#include "op/qwen35_ops.h"
#include "op/swiglu.h"
#include "qwen35_config.h"

namespace model {

// Weights and sublayers of one full-attention (GQA) layer.
//
// Differs from Qwen3's attention block in three ways: q_proj emits 2x the head
// width because the second half is a sigmoid output gate, RoPE only rotates the
// leading rotary_dim of each head, and head_dim is independent of
// hidden_size / head_num.
// Layers are held as op::Layer pointers, matching the rest of the project. The
// multi-argument forward() overloads live on op::Layer and are hidden by the
// derived classes' own forward(), so calling through the base type is what makes
// forward(in, out) resolve.
struct Qwen35FullAttnLayer {
  std::shared_ptr<op::Layer> wq;  // [head_num * head_dim * 2, hidden]
  std::shared_ptr<op::Layer> wk;
  std::shared_ptr<op::Layer> wv;
  std::shared_ptr<op::Layer> wo;
  std::shared_ptr<op::Layer> q_norm;  // per-head, over head_dim
  std::shared_ptr<op::Layer> k_norm;
};

// Weights and sublayers of one linear-attention (Gated DeltaNet) layer.
struct Qwen35LinearAttnLayer {
  std::shared_ptr<op::Layer> in_proj_qkv;  // [conv_dim, hidden]
  std::shared_ptr<op::Layer> in_proj_z;    // [v_dim, hidden], gate
  std::shared_ptr<op::Layer> in_proj_a;    // [num_v_heads, hidden]
  std::shared_ptr<op::Layer> in_proj_b;    // [num_v_heads, hidden]
  std::shared_ptr<op::Layer> conv;
  std::shared_ptr<op::Layer> decay;  // owns A_log and dt_bias
  std::shared_ptr<op::Layer> norm;   // gated, over v_head_dim
  std::shared_ptr<op::Layer> out_proj;  // [hidden, v_dim]
};

struct Qwen35Layers {
  // Shared across both layer kinds.
  std::shared_ptr<op::Layer> add_layer_;
  std::shared_ptr<op::Layer> swiglu_layer_;
  std::shared_ptr<op::Layer> embedding_layer_;
  std::shared_ptr<op::Layer> cls_layer_;
  std::shared_ptr<op::Layer> final_norm_;

  // Per layer, indexed by absolute layer index.
  std::vector<std::shared_ptr<op::Layer>> input_norms_;
  std::vector<std::shared_ptr<op::Layer>> post_attn_norms_;
  std::vector<std::shared_ptr<op::Layer>> w1_layers_;  // gate_proj
  std::vector<std::shared_ptr<op::Layer>> w2_layers_;  // down_proj
  std::vector<std::shared_ptr<op::Layer>> w3_layers_;  // up_proj

  // Sparse: only the entries for layers of the matching kind are populated,
  // indexed by Qwen35Config::type_local_idx.
  std::vector<Qwen35FullAttnLayer> full_layers_;
  std::vector<Qwen35LinearAttnLayer> linear_layers_;

  // Stateless ops, shared by every layer of the relevant kind.
  std::shared_ptr<op::Layer> mha_layer_;
  std::shared_ptr<op::Layer> rope_layer_;
  std::shared_ptr<op::Layer> sigmoid_layer_;
  std::shared_ptr<op::Layer> mul_layer_;
  std::shared_ptr<op::Layer> q_l2norm_;
  std::shared_ptr<op::Layer> k_l2norm_;
  std::shared_ptr<op::Layer> gated_delta_;
  std::shared_ptr<op::Layer> split_qgate_;

  void to_cuda(std::shared_ptr<kernel::CudaConfig> config);
};

// Buffers that only Qwen3.5 needs. The base ModelBufferType enum covers the
// shared ones; these are kept in a separate map so base/base.h stays untouched.
enum class Qwen35Buffer {
  // q_proj writes [q_dim * 2]: first half query, second half the output gate.
  // One buffer, two views, so the projection stays a single matmul.
  kQProj,         // [q_dim * 2] raw, per-head interleaved [q|gate] pairs
  kQuery,         // [q_dim] deinterleaved query
  kQueryGate,     // [q_dim] deinterleaved gate
  kAttnOut,       // [q_dim] MHA output before o_proj
  kMixedQKV,      // [conv_dim] in_proj_qkv output
  kConvOut,       // [conv_dim] post-conv; q/k/v are views into this
  kGdnZ,          // [linear_v_dim]
  kGdnA,          // [num_v_heads]
  kGdnB,          // [num_v_heads]
  kGdnG,          // [num_v_heads]
  kGdnBeta,       // [num_v_heads]
  kGdnCore,       // [linear_v_dim] delta-rule output
  kGdnNormed,     // [linear_v_dim] after gated rmsnorm
  kRecurrentState,  // [linear_layer_num, num_v_heads, k_head_dim, v_head_dim]
  kConvState,       // [linear_layer_num, conv_dim, k-1]
};

class Qwen35Model : public Model {
 public:
  explicit Qwen35Model(base::TokenizerType tokenizer_type, std::string token_path,
                       std::string model_path, bool is_quant_model);

  base::Status init(base::DeviceType device_type) override;

  base::Status predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                       bool is_prompt, int& next) const override;

  base::Status forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                       int& next) const override;

  op::EmbeddingOutput embedding(const std::vector<int>& tokens) const override;

  const Qwen35Config& qwen35_config() const { return q35_; }

  // Zeroes the recurrent and conv states. Must be called before starting a new
  // sequence: unlike a KV cache, which is overwritten position by position, the
  // GDN state accumulates and would otherwise leak across sequences.
  void reset_state() const;

 private:
  void init_mem() override;

  base::Status create_layers() override;

  void create_param_layers() override;

  void create_nonparam_layers() override;

  void create_param_quant_layers() override;

  // Qwen3.5's header is a different format from the base class's 7-int one, so
  // reading it is overridden wholesale rather than shimmed into ModelConfig.
  base::Status read_model_file() override;

  base::Status gen_model_from_file() override;

  void attention_full(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

  void attention_linear(int32_t layer_idx) const;

  void feed_forward(int32_t layer_idx, const tensor::Tensor& input) const;

  void cls_logits(const tensor::Tensor& input) const;

  int32_t post_processing(const tensor::Tensor& pos, bool is_prompt) const override;

  tensor::Tensor& q35_buffer(Qwen35Buffer idx) const;

  base::Status insert_q35_buffer(Qwen35Buffer idx, const tensor::Tensor& tensor);

  // A non-owning 1-D view of `count` floats starting at `offset` inside a buffer,
  // following the same pattern as Model::slice_kv_cache. Used to split packed
  // projections (q_proj -> query|gate, conv output -> q|k|v) and to address one
  // layer's slot in the recurrent/conv state without copying.
  tensor::Tensor view(Qwen35Buffer idx, int32_t offset, int32_t count) const;

 private:
  Qwen35Config q35_;
  std::shared_ptr<kernel::CudaConfig> cuda_config_;
  std::unique_ptr<Qwen35Layers> layers_;
  mutable std::map<Qwen35Buffer, tensor::Tensor> q35_buffers_;
};

}  // namespace model
#endif  // KUIPER_INCLUDE_MODEL_QWEN35_H_
