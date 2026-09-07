#ifndef KUIPER_INCLUDE_MODEL_QWEN35_CONFIG_H_
#define KUIPER_INCLUDE_MODEL_QWEN35_CONFIG_H_
#include <cstdint>

namespace model {

// Qwen3.5 mixes two kinds of attention layers. `layer_types` in the HF config
// spells the pattern out per layer; it is always (interval-1) linear layers
// followed by one full-attention layer, so we recompute it from the interval
// instead of storing the list.
enum class Qwen35LayerType : int32_t {
  kLinearAttention = 0,  // Gated DeltaNet
  kFullAttention = 1,    // GQA + QK-norm + partial RoPE + output gate
};

// On-disk header for the text tower of a Qwen3.5 checkpoint. Written by
// tools/export_qwen35/export.py, read by Qwen35Model::read_model_file.
// All ints are int32 little-endian, all floats are fp32 little-endian, so the
// struct maps 1:1 onto the file bytes on the platforms we support.
//
// The existing 7-int LLama/Qwen2 header cannot describe a hybrid model, so this
// is a separate versioned format rather than an extension of that one.
struct Qwen35RawConfig {
  int32_t magic;    // 'K'|'3'<<8|'5'<<16|'D'<<24
  int32_t version;  // 2

  int32_t hidden_size;
  int32_t intermediate_size;
  int32_t layer_num;
  int32_t vocab_size;
  int32_t max_seq_len;  // clamped at export time, not the config's 262144

  // full-attention layers
  int32_t head_num;
  int32_t kv_head_num;
  int32_t head_dim;    // 256, independent of hidden_size / head_num
  int32_t rotary_dim;  // head_dim * partial_rotary_factor, i.e. 64

  int32_t full_attention_interval;  // 4 -> layers 3, 7, 11, ... are full

  // linear-attention (Gated DeltaNet) layers
  int32_t linear_num_k_heads;
  int32_t linear_num_v_heads;
  int32_t linear_k_head_dim;
  int32_t linear_v_head_dim;
  int32_t conv_kernel_size;

  int32_t tie_word_embeddings;  // 1 -> no lm_head in the file, reuse embedding

  float rope_theta;
  float rms_norm_eps;
};

static constexpr int32_t kQwen35Magic = 'K' | ('3' << 8) | ('5' << 16) | ('D' << 24);
static constexpr int32_t kQwen35Version = 2;

// Runtime view of the above, with the derived sizes the layers actually need.
struct Qwen35Config {
  int32_t hidden_size = 0;
  int32_t intermediate_size = 0;
  int32_t layer_num = 0;
  int32_t vocab_size = 0;
  int32_t max_seq_len = 0;

  int32_t head_num = 0;
  int32_t kv_head_num = 0;
  int32_t head_dim = 0;
  int32_t rotary_dim = 0;
  int32_t full_attention_interval = 0;

  int32_t linear_num_k_heads = 0;
  int32_t linear_num_v_heads = 0;
  int32_t linear_k_head_dim = 0;
  int32_t linear_v_head_dim = 0;
  int32_t conv_kernel_size = 0;

  bool tie_word_embeddings = false;
  float rope_theta = 0.f;
  float rms_norm_eps = 0.f;

  // ---- derived: full attention ----
  int32_t q_dim = 0;         // head_num * head_dim
  int32_t q_proj_out = 0;    // q_dim * 2, second half is the output gate
  int32_t kv_dim = 0;        // kv_head_num * head_dim
  int32_t kv_mul = 0;        // head_num / kv_head_num
  int32_t full_layer_num = 0;  // how many layers own a KV cache

  // ---- derived: linear attention ----
  int32_t linear_k_dim = 0;   // linear_num_k_heads * linear_k_head_dim
  int32_t linear_v_dim = 0;   // linear_num_v_heads * linear_v_head_dim
  int32_t conv_dim = 0;       // linear_k_dim * 2 + linear_v_dim
  int32_t v_per_k = 0;        // linear_num_v_heads / linear_num_k_heads
  int32_t state_size = 0;     // per layer: num_v_heads * k_head_dim * v_head_dim
  int32_t conv_state_size = 0;  // per layer: conv_dim * (conv_kernel_size - 1)
  int32_t linear_layer_num = 0;

  Qwen35LayerType layer_type(int32_t layer_idx) const {
    const bool is_full = ((layer_idx + 1) % full_attention_interval) == 0;
    return is_full ? Qwen35LayerType::kFullAttention : Qwen35LayerType::kLinearAttention;
  }

  // Index of this layer among layers of its own kind. Full-attention layers use
  // it to address the KV cache, linear layers to address the recurrent state,
  // so neither wastes memory on the other's slots.
  int32_t type_local_idx(int32_t layer_idx) const {
    const int32_t full_seen = (layer_idx + 1) / full_attention_interval;
    if (layer_type(layer_idx) == Qwen35LayerType::kFullAttention) {
      return full_seen - 1;
    }
    return layer_idx - full_seen;
  }

  void derive() {
    q_dim = head_num * head_dim;
    q_proj_out = q_dim * 2;
    kv_dim = kv_head_num * head_dim;
    kv_mul = kv_head_num ? head_num / kv_head_num : 0;

    linear_k_dim = linear_num_k_heads * linear_k_head_dim;
    linear_v_dim = linear_num_v_heads * linear_v_head_dim;
    conv_dim = linear_k_dim * 2 + linear_v_dim;
    v_per_k = linear_num_k_heads ? linear_num_v_heads / linear_num_k_heads : 0;
    state_size = linear_num_v_heads * linear_k_head_dim * linear_v_head_dim;
    conv_state_size = conv_dim * (conv_kernel_size - 1);

    full_layer_num = full_attention_interval > 0 ? layer_num / full_attention_interval : 0;
    linear_layer_num = layer_num - full_layer_num;
  }
};

}  // namespace model
#endif  // KUIPER_INCLUDE_MODEL_QWEN35_CONFIG_H_
