#ifndef KUIPER_INCLUDE_OP_QWEN35_OPS_H_
#define KUIPER_INCLUDE_OP_QWEN35_OPS_H_
#include "layer.h"

// Layer wrappers for the Qwen3.5-only kernels. The existing RmsNormLayer,
// RoPELayer, MultiHeadAttention and SwiGLULayer are left untouched: Qwen3.5's
// variants differ in ways (no weight, partial rotation, gating, recurrent state)
// that would have meant changing their semantics for every other model.
namespace op {

// Per-row L2 normalisation, no weight. Applied to the GDN query/key.
// forward(in, out); rows are `dim` wide and inferred from the tensor size.
class L2NormLayer : public Layer {
 public:
  explicit L2NormLayer(base::DeviceType device_type, int32_t dim, float eps);

  // Declaring forward() would otherwise hide op::Layer's multi-argument
  // overloads, which is how callers pass inputs and outputs.
  using Layer::forward;

  base::Status check() const override;
  base::Status forward() override;

 private:
  int32_t dim_ = 0;
  float eps_ = 0.f;
};

// Qwen3.5's zero-centered RMSNorm:
//   out = rmsnorm(in) * (1 + weight)
//
// forward(in, out); rows are `dim` wide and inferred from the tensor size. The
// same implementation serves decoder/final norms (one row) and QK-norm (one
// row per head). It is separate from the existing RmsNormLayer because that
// layer uses the Llama/Qwen2 convention `rmsnorm(in) * weight` and has a
// compile-time epsilon.
class ZeroCenteredRMSNormLayer : public LayerParam {
 public:
  explicit ZeroCenteredRMSNormLayer(base::DeviceType device_type, int32_t dim, float eps);

  using Layer::forward;

  base::Status check() const override;
  base::Status forward() override;

 private:
  int32_t dim_ = 0;
  float eps_ = 0.f;
};

// out = rmsnorm(in) * weight * silu(gate).
// forward(in, gate, out); weight index 0 is [dim].
class GatedRMSNormLayer : public LayerParam {
 public:
  explicit GatedRMSNormLayer(base::DeviceType device_type, int32_t dim, float eps);

  // Declaring forward() would otherwise hide op::Layer's multi-argument
  // overloads, which is how callers pass inputs and outputs.
  using Layer::forward;

  base::Status check() const override;
  base::Status forward() override;

 private:
  int32_t dim_ = 0;
  float eps_ = 0.f;
};

// Depthwise causal conv1d, one decode step, silu fused on the output.
// forward(in, state, out); weight index 0 is [dim, k]. `state` is [dim, k-1] and
// is updated in place, so it is passed as an input tensor rather than owned here
// (the model keeps one state buffer per linear layer).
class CausalConv1DLayer : public LayerParam {
 public:
  explicit CausalConv1DLayer(base::DeviceType device_type, int32_t dim, int32_t kernel_size);

  // Declaring forward() would otherwise hide op::Layer's multi-argument
  // overloads, which is how callers pass inputs and outputs.
  using Layer::forward;

  base::Status check() const override;
  base::Status forward() override;

 private:
  int32_t dim_ = 0;
  int32_t kernel_size_ = 0;
};

// One decode step of the gated delta rule.
// forward(q, k, v, g, beta, state, out) -- 6 inputs, 1 output.
// q/k are expected already L2-normalised; the 1/sqrt(k_head_dim) query scale is
// applied inside. `state` is [num_v_heads, k_head_dim, v_head_dim], in place.
class GatedDeltaLayer : public Layer {
 public:
  explicit GatedDeltaLayer(base::DeviceType device_type, int32_t num_k_heads, int32_t num_v_heads,
                           int32_t k_head_dim, int32_t v_head_dim);

  // Declaring forward() would otherwise hide op::Layer's multi-argument
  // overloads, which is how callers pass inputs and outputs.
  using Layer::forward;

  base::Status check() const override;
  base::Status forward() override;

 private:
  int32_t num_k_heads_ = 0;
  int32_t num_v_heads_ = 0;
  int32_t k_head_dim_ = 0;
  int32_t v_head_dim_ = 0;
};

// RoPE over the leading rotary_dim of each head, rotate-half pairing.
// forward(q, k, pos, sin_cache, cos_cache) -- q and k are updated in place.
class RoPEPartialLayer : public Layer {
 public:
  explicit RoPEPartialLayer(base::DeviceType device_type, int32_t num_q_heads,
                            int32_t num_k_heads, int32_t head_dim, int32_t rotary_dim);

  // Declaring forward() would otherwise hide op::Layer's multi-argument
  // overloads, which is how callers pass inputs and outputs.
  using Layer::forward;

  base::Status check() const override;
  base::Status forward() override;

 private:
  int32_t num_q_heads_ = 0;
  int32_t num_k_heads_ = 0;
  int32_t head_dim_ = 0;
  int32_t rotary_dim_ = 0;
};

// Splits a per-head interleaved [num_heads, width*2] tensor into two contiguous
// [num_heads, width] tensors. forward(in, first, second) -- note both outputs.
class SplitHeadInterleavedLayer : public Layer {
 public:
  explicit SplitHeadInterleavedLayer(base::DeviceType device_type, int32_t num_heads,
                                     int32_t width);

  using Layer::forward;

  base::Status check() const override;
  base::Status forward() override;

 private:
  int32_t num_heads_ = 0;
  int32_t width_ = 0;
};

// forward(in, out): elementwise sigmoid. Used for GDN beta.
class SigmoidLayer : public Layer {
 public:
  explicit SigmoidLayer(base::DeviceType device_type);

  // Declaring forward() would otherwise hide op::Layer's multi-argument
  // overloads, which is how callers pass inputs and outputs.
  using Layer::forward;

  base::Status check() const override;
  base::Status forward() override;
};

// forward(a, b, out): elementwise product. Used for the full-attention output
// gate, attn_out *= sigmoid(gate).
class MulLayer : public Layer {
 public:
  explicit MulLayer(base::DeviceType device_type);

  // Declaring forward() would otherwise hide op::Layer's multi-argument
  // overloads, which is how callers pass inputs and outputs.
  using Layer::forward;

  base::Status check() const override;
  base::Status forward() override;
};

// g = -exp(A_log) * softplus(a + dt_bias), one value per v-head.
// forward(a, out); weight 0 is A_log [n], weight 1 is dt_bias [n].
class SoftplusDecayLayer : public LayerParam {
 public:
  explicit SoftplusDecayLayer(base::DeviceType device_type, int32_t num_v_heads);

  // Declaring forward() would otherwise hide op::Layer's multi-argument
  // overloads, which is how callers pass inputs and outputs.
  using Layer::forward;

  base::Status check() const override;
  base::Status forward() override;

 private:
  int32_t num_v_heads_ = 0;
};

}  // namespace op
#endif  // KUIPER_INCLUDE_OP_QWEN35_OPS_H_
