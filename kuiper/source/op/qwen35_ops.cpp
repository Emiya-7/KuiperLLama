#include "op/qwen35_ops.h"
#include <cuda_runtime_api.h>
#include "kernels/cpu/qwen35_kernel.h"
#include "kernels/cuda/qwen35_kernel.cuh"

namespace op {

namespace {
// These kernels take raw pointers, so dispatch is a two-way branch here rather
// than through kernels_interface.h's function-pointer tables.
bool is_cuda(base::DeviceType t) { return t == base::DeviceType::kDeviceCUDA; }

base::Status expect_device(const tensor::Tensor& t, base::DeviceType want, const char* what) {
  if (t.is_empty()) {
    return base::error::InvalidArgument(std::string(what) + " is empty.");
  }
  if (t.device_type() != want) {
    return base::error::InvalidArgument(std::string(what) + " is on the wrong device.");
  }
  return base::error::Success();
}
}  // namespace

// ------------------------------------------------------------------- L2Norm
L2NormLayer::L2NormLayer(base::DeviceType device_type, int32_t dim, float eps)
    : Layer(device_type, LayerType::kLayerL2Norm, "L2Norm"), dim_(dim), eps_(eps) {
  reset_input_size(1);
  reset_output_size(1);
}

base::Status L2NormLayer::check() const {
  auto st = expect_device(get_input(0), device_type_, "L2Norm input");
  if (!st) return st;
  st = expect_device(get_output(0), device_type_, "L2Norm output");
  if (!st) return st;
  if (static_cast<int32_t>(get_input(0).size()) % dim_ != 0) {
    return base::error::InvalidArgument("L2Norm input size is not a multiple of dim.");
  }
  return base::error::Success();
}

base::Status L2NormLayer::forward() {
  auto status = check();
  if (!status) return status;
  auto in = get_input(0);
  auto out = get_output(0);
  const int32_t n = static_cast<int32_t>(in.size()) / dim_;
  if (is_cuda(device_type_)) {
    CHECK(cuda_config_ != nullptr);
    kernel::l2norm_cu(in.ptr<float>(), const_cast<float*>(out.ptr<float>()), n, dim_, eps_,
                      cuda_config_->stream);
  } else {
    kernel::l2norm_cpu(in.ptr<float>(), const_cast<float*>(out.ptr<float>()), n, dim_, eps_);
  }
  return base::error::Success();
}

// ----------------------------------------------- ZeroCenteredRMSNorm
ZeroCenteredRMSNormLayer::ZeroCenteredRMSNormLayer(base::DeviceType device_type, int32_t dim,
                                                   float eps)
    : LayerParam(device_type, LayerType::kLayerZeroCenteredRMSNorm, false,
                 "ZeroCenteredRMSNorm"),
      dim_(dim),
      eps_(eps) {
  reset_input_size(1);
  reset_output_size(1);
  reset_weight_size(1);
}

base::Status ZeroCenteredRMSNormLayer::check() const {
  auto st = expect_device(get_input(0), device_type_, "ZeroCenteredRMSNorm input");
  if (!st) return st;
  st = expect_device(get_output(0), device_type_, "ZeroCenteredRMSNorm output");
  if (!st) return st;
  st = expect_device(get_weight(0), device_type_, "ZeroCenteredRMSNorm weight");
  if (!st) return st;
  if (dim_ <= 0 || eps_ < 0.f) {
    return base::error::InvalidArgument(
        "ZeroCenteredRMSNorm requires dim > 0 and eps >= 0.");
  }
  if (get_input(0).data_type() != base::DataType::kDataTypeFp32 ||
      get_output(0).data_type() != base::DataType::kDataTypeFp32 ||
      get_weight(0).data_type() != base::DataType::kDataTypeFp32) {
    return base::error::InvalidArgument("ZeroCenteredRMSNorm tensors must be fp32.");
  }
  if (static_cast<int32_t>(get_weight(0).size()) != dim_) {
    return base::error::InvalidArgument("ZeroCenteredRMSNorm weight size != dim.");
  }
  if (static_cast<int32_t>(get_input(0).size()) % dim_ != 0) {
    return base::error::InvalidArgument(
        "ZeroCenteredRMSNorm input size is not a multiple of dim.");
  }
  if (get_output(0).size() != get_input(0).size()) {
    return base::error::InvalidArgument(
        "ZeroCenteredRMSNorm output size differs from input size.");
  }
  return base::error::Success();
}

base::Status ZeroCenteredRMSNormLayer::forward() {
  auto status = check();
  if (!status) return status;
  auto in = get_input(0);
  auto weight = get_weight(0);
  auto out = get_output(0);
  const int32_t n = static_cast<int32_t>(in.size()) / dim_;
  if (is_cuda(device_type_)) {
    CHECK(cuda_config_ != nullptr);
    kernel::zero_centered_rmsnorm_cu(in.ptr<float>(), weight.ptr<float>(),
                                     const_cast<float*>(out.ptr<float>()), n, dim_, eps_,
                                     cuda_config_->stream);
  } else {
    kernel::zero_centered_rmsnorm_cpu(in.ptr<float>(), weight.ptr<float>(),
                                      const_cast<float*>(out.ptr<float>()), n, dim_, eps_);
  }
  return base::error::Success();
}

// ------------------------------------------------------------ GatedRMSNorm
GatedRMSNormLayer::GatedRMSNormLayer(base::DeviceType device_type, int32_t dim, float eps)
    : LayerParam(device_type, LayerType::kLayerGatedRMSNorm, false, "GatedRMSNorm"),
      dim_(dim),
      eps_(eps) {
  reset_input_size(2);  // in, gate
  reset_output_size(1);
  reset_weight_size(1);
}

base::Status GatedRMSNormLayer::check() const {
  for (int i = 0; i < 2; ++i) {
    auto st = expect_device(get_input(i), device_type_, "GatedRMSNorm input");
    if (!st) return st;
  }
  auto st = expect_device(get_output(0), device_type_, "GatedRMSNorm output");
  if (!st) return st;
  if (get_input(0).size() != get_input(1).size()) {
    return base::error::InvalidArgument("GatedRMSNorm gate size differs from input size.");
  }
  if (static_cast<int32_t>(get_weight(0).size()) != dim_) {
    return base::error::InvalidArgument("GatedRMSNorm weight size != dim.");
  }
  return base::error::Success();
}

base::Status GatedRMSNormLayer::forward() {
  auto status = check();
  if (!status) return status;
  auto in = get_input(0);
  auto gate = get_input(1);
  auto weight = get_weight(0);
  auto out = get_output(0);
  const int32_t n = static_cast<int32_t>(in.size()) / dim_;
  if (is_cuda(device_type_)) {
    CHECK(cuda_config_ != nullptr);
    kernel::gated_rmsnorm_cu(in.ptr<float>(), gate.ptr<float>(), weight.ptr<float>(),
                             const_cast<float*>(out.ptr<float>()), n, dim_, eps_,
                             cuda_config_->stream);
  } else {
    kernel::gated_rmsnorm_cpu(in.ptr<float>(), gate.ptr<float>(), weight.ptr<float>(),
                              const_cast<float*>(out.ptr<float>()), n, dim_, eps_);
  }
  return base::error::Success();
}

// ------------------------------------------------------------ CausalConv1D
CausalConv1DLayer::CausalConv1DLayer(base::DeviceType device_type, int32_t dim,
                                     int32_t kernel_size)
    : LayerParam(device_type, LayerType::kLayerCausalConv1D, false, "CausalConv1D"),
      dim_(dim),
      kernel_size_(kernel_size) {
  reset_input_size(2);  // in, state (mutated in place)
  reset_output_size(1);
  reset_weight_size(1);
}

base::Status CausalConv1DLayer::check() const {
  auto st = expect_device(get_input(0), device_type_, "Conv1D input");
  if (!st) return st;
  st = expect_device(get_input(1), device_type_, "Conv1D state");
  if (!st) return st;
  st = expect_device(get_output(0), device_type_, "Conv1D output");
  if (!st) return st;
  if (static_cast<int32_t>(get_input(0).size()) != dim_) {
    return base::error::InvalidArgument("Conv1D input size != dim.");
  }
  if (static_cast<int32_t>(get_input(1).size()) != dim_ * (kernel_size_ - 1)) {
    return base::error::InvalidArgument("Conv1D state size != dim * (k-1).");
  }
  if (static_cast<int32_t>(get_weight(0).size()) != dim_ * kernel_size_) {
    return base::error::InvalidArgument("Conv1D weight size != dim * k.");
  }
  return base::error::Success();
}

base::Status CausalConv1DLayer::forward() {
  auto status = check();
  if (!status) return status;
  auto in = get_input(0);
  auto state = get_input(1);
  auto weight = get_weight(0);
  auto out = get_output(0);
  if (is_cuda(device_type_)) {
    CHECK(cuda_config_ != nullptr);
    kernel::causal_conv1d_decode_cu(in.ptr<float>(), const_cast<float*>(state.ptr<float>()),
                                    weight.ptr<float>(), const_cast<float*>(out.ptr<float>()),
                                    dim_, kernel_size_, cuda_config_->stream);
  } else {
    kernel::causal_conv1d_decode_cpu(in.ptr<float>(), const_cast<float*>(state.ptr<float>()),
                                     weight.ptr<float>(), const_cast<float*>(out.ptr<float>()),
                                     dim_, kernel_size_);
  }
  return base::error::Success();
}

// -------------------------------------------------------------- GatedDelta
GatedDeltaLayer::GatedDeltaLayer(base::DeviceType device_type, int32_t num_k_heads,
                                 int32_t num_v_heads, int32_t k_head_dim, int32_t v_head_dim)
    : Layer(device_type, LayerType::kLayerGatedDelta, "GatedDelta"),
      num_k_heads_(num_k_heads),
      num_v_heads_(num_v_heads),
      k_head_dim_(k_head_dim),
      v_head_dim_(v_head_dim) {
  reset_input_size(6);  // q, k, v, g, beta, state
  reset_output_size(1);
}

base::Status GatedDeltaLayer::check() const {
  static const char* kNames[6] = {"q", "k", "v", "g", "beta", "state"};
  for (int i = 0; i < 6; ++i) {
    auto st = expect_device(get_input(i), device_type_, kNames[i]);
    if (!st) return st;
  }
  auto st = expect_device(get_output(0), device_type_, "GatedDelta output");
  if (!st) return st;
  if (num_k_heads_ <= 0 || num_v_heads_ <= 0 || num_v_heads_ % num_k_heads_ != 0) {
    return base::error::InvalidArgument(
        "num_k_heads and num_v_heads must be positive, with v heads grouped by k heads.");
  }
  if (k_head_dim_ <= 0 || v_head_dim_ <= 0) {
    return base::error::InvalidArgument("GatedDelta head dimensions must be positive.");
  }
  const int64_t k_dim = static_cast<int64_t>(num_k_heads_) * k_head_dim_;
  const int64_t v_dim = static_cast<int64_t>(num_v_heads_) * v_head_dim_;
  if (static_cast<int64_t>(get_input(0).size()) != k_dim ||
      static_cast<int64_t>(get_input(1).size()) != k_dim) {
    return base::error::InvalidArgument("GatedDelta q/k size != num_k_heads * k_head_dim.");
  }
  if (static_cast<int64_t>(get_input(2).size()) != v_dim ||
      static_cast<int64_t>(get_output(0).size()) != v_dim) {
    return base::error::InvalidArgument("GatedDelta v/out size != num_v_heads * v_head_dim.");
  }
  if (static_cast<int32_t>(get_input(3).size()) != num_v_heads_ ||
      static_cast<int32_t>(get_input(4).size()) != num_v_heads_) {
    return base::error::InvalidArgument("GatedDelta g/beta size != num_v_heads.");
  }
  const int64_t want_state =
      static_cast<int64_t>(num_v_heads_) * k_head_dim_ * v_head_dim_;
  if (static_cast<int64_t>(get_input(5).size()) != want_state) {
    return base::error::InvalidArgument("GatedDelta state size mismatch.");
  }
  return base::error::Success();
}

base::Status GatedDeltaLayer::forward() {
  auto status = check();
  if (!status) return status;
  auto q = get_input(0);
  auto k = get_input(1);
  auto v = get_input(2);
  auto g = get_input(3);
  auto beta = get_input(4);
  auto state = get_input(5);
  auto out = get_output(0);
  if (is_cuda(device_type_)) {
    CHECK(cuda_config_ != nullptr);
    kernel::gated_delta_step_cu(q.ptr<float>(), k.ptr<float>(), v.ptr<float>(), g.ptr<float>(),
                                beta.ptr<float>(), const_cast<float*>(state.ptr<float>()),
                                const_cast<float*>(out.ptr<float>()), num_k_heads_, num_v_heads_,
                                k_head_dim_, v_head_dim_, cuda_config_->stream);
  } else {
    kernel::gated_delta_step_cpu(q.ptr<float>(), k.ptr<float>(), v.ptr<float>(), g.ptr<float>(),
                                 beta.ptr<float>(), const_cast<float*>(state.ptr<float>()),
                                 const_cast<float*>(out.ptr<float>()), num_k_heads_, num_v_heads_,
                                 k_head_dim_, v_head_dim_);
  }
  return base::error::Success();
}

// ------------------------------------------------------------- RoPEPartial
RoPEPartialLayer::RoPEPartialLayer(base::DeviceType device_type, int32_t num_q_heads,
                                   int32_t num_k_heads, int32_t head_dim, int32_t rotary_dim)
    : Layer(device_type, LayerType::kLayerRoPePartial, "RoPEPartial"),
      num_q_heads_(num_q_heads),
      num_k_heads_(num_k_heads),
      head_dim_(head_dim),
      rotary_dim_(rotary_dim) {
  reset_input_size(5);  // q, k, pos, sin_cache, cos_cache
  reset_output_size(1);  // unused: q and k are rotated in place
}

base::Status RoPEPartialLayer::check() const {
  auto st = expect_device(get_input(0), device_type_, "RoPEPartial q");
  if (!st) return st;
  st = expect_device(get_input(1), device_type_, "RoPEPartial k");
  if (!st) return st;
  // pos lives on the CPU: the kernel takes it as a scalar launch argument.
  if (get_input(2).is_empty()) {
    return base::error::InvalidArgument("RoPEPartial pos is empty.");
  }
  st = expect_device(get_input(3), device_type_, "RoPEPartial sin cache");
  if (!st) return st;
  st = expect_device(get_input(4), device_type_, "RoPEPartial cos cache");
  if (!st) return st;
  if (rotary_dim_ <= 0 || rotary_dim_ % 2 != 0 || rotary_dim_ > head_dim_) {
    return base::error::InvalidArgument("RoPEPartial rotary_dim must be even and <= head_dim.");
  }
  if (static_cast<int32_t>(get_input(0).size()) != num_q_heads_ * head_dim_) {
    return base::error::InvalidArgument("RoPEPartial q size != num_q_heads * head_dim.");
  }
  if (static_cast<int32_t>(get_input(1).size()) != num_k_heads_ * head_dim_) {
    return base::error::InvalidArgument("RoPEPartial k size != num_k_heads * head_dim.");
  }
  return base::error::Success();
}

base::Status RoPEPartialLayer::forward() {
  auto status = check();
  if (!status) return status;
  auto q = get_input(0);
  auto k = get_input(1);
  const int32_t pos = get_input(2).index<int32_t>(0);
  auto sin_cache = get_input(3);
  auto cos_cache = get_input(4);
  if (is_cuda(device_type_)) {
    CHECK(cuda_config_ != nullptr);
    kernel::rope_partial_cu(sin_cache.ptr<float>(), cos_cache.ptr<float>(),
                            const_cast<float*>(q.ptr<float>()),
                            const_cast<float*>(k.ptr<float>()), pos, num_q_heads_, num_k_heads_,
                            head_dim_, rotary_dim_, cuda_config_->stream);
  } else {
    kernel::rope_partial_cpu(sin_cache.ptr<float>(), cos_cache.ptr<float>(),
                             const_cast<float*>(q.ptr<float>()),
                             const_cast<float*>(k.ptr<float>()), pos, num_q_heads_, num_k_heads_,
                             head_dim_, rotary_dim_);
  }
  return base::error::Success();
}

// --------------------------------------------------- SplitHeadInterleaved
SplitHeadInterleavedLayer::SplitHeadInterleavedLayer(base::DeviceType device_type,
                                                     int32_t num_heads, int32_t width)
    : Layer(device_type, LayerType::kLayerSplitHeadInterleaved, "SplitHeadInterleaved"),
      num_heads_(num_heads),
      width_(width) {
  reset_input_size(1);
  reset_output_size(2);
}

base::Status SplitHeadInterleavedLayer::check() const {
  auto st = expect_device(get_input(0), device_type_, "SplitHeadInterleaved input");
  if (!st) return st;
  for (int i = 0; i < 2; ++i) {
    st = expect_device(get_output(i), device_type_, "SplitHeadInterleaved output");
    if (!st) return st;
    if (static_cast<int32_t>(get_output(i).size()) != num_heads_ * width_) {
      return base::error::InvalidArgument("SplitHeadInterleaved output size != num_heads * width.");
    }
  }
  if (static_cast<int32_t>(get_input(0).size()) != num_heads_ * width_ * 2) {
    return base::error::InvalidArgument("SplitHeadInterleaved input size != num_heads*width*2.");
  }
  return base::error::Success();
}

base::Status SplitHeadInterleavedLayer::forward() {
  auto status = check();
  if (!status) return status;
  auto in = get_input(0);
  auto first = get_output(0);
  auto second = get_output(1);
  if (is_cuda(device_type_)) {
    CHECK(cuda_config_ != nullptr);
    kernel::split_head_interleaved_cu(in.ptr<float>(),
                                      const_cast<float*>(first.ptr<float>()),
                                      const_cast<float*>(second.ptr<float>()), num_heads_, width_,
                                      cuda_config_->stream);
  } else {
    kernel::split_head_interleaved_cpu(in.ptr<float>(),
                                       const_cast<float*>(first.ptr<float>()),
                                       const_cast<float*>(second.ptr<float>()), num_heads_,
                                       width_);
  }
  return base::error::Success();
}

// ----------------------------------------------------------------- Sigmoid
SigmoidLayer::SigmoidLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerSigmoid, "Sigmoid") {
  reset_input_size(1);
  reset_output_size(1);
}

base::Status SigmoidLayer::check() const {
  auto st = expect_device(get_input(0), device_type_, "Sigmoid input");
  if (!st) return st;
  st = expect_device(get_output(0), device_type_, "Sigmoid output");
  if (!st) return st;
  if (get_input(0).size() != get_output(0).size()) {
    return base::error::InvalidArgument("Sigmoid input and output sizes differ.");
  }
  return base::error::Success();
}

base::Status SigmoidLayer::forward() {
  auto status = check();
  if (!status) return status;
  auto in = get_input(0);
  auto out = get_output(0);
  const int32_t n = static_cast<int32_t>(in.size());
  if (is_cuda(device_type_)) {
    CHECK(cuda_config_ != nullptr);
    kernel::sigmoid_cu(in.ptr<float>(), const_cast<float*>(out.ptr<float>()), n,
                       cuda_config_->stream);
  } else {
    kernel::sigmoid_cpu(in.ptr<float>(), const_cast<float*>(out.ptr<float>()), n);
  }
  return base::error::Success();
}

// --------------------------------------------------------------------- Mul
MulLayer::MulLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerMul, "Mul") {
  reset_input_size(2);
  reset_output_size(1);
}

base::Status MulLayer::check() const {
  for (int i = 0; i < 2; ++i) {
    auto st = expect_device(get_input(i), device_type_, "Mul input");
    if (!st) return st;
  }
  auto st = expect_device(get_output(0), device_type_, "Mul output");
  if (!st) return st;
  if (get_input(0).size() != get_input(1).size() ||
      get_input(0).size() != get_output(0).size()) {
    return base::error::InvalidArgument("Mul operand sizes differ.");
  }
  return base::error::Success();
}

base::Status MulLayer::forward() {
  auto status = check();
  if (!status) return status;
  auto a = get_input(0);
  auto b = get_input(1);
  auto out = get_output(0);
  const int32_t n = static_cast<int32_t>(a.size());
  if (is_cuda(device_type_)) {
    CHECK(cuda_config_ != nullptr);
    kernel::mul_cu(a.ptr<float>(), b.ptr<float>(), const_cast<float*>(out.ptr<float>()), n,
                   cuda_config_->stream);
  } else {
    kernel::mul_cpu(a.ptr<float>(), b.ptr<float>(), const_cast<float*>(out.ptr<float>()), n);
  }
  return base::error::Success();
}

// ---------------------------------------------------------- SoftplusDecay
SoftplusDecayLayer::SoftplusDecayLayer(base::DeviceType device_type, int32_t num_v_heads)
    : LayerParam(device_type, LayerType::kLayerSoftplusDecay, false, "SoftplusDecay"),
      num_v_heads_(num_v_heads) {
  reset_input_size(1);
  reset_output_size(1);
  reset_weight_size(2);  // A_log, dt_bias
}

base::Status SoftplusDecayLayer::check() const {
  auto st = expect_device(get_input(0), device_type_, "SoftplusDecay input");
  if (!st) return st;
  st = expect_device(get_output(0), device_type_, "SoftplusDecay output");
  if (!st) return st;
  if (static_cast<int32_t>(get_input(0).size()) != num_v_heads_ ||
      static_cast<int32_t>(get_output(0).size()) != num_v_heads_) {
    return base::error::InvalidArgument("SoftplusDecay in/out size != num_v_heads.");
  }
  for (int i = 0; i < 2; ++i) {
    if (static_cast<int32_t>(get_weight(i).size()) != num_v_heads_) {
      return base::error::InvalidArgument("SoftplusDecay weight size != num_v_heads.");
    }
  }
  return base::error::Success();
}

base::Status SoftplusDecayLayer::forward() {
  auto status = check();
  if (!status) return status;
  auto a = get_input(0);
  auto A_log = get_weight(0);
  auto dt_bias = get_weight(1);
  auto out = get_output(0);
  if (is_cuda(device_type_)) {
    CHECK(cuda_config_ != nullptr);
    kernel::softplus_decay_cu(a.ptr<float>(), A_log.ptr<float>(), dt_bias.ptr<float>(),
                              const_cast<float*>(out.ptr<float>()), num_v_heads_,
                              cuda_config_->stream);
  } else {
    kernel::softplus_decay_cpu(a.ptr<float>(), A_log.ptr<float>(), dt_bias.ptr<float>(),
                               const_cast<float*>(out.ptr<float>()), num_v_heads_);
  }
  return base::error::Success();
}

}  // namespace op
