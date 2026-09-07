#include "qwen35_kernel.h"
#include <cmath>
#include <vector>

namespace kernel {

void l2norm_cpu(const float* in, float* out, int32_t n, int32_t dim, float eps) {
  for (int32_t r = 0; r < n; ++r) {
    const float* src = in + r * dim;
    float* dst = out + r * dim;
    float sum = 0.f;
    for (int32_t i = 0; i < dim; ++i) {
      sum += src[i] * src[i];
    }
    // rsqrt(sum + eps), matching l2norm in flash-linear-attention: eps is added
    // to the sum of squares, not to the norm.
    const float scale = 1.f / std::sqrt(sum + eps);
    for (int32_t i = 0; i < dim; ++i) {
      dst[i] = src[i] * scale;
    }
  }
}

void split_head_interleaved_cpu(const float* in, float* first, float* second, int32_t num_heads,
                                int32_t width) {
  for (int32_t h = 0; h < num_heads; ++h) {
    const float* src = in + static_cast<int64_t>(h) * width * 2;
    float* d1 = first + static_cast<int64_t>(h) * width;
    float* d2 = second + static_cast<int64_t>(h) * width;
    for (int32_t i = 0; i < width; ++i) {
      d1[i] = src[i];
      d2[i] = src[width + i];
    }
  }
}

void silu_cpu(const float* in, float* out, int32_t n) {
  for (int32_t i = 0; i < n; ++i) {
    out[i] = in[i] / (1.f + std::exp(-in[i]));
  }
}

void sigmoid_cpu(const float* in, float* out, int32_t n) {
  for (int32_t i = 0; i < n; ++i) {
    out[i] = 1.f / (1.f + std::exp(-in[i]));
  }
}

void mul_cpu(const float* a, const float* b, float* out, int32_t n) {
  for (int32_t i = 0; i < n; ++i) {
    out[i] = a[i] * b[i];
  }
}

static inline float softplus(float x) {
  // log1p(exp(x)) with the standard large-x guard, so a+dt_bias well above zero
  // does not overflow exp.
  if (x > 20.f) {
    return x;
  }
  return std::log1p(std::exp(x));
}

void softplus_decay_cpu(const float* a, const float* A_log, const float* dt_bias, float* g,
                        int32_t n) {
  for (int32_t i = 0; i < n; ++i) {
    g[i] = -std::exp(A_log[i]) * softplus(a[i] + dt_bias[i]);
  }
}

void gated_rmsnorm_cpu(const float* in, const float* gate, const float* weight, float* out,
                       int32_t n, int32_t dim, float eps) {
  for (int32_t r = 0; r < n; ++r) {
    const float* src = in + r * dim;
    const float* gsrc = gate + r * dim;
    float* dst = out + r * dim;
    float sum = 0.f;
    for (int32_t i = 0; i < dim; ++i) {
      sum += src[i] * src[i];
    }
    const float scale = 1.f / std::sqrt(sum / static_cast<float>(dim) + eps);
    for (int32_t i = 0; i < dim; ++i) {
      // Norm and weight first, gate second: Qwen3NextRMSNormGated normalises
      // before applying silu(gate), so the gate never affects the variance.
      const float normed = src[i] * scale * weight[i];
      const float gv = gsrc[i];
      dst[i] = normed * (gv / (1.f + std::exp(-gv)));
    }
  }
}

void zero_centered_rmsnorm_cpu(const float* in, const float* weight, float* out, int32_t n,
                               int32_t dim, float eps) {
  for (int32_t r = 0; r < n; ++r) {
    const float* src = in + static_cast<int64_t>(r) * dim;
    float* dst = out + static_cast<int64_t>(r) * dim;
    float sum = 0.f;
    for (int32_t i = 0; i < dim; ++i) {
      sum += src[i] * src[i];
    }
    const float scale = 1.f / std::sqrt(sum / static_cast<float>(dim) + eps);
    for (int32_t i = 0; i < dim; ++i) {
      dst[i] = src[i] * scale * (1.f + weight[i]);
    }
  }
}

void causal_conv1d_decode_cpu(const float* in, float* state, const float* weight, float* out,
                              int32_t dim, int32_t k) {
  const int32_t hist = k - 1;
  for (int32_t c = 0; c < dim; ++c) {
    float* srow = state + c * hist;
    const float* w = weight + c * k;
    // Window is [oldest ... newest, in[c]]; accumulate against the kernel while
    // shifting the history left so the new sample lands at the end.
    float acc = 0.f;
    for (int32_t j = 0; j < hist; ++j) {
      acc += srow[j] * w[j];
    }
    acc += in[c] * w[hist];
    for (int32_t j = 0; j + 1 < hist; ++j) {
      srow[j] = srow[j + 1];
    }
    if (hist > 0) {
      srow[hist - 1] = in[c];
    }
    out[c] = acc / (1.f + std::exp(-acc));  // silu, fused
  }
}

void gated_delta_step_cpu(const float* q, const float* k, const float* v, const float* g,
                          const float* beta, float* state, float* out, int32_t num_k_heads,
                          int32_t num_v_heads, int32_t k_head_dim, int32_t v_head_dim) {
  const int32_t v_per_k = num_v_heads / num_k_heads;
  const float q_scale = 1.f / std::sqrt(static_cast<float>(k_head_dim));
  const int32_t state_stride = k_head_dim * v_head_dim;
  // One scratch row is enough for all heads. A dynamic buffer avoids the old
  // fixed 512-element limit, which silently skipped state/output columns when
  // a future model used v_head_dim > 512.
  std::vector<float> delta(static_cast<size_t>(v_head_dim));

  for (int32_t h = 0; h < num_v_heads; ++h) {
    const int32_t kh = h / v_per_k;
    const float* q_h = q + kh * k_head_dim;
    const float* k_h = k + kh * k_head_dim;
    const float* v_h = v + h * v_head_dim;
    float* S = state + static_cast<int64_t>(h) * state_stride;  // [k_head_dim, v_head_dim]
    float* out_h = out + h * v_head_dim;

    const float decay = std::exp(g[h]);
    const float beta_h = beta[h];

    // Fuse the decay into the read: kv_mem = k^T (S * decay). Writing the decay
    // back separately would touch S twice.
    for (int32_t j = 0; j < v_head_dim; ++j) {
      out_h[j] = 0.f;
    }
    // kv_mem reuses out_h as scratch, then is turned into the real output below.
    for (int32_t i = 0; i < k_head_dim; ++i) {
      const float ki = k_h[i];
      if (ki == 0.f) {
        continue;
      }
      const float* srow = S + static_cast<int64_t>(i) * v_head_dim;
      for (int32_t j = 0; j < v_head_dim; ++j) {
        out_h[j] += ki * srow[j] * decay;
      }
    }
    // delta = (v - kv_mem) * beta. It must remain available while out_h is
    // rewritten as the query read below.
    for (int32_t j = 0; j < v_head_dim; ++j) {
      delta[j] = (v_h[j] - out_h[j]) * beta_h;
    }

    // S = S * decay + outer(k, delta), then out = S^T q in the same sweep.
    for (int32_t j = 0; j < v_head_dim; ++j) {
      out_h[j] = 0.f;
    }
    for (int32_t i = 0; i < k_head_dim; ++i) {
      float* srow = S + static_cast<int64_t>(i) * v_head_dim;
      const float ki = k_h[i];
      const float qi = q_h[i] * q_scale;
      for (int32_t j = 0; j < v_head_dim; ++j) {
        const float s = srow[j] * decay + ki * delta[j];
        srow[j] = s;
        out_h[j] += qi * s;
      }
    }
  }
}

void rope_partial_cache_cpu(float* sin_cache, float* cos_cache, int32_t max_seq_len,
                            int32_t rotary_dim, float theta) {
  const int32_t half = rotary_dim / 2;
  for (int32_t pos = 0; pos < max_seq_len; ++pos) {
    for (int32_t i = 0; i < half; ++i) {
      // inv_freq over rotary_dim, i.e. the partial head slice, not head_dim.
      const float freq =
          1.f / std::pow(theta, static_cast<float>(2 * i) / static_cast<float>(rotary_dim));
      const float val = static_cast<float>(pos) * freq;
      sin_cache[pos * half + i] = std::sin(val);
      cos_cache[pos * half + i] = std::cos(val);
    }
  }
}

// Rotate-half layout: pairs element i with i+half inside the rotary slice, which
// is what HF's rotate_half does. The project's existing rope_kernel_cpu pairs
// (2i, 2i+1) instead, so the two are not interchangeable.
static inline void rope_partial_one(float* vec, const float* sin_row, const float* cos_row,
                                    int32_t num_heads, int32_t head_dim, int32_t rotary_dim) {
  const int32_t half = rotary_dim / 2;
  for (int32_t h = 0; h < num_heads; ++h) {
    float* base = vec + h * head_dim;
    for (int32_t i = 0; i < half; ++i) {
      const float x1 = base[i];
      const float x2 = base[i + half];
      const float c = cos_row[i];
      const float s = sin_row[i];
      base[i] = x1 * c - x2 * s;
      base[i + half] = x2 * c + x1 * s;
    }
    // dims [rotary_dim, head_dim) are left untouched by design.
  }
}

void rope_partial_cpu(const float* sin_cache, const float* cos_cache, float* q, float* k,
                      int32_t pos, int32_t num_q_heads, int32_t num_k_heads, int32_t head_dim,
                      int32_t rotary_dim) {
  const int32_t half = rotary_dim / 2;
  const float* sin_row = sin_cache + static_cast<int64_t>(pos) * half;
  const float* cos_row = cos_cache + static_cast<int64_t>(pos) * half;
  rope_partial_one(q, sin_row, cos_row, num_q_heads, head_dim, rotary_dim);
  rope_partial_one(k, sin_row, cos_row, num_k_heads, head_dim, rotary_dim);
}

}  // namespace kernel
