#ifndef KUIPER_INCLUDE_BASE_BFLOAT16_H_
#define KUIPER_INCLUDE_BASE_BFLOAT16_H_

#include <cstdint>
#include <cstring>

namespace base {

// BF16 is stored as its raw IEEE-754 upper 16 bits. Activations remain FP32;
// these helpers are for CPU kernels and tests that consume BF16 weights.
inline float bfloat16_to_float(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

inline uint16_t float_to_bfloat16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  // Round to nearest, ties to even, matching common BF16 conversions.
  const uint32_t rounding_bias = 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

}  // namespace base

#endif  // KUIPER_INCLUDE_BASE_BFLOAT16_H_
