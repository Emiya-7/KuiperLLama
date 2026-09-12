#include "matmul_kernel.h"
#include "../kernels_interface.h"
#include "base/base.h"
#include "base/bfloat16.h"
namespace kernel {
void matmul_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, float scale,
                       const CudaConfig* config) {
  UNUSED(config);
  CHECK(input.is_empty() == false);
  CHECK(weight.is_empty() == false);
  CHECK(output.is_empty() == false);
  CHECK(input.device_type() == base::DeviceType::kDeviceCPU);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCPU);
  CHECK(output.device_type() == base::DeviceType::kDeviceCPU);

  const float* input_ptr = input.ptr<float>();
  const float* output_ptr = output.ptr<float>();

  int32_t batch_size = 1;
  int32_t input_size = 1;
  if (input.dims_size() == 2) {
    batch_size = input.get_dim(0);
    input_size = input.get_dim(1);
  } else if (input.dims_size() == 1) {
    input_size = input.get_dim(0);
  } else {
    LOG(FATAL) << "The input tensor has a wrong dim size.";
  }

  CHECK_EQ(weight.dims_size(), 2);
  const int32_t wei_dim0 = weight.get_dim(0);
  const int32_t wei_dim1 = weight.get_dim(1);
  CHECK_EQ(input_size, wei_dim1);

  CHECK_EQ(output.size(), static_cast<size_t>(wei_dim0) * batch_size);
  if (weight.data_type() == base::DataType::kDataTypeFp32) {
    const float* weight_ptr = weight.ptr<float>();
    // Row-major [N,M], [K,M] and [N,K] buffers are the transposes of the
    // column-major Armadillo views below.
    arma::fmat input_transposed(const_cast<float*>(input_ptr), input_size, batch_size, false,
                                true);
    arma::fmat weight_transposed(const_cast<float*>(weight_ptr), input_size, wei_dim0, false,
                                 true);
    arma::fmat output_transposed(const_cast<float*>(output_ptr), wei_dim0, batch_size, false,
                                 true);
    output_transposed = (weight_transposed.t() * input_transposed) * scale;
    return;
  }

  CHECK(weight.data_type() == base::DataType::kDataTypeBf16);
  const uint16_t* weight_ptr = weight.ptr<uint16_t>();
  float* mutable_output = const_cast<float*>(output_ptr);
  for (int32_t batch = 0; batch < batch_size; ++batch) {
    const float* input_row = input_ptr + static_cast<size_t>(batch) * input_size;
    for (int32_t row = 0; row < wei_dim0; ++row) {
      const uint16_t* weight_row = weight_ptr + static_cast<size_t>(row) * wei_dim1;
      float sum = 0.f;
      for (int32_t i = 0; i < input_size; ++i) {
        sum += input_row[i] * base::bfloat16_to_float(weight_row[i]);
      }
      mutable_output[static_cast<size_t>(batch) * wei_dim0 + row] = sum * scale;
    }
  }
}
}  // namespace kernel
