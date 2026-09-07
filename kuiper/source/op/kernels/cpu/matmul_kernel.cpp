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

  int32_t in_dim1 = 1;
  int32_t in_dim0 = 1;
  if (input.dims_size() == 2) {
    in_dim0 = input.get_dim(0);
    in_dim1 = input.get_dim(1);
  } else if (input.dims_size() == 1) {
    in_dim0 = input.get_dim(0);
  } else {
    LOG(FATAL) << "The input tensor has a wrong dim size.";
  }

  CHECK_EQ(weight.dims_size(), 2);
  const int32_t wei_dim0 = weight.get_dim(0);
  const int32_t wei_dim1 = weight.get_dim(1);
  CHECK_EQ(in_dim0, wei_dim1);

  CHECK_EQ(output.size(), wei_dim0 * in_dim1);
  if (weight.data_type() == base::DataType::kDataTypeFp32) {
    const float* weight_ptr = weight.ptr<float>();
    arma::fmat input_mat(const_cast<float*>(input_ptr), in_dim1, in_dim0, false, true);
    arma::fmat weight_mat(const_cast<float*>(weight_ptr), wei_dim1, wei_dim0, false, true);
    arma::fmat output_mat(const_cast<float*>(output_ptr), in_dim1, wei_dim0, false, true);
    output_mat = ((input_mat * weight_mat)) * scale;
    return;
  }

  CHECK(weight.data_type() == base::DataType::kDataTypeBf16);
  const uint16_t* weight_ptr = weight.ptr<uint16_t>();
  float* mutable_output = const_cast<float*>(output_ptr);
  for (int32_t row = 0; row < wei_dim0; ++row) {
    const uint16_t* weight_row = weight_ptr + static_cast<size_t>(row) * wei_dim1;
    for (int32_t column = 0; column < in_dim1; ++column) {
      float sum = 0.f;
      for (int32_t i = 0; i < in_dim0; ++i) {
        sum += input_ptr[static_cast<size_t>(i) * in_dim1 + column] *
               base::bfloat16_to_float(weight_row[i]);
      }
      mutable_output[static_cast<size_t>(row) * in_dim1 + column] = sum * scale;
    }
  }
}
}  // namespace kernel
