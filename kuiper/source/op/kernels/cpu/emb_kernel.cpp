#include "emb_kernel.h"
#include "base/bfloat16.h"
namespace kernel {

void emb_kernel_normal(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, int32_t vocab_size, void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  const int32_t input_num = static_cast<int32_t>(input.size());
  const int32_t weight_dim = weight.get_dim(1);
  CHECK(weight.device_type() == output.device_type());
  CHECK(input.device_type() == base::DeviceType::kDeviceCPU);

  const auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
  for (int32_t i = 0; i < input_num; ++i) {
    int32_t token = *input.ptr<int32_t>(i);
    if (token > vocab_size) {
      LOG(FATAL) << "Token index is greater than vocab size.";
    } else {
      float* dest_ptr = const_cast<float*>(output.ptr<float>(i * weight_dim));
      if (weight.device_type() == base::DeviceType::kDeviceCPU) {
        if (weight.data_type() == base::DataType::kDataTypeFp32) {
          const float* src_ptr = weight.ptr<float>(token * weight_dim);
          allocator->memcpy(src_ptr, dest_ptr, weight_dim * sizeof(float),
                            base::MemcpyKind::kMemcpyCPU2CPU);
        } else {
          CHECK(weight.data_type() == base::DataType::kDataTypeBf16);
          const uint16_t* src_ptr = weight.ptr<uint16_t>(token * weight_dim);
          for (int32_t column = 0; column < weight_dim; ++column) {
            dest_ptr[column] = base::bfloat16_to_float(src_ptr[column]);
          }
        }
      } else {
        LOG(FATAL) << "Unknown device type of weight tensor in the embedding layer.";
      }
    }
  }
}

}  // namespace kernel
