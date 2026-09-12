#ifndef KUIPER_SOURCE_OP_KERNELS_CUDA_CUDA_LAUNCH_CHECK_CUH_
#define KUIPER_SOURCE_OP_KERNELS_CUDA_CUDA_LAUNCH_CHECK_CUH_

#include <cuda_runtime_api.h>
#include <glog/logging.h>

namespace kernel {

// Kernel launches are asynchronous, but launch-configuration and argument
// errors are available immediately. Clear and report that error at the launch
// site so a later synchronization does not misattribute it to another kernel.
inline void check_cuda_kernel_launch(const char* kernel_name) {
  const cudaError_t status = cudaGetLastError();
  CHECK(status == cudaSuccess)
      << kernel_name << " launch failed: " << cudaGetErrorString(status);
}

}  // namespace kernel

#endif  // KUIPER_SOURCE_OP_KERNELS_CUDA_CUDA_LAUNCH_CHECK_CUH_
