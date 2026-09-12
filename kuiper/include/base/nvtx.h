#ifndef KUIPER_INCLUDE_BASE_NVTX_H_
#define KUIPER_INCLUDE_BASE_NVTX_H_

#ifdef KUIPER_ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace base {

// Lightweight scoped ranges shared by the model and benchmark executable.
// Keeping the domain stable makes Nsight Compute filters reproducible:
//   --nvtx --nvtx-include 'qwen35@prefill/'
class ScopedNvtxRange {
 public:
  explicit ScopedNvtxRange(const char* name) {
#ifdef KUIPER_ENABLE_NVTX
    nvtxEventAttributes_t event{};
    event.version = NVTX_VERSION;
    event.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    event.messageType = NVTX_MESSAGE_TYPE_ASCII;
    event.message.ascii = name;
    nvtxDomainRangePushEx(domain(), &event);
#else
    (void)name;
#endif
  }

  ~ScopedNvtxRange() {
#ifdef KUIPER_ENABLE_NVTX
    nvtxDomainRangePop(domain());
#endif
  }

  ScopedNvtxRange(const ScopedNvtxRange&) = delete;
  ScopedNvtxRange& operator=(const ScopedNvtxRange&) = delete;

 private:
#ifdef KUIPER_ENABLE_NVTX
  static nvtxDomainHandle_t domain() {
    static nvtxDomainHandle_t handle = nvtxDomainCreateA("qwen35");
    return handle;
  }
#endif
};

}  // namespace base

#endif  // KUIPER_INCLUDE_BASE_NVTX_H_
