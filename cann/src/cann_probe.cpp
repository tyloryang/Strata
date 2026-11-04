// strata_cann_probe: Phase 0 self-test for the CANN backend.
//
// Calls only ACL runtime APIs whose signatures were checked against
// /usr/local/Ascend/cann-9.0.0/include/acl/acl_rt.h on the first
// verified target (910B2C, CANN 9.0.0, 2026-11-04):
//   aclInit(const char*), aclrtGetDeviceCount(uint32_t*),
//   aclrtSetDevice(int32_t), aclrtGetDevice(int32_t*), aclFinalize().
// No model is loaded and no kernel runs. Exit 0 means the CANN
// runtime initialises and sees at least one NPU; anything else is a
// plain error on stderr. This is the "engine self-test with no model"
// milestone from docs/CANN_PORT_PLAN.md §1, in its smallest form.
#include <cstdint>
#include <cstdio>

#include "acl/acl_rt.h"

int main() {
  aclError err = aclInit(nullptr);
  if (err != ACL_SUCCESS) {
    std::fprintf(stderr, "strata_cann_probe: aclInit failed: %d\n",
                 static_cast<int>(err));
    return 1;
  }
  std::uint32_t count = 0;
  err = aclrtGetDeviceCount(&count);
  if (err != ACL_SUCCESS) {
    std::fprintf(stderr, "strata_cann_probe: aclrtGetDeviceCount failed: %d\n",
                 static_cast<int>(err));
    aclFinalize();
    return 1;
  }
  std::printf("strata_cann_probe: CANN runtime ok, NPU count=%u\n", count);
  if (count == 0) {
    std::fprintf(stderr, "strata_cann_probe: no NPU visible to the runtime\n");
    aclFinalize();
    return 2;
  }
  err = aclrtSetDevice(0);
  if (err != ACL_SUCCESS) {
    std::fprintf(stderr, "strata_cann_probe: aclrtSetDevice(0) failed: %d\n",
                 static_cast<int>(err));
    aclFinalize();
    return 1;
  }
  std::int32_t current = -1;
  err = aclrtGetDevice(&current);
  if (err != ACL_SUCCESS) {
    std::fprintf(stderr, "strata_cann_probe: aclrtGetDevice failed: %d\n",
                 static_cast<int>(err));
    aclFinalize();
    return 1;
  }
  std::printf("strata_cann_probe: current device=%d\n", current);
  aclFinalize();
  return 0;
}
