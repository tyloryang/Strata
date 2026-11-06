// cann/src/aclnn_call.hpp - the one place that knows how to run a two-phase aclnn op.
//
// Operators reuse one workspace per device. Synchronous completion is the
// default; audited wrappers may opt into a stream's explicit deferred scope.
// ops.cpp owns that machinery so external wrappers do not duplicate it.
#pragma once

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>

#include <cstdint>
#include <functional>

#include "strata/cann/device.hpp"

namespace strata::cann {

using GetWs = std::function<aclnnStatus(uint64_t *, aclOpExecutor **)>;
using Launch = std::function<aclnnStatus(void *, uint64_t, aclOpExecutor *, aclrtStream)>;

// Only wrappers that defer all ACL descriptor destruction may opt into queueing.
void execute(Stream & stream, GetWs get_ws, Launch launch, const char * what,
             bool deferred_metadata = false);

} // namespace strata::cann
