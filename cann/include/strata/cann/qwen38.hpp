#pragma once
#include "strata/cann/engine.hpp"

namespace strata::cann {
class Qwen38NpuModel;
// Native Qwen3.8 dispatcher. STRATA_CANN_NPU_ONLY=1 selects device-resident
// decoding and forward computation; the default retains the host reference.
// In NPU mode step() returns an empty handle and top() consumes device hidden
// state directly. No Torch, ggml or external inference process is used.
class Qwen38Model {
public:
    Qwen38Model(const std::string & path, int64_t capacity);
    ~Qwen38Model();
    std::vector<float> step(int64_t token, int64_t position);
    Model::Top top(const std::vector<float> & hidden, int64_t k);
    void reset();
    int64_t capacity() const;
    int64_t vocab() const;
    int64_t cache_bytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
#ifdef STRATA_CANN_HAS_NPU_GRAPH
    std::unique_ptr<Qwen38NpuModel> npu_impl_;
#endif
};
}
