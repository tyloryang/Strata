#pragma once
#include "strata/cann/engine.hpp"

namespace strata::cann {
// Native correctness path: CANN FP32 weight matmuls, host FP32 state/elementwise
// operations. No Torch, ggml or external inference process is used.
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
};
}
