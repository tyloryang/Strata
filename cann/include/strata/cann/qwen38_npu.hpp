#pragma once
#include "strata/cann/engine.hpp"

namespace strata::cann {
// Opt-in device-resident forward path. Host code only performs I/O and dispatch.
class Qwen38NpuModel {
public:
    Qwen38NpuModel(const std::string& path, int64_t capacity);
    ~Qwen38NpuModel();
    void step(int64_t token, int64_t position);
    Model::Top top(int64_t k);
    void reset();
    int64_t capacity() const;
    int64_t vocab() const;
    int64_t cache_bytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
