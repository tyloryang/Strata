#pragma once
// CPU decoding and double dot products here are test oracles only.
#include "strata/cann/dequant_npu.hpp"
#include "strata/cann/npu_tensor.hpp"
#include "strata/cann/ops.hpp"
#include "strata/cann/quants.hpp"
#include <acl/acl.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace fused_test {
using namespace strata::cann;
struct Format { int type; const char * name; };
inline constexpr Format formats[] = {
    {18,"IQ3_XXS"},{21,"IQ3_S"},{13,"Q5_K"},{14,"Q6_K"},
    {8,"Q8_0"},{12,"Q4_K"},{16,"IQ2_XXS"},{17,"IQ2_XS"},
    {20,"IQ4_NL"},{22,"IQ2_S"},{23,"IQ4_XS"},{42,"Q2_0"},
    {0,"F32"},{1,"F16"},{30,"BF16"}
};
inline Format format(int type) {
    for (auto f : formats) if (f.type == type) return f;
    throw std::runtime_error("unsupported format " + std::to_string(type));
}
inline uint64_t block_elements(int type) {
    return type == 0 || type == 1 || type == 30 ? 1 : quant_block_elements(type);
}
inline uint64_t positive(const std::string & text, bool zero = false) {
    size_t end = 0;
    if (text.empty() || text.front() == '-') throw std::runtime_error("invalid integer: " + text);
    const auto value = std::stoull(text, &end);
    if (end != text.size() || (!zero && !value)) throw std::runtime_error("invalid integer: " + text);
    return value;
}
inline std::vector<Format> parse_formats(const std::string & value) {
    std::vector<Format> result;
    size_t start = 0;
    do {
        const auto end = value.find(',', start);
        const auto item = value.substr(start, end == std::string::npos ? std::string::npos : end - start);
        const Format * found = nullptr;
        for (const auto & f : formats) if (item == f.name || item == std::to_string(f.type)) found = &f;
        if (!found) throw std::runtime_error("unknown format " + item);
        if (std::any_of(result.begin(), result.end(), [&](Format f) { return f.type == found->type; }))
            throw std::runtime_error("duplicate format " + item);
        result.push_back(*found);
        if (end == std::string::npos) break;
        start = end + 1;
    } while (true);
    return result;
}
inline void put_half(uint8_t * dst, float value) {
    uint16_t half; f32_to_f16(&value, &half, 1); std::memcpy(dst, &half, 2);
}
struct Input {
    Format f;
    std::string label;
    uint64_t rows, cols;
    std::vector<uint8_t> raw;
    std::vector<float> weights, x;
    std::vector<double> expected, sum_abs;
};
inline void oracle(Input & in) {
    in.x.resize(in.cols);
    uint32_t state = 0x9e3779b9u;
    for (auto & x : in.x) {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        x = float(int32_t(state & 65535) - 32768) / 32768.0f;
    }
    in.expected.assign(in.rows, 0);
    in.sum_abs.assign(in.rows, 0);
    for (uint64_t r = 0; r < in.rows; ++r) {
        double sum = 0, abs_sum = 0;
        for (uint64_t c = 0; c < in.cols; ++c) {
            const double product = double(in.weights[r * in.cols + c]) * in.x[c];
            sum += product; abs_sum += std::abs(product);
        }
        in.expected[r] = sum; in.sum_abs[r] = abs_sum;
    }
}
inline void decode_cpu(Input & in) {
    const auto count = in.rows * in.cols;
    in.weights.resize(count);
    if (in.f.type == 0) std::memcpy(in.weights.data(), in.raw.data(), count * sizeof(float));
    else if (in.f.type == 1) {
        for (uint64_t i = 0; i < count; ++i) {
            uint16_t half; std::memcpy(&half, in.raw.data() + 2*i, 2);
            f16_to_f32(&half, &in.weights[i], 1);
        }
    } else if (in.f.type == 30) {
        for (uint64_t i = 0; i < count; ++i) {
            uint16_t half; std::memcpy(&half, in.raw.data() + 2*i, 2);
            const uint32_t bits = uint32_t(half) << 16;
            std::memcpy(&in.weights[i], &bits, 4);
        }
    } else if (!dequantize(in.f.type, in.raw.data(), in.weights.data(), int64_t(count)))
        throw std::runtime_error("CPU test decoder rejected " + std::string(in.f.name));
    oracle(in);
}
inline void check_shape(uint64_t rows, uint64_t cols) {
    if (!rows || !cols || rows > uint64_t(std::numeric_limits<int64_t>::max()) / cols ||
        rows * cols > std::numeric_limits<size_t>::max() / sizeof(float))
        throw std::runtime_error("matrix shape overflow");
}
inline Input synthetic(Format f, uint64_t rows, uint64_t cols) {
    check_shape(rows, cols);
    if (cols % 8 || rows * cols % block_elements(f.type))
        throw std::runtime_error("synthetic matrix not kernel/block aligned");
    Input in{f, "synthetic", rows, cols, {}, {}, {}, {}, {}};
    const uint64_t count = rows * cols;
    in.raw.resize(quant_storage_bytes(f.type, count));
    if (f.type == 0 || f.type == 1 || f.type == 30) {
        for (uint64_t i = 0; i < count; ++i) {
            float value = float(int((i * 73 + i / cols * 19) % 1021) - 510) / 256.0f;
            if (f.type == 0) std::memcpy(in.raw.data() + 4*i, &value, 4);
            else if (f.type == 1) put_half(in.raw.data() + 2*i, value);
            else {
                uint32_t bits; std::memcpy(&bits, &value, 4);
                const uint16_t half = uint16_t(bits >> 16);
                std::memcpy(in.raw.data() + 2*i, &half, 2);
            }
        }
    } else {
        uint32_t state = 0x19660dU + uint32_t(f.type);
        for (auto & byte : in.raw) {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5; byte = uint8_t(state);
        }
        const float scales[] = {0.0123f, -0.0141f, 0x1p-24f, 0.5f};
        const auto block = block_elements(f.type), bytes = uint64_t(quant_type_size(f.type));
        for (uint64_t b = 0; b < count / block; ++b) {
            auto * dst = in.raw.data() + b*bytes;
            put_half(dst + (f.type == 14 ? 208 : 0), scales[b % 4]);
            if (f.type == 12 || f.type == 13) put_half(dst + 2, 0.0071f);
        }
    }
    decode_cpu(in);
    return in;
}
inline Input gguf_input(const Gguf & gguf, const GgufTensor & tensor,
                        uint64_t row_offset, uint64_t row_limit) {
    if (tensor.dims.size() < 2 || !tensor.dims[0]) throw std::runtime_error("not a matrix: " + tensor.name);
    const uint64_t cols = tensor.dims[0], total_rows = tensor.elements() / cols;
    if (row_offset >= total_rows) throw std::runtime_error("row offset outside tensor");
    const auto f = format(int(tensor.type));
    const auto block = block_elements(f.type);
    if (cols % 8 || row_offset * cols % block)
        throw std::runtime_error("GGUF slice start not kernel/block aligned: " + tensor.name);
    uint64_t rows = row_limit ? std::min(row_limit, total_rows - row_offset) : total_rows - row_offset;
    // Preserve complete flat blocks when a row spans a fractional block.
    while (rows && rows * cols % block) --rows;
    check_shape(rows, cols);
    Input in{f, tensor.name, rows, cols, {}, {}, {}, {}, {}};
    const auto byte_offset = row_offset ? quant_storage_bytes(f.type, row_offset * cols) : 0;
    in.raw = gguf.tensor_bytes(tensor, byte_offset, quant_storage_bytes(f.type, rows * cols));
    decode_cpu(in);
    return in;
}
struct Check {
    double max_abs = 0, max_rel = 0, rms = 0, normalized = 0;
    size_t actual_argmax = 0, reference_argmax = 0;
    bool argmax_stable = false;
};
inline double tolerance(double sum_abs) {
    // FP32 products + tree reductions have different ordering than the double
    // oracle. Scale by the sum of absolute products, not a near-zero dot result.
    return 2e-5 + 8.0 * std::numeric_limits<float>::epsilon() * sum_abs;
}
inline Check verify(const Input & in, const std::vector<float> & actual, const char * path) {
    if (actual.size() != in.expected.size()) throw std::runtime_error("matvec result shape mismatch");
    Check out;
    double sum_square = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(in.expected[i]) || !std::isfinite(in.sum_abs[i]))
            throw std::runtime_error(std::string(path) + ": nonfinite output/oracle at " + std::to_string(i));
        const double error = std::abs(double(actual[i]) - in.expected[i]);
        out.max_abs = std::max(out.max_abs, error);
        out.max_rel = std::max(out.max_rel, error / std::max(1e-12, std::abs(in.expected[i])));
        out.normalized = std::max(out.normalized, error / std::max(1e-12, in.sum_abs[i]));
        sum_square += error * error;
        if (error > tolerance(in.sum_abs[i])) {
            std::fprintf(stderr, "%s %s row=%zu actual=%.9g double=%.17g abs=%.9g bound=%.9g\n",
                         path, in.label.c_str(), i, actual[i], in.expected[i], error, tolerance(in.sum_abs[i]));
            throw std::runtime_error(std::string(path) + ": FP32 dot comparison failed");
        }
        if (actual[i] > actual[out.actual_argmax]) out.actual_argmax = i;
        if (in.expected[i] > in.expected[out.reference_argmax]) out.reference_argmax = i;
    }
    out.rms = std::sqrt(sum_square / actual.size());
    double runner_up = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < actual.size(); ++i) if (i != out.reference_argmax) runner_up = std::max(runner_up, in.expected[i]);
    double max_bound = 0;
    for (double a : in.sum_abs) max_bound = std::max(max_bound, tolerance(a));
    out.argmax_stable = in.expected[out.reference_argmax] - runner_up > 2 * max_bound;
    if (out.argmax_stable && out.actual_argmax != out.reference_argmax)
        throw std::runtime_error(std::string(path) + ": stable argmax mismatch");
    return out;
}
inline void print_check(const Input & in, const char * path, const Check & c) {
    std::printf("PARITY path=%s tensor=%s type=%s rows=%llu cols=%llu max_abs=%.9g max_rel=%.9g rms=%.9g normalized=%.9g argmax=%zu/%zu stable=%s PASS\n",
        path, in.label.c_str(), in.f.name, (unsigned long long)in.rows, (unsigned long long)in.cols,
        c.max_abs, c.max_rel, c.rms, c.normalized, c.actual_argmax, c.reference_argmax, c.argmax_stable ? "true" : "false");
}
inline void poison(Buffer & output) {
    output.activate();
    const auto status = aclrtMemset(output.data(), output.size(), 0xff, output.size());
    if (status != ACL_SUCCESS) throw Error("matvec output poison failed", status);
}
inline std::vector<float> read(const Buffer & b, uint64_t count) {
    std::vector<float> result(count); b.copy_to_host(result.data(), count * sizeof(float)); return result;
}
} // namespace fused_test
