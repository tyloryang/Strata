// CPU computations in this file are test oracles only. Production decode is
// raw GGUF bytes -> HBM -> AscendC -> F32 HBM, with no host dequantization.
#include "strata/cann/dequant_npu.hpp"
#include "strata/cann/ops.hpp"
#include "strata/cann/quants.hpp"
#include "strata/cann/tensor_load.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef STRATA_CANN_HAVE_GGML
#include <ggml.h>
#endif

using namespace strata::cann;

namespace {
void check_values(const std::string & label, const std::vector<float> & actual, const std::vector<float> & expected,
                  bool require_exact = false) {
    if (actual.size() != expected.size()) throw std::runtime_error(label + ": shape mismatch");
    double maximum = 0;
    bool exact = true;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]))
            throw std::runtime_error(label + ": non-finite value at " + std::to_string(i));
        const double difference = std::fabs(double(actual[i]) - expected[i]);
        maximum = std::max(maximum, difference);
        const bool same_bits = std::memcmp(&actual[i], &expected[i], sizeof(float)) == 0;
        exact = exact && same_bits;
        if ((require_exact && !same_bits) || difference > 1e-6 + 2e-6 * std::fabs(double(expected[i]))) {
            std::fprintf(stderr, "%s mismatch at %zu: NPU=%.9g CPU=%.9g\n", label.c_str(), i, actual[i], expected[i]);
            if (require_exact) {
                uint32_t got, want;
                std::memcpy(&got, &actual[i], sizeof(got));
                std::memcpy(&want, &expected[i], sizeof(want));
                std::fprintf(stderr, "  exact bits required: NPU=0x%08x CPU=0x%08x\n", got, want);
            }
            throw std::runtime_error(label + ": NPU dequant mismatch");
        }
    }
    std::printf("%-20s count=%zu max_abs=%.9g %s PASS\n", label.c_str(), actual.size(), maximum, exact ? "BIT-EXACT" : "within FP32 tolerance");
}

std::vector<float> run(Stream & stream, int type, const std::vector<uint8_t> & bytes, uint64_t count) {
    Buffer raw(bytes.size()), decoded(count * sizeof(float));
    raw.copy_from_host(bytes.data(), bytes.size());
    dequantize_device(stream, type, raw.data(), count, decoded.data());
    std::vector<float> result(count);
    decoded.copy_to_host(result.data(), result.size() * sizeof(float));
    return result;
}

void put_half(uint8_t * p, float value) {
    uint16_t bits;
    f32_to_f16(&value, &bits, 1);
    std::memcpy(p, &bits, 2);
}

void synthetic(Stream & stream) {
    const int types[] = {8, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 42};
    for (const int type : types) {
        const int block = quant_block_elements(type), size = quant_type_size(type);
        // Exceeds two 32-core x 16-block tiles, including an incomplete final
        // tile: this catches multi-tile indexing and tail handling regressions.
        constexpr int blocks = 1073;
        const uint64_t count = uint64_t(blocks) * block;
        std::mt19937 random(100 + type);
        std::uniform_int_distribution<int> byte(0, 255);
        std::vector<uint8_t> raw(blocks * size);
        for (auto & value : raw) value = uint8_t(byte(random));
        for (int b = 0; b < blocks; ++b) {
            const float scales[] = {0.0123f, -0.0141f, 0x1p-24f, 0.5f};
            put_half(raw.data() + b * size + (type == 14 ? 208 : 0), scales[b % 4]);
            if (type == 12 || type == 13) put_half(raw.data() + b * size + 2, 0.0071f);
        }
        std::vector<float> expected(count);
#ifdef STRATA_CANN_HAVE_GGML
        const auto * traits = ggml_get_type_traits(static_cast<ggml_type>(type));
        if (!traits || !traits->to_float) throw std::runtime_error("ggml decoder missing for type " + std::to_string(type));
        traits->to_float(raw.data(), expected.data(), count);
#else
        if (!dequantize(type, raw.data(), expected.data(), count)) throw std::runtime_error("CPU test oracle rejected type");
#endif
        check_values("synthetic type=" + std::to_string(type), run(stream, type, raw, count), expected);
    }
    // Exercise plain element types and a partial last device work group.
    for (const int type : {0, 1, 30}) {
        const size_t count = 515, width = type == 0 ? 4 : 2;
        std::vector<float> expected(count);
        std::vector<uint8_t> bytes(count * width);
        for (size_t i = 0; i < count; ++i) {
            float value = (float(int(i % 257) - 128)) / 64.0f;
            if (type == 0) {
                // Plain F32 copies must retain both zero signs and subnormals.
                constexpr uint32_t edges[] = {0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u,
                    0x007fffffu, 0x807fffffu, 0x00800000u, 0x80800000u, 0x7f7fffffu, 0xff7fffffu};
                if (i < sizeof(edges) / sizeof(edges[0])) std::memcpy(&value, &edges[i], sizeof(value));
                std::memcpy(bytes.data() + i * width, &value, 4);
            }
            else if (type == 1) {
                uint16_t half;
                f32_to_f16(&value, &half, 1);
                std::memcpy(bytes.data() + i * width, &half, 2);
                f16_to_f32(&half, &value, 1);
            } else {
                uint32_t bits;
                std::memcpy(&bits, &value, 4);
                const uint16_t half = uint16_t(bits >> 16);
                std::memcpy(bytes.data() + i * width, &half, 2);
                bits &= 0xffff0000u;
                std::memcpy(&value, &bits, 4);
            }
            expected[i] = value;
        }
        check_values("element type=" + std::to_string(type), run(stream, type, bytes, count), expected, true);
    }
    Buffer raw(256), out(1024);
    bool rejected = false;
    try { dequantize_device(stream, 999, raw.data(), 256, out.data()); }
    catch (const std::runtime_error &) { rejected = true; }
    if (!rejected) throw std::runtime_error("unknown type was not rejected");
    rejected = false;
    try { dequantize_device(stream, 18, raw.data(), 255, out.data()); }
    catch (const std::runtime_error &) { rejected = true; }
    if (!rejected) throw std::runtime_error("unaligned IQ3_XXS count was not rejected");
    std::puts("Unsupported type and unaligned count: rejected without fallback PASS");
}

void every_finite_half(Stream & stream) {
    for (const int type : {1, 30}) {
        std::vector<uint8_t> bytes;
        std::vector<float> expected;
        size_t skipped = 0, subnormals = 0, zeros = 0;
        for (uint32_t raw = 0; raw < 65536; ++raw) {
            const uint16_t bits = uint16_t(raw);
            const uint32_t exponent = type == 1 ? ((raw >> 10) & 31) : ((raw >> 7) & 255);
            const uint32_t fraction = type == 1 ? (raw & 1023) : (raw & 127);
            if (exponent == uint32_t(type == 1 ? 31 : 255)) {
                ++skipped; // NaN/Inf are intentionally outside this finite-only oracle.
                continue;
            }
            if (exponent == 0) {
                if (fraction == 0) ++zeros; else ++subnormals;
            }
            float value;
            if (type == 1) {
                // Independent numeric IEEE definition, not the production Cast
                // or the shared half converter. Unary minus retains negative zero.
                value = exponent == 0 ? std::ldexp(float(fraction), -24)
                                      : std::ldexp(float(1024 + fraction), int(exponent) - 25);
                if (raw & 0x8000) value = -value;
            } else {
                const uint32_t expanded = raw << 16;
                std::memcpy(&value, &expanded, sizeof(value));
            }
            const size_t at = bytes.size(); bytes.resize(at + sizeof(bits));
            std::memcpy(bytes.data() + at, &bits, sizeof(bits));
            expected.push_back(value);
        }
        const size_t wanted = type == 1 ? 63488 : 65280;
        const size_t wanted_subnormals = type == 1 ? 2046 : 254;
        if (expected.size() != wanted || skipped != 65536 - wanted || zeros != 2 || subnormals != wanted_subnormals)
            throw std::runtime_error("exhaustive finite half enumeration is incomplete");
        check_values(type == 1 ? "all finite F16" : "all finite BF16",
                     run(stream, type, bytes, expected.size()), expected, true);
        std::printf("  finite=%zu subnormals=%zu signed_zeros=%zu skipped_NaN_Inf=%zu\n",
                    expected.size(), subnormals, zeros, skipped);
    }
}

void ple(Stream & stream) {
    uint64_t context[3] = {760, 6511, 314};
    uint64_t mult[3] = {0xffffffffffffffffULL, 0xf123456789abcdefULL, 0x8923456789abcdefULL};
    uint64_t offsets[16], vocabs[16];
    for (int h = 0; h < 16; ++h) {
        offsets[h] = uint64_t(h) * 20000001;
        vocabs[h] = 19999999 + h * 2;
    }
    // Include divisors beyond 2^63 to test unsigned modulo without FP conversion.
    vocabs[1] = 0x8000000000000001ULL;
    vocabs[7] = 0xffffffffffffffffULL;
    for (int trial = 0; trial < 3; ++trial) {
        const auto got = ple_rows_device(stream, context, mult, offsets, vocabs);
        const uint64_t h2 = context[0] * mult[0] ^ context[1] * mult[1];
        const uint64_t h3 = h2 ^ context[2] * mult[2];
        for (int h = 0; h < 16; ++h)
            if (got[h] != (h < 8 ? h2 : h3) % vocabs[h] + offsets[h])
                throw std::runtime_error("NPU PLE uint64 hash mismatch");
        context[trial] = std::numeric_limits<uint64_t>::max() - trial;
    }
    std::puts("PLE uint64 wrapping hash and modulo: 48/48 row IDs exact PASS");
}

void real_weights(Stream & stream, const std::string & path) {
    Gguf gguf(path);
    std::set<int> tested;
    for (const auto & tensor : gguf.tensors()) {
        if (!tested.insert(int(tensor.type)).second) continue;
        const uint64_t block = tensor.type == 0 || tensor.type == 1 || tensor.type == 30 ? 1 : quant_block_elements(tensor.type);
        if (!block) throw std::runtime_error("real tensor has unsupported type " + std::to_string(tensor.type));
        const uint64_t count = std::min<uint64_t>(tensor.elements() / block, 4) * block;
        const auto device = load_quant_device(stream, gguf, tensor, 0, count);
        std::vector<float> got(count);
        device->copy_to_host(got.data(), got.size() * sizeof(float));
        const auto expected = load_tensor_slice(gguf, tensor, 0, count);
        check_values("real type=" + std::to_string(tensor.type), got, expected,
                     tensor.type == 0 || tensor.type == 1 || tensor.type == 30);
        std::printf("  tensor=%s\n", tensor.name.c_str());
    }
    std::printf("Real GGUF: %zu distinct types checked\n", tested.size());
}
} // namespace

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        init();
        Device device(std::getenv("STRATA_CANN_DEVICE") ? std::stoi(std::getenv("STRATA_CANN_DEVICE")) : 0);
        Stream stream(device);
        synthetic(stream);
        every_finite_half(stream);
        ple(stream);
        for (int i = 1; i < argc; ++i) real_weights(stream, argv[i]);
        std::puts("All NPU decode checks passed; CPU code was used only as a test oracle.");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAILED: %s\n", error.what());
        return 1;
    }
}
