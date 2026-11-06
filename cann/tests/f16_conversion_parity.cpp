// Host-only checks of the conversion used by native CANN KV storage.
// The executable links the CANN library but does not initialize or use an NPU.
#include "strata/cann/ops.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>

using strata::cann::f16_to_f32;
using strata::cann::f32_to_f16;

namespace {
uint32_t bits(float value) {
    uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

float from_bits(uint32_t value) {
    float result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

uint16_t encode(float value) {
    uint16_t result;
    f32_to_f16(&value, &result, 1);
    return result;
}

void expect(float value, uint16_t wanted) {
    const uint16_t actual = encode(value);
    if (actual != wanted) {
        std::fprintf(stderr, "f32 0x%08x -> f16 0x%04x, expected 0x%04x\n",
                     bits(value), actual, wanted);
        throw std::runtime_error("FP16 conversion mismatch");
    }
}

void known_vectors() {
    expect(0.0f, 0x0000);
    expect(-0.0f, 0x8000);
    expect(1.0f, 0x3c00);
    expect(-2.0f, 0xc000);
    // Ties on either side of an odd mantissa, plus a value above a tie.
    expect(1.0f + 0x1p-11f, 0x3c00);
    expect(1.0f + 3 * 0x1p-11f, 0x3c02);
    expect(std::nextafter(1.0f + 0x1p-11f, 2.0f), 0x3c01);
    expect(-1.0f - 3 * 0x1p-11f, 0xbc02);
    expect(2.0f - 0x1p-11f, 0x4000); // carry into exponent
    expect(0x1p-24f, 0x0001);
    expect(-0x1p-24f, 0x8001);
    expect(0x1p-25f, 0x0000); // halfway to the smallest subnormal
    expect(-0x1p-25f, 0x8000);
    expect(std::nextafter(0x1p-25f, 1.0f), 0x0001);
    expect(3 * 0x1p-25f, 0x0002);
    expect(0x1p-14f - 0x1p-25f, 0x0400); // subnormal -> normal tie
    expect(0x1p-14f, 0x0400);
    expect(65504.0f, 0x7bff);
    expect(65519.0f, 0x7bff);
    expect(65520.0f, 0x7c00); // nearest-even overflow threshold
    expect(-65520.0f, 0xfc00);
    expect(std::numeric_limits<float>::max(), 0x7c00);
    expect(std::numeric_limits<float>::denorm_min(), 0x0000);
    expect(std::numeric_limits<float>::infinity(), 0x7c00);
    expect(-std::numeric_limits<float>::infinity(), 0xfc00);
    expect(from_bits(0x7fc12345), 0x7e00);
    expect(from_bits(0xffc12345), 0xfe00);
    expect(from_bits(0x7f800001), 0x7e00); // signaling NaN stays NaN
}

void every_half_value() {
    for (uint32_t raw = 0; raw < 65536; ++raw) {
        const uint16_t half = static_cast<uint16_t>(raw);
        float actual;
        f16_to_f32(&half, &actual, 1);
        const int exponent = (raw >> 10) & 31;
        const int fraction = raw & 1023;
        if (exponent == 31) {
            if (fraction ? !std::isnan(actual) : !std::isinf(actual)) {
                throw std::runtime_error("FP16 infinity/NaN expansion mismatch");
            }
            if (std::signbit(actual) != ((raw & 0x8000) != 0)) {
                throw std::runtime_error("FP16 infinity/NaN sign mismatch");
            }
            expect(actual, static_cast<uint16_t>((raw & 0x8000) | (fraction ? 0x7e00 : 0x7c00)));
        } else {
            // Numeric IEEE definition, independent of the bit converter.
            float wanted = exponent == 0 ? std::ldexp(float(fraction), -24)
                                         : std::ldexp(float(1024 + fraction), exponent - 25);
            if (raw & 0x8000) wanted = -wanted;
            if (bits(actual) != bits(wanted)) {
                throw std::runtime_error("FP16 finite expansion mismatch");
            }
            expect(actual, half);
        }
    }
}

#if defined(__FLT16_MANT_DIG__) && __FLT16_MANT_DIG__ == 11
void compiler_oracle() {
    uint32_t state = 0x7bd05a91u;
    for (int i = 0; i < 1000000; ++i) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        // The shared converter canonicalizes NaN payloads; compiler payload
        // policies differ. NaNs are already checked explicitly above.
        if ((state & 0x7f800000u) == 0x7f800000u && (state & 0x007fffffu)) continue;
        const float value = from_bits(state);
        const _Float16 reference = static_cast<_Float16>(value);
        uint16_t wanted;
        static_assert(sizeof(reference) == sizeof(wanted));
        std::memcpy(&wanted, &reference, sizeof(wanted));
        expect(value, wanted);
    }
    std::puts("1,000,000 sampled F32 bit patterns vs compiler _Float16: PASS");
}
#endif
} // namespace

int main() {
    try {
        known_vectors();
        every_half_value();
        std::puts("Known RNE vectors and all 65,536 FP16 encodings: PASS");
#if defined(__FLT16_MANT_DIG__) && __FLT16_MANT_DIG__ == 11
        compiler_oracle();
#else
        std::puts("Compiler _Float16 oracle unavailable; known-vector/exhaustive checks passed");
#endif
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAILED: %s\n", error.what());
        return 1;
    }
}
