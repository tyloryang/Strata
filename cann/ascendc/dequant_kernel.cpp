// Tiled UB/DMA implementation of the canonical GGUF block layouts. Quant
// fields are decoded once per block/group on AI Cores; no CPU decoder is used.
// Keep FP32 arithmetic order consistent with cann/src/{iq_,}quants.cpp.
#include "kernel_operator.h"

namespace {
constexpr uint64_t SIGNS = 0;
constexpr uint64_t IQ2XXS = 128;
constexpr uint64_t IQ2XS = IQ2XXS + 256 * 8;
constexpr uint64_t IQ2S = IQ2XS + 512 * 8;
constexpr uint64_t IQ3XXS = IQ2S + 1024 * 8;
constexpr uint64_t IQ3S = IQ3XXS + 256 * 4;
constexpr uint64_t IQ4NL = IQ3S + 512 * 4;

__aicore__ inline float float_bits(uint32_t bits) {
    union Value { uint32_t u; float f; } value;
    value.u = bits;
    return value.f;
}

__aicore__ inline float half_value(uint32_t bits) {
    const uint32_t sign = (bits & 0x8000u) << 16;
    uint32_t exponent = (bits >> 10) & 31u;
    uint32_t fraction = bits & 1023u;
    if (exponent == 0) {
        if (fraction == 0) return float_bits(sign);
        exponent = 113;
        while ((fraction & 1024u) == 0) { fraction <<= 1; --exponent; }
        return float_bits(sign | (exponent << 23) | ((fraction & 1023u) << 13));
    }
    if (exponent == 31) return float_bits(sign | 0x7f800000u | (fraction << 13));
    return float_bits(sign | ((exponent + 112u) << 23) | (fraction << 13));
}

#include "dequant_decoder.hpp"

class OrderedExpertSummer {
public:
    __aicore__ inline void Run(GM_ADDR input, GM_ADDR scores, GM_ADDR output,
        uint64_t rows, uint32_t selected) {
        constexpr uint32_t TILE = 256;
        AscendC::GlobalTensor<float> source, coefficients, destination;
        source.SetGlobalBuffer((__gm__ float *)input);
        coefficients.SetGlobalBuffer((__gm__ float *)scores);
        destination.SetGlobalBuffer((__gm__ float *)output);
        pipe_.InitBuffer(input_buffer_, TILE * sizeof(float));
        pipe_.InitBuffer(accumulator_buffer_, TILE * sizeof(float));
        pipe_.InitBuffer(scores_buffer_, (selected * uint32_t(sizeof(float)) + 31) / 32 * 32);
        auto values = input_buffer_.Get<float>();
        auto accumulated = accumulator_buffer_.Get<float>();
        auto weights = scores_buffer_.Get<float>();
        AscendC::DataCopyPad(weights, coefficients,
            AscendC::DataCopyExtParams{1, selected * uint32_t(sizeof(float)), 0, 0, 0},
            AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
        Sync<AscendC::HardEvent::MTE2_S>();
        const uint64_t tiles = (rows + TILE - 1) / TILE;
        const uint64_t cores = AscendC::GetBlockNum(), core = AscendC::GetBlockIdx();
        const uint64_t quotient = tiles / cores, remainder = tiles % cores;
        const uint64_t begin = core * quotient + (core < remainder ? core : remainder);
        const uint64_t end = begin + quotient + (core < remainder ? 1 : 0);
        for (uint64_t tile = begin; tile < end; ++tile) {
            const uint64_t first = tile * TILE;
            const uint32_t count = uint32_t(rows - first < TILE ? rows - first : TILE);
            AscendC::Duplicate(accumulated, 0.0f, int32_t(count));
            for (uint32_t expert = 0; expert < selected; ++expert) {
                const float coefficient = weights.GetValue(expert);
                AscendC::DataCopyPad(values, source[uint64_t(expert) * rows + first],
                    AscendC::DataCopyExtParams{1, count * uint32_t(sizeof(float)), 0, 0, 0},
                    AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
                Sync<AscendC::HardEvent::MTE2_V>();
                AscendC::Muls(values, values, coefficient, int32_t(count));
                AscendC::PipeBarrier<PIPE_V>();
                // Match the former per-expert ACLNN Mul then Add, including
                // the first addition to +0. Never use FMA or parallel reduce.
                AscendC::Add(accumulated, accumulated, values, count);
                AscendC::PipeBarrier<PIPE_V>();
                Sync<AscendC::HardEvent::V_MTE2>();
            }
            Sync<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopyPad(destination[first], accumulated,
                AscendC::DataCopyExtParams{1, count * uint32_t(sizeof(float)), 0, 0, 0});
            Sync<AscendC::HardEvent::MTE3_V>();
        }
    }
private:
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> input_buffer_, accumulator_buffer_, scores_buffer_;
    template <AscendC::HardEvent Event>
    __aicore__ inline void Sync() {
        const auto id = static_cast<event_t>(pipe_.FetchEventID(Event));
        AscendC::SetFlag<Event>(id);
        AscendC::WaitFlag<Event>(id);
    }
};

__aicore__ inline uint64_t remainder_u64(uint64_t value, uint64_t divisor) {
    uint64_t remainder = 0;
    for (int bit = 63; bit >= 0; --bit) {
        const bool carry = (remainder >> 63) != 0;
        remainder = (remainder << 1) | ((value >> bit) & 1);
        if (carry || remainder >= divisor) remainder -= divisor;
    }
    return remainder;
}
} // namespace

extern "C" __global__ __aicore__ void strata_dequant_kernel(
    GM_ADDR raw, GM_ADDR tables, GM_ADDR output, uint64_t count,
    int32_t type, uint32_t block_elements, uint32_t block_bytes) {
    Decoder decoder;
    decoder.Init(raw, tables, output, type);
    decoder.Run(type, count, block_elements, block_bytes);
}

extern "C" __global__ __aicore__ void strata_matvec_kernel(
    GM_ADDR raw, GM_ADDR tables, GM_ADDR activation, GM_ADDR output,
    uint64_t rows, uint64_t cols, int32_t type,
    uint32_t block_elements, uint32_t block_bytes) {
    Decoder decoder;
    decoder.InitMatvec(raw, tables, output, type, block_elements, block_bytes);
    decoder.RunMatvec(activation, rows, cols, type, block_elements, block_bytes);
}

extern "C" __global__ __aicore__ void strata_grouped_matvec_kernel(
    GM_ADDR raw, GM_ADDR tables, GM_ADDR activation, GM_ADDR ids, GM_ADDR output,
    uint64_t rows, uint64_t cols, uint32_t selected, uint32_t experts_total,
    int32_t type, uint32_t block_elements, uint32_t block_bytes, int32_t x_perexpert) {
    Decoder decoder;
    decoder.InitMatvec(raw, tables, output, type, block_elements, block_bytes);
    decoder.RunGroupedMatvec(raw, activation, ids, rows, cols, selected, experts_total,
        type, block_elements, block_bytes, x_perexpert != 0);
}

extern "C" __global__ __aicore__ void strata_ordered_expert_sum_kernel(
    GM_ADDR input, GM_ADDR scores, GM_ADDR output, uint64_t rows, uint32_t selected) {
    OrderedExpertSummer sum;
    sum.Run(input, scores, output, rows, selected);
}

extern "C" __global__ __aicore__ void strata_ple_rows_kernel(GM_ADDR input, GM_ADDR output) {
    AscendC::GlobalTensor<uint64_t> in, out;
    in.SetGlobalBuffer((__gm__ uint64_t *)input);
    out.SetGlobalBuffer((__gm__ uint64_t *)output);
    AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE>(in);
    const uint64_t h2 = in.GetValue(0) * in.GetValue(3) ^ in.GetValue(1) * in.GetValue(4);
    const uint64_t h3 = h2 ^ in.GetValue(2) * in.GetValue(5);
    for (uint32_t head = 0; head < 16; ++head)
        out.SetValue(head, remainder_u64(head < 8 ? h2 : h3, in.GetValue(22 + head)) + in.GetValue(6 + head));
    AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE>(out);
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void strata_cann_launch_dequant(uint32_t cores, void * stream,
    const void * raw, const void * tables, void * output, uint64_t count,
    int32_t type, uint32_t block_elements, uint32_t block_bytes) {
    strata_dequant_kernel<<<cores, nullptr, stream>>>((uint8_t *)raw, (uint8_t *)tables,
        (uint8_t *)output, count, type, block_elements, block_bytes);
}

extern "C" void strata_cann_launch_ple_rows(void * stream, const void * input, void * output) {
    strata_ple_rows_kernel<<<1, nullptr, stream>>>((uint8_t *)input, (uint8_t *)output);
}

extern "C" void strata_cann_launch_matvec(uint32_t cores, void * stream,
    const void * raw, const void * tables, const float * activation, float * output,
    uint64_t rows, uint64_t cols, int32_t type,
    uint32_t block_elements, uint32_t block_bytes) {
    strata_matvec_kernel<<<cores, nullptr, stream>>>((uint8_t *)raw, (uint8_t *)tables,
        (uint8_t *)activation, (uint8_t *)output, rows, cols, type, block_elements, block_bytes);
}

extern "C" void strata_cann_launch_grouped_matvec(uint32_t cores, void * stream,
    const void * raw, const void * tables, const float * activation, const int64_t * ids, float * output,
    uint64_t rows, uint64_t cols, uint32_t selected, uint32_t experts_total, int32_t type,
    uint32_t block_elements, uint32_t block_bytes, int32_t x_perexpert) {
    strata_grouped_matvec_kernel<<<cores, nullptr, stream>>>((uint8_t *)raw, (uint8_t *)tables,
        (uint8_t *)activation, (uint8_t *)ids, (uint8_t *)output, rows, cols, selected, experts_total,
        type, block_elements, block_bytes, x_perexpert);
}

extern "C" void strata_cann_launch_ordered_expert_sum(uint32_t cores, void * stream,
    const float * input, const float * scores, float * output, uint64_t rows, uint32_t selected) {
    strata_ordered_expert_sum_kernel<<<cores, nullptr, stream>>>((uint8_t *)input,
        (uint8_t *)scores, (uint8_t *)output, rows, selected);
}
#endif
