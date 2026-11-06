// Included inside Decoder's private section. The three supported formats use
// only vector Gather/Shift/And/Cast/Mul in their per-tile decode path. Offsets
// are created once per kernel on the device, then reused for every 1024 lanes.
// InitVectorFast must run from Init/InitMatvec. Call DecodeVectorFast after
// MTE2_V, and skip the ordinary PackBlock/Cast/scale path when it returns true.
    static constexpr uint32_t FAST_LANES = 1024;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> fast_offsets_buffer_, fast_values_buffer_,
        fast_float_buffer_, fast_constants_buffer_;
    AscendC::LocalTensor<int32_t> fast_raw_offsets_, fast_d_offsets_, fast_low_offsets_,
        fast_high_offsets_, fast_scratch_;
    AscendC::LocalTensor<uint16_t> fast_a_, fast_b_, fast_mask_;
    AscendC::LocalTensor<float> fast_scale_, fast_lut_;

    __aicore__ inline bool IsVectorFast(int type) {
        return type == 20 || type == 23 || type == 42;
    }
    __aicore__ inline void FastBarrier() { AscendC::PipeBarrier<PIPE_V>(); }

    // A2 has only 16-bit vector And. For 32-bit offsets reinterpret both words;
    // a repeated {low16,high16} constant preserves their exact integer bits.
    __aicore__ inline void FastAnd32(const AscendC::LocalTensor<int32_t>& values,
        int32_t mask) {
        AscendC::Duplicate(fast_mask_.ReinterpretCast<int32_t>(), mask, int32_t(8));
        FastBarrier();
        auto words = values.ReinterpretCast<uint16_t>();
        AscendC::And(words, words, fast_mask_, uint64_t(128), uint8_t(FAST_LANES / 64),
            AscendC::BinaryRepeatParams{1, 1, 0, 8, 8, 0});
        FastBarrier();
    }
    __aicore__ inline void FastAnd16(const AscendC::LocalTensor<uint16_t>& values,
        uint16_t mask, uint32_t count) {
        AscendC::Duplicate(fast_mask_, mask, int32_t(16));
        FastBarrier();
        const uint8_t repeats = uint8_t(count / 128);
        if (repeats) AscendC::And(values, values, fast_mask_, uint64_t(128), repeats,
            AscendC::BinaryRepeatParams{1, 1, 0, 8, 8, 0});
        if (count % 128) AscendC::And(values[uint32_t(repeats) * 128],
            values[uint32_t(repeats) * 128], fast_mask_, uint64_t(count % 128), uint8_t(1),
            AscendC::BinaryRepeatParams{1, 1, 0, 8, 8, 0});
        FastBarrier();
    }

    // Mask words are low lanes first, as required by AscendC's L0 interface.
    // step=256 handles the eight IQ4_XS scale groups, two vector repeats apart.
    __aicore__ inline void FastMaskedShift(const AscendC::LocalTensor<uint16_t>& values,
        uint16_t amount, uint64_t low_mask, uint64_t high_mask, uint32_t count,
        uint32_t offset = 0, uint32_t step = 128) {
        if (offset >= count) return;
        const uint32_t remaining = count - offset;
        const uint8_t repeats = remaining < 128 ? 0 : uint8_t(1 + (remaining - 128) / step);
        uint64_t mask[2] = {low_mask, high_mask};
        const uint8_t stride = uint8_t(step / 16);
        if (repeats) AscendC::ShiftRight(values[offset], values[offset], amount, mask, repeats,
            AscendC::UnaryRepeatParams{1, 1, stride, stride});
        const uint32_t tail_start = offset + uint32_t(repeats) * step;
        if (tail_start < count) {
            const uint32_t tail = count - tail_start < 128 ? count - tail_start : 128;
            if (tail < 64) { mask[0] &= (uint64_t(1) << tail) - 1; mask[1] = 0; }
            else if (tail == 64) mask[1] = 0;
            else if (tail < 128) mask[1] &= (uint64_t(1) << (tail - 64)) - 1;
            if (mask[0] || mask[1]) AscendC::ShiftRight(values[tail_start], values[tail_start],
                amount, mask, uint8_t(1), AscendC::UnaryRepeatParams{1, 1, stride, stride});
        }
        FastBarrier();
    }

    __aicore__ inline void InitVectorFast(int type, uint32_t /*capacity*/) {
        if (!IsVectorFast(type)) return;
        // 28,992 bytes independent of the ordinary/fused tile capacity.
        pipe_.InitBuffer(fast_offsets_buffer_, 5 * FAST_LANES * sizeof(int32_t));
        pipe_.InitBuffer(fast_values_buffer_, 2 * FAST_LANES * sizeof(uint16_t));
        pipe_.InitBuffer(fast_float_buffer_, FAST_LANES * sizeof(float));
        pipe_.InitBuffer(fast_constants_buffer_, 320);
        auto offsets = fast_offsets_buffer_.Get<int32_t>();
        fast_raw_offsets_ = offsets;
        fast_d_offsets_ = offsets[FAST_LANES];
        fast_low_offsets_ = offsets[2 * FAST_LANES];
        fast_high_offsets_ = offsets[3 * FAST_LANES];
        fast_scratch_ = offsets[4 * FAST_LANES];
        fast_a_ = fast_values_buffer_.Get<uint16_t>();
        fast_b_ = fast_a_[FAST_LANES];
        fast_scale_ = fast_float_buffer_.Get<float>();
        fast_lut_ = fast_constants_buffer_.Get<float>();
        fast_mask_ = fast_constants_buffer_.Get<uint16_t>()[32];
        // Canonical IQ4_NL codebook (also used by IQ4_XS). Constants are exact
        // FP32 integers; these 16 scalar stores happen only once per kernel.
        constexpr int16_t table[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
            1, 13, 25, 38, 53, 69, 89, 113};
        for (uint32_t i = 0; i < 16; ++i) fast_lut_.SetValue(i, float(table[i]));
        Sync<AscendC::HardEvent::S_V>();
        AscendC::CreateVecIndex(fast_scratch_, int32_t(0), FAST_LANES);
        FastBarrier();
        const int32_t block_shift = type == 20 ? 5 : type == 42 ? 6 : 8;
        const int32_t block_bytes = type == 23 ? 136 : 18;
        AscendC::ShiftRight(fast_d_offsets_, fast_scratch_, block_shift, int32_t(FAST_LANES));
        AscendC::ShiftRight(fast_raw_offsets_, fast_scratch_, int32_t(type == 42 ? 3 : 1),
            int32_t(FAST_LANES));
        FastBarrier();
        AscendC::Muls(fast_d_offsets_, fast_d_offsets_, block_bytes, int32_t(FAST_LANES));
        FastAnd32(fast_raw_offsets_, 7);
        AscendC::ShiftLeft(fast_raw_offsets_, fast_raw_offsets_, int32_t(1), int32_t(FAST_LANES));
        FastBarrier();
        AscendC::Add(fast_raw_offsets_, fast_raw_offsets_, fast_d_offsets_, int32_t(FAST_LANES));
        FastBarrier();
        if (type == 23) {
            AscendC::ShiftRight(fast_low_offsets_, fast_scratch_, int32_t(5), int32_t(FAST_LANES));
            FastBarrier();
            FastAnd32(fast_low_offsets_, 7);
            AscendC::ShiftLeft(fast_low_offsets_, fast_low_offsets_, int32_t(4), int32_t(FAST_LANES));
            FastBarrier();
            AscendC::Add(fast_raw_offsets_, fast_raw_offsets_, fast_low_offsets_, int32_t(FAST_LANES));
            FastBarrier();
            AscendC::Adds(fast_raw_offsets_, fast_raw_offsets_, int32_t(8), int32_t(FAST_LANES));
            AscendC::ShiftRight(fast_low_offsets_, fast_scratch_, int32_t(7), int32_t(FAST_LANES));
            FastBarrier();
            FastAnd32(fast_low_offsets_, 1);
            AscendC::ShiftLeft(fast_low_offsets_, fast_low_offsets_, int32_t(1), int32_t(FAST_LANES));
            FastBarrier();
            AscendC::Add(fast_low_offsets_, fast_low_offsets_, fast_d_offsets_, int32_t(FAST_LANES));
            FastBarrier();
            AscendC::Adds(fast_low_offsets_, fast_low_offsets_, int32_t(4), int32_t(FAST_LANES));
            AscendC::Adds(fast_high_offsets_, fast_d_offsets_, int32_t(2), int32_t(FAST_LANES));
        } else {
            AscendC::Adds(fast_raw_offsets_, fast_raw_offsets_, int32_t(2), int32_t(FAST_LANES));
        }
        FastBarrier();
    }

    __aicore__ inline bool DecodeVectorFast(int type, uint32_t elements) {
        if (!IsVectorFast(type)) return false;
        const uint32_t block_elements = type == 20 ? 32 : type == 42 ? 64 : 256;
        const uint32_t block_bytes = type == 23 ? 136 : 18;
        auto raw_words = raw_local_.ReinterpretCast<uint16_t>();
        auto raw_halves = raw_local_.ReinterpretCast<half>();
        for (uint32_t start = 0; start < elements; start += FAST_LANES) {
            const uint32_t count = elements - start < FAST_LANES ? elements - start : FAST_LANES;
            const uint32_t base = (start / block_elements) * block_bytes;
            auto result = output_local_[start];
            AscendC::Gather(fast_a_, raw_words, fast_raw_offsets_.ReinterpretCast<uint32_t>(), base, count);
            FastBarrier();
            if (type == 42) {
                for (uint16_t lane = 1; lane < 8; ++lane) {
                    const uint64_t mask = 0x0101010101010101ull << lane;
                    FastMaskedShift(fast_a_, uint16_t(2 * lane), mask, mask, count);
                }
                FastAnd16(fast_a_, uint16_t(3), count);
                auto codes = fast_a_.ReinterpretCast<int16_t>();
                AscendC::Adds(codes, codes, int16_t(-1), int32_t(count));
                FastBarrier();
            } else {
                FastMaskedShift(fast_a_, uint16_t(8), 0x0000aaaa0000aaaaull, 0x0000aaaa0000aaaaull, count);
                FastMaskedShift(fast_a_, uint16_t(4), 0x5555000055550000ull, 0x5555000055550000ull, count);
                FastMaskedShift(fast_a_, uint16_t(12), 0xaaaa0000aaaa0000ull, 0xaaaa0000aaaa0000ull, count);
                FastAnd16(fast_a_, uint16_t(15), count);
            }
            AscendC::Cast(fast_b_.ReinterpretCast<half>(), fast_a_.ReinterpretCast<int16_t>(),
                AscendC::RoundMode::CAST_NONE, count);
            FastBarrier();
            if (type == 42) {
                AscendC::Cast(result, fast_b_.ReinterpretCast<half>(), AscendC::RoundMode::CAST_NONE, count);
            } else {
                AscendC::Cast(fast_scratch_, fast_b_.ReinterpretCast<half>(), AscendC::RoundMode::CAST_RINT, count);
                FastBarrier();
                AscendC::ShiftLeft(fast_scratch_.ReinterpretCast<uint32_t>(),
                    fast_scratch_.ReinterpretCast<uint32_t>(), uint32_t(2), int32_t(count));
                FastBarrier();
                AscendC::Gather(result, fast_lut_, fast_scratch_.ReinterpretCast<uint32_t>(), uint32_t(0), count);
            }
            FastBarrier();

            if (type == 23) {
                AscendC::Gather(fast_a_, raw_words, fast_low_offsets_.ReinterpretCast<uint32_t>(), base, count);
                AscendC::Gather(fast_b_, raw_words, fast_high_offsets_.ReinterpretCast<uint32_t>(), base, count);
                FastBarrier();
                FastMaskedShift(fast_a_, uint16_t(4), 0xffffffff00000000ull, 0, count);
                FastMaskedShift(fast_a_, uint16_t(8), 0, 0x00000000ffffffffull, count);
                FastMaskedShift(fast_a_, uint16_t(12), 0, 0xffffffff00000000ull, count);
                FastAnd16(fast_a_, uint16_t(15), count);
                for (uint16_t group = 1; group < 8; ++group) {
                    const uint32_t local_group = group % 4;
                    const uint64_t mask = uint64_t(0xffffffffu) << ((local_group % 2) * 32);
                    FastMaskedShift(fast_b_, uint16_t(group * 2), local_group < 2 ? mask : 0,
                        local_group >= 2 ? mask : 0, count, uint32_t(group / 4) * 128, 256);
                }
                FastAnd16(fast_b_, uint16_t(3), count);
                AscendC::ShiftLeft(fast_b_, fast_b_, uint16_t(4), int32_t(count));
                FastBarrier();
                auto signed_a = fast_a_.ReinterpretCast<int16_t>();
                AscendC::Add(signed_a, signed_a, fast_b_.ReinterpretCast<int16_t>(), int32_t(count));
                FastBarrier();
                AscendC::Adds(signed_a, signed_a, int16_t(-32), int32_t(count));
                FastBarrier();
                AscendC::Cast(fast_b_.ReinterpretCast<half>(), signed_a, AscendC::RoundMode::CAST_NONE, count);
                FastBarrier();
                AscendC::Cast(fast_scale_, fast_b_.ReinterpretCast<half>(), AscendC::RoundMode::CAST_NONE, count);
                AscendC::Gather(fast_a_.ReinterpretCast<half>(), raw_halves,
                    fast_d_offsets_.ReinterpretCast<uint32_t>(), base, count);
                FastBarrier();
                AscendC::Cast(fast_scratch_.ReinterpretCast<float>(), fast_a_.ReinterpretCast<half>(),
                    AscendC::RoundMode::CAST_NONE, count);
                FastBarrier();
                // Preserve the oracle's (d * signed_scale) * code rounding.
                AscendC::Mul(fast_scale_, fast_scratch_.ReinterpretCast<float>(), fast_scale_, count);
            } else {
                AscendC::Gather(fast_a_.ReinterpretCast<half>(), raw_halves,
                    fast_d_offsets_.ReinterpretCast<uint32_t>(), base, count);
                FastBarrier();
                AscendC::Cast(fast_scale_, fast_a_.ReinterpretCast<half>(), AscendC::RoundMode::CAST_NONE, count);
            }
            FastBarrier();
            AscendC::Mul(result, fast_scale_, result, count);
            FastBarrier();
        }
        return true;
    }
