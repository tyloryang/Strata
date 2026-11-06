// Decoder private methods for IQ2_XS (17) and IQ2_S (22). Model bytes,
// codebook indices, signs and scales are unpacked entirely by vector ops.
// At capacity 8192 the extra maps/factors use 46,368 bytes; only the active
// dtype initializes these buffers. Final FP32 order remains
// ((d * (0.5 + scale_nibble)) * 0.25) * signed_grid_value.
    AscendC::TBuf<AscendC::QuePosition::VECCALC> iq2_maps_buffer_, iq2_factors_buffer_,
        iq2_powers_buffer_, iq2_sign_table_buffer_;
    AscendC::LocalTensor<uint32_t> iq2_maps_;
    AscendC::LocalTensor<half> iq2_factors_, iq2_powers_, iq2_sign_table_;
    uint32_t iq2_capacity_ = 0;

    __aicore__ inline void Iq2Barrier() { AscendC::PipeBarrier<PIPE_V>(); }
    __aicore__ inline void Iq2AndMask(const AscendC::LocalTensor<uint32_t>& value,
        uint16_t bits, uint32_t count, const AscendC::LocalTensor<uint16_t>& scratch) {
        AscendC::Duplicate(scratch, bits, int32_t(2 * count));
        Iq2Barrier();
        // All indices are less than 65536. Their upper halfwords are zero.
        AscendC::And(value.ReinterpretCast<uint16_t>(), value.ReinterpretCast<uint16_t>(),
            scratch, int32_t(2 * count));
        Iq2Barrier();
    }

    __aicore__ inline void InitVectorIq2(int type, uint32_t capacity) {
        iq2_capacity_ = capacity;
        const uint32_t words = capacity / 4, groups = capacity / 16;
        const int32_t bytes = type == 17 ? 74 : 82;
        pipe_.InitBuffer(iq2_maps_buffer_, (4 * words + 2 * groups) * sizeof(uint32_t));
        pipe_.InitBuffer(iq2_factors_buffer_, (2 * words + groups) * sizeof(half));
        pipe_.InitBuffer(iq2_powers_buffer_, 32);
        pipe_.InitBuffer(iq2_sign_table_buffer_, 128 * sizeof(half));
        iq2_maps_ = iq2_maps_buffer_.Get<uint32_t>();
        iq2_factors_ = iq2_factors_buffer_.Get<half>();
        iq2_powers_ = iq2_powers_buffer_.Get<half>();
        iq2_sign_table_ = iq2_sign_table_buffer_.Get<half>();
        for (uint32_t bit = 0; bit < 8; ++bit)
            iq2_powers_.SetValue(bit, half(1.0f / float(int32_t(1u << bit))));
        Sync<AscendC::HardEvent::S_V>();
        if (type == 17) {
            // The 128 byte sign codebook is immutable format data already
            // copied by InitMatvec; wait before its one-time numeric expansion.
            Sync<AscendC::HardEvent::MTE2_V>();
            AscendC::Cast(iq2_sign_table_, table_local_[SIGNS], AscendC::RoundMode::CAST_NONE, uint32_t(128));
            Iq2Barrier();
        }
        auto work = output_local_.ReinterpretCast<uint32_t>();
        auto index = work, block = work[words];
        auto mask = work[3 * words].ReinterpretCast<uint16_t>();
        auto low_offsets = iq2_maps_;
        auto high_offsets = iq2_maps_[words];
        auto sign_offsets = iq2_maps_[2 * words];
        auto grid_halves = iq2_maps_[3 * words];
        auto d_offsets = iq2_maps_[4 * words];
        auto scale_offsets = iq2_maps_[4 * words + groups];

        AscendC::CreateVecIndex(index.ReinterpretCast<int32_t>(), int32_t(0), words);
        Iq2Barrier();
        AscendC::ShiftRight(block, index, uint32_t(6), int32_t(words));
        Iq2Barrier();
        AscendC::Muls(block.ReinterpretCast<int32_t>(), block.ReinterpretCast<int32_t>(), bytes, int32_t(words));
        Iq2Barrier();
        Iq2AndMask(index, uint16_t(63), words, mask);
        AscendC::ShiftRight(index, index, uint32_t(1), int32_t(words));
        Iq2Barrier();
        if (type == 17) {
            // One 16-bit packed index per eight output values. Keeping its
            // bytes separate avoids a lossy uint16 -> FP16 conversion.
            AscendC::ShiftLeft(index, index, uint32_t(1), int32_t(words));
            Iq2Barrier();
            AscendC::Add(low_offsets.ReinterpretCast<int32_t>(), block.ReinterpretCast<int32_t>(), index.ReinterpretCast<int32_t>(), int32_t(words));
            AscendC::Add(high_offsets.ReinterpretCast<int32_t>(), block.ReinterpretCast<int32_t>(), index.ReinterpretCast<int32_t>(), int32_t(words));
            AscendC::Add(sign_offsets.ReinterpretCast<int32_t>(), block.ReinterpretCast<int32_t>(), index.ReinterpretCast<int32_t>(), int32_t(words));
            Iq2Barrier();
            AscendC::Adds(low_offsets.ReinterpretCast<int32_t>(), low_offsets.ReinterpretCast<int32_t>(), int32_t(2), int32_t(words));
            AscendC::Adds(high_offsets.ReinterpretCast<int32_t>(), high_offsets.ReinterpretCast<int32_t>(), int32_t(3), int32_t(words));
            AscendC::Adds(sign_offsets.ReinterpretCast<int32_t>(), sign_offsets.ReinterpretCast<int32_t>(), int32_t(3), int32_t(words));
        } else {
            AscendC::Add(low_offsets.ReinterpretCast<int32_t>(), block.ReinterpretCast<int32_t>(), index.ReinterpretCast<int32_t>(), int32_t(words));
            AscendC::Add(sign_offsets.ReinterpretCast<int32_t>(), block.ReinterpretCast<int32_t>(), index.ReinterpretCast<int32_t>(), int32_t(words));
            Iq2Barrier();
            AscendC::Adds(low_offsets.ReinterpretCast<int32_t>(), low_offsets.ReinterpretCast<int32_t>(), int32_t(2), int32_t(words));
            AscendC::Adds(sign_offsets.ReinterpretCast<int32_t>(), sign_offsets.ReinterpretCast<int32_t>(), int32_t(34), int32_t(words));
            AscendC::ShiftRight(index, index, uint32_t(2), int32_t(words));
            Iq2Barrier();
            AscendC::Add(high_offsets.ReinterpretCast<int32_t>(), block.ReinterpretCast<int32_t>(), index.ReinterpretCast<int32_t>(), int32_t(words));
            Iq2Barrier();
            AscendC::Adds(high_offsets.ReinterpretCast<int32_t>(), high_offsets.ReinterpretCast<int32_t>(), int32_t(66), int32_t(words));
        }
        Iq2Barrier();
        // raw_values represents each source byte by an FP16 value.
        AscendC::ShiftLeft(low_offsets, low_offsets, uint32_t(1), int32_t(words));
        AscendC::ShiftLeft(high_offsets, high_offsets, uint32_t(1), int32_t(words));
        AscendC::ShiftLeft(sign_offsets, sign_offsets, uint32_t(1), int32_t(words));
        Iq2Barrier();

        AscendC::CreateVecIndex(index.ReinterpretCast<int32_t>(), int32_t(0), words);
        Iq2Barrier();
        Iq2AndMask(index, uint16_t(1), words, mask);
        AscendC::ShiftLeft(grid_halves, index, uint32_t(2), int32_t(words));
        Iq2Barrier();
        AscendC::ShiftLeft(index, index, uint32_t(3), int32_t(words));
        Iq2Barrier();
        AscendC::Gather(iq2_factors_[words], iq2_powers_, index, uint32_t(0), words);
        Iq2Barrier();
        if (type == 22) {
            AscendC::CreateVecIndex(index.ReinterpretCast<int32_t>(), int32_t(0), words);
            Iq2Barrier();
            AscendC::ShiftRight(index, index, uint32_t(1), int32_t(words));
            Iq2Barrier();
            Iq2AndMask(index, uint16_t(3), words, mask);
            AscendC::ShiftLeft(index, index, uint32_t(2), int32_t(words));
            Iq2Barrier();
            AscendC::Gather(iq2_factors_, iq2_powers_, index, uint32_t(0), words);
            Iq2Barrier();
        }

        AscendC::CreateVecIndex(index.ReinterpretCast<int32_t>(), int32_t(0), groups);
        Iq2Barrier();
        AscendC::ShiftRight(d_offsets, index, uint32_t(4), int32_t(groups));
        Iq2Barrier();
        AscendC::Muls(d_offsets.ReinterpretCast<int32_t>(), d_offsets.ReinterpretCast<int32_t>(), bytes, int32_t(groups));
        Iq2Barrier();
        Iq2AndMask(index, uint16_t(15), groups, mask);
        AscendC::ShiftRight(index, index, uint32_t(1), int32_t(groups));
        Iq2Barrier();
        AscendC::Add(scale_offsets.ReinterpretCast<int32_t>(), d_offsets.ReinterpretCast<int32_t>(), index.ReinterpretCast<int32_t>(), int32_t(groups));
        Iq2Barrier();
        AscendC::Adds(scale_offsets.ReinterpretCast<int32_t>(), scale_offsets.ReinterpretCast<int32_t>(), int32_t(type == 17 ? 66 : 74), int32_t(groups));
        Iq2Barrier();
        AscendC::ShiftLeft(scale_offsets, scale_offsets, uint32_t(1), int32_t(groups));
        AscendC::CreateVecIndex(index.ReinterpretCast<int32_t>(), int32_t(0), groups);
        Iq2Barrier();
        Iq2AndMask(index, uint16_t(1), groups, mask);
        AscendC::ShiftLeft(index, index, uint32_t(3), int32_t(groups));
        Iq2Barrier();
        AscendC::Gather(iq2_factors_[2 * words], iq2_powers_, index, uint32_t(0), groups);
        Iq2Barrier();
    }

    __aicore__ inline void DecodeVectorIq2(int type, uint32_t elements) {
        const uint32_t words = elements / 4, groups = elements / 16;
        const uint32_t capacity_words = iq2_capacity_ / 4, capacity_groups = iq2_capacity_ / 16;
        const uint32_t bytes = type == 17 ? 74 : 82;
        auto work = output_local_.ReinterpretCast<uint32_t>();
        auto index = work, tmp = work[capacity_words], sign = work[2 * capacity_words];
        auto mask = work[3 * capacity_words].ReinterpretCast<uint16_t>();
        auto raw_values = scales_local_.ReinterpretCast<half>();
        auto low = halves_local_, other = halves_local_[capacity_words];
        auto codes = codes_local_.ReinterpretCast<uint32_t>();
        AscendC::Cast(raw_values, raw_local_, AscendC::RoundMode::CAST_NONE, elements / 256 * bytes);
        Iq2Barrier();
        AscendC::Gather(low, raw_values, iq2_maps_, uint32_t(0), words);
        AscendC::Gather(other, raw_values, iq2_maps_[capacity_words], uint32_t(0), words);
        Iq2Barrier();
        AscendC::Cast(index.ReinterpretCast<int32_t>(), low, AscendC::RoundMode::CAST_RINT, words);
        if (type == 22) AscendC::Mul(other, other, iq2_factors_, words);
        Iq2Barrier();
        AscendC::Cast(tmp.ReinterpretCast<int32_t>(), other, AscendC::RoundMode::CAST_FLOOR, words);
        Iq2Barrier();
        Iq2AndMask(tmp, uint16_t(type == 17 ? 1 : 3), words, mask);
        AscendC::ShiftLeft(tmp, tmp, uint32_t(8), int32_t(words));
        Iq2Barrier();
        AscendC::Add(index.ReinterpretCast<int32_t>(), index.ReinterpretCast<int32_t>(), tmp.ReinterpretCast<int32_t>(), int32_t(words));
        Iq2Barrier();
        AscendC::ShiftLeft(index, index, uint32_t(3), int32_t(words));
        Iq2Barrier();
        AscendC::Add(index.ReinterpretCast<int32_t>(), index.ReinterpretCast<int32_t>(),
            iq2_maps_[3 * capacity_words].ReinterpretCast<int32_t>(), int32_t(words));
        Iq2Barrier();
        AscendC::Gather(codes, table_local_.ReinterpretCast<uint32_t>()[uint32_t(type == 17 ? IQ2XS : IQ2S) / 4],
            index, uint32_t(0), words);
        AscendC::Gather(other, raw_values, iq2_maps_[2 * capacity_words], uint32_t(0), words);
        Iq2Barrier();
        if (type == 17) {
            AscendC::Muls(other, other, half(0.5f), int32_t(words));
            Iq2Barrier();
            AscendC::Cast(tmp.ReinterpretCast<int32_t>(), other, AscendC::RoundMode::CAST_FLOOR, words);
            Iq2Barrier();
            AscendC::ShiftLeft(tmp, tmp, uint32_t(1), int32_t(words));
            Iq2Barrier();
            AscendC::Gather(other, iq2_sign_table_, tmp, uint32_t(0), words);
            Iq2Barrier();
        }
        AscendC::Mul(other, other, iq2_factors_[capacity_words], words);
        Iq2Barrier();
        AscendC::Cast(index.ReinterpretCast<int32_t>(), other, AscendC::RoundMode::CAST_FLOOR, words);
        Iq2Barrier();
        Iq2AndMask(index, uint16_t(15), words, mask);
        AscendC::ShiftLeft(index, index, uint32_t(2), int32_t(words));
        Iq2Barrier();
        AscendC::Gather(sign, sign_masks_local_, index, uint32_t(0), words);
        Iq2Barrier();
        AscendC::Or(index.ReinterpretCast<uint16_t>(), codes.ReinterpretCast<uint16_t>(), sign.ReinterpretCast<uint16_t>(), int32_t(2 * words));
        AscendC::And(tmp.ReinterpretCast<uint16_t>(), codes.ReinterpretCast<uint16_t>(), sign.ReinterpretCast<uint16_t>(), int32_t(2 * words));
        Iq2Barrier();
        AscendC::Sub(codes.ReinterpretCast<int32_t>(), index.ReinterpretCast<int32_t>(), tmp.ReinterpretCast<int32_t>(), int32_t(words));
        AscendC::Duplicate(mask, uint16_t(0x0101), int32_t(2 * words));
        Iq2Barrier();
        AscendC::And(tmp.ReinterpretCast<uint16_t>(), sign.ReinterpretCast<uint16_t>(), mask, int32_t(2 * words));
        Iq2Barrier();
        AscendC::Add(codes.ReinterpretCast<int32_t>(), codes.ReinterpretCast<int32_t>(), tmp.ReinterpretCast<int32_t>(), int32_t(words));
        Iq2Barrier();

        AscendC::Gather(low, raw_local_.ReinterpretCast<half>(), iq2_maps_[4 * capacity_words], uint32_t(0), groups);
        AscendC::Gather(other, raw_values, iq2_maps_[4 * capacity_words + capacity_groups], uint32_t(0), groups);
        Iq2Barrier();
        AscendC::Cast(index.ReinterpretCast<float>(), low, AscendC::RoundMode::CAST_NONE, groups);
        AscendC::Mul(other, other, iq2_factors_[2 * capacity_words], groups);
        Iq2Barrier();
        AscendC::Cast(tmp.ReinterpretCast<int32_t>(), other, AscendC::RoundMode::CAST_FLOOR, groups);
        Iq2Barrier();
        Iq2AndMask(tmp, uint16_t(15), groups, mask);
        AscendC::Cast(sign.ReinterpretCast<float>(), tmp.ReinterpretCast<int32_t>(), AscendC::RoundMode::CAST_RINT, groups);
        Iq2Barrier();
        AscendC::Adds(sign.ReinterpretCast<float>(), sign.ReinterpretCast<float>(), 0.5f, int32_t(groups));
        Iq2Barrier();
        AscendC::Mul(index.ReinterpretCast<float>(), index.ReinterpretCast<float>(), sign.ReinterpretCast<float>(), groups);
        Iq2Barrier();
        AscendC::Muls(index.ReinterpretCast<float>(), index.ReinterpretCast<float>(), 0.25f, int32_t(groups));
        Iq2Barrier();
        // The temporary index range spans the now-dead tmp/sign regions.
        AscendC::CreateVecIndex(tmp.ReinterpretCast<int32_t>(), int32_t(0), elements / 2);
        Iq2Barrier();
        AscendC::ShiftRight(tmp, tmp, uint32_t(3), int32_t(elements / 2));
        Iq2Barrier();
        AscendC::ShiftLeft(tmp, tmp, uint32_t(2), int32_t(elements / 2));
        Iq2Barrier();
        AscendC::Gather(scales_local_, index.ReinterpretCast<float>(), tmp, uint32_t(0), elements / 2);
        Iq2Barrier();
        AscendC::Cast(halves_local_, codes_local_, AscendC::RoundMode::CAST_NONE, elements);
        Iq2Barrier();
        AscendC::Cast(output_local_, halves_local_, AscendC::RoundMode::CAST_NONE, elements);
        Iq2Barrier();
        for (uint32_t group = 0; group < groups; group += 128) {
            const uint8_t repeats = uint8_t(groups - group < 128 ? groups - group : 128);
            AscendC::Mul(output_local_[group * 16], output_local_[group * 16], scales_local_[group * 8],
                uint64_t(16), repeats, AscendC::BinaryRepeatParams{1, 1, 0, 2, 2, 1});
        }
        Iq2Barrier();
    }
