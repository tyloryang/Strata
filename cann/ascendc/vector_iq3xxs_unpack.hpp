// Decoder private methods for IQ2_XXS (type 16) and IQ3_XXS (type 18).
// Both formats encode four 7-bit sign-table indices and a 4-bit scale in each
// 32-value chunk's aux word. Only the grid layout and final scale differ.
// Additional UB at capacity 8192: 31008 bytes. Existing output/half/scales
// buffers are scratch until the final signed-code conversion and scale Mul.
    AscendC::TBuf<AscendC::QuePosition::VECCALC> xxs_maps_buffer_, xxs_factors_buffer_,
        xxs_constants_buffer_, xxs_signs_buffer_;
    AscendC::LocalTensor<uint32_t> xxs_maps_;
    AscendC::LocalTensor<half> xxs_factors_, xxs_constants_, xxs_signs_;
    uint32_t xxs_capacity_ = 0;

    __aicore__ inline void XXSBarrier() { AscendC::PipeBarrier<PIPE_V>(); }

    // All integer values masked here are nonnegative and fit their low half.
    // Reinterpreting as u16 avoids the A2 uint32 And/Or compatibility path.
    __aicore__ inline void XXSAndMask(const AscendC::LocalTensor<uint32_t>& value,
        uint16_t mask, uint32_t count, const AscendC::LocalTensor<uint16_t>& scratch) {
        AscendC::Duplicate(scratch, mask, int32_t(2 * count));
        XXSBarrier();
        AscendC::And(value.ReinterpretCast<uint16_t>(), value.ReinterpretCast<uint16_t>(),
            scratch, int32_t(2 * count));
        XXSBarrier();
    }

    __aicore__ inline void InitVectorXXS(int type, uint32_t capacity) {
        xxs_capacity_ = capacity;
        const uint32_t words = capacity / 4, groups = capacity / 32;
        const int32_t block_bytes = type == 16 ? 66 : 98;
        pipe_.InitBuffer(xxs_maps_buffer_, (2 * words + 2 * groups) * sizeof(uint32_t));
        pipe_.InitBuffer(xxs_factors_buffer_, 3 * words * sizeof(half));
        pipe_.InitBuffer(xxs_constants_buffer_, 32);
        pipe_.InitBuffer(xxs_signs_buffer_, 128 * sizeof(half));
        xxs_maps_ = xxs_maps_buffer_.Get<uint32_t>();
        xxs_factors_ = xxs_factors_buffer_.Get<half>();
        xxs_constants_ = xxs_constants_buffer_.Get<half>();
        xxs_signs_ = xxs_signs_buffer_.Get<half>();
        // Format constants only. 7*l has byte shifts 0,7,6,5 for l=0,1,2,3.
        // A two-byte sign field is reconstructed numerically, avoiding both
        // unaligned u32 Gather and unsupported per-lane variable shifts.
        xxs_constants_.SetValue(0, half(1.0f));
        xxs_constants_.SetValue(1, half(1.0f / 128.0f));
        xxs_constants_.SetValue(2, half(1.0f / 64.0f));
        xxs_constants_.SetValue(3, half(1.0f / 32.0f));
        xxs_constants_.SetValue(4, half(256.0f));
        xxs_constants_.SetValue(5, half(2.0f));
        xxs_constants_.SetValue(6, half(4.0f));
        xxs_constants_.SetValue(7, half(8.0f));
        xxs_constants_.SetValue(8, half(1.0f));
        xxs_constants_.SetValue(9, half(1.0f / 16.0f));
        Sync<AscendC::HardEvent::S_V>();
        // InitMatvec has issued SIGNS and grid-table DMA before this call.
        Sync<AscendC::HardEvent::MTE2_V>();
        AscendC::Cast(xxs_signs_, table_local_[SIGNS], AscendC::RoundMode::CAST_NONE, uint32_t(128));
        XXSBarrier();

        auto work = output_local_.ReinterpretCast<uint32_t>();
        auto index = work, block = work[words], lane = work[2 * words];
        auto mask = work[3 * words].ReinterpretCast<uint16_t>();
        auto grid_offsets = xxs_maps_, aux_offsets = xxs_maps_[words];
        auto d_offsets = xxs_maps_[2 * words], scale_offsets = xxs_maps_[2 * words + groups];
        AscendC::CreateVecIndex(index.ReinterpretCast<int32_t>(), int32_t(0), words);
        XXSBarrier();
        AscendC::ShiftRight(block, index, uint32_t(6), int32_t(words));
        XXSBarrier();
        AscendC::Muls(block.ReinterpretCast<int32_t>(), block.ReinterpretCast<int32_t>(), block_bytes, int32_t(words));
        XXSBarrier();
        XXSAndMask(index, uint16_t(63), words, mask);
        if (type == 16) {
            AscendC::ShiftRight(lane, index, uint32_t(1), int32_t(words));
            XXSBarrier();
            XXSAndMask(lane, uint16_t(3), words, mask);
            AscendC::ShiftRight(grid_offsets, index, uint32_t(3), int32_t(words));
            XXSBarrier();
            AscendC::ShiftLeft(grid_offsets, grid_offsets, uint32_t(3), int32_t(words));
            XXSBarrier();
            AscendC::Add(grid_offsets.ReinterpretCast<int32_t>(), grid_offsets.ReinterpretCast<int32_t>(), lane.ReinterpretCast<int32_t>(), int32_t(words));
            XXSBarrier();
            AscendC::Add(grid_offsets.ReinterpretCast<int32_t>(), grid_offsets.ReinterpretCast<int32_t>(), block.ReinterpretCast<int32_t>(), int32_t(words));
        } else {
            AscendC::Add(grid_offsets.ReinterpretCast<int32_t>(), block.ReinterpretCast<int32_t>(), index.ReinterpretCast<int32_t>(), int32_t(words));
        }
        XXSBarrier();
        AscendC::Adds(grid_offsets.ReinterpretCast<int32_t>(), grid_offsets.ReinterpretCast<int32_t>(), int32_t(2), int32_t(words));
        XXSBarrier();
        AscendC::ShiftLeft(grid_offsets, grid_offsets, uint32_t(1), int32_t(words));
        XXSBarrier();

        AscendC::ShiftRight(aux_offsets, index, uint32_t(3), int32_t(words));
        XXSBarrier();
        AscendC::ShiftLeft(aux_offsets, aux_offsets, uint32_t(type == 16 ? 3 : 2), int32_t(words));
        XXSBarrier();
        AscendC::Add(aux_offsets.ReinterpretCast<int32_t>(), aux_offsets.ReinterpretCast<int32_t>(), block.ReinterpretCast<int32_t>(), int32_t(words));
        AscendC::ShiftRight(lane, index, uint32_t(1), int32_t(words));
        XXSBarrier();
        XXSAndMask(lane, uint16_t(3), words, mask);
        AscendC::Muls(lane.ReinterpretCast<int32_t>(), lane.ReinterpretCast<int32_t>(), int32_t(7), int32_t(words));
        XXSBarrier();
        AscendC::ShiftRight(lane, lane, uint32_t(3), int32_t(words));
        XXSBarrier();
        AscendC::Add(aux_offsets.ReinterpretCast<int32_t>(), aux_offsets.ReinterpretCast<int32_t>(), lane.ReinterpretCast<int32_t>(), int32_t(words));
        XXSBarrier();
        AscendC::Adds(aux_offsets.ReinterpretCast<int32_t>(), aux_offsets.ReinterpretCast<int32_t>(), int32_t(type == 16 ? 6 : 66), int32_t(words));
        XXSBarrier();
        AscendC::ShiftLeft(aux_offsets, aux_offsets, uint32_t(1), int32_t(words));
        XXSBarrier();

        // Factor l is selected for each 4-code word; each 8-code sign byte uses
        // the same l twice. The final nibble selector alternates by word.
        XXSAndMask(index, uint16_t(7), words, mask);
        AscendC::ShiftRight(index, index, uint32_t(1), int32_t(words));
        XXSBarrier();
        AscendC::ShiftLeft(index, index, uint32_t(1), int32_t(words));
        XXSBarrier();
        AscendC::Gather(xxs_factors_, xxs_constants_, index, uint32_t(0), words);
        AscendC::Gather(xxs_factors_[words], xxs_constants_, index, uint32_t(8), words);
        XXSBarrier();
        AscendC::CreateVecIndex(index.ReinterpretCast<int32_t>(), int32_t(0), words);
        XXSBarrier();
        XXSAndMask(index, uint16_t(1), words, mask);
        AscendC::ShiftLeft(index, index, uint32_t(1), int32_t(words));
        XXSBarrier();
        AscendC::Gather(xxs_factors_[2 * words], xxs_constants_, index, uint32_t(16), words);
        XXSBarrier();

        AscendC::CreateVecIndex(index.ReinterpretCast<int32_t>(), int32_t(0), groups);
        XXSBarrier();
        AscendC::ShiftRight(d_offsets, index, uint32_t(3), int32_t(groups));
        XXSBarrier();
        AscendC::Muls(d_offsets.ReinterpretCast<int32_t>(), d_offsets.ReinterpretCast<int32_t>(), block_bytes, int32_t(groups));
        XXSBarrier();
        XXSAndMask(index, uint16_t(7), groups, mask);
        AscendC::ShiftLeft(scale_offsets, index, uint32_t(type == 16 ? 3 : 2), int32_t(groups));
        XXSBarrier();
        AscendC::Add(scale_offsets.ReinterpretCast<int32_t>(), scale_offsets.ReinterpretCast<int32_t>(), d_offsets.ReinterpretCast<int32_t>(), int32_t(groups));
        XXSBarrier();
        AscendC::Adds(scale_offsets.ReinterpretCast<int32_t>(), scale_offsets.ReinterpretCast<int32_t>(), int32_t(type == 16 ? 9 : 69), int32_t(groups));
        XXSBarrier();
        AscendC::ShiftLeft(scale_offsets, scale_offsets, uint32_t(1), int32_t(groups));
        XXSBarrier();
    }

    // The caller orders compressed-input DMA with MTE2_V. Scalar accesses
    // below are dimensions/type metadata only, never packed model values.
    __aicore__ inline void DecodeVectorXXS(int type, uint32_t elements) {
        const uint32_t words = elements / 4, groups = elements / 32;
        const uint32_t capacity_words = xxs_capacity_ / 4, capacity_groups = xxs_capacity_ / 32;
        auto work = output_local_.ReinterpretCast<uint32_t>();
        auto index = work, tmp = work[capacity_words], sign = work[2 * capacity_words];
        auto mask = work[3 * capacity_words].ReinterpretCast<uint16_t>();
        auto raw_values = scales_local_.ReinterpretCast<half>();
        auto low = halves_local_, high = halves_local_[capacity_words];
        auto codes = codes_local_.ReinterpretCast<uint32_t>();
        const uint32_t raw_count = elements / 256 * uint32_t(type == 16 ? 66 : 98);
        AscendC::Cast(raw_values, raw_local_, AscendC::RoundMode::CAST_NONE, raw_count);
        XXSBarrier();
        AscendC::Gather(low, raw_values, xxs_maps_, uint32_t(0), words);
        XXSBarrier();
        AscendC::Cast(index.ReinterpretCast<int32_t>(), low, AscendC::RoundMode::CAST_RINT, words);
        XXSBarrier();
        AscendC::ShiftLeft(index, index, uint32_t(type == 16 ? 3 : 2), int32_t(words));
        XXSBarrier();
        if (type == 16) {
            AscendC::CreateVecIndex(tmp.ReinterpretCast<int32_t>(), int32_t(0), words);
            XXSBarrier();
            XXSAndMask(tmp, uint16_t(1), words, mask);
            AscendC::ShiftLeft(tmp, tmp, uint32_t(2), int32_t(words));
            XXSBarrier();
            AscendC::Add(index.ReinterpretCast<int32_t>(), index.ReinterpretCast<int32_t>(), tmp.ReinterpretCast<int32_t>(), int32_t(words));
            XXSBarrier();
        }
        const uint32_t grid = type == 16 ? IQ2XXS : IQ3XXS;
        AscendC::Gather(codes, table_local_.ReinterpretCast<uint32_t>()[grid / 4], index, uint32_t(0), words);
        // A 7-bit sign field occupies at most two bytes. Each half product is
        // exact (maximum 255*256=65280), then integer Add restores the field.
        AscendC::Gather(low, raw_values, xxs_maps_[capacity_words], uint32_t(0), words);
        AscendC::Gather(high, raw_values, xxs_maps_[capacity_words], uint32_t(2), words);
        XXSBarrier();
        AscendC::Mul(low, low, xxs_factors_, words);
        AscendC::Mul(high, high, xxs_factors_[capacity_words], words);
        XXSBarrier();
        AscendC::Cast(index.ReinterpretCast<int32_t>(), low, AscendC::RoundMode::CAST_FLOOR, words);
        AscendC::Cast(tmp.ReinterpretCast<int32_t>(), high, AscendC::RoundMode::CAST_RINT, words);
        XXSBarrier();
        AscendC::Add(index.ReinterpretCast<int32_t>(), index.ReinterpretCast<int32_t>(), tmp.ReinterpretCast<int32_t>(), int32_t(words));
        XXSBarrier();
        XXSAndMask(index, uint16_t(127), words, mask);
        AscendC::ShiftLeft(index, index, uint32_t(1), int32_t(words));
        XXSBarrier();
        AscendC::Gather(low, xxs_signs_, index, uint32_t(0), words);
        XXSBarrier();
        AscendC::Mul(low, low, xxs_factors_[2 * capacity_words], words);
        XXSBarrier();
        AscendC::Cast(index.ReinterpretCast<int32_t>(), low, AscendC::RoundMode::CAST_FLOOR, words);
        XXSBarrier();
        XXSAndMask(index, uint16_t(15), words, mask);
        AscendC::ShiftLeft(index, index, uint32_t(2), int32_t(words));
        XXSBarrier();
        AscendC::Gather(sign, sign_masks_local_, index, uint32_t(0), words);
        XXSBarrier();
        AscendC::Or(index.ReinterpretCast<uint16_t>(), codes.ReinterpretCast<uint16_t>(), sign.ReinterpretCast<uint16_t>(), int32_t(2 * words));
        AscendC::And(tmp.ReinterpretCast<uint16_t>(), codes.ReinterpretCast<uint16_t>(), sign.ReinterpretCast<uint16_t>(), int32_t(2 * words));
        XXSBarrier();
        AscendC::Sub(codes.ReinterpretCast<int32_t>(), index.ReinterpretCast<int32_t>(), tmp.ReinterpretCast<int32_t>(), int32_t(words));
        AscendC::Duplicate(mask, uint16_t(0x0101), int32_t(2 * words));
        XXSBarrier();
        AscendC::And(tmp.ReinterpretCast<uint16_t>(), sign.ReinterpretCast<uint16_t>(), mask, int32_t(2 * words));
        XXSBarrier();
        AscendC::Add(codes.ReinterpretCast<int32_t>(), codes.ReinterpretCast<int32_t>(), tmp.ReinterpretCast<int32_t>(), int32_t(words));
        XXSBarrier();

        AscendC::Gather(low, raw_local_.ReinterpretCast<half>(), xxs_maps_[2 * capacity_words], uint32_t(0), groups);
        AscendC::Gather(high, raw_values, xxs_maps_[2 * capacity_words + capacity_groups], uint32_t(0), groups);
        XXSBarrier();
        AscendC::Cast(index.ReinterpretCast<float>(), low, AscendC::RoundMode::CAST_NONE, groups);
        AscendC::Muls(high, high, half(1.0f / 16.0f), int32_t(groups));
        XXSBarrier();
        AscendC::Cast(tmp.ReinterpretCast<int32_t>(), high, AscendC::RoundMode::CAST_FLOOR, groups);
        XXSBarrier();
        AscendC::Cast(sign.ReinterpretCast<float>(), tmp.ReinterpretCast<int32_t>(), AscendC::RoundMode::CAST_RINT, groups);
        XXSBarrier();
        AscendC::Adds(sign.ReinterpretCast<float>(), sign.ReinterpretCast<float>(), 0.5f, int32_t(groups));
        XXSBarrier();
        // Preserve both FP32 multiplies in d*(0.5f+aux>>28)*format_factor.
        AscendC::Mul(index.ReinterpretCast<float>(), index.ReinterpretCast<float>(), sign.ReinterpretCast<float>(), groups);
        XXSBarrier();
        AscendC::Muls(index.ReinterpretCast<float>(), index.ReinterpretCast<float>(), type == 16 ? 0.25f : 0.5f, int32_t(groups));
        XXSBarrier();
        AscendC::CreateVecIndex(tmp.ReinterpretCast<int32_t>(), int32_t(0), words);
        XXSBarrier();
        AscendC::ShiftRight(tmp, tmp, uint32_t(3), int32_t(words));
        XXSBarrier();
        AscendC::ShiftLeft(tmp, tmp, uint32_t(2), int32_t(words));
        XXSBarrier();
        AscendC::Gather(scales_local_, index.ReinterpretCast<float>(), tmp, uint32_t(0), words);
        XXSBarrier();
        AscendC::Cast(halves_local_, codes_local_, AscendC::RoundMode::CAST_NONE, elements);
        XXSBarrier();
        AscendC::Cast(output_local_, halves_local_, AscendC::RoundMode::CAST_NONE, elements);
        XXSBarrier();
        for (uint32_t group = 0; group < groups; group += 128) {
            const uint8_t repeats = uint8_t(groups - group < 128 ? groups - group : 128);
            AscendC::Mul(output_local_[group * 32], output_local_[group * 32], scales_local_[group * 8],
                uint64_t(32), repeats, AscendC::BinaryRepeatParams{1, 1, 0, 4, 4, 1});
        }
        XXSBarrier();
    }
