// Included in Decoder's public section. The host validates dtype geometry,
// cols % 8 == 0 and whole-tensor quant-block alignment. Rows may cross blocks.
    static constexpr uint32_t MATVEC_ELEMENTS = 8192;
    static constexpr uint32_t MATVEC_X_CACHE = 10240;

    __aicore__ inline void InitMatvec(GM_ADDR raw, GM_ADDR tables, GM_ADDR output,
        int type, uint32_t block_elements, uint32_t block_bytes) {
        const bool quant = VectorQuant(type);
        const uint32_t raw_bytes = (MATVEC_ELEMENTS / block_elements) * block_bytes;
        raw_.SetGlobalBuffer((__gm__ uint8_t *)raw);
        tables_.SetGlobalBuffer((__gm__ uint8_t *)tables);
        output_.SetGlobalBuffer((__gm__ float *)output);
        pipe_.InitBuffer(raw_buffer_, (raw_bytes + 31) / 32 * 32);
        pipe_.InitBuffer(table_buffer_, quant ? 17568 : 32);
        pipe_.InitBuffer(output_buffer_, type == 0 ? 32 : MATVEC_ELEMENTS * sizeof(float));
        pipe_.InitBuffer(codes_buffer_, quant ? MATVEC_ELEMENTS : 32);
        pipe_.InitBuffer(half_buffer_, quant ? MATVEC_ELEMENTS * sizeof(half) : 32);
        pipe_.InitBuffer(scales_buffer_, quant ? MATVEC_ELEMENTS / 2 * sizeof(float) : 32);
        pipe_.InitBuffer(minima_buffer_, (type == 12 || type == 13) ? MATVEC_ELEMENTS / 2 * sizeof(float) : 32);
        pipe_.InitBuffer(sign_masks_buffer_, 16 * sizeof(uint32_t));
        raw_local_ = raw_buffer_.Get<uint8_t>();
        table_local_ = table_buffer_.Get<uint8_t>();
        output_local_ = output_buffer_.Get<float>();
        codes_local_ = codes_buffer_.Get<int8_t>();
        halves_local_ = half_buffer_.Get<half>();
        scales_local_ = scales_buffer_.Get<float>();
        minima_local_ = minima_buffer_.Get<float>();
        sign_masks_local_ = sign_masks_buffer_.Get<uint32_t>();
        if (quant) for (uint32_t bits = 0; bits < 16; ++bits) {
            const uint32_t lanes = (bits & 1) | ((bits & 2) << 7) | ((bits & 4) << 14) | ((bits & 8) << 21);
            sign_masks_local_.SetValue(bits, lanes * 255u);
        }
        if (type == 16 || type == 17 || type == 18) LoadTable(SIGNS, 128);
        if (type == 16) LoadTable(IQ2XXS, 256 * 8);
        if (type == 17) LoadTable(IQ2XS, 512 * 8);
        if (type == 22) LoadTable(IQ2S, 1024 * 8);
        if (type == 18) LoadTable(IQ3XXS, 256 * 4);
        if (type == 21) LoadTable(IQ3S, 512 * 4);
        if (type == 20 || type == 23) LoadTable(IQ4NL, 16);
        InitVectorFast(type, MATVEC_ELEMENTS);
        if (type == 21) InitVectorIq3S(MATVEC_ELEMENTS);
        if (type == 17 || type == 22) InitVectorIq2(type, MATVEC_ELEMENTS);
        if (type == 16 || type == 18) InitVectorXXS(type, MATVEC_ELEMENTS);
    }

    __aicore__ inline AscendC::LocalTensor<float> MatvecDecode(int type,
        uint64_t first_block, uint32_t tile_blocks, uint32_t elements,
        uint32_t block_elements, uint32_t block_bytes) {
        const bool plain = type == 0 || type == 1 || type == 30;
        const uint32_t input_bytes = plain ? elements * (type == 0 ? 4 : 2) : tile_blocks * block_bytes;
        AscendC::DataCopyPad(raw_local_, raw_[first_block * block_bytes],
            AscendC::DataCopyExtParams{1, input_bytes, 0, 0, 0},
            AscendC::DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
        auto weights = output_local_;
        if (type == 0) {
            Sync<AscendC::HardEvent::MTE2_V>();
            return raw_local_.ReinterpretCast<float>();
        }
        if (IsVectorFast(type)) {
            Sync<AscendC::HardEvent::MTE2_V>();
            DecodeVectorFast(type, elements);
            return weights;
        }
        if (type == 21) {
            Sync<AscendC::HardEvent::MTE2_V>();
            DecodeVectorIq3S(elements);
            return weights;
        }
        if (type == 17 || type == 22) {
            Sync<AscendC::HardEvent::MTE2_V>();
            DecodeVectorIq2(type, elements);
            return weights;
        }
        if (type == 16 || type == 18) {
            Sync<AscendC::HardEvent::MTE2_V>();
            DecodeVectorXXS(type, elements);
            return weights;
        }
        if (type == 1 || type == 30) {
            Sync<AscendC::HardEvent::MTE2_V>();
            if (type == 1)
                AscendC::Cast(output_local_, raw_local_.ReinterpretCast<half>(), AscendC::RoundMode::CAST_NONE, elements);
            else
                AscendC::Cast(output_local_, raw_local_.ReinterpretCast<bfloat16_t>(), AscendC::RoundMode::CAST_NONE, elements);
        } else {
            Sync<AscendC::HardEvent::MTE2_S>();
            for (uint32_t b = 0; b < tile_blocks; ++b)
                PackBlock(type, b * block_bytes, b * block_elements);
            Sync<AscendC::HardEvent::S_V>();
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(halves_local_, codes_local_, AscendC::RoundMode::CAST_NONE, elements);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(output_local_, halves_local_, AscendC::RoundMode::CAST_NONE, elements);
            AscendC::PipeBarrier<PIPE_V>();
            const uint32_t groups = elements / 16;
            for (uint32_t g = 0; g < groups; g += 128) {
                const uint8_t repeats = uint8_t(groups - g < 128 ? groups - g : 128);
                AscendC::Mul(output_local_[g * 16], output_local_[g * 16], scales_local_[g * 8],
                    uint64_t(16), repeats, AscendC::BinaryRepeatParams{1, 1, 0, 2, 2, 1});
            }
            if (type == 12 || type == 13) {
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t g = 0; g < groups; g += 128) {
                    const uint8_t repeats = uint8_t(groups - g < 128 ? groups - g : 128);
                    AscendC::Sub(output_local_[g * 16], output_local_[g * 16], minima_local_[g * 8],
                        uint64_t(16), repeats, AscendC::BinaryRepeatParams{1, 1, 0, 2, 2, 1});
                }
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
        return weights;
    }

    __aicore__ inline void MatvecReduce(const AscendC::LocalTensor<float>& result,
        const AscendC::LocalTensor<float>& values, const AscendC::LocalTensor<float>& work,
        uint32_t elements) {
        // The model's column widths are multiples of 64. Unlike ReduceSum,
        // WholeReduceSum has no internal scalar load/store synchronization.
        if (elements % 64 != 0) {
            AscendC::ReduceSum(result, values, work, int32_t(elements));
            AscendC::PipeBarrier<PIPE_V>();
            return;
        }
        const uint32_t repeats = elements / 64;
        AscendC::WholeReduceSum(work, values, int32_t(64), int32_t(repeats), 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        if (repeats <= 64) {
            AscendC::WholeReduceSum(result, work, int32_t(repeats), 1, 1, 1, 8);
        } else {
            AscendC::WholeReduceSum(work[128], work, int32_t(64), 1, 1, 1, 8);
            AscendC::WholeReduceSum(work[136], work[64], int32_t(repeats - 64), 1, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(result, work[128], work[136], uint32_t(1));
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void RunMatvec(GM_ADDR activation, uint64_t rows, uint64_t cols,
        int type, uint32_t block_elements, uint32_t block_bytes) {
        constexpr uint32_t ROWS_PER_GROUP = 8;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> activation_buffer, reduction_work_buffer,
            reduction_buffer, row_results_buffer;
        const bool cached_x = cols <= MATVEC_X_CACHE;
        const uint32_t x_elements = cached_x ? uint32_t(cols) : MATVEC_ELEMENTS;
        pipe_.InitBuffer(activation_buffer, x_elements * sizeof(float));
        // 128 first-level FP32 sums plus two aligned second-level sums. The
        // fallback ReduceSum needs at most ceil(8192/64) temporary floats too.
        pipe_.InitBuffer(reduction_work_buffer, 1024);
        pipe_.InitBuffer(reduction_buffer, ROWS_PER_GROUP * 32);
        pipe_.InitBuffer(row_results_buffer, 32);
        auto x_local = activation_buffer.Get<float>();
        auto work = reduction_work_buffer.Get<float>();
        auto reductions = reduction_buffer.Get<float>();
        auto row_results = row_results_buffer.Get<float>();
        AscendC::GlobalTensor<float> x;
        x.SetGlobalBuffer((__gm__ float *)activation);
        if (cached_x) {
            AscendC::DataCopyPad(x_local, x,
                AscendC::DataCopyExtParams{1, x_elements * uint32_t(sizeof(float)), 0, 0, 0},
                AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
        }
        const uint64_t groups = (rows + ROWS_PER_GROUP - 1) / ROWS_PER_GROUP;
        const uint64_t cores = AscendC::GetBlockNum(), core = AscendC::GetBlockIdx();
        const uint64_t quotient = groups / cores, remainder = groups % cores;
        const uint64_t begin = core * quotient + (core < remainder ? core : remainder);
        const uint64_t end = begin + quotient + (core < remainder ? 1 : 0);
        const uint64_t total = rows * cols;
        for (uint64_t group = begin; group < end; ++group) {
            const uint64_t row_begin = group * ROWS_PER_GROUP;
            const uint32_t group_rows = uint32_t(rows - row_begin < ROWS_PER_GROUP ? rows - row_begin : ROWS_PER_GROUP);
            uint32_t local_row = 0;
            while (local_row < group_rows) {
                const uint64_t row_flat = (row_begin + local_row) * cols;
                const uint64_t first_block = row_flat / block_elements;
                const uint32_t leading = uint32_t(row_flat % block_elements);
                uint32_t tile_rows = uint32_t((MATVEC_ELEMENTS - leading) / cols);
                if (tile_rows > group_rows - local_row) tile_rows = group_rows - local_row;
                if (tile_rows) {
                    const uint32_t tile_blocks = uint32_t((leading + tile_rows * cols + block_elements - 1) / block_elements);
                    const uint64_t first_element = first_block * block_elements;
                    const uint32_t elements = uint32_t(total - first_element < uint64_t(tile_blocks) * block_elements
                        ? total - first_element : uint64_t(tile_blocks) * block_elements);
                    Sync<AscendC::HardEvent::S_MTE2>();
                    auto weights = MatvecDecode(type, first_block, tile_blocks, elements, block_elements, block_bytes);
                    for (uint32_t r = 0; r < tile_rows; ++r) {
                        auto row = weights[leading + r * uint32_t(cols)];
                        AscendC::Mul(row, row, x_local, uint32_t(cols));
                        AscendC::PipeBarrier<PIPE_V>();
                        MatvecReduce(reductions[r * 8], row, work, uint32_t(cols));
                    }
                    Sync<AscendC::HardEvent::V_S>();
                    for (uint32_t r = 0; r < tile_rows; ++r)
                        row_results.SetValue(local_row + r, reductions.GetValue(r * 8));
                    local_row += tile_rows;
                } else {
                    // Wide rows stream chunks; only x wider than the cache is
                    // copied again. No full decoded row is stored in HBM.
                    float sum = 0.0f;
                    for (uint64_t col = 0; col < cols;) {
                        const uint64_t flat = row_flat + col;
                        const uint64_t block = flat / block_elements;
                        const uint32_t skip = uint32_t(flat % block_elements);
                        const uint32_t take = uint32_t(cols - col < MATVEC_ELEMENTS - skip
                            ? cols - col : MATVEC_ELEMENTS - skip);
                        const uint32_t tile_blocks = (skip + take + block_elements - 1) / block_elements;
                        const uint64_t first_element = block * block_elements;
                        const uint32_t elements = uint32_t(total - first_element < uint64_t(tile_blocks) * block_elements
                            ? total - first_element : uint64_t(tile_blocks) * block_elements);
                        Sync<AscendC::HardEvent::S_MTE2>();
                        if (!cached_x) AscendC::DataCopyPad(x_local, x[col],
                            AscendC::DataCopyExtParams{1, take * uint32_t(sizeof(float)), 0, 0, 0},
                            AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
                        auto weights = MatvecDecode(type, block, tile_blocks, elements, block_elements, block_bytes);
                        auto row = weights[skip];
                        AscendC::Mul(row, row, x_local[cached_x ? uint32_t(col) : 0], take);
                        AscendC::PipeBarrier<PIPE_V>();
                        MatvecReduce(reductions, row, work, take);
                        Sync<AscendC::HardEvent::V_S>();
                        sum += reductions.GetValue(0);
                        col += take;
                    }
                    row_results.SetValue(local_row++, sum);
                }
            }
            Sync<AscendC::HardEvent::S_MTE3>();
            AscendC::DataCopyPad(output_[row_begin], row_results,
                AscendC::DataCopyExtParams{1, group_rows * uint32_t(sizeof(float)), 0, 0, 0});
            Sync<AscendC::HardEvent::MTE3_S>();
        }
    }

    // Grouped MoE projections keep routing IDs and all arithmetic on the NPU.
    // The old single-matrix RunMatvec remains an independent, unchanged path.
    __aicore__ inline void RunGroupedMatvec(GM_ADDR raw, GM_ADDR activation, GM_ADDR routing,
        uint64_t rows, uint64_t cols, uint32_t selected, uint32_t experts_total,
        int type, uint32_t block_elements, uint32_t block_bytes, bool x_perexpert) {
        constexpr uint32_t ROWS_PER_GROUP = 8;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> activation_buffer, reduction_work_buffer,
            reduction_buffer, row_results_buffer, routing_buffer;
        const bool cached_x = cols <= MATVEC_X_CACHE;
        const uint32_t x_elements = cached_x ? uint32_t(cols) : MATVEC_ELEMENTS;
        pipe_.InitBuffer(activation_buffer, x_elements * sizeof(float));
        pipe_.InitBuffer(reduction_work_buffer, 1024);
        pipe_.InitBuffer(reduction_buffer, ROWS_PER_GROUP * 32);
        pipe_.InitBuffer(row_results_buffer, 32);
        pipe_.InitBuffer(routing_buffer, (selected * uint32_t(sizeof(int64_t)) + 31) / 32 * 32);
        auto x_local = activation_buffer.Get<float>();
        auto work = reduction_work_buffer.Get<float>();
        auto reductions = reduction_buffer.Get<float>();
        auto row_results = row_results_buffer.Get<float>();
        auto ids = routing_buffer.Get<int64_t>();
        AscendC::GlobalTensor<float> x;
        AscendC::GlobalTensor<int64_t> route;
        x.SetGlobalBuffer((__gm__ float *)activation);
        route.SetGlobalBuffer((__gm__ int64_t *)routing);
        // DMA rather than scalar GM GetValue avoids stale scalar cache lines
        // when TopK reuses its ID buffer on the next token.
        AscendC::DataCopyPad(ids, route,
            AscendC::DataCopyExtParams{1, selected * uint32_t(sizeof(int64_t)), 0, 0, 0},
            AscendC::DataCopyPadExtParams<int64_t>{false, 0, 0, 0});
        if (cached_x && !x_perexpert) {
            AscendC::DataCopyPad(x_local, x,
                AscendC::DataCopyExtParams{1, x_elements * uint32_t(sizeof(float)), 0, 0, 0},
                AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
        }
        Sync<AscendC::HardEvent::MTE2_S>();
        const uint64_t total_rows = uint64_t(selected) * rows;
        // Flatten before assigning groups. Even when rows%8!=0, different
        // cores never write portions of the same 32-byte output region.
        const uint64_t groups = (total_rows + ROWS_PER_GROUP - 1) / ROWS_PER_GROUP;
        const uint64_t cores = AscendC::GetBlockNum(), core = AscendC::GetBlockIdx();
        const uint64_t quotient = groups / cores, remainder = groups % cores;
        const uint64_t begin = core * quotient + (core < remainder ? core : remainder);
        const uint64_t end = begin + quotient + (core < remainder ? 1 : 0);
        const uint64_t total = rows * cols;
        const bool plain = type == 0 || type == 1 || type == 30;
        const uint64_t expert_bytes = plain ? total * uint64_t(type == 0 ? 4 : 2)
            : (total / block_elements) * block_bytes;
        uint32_t current_rank = selected;
        uint64_t activation_begin = 0;
        bool valid_expert = false;

        for (uint64_t group = begin; group < end; ++group) {
            const uint64_t group_begin = group * ROWS_PER_GROUP;
            const uint32_t group_rows = uint32_t(total_rows - group_begin < ROWS_PER_GROUP
                ? total_rows - group_begin : ROWS_PER_GROUP);
            uint32_t local_row = 0;
            while (local_row < group_rows) {
                const uint64_t flat_row = group_begin + local_row;
                const uint32_t rank = uint32_t(flat_row / rows);
                const uint64_t expert_row = flat_row % rows;
                const uint32_t rank_rows = uint32_t(rows - expert_row < group_rows - local_row
                    ? rows - expert_row : group_rows - local_row);
                if (rank != current_rank) {
                    current_rank = rank;
                    const int64_t expert = ids.GetValue(rank);
                    valid_expert = expert >= 0 && uint64_t(expert) < experts_total;
                    if (valid_expert) {
                        raw_.SetGlobalBuffer((__gm__ uint8_t *)raw + uint64_t(expert) * expert_bytes);
                        activation_begin = x_perexpert ? uint64_t(rank) * cols : 0;
                        if (cached_x && x_perexpert) {
                            Sync<AscendC::HardEvent::S_MTE2>();
                            AscendC::DataCopyPad(x_local, x[activation_begin],
                                AscendC::DataCopyExtParams{1, x_elements * uint32_t(sizeof(float)), 0, 0, 0},
                                AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
                        }
                    }
                }
                if (!valid_expert) {
                    // TopK-generated IDs satisfy the range contract. Still
                    // guard raw memory for direct API callers: invalid ranks
                    // produce NaNs rather than reading outside the weights.
                    for (uint32_t r = 0; r < rank_rows; ++r)
                        row_results.SetValue(local_row + r, float_bits(0x7fc00000u));
                    local_row += rank_rows;
                    continue;
                }
                const uint64_t row_flat = expert_row * cols;
                const uint64_t first_block = row_flat / block_elements;
                const uint32_t leading = uint32_t(row_flat % block_elements);
                uint32_t tile_rows = uint32_t((MATVEC_ELEMENTS - leading) / cols);
                if (tile_rows > rank_rows) tile_rows = rank_rows;
                if (tile_rows) {
                    const uint32_t tile_blocks = uint32_t((leading + tile_rows * cols + block_elements - 1) / block_elements);
                    const uint64_t first_element = first_block * block_elements;
                    const uint32_t elements = uint32_t(total - first_element < uint64_t(tile_blocks) * block_elements
                        ? total - first_element : uint64_t(tile_blocks) * block_elements);
                    Sync<AscendC::HardEvent::S_MTE2>();
                    auto weights = MatvecDecode(type, first_block, tile_blocks, elements, block_elements, block_bytes);
                    for (uint32_t r = 0; r < tile_rows; ++r) {
                        auto row = weights[leading + r * uint32_t(cols)];
                        AscendC::Mul(row, row, x_local, uint32_t(cols));
                        AscendC::PipeBarrier<PIPE_V>();
                        MatvecReduce(reductions[r * 8], row, work, uint32_t(cols));
                    }
                    Sync<AscendC::HardEvent::V_S>();
                    for (uint32_t r = 0; r < tile_rows; ++r)
                        row_results.SetValue(local_row + r, reductions.GetValue(r * 8));
                    local_row += tile_rows;
                } else {
                    float sum = 0.0f;
                    for (uint64_t col = 0; col < cols;) {
                        const uint64_t flat = row_flat + col;
                        const uint64_t block = flat / block_elements;
                        const uint32_t skip = uint32_t(flat % block_elements);
                        const uint32_t take = uint32_t(cols - col < MATVEC_ELEMENTS - skip
                            ? cols - col : MATVEC_ELEMENTS - skip);
                        const uint32_t tile_blocks = (skip + take + block_elements - 1) / block_elements;
                        const uint64_t first_element = block * block_elements;
                        const uint32_t elements = uint32_t(total - first_element < uint64_t(tile_blocks) * block_elements
                            ? total - first_element : uint64_t(tile_blocks) * block_elements);
                        Sync<AscendC::HardEvent::S_MTE2>();
                        if (!cached_x) AscendC::DataCopyPad(x_local, x[activation_begin + col],
                            AscendC::DataCopyExtParams{1, take * uint32_t(sizeof(float)), 0, 0, 0},
                            AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
                        auto weights = MatvecDecode(type, block, tile_blocks, elements, block_elements, block_bytes);
                        auto row = weights[skip];
                        AscendC::Mul(row, row, x_local[cached_x ? uint32_t(col) : 0], take);
                        AscendC::PipeBarrier<PIPE_V>();
                        MatvecReduce(reductions, row, work, take);
                        Sync<AscendC::HardEvent::V_S>();
                        sum += reductions.GetValue(0);
                        col += take;
                    }
                    row_results.SetValue(local_row++, sum);
                }
            }
            Sync<AscendC::HardEvent::S_MTE3>();
            AscendC::DataCopyPad(output_[group_begin], row_results,
                AscendC::DataCopyExtParams{1, group_rows * uint32_t(sizeof(float)), 0, 0, 0});
            Sync<AscendC::HardEvent::MTE3_S>();
        }
    }
