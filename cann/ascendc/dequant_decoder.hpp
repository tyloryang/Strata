// Included inside dequant_kernel.cpp's anonymous namespace. All packed bytes,
// codebooks and results are accessed in UB; only tile DMA touches GM.
class Decoder {
public:
    __aicore__ inline void Init(GM_ADDR raw, GM_ADDR tables, GM_ADDR output, int type) {
        raw_.SetGlobalBuffer((__gm__ uint8_t *)raw);
        tables_.SetGlobalBuffer((__gm__ uint8_t *)tables);
        output_.SetGlobalBuffer((__gm__ float *)output);
        pipe_.InitBuffer(raw_buffer_, TILE_BLOCKS * 1024);
        pipe_.InitBuffer(table_buffer_, 17568);
        pipe_.InitBuffer(output_buffer_, TILE_BLOCKS * 256 * sizeof(float));
        pipe_.InitBuffer(codes_buffer_, TILE_BLOCKS * 256);
        pipe_.InitBuffer(half_buffer_, TILE_BLOCKS * 256 * sizeof(half));
        pipe_.InitBuffer(scales_buffer_, TILE_BLOCKS * 128 * sizeof(float));
        pipe_.InitBuffer(minima_buffer_, TILE_BLOCKS * 128 * sizeof(float));
        pipe_.InitBuffer(sign_masks_buffer_, 16 * sizeof(uint32_t));
        raw_local_ = raw_buffer_.Get<uint8_t>();
        table_local_ = table_buffer_.Get<uint8_t>();
        output_local_ = output_buffer_.Get<float>();
        codes_local_ = codes_buffer_.Get<int8_t>();
        halves_local_ = half_buffer_.Get<half>();
        scales_local_ = scales_buffer_.Get<float>();
        minima_local_ = minima_buffer_.Get<float>();
        sign_masks_local_ = sign_masks_buffer_.Get<uint32_t>();
        if (VectorQuant(type)) for (uint32_t bits = 0; bits < 16; ++bits) {
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
    }

    __aicore__ inline void Run(int type, uint64_t count, uint32_t block_elements, uint32_t block_bytes) {
        const uint64_t blocks = (count + block_elements - 1) / block_elements;
        const uint64_t cores = AscendC::GetBlockNum(), core = AscendC::GetBlockIdx();
        const uint64_t quotient = blocks / cores, remainder = blocks % cores;
        const uint64_t begin = core * quotient + (core < remainder ? core : remainder);
        const uint64_t end = begin + quotient + (core < remainder ? 1 : 0);
        for (uint64_t first_block = begin; first_block < end; first_block += TILE_BLOCKS) {
            const uint32_t tile_blocks = uint32_t(end - first_block < TILE_BLOCKS ? end - first_block : TILE_BLOCKS);
            const uint64_t first_element = first_block * block_elements;
            const uint32_t elements = uint32_t(count - first_element < uint64_t(tile_blocks) * block_elements
                ? count - first_element : uint64_t(tile_blocks) * block_elements);
            const bool plain = type == 0 || type == 1 || type == 30;
            const bool vector_quant = VectorQuant(type);
            const uint32_t input_bytes = plain ? elements * (type == 0 ? 4 : 2) : tile_blocks * block_bytes;
            // Reusing UB is ordered against both scalar accesses and prior DMA.
            Sync<AscendC::HardEvent::S_MTE2>();
            AscendC::DataCopyPad(raw_local_, raw_[first_block * block_bytes],
                AscendC::DataCopyExtParams{1, input_bytes, 0, 0, 0},
                AscendC::DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
            auto result = output_local_;
            if (type == 0) {
                Sync<AscendC::HardEvent::MTE2_MTE3>();
                result = raw_local_.ReinterpretCast<float>();
            } else if (type == 1 || type == 30) {
                Sync<AscendC::HardEvent::MTE2_V>();
                if (type == 1)
                    AscendC::Cast(output_local_, raw_local_.ReinterpretCast<half>(), AscendC::RoundMode::CAST_NONE, elements);
                else
                    AscendC::Cast(output_local_, raw_local_.ReinterpretCast<bfloat16_t>(), AscendC::RoundMode::CAST_NONE, elements);
                Sync<AscendC::HardEvent::V_MTE3>();
            } else if (vector_quant) {
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
                for (uint32_t group = 0; group < groups; group += 128) {
                    const uint8_t repeats = uint8_t(groups - group < 128 ? groups - group : 128);
                    AscendC::Mul(output_local_[group * 16], output_local_[group * 16], scales_local_[group * 8],
                        uint64_t(16), repeats, AscendC::BinaryRepeatParams{1, 1, 0, 2, 2, 1});
                }
                if (type == 12 || type == 13) {
                    // Preserve the oracle's two FP32 roundings: multiply, then
                    // subtract. A fused multiply-add changes some Q4/Q5 values.
                    AscendC::PipeBarrier<PIPE_V>();
                    for (uint32_t group = 0; group < groups; group += 128) {
                        const uint8_t repeats = uint8_t(groups - group < 128 ? groups - group : 128);
                        AscendC::Sub(output_local_[group * 16], output_local_[group * 16], minima_local_[group * 8],
                            uint64_t(16), repeats, AscendC::BinaryRepeatParams{1, 1, 0, 2, 2, 1});
                    }
                }
                Sync<AscendC::HardEvent::V_MTE3>();
            } else {
                Sync<AscendC::HardEvent::MTE2_S>();
                for (uint32_t b = 0; b < tile_blocks; ++b)
                    DecodeBlock(type, b * block_bytes, b * block_elements);
                Sync<AscendC::HardEvent::S_MTE3>();
            }
            AscendC::DataCopyPad(output_[first_element], result,
                AscendC::DataCopyExtParams{1, elements * uint32_t(sizeof(float)), 0, 0, 0});
            Sync<AscendC::HardEvent::MTE3_S>();
            Sync<AscendC::HardEvent::MTE3_MTE2>();
            if (type == 1 || type == 30 || vector_quant) Sync<AscendC::HardEvent::MTE3_V>();
        }
    }

#include "fused_matvec.hpp"

private:
    static constexpr uint32_t TILE_BLOCKS = 16;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> raw_buffer_, table_buffer_, output_buffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> codes_buffer_, half_buffer_, scales_buffer_, minima_buffer_, sign_masks_buffer_;
    AscendC::GlobalTensor<uint8_t> raw_, tables_;
    AscendC::GlobalTensor<float> output_;
    AscendC::LocalTensor<uint8_t> raw_local_, table_local_;
    AscendC::LocalTensor<float> output_local_;
    AscendC::LocalTensor<int8_t> codes_local_;
    AscendC::LocalTensor<half> halves_local_;
    AscendC::LocalTensor<float> scales_local_, minima_local_;
    AscendC::LocalTensor<uint32_t> sign_masks_local_;

    template <AscendC::HardEvent Event>
    __aicore__ inline void Sync() {
        const auto id = static_cast<event_t>(pipe_.FetchEventID(Event));
        AscendC::SetFlag<Event>(id);
        AscendC::WaitFlag<Event>(id);
    }
    __aicore__ inline void LoadTable(uint32_t offset, uint32_t bytes) {
        AscendC::DataCopyPad(table_local_[offset], tables_[offset],
            AscendC::DataCopyExtParams{1, bytes, 0, 0, 0},
            AscendC::DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
    }
    __aicore__ inline uint32_t U8(uint32_t at) { return raw_local_.GetValue(at); }
    __aicore__ inline uint32_t U16(uint32_t at) { return U8(at) | (U8(at + 1) << 8); }
    __aicore__ inline uint32_t U32(uint32_t at) { return U16(at) | (U16(at + 2) << 16); }
    __aicore__ inline int I8(uint32_t value) { return value < 128 ? int(value) : int(value) - 256; }
    __aicore__ inline uint32_t Table(uint32_t at) { return table_local_.GetValue(at); }
    __aicore__ inline float D(uint32_t at) { return half_value(U16(at)); }
    __aicore__ inline void Store(uint32_t at, float value) { output_local_.SetValue(at, value); }
    __aicore__ inline float Signed(float scale, uint32_t value, uint32_t signs, uint32_t lane) {
        const float decoded = scale * float(int32_t(value));
        return signs & (1u << lane) ? -decoded : decoded;
    }
    __aicore__ inline void ScaleMin(uint32_t at, int sub, uint32_t &scale, uint32_t &minimum) {
        if (sub < 4) { scale = U8(at + sub) & 63; minimum = U8(at + sub + 4) & 63; }
        else {
            scale = (U8(at + sub + 4) & 15) | ((U8(at + sub - 4) >> 6) << 4);
            minimum = (U8(at + sub + 4) >> 4) | ((U8(at + sub) >> 6) << 4);
        }
    }

#include "dequant_vector.hpp"
#include "vector_fast_unpack.hpp"
#include "vector_iq3s_unpack.hpp"
#include "vector_iq2_unpack.hpp"
#include "vector_iq3xxs_unpack.hpp"

    __aicore__ inline void DecodeBlock(int type, uint32_t p, uint32_t out) {
        const float d = D(p + (type == 14 ? 208 : 0));
        if (type == 8) {
            for (uint32_t j = 0; j < 32; ++j) Store(out + j, d * float(I8(U8(p + 2 + j))));
            return;
        }
        if (type == 42) {
            for (uint32_t j = 0; j < 16; ++j) {
                const uint32_t packed = U8(p + 2 + j);
                for (uint32_t lane = 0; lane < 4; ++lane)
                    Store(out + j * 4 + lane, float(int32_t((packed >> (2 * lane)) & 3) - 1) * d);
            }
            return;
        }
        if (type == 12 || type == 13) {
            const float minimum = D(p + 2);
            for (int segment = 0; segment < 4; ++segment) {
                uint32_t s0, m0, s1, m1;
                ScaleMin(p + 4, 2 * segment, s0, m0);
                ScaleMin(p + 4, 2 * segment + 1, s1, m1);
                const float ds0 = d * float(int32_t(s0)), dm0 = minimum * float(int32_t(m0));
                const float ds1 = d * float(int32_t(s1)), dm1 = minimum * float(int32_t(m1));
                for (uint32_t lane = 0; lane < 32; ++lane) {
                    const uint32_t packed = U8(p + (type == 12 ? 16 : 48) + 32 * segment + lane);
                    const uint32_t high = type == 13 ? U8(p + 16 + lane) : 0;
                    const uint32_t q0 = (packed & 15) | (((high >> (2 * segment)) & 1) << 4);
                    const uint32_t q1 = (packed >> 4) | (((high >> (2 * segment + 1)) & 1) << 4);
                    Store(out + 64 * segment + lane, ds0 * float(int32_t(q0)) - dm0);
                    Store(out + 64 * segment + 32 + lane, ds1 * float(int32_t(q1)) - dm1);
                }
            }
            return;
        }
        if (type == 14) {
            for (uint32_t half = 0; half < 2; ++half) for (uint32_t group = 0; group < 4; ++group) {
                const float ds0 = d * float(I8(U8(p + 192 + 8 * half + 2 * group)));
                const float ds1 = d * float(I8(U8(p + 192 + 8 * half + 2 * group + 1)));
                for (uint32_t lane = 0; lane < 32; ++lane) {
                    const uint32_t low = U8(p + 64 * half + 32 * (group % 2) + lane);
                    const uint32_t high = U8(p + 128 + 32 * half + lane);
                    const int q = int(((low >> (group >= 2 ? 4 : 0)) & 15) | (((high >> (2 * group)) & 3) << 4)) - 32;
                    Store(out + 128 * half + 32 * group + lane, (lane < 16 ? ds0 : ds1) * float(q));
                }
            }
            return;
        }
        if (type == 20 || type == 23) {
            const uint32_t groups = type == 20 ? 1 : 8;
            const uint32_t high = type == 23 ? U16(p + 2) : 0;
            for (uint32_t group = 0; group < groups; ++group) {
                float scale = d;
                if (type == 23) {
                    const uint32_t low = U8(p + 4 + group / 2);
                    const uint32_t ls = ((low >> (4 * (group % 2))) & 15) | (((high >> (2 * group)) & 3) << 4);
                    scale = d * float(int32_t(ls) - 32);
                }
                for (uint32_t lane = 0; lane < 16; ++lane) {
                    const uint32_t packed = U8(p + (type == 20 ? 2 : 8) + group * 16 + lane);
                    Store(out + group * 32 + lane, scale * float(I8(Table(IQ4NL + (packed & 15)))));
                    Store(out + group * 32 + 16 + lane, scale * float(I8(Table(IQ4NL + (packed >> 4)))));
                }
            }
            return;
        }
        for (uint32_t chunk = 0; chunk < 8; ++chunk) {
            if (type == 16 || type == 18) {
                const uint32_t aux = U32(p + (type == 16 ? 6 + chunk * 8 : 66 + chunk * 4));
                const float scale = d * (0.5f + float(int32_t(aux >> 28))) * (type == 16 ? 0.25f : 0.5f);
                for (uint32_t l = 0; l < 4; ++l) {
                    const uint32_t signs = Table(SIGNS + ((aux >> (7 * l)) & 127));
                    if (type == 16) {
                        const uint32_t grid = IQ2XXS + 8 * U8(p + 2 + chunk * 8 + l);
                        for (uint32_t lane = 0; lane < 8; ++lane)
                            Store(out + chunk * 32 + l * 8 + lane, Signed(scale, Table(grid + lane), signs, lane));
                    } else {
                        for (uint32_t half = 0; half < 2; ++half) {
                            const uint32_t grid = IQ3XXS + 4 * U8(p + 2 + chunk * 8 + 2 * l + half);
                            for (uint32_t lane = 0; lane < 4; ++lane)
                                Store(out + chunk * 32 + l * 8 + half * 4 + lane, Signed(scale, Table(grid + lane), signs, half * 4 + lane));
                        }
                    }
                }
            } else if (type == 17 || type == 22) {
                const uint32_t sc = U8(p + (type == 17 ? 66 : 74) + chunk);
                const float scale0 = d * (0.5f + float(int32_t(sc & 15))) * 0.25f;
                const float scale1 = d * (0.5f + float(int32_t(sc >> 4))) * 0.25f;
                const uint32_t high = type == 22 ? U8(p + 66 + chunk) : 0;
                for (uint32_t l = 0; l < 4; ++l) {
                    uint32_t grid, signs;
                    if (type == 17) {
                        const uint32_t index = U16(p + 2 + (4 * chunk + l) * 2);
                        grid = IQ2XS + 8 * (index & 511);
                        signs = Table(SIGNS + (index >> 9));
                    } else {
                        const uint32_t index = U8(p + 2 + chunk * 4 + l) | ((high << (8 - 2 * l)) & 0x300);
                        grid = IQ2S + 8 * index;
                        signs = U8(p + 34 + chunk * 4 + l);
                    }
                    const float scale = l < 2 ? scale0 : scale1;
                    for (uint32_t lane = 0; lane < 8; ++lane)
                        Store(out + chunk * 32 + l * 8 + lane, Signed(scale, Table(grid + lane), signs, lane));
                }
            } else if (type == 21) {
                const uint32_t sc = U8(p + 106 + chunk / 2), high = U8(p + 66 + chunk);
                const float scale = d * float(int32_t(1 + 2 * ((sc >> (4 * (chunk % 2))) & 15)));
                for (uint32_t l = 0; l < 4; ++l) {
                    const uint32_t signs = U8(p + 74 + chunk * 4 + l);
                    for (uint32_t half = 0; half < 2; ++half) {
                        const uint32_t index = U8(p + 2 + chunk * 8 + 2 * l + half) | ((high << (8 - 2 * l - half)) & 256);
                        for (uint32_t lane = 0; lane < 4; ++lane)
                            Store(out + chunk * 32 + l * 8 + half * 4 + lane, Signed(scale, Table(IQ3S + 4 * index + lane), signs, half * 4 + lane));
                    }
                }
            }
        }
    }
};
