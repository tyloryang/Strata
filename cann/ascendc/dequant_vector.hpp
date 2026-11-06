// Decoder private methods: pack signed integer codes in UB four at a time.
// Vector Cast and broadcast Mul produce FP32 values in the original order.
    __aicore__ inline bool VectorQuant(int type) {
        return type == 8 || type == 12 || type == 13 || type == 14 || type == 16 || type == 17 ||
               type == 18 || type == 20 || type == 21 || type == 22 || type == 23 || type == 42;
    }
    __aicore__ inline uint32_t RawWord(uint32_t offset) {
        auto words = raw_local_.ReinterpretCast<uint32_t>();
        const uint32_t low = words.GetValue(offset / 4), shift = (offset % 4) * 8;
        return shift == 0 ? low : (low >> shift) | (words.GetValue(offset / 4 + 1) << (32 - shift));
    }
    __aicore__ inline uint32_t GridWord(uint32_t offset) {
        return table_local_.ReinterpretCast<uint32_t>().GetValue(offset / 4);
    }
    __aicore__ inline void Codes(uint32_t out, uint32_t packed) {
        codes_local_.ReinterpretCast<uint32_t>().SetValue(out / 4, packed);
    }
    __aicore__ inline uint32_t SignedWord(uint32_t grid, uint32_t signs) {
        const uint32_t mask = sign_masks_local_.GetValue(signs & 15);
        // Every lattice magnitude is in [1,127]. Negating bytes independently
        // cannot carry into a neighbouring lane because none of them is zero.
        return (grid ^ mask) + (mask & 0x01010101u);
    }
    __aicore__ inline void Scale16(uint32_t out, float value) {
        AscendC::Duplicate(scales_local_[(out / 16) * 8], value, int32_t(8));
    }
    __aicore__ inline void Scale32(uint32_t out, float value) {
        Scale16(out, value);
        Scale16(out + 16, value);
    }
    __aicore__ inline void Minimum32(uint32_t out, float value) {
        AscendC::Duplicate(minima_local_[(out / 16) * 8], value, int32_t(8));
        AscendC::Duplicate(minima_local_[(out / 16 + 1) * 8], value, int32_t(8));
    }
    __aicore__ inline uint32_t NibblesToCodes(uint32_t packed) {
        return Table(IQ4NL + (packed & 15)) |
               (Table(IQ4NL + ((packed >> 8) & 15)) << 8) |
               (Table(IQ4NL + ((packed >> 16) & 15)) << 16) |
               (Table(IQ4NL + ((packed >> 24) & 15)) << 24);
    }
    __aicore__ inline uint32_t TwoBitCodes(uint32_t packed) {
        return (((packed & 3) - 1u) & 255u) |
               (((((packed >> 2) & 3) - 1u) & 255u) << 8) |
               (((((packed >> 4) & 3) - 1u) & 255u) << 16) |
               (((((packed >> 6) & 3) - 1u) & 255u) << 24);
    }
    __aicore__ inline void Grid8(uint32_t out, uint32_t grid, uint32_t signs) {
        Codes(out, SignedWord(GridWord(grid), signs));
        Codes(out + 4, SignedWord(GridWord(grid + 4), signs >> 4));
    }
    __aicore__ inline void PackBlock(int type, uint32_t p, uint32_t out) {
        const float d = D(p + (type == 14 ? 208 : 0));
        if (type == 8) {
            Scale32(out, d);
            for (uint32_t lane = 0; lane < 32; lane += 4) Codes(out + lane, RawWord(p + 2 + lane));
            return;
        }
        if (type == 42) {
            Scale32(out, d);
            Scale32(out + 32, d);
            for (uint32_t group = 0; group < 16; group += 4) {
                const uint32_t packed = RawWord(p + 2 + group);
                Codes(out + group * 4, TwoBitCodes(packed));
                Codes(out + group * 4 + 4, TwoBitCodes(packed >> 8));
                Codes(out + group * 4 + 8, TwoBitCodes(packed >> 16));
                Codes(out + group * 4 + 12, TwoBitCodes(packed >> 24));
            }
            return;
        }
        if (type == 12 || type == 13) {
            const float minimum = D(p + 2);
            for (uint32_t group = 0; group < 4; ++group) {
                uint32_t s0, m0, s1, m1;
                ScaleMin(p + 4, 2 * group, s0, m0);
                ScaleMin(p + 4, 2 * group + 1, s1, m1);
                Scale32(out + group * 64, d * float(int32_t(s0)));
                Scale32(out + group * 64 + 32, d * float(int32_t(s1)));
                Minimum32(out + group * 64, minimum * float(int32_t(m0)));
                Minimum32(out + group * 64 + 32, minimum * float(int32_t(m1)));
                for (uint32_t lane = 0; lane < 32; lane += 4) {
                    const uint32_t packed = RawWord(p + (type == 12 ? 16 : 48) + group * 32 + lane);
                    const uint32_t high = type == 13 ? RawWord(p + 16 + lane) : 0;
                    Codes(out + group * 64 + lane,
                        (packed & 0x0f0f0f0fu) | (((high >> (2 * group)) & 0x01010101u) << 4));
                    Codes(out + group * 64 + 32 + lane,
                        ((packed >> 4) & 0x0f0f0f0fu) | (((high >> (2 * group + 1)) & 0x01010101u) << 4));
                }
            }
            return;
        }
        if (type == 20 || type == 23) {
            const uint32_t groups = type == 20 ? 1 : 8, high = type == 23 ? U16(p + 2) : 0;
            for (uint32_t group = 0; group < groups; ++group) {
                float scale = d;
                if (type == 23) {
                    const uint32_t low = U8(p + 4 + group / 2);
                    const uint32_t ls = ((low >> (4 * (group % 2))) & 15) | (((high >> (2 * group)) & 3) << 4);
                    scale = d * float(int32_t(ls) - 32);
                }
                Scale32(out + group * 32, scale);
                for (uint32_t lane = 0; lane < 16; lane += 4) {
                    const uint32_t packed = RawWord(p + (type == 20 ? 2 : 8) + group * 16 + lane);
                    Codes(out + group * 32 + lane, NibblesToCodes(packed));
                    Codes(out + group * 32 + 16 + lane, NibblesToCodes(packed >> 4));
                }
            }
            return;
        }
        if (type == 14) {
            for (uint32_t half = 0; half < 2; ++half) {
                for (uint32_t group = 0; group < 8; ++group)
                    Scale16(out + half * 128 + group * 16, d * float(I8(U8(p + 192 + half * 8 + group))));
                for (uint32_t lane = 0; lane < 32; lane += 4) {
                    const uint32_t low0 = RawWord(p + half * 64 + lane);
                    const uint32_t low1 = RawWord(p + half * 64 + 32 + lane);
                    const uint32_t high = RawWord(p + 128 + half * 32 + lane);
                    const uint32_t q0 = (low0 & 0x0f0f0f0fu) | ((high & 0x03030303u) << 4);
                    const uint32_t q1 = (low1 & 0x0f0f0f0fu) | ((high & 0x0c0c0c0cu) << 2);
                    const uint32_t q2 = ((low0 >> 4) & 0x0f0f0f0fu) | (high & 0x30303030u);
                    const uint32_t q3 = ((low1 >> 4) & 0x0f0f0f0fu) | ((high & 0xc0c0c0c0u) >> 2);
                    // q is [0,63]. Bias each byte by 128 before subtracting 32,
                    // preventing inter-byte borrow; XOR restores signed int8.
                    Codes(out + half * 128 + lane, ((q0 | 0x80808080u) - 0x20202020u) ^ 0x80808080u);
                    Codes(out + half * 128 + 32 + lane, ((q1 | 0x80808080u) - 0x20202020u) ^ 0x80808080u);
                    Codes(out + half * 128 + 64 + lane, ((q2 | 0x80808080u) - 0x20202020u) ^ 0x80808080u);
                    Codes(out + half * 128 + 96 + lane, ((q3 | 0x80808080u) - 0x20202020u) ^ 0x80808080u);
                }
            }
            return;
        }
        for (uint32_t chunk = 0; chunk < 8; ++chunk) {
            const uint32_t row = out + chunk * 32;
            if (type == 16 || type == 18) {
                const uint32_t aux = RawWord(p + (type == 16 ? 6 + chunk * 8 : 66 + chunk * 4));
                const float scale = d * (0.5f + float(int32_t(aux >> 28))) * (type == 16 ? 0.25f : 0.5f);
                Scale32(row, scale);
                for (uint32_t l = 0; l < 4; ++l) {
                    const uint32_t signs = Table(SIGNS + ((aux >> (7 * l)) & 127));
                    if (type == 16) Grid8(row + l * 8, IQ2XXS + 8 * U8(p + 2 + chunk * 8 + l), signs);
                    else {
                        const uint32_t g0 = IQ3XXS + 4 * U8(p + 2 + chunk * 8 + 2 * l);
                        const uint32_t g1 = IQ3XXS + 4 * U8(p + 3 + chunk * 8 + 2 * l);
                        Codes(row + l * 8, SignedWord(GridWord(g0), signs));
                        Codes(row + l * 8 + 4, SignedWord(GridWord(g1), signs >> 4));
                    }
                }
            } else if (type == 17 || type == 22) {
                const uint32_t sc = U8(p + (type == 17 ? 66 : 74) + chunk);
                Scale16(row, d * (0.5f + float(int32_t(sc & 15))) * 0.25f);
                Scale16(row + 16, d * (0.5f + float(int32_t(sc >> 4))) * 0.25f);
                const uint32_t high = type == 22 ? U8(p + 66 + chunk) : 0;
                for (uint32_t l = 0; l < 4; ++l) {
                    if (type == 17) {
                        const uint32_t index = U16(p + 2 + (4 * chunk + l) * 2);
                        Grid8(row + l * 8, IQ2XS + 8 * (index & 511), Table(SIGNS + (index >> 9)));
                    } else {
                        const uint32_t index = U8(p + 2 + chunk * 4 + l) | ((high << (8 - 2 * l)) & 0x300);
                        Grid8(row + l * 8, IQ2S + 8 * index, U8(p + 34 + chunk * 4 + l));
                    }
                }
            } else if (type == 21) {
                const uint32_t sc = U8(p + 106 + chunk / 2), high = U8(p + 66 + chunk);
                Scale32(row, d * float(int32_t(1 + 2 * ((sc >> (4 * (chunk % 2))) & 15))));
                for (uint32_t l = 0; l < 4; ++l) {
                    const uint32_t signs = U8(p + 74 + chunk * 4 + l);
                    const uint32_t i0 = U8(p + 2 + chunk * 8 + 2 * l) | ((high << (8 - 2 * l)) & 256);
                    const uint32_t i1 = U8(p + 3 + chunk * 8 + 2 * l) | ((high << (7 - 2 * l)) & 256);
                    Codes(row + l * 8, SignedWord(GridWord(IQ3S + 4 * i0), signs));
                    Codes(row + l * 8 + 4, SignedWord(GridWord(IQ3S + 4 * i1), signs >> 4));
                }
            }
        }
    }
