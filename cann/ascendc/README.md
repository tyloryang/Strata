# Native device decoding

`dequant_kernel.cpp` decodes the original GGUF bytes on an Ascend AI Core. It
covers F32, F16, BF16, Q8_0, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_XS, IQ2_S,
IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS and Q2_0. The wrapper rejects unsupported types;
it never calls the CPU decoder. PLE hash multiplication, XOR and unsigned
modulo also execute in a device kernel. Only the row IDs return for file I/O.

The decoder assigns contiguous block ranges to the available vector cores,
loads packed tiles and the required codebook sections into UB with DMA, and
writes complete FP32 tiles back with DMA. Plain F32 uses a copy path; F16 and
BF16 use vector casts. Quantized paths pack signed integer codes and apply
FP32 scales on the device. `dequant_decoder.hpp` owns tiling and pipeline
ordering; `dequant_vector.hpp` owns packed-code extraction. Decoding and matrix
multiplication remain separate kernels. Codebooks are immutable format
constants uploaded from the host; model weights are never decoded there.

The host queries the device's vector-core count (48 on the tested 910B2C).
`STRATA_CANN_DEQUANT_CORES` can lower that count for controlled comparisons;
invalid values fail. `STRATA_CANN_DEQUANT_STATS=1` reports cumulative decoder
API time and element counts by GGML type. These times include setup and stream
synchronization; they are not profiler kernel durations.

The parent CMake project integrates it with:

```cmake
add_subdirectory(ascendc)
target_sources(strata_cann PRIVATE src/dequant_npu.cpp)
target_link_libraries(strata_cann PUBLIC strata_cann_kernels)
add_executable(dequant_npu_parity tests/dequant_npu_parity.cpp)
target_link_libraries(dequant_npu_parity PRIVATE strata_cann)
```

`ASCEND_HOME_PATH` selects the installed CANN package. The subproject finds
the installed `ascendc_kernel_cmake/ascendc.cmake`, sets `RUN_MODE=npu`, and
defaults `SOC_VERSION=Ascend910B2`. CPU simulation is not part of the
production path. If linking the test with `libggml-base`, also define
`STRATA_CANN_HAVE_GGML=1` and add the ggml include directory; otherwise it uses
the existing host decoder solely as the test oracle.

```bash
cmake --build build-cann --target dequant_npu_parity dequant_npu_bench -j8
STRATA_CANN_DEVICE=1 ./build-cann/dequant_npu_parity \
  /path/to/model-00001-of-00002.gguf /path/to/model-00002-of-00002.gguf
STRATA_CANN_DEVICE=1 ./build-cann/dequant_npu_bench --label current
STRATA_CANN_DEVICE=1 ./build-cann/dequant_npu_bench --label large \
  --elements 67108864 --types IQ3_XXS,IQ3_S,Q5_K,Q6_K
```

The test covers 1073 packed blocks per quantized format, normal/subnormal scales,
multi-core tile boundaries, every finite F16/BF16 encoding (including signed
zero and subnormals), F32 boundary patterns, rejected inputs, unsigned PLE
overflow/modulo, and small real tensor slices for every format present in
the supplied GGUF files. The runtime API synchronizes before returning so
temporary packed input and codebook allocations can safely be released.
The microbenchmark reports JSONL with individual samples, median API time,
byte counts and a full CPU-reference comparison. CPU reference work and data
upload/readback are outside the timed region and outside production inference.
