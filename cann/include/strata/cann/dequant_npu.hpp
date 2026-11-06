#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "strata/cann/device.hpp"
#include "strata/cann/gguf.hpp"

namespace strata::cann {

// raw_device and out_f32 are distinct HBM allocations. The output must be
// 64-byte aligned (Buffer guarantees this). All decoding arithmetic executes
// in the AscendC kernel. Unsupported types and unaligned blocks throw.
// This initial implementation synchronizes before returning, so callers may
// immediately release the compressed input allocation.
void dequantize_device(Stream & stream, int ggml_type, const void * raw_device,
                       uint64_t count, void * out_f32,
                       const void * tables_device = nullptr);

// Upload immutable format tables once and retain them with the model/stream.
// With deferred matvec execution the caller retains borrowed tables until the
// stream is drained; nullptr retains the synchronous self-contained API path.
std::shared_ptr<Buffer> make_quant_tables(Stream & stream);

// Size of an original, block-aligned GGUF payload. Validates the format and
// count without touching model values; overflow and partial blocks throw.
uint64_t quant_storage_bytes(int ggml_type, uint64_t elements);

// y[rows] = original_quant_weights[rows,cols] * x[cols]. The compressed
// weights remain in HBM; dequantized tiles and reductions stay inside the NPU.
// All pointers belong to stream's device and y is distinct, 64-byte aligned.
// cols is a multiple of eight; rows*cols contains whole quant blocks. A row
// may straddle blocks (for example cols=640 with block size 256).
// Normally synchronous. Inside DeferredScope with borrowed tables, queues on
// that stream; all pointers remain valid through the next stream drain (pooled
// addresses may be reused only in stream order). No CPU arithmetic or fallback.
void quant_matvec_device(Stream & stream, int ggml_type, const void * raw_device,
                         uint64_t rows, uint64_t cols,
                         const float * x_device, float * y_device,
                         const void * tables_device = nullptr);

// raw contains experts_total consecutive [rows,cols] compressed matrices.
// IDs remain in HBM; repeated/out-of-order IDs are supported. y is contiguous
// [selected,rows], and x is [cols] or [selected,cols] when x_perexpert=true
// (indexed by selection rank, not expert ID). Invalid IDs produce a NaN row
// group on the NPU without reading outside raw. selected is in [1,512].
// Each expert contains complete quant blocks; cols is a multiple of eight.
// Every pointer belongs to stream's device, y is 64-byte aligned and does not
// overlap an input. Borrowed-table DeferredScope behavior matches matvec above.
void quant_matvec_grouped_device(Stream& stream, int ggml_type, const void* raw_device,
                                uint64_t rows, uint64_t cols, uint32_t selected,
                                uint32_t experts_total, const int64_t* ids_device,
                                const float* x_device, float* y_device, bool x_perexpert,
                                const void* tables_device = nullptr);

// y[rows] = sum_i scores[i] * expert_outputs[i,rows], starting at positive
// zero and processing ranks in order. FP32 multiplication and addition remain
// separate, matching the original MoE accumulator. All pointers are on the
// stream's NPU; y is 64-byte aligned and must not overlap either input.
// Synchronous by default, queued in DeferredScope until the next stream drain.
void weighted_sum_experts_device(Stream& stream, uint32_t selected, uint64_t rows,
                                 const float* expert_outputs, const float* scores,
                                 float* y_device);

// Host work consists only of file I/O and moving the original compressed bytes.
// F32/F16/BF16, all model IQ/K formats, Q2_0 and Q8_0 are supported.
std::shared_ptr<Buffer> load_quant_device(Stream & stream, const Gguf & gguf,
                                        const GgufTensor & tensor,
                                        uint64_t offset, uint64_t count);

// Hash arithmetic runs on NPU; only the sixteen row IDs return to the host for
// GGUF byte-range I/O. Multiplication wraps modulo 2^64, matching the model.
std::vector<uint64_t> ple_rows_device(Stream & stream, const uint64_t ctx[3],
                                     const uint64_t multipliers[3],
                                     const uint64_t offsets[16],
                                     const uint64_t vocabs[16]);

} // namespace strata::cann
