#include "strata/cann/dequant_npu.hpp"
#include "strata/cann/quants.hpp"
#include <acl/acl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "../../third_party/ggml/ggml-common.h"
#undef QK_K

extern "C" void strata_cann_launch_dequant(uint32_t, void *, const void *, const void *,
    void *, uint64_t, int32_t, uint32_t, uint32_t);
extern "C" void strata_cann_launch_ple_rows(void *, const void *, void *);
extern "C" void strata_cann_launch_matvec(uint32_t, void *, const void *, const void *,
    const float *, float *, uint64_t, uint64_t, int32_t, uint32_t, uint32_t);
extern "C" void strata_cann_launch_grouped_matvec(uint32_t, void*, const void*, const void*,
    const float*, const int64_t*, float*, uint64_t, uint64_t, uint32_t, uint32_t,
    int32_t, uint32_t, uint32_t, int32_t);
extern "C" void strata_cann_launch_ordered_expert_sum(uint32_t, void*, const float*,
    const float*, float*, uint64_t, uint32_t);

namespace strata::cann {
namespace {
uint32_t decoder_core_count() {
    int32_t device = -1;
    if (aclrtGetDevice(&device) != ACL_SUCCESS)
        throw std::runtime_error("NPU dequant: cannot query current device");
    // Core counts are immutable hardware properties; cache per calling thread
    // and refresh when switching devices. Never assume 32 vector cores.
    thread_local int32_t cached_device = -1;
    thread_local uint32_t available = 0;
    if (device != cached_device) {
        int64_t count = 0;
        if (aclrtGetDeviceInfo(uint32_t(device), ACL_DEV_ATTR_VECTOR_CORE_NUM, &count) != ACL_SUCCESS || count < 1)
            throw std::runtime_error("NPU dequant: cannot query vector core count");
        available = uint32_t(count);
        cached_device = device;
    }
    if (const char * value = std::getenv("STRATA_CANN_DEQUANT_CORES")) {
        char * end = nullptr;
        const auto requested = std::strtol(value, &end, 10);
        if (!*value || *end || requested < 1 || requested > available)
            throw std::runtime_error("NPU dequant: core override exceeds available vector cores");
        return uint32_t(requested);
    }
    return available;
}
struct DecodeStats {
    const char* name;
    explicit DecodeStats(const char* label = "DEQUANT_STATS") : name(label) {}
    struct Entry { uint64_t calls = 0, elements = 0; double seconds = 0; };
    std::array<Entry, 43> entries{};
    std::mutex mutex;
    ~DecodeStats() {
        for (size_t type = 0; type < entries.size(); ++type) {
            const auto & e = entries[type];
            if (e.calls) std::fprintf(stderr,
                "%s type=%zu calls=%llu elements=%llu seconds=%.6f\n", name, type,
                (unsigned long long)e.calls, (unsigned long long)e.elements, e.seconds);
        }
    }
};
void record_decode(int type, uint64_t count, double seconds) {
    static DecodeStats stats;
    std::lock_guard<std::mutex> lock(stats.mutex);
    auto & entry = stats.entries.at(size_t(type));
    ++entry.calls; entry.elements += count; entry.seconds += seconds;
}
void record_matvec(int type, uint64_t count, double seconds) {
    static DecodeStats stats("MATVEC_STATS");
    std::lock_guard<std::mutex> lock(stats.mutex);
    auto& entry = stats.entries.at(size_t(type));
    ++entry.calls; entry.elements += count; entry.seconds += seconds;
}
void record_matvec_enqueue(int type, uint64_t count, double seconds) {
    static DecodeStats stats("MATVEC_ENQUEUE_ONLY_STATS");
    std::lock_guard<std::mutex> lock(stats.mutex);
    auto& entry = stats.entries.at(size_t(type));
    ++entry.calls; entry.elements += count; entry.seconds += seconds;
}
void record_grouped(int type, uint64_t count, double seconds, bool deferred) {
    static DecodeStats completed("GROUPED_MATVEC_STATS");
    static DecodeStats queued("GROUPED_MATVEC_ENQUEUE_ONLY_STATS");
    auto& stats = deferred ? queued : completed;
    std::lock_guard<std::mutex> lock(stats.mutex);
    auto& entry = stats.entries.at(size_t(type));
    ++entry.calls; entry.elements += count; entry.seconds += seconds;
}
void record_expert_sum(uint64_t count, double seconds, bool deferred) {
    static DecodeStats completed("ORDERED_EXPERT_SUM_STATS");
    static DecodeStats queued("ORDERED_EXPERT_SUM_ENQUEUE_ONLY_STATS");
    auto& stats = deferred ? queued : completed;
    std::lock_guard<std::mutex> lock(stats.mutex);
    auto& entry = stats.entries[0]; // Aggregation is always FP32.
    ++entry.calls; entry.elements += count; entry.seconds += seconds;
}

uint64_t checked_product(uint64_t a, uint64_t b, const char* what) {
    if (b && a > std::numeric_limits<uint64_t>::max() / b)
        throw std::overflow_error(std::string("NPU grouped: ") + what + " overflow");
    return a * b;
}
size_t checked_bytes(uint64_t count, size_t width, const char* what) {
    if (count > std::numeric_limits<size_t>::max() / width)
        throw std::overflow_error(std::string("NPU grouped: ") + what + " exceeds address space");
    return size_t(count) * width;
}
uintptr_t checked_end(const void* pointer, size_t bytes) {
    const uintptr_t start = reinterpret_cast<uintptr_t>(pointer);
    if (bytes > std::numeric_limits<uintptr_t>::max() - start)
        throw std::overflow_error("NPU grouped: pointer range wraps address space");
    return start + bytes;
}
void require_disjoint(const void* output, size_t output_bytes, const void* input, size_t input_bytes) {
    const uintptr_t output_end = checked_end(output, output_bytes), input_end = checked_end(input, input_bytes);
    if (reinterpret_cast<uintptr_t>(output) < input_end && reinterpret_cast<uintptr_t>(input) < output_end)
        throw std::runtime_error("NPU grouped: output overlaps an input");
}
void require_device_pointer(const Stream& stream, const void* pointer, size_t alignment, const char* what) {
    if (!pointer || reinterpret_cast<uintptr_t>(pointer) % alignment)
        throw std::runtime_error(std::string("NPU grouped: null or misaligned ") + what);
    aclrtPtrAttributes attributes{};
    const aclError status = aclrtPointerGetAttributes(pointer, &attributes);
    if (status != ACL_SUCCESS)
        throw Error(std::string("NPU grouped: cannot query ") + what + " owner", status);
    if (attributes.location.type != ACL_MEM_LOCATION_TYPE_DEVICE ||
        int(attributes.location.id) != stream.device_ordinal())
        throw std::runtime_error(std::string("NPU grouped: ") + what + " is not on the stream's NPU");
}
struct Geometry { uint32_t elements, bytes; };
Geometry geometry(int type) {
    if (type == 0) return {1, 4};
    if (type == 1 || type == 30) return {1, 2};
    switch (type) {
        case 8: case 12: case 13: case 14: case 16: case 17: case 18:
        case 20: case 21: case 22: case 23: case 42: break;
        default: throw std::runtime_error("NPU dequant: unsupported GGML type " + std::to_string(type));
    }
    const int elements = quant_block_elements(type), bytes = quant_type_size(type);
    if (!elements || !bytes) throw std::runtime_error("NPU dequant: unsupported GGML type " + std::to_string(type));
    return {uint32_t(elements), uint32_t(bytes)};
}

const std::vector<uint8_t> & codebook_bytes() {
    // These are immutable format constants, never decoded model weights.
    static const auto data = [] {
        std::vector<uint8_t> result;
        const auto append = [&](const void * pointer, size_t size) {
            const auto * begin = static_cast<const uint8_t *>(pointer);
            result.insert(result.end(), begin, begin + size);
        };
        append(ksigns_iq2xs, sizeof(ksigns_iq2xs));
        append(iq2xxs_grid, sizeof(iq2xxs_grid));
        append(iq2xs_grid, sizeof(iq2xs_grid));
        append(iq2s_grid, sizeof(iq2s_grid));
        append(iq3xxs_grid, sizeof(iq3xxs_grid));
        append(iq3s_grid, sizeof(iq3s_grid));
        append(kvalues_iq4nl, sizeof(kvalues_iq4nl));
        if (result.size() != 17552) throw std::runtime_error("NPU dequant: unexpected codebook layout");
        return result;
    }();
    return data;
}
} // namespace

std::shared_ptr<Buffer> make_quant_tables(Stream & stream) {
    stream.activate();
    const auto & table = codebook_bytes();
    auto device_table = std::make_shared<Buffer>(table.size());
    device_table->copy_from_host(table.data(), table.size());
    return device_table;
}

void dequantize_device(Stream & stream, int type, const void * raw_device, uint64_t count, void * out_f32,
                       const void * tables_device) {
    stream.activate();
    const bool stats = std::getenv("STRATA_CANN_DEQUANT_STATS") != nullptr;
    const auto begin = stats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto g = geometry(type);
    if (!raw_device || !out_f32 || !count || count % g.elements ||
        count > std::numeric_limits<size_t>::max() / sizeof(float))
        throw std::runtime_error("NPU dequant: invalid pointer/count or unaligned quant block");
    if (raw_device == out_f32 || reinterpret_cast<uintptr_t>(out_f32) % 64)
        throw std::runtime_error("NPU dequant: output must be distinct and 64-byte aligned");
    std::shared_ptr<Buffer> owned_tables;
    if (!tables_device) { owned_tables = make_quant_tables(stream); tables_device = owned_tables->data(); }
    const uint32_t kernel_elements = g.elements == 1 ? 256 : g.elements;
    const uint32_t kernel_bytes = g.elements == 1 ? 256 * g.bytes : g.bytes;
    const uint64_t blocks = (count + kernel_elements - 1) / kernel_elements;
    const uint32_t cores = uint32_t(std::min<uint64_t>(decoder_core_count(), blocks));
    stream.submitted();
    strata_cann_launch_dequant(cores, stream.handle(), raw_device, tables_device, out_f32,
                              count, type, kernel_elements, kernel_bytes);
    stream.synchronize();
    if (stats) record_decode(type, count, std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());
}

uint64_t quant_storage_bytes(int type, uint64_t elements) {
    const auto g = geometry(type);
    if (!elements || elements % g.elements)
        throw std::runtime_error("NPU quant: empty count or partial quant block");
    if (elements / g.elements > std::numeric_limits<uint64_t>::max() / g.bytes)
        throw std::overflow_error("NPU quant: compressed payload size overflow");
    return elements / g.elements * g.bytes;
}

void quant_matvec_device(Stream & stream, int type, const void * raw_device,
                         uint64_t rows, uint64_t cols,
                         const float * x_device, float * y_device, const void * tables_device) {
    stream.activate();
    const bool stats = std::getenv("STRATA_CANN_MATVEC_STATS") != nullptr;
    const auto begin = stats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto g = geometry(type);
    if (!raw_device || !x_device || !y_device || !rows || !cols ||
        rows > std::numeric_limits<uint64_t>::max() / cols ||
        rows > std::numeric_limits<size_t>::max() / sizeof(float) ||
        cols > std::numeric_limits<size_t>::max() / sizeof(float))
        throw std::runtime_error("NPU matvec: invalid pointer or dimensions");
    if (cols % 8)
        throw std::runtime_error("NPU matvec: columns must be a multiple of eight");
    if (quant_storage_bytes(type, rows * cols) > std::numeric_limits<size_t>::max())
        throw std::overflow_error("NPU matvec: compressed payload exceeds address space");
    if (raw_device == y_device || x_device == y_device || reinterpret_cast<uintptr_t>(y_device) % 64)
        throw std::runtime_error("NPU matvec: output must be distinct and 64-byte aligned");
    std::shared_ptr<Buffer> owned_tables;
    if (!tables_device) { owned_tables = make_quant_tables(stream); tables_device = owned_tables->data(); }
    const uint32_t cores = uint32_t(std::min<uint64_t>(decoder_core_count(), (rows + 7) / 8));
    const uint32_t kernel_elements = g.elements == 1 ? 256 : g.elements;
    const uint32_t kernel_bytes = g.elements == 1 ? 256 * g.bytes : g.bytes;
    stream.submitted();
    strata_cann_launch_matvec(cores, stream.handle(), raw_device, tables_device, x_device, y_device,
                             rows, cols, type, kernel_elements, kernel_bytes);
    // Owned tables must survive until execution completes. A model may enqueue
    // with its persistent borrowed tables only inside the explicit scope.
    const bool deferred = stream.deferred() && !owned_tables;
    if (!deferred) stream.synchronize();
    if (stats) {
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        if (deferred) record_matvec_enqueue(type, rows * cols, seconds);
        else record_matvec(type, rows * cols, seconds);
    }
}

void quant_matvec_grouped_device(Stream& stream, int type, const void* raw_device,
                                uint64_t rows, uint64_t cols, uint32_t selected,
                                uint32_t experts_total, const int64_t* ids_device,
                                const float* x_device, float* y_device, bool x_perexpert,
                                const void* tables_device) {
    stream.activate();
    const bool stats = std::getenv("STRATA_CANN_MATVEC_STATS") != nullptr;
    const auto begin = stats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto g = geometry(type);
    if (!rows || !cols || cols % 8 || !selected || selected > 512 || !experts_total)
        throw std::runtime_error("NPU grouped: invalid dimensions, expert count or columns not divisible by eight");
    const uint64_t matrix_elements = checked_product(rows, cols, "expert matrix");
    const uint64_t output_elements = checked_product(rows, selected, "grouped output");
    const uint64_t activation_elements = checked_product(cols, x_perexpert ? selected : 1, "activation");
    const uint64_t logical_elements = checked_product(matrix_elements, selected, "logical matrix work");
    const uint64_t expert_bytes = quant_storage_bytes(type, matrix_elements);
    const size_t raw_bytes = checked_bytes(checked_product(expert_bytes, experts_total, "all expert payloads"), 1, "weights");
    const size_t output_bytes = checked_bytes(output_elements, sizeof(float), "output");
    const size_t input_bytes = checked_bytes(activation_elements, sizeof(float), "activation");
    const size_t id_bytes = checked_bytes(selected, sizeof(int64_t), "IDs");
    if (!raw_device || !x_device || !ids_device || !y_device)
        throw std::runtime_error("NPU grouped: null pointer");
    require_disjoint(y_device, output_bytes, raw_device, raw_bytes);
    require_disjoint(y_device, output_bytes, x_device, input_bytes);
    require_disjoint(y_device, output_bytes, ids_device, id_bytes);
    require_device_pointer(stream, raw_device, 1, "weights");
    require_device_pointer(stream, ids_device, alignof(int64_t), "IDs");
    require_device_pointer(stream, x_device, alignof(float), "activation");
    require_device_pointer(stream, y_device, 64, "output");
    std::shared_ptr<Buffer> owned_tables;
    if (!tables_device) { owned_tables = make_quant_tables(stream); tables_device = owned_tables->data(); }
    require_disjoint(y_device, output_bytes, tables_device, codebook_bytes().size());
    require_device_pointer(stream, tables_device, 1, "codebook");
    const uint64_t groups = output_elements / 8 + (output_elements % 8 != 0);
    const uint32_t cores = uint32_t(std::min<uint64_t>(decoder_core_count(), groups));
    const uint32_t kernel_elements = g.elements == 1 ? 256 : g.elements;
    const uint32_t kernel_bytes = g.elements == 1 ? 256 * g.bytes : g.bytes;
    stream.submitted();
    strata_cann_launch_grouped_matvec(cores, stream.handle(), raw_device, tables_device,
        x_device, ids_device, y_device, rows, cols, selected, experts_total, type,
        kernel_elements, kernel_bytes, x_perexpert ? 1 : 0);
    const bool deferred = stream.deferred() && !owned_tables;
    if (!deferred) stream.synchronize();
    if (stats) record_grouped(type, logical_elements,
        std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count(), deferred);
}

void weighted_sum_experts_device(Stream& stream, uint32_t selected, uint64_t rows,
                                 const float* expert_outputs, const float* scores, float* y_device) {
    stream.activate();
    const bool stats = std::getenv("STRATA_CANN_MATVEC_STATS") != nullptr;
    const auto begin = stats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (!rows || !selected || selected > 512)
        throw std::runtime_error("NPU expert sum: rows must be positive and selected must be in [1,512]");
    const uint64_t input_elements = checked_product(rows, selected, "expert outputs");
    const size_t input_bytes = checked_bytes(input_elements, sizeof(float), "expert outputs");
    const size_t output_bytes = checked_bytes(rows, sizeof(float), "aggregate output");
    const size_t score_bytes = checked_bytes(selected, sizeof(float), "scores");
    if (!expert_outputs || !scores || !y_device)
        throw std::runtime_error("NPU expert sum: null pointer");
    require_disjoint(y_device, output_bytes, expert_outputs, input_bytes);
    require_disjoint(y_device, output_bytes, scores, score_bytes);
    require_device_pointer(stream, expert_outputs, alignof(float), "expert outputs");
    require_device_pointer(stream, scores, alignof(float), "scores");
    require_device_pointer(stream, y_device, 64, "aggregate output");
    const uint64_t groups = rows / 256 + (rows % 256 != 0);
    const uint32_t cores = uint32_t(std::min<uint64_t>(decoder_core_count(), groups));
    stream.submitted();
    strata_cann_launch_ordered_expert_sum(cores, stream.handle(), expert_outputs, scores, y_device, rows, selected);
    const bool deferred = stream.deferred();
    if (!deferred) stream.synchronize();
    if (stats) record_expert_sum(input_elements,
        std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count(), deferred);
}

std::shared_ptr<Buffer> load_quant_device(Stream & stream, const Gguf & gguf,
                                        const GgufTensor & tensor, uint64_t offset, uint64_t count) {
    stream.activate();
    const auto g = geometry(int(tensor.type));
    if (!count || offset > tensor.elements() || count > tensor.elements() - offset ||
        offset % g.elements || count % g.elements ||
        count > std::numeric_limits<size_t>::max() / sizeof(float))
        throw std::runtime_error("NPU dequant: invalid tensor slice " + tensor.name);
    const auto bytes = gguf.tensor_bytes(tensor, offset / g.elements * g.bytes, count / g.elements * g.bytes);
    Buffer compressed(bytes.size());
    compressed.copy_from_host(bytes.data(), bytes.size());
    auto result = std::make_shared<Buffer>(count * sizeof(float));
    dequantize_device(stream, int(tensor.type), compressed.data(), count, result->data());
    return result;
}

std::vector<uint64_t> ple_rows_device(Stream & stream, const uint64_t ctx[3], const uint64_t multipliers[3],
                                     const uint64_t offsets[16], const uint64_t vocabs[16]) {
    stream.activate();
    if (!ctx || !multipliers || !offsets || !vocabs) throw std::runtime_error("NPU PLE: null metadata");
    for (int h = 0; h < 16; ++h)
        if (!vocabs[h]) throw std::runtime_error("NPU PLE: empty hash vocabulary");
    uint64_t input[38];
    std::memcpy(input, ctx, 3 * sizeof(uint64_t));
    std::memcpy(input + 3, multipliers, 3 * sizeof(uint64_t));
    std::memcpy(input + 6, offsets, 16 * sizeof(uint64_t));
    std::memcpy(input + 22, vocabs, 16 * sizeof(uint64_t));
    Buffer packed(sizeof(input)), rows(16 * sizeof(uint64_t));
    packed.copy_from_host(input, sizeof(input));
    stream.submitted();
    strata_cann_launch_ple_rows(stream.handle(), packed.data(), rows.data());
    stream.synchronize();
    std::vector<uint64_t> result(16);
    rows.copy_to_host(result.data(), 16 * sizeof(uint64_t));
    return result;
}
} // namespace strata::cann
