#include "fused_matvec_test_common.hpp"
#include <chrono>
#include <functional>

using namespace fused_test;
namespace {
constexpr uint32_t experts = 11;
struct Options {
    std::vector<Format> types{std::begin(formats), std::end(formats)};
    std::vector<uint32_t> selected{1, 3, 10};
    uint64_t rows = 0, cols = 0;
    int warmup = 2, repeats = 7;
    bool bench = false, small = false, deferred = false;
    std::string input = "both";
};
Options parse(int argc, char ** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") {
            std::puts("usage: grouped_matvec_parity [--types IQ3_XXS,...] [--small] [--rows N --cols N] [--selected 1|3|10] [--input shared|per-rank|both] [--bench --warmup N --repeats N --deferred]");
            std::puts("Default: 15 formats, selected=1/3/10, shared and per-rank x, shapes 49x256/50x640/640x2560/2560x640. --small omits the last two. Benchmark uses the same resident packed weights and codebooks for grouped and separate launches.");
            std::exit(0);
        }
        if (key == "--bench") { o.bench = true; continue; }
        if (key == "--small") { o.small = true; continue; }
        if (key == "--deferred") { o.deferred = true; continue; }
        if (++i >= argc) throw std::runtime_error("missing value for " + key);
        const std::string value = argv[i];
        if (key == "--types") o.types = parse_formats(value);
        else if (key == "--rows") o.rows = positive(value);
        else if (key == "--cols") o.cols = positive(value);
        else if (key == "--selected") {
            const auto n = positive(value);
            if (n != 1 && n != 3 && n != 10) throw std::runtime_error("--selected must be 1, 3 or 10");
            o.selected = {uint32_t(n)};
        } else if (key == "--input") {
            if (value != "shared" && value != "per-rank" && value != "both") throw std::runtime_error("invalid --input");
            o.input = value;
        } else if (key == "--warmup" || key == "--repeats") {
            const auto n = positive(value, key == "--warmup");
            if (n > 1000) throw std::runtime_error("warmup/repeats must be <=1000");
            (key == "--warmup" ? o.warmup : o.repeats) = int(n);
        } else throw std::runtime_error("unknown option " + key);
    }
    if (bool(o.rows) != bool(o.cols)) throw std::runtime_error("--rows and --cols must be supplied together");
    return o;
}
std::vector<int64_t> selection(uint32_t selected) {
    if (selected == 1) return {7};
    if (selected == 3) return {8, 1, 8};
    return {9, 0, 7, 2, 2, 10, 3, 1, 8, 5};
}
std::vector<float> activation(uint64_t cols, uint32_t selected, bool per_rank) {
    std::vector<float> result(cols * (per_rank ? selected : 1));
    uint32_t state = 0x27d4eb2du;
    for (auto & value : result) {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        value = float(int32_t(state & 65535) - 32768) / 32768.0f;
    }
    return result;
}
Input reference(const Input & bank, uint64_t rows, const std::vector<int64_t> & ids,
                const std::vector<float> & x, bool per_rank) {
    Input out{bank.f, "grouped-" + std::to_string(ids.size()) + (per_rank ? "-per-rank" : "-shared"),
              rows * ids.size(), bank.cols, {}, {}, {}, {}, {}};
    out.expected.assign(out.rows, 0); out.sum_abs.assign(out.rows, 0);
    for (size_t rank = 0; rank < ids.size(); ++rank) for (uint64_t row = 0; row < rows; ++row) {
        const auto base = (uint64_t(ids[rank]) * rows + row) * bank.cols;
        const auto xbase = per_rank ? rank * bank.cols : 0;
        double sum = 0, abs_sum = 0;
        for (uint64_t col = 0; col < bank.cols; ++col) {
            const double product = double(bank.weights[base + col]) * x[xbase + col];
            sum += product; abs_sum += std::abs(product);
        }
        out.expected[rank * rows + row] = sum;
        out.sum_abs[rank * rows + row] = abs_sum;
    }
    return out;
}
void same_bits(const std::vector<float> & expected, const std::vector<float> & actual, const char * label) {
    if (expected.size() != actual.size()) throw std::runtime_error(std::string(label) + " size mismatch");
    for (size_t i = 0; i < actual.size(); ++i) if (std::memcmp(&expected[i], &actual[i], sizeof(float))) {
        uint32_t a, e; std::memcpy(&a, &actual[i], 4); std::memcpy(&e, &expected[i], 4);
        std::fprintf(stderr, "%s index=%zu actual=%.9g/%08x expected=%.9g/%08x\n", label, i, actual[i], a, expected[i], e);
        throw std::runtime_error(std::string(label) + " bit mismatch");
    }
}
std::vector<float> ordered_reference(const std::vector<float> & values, const std::vector<float> & scores, uint64_t rows) {
    std::vector<float> result(rows);
    for (uint64_t row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (size_t rank = 0; rank < scores.size(); ++rank) {
            // Volatile stores force the two IEEE FP32 roundings, including under -O3.
            volatile float product = values[rank * rows + row] * scores[rank];
            volatile float next = sum + product;
            sum = next;
        }
        result[row] = sum;
    }
    return result;
}
npu::ND ordered_aclnn(Stream & stream, const npu::ND & values, const npu::ND & scores, uint64_t rows, uint32_t selected) {
    auto sum = npu::zeros({int64_t(rows)});
    for (uint32_t rank = 0; rank < selected; ++rank) {
        const auto row = npu::reshape(stream, npu::slice(values, 0, rank, 1), {int64_t(rows)});
        const auto product = npu::binary(stream, row, npu::slice(scores, 0, rank, 1), npu::Binary::Mul);
        sum = npu::binary(stream, sum, product, npu::Binary::Add);
    }
    return sum;
}
void aggregate_check(Stream & stream, const npu::ND & values, const std::vector<float> & host_values,
                     uint64_t rows, const std::vector<float> & scores) {
    const auto selected = uint32_t(scores.size());
    const auto score_device = npu::from_host(scores.data(), {selected});
    const auto expected = ordered_reference(host_values, scores, rows);
    Buffer sum(rows * sizeof(float));
    for (int repeat = 0; repeat < 3; ++repeat) {
        poison(sum);
        {
            DeferredScope deferred(stream, repeat == 1);
            weighted_sum_experts_device(stream, selected, rows, static_cast<const float *>(values.data()),
                static_cast<const float *>(score_device.data()), static_cast<float *>(sum.data()));
            deferred.finish();
        }
        same_bits(expected, read(sum, rows), "ordered-aggregate");
    }
    const auto sequential = ordered_aclnn(stream, values, score_device, rows, selected);
    same_bits(expected, npu::read_f32(stream, sequential), "sequential-ACLNN-aggregate");
}
template<class F> double timed(Stream & stream, bool deferred, F && operation) {
    stream.synchronize();
    const auto start = std::chrono::steady_clock::now();
    {
        DeferredScope scope(stream, deferred);
        operation();
        scope.finish();
    }
    stream.synchronize();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}
double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const auto mid = values.size() / 2;
    return values.size() % 2 ? values[mid] : (values[mid - 1] + values[mid]) * 0.5;
}
void print_samples(const char * key, const std::vector<double> & samples) {
    std::printf(",\"%s\":[", key);
    for (size_t i = 0; i < samples.size(); ++i) std::printf("%s%.6f", i ? "," : "", samples[i]);
    std::putchar(']');
}
void run(Stream & stream, const Options & options, const Input & bank, Buffer & packed,
         uint64_t rows, uint32_t selected, bool per_rank, const void * tables) {
    const auto ids = selection(selected);
    const auto x = activation(bank.cols, selected, per_rank);
    const auto oracle = reference(bank, rows, ids, x, per_rank);
    const auto ids_device = npu::from_host(ids.data(), {selected}, npu::Type::I64);
    const auto x_device = npu::from_host(x.data(), {int64_t(x.size())});
    const auto grouped = npu::empty({selected, int64_t(rows)});
    const auto expert_bytes = quant_storage_bytes(bank.f.type, rows * bank.cols);
    std::vector<std::shared_ptr<Buffer>> separate;
    for (uint32_t rank = 0; rank < selected; ++rank) separate.push_back(std::make_shared<Buffer>(rows * sizeof(float)));
    const auto grouped_launch = [&](const void * borrowed_tables) {
        quant_matvec_grouped_device(stream, bank.f.type, packed.data(), rows, bank.cols, selected, experts,
            static_cast<const int64_t *>(ids_device.data()), static_cast<const float *>(x_device.data()),
            static_cast<float *>(grouped.data()), per_rank, borrowed_tables);
    };
    const auto separate_launch = [&] {
        for (uint32_t rank = 0; rank < selected; ++rank)
            quant_matvec_device(stream, bank.f.type, static_cast<const uint8_t *>(packed.data()) + ids[rank] * expert_bytes,
                rows, bank.cols, static_cast<const float *>(x_device.data()) + (per_rank ? rank * bank.cols : 0),
                static_cast<float *>(separate[rank]->data()), tables);
    };
    std::vector<float> first;
    Check grouped_check;
    for (int repeat = 0; repeat < 3; ++repeat) {
        poison(*grouped.storage);
        {
            DeferredScope deferred(stream, repeat == 1);
            grouped_launch(repeat == 2 ? nullptr : tables);
            deferred.finish();
        }
        const auto actual = npu::read_f32(stream, grouped);
        grouped_check = verify(oracle, actual, "grouped");
        if (!repeat) first = actual;
        else same_bits(first, actual, "grouped-repeat");
    }
    for (auto & output : separate) poison(*output);
    separate_launch();
    const auto read_separate = [&] {
        std::vector<float> result;
        result.reserve(selected * rows);
        for (const auto & output : separate) {
            const auto part = read(*output, rows);
            result.insert(result.end(), part.begin(), part.end());
        }
        return result;
    };
    const auto singles = read_separate();
    const auto single_check = verify(oracle, singles, "separate-matvec");
    double pairwise = 0;
    for (size_t i = 0; i < singles.size(); ++i) pairwise = std::max(pairwise, std::abs(double(first[i]) - singles[i]));
    std::vector<float> scores(selected);
    for (uint32_t rank = 0; rank < selected; ++rank) scores[rank] = float(int(rank % 3) - 1) * 0.1328125f + 0.1015625f;
    aggregate_check(stream, grouped, first, rows, scores);
    print_check(oracle, "grouped", grouped_check);
    print_check(oracle, "separate-matvec", single_check);
    std::printf("GROUPED type=%s rows=%llu cols=%llu selected=%u input=%s max_pairwise_abs=%.9g repeat=bitexact ordered_aggregate=CPU_and_ACLNN_bitexact PASS\n",
        bank.f.name, (unsigned long long)rows, (unsigned long long)bank.cols, selected, per_rank ? "per-rank" : "shared", pairwise);
    if (!options.bench) return;
    const auto scores_device = npu::from_host(scores.data(), {selected});
    Buffer aggregate(rows * sizeof(float));
    npu::ND aclnn_sum;
    std::vector<double> group_ms, single_ms, aggregate_ms, aclnn_ms;
    for (int trial = -options.warmup; trial < options.repeats; ++trial) {
        double tg = 0, ts = 0;
        const auto group_step = [&] { poison(*grouped.storage); tg = timed(stream, options.deferred, [&] { grouped_launch(tables); }); };
        const auto single_step = [&] { for (auto & output : separate) poison(*output); ts = timed(stream, options.deferred, separate_launch); };
        if (trial % 2) { group_step(); single_step(); } else { single_step(); group_step(); }
        poison(aggregate);
        const auto ta = timed(stream, options.deferred, [&] {
            weighted_sum_experts_device(stream, selected, rows, static_cast<const float *>(grouped.data()),
                static_cast<const float *>(scores_device.data()), static_cast<float *>(aggregate.data()));
        });
        const auto tc = timed(stream, options.deferred, [&] {
            aclnn_sum = ordered_aclnn(stream, grouped, scores_device, rows, selected);
        });
        if (trial >= 0) { group_ms.push_back(tg); single_ms.push_back(ts); aggregate_ms.push_back(ta); aclnn_ms.push_back(tc); }
    }
    const auto final_grouped = npu::read_f32(stream, grouped);
    verify(oracle, final_grouped, "timed-grouped");
    verify(oracle, read_separate(), "timed-separate");
    const auto final_sum = ordered_reference(final_grouped, scores, rows);
    same_bits(final_sum, read(aggregate, rows), "timed-aggregate");
    same_bits(final_sum, npu::read_f32(stream, aclnn_sum), "timed-ACLNN-aggregate");
    std::printf("BENCH {\"type\":%d,\"name\":\"%s\",\"rows\":%llu,\"cols\":%llu,\"experts\":%u,\"selected\":%u,\"input\":\"%s\",\"deferred\":%s,\"packed_bank_bytes\":%zu,\"warmup\":%d,\"repeats\":%d,\"grouped_median_ms\":%.6f,\"separate_median_ms\":%.6f,\"matvec_speedup\":%.6f,\"ordered_aggregate_median_ms\":%.6f,\"ACLNN_aggregate_median_ms\":%.6f",
        bank.f.type, bank.f.name, (unsigned long long)rows, (unsigned long long)bank.cols, experts, selected, per_rank ? "per-rank" : "shared",
        options.deferred ? "true" : "false", bank.raw.size(), options.warmup, options.repeats, median(group_ms), median(single_ms), median(single_ms) / median(group_ms), median(aggregate_ms), median(aclnn_ms));
    print_samples("grouped_ms", group_ms); print_samples("separate_ms", single_ms);
    print_samples("aggregate_ms", aggregate_ms); print_samples("ACLNN_aggregate_ms", aclnn_ms);
    std::puts(",\"verified\":true}");
}
void invalid_ids(Stream & stream, const Input & bank, Buffer & packed, uint64_t rows, const void * tables) {
    const std::vector<int64_t> ids{0, -1, experts, 3, std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()};
    const auto x = activation(bank.cols, uint32_t(ids.size()), true);
    const auto id_device = npu::from_host(ids.data(), {int64_t(ids.size())}, npu::Type::I64);
    const auto x_device = npu::from_host(x.data(), {int64_t(x.size())});
    Buffer output(ids.size() * rows * sizeof(float));
    for (int repeat = 0; repeat < 2; ++repeat) {
        // A NaN poison would let an unwritten invalid rank pass this assertion.
        const std::vector<float> finite_poison(ids.size() * rows, repeat ? -123.25f : 456.75f);
        output.copy_from_host(finite_poison.data(), finite_poison.size() * sizeof(float));
        quant_matvec_grouped_device(stream, bank.f.type, packed.data(), rows, bank.cols, uint32_t(ids.size()), experts,
            static_cast<const int64_t *>(id_device.data()), static_cast<const float *>(x_device.data()), static_cast<float *>(output.data()), true, tables);
        const auto actual = read(output, ids.size() * rows);
        for (size_t rank = 0; rank < ids.size(); ++rank) {
            if (ids[rank] < 0 || ids[rank] >= experts) {
                for (uint64_t row = 0; row < rows; ++row) if (!std::isnan(actual[rank * rows + row])) throw std::runtime_error("invalid ID did not produce NaN");
            } else {
                const std::vector<float> xi(x.begin() + rank * bank.cols, x.begin() + (rank + 1) * bank.cols);
                const auto expected = reference(bank, rows, {ids[rank]}, xi, false);
                verify(expected, std::vector<float>(actual.begin() + rank * rows, actual.begin() + (rank + 1) * rows), "invalid-ID-valid-neighbor");
            }
        }
    }
    std::printf("INVALID_IDS type=%s rows=%llu cols=%llu negative_and_upper_bound=NaN valid_neighbors=verified PASS\n",
        bank.f.name, (unsigned long long)rows, (unsigned long long)bank.cols);
}
void aggregate_rounding(Stream & stream) {
    constexpr uint64_t rows = 49;
    const std::vector<float> scores{1.0f, 0x1.000002p+0f, 1.0f};
    std::vector<float> values(3 * rows);
    for (uint64_t row = 0; row < rows; ++row) {
        // Even rows distinguish separate Mul/Add from FMA; odd rows distinguish reduction ordering.
        values[row] = row % 2 ? 1e8f : -0x1.000004p+0f;
        values[rows + row] = row % 2 ? 1.0f : 0x1.000002p+0f;
        values[2 * rows + row] = row % 2 ? -1e8f : 0.0f;
    }
    const auto tensor = npu::from_host(values.data(), {3, int64_t(rows)});
    aggregate_check(stream, tensor, values, rows, scores);
    std::puts("AGGREGATE_ROUNDING rows=49 selected=3 FMA_sensitive_and_order_sensitive=CPU_and_ACLNN_bitexact PASS");
}
void rejection(Stream & stream, const void * tables) {
    Buffer raw(4096), x(4096), y(4096), ids(4096), scores(4096);
    const auto expect = [](const std::function<void()> & action) {
        try { action(); } catch (const std::runtime_error &) { return; }
        throw std::runtime_error("invalid grouped metadata was not rejected");
    };
    const auto call = [&](int type, uint64_t rows, uint64_t cols, uint32_t selected, uint32_t count) {
        quant_matvec_grouped_device(stream, type, raw.data(), rows, cols, selected, count,
            static_cast<const int64_t *>(ids.data()), static_cast<const float *>(x.data()), static_cast<float *>(y.data()), false, tables);
    };
    expect([&] { call(999, 8, 256, 1, 11); });
    expect([&] { call(18, 1, 640, 1, 11); });
    expect([&] { call(0, 8, 15, 1, 11); });
    expect([&] { call(0, 0, 8, 1, 11); });
    expect([&] { call(0, 8, 8, 0, 11); });
    expect([&] { call(0, 8, 8, 513, 11); });
    expect([&] { call(0, 8, 8, 1, 0); });
    expect([&] { call(0, std::numeric_limits<uint64_t>::max(), 8, 1, 11); });
    expect([&] { weighted_sum_experts_device(stream, 0, 8, static_cast<const float *>(x.data()), static_cast<const float *>(scores.data()), static_cast<float *>(y.data())); });
    expect([&] { weighted_sum_experts_device(stream, 513, 8, static_cast<const float *>(x.data()), static_cast<const float *>(scores.data()), static_cast<float *>(y.data())); });
    std::puts("METADATA invalid_type/partial_expert_block/column_alignment/empty/selection_limit/overflow rejected PASS");
}
} // namespace
int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        const auto options = parse(argc, argv);
        Device device(std::getenv("STRATA_CANN_DEVICE") ? std::stoi(std::getenv("STRATA_CANN_DEVICE")) : 0);
        Stream stream(device);
        const auto tables = make_quant_tables(stream);
        std::printf("Device: %s ordinal=%d. CPU decoding/double dots/ordered FP32 are test oracles only.\n", device_info(device.ordinal()).soc.c_str(), device.ordinal());
        std::puts("Benchmark: synchronized API wall time; packed bank, input, IDs, scores and codebooks resident. Excludes upload, CPU oracle, poison and readback; includes launches and waits. ACLNN aggregate includes its normal output allocations. Per-rank x uses selected rank, not expert ID.");
        rejection(stream, tables->data());
        aggregate_rounding(stream);
        size_t cases = 0;
        std::vector<std::pair<uint64_t, uint64_t>> shapes{{49, 256}, {50, 640}};
        if (!options.small) { shapes.emplace_back(640, 2560); shapes.emplace_back(2560, 640); }
        if (options.rows) shapes = {{options.rows, options.cols}};
        for (auto f : options.types) for (const auto & [rows, cols] : shapes) {
            check_shape(rows, cols);
            quant_storage_bytes(f.type, rows * cols); // Every expert independently starts on a quant block.
            if (rows > std::numeric_limits<uint64_t>::max() / experts) throw std::runtime_error("expert bank shape overflow");
            check_shape(rows * experts, cols);
            const auto bank = synthetic(f, rows * experts, cols);
            Buffer packed(bank.raw.size());
            packed.copy_from_host(bank.raw.data(), bank.raw.size());
            for (const auto selected : options.selected) for (bool per_rank : {false, true}) {
                if (options.input == "shared" && per_rank) continue;
                if (options.input == "per-rank" && !per_rank) continue;
                run(stream, options, bank, packed, rows, selected, per_rank, tables->data());
                ++cases;
            }
            invalid_ids(stream, bank, packed, rows, tables->data());
        }
        std::printf("GROUPED_ALL_PASS cases=%zu types=%zu CPU_double/separate_matvec/ordered_aggregate/repeat_poison/invalid_ids verified\n", cases, options.types.size());
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAILED: %s\n", error.what()); return 1;
    }
}
