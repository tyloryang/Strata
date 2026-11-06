#include "fused_matvec_test_common.hpp"
#include <chrono>

using namespace fused_test;
namespace {
struct Options {
    uint64_t rows = 640, cols = 2560, row_offset = 0, row_limit = 0;
    int warmup = 2, repeats = 7;
    std::string label = "current", path, tensor;
    std::vector<Format> selected{format(18)};
};
Options parse(int argc, char ** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") {
            std::puts("usage: fused_matvec_bench [--types IQ3_XXS,...] [--rows N] [--cols N] [--warmup N] [--repeats N] [--label text] [--gguf file --tensor name --row-offset N --row-limit N]");
            std::puts("Default IQ3_XXS 640x2560, warmup=2,repeats=7. Shapes 2560x640,6144x2560,248320x2560 require explicit selection. GGUF --row-limit defaults to all rows.");
            std::exit(0);
        }
        if (++i >= argc) throw std::runtime_error("missing value for " + key);
        const std::string value = argv[i];
        if (key == "--types") o.selected = parse_formats(value);
        else if (key == "--rows") o.rows = positive(value);
        else if (key == "--cols") o.cols = positive(value);
        else if (key == "--row-offset") o.row_offset = positive(value, true);
        else if (key == "--row-limit") o.row_limit = positive(value, true);
        else if (key == "--gguf") o.path = value;
        else if (key == "--tensor") o.tensor = value;
        else if (key == "--warmup" || key == "--repeats") {
            const auto n = positive(value, key == "--warmup");
            if (n > 1000) throw std::runtime_error("warmup/repeats must be <=1000");
            (key == "--warmup" ? o.warmup : o.repeats) = int(n);
        } else if (key == "--label") {
            if (value.empty() || value.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-") != std::string::npos)
                throw std::runtime_error("label requires letters/digits/dot/underscore/hyphen");
            o.label = value;
        } else throw std::runtime_error("unknown option " + key);
    }
    if (o.path.empty() != o.tensor.empty()) throw std::runtime_error("--gguf and --tensor must be supplied together");
    return o;
}
template<class F> double timed(Stream & stream, F && operation) {
    stream.synchronize();
    const auto begin = std::chrono::steady_clock::now();
    operation();
    stream.synchronize();
    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - begin).count();
}
double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    return values.size() % 2 ? values[mid] : (values[mid - 1] + values[mid]) * 0.5;
}
void samples(const char * name, const std::vector<double> & values) {
    std::printf(",\"%s\":[", name);
    for (size_t i = 0; i < values.size(); ++i) std::printf("%s%.6f", i ? "," : "", values[i]);
    std::putchar(']');
}
void bench(Stream & stream, const Options & o, const Input & in) {
    stream.activate();
    const auto tables = make_quant_tables(stream);
    Buffer packed(in.raw.size()), fused(in.rows * sizeof(float));
    auto decoded = std::make_shared<Buffer>(in.rows * in.cols * sizeof(float));
    packed.copy_from_host(in.raw.data(), in.raw.size());
    const auto x = npu::from_host(in.x.data(), {int64_t(in.cols), 1});
    const auto matrix = npu::wrap(decoded, {int64_t(in.rows), int64_t(in.cols)});
    std::vector<double> fused_ms, decode_ms, decoded_matmul_ms, pipeline_ms;
    npu::ND decoded_result, pipeline_result;
    for (int trial = -o.warmup; trial < o.repeats; ++trial) {
        double tf = 0, td = 0, tm = 0, tp = 0;
        const auto fused_step = [&] {
            poison(fused);
            tf = timed(stream, [&] {
                quant_matvec_device(stream, in.f.type, packed.data(), in.rows, in.cols,
                                    static_cast<const float *>(x.data()), static_cast<float *>(fused.data()), tables->data());
            });
        };
        const auto pipeline_step = [&] {
            poison(*decoded);
            tp = timed(stream, [&] {
                dequantize_device(stream, in.f.type, packed.data(), in.rows * in.cols, decoded->data(), tables->data());
                pipeline_result = npu::matmul(stream, matrix, x);
            });
        };
        // Alternate the compared paths to avoid always giving one the warmer card.
        if (trial % 2) { fused_step(); pipeline_step(); }
        else { pipeline_step(); fused_step(); }
        poison(*decoded);
        td = timed(stream, [&] { dequantize_device(stream, in.f.type, packed.data(), in.rows * in.cols, decoded->data(), tables->data()); });
        tm = timed(stream, [&] { decoded_result = npu::matmul(stream, matrix, x); });
        if (trial >= 0) {
            fused_ms.push_back(tf); pipeline_ms.push_back(tp); decode_ms.push_back(td); decoded_matmul_ms.push_back(tm);
        }
    }
    const auto cf = verify(in, read(fused, in.rows), "fused");
    const auto cp = verify(in, npu::read_f32(stream, pipeline_result), "dequant+aclnn_f32");
    const auto cm = verify(in, npu::read_f32(stream, decoded_result), "resident_aclnn_f32");
    print_check(in, "fused", cf);
    print_check(in, "dequant+aclnn_f32", cp);
    print_check(in, "resident_aclnn_f32", cm);
    const double fused_median = median(fused_ms), pipeline_median = median(pipeline_ms);
    std::printf("BENCH {\"label\":\"%s\",\"type\":%d,\"name\":\"%s\",\"rows\":%llu,\"cols\":%llu,\"packed_bytes\":%zu,\"decoded_bytes\":%zu,\"warmup\":%d,\"repeats\":%d,\"fused_median_ms\":%.6f,\"dequant_median_ms\":%.6f,\"resident_f32_matmul_median_ms\":%.6f,\"dequant_matmul_median_ms\":%.6f,\"pipeline_speedup\":%.6f,\"fused_max_abs_error\":%.9g,\"fused_max_rel_error\":%.9g,\"fused_argmax\":%zu,\"oracle_argmax\":%zu",
        o.label.c_str(), in.f.type, in.f.name, (unsigned long long)in.rows, (unsigned long long)in.cols,
        in.raw.size(), decoded->size(), o.warmup, o.repeats, fused_median, median(decode_ms), median(decoded_matmul_ms), pipeline_median,
        pipeline_median / fused_median, cf.max_abs, cf.max_rel, cf.actual_argmax, cf.reference_argmax);
    samples("fused_ms", fused_ms); samples("dequant_ms", decode_ms);
    samples("resident_f32_matmul_ms", decoded_matmul_ms); samples("dequant_matmul_ms", pipeline_ms);
    std::puts(",\"verified\":true}");
}
}
int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        const auto o = parse(argc, argv);
        Device device(std::getenv("STRATA_CANN_DEVICE") ? std::stoi(std::getenv("STRATA_CANN_DEVICE")) : 0);
        Stream stream(device);
        std::printf("Device: %s ordinal=%d. Scope: synchronized API wall time, packed W and x resident in all paths.\n", device_info(device.ordinal()).soc.c_str(), device.ordinal());
        std::puts("Excluded: generation/GGUF I/O, input/codebook H2D, output poison/readback, CPU-double oracle. Codebooks remain resident and are shared by all compared paths. Included: API setup, ACLNN output allocation, synchronization. Resident F32 matmul is separately reported; it requires expanded weight HBM.");
        if (o.path.empty()) for (auto f : o.selected) bench(stream, o, synthetic(f, o.rows, o.cols));
        else {
            const Gguf gguf(o.path);
            const auto * tensor = gguf.find(o.tensor);
            if (!tensor) throw std::runtime_error("GGUF tensor missing: " + o.tensor);
            bench(stream, o, gguf_input(gguf, *tensor, o.row_offset, o.row_limit));
        }
        std::puts("All timed paths verified against CPU-double oracle after timing.");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAILED: %s\n", error.what()); return 1;
    }
}
