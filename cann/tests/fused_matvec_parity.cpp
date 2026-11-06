#include "fused_matvec_test_common.hpp"

using namespace fused_test;
namespace {
void run(Stream & stream, const Input & in, const void * tables) {
    stream.activate();
    Buffer packed(in.raw.size()), fused(in.rows * sizeof(float));
    auto decoded = std::make_shared<Buffer>(in.rows * in.cols * sizeof(float));
    packed.copy_from_host(in.raw.data(), in.raw.size());
    const auto x = npu::from_host(in.x.data(), {int64_t(in.cols), 1});
    // Re-poison and repeat to catch missed writes and cross-core output races.
    Check fused_check;
    std::vector<float> first;
    for (int repeat = 0; repeat < 3; ++repeat) {
        poison(fused);
        quant_matvec_device(stream, in.f.type, packed.data(), in.rows, in.cols,
                            static_cast<const float *>(x.data()), static_cast<float *>(fused.data()), repeat == 2 ? nullptr : tables);
        const auto result = read(fused, in.rows);
        fused_check = verify(in, result, "fused");
        if (repeat == 0) first = result;
        else if (std::memcmp(first.data(), result.data(), result.size() * sizeof(float)))
            throw std::runtime_error("nondeterministic fused output: " + in.label);
    }
    print_check(in, "fused", fused_check);
    dequantize_device(stream, in.f.type, packed.data(), in.rows * in.cols, decoded->data(), tables);
    const auto matrix = npu::wrap(decoded, {int64_t(in.rows), int64_t(in.cols)});
    const auto reference = npu::read_f32(stream, npu::matmul(stream, matrix, x));
    print_check(in, "dequant+aclnn_f32", verify(in, reference, "dequant+aclnn_f32"));
    double maximum = 0;
    for (size_t i = 0; i < first.size(); ++i) maximum = std::max(maximum, std::abs(double(first[i]) - reference[i]));
    std::printf("PAIRWISE fused_vs_aclnn max_abs=%.9g\n", maximum);
}
void rejection(Stream & stream) {
    Buffer packed(1024), x(4096), y(4096);
    auto expect = [&](int type, uint64_t rows, uint64_t cols) {
        bool rejected = false;
        try { quant_matvec_device(stream, type, packed.data(), rows, cols,
                  static_cast<const float *>(x.data()), static_cast<float *>(y.data())); }
        catch (const std::runtime_error &) { rejected = true; }
        if (!rejected) throw std::runtime_error("invalid matvec input was not rejected");
    };
    expect(999, 1, 256);
    expect(18, 1, 640); // valid column alignment, incomplete flat block
    expect(0, 1, 15);
    expect(0, std::numeric_limits<uint64_t>::max(), 8);
    expect(0, 0, 8);
    std::puts("Invalid type, partial flat block, column alignment, overflow and empty shape rejected PASS");
}
}
int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        std::vector<Format> selected(std::begin(formats), std::end(formats));
        std::vector<std::string> paths, tensors;
        uint64_t row_offset = 0, row_limit = 64;
        bool synthetic_enabled = true, large = false;
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--help") {
                std::puts("usage: fused_matvec_parity [--types names] [--large] [--no-synthetic] [--gguf path ...] [--tensor name ...] [--row-offset N] [--row-limit N]");
                std::puts("Default: all 15 formats; tiny/tail/real dimensions, including cross-block cols=640. GGUF defaults to first matrix of each selected type, <=64 rows; --row-limit 0 uses all rows. --large adds 6144x2560.");
                return 0;
            }
            if (key == "--no-synthetic") { synthetic_enabled = false; continue; }
            if (key == "--large") { large = true; continue; }
            if (++i >= argc) throw std::runtime_error("missing value for " + key);
            const std::string value = argv[i];
            if (key == "--types") selected = parse_formats(value);
            else if (key == "--gguf") paths.push_back(value);
            else if (key == "--tensor") tensors.push_back(value);
            else if (key == "--row-offset") row_offset = positive(value, true);
            else if (key == "--row-limit") row_limit = positive(value, true);
            else throw std::runtime_error("unknown option " + key);
        }
        if (!synthetic_enabled && paths.empty()) throw std::runtime_error("no test cases requested");
        Device device(std::getenv("STRATA_CANN_DEVICE") ? std::stoi(std::getenv("STRATA_CANN_DEVICE")) : 0);
        Stream stream(device);
        const auto tables = make_quant_tables(stream);
        std::printf("Device: %s ordinal=%d. CPU double arithmetic is test oracle only.\n", device_info(device.ordinal()).soc.c_str(), device.ordinal());
        rejection(stream);
        size_t count = 0;
        if (synthetic_enabled) for (auto f : selected) {
            // Vary row count around core/output-line boundaries and tile lengths.
            for (const auto & shape : std::vector<std::pair<uint64_t,uint64_t>>{
                    {1,256}, {3,2560}, {49,2560}, {65,6144}, {640,2560}, {2560,640}, {8,16384}}) {
                run(stream, synthetic(f, shape.first, shape.second), tables->data()); ++count;
            }
            if (f.type == 0 || f.type == 1 || f.type == 30) {
                run(stream, synthetic(f, 3, 24), tables->data()); ++count;
                run(stream, synthetic(f, 65, 320), tables->data()); ++count;
                run(stream, synthetic(f, 3, 264), tables->data()); ++count;
            }
            if (large) { run(stream, synthetic(f, 6144, 2560), tables->data()); ++count; }
        }
        std::set<int> seen_types;
        std::set<std::string> seen_tensors;
        for (const auto & path : paths) {
            Gguf gguf(path);
            for (const auto & tensor : gguf.tensors()) {
                if (tensor.dims.size() < 2 || tensor.dims[0] % 8) continue;
                if (std::none_of(selected.begin(), selected.end(), [&](Format f) { return f.type == int(tensor.type); })) continue;
                if (tensors.empty()) { if (seen_types.count(int(tensor.type))) continue; }
                else if (std::find(tensors.begin(), tensors.end(), tensor.name) == tensors.end()) continue;
                const auto input = gguf_input(gguf, tensor, row_offset, row_limit);
                run(stream, input, tables->data()); ++count;
                seen_types.insert(int(tensor.type)); seen_tensors.insert(tensor.name);
            }
        }
        for (const auto & name : tensors) if (!seen_tensors.count(name)) throw std::runtime_error("requested tensor not tested: " + name);
        if (!paths.empty() && seen_types.empty()) throw std::runtime_error("no matching GGUF matrices tested");
        std::printf("All %zu fused matvec cases passed CPU-double and independent NPU-dequant+ACLNN comparisons.\n", count);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAILED: %s\n", error.what()); return 1;
    }
}
