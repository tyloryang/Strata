// Device-resident Qwen4Exp forward. Quantized bytes are decoded on the NPU;
// all floating-point activations, recurrent states and KV caches remain in HBM.
#include "strata/cann/qwen38_npu.hpp"
#include "strata/cann/dequant_npu.hpp"
#include "strata/cann/gguf.hpp"
#include "strata/cann/npu_tensor.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <list>
#include <stdexcept>
#include <unordered_map>

namespace strata::cann {
namespace {
using npu::ND;
using npu::Type;
using npu::Unary;
using npu::Binary;
void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error("qwen4exp NPU: " + message);
}
std::string second_shard(std::string path) {
    const std::string marker = "00001-of-00002";
    const auto at = path.rfind(marker);
    require(at != std::string::npos, "pass the original first GGUF shard");
    path.replace(at, marker.size(), "00002-of-00002");
    return path;
}
}

struct Qwen38NpuModel::Impl {
    Gguf gguf, table;
    std::vector<std::unique_ptr<Device>> devices;
    std::vector<std::unique_ptr<Stream>> device_streams;
    size_t active_slot = 0;
    bool packed_mode = false, fused_mode = false, dense_cache = false, deferred_mode = false;
    bool grouped_experts = false;
    struct Packed { std::shared_ptr<Buffer> bytes; size_t owner; };
    std::unordered_map<std::string, Packed> packed;
    std::vector<uint64_t> packed_sizes;
    std::vector<std::shared_ptr<Buffer>> quant_tables;
    uint64_t peer_bytes = 0, fused_calls = 0, dense_hits = 0, dense_bytes = 0;
    uint64_t grouped_launches = 0, aggregate_launches = 0;
    std::unordered_map<std::string, ND> dense_weights;
    Stream& current_stream() { return *device_streams.at(active_slot); }
    std::vector<std::unique_ptr<DeferredScope>> defer_submission() {
        std::vector<std::unique_ptr<DeferredScope>> scopes;
        if (deferred_mode) for (auto& stream : device_streams)
            scopes.push_back(std::make_unique<DeferredScope>(*stream, true));
        return scopes;
    }
    void activate(size_t slot) { active_slot = slot; devices.at(slot)->activate(); }
    size_t layer_slot(int64_t layer) const { return std::min(devices.size() - 1, size_t(layer) * devices.size() / size_t(layers)); }
    size_t weight_slot(const std::string& name) const {
        if (name.rfind("blk.", 0) == 0) return layer_slot(std::stoll(name.substr(4)));
        return name.rfind("output", 0) == 0 ? devices.size() - 1 : 0;
    }
    int64_t cap, n, layers, hc, kh, vh, sd, conv_k, heads, kvh, hd, rot;
    int64_t interval, experts, top_k, idx_dim, idx_heads, idx_topk, vocab_count, position = 0;
    float eps, base;
    size_t budget, used = 0;
    uint64_t matmuls = 0, hits = 0, misses = 0;
    struct Cached { ND value; std::list<std::string>::iterator age; };
    std::unordered_map<std::string, Cached> weights;
    std::list<std::string> ages;
    std::unordered_map<std::string, ND> vectors;
    std::unordered_map<std::string, ND> inverse_roots;
    struct State {
        std::deque<ND> conv, ple;
        ND recurrent, keys, values, index;
    };
    std::vector<State> states;
    std::vector<int64_t> tokens;
    ND hidden;
    std::vector<ND> frequencies, gdn_mapping, attention_mapping, ple_embeddings;

    int64_t integer(const std::string& key) const { return gguf.metadata_int("qwen4exp." + key); }
    const std::vector<uint64_t>& array(const std::string& key) const {
        return gguf.metadata_uints("qwen4exp." + key);
    }
    const GgufTensor& tensor(const std::string& name) const {
        const auto* value = gguf.find(name);
        require(value != nullptr, "missing tensor " + name);
        return *value;
    }
    ND flat(const ND& x) { return npu::reshape(current_stream(), x, {x.numel()}); }
    ND shape(const ND& x, std::vector<int64_t> dims) { return npu::reshape(current_stream(), x, dims); }
    ND add(const ND& a, const ND& b) { return npu::binary(current_stream(), a, b, Binary::Add); }
    ND sub(const ND& a, const ND& b) { return npu::binary(current_stream(), a, b, Binary::Sub); }
    ND mul(const ND& a, const ND& b) { return npu::binary(current_stream(), a, b, Binary::Mul); }
    ND div(const ND& a, const ND& b) { return npu::binary(current_stream(), a, b, Binary::Div); }
    ND unary(const ND& x, Unary op) { return npu::unary(current_stream(), x, op); }
    ND scale(const ND& x, float scalar) { return npu::scale(current_stream(), x, scalar); }
    ND scalar(float value) { return npu::from_host(&value, {1}); }
    const ND& inverse_root(int64_t value) {
        const auto key = std::to_string(active_slot) + ":" + std::to_string(value);
        auto it = inverse_roots.find(key);
        if (it == inverse_roots.end()) {
            it = inverse_roots.emplace(key, unary(scalar(float(value)), Unary::Rsqrt)).first;
        }
        return it->second;
    }
    ND load(const Gguf& source, const GgufTensor& t, uint64_t offset, uint64_t count) {
        const auto it = packed.find(t.name);
        if (it == packed.end()) {
            require(!packed_mode, "packed weight missing: " + t.name);
            return npu::wrap(load_quant_device(current_stream(), source, t, offset, count), {int64_t(count)});
        }
        const uint64_t begin = offset ? quant_storage_bytes(t.type, offset) : 0;
        const uint64_t bytes = quant_storage_bytes(t.type, count);
        require(begin <= it->second.bytes->size() && bytes <= it->second.bytes->size() - begin, "packed slice out of range");
        const auto* raw = static_cast<const uint8_t*>(it->second.bytes->data()) + begin;
        std::shared_ptr<Buffer> local;
        if (it->second.owner != active_slot) {
            local = std::make_shared<Buffer>(bytes);
            copy_peer(local->data(), devices[active_slot]->ordinal(), raw, devices[it->second.owner]->ordinal(), bytes);
            peer_bytes += bytes; raw = static_cast<const uint8_t*>(local->data());
        }
        auto output = npu::empty({int64_t(count)});
        dequantize_device(current_stream(), t.type, raw, count, output.data(), quant_tables[active_slot]->data());
        return output;
    }
    const ND& vector(const std::string& name) {
        auto it = vectors.find(name);
        if (it == vectors.end()) {
            const auto& t = tensor(name);
            it = vectors.emplace(name, load(gguf, t, 0, t.elements())).first;
        }
        return it->second;
    }
    void preload(const Gguf& source, const GgufTensor& t, size_t owner) {
        activate(owner);
        const uint64_t bytes = quant_storage_bytes(t.type, t.elements());
        auto storage = std::make_shared<Buffer>(bytes);
        constexpr uint64_t chunk = 64ULL * 1024 * 1024;
        for (uint64_t at = 0; at < bytes; at += chunk) {
            const auto host = source.tensor_bytes(t, at, std::min(chunk, bytes - at));
            copy_to_device(static_cast<uint8_t*>(storage->data()) + at, host.data(), host.size());
        }
        require(packed.emplace(t.name, Packed{std::move(storage), owner}).second, "duplicate packed tensor " + t.name);
        packed_sizes[owner] += bytes;
    }
    Impl(const std::string& path, int64_t capacity)
        : gguf(path), table(second_shard(path)), cap(capacity) {
        std::string selected = std::getenv("STRATA_CANN_DEVICES") ? std::getenv("STRATA_CANN_DEVICES") :
            (std::getenv("STRATA_CANN_DEVICE") ? std::getenv("STRATA_CANN_DEVICE") : "0");
        std::stringstream spec(selected);
        std::string part;
        while (std::getline(spec, part, ',')) {
            size_t end = 0; const int ordinal = std::stoi(part, &end);
            require(end == part.size() && ordinal >= 0, "invalid device list");
            for (const auto& d : devices) require(d->ordinal() != ordinal, "duplicate device");
            devices.push_back(std::make_unique<Device>(ordinal));
            device_streams.push_back(std::make_unique<Stream>(*devices.back()));
        }
        require(!devices.empty() && devices.size() <= 2, "select one or two devices");
        packed_mode = std::getenv("STRATA_CANN_PACKED_WEIGHTS") && std::string(std::getenv("STRATA_CANN_PACKED_WEIGHTS")) == "1";
        fused_mode = std::getenv("STRATA_CANN_FUSED_MATVEC") && std::string(std::getenv("STRATA_CANN_FUSED_MATVEC")) == "1";
        require(!fused_mode || packed_mode, "fused matvec requires compressed residency");
        dense_cache = std::getenv("STRATA_CANN_DENSE_CACHE") && std::string(std::getenv("STRATA_CANN_DENSE_CACHE")) == "1";
        require(!dense_cache || fused_mode, "dense cache requires the packed fused path");
        deferred_mode = std::getenv("STRATA_CANN_DEFERRED") && std::string(std::getenv("STRATA_CANN_DEFERRED")) == "1";
        require(!deferred_mode || fused_mode, "deferred submission requires the packed fused path");
        grouped_experts = std::getenv("STRATA_CANN_GROUPED_EXPERTS") && std::string(std::getenv("STRATA_CANN_GROUPED_EXPERTS")) == "1";
        require(!grouped_experts || fused_mode, "grouped experts require the packed fused path");
        for (const auto& a : devices) for (const auto& b : devices)
            if (a->ordinal() != b->ordinal()) enable_peer_access(a->ordinal(), b->ordinal());
        packed_sizes.resize(devices.size());
        quant_tables.resize(devices.size());
        frequencies.resize(devices.size()); gdn_mapping.resize(devices.size()); attention_mapping.resize(devices.size());
        ple_embeddings.resize(devices.size());
        activate(0);
        require(std::getenv("STRATA_CANN_NO_SYNC") == nullptr,
                "STRATA_CANN_NO_SYNC is unsafe with the current allocator");
        n = integer("embedding_length"); layers = integer("block_count");
        hc = integer("hyper_connection.count"); kh = integer("ssm.group_count");
        vh = integer("ssm.time_step_rank"); sd = integer("ssm.state_size"); conv_k = integer("ssm.conv_kernel");
        heads = integer("attention.head_count"); kvh = integer("attention.head_count_kv");
        hd = integer("attention.key_length"); rot = integer("rope.dimension_count");
        interval = integer("full_attention_interval"); experts = integer("expert_count");
        top_k = integer("expert_used_count"); idx_dim = integer("attention.indexer.key_length");
        idx_heads = integer("attention.indexer.head_count"); idx_topk = integer("attention.indexer.top_k");
        eps = float(gguf.metadata_flt("qwen4exp.attention.layer_norm_rms_epsilon"));
        base = float(gguf.metadata_flt("qwen4exp.rope.freq_base"));
        require(n == 2560 && layers == 48 && hc == 4 && kh == 16 && vh == 48 && sd == 128 && conv_k == 4 &&
                heads == 24 && kvh == 2 && hd == 256 && rot == 64 && interval == 4 && experts == 512 && top_k == 10 &&
                idx_dim == 128 && idx_heads == 4 && idx_topk > 0 && eps > 0 && base > 1,
                "only original Flash-Next geometry is supported");
        require(cap > 0 && cap <= integer("context_length"), "invalid context capacity");
        require(table.find("per_layer_token_embd.weight") != nullptr, "second shard lacks PLE table");
        require(array("ple.layer_multipliers").size() >= 3 && array("ple.head_offsets").size() == 16 &&
                array("ple.head_vocab_sizes").size() == 16 && array("attention.compress_ratios").size() == size_t(layers),
                "invalid PLE/QSA metadata");
        vocab_count = tensor("output.weight").dims.at(1);
        const double gib = std::getenv("STRATA_BUDGET_GB") ? std::stod(std::getenv("STRATA_BUDGET_GB")) : 12;
        require(std::isfinite(gib) && gib >= 3 && gib <= 48, "weight budget must be between 3 and 48 GiB");
        budget = size_t(gib * 1073741824.0);
        states.resize(layers);
        // These host arrays contain geometry indices only, not computed activations.
        for (size_t slot = 0; slot < devices.size(); ++slot) {
        activate(slot);
        quant_tables[slot] = make_quant_tables(current_stream());
        std::vector<float> frequency_indices(rot / 2);
        for (int64_t i = 0; i < rot / 2; ++i) frequency_indices[i] = float(i);
        const auto indices = npu::from_host(frequency_indices.data(), {rot / 2});
        frequencies[active_slot] = unary(mul(scale(indices, -2.0f / float(rot)), unary(scalar(base), Unary::Log)), Unary::Exp);
        std::vector<int64_t> mapping(vh);
        for (int64_t i = 0; i < vh; ++i) mapping[i] = i % kh;
        gdn_mapping[active_slot] = npu::from_host(mapping.data(), {vh}, Type::I64);
        mapping.resize(heads);
        for (int64_t i = 0; i < heads; ++i) mapping[i] = i / (heads / kvh);
        attention_mapping[active_slot] = npu::from_host(mapping.data(), {heads}, Type::I64);
        }
        if (packed_mode) {
            const auto start = std::chrono::steady_clock::now();
            // The original packed representation stays in HBM. The large shared
            // PLE table lives on card 0; selected rows travel by P2P to card 1.
            std::vector<uint64_t> required(devices.size(), 0), dense_required(devices.size(), 0);
            const auto cacheable = [](const GgufTensor& t) {
                return t.dims.size() == 2 && t.name != "token_embd.weight" && t.dims[0] >= 320;
            };
            for (const auto& t : gguf.tensors()) {
                required[weight_slot(t.name)] += quant_storage_bytes(t.type, t.elements());
                if (dense_cache && cacheable(t)) dense_required[weight_slot(t.name)] += t.elements() * sizeof(float);
            }
            for (const auto& t : table.tensors()) required[0] += quant_storage_bytes(t.type, t.elements());
            for (size_t slot = 0; slot < devices.size(); ++slot) {
                const auto info = device_info(devices[slot]->ordinal());
                const uint64_t reserve = 3ULL * 1024 * 1024 * 1024 + (fused_mode ? 0 : budget) + dense_required[slot];
                require(required[slot] + reserve < info.free_bytes,
                        "packed residency plus workspace/cache exceeds device " + std::to_string(info.ordinal) +
                        " free HBM; select both devices, disable dense cache, reduce a non-fused cache budget, or free HBM");
                fprintf(stderr, "PACKED_PLAN device=%d bytes=%llu free=%llu\n", info.ordinal,
                        (unsigned long long)required[slot], (unsigned long long)info.free_bytes);
            }
            for (const auto& t : gguf.tensors()) preload(gguf, t, weight_slot(t.name));
            for (const auto& t : table.tensors()) preload(table, t, 0);
            if (dense_cache) for (const auto& t : gguf.tensors()) if (cacheable(t)) {
                activate(weight_slot(t.name));
                auto value = shape(load(gguf, t, 0, t.elements()), {int64_t(t.dims[1]), int64_t(t.dims[0])});
                dense_bytes += t.elements() * sizeof(float);
                dense_weights.emplace(t.name, std::move(value));
            }
            if (dense_cache) for (size_t slot = 0; slot < devices.size(); ++slot)
                fprintf(stderr, "DENSE_RESIDENT device=%d bytes=%llu GiB=%.3f decode=NPU dtype=F32\n",
                        devices[slot]->ordinal(), (unsigned long long)dense_required[slot], dense_required[slot] / 1073741824.0);
            for (size_t slot = 0; slot < devices.size(); ++slot)
                fprintf(stderr, "PACKED_RESIDENT device=%d bytes=%llu GiB=%.3f\n", devices[slot]->ordinal(),
                        (unsigned long long)packed_sizes[slot], packed_sizes[slot] / 1073741824.0);
            fprintf(stderr, "PACKED_LOAD seconds=%.6f tensors=%zu\n",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(), packed.size());
        }
        activate(0);
        fprintf(stderr, "native strata-cann qwen4exp NPU_ONLY: device=%d layers=%lld hidden=%lld context=%lld "
                        "weight_decode=NPU projections=NPU activation_state=NPU CPU_fallback=disabled\n",
                devices[0]->ordinal(), (long long)layers, (long long)n, (long long)cap);
    }

    ND linear(const std::string& name, const ND& input, int expert = -1) {
        const auto& t = tensor(name);
        const int64_t columns = int64_t(t.dims.at(0));
        require(columns == input.numel(), "linear input shape for " + name);
        const int64_t rows = t.dims.size() > 1 ? int64_t(t.dims[1]) : 1;
        const uint64_t count = uint64_t(rows) * columns;
        if (expert >= 0) require(t.dims.size() == 3 && uint64_t(expert) < t.dims[2], "invalid expert " + name);
        else require(t.dims.size() <= 2, "expert selection required for " + name);
        const auto dense = dense_weights.find(name);
        if (dense != dense_weights.end()) {
            require(expert < 0, "dense cache cannot contain routed expert tensors");
            require(dense->second.storage->device_ordinal() == devices[active_slot]->ordinal(), "dense cache on wrong device");
            ++matmuls; ++dense_hits;
            return flat(npu::matmul(current_stream(), dense->second, shape(input, {columns, 1})));
        }
        if (fused_mode) {
            const auto& weight = packed.at(name);
            require(weight.owner == active_slot, "projection on wrong device: " + name);
            require(input.storage->device_ordinal() == devices[active_slot]->ordinal(), "projection input on wrong device");
            const uint64_t begin = expert <= 0 ? 0 : quant_storage_bytes(t.type, uint64_t(expert) * count);
            const auto x = npu::contiguous(current_stream(), input);
            auto y = npu::empty({rows});
            quant_matvec_device(current_stream(), t.type, static_cast<const uint8_t*>(weight.bytes->data()) + begin,
                                rows, columns, static_cast<const float*>(x.data()), static_cast<float*>(y.data()), quant_tables[active_slot]->data());
            ++matmuls; ++fused_calls;
            return y;
        }
        const std::string key = name + "#" + std::to_string(expert);
        auto it = weights.find(key);
        if (it == weights.end()) {
            ++misses;
            require(count * sizeof(float) <= budget, "weight exceeds HBM cache budget: " + name);
            while (used + count * sizeof(float) > budget && !ages.empty()) {
                const auto victim = ages.back(); ages.pop_back();
                used -= weights.at(victim).value.storage->size(); weights.erase(victim);
            }
            auto value = shape(load(gguf, t, expert < 0 ? 0 : uint64_t(expert) * count, count), {rows, columns});
            ages.push_front(key);
            it = weights.emplace(key, Cached{std::move(value), ages.begin()}).first;
            used += count * sizeof(float);
        } else {
            ++hits;
            ages.splice(ages.begin(), ages, it->second.age);
        }
        ++matmuls;
        return flat(npu::matmul(current_stream(), it->second.value, shape(input, {columns, 1})));
    }
    ND grouped_linear(const std::string& name, const ND& input, const ND& indices, bool separate_inputs) {
        const auto& t = tensor(name);
        require(t.dims.size() == 3 && int64_t(t.dims[2]) == experts, "grouped expert tensor shape: " + name);
        const int64_t columns = int64_t(t.dims[0]), rows = int64_t(t.dims[1]);
        require(input.type == Type::F32 && input.numel() == columns * (separate_inputs ? top_k : 1), "grouped input shape: " + name);
        require(indices.type == Type::I64 && indices.numel() == top_k, "grouped indices shape");
        const auto& weight = packed.at(name);
        require(weight.owner == active_slot && input.storage->device_ordinal() == devices[active_slot]->ordinal() &&
                indices.storage->device_ordinal() == devices[active_slot]->ordinal(), "grouped input on wrong device");
        const auto x = npu::contiguous(current_stream(), input);
        const auto ids = npu::contiguous(current_stream(), indices);
        auto y = npu::empty({top_k, rows});
        quant_matvec_grouped_device(current_stream(), t.type, weight.bytes->data(), rows, columns,
            uint32_t(top_k), uint32_t(experts), static_cast<const int64_t*>(ids.data()),
            static_cast<const float*>(x.data()), static_cast<float*>(y.data()), separate_inputs, quant_tables[active_slot]->data());
        matmuls += top_k; fused_calls += top_k; ++grouped_launches;
        return y;
    }
    ND norm(const ND& x, const ND& gamma, int64_t width) {
        require(width > 0 && x.numel() % width == 0 &&
                (gamma.numel() == width || gamma.numel() == x.numel()), "RMS norm shape");
        const auto rows = shape(x, {x.numel() / width, width});
        const auto variance = npu::mean(current_stream(), mul(rows, rows), -1, true);
        const auto normalized = mul(rows, unary(npu::add_scalar(current_stream(), variance, eps), Unary::Rsqrt));
        return shape(mul(normalized, shape(gamma, {gamma.numel() / width, width})), x.shape);
    }
    ND l2_norm(const ND& rows) {
        const auto squares = npu::sum(current_stream(), mul(rows, rows), -1, true);
        return mul(rows, unary(npu::add_scalar(current_stream(), squares, eps), Unary::Rsqrt));
    }
    ND rope(const ND& x, int64_t width, int64_t pos) {
        require(width >= rot && x.numel() % width == 0, "RoPE shape");
        const auto rows = shape(x, {x.numel() / width, width});
        const auto angles = scale(frequencies[active_slot], float(pos));
        const auto c = unary(angles, Unary::Cos), s = unary(angles, Unary::Sin);
        const auto left = npu::slice(rows, 1, 0, rot / 2), right = npu::slice(rows, 1, rot / 2, rot / 2);
        std::vector<ND> pieces{sub(mul(left, c), mul(right, s)), add(mul(left, s), mul(right, c))};
        if (width > rot) pieces.push_back(npu::slice(rows, 1, rot, width - rot));
        return shape(npu::cat(current_stream(), pieces, 1), x.shape);
    }
    std::pair<ND, ND> hc_mix(const ND& residual, const std::string& prefix, bool inject = true) {
        const auto normalized = norm(residual, vector(prefix + "norm.weight"), n);
        const auto low = unary(scale(linear(prefix + "down.weight", normalized), 1.0f / float(hc)), Unary::Silu);
        const auto gate = shape(linear(prefix + "up.weight", low), {hc, n});
        const auto mixed = npu::mean(current_stream(), mul(normalized, unary(gate, Unary::Sigmoid)), 0, false);
        return {mixed, inject ? linear(prefix + "inject.weight", normalized) : ND{}};
    }
    ND combine(const ND& residual, const ND& output, const ND& inject) {
        require(output.numel() == n && inject.numel() == hc, "HC combine shape");
        const auto gate = scale(unary(scale(inject, 1.0f / float(hc)), Unary::Sigmoid), 2.0f);
        return add(residual, mul(shape(output, {1, n}), shape(gate, {hc, 1})));
    }
    ND ple(const ND& residual, int64_t token, int il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const uint64_t eos = integer("ple.eos_token_id"), width = integer("embedding_length_per_layer_input");
        const auto& multipliers = array("ple.layer_multipliers");
        const auto& offsets = array("ple.head_offsets");
        const auto& vocabs = array("ple.head_vocab_sizes");
        uint64_t ctx[3] = {uint64_t(token), eos, eos}; bool cut = false;
        for (int back = 1; back <= 2; ++back) {
            cut = cut || tokens.size() < size_t(back) || uint64_t(tokens[tokens.size() - back]) == eos;
            ctx[back] = cut ? eos : uint64_t(tokens[tokens.size() - back]);
        }
        if (!ple_embeddings[active_slot].storage) {
            if (active_slot && ple_embeddings[0].storage) {
                // PLE hashes depend on token history, not the layer. Decode the
                // selected rows once on NPU 0 and reuse one NPU-to-NPU embedding.
                auto emb = npu::empty(ple_embeddings[0].shape);
                const auto& origin = ple_embeddings[0];
                const size_t bytes = size_t(origin.numel()) * sizeof(float);
                copy_peer(emb.data(), devices[active_slot]->ordinal(), origin.data(), devices[0]->ordinal(), bytes);
                peer_bytes += bytes;
                ple_embeddings[active_slot] = std::move(emb);
            } else {
                const auto row_ids = ple_rows_device(current_stream(), ctx, multipliers.data(), offsets.data(), vocabs.data());
                require(row_ids.size() == 16, "PLE device row count");
                std::vector<ND> embeddings;
                const auto& t = *table.find("per_layer_token_embd.weight");
                for (uint64_t row : row_ids) embeddings.push_back(load(table, t, row * width, width));
                ple_embeddings[active_slot] = npu::cat(current_stream(), embeddings, 0);
            }
        }
        const auto& emb = ple_embeddings[active_slot];
        const auto key = norm(shape(linear(p + "ple_key.weight", emb), {hc, n}), vector(p + "ple_norm_key.weight"), n);
        const auto value = linear(p + "ple_value.weight", emb);
        const auto query = norm(residual, vector(p + "ple_norm_query.weight"), n);
        const auto score = mul(npu::sum(current_stream(), mul(key, query), -1, true), inverse_root(n));
        const auto transformed = mul(unary(score, Unary::Sign),
            unary(npu::clamp_min(current_stream(), unary(score, Unary::Abs), 1e-6f), Unary::Sqrt));
        const auto gated = mul(shape(value, {1, n}), unary(transformed, Unary::Sigmoid));
        const auto normalized = norm(gated, vector(p + "ple_norm_conv.weight"), n);
        const auto kernel = shape(vector(p + "ple_conv1d.weight"), {hc * n, conv_k});
        auto convolved = mul(flat(normalized), flat(npu::slice(kernel, 1, conv_k - 1, 1)));
        auto& history = states[il].ple;
        const int64_t dilation = integer("ple.ngram_size");
        require(dilation > 0, "invalid PLE dilation");
        for (int tap = 0; tap < conv_k - 1; ++tap) {
            const size_t back = size_t((conv_k - 1 - tap) * dilation);
            if (history.size() >= back) {
                convolved = add(convolved, mul(flat(history[history.size() - back]), flat(npu::slice(kernel, 1, tap, 1))));
            }
        }
        history.push_back(normalized);
        if (history.size() > size_t(dilation * (conv_k - 1))) history.pop_front();
        return add(add(residual, gated), shape(unary(convolved, Unary::Silu), {hc, n}));
    }
    ND gdn(const ND& x, int il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const auto raw = linear(p + "attn_qkv.weight", x), z = linear(p + "attn_gate.weight", x);
        const int64_t kd = kh * sd, vd = vh * sd;
        require(raw.numel() == 2 * kd + vd && z.numel() == vd, "GDN projection shape");
        const auto kernel = shape(vector(p + "ssm_conv1d.weight"), {raw.numel(), conv_k});
        auto& st = states[il];
        auto convolved = mul(raw, flat(npu::slice(kernel, 1, conv_k - 1, 1)));
        for (int back = 1; back < conv_k; ++back) {
            if (st.conv.size() >= size_t(back))
                convolved = add(convolved, mul(st.conv[st.conv.size() - back], flat(npu::slice(kernel, 1, conv_k - 1 - back, 1))));
        }
        st.conv.push_back(raw);
        if (st.conv.size() >= size_t(conv_k)) st.conv.pop_front();
        convolved = unary(convolved, Unary::Silu);
        const auto q = npu::index_select(current_stream(), l2_norm(shape(npu::slice(convolved, 0, 0, kd), {kh, sd})), 0, gdn_mapping[active_slot]);
        const auto k = npu::index_select(current_stream(), l2_norm(shape(npu::slice(convolved, 0, kd, kd), {kh, sd})), 0, gdn_mapping[active_slot]);
        const auto v = shape(npu::slice(convolved, 0, 2 * kd, vd), {vh, sd});
        const auto beta = unary(linear(p + "ssm_beta.weight", x), Unary::Sigmoid);
        const auto alpha = add(linear(p + "ssm_alpha.weight", x), vector(p + "ssm_dt.bias"));
        const auto decay = unary(mul(unary(alpha, Unary::Softplus), vector(p + "ssm_a")), Unary::Exp);
        if (!st.recurrent.storage) st.recurrent = npu::zeros({vh, sd, sd});
        st.recurrent = mul(st.recurrent, shape(decay, {vh, 1, 1}));
        const auto prediction = shape(npu::matmul(current_stream(), shape(k, {vh, 1, sd}), st.recurrent), {vh, sd});
        const auto delta = mul(sub(v, prediction), shape(beta, {vh, 1}));
        st.recurrent = add(st.recurrent, mul(shape(k, {vh, sd, 1}), shape(delta, {vh, 1, sd})));
        const auto projected = shape(npu::matmul(current_stream(), shape(q, {vh, 1, sd}), st.recurrent), {vh, sd});
        const auto result = mul(norm(mul(projected, inverse_root(sd)), vector(p + "ssm_norm.weight"), sd),
                                unary(shape(z, {vh, sd}), Unary::Sigmoid));
        return linear(p + "ssm_out.weight", result);
    }
    ND attention(const ND& x, int il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const auto qg = shape(linear(p + "attn_q.weight", x), {heads, 2, hd});
        const auto q = rope(norm(shape(npu::slice(qg, 1, 0, 1), {heads, hd}), vector(p + "attn_q_norm.weight"), hd), hd, position);
        const auto k = rope(norm(shape(linear(p + "attn_k.weight", x), {kvh, hd}), vector(p + "attn_k_norm.weight"), hd), hd, position);
        const auto v = shape(linear(p + "attn_v.weight", x), {kvh, hd});
        const auto idx = linear(p + "indexer.k_proj.weight", x);
        auto& st = states[il];
        if (!st.keys.storage) {
            st.keys = npu::empty({cap, kvh, hd}, Type::F16);
            st.values = npu::empty({cap, kvh, hd}, Type::F16);
            st.index = npu::empty({cap, idx_dim}, Type::F16);
        }
        auto dst = npu::slice(st.keys, 0, position, 1);
        npu::copy(current_stream(), npu::cast(current_stream(), shape(k, {1, kvh, hd}), Type::F16), dst);
        dst = npu::slice(st.values, 0, position, 1);
        npu::copy(current_stream(), npu::cast(current_stream(), shape(v, {1, kvh, hd}), Type::F16), dst);
        dst = npu::slice(st.index, 0, position, 1);
        npu::copy(current_stream(), npu::cast(current_stream(), shape(idx, {1, idx_dim}), Type::F16), dst);
        const int64_t count = position + 1;
        ND selected;
        if (count > idx_topk + 3) {
            const int64_t ratio = int64_t(array("attention.compress_ratios").at(il));
            require(ratio == 4, "QSA compression ratio must be four");
            const int64_t full = count / ratio;
            const auto indices = npu::cast(current_stream(), npu::slice(st.index, 0, 0, full * ratio), Type::F32);
            auto pooled = npu::mean(current_stream(), shape(indices, {full, ratio, idx_dim}), 1, false);
            if (count % ratio) pooled = npu::cat(current_stream(), {pooled, npu::cast(current_stream(), npu::slice(st.index, 0, 0, 1), Type::F32)}, 0);
            const int64_t blocks = pooled.shape[0];
            pooled = norm(pooled, vector(p + "indexer.k_norm.weight"), idx_dim);
            std::vector<ND> rotated;
            for (int64_t b = 0; b < blocks; ++b)
                rotated.push_back(rope(npu::slice(pooled, 0, b, 1), idx_dim, b == full ? 0 : b * ratio));
            pooled = npu::cat(current_stream(), rotated, 0);
            const auto iq = rope(norm(shape(linear(p + "indexer.q_proj.weight", x), {idx_heads, idx_dim}),
                                     vector(p + "indexer.q_norm.weight"), idx_dim), idx_dim, position);
            auto scores = npu::sum(current_stream(), unary(npu::matmul(current_stream(), iq, npu::permute(pooled, {1, 0})), Unary::Relu), 0, false);
            if (count % ratio) {
                auto bias = npu::zeros({blocks}); auto last = npu::slice(bias, 0, blocks - 1, 1);
                npu::copy(current_stream(), scalar(1e9f), last); scores = add(scores, bias);
            }
            const auto ones = npu::add_scalar(current_stream(), npu::zeros({1, ratio}), 1.0f);
            const auto expanded = npu::slice(flat(mul(shape(scores, {blocks, 1}), ones)), 0, 0, count);
            selected = npu::topk(current_stream(), expanded, std::min(count, idx_topk + ratio - 1)).second;
        }
        auto keys = npu::slice(st.keys, 0, 0, count), values = npu::slice(st.values, 0, 0, count);
        if (selected.storage) {
            keys = npu::index_select(current_stream(), keys, 0, selected);
            values = npu::index_select(current_stream(), values, 0, selected);
        }
        keys = npu::cast(current_stream(), npu::permute(npu::index_select(current_stream(), keys, 1, attention_mapping[active_slot]), {1, 0, 2}), Type::F32);
        values = npu::cast(current_stream(), npu::permute(npu::index_select(current_stream(), values, 1, attention_mapping[active_slot]), {1, 0, 2}), Type::F32);
        const auto scores = mul(npu::matmul(current_stream(), shape(q, {heads, 1, hd}), npu::permute(keys, {0, 2, 1})), inverse_root(hd));
        const auto mixed = shape(npu::matmul(current_stream(), npu::softmax(current_stream(), scores), values), {heads, hd});
        const auto gate = unary(shape(npu::slice(qg, 1, 1, 1), {heads, hd}), Unary::Sigmoid);
        return linear(p + "attn_output.weight", mul(mixed, gate));
    }
    ND moe(const ND& x, int il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const auto router = linear(p + "ffn_gate_inp.weight", x);
        require(router.numel() == experts, "MoE router shape");
        auto [scores, indices] = npu::topk(current_stream(), npu::softmax(current_stream(), router), top_k);
        scores = div(scores, npu::sum(current_stream(), scores, 0, true));
        auto expert = [&](const std::string& suffix, int id) {
            const auto gate = linear(p + "ffn_gate_" + suffix + ".weight", x, id);
            const auto up = linear(p + "ffn_up_" + suffix + ".weight", x, id);
            return linear(p + "ffn_down_" + suffix + ".weight", mul(unary(gate, Unary::Silu), up), id);
        };
        ND out;
        if (grouped_experts) {
            // Routing indices stay in HBM. Each projection consumes all selected
            // experts in one launch; the final sum keeps the original rank order.
            const auto gate = grouped_linear(p + "ffn_gate_exps.weight", x, indices, false);
            const auto up = grouped_linear(p + "ffn_up_exps.weight", x, indices, false);
            const auto down = grouped_linear(p + "ffn_down_exps.weight", mul(unary(gate, Unary::Silu), up), indices, true);
            out = npu::empty({n});
            weighted_sum_experts_device(current_stream(), uint32_t(top_k), uint64_t(n),
                static_cast<const float*>(down.data()), static_cast<const float*>(scores.data()), static_cast<float*>(out.data()));
            ++aggregate_launches;
        } else {
            // The ungrouped path reads only IDs, never model floating-point data.
            const auto ids = npu::read_i64(current_stream(), indices);
            out = npu::zeros({n});
            for (int64_t i = 0; i < top_k; ++i) {
                require(ids.at(i) >= 0 && ids.at(i) < experts, "NPU router returned invalid expert");
                out = add(out, mul(expert("exps", int(ids[i])), npu::slice(scores, 0, i, 1)));
            }
        }
        const auto shared = expert("shexp", -1);
        return add(out, mul(shared, unary(linear(p + "ffn_gate_inp_shexp.weight", x), Unary::Sigmoid)));
    }
    void trace(const ND& value, const std::string& suffix) {
        if (const char* dir = std::getenv("STRATA_CANN_TRACE")) {
            // Optional diagnostics copy data out; no inference arithmetic runs on host.
            std::filesystem::create_directories(dir);
            const auto values = npu::read_f32(current_stream(), value);
            const auto name = std::filesystem::path(dir) / suffix;
            std::ofstream out(name, std::ios::binary);
            out.write(reinterpret_cast<const char*>(values.data()), std::streamsize(values.size() * sizeof(float)));
            require(bool(out), "cannot write device trace");
        }
    }
    void step(int64_t token, int64_t pos) {
        require(pos == position && pos < cap && token >= 0 && token < vocab_count, "invalid token/position");
        // The scopes drain before return, including exception unwinding.
        // Model initialization and dense cache construction stay synchronous.
        auto submission = defer_submission();
        activate(0);
        for (auto& embedding : ple_embeddings) embedding = {};
        const auto embedding = shape(load(gguf, tensor("token_embd.weight"), uint64_t(token) * n, n), {1, n});
        std::vector<ND> streams(size_t(hc), embedding);
        auto residual = npu::cat(current_stream(), streams, 0);
        const auto& ple_layers = array("ple.layers");
        for (int il = 0; il < layers; ++il) {
            const auto slot = layer_slot(il);
            if (slot != active_slot) {
                const auto source = npu::contiguous(current_stream(), residual);
                current_stream().synchronize();
                activate(slot);
                auto target = npu::empty(source.shape, source.type);
                const size_t bytes = size_t(source.numel()) * npu::type_bytes(source.type);
                copy_peer(target.data(), devices[slot]->ordinal(), source.data(), source.storage->device_ordinal(), bytes);
                peer_bytes += bytes;
                residual = std::move(target);
            }
            if (std::find(ple_layers.begin(), ple_layers.end(), il) != ple_layers.end()) residual = ple(residual, token, il);
            const std::string p = "blk." + std::to_string(il) + ".";
            auto [x, inject] = hc_mix(residual, p + "hc_attn_");
            residual = combine(residual, (il + 1) % interval == 0 ? attention(x, il) : gdn(x, il), inject);
            auto [f, scatter] = hc_mix(residual, p + "hc_ffn_");
            residual = combine(residual, moe(f, il), scatter);
            trace(residual, "p" + std::to_string(pos) + "-l" + std::to_string(il) + ".f32");
        }
        hidden = hc_mix(residual, "output_hc_", false).first;
        tokens.push_back(token); ++position;
        for (auto& scope : submission) scope->finish();
    }
    Model::Top top(int64_t k) {
        require(hidden.storage != nullptr && k > 0 && k <= vocab_count, "top before step or invalid k");
        auto submission = defer_submission();
        activate(devices.size() - 1);
        const auto logits = linear("output.weight", hidden);
        trace(logits, "p" + std::to_string(position - 1) + "-logits.f32");
        const auto [values, indices] = npu::topk(current_stream(), logits, k);
        const auto log_z = npu::logsumexp(current_stream(), logits);
        const auto best = npu::read_f32(current_stream(), values);
        const auto ids = npu::read_i64(current_stream(), indices);
        Model::Top result;
        result.log_z = npu::read_f32(current_stream(), log_z).at(0);
        for (int64_t i = 0; i < k; ++i) result.best.emplace_back(best.at(i), int(ids.at(i)));
        fprintf(stderr, "native NPU_ONLY stats: NPU matmuls=%llu cache_hits=%llu cache_misses=%llu weight_GiB=%.3f "
                        "host_float_forward=disabled\n", (unsigned long long)matmuls,
                (unsigned long long)hits, (unsigned long long)misses, used / 1073741824.0);
        fprintf(stderr, "PACKED_STATS devices=%zu fused_calls=%llu peer_bytes=%llu packed=%d dense_hits=%llu dense_bytes=%llu\n",
                devices.size(), (unsigned long long)fused_calls, (unsigned long long)peer_bytes, int(packed_mode),
                (unsigned long long)dense_hits, (unsigned long long)dense_bytes);
        if (grouped_experts) fprintf(stderr, "GROUPED_STATS projections=%llu weighted_sums=%llu router_ids=NPU\n",
                (unsigned long long)grouped_launches, (unsigned long long)aggregate_launches);
        for (auto& scope : submission) scope->finish();
        return result;
    }
};

Qwen38NpuModel::Qwen38NpuModel(const std::string& path, int64_t capacity) : impl_(std::make_unique<Impl>(path, capacity)) {}
Qwen38NpuModel::~Qwen38NpuModel() = default;
void Qwen38NpuModel::step(int64_t token, int64_t position) { impl_->step(token, position); }
Model::Top Qwen38NpuModel::top(int64_t k) { return impl_->top(k); }
void Qwen38NpuModel::reset() {
    impl_->tokens.clear(); impl_->states.clear(); impl_->states.resize(impl_->layers);
    impl_->hidden = {}; impl_->position = 0;
}
int64_t Qwen38NpuModel::capacity() const { return impl_->cap; }
int64_t Qwen38NpuModel::vocab() const { return impl_->vocab_count; }
int64_t Qwen38NpuModel::cache_bytes() const { uint64_t bytes = impl_->used + impl_->dense_bytes; for (auto n : impl_->packed_sizes) bytes += n; return int64_t(bytes); }
}
