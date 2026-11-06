#pragma once
#include "strata/cann/device.hpp"
#include <memory>
#include <vector>
#include <cstdint>

namespace strata::cann::npu {
enum class Type { F32, F16, BF16, I64, I32, U8 };
struct ND {
    std::shared_ptr<Buffer> storage;
    std::vector<int64_t> shape, strides;
    Type type = Type::F32;
    int64_t offset = 0;
    int64_t numel() const;
    void * data() const;
};
size_t type_bytes(Type type);
ND empty(const std::vector<int64_t>& shape, Type type = Type::F32);
ND zeros(const std::vector<int64_t>& shape, Type type = Type::F32);
ND from_host(const void* data, const std::vector<int64_t>& shape, Type type = Type::F32);
ND wrap(std::shared_ptr<Buffer> storage, const std::vector<int64_t>& shape, Type type = Type::F32);
void copy(Stream&, const ND& src, ND& dst);
ND contiguous(Stream&, const ND&);
ND reshape(Stream&, const ND&, const std::vector<int64_t>& shape);
ND permute(const ND&, const std::vector<int64_t>& dims);
ND slice(const ND&, int64_t dim, int64_t start, int64_t length);
ND cast(Stream&, const ND&, Type);
enum class Unary { Silu, Sigmoid, Exp, Log, Sqrt, Rsqrt, Sin, Cos, Abs, Sign, Relu, Softplus, Neg };
ND unary(Stream&, const ND&, Unary);
enum class Binary { Add, Sub, Mul, Div };
ND binary(Stream&, const ND&, const ND&, Binary);
ND scale(Stream&, const ND&, float scalar);
ND add_scalar(Stream&, const ND&, float scalar);
ND pow_scalar(Stream&, const ND&, float scalar);
ND clamp_min(Stream&, const ND&, float scalar);
ND sum(Stream&, const ND&, int64_t dim, bool keepdim = true);
ND mean(Stream&, const ND&, int64_t dim, bool keepdim = true);
ND softmax(Stream&, const ND&, int64_t dim = -1);
ND matmul(Stream&, const ND&, const ND&);
ND cat(Stream&, const std::vector<ND>&, int64_t dim);
ND index_select(Stream&, const ND&, int64_t dim, const ND& indices);
std::pair<ND, ND> topk(Stream&, const ND&, int64_t k, int64_t dim = -1);
ND logsumexp(Stream&, const ND&, int64_t dim = -1, bool keepdim = false);
std::vector<float> read_f32(Stream&, const ND&);
std::vector<int64_t> read_i64(Stream&, const ND&);
}
