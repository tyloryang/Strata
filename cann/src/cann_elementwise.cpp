// cann/src/cann_elementwise.cpp - RMSNorm / SiLU / NEOX RoPE on Ascend via aclnn.
// Same two-stage aclnn pattern and no-aliasing rule as cann_attention.cpp.
#include "cann_elementwise.h"

#include <cmath>
#include <vector>

#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_mul.h"
#include "aclnnop/aclnn_reduce_sum.h"
#include "aclnnop/aclnn_rsqrt.h"
#include "aclnnop/aclnn_silu.h"
#include "aclnnop/aclnn_sub.h"

namespace strata::cann {
namespace {

bool fail(std::string* err, const std::string& msg) {
  if (err != nullptr) *err = msg;
  return false;
}

struct Workspace {
  aclrtStream stream = nullptr;
  void* ptr = nullptr;
  std::uint64_t size = 0;
  ~Workspace() {
    if (ptr != nullptr) aclrtFree(ptr);
  }
  Workspace() = default;
  Workspace(const Workspace&) = delete;
  Workspace& operator=(const Workspace&) = delete;
  bool ensure(std::uint64_t need, std::string* err) {
    if (need <= size) return true;
    if (ptr != nullptr) {
      aclrtFree(ptr);
      ptr = nullptr;
      size = 0;
    }
    if (aclrtMalloc(&ptr, need, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
      return fail(err, "cann: aclrtMalloc workspace failed");
    }
    size = need;
    return true;
  }
};

/// Runs one aclnn op: getws fills (need, executor), run invokes the op.
///
/// The executor is deliberately NOT destroyed here. On CANN 9.0.0 / 910B2C
/// (verified 2026-11-05 with minimal repros) the executor is a
/// runtime-shared object: GetWorkspaceSize for a different op returns the
/// same pointer, and destroying an executor whose op has executed makes the
/// NEXT GetWorkspaceSize for a different op type segfault (Add->Abs,
/// Add->Neg, Add->Exp, Abs->Add all reproduced; a stream sync before the
/// destroy does not help, and destroying only the aclScalar is harmless).
/// Same-op-type reuse after destroy happens to work, which is why the
/// GEMM-only test (Matmul throughout) never saw this. Executors are
/// therefore leaked for the life of the process; a future CANN version or
/// a documented destroy discipline may revisit this.
template <typename GetWs, typename Run>
bool run_op(Workspace& ws, const char* name, GetWs&& getws, Run&& run, std::string* err) {
  std::uint64_t need = 0;
  aclOpExecutor* ex = nullptr;
  aclnnStatus st = getws(&need, &ex);
  if (st != 0 || ex == nullptr) {
    return fail(err, std::string("cann: ") + name + " GetWorkspaceSize failed: " + std::to_string(st));
  }
  if (need > 0 && !ws.ensure(need, err)) {
    return false;
  }
  st = run(need > 0 ? ws.ptr : nullptr, need, ex);
  if (st != 0) {
    return fail(err, std::string("cann: ") + name + " failed: " + std::to_string(st));
  }
  return true;
}

struct Tensor {
  aclTensor* t = nullptr;
  Tensor() = default;
  Tensor(void* data, std::vector<std::int64_t> dims, std::vector<std::int64_t> strides,
         std::int64_t offset, std::vector<std::int64_t> storage) {
    t = aclCreateTensor(dims.data(), dims.size(), ACL_FLOAT, strides.data(), offset, ACL_FORMAT_ND,
                        storage.data(), storage.size(), data);
  }
  static Tensor contig(void* data, std::vector<std::int64_t> dims) {
    std::vector<std::int64_t> strides(dims.size(), 1);
    for (std::int64_t i = static_cast<std::int64_t>(dims.size()) - 2; i >= 0; --i)
      strides[i] = strides[i + 1] * dims[i + 1];
    return Tensor(data, dims, strides, 0, dims);
  }
  ~Tensor() {
    if (t != nullptr) aclDestroyTensor(t);
  }
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;
  Tensor(Tensor&& o) noexcept : t(o.t) { o.t = nullptr; }
  Tensor& operator=(Tensor&& o) noexcept {
    if (this != &o) {
      if (t != nullptr) aclDestroyTensor(t);
      t = o.t;
      o.t = nullptr;
    }
    return *this;
  }
  operator const aclTensor*() const { return t; }
  operator aclTensor*() { return t; }
};

struct Scalar {
  aclScalar* s = nullptr;
  float v = 0.0f;
  explicit Scalar(float x) : v(x) { s = aclCreateScalar(&v, ACL_FLOAT); }
  ~Scalar() {
    if (s != nullptr) aclDestroyScalar(s);
  }
  Scalar(const Scalar&) = delete;
  Scalar& operator=(const Scalar&) = delete;
  operator const aclScalar*() const { return s; }
};

struct IntArray {
  aclIntArray* a = nullptr;
  explicit IntArray(std::vector<std::int64_t> v) { a = aclCreateIntArray(v.data(), v.size()); }
  ~IntArray() {
    if (a != nullptr) aclDestroyIntArray(a);
  }
  IntArray(const IntArray&) = delete;
  IntArray& operator=(const IntArray&) = delete;
  operator const aclIntArray*() const { return a; }
};

bool op_mul(Workspace& ws, const aclTensor* a, const aclTensor* b, aclTensor* out,
            std::string* err) {
  return run_op(
      ws, "aclnnMul",
      [&](std::uint64_t* n, aclOpExecutor** e) {
        return aclnnMulGetWorkspaceSize(a, b, out, n, e);
      },
      [&](void* w, std::uint64_t s, aclOpExecutor* e) { return aclnnMul(w, s, e, ws.stream); }, err);
}
bool op_muls(Workspace& ws, const aclTensor* a, float other, aclTensor* out, std::string* err) {
  Scalar o(other);
  return run_op(
      ws, "aclnnMuls",
      [&](std::uint64_t* n, aclOpExecutor** e) {
        return aclnnMulsGetWorkspaceSize(a, o, out, n, e);
      },
      [&](void* w, std::uint64_t s, aclOpExecutor* e) { return aclnnMuls(w, s, e, ws.stream); },
      err);
}
bool op_adds(Workspace& ws, const aclTensor* a, float other, float alpha, aclTensor* out,
             std::string* err) {
  Scalar o(other), al(alpha);
  return run_op(
      ws, "aclnnAdds",
      [&](std::uint64_t* n, aclOpExecutor** e) {
        return aclnnAddsGetWorkspaceSize(a, o, al, out, n, e);
      },
      [&](void* w, std::uint64_t s, aclOpExecutor* e) { return aclnnAdds(w, s, e, ws.stream); },
      err);
}
bool op_add(Workspace& ws, const aclTensor* a, const aclTensor* b, aclTensor* out,
            std::string* err) {
  Scalar al(1.0f);
  return run_op(
      ws, "aclnnAdd",
      [&](std::uint64_t* n, aclOpExecutor** e) {
        return aclnnAddGetWorkspaceSize(a, b, al, out, n, e);
      },
      [&](void* w, std::uint64_t s, aclOpExecutor* e) { return aclnnAdd(w, s, e, ws.stream); }, err);
}
bool op_sub(Workspace& ws, const aclTensor* a, const aclTensor* b, aclTensor* out,
            std::string* err) {
  Scalar al(1.0f);
  return run_op(
      ws, "aclnnSub",
      [&](std::uint64_t* n, aclOpExecutor** e) {
        return aclnnSubGetWorkspaceSize(a, b, al, out, n, e);
      },
      [&](void* w, std::uint64_t s, aclOpExecutor* e) { return aclnnSub(w, s, e, ws.stream); }, err);
}
bool op_rsqrt(Workspace& ws, const aclTensor* a, aclTensor* out, std::string* err) {
  return run_op(
      ws, "aclnnRsqrt",
      [&](std::uint64_t* n, aclOpExecutor** e) {
        return aclnnRsqrtGetWorkspaceSize(a, out, n, e);
      },
      [&](void* w, std::uint64_t s, aclOpExecutor* e) { return aclnnRsqrt(w, s, e, ws.stream); },
      err);
}
bool op_silu(Workspace& ws, const aclTensor* a, aclTensor* out, std::string* err) {
  return run_op(
      ws, "aclnnSilu",
      [&](std::uint64_t* n, aclOpExecutor** e) {
        return aclnnSiluGetWorkspaceSize(a, out, n, e);
      },
      [&](void* w, std::uint64_t s, aclOpExecutor* e) { return aclnnSilu(w, s, e, ws.stream); },
      err);
}
bool op_reduce_sum(Workspace& ws, const aclTensor* a, std::int64_t dim, bool keep_dims,
                   aclTensor* out, std::string* err) {
  IntArray dims({dim});
  return run_op(
      ws, "aclnnReduceSum",
      [&](std::uint64_t* n, aclOpExecutor** e) {
        return aclnnReduceSumGetWorkspaceSize(a, dims, keep_dims, ACL_FLOAT, out, n, e);
      },
      [&](void* w, std::uint64_t s, aclOpExecutor* e) { return aclnnReduceSum(w, s, e, ws.stream); },
      err);
}

bool memcpy_d2d(void* dst, const void* src, std::size_t bytes, aclrtStream stream,
                std::string* err) {
  if (aclrtMemcpyAsync(dst, bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream) !=
      ACL_SUCCESS) {
    return fail(err, "cann: aclrtMemcpyAsync device-to-device failed");
  }
  return true;
}

}  // namespace

bool cann_rmsnorm(const float* x, const float* w, float* y, int rows, int n, float eps,
                  aclrtStream stream, float* scratch, std::string* err) {
  if (x == nullptr || w == nullptr || y == nullptr || rows <= 0 || n <= 0 || stream == nullptr ||
      scratch == nullptr) {
    return fail(err, "cann_rmsnorm: invalid arguments");
  }
  const std::int64_t R = rows, N = n;
  float* sq = scratch;              // [R][N]
  float* xs = scratch + R * N;      // [R][N]
  float* ss = xs + R * N;           // [R][1]
  float* sc = ss + R;               // [R][1]
  Workspace ws;
  ws.stream = stream;
  Tensor v_x = Tensor::contig(const_cast<float*>(x), {R, N});
  Tensor v_w = Tensor::contig(const_cast<float*>(w), {N});
  Tensor v_y = Tensor::contig(y, {R, N});
  Tensor v_sq = Tensor::contig(sq, {R, N});
  Tensor v_xs = Tensor::contig(xs, {R, N});
  Tensor v_ss = Tensor::contig(ss, {R, 1});
  Tensor v_sc = Tensor::contig(sc, {R, 1});
  if (!op_mul(ws, v_x, v_x, v_sq, err)) return false;
  if (!op_reduce_sum(ws, v_sq, 1, true, v_ss, err)) return false;
  if (!op_muls(ws, v_ss, 1.0f / static_cast<float>(n), v_sc, err)) return false;
  if (!op_adds(ws, v_sc, eps, 1.0f, v_ss, err)) return false;
  if (!op_rsqrt(ws, v_ss, v_sc, err)) return false;
  if (!op_mul(ws, v_x, v_sc, v_xs, err)) return false;
  if (!op_mul(ws, v_xs, v_w, v_y, err)) return false;
  return true;
}

bool cann_silu(const float* x, float* y, std::int64_t n, aclrtStream stream, std::string* err) {
  if (x == nullptr || y == nullptr || n <= 0 || stream == nullptr) {
    return fail(err, "cann_silu: invalid arguments");
  }
  Workspace ws;
  ws.stream = stream;
  Tensor v_x = Tensor::contig(const_cast<float*>(x), {n});
  Tensor v_y = Tensor::contig(y, {n});
  return op_silu(ws, v_x, v_y, err);
}

void cann_build_rope_table(int n_rot, double theta, int max_pos, float* cos_tab,
                           float* sin_tab) {
  const int half = n_rot / 2;
  for (int p = 0; p < max_pos; ++p) {
    for (int i = 0; i < half; ++i) {
      const double inv = std::pow(theta, -2.0 * static_cast<double>(i) / static_cast<double>(n_rot));
      const double ang = static_cast<double>(p) * inv;
      cos_tab[static_cast<size_t>(p) * half + i] = static_cast<float>(std::cos(ang));
      sin_tab[static_cast<size_t>(p) * half + i] = static_cast<float>(std::sin(ang));
    }
  }
}

bool cann_rope_neox(float* x, const float* cos_t, const float* sin_t, int rows, int heads,
                    int head_dim, int n_rot, aclrtStream stream, float* scratch,
                    std::string* err) {
  if (x == nullptr || cos_t == nullptr || sin_t == nullptr || rows <= 0 || heads <= 0 ||
      head_dim <= 0 || n_rot <= 0 || n_rot > head_dim || (n_rot % 2) != 0 || stream == nullptr ||
      scratch == nullptr) {
    return fail(err, "cann_rope_neox: invalid arguments");
  }
  const std::int64_t R = rows, H = heads, HD = head_dim, NR = n_rot, HF = n_rot / 2;
  const std::int64_t total = R * H * HD;
  float* buf = scratch;                    // [R][H][HD] working copy
  float* p1 = buf + total;                 // [R][H][HF] products, x4
  float* p2 = p1 + R * H * HF;
  float* p3 = p2 + R * H * HF;
  float* p4 = p3 + R * H * HF;
  Workspace ws;
  ws.stream = stream;
  if (aclrtMemcpyAsync(buf, static_cast<size_t>(total) * sizeof(float), x,
                       static_cast<size_t>(total) * sizeof(float), ACL_MEMCPY_DEVICE_TO_DEVICE,
                       stream) != ACL_SUCCESS) {
    return fail(err, "cann_rope_neox: copy x to scratch failed");
  }
  // First half a = buf[..., 0:HF], second half b = buf[..., HF:NR], both
  // strided views with the head's HD stride. cos/sin are [R][HF], broadcast
  // over heads as [R][1][HF].
  Tensor v_a = Tensor(buf, {R, H, HF}, {H * HD, HD, 1}, 0, {R, H, HD});
  Tensor v_b = Tensor(buf, {R, H, HF}, {H * HD, HD, 1}, HF, {R, H, HD});
  Tensor v_cos = Tensor(const_cast<float*>(cos_t), {R, 1, HF}, {HF, HF, 1}, 0, {R, HF});
  Tensor v_sin = Tensor(const_cast<float*>(sin_t), {R, 1, HF}, {HF, HF, 1}, 0, {R, HF});
  Tensor v_p1 = Tensor::contig(p1, {R, H, HF});
  Tensor v_p2 = Tensor::contig(p2, {R, H, HF});
  Tensor v_p3 = Tensor::contig(p3, {R, H, HF});
  Tensor v_p4 = Tensor::contig(p4, {R, H, HF});
  // All four products first: writing either output destroys an input half.
  if (!op_mul(ws, v_a, v_cos, v_p1, err)) return false;  // a*cos
  if (!op_mul(ws, v_b, v_sin, v_p2, err)) return false;  // b*sin
  if (!op_mul(ws, v_a, v_sin, v_p3, err)) return false;  // a*sin
  if (!op_mul(ws, v_b, v_cos, v_p4, err)) return false;  // b*cos
  if (!op_sub(ws, v_p1, v_p2, v_a, err)) return false;   // out[:HF] = a*cos - b*sin
  if (!op_add(ws, v_p3, v_p4, v_b, err)) return false;   // out[HF:NR] = a*sin + b*cos
  if (!memcpy_d2d(x, buf, static_cast<size_t>(total) * sizeof(float), stream, err)) return false;
  return true;
}

}  // namespace strata::cann
