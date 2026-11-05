// cann/src/cann_attention.cpp - Phase 1: Gated DeltaNet attention on Ascend via aclnn.
//
// See cann_attention.h for the math and the state-layout note. Implementation
// notes (all verified against the CANN 9.0.0 headers on the target box):
//   * Every aclnn op is the standard two-stage call: <Op>GetWorkspaceSize,
//     then <Op>(workspace, size, executor, stream). Executors are NOT
//     destroyed after each call (CANN 9.0.0 reuses one shared executor
//     object; destroying it breaks the next different op's GetWorkspaceSize
//     -- see run_op); one reusable device workspace buffer is grown on demand.
//   * Only elementwise ops (Add/Adds/Mul/Muls/Sub/Neg/Abs/Relu/Exp/Log1p/
//     Sigmoid/Silu/Rsqrt) and ReduceSum are used, all FP32, all with explicit
//     broadcast shapes. No matmul: the (S^T k) and (S_new^T q) contractions
//     are Mul with a broadcast head vector followed by ReduceSum over the
//     state-row axis, which is the same arithmetic in a different order.
//   * aclnn does not guarantee out == in aliasing, so every op writes to a
//     distinct scratch buffer and results are copied back with
//     aclrtMemcpyAsync (device-to-device) where an in-place result is needed.
#include "cann_attention.h"

#include <cmath>
#include <functional>
#include <utility>
#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_abs.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_exp.h"
#include "aclnnop/aclnn_log1p.h"
#include "aclnnop/aclnn_mul.h"
#include "aclnnop/aclnn_neg.h"
#include "aclnnop/aclnn_reduce_sum.h"
#include "aclnnop/aclnn_relu.h"
#include "aclnnop/aclnn_rsqrt.h"
#include "aclnnop/aclnn_sigmoid.h"
#include "aclnnop/aclnn_silu.h"
#include "aclnnop/aclnn_sub.h"

namespace strata::cann {
namespace {

bool fail(std::string* err, const std::string& msg) {
  if (err != nullptr) *err = msg;
  return false;
}

// ---------------------------------------------------------------- aclnn plumbing

/// One reusable device workspace for the two-stage aclnn calls.
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
      return fail(err, "cann: aclrtMalloc workspace (" + std::to_string(need) + " bytes) failed");
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

/// RAII aclTensor. `strides` are in elements; `offset` in elements.
struct Tensor {
  aclTensor* t = nullptr;
  Tensor() = default;
  Tensor(void* data, std::vector<std::int64_t> dims, std::vector<std::int64_t> strides,
         std::int64_t offset, std::vector<std::int64_t> storage) {
    t = aclCreateTensor(dims.data(), dims.size(), ACL_FLOAT, strides.data(), offset, ACL_FORMAT_ND,
                        storage.data(), storage.size(), data);
  }
  /// Contiguous row-major tensor.
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

bool memcpy_d2d(void* dst, const void* src, std::size_t bytes, aclrtStream stream,
                std::string* err) {
  if (aclrtMemcpyAsync(dst, bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream) !=
      ACL_SUCCESS) {
    return fail(err, "cann: aclrtMemcpyAsync device-to-device failed");
  }
  return true;
}

// Unary / binary op shims: each takes pre-built tensors and runs the op.
bool op_add(Workspace& ws, const aclTensor* a, const aclTensor* b, float alpha, aclTensor* out,
            std::string* err) {
  Scalar al(alpha);
  return run_op(
      ws, "aclnnAdd",
      [&](std::uint64_t* n, aclOpExecutor** e) {
        return aclnnAddGetWorkspaceSize(a, b, al, out, n, e);
      },
      [&](void* w, std::uint64_t s, aclOpExecutor* e) { return aclnnAdd(w, s, e, ws.stream); }, err);
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
bool op_exp(Workspace& ws, const aclTensor* a, aclTensor* out, std::string* err) {
  return run_op(
      ws, "aclnnExp",
      [&](std::uint64_t* n, aclOpExecutor** e) {
        return aclnnExpGetWorkspaceSize(a, out, n, e);
      },
      [&](void* w, std::uint64_t s, aclOpExecutor* e) { return aclnnExp(w, s, e, ws.stream); }, err);
}
bool op_log1p(Workspace& ws, const aclTensor* a, aclTensor* out, std::string* err) {
  return run_op(
      ws, "aclnnLog1p",
      [&](std::uint64_t* n, aclOpExecutor** e) {
        return aclnnLog1pGetWorkspaceSize(a, out, n, e);
      },
      [&](void* w, std::uint64_t s, aclOpExecutor* e) { return aclnnLog1p(w, s, e, ws.stream); },
      err);
}
bool op_sigmoid(Workspace& ws, const aclTensor* a, aclTensor* out, std::string* err) {
  return run_op(
      ws, "aclnnSigmoid",
      [&](std::uint64_t* n, aclOpExecutor** e) {
        return aclnnSigmoidGetWorkspaceSize(a, out, n, e);
      },
      [&](void* w, std::uint64_t s, aclOpExecutor* e) { return aclnnSigmoid(w, s, e, ws.stream); },
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
bool op_rsqrt(Workspace& ws, const aclTensor* a, aclTensor* out, std::string* err) {
  return run_op(
      ws, "aclnnRsqrt",
      [&](std::uint64_t* n, aclOpExecutor** e) {
        return aclnnRsqrtGetWorkspaceSize(a, out, n, e);
      },
      [&](void* w, std::uint64_t s, aclOpExecutor* e) { return aclnnRsqrt(w, s, e, ws.stream); },
      err);
}
bool op_abs(Workspace& ws, const aclTensor* a, aclTensor* out, std::string* err) {
  return run_op(
      ws, "aclnnAbs",
      [&](std::uint64_t* n, aclOpExecutor** e) {
        return aclnnAbsGetWorkspaceSize(a, out, n, e);
      },
      [&](void* w, std::uint64_t s, aclOpExecutor* e) { return aclnnAbs(w, s, e, ws.stream); }, err);
}
bool op_neg(Workspace& ws, const aclTensor* a, aclTensor* out, std::string* err) {
  return run_op(
      ws, "aclnnNeg",
      [&](std::uint64_t* n, aclOpExecutor** e) {
        return aclnnNegGetWorkspaceSize(a, out, n, e);
      },
      [&](void* w, std::uint64_t s, aclOpExecutor* e) { return aclnnNeg(w, s, e, ws.stream); }, err);
}
bool op_relu(Workspace& ws, const aclTensor* a, aclTensor* out, std::string* err) {
  return run_op(
      ws, "aclnnRelu",
      [&](std::uint64_t* n, aclOpExecutor** e) {
        return aclnnReluGetWorkspaceSize(a, out, n, e);
      },
      [&](void* w, std::uint64_t s, aclOpExecutor* e) { return aclnnRelu(w, s, e, ws.stream); },
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

}  // namespace

GdnGeometry qwen36_geometry() {
  GdnGeometry g;
  g.n_embd = 2048;
  g.state_size = 128;
  g.k_heads = 16;
  g.v_heads = 32;
  g.d_conv = 4;
  return g;
}

bool cann_init(int device, aclrtStream* stream, std::string* err) {
  if (stream == nullptr) return fail(err, "cann_init: null stream out-param");
  if (aclInit(nullptr) != ACL_SUCCESS) return fail(err, "cann_init: aclInit failed");
  if (aclrtSetDevice(device) != ACL_SUCCESS) {
    aclFinalize();
    return fail(err, "cann_init: aclrtSetDevice failed");
  }
  if (aclrtCreateStream(stream) != ACL_SUCCESS) {
    aclFinalize();
    return fail(err, "cann_init: aclrtCreateStream failed");
  }
  return true;
}

void cann_shutdown(aclrtStream stream) {
  if (stream != nullptr) aclrtDestroyStream(stream);
  aclFinalize();
}

size_t gdn_scratch_floats(const GdnGeometry& g, int max_tokens) {
  if (!g.valid() || max_tokens <= 0) return 0;
  const size_t S = static_cast<size_t>(g.state_size);
  const size_t HV = static_cast<size_t>(g.v_heads);
  const size_t C = static_cast<size_t>(g.conv_channels());
  const size_t step = 2 * HV * S * S + 6 * HV * S + 4 * HV + 64;
  const size_t conv = 6 * C + 2 * static_cast<size_t>(g.k_heads) * S + 64;
  const size_t gate = 4 * static_cast<size_t>(max_tokens) * HV + 64;
  size_t m = step;
  if (conv > m) m = conv;
  if (gate > m) m = gate;
  return m;
}

bool cann_gdn_gate_beta(const GdnGeometry& g, float* alpha, float* beta, const float* dt,
                        const float* ssm_a, int tokens, aclrtStream stream, float* scratch,
                        std::string* err) {
  if (!g.valid() || alpha == nullptr || beta == nullptr || dt == nullptr || ssm_a == nullptr ||
      tokens <= 0 || stream == nullptr || scratch == nullptr) {
    return fail(err, "cann_gdn_gate_beta: invalid arguments");
  }
  const std::int64_t HV = g.v_heads;
  const std::int64_t n = static_cast<std::int64_t>(tokens) * HV;
  float* t0 = scratch;
  float* t1 = scratch + n;
  float* t2 = scratch + 2 * n;
  float* t3 = scratch + 3 * n;
  Workspace ws;
  ws.stream = stream;

  Tensor a_alpha = Tensor::contig(alpha, {tokens, HV});
  Tensor a_beta = Tensor::contig(beta, {tokens, HV});
  Tensor a_dt = Tensor::contig(const_cast<float*>(dt), {HV});
  Tensor a_ssm_a = Tensor::contig(const_cast<float*>(ssm_a), {HV});
  Tensor v_t0 = Tensor::contig(t0, {tokens, HV});
  Tensor v_t1 = Tensor::contig(t1, {tokens, HV});
  Tensor v_t2 = Tensor::contig(t2, {tokens, HV});
  Tensor v_t3 = Tensor::contig(t3, {tokens, HV});

  // alpha_dt = alpha + dt (dt broadcasts over the token rows)
  if (!op_add(ws, a_alpha, a_dt, 1.0f, v_t0, err)) return false;
  // softplus(x) = max(x, 0) + log1p(exp(-|x|))
  if (!op_abs(ws, v_t0, v_t1, err)) return false;
  if (!op_neg(ws, v_t1, v_t2, err)) return false;
  if (!op_exp(ws, v_t2, v_t3, err)) return false;
  if (!op_log1p(ws, v_t3, v_t2, err)) return false;
  if (!op_relu(ws, v_t0, v_t3, err)) return false;
  if (!op_add(ws, v_t3, v_t2, 1.0f, v_t0, err)) return false;
  if (!op_mul(ws, v_t0, a_ssm_a, v_t1, err)) return false;
  if (!memcpy_d2d(alpha, t1, static_cast<size_t>(n) * sizeof(float), stream, err)) return false;
  // beta = sigmoid(beta)
  if (!op_sigmoid(ws, a_beta, v_t0, err)) return false;
  if (!memcpy_d2d(beta, t0, static_cast<size_t>(n) * sizeof(float), stream, err)) return false;
  return true;
}

bool cann_gdn_conv(const GdnGeometry& g, GdnConvState& st, const float* qkv, const float* conv_w,
                   float* h_out, int tokens, float eps, aclrtStream stream, float* scratch,
                   std::string* err) {
  if (!g.valid() || st.h0 == nullptr || st.h1 == nullptr || st.h2 == nullptr || qkv == nullptr ||
      conv_w == nullptr || h_out == nullptr || tokens <= 0 || stream == nullptr ||
      scratch == nullptr) {
    return fail(err, "cann_gdn_conv: invalid arguments");
  }
  const std::int64_t C = g.conv_channels();
  const std::int64_t S = g.state_size;
  const std::int64_t HK = g.k_heads;
  const std::int64_t qk = 2 * HK * S;  // L2-normalised channels: q heads then k heads
  // Scratch carve-up: b0..b3, b4, bA are [C]; bSq is [2*HK*S]; sums/scale are [2*HK].
  float* b0 = scratch;
  float* b1 = scratch + C;
  float* b2 = scratch + 2 * C;
  float* b3 = scratch + 3 * C;
  float* b4 = scratch + 4 * C;
  float* bA = scratch + 5 * C;
  float* bSq = scratch + 6 * C;
  float* sums = bSq + qk;
  float* scale = sums + 2 * HK;
  Workspace ws;
  ws.stream = stream;

  Tensor v_b0 = Tensor::contig(b0, {C});
  Tensor v_b1 = Tensor::contig(b1, {C});
  Tensor v_b2 = Tensor::contig(b2, {C});
  Tensor v_b3 = Tensor::contig(b3, {C});
  Tensor v_b4 = Tensor::contig(b4, {C});
  Tensor v_bA = Tensor::contig(bA, {C});
  // q/k slice of bA and its square buffer, as [2*HK, S].
  Tensor v_qk = Tensor::contig(bA, {2 * HK, S});
  Tensor v_bSq = Tensor::contig(bSq, {2 * HK, S});
  Tensor v_sums = Tensor::contig(sums, {2 * HK});
  Tensor v_scale = Tensor::contig(scale, {2 * HK});
  // The rsqrt output lands back in `sums` (free by then); v_scale2 reads it.
  Tensor v_scale2 = Tensor(sums, {2 * HK, 1}, {1, 1}, 0, {2 * HK});
  // conv weight taps: conv_w is [C][4]; tap i is a strided [C] view.
  Tensor w0 = Tensor(const_cast<float*>(conv_w), {C}, {4}, 0, {C * 4});
  Tensor w1 = Tensor(const_cast<float*>(conv_w), {C}, {4}, 1, {C * 4});
  Tensor w2 = Tensor(const_cast<float*>(conv_w), {C}, {4}, 2, {C * 4});
  Tensor w3 = Tensor(const_cast<float*>(conv_w), {C}, {4}, 3, {C * 4});

  for (int t = 0; t < tokens; ++t) {
    const float* x_t = qkv + static_cast<size_t>(t) * static_cast<size_t>(C);
    float* out_t = h_out + static_cast<size_t>(t) * static_cast<size_t>(C);
    Tensor v_h0 = Tensor::contig(st.h0, {C});
    Tensor v_h1 = Tensor::contig(st.h1, {C});
    Tensor v_h2 = Tensor::contig(st.h2, {C});
    Tensor v_x = Tensor::contig(const_cast<float*>(x_t), {C});
    // sum = h0*w0 + h1*w1 + h2*w2 + x*w3 (oldest tap first, as k_conv_seq)
    if (!op_mul(ws, v_h0, w0, v_b0, err)) return false;
    if (!op_mul(ws, v_h1, w1, v_b1, err)) return false;
    if (!op_mul(ws, v_h2, w2, v_b2, err)) return false;
    if (!op_mul(ws, v_x, w3, v_b3, err)) return false;
    if (!op_add(ws, v_b0, v_b1, 1.0f, v_b4, err)) return false;
    if (!op_add(ws, v_b2, v_b3, 1.0f, v_b0, err)) return false;
    if (!op_add(ws, v_b4, v_b0, 1.0f, v_b1, err)) return false;
    if (!op_silu(ws, v_b1, v_bA, err)) return false;
    // L2 norm of the q/k heads: y *= rsqrt(sum(y^2) + eps) per 128-channel head.
    if (!op_mul(ws, v_qk, v_qk, v_bSq, err)) return false;
    if (!op_reduce_sum(ws, v_bSq, 1, false, v_sums, err)) return false;
    if (!op_adds(ws, v_sums, eps, 1.0f, v_scale, err)) return false;
    if (!op_rsqrt(ws, v_scale, v_sums, err)) return false;
    if (!op_mul(ws, v_qk, v_scale2, v_bSq, err)) return false;
    if (!memcpy_d2d(bA, bSq, static_cast<size_t>(qk) * sizeof(float), stream, err)) return false;
    if (!memcpy_d2d(out_t, bA, static_cast<size_t>(C) * sizeof(float), stream, err)) return false;
    // Rotate the history: (h0, h1, h2) <- (h1, h2, x_t).
    float* tmp = st.h0;
    st.h0 = st.h1;
    st.h1 = st.h2;
    st.h2 = tmp;
    if (!memcpy_d2d(st.h2, x_t, static_cast<size_t>(C) * sizeof(float), stream, err)) return false;
  }
  return true;
}

bool cann_gdn_step(const GdnGeometry& g, float* state, const float* h, const float* gate,
                   const float* beta, const float* z, const float* gamma, float eps, float* y,
                   int tokens, aclrtStream stream, float* scratch, std::string* err) {
  if (!g.valid() || state == nullptr || h == nullptr || gate == nullptr || beta == nullptr ||
      z == nullptr || gamma == nullptr || y == nullptr || tokens <= 0 || stream == nullptr ||
      scratch == nullptr) {
    return fail(err, "cann_gdn_step: invalid arguments");
  }
  const std::int64_t S = g.state_size;
  const std::int64_t HK = g.k_heads;
  const std::int64_t HV = g.v_heads;
  const std::int64_t R = HV / HK;  // value heads per key head (2 for Qwen3.6)
  const std::int64_t C = g.conv_channels();
  const std::int64_t ZV = HV * S;
  const float inv_sqrt_S = 1.0f / std::sqrt(static_cast<float>(S));
  const float inv_S = 1.0f / static_cast<float>(S);
  // Scratch carve-up: tmpA/tmpB are [HV][S][S]; the rest are per-token vectors.
  float* tmpA = scratch;
  float* tmpB = scratch + HV * S * S;
  float* kvbuf = tmpB + HV * S * S;
  float* deltabuf = kvbuf + HV * S;
  float* obuf = deltabuf + HV * S;
  float* tmpC = obuf + HV * S;
  float* tmpD = tmpC + HV * S;
  float* gbuf = tmpD + HV * S;
  float* ssbuf = gbuf + HV;
  float* scalebuf = ssbuf + HV;
  Workspace ws;
  ws.stream = stream;

  // State as [R][HK][S][S]: flat value head hd = r*HK + qh, so qh = hd % HK,
  // matching k_gdn_seq / k_gdn_seq_cols in src/qwen35/q36_engine.cu
  // (`const int qh = head % h_k`). The [HV][S][S] buffer re-reads as
  // [R][HK][S][S]; the flat head order is unchanged, so cann_state_to_logical
  // in the test still applies. (An earlier [HK][R] layout paired hd with
  // qh = hd / R, which is wrong for half the value heads.)
  Tensor v_state4 = Tensor(state, {R, HK, S, S}, {HK * S * S, S * S, S, 1}, 0, {HV, S, S});
  Tensor v_tmpA4 = Tensor(tmpA, {R, HK, S, S}, {HK * S * S, S * S, S, 1}, 0, {HV, S, S});
  Tensor v_tmpB4 = Tensor(tmpB, {R, HK, S, S}, {HK * S * S, S * S, S, 1}, 0, {HV, S, S});
  Tensor v_kv = Tensor::contig(kvbuf, {HV, S});
  Tensor v_delta = Tensor::contig(deltabuf, {HV, S});
  Tensor v_o = Tensor::contig(obuf, {HV, S});
  // 3D views of kvbuf/obuf for the ReduceSum outputs: reducing the 4D
  // {HK,R,S,S} contraction over dim 2 yields {HK,R,S}, so the output
  // tensor must be 3D -- a {HV,S} view is rejected with 161002
  // (ACLNN_ERR_PARAM_INVALID) even though the element count matches.
  Tensor v_kv3 = Tensor::contig(kvbuf, {R, HK, S});
  Tensor v_o3 = Tensor::contig(obuf, {R, HK, S});
  Tensor v_tmpC = Tensor::contig(tmpC, {HV, S});
  Tensor v_tmpD = Tensor::contig(tmpD, {HV, S});
  Tensor v_g = Tensor::contig(gbuf, {HV});
  Tensor v_g1 = Tensor(gbuf, {HV, 1}, {1, 1}, 0, {HV});
  Tensor v_g4 = Tensor(gbuf, {R, HK, 1, 1}, {HK, 1, 1, 1}, 0, {HV});
  Tensor v_ss = Tensor::contig(ssbuf, {HV, 1});
  Tensor v_scale = Tensor::contig(scalebuf, {HV, 1});
  Tensor v_gamma = Tensor::contig(const_cast<float*>(gamma), {S});

  for (int t = 0; t < tokens; ++t) {
    const float* h_t = h + static_cast<size_t>(t) * static_cast<size_t>(C);
    const float* gate_t = gate + static_cast<size_t>(t) * static_cast<size_t>(HV);
    const float* beta_t = beta + static_cast<size_t>(t) * static_cast<size_t>(HV);
    const float* z_t = z + static_cast<size_t>(t) * static_cast<size_t>(ZV);
    float* y_t = y + static_cast<size_t>(t) * static_cast<size_t>(ZV);
    Tensor v_gate = Tensor::contig(const_cast<float*>(gate_t), {HV});
    Tensor v_beta1 = Tensor(const_cast<float*>(beta_t), {HV, 1}, {1, 1}, 0, {HV});
    Tensor v_v = Tensor::contig(const_cast<float*>(h_t) + 2 * HK * S, {HV, S});
    Tensor v_z = Tensor::contig(const_cast<float*>(z_t), {HV, S});
    Tensor v_y = Tensor::contig(y_t, {HV, S});
    // q / k slices as [HK][S], then as broadcast views [1][HK][S][1] (r axis).
    Tensor v_q4 = Tensor(const_cast<float*>(h_t), {1, HK, S, 1}, {HK * S, S, 1, 1}, 0, {HK, S});
    Tensor v_k4 = Tensor(const_cast<float*>(h_t) + HK * S, {1, HK, S, 1}, {HK * S, S, 1, 1}, 0,
                         {HK, S});
    Tensor v_delta4 = Tensor(deltabuf, {R, HK, 1, S}, {HK * S, S, S, 1}, 0, {HV, S});

    // g = exp(gate)
    if (!op_exp(ws, v_gate, v_g, err)) return false;
    // kv = sum_i S[h][i][j] * k[qh][i]  (Mul broadcast, then ReduceSum over i)
    if (!op_mul(ws, v_state4, v_k4, v_tmpA4, err)) return false;
    if (!op_reduce_sum(ws, v_tmpA4, 2, false, v_kv3, err)) return false;
    // delta = (v - g * kv) * beta
    if (!op_mul(ws, v_g1, v_kv, v_tmpC, err)) return false;
    if (!op_sub(ws, v_v, v_tmpC, v_tmpD, err)) return false;
    if (!op_mul(ws, v_tmpD, v_beta1, v_delta, err)) return false;
    // S <- g * S + k (x) delta
    if (!op_mul(ws, v_state4, v_g4, v_tmpA4, err)) return false;
    if (!op_mul(ws, v_k4, v_delta4, v_tmpB4, err)) return false;
    if (!op_add(ws, v_tmpA4, v_tmpB4, 1.0f, v_state4, err)) return false;
    // o = sum_i S_new[h][i][j] * q[qh][i]; o_raw = o / sqrt(S)
    if (!op_mul(ws, v_state4, v_q4, v_tmpA4, err)) return false;
    if (!op_reduce_sum(ws, v_tmpA4, 2, false, v_o3, err)) return false;
    if (!op_muls(ws, v_o, inv_sqrt_S, v_tmpC, err)) return false;
    // RMS norm over each head: scale = rsqrt(mean(o_raw^2) + eps)
    if (!op_mul(ws, v_tmpC, v_tmpC, v_tmpD, err)) return false;
    if (!op_reduce_sum(ws, v_tmpD, 1, true, v_ss, err)) return false;
    if (!op_muls(ws, v_ss, inv_S, v_scale, err)) return false;
    if (!op_adds(ws, v_scale, eps, 1.0f, v_ss, err)) return false;
    if (!op_rsqrt(ws, v_ss, v_scale, err)) return false;
    // y = o_raw * scale * gamma * silu(z)
    if (!op_mul(ws, v_tmpC, v_scale, v_tmpD, err)) return false;
    if (!op_mul(ws, v_tmpD, v_gamma, v_tmpC, err)) return false;
    if (!op_silu(ws, v_z, v_tmpD, err)) return false;
    if (!op_mul(ws, v_tmpC, v_tmpD, v_y, err)) return false;
  }
  return true;
}

}  // namespace strata::cann
