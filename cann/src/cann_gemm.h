// cann_gemm.h: CANN matmul for the Strata Ascend backend (Phase 1).
//
// A thin wrapper over aclnnMatmul (CANN 9.0.0, Ascend 910B2C, verified on
// the first target 2026-11-05). It covers the GEMMs the Qwen3.6 layers
// actually use — read from src/qwen35/q36_engine.cu and verified against
// the GGUF on the target box (see cann/PHASE1_GEMM.md for the shape table):
//   q/k/v/o projections, gate/up/down (shared + routed experts), the
//   router (FP32), the GDN alpha/beta (FP32), the lm head, eh_proj.
// The engine computes y[T][n_out] = W[n_out][n_in] . x[T][n_in], i.e.
// C[M, N] = A[M, K] . B^T with B stored row-major [N, K] — that is the
// b_transposed=true path here. The plain C = A . B path is also supported.
//
// Dtypes follow the engine: the main projections run on FP16 activations
// with FP32 accumulation (the engine's prompt-chunk path dequantizes the
// GGUF weights to FP16 and calls cuBLAS with COMPUTE_32F); the router
// and the GDN alpha/beta stay FP32. This wrapper does NOT dequantize
// GGUF weights — the caller supplies already-dequantized device buffers
// (dequant is a later phase). Out is FP32 for the engine's use; FP16 out
// is supported for completeness.
//
// Cube math type (aclnnMatmul's cubeMathType argument):
//   0 KEEP_DTYPE — compute in the input dtype. An aclnn_mv.h header
//     comment says FP32 + KEEP_DTYPE is rejected on Ascend 910, but
//     that concerns the Mv op, not Matmul: aclnnMatmul accepts FP32
//     with cubeMathType=0 on the first target, and the test binary
//     exercises it as the primary FP32 path (true FP32, ~1e-6
//     scale-relative error). Use 0 for the FP32 router / alpha-beta.
//   1 ALLOW_FP32_COMPUTE — FP16 inputs accumulate in FP32 (what the
//     engine wants for its main projections); FP32 inputs are
//     down-converted to FP16 for the cube, so an FP32 GEMM with
//     cube=1 runs at FP16 precision. The test keeps cube=1 FP32
//     shapes only to document that fallback.
//
// One call = one GetWorkspaceSize + one aclnnMatmul + one stream sync,
// with a freshly allocated workspace each time. Correct first, fast
// later: reusing the workspace / executor across calls is open work
// (see cann/PHASE1_GEMM.md). All pointers are device pointers owned by
// the caller; the class owns only the ACL init, the device and a stream.
#pragma once

#include <cstdint>
#include <string>

#include "acl/acl_rt.h"

namespace strata::cann {

enum class DType {
  kFp16,
  kFp32,
};

class CannGemm {
 public:
  // cubeMathType values, as documented in the aclnn headers on the
  // target (aclnnop/aclnn_mv.h: 0 KEEP_DTYPE, 1 ALLOW_FP32_*).
  static constexpr std::int8_t kCubeKeepDtype = 0;
  static constexpr std::int8_t kCubeAllowFp32 = 1;

  CannGemm() = default;
  ~CannGemm();
  CannGemm(const CannGemm&) = delete;
  CannGemm& operator=(const CannGemm&) = delete;

  // aclInit + aclrtSetDevice(device) + one stream. `device` is the
  // runtime-visible id (0 inside the container on the first target;
  // never hard-code the host's /dev/davinci number).
  bool init(int device, std::string& err);
  void shutdown();
  bool initialized() const { return inited_; }
  int device() const { return device_; }
  aclrtStream stream() const { return stream_; }

  // C[M, N] = A[M, K] . B[K, N] (b_transposed=false, B stored [K, N])
  //      or = A[M, K] . B^T    (b_transposed=true,  B stored [N, K],
  //      the engine's W layout). A, B, out are contiguous row-major
  // device buffers. in_dtype applies to A and B; out has out_dtype.
  // Returns false and fills `err` (with the aclnn status code) on any
  // CANN failure. The call is synchronous: the stream is synchronized
  // before it returns.
  bool gemm(const void* a_dev, const void* b_dev, void* out_dev, std::int64_t m, std::int64_t n,
            std::int64_t k, DType in_dtype, DType out_dtype, bool b_transposed,
            std::int8_t cube_math_type, std::string& err);

  static std::size_t dtype_size(DType t);
  static const char* dtype_name(DType t);
  static aclDataType acl_dtype(DType t);

 private:
  aclrtStream stream_ = nullptr;
  bool inited_ = false;
  int device_ = 0;
};

}  // namespace strata::cann
