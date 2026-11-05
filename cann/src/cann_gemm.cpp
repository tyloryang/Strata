// cann_gemm.cpp: aclnnMatmul wrapper (Phase 1).
//
// API signatures checked against the headers on the first target
// (CANN 9.0.0, /usr/local/Ascend/cann-9.0.0/include):
//   aclnnMatmulGetWorkspaceSize(const aclTensor*, const aclTensor*,
//       aclTensor*, int8_t cubeMathType, uint64_t*, aclOpExecutor**)
//   aclnnMatmul(void*, uint64_t, aclOpExecutor*, aclrtStream)
//     in aclnnop/aclnn_matmul.h; symbols live in libopapi_nn.so.
//   aclCreateTensor(viewDims, viewDimsNum, aclDataType, stride, offset,
//       aclFormat, storageDims, storageDimsNum, tensorData)
//   aclDestroyTensor(const aclTensor*)
//     in aclnn/opdev/common_types.h.
//   aclrtMalloc(void**, size_t, aclrtMemMallocPolicy),
//   aclrtFree(void*), aclrtMemcpy, aclrtCreateStream, aclrtSynchronizeStream
//     in acl/acl_rt.h.
// CANN 9.0.0 exposes no aclDestroyExecutor: the official aclnn samples
// destroy only the tensors after the second-stage call, and this file
// does the same (the executor is owned by the aclnn runtime once the
// second-stage call returns).
#include "cann_gemm.h"

#include <cstdio>

#include "acl/acl_base_rt.h"
#include "aclnn/aclnn_base.h"
#include "aclnn/opdev/op_errno.h"  // ACLNN_SUCCESS (not pulled in by aclnn_matmul.h)
#include "aclnnop/aclnn_matmul.h"

namespace strata::cann {

CannGemm::~CannGemm() { shutdown(); }

bool CannGemm::init(int device, std::string& err) {
  if (inited_) return true;
  aclError rerr = aclInit(nullptr);
  if (rerr != ACL_SUCCESS) {
    err = "aclInit failed: " + std::to_string(static_cast<int>(rerr));
    return false;
  }
  rerr = aclrtSetDevice(device);
  if (rerr != ACL_SUCCESS) {
    err = "aclrtSetDevice(" + std::to_string(device) +
          ") failed: " + std::to_string(static_cast<int>(rerr));
    aclFinalize();
    return false;
  }
  rerr = aclrtCreateStream(&stream_);
  if (rerr != ACL_SUCCESS) {
    err = "aclrtCreateStream failed: " + std::to_string(static_cast<int>(rerr));
    aclFinalize();
    return false;
  }
  device_ = device;
  inited_ = true;
  return true;
}

void CannGemm::shutdown() {
  if (stream_ != nullptr) {
    aclrtDestroyStream(stream_);
    stream_ = nullptr;
  }
  if (inited_) {
    aclFinalize();
    inited_ = false;
  }
}

std::size_t CannGemm::dtype_size(DType t) {
  switch (t) {
    case DType::kFp16: return 2;
    case DType::kFp32: return 4;
  }
  return 0;
}

const char* CannGemm::dtype_name(DType t) {
  switch (t) {
    case DType::kFp16: return "fp16";
    case DType::kFp32: return "fp32";
  }
  return "?";
}

aclDataType CannGemm::acl_dtype(DType t) {
  switch (t) {
    case DType::kFp16: return ACL_FLOAT16;
    case DType::kFp32: return ACL_FLOAT;
  }
  return ACL_FLOAT;
}

bool CannGemm::gemm(const void* a_dev, const void* b_dev, void* out_dev, std::int64_t m,
                    std::int64_t n, std::int64_t k, DType in_dtype, DType out_dtype,
                    bool b_transposed, std::int8_t cube_math_type, std::string& err) {
  if (!inited_) {
    err = "CannGemm::gemm called before init";
    return false;
  }
  if (m <= 0 || n <= 0 || k <= 0 || a_dev == nullptr || b_dev == nullptr || out_dev == nullptr) {
    err = "CannGemm::gemm: bad shape or null pointer";
    return false;
  }
  const aclDataType in_dt = acl_dtype(in_dtype);
  const aclDataType out_dt = acl_dtype(out_dtype);

  // A: [M, K], contiguous row-major.
  const std::int64_t a_dims[2] = {m, k};
  aclTensor* a = aclCreateTensor(a_dims, 2, in_dt, nullptr, 0, ACL_FORMAT_ND, a_dims, 2,
                                 const_cast<void*>(a_dev));
  if (a == nullptr) {
    err = "aclCreateTensor(A) failed";
    return false;
  }
  // B: plain [K, N] contiguous, or the engine's [N, K] storage viewed
  // as [K, N] through strides (element (i, j) at i*1 + j*K, i.e. B^T).
  // aclnnMatmul itself has no transpose flag; the strided view is how
  // the transpose is expressed.
  const std::int64_t b_dims[2] = {k, n};
  const std::int64_t b_storage[2] = {b_transposed ? n : k, b_transposed ? k : n};
  const std::int64_t b_stride_nt[2] = {1, k};
  aclTensor* b = aclCreateTensor(b_dims, 2, in_dt, b_transposed ? b_stride_nt : nullptr, 0,
                                 ACL_FORMAT_ND, b_storage, 2, const_cast<void*>(b_dev));
  if (b == nullptr) {
    aclDestroyTensor(a);
    err = "aclCreateTensor(B) failed";
    return false;
  }
  // Out: [M, N], contiguous row-major.
  const std::int64_t o_dims[2] = {m, n};
  aclTensor* out = aclCreateTensor(o_dims, 2, out_dt, nullptr, 0, ACL_FORMAT_ND, o_dims, 2, out_dev);
  if (out == nullptr) {
    aclDestroyTensor(a);
    aclDestroyTensor(b);
    err = "aclCreateTensor(out) failed";
    return false;
  }

  auto fail = [&](const char* what, aclnnStatus st) {
    err = std::string(what) + " failed: aclnn status " + std::to_string(static_cast<int>(st));
    aclDestroyTensor(a);
    aclDestroyTensor(b);
    aclDestroyTensor(out);
    return false;
  };

  std::uint64_t ws_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus st = aclnnMatmulGetWorkspaceSize(a, b, out, cube_math_type, &ws_size, &executor);
  if (st != ACLNN_SUCCESS) return fail("aclnnMatmulGetWorkspaceSize", st);

  void* ws = nullptr;
  if (ws_size > 0) {
    aclError rerr = aclrtMalloc(&ws, ws_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (rerr != ACL_SUCCESS) {
      aclDestroyTensor(a);
      aclDestroyTensor(b);
      aclDestroyTensor(out);
      err = "aclrtMalloc(workspace " + std::to_string(ws_size) +
            ") failed: " + std::to_string(static_cast<int>(rerr));
      return false;
    }
  }
  st = aclnnMatmul(ws, ws_size, executor, stream_);
  if (st != ACLNN_SUCCESS) {
    if (ws != nullptr) aclrtFree(ws);
    return fail("aclnnMatmul", st);
  }
  aclError rerr = aclrtSynchronizeStream(stream_);
  if (ws != nullptr) aclrtFree(ws);
  aclDestroyTensor(a);
  aclDestroyTensor(b);
  aclDestroyTensor(out);
  if (rerr != ACL_SUCCESS) {
    err = "aclrtSynchronizeStream failed: " + std::to_string(static_cast<int>(rerr));
    return false;
  }
  return true;
}

}  // namespace strata::cann
