// test_cann_gemm.cpp: standalone on-device check of strata::cann::CannGemm.
//
// Runs ON the Ascend box (it needs a real NPU) and compares aclnnMatmul
// output against a CPU reference computed in FP32, printing per-shape
// max abs error, mean abs error and max relative error, and exiting
// non-zero on any failure.
//
// Shapes are the GEMMs the Qwen3.6 layers actually use, read from
// src/qwen35/q36_engine.cu and verified against the GGUF on the target
// box (/data/strata-qwen36/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf):
//   geometry: n_embd 2048, n_head 16, n_head_kv 2, head_dim 256,
//   n_ff 512, n_expert 256, top_k 8, ssm_state_size 128,
//   ssm_k_heads 16, ssm_v_heads 32, ssm_d_conv 4, ssm_value_dim 4096,
//   ssm_conv_channels 8192, n_layers 41, vocab 248320.
// The engine computes y[T][n_out] = W[n_out][n_in] . x[T][n_in]
// (f32mm / prefill::Gemm::native), i.e. C[M, N] = A[M, K] . B^T with
// B stored row-major [N, K] — the b_transposed=true path. T > 8 is a
// prompt chunk (FP16 activations, FP32 accumulation on CUDA); T <= 8
// is a decode batch (the engine uses its MMVQ there, but the GEMM
// must still be correct at those shapes for the prompt path's tail).
// The weights here are synthetic FP16 (the engine's dequantized
// weights); GGUF dequant itself is a later phase and is NOT tested.
//
// Metric: max relative error is ||computed - ref||_inf / ||ref||_inf
// (max abs error divided by the output scale). A per-element relative
// error was tried first and rejected: with FP32 inputs down-converted
// to FP16 in the cube, the absolute error floor (~1e-2 at output scale
// ~60) makes near-zero outputs report rel errors up to 0.11 even
// though every output is within 1.5e-4 of the output scale — the
// per-element metric measures the test's near-zero outputs, not the
// GEMM. The scale-relative metric is the standard GEMM check, and
// max abs / mean abs are printed alongside so the absolute scale is
// visible.
// Tolerances (stated before the run, not tuned to it):
//   FP16 in -> FP32 out, cubeMathType=1 (FP32 accumulation, same as
//     the engine's cuBLAS COMPUTE_32F): max rel err < 1e-3. FP16
//     inputs are exact in FP32 and only accumulation rounds.
//   FP16 in -> FP16 out: max rel err < 1e-2 (FP16 output rounding).
//   FP32 in -> FP32 out, cubeMathType=0 (KEEP_DTYPE): max rel err
//     < 1e-3. The aclnn_mv.h header comment ("FP32 + KEEP_DTYPE is
//     rejected on Ascend 910") turns out to concern the Mv op, not
//     Matmul: aclnnMatmul accepts FP32 with cubeMathType=0 on this
//     box, and this table exercises it as the primary FP32 path.
//   FP32 in -> FP32 out, cubeMathType=1 (ALLOW_FP32_*): max rel err
//     < 2e-2. With cube=1 the FP32 inputs are down-converted to FP16
//     in the cube, so the error floor is FP16 input rounding
//     (~1.5e-4 of output scale, measured). Kept as documentation of
//     the down-precision fallback, not the recommended path.
// The reference is a plain CPU triple loop in FP32 (OpenMP over rows
// when available); it is slow but independent of aclnn.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "acl/acl_rt.h"
#include "cann_gemm.h"

namespace {

// Deterministic PRNG (splitmix64); uniform floats in [-1, 1].
struct Rng {
  std::uint64_t s;
  explicit Rng(std::uint64_t seed) : s(seed) {}
  std::uint64_t next() {
    std::uint64_t z = (s += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
  }
  float uniform() {  // [-1, 1)
    return static_cast<float>((next() >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0);
  }
};

// IEEE binary16 <-> float, by hand (no <half> dependency).
std::uint16_t f32_to_f16(float f) {
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  const std::uint32_t sign = (u >> 16) & 0x8000;
  const int exp = static_cast<int>((u >> 23) & 0xff) - 112;  // 127-15
  const std::uint32_t mant = u & 0x7fffff;
  if (exp <= 0) {
    if (exp < -10) return static_cast<std::uint16_t>(sign);
    const std::uint32_t m = mant | 0x800000;
    const int shift = 14 - exp;
    std::uint32_t h = m >> shift;
    if ((m >> (shift - 1)) & 1) ++h;  // round to nearest (ties not exact, fine for a test)
    return static_cast<std::uint16_t>(sign | h);
  }
  if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7c00 | (mant ? 0x200 : 0));
  return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) | (mant >> 13));
}

float f16_to_f32(std::uint16_t h) {
  const std::uint32_t sign = (static_cast<std::uint32_t>(h & 0x8000)) << 16;
  const std::uint32_t exp = (h >> 10) & 0x1f;
  const std::uint32_t mant = h & 0x3ff;
  std::uint32_t u;
  if (exp == 0) {
    if (mant == 0) {
      u = sign;
    } else {
      std::uint32_t m = mant, shift = 0;
      while ((m & 0x400) == 0) {
        m <<= 1;
        ++shift;
      }
      u = sign | ((127 - 14 - static_cast<int>(shift)) << 23) | ((m & 0x3ff) << 13);
    }
  } else if (exp == 0x1f) {
    u = sign | 0x7f800000u | (mant << 13);
  } else {
    u = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// NOTE on the NT reference: C[M, N] = A[M, K] . B^T with B stored
// [N, K]. A direct NT loop strides through B with step K and re-reads
// it M times, which is orders of magnitude slower and dominated the
// first test run. Instead the caller pre-transposes B into BT [K, N]
// once (a single pass; a permutation, no rounding) and runs the
// contiguous ref_gemm_nn kernel below.

// Same, plain C = A[M, K] . B[K, N], B stored [K, N].
void ref_gemm_nn(const float* a, const float* b, float* c, std::int64_t m, std::int64_t n,
                 std::int64_t k) {
  std::fill(c, c + static_cast<size_t>(m) * static_cast<size_t>(n), 0.0f);
#ifdef STRATA_CANN_GEMM_TEST_OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (std::int64_t i = 0; i < m; ++i) {
    float* crow = c + static_cast<size_t>(i) * static_cast<size_t>(n);
    const float* arow = a + static_cast<size_t>(i) * static_cast<size_t>(k);
    for (std::int64_t kk = 0; kk < k; ++kk) {
      const float av = arow[kk];
      const float* brow = b + static_cast<size_t>(kk) * static_cast<size_t>(n);
      for (std::int64_t j = 0; j < n; ++j) crow[j] += av * brow[j];
    }
  }
}

struct Shape {
  const char* name;      // engine tensor / use
  std::int64_t m;        // T (tokens)
  std::int64_t n;        // n_out
  std::int64_t k;        // n_in
  strata::cann::DType in_dtype;
  strata::cann::DType out_dtype;
  bool b_transposed;     // true: B stored [N, K] (engine W layout)
  std::int8_t cube;      // cubeMathType
  double tol;            // max-rel-error pass threshold
};

bool dev_alloc(void** p, std::size_t bytes, std::string& err) {
  if (aclrtMalloc(p, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
    err = "aclrtMalloc failed";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  std::printf("test_cann_gemm: CANN 9.0.0 aclnnMatmul vs CPU FP32 reference\n");
  strata::cann::CannGemm gemm;
  std::string err;
  if (!gemm.init(0, err)) {
    std::fprintf(stderr, "test_cann_gemm: init failed: %s\n", err.c_str());
    return 1;
  }
  std::uint32_t ndev = 0;
  aclrtGetDeviceCount(&ndev);
  std::printf("test_cann_gemm: device=%d, NPU count=%u\n", gemm.device(), ndev);

  // FP16 shapes: the engine's main projections (W stored [N, K]).
  // FP32 shapes: router / GDN alpha-beta / shared-expert gate input.
  const Shape shapes[] = {
      // name,            T,    N,     K,    in,    out,   Bt, cube, tol
      {"qkv T=1",         1,    8192,  2048, strata::cann::DType::kFp16, strata::cann::DType::kFp32, true, 1, 1e-3},
      {"qkv T=2",         2,    8192,  2048, strata::cann::DType::kFp16, strata::cann::DType::kFp32, true, 1, 1e-3},
      {"qkv T=8",         8,    8192,  2048, strata::cann::DType::kFp16, strata::cann::DType::kFp32, true, 1, 1e-3},
      {"qkv T=512",       512,  8192,  2048, strata::cann::DType::kFp16, strata::cann::DType::kFp32, true, 1, 1e-3},
      {"qkv T=1024",      1024, 8192,  2048, strata::cann::DType::kFp16, strata::cann::DType::kFp32, true, 1, 1e-3},
      {"qkv T=4096",      4096, 8192,  2048, strata::cann::DType::kFp16, strata::cann::DType::kFp32, true, 1, 1e-3},
      {"attn_gate T=512", 512,  4096,  2048, strata::cann::DType::kFp16, strata::cann::DType::kFp32, true, 1, 1e-3},
      {"ssm_out T=512",   512,  2048,  4096, strata::cann::DType::kFp16, strata::cann::DType::kFp32, true, 1, 1e-3},
      {"attn_q T=512",    512,  8192,  2048, strata::cann::DType::kFp16, strata::cann::DType::kFp32, true, 1, 1e-3},
      {"attn_k T=512",    512,  512,   2048, strata::cann::DType::kFp16, strata::cann::DType::kFp32, true, 1, 1e-3},
      {"attn_v T=512",    512,  512,   2048, strata::cann::DType::kFp16, strata::cann::DType::kFp32, true, 1, 1e-3},
      {"attn_out T=512",  512,  2048,  4096, strata::cann::DType::kFp16, strata::cann::DType::kFp32, true, 1, 1e-3},
      {"sh_gate T=8",     8,    512,   2048, strata::cann::DType::kFp16, strata::cann::DType::kFp32, true, 1, 1e-3},
      {"sh_gate T=512",   512,  512,   2048, strata::cann::DType::kFp16, strata::cann::DType::kFp32, true, 1, 1e-3},
      {"sh_down T=512",   512,  2048,  512,  strata::cann::DType::kFp16, strata::cann::DType::kFp32, true, 1, 1e-3},
      {"expert_gate T=8", 8,    512,   2048, strata::cann::DType::kFp16, strata::cann::DType::kFp32, true, 1, 1e-3},
      {"head T=1",        1,    248320, 2048, strata::cann::DType::kFp16, strata::cann::DType::kFp32, true, 1, 1e-3},
      {"head T=8",        8,    248320, 2048, strata::cann::DType::kFp16, strata::cann::DType::kFp32, true, 1, 1e-3},
      {"head T=512",      512,  248320, 2048, strata::cann::DType::kFp16, strata::cann::DType::kFp32, true, 1, 1e-3},
      {"eh_proj T=512",   512,  2048,  4096, strata::cann::DType::kFp16, strata::cann::DType::kFp32, true, 1, 1e-3},
      // Plain (non-transposed) B, same qkv sizes: covers the NN path.
      {"qkv-nn T=512",    512,  8192,  2048, strata::cann::DType::kFp16, strata::cann::DType::kFp32, false, 1, 1e-3},
      // FP16 out (not the engine's path, but the wrapper supports it).
      {"qkv-fp16out T=8", 8,    8192,  2048, strata::cann::DType::kFp16, strata::cann::DType::kFp16, true, 1, 1e-2},
      // FP32 shapes (router / alpha-beta / shared-expert gate input),
      // primary path cube=0 (KEEP_DTYPE, true FP32 on this box).
      {"router T=1",      1,    256,   2048, strata::cann::DType::kFp32, strata::cann::DType::kFp32, true, 0, 1e-3},
      {"router T=8",      8,    256,   2048, strata::cann::DType::kFp32, strata::cann::DType::kFp32, true, 0, 1e-3},
      {"router T=512",    512,  256,   2048, strata::cann::DType::kFp32, strata::cann::DType::kFp32, true, 0, 1e-3},
      {"router T=1024",   1024, 256,   2048, strata::cann::DType::kFp32, strata::cann::DType::kFp32, true, 0, 1e-3},
      {"router_ext T=8",  8,    257,   2048, strata::cann::DType::kFp32, strata::cann::DType::kFp32, true, 0, 1e-3},
      {"alpha/beta T=512",512,  32,    2048, strata::cann::DType::kFp32, strata::cann::DType::kFp32, true, 0, 1e-3},
      {"sh_gate_inp T=8", 8,    1,     2048, strata::cann::DType::kFp32, strata::cann::DType::kFp32, true, 0, 1e-3},
      // Same FP32 shapes with cube=1: documents the down-precision
      // fallback (FP32 inputs converted to FP16 in the cube).
      {"router cube1 T=8",   8,    256, 2048, strata::cann::DType::kFp32, strata::cann::DType::kFp32, true, 1, 2e-2},
      {"router cube1 T=512", 512,  256, 2048, strata::cann::DType::kFp32, strata::cann::DType::kFp32, true, 1, 2e-2},
      {"alpha/beta c1 T=512",512,  32,  2048, strata::cann::DType::kFp32, strata::cann::DType::kFp32, true, 1, 2e-2},
  };

  int failures = 0;
  for (const Shape& s : shapes) {
    const std::size_t na = static_cast<std::size_t>(s.m) * static_cast<std::size_t>(s.k);
    const std::size_t nb = s.b_transposed
        ? static_cast<std::size_t>(s.n) * static_cast<std::size_t>(s.k)
        : static_cast<std::size_t>(s.k) * static_cast<std::size_t>(s.n);
    const std::size_t nc = static_cast<std::size_t>(s.m) * static_cast<std::size_t>(s.n);
    // Host float copies (the reference inputs).
    std::vector<float> ha(na), hb(nb), ref(nc), got_f(nc);
    Rng rng(0x1234abcdull + static_cast<std::uint64_t>(s.m) * 1315423911ull +
            static_cast<std::uint64_t>(s.n));
    for (float& v : ha) v = rng.uniform();
    for (float& v : hb) v = rng.uniform();
    // Device buffers in the test dtype.
    const std::size_t in_sz = strata::cann::CannGemm::dtype_size(s.in_dtype);
    const std::size_t out_sz = strata::cann::CannGemm::dtype_size(s.out_dtype);
    std::vector<std::uint8_t> h_a_in(na * in_sz), h_b_in(nb * in_sz);
    if (s.in_dtype == strata::cann::DType::kFp16) {
      auto* pa = reinterpret_cast<std::uint16_t*>(h_a_in.data());
      auto* pb = reinterpret_cast<std::uint16_t*>(h_b_in.data());
      for (std::size_t i = 0; i < na; ++i) pa[i] = f32_to_f16(ha[i]);
      for (std::size_t i = 0; i < nb; ++i) pb[i] = f32_to_f16(hb[i]);
      // Reference uses the widened FP16 values, not the original
      // floats, so input rounding is not counted as GEMM error.
      for (std::size_t i = 0; i < na; ++i) ha[i] = f16_to_f32(pa[i]);
      for (std::size_t i = 0; i < nb; ++i) hb[i] = f16_to_f32(pb[i]);
    } else {
      std::memcpy(h_a_in.data(), ha.data(), na * 4);
      std::memcpy(h_b_in.data(), hb.data(), nb * 4);
    }
    if (s.b_transposed) {
      // Transpose B [N, K] -> BT [K, N] once, then the contiguous NN kernel.
      std::vector<float> bt(nb);
#ifdef STRATA_CANN_GEMM_TEST_OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (std::int64_t j = 0; j < s.n; ++j)
        for (std::int64_t kk = 0; kk < s.k; ++kk)
          bt[static_cast<size_t>(kk) * static_cast<size_t>(s.n) +
             static_cast<size_t>(j)] =
              hb[static_cast<size_t>(j) * static_cast<size_t>(s.k) + static_cast<size_t>(kk)];
      ref_gemm_nn(ha.data(), bt.data(), ref.data(), s.m, s.n, s.k);
    } else ref_gemm_nn(ha.data(), hb.data(), ref.data(), s.m, s.n, s.k);

    void *d_a = nullptr, *d_b = nullptr, *d_o = nullptr;
    if (!dev_alloc(&d_a, na * in_sz, err) || !dev_alloc(&d_b, nb * in_sz, err) ||
        !dev_alloc(&d_o, nc * out_sz, err)) {
      std::fprintf(stderr, "FAIL %-18s: %s\n", s.name, err.c_str());
      ++failures;
      continue;
    }
    aclrtMemcpy(d_a, na * in_sz, h_a_in.data(), na * in_sz, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(d_b, nb * in_sz, h_b_in.data(), nb * in_sz, ACL_MEMCPY_HOST_TO_DEVICE);
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = gemm.gemm(d_a, d_b, d_o, s.m, s.n, s.k, s.in_dtype, s.out_dtype,
                              s.b_transposed, s.cube, err);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (!ok) {
      std::fprintf(stderr, "FAIL %-18s: gemm: %s\n", s.name, err.c_str());
      ++failures;
      aclrtFree(d_a);
      aclrtFree(d_b);
      aclrtFree(d_o);
      continue;
    }
    std::vector<std::uint8_t> h_o(nc * out_sz);
    aclrtMemcpy(h_o.data(), nc * out_sz, d_o, nc * out_sz, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtFree(d_a);
    aclrtFree(d_b);
    aclrtFree(d_o);
    if (s.out_dtype == strata::cann::DType::kFp16) {
      const auto* po = reinterpret_cast<const std::uint16_t*>(h_o.data());
      for (std::size_t i = 0; i < nc; ++i) got_f[i] = f16_to_f32(po[i]);
    } else {
      std::memcpy(got_f.data(), h_o.data(), nc * 4);
    }
    double max_abs = 0, sum_abs = 0, cmax = 0;
    for (float v : ref) cmax = std::max(cmax, static_cast<double>(std::fabs(v)));
    for (std::size_t i = 0; i < nc; ++i) {
      const double d = std::fabs(static_cast<double>(got_f[i]) - static_cast<double>(ref[i]));
      max_abs = std::max(max_abs, d);
      sum_abs += d;
    }
    const double mean_abs = sum_abs / static_cast<double>(nc);
    const double max_rel = cmax > 0 ? max_abs / cmax : max_abs;  // ||d||_inf / ||ref||_inf
    const bool pass = max_rel < s.tol;
    std::printf("%-18s M=%-5lld N=%-6lld K=%-5lld %s->%s %s cube=%d %8.2f ms  "
                "max|ref|=%.4g max_abs=%.4g mean_abs=%.4g max_rel=%.4g tol=%.0e  %s\n",
                s.name, static_cast<long long>(s.m), static_cast<long long>(s.n),
                static_cast<long long>(s.k),
                strata::cann::CannGemm::dtype_name(s.in_dtype),
                strata::cann::CannGemm::dtype_name(s.out_dtype), s.b_transposed ? "NT" : "NN",
                static_cast<int>(s.cube), ms, cmax, max_abs, mean_abs, max_rel, s.tol,
                pass ? "PASS" : "FAIL");
    if (!pass) ++failures;
  }

  gemm.shutdown();
  std::printf("test_cann_gemm: %s (%d failures)\n", failures == 0 ? "ALL PASS" : "FAILURES",
              failures);
  return failures == 0 ? 0 : 1;
}
