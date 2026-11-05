// cann/src/cann_elementwise.h - Phase 1: small elementwise ops that fall out of the
// GDN work, on Ascend via aclnn (FP32). These are the same patterns as the
// attention file (elementwise ops + ReduceSum), so they share its OpRunner
// machinery only through a copy of the same small helpers in the .cpp.
//
// Covered:
//   * RMSNorm: y[r] = rms(x[r]) * w, rms over the row, scale = rsqrt(mean(x^2) + eps)
//     (src/qwen35/q36_engine.cu: k_rms / k_rms_ld; the engine's per-layer norms)
//   * SiLU: y = x * sigmoid(x) (the SwiGLU gate's activation and the conv's)
//   * NEOX partial RoPE: the first n_rot of each head is rotated in (i, i + n_rot/2)
//     pairs (src/kernels/cuda/rope.cu); the cos/sin table is built on the host
//     in float64, exactly like the CUDA build_rope_table, and the per-token
//     cos/sin rows are passed in already gathered.
//
// NOT covered here: the full-attention layers (native_flash_attn.cu: QK^T +
// softmax + PV over a KV cache, plus the QSA indexer/pooling) -- that is the
// next step, listed in cann/PHASE1_ATTENTION.md.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "acl/acl_rt.h"

namespace strata::cann {

/// y = rmsnorm(x) * w, row-wise over [rows][n]. x / y are [rows*n] device
/// floats, w is [n]. eps is the model's rms eps (1e-6 for Qwen3.6).
/// scratch must hold at least 2*rows*n + 2*rows floats.
bool cann_rmsnorm(const float* x, const float* w, float* y, int rows, int n, float eps,
                  aclrtStream stream, float* scratch, std::string* err);

/// y = silu(x) = x * sigmoid(x), elementwise over n floats.
/// scratch is unused (may be null).
bool cann_silu(const float* x, float* y, std::int64_t n, aclrtStream stream, std::string* err);

/// Host-side RoPE table, mirroring CUDA's build_rope_table (rope.cu):
/// inv[i] = theta^(-2i/n_rot), ang = pos * inv, cos/sin in float64, cast to
/// float. cos_tab / sin_tab are [max_pos][n_rot/2] host floats.
void cann_build_rope_table(int n_rot, double theta, int max_pos, float* cos_tab,
                           float* sin_tab);

/// NEOX partial RoPE in place on x [rows][heads*head_dim]: for each row and
/// head, the first n_rot dims rotate in (i, i + n_rot/2) pairs using that
/// row's cos/sin. cos_t / sin_t are [rows][n_rot/2] device floats (already
/// gathered for each row's position). The tail [n_rot:head_dim] is untouched.
/// scratch must hold rows*heads*head_dim + 4*rows*heads*(n_rot/2) floats.
bool cann_rope_neox(float* x, const float* cos_t, const float* sin_t, int rows, int heads,
                    int head_dim, int n_rot, aclrtStream stream, float* scratch,
                    std::string* err);

}  // namespace strata::cann
