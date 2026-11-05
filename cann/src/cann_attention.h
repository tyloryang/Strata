// cann/src/cann_attention.h - Phase 1: Gated DeltaNet attention on Ascend (CANN 9.0.0, aclnn).
//
// Transcribes the CUDA GDN path for Qwen3.6-35B-A3B (qwen35moe) into aclnn
// elementwise / reduction ops, FP32 throughout:
//
//   * gate / beta epilogue:  gate = softplus(alpha + dt) * ssm_a, beta = sigmoid(beta)
//     (src/qwen35/q36_engine.cu: k_gdn_ab / k_gdn_gb; src/kernels/cuda/fused_gdn.cu: gdn_ab_kernel)
//   * 4-tap causal conv + SiLU + per-head L2 norm of the q/k heads
//     (q36_engine.cu: k_conv_seq; fused_gdn.cu: gdn_conv_l2_kernel)
//   * gated delta step + SiLU-gated RMS norm, one token at a time, state carried across calls
//     (q36_engine.cu: k_gdn_seq / k_gdn_seq_cols / k_gdn_norm; fused_gdn.cu: gdn_step_norm_kernel)
//
// Per-token recurrence, per value head h (qh = h % h_k), S = 128:
//     g     = exp(gate[h])
//     kv[j] = sum_i S[h][i][j] * k[qh][i]            (S^T k, one column at a time)
//     delta[j] = (v[h][j] - g * kv[j]) * beta[h]
//     S[h][i][j] <- g * S[h][i][j] + k[qh][i] * delta[j]
//     o[j]  = sum_i S_new[h][i][j] * q[qh][i]        (S_new^T q)
//     o_raw = o / sqrt(S);  y = rmsnorm(o_raw) * gamma * silu(z)
//
// STATE LAYOUT. The CUDA kernels store the state as [row][head][col]
// (q36_engine.cu: state + ((row) * h_v + head) * S + col). This port stores the
// same logical tensor head-major as [head][row][col]: S[h][i][j] lives at
// h*S*S + i*S + j. Same size (h_v * S * S floats, cf. the per-layer allocation
// in q36_engine.cu ~line 1457), same values, different storage order; the state
// is private to a layer, so no other kernel observes the order. The test
// compares values by logical (head, row, col) index against a CPU reference
// kept in the CUDA [row][head][col] order, so a transposition bug is caught.
//
// The prompt path (T > 1) is a host-side loop over tokens launching the same
// per-token aclnn sequence as the decode path (T = 1); the recurrence is
// sequential across tokens, so chunking cannot change the result (the test
// checks a 16+16 chunked run against a single 32-token run). A fused AscendC
// kernel is the follow-up for speed; this phase is a correct foundation, not
// a fast one.
//
// All pointers are device pointers on the current ACL device. `stream` is the
// caller's aclrtStream. `scratch` is a caller-owned device buffer of at least
// gdn_scratch_floats(g, max_tokens) floats; functions fail cleanly (false +
// *err) instead of aborting. No CUDA/HIP code is touched: this lives in cann/
// only, behind -DSTRATA_ENABLE_CANN=ON.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "acl/acl_rt.h"

namespace strata::cann {

/// Geometry of one GDN layer. Defaults are Qwen3.6-35B-A3B as read from the
/// GGUF on the target box (see qwen36_geometry()).
struct GdnGeometry {
  int n_embd = 2048;     ///< embedding_length (qwen35moe.embedding_length)
  int state_size = 128;  ///< ssm.state_size
  int k_heads = 16;      ///< ssm.group_count
  int v_heads = 32;      ///< ssm.time_step_rank
  int d_conv = 4;        ///< ssm.conv_kernel
  int value_dim() const { return state_size * v_heads; }  ///< ssm.inner_size
  /// 2 * S * HK + HV * S: the qkv conv channels (q | k | v).
  int conv_channels() const { return 2 * state_size * k_heads + value_dim(); }
  size_t state_floats() const {
    return static_cast<size_t>(v_heads) * static_cast<size_t>(state_size) *
           static_cast<size_t>(state_size);
  }
  bool valid() const {
    return n_embd > 0 && state_size > 0 && k_heads > 0 && v_heads > 0 && d_conv > 0 &&
           (v_heads % k_heads) == 0;
  }
};

/// Qwen3.6-35B-A3B geometry, verified 2026-11-05 by parsing the GGUF metadata
/// of /data/strata-qwen36/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf on the target box:
/// embedding_length 2048, ssm.state_size 128, ssm.group_count 16,
/// ssm.time_step_rank 32, ssm.inner_size 4096, ssm.conv_kernel 4,
/// attention.head_count 16, head_dim 256, full_attention_interval 4,
/// expert_count 256, expert_used_count 8, block_count 41 (40 trunk + 1 MTP),
/// rms eps 1e-6, rope.freq_base 1e7. Note v_heads is 32 here, NOT the 48 of
/// Flash-Next's compiled defaults in include/strata/core/geometry.hpp.
GdnGeometry qwen36_geometry();

/// aclInit + aclrtSetDevice(device) + one stream. Returns false + *err on failure.
bool cann_init(int device, aclrtStream* stream, std::string* err);
/// Destroys the stream and calls aclFinalize().
void cann_shutdown(aclrtStream stream);

/// Device scratch (floats) covering every function below for up to `max_tokens`
/// tokens in one call: 2 state-sized temporaries for the step, 6 conv-channel
/// temporaries for the conv, 4 * max_tokens * v_heads for the gate/beta
/// epilogue, plus small head-sized buffers.
size_t gdn_scratch_floats(const GdnGeometry& g, int max_tokens);

/// Gate / beta epilogue, in place: alpha [T][HV] -> softplus(alpha + dt) * ssm_a,
/// beta [T][HV] -> sigmoid(beta). dt / ssm_a are [HV] device tensors.
/// softplus is computed in the overflow-safe form max(x,0) + log1p(exp(-|x|)),
/// mathematically identical to CUDA's `x > 20 ? x : log1p(exp(x))`.
bool cann_gdn_gate_beta(const GdnGeometry& g, float* alpha, float* beta, const float* dt,
                        const float* ssm_a, int tokens, aclrtStream stream, float* scratch,
                        std::string* err);

/// The conv history: the three previous qkv inputs, each [C] floats, oldest
/// first. Caller allocates three zero-filled [C] device buffers per layer at
/// session start; cann_gdn_conv rotates the pointers each token and copies
/// the current input into the newest slot, so the struct must be kept across
/// calls for the same layer. (CUDA: history [channel][3], q36_engine.cu
/// k_conv_seq; here three separate [C] buffers instead of interleaved [C][3],
/// same values.)
struct GdnConvState {
  float* h0 = nullptr;  ///< x[t-3]
  float* h1 = nullptr;  ///< x[t-2]
  float* h2 = nullptr;  ///< x[t-1]
};

/// 4-tap causal conv + SiLU + L2 norm of the q/k heads, for `tokens` tokens.
/// qkv [T][C] (raw qkv projection output), conv_w [C][4] (w[c*4+i], tap i,
/// oldest first, matching the GGUF ssm_conv1d.weight layout), h_out [T][C].
/// eps is the model's rms eps (1e-6 for Qwen3.6); the L2 norm is
/// y *= rsqrt(sum(y^2) + eps) per 128-channel q/k head, exactly as k_conv_seq.
bool cann_gdn_conv(const GdnGeometry& g, GdnConvState& st, const float* qkv,
                   const float* conv_w, float* h_out, int tokens, float eps, aclrtStream stream,
                   float* scratch, std::string* err);

/// Gated delta step + SiLU-gated RMS norm, for `tokens` tokens.
/// state [HV][S][S] head-major (see the layout note above), updated in place.
/// h [T][C] is the conv output (q | k | v), gate / beta [T][HV] are the
/// epilogued values from cann_gdn_gate_beta, z [T][HV*S] is the gate
/// projection output, gamma [S] is the ssm_norm weight, y [T][HV*S] is the
/// output. eps is the model's rms eps. The state is read, updated and written
/// back for every token (unlike the engine's verify path, which reads without
/// writing); there is no separate "verify" mode in this phase.
bool cann_gdn_step(const GdnGeometry& g, float* state, const float* h, const float* gate,
                   const float* beta, const float* z, const float* gamma, float eps, float* y,
                   int tokens, aclrtStream stream, float* scratch, std::string* err);

}  // namespace strata::cann
