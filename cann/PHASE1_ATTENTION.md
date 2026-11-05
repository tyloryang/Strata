# Phase 1 — CANN Gated DeltaNet attention + RMSNorm / SiLU / RoPE (aclnn)

Status: **verified on the target box** (Ascend 910B2C, CANN 9.0.0,
2026-11-05). `test_cann_attention` passes **all cases, exit 0**,
against an independent CPU FP32 reference transcribed from the CUDA
kernels. No CUDA comparison exists yet — a single-layer bit-close
against a CUDA build still needs a CUDA machine and is explicitly
still open.

## What was implemented

- `cann/src/cann_attention.h` / `cann/src/cann_attention.cpp` —
  `strata::cann::cann_gdn_gate_beta`, `cann_gdn_conv`,
  `cann_gdn_step`, plus `cann_init` / `cann_shutdown` and
  `gdn_scratch_floats`:
  - gate / beta epilogue: `gate = softplus(alpha + dt) * ssm_a`
    (overflow-safe form `max(x,0) + log1p(exp(-|x|))`),
    `beta = sigmoid(beta)`.
  - 4-tap causal conv + SiLU + per-head L2 norm of the q/k heads
    (`k_conv_seq` / `gdn_conv_l2_kernel`).
  - gated delta step + SiLU-gated RMS norm, one token at a time,
    state carried across calls (`k_gdn_seq` / `k_gdn_seq_cols` /
    `k_gdn_norm` / `gdn_step_norm_kernel`). The (S^T k) and
    (S_new^T q) contractions are Mul with a broadcast head vector
    followed by ReduceSum over the state-row axis — the same
    arithmetic in a different order, no matmul.
  - All FP32, all aclnn elementwise ops + ReduceSum, explicit
    broadcast shapes. No CUDA/HIP code is touched.
- `cann/src/cann_elementwise.h` / `cann/src/cann_elementwise.cpp` —
  `cann_rmsnorm`, `cann_silu`, `cann_build_rope_table` and the
  RoPE apply path used by the test.
- `cann/tests/test_cann_attention.cpp` — standalone on-device test:
  deterministic synthetic inputs (xorshift64star, fixed seed), CPU
  FP32 reference, per-case max abs / mean abs / max relative error,
  exit non-zero on any failure.
- `cann/CMakeLists.txt` — one delimited block at the end (libraries
  `strata_cann_attention` / `strata_cann_elementwise`, executable
  `test_cann_attention`). It finds two extra libs the probe did not
  need: `libopapi.so` (the elementwise aclnn ops) and
  `libnnopbase.so` (aclCreateTensor). Nothing above the block
  changed; no upstream file was touched.

## Method

The CPU reference keeps the GDN state in the CUDA storage order
`[row][head][col]` while the CANN code stores it head-major
`[head][row][col]`; the comparison maps logical (head, row, col)
indices, so a transposition bug fails loudly. Tolerances (FP32,
eps 1.2e-7; bars ~10-50x above the expected ~1e-5..1e-4
disagreement from summation-order differences): outputs 1e-3 abs /
2e-3 rel; carried state 5e-3 abs / 5e-3 rel; chunk identity 1e-6
(the two runs execute the identical per-token op sequence, so they
must agree bitwise). Relative error is `|a-b| / max(|ref|, 0.1)`.

## Results (target box, 910B2C, CANN 9.0.0, 2026-11-05)

| Case | max_abs | Verdict |
| --- | --- | --- |
| small gate (S=8, HK=2, HV=4, T=5) | 2.980e-08 | PASS |
| small beta | 5.960e-08 | PASS |
| small conv h | 1.192e-07 | PASS |
| small step y | 1.788e-07 | PASS |
| small step state | 1.863e-09 | PASS |
| small decode conv h (T=1, carried state) | 2.980e-08 | PASS |
| small decode y | 1.788e-07 | PASS |
| small decode state | 1.863e-09 | PASS |
| qwen36 gate (real geometry, T=32) | 1.907e-06 | PASS |
| qwen36 beta | 5.960e-08 | PASS |
| qwen36 conv h (T=32) | 8.941e-08 | PASS |
| qwen36 step y (T=32) | 4.172e-07 | PASS |
| qwen36 step state (T=32) | 1.118e-08 | PASS |
| qwen36 decode conv h (T=1) | 7.451e-08 | PASS |
| qwen36 decode y (T=1) | 3.576e-07 | PASS |
| qwen36 decode state | 1.118e-08 | PASS |
| chunk identity y (16+16 vs 32) | 0.000e+00 (bitwise) | PASS |
| chunk identity state (16+16 vs 32) | 0.000e+00 (bitwise) | PASS |
| chunk run A y vs ref | 4.172e-07 | PASS |
| chunk run A state vs ref | 1.118e-08 | PASS |
| rmsnorm y (4x2048) | 9.537e-07 | PASS |
| silu y (n=100000) | 4.768e-07 | PASS |
| rope y (4x16x256, n_rot=64) | 0.000e+00 | PASS |

`qwen36 conv+step T=32 wall: 9.0 ms` (informational only, not a
benchmark — one wall time for the whole prompt loop, no timing
methodology behind it).

## Bugs found during verification (all fixed here, all measured)

1. **Destroying an executed aclOpExecutor segfaults the next
   different op's GetWorkspaceSize** (CANN 9.0.0, 910B2C). Minimal
   repros: Add->Abs, Add->Neg, Add->Exp, Abs->Add all segfault at
   the second op's GetWorkspaceSize (exit 139); same-op-after-destroy
   and getws-without-exec-after-destroy pass; a stream sync before
   the destroy does not help; destroying only the aclScalar is
   harmless; NOT destroying the executor makes everything pass, and
   the next op's GetWorkspaceSize returns the *same* executor
   pointer — the executor is a runtime-shared object. This is why
   the GEMM-only test never saw it (Matmul throughout). Fix: `run_op`
   never destroys the executor (leaked for the life of the process,
   documented in the code); revisit if a future CANN version
   documents a safe destroy discipline. The same latent bug exists
   in `cann_gemm.cpp`'s copy of `run_op` and is fixed separately
   when the branches merge (it works today only because GEMM uses a
   single op type).
2. **ReduceSum output must match the reduced rank.** Reducing the 4D
   `{HK,R,S,S}` contraction over dim 2 yields `{HK,R,S}`; passing a
   `{HV,S}` view (same element count, different rank) is rejected
   with 161002 (`ACLNN_ERR_PARAM_INVALID`). Fix: 3D `{R,HK,S}` views
   (`v_kv3`, `v_o3`) over the same buffers for the two ReduceSum
   outputs in `cann_gdn_step`.
3. **Value-head to q/k-head mapping is interleaved, not blocked.**
   The step first stored the state as `[HK][R][S][S]`, pairing value
   head `hd` with q head `hd / R`. CUDA (`k_gdn_seq`,
   `k_gdn_seq_cols` in `src/qwen35/q36_engine.cu`) uses
   `const int qh = head % h_k`, and the test reference transcribes
   that. The mismatch made every step case fail (y max_abs ~1.6,
   state ~0.03) while chunk identity still passed bitwise. Fix: the
   state views are now `[R][HK][S][S]`, so flat head `hd = r*HK+qh`
   and `qh = hd % HK`; the flat head order is unchanged, so the
   test's `cann_state_to_logical` still applies.
4. **Test-harness race: synchronous aclrtMemcpy does not wait for a
   custom stream.** The test's `DevBuf::d2h` read `d_h` right after
   the conv and got max_abs 0.77; after an explicit
   `aclrtSynchronizeStream` the same buffer read 1.2e-07. Fix:
   `run_case` now synchronizes the stream on success before any
   report reads a device buffer. (Library functions stay async on
   the caller's stream, as documented; the caller syncs.)

## Not done (explicitly)

- **Full attention is NOT in this scope**: QK^T + softmax + PV,
  KV cache and QSA (every-4th-layer full attention) are the next
  step; this phase covers GDN + RMSNorm / SiLU / RoPE only.
- Single-layer bit-close against a CUDA build (needs a CUDA
  machine; the CPU reference here is transcribed from the CUDA
  kernels, not a CUDA-binary comparison).
- No BF16 path, no real timing/benchmark, prompt max T=32 + 1
  decode step in the test, no verify mode, no invalid-arg tests.
  The RoPE test is self-referential (it uses
  `cann_build_rope_table` to build its own reference cos/sin — only
  the rotation arithmetic is independently checked). The test fill
  ranges do not reach the commented real extremes for `ssm_a` /
  `dt` (recorded in the test file's limits, not re-tested here).

## Merge note

Merging this branch into `ascend-cann` conflicts with `cann-gemm`
in exactly one place, `cann/CMakeLists.txt`: both branches define
`STRATA_CANN_OPAPI_LIB` but find different libraries (`opapi_nn`
vs `opapi`); `find_library` caching means one block's result wins
and the other target links the wrong `.so`. At merge: rename to
`STRATA_CANN_OPAPI_LIB` -> `opapi` (attention) and a new
`STRATA_CANN_OPAPI_NN_LIB` -> `opapi_nn` (GEMM); find
`STRATA_CANN_NNOPBASE_LIB` (nnopbase, same both sides) once;
`strata_cann_gemm` links ascendcl + opapi_nn + nnopbase,
`strata_cann_attention` links ascendcl + opapi + nnopbase.
Post-merge, re-run both `test_cann_gemm` (regression) and
`test_cann_attention` on the box.
