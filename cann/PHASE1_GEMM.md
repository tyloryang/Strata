# Phase 1 — CANN matmul (aclnn) for Qwen3.6

Status: **verified on the target box** (Ascend 910B2C, CANN 9.0.0,
2026-11-05). `test_cann_gemm` passes all 32 shapes against an
independent CPU FP32 reference, twice (two full runs, same verdict;
per-shape numbers agree to ~1e-6). No CUDA comparison exists yet —
that needs a CUDA machine and is explicitly still open.

## What was implemented

- `cann/src/cann_gemm.h` / `cann/src/cann_gemm.cpp` —
  `strata::cann::CannGemm`, a thin wrapper over `aclnnMatmul`:
  `aclInit` + `aclrtSetDevice` + one stream in `init()`; `gemm()`
  computes `C[M, N] = A[M, K] . B[K, N]` (plain) or
  `C[M, N] = A[M, K] . B^T` with `B` stored row-major `[N, K]` (the
  engine's `W` layout, expressed to aclnn as a strided `[K, N]` view:
  dims `{K, N}`, strides `{1, K}`). Dtypes FP16/FP32 in, FP16/FP32
  out; `cubeMathType` is a caller parameter. One call = one
  `aclnnMatmulGetWorkspaceSize` + one `aclnnMatmul` + one stream sync,
  with a freshly allocated workspace each time (correct first, fast
  later). All pointers are caller-owned device pointers.
- `cann/tests/test_cann_gemm.cpp` — standalone on-device test:
  deterministic synthetic inputs, CPU FP32 reference, per-shape
  max abs / mean abs / max relative error (`||d||_inf / ||ref||_inf`),
  exit non-zero on any failure. The reference pre-transposes `B`
  once and runs a contiguous NN kernel (a direct strided NT loop
  re-reads `B` M times and dominated the first run).
- `cann/CMakeLists.txt` — one delimited block at the end (library
  `strata_cann_gemm`, executable `test_cann_gemm`). It finds two
  extra libs the probe did not need: `libopapi_nn.so` (aclnnMatmul)
  and `libnnopbase.so` (aclCreateTensor). Nothing above the block
  changed; no upstream file was touched.

## Shapes covered (read from the engine, verified against the GGUF)

Geometry from `/data/strata-qwen36/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`
on the box (parsed, not assumed): `n_embd` 2048, `n_head` 16,
`n_head_kv` 2, `head_dim` 256, `n_ff` 512, `n_expert` 256, `top_k` 8,
`ssm_state_size` 128, `ssm_k_heads` 16, `ssm_v_heads` 32,
`ssm_d_conv` 4, `ssm_value_dim` 4096, `ssm_conv_channels` 8192,
`n_layers` 41, vocab 248320. Full-attention layers are
`layer % 4 == 3` (`is_qsa_layer`). Tensor shapes in the GGUF are
`[n_in, n_out]` (the engine's `mat()` reads `shape[0]`/`shape[1]`
that way); the engine computes `y[T][n_out] = W[n_out][n_in] .
x[T][n_in]`. Source: `src/qwen35/q36_engine.cu` (`mm`/`f32mm`,
`kGemmT = 8`) + `include/strata/core/family_layout.hpp`.

| Use | K (n_in) | N (n_out) | T tested | dtype / cube |
| --- | --- | --- | --- | --- |
| attn_qkv (GDN) | 2048 | 8192 | 1, 2, 8, 512, 1024, 4096 | fp16->fp32, 1 |
| attn_gate (GDN) | 2048 | 4096 | 512 | fp16->fp32, 1 |
| ssm_out | 4096 | 2048 | 512 | fp16->fp32, 1 |
| attn_q (full attn) | 2048 | 8192 | 512 | fp16->fp32, 1 |
| attn_k / attn_v | 2048 | 512 | 512 | fp16->fp32, 1 |
| attn_output | 4096 | 2048 | 512 | fp16->fp32, 1 |
| sh_gate / sh_up (= expert gate/up per expert) | 2048 | 512 | 8, 512 | fp16->fp32, 1 |
| sh_down (= expert down per expert) | 512 | 2048 | 512 | fp16->fp32, 1 |
| lm head (`output.weight`) | 2048 | 248320 | 1, 8, 512 | fp16->fp32, 1 |
| eh_proj (MTP) | 4096 | 2048 | 512 | fp16->fp32, 1 |
| qkv, plain NN (B stored [K, N]) | 2048 | 8192 | 512 | fp16->fp32, 1 |
| qkv, FP16 out | 2048 | 8192 | 8 | fp16->fp16, 1 |
| router | 2048 | 256 | 1, 8, 512, 1024 | fp32->fp32, 0 |
| router_ext (shared-expert row appended) | 2048 | 257 | 8 | fp32->fp32, 0 |
| ssm_alpha / ssm_beta | 2048 | 32 | 512 | fp32->fp32, 0 |
| sh_gate_inp | 2048 | 1 | 8 | fp32->fp32, 0 |
| router / alpha-beta, cube=1 fallback | 2048 | 256 / 32 | 8, 512 | fp32->fp32, 1 |

Not covered here (other phases): the batched attention score/value
GEMMs (`cublasGemmStridedBatchedEx`, FP16, `P x nt x HD`), GGUF
dequant (the weights here are synthetic FP16, not dequantized Q8_0 /
Q4_K / Q5_K), MoE top-k dispatch, BF16.

## Build and test (exact commands, on the box)

Box: `<服务器地址>:32222` (SSHPiper, password auth via the
expect helper; direct TCP is reset by the egress proxy). Build dir
on the box: `/data/cann-phase1/gemm` (884 KiB total; `/data` still
34 GiB free afterwards). The running llama-server was not touched.

```sh
# transfer (scp is blocked by SSHPiper; base64 over SSH instead)
#   local: tar czf cann-gemm.tar.gz cann/
#   remote: base64 -d > cann-gemm.tar.gz && tar xzf cann-gemm.tar.gz
cd /data/cann-phase1/gemm
source /usr/local/Ascend/cann-9.0.0/set_env.sh
cmake -S cann -B build-cann -G 'Unix Makefiles' -DCMAKE_BUILD_TYPE=Release
cmake --build build-cann -j 8        # exit 0; see build notes below
./build-cann/strata_cann_probe       # exit 0: NPU count=1, current device=0
./build-cann/test_cann_gemm          # exit 0: ALL PASS (0 failures)
```

Full output of the verified run (2026-11-05 ~23:37 CST; a second
full run afterwards gave the same verdict and per-shape numbers
agreeing to ~1e-6 — e.g. attn_q max_abs 0.0001526 vs 0.0001602):

```
test_cann_gemm: CANN 9.0.0 aclnnMatmul vs CPU FP32 reference
test_cann_gemm: device=0, NPU count=1
qkv T=1            M=1     N=8192   K=2048  fp16->fp32 NT cube=1  2405.01 ms  max|ref|=59.36 max_abs=9.155e-05 mean_abs=8.822e-06 max_rel=1.542e-06 tol=1e-03  PASS
qkv T=2            M=2     N=8192   K=2048  fp16->fp32 NT cube=1     0.39 ms  max|ref|=56.11 max_abs=0.000103 mean_abs=8.678e-06 max_rel=1.836e-06 tol=1e-03  PASS
qkv T=8            M=8     N=8192   K=2048  fp16->fp32 NT cube=1     0.42 ms  max|ref|=65.27 max_abs=0.0001221 mean_abs=8.65e-06 max_rel=1.87e-06 tol=1e-03  PASS
qkv T=512          M=512   N=8192   K=2048  fp16->fp32 NT cube=1     0.48 ms  max|ref|=80.71 max_abs=0.0001526 mean_abs=8.6e-06 max_rel=1.891e-06 tol=1e-03  PASS
qkv T=1024         M=1024  N=8192   K=2048  fp16->fp32 NT cube=1   298.92 ms  max|ref|=82.3 max_abs=0.0001564 mean_abs=8.733e-06 max_rel=1.9e-06 tol=1e-03  PASS
qkv T=4096         M=4096  N=8192   K=2048  fp16->fp32 NT cube=1     1.19 ms  max|ref|=83.34 max_abs=0.000164 mean_abs=8.734e-06 max_rel=1.968e-06 tol=1e-03  PASS
attn_gate T=512    M=512   N=4096   K=2048  fp16->fp32 NT cube=1     0.57 ms  max|ref|=77.89 max_abs=0.0001526 mean_abs=8.59e-06 max_rel=1.959e-06 tol=1e-03  PASS
ssm_out T=512      M=512   N=2048   K=4096  fp16->fp32 NT cube=1    87.40 ms  max|ref|=111.1 max_abs=0.000267 mean_abs=1.719e-05 max_rel=2.404e-06 tol=1e-03  PASS
attn_q T=512       M=512   N=8192   K=2048  fp16->fp32 NT cube=1     0.32 ms  max|ref|=80.71 max_abs=0.0001526 mean_abs=8.6e-06 max_rel=1.891e-06 tol=1e-03  PASS
attn_k T=512       M=512   N=512    K=2048  fp16->fp32 NT cube=1     0.46 ms  max|ref|=71.09 max_abs=0.0001259 mean_abs=8.601e-06 max_rel=1.771e-06 tol=1e-03  PASS
attn_v T=512       M=512   N=512    K=2048  fp16->fp32 NT cube=1     0.17 ms  max|ref|=71.09 max_abs=0.0001259 mean_abs=8.602e-06 max_rel=1.771e-06 tol=1e-03  PASS
attn_out T=512     M=512   N=2048   K=4096  fp16->fp32 NT cube=1     0.14 ms  max|ref|=111.1 max_abs=0.000267 mean_abs=1.719e-05 max_rel=2.404e-06 tol=1e-03  PASS
sh_gate T=8        M=8     N=512    K=2048  fp16->fp32 NT cube=1     0.42 ms  max|ref|=52.75 max_abs=7.629e-05 mean_abs=8.64e-06 max_rel=1.446e-06 tol=1e-03  PASS
sh_gate T=512      M=512   N=512    K=2048  fp16->fp32 NT cube=1     0.15 ms  max|ref|=71.09 max_abs=0.0001259 mean_abs=8.601e-06 max_rel=1.771e-06 tol=1e-03  PASS
sh_down T=512      M=512   N=2048   K=512   fp16->fp32 NT cube=1     0.56 ms  max|ref|=37.85 max_abs=3.052e-05 mean_abs=2.134e-06 max_rel=8.062e-07 tol=1e-03  PASS
expert_gate T=8    M=8     N=512    K=2048  fp16->fp32 NT cube=1     0.10 ms  max|ref|=52.75 max_abs=8.011e-05 mean_abs=8.644e-06 max_rel=1.519e-06 tol=1e-03  PASS
head T=1           M=1     N=248320 K=2048  fp16->fp32 NT cube=1     1.40 ms  max|ref|=82.57 max_abs=0.0001526 mean_abs=8.668e-06 max_rel=1.848e-06 tol=1e-03  PASS
head T=8           M=8     N=248320 K=2048  fp16->fp32 NT cube=1    76.90 ms  max|ref|=77.81 max_abs=0.0001526 mean_abs=8.717e-06 max_rel=1.961e-06 tol=1e-03  PASS
head T=512         M=512   N=248320 K=2048  fp16->fp32 NT cube=1     4.35 ms  max|ref|=90.41 max_abs=0.0001907 mean_abs=8.602e-06 max_rel=2.11e-06 tol=1e-03  PASS
eh_proj T=512      M=512   N=2048   K=4096  fp16->fp32 NT cube=1     0.22 ms  max|ref|=111.1 max_abs=0.0002594 mean_abs=1.719e-05 max_rel=2.335e-06 tol=1e-03  PASS
qkv-nn T=512       M=512   N=8192   K=2048  fp16->fp32 NN cube=1    96.40 ms  max|ref|=75.17 max_abs=0.0001869 mean_abs=8.599e-06 max_rel=2.487e-06 tol=1e-03  PASS
qkv-fp16out T=8    M=8     N=8192   K=2048  fp16->fp16 NT cube=1    99.51 ms  max|ref|=65.27 max_abs=0.02235 mean_abs=0.002114 max_rel=0.0003424 tol=1e-02  PASS
router T=1         M=1     N=256    K=2048  fp32->fp32 NT cube=0     3.51 ms  max|ref|=41.83 max_abs=4.005e-05 mean_abs=8.776e-06 max_rel=9.575e-07 tol=1e-03  PASS
router T=8         M=8     N=256    K=2048  fp32->fp32 NT cube=0     0.35 ms  max|ref|=62.47 max_abs=8.392e-05 mean_abs=1.003e-05 max_rel=1.343e-06 tol=1e-03  PASS
router T=512       M=512   N=256    K=2048  fp32->fp32 NT cube=0     0.42 ms  max|ref|=63.83 max_abs=0.0001526 mean_abs=9.695e-06 max_rel=2.39e-06 tol=1e-03  PASS
router T=1024      M=1024  N=256    K=2048  fp32->fp32 NT cube=0     0.47 ms  max|ref|=71.54 max_abs=0.0001602 mean_abs=9.719e-06 max_rel=2.24e-06 tol=1e-03  PASS
router_ext T=8     M=8     N=257    K=2048  fp32->fp32 NT cube=0     0.37 ms  max|ref|=53.73 max_abs=9.155e-05 mean_abs=9.904e-06 max_rel=1.704e-06 tol=1e-03  PASS
alpha/beta T=512   M=512   N=32     K=2048  fp32->fp32 NT cube=0     0.43 ms  max|ref|=60.5 max_abs=0.0001202 mean_abs=9.572e-06 max_rel=1.986e-06 tol=1e-03  PASS
sh_gate_inp T=8    M=8     N=1      K=2048  fp32->fp32 NT cube=0     0.35 ms  max|ref|=21.02 max_abs=1.383e-05 mean_abs=6.372e-06 max_rel=6.577e-07 tol=1e-03  PASS
router cube1 T=8   M=8     N=256    K=2048  fp32->fp32 NT cube=1     0.41 ms  max|ref|=62.47 max_abs=0.007587 mean_abs=0.001546 max_rel=0.0001215 tol=2e-02  PASS
router cube1 T=512 M=512   N=256    K=2048  fp32->fp32 NT cube=1     0.44 ms  max|ref|=63.83 max_abs=0.009649 mean_abs=0.001567 max_rel=0.0001512 tol=2e-02  PASS
alpha/beta c1 T=512 M=512   N=32     K=2048  fp32->fp32 NT cube=1     0.37 ms  max|ref|=60.5 max_abs=0.007477 mean_abs=0.001576 max_rel=0.0001236 tol=2e-02  PASS
test_cann_gemm: ALL PASS (0 failures)
```

Summary of the numbers: FP16->FP32 max rel err 8.1e-07–2.5e-06
(tol 1e-3); FP32 cube=0 max rel err 6.6e-07–2.4e-06 (tol 1e-3);
FP32 cube=1 max rel err 1.2e-04–1.5e-04 (tol 2e-2); FP16 out
3.4e-04 (tol 1e-2). Inputs are synthetic uniform [-1, 1]; the
reference is a CPU FP32 triple loop (OpenMP, `-march=native` for
the test binary only), so the residual is accumulation-order noise
plus FP16 input rounding where applicable — not a model result.

## Findings that change the plan

1. **FP32 + `cubeMathType=0` (KEEP_DTYPE) works for aclnnMatmul
   on this box — true FP32, ~1e-6, and no slower than cube=1
   (0.35–0.47 ms vs 0.37–0.44 ms on the router shapes).** The
   "FP32 + KEEP_DTYPE is rejected on Ascend 910" note in
   `aclnnop/aclnn_mv.h` concerns the Mv op, not Matmul. The
   router / alpha-beta path should therefore use cube=0, not the
   down-precision cube=1 path the plan assumed. With cube=1 the
   FP32 inputs are converted to FP16 in the cube: max abs err
   ~0.0076–0.0096 at output scale ~60 (scale-relative 1.2e-4–
   1.5e-4), i.e. the FP16 input-rounding floor. The first version
   of this test failed all 7 FP32 shapes under a per-element
   relative metric because near-zero outputs report rel errors up
   to 0.11 at that absolute floor; the metric was changed to the
   scale-relative `||d||_inf / ||ref||_inf` (standard GEMM check),
   with max abs / mean abs printed alongside — the FP32 cube=1
   error is real but it is an absolute floor, not a relative one.
2. **Timings are single-call wall times, not benchmarks.** Most
   steady-state calls are 0.1–1.5 ms, but the first call of a run
   took 0.5–2.4 s and a few first-uses of a shape class spiked
   (qkv T=1024: 75–299 ms, ssm_out T=512: 87 ms, head T=8:
   77 ms, qkv-nn T=512: 96 ms, qkv-fp16out T=8: 99 ms) — most
   plausibly one-off tiling/autotune or cold-cache costs. Do not
   quote the ms column as throughput. The NN (non-transposed) path
   and the FP16-out path were both slower than NT FP32-out on the
   same shape (3.95/96.4 ms vs 0.5 ms; 9.4/99.5 ms vs 0.4 ms) —
   also single samples, worth a proper timing pass later, not a
   conclusion now.
3. **API facts found on the box** (headers checked, not guessed):
   `aclnnMatmul` lives in `include/aclnnop/aclnn_matmul.h`,
   symbols in `libopapi_nn.so`; `aclCreateTensor` /
   `aclDestroyTensor` in `libnnopbase.so`; `ACLNN_SUCCESS` is
   defined in `aclnn/opdev/op_errno.h`, which `aclnn_matmul.h`
   does not pull in; CANN 9.0.0 exposes no executor-destroy API
   (official samples destroy only the tensors). `aclnnMatmul`
   has no transpose flag — transposition is expressed through the
   tensor strides. `aclrtMalloc` takes
   `ACL_MEM_MALLOC_HUGE_FIRST` here.

## Build notes (failures seen and fixed)

- `ACLNN_SUCCESS was not declared` — fixed by including
  `aclnn/opdev/op_errno.h`.
- `undefined reference to aclCreateTensor` / `DSO missing` —
  fixed by linking `libnnopbase.so`.
- Test target needed the CANN include dir (it includes
  `acl/acl_rt.h` directly).
- The CPU reference's first NT loop was cache-hostile (strided
  `B` re-read M times); the test now pre-transposes `B` once.
  The reference is still the runtime bottleneck (the head T=512
  reference is ~500 GFLOP on CPU); the NPU calls themselves are
  sub-ms to a few ms.
- SSH notes for the next worker: `/tmp/ssh_expect.exp` was wiped
  once mid-task (VM replacement) and had to be recreated from the
  task text; the helper's `set timeout 45` kills long silent
  commands — run long tests detached (`nohup … &`) and poll the
  log file instead of waiting on the SSH session.

## Still open

- GGUF dequant (Q8_0 / Q4_K / Q5_K -> FP16 device buffers) so
  the GEMM can consume real weights; then a single Qwen3.6 layer
  vs the CUDA build (the plan's bit-close milestone — needs a
  CUDA machine, not done here).
- Wiring `CannGemm` into the engine (a later phase; this is a
  standalone library + test, the engine still builds/runs on
  CUDA only and nothing upstream changed).
- Workspace / executor reuse across calls (currently allocated
  per call), a proper timing pass (steady-state, warmed up,
  interleaved), BF16, the batched attention score/value GEMMs,
  and 910C (2 dies) — none verified here.
- No torch-npu reference was needed: the CPU reference was
  sufficient and aclnn worked first try once linked correctly.
