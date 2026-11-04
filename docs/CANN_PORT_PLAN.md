# Strata on Ascend (910B / 910C) — CANN port plan

Branch: `ascend-cann` — fork of Niko1221/Strata, base `ffa80d73` (2026-11-02).
Goal: run Qwen3.8-Flash-Next (IQ3_XXS and friends) on Huawei Ascend 910B/910C
via a native CANN backend. Upstream Strata has no Ascend backend today
(CUDA / HIP / SYCL / Metal only).

## 0. Ground rules

- Do not break the existing backends. The CANN backend lives in its own
  directory (`cann/`, mirroring `sycl/` and `metal/`) behind a build flag
  (e.g. `./setup.sh --backend cann`), same pattern as `--backend sycl`.
- One variable at a time: bring up in the phase order below, bench after
  each phase against the A100 baseline (IQ3_XXS, 262K, same prompts).
- Atlas servers are Kunpeng aarch64. Everything (C++ engine, Python
  server, tokenizer) must build and run on aarch64; no x86/AVX2
  assumptions in the new code.
- Upstream rules (this repo's AGENTS.md / CONTRIBUTING.md): agree the
  scope in a GitHub issue before a large feature; PR titles start with
  an area (`[Engine]`, `[Docs]`, …); one problem per PR; the new
  backend is opt-in and the default path stays byte-identical to the
  last release; build every backend a touched file affects (CUDA, HIP,
  SYCL) before review; never submit measurements you did not produce —
  mark code-reading findings as unconfirmed static analysis.
- Read first: `docs/AI_REFERENCE.md` (models/APIs/hardware limits),
  `docs/DETAILS.md` (how the engine works), `docs/AMD_HIP.md` (template
  for a new-backend doc — a future `docs/CANN.md` follows it),
  `docs/MULTI_GPU.md`.

## 1. Toolchain & build

- CANN toolkit version to pin: TBD (check `npu-smi info` / CANN version
  on the target machine first; 910B/910C need CANN 8.x).
- CMake: add a `STRATA_BACKEND=cann` option; link `libascendcl` /
  `aclnn`; add a `cann/` target next to `sycl/`.
- Get `strata --help` / engine self-test to run on the NPU host with no
  model loaded before touching any kernel.

## 2. Core compute (dense path first)

- GEMM / matmul via aclnn (or `torch-npu` as a shortcut for the first
  bring-up, then move hot paths to aclnn).
- Attention incl. Gated DeltaNet (GDN) recurrence — Strata already ships
  CUDA GDN kernels; these need Ascend equivalents (chunked recurrence,
  same numerics, same state layout).
- RMSNorm / SiLU / rotary embedding / sampling.
- Milestone: a tiny dense model (or a single Qwen3.8-27B layer) produces
  bit-close logits vs the CUDA build.

## 3. GGUF quants

- Dequant kernels for IQ3_XXS / IQ2_XS / Q4_K / Q6_K (whatever the target
  model uses). Start with the exact quant the bench model uses (IQ3_XXS),
  add others later.
- Verify dequant output against the CUDA build element-wise before
  wiring it into decode.

## 4. MoE expert routing & PLE streaming

- Router / top-k gating, expert dispatch.
- PLE (the shared-expert weights, ~28.8 GB second shard) streaming:
  RAM/disk -> NPU, same role as on CUDA. Host RAM size and disk speed
  matter as much as on the A100 box.
- Expert cache: same slot accounting as CUDA (`--expert-cache`), sized
  to the card (910B 64 GiB, 910C 2 dies x 64 GiB).

## 5. KV cache, prefix cache, MTP

- KV cache ops on device; prefix/conversation cache same semantics.
- MTP speculative draft verification (the draft layer + accept logic).
- `spec` / `spec-min-p` behaviour must match CUDA bit-for-bit on the
  same prompts, or the acceptance-rate comparison is meaningless.

## 6. Multi-device

- 910B: 1 die per card. 910C: 2 dies per board, seen as 2 devices;
  layer-split must handle the 2-die case (HCCS between dies).
- `--layer-split` semantics carry over; verify with 2 devices before
  claiming multi-card works.

## 7. Bench & accept

- Same protocol as the A100 rounds: evalscope c2000 (prompt 2000,
  out 100, conc 1/2/3/4/6/8, thinking off), then the long-prompt
  distribution (median 72K, p95 177K).
- Noise floor first: repeat the baseline on the Ascend box before
  judging any optimisation (do not reuse the A100/3080 thresholds).
- Ship criteria: same model, same prompts, correct answers; then
  speed.

## Open questions (fill in before phase 1)

- Exact CANN version and driver on the target machine.
- Target machine: Atlas 800I A2 (910B) or A3 (910C)? How many cards/dies?
- Does the box have enough host RAM + fast disk for the PLE streaming
  path (Strata loads 35–55 GB into RAM on CUDA boxes)?
