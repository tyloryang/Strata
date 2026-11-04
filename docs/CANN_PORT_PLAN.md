# Strata on Ascend (910B / 910C) — CANN port plan

Branch: `ascend-cann` — fork of Niko1221/Strata, base `ffa80d73` (2026-11-02).
Goal: run Qwen3.8-Flash-Next (IQ3_XXS and friends) on Huawei Ascend 910B/910C
via a native CANN backend. Upstream Strata has no Ascend backend today
(CUDA / HIP / SYCL / Metal only).

## 0. Ground rules

- Do not break the existing backends. The CANN backend lives in its own
  directory (`cann/`, mirroring `sycl/` and `metal/`) behind a build flag
  (`-DSTRATA_ENABLE_CANN=ON`, same pattern as `STRATA_ENABLE_SYCL`:
  a separate backend, mutually exclusive with CUDA/HIP/SYCL, and the
  top-level build returns after `add_subdirectory(cann)`).
  The default build (no flag) is unchanged.
- One variable at a time: bring up in the phase order below, bench after
  each phase against the A100 baseline (IQ3_XXS, 262K, same prompts).
- Do NOT assume Kunpeng aarch64 for every Atlas box. The first verified
  target (2026-11-04, see §8) is an x86_64 openEuler container host
  with a 910B2C on PCIe. New code must build on x86_64 first and must
  not bake in x86/AVX2 or aarch64 assumptions either way; check
  `uname -m` on each new target before claiming aarch64.
- Upstream rules (this repo's AGENTS.md / CONTRIBUTING.md): agree the
  scope in a GitHub issue before a large feature; PR titles start with
  an area (`[Engine]`, `[Docs]`, …); one problem per PR; the new
  backend is opt-in and the default path stays byte-identical to the
  last release; build every backend a touched file affects (CUDA, HIP,
  SYCL) before review; never submit measurements you did not produce —
  mark code-reading findings as unconfirmed static analysis.
- Read first: `docs/AI_REFERENCE.md` (models/APIs/hardware limits),
  `docs/DETAILS.md` (how the engine works), `docs/AMD_HIP.md` (template
  for a new-backend doc — `docs/CANN.md` follows it),
  `docs/MULTI_GPU.md`.
- Backend survey (2026-11-04, code reading, unconfirmed static
  analysis unless noted): the top-level CMake has three mutually
  exclusive backends — `STRATA_ENABLE_CUDA`, `STRATA_ENABLE_HIP`,
  `STRATA_ENABLE_SYCL` (the SYCL port in `sycl/` is its own project
  that never edits upstream files and migrates `.cu` to `.dp.cpp`
  via `sycl/migrate.sh`; the CANN port follows that pattern).
  The engine core is CUDA: 100 `.cu` files, 49 in `src/kernels/cuda`,
  plus `src/core/device.cu` (`cudaMalloc` etc.) and the Qwen3.6 path
  in `src/qwen35` (`q36_engine.cu`). There is no generic device
  abstraction to plug a backend into — the port is a per-kernel
  migration, not a dispatch-table swap. A HIP-style compat layer
  (mapping CUDA runtime calls to ACL, as `include/strata/platform/
  hip_compat/` does for HIP) can cover host-side runtime calls only;
  the custom kernels (fused GDN, IQ dequant, MoE routing) still need
  Ascend equivalents.

## 1. Toolchain & build

- CANN toolkit version to pin: **9.0.0 on the first verified target**
  (see §8; the earlier "CANN 8.x" guess is superseded). Do not pin
  8.x without a machine that actually runs it.
- CMake: `STRATA_ENABLE_CANN` (done in this branch, Phase 0);
  link `libascendcl` / `libacl_rt`; add a `cann/` target next to
  `sycl/`. First bring-up shortcut is `torch-npu` (already installed
  on the target, 2.10.0, see §8) for a Python-level smoke test, then
  move hot paths to aclnn / AscendC.
- Phase 0 milestone (done and verified on the target, 2026-11-04):
  `cann/` builds a `strata_cann_probe` binary that calls only
  verified ACL runtime APIs (`aclInit`, `aclrtGetDeviceCount`,
  `aclrtSetDevice`, `aclrtGetDevice`, `aclFinalize` — signatures
  checked against `/usr/local/Ascend/cann-9.0.0/include/acl/acl_rt.h`
  on the target) and prints the visible NPU count. No model loaded,
  no kernel yet. Verified run on the 910B2C box: compiled with
  `g++ -std=c++20 -I/usr/local/Ascend/cann-9.0.0/include
  -L/usr/local/Ascend/cann-9.0.0/lib64 -lascendcl`, exit 0,
  output `CANN runtime ok, NPU count=1` and `current device=0`.
  Note: the runtime sees 1 device (id 0) even though the host
  device is `/dev/davinci14` / NPU 14 — inside the container the
  visible device is 0, which settles the
  `ASCEND_RT_VISIBLE_DEVICES` mapping question for this box.
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

## 8. First verified target (2026-11-04, measured on the machine, not docs)

- Host: <服务器地址>:32222 via SSHPiper, openEuler 24.03 LTS-SP2,
  kernel 5.15, **x86_64** (not Kunpeng aarch64), container hostname
  9167159da7a3, 128 vCPUs (reported), 64 GiB host RAM, `/` 30 GiB,
  `/data` 59 GiB (56 GiB free), NFS `/mnt/moark-models` 3.5 TiB
  (763 GiB free), no Docker, no NVIDIA driver, Jupyter on 3000.
- NPU: **Ascend 910B2C, one device visible as `/dev/davinci14`**,
  NPU id 14, bus 0000:BE:00.0, HBM 64 GiB (65536 MiB, ~3.4 GiB used
  idle), `npu-smi` 26.0.rc1, no running NPU process at check time.
  910C (2 dies per board) is NOT on this box — multi-die work in §6
  is unvalidated until a 910C box appears.
- CANN: **9.0.0** at `/usr/local/Ascend/cann-9.0.0` (`ASCEND_HOME_PATH`,
  `ASCEND_TOOLKIT_HOME`, `set_env.sh` present), headers
  `include/acl/acl.h`, `include/acl/acl_rt.h`, `include/aclnn/`
  present, `libascendcl.so` / `libacl_rt.so` present.
  Toolchain: gcc/g++ 12.3.1, CMake 3.27.9, **no ninja, no rustc/cargo**
  (install ninja before building; the engine is C++, Rust is not
  needed for it). Python 3.11.15 at
  `/usr/local/python3.11.15/bin/python3`, torch 2.10.0+cpu,
  **torch_npu 2.10.0**.
- Models already on the box: `/mnt/moark-models/Qwen3-0.6B` (2.9 GiB)
  and **`/mnt/moark-models/Qwen3.6-35B-A3B` (134 GiB BF16, 26
  safetensors shards)** — `Qwen3_5MoeForConditionalGeneration`,
  hidden 2048, head_dim 256, `full_attention_interval` 4, mostly
  linear-attention (Gated DeltaNet) layers with full attention every
  4th layer. This is Strata's `qwen36` family (`src/qwen35`,
  `docs/QWEN36.md`), so the first bring-up model is Qwen3.6-35B-A3B,
  not Flash-Next — but note the on-disk weights are safetensors BF16,
  while Strata's qwen36 path runs Unsloth GGUF
  (UD-Q4_K_XL / UD-Q6_K_XL / Q8_0); a GGUF copy still has to be
  obtained before any Strata end-to-end run. The original
  Flash-Next IQ3_XXS goal stands, it is just not the model on this box.
- Host RAM / disk for PLE streaming: **tight**. 64 GiB host RAM is
  barely above the 35–55 GiB Strata loads into RAM on CUDA boxes,
  and `/data` (56 GiB free) cannot hold a second model copy — use the
  NFS mount for model storage and measure its read speed before
  trusting the PLE path.
- Reference deployment (not Strata, for while the port is in progress,
  web research 2026-11-04): vLLM-Ascend is the only ready-made route
  that supports Qwen3.6-35B-A3B including Gated DeltaNet; **v0.22.1
  aligns with this box (CANN 9.0.0 + torch_npu 2.10.0)** — do not
  install v0.23 (needs CANN 9.0.1) or v0.30 (needs CANN 9.2.0).
  BF16 Qwen3.6 does NOT fit a single 64 GiB card (~70 GiB weights);
  the W8A8 weights (~39.81 GiB, separate download) are the only
  single-card candidate and are officially unverified on a single
  910B. SGLang's current release wants CANN 9.1.0 + Python 3.12
  (mismatched here); MindIE's support list as checked did not list
  Qwen3.6.

## Open questions (fill in before phase 1)

- 910C box: still no target machine. §6 stays unvalidated.
- GGUF copy of Qwen3.6-35B-A3B for the Strata qwen36 path: not yet
  obtained (the BF16 safetensors on the box cannot be fed to Strata
  as-is).
- NFS read speed for the PLE streaming path: not yet measured.
- `ASCEND_RT_VISIBLE_DEVICES` mapping for `/dev/davinci14`: verified
  2026-11-04 — the runtime inside the container sees 1 device,
  id 0 (`strata_cann_probe` output above). Use device 0 in code;
  do not hard-code 14.
