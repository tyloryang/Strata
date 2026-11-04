# Strata on Ascend — the CANN backend

Experimental, opt-in, Phase 0. It does not serve a model yet.

Build it (on a machine with CANN installed; the first verified target
is a 910B2C with CANN 9.0.0, see `docs/CANN_PORT_PLAN.md` §8):

```sh
cmake -S cann -B build-cann -G Ninja
cmake --build build-cann
./build-cann/strata_cann_probe
```

Or from the top level, the same pattern as the SYCL backend:

```sh
cmake -S . -B build -DSTRATA_ENABLE_CANN=ON
cmake --build build
```

Without `-DSTRATA_ENABLE_CANN=ON` the top-level build never enters
this directory, and the default CUDA/HIP/SYCL builds are unchanged.

What exists so far:

- `src/cann_probe.cpp` — a no-model self-test. It initialises the
  CANN runtime and prints the visible NPU count. Exit 0 means the
  runtime sees at least one NPU.
- `CMakeLists.txt` — finds CANN via `ASCEND_HOME_PATH`,
  `ASCEND_TOOLKIT_HOME`, `/usr/local/Ascend/cann-9.0.0`, or
  `-DSTRATA_CANN_ROOT=<path>`, and fails configure with a plain
  message when CANN is absent instead of building a stub.

What does not exist yet: GEMM, attention (including Gated DeltaNet),
GGUF dequantisation, MoE routing, KV cache, MTP, multi-device. See
`docs/CANN_PORT_PLAN.md` for the phase order, and `docs/CANN.md`
for the user-facing status.
